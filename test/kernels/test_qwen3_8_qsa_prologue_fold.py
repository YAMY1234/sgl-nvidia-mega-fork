"""QSA prologue fold bitwise gate against SGLang's two-launch indexer prologue; run on SM10x.

The provider's one launch (q_prep and the group compression) must leave q (padded
heads included), the ring rows, the RoPE positions and the compressed rows bit-identical
to qsa_index_q_norm_rope_store followed by qsa_index_k_compress_store. Groups mix
members this step writes with older ring rows.
"""

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    QsaPrologueFoldIO,
    QsaPrologueFoldSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers

HEAD_DIM = 128
Q_HEADS = 4
Q_HEADS_PADDED = 8
ROTARY = 64
RATIO = 4
SLOTS = 512
POSITIONS = 4096
EPS = 1e-6


def _inputs(tokens, groups, out_dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    def randint(high, *shape):
        return torch.randint(0, high, shape, generator=g, device="cuda")

    bf16 = torch.bfloat16
    slots = torch.randperm(SLOTS, generator=g, device="cuda")
    cache_loc = slots[:tokens].contiguous()
    # Each group starts at a slot this step writes or an older one; the other
    # members are drawn from both, so the fold must resolve each against cache_loc.
    pool = slots[: tokens + 2 * groups * RATIO]
    group_locs = torch.stack(
        [
            pool[torch.randperm(pool.numel(), generator=g, device="cuda")[:RATIO]]
            for _ in range(groups)
        ]
    ).to(torch.int32)
    write_locs = torch.randperm(SLOTS, generator=g, device="cuda")[:groups].to(
        torch.int32
    )
    axis_map = (torch.arange(ROTARY // 2, device="cuda") % 3).to(torch.int32)
    return dict(
        qk=randn(tokens, (Q_HEADS + 1) * HEAD_DIM, dtype=bf16),
        positions=randint(POSITIONS, 3, tokens),
        cos_sin_cache=randn(POSITIONS, ROTARY),
        axis_map=axis_map,
        q_weight=0.1 * randn(HEAD_DIM, dtype=bf16),
        cache_loc=cache_loc,
        key_state_buffer=randn(SLOTS, HEAD_DIM, dtype=bf16),
        rope_position_buffer=randint(POSITIONS, SLOTS, 3),
        group_locs=group_locs.contiguous(),
        k_weight=0.1 * randn(HEAD_DIM, dtype=bf16),
        write_locs=write_locs,
        compressed_k_buffer=torch.zeros(SLOTS, HEAD_DIM, device="cuda").to(out_dtype),
    )


def _bits(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="explicit SM10x qualification",
)
@pytest.mark.parametrize(
    "tokens,groups,out_dtype,neox",
    [
        (4, 1, torch.bfloat16, True),
        (32, 8, torch.bfloat16, True),
        (33, 9, torch.float8_e4m3fn, True),
        (128, 32, torch.bfloat16, False),
        (128, 32, torch.float8_e4m3fn, True),
    ],
)
def test_prologue_fold_bits(tmp_path, monkeypatch, tokens, groups, out_dtype, neox):
    from sglang.kernels.ops.attention.qsa_indexer import (
        qsa_index_k_compress_store,
        qsa_index_q_norm_rope_store,
    )

    monkeypatch.setenv("SGLANG_QSA_QPF_SRC_DIR", str(tmp_path))
    out_scalar = D.FP8_E4M3 if out_dtype == torch.float8_e4m3fn else D.BF16
    signature = QsaPrologueFoldSignature(
        O.QSA_PROLOGUE_FOLD,
        0,
        Q_HEADS,
        Q_HEADS_PADDED,
        HEAD_DIM,
        ROTARY,
        3,
        RATIO,
        D.BF16,
        out_scalar,
        int(neox),
    )
    offer = providers.make_offer(signature)
    assert offer is not None

    x = _inputs(tokens, groups, out_dtype, seed=tokens * 31 + groups)
    ref = {k: v.clone() for k, v in x.items()}
    q_ref = qsa_index_q_norm_rope_store(
        ref["qk"],
        ref["positions"],
        ref["cos_sin_cache"],
        ref["axis_map"],
        ref["q_weight"],
        ref["cache_loc"],
        ref["key_state_buffer"],
        ref["rope_position_buffer"],
        Q_HEADS,
        ROTARY,
        EPS,
        neox,
        q_heads_padded=Q_HEADS_PADDED,
        out_dtype=out_dtype,
    )
    qsa_index_k_compress_store(
        ref["key_state_buffer"],
        ref["group_locs"],
        ref["rope_position_buffer"],
        ref["cos_sin_cache"],
        ref["axis_map"],
        ref["k_weight"],
        ref["write_locs"],
        ref["compressed_k_buffer"],
        RATIO,
        ROTARY,
        EPS,
        neox,
    )

    q = torch.empty(tokens, Q_HEADS_PADDED, HEAD_DIM, device="cuda", dtype=out_dtype)
    offer.run(QsaPrologueFoldIO(**x, q_out=q, q_eps=EPS, k_eps=EPS))
    torch.cuda.synchronize()

    assert torch.equal(_bits(q), _bits(q_ref))
    assert torch.equal(_bits(x["key_state_buffer"]), _bits(ref["key_state_buffer"]))
    assert torch.equal(x["rope_position_buffer"], ref["rope_position_buffer"])
    assert torch.equal(
        _bits(x["compressed_k_buffer"]), _bits(ref["compressed_k_buffer"])
    )
    # The step wrote what it had to: each written compressed row changed.
    written = ref["compressed_k_buffer"][ref["write_locs"].long()].float()
    assert bool(written.abs().sum(dim=1).gt(0).all())
