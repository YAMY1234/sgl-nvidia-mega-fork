# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FN TP4 T4 native conv and ReplaySSM parity (including PDL).

Native checkpoint/output/ring stores must be bitwise. This file does not qualify
numeric MK/front, service smoke, performance or full-model accuracy.
"""

import os

import pytest
import torch

from types import SimpleNamespace
from sgl_nvidia_mega.qwen3_8.gdn.config import shapes

HF = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560,
    linear_num_key_heads=16, linear_num_value_heads=48,
    linear_key_head_dim=128, linear_value_head_dim=128))
SHAPE = shapes(HF, tokens_per_request=4, tp=4)
H, K, HV, V = SHAPE.hk, SHAPE.key_dim, SHAPE.hv, SHAPE.value_dim
D, DRAFT, WIDTH, SLOTS = SHAPE.channels, SHAPE.tokens_per_request, 4, 9


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _projection(bs, views, g):
    """A verify step's mixed q / k / v rows [bs * DRAFT, D]: a contiguous copy, or the
    token-major view of the q / k / v / z projection."""
    qkvz = torch.randn(bs * DRAFT, D + HV * V, device="cuda", generator=g).to(
        torch.bfloat16
    )
    return qkvz[:, :D] if views else qkvz[:, :D].contiguous()


# a covered step's requests (slot gdn_verify covers B1, B2 and B3) and their slots
REQUESTS = (1, 2, 3)
STATE_SLOTS = (4, 7, 0, 8)
RING_SLOTS = (3, 5, 0, 8)
REPLAY_SLOTS = (1, 2, 4, 6)


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("views", [False, True])
@pytest.mark.parametrize("block_n", [256, 128])
def test_conv_update_is_sglangs(views, block_n, bs):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    from sgl_nvidia_mega.qwen3_8.gdn.conv import conv_update

    g = torch.Generator(device="cuda").manual_seed(block_n + views + 10 * bs + 1000 * int(os.getenv("GDN_TEST_RANK", "0")))
    rows = _projection(bs, views, g)
    weight = torch.randn(D, WIDTH, device="cuda", generator=g).to(torch.bfloat16)
    bias = torch.randn(D, device="cuda", generator=g).to(torch.bfloat16)
    state = torch.randn(SLOTS, D, WIDTH - 1, device="cuda", generator=g).to(
        torch.bfloat16
    )
    window = torch.zeros(
        SLOTS, DRAFT, D, WIDTH - 1, dtype=torch.bfloat16, device="cuda"
    )
    slots = torch.tensor(STATE_SLOTS[:bs], dtype=torch.int32, device="cuda")
    inter = torch.arange(bs, dtype=torch.int32, device="cuda")
    x = rows.view(bs, DRAFT, D).transpose(1, 2)
    want_state, want_window = state.clone(), window.clone()
    want = causal_conv1d_update(
        x,
        want_state,
        weight,
        bias,
        "silu",
        conv_state_indices=slots,
        intermediate_conv_window=want_window,
        intermediate_state_indices=inter,
    )
    got = conv_update(
        x,
        state,
        weight,
        bias,
        "silu",
        slots,
        window,
        inter,
        block_n=block_n,
        token_major_out=views,
    )
    assert torch.equal(got.contiguous(), want.contiguous())
    assert torch.equal(state, want_state) and torch.equal(window, want_window)
    # the slot reads the output back as token rows without a copy
    if views:
        assert got.transpose(1, 2).is_contiguous()


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("pdl, gate", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("block_v,num_warps", [(64, 1), (32, 2)])
def test_replayssm_verify_is_sglangs(pdl, gate, bs, block_v, num_warps):
    """The slot's verify launch (32-wide value tiles on 2 warps, and with or without the
    dependent launch, waiting before the gates with ``gate``) against
    gdn_replayssm_spec_decode's verify launch: the output and this step's ring entries."""
    try:
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )
    except ImportError:  # in fla/ before sgl-project/sglang#45112
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

    from sgl_nvidia_mega.qwen3_8.gdn import replayssm

    g = torch.Generator(device="cuda").manual_seed(
        3 + int(pdl) + 2 * int(gate) + 10 * bs + 1000 * int(os.getenv("GDN_TEST_RANK", "0"))
    )
    ring_len, slots = 16, 9
    tokens = bs * DRAFT

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=g).to(dtype)

    qkv = rand(tokens, D)  # strided q / k / v, as the slot's split leaves them
    q = qkv[:, : H * K].view(tokens, H, K)
    k = qkv[:, H * K : 2 * H * K].view(tokens, H, K)
    v = qkv[:, 2 * H * K :].view(tokens, HV, V)
    ba = rand(tokens, 2 * HV)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log, dt_bias = rand(HV, dtype=torch.float32), rand(HV, dtype=torch.float32)
    state = rand(slots, HV, V, K)
    ring = [
        rand(slots, HV, ring_len, V),
        rand(slots, H, ring_len, K),
        -torch.rand(slots, HV, ring_len, device="cuda", generator=g),
        rand(slots, HV, ring_len, V),
        rand(slots, H, ring_len, K),
    ]
    write_pos = torch.randint(0, ring_len - DRAFT, (slots,), device="cuda", generator=g)
    write_pos = write_pos.to(torch.int32)
    cache_base = torch.randint(0, ring_len, (slots,), device="cuda", generator=g)
    cache_base = cache_base.to(torch.int32)
    is_flush = torch.zeros(slots, dtype=torch.int8, device="cuda")
    qsl = torch.arange(0, tokens + 1, DRAFT, dtype=torch.int32, device="cuda")
    states = torch.tensor(RING_SLOTS[:bs], dtype=torch.int32, device="cuda")
    replay = torch.tensor(REPLAY_SLOTS[:bs], dtype=torch.int32, device="cuda")
    want_ring, got_ring = [t.clone() for t in ring], [t.clone() for t in ring]
    want, got = (q.new_empty(tokens, HV, V) for _ in range(2))
    gdn_replayssm_spec_decode(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        checkpoint_state=state,
        d_cache=want_ring[0],
        k_cache=want_ring[1],
        g_cache=want_ring[2],
        rawv_cache=want_ring[3],
        rawk_cache=want_ring[4],
        beta_cache=None,
        out=want,
        query_start_loc=qsl,
        ssm_state_indices=states,
        replay_indices=replay,
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
        launch_mode="verify",
        block_v=block_v,
        num_warps=num_warps,
    )
    replayssm.verify(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        got,
        state,
        *got_ring,
        None,
        qsl,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=block_v,
        num_warps=num_warps,
        pdl=pdl,
        wait_for_gate_producer=gate,
    )
    assert torch.equal(got, want)
    assert all(torch.equal(x, y) for x, y in zip(got_ring, want_ring))


