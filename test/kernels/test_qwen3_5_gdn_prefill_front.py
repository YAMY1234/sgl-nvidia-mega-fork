# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random

import pytest
import torch

NK, NV, D = 16, 64, 128
QKV = (2 * NK + NV) * D
QKVZ = QKV + NV * D
SLOTS = 48


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import sglang.kernels.ops.attention.fla.l2norm  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM100-family GPU + SGLang"
)

CASES = {
    "chunk_32k_7req": ([4681] * 6 + [4682], [True] * 7),
    "cold_32k": ([32768], [False]),
    "ragged": (
        [1, 2, 3, 0, 5, 63, 64, 65, 129, 1000, 4, 2, 700],
        [
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
        ],
    ),
}


def _batch(lens, hinit, seed, extreme, pad_seq=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    qkvz = torch.randn(T, QKVZ, generator=g, device="cuda") * 1.5
    if extreme:
        n = T * QKV // 50
        flat = qkvz[:, :QKV].contiguous().view(-1)
        flat[torch.randint(0, T * QKV, (n,), generator=g, device="cuda")] = (
            torch.randn(n, generator=g, device="cuda") * 200.0
        )
        qkvz[:, :QKV] = flat.view(T, QKV)
    qkvz = qkvz.to(torch.bfloat16)
    w = (torch.randn(QKV, 4, generator=g, device="cuda") * 0.35).to(torch.bfloat16)
    if extreme:
        w[::97] *= 40
    pool = (torch.randn(SLOTS, QKV, 3, generator=g, device="cuda") * 1.5).to(
        torch.bfloat16
    )
    slots = random.Random(seed).sample(range(1, SLOTS), len(lens))
    if pad_seq is not None:
        slots[pad_seq] = -1
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return dict(
        T=T,
        lens=lens,
        mixed_qkv=qkvz[:, :QKV],
        w=w,
        pool=pool,
        cidx=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        qsl=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        hinit=torch.tensor(hinit, dtype=torch.bool, device="cuda"),
    )


def _block_table(lens):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        CAUSAL_CONV1D_FWD_BLOCK_M as M,
    )

    counts = [(n + M - 1) // M for n in lens]
    if len(counts) * max(counts) <= 4 * sum(counts):
        return None
    pairs = [(s, b) for s, c in enumerate(counts) for b in range(c)]
    return torch.tensor(pairs, dtype=torch.int32, device="cuda")


def _stock(x, pool):
    from sglang.kernels.ops.attention.fla.l2norm import gdn_prefill_qkv_prepare_fwd
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    bt = _block_table(x["lens"])
    out = causal_conv1d_fn(
        x["mixed_qkv"].transpose(0, 1),
        x["w"],
        None,
        activation="silu",
        conv_states=pool,
        has_initial_state=x["hinit"],
        cache_indices=x["cidx"],
        query_start_loc=x["qsl"],
        seq_lens_cpu=x["lens"],
        **({"block_table": bt} if bt is not None else {}),
    ).transpose(0, 1)[: x["T"]]
    q, k, v = torch.split(out, [NK * D, NK * D, NV * D], dim=-1)
    q, k, v = (t.view(x["T"], -1, D) for t in (q, k, v))
    return out, gdn_prefill_qkv_prepare_fwd(q, k, v)


def _front(x, pool):
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import prefill_front as PF

    assert PF.supported(
        num_k_heads=NK,
        num_v_heads=NV,
        head_k_dim=D,
        head_v_dim=D,
        conv_width=4,
        activation="silu",
    )
    assert PF.eligible(x["mixed_qkv"], x["w"], None, pool, x["qsl"], x["cidx"]) == ""
    return PF.gdn_prefill_front(
        x["mixed_qkv"],
        x["w"],
        None,
        pool,
        x["qsl"],
        x["cidx"],
        x["hinit"],
        num_k_heads=NK,
        num_v_heads=NV,
        activation="silu",
    )


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


@pytest.mark.parametrize("extreme", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_front_bitwise(case, extreme):
    lens, hinit = CASES[case]
    x = _batch(
        lens, hinit, seed=2 * list(CASES).index(case) + int(extreme), extreme=extreme
    )
    p_s, p_f = x["pool"].clone(), x["pool"].clone()
    _, (qs, ks, vs) = _stock(x, p_s)
    qf, kf, vf = _front(x, p_f)
    for a, b in ((qs, qf), (ks, kf), (vs, vf), (p_s, p_f)):
        assert torch.equal(_bits(a), _bits(b))


def test_padded_slot_rows_and_state():
    lens, hinit = CASES["ragged"]
    x = _batch(lens, hinit, seed=5, extreme=True, pad_seq=9)
    p_s, p_f = x["pool"].clone(), x["pool"].clone()
    _, (qs, ks, vs) = _stock(x, p_s)
    qf, kf, vf = _front(x, p_f)
    keep = torch.ones(x["T"], dtype=torch.bool, device="cuda")
    keep[x["qsl"][9].item() : x["qsl"][10].item()] = False
    for a, b in ((qs, qf), (ks, kf), (vs, vf)):
        assert torch.equal(_bits(a[keep]), _bits(b[keep]))
    assert torch.equal(_bits(p_s), _bits(p_f))
    assert not torch.equal(
        _bits(p_f), _bits(x["pool"])
    )  # the other sequences' states moved


@pytest.mark.parametrize("checkpoints", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_flashinfer_prefill_on_front_inputs(state_dtype, checkpoints):
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import prefill_front as PF

    lens, hinit = CASES["chunk_32k_7req"]
    x = _batch(lens, hinit, seed=21, extreme=False)
    g = torch.Generator(device="cuda").manual_seed(3)
    ba = (torch.randn(x["T"], 2 * NV, generator=g, device="cuda") * 2).to(
        torch.bfloat16
    )
    A_log = torch.rand(NV, generator=g, device="cuda") * 2.7
    dt_bias = (torch.randn(NV, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    ssm = (torch.randn(SLOTS, NV, D, D, generator=g, device="cuda") * 0.05).to(
        state_dtype
    )
    every = 64
    counts = [n // every for n in lens] if checkpoints else [0] * len(lens)
    starts = [0]
    for n in counts:
        starts.append(starts[-1] + n)
    plan = dict(
        state_checkpoint_cu_starts=(
            torch.tensor(starts, dtype=torch.int64, device="cuda")
            if checkpoints
            else None
        ),
        num_state_checkpoints=starts[-1],
        state_checkpoint_every_n_tokens=every if checkpoints else 0,
    )
    kernel = FlashInferGDNKernel()

    out, _ = _stock(x, x["pool"].clone())
    q, k, v = torch.split(out, [NK * D, NK * D, NV * D], dim=-1)
    q, k, v = (t.reshape(x["T"], -1, D).unsqueeze(0) for t in (q, k, v))
    gate, beta = fused_gdn_gating(A_log, ba[:, NV:], ba[:, :NV], dt_bias)
    state_s = ssm.clone()
    o_s, _, h_s = kernel.extend(
        q,
        k,
        v,
        gate,
        beta,
        ssm_states=state_s,
        cache_indices=x["cidx"],
        query_start_loc=x["qsl"],
        **plan,
    )

    qf, kf, vf = _front(x, x["pool"].clone())
    gf, bf = PF.gdn_prefill_gates(A_log, ba[:, NV:], ba[:, :NV], dt_bias)
    state_f = ssm.clone()
    view = SimpleNamespace(state_indices=x["cidx"], query_start_loc=x["qsl"], **plan)
    helpers = prefill_fi._state_copies()
    copies = None if helpers is None else helpers[0]
    o_f, h_f, _ = prefill_fi._flashinfer_extend(
        kernel, qf, kf, vf, gf, bf, state_f, view, None, copies
    )
    assert torch.equal(_bits(o_s), _bits(o_f))
    assert torch.equal(_bits(state_s), _bits(state_f))
    assert (h_s is None) == (h_f is None) == (not checkpoints)
    if checkpoints:
        assert torch.equal(_bits(h_s), _bits(h_f))


@pytest.mark.parametrize("copies", ["index-ops", "one-launch"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_flashinfer_prefill_track_snapshot_on_front_inputs(
    monkeypatch, state_dtype, copies
):
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid
    from sglang.srt.layers.attention.linear import gdn_extend_track
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot, prefill_fi

    if copies == "index-ops":
        monkeypatch.setattr(gdn_extend_track, "track_state_rows_ok", lambda **kw: False)
    lens, hinit = CASES["chunk_32k_7req"]
    x = _batch(lens, hinit, seed=23, extreme=False)
    g = torch.Generator(device="cuda").manual_seed(5)
    ba = (torch.randn(x["T"], 2 * NV, generator=g, device="cuda") * 2).to(
        torch.bfloat16
    )
    A_log = torch.rand(NV, generator=g, device="cuda") * 2.7
    dt_bias = (torch.randn(NV, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    ssm = (torch.randn(SLOTS, NV, D, D, generator=g, device="cuda") * 0.05).to(
        state_dtype
    )
    every = 64
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n // every)
    cu = x["qsl"].tolist()
    tracked = [(0, 10), (2, lens[2] // every - 1), (4, 0), (6, lens[6] // every - 1)]
    free = [s for s in range(1, SLOTS) if s not in x["cidx"].tolist()][:5]
    ends = [cu[i] + every * (j + 1) for i, j in tracked] + [cu[4]]
    on = dict(device="cuda", dtype=torch.int64)
    view = SimpleNamespace(
        state_indices=x["cidx"],
        query_start_loc=x["qsl"],
        has_track=True,
        track_conv_indices=torch.tensor([[e - 3, e - 2, e - 1] for e in ends], **on),
        track_conv_dst=torch.tensor(free, **on),
        track_ssm_h_src=torch.tensor([starts[i] + j for i, j in tracked], **on),
        track_ssm_h_dst=torch.tensor(free[:4], **on),
        track_ssm_final_src=x["cidx"][3:4].long(),
        track_ssm_final_dst=torch.tensor(free[4:], **on),
        track_ssm_recompute_dst=None,
        state_checkpoint_cu_starts=torch.tensor(starts, **on),
        num_state_checkpoints=starts[-1],
        state_checkpoint_every_n_tokens=every,
    )
    metadata = prefill_fi._track_metadata(view)
    metadata.has_mamba_track_mask = True
    kernel = FlashInferGDNKernel()
    a, b = ba[:, NV:], ba[:, :NV]

    pool_s, state_s = x["pool"].clone(), ssm.clone()
    gdn_extend_track.track_conv_windows(
        conv_states=pool_s, mixed_qkv=x["mixed_qkv"], metadata=metadata
    )
    out, _ = _stock(x, pool_s)
    q, k, v = torch.split(out, [NK * D, NK * D, NV * D], dim=-1)
    q, k, v = (t.reshape(x["T"], -1, D).unsqueeze(0) for t in (q, k, v))
    gate, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    o_s, _, h_s = kernel.extend(
        q,
        k,
        v,
        gate,
        beta,
        ssm_states=state_s,
        cache_indices=x["cidx"],
        query_start_loc=x["qsl"],
        state_checkpoint_cu_starts=view.state_checkpoint_cu_starts,
        num_state_checkpoints=view.num_state_checkpoints,
        state_checkpoint_every_n_tokens=every,
    )
    one_launch = gdn_extend_track.track_ssm_states_in_one_launch(
        ssm_states=state_s, h=h_s, h_track_buf=None, metadata=metadata
    )
    assert one_launch is (copies == "one-launch")
    if not one_launch:
        hybrid.MambaAttnBackendBase._track_mamba_state_extend(
            None, None, h_s, state_s, metadata
        )

    # the slot: prefill_fi's extend with the snapshot
    pool_f, state_f = x["pool"].clone(), ssm.clone()
    rows = []
    prefill = kernel._prefill_fn
    monkeypatch.setattr(
        kernel,
        "_prefill_fn",
        lambda **kw: rows.append(kw.get("checkpoint_state_indices")) or prefill(**kw),
    )
    attn = SimpleNamespace(
        layer_id=0,
        conv_weights=x["w"],
        bias=None,
        num_k_heads=NK,
        num_v_heads=NV,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
    )
    linear = SimpleNamespace(
        state_layer_cache=lambda _: SimpleNamespace(conv=(pool_f,), temporal=state_f),
        kernel_dispatcher=SimpleNamespace(extend_kernel=kernel),
    )
    o_f = prefill_fi.extend(
        attn,
        x["mixed_qkv"],
        a,
        b,
        view,
        SimpleNamespace(extend_prefix_lens=x["hinit"].to(torch.int32)),
        linear,
        snapshot=fi_track_snapshot.TrackSnapshot(),
    )
    (got,) = rows
    assert got is not None and int((got >= 0).sum()) == len(tracked)
    assert torch.equal(_bits(o_s), _bits(o_f))
    assert torch.equal(_bits(pool_s), _bits(pool_f))
    assert torch.equal(_bits(state_s), _bits(state_f))
    for slot in free:
        assert not torch.equal(_bits(state_f[slot]), _bits(ssm[slot]))


@pytest.mark.parametrize("scale", [2.0, 60.0])
@pytest.mark.parametrize("tokens", [1, 33, 32768])
def test_gates_bitwise(tokens, scale):
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating

    from sgl_nvidia_mega.qwen3_5.sm103.gdn import prefill_front as PF

    g = torch.Generator(device="cuda").manual_seed(tokens)
    ba = (torch.randn(tokens, 2 * NV, generator=g, device="cuda") * scale).to(
        torch.bfloat16
    )
    b, a = ba[:, :NV], ba[:, NV:]
    A_log = torch.rand(NV, generator=g, device="cuda") * 2.7
    dt_bias = (torch.randn(NV, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    assert PF.gates_eligible(A_log, a, b, dt_bias)
    gs, bs = fused_gdn_gating(A_log, a, b, dt_bias)
    gf, bf = PF.gdn_prefill_gates(A_log, a, b, dt_bias)
    assert gs.shape == gf.shape and bs.shape == bf.shape
    assert torch.equal(_bits(gs), _bits(gf))
    assert torch.equal(_bits(bs), _bits(bf))


def test_negative_control_detects_norm_order():
    lens, hinit = CASES["chunk_32k_7req"]
    x = _batch(lens, hinit, seed=11, extreme=False)
    out, (qs, _, _) = _stock(x, x["pool"].clone())
    q = out[:, : NK * D].reshape(x["T"], NK, D).float()
    ref = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    assert not torch.equal(_bits(ref), _bits(qs))
