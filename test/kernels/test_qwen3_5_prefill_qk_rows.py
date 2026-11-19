# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

H, KVH, D, ROT, EPS = 32, 2, 256, 64, 1e-6
WIDTH = H * 2 * D + 2 * KVH * D
MROPE_SECTIONS = (11, 11, 10)
MAXPOS = 131072
MROPE_OFFSETS = (0, 3, 7)


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate  # noqa: F401
        import sglang.kernels.ops.elementwise.elementwise  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs a GPU + SGLang")

if triton is not None:

    @triton.jit
    def _producer(src, dst, n, BLOCK: tl.constexpr):
        # releases its dependents at entry, then writes the projection rows
        tl.extra.cuda.gdc_launch_dependents()
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def _cos_sin(maxpos=MAXPOS, theta=1e7):
    inv = 1.0 / (
        theta ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT)
    )
    f = torch.outer(torch.arange(maxpos, dtype=torch.float32, device="cuda"), inv)
    return torch.cat([f.cos(), f.sin()], dim=-1).to(torch.bfloat16)


def _axis_map(layout="interleaved", dtype=torch.long):
    pairs = ROT // 2
    if layout == "interleaved":
        axes = [0] * pairs
        for axis in (1, 2):
            for lane in range(axis, min(3 * MROPE_SECTIONS[axis], pairs), 3):
                axes[lane] = axis
    else:
        axes = [axis for axis, n in enumerate(MROPE_SECTIONS) for _ in range(n)]
    return torch.tensor(axes, dtype=dtype, device="cuda")


def _inputs(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ch = torch.exp(torch.randn(WIDTH, generator=g, device="cuda") * 0.6)
    qkv = (torch.randn(T, WIDTH, generator=g, device="cuda") * ch).to(torch.bfloat16)
    q_gate, k, _ = qkv.split([H * 2 * D, KVH * D, KVH * D], dim=-1)
    qw = (torch.randn(D, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    kw = (torch.randn(D, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    high = MAXPOS - T - MROPE_OFFSETS[-1]
    start = torch.randint(0, high, (1,), generator=g, device="cuda")
    pos = (start + torch.arange(T, device="cuda")).to(torch.int64)
    return qkv, q_gate, k, qw, kw, pos


def _positions(pos, mrope):
    if not mrope:
        return pos, None
    rows = torch.stack([pos + offset for offset in MROPE_OFFSETS]).contiguous()
    return rows, _axis_map() if mrope is True else _axis_map(*mrope)


def _both(q_gate, k, qw, kw, cs, positions, axis):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

    args = (q_gate, k, qw, kw, cs, positions, EPS, H, KVH, D, ROT, True, axis)
    return fused_qk_gemma_rmsnorm_rope_gate(*args), rows.qk_gemma_rmsnorm_rope_rows(
        *args
    )


@pytest.mark.parametrize(
    "T, mrope",
    [
        (32768, True),
        (4096, False),
        (97, True),
        (97, ("in_order", torch.long)),
        (97, ("in_order", torch.int32)),
        (17, False),
        (1, True),
    ],
)
def test_the_rows_differ_from_sglangs_kernel_only_in_rare_roundings(T, mrope):
    from sgl_nvidia_mega.qwen3_5 import spec
    from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

    bar = spec.NUMERIC_TOLERANCES["attn_prefill_qk_rows"]
    differ_bar = bar["attn_prefill_qk_rows.q_k.bf16_differ_fraction_max"]
    cs = _cos_sin()
    _, q_gate, k, qw, kw, pos = _inputs(T, 7 + T)
    assert rows.qk_norm_rope_rows_supported(q_gate, k, D, ROT)
    positions, axis = _positions(pos, mrope)
    (qs, ks, gs), (qn, kn, gn) = _both(q_gate, k, qw, kw, cs, positions, axis)
    assert gn.data_ptr() == q_gate.data_ptr() + D * q_gate.element_size()
    assert torch.equal(gs, gn)
    for new, old in ((qn, qs), (kn, ks)):
        differ = (new.view(torch.int16) != old.view(torch.int16)).sum().item()
        assert differ <= max(4, new.numel() * differ_bar), differ


@pytest.mark.parametrize("T", [4096, 1])
def test_the_output_gate_reads_the_strided_gate_to_the_same_bytes(T):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    cs = _cos_sin()
    _, q_gate, k, qw, kw, pos = _inputs(T, 3)
    (_, _, gs), (_, _, gn) = _both(q_gate, k, qw, kw, cs, pos, None)
    g = torch.Generator(device="cuda").manual_seed(5)
    attn = (torch.randn(T, H * D, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    a = fused_sigmoid_mul(attn.clone(), gs.view(T, -1), inplace=True)
    b = fused_sigmoid_mul(attn.clone(), gn, inplace=True)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    (qa, sa), (qb, sb) = (
        fused_sigmoid_mul(attn, gs.view(T, -1), quant=True),
        fused_sigmoid_mul(attn, gn, quant=True),
    )
    assert torch.equal(qa.view(torch.uint8), qb.view(torch.uint8))
    assert torch.equal(sa, sb)


def test_the_launch_waits_for_the_projection_before_it():
    from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

    if not rows._ENABLE_PDL:
        pytest.skip("no programmatic dependent launch on this GPU")
    T = 8192
    cs = _cos_sin()
    qkv, q_gate, k, qw, kw, pos = _inputs(T, 13)
    positions, axis = _positions(pos, True)
    args = (cs, positions, EPS, H, KVH, D, ROT, True, axis)
    want_q, want_k, _ = rows.qk_gemma_rmsnorm_rope_rows(q_gate, k, qw, kw, *args)
    torch.cuda.synchronize()
    dst = torch.zeros_like(qkv)
    dq, dk, _ = dst.split([H * 2 * D, KVH * D, KVH * D], dim=-1)
    for _ in range(20):
        dst.zero_()
        _producer[(triton.cdiv(qkv.numel(), 1024),)](qkv, dst, qkv.numel(), BLOCK=1024)
        q, kk, _ = rows.qk_gemma_rmsnorm_rope_rows(dq, dk, qw, kw, *args)
        torch.cuda.synchronize()
        assert torch.equal(q, want_q) and torch.equal(kk, want_k)


def test_a_graph_replay_equals_the_eager_launch():
    from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

    T = 2048
    cs = _cos_sin()
    qkv, q_gate, k, qw, kw, pos = _inputs(T, 17)
    positions, axis = _positions(pos, True)
    args = (q_gate, k, qw, kw, cs, positions, EPS, H, KVH, D, ROT, True, axis)
    rows.warm(qw, kw, cs, EPS, H, KVH, D, ROT, axis)
    want_q, want_k, _ = rows.qk_gemma_rmsnorm_rope_rows(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, kk, _ = rows.qk_gemma_rmsnorm_rope_rows(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q, want_q) and torch.equal(kk, want_k)
