# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H = 4096
TOKENS = [1, 3, 7, 127, 1000, 1023, 1024, 4681, 32616, 32768]


def _inputs(m, amp, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, H, device="cuda", generator=g) * 1.5).to(torch.bfloat16)
    s = (torch.randn(m, H, device="cuda", generator=g) * amp).to(torch.bfloat16)
    f = (torch.randn(m, H, device="cuda", generator=g) * amp).to(torch.bfloat16)
    r = (torch.randn(m, H, device="cuda", generator=g) * amp * 4).to(torch.bfloat16)
    wg = (torch.randn(H, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    wn = (torch.randn(H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    if m >= 8:  # edge rows: zero output, a huge residual, tiny values, a cancelling sum
        f[1].zero_()
        s[1].zero_()
        r[2] *= 1e4
        f[3] *= 1e-30
        s[3] *= 1e-30
        r[3] *= 1e-30
        r[4] = -(f[4].float() + s[4].float() * 0.5).to(torch.bfloat16)
        r[5, :6] = 3000.0  # massive channels
    return x, s, f, r, wg, wn


def _stock(x, s, f, r, wg, wn, eps, scale=None):
    from flashinfer.norm import gemma_fused_add_rmsnorm
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    f, r = f.clone(), r.clone()
    fused_gate_sigmoid_mul_add(x, wg, s, f)
    gemma_fused_add_rmsnorm(f, r, wn, eps)
    q = static_quant_fp8(f, scale, repeat_scale=False)[0] if scale is not None else None
    return f, r, q


def _fused(x, s, f, r, wg, wn, eps, scale=None):
    from sgl_nvidia_mega.qwen3_5.moe.prefill import gate_fp32
    from sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_output_add_norm import (
        gated_add_rmsnorm,
        supported,
    )

    f, r = f.clone(), r.clone()
    g = gate_fp32(x, wg)
    assert supported(f, s, g, r)
    q = gated_add_rmsnorm(f, s, g, r, wn, eps, fp8_scale=scale)
    return f, r, q


def _eq(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("amp", [0.05, 1.0, 40.0])
def test_bitwise(m, amp):
    x, s, f, r, wg, wn = _inputs(m, amp, m * 31 + int(amp * 100))
    fs, rs, _ = _stock(x, s, f, r, wg, wn, 1e-6)
    ff, rf, q = _fused(x, s, f, r, wg, wn, 1e-6)
    assert q is None
    assert _eq(rf, rs), (
        f"residual: {int((rf.view(torch.int16) != rs.view(torch.int16)).sum())} differ"
    )
    assert _eq(ff, fs), (
        f"norm output: {int((ff.view(torch.int16) != fs.view(torch.int16)).sum())} differ"
    )


@pytest.mark.parametrize("m", [1, 7, 1024, 4681, 32768])
@pytest.mark.parametrize("s8", [0.0109, 1e-3, 3.0])
def test_bitwise_fp8(m, s8):
    x, s, f, r, wg, wn = _inputs(m, 1.0, m + int(s8 * 1000))
    sc = torch.tensor([s8], device="cuda", dtype=torch.float32)
    fs, rs, qs = _stock(x, s, f, r, wg, wn, 1e-6, sc)
    ff, rf, qf = _fused(x, s, f, r, wg, wn, 1e-6, sc)
    assert _eq(rf, rs) and _eq(ff, fs)
    assert _eq(qf, qs), (
        f"FP8: {int((qf.view(torch.uint8) != qs.view(torch.uint8)).sum())} differ"
    )


def test_fma_is_the_contraction():
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

    from sgl_nvidia_mega.qwen3_5.moe.prefill import gate_fp32

    x, s, f, r, wg, wn = _inputs(4096, 1.0, 5)
    ref = f.clone()
    fused_gate_sigmoid_mul_add(x, wg, s, ref)
    g = gate_fp32(x, wg)
    two = ((g[:, None] * s.float()) + f.float()).to(torch.bfloat16)  # torch: mul, add
    assert not torch.equal(two, ref)
