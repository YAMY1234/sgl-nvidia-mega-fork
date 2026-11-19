# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H = 4096
TOKENS = [1, 3, 64, 127, 1000, 4096, 32768]
GSCALES = [1270.0, 30000.0, 5.0]


def _stock(x, r, w, eps, gs):
    from flashinfer.norm import gemma_fused_add_rmsnorm
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    gemma_fused_add_rmsnorm(x, r, w, eps)
    q, sf = fp4_quantize(x, gs, sf_vec_size=16, is_sf_swizzled_layout=False)
    return (
        x,
        r,
        q.view(torch.uint8).reshape(x.shape[0], -1),
        sf.view(torch.uint8).reshape(x.shape[0], -1),
    )


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("gscale", GSCALES)
@pytest.mark.parametrize("amp", [1.0, 30.0])
def test_bitwise(m, gscale, amp):
    from sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_input_nvfp4 import (
        gemma_fused_add_rmsnorm_nvfp4,
        supported,
    )

    g = torch.Generator(device="cuda").manual_seed(m * 7 + int(gscale) + int(amp))
    x = (torch.randn(m, H, device="cuda", generator=g) * amp).to(torch.bfloat16)
    r = (torch.randn(m, H, device="cuda", generator=g) * amp).to(torch.bfloat16)
    w = (torch.randn(H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    gs = torch.tensor([gscale], device="cuda", dtype=torch.float32)
    eps = 1e-6
    assert supported(x, r)
    xs, rs, qs, sfs = _stock(x.clone(), r.clone(), w, eps, gs)
    xf, rf = x.clone(), r.clone()
    qf, sff = gemma_fused_add_rmsnorm_nvfp4(xf, rf, w, eps, gs)
    assert torch.equal(xf.view(torch.int16), xs.view(torch.int16)), (
        "BF16 output differs"
    )
    assert torch.equal(rf.view(torch.int16), rs.view(torch.int16)), "residual differs"
    bad_sf = int((sff != sfs).sum())
    bad_q = int((qf != qs).sum())
    assert bad_sf == 0 and bad_q == 0, (
        f"sf {bad_sf} / {sfs.numel()}, q {bad_q} / {qs.numel()} bytes differ"
    )


@pytest.mark.parametrize("m", [1, 127, 4096, 32768])
@pytest.mark.parametrize("s8", [0.0109, 1e-3, 3.0])
def test_bitwise_dual_fp8(m, s8):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_input_nvfp4 import (
        gemma_fused_add_rmsnorm_nvfp4,
    )

    g = torch.Generator(device="cuda").manual_seed(m + int(s8 * 1000))
    x = (torch.randn(m, H, device="cuda", generator=g) * 4).to(torch.bfloat16)
    r = (torch.randn(m, H, device="cuda", generator=g) * 4).to(torch.bfloat16)
    w = (torch.randn(H, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    gs = torch.tensor([1270.0], device="cuda", dtype=torch.float32)
    s = torch.tensor([s8], device="cuda", dtype=torch.float32)
    xs, rs, qs, sfs = _stock(x.clone(), r.clone(), w, 1e-6, gs)
    q8s, _ = static_quant_fp8(xs, s, repeat_scale=False)
    xf, rf = x.clone(), r.clone()
    qf, sff, q8f = gemma_fused_add_rmsnorm_nvfp4(xf, rf, w, 1e-6, gs, fp8_scale=s)
    assert torch.equal(xf, xs) and torch.equal(rf, rs)
    assert torch.equal(qf, qs) and torch.equal(sff, sfs)
    assert torch.equal(q8f.view(torch.uint8), q8s.view(torch.uint8)), (
        "FP8 output differs"
    )
