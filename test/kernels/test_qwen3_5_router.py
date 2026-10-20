# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 router front on one SM100-family GPU (GB300): router_gemm_quant's logits
against the cuBLAS router GEMM SGLang runs (ReplicatedLinear, bf16), and its NVFP4
input against the trtllm runner's quantizer (flashinfer.fp4_quantize, linear scale
layout), bit for bit at 1-16 rows, eager and replayed from a CUDA graph."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103.moe import router


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _inputs(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, router.HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero quantization block
    x[-1, 32:48] *= 1e4  # a block far above the others
    w = (
        torch.randn(router.EXPERTS, router.HIDDEN, device="cuda", generator=g) * 0.02
    ).to(torch.bfloat16)
    scale = (448.0 * 6.0 / x.float().abs().max()).reshape(1)
    return x, w, scale


def _stock(x, w, scale):
    from flashinfer import fp4_quantize

    logits = torch.nn.functional.linear(x, w)
    fp4, sf = fp4_quantize(x, scale, 16, False, False)
    return logits, fp4.view(torch.uint8).reshape(x.shape[0], -1), sf.view(torch.uint8)


@pytest.mark.parametrize("m", list(range(1, router.MAX_TOKENS + 1)))
def test_router_gemm_quant_is_the_stock_router_gemm_and_quantizer(m):
    x, w, scale = _inputs(m, seed=m)
    got = router.router_gemm_quant(x, w, scale)
    want = _stock(x, w, scale)
    for name, a, b in zip(("logits", "x_fp4", "x_sf"), got, want):
        assert torch.equal(a.view(torch.uint8), b.reshape(a.shape).view(torch.uint8)), (
            name
        )


def test_router_gemm_quant_replays_from_a_cuda_graph():
    x, w, scale = _inputs(7, seed=0)
    router.build(x.device.index, 7)
    eager = router.router_gemm_quant(x, w, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.router_gemm_quant(x, w, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))
