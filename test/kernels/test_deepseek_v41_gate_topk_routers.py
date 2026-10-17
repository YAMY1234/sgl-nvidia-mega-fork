# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""gate_topk.cuh's two routers on one SM100-family GPU (GB300): the target's (384 experts,
top-6, gate_topk.cu) and the DSpark draft's (128 experts, top-3, gate_topk_draft.cu)
against SGLang's moe_fused_gate, bit for bit (ids and weights), at the fast-gate widths
(1-16 and 48 rows), including ties, both softplus branches, NaN logits, padded rows and
no padded-row count; the ids also against torch's top-k, and replayed from a CUDA graph.
"""

import ctypes

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

ROUTERS = [(384, 6), (128, 3)]  # target, DSpark draft
ROWS = [*range(1, 17), 48]
SCALE = 1.5


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _stock(logits, bias, num_valid, top_k, scale):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    w, ids = moe_fused_gate(
        logits,
        bias,
        topk=top_k,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=True,
        num_token_non_padded=num_valid,
        sqrtsoftplus_log1p=True,
    )
    return w, ids.to(torch.int32)


def _fast(logits, bias, num_valid, top_k, scale):
    rows, experts = logits.shape
    weights = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    reg._launch(
        reg.load_gate_topk(logits.device.index, experts, top_k),
        (rows, 1, 1),
        [
            reg._pointer(logits),
            reg._pointer(bias),
            reg._null_or_pointer(None),
            reg._null_or_pointer(None),
            ctypes.c_int32(0),
            ctypes.c_int64(-1),
            reg._null_or_pointer(num_valid),
            reg._pointer(weights),
            reg._pointer(ids),
            ctypes.c_float(scale),
            ctypes.c_float(0.0),
        ],
        torch.cuda.current_stream().cuda_stream,
    )
    return weights, ids


def _logits(kind, rows, experts, gen):
    shape = (rows, experts)
    if kind == "normal":
        return torch.randn(shape, device="cuda", generator=gen) * 2
    if kind == "wide":  # both softplus branches (> 20 passes through)
        return torch.randn(shape, device="cuda", generator=gen) * 15
    if kind == "ties":  # few distinct values: ties broken by the lowest id
        return torch.randint(-3, 4, shape, device="cuda", generator=gen).float() * 0.5
    if kind == "correlated":  # the rows of one request choose overlapping experts
        base = torch.randn((1, experts), device="cuda", generator=gen)
        return base + 0.2 * torch.randn(shape, device="cuda", generator=gen)
    if kind == "nan":
        x = torch.randn(shape, device="cuda", generator=gen)
        x[torch.rand(shape, device="cuda", generator=gen) < 0.01] = float("nan")
        return x
    raise ValueError(kind)


@pytest.mark.parametrize("router", ROUTERS, ids=lambda r: f"{r[0]}x{r[1]}")
@pytest.mark.parametrize("scale", [SCALE, 1.0])
@pytest.mark.parametrize("kind", ["normal", "wide", "ties", "correlated", "nan"])
def test_each_router_matches_moe_fused_gate(router, scale, kind):
    experts, top_k = router
    gen = torch.Generator(device="cuda").manual_seed(4321 + experts)
    for rows in ROWS:
        for trial in range(4):
            logits = _logits(kind, rows, experts, gen).contiguous()
            bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
            valid = rows if trial % 2 == 0 else max(1, rows - 1 - trial)
            num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
            if trial == 2:  # no num_token_non_padded: every row is valid
                num_valid = None
            sw, sid = _stock(logits, bias, num_valid, top_k, scale)
            fw, fid = _fast(logits, bias, num_valid, top_k, scale)
            torch.cuda.synchronize()
            where = f"{kind} rows={rows} valid={valid} trial={trial}"
            assert torch.equal(fid, sid), where
            assert torch.equal(fw.view(torch.int32), sw.view(torch.int32)), where


@pytest.mark.parametrize("router", ROUTERS, ids=lambda r: f"{r[0]}x{r[1]}")
def test_each_router_picks_torchs_top_k(router):
    """On well-separated scores the ids are torch's top-k of the biased sqrtsoftplus."""
    experts, top_k = router
    gen = torch.Generator(device="cuda").manual_seed(17)
    for rows in ROWS:
        logits = _logits("normal", rows, experts, gen).contiguous()
        bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
        scores = torch.nn.functional.softplus(logits.double()).sqrt() + bias.double()
        _, fid = _fast(logits, bias, None, top_k, SCALE)
        assert torch.equal(fid.long(), scores.topk(top_k, dim=-1).indices), rows


@pytest.mark.parametrize("router", ROUTERS, ids=lambda r: f"{r[0]}x{r[1]}")
@pytest.mark.parametrize("rows", [3, 12, 48])
def test_each_router_follows_live_inputs_in_a_graph(router, rows):
    experts, top_k = router
    gen = torch.Generator(device="cuda").manual_seed(11 + rows)
    logits = torch.randn((rows, experts), device="cuda", generator=gen)
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
    num_valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    _fast(logits, bias, num_valid, top_k, SCALE)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fast(logits, bias, num_valid, top_k, SCALE)
    for _ in range(3):
        logits.copy_(torch.randn((rows, experts), device="cuda", generator=gen))
        graph.replay()
        ref = _stock(logits, bias, num_valid, top_k, SCALE)
        torch.cuda.synchronize()
        for got, want in zip(out, ref):
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))
