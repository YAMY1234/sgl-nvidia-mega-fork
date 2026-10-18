# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft's low-M LM head on one SM100-family GPU (GB300): FlashInfer's cute-dsl
``mm_bf16`` against the BF16 projection SGLang's draft runs, on the TP4 shard
(N 32320, K 5120) at the rows a DSpark draft step projects (5 a request, 1-3 requests,
and the 16-row bound). Only the K summation order differs, so the bound is
``sampler.LM_HEAD_LOGITS_REL``.

Run with ``python -m pytest test/kernels/test_deepseek_v41_draft_lm_head.py -s``.
"""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import sampler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

N, K = 32320, 5120
ROWS = (5, 10, 15, 16)


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(0)
    weight = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    lm_head = sampler.DraftLmHead(SimpleNamespace(weight=weight), low_m=True)
    assert lm_head.low_m, "FlashInfer cute-dsl mm_bf16 is unavailable"
    lm_head.tune(ROWS)
    return lm_head


def _us(fn, iters=50) -> float:
    """Per-call time from CUDA graph replay, as the draft graph runs it (the CuTe DSL
    host launch path would dominate an eager loop)."""
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (5 * iters)


@pytest.mark.parametrize("rows", ROWS)
def test_low_m_logits_are_the_bf16_projection_within_the_bound(head, rows):
    assert head.takes(rows) and not head.takes(sampler.LM_HEAD_MAX_ROWS + 1)
    x = torch.randn(rows, K, device="cuda").to(torch.bfloat16)
    weight = head.lm_head.weight
    want = torch.matmul(x, weight.T).float()
    got = head(x)
    assert got.dtype == torch.bfloat16 and got.shape == (rows, N)
    got = got.float()
    rel = ((got - want).abs().max() / want.abs().max()).item()
    assert rel <= sampler.LM_HEAD_LOGITS_REL, rel
    # greedy drafting reads the argmax; near-ties may flip, nothing else may
    flips = (got.argmax(-1) != want.argmax(-1)).sum().item()
    top2 = want.topk(2, dim=-1).values
    near_tie = ((top2[:, 0] - top2[:, 1]) <= 2 * rel * want.abs().max()).sum().item()
    assert flips <= near_tie
    print(
        f"rows {rows}: rel {rel:.2e}, argmax flips {flips}; "
        f"low-M {_us(lambda: head(x)):.1f} us, "
        f"cuBLAS {_us(lambda: torch.matmul(x, weight.T)):.1f} us"
    )


def test_low_m_logits_replay_in_a_cuda_graph(head):
    rows = 5
    x = torch.randn(rows, K, device="cuda").to(torch.bfloat16)
    eager = head(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = head(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, head(x))
    assert not torch.equal(out, eager)
