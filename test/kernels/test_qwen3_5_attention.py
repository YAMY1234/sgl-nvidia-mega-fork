# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The full-attention kernels of slot skv2, on one SM100-family GPU (GB300), at Qwen3.5's
TP8 verify shape (7 rows of 4 query heads over one KV head of 256, FP8 pages of 64):

- SKV2 against an FP64 reference of the masked attention: at most twice the error of
  SGLang's attention (TRT-LLM MHA's decode kernel) on the same FP8 inputs, at KV lengths
  from one page to past 128K;
- SKV2 replayed in a CUDA graph on new inputs, bit for bit its eager result.

Its bits against our 2.2 branch's SKV2 are checked with the branch's tree
(``slot_vs_branch``), not here.
"""

import math

import pytest
import torch

ROWS, HEADS, DIM, PAGE = 7, 4, 256, 64
SCALE = DIM**-0.5


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _hnd(pool):
    """The pool's NHD rows as HND pages of one KV head, with TRT-LLM MHA's canonical
    strides (as its decode kernel and SKV2 get them)."""
    pages = pool.shape[0] // PAGE
    return pool.as_strided((pages, 1, PAGE, DIM), (PAGE * DIM, PAGE * DIM, DIM, 1))


def _fill(q, pools, table, lengths, length, g):
    """New FP8 query rows and pool contents, the request's pages scattered over the
    pool, and its length, written in place."""
    q.copy_((torch.randn(q.shape, device="cuda", generator=g) * 2).to(q.dtype))
    for pool in pools:
        pool.copy_(
            (torch.randn(pool.shape, device="cuda", generator=g) * 2).to(q.dtype)
        )
    pages = math.ceil(length / PAGE)
    total = pools[0].shape[0] // PAGE
    table.zero_()
    table[0, :pages] = torch.randperm(total, device="cuda", generator=g)[:pages].int()
    lengths.fill_(length)


def _inputs(length, g, spare_pages=8):
    """FP8 query rows, a paged FP8 KV pool with the request's pages scattered, its page
    table and its length."""
    total = math.ceil(length / PAGE) + spare_pages
    q = torch.empty(ROWS, HEADS, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    pools = [
        torch.empty(total * PAGE, 1, DIM, dtype=torch.float8_e4m3fn, device="cuda")
        for _ in range(2)
    ]
    table = torch.zeros(1, total + 4, dtype=torch.int32, device="cuda")
    lengths = torch.zeros(1, dtype=torch.int32, device="cuda")
    _fill(q, pools, table, lengths, length, g)
    return q, pools, table, lengths


def _reference(q, kv, table, length, qk_scale, v_scale):
    """FP64 attention of the 7 verify rows, row t seeing the first length - 7 + t + 1
    tokens."""
    k, v = kv
    pages = table[0, : math.ceil(length / PAGE)].long()
    keys = k[pages, 0].reshape(-1, DIM)[:length].double()
    values = v[pages, 0].reshape(-1, DIM)[:length].double()
    scores = q.double() @ keys.T * qk_scale  # [rows, heads, length]
    tokens = torch.arange(length, device="cuda")
    for t in range(ROWS):
        scores[t, :, tokens > length - ROWS + t] = -math.inf
    return torch.softmax(scores, dim=-1) @ values * v_scale


def _sglang(q, kv, table, lengths, qk_scale, v_scale):
    """SGLang's attention on the same inputs: TRT-LLM MHA's decode kernel at 7 rows."""
    import flashinfer

    workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv,
        workspace_buffer=workspace,
        block_tables=table,
        seq_lens=lengths,
        max_seq_len=table.shape[1] * PAGE,
        bmm1_scale=qk_scale,
        bmm2_scale=v_scale,
        out_dtype=torch.bfloat16,
        q_len_per_req=ROWS,
    )


@pytest.mark.parametrize("length", [7, 64, 2048, 8192, 32769, 131072, 150007])
def test_skv2_is_within_twice_sglangs_error(length):
    from sgl_nvidia_mega.qwen3_5.sm103.attention.skv2 import StockSplitPlan

    g = torch.Generator(device="cuda").manual_seed(length)
    q, pools, table, lengths = _inputs(length, g)
    kv = tuple(map(_hnd, pools))
    qk_scale, v_scale = 0.75 * SCALE, 0.5
    out = torch.empty(ROWS, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    plan = StockSplitPlan(q, kv, table, lengths)
    plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out)
    want = _reference(q, kv, table, length, qk_scale, v_scale)
    assert torch.isfinite(out).all()
    err = (out.double() - want).square().mean().sqrt().item()
    stock = _sglang(q, kv, table, lengths, qk_scale, v_scale)
    stock_err = (stock.double() - want).square().mean().sqrt().item()
    assert err <= 2 * stock_err + 1e-6, (length, err, stock_err)


def test_skv2_replays_in_a_cuda_graph_bit_for_bit():
    from sgl_nvidia_mega.qwen3_5.sm103.attention.skv2 import StockSplitPlan

    g = torch.Generator(device="cuda").manual_seed(1)
    q, pools, table, lengths = _inputs(150007, g)
    kv = tuple(map(_hnd, pools))
    qk_scale, v_scale = 0.75 * SCALE, 0.5
    out = torch.empty(ROWS, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    plan = StockSplitPlan(q, kv, table, lengths)
    plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out)
    for length in (131072, 9000, 7, 150007):
        _fill(q, pools, table, lengths, length, g)
        graph.replay()
        want = torch.empty_like(out)
        plan.run(q, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=want)
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), length
