# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TOKENS_PER_REQUEST = 7


def _source(rows, hidden, seed, *, pad=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    full = torch.randn(rows, hidden + pad, generator=g, device="cuda").bfloat16()
    bits = full.view(torch.int16)
    special = torch.tensor(
        [-32768, 0x7F80, -128, 0x7FC1], dtype=torch.int16, device="cuda"
    )
    bits[:, :4] = special
    bits[:, hidden - 4 : hidden] = special
    return full[:, :hidden]


def _index(requests, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    last = torch.arange(requests, device="cuda") * TOKENS_PER_REQUEST
    jitter = torch.randint(
        0, TOKENS_PER_REQUEST, (requests,), generator=g, device="cuda"
    )
    return (last + jitter).clamp(max=rows - 1)


def _same(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@cuda
@pytest.mark.parametrize("requests", [1, 2, 7, 16, 52, 128])
@pytest.mark.parametrize("hidden, pad", [(4096, 0), (4096, 64), (1000, 0)])
def test_copy_rows_is_index_select(requests, hidden, pad):
    from sgl_nvidia_mega.qwen3_5.sm103.draft.rows import copy_rows

    rows = requests * TOKENS_PER_REQUEST
    src = _source(rows, hidden, requests + hidden, pad=pad)
    index = _index(requests, rows, requests)
    want = torch.index_select(src, 0, index)
    out = torch.empty_like(want)
    assert copy_rows(src, index, out) is out
    _same(out, want)


@cuda
def test_copy_rows_replays_from_a_cuda_graph():
    from sgl_nvidia_mega.qwen3_5.sm103.draft.rows import copy_rows

    requests, hidden = 16, 4096
    rows = requests * TOKENS_PER_REQUEST
    src = _source(rows, hidden, 1)
    index = _index(requests, rows, 1)
    out = torch.empty(requests, hidden, dtype=torch.bfloat16, device="cuda")
    copy_rows(src, index, out)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_rows(src, index, out)
    for seed in (2, 3):
        src.copy_(_source(rows, hidden, seed))
        index.copy_(_index(requests, rows, seed))
        graph.replay()
        torch.cuda.synchronize()
        _same(out, torch.index_select(src, 0, index))
