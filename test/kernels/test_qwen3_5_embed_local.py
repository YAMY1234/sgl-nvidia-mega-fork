# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _table(vocab, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.randn(vocab, hidden, generator=g, device="cuda").bfloat16()
    bits = table.view(torch.int16)
    # -0.0, +inf, -inf and the smallest subnormal at both ends of every row
    special = torch.tensor(
        [-32768, 0x7F80, -128, 0x0001], dtype=torch.int16, device="cuda"
    )
    bits[:, :4] = special
    bits[:, hidden - 4 :] = special
    return table


def _lookup(table, ids):
    """The vocabulary-parallel lookup's rows: +0.0 plus the owner's row in fp32."""
    valid = (ids >= 0) & (ids < table.shape[0])
    rows = torch.zeros(ids.numel(), table.shape[1], device="cuda")
    rows[valid] += table[ids[valid]].float()
    return rows.bfloat16()


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@cuda
@pytest.mark.parametrize("hidden", [4096, 1000])
@pytest.mark.parametrize("rows", [1, 7, 64, 896])
def test_the_gather_equals_the_vocabulary_parallel_lookup(hidden, rows):
    from sgl_nvidia_mega.qwen3_5.sm103.draft.embed_local import gather_rows

    vocab = 4099
    table = _table(vocab, hidden, rows)
    g = torch.Generator(device="cuda").manual_seed(rows + hidden)
    ids = torch.randint(0, vocab, (rows,), generator=g, device="cuda")
    ids[0] = vocab - 1
    if rows > 2:
        ids[1], ids[2] = -1, vocab
    assert _same(gather_rows(table, ids), _lookup(table, ids))


@cuda
def test_the_gather_reads_ids_at_their_stride():
    from sgl_nvidia_mega.qwen3_5.sm103.draft.embed_local import gather_rows

    table = _table(512, 256, 0)
    tokens = torch.randint(0, 512, (16, 3), device="cuda")
    ids = tokens[:, 1]
    assert ids.stride(0) == 3
    assert _same(gather_rows(table, ids), _lookup(table, ids.contiguous()))


@cuda
def test_the_gather_replays_in_a_cuda_graph():
    from sgl_nvidia_mega.qwen3_5.sm103.draft.embed_local import gather_rows

    table = _table(1024, 4096, 1)
    ids = torch.randint(0, 1024, (8,), device="cuda")
    out = torch.empty(8, 4096, dtype=torch.bfloat16, device="cuda")
    gather_rows(table, ids, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_rows(table, ids, out)
    for seed in range(3):
        g = torch.Generator(device="cuda").manual_seed(seed)
        ids.copy_(torch.randint(0, 1024, (8,), generator=g, device="cuda"))
        graph.replay()
        assert _same(out, _lookup(table, ids))
