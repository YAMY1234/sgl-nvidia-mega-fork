# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, VOCAB, EPS = 4096, 4099, 1e-6


def _reference(table, ids, hidden, w_emb, w_hid):
    valid = (ids >= 0) & (ids < table.shape[0])
    embeds = torch.zeros(
        ids.numel(), table.shape[1], dtype=torch.float64, device="cuda"
    )
    embeds[valid] = table[ids[valid]].double()

    def norm(x, w):
        x = x.double()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.double())

    return torch.cat([norm(embeds, w_emb), norm(hidden, w_hid)], -1)


def _inputs(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").bfloat16()
    hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    w_emb = (0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).bfloat16()
    w_hid = (0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).bfloat16()
    ids = torch.randint(0, VOCAB, (rows, 3), generator=g, device="cuda")[:, 1]
    if rows > 2:
        ids[0], ids[1] = -1, VOCAB
    return table, ids, hidden, w_emb, w_hid


@cuda
@pytest.mark.parametrize("rows", [1, 4, 16, 112])
def test_the_fused_input_is_within_one_rounding_of_fp64(rows):
    from sgl_nvidia_mega.qwen3_5.draft.fc_input import mtp_fused_input

    table, ids, hidden, w_emb, w_hid = _inputs(rows, rows)
    got = mtp_fused_input(ids, table, hidden, w_emb, w_hid, EPS, EPS)
    want = _reference(table, ids, hidden, w_emb, w_hid)
    assert got.shape == (rows, 2 * HIDDEN) and got.dtype == torch.bfloat16
    err = (got.double() - want).abs()
    assert bool((err <= want.abs() * 2**-8 + 1e-6).all())
    if rows > 2:
        assert not got[:2, :HIDDEN].any()


@cuda
@pytest.mark.parametrize("world", [2, 4, 8])
def test_every_chunk_is_its_slice_of_the_full_concat(world):
    from sgl_nvidia_mega.qwen3_5.draft.fc_input import mtp_fused_input

    table, ids, hidden, w_emb, w_hid = _inputs(7, world)
    full = mtp_fused_input(ids, table, hidden, w_emb, w_hid, EPS, EPS)
    chunks = [
        mtp_fused_input(ids, table, hidden, w_emb, w_hid, EPS, EPS, world, rank)
        for rank in range(world)
    ]
    assert torch.equal(torch.cat(chunks, -1).view(torch.int16), full.view(torch.int16))
