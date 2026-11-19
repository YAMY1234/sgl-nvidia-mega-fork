# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gumbel_fast: sharded_gumbel's four-draws-per-Philox keys. One GPU."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from sglang.srt.speculative.sharded_verify import VocabShard, shard_keys  # noqa: E402

from sgl_nvidia_mega.qwen3_5.sm103.sharded_gumbel import (  # noqa: E402
    gumbel_shard_keys_fast,
)

VOCAB, WORLD = 1024, 8
LOCAL = VOCAB // WORLD


def _tokens(keys: torch.Tensor) -> torch.Tensor:
    if keys.dim() == 2:
        keys = keys.amax(dim=-1)
    return 0xFFFFFFFF - (keys & 0xFFFFFFFF)


@pytest.mark.parametrize("start, splits", [(0, 4), (3 * LOCAL, 2), (7 * LOCAL, 4)])
def test_plain_keys_match_the_runtime_greedy_keys(start, splits):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(6, LOCAL, generator=g) * 3).bfloat16()
    x[0, 5] = x[0, 77] = x[0].float().max() + 1  # tie
    x[1, :] = float("nan")
    x[1, 9] = -1.0
    x[2, :] = -0.0
    x[2, 4] = 0.0
    x[3, :] = float("-inf")
    x = x.cuda()
    temps = torch.tensor([1.0, 0.7, 0.0, 1.0, -1.0, 2.0], device="cuda")
    seeds = torch.arange(6, dtype=torch.int64, device="cuda") * 977 + 5
    greedy = torch.empty(6, splits, dtype=torch.int64, device="cuda")
    keys = gumbel_shard_keys_fast(
        x,
        temps,
        seeds,
        vocab_start=start,
        valid=LOCAL,
        vocab_size=VOCAB,
        splits=splits,
        greedy_out=greedy,
    )
    want = shard_keys(x, VocabShard(start=start, valid=LOCAL, vocab_size=VOCAB), splits)
    assert torch.equal(greedy, want)
    # T <= 0 rows keep the plain key
    for row in (2, 4):
        assert torch.equal(keys[row], want[row]), row


def test_winners_do_not_depend_on_the_sharding():
    g = torch.Generator().manual_seed(2)
    rows = 64
    full = (torch.randn(rows, VOCAB, generator=g) * 2).bfloat16().cuda()
    temps = torch.full((rows,), 1.0, device="cuda")
    seeds = torch.randint(0, 2**31 - 1, (rows,), generator=g, dtype=torch.int64).cuda()
    whole = gumbel_shard_keys_fast(
        full, temps, seeds, vocab_start=0, valid=VOCAB, vocab_size=VOCAB, splits=8
    )
    parts = [
        gumbel_shard_keys_fast(
            full[:, w * LOCAL : (w + 1) * LOCAL].contiguous(),
            temps,
            seeds,
            vocab_start=w * LOCAL,
            valid=LOCAL,
            vocab_size=VOCAB,
            splits=2,
        )
        for w in range(WORLD)
    ]
    assert torch.equal(_tokens(whole), _tokens(torch.cat(parts, dim=1)))


@pytest.mark.parametrize("temperature", [1.0, 0.8])
def test_samples_follow_the_softmax_distribution(temperature):
    from scipy.stats import chi2

    samples, ids = 40_000, 16
    row = torch.linspace(-2.0, 1.0, ids).bfloat16()
    logits = row.repeat(samples, 1).cuda()
    g = torch.Generator().manual_seed(3)
    seeds = torch.randint(0, 2**31 - 1, (samples,), generator=g, dtype=torch.int64)
    temps = torch.full((samples,), temperature, device="cuda")
    picks = _tokens(
        gumbel_shard_keys_fast(
            logits,
            temps,
            seeds.cuda(),
            vocab_start=0,
            valid=ids,
            vocab_size=ids,
            splits=1,
        )
    )
    observed = torch.bincount(picks.cpu(), minlength=ids).double()
    expected = torch.softmax(row.double() / temperature, dim=0) * samples
    statistic = ((observed - expected).square() / expected).sum().item()
    p_value = chi2.sf(statistic, ids - 1)
    assert p_value >= 1e-4, f"chi-square {statistic:.2f}, p-value {p_value:.3g}"
