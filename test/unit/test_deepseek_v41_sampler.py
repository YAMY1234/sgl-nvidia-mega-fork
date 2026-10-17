# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft sampler's step on CPU: what it hands the chain and SGLang's RNG contract."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import sampler

BS, GAMMA, VOCAB, HIDDEN = 2, 5, 12, 8


class FakeChain:
    def __init__(self):
        self.calls = []

    def fork_noise(self, *, rows, key, offset):
        self.calls.append(("fork", rows, key))

    def join_noise(self):
        self.calls.append(("join",))

    def sample(
        self, *, base_logits, anchor_tokens, temperatures, logits_cache, draft_tokens
    ):
        self.calls.append(("sample", tuple(base_logits.shape), logits_cache))
        draft_tokens.fill_(7)


def _step(*, greedy_draft):
    weight = torch.randn(VOCAB, HIDDEN)
    lm = SimpleNamespace(stages=[SimpleNamespace(norm=lambda x: x * 2)])
    chain = FakeChain()
    noise = SimpleNamespace(key=9, offset=torch.zeros(1, dtype=torch.int64))
    step_logits = None if greedy_draft else torch.zeros(4, GAMMA, VOCAB)
    step = sampler._Step(
        lm=lm,
        chain=chain,
        lm_head=lambda x: x @ weight.T,
        bs=BS,
        gamma=GAMMA,
        greedy_draft=greedy_draft,
        noise=noise,
        draft_tokens=torch.zeros(4 * GAMMA, dtype=torch.int64),
        step_logits=step_logits,
        step_temperatures=torch.zeros(4),
    )
    return step, chain, noise, step_logits


@pytest.mark.parametrize("greedy_draft", [False, True])
def test_a_block_draws_noise_and_advances_the_counter_as_sglang_does(
    greedy_draft, monkeypatch
):
    """draft_markov_block advances noise.offset once per sampled block and never for a
    greedy draft; a different count would desynchronize the accept's noise."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    step, chain, noise, step_logits = _step(greedy_draft=greedy_draft)
    hidden = torch.randn(BS * GAMMA, HIDDEN)
    tokens, features = step(hidden, torch.arange(BS))
    assert tokens.shape == (BS, GAMMA) and bool(tokens.eq(7).all())
    # the confidence head reads the collapsed hidden states, as compute_base_logits returns
    assert features is hidden
    sample = ("sample", (BS, GAMMA, VOCAB), step_logits)
    if greedy_draft:
        assert chain.calls == [sample] and int(noise.offset) == 0
    else:
        assert chain.calls == [("fork", BS, 9), ("join",), sample]
        assert int(noise.offset) == 1
