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
    lm = SimpleNamespace(
        stages=[SimpleNamespace(norm=lambda x: x * 2)], collapse_hc_head=lambda x: x
    )
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


def test_a_rank_that_cannot_run_the_chain_refuses_on_every_rank():
    """The slot was asked for, so an unsupported setup refuses at startup instead of
    falling back to SGLang's proposal; every rank's reason is listed."""
    assert sampler.refusal_message(["", ""]) == ""
    message = sampler.refusal_message(["", "an FP32 markov_w2", "", "a TP size"])
    assert "rank 1: an FP32 markov_w2" in message and "rank 3: a TP size" in message
    assert "rank 0" not in message


def test_ranks_that_built_different_proposals_refuse():
    """Ranks serving different batch sizes would capture graphs that wait on each other."""
    assert sampler.mismatch_message([(64, True)] * 4) == ""
    message = sampler.mismatch_message([(64, True), (8, True), (64, True), (64, False)])
    assert (
        "rank 1: batch 1-8" in message
        and "rank 3: batch 1-64, low-M LM head False" in message
    )


def test_buffers_the_chain_cannot_write_into_are_refused():
    """The chain writes int64 draft tokens and reads float32 step temperatures in place;
    another SGLang layout must refuse at startup, not corrupt the proposal."""

    def reason(tokens, temperatures):
        return sampler.buffer_reason(
            max_bs=4, gamma=GAMMA, draft_tokens=tokens, step_temperatures=temperatures
        )

    assert reason(torch.zeros(4 * GAMMA, dtype=torch.int64), torch.zeros(4)) == ""
    narrow = reason(torch.zeros(4 * GAMMA, dtype=torch.int32), torch.zeros(4))
    assert "draft tokens" in narrow
    half = reason(torch.zeros(4 * GAMMA, dtype=torch.int64), torch.zeros(4).half())
    assert "step temperatures" in half


@pytest.mark.parametrize("slots", [frozenset(), frozenset({"draft_lm_head"})])
def test_without_the_markov_slot_the_draft_keeps_sglang_s_proposal(slots):
    from sgl_nvidia_mega.deepseek_v41 import draft

    built = draft.DraftProposal(SimpleNamespace(), deadline_s=30.0, slots=slots)
    assert built.draft_sampler is None
    # the draft's forward stays SGLang's at every step
    assert built.widths == () and built.plan(None, None) is None


def test_the_draft_sampler_builds_the_chain_over_sglang_s_buffers(monkeypatch):
    """SGLang calls the factory once with its own buffers; they reach the chain as
    given, with the spec's deadline and LM-head choice."""
    from sgl_nvidia_mega.deepseek_v41 import draft

    calls = []
    monkeypatch.setattr(
        sampler, "build_steps", lambda lm, **kw: calls.append((lm, kw)) or {2: "step"}
    )
    lm = SimpleNamespace()
    built = draft.DraftProposal(
        lm, deadline_s=30.0, slots=frozenset({"markov", "draft_lm_head"})
    )
    buffers = dict(
        gamma=GAMMA,
        max_bs=4,
        greedy_draft=False,
        noise="noise",
        draft_tokens="tokens",
        step_logits="logits",
        step_temperatures="temperatures",
    )
    steps = built.draft_sampler(**buffers)
    assert calls == [(lm, {"low_m": True, "deadline_s": 30.0, **buffers})]
    assert steps(2) == "step" and steps(3) is None


def test_the_draft_spec_follows_the_target_spec():
    """The draft shares the target's GPUs and contract version; a drifted copy would
    accept a setup the target refuses, or pass an API check the target fails."""
    from sgl_nvidia_mega.deepseek_v41 import draft, spec

    assert draft.SPEC.api_version == spec.API_VERSION
    for field in ("isas", "tp_sizes", "dtypes", "decode_graph_backends"):
        assert getattr(draft.SPEC.requires, field) == getattr(spec.SPEC.requires, field)
    assert draft.SPEC.max_rows == 0  # the draft's forward stays SGLang's
