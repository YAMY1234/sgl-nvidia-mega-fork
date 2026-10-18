# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel draft proposal through SGLang's own DsparkDraftSampler, one SM100-family
GPU: maybe_build_draft_sampler gets the draft spec's draft_sampler (the framework's
megakernel_draft_sampler), and a sampler built with it draws the same tokens, keeps the
same step logits and advances the noise counter as one built without it (SGLang's
proposal), eagerly and under CUDA graph replay. The Markov weights sum exactly in any
order, so the two must be bit for bit equal.
"""

from types import SimpleNamespace

import pytest
import torch

VOCAB, RANK, HIDDEN, GAMMA, MAX_BS = 129_280, 256, 64, 5, 8
KEY = 0x2468_ACE0_1357_9BDF
TEMPERATURES = (0.7, 0.0, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def _ternary(g, *shape):
    """Sparse {-1, 0, 1} BF16: the bias sums exactly in any order."""
    values = torch.randint(-1, 2, shape, generator=g) * (
        torch.rand(shape, generator=g) < 0.25
    )
    return values.to("cuda", torch.bfloat16)


def _draft_model():
    """The fields SGLang's sampler and the hooks read, on a vanilla BF16 Markov head
    whose step logits are FP32, as DeepSeek-V4.1's are."""
    from sglang.srt.models.dspark import VanillaMarkov, project_through_lm_head

    class Head(VanillaMarkov):
        def step_logits_dtype(self, base_dtype):
            return torch.float32

    g = torch.Generator().manual_seed(0)
    head = Head(vocab_size=VOCAB, markov_rank=RANK).to("cuda", torch.bfloat16)
    head.markov_w1.weight.data.copy_(_ternary(g, VOCAB, RANK))
    head.markov_w2.weight.data.copy_(_ternary(g, VOCAB, RANK))
    lm_head = SimpleNamespace(
        weight=(torch.randn(VOCAB, HIDDEN, generator=g) * 0.05).to(
            "cuda", torch.bfloat16
        ),
        quant_method=None,
        org_vocab_size=VOCAB,
        num_embeddings_per_partition=VOCAB,
        shard_indices=SimpleNamespace(org_vocab_start_index=0),
    )
    norm = SimpleNamespace(norm=lambda x: x * 0.5)
    return SimpleNamespace(
        markov_head=head,
        sample_from_anchor=True,
        confidence_head=None,
        lm_head=lm_head,
        _use_fp32_lm_head=False,
        stages=[norm],
        collapse_hc_head=lambda x: x,
        compute_base_logits=lambda h: (
            project_through_lm_head(norm.norm(h), lm_head),
            h,
        ),
    )


@pytest.fixture(scope="module")
def samplers():
    import sglang.srt.runtime_context as context
    from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise

    from sgl_nvidia_mega.deepseek_v41 import draft

    patch = pytest.MonkeyPatch()
    one_rank = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=None)
    patch.setattr(context, "get_parallel", lambda: SimpleNamespace(tp_group=one_rank))
    model = _draft_model()
    built = draft.DraftProposal(model, deadline_s=30.0, slots=frozenset({"markov"}))

    def build(factory):
        # the framework's lookup of the draft spec's built object
        patch.setattr(ds, "megakernel_draft_sampler", lambda draft_model: factory)
        return ds.maybe_build_draft_sampler(
            draft_model=model,
            gamma=GAMMA,
            max_bs=MAX_BS,
            device=torch.device("cuda"),
            tp_rank=0,
            noise=DraftNoise(
                key=KEY, offset=torch.zeros(1, dtype=torch.int64, device="cuda")
            ),
            greedy_draft=False,
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
        )

    stock, mega = build(None), build(built.draft_sampler)
    assert stock._megakernel_steps is None and mega._megakernel_steps is not None
    yield stock, mega
    patch.undo()


def _inputs(bs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(bs * GAMMA, HIDDEN, generator=g).to("cuda", torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (bs * GAMMA,), generator=g).to("cuda")
    return hidden, input_ids


def _stage(samplers, bs: int) -> torch.Tensor:
    temperatures = torch.tensor(TEMPERATURES[:bs], device="cuda")
    for s in samplers:
        s.params.step_temperatures[:bs].copy_(temperatures)
    return temperatures


def _assert_same(stock, mega, bs: int, temperatures, what) -> None:
    rows = bs * GAMMA
    assert torch.equal(mega.out[:rows], stock.out[:rows]), what
    kept = temperatures > 0
    assert torch.equal(mega.step_logits[:bs][kept], stock.step_logits[:bs][kept]), what
    assert int(mega._noise.offset) == int(stock._noise.offset), what


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
def test_the_hooked_proposal_is_sglang_s_proposal(samplers, bs):
    stock, mega = samplers
    temperatures = _stage(samplers, bs)
    for launch in range(2):
        hidden, input_ids = _inputs(bs, seed=10 * bs + launch)
        stock(hidden, input_ids)
        mega(hidden, input_ids)
        torch.cuda.synchronize()
        _assert_same(stock, mega, bs, temperatures, (bs, launch))


def test_the_hooked_proposal_replays_in_a_cuda_graph(samplers):
    stock, mega = samplers
    bs = 4
    temperatures = _stage(samplers, bs)
    hidden, input_ids = _inputs(bs, seed=99)
    mega(hidden, input_ids)  # SGLang's runner warms up eagerly before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega(hidden, input_ids)
    for replay in range(3):
        new_hidden, new_ids = _inputs(bs, seed=100 + replay)
        hidden.copy_(new_hidden)
        input_ids.copy_(new_ids)
        # both draw at the same noise offset
        stock._noise.offset.copy_(mega._noise.offset)
        graph.replay()
        stock(hidden, input_ids)
        torch.cuda.synchronize()
        _assert_same(stock, mega, bs, temperatures, replay)
