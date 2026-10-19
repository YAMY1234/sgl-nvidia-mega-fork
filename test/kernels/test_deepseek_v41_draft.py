# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft and its logits on SM100-family GPUs (GB300).

- The draft's low-M LM head: FlashInfer's cute-dsl ``mm_bf16`` against the BF16
  projection SGLang's draft runs, on the TP4 shard (N 32320, K 5120) at the rows a draft
  step projects (5 a request, 1-3 requests, and the 16-row bound). Only the K summation
  order differs, so the bound is ``sampler.LM_HEAD_LOGITS_REL``.
- The sharded Markov chain (sm103/markov.py) draws SGLang's MarkovSampleStep tokens and
  keeps its step logits bit for bit when the bias products sum exactly: one rank and four
  ranks (spawned, symmetric memory), 1 to 64 rows (direct exchange, rank-local fold and
  the wide kernel), greedy, sampled and mixed temperatures, BF16 and FP32 base logits,
  local-shard and full-vocab base logits, three launches and CUDA graph replay with the
  noise forked onto its side stream. Rows at temperature 0 leave their step logits
  untouched. Long graph replays of mixed row counts and launch counters crossing 2**32
  exercise the epoch-tag reuse. With real-valued weights the bias sums in another order
  than SGLang's F.linear, so kept logits are bounded (one BF16 ulp of the step's bias).
  A missing rank's fault is in test_deepseek_v41_fail_stop.py.
- The row argmax from per-rank keys (``common/argmax_keys.py``): four vocab shards'
  keys, maxed over ranks, give exactly SGLang's ids (``_row_argmax`` on the gathered
  fp32 logits, the fused kernel the verify epilogue uses), including ties across shards,
  -0.0 against +0.0, and columns past the vocabulary.
- The megakernel draft proposal through SGLang's own DsparkDraftSampler:
  maybe_build_draft_sampler gets the draft spec's draft_sampler, and a sampler built with
  it draws the same tokens, keeps the same step logits and advances the noise counter as
  one built without it, eagerly and under CUDA graph replay, bit for bit (the Markov
  weights sum exactly in any order).

Run with ``python -m pytest test/kernels/test_deepseek_v41_draft.py -s``.
"""

import os
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.common.argmax_keys import argmax_keys, keys_to_ids
from sgl_nvidia_mega.deepseek_v41 import sampler
from sgl_nvidia_mega.deepseek_v41.sm103 import markov

try:
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        _row_argmax,
    )
except ImportError:
    _row_argmax = None


# ------------------------------------------------------------------------ low-M LM head

needs_lm_head = pytest.mark.skipif(
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


@needs_lm_head
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


@needs_lm_head
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


# ----------------------------------------------------------------- sharded Markov chain

VOCAB = 129_280  # DeepSeek-V4.1
RANK = 256
STEPS = 5
MAX_BS = 8
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = ((0.0,) * 8, (1.0,) * 8, (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1))
# Far above any launch, so only a missing peer reaches it.
DEADLINE_S = 20.0
# Graph replays of the long test: LONG_LAUNCHES launches per replay.
LONG_LAUNCHES = 64
LONG_REPLAYS = 1100
LONG_ROWS = (8, 1, 3, 8, 2)
# The wide kernel (9-64 rows at TP4).
WIDE_BS = 64
WIDE_ROWS = (9, 16, 32, 64)
WIDE_GRAPH_ROWS = (64, 9, 1, 8, 33)
WIDE_REPLAYS = 14_000
# Per-CTA launch counters (markov.cu).
NARROW_COUNT_WORDS = slice(256, None)
WIDE_COUNT_WORDS = slice(1280, None)
ARRIVAL_WORDS = slice(128, 136)


def _markov_ready() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in markov.SUPPORTED_CAPABILITIES
    )


needs_gpu = pytest.mark.skipif(not _markov_ready(), reason="needs an SM100/SM103 GPU")
needs_4_gpus = pytest.mark.skipif(
    not _markov_ready() or torch.cuda.device_count() < 4,
    reason="needs 4 SM100/SM103 GPUs",
)


def _ternary(g: torch.Generator, device, *shape: int) -> torch.Tensor:
    """Sparse {-1, 0, 1} BF16: |bias| <= 256, exact in BF16 in any summation order."""
    values = torch.randint(-1, 2, shape, generator=g)
    values *= torch.rand(shape, generator=g) < 0.25
    return values.to(device, torch.bfloat16)


def _reference(w1, w2, base, anchor, temperatures, offset, cache):
    """SGLang's MarkovSampleStep per step over the full vocab, as the unsharded draft runs it."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    prev, tokens = anchor.long(), []
    for step in range(base.shape[1]):
        bias = F.linear(w1[prev], w2)
        prev = MarkovSampleStep.execute(
            base_logits=base[:, step],
            bias=bias,
            temperatures=temperatures,
            key=KEY,
            offset=offset,
            step=step,
            step_dtype=torch.float32,
            logits_out=None if cache is None else cache[: base.shape[0], step],
            has_sampled_rows=None if cache is not None else False,
        )
        tokens.append(prev)
    return torch.stack(tokens, 1)


def _assert_cache(cache, want_cache, temperatures, rows: int, what) -> None:
    """Rows that keep logits match; temperature-0 rows stay zero."""
    keeps = temperatures[:rows] != 0
    assert torch.equal(cache[:rows][keeps], want_cache[:rows][keeps]), what
    assert not cache[:rows][~keeps].any(), what


def _make(group, rank, world, w1, w2, max_bs=MAX_BS, keeps_logits=True):
    return markov.ShardedMarkov(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=max_bs,
        deadline_s=DEADLINE_S,
        keeps_logits=keeps_logits,
    )


def _check_equal(device, group, rank: int, world: int, base_dtype, seed: int):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(group, rank, world, w1, w2)
    # A greedy draft's chain has no logit staging; it serves greedy blocks only.
    greedy_chain = _make(group, rank, world, w1, w2, keeps_logits=False)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    for rows in (1, 2, 4, 8):
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temps in TEMPERATURE_SETS:
            temperatures = torch.tensor(temps, device=device)
            for keep in (True, False):  # sampled draft vs greedy draft (no logits)
                want_cache = (
                    torch.zeros(MAX_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for sampler in (chain,) if keep else (chain, greedy_chain):
                    for local in (True, False):
                        inputs = base[:, :, shard].contiguous() if local else base
                        for _ in range(3):
                            cache = (
                                torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
                                if keep
                                else None
                            )
                            tokens = torch.full(
                                (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                            )
                            if keep:
                                sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                            sampler.sample(
                                base_logits=inputs,
                                anchor_tokens=anchor,
                                temperatures=temperatures,
                                logits_cache=cache,
                                draft_tokens=tokens,
                            )
                            torch.cuda.synchronize()
                            what = (rows, temps, keep, local, sampler is chain)
                            assert torch.equal(tokens[:rows], want), what
                            if keep:
                                _assert_cache(
                                    cache, want_cache, temperatures, rows, what
                                )

    _check_forked_graph(device, chain, w1, w2, shard, base_dtype, g)
    assert int(chain.control[ARRIVAL_WORDS].abs().sum()) == 0  # arrivals reset


def _check_forked_graph(device, chain, w1, w2, shard, base_dtype, g):
    """CUDA graph: noise forked onto the side stream, replayed with new offsets."""
    rows = 2
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
    anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)
    local = base[:, :, shard].contiguous()

    def forked_sample():
        chain.fork_noise(rows=rows, key=KEY, offset=offset)
        chain.join_noise()
        chain.sample(
            base_logits=local,
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        forked_sample()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forked_sample()
    for step_offset in (5, 6, 1 << 33):
        offset.fill_(step_offset)
        graph.replay()
        want_cache = torch.zeros_like(cache)
        want = _reference(w1, w2, base, anchor, temperatures[:rows], offset, want_cache)
        torch.cuda.synchronize()
        assert torch.equal(tokens[:rows], want), step_offset
        _assert_cache(cache, want_cache, temperatures, rows, step_offset)


def _check_long_replay(device, group, rank: int, world: int, seed: int):
    """Graphs of mixed row counts share the launch counters; > 2 tag periods of
    launches, then a fresh chain whose launch counters cross 2**32."""
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    base = (torch.randn(MAX_BS, STEPS, VOCAB, generator=g) * 3).to(
        device, torch.bfloat16
    )
    local = base[:, :, shard].contiguous()
    anchor = torch.randint(0, VOCAB, (MAX_BS,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([11], dtype=torch.int64, device=device)
    want_cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    want_sampled = _reference(w1, w2, base, anchor, temperatures, offset, want_cache)
    want_greedy = _reference(w1, w2, base, anchor, temperatures, offset, None)

    chain = _make(group, rank, world, w1, w2)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full(
        (LONG_LAUNCHES, MAX_BS, STEPS), -7, dtype=torch.int64, device=device
    )

    def launches():
        for i in range(LONG_LAUNCHES):
            rows, keep = LONG_ROWS[i % len(LONG_ROWS)], i % 2 == 0
            if keep:
                chain.launch_noise(rows=rows, key=KEY, offset=offset)
            chain.sample(
                base_logits=local[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=cache if keep else None,
                draft_tokens=tokens[i],
            )

    def check(what):
        torch.cuda.synchronize()
        for i in range(LONG_LAUNCHES):
            rows, keep = LONG_ROWS[i % len(LONG_ROWS)], i % 2 == 0
            want = want_sampled if keep else want_greedy
            assert torch.equal(tokens[i, :rows], want[:rows]), (what, i)
        tokens.fill_(-7)

    launches()
    check("eager")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launches()
    for replay in range(LONG_REPLAYS):
        graph.replay()
        if replay % 100 == 99:
            check(replay)
    _assert_cache(cache, want_cache, temperatures, MAX_BS, "long")
    del graph

    # A fresh mailbox and counters three launches short of 2**32.
    chain = _make(group, rank, world, w1, w2)
    chain.control[NARROW_COUNT_WORDS] = -3
    for launch in range(6):
        for rows in (1, MAX_BS):  # the direct exchange and the fold
            out = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)
            cache.zero_()
            chain.launch_noise(rows=rows, key=KEY, offset=offset)
            chain.sample(
                base_logits=local[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=cache,
                draft_tokens=out,
            )
            torch.cuda.synchronize()
            assert torch.equal(out[:rows], want_sampled[:rows]), ("wrap", launch, rows)
            _assert_cache(cache, want_cache, temperatures, rows, ("wrap", launch))


def _check_wide(device, group, rank: int, world: int, seed: int):
    """9-64 rows on the wide kernel: tokens and kept logits, a graph replay of mixed
    row counts across all three kernels, and a wide counter wrap."""
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(group, rank, world, w1, w2, max_bs=WIDE_BS)
    assert chain.wide_rows == WIDE_BS and chain.max_rows == WIDE_BS
    greedy_chain = _make(group, rank, world, w1, w2, max_bs=WIDE_BS, keeps_logits=False)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([5], dtype=torch.int64, device=device)
    mixed = torch.tensor(TEMPERATURE_SETS[2] * (WIDE_BS // 8), device=device)
    for rows in WIDE_ROWS:
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(
            device, torch.bfloat16
        )
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temperatures in (torch.zeros(WIDE_BS, device=device), mixed):
            for keep in (True, False):
                want_cache = (
                    torch.zeros(WIDE_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for sampler in (chain,) if keep else (chain, greedy_chain):
                    for local in (True, False):
                        inputs = base[:, :, shard].contiguous() if local else base
                        for _ in range(3):
                            cache = (
                                torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
                                if keep
                                else None
                            )
                            tokens = torch.full(
                                (WIDE_BS, STEPS), -7, dtype=torch.int64, device=device
                            )
                            if keep:
                                sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                            sampler.sample(
                                base_logits=inputs,
                                anchor_tokens=anchor,
                                temperatures=temperatures,
                                logits_cache=cache,
                                draft_tokens=tokens,
                            )
                            torch.cuda.synchronize()
                            what = ("wide", rows, keep, local, bool(temperatures.any()))
                            assert torch.equal(tokens[:rows], want), what
                            if keep:
                                _assert_cache(
                                    cache, want_cache, temperatures, rows, what
                                )

    # One graph per row count; replays interleave the direct, narrow and wide kernels.
    base = (torch.randn(WIDE_BS, STEPS, VOCAB, generator=g) * 3).to(
        device, torch.bfloat16
    )
    local = base[:, :, shard].contiguous()
    anchor = torch.randint(0, VOCAB, (WIDE_BS,), generator=g).to(device)
    want_cache = torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
    want = _reference(w1, w2, base, anchor, mixed, offset, want_cache)
    cache = torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
    outs = [
        torch.full((rows, STEPS), -7, dtype=torch.int64, device=device)
        for rows in WIDE_GRAPH_ROWS
    ]

    def launch(rows, out):
        chain.launch_noise(rows=rows, key=KEY, offset=offset)
        chain.sample(
            base_logits=local[:rows],
            anchor_tokens=anchor[:rows],
            temperatures=mixed,
            logits_cache=cache,
            draft_tokens=out,
        )

    graphs = []
    for rows, out in zip(WIDE_GRAPH_ROWS, outs):
        launch(rows, out)  # warm-up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(rows, out)
        graphs.append(graph)
    for replay in range(WIDE_REPLAYS):
        for graph in graphs:
            graph.replay()
        if replay % 2000 == 1999:
            torch.cuda.synchronize()
            for rows, out in zip(WIDE_GRAPH_ROWS, outs):
                assert torch.equal(out, want[:rows]), (replay, rows)
                out.fill_(-7)
    _assert_cache(cache, want_cache, mixed, WIDE_BS, "wide graph")
    del graphs

    # A fresh chain whose wide counters are three launches short of 2**32.
    chain = _make(group, rank, world, w1, w2, max_bs=WIDE_BS)
    chain.wide_control[WIDE_COUNT_WORDS] = -3
    for launch_index in range(6):
        out = torch.full((WIDE_BS, STEPS), -7, dtype=torch.int64, device=device)
        cache.zero_()
        launch(WIDE_BS, out)
        torch.cuda.synchronize()
        assert torch.equal(out, want), ("wide wrap", launch_index)
        _assert_cache(cache, want_cache, mixed, WIDE_BS, ("wide wrap", launch_index))


def test_unsupported_reason():
    def reason(**kw):
        args = dict(
            world=4,
            vocab_size=VOCAB,
            num_steps=STEPS,
            embed_weight=torch.empty(VOCAB, RANK, dtype=torch.bfloat16),
            head_weight=torch.empty(VOCAB, RANK, dtype=torch.bfloat16),
            cache_dtype=torch.float32,
            capability=(10, 3),
        )
        args.update(kw)
        return markov.unsupported_reason(**args)

    assert reason() is None
    assert reason(capability=(10, 0)) is None
    assert "tensor-parallel" in reason(world=3)
    assert "capability" in reason(capability=(9, 0))
    assert "draft positions" in reason(num_steps=9)
    assert "vocabulary" in reason(vocab_size=VOCAB + 16)
    assert "step-logits" in reason(cache_dtype=torch.float16)
    assert "embedding" in reason(embed_weight=torch.empty(VOCAB, 128))
    assert markov.narrow_row_limit(world=4, max_bs=64) == 8
    assert markov.narrow_row_limit(world=8, max_bs=64) == 4
    assert markov.narrow_row_limit(world=1, max_bs=2) == 2


def _nvrtc_available() -> bool:
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _nvrtc_available(), reason="needs NVRTC (cuda.bindings)")
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_every_variant_compiles(arch, monkeypatch):
    if not torch.cuda.is_available():  # a CPU node: no stream can be capturing
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for steps in (1, STEPS, markov.MAX_STEPS):
        for base_dtype in markov.LOGIT_DTYPES:
            for cache_dtype in markov.LOGIT_DTYPES:
                for direct in (True, False):
                    cubin = markov.compile_cubin(
                        steps=steps,
                        base_dtype=base_dtype,
                        cache_dtype=cache_dtype,
                        direct=direct,
                        arch=arch,
                    )
                    assert markov.NARROW_SYMBOL.encode() in cubin
                    assert (markov.WIDE_SYMBOL.encode() in cubin) == (not direct)


@needs_gpu
@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_one_rank_matches_markov_sample_step(base_dtype):
    _check_equal(torch.device("cuda"), None, 0, 1, base_dtype, seed=1)


@needs_gpu
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_real_weights_bound_the_kept_logits(rows):
    """With real-valued weights the kernel's MMA bias and SGLang's F.linear bias sum in
    different orders, so a BF16-rounded bias may differ by one ulp: each kept logit is
    within 2**-7 of the step's largest |bias| of SGLang's, until the row's first token
    difference."""
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(rows)
    w1, w2 = (
        (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
        for _ in range(2)
    )
    chain = _make(None, 0, 1, w1, w2)
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
    temperatures = torch.ones(rows, device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    want_cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    want = _reference(w1, w2, base, anchor, temperatures, offset, want_cache)
    cache = torch.zeros_like(want_cache)
    tokens = torch.empty(MAX_BS, STEPS, dtype=torch.int64, device=device)
    chain.launch_noise(rows=rows, key=KEY, offset=offset)
    chain.sample(
        base_logits=base,
        anchor_tokens=anchor,
        temperatures=temperatures,
        logits_cache=cache,
        draft_tokens=tokens,
    )
    torch.cuda.synchronize()
    differs = tokens[:rows] != want
    # a step's logits are comparable up to the row's first token difference
    first = torch.where(differs.any(1), differs.int().argmax(1), STEPS - 1)
    comparable = torch.arange(STEPS, device=device)[None] <= first[:, None]
    # the bias's scale in each (row, step): near-zero biases cancel, so a per-element
    # relative bound does not hold
    scale = (want_cache[:rows] - base.float()).abs().amax(-1)
    delta = (cache[:rows] - want_cache[:rows]).abs().amax(-1)
    ratio = (delta / scale)[comparable]
    print(
        f"rows {rows}: token differences {int(differs.sum())}, max |logit difference| "
        f"/ max |bias| {float(ratio.max()):.3g}"
    )
    # one BF16 ulp: two roundings of nearly equal FP32 sums differ by at most that
    assert bool((ratio <= 2**-7).all())


@needs_gpu
def test_one_rank_long_replay_and_counter_wrap():
    _check_long_replay(torch.device("cuda"), None, 0, 1, seed=3)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _four_rank_worker(rank: int, world: int, port: int, mode: str) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    try:
        if mode == "long":
            _check_long_replay(device, group, rank, world, seed=9)
        elif mode == "wide":
            _check_wide(device, group, rank, world, seed=11)
        else:
            for base_dtype in (torch.bfloat16, torch.float32):
                _check_equal(device, group, rank, world, base_dtype, seed=7)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@needs_4_gpus
@pytest.mark.parametrize("mode", ["equal", "long", "wide"])
def test_four_ranks_match_markov_sample_step(mode):
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port(), mode), nprocs=4, join=True
    )


# -------------------------------------------------------------------------- argmax keys

TP = 4

needs_argmax = pytest.mark.skipif(
    not torch.cuda.is_available() or _row_argmax is None,
    reason="needs CUDA and SGLang's DSpark accept kernels",
)


def _ids(logits, *, vocab=VOCAB, dtype=torch.float32):
    width = logits.shape[1] // TP
    keys = torch.stack(
        [
            argmax_keys(
                logits[:, r * width : (r + 1) * width].to(dtype).contiguous(),
                vocab_offset=r * width,
                vocab=vocab,
            )
            for r in range(TP)
        ]
    )
    return keys_to_ids(keys)


def _logits(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, VOCAB, device="cuda", generator=g) * 4


@needs_argmax
@pytest.mark.parametrize("rows", [6 * b for b in range(1, 9)])
def test_random_rows_are_sglangs_ids(rows):
    logits = _logits(rows, rows)
    assert torch.equal(_ids(logits), _row_argmax(logits, fused=True))


@needs_argmax
def test_ties_across_shards_take_the_lowest_global_index():
    logits = _logits(6, 1)
    top = logits.max() + 1
    # row r: equal maxima in shards (r % 4) and 3, the lower index must win
    for r in range(6):
        lo = (r % 3) * (VOCAB // TP) + 17 * r
        logits[r, lo] = top
        logits[r, 3 * (VOCAB // TP) + 5] = top
        logits[r, lo + 1] = top  # a tie inside the shard too
    want = _row_argmax(logits, fused=True)
    assert torch.equal(_ids(logits), want)
    assert want.tolist() == [(r % 3) * (VOCAB // TP) + 17 * r for r in range(6)]


@needs_argmax
def test_minus_zero_equals_plus_zero():
    logits = -torch.rand(6, VOCAB, device="cuda") - 1
    logits[:, 100] = -0.0  # lower index, -0.0
    logits[:, 40000] = 0.0  # higher index, +0.0: equal value, the lower index wins
    want = _row_argmax(logits, fused=True)
    assert want.tolist() == [100] * 6
    assert torch.equal(_ids(logits), want)


@needs_argmax
def test_columns_past_the_vocabulary_never_win():
    logits = _logits(6, 2)
    logits[:, -3:] = 1e9  # padding columns of the last shard
    want = _row_argmax(logits[:, : VOCAB - 3].contiguous(), fused=True)
    assert torch.equal(_ids(logits, vocab=VOCAB - 3), want)


@needs_argmax
def test_bf16_shards_give_the_ids_of_their_fp32_gather():
    logits = _logits(12, 3).bfloat16().float()
    assert torch.equal(
        _ids(logits, dtype=torch.bfloat16), _row_argmax(logits, fused=True)
    )


@needs_argmax
def test_graph_replay_into_the_exchange_pairs_reads_new_logits():
    """Keys land in a column of the (key, 0) pairs the exchange gathers (row stride 2),
    as the verify tail's fork writes them, and a replay reads the new logits."""
    logits = _logits(6, 4)
    pairs = torch.zeros(TP, 6, 2, dtype=torch.int64, device="cuda")
    width = VOCAB // TP
    shards = [logits[:, r * width : (r + 1) * width] for r in range(TP)]

    def run():
        for r in range(TP):
            argmax_keys(
                shards[r], vocab_offset=r * width, vocab=VOCAB, out=pairs[r, :, 0]
            )

    run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    logits.copy_(_logits(6, 5))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(keys_to_ids(pairs[..., 0]), _row_argmax(logits, fused=True))
    assert not pairs[..., 1].any()


# ----------------------------------------------------------------------- draft proposal

HIDDEN, GAMMA = 64, 5  # VOCAB, RANK and MAX_BS: the Markov chain's
PROPOSAL_KEY = 0x2468_ACE0_1357_9BDF
TEMPERATURES = (0.7, 0.0, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9)

needs_proposal = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def _proposal_ternary(g, *shape):
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
    head.markov_w1.weight.data.copy_(_proposal_ternary(g, VOCAB, RANK))
    head.markov_w2.weight.data.copy_(_proposal_ternary(g, VOCAB, RANK))
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
    # the sampler agrees its batch sizes over the DP-TP group
    patch.setattr(ds, "get_dp_tp_group", lambda: one_rank)
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
                key=PROPOSAL_KEY,
                offset=torch.zeros(1, dtype=torch.int64, device="cuda"),
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


@needs_proposal
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


@needs_proposal
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
