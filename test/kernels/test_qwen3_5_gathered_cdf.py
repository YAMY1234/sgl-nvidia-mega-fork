# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging

import pytest
import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.qwen3_5.sm103 import gathered_cdf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

logger = logging.getLogger(__name__)

_BLOCK = gathered_cdf._BLOCK
_RAND_SCALE = 4.6566127342e-10
_TOP_GRID = 2.0**-30
_TOP_REL = 2.0**-20
_TAIL_REL_L2_MAX = 2.0**-14


def _chi_square_p_value(counts, expected):
    from scipy.stats import chi2

    large = expected >= 5
    observed_bins = torch.cat([counts[large], counts[~large].sum().view(1)])
    expected_bins = torch.cat([expected[large], expected[~large].sum().view(1)])
    statistic = (
        (observed_bins - expected_bins).square() / expected_bins.clamp_min(1e-12)
    ).sum()
    return chi2.sf(statistic.item(), observed_bins.numel() - 1)


def _draw_counts(logits, temperature, samples):
    # every sample the same distribution; its seed is its own draw
    expanded = logits.view(1, -1).expand(samples, -1)
    temperatures = torch.full((samples,), temperature, device="cuda")
    seeds = torch.arange(samples, device="cuda", dtype=torch.int64) * 7919 + 1
    tokens = gathered_cdf.cdf_tokens(expanded, temperatures, seeds)
    counts = torch.bincount(tokens, minlength=logits.numel()).double()
    assert counts.numel() == logits.numel()
    expected = torch.softmax(logits.double() / temperature, dim=-1) * samples
    return counts, expected


@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_draws_follow_the_softmax_distribution(temperature, dtype):
    generator = torch.Generator(device="cuda").manual_seed(1)
    logits = (torch.randn(5000, device="cuda", generator=generator) * 3).to(dtype)
    counts, expected = _draw_counts(logits, temperature, 200_000)
    p_value = _chi_square_p_value(counts, expected)
    assert p_value > 1e-4, f"chi-square p-value {p_value:.3g}"


def test_a_confident_row_follows_the_softmax_distribution():
    generator = torch.Generator(device="cuda").manual_seed(5)
    logits = torch.randn(5000, device="cuda", generator=generator) * 2
    logits[2600] += 16.0
    counts, expected = _draw_counts(logits, 1.0, 200_000)
    tail = torch.ones_like(counts, dtype=torch.bool)
    tail[2600] = False
    group = torch.arange(counts.numel(), device="cuda") // 1024
    pooled_counts = [counts[2600].view(1)]
    pooled_expected = [expected[2600].view(1)]
    for g in range(int(group.max()) + 1):
        pooled_counts.append(counts[tail & (group == g)].sum().view(1))
        pooled_expected.append(expected[tail & (group == g)].sum().view(1))
    p_value = _chi_square_p_value(torch.cat(pooled_counts), torch.cat(pooled_expected))
    assert p_value > 1e-4, f"chi-square p-value {p_value:.3g}"


def test_a_max_tied_across_two_splits_follows_the_softmax_distribution():
    generator = torch.Generator(device="cuda").manual_seed(6)
    logits = torch.randn(5000, device="cuda", generator=generator) * 2
    # the row max in split 0 and again, equal, as split 1's max
    logits[700] = 9.0
    logits[3000] = 9.0
    counts, expected = _draw_counts(logits, 1.0, 200_000)
    p_value = _chi_square_p_value(counts, expected)
    assert p_value > 1e-4, f"chi-square p-value {p_value:.3g}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_maxima_tied_within_one_split_follow_the_softmax_distribution(dtype):
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(5000, device="cuda", generator=generator)
    logits[[700, 1500]] = 9.0
    logits[[2500, 3000]] = 7.0
    counts, expected = _draw_counts(logits.to(dtype), 1.0, 200_000)
    p_value = _chi_square_p_value(counts, expected)
    assert p_value > 1e-4, f"chi-square p-value {p_value:.3g}"


def test_masked_columns_are_never_drawn():
    rows, vocabulary = 4096, 4100
    logits = torch.full((rows, vocabulary), float("-inf"), device="cuda")
    live = torch.tensor([3, 2049, 4099], device="cuda")
    logits[:, live] = torch.tensor([0.0, 1.0, -1.0], device="cuda")
    logits[0, :] = float("-inf")
    logits[0, 2050] = 0.0
    temperatures = torch.full((rows,), 0.8, device="cuda")
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64)
    tokens = gathered_cdf.cdf_tokens(logits, temperatures, seeds)
    assert int(tokens[0]) == 2050
    assert torch.isin(tokens[1:], live).all()


@triton.jit
def _uniforms_kernel(seed_ptr, u_ptr):
    i = tl.program_id(0).to(tl.int64)
    r1, r2, r3, r4 = tl.rand4x(tl.load(seed_ptr + i), tl.arange(0, 1))
    tl.store(u_ptr + i * 4, tl.sum(r1, axis=0))
    tl.store(u_ptr + i * 4 + 1, tl.sum(r2, axis=0))
    tl.store(u_ptr + i * 4 + 2, tl.sum(r3, axis=0))
    tl.store(u_ptr + i * 4 + 3, tl.sum(r4, axis=0))


@triton.jit
def _draw_at_kernel(
    logits_ptr,
    stride_row,
    temperature_ptr,
    m_ptr,
    z_ptr,
    row_ptr,
    u_ptr,
    out_ptr,
    vocab_size,
    nparts,
    BLOCK: tl.constexpr,
    PBLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    token = gathered_cdf._cdf_draw(
        logits_ptr,
        stride_row,
        temperature_ptr,
        m_ptr,
        z_ptr,
        tl.load(row_ptr + i),
        vocab_size,
        nparts,
        tl.load(u_ptr + i * 4),
        tl.load(u_ptr + i * 4 + 1),
        tl.load(u_ptr + i * 4 + 2),
        tl.load(u_ptr + i * 4 + 3),
        BLOCK,
        PBLOCK,
    )
    tl.store(out_ptr + i, token.to(tl.int64))


class _StagedDraw:
    """The served draw of given rows at given uniforms (pass 1 once, then the draw)."""

    def __init__(self, logits, temperatures):
        self.logits = logits
        self.temperatures = temperatures.contiguous()
        self.m, self.z = gathered_cdf._split_stats(logits, self.temperatures)
        self.nparts = self.m.shape[1]

    def __call__(self, rows, uniforms):
        n = rows.numel()
        out = torch.empty(n, dtype=torch.int64, device="cuda")
        _draw_at_kernel[(n,)](
            self.logits,
            self.logits.stride(0),
            self.temperatures,
            self.m,
            self.z,
            rows.to(torch.int64).contiguous(),
            uniforms.to(torch.float32).contiguous(),
            out,
            self.logits.shape[1],
            self.nparts,
            BLOCK=_BLOCK,
            PBLOCK=triton.next_power_of_2(self.nparts),
            num_warps=8,
        )
        return out


def _served_uniform(x):
    scale = torch.tensor(_RAND_SCALE, dtype=torch.float32, device=x.device)
    return x.to(torch.float32) * scale


def _coarse_uniform(bits):
    def uniform(x):
        return x.to(torch.float32) * 2.0**-bits

    return uniform


def _first(predicate, items, size):
    lo = torch.zeros(items, dtype=torch.int64, device="cuda")
    hi = torch.full((items,), size, dtype=torch.int64, device="cuda")
    while bool((lo < hi).any()):
        active = lo < hi
        mid = (lo + hi) // 2
        ok = predicate(mid.clamp(max=size - 1))
        hi = torch.where(active & ok, mid, hi)
        lo = torch.where(active & ~ok, mid + 1, lo)
    return hi


def _staged_law(draw, vocabulary, uniform, size):
    weight = 1.0 / size
    zero = uniform(torch.zeros(1, dtype=torch.int64, device="cuda"))
    one = torch.ones(1, dtype=torch.float32, device="cuda")

    def at(us):
        n = max(u.numel() for u in us)
        us = torch.stack([u.expand(n) for u in us], dim=1)
        return draw(torch.zeros(n, dtype=torch.int64, device="cuda"), us)

    law = torch.zeros(vocabulary, dtype=torch.float64, device="cuda")
    top = int(at([one, zero, zero, zero]))
    top_split = top // _BLOCK
    # stage 1: the max column for u1 at or above the threshold
    x1 = int(_first(lambda x: at([uniform(x), zero, zero, zero]) == top, 1, size))
    p_top = (size - x1) * weight
    law[top] = p_top
    if x1 == 0:
        return law, top
    nparts = -(-vocabulary // _BLOCK)
    # stage 2: the split boundaries (any u3 / u4: the token's split is the stage's)
    parts = torch.arange(nparts + 1, device="cuda")
    xs = _first(
        lambda x: at([zero, uniform(x), zero, zero]) // _BLOCK >= parts,
        nparts + 1,
        size,
    )
    p_split = (xs[1:] - xs[:-1]).double() * weight
    for s in torch.nonzero(p_split > 0).flatten().tolist():
        u2 = uniform(xs[s].view(1))
        p_rest = 1.0
        if s != top_split:
            # stage 3: the split's max column for u3 at or above its threshold
            max_col = int(at([zero, u2, one, zero]))
            x3 = int(
                _first(lambda x: at([zero, u2, uniform(x), zero]) == max_col, 1, size)
            )
            p_rest = x3 * weight
            law[max_col] += (1 - p_top) * p_split[s] * (1 - p_rest)
        if p_rest == 0:
            continue
        # stage 4: the split's other columns, u4's boundaries
        cols = torch.arange(_BLOCK + 1, device="cuda")
        x4 = _first(
            lambda x: at([zero, u2, zero, uniform(x)]) - s * _BLOCK >= cols,
            _BLOCK + 1,
            size,
        )
        # monotone: the draw just below each boundary is an earlier column
        inner = (x4[:-1] > 0) & (x4[:-1] < size)
        below = at([zero, u2, zero, uniform((x4[:-1] - 1).clamp(min=0))]) - s * _BLOCK
        assert bool((below[inner] < cols[:-1][inner]).all())
        width = min(_BLOCK, vocabulary - s * _BLOCK)
        p_col = (x4[1:] - x4[:-1]).double() * weight
        law[s * _BLOCK : s * _BLOCK + width] += (
            (1 - p_top) * p_split[s] * p_rest * p_col[:width]
        )
    return law, top


def _law_errors(law, target, top):
    rest = torch.ones_like(law, dtype=torch.bool)
    rest[top] = False
    law_tail = law[rest] / law[rest].sum()
    target_tail = target[rest] / target[rest].sum()
    tail = float((law_tail - target_tail).norm() / target_tail.norm())
    return abs(float(law[top] - target[top])), tail


def test_the_served_draw_is_the_staged_draw_of_its_seeds_uniforms():
    generator = torch.Generator(device="cuda").manual_seed(7)
    vocabulary, rows = 3 * _BLOCK + 300, 24
    logits = torch.randn(rows, vocabulary, device="cuda", generator=generator) * 2
    peaks = torch.randint(0, vocabulary, (rows,), device="cuda", generator=generator)
    logits[torch.arange(rows, device="cuda"), peaks] += torch.linspace(
        0, 24, rows, device="cuda"
    )
    temperatures = torch.rand(rows, device="cuda", generator=generator) + 0.5
    draw = _StagedDraw(logits, temperatures)
    for round_ in range(64):
        seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + round_ * (
            1 << 16
        )
        uniforms = torch.empty((rows, 4), dtype=torch.float32, device="cuda")
        _uniforms_kernel[(rows,)](seeds, uniforms)
        served = gathered_cdf.cdf_tokens(logits, temperatures, seeds)
        assert torch.equal(served, draw(torch.arange(rows, device="cuda"), uniforms))


@pytest.mark.parametrize(
    "gap, temperature, dtype",
    [(12.0, 1.0, torch.float32), (5.0, 0.8, torch.bfloat16)],
    ids=["confident", "peaky-t0.8-bf16"],
)
def test_the_staged_law_resolves_the_max_and_the_tail_behind_it(
    gap, temperature, dtype
):
    generator = torch.Generator(device="cuda").manual_seed(8)
    vocabulary = 3 * _BLOCK + 300
    logits = torch.randn(vocabulary, device="cuda", generator=generator) * 2
    logits[2500] = logits.max() + gap
    logits = logits.to(dtype).view(1, -1)
    target = torch.softmax(logits.double().flatten() / temperature, dim=-1)
    draw = _StagedDraw(logits, torch.full((1,), temperature, device="cuda"))
    law, top = _staged_law(draw, vocabulary, _served_uniform, 1 << 31)
    assert top == 2500
    assert abs(float(law.sum()) - 1.0) < 1e-12
    top_bound = _TOP_GRID + _TOP_REL * (1.0 - float(target[top]))
    top_err, tail_err = _law_errors(law, target, top)
    control, _ = _staged_law(draw, vocabulary, _coarse_uniform(14), 1 << 14)
    control_top, control_tail = _law_errors(control, target, top)
    logger.info(
        "gathered_cdf law vs fp64: max column p %.9f |err| %.3e (bound %.3e), tail "
        "rel-L2 %.3e (bound %.3e); 14-bit control %.3e / %.3e",
        float(target[top]),
        top_err,
        top_bound,
        tail_err,
        _TAIL_REL_L2_MAX,
        control_top,
        control_tail,
    )
    assert top_err <= top_bound
    assert tail_err <= _TAIL_REL_L2_MAX
    assert control_top > top_bound and control_tail > _TAIL_REL_L2_MAX


# ------------------------------------------------------------- seeds and graphs


def test_draws_repeat_with_their_seeds_and_the_counter_advances():
    generator = torch.Generator(device="cuda").manual_seed(2)
    logits = torch.randn(64, 3000, device="cuda", generator=generator)
    temperatures = torch.rand(64, device="cuda", generator=generator) + 0.5
    counter = gathered_cdf.RoundSeeds(torch.device("cuda"))
    first = counter.next(64)
    second = counter.next(64)
    assert int(counter.round) == 2
    assert torch.equal(second - first, torch.full_like(first, 1 << 16))
    a = gathered_cdf.cdf_tokens(logits, temperatures, first)
    b = gathered_cdf.cdf_tokens(logits, temperatures, first)
    assert torch.equal(a, b)
    assert not torch.equal(a, gathered_cdf.cdf_tokens(logits, temperatures, second))


def test_a_graph_replay_advances_the_counter_and_equals_eager():
    generator = torch.Generator(device="cuda").manual_seed(3)
    logits = torch.randn(28, 6000, device="cuda", generator=generator)
    temperatures = torch.rand(28, device="cuda", generator=generator) + 0.5
    eager = gathered_cdf.RoundSeeds(torch.device("cuda"))
    want = [
        gathered_cdf.cdf_tokens(logits, temperatures, eager.next(28)) for _ in range(3)
    ]

    graphed = gathered_cdf.RoundSeeds(torch.device("cuda"))
    out = torch.empty(28, dtype=torch.int64, device="cuda")

    def run():
        out.copy_(gathered_cdf.cdf_tokens(logits, temperatures, graphed.next(28)))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graphed.round.zero_()
    for round_tokens in want:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, round_tokens)
    assert int(graphed.round) == 3


def test_a_built_counter_is_captured_without_an_eager_round():
    generator = torch.Generator(device="cuda").manual_seed(4)
    logits = torch.randn(28, 6000, device="cuda", generator=generator)
    temperatures = torch.rand(28, device="cuda", generator=generator) + 0.5
    seeds = gathered_cdf.build(torch.device("cuda"), 6000)
    out = torch.empty(28, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(gathered_cdf.cdf_tokens(logits, temperatures, seeds.next(28)))
        with pytest.raises(RuntimeError, match="inside CUDA-graph capture"):
            gathered_cdf.build(torch.device("cuda"), 6000)
    eager = gathered_cdf.RoundSeeds(torch.device("cuda"))
    for _ in range(3):
        graph.replay()
        want = gathered_cdf.cdf_tokens(logits, temperatures, eager.next(28))
        torch.cuda.synchronize()
        assert torch.equal(out, want)
    assert int(seeds.round) == 3
