# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import inspect
import random
import textwrap

import pytest

from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

PAGE, CP = F.PAGE, 4


def _tiles(lens, rank):
    return [((((L + PAGE - 1) // PAGE) - rank + CP - 1) // CP + 1) // 2 for L in lens]


def _plan_without_floor(lens, rank, n_ctas):
    t = _tiles(lens, rank)
    b, total, tmax = len(t), sum(t), max(t)
    lo = max(-(-total // n_ctas), 1)
    hi = max(min(tmax, -(-total // (n_ctas - b))), lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(-(-x // mid), 1) for x in t) <= n_ctas:
            hi = mid
        else:
            lo = mid + 1
    c = [max(-(-x // lo), 1) for x in t]
    top = max(r for r in range(b) if t[r] == tmax)
    c[top] += n_ctas - sum(c)
    return lo, c


def _batches():
    rng = random.Random(7)
    out = [
        [3800, 120000, 200000, 262144, 3100, 90000, 150000, 4700]
        + [100000] * 8,  # mixed short / long
        [8] * 16,  # idle graph
        [7, 8, 9, 1000, 263, 64 * 7 + 7, 120000, 9],  # ragged short
        [262144] * 15 + [8],
    ]
    for b in (1, 2, 8, 15, 16, 32, 33, 64, 100, 128):
        for _ in range(6):
            out.append(
                [
                    rng.choice((8, 300, 3000, 9000, 30000, 100000, 262144))
                    + rng.randrange(64)
                    for _ in range(b)
                ]
            )
    return out


@pytest.mark.parametrize("n_ctas", [148, 152, 160])
def test_floor_one_is_the_plan_without_the_floor(n_ctas):
    for lens in _batches():
        if len(lens) >= n_ctas:
            continue
        for rank in range(CP):
            T, c, _ = F.balanced_split_plan(lens, rank, CP, n_ctas, min_ctas=1)
            assert (T, c) == _plan_without_floor(lens, rank, n_ctas)


@pytest.mark.parametrize("floor", [2, 4, 8])
@pytest.mark.parametrize("n_ctas", [148, 152])
def test_every_request_gets_the_floor_and_t_is_minimal(floor, n_ctas):
    for lens in _batches():
        b = len(lens)
        if b >= n_ctas:
            continue
        k = max(1, min(floor, (n_ctas - 1) // b))
        for rank in range(CP):
            t = _tiles(lens, rank)
            T, c, ctas = F.balanced_split_plan(lens, rank, CP, n_ctas, min_ctas=floor)
            assert sum(c) == n_ctas and len(ctas) == n_ctas
            assert all(cr >= max(min(tr, k), 1) for cr, tr in zip(c, t))
            # every CTA's block is at most T tiles, and T - 1 would not fit the floored counts
            assert max(n for (_, _, _, n) in ctas) <= T
            if T > 1:
                need = sum(max(-(-x // (T - 1)), min(x, k), 1) for x in t)
                assert need > n_ctas
            # the blocks cover each request's tiles exactly once, in order
            for r in range(b):
                blocks = [(f, n) for (rr, _, f, n) in ctas if rr == r]
                assert sum(n for _, n in blocks) == t[r]
                assert all(
                    blocks[j][0] + blocks[j][1] == blocks[j + 1][0]
                    for j in range(len(blocks) - 1)
                )


def test_the_default_floor_is_the_one_slot_plans_floor():
    lens = [3800, 120000, 200000, 262144, 3100, 90000, 150000, 4700] + [100000] * 8
    assert F._MIN_CTAS == 6
    for rank in range(CP):
        assert F.balanced_split_plan(lens, rank, CP, 152) == F.balanced_split_plan(
            lens, rank, CP, 152, min_ctas=F._MIN_CTAS
        )
        _, c, _ = F.balanced_split_plan(lens, rank, CP, 152)
        t = _tiles(lens, rank)
        assert all(cr >= min(tr, F._MIN_CTAS) for cr, tr in zip(c, t))


def test_short_request_beside_long_ones_gets_the_floor():
    lens = [3800, 120000, 200000, 262144, 3100, 90000, 150000, 4700] + [100000] * 8
    pre = [L - 7 for L in lens]
    for rank in range(CP):
        t = _tiles(pre, rank)
        T1, c1, _ = F.balanced_split_plan(pre, rank, CP, 152, min_ctas=1)
        T8, c8, _ = F.balanced_split_plan(pre, rank, CP, 152, min_ctas=8)
        for r in (0, 4, 7):
            assert c1[r] == 1 and c8[r] == min(t[r], 8)
        # the long requests keep the balanced share: T rises only by the floor's CTAs
        extra = sum(c8[r] - c1[r] for r in (0, 4, 7))
        long_ctas = 152 - sum(c1[r] for r in (0, 4, 7))
        assert T8 <= -(-T1 * long_ctas // (long_ctas - extra)) + 1


def test_the_balanced_partial_compiles_with_the_one_slot_floor():
    names = list(inspect.signature(F._compiled.__wrapped__).parameters)
    fn = ast.parse(textwrap.dedent(inspect.getsource(F.q35_cp_fused_partial)))
    floors = []
    for c in ast.walk(fn):
        if not (isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_compiled"):
            continue
        args = {}
        for name, a in zip(names, c.args):
            if isinstance(a, ast.Starred):
                break
            args[name] = a
        args.update({k.arg: k.value for k in c.keywords if k.arg})
        if "min_ctas" in args:
            floors.append(ast.Expression(args["min_ctas"]))
    assert len(floors) == 1  # the balanced call; the equal split builds without a floor
    expr = compile(ast.fix_missing_locations(floors[0]), "<min_ctas>", "eval")
    for two, want in ((False, F._MIN_CTAS), (True, 1)):
        assert eval(expr, {"two": two, "_MIN_CTAS": F._MIN_CTAS}) == want
