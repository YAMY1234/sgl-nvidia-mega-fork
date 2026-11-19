# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

HQ, D, PAGE, CP = 4, 256, 64, 4
FORMS = {"bf16": {}, "p2f16": dict(p_hilo=True, o_f16=True)}
ARMS = {
    "lean": dict(plan_cache=False, comb_w=False),
    "lean_comb": dict(plan_cache=False, comb_w=True),
    "on": dict(plan_cache=True, comb_w=True),
}


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


@pytest.fixture(scope="module")
def F():
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

    dev = torch.device("cuda")
    F.q35_cp_fused_prepare(
        dev, torch.cuda.get_device_properties(dev).multi_processor_count
    )
    yield F
    F.set_cp_phase(False)


def _inputs(lens, ql, seed):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    b = len(lens)
    pages = (max(lens) + PAGE - 1) // PAGE
    pool = b * pages + 1
    rows = b * ql

    def fp8(*shape, s=2.0):
        x = torch.randn(*shape, device=dev, generator=g) * s
        return x.to(torch.float8_e4m3fn)

    table = torch.randperm(pool - 1, device=dev, generator=g)[: b * pages]
    return dict(
        k=fp8(pool, 1, PAGE, D),
        v=fp8(pool, 1, PAGE, D),
        table=table.int().view(b, pages).contiguous(),
        lens=torch.tensor(lens, dtype=torch.int32, device=dev),
        q_all=fp8(rows, CP * HQ, D, s=1.0),
        po=torch.zeros(CP, rows, HQ, D, dtype=torch.bfloat16, device=dev),
        pl=torch.zeros(CP, rows, HQ, device=dev),
    )


def _call(F, x, ql, me, form="bf16"):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    b = x["lens"].numel()
    F.q35_cp_fused_partial(
        x["q_all"].view(b, CP, ql * HQ, D).permute(1, 0, 2, 3),
        x["k"],
        x["table"],
        x["lens"],
        1,
        me,
        CP,
        0.75 * D**-0.5,
        0.5,
        x["po"].view(CP, b, ql * HQ, D),
        x["pl"].view(CP, b, ql * HQ),
        enable_pdl=True,
        v_pool=x["v"],
        p_shift=L._EXACT_P_SHIFT,
        rescale_threshold=L._EXACT_RESCALE_THRESHOLD,
        len_sub=ql,
        early_trigger=True,
        **FORMS[form],
    )


def _run(F, x, ql, me, arm, form="bf16"):
    if arm == "off":
        F.set_cp_phase(False)
    else:
        F.set_cp_phase(True, **ARMS[arm])
    x["po"].zero_()
    x["pl"].zero_()
    _call(F, x, ql, me, form)
    torch.cuda.synchronize()
    return x["po"].view(torch.int16).clone(), x["pl"].view(torch.int32).clone()


def _same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


VERIFY = {
    "batch13": [3_000 + 977 * i for i in range(12)]
    + [7],  # the last request has no prefix
    "batch14": [98_000 + 977 * i for i in range(13)] + [707],
    "batch16": [98_000 + 977 * i for i in range(16)],
    "batch16_long": [300_000] * 2 + [120_000 + 3_333 * i for i in range(14)],
}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("case", sorted(VERIFY))
def test_verify_every_rank_writes_the_same_bytes(F, case, form):
    lens = VERIFY[case]
    x = _inputs(lens, 7, seed=len(lens))
    for me in range(CP):
        ref = _run(F, x, 7, me, "off", form)
        for arm in ("lean", "lean_comb", "on", "on"):  # the second "on" call hits
            assert _same(_run(F, x, 7, me, arm, form), ref), (case, me, arm, form)


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("b", [8, 16])
def test_one_token_calls_write_the_same_bytes(F, b, form):
    lens = [60_000 + 4_001 * i for i in range(b - 1)] + [1_500]
    x = _inputs(lens, 1, seed=100 + b)
    ref = _run(F, x, 1, 1, "off", form)
    for arm in ("lean", "lean_comb", "on", "on"):
        assert _same(_run(F, x, 1, 1, arm, form), ref), (b, arm, form)


def _rows(F):
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return F._state(dev, sms).plan_cache


def test_plan_cache_hits_misses_and_torn_rows(F):
    lens = VERIFY["batch16"]
    x = _inputs(lens, 7, seed=7)
    ref_a = _run(F, x, 7, 1, "off")
    assert _same(_run(F, x, 7, 1, "on"), ref_a)
    assert _rows(F), "the plan cache was not allocated"

    def gens():
        return torch.stack([t[:, 33] & 0xFFFFFFFF for t in _rows(F).values()]).clone()

    g0 = gens()
    assert _same(_run(F, x, 7, 1, "on"), ref_a)
    assert torch.equal(gens(), g0), "repeated lengths must hit (no publish)"
    x["lens"][5] += 1
    ref_b = _run(F, x, 7, 1, "off")
    assert _same(_run(F, x, 7, 1, "on"), ref_b)
    g1 = gens()
    assert bool((g1 >= g0).all()) and int((g1 - g0).sum()) > 0
    # back to the first lengths: miss again (the rows hold the second ones)
    x["lens"][5] -= 1
    assert _same(_run(F, x, 7, 1, "on"), ref_a)
    for t in _rows(F).values():
        gen = ((t[:, 32] >> 32) + 1) & 0xFFFFFFFF
        t[:, 32] = (t[:, 32] & 0xFFFFFFFF) | (gen << 32)
    g2 = gens()
    assert _same(_run(F, x, 7, 1, "on"), ref_a)
    assert bool((gens() > g2).any()), "a torn row must miss and republish"
    for t in _rows(F).values():
        gen = ((t[:, 3] >> 32) + 7) & 0xFFFFFFFF
        t[:, 3] = (t[:, 3] & 0xFFFFFFFF) | (gen << 32)
    g3 = gens()
    assert _same(_run(F, x, 7, 1, "on"), ref_a)
    assert bool((gens() > g3).any())
    # another CP rank on the same rows
    ref_r = _run(F, x, 7, 2, "off")
    assert _same(_run(F, x, 7, 2, "on"), ref_r)
    y = _inputs(lens[:15], 7, seed=7)
    ref_c = _run(F, y, 7, 1, "off")
    assert _same(_run(F, y, 7, 1, "on"), ref_c)
    assert _same(_run(F, x, 7, 1, "on"), ref_a)


def test_plan_cache_graph_replay_on_new_lengths(F):
    lens = VERIFY["batch14"]
    x = _inputs([v + 64 for v in lens], 7, seed=21)  # sized for the longest lengths
    variants = [lens, [v + 1 for v in lens], [v + 64 for v in lens], lens]
    refs = []
    for v in variants:
        x["lens"].copy_(torch.tensor(v, dtype=torch.int32))
        refs.append(_run(F, x, 7, 2, "off"))
    F.set_cp_phase(True)
    x["lens"].copy_(torch.tensor(lens, dtype=torch.int32))
    _call(F, x, 7, 2)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        _call(F, x, 7, 2)
    for _ in range(2):
        for v, ref in zip(variants, refs):
            x["lens"].copy_(torch.tensor(v, dtype=torch.int32))
            x["po"].zero_()
            x["pl"].zero_()
            gr.replay()
            torch.cuda.synchronize()
            got = (x["po"].view(torch.int16), x["pl"].view(torch.int32))
            assert _same(got, ref), v[:2]
    F.set_cp_phase(False)


def test_plan_cache_with_overlapping_launches_in_a_graph(F):
    layers, ql, me, replays = 15, 7, 3, 100
    base = [20_000 + 1_231 * i for i in range(16)]
    sets = [list(base), list(base), list(base)]
    sets[1][4] += 1
    sets[2][11] += 65
    xs = [_inputs([v + 128 for v in base], ql, seed=300 + k) for k in range(layers)]
    refs = {}
    for k, x in enumerate(xs):
        for si, sv in enumerate(sets):
            x["lens"].copy_(torch.tensor(sv, dtype=torch.int32))
            refs[k, si] = _run(F, x, ql, me, "off")
    F.set_cp_phase(True)
    for x in xs:
        _call(F, x, ql, me)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for x in xs:
            _call(F, x, ql, me)
    bad = 0
    for r in range(replays):
        pick = [((k // 2) + r) % 3 for k in range(layers)]
        for x, si in zip(xs, pick):
            x["lens"].copy_(torch.tensor(sets[si], dtype=torch.int32))
            x["po"].zero_()
            x["pl"].zero_()
        gr.replay()
        torch.cuda.synchronize()
        for k, (x, si) in enumerate(zip(xs, pick)):
            got = (x["po"].view(torch.int16), x["pl"].view(torch.int32))
            if not _same(got, refs[k, si]):
                bad += 1
    F.set_cp_phase(False)
    assert bad == 0, f"{bad} of {replays * layers} call outputs differ"
