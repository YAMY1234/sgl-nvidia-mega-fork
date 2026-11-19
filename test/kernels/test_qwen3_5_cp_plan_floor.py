# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random

import pytest
import torch

HQ, D, PAGE, CP, QL = 4, 256, 64, 4, 7


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


def _mixes(b: int, seed: int) -> dict:
    """Prefix lengths (tokens before the call's QL rows) of b requests."""
    rng = random.Random(1000 * b + seed)
    long = [rng.randrange(100_000, 260_000) for _ in range(b)]
    short = [rng.randrange(64, 4_096) for _ in range(b)]
    tiny = [rng.randrange(0, 1_024) for _ in range(b)]  # 0-2 tiles a rank, some none
    return {
        "split": [s if i % 2 == 0 else x for i, (s, x) in enumerate(zip(short, long))],
        "tiny": [t if i % 2 == 0 else x for i, (t, x) in enumerate(zip(tiny, long))],
        "one_short": [short[0]] + long[1:],
        "equal": [50_000] * b,
        "wide": [rng.randrange(1_024, 260_000) for _ in range(b)],
    }


def _plan_rows(F, prefixes, me, floor, plan, seed):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    b = len(prefixes)
    lens = [p + QL for p in prefixes]
    pages = (max(lens) + PAGE - 1) // PAGE
    pool = b * pages + 1

    def fp8(*shape):
        return (torch.randn(*shape, device=dev, generator=g) * 2).to(
            torch.float8_e4m3fn
        )

    table = torch.randperm(pool - 1, device=dev, generator=g)[: b * pages]
    rows = b * QL
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    n_ctas = F._num_ctas(sms)
    debug = torch.zeros(n_ctas, 16, dtype=torch.int64, device=dev)
    po = torch.zeros(CP, rows, HQ, D, dtype=torch.bfloat16, device=dev)
    pl = torch.zeros(CP, rows, HQ, device=dev)
    # the partial reads its one-slot floor from the module (no call takes another)
    floor_was = F._MIN_CTAS
    if floor is not None:
        F._MIN_CTAS = floor
    try:
        F.q35_cp_fused_partial(
            fp8(rows, CP * HQ, D).view(b, CP, QL * HQ, D).permute(1, 0, 2, 3),
            fp8(pool, 1, PAGE, D),
            table.int().view(b, pages).contiguous(),
            torch.tensor(lens, dtype=torch.int32, device=dev),
            1,
            me,
            CP,
            0.75 * D**-0.5,
            0.5,
            po.view(CP, b, QL * HQ, D),
            pl.view(CP, b, QL * HQ),
            enable_pdl=True,
            debug=debug,
            v_pool=fp8(pool, 1, PAGE, D),
            p_shift=L._EXACT_P_SHIFT,
            rescale_threshold=L._EXACT_RESCALE_THRESHOLD,
            len_sub=QL,
            early_trigger=True,
            plan=plan,
        )
    finally:
        F._MIN_CTAS = floor_was
    torch.cuda.synchronize()
    return debug.cpu(), n_ctas


def _host_rows(F, prefixes, me, n_ctas, floor):
    _, c, ctas = F.balanced_split_plan(prefixes, me, CP, n_ctas, min_ctas=floor)
    first = [sum(c[:r]) for r in range(len(c))]
    return [
        ((((r * 256 + j) * 256 + c[r]) * 256 + first[r]), k_lo, n_t)
        for r, j, k_lo, n_t in ctas
    ]


def _plan_name(b: int) -> str:
    return "lane" if b <= 32 else ("wide" if b <= 64 else "quad")


@pytest.mark.parametrize("floor", [1, None])
@pytest.mark.parametrize("b", [1, 2, 9, 13, 32, 33, 47, 64, 65, 96, 128])
def test_the_in_kernel_plan_is_the_host_mirrors(F, b, floor):
    F.set_cp_phase(False)
    plan = _plan_name(b)
    for name, prefixes in _mixes(b, 0).items():
        if b == 1 and name != "wide":
            continue
        for me in range(CP):
            got, n_ctas = _plan_rows(F, prefixes, me, floor, plan, seed=b + me)
            want = _host_rows(F, prefixes, me, n_ctas, floor)
            rows = [tuple(int(x) for x in got[i, 13:16]) for i in range(n_ctas)]
            assert rows == want, (b, floor, name, me)


@pytest.mark.parametrize("b", [9, 32, 64])
def test_the_lean_phase_plans_as_the_default_build(F, b):
    prefixes = _mixes(b, 1)["tiny"]
    for me in range(CP):
        F.set_cp_phase(False)
        ref, n_ctas = _plan_rows(F, prefixes, me, None, _plan_name(b), seed=7)
        F.set_cp_phase(True)
        try:
            got, _ = _plan_rows(F, prefixes, me, None, _plan_name(b), seed=7)
        finally:
            F.set_cp_phase(False)
        assert torch.equal(got[:, 13:16], ref[:, 13:16]), (b, me)
        assert ref[:, 13:16].tolist() == [
            list(t) for t in _host_rows(F, prefixes, me, n_ctas, None)
        ]
