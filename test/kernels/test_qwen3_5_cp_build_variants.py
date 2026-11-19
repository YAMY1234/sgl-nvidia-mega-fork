# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

HQ, D, PAGE, CP = 4, 256, 64, 4


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")
_LD_RED = _gpu_ready() and torch.cuda.get_device_capability() == (10, 3)
BUILDS = ((2, ""), (0, "-O1"), (2, "-O1")) if _LD_RED else ((0, "-O1"),)


@pytest.fixture(scope="module")
def F():
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

    dev = torch.device("cuda")
    F.q35_cp_fused_prepare(
        dev, torch.cuda.get_device_properties(dev).multi_processor_count
    )
    yield F
    F.set_partial_build()


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


def _run(F, x, ql, me, build, exact, balanced, form=None):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    F.set_partial_build(*build)
    b = x["lens"].numel()
    x["po"].zero_()
    x["pl"].zero_()
    numerics = (
        dict(p_shift=L._EXACT_P_SHIFT, rescale_threshold=L._EXACT_RESCALE_THRESHOLD)
        if exact
        else {}
    )
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
        len_sub=ql,
        **numerics,
        **(form or {}),
        **(
            {}
            if balanced
            else dict(
                balanced=False,
                counters=torch.zeros(
                    F.MAX_BATCH + 1, F.MAX_BATCH, dtype=torch.int64, device="cuda"
                ),
            )
        ),
    )
    torch.cuda.synchronize()
    return x["po"].view(torch.int16).clone(), x["pl"].view(torch.int32).clone()


def _ragged(b, long, ql):
    return [long + 977 * i + 13 * (i % 5) for i in range(b - 1)] + [ql]


@pytest.mark.parametrize("p2f16", [False, True])
@pytest.mark.parametrize("ql", [7, 1])
@pytest.mark.parametrize("b", [2, 16, 40, 80])
@pytest.mark.parametrize("exact", [True, False])
def test_balanced_builds_write_the_default_builds_bytes(F, ql, b, exact, p2f16):
    if p2f16 and not (exact and b <= F._WIDE_PLAN_MAX_BATCH):
        pytest.skip("p2f16: the exact call on the one-slot plans")
    form = dict(p_hilo=True, o_f16=True) if p2f16 else None
    x = _inputs(_ragged(b, 9_000, ql), ql, seed=b * 10 + ql)
    for me in range(CP):
        ref = _run(F, x, ql, me, (0, ""), exact, True, form)
        for build in BUILDS:
            got = _run(F, x, ql, me, build, exact, True, form)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), (
                b,
                me,
                build,
            )


def test_the_equal_split_builds_write_the_default_builds_bytes(F):
    x = _inputs(_ragged(4, 20_000, 7), 7, seed=3)
    for me in range(CP):
        ref = _run(F, x, 7, me, (0, ""), True, False)
        for build in BUILDS:
            got = _run(F, x, 7, me, build, True, False)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), (
                me,
                build,
            )
