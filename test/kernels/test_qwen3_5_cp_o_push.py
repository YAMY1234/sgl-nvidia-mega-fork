# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

HQ, D, PAGE, CP, QL = 4, 256, 64, 4, 7

FORMS = {
    "bf16": ({}, {}),
    "p2f16": (dict(p_hilo=True, o_f16=True), dict(F16=True)),
}

BATCHES = {
    "batch4": [98_000 + 977 * i for i in range(4)],
    "batch13": [3_000 + 977 * i for i in range(12)] + [7],
    "batch16": [98_000 + 977 * i for i in range(16)],
}


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import triton  # noqa: F401
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


def _inputs(lens, seed):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    b = len(lens)
    pages = (max(lens) + PAGE - 1) // PAGE
    pool = b * pages + 1
    rows = b * QL

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
        q=fp8(rows, HQ, D, s=1.0),
        gate=torch.randn(rows, HQ * D, device=dev, generator=g).to(torch.bfloat16),
    )


def _partial(F, x, me, push=None, form="bf16"):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    b = x["lens"].numel()
    po = torch.zeros(CP, b * QL, HQ, D, dtype=torch.bfloat16, device="cuda")
    pl = torch.zeros(CP, b * QL, HQ, device="cuda")
    F.q35_cp_fused_partial(
        x["q_all"].view(b, CP, QL * HQ, D).permute(1, 0, 2, 3),
        x["k"],
        x["table"],
        x["lens"],
        1,
        me,
        CP,
        0.75 * D**-0.5,
        0.5,
        po.view(CP, b, QL * HQ, D),
        pl.view(CP, b, QL * HQ),
        enable_pdl=True,
        v_pool=x["v"],
        p_shift=L._EXACT_P_SHIFT,
        rescale_threshold=L._EXACT_RESCALE_THRESHOLD,
        len_sub=QL,
        early_trigger=push is None,
        **({} if push is None else push),
        **FORMS[form][0],
    )
    torch.cuda.synchronize()
    return po, pl


def _ll(epoch, words):
    return (epoch << 32) | (words.to(torch.int64) & 0xFFFFFFFF)


def _mailboxes(rows):
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import _LL_PWORDS

    boxes = [
        torch.zeros(rows, CP, _LL_PWORDS, dtype=torch.int64, device="cuda")
        for _ in range(CP)
    ]
    tab = torch.tensor([m.data_ptr() for m in boxes], dtype=torch.int64, device="cuda")
    return boxes, tab


def _push_words(po, pl, d):
    rows = po.shape[1]
    o32 = po[d].contiguous().view(torch.int32).view(rows, HQ * D // 2)
    l32 = pl[d].contiguous().view(torch.int32)
    return torch.cat([_ll(6, o32), _ll(6, l32)], dim=1)


PHASES = {
    "off": dict(on=False),
    "lean": dict(on=True, ep_early=False),
    "lean_ep_early": dict(on=True),
}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("phase", sorted(PHASES))
@pytest.mark.parametrize("case", sorted(BATCHES))
def test_the_partial_pushes_the_merge_push_words(F, case, phase, form):
    F.set_cp_phase(**PHASES[phase])
    lens = BATCHES[case]
    x = _inputs(lens, seed=len(lens))
    rows = len(lens) * QL
    NO = HQ * D // 2
    for me in range(CP):
        po_ref, pl_ref = _partial(F, x, me, form=form)
        boxes, tab = _mailboxes(rows)
        cnt_h = torch.full((rows * HQ,), 5, dtype=torch.int32, device="cuda")
        push = dict(o_push=True, ll=tab, cnt_h=cnt_h, late_trigger=True)
        po, pl = _partial(F, x, me, push, form=form)
        assert not po.view(torch.int16).any(), "o_push writes no out_o row"
        assert torch.equal(pl.view(torch.int32), pl_ref.view(torch.int32))
        assert bool((cnt_h == 5).all())
        for d in range(CP):
            got = boxes[d][:, me, : NO + HQ]
            assert torch.equal(got, _push_words(po_ref, pl_ref, d)), (case, me, d)
            others = [c for c in range(CP) if c != me]
            assert not boxes[d][:, others].any(), (case, me, d)
            assert not boxes[d][:, me, NO + HQ :].any()


def _merge(x, peers, me, counter, out, oq, po, pl, **options):
    import triton

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _LL_PWORDS,
        _cp_merge_h_ll,
    )

    rows = x["q"].shape[0]
    b = x["lens"].numel()
    _cp_merge_h_ll[(rows, HQ)](
        po.view(torch.int32),
        pl.view(torch.int32),
        HQ * D // 2,
        po.stride(0) // 2,
        HQ,
        pl.stride(0),
        x["q"],
        x["q"].stride(0),
        x["k"],
        x["v"],
        x["k"].stride(0),
        x["k"].stride(2),
        x["table"],
        x["table"].stride(0),
        x["lens"],
        out,
        out.stride(0),
        peers,
        me,
        0,
        counter,
        0.75 * D**-0.5,
        0.5,
        b,
        0,
        QL=QL,
        HQ=HQ,
        CP=CP,
        D=D,
        PG=PAGE,
        NWP=_LL_PWORDS,
        NB=triton.next_power_of_2(b),
        GMAX=False,
        WAIT=True,
        num_warps=4,
        launch_pdl=True,
        gate=x["gate"],
        gate_stride=x["gate"].stride(0),
        o_proj_input_scale=torch.tensor([0.05], device="cuda"),
        o_proj_input=oq,
        o_proj_input_stride=oq.stride(0),
        GATE_EPILOGUE=True,
        EARLY=True,
        PLAIN_WAIT=True,
        CG=True,
        **options,
    )


def _land_late(arm, attempts=3):
    side = torch.cuda.Stream(priority=-1)
    torch.cuda._sleep(1)  # load the delay's kernel
    for attempt in range(attempts):
        land, launch, check = arm(attempt)
        torch.cuda.synchronize()
        started, landed, done = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(1 << (21 + attempt))
            landed.record()
            land()
        started.record()
        launch()
        done.record()
        torch.cuda.synchronize()
        assert landed.elapsed_time(done) > 0, "the merge ended before the words"
        check()
        if started.elapsed_time(landed) > 0:
            return
    pytest.fail(f"the words landed first in all {attempts} attempts")


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("case", sorted(BATCHES))
def test_the_polling_merge_writes_the_plain_wait_merges_bytes(F, case, late, form):
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _PUSHED_MERGE_SLEEP_NS,
    )

    F.set_cp_phase(True)
    lens = BATCHES[case]
    x = _inputs(lens, seed=100 + len(lens))
    rows = len(lens) * QL
    me = 1
    boxes, tab = _mailboxes(rows)
    refs = []
    for c in range(CP):
        cnt_h = torch.full((rows * HQ,), 5, dtype=torch.int32, device="cuda")
        push = dict(o_push=True, ll=tab, cnt_h=cnt_h, late_trigger=True)
        _partial(F, x, c, push, form=form)
        refs.append(_partial(F, x, c, form=form))
    pushed = boxes[me].clone()

    def outputs():
        return (
            torch.full((rows * HQ,), 5, dtype=torch.int32, device="cuda"),
            torch.empty(rows, HQ, D, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(rows, HQ * D, dtype=torch.float8_e4m3fn, device="cuda"),
        )

    region = pushed.clone()
    region[:, me] = 0
    sinks = [torch.zeros_like(region) for _ in range(CP)]
    peers = torch.tensor(
        [(region if c == me else sinks[c]).data_ptr() for c in range(CP)],
        dtype=torch.int64,
        device="cuda",
    )
    cnt_ref, out_ref, oq_ref = outputs()
    po_me, pl_me = refs[me]
    merge_form = FORMS[form][1]
    _merge(
        x,
        peers,
        me,
        cnt_ref,
        out_ref,
        oq_ref,
        po_me,
        pl_me,
        PUSH_ROT=True,
        **merge_form,
    )
    torch.cuda.synchronize()
    assert torch.equal(region, pushed), "the push writes the partial's words"
    assert oq_ref.view(torch.uint8).any()

    # the polling merge: no push, one pass over every source, the wait at its end
    region = pushed.clone()
    others = [c for c in range(CP) if c != me]
    peers = torch.tensor(
        [region.data_ptr() if c == me else 0 for c in range(CP)],
        dtype=torch.int64,
        device="cuda",
    )

    def polling(cnt, out, oq):
        return lambda: _merge(
            x,
            peers,
            me,
            cnt,
            out,
            oq,
            po_me,
            pl_me,
            NOPUSH=True,
            MSLEEP=_PUSHED_MERGE_SLEEP_NS,
            **merge_form,
        )

    def check(cnt, out, oq):
        assert torch.equal(out.view(torch.int16), out_ref.view(torch.int16))
        assert torch.equal(oq.view(torch.uint8), oq_ref.view(torch.uint8))
        assert torch.equal(cnt, cnt_ref) and bool((cnt == 6).all())

    cnt, out, oq = outputs()
    polling(cnt, out, oq)()
    torch.cuda.synchronize()
    check(cnt, out, oq)
    if late:

        def arm(attempt):
            region.copy_(pushed)
            region[:, others] = 0
            cnt, out, oq = outputs()

            def land():
                for c in others:
                    region[:, c].copy_(pushed[:, c])

            return land, polling(cnt, out, oq), lambda: check(cnt, out, oq)

        _land_late(arm)
