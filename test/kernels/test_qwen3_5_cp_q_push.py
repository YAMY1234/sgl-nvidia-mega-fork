# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

HEADS, DIM, CP, QL = 4, 256, 4, 7
WORDS = HEADS * DIM // 4
TMO_NS = 30_000_000_000  # the deadline arms' wait deadline: far past any landing here


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _front_inputs(rows, g, slots=64 * 32):
    widths = (HEADS * 2 * DIM, DIM, DIM)
    qkv = (torch.randn(rows, sum(widths), device="cuda", generator=g) * 3).bfloat16()
    q_gate, k, v = qkv.split(widths, dim=-1)
    norms = [
        (torch.randn(DIM, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
        for _ in range(2)
    ]
    angles = torch.rand(4096, 32, device="cuda", generator=g) * 6.3
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).contiguous()
    pos = torch.randint(0, 4096, (rows,), device="cuda", generator=g)
    pools = tuple(
        torch.randn(slots, 1, DIM, device="cuda", generator=g).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    loc = torch.randperm(slots, device="cuda", generator=g)[:rows]
    args = (q_gate, k, *norms, cos_sin, pos, 1e-6, HEADS, 1, DIM, 64)
    scales = [torch.full((1,), s, device="cuda") for s in (0.8, 1.3)]
    return args, v, pools, loc, scales


def _front(inputs, q_push=None, entry=False):
    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    args, v, pools, loc, scales = inputs
    pools = tuple(p.clone() for p in pools)
    q8, _, gate = fused_qk_gemma_rmsnorm_rope_gate(
        *args,
        has_gate=True,
        v=v,
        kv_cache=pools,
        cache_loc=loc,
        k_scale=scales[0],
        v_scale=scales[1],
        q_push=q_push,
        early_trigger=entry,
    )
    return [
        q8.view(torch.uint8),
        gate.view(torch.int16),
        *(p.view(torch.uint8) for p in pools),
    ]


def _gather_args(cnt, region_ptrs, me, q_all, lens, pre_lens, rows):
    import triton

    b = lens.shape[0]
    peers = torch.tensor(region_ptrs, dtype=torch.int64, device="cuda")
    return (
        torch.zeros(rows, WORDS, dtype=torch.int32, device="cuda"),
        WORDS,
        peers,
        me,
        0,
        q_all,
        cnt,
        lens,
        pre_lens,
        torch.zeros_like(lens),
    ), dict(
        nb=b,
        thr=0,
        GATE=False,
        NB=triton.next_power_of_2(b),
        GMAX=False,
        CP=CP,
        WORDS=WORDS,
        QL=QL,
        FOLD=True,
        GWRITE=False,
        TRIG=True,
        WAIT=True,
        num_warps=4,
        launch_pdl=True,
    )


def _deadline(on: bool) -> dict:
    """The wait half's launch arguments of CP-lite's wait deadline (none while off)."""
    return dict(tmo=TMO_NS, DEADLINE=True) if on else {}


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
        assert landed.elapsed_time(done) > 0, "the wait ended before the words"
        check()
        if started.elapsed_time(landed) > 0:
            return
    pytest.fail(f"the words landed first in all {attempts} attempts")


@pytest.mark.parametrize("deadline", [False, True])
@pytest.mark.parametrize("requests", [2, 4, 8])
def test_the_pushed_words_and_the_wait_half_give_the_full_gathers_bytes(
    requests, deadline
):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    rows = requests * QL
    g = torch.Generator(device="cuda").manual_seed(100 * requests)
    inputs = [_front_inputs(rows, g) for _ in range(CP)]
    want = [_front(x) for x in inputs]
    q8 = [w[0].view(rows, WORDS * 4) for w in want]
    regions = [
        torch.zeros(rows, CP, WORDS, dtype=torch.int64, device="cuda")
        for _ in range(CP)
    ]
    ptrs = [r.data_ptr() for r in regions]
    cnts = [torch.full((rows,), 5, dtype=torch.int32, device="cuda") for _ in range(CP)]
    lens = torch.randint(
        QL, 9000, (requests,), dtype=torch.int32, device="cuda", generator=g
    )
    me = 1

    def words(c, epoch):
        data = q8[c].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        return (epoch << 32) | data

    def push(c, entry=False):
        args = dict(
            peers=torch.tensor(ptrs, dtype=torch.int64, device="cuda"),
            me=c,
            off=0,
            cnt=cnts[c],
            cp=CP,
            words=WORDS,
        )
        got = _front(inputs[c], args, entry)
        assert all(torch.equal(a, b) for a, b in zip(want[c], got)), c

    def wait_half():
        q_all = torch.zeros(requests * CP * QL, WORDS, dtype=torch.int32, device="cuda")
        pre_lens = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
        largs, kwargs = _gather_args(cnts[me], ptrs, me, q_all, lens, pre_lens, rows)
        pushed = type("Pushed", (), {"_q_pushed": rows, "_q_gathered": None})()
        assert L._cp_gather_q_ll_or_wait(pushed, rows) is L._WAIT_ONLY_GATHER
        dl = _deadline(deadline)
        return (
            q_all,
            pre_lens,
            lambda: L._WAIT_ONLY_GATHER[(rows,)](*largs, **kwargs, **dl),
        )

    want_all = torch.empty(requests * CP * QL, WORDS, dtype=torch.int32, device="cuda")
    for c in range(CP):
        for i in range(rows):
            want_all[((i // QL) * CP + c) * QL + i % QL] = q8[c][i].view(torch.int32)

    for c in range(CP):
        push(c, entry=c % 2 == 1)
    torch.cuda.synchronize()
    for d in range(CP):
        for c in range(CP):
            assert torch.equal(regions[d][:, c], words(c, 6)), (d, c)
    q_all, pre_lens, launch = wait_half()
    launch()
    torch.cuda.synchronize()
    assert torch.equal(q_all, want_all)
    assert torch.equal(pre_lens, torch.clamp(lens - QL, min=0))
    assert bool((cnts[me] == 6).all())

    def arm(attempt):
        epoch = 7 + attempt
        push(me, entry=True)
        late = {c: words(c, epoch) for c in range(CP) if c != me}
        q_all, pre_lens, launch = wait_half()

        def land():
            for c, w in late.items():
                regions[me][:, c].copy_(w)

        def check():
            assert torch.equal(q_all, want_all)
            assert bool((cnts[me] == epoch).all())

        return land, launch, check

    _land_late(arm)


@pytest.mark.parametrize("deadline", [False, True])
@pytest.mark.parametrize("requests", [2, 4, 8])
def test_lamport_words_give_the_full_gathers_bytes_and_rearm(requests, deadline):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    rows = requests * QL
    g = torch.Generator(device="cuda").manual_seed(100 * requests + 1)
    inputs = [_front_inputs(rows, g) for _ in range(CP)]
    want = [_front(x) for x in inputs]
    q8 = [w[0].view(rows, WORDS * 4).contiguous().view(torch.int32) for w in want]
    regions = [
        torch.full((rows, CP, WORDS), -1, dtype=torch.int32, device="cuda")
        for _ in range(CP)
    ]
    ptrs = [r.data_ptr() for r in regions]
    cnts = [torch.full((rows,), 5, dtype=torch.int32, device="cuda") for _ in range(CP)]
    lens = torch.randint(
        QL, 9000, (requests,), dtype=torch.int32, device="cuda", generator=g
    )
    me = 2

    def push(c, entry=False):
        args = dict(
            peers=torch.tensor(ptrs, dtype=torch.int64, device="cuda"),
            me=c,
            off=0,
            cnt=cnts[c],
            cp=CP,
            words=WORDS,
            lam=True,
        )
        got = _front(inputs[c], args, entry)
        assert all(torch.equal(a, b) for a, b in zip(want[c], got)), c

    def wait_half():
        q_all = torch.zeros(requests * CP * QL, WORDS, dtype=torch.int32, device="cuda")
        pre_lens = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
        largs, kwargs = _gather_args(cnts[me], ptrs, me, q_all, lens, pre_lens, rows)
        dl = _deadline(deadline)
        return (
            q_all,
            pre_lens,
            lambda: L._WAIT_ONLY_GATHER[(rows,)](*largs, **kwargs, LAM=True, **dl),
        )

    want_all = torch.empty(requests * CP * QL, WORDS, dtype=torch.int32, device="cuda")
    for c in range(CP):
        for i in range(rows):
            want_all[((i // QL) * CP + c) * QL + i % QL] = q8[c][i]

    for c in range(CP):
        push(c, entry=c % 2 == 0)
    torch.cuda.synchronize()
    for d in range(CP):
        for c in range(CP):
            assert torch.equal(regions[d][:, c], q8[c]), (d, c)
    q_all, pre_lens, launch = wait_half()
    launch()
    torch.cuda.synchronize()
    assert torch.equal(q_all, want_all)
    assert torch.equal(pre_lens, torch.clamp(lens - QL, min=0))
    assert bool((cnts[me] == 6).all())
    assert bool((regions[me] == -1).all()), "the wait half re-arms every polled word"

    def arm(attempt):
        push(me, entry=True)
        q_all, pre_lens, launch = wait_half()

        def land():
            for c in range(CP):
                if c != me:
                    regions[me][:, c].copy_(q8[c])

        def check():
            assert torch.equal(q_all, want_all)
            assert bool((cnts[me] == 7 + attempt).all())
            assert bool((regions[me] == -1).all())

        return land, launch, check

    _land_late(arm)


@pytest.mark.parametrize("deadline", [False, True])
@pytest.mark.parametrize("requests", [2, 5, 9])
def test_the_gathering_q_writer_lands_the_full_gathers_bytes(requests, deadline):
    rows = requests * QL
    g = torch.Generator(device="cuda").manual_seed(100 * requests + 2)
    inputs = [_front_inputs(rows, g) for _ in range(CP)]
    want = [_front(x) for x in inputs]
    q8 = [w[0].view(rows, WORDS * 4).contiguous().view(torch.int32) for w in want]
    regions = [
        torch.full((rows, CP, WORDS), -1, dtype=torch.int32, device="cuda")
        for _ in range(CP)
    ]
    peers = torch.tensor(
        [r.data_ptr() for r in regions], dtype=torch.int64, device="cuda"
    )
    cnts = [torch.full((rows,), 5, dtype=torch.int32, device="cuda") for _ in range(CP)]
    lens = torch.randint(
        QL, 9000, (requests,), dtype=torch.int32, device="cuda", generator=g
    )
    me = 3

    want_all = torch.empty(requests * CP * QL, WORDS, dtype=torch.int32, device="cuda")
    for c in range(CP):
        for i in range(rows):
            want_all[((i // QL) * CP + c) * QL + i % QL] = q8[c][i]

    def push(c, gq=None):
        args = dict(peers=peers, me=c, off=0, cnt=cnts[c], cp=CP, words=WORDS, lam=True)
        if gq is not None:
            args["gather"] = gq
        got = _front(inputs[c], args)
        assert all(torch.equal(a, b) for a, b in zip(want[c], got)), c

    def gather_args():
        q_all = torch.zeros(requests * CP * QL, WORDS, dtype=torch.int32, device="cuda")
        pre_lens = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
        gq = dict(q_all=q_all, seq_lens=lens, pre_lens=pre_lens, ql=QL)
        if deadline:
            gq["tmo"] = TMO_NS  # the in-front gather's poll takes the wait deadline
        return gq

    for c in range(CP):
        if c != me:
            push(c)
    gq = gather_args()
    push(me, gq)
    torch.cuda.synchronize()
    assert torch.equal(gq["q_all"], want_all)
    assert torch.equal(gq["pre_lens"], torch.clamp(lens - QL, min=0))
    assert bool((cnts[me] == 6).all())
    assert bool((regions[me] == -1).all()), "the writer re-arms every polled word"
    for d in range(CP):
        if d != me:
            assert torch.equal(regions[d][:, me], q8[me]), d

    def arm(attempt):
        gq = gather_args()

        def land():
            for c in range(CP):
                if c != me:
                    regions[me][:, c].copy_(q8[c])

        def check():
            assert torch.equal(gq["q_all"], want_all)
            assert bool((cnts[me] == 7 + attempt).all())
            assert bool((regions[me] == -1).all())

        return land, lambda: push(me, gq), check

    _land_late(arm)
