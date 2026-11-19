# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

# the emulated group and the target verify's call
import test_qwen3_5_cp_p2f16_fp64 as T  # noqa: E402

P = T.P
pytestmark = P.pytestmark
logger = logging.getLogger(__name__)
DEADLINE_S = 0.2
RETURNED = 3  # the child's exit code when its call finished (no fault)
TIMEOUT_S = 600
Q_LEN, B, ME, GONE = 7, 4, 0, 2
LENGTHS = [20_000 + 977 * i for i in range(B)]

FAULT_FORMS = {
    "gather": dict(options=dict(min_bs=None, p2f16=True), q=False),
    "gather_wait_half": dict(
        options=dict(min_bs=None, p2f16=True, plain_wait=True), q=False, pushed=True
    ),
    "gather_lamport": dict(
        options=dict(min_bs=None, p2f16=True, plain_wait=True),
        q=False,
        pushed=True,
        lam=True,
    ),
    "merge_spin_f16": dict(options=dict(min_bs=None, p2f16=True)),
    "merge_spin_bf16": dict(options=dict(min_bs=None)),
    "merge_plain_f16": dict(options=dict(min_bs=None, p2f16=True, plain_wait=True)),
    "merge_pushed": dict(
        options=dict(min_bs=None, p2f16=True, plain_wait=True), pushed=True
    ),
    "merge_residual": dict(
        options=dict(min_bs=None, early_merge=False, merge_h=True, precise=True)
    ),
    "merge_r11_plain": dict(options=dict(min_bs=None, early_merge=False)),
    "tp4_gather": dict(tp4=True),
    "tp4_merge": dict(tp4=True, arrived=True),
    "front_gather": dict(front=True),
}


def _call(grp, r, q8, kv, table, lens, options):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    rows = q8.shape[0]
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    q_r = P.per_rank(q8, grp.cp)[r]
    with grp.as_rank(r):
        return split_kv.cp_core(
            P.layer(),
            q_r.reshape(rows, P.HQ * P.DIM),
            P.kv_view(kv),
            meta,
            P._Fb(),
            Q_LEN,
            **options,
        )


def _withhold_partials(grp, r: int, gone: int, rows: int) -> None:
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _LL_PWORDS,
        _MAX_ROWS,
    )

    inst = grp.ranks[r]
    buf = grp.bufs[r].view(torch.int64)
    hr = min(2 * rows, _MAX_ROWS)
    regh = buf[inst.llh_off // 8 :][: hr * grp.cp * _LL_PWORDS]
    regh.view(hr, grp.cp, _LL_PWORDS)[:, gone].zero_()


def _tp4_withheld(arrived: bool) -> None:
    grp = T.tp4_group()
    q8, kv, table, lens = P.make_inputs(LENGTHS, Q_LEN, "flat", seed=5)
    rows = q8.shape[0]
    if arrived:
        T.tp4_arrived(grp, ME, rows, kinds=(0,))
    torch.cuda.synchronize()
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    q_r = T.heads_of(q8, T.TP4_HQ)[ME]
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    with grp.as_rank(ME):
        split_kv.cp_core(
            T.tp4_layer(),
            q_r.reshape(rows, T.TP4_HQ * P.DIM),
            P.kv_view(kv),
            meta,
            P._Fb(),
            Q_LEN,
            **T.core_options(B),
        )


def _front_gather_withheld() -> None:
    import test_qwen3_5_cp_q_push as Q

    rows = B * Q.QL
    g = torch.Generator(device="cuda").manual_seed(11)
    inputs = Q._front_inputs(rows, g)
    regions = [
        torch.full((rows, Q.CP, Q.WORDS), -1, dtype=torch.int32, device="cuda")
        for _ in range(Q.CP)
    ]
    for c in range(Q.CP):
        if c not in (ME, GONE):
            regions[ME][:, c] = c + 1
    gq = dict(
        q_all=torch.zeros(B * Q.CP * Q.QL, Q.WORDS, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(LENGTHS, dtype=torch.int32, device="cuda"),
        pre_lens=torch.zeros(B, dtype=torch.int32, device="cuda"),
        ql=Q.QL,
        tmo=int(DEADLINE_S * 1e9),
    )
    push = dict(
        peers=torch.tensor(
            [r.data_ptr() for r in regions], dtype=torch.int64, device="cuda"
        ),
        me=ME,
        off=0,
        cnt=torch.full((rows,), 5, dtype=torch.int32, device="cuda"),
        cp=Q.CP,
        words=Q.WORDS,
        lam=True,
        gather=gq,
    )
    torch.cuda.synchronize()
    Q._front(inputs, push)


def withheld(form: str) -> None:
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    spec = FAULT_FORMS[form]
    L.set_wait_deadline(DEADLINE_S)
    if spec.get("tp4") or spec.get("front"):
        if spec.get("tp4"):
            _tp4_withheld(spec.get("arrived", False))
        else:
            _front_gather_withheld()
        torch.cuda.synchronize()
        sys.exit(RETURNED)
    if spec.get("lam"):
        with mock.patch.object(L.CpLite, "lam_q", True):
            grp = P.CpGroup()
    else:
        grp = P.CpGroup()
    q8, kv, table, lens = P.make_inputs(LENGTHS, Q_LEN, "flat", seed=5)
    rows = q8.shape[0]
    inst = grp.ranks[ME]
    if spec.get("lam"):
        inst.lam_q = True
        assert inst.lam_q_end > inst.lamq_off
    if spec.get("pushed"):
        inst.o_push = True
        assert inst.q_push_args(rows, B, Q_LEN) is not None
    if spec.get("q", True):
        grp.publish(ME, P.per_rank(q8, grp.cp), rows)
        if spec.get("pushed"):
            T._publish_own_q(grp, ME, P.per_rank(q8, grp.cp)[ME], rows)
        _withhold_partials(grp, ME, GONE, rows)
    torch.cuda.synchronize()
    _call(grp, ME, q8, kv, table, lens, spec["options"])
    torch.cuda.synchronize()
    sys.exit(RETURNED)


@pytest.mark.parametrize("form", sorted(FAULT_FORMS))
def test_a_withheld_peer_ends_the_wait_with_a_fault(form):
    t0 = time.monotonic()
    try:
        res = subprocess.run(
            [sys.executable, __file__, "withheld", form],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{form}: the call hung past {TIMEOUT_S} s (no deadline)")
    took = time.monotonic() - t0
    text = res.stdout + res.stderr
    assert res.returncode not in (0, RETURNED), (form, res.returncode, text[-2000:])
    assert (
        "illegal memory access" in text
        or "cudaErrorIllegalAddress" in text
        or "unspecified launch failure" in text
        or "an illegal instruction" in text
    ), (form, took, text[-2000:])


@pytest.mark.parametrize("form", ["spin", "plain", "pushed"])
@pytest.mark.parametrize("b", [2, 16, 64])
def test_with_every_peer_present_the_deadline_changes_no_byte(b, form):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    lengths = [max(x, Q_LEN) for x in P.mix(b, 1, "split")]
    q8, kv, table, lens = P.make_inputs(lengths, Q_LEN, "peaky", seed=41 * b)
    grp = P.group()
    L.set_wait_deadline(None)
    off = T.target_call(grp, q8, kv, table, lens, form)
    try:
        L.set_wait_deadline(30.0)
        on = T.target_call(grp, q8, kv, table, lens, form)
    finally:
        L.set_wait_deadline(None)
    assert off is not None and on is not None
    assert torch.equal(on.view(torch.int16), off.view(torch.int16)), (b, form)


@pytest.mark.parametrize(
    "options",
    [
        dict(min_bs=None, early_merge=False, merge_h=True, precise=True),
        dict(min_bs=None, early_merge=False),
        dict(precise=True),
    ],
    ids=["residual", "r11_plain", "one_request"],
)
def test_the_other_merge_forms_give_the_same_bytes_with_the_deadline(options):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    b = 1 if options == dict(precise=True) else 9
    lengths = [max(x, Q_LEN) for x in P.mix(b, 2, "split")] if b > 1 else [100003]
    q8, kv, table, lens = P.make_inputs(lengths, Q_LEN, "flat", seed=53 + b)
    grp = P.group()
    rows = q8.shape[0]
    fb = P._Fb()
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)

    def run():
        from sgl_nvidia_mega.qwen3_5.attention import split_kv

        def call(r, q_r):
            return split_kv.cp_core(
                P.layer(),
                q_r.reshape(rows, P.HQ * P.DIM),
                P.kv_view(kv),
                meta,
                fb,
                Q_LEN,
                **options,
            )

        outs = grp.run(call, P.per_rank(q8, grp.cp), rows)
        assert all(o is not None for o in outs)
        return torch.cat([o.view(rows, P.HQ, P.DIM) for o in outs], dim=1)

    L.set_wait_deadline(None)
    off = run()
    try:
        L.set_wait_deadline(30.0)
        on = run()
    finally:
        L.set_wait_deadline(None)
    assert torch.equal(on.view(torch.int16), off.view(torch.int16))


@pytest.mark.parametrize("b", [1, 9, 64, 128])
def test_tp4s_fold_path_gives_the_same_bytes_with_the_deadline(b):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    lengths = [100003] if b == 1 else [max(x, Q_LEN) for x in T.short_mix(b, 3)]
    q8, kv, table, lens = P.make_inputs(lengths, Q_LEN, "flat", seed=61 + b)
    grp = T.tp4_group()
    options = T.core_options(b)
    L.set_wait_deadline(None)
    off = T.tp4_call(grp, q8, kv, table, lens, options)
    again = T.tp4_call(grp, q8, kv, table, lens, options)
    try:
        L.set_wait_deadline(30.0)
        on = T.tp4_call(grp, q8, kv, table, lens, options)
    finally:
        L.set_wait_deadline(None)
    assert off is not None and on is not None
    assert torch.equal(again.view(torch.int16), off.view(torch.int16)), b
    assert torch.equal(on.view(torch.int16), off.view(torch.int16)), b


# the per-head merge's forms in the target verify's ungated call (FP16 partials)
TIMING_FORMS = {
    "spin": dict(EARLY=True, F16=True),
    "plain": dict(EARLY=True, F16=True, PLAIN_WAIT=True),
    "pushed": dict(EARLY=True, F16=True, PLAIN_WAIT=True, CG=True, NOPUSH=True),
}


def _merge_timing(requests: int, form: str, deadline: bool, late: bool) -> float:
    import triton

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _LL_PWORDS,
        _PUSHED_MERGE_SLEEP_NS,
        _cp_merge_h_ll,
    )

    heads, dim, page, cp, me = P.HQ, P.DIM, P.PAGE, P.CP, 1
    rows, no = requests * Q_LEN, heads * dim // 2
    g = torch.Generator(device="cuda").manual_seed(requests)
    q = (torch.randn(rows, heads, dim, device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    po = (torch.randn(cp, rows, heads, dim, device="cuda", generator=g) * 0.1).to(
        torch.float16
    )
    plse = torch.randn(cp, rows, heads, device="cuda", generator=g)
    pages = 5
    k, v = (
        (
            torch.randn(requests * pages, 1, page, dim, device="cuda", generator=g) * 2
        ).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    table = torch.randperm(requests * pages, device="cuda", generator=g)
    table = table.to(torch.int32).view(requests, pages)
    lengths = torch.full(
        (requests,), 4 * page + Q_LEN, dtype=torch.int32, device="cuda"
    )
    out = torch.empty(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    counter = torch.zeros(rows * heads, dtype=torch.int32, device="cuda")
    ll = torch.zeros(rows, cp, _LL_PWORDS, dtype=torch.int64, device="cuda")
    sinks = [torch.zeros_like(ll) for _ in range(cp)]
    pushed = form == "pushed"

    def mailbox(c):
        # the pushed form reads only its own mailbox; the others push into the members'
        if c == me:
            return ll.data_ptr()
        return 0 if pushed else sinks[c].data_ptr()

    peers = torch.tensor(
        [mailbox(c) for c in range(cp)], dtype=torch.int64, device="cuda"
    )
    o32 = po.view(torch.int32).view(cp, rows, no).long() & 0xFFFFFFFF
    l32 = plse.view(torch.int32).long() & 0xFFFFFFFF
    sources = [c for c in range(cp) if pushed or c != me]

    def land(epoch):
        flag = epoch << 32
        for c in sources:
            ll[:, c, :no] = flag | o32[c]
            ll[:, c, no : no + heads] = flag | l32[c]

    options = dict(TIMING_FORMS[form])
    if pushed:
        options["MSLEEP"] = _PUSHED_MERGE_SLEEP_NS
    if deadline:
        options.update(tmo=30_000_000_000, DEADLINE=True)

    def launch():
        _cp_merge_h_ll[(rows, heads)](
            po.view(torch.int32),
            plse.view(torch.int32),
            no,
            po.stride(0) // 2,
            heads,
            plse.stride(0),
            q,
            q.stride(0),
            k,
            v,
            k.stride(0),
            k.stride(2),
            table,
            table.stride(0),
            lengths,
            out,
            out.stride(0),
            peers,
            me,
            0,
            counter,
            0.75 * dim**-0.5,
            0.5,
            requests,
            0,
            QL=Q_LEN,
            HQ=heads,
            CP=cp,
            D=dim,
            PG=page,
            NWP=_LL_PWORDS,
            NB=triton.next_power_of_2(requests),
            GMAX=False,
            WAIT=True,
            num_warps=4,
            launch_pdl=True,
            **options,
        )

    main = torch.cuda.current_stream()
    side = torch.cuda.Stream(priority=-1)
    torch.cuda._sleep(1)
    land(0)
    torch.cuda.synchronize()
    times, first_landed, it = [], 0, 0
    while len(times) < 50:
        assert it < 120, f"the words landed first at {first_landed} of {it} launches"
        start, landed, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        if late:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                torch.cuda._sleep(1 << 21)
                landed.record()
                land(it + 1)
        else:
            land(it + 1)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        it += 1
        if late:
            assert landed.elapsed_time(end) > 0, "the merge ended before the words"
            if not start.elapsed_time(landed) > 0:
                first_landed += 1
                continue
        if it > 10:
            first = landed if late else start
            times.append(first.elapsed_time(end) * 1e3)
    times.sort()
    return times[len(times) // 2]


@pytest.mark.parametrize("form", sorted(TIMING_FORMS))
@pytest.mark.parametrize(
    "requests, late",
    [(4, False), (16, False), (64, False), (4, True), (16, True)],
)
def test_the_deadline_costs_the_merge_a_timer_read(requests, late, form):
    off = _merge_timing(requests, form, False, late)
    on = _merge_timing(requests, form, True, late)
    logger.info(
        "per-head merge (%s), %d requests, words landed %s: %.2f us without the "
        "deadline, %.2f us with it",
        form,
        requests,
        "while it polls" if late else "first",
        off,
        on,
    )
    assert on <= 1.25 * off + 2.0, (requests, late, form, off, on)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "withheld":
    withheld(sys.argv[2])
