# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

HEADS, DIM, PAGE = 4, 256, 64
SCALE = DIM**-0.5


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")
FORMS = {"bf16": (torch.bfloat16, {}), "p2f16": (torch.float16, {"F16": True})}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("cp", [1, 4])
@pytest.mark.parametrize("requests", [4, 6, 8, 12, 16])
def test_cp_verify_merge_launch_variants_give_the_same_bytes(requests, cp, form):
    import triton

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _LL_PWORDS,
        _cp_merge_h_ll,
    )

    q_len, heads, dim, page, words = 7, HEADS, DIM, PAGE, _LL_PWORDS
    rows, me = requests * q_len, (1 if cp == 4 else 0)
    pdt, f16 = FORMS[form]
    g = torch.Generator(device="cuda").manual_seed(1000 + 10 * requests + cp)
    q = (torch.randn(rows, heads, dim, device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    po = (torch.randn(cp, rows, heads, dim, device="cuda", generator=g) * 0.1).to(pdt)
    plse = torch.randn(cp, rows, heads, device="cuda", generator=g)
    peer_o = (torch.randn(cp, rows, heads, dim, device="cuda", generator=g) * 0.1).to(
        pdt
    )
    peer_lse = torch.randn(cp, rows, heads, device="cuda", generator=g)
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
        (requests,), 4 * page + q_len, dtype=torch.int32, device="cuda"
    )
    gate = torch.randn(rows, heads * dim, device="cuda", generator=g).to(torch.bfloat16)
    o_scale = torch.tensor([0.05], device="cuda")

    def published():
        ll = torch.zeros(rows, cp, words, dtype=torch.int64, device="cuda")
        flag = 1 << 32
        o32 = peer_o.view(torch.int32).view(cp, rows, heads * dim // 2)
        lse32 = peer_lse.view(torch.int32)
        for c in range(cp):
            if c == me:
                continue
            ll[:, c, : heads * dim // 2] = flag | (o32[c].to(torch.int64) & 0xFFFFFFFF)
            ll[:, c, heads * dim // 2 : heads * dim // 2 + heads] = flag | (
                lse32[c].to(torch.int64) & 0xFFFFFFFF
            )
        return ll

    def run(kernel, epilogue, late=False, **options):
        options.update(f16)
        ll = published()
        if late:
            peer_words, ll = ll, torch.zeros_like(ll)
            torch.cuda._sleep(1)
            torch.empty_like(ll)[:, 1 - me].copy_(peer_words[:, 1 - me])
        sinks = [torch.zeros_like(ll) for _ in range(cp)]
        peers = torch.tensor(
            [(ll if c == me else sinks[c]).data_ptr() for c in range(cp)],
            dtype=torch.int64,
            device="cuda",
        )
        counter = torch.zeros(rows * heads, dtype=torch.int32, device="cuda")
        out = torch.empty(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
        oq = torch.zeros(rows, heads * dim, dtype=torch.float8_e4m3fn, device="cuda")
        if late:
            started, landed, done = (
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            )
            side = torch.cuda.Stream(priority=-1)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                torch.cuda._sleep(1 << 21)
                landed.record()
                for c in range(cp):
                    if c != me:
                        ll[:, c].copy_(peer_words[:, c])
            started.record()
        if epilogue:
            options.update(
                gate=gate,
                gate_stride=gate.stride(0),
                o_proj_input_scale=o_scale,
                o_proj_input=oq,
                o_proj_input_stride=oq.stride(0),
                GATE_EPILOGUE=True,
            )
        kernel[(rows, heads)](
            po.view(torch.int32),
            plse.view(torch.int32),
            heads * dim // 2,
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
            QL=q_len,
            HQ=heads,
            CP=cp,
            D=dim,
            PG=page,
            NWP=words,
            NB=triton.next_power_of_2(requests),
            GMAX=False,
            WAIT=True,
            num_warps=4,
            launch_pdl=True,
            **options,
        )
        if late:
            done.record()
        torch.cuda.synchronize()
        if late:
            assert started.elapsed_time(landed) > 0, "the words landed first"
            assert landed.elapsed_time(done) > 0, "the merge ended before the words"
        assert torch.isfinite(out.float()).all()
        assert bool((counter == 1).all())
        # this rank's pushes into the other members' regions come last
        return [out.view(torch.int16), oq.view(torch.uint8), *sinks]

    def same(a, b):
        return all(torch.equal(x, y) for x, y in zip(a, b))

    # the early spin merge, the late launch, the plain wait, both
    early = run(_cp_merge_h_ll, True, EARLY=True)
    assert early[1].any()
    assert same(run(_cp_merge_h_ll, True, EARLY=False), early)
    assert same(run(_cp_merge_h_ll, True, EARLY=True, PLAIN_WAIT=True), early)
    assert same(run(_cp_merge_h_ll, True, EARLY=False, PLAIN_WAIT=True), early)
    rot = {"PLAIN_WAIT": True, "PUSH_ROT": True}
    assert same(run(_cp_merge_h_ll, True, EARLY=True, **rot), early)
    assert same(run(_cp_merge_h_ll, True, EARLY=True, CG=True), early)
    assert same(run(_cp_merge_h_ll, True, EARLY=True, CG=True, **rot), early)
    if cp > 1:
        for options in (
            {"EARLY": True},
            {"EARLY": False},
            {"EARLY": True, "PLAIN_WAIT": True},
            {"EARLY": False, "PLAIN_WAIT": True},
            {"EARLY": True, **rot},
            {"EARLY": True, "CG": True, **rot},
        ):
            assert same(run(_cp_merge_h_ll, True, late=True, **options), early)
    # no epilogue: the per-head merge at either ordering and wait
    bare = run(_cp_merge_h_ll, False, EARLY=True)
    assert torch.equal(bare[0], early[0])
    assert same(run(_cp_merge_h_ll, False, EARLY=False), bare)
    assert same(run(_cp_merge_h_ll, False, EARLY=True, PLAIN_WAIT=True), bare)


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("requests", [2, 4, 8, 12, 16])
def test_cp_verify_plain_wait_merge_rounds_as_the_spin_merge(requests, form):
    import triton

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
        _LL_PWORDS,
        _cp_merge_h_ll,
    )

    q_len, heads, dim, page, words, cp, me = 7, HEADS, DIM, PAGE, _LL_PWORDS, 4, 1
    pdt, f16 = FORMS[form]
    rows, no, max_pages = requests * q_len, heads * dim // 2, 5
    others = [c for c in range(cp) if c != me]
    prefix_choices = (17, 64, 65, 150, 200, 256, 300)
    trials = math.ceil(1_500_000 / (rows * heads * dim))
    sleep_us = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda._sleep(1)
    sleep_us[0].record()
    torch.cuda._sleep(1 << 16)
    sleep_us[1].record()
    sleep_us[1].synchronize()
    cycles_us = (1 << 16) / (sleep_us[0].elapsed_time(sleep_us[1]) * 1e3)
    side = torch.cuda.Stream(priority=-1)
    # load the publisher's kernels before any merge polls (lazy loading would block)
    scratch = torch.zeros(rows, cp, words, dtype=torch.int64, device="cuda")
    scratch[:, 0, :no].copy_(scratch[:, 2, :no])
    torch.cuda.synchronize()
    bad = {}
    for trial in range(trials):
        g = torch.Generator(device="cuda").manual_seed(2000 + 1000 * requests + trial)
        prefixes = [prefix_choices[(j + trial) % 7] for j in range(requests)]
        lengths = torch.tensor(
            [p + q_len for p in prefixes], dtype=torch.int32, device="cuda"
        )
        n_pages = torch.tensor(
            [(p + page - 1) // page for p in prefixes], device="cuda"
        ).repeat_interleave(q_len)
        q = (torch.randn(rows, heads, dim, device="cuda", generator=g) * 2).to(
            torch.float8_e4m3fn
        )
        k, v = (
            (
                torch.randn(
                    requests * max_pages, 1, page, dim, device="cuda", generator=g
                )
                * 2
            ).to(torch.float8_e4m3fn)
            for _ in range(2)
        )
        table = torch.randperm(requests * max_pages, device="cuda", generator=g)
        table = table.to(torch.int32).view(requests, max_pages)
        bmm1, bmm2 = SCALE, 1.0
        # each row's causal-block max (natural log) over this step's keys
        pos = (lengths - q_len).long()[:, None] + torch.arange(q_len, device="cuda")
        slot = table.gather(1, (pos // page).to(torch.int64)).long()
        keys = k[slot, 0, pos % page].float()  # [requests, q_len keys, dim]
        scores = torch.einsum(
            "rthd,rkd->rthk", q.float().view(requests, q_len, heads, dim), keys
        )
        causal = torch.ones(q_len, q_len, device="cuda").tril().bool()
        scores = scores.masked_fill(~causal[None, :, None, :], float("-inf")) * bmm1
        peak = scores.amax(-1).view(rows, heads)

        def partials():
            o = (
                torch.randn(cp, rows, heads, dim, device="cuda", generator=g) * 0.5
            ).to(pdt)
            lse = (
                peak[None] + torch.randn(cp, rows, heads, device="cuda", generator=g)
            ) / math.log(2.0)
            owns = (n_pages[None, :] > torch.arange(cp, device="cuda")[:, None])[
                ..., None
            ]
            if trial % 2 == 0:
                o = o.masked_fill(~owns[..., None], 0.0)
                lse = lse.masked_fill(~owns, float("-inf"))
            else:
                junk = torch.randn(cp, rows, heads, device="cuda", generator=g) * 30
                lse = torch.where(owns, lse, junk)
            return o, lse

        po, plse = partials()  # this rank's partials (pushed to every group rank)
        peer_o, peer_lse = partials()  # the other sources' partials for this rank
        epoch = 1000 + trial
        flag = epoch << 32
        published = torch.zeros(rows, cp, words, dtype=torch.int64, device="cuda")
        o32 = peer_o.view(torch.int32).view(cp, rows, no).long() & 0xFFFFFFFF
        l32 = peer_lse.view(torch.int32).long() & 0xFFFFFFFF
        for c in others:
            published[:, c, :no] = flag | o32[c]
            published[:, c, no : no + heads] = flag | l32[c]
        stale = torch.randint(
            0, 2**32, (rows, cp, words), dtype=torch.int64, device="cuda", generator=g
        )
        stale |= (epoch - 1) << 32
        gate = torch.randn(rows, heads * dim, device="cuda", generator=g).to(
            torch.bfloat16
        )
        o_scale = torch.tensor([0.05], device="cuda")

        def run(epilogue, late, **options):
            options.update(f16)
            ll = stale.clone()
            if not late:
                for c in others:
                    ll[:, c] = published[:, c]
            sinks = [torch.zeros_like(ll) for _ in range(cp)]
            peers = torch.tensor(
                [(ll if c == me else sinks[c]).data_ptr() for c in range(cp)],
                dtype=torch.int64,
                device="cuda",
            )
            counter = torch.full(
                (rows * heads,), epoch - 1, dtype=torch.int32, device="cuda"
            )
            out = torch.empty(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
            oq = torch.zeros(
                rows, heads * dim, dtype=torch.float8_e4m3fn, device="cuda"
            )
            started, first, last, done = (
                torch.cuda.Event(enable_timing=True) for _ in range(4)
            )
            started.record()
            if late:
                lse_words, lo, hi = (
                    slice(no, no + heads),
                    slice(0, no // 2),
                    slice(no // 2, no),
                )
                side.wait_event(started)
                with torch.cuda.stream(side):
                    torch.cuda._sleep(int(100 * cycles_us))
                    first.record()
                    for c, parts in zip(
                        others,
                        ((lse_words, lo, hi), (lo, hi, lse_words), (lo, lse_words, hi)),
                    ):
                        for part in parts:
                            ll[:, c, part].copy_(published[:, c, part])
                            torch.cuda._sleep(int(3 * cycles_us))
                    last.record()
            if epilogue:
                options.update(
                    gate=gate,
                    gate_stride=gate.stride(0),
                    o_proj_input_scale=o_scale,
                    o_proj_input=oq,
                    o_proj_input_stride=oq.stride(0),
                    GATE_EPILOGUE=True,
                )
            _cp_merge_h_ll[(rows, heads)](
                po.view(torch.int32),
                plse.view(torch.int32),
                heads * dim // 2,
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
                bmm1,
                bmm2,
                requests,
                0,
                QL=q_len,
                HQ=heads,
                CP=cp,
                D=dim,
                PG=page,
                NWP=words,
                NB=triton.next_power_of_2(requests),
                GMAX=False,
                WAIT=True,
                num_warps=4,
                launch_pdl=True,
                **options,
            )
            done.record()
            torch.cuda.synchronize()
            if late:
                assert started.elapsed_time(first) > 0, "the words landed first"
                assert last.elapsed_time(done) > 0, "the merge ended before the words"
            assert bool((counter == epoch).all())
            assert torch.isfinite(out.float()).all()
            return out.view(torch.int16), oq.view(torch.uint8)

        for epilogue in (True, False):
            stock = run(epilogue, False, EARLY=True)
            for name, options in (
                ("late launch", {"EARLY": False}),
                ("plain wait", {"EARLY": True, "PLAIN_WAIT": True}),
                ("both", {"EARLY": False, "PLAIN_WAIT": True}),
            ):
                got = run(epilogue, True, **options)
                key = (name, epilogue)
                n = bad.setdefault(key, [0, 0])
                n[0] += int((got[0] != stock[0]).sum())
                n[1] += int((got[1] != stock[1]).sum())
    assert all(n == [0, 0] for n in bad.values()), {
        f"{name}, epilogue {epilogue}": f"{n[0]} merged values, {n[1]} o_proj inputs differ"
        for (name, epilogue), n in bad.items()
    }
