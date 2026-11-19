# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

import test_qwen3_5_cp_precise_fp64 as P  # noqa: E402

pytestmark = P.pytestmark
Q_LEN = 7
FORMS = ("spin", "plain", "pushed")


TP4_CP, TP4_HQ = 2, 8  # TP4's CP-lite: cp_lite_topology(4)


def tiny_mix(b: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(7000 + b + seed)
    tiny = [int(x) for x in torch.randint(129, 1025, ((b + 1) // 2,), generator=g)]
    long = [int(x) for x in torch.randint(100000, P.CONTEXT, (b // 2,), generator=g)]
    return [x for pair in zip(tiny, long + [0]) for x in pair if x][:b]


def short_mix(b: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(9000 + b + seed)
    lens = [int(x) for x in torch.randint(1024, 16384, (b,), generator=g)]
    for i in range(0, b, 16):
        lens[i] = int(torch.randint(32768, 65536, (1,), generator=g))
    return lens


def _publish_own_q(grp, r: int, q_r: torch.Tensor, rows: int) -> None:
    inst = grp.ranks[r]
    words = grp.hq * P.DIM // 4
    buf = grp.bufs[r].view(torch.int64)
    reg = buf[inst.llq_off // 8 :][: rows * grp.cp * words].view(rows, grp.cp, words)
    e_q = inst.cnt[0][:rows] + 1
    q32 = q_r[:rows].reshape(rows, -1).view(torch.int32)
    reg[:, r, :] = grp._pack(e_q[:, None], q32)


def target_call(grp, q8, kv, table, lens, form: str, q_len: int = Q_LEN):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    attn, kvv = P.layer(), P.kv_view(kv)
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    rows, b = q8.shape[0], lens.shape[0]
    if b > 1:
        options = dict(min_bs=None, p2f16=True)
        if form != "spin":
            options["plain_wait"] = True  # split_kv.merge_options() under the slot
    else:
        options = dict(precise=True)
    pushed = form == "pushed"
    for inst in grp.ranks:
        inst.o_push = pushed
    fb = P._Fb()

    def call(r, q_r):
        if pushed:
            assert grp.ranks[r].q_push_args(rows, b, q_len) is not None
            _publish_own_q(grp, r, q_r, rows)
        return split_kv.cp_core(
            attn, q_r.reshape(rows, P.HQ * P.DIM), kvv, meta, fb, q_len, **options
        )

    try:
        outs = grp.run(call, P.per_rank(q8, grp.cp), rows)
    finally:
        for inst in grp.ranks:
            inst.o_push = False
    if any(o is None for o in outs):
        return None
    return torch.cat([o.view(rows, P.HQ, P.DIM) for o in outs], dim=1)


def heads_of(q8: torch.Tensor, hq: int) -> list[torch.Tensor]:
    return [q8[:, r * hq : (r + 1) * hq].contiguous() for r in range(P.HEADS // hq)]


MIXES = {
    "wide_flat": (lambda b, s: P.mix(b, s, "wide"), "flat"),
    "split_flat": (lambda b, s: P.mix(b, s, "split"), "flat"),
    "split_peaky": (lambda b, s: P.mix(b, s, "split"), "peaky"),
    "tiny_flat": (tiny_mix, "flat"),
    "tiny_peaky": (tiny_mix, "peaky"),
}


@pytest.mark.parametrize("mix", sorted(MIXES))
@pytest.mark.parametrize("b", [2, 9, 16, 33, 64])
def test_every_form_of_the_multi_request_verify_gives_the_same_bytes(b, mix):
    make, dist = MIXES[mix]
    lengths = [max(L, Q_LEN) for L in make(b, 0)]
    q8, kv, table, lens = P.make_inputs(lengths, Q_LEN, dist, seed=17 * b + len(mix))
    grp = P.group()
    got = {form: target_call(grp, q8, kv, table, lens, form) for form in FORMS}
    for form, out in got.items():
        assert out is not None, (b, mix, form, "CP-lite declined")
    for form in FORMS[1:]:
        assert torch.equal(
            got[form].view(torch.int16), got["spin"].view(torch.int16)
        ), (b, mix, form)
    assert bool(torch.isfinite(got["spin"].float()).all())


_TP4: list = []


def tp4_group():
    if not _TP4:
        grp = P.CpGroup(cp=TP4_CP, hq=TP4_HQ)
        assert not any(inst.ll or inst.exact for inst in grp.ranks)
        _TP4.append(grp)
    return _TP4[0]


def tp4_layer():
    return SimpleNamespace(**{**vars(P.layer()), "tp_q_head_num": TP4_HQ})


def tp4_flags(grp, r: int) -> torch.Tensor:
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import _KINDS, _MAX_ROWS

    inst = grp.ranks[r]
    n = _KINDS * _MAX_ROWS * grp.cp
    words = grp.bufs[r][inst.f_off : inst.f_off + 4 * n].view(torch.int32)
    return words.view(_KINDS * _MAX_ROWS, grp.cp)


def tp4_arrived(grp, r: int, rows: int, kinds=(0, 1)) -> None:
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import _MAX_ROWS

    flags = tp4_flags(grp, r)
    for k in kinds:
        flags[k * _MAX_ROWS : k * _MAX_ROWS + rows, r + 1 :] = 1


def tp4_call(grp, q8, kv, table, lens, options, q_len: int = Q_LEN):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    rows = q8.shape[0]
    attn, kvv = tp4_layer(), P.kv_view(kv)
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    fb = P._Fb()
    qs = heads_of(q8, TP4_HQ)
    for r in range(grp.cp):
        tp4_flags(grp, r).zero_()
        tp4_arrived(grp, r, rows)
    outs = None
    for _ in range(2):
        outs = []
        for r in range(grp.cp):
            with grp.as_rank(r):
                outs.append(
                    split_kv.cp_core(
                        attn,
                        qs[r].reshape(rows, TP4_HQ * P.DIM),
                        kvv,
                        meta,
                        fb,
                        q_len,
                        **options,
                    )
                )
    if any(o is None for o in outs):
        return None
    return torch.cat([o.view(rows, TP4_HQ, P.DIM) for o in outs], dim=1)


def core_options(b: int) -> dict:
    if b > 1:
        return dict(min_bs=None, p2f16=True, plain_wait=True)
    return dict(precise=True)
