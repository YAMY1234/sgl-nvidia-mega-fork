# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_cp's attention calls, on one SM100-family GPU."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_qwen3_5_cp_precise_fp64 as H  # noqa: E402

pytestmark = pytest.mark.skipif(not H._gpu_ready(), reason="needs an SM100-family GPU")

DRAFT_SCALES = (1.0, 1.0)


class _Attention:
    def __init__(self):
        self.layer = SimpleNamespace(attn=H.layer())


def draft_arm(grp, q8, kv, table, lens, q_len, scales):
    """DraftCP.run (precise) on every rank; None where CP-lite declines."""
    from sgl_nvidia_mega.qwen3_5.attention import route, split_kv
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    d = DraftCP(split_kv, route, True, True, True, precise=True)
    rows = q8.shape[0]
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    kvv = H.kv_view(kv, scales)
    att = _Attention()

    def call(r, q_r):
        return d.run(att, q_r.reshape(rows, H.HQ * H.DIM), meta, kvv, q_len)

    outs = grp.run(call, H.per_rank(q8, grp.cp), rows)
    if any(o is None for o in outs):
        return None
    return torch.cat([o.view(rows, H.HQ, H.DIM) for o in outs], dim=1)


def entry_arm(grp, q8, kv, table, lens, q_len, scales, real_rows=None):
    """attention/draft_cp.attend on every rank's view; None where it declines."""
    from sgl_nvidia_mega.qwen3_5.attention import draft_cp

    rows = q8.shape[0]
    real = rows if real_rows is None else real_rows

    def call(r, q_r):
        view = SimpleNamespace(
            layer=H.layer(),
            q=q_r.reshape(rows, H.HQ, H.DIM),
            k_pool=kv[0],
            v_pool=kv[1],
            k_scale=scales[0],
            v_scale=scales[1],
            page_size=H.PAGE,
            page_table=table,
            cache_seqlens=lens,
            q_len=q_len,
            out_dtype=torch.bfloat16,
            step=H._Fb(),
        )
        return draft_cp.attend(view)

    outs = grp.run(call, H.per_rank(q8, grp.cp), real)
    if any(o is None for o in outs):
        return None
    return torch.cat([o.view(rows, H.HQ, H.DIM) for o in outs], dim=1)


ARMS = {"draft": draft_arm, "entry": entry_arm}


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("q_len", [1, 7])
@pytest.mark.parametrize("b", [2, 4, 9, 13, 16, 32, 64])
@pytest.mark.parametrize(
    "regime,dist", [("wide", "flat"), ("split", "flat"), ("split", "peaky")]
)
def test_the_drafts_calls_are_accepted_and_finite(arm, b, q_len, regime, dist):
    lengths = [max(L, q_len) for L in H.mix(b, 1, regime)]
    q8, kv, table, lens = H.make_inputs(lengths, q_len, dist, seed=17 * b + q_len)
    got = ARMS[arm](H.group(), q8, kv, table, lens, q_len, DRAFT_SCALES)
    assert got is not None, (arm, b, q_len, "CP-lite declined")
    assert bool(torch.isfinite(got.float()).all())


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("q_len", [1, 7])
def test_the_drafts_calls_at_scaled_kv_are_accepted_and_finite(arm, q_len):
    lengths = [max(L, q_len) for L in H.mix(15, 2, "split")]
    q8, kv, table, lens = H.make_inputs(lengths, q_len, "flat", seed=5 + q_len)
    scales = (H.K_SCALE, H.V_SCALE)
    got = ARMS[arm](H.group(), q8, kv, table, lens, q_len, scales)
    assert got is not None
    assert bool(torch.isfinite(got.float()).all())


@pytest.mark.parametrize("q_len", [1, 7])
@pytest.mark.parametrize("length", [32768, 131079, H.CONTEXT])
def test_one_request_runs_precise_in_the_draft_and_stock_through_the_entry(
    q_len, length
):
    q8, kv, table, lens = H.make_inputs([length], q_len, "peaky", seed=length + q_len)
    grp = H.group()
    got = draft_arm(grp, q8, kv, table, lens, q_len, DRAFT_SCALES)
    assert got is not None
    assert bool(torch.isfinite(got.float()).all())
    # one request: CP-lite serves two or more through the entry, SGLang's kernel runs
    assert entry_arm(grp, q8, kv, table, lens, q_len, DRAFT_SCALES) is None


def test_the_entry_serves_a_padded_calls_requests_and_zero_pad_rows():
    grp = H.group()
    lengths = [50000 + 7919 * i for i in range(9)]
    q8, kv, table, lens = H.make_inputs(lengths, 1, "flat", seed=7, pad_rows=7)
    want = entry_arm(grp, q8[:9].contiguous(), kv, table, lens, 1, DRAFT_SCALES)
    got = entry_arm(grp, q8, kv, table, lens, 1, DRAFT_SCALES, real_rows=9)
    assert got is not None and got.shape[0] == 16
    assert torch.equal(got[:9].view(torch.int16), want.view(torch.int16))
    assert not got[9:].view(torch.int16).any()


@pytest.mark.parametrize("q_len", [1, 7])
def test_past_64_requests_both_ways_leave_the_call_to_stock(q_len):
    from sgl_nvidia_mega.qwen3_5.attention import draft_cp, route, split_kv
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    d = DraftCP(split_kv, route, True, True, True, precise=True)
    b = draft_cp.PRECISE_MAX_REQUESTS + 1
    assert not d.takes(q_len, b * q_len)
    lengths = [4096 + 97 * i for i in range(b)]
    q8, kv, table, lens = H.make_inputs(lengths, q_len, "flat", seed=b + q_len)
    assert entry_arm(H.group(), q8, kv, table, lens, q_len, DRAFT_SCALES) is None
