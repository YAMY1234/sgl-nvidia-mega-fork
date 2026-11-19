# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime  # noqa: F401
from sgl_nvidia_mega.qwen3_5.attention import split_kv
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L


def _cp(**over):
    cp = L.CpLite.__new__(L.CpLite)
    fields = dict(
        ll=True,
        cp_ungated=True,
        exact=True,
        min_kv={},
        min_bs=2,
        peers=torch.tensor([1 << 40, 2 << 40, 3 << 40, 4 << 40]),
        me=1,
        llq_off=4096,
        llh_off=1 << 20,
        cnt=["cnt0"],
        cnt_h="cnt_h",
        cp=4,
        hq=4,
        sms=152,
        o_push=True,
        early_merge=True,
        gate_max=False,
        q_all=torch.zeros(4, dtype=torch.int8),
        pre_lens="pre",
        stock_lens="stock",
        gather_pdl=True,
        device="cpu",
        _gate_epilogue=None,
        _logged=set(),
        _exact_supported=lambda q_len, b: True,
    )
    fields.update(over)
    for k, v in fields.items():
        setattr(cp, k, v)
    return cp


def _attend(cp, b, q_len, **kw):
    seen = {}

    def ungated(*args, **kwargs):
        seen.update(kwargs)
        return "ran"

    rows = b * q_len
    q = torch.zeros(rows, cp.hq, L.HEAD_DIM, dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(tp_k_head_num=1, sliding_window_size=None, layer_id=3)
    with patch.object(L.CpLite, "_ungated_main", side_effect=ungated):
        res = cp.attend(
            q,
            (None, None),
            torch.zeros(b, 1, dtype=torch.int32),
            torch.full((b,), 65536, dtype=torch.int32),
            q_len,
            1.0,
            1.0,
            None,
            layer,
            None,
            **kw,
        )
    assert res == "ran"
    return seen


@pytest.mark.parametrize(
    "b, q_len, hilo, p_hilo",
    [
        (15, 1, True, True),
        (15, 7, True, True),  # a draft extend
        (64, 2, True, True),
        (64, 7, True, True),
        (33, 14, True, True),
        (65, 2, False, False),
        (100, 7, False, False),
    ],
)
def test_precise_features_stay_on_the_one_slot_plans(b, q_len, hilo, p_hilo):
    if 2 * b * q_len > L._MAX_ROWS:
        hilo = False
    kw = _attend(_cp(), b, q_len, precise=True)
    assert (kw["hilo"], kw["p_hilo"]) == (hilo, p_hilo)
    assert kw["plain_wait"] is True
    off = _attend(_cp(), b, q_len)
    assert (off["hilo"], off["p_hilo"], off["plain_wait"]) == (False, False, False)


def test_the_precise_bound_is_the_wide_plans():
    assert L._PRECISE_MAX_BATCH == F._WIDE_PLAN_MAX_BATCH


def test_attend_writes_into_out_and_refuses_a_foreign_one():
    cp = _cp()
    out = torch.zeros(14, 4, L.HEAD_DIM, dtype=torch.bfloat16)
    kw = _attend(cp, 2, 7, out=out)
    assert kw["out"] is out
    layer = SimpleNamespace(tp_k_head_num=1, sliding_window_size=None, layer_id=3)
    q = torch.zeros(14, 4, L.HEAD_DIM, dtype=torch.float8_e4m3fn)
    bad = torch.zeros(14, 4, L.HEAD_DIM, dtype=torch.float32)
    lens = torch.full((2,), 65536, dtype=torch.int32)
    assert (
        cp.attend(
            q,
            (None, None),
            lens[:, None],
            lens,
            7,
            1.0,
            1.0,
            None,
            layer,
            None,
            out=bad,
        )
        is None
    )


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


_GATE = dict(nb=4, thr=0, GATE=False, NB=4, GMAX=False)


def _ungated(monkeypatch, *, b=4, pushed=False, **kw):
    monkeypatch.setattr(L, "_cp_gather_q_ll", _Launch())
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", _Launch())
    cp = _cp()
    partials, merges = [], []
    cp._exact_partial = lambda *a, **k: partials.append(k)
    cp._merge_h = lambda *a, **k: merges.append((a, k))
    rows = 7 * b
    if pushed:
        cp._q_pushed = rows
    q = torch.zeros(rows, 4, 256, dtype=torch.int8)
    out = cp._ungated_main(
        q, "k", "v", "table", "lens", 7, b, rows, 1.0, 1.0, _GATE, **kw
    )
    return out, partials[0], merges[0]


@pytest.mark.parametrize(
    "kw, hilo",
    [
        (dict(hilo=True, p_hilo=True), True),
        (dict(hilo=True, p_hilo=True, early_merge=False), False),
        # the per-head merge kept late (merge_h): the residual partials reach it
        (dict(hilo=True, p_hilo=True, early_merge=False, merge_h=True), True),
        (dict(), False),
    ],
)
def test_the_ungated_call_hands_the_precise_features_on(monkeypatch, kw, hilo):
    p_hilo = kw.get("p_hilo", False)
    _, partial, (_, merge) = _ungated(monkeypatch, plain_wait=True, **kw)
    assert partial["o_hilo"] is hilo and partial["p_hilo"] is p_hilo
    assert merge["hilo"] is hilo
    assert merge["plain"] is (kw.get("early_merge") is False and not kw.get("merge_h"))


def test_a_precise_call_keeps_the_merges_own_push(monkeypatch):
    _, partial, (_, merge) = _ungated(
        monkeypatch, pushed=True, plain_wait=True, hilo=True, p_hilo=True
    )
    assert partial["o_push"] is False and merge["nopush"] is False
    _, partial, (_, merge) = _ungated(monkeypatch, pushed=True, plain_wait=True)
    assert partial["o_push"] is True and merge["nopush"] is True


def test_the_ungated_call_merges_into_out_and_logs_once(monkeypatch, caplog):
    out = torch.zeros(28, 4, 256, dtype=torch.bfloat16)
    with caplog.at_level(logging.INFO):
        got, _, (args, _) = _ungated(
            monkeypatch, plain_wait=True, hilo=True, p_hilo=True, out=out
        )
    assert got is out and args[9] is out
    assert "precise call at 28 rows: two-term E4M3 P True" in caplog.text


def _merge(monkeypatch, rows=14, **kw):
    launch = _Launch()
    monkeypatch.setattr(L, "_cp_merge_h_ll", launch)
    monkeypatch.setattr(L, "_cp_merge_h_ll_plain", launch)
    me = SimpleNamespace(
        cp_pdl=True,
        merge_cg=False,
        hq=4,
        cp=4,
        _logged=set(),
        po_x=torch.zeros(2, 8, dtype=torch.int32),
        pl_x=torch.zeros(2, 4),
        peers=None,
        me=0,
        llh_off=0,
        cnt_h=None,
    )
    t = torch.zeros(1, 1, 1, 1)
    gate = {"nb": 0, "thr": 0, "NB": 1, "GMAX": 1}
    args = (t, t, t, t, t, 7, rows, 1.0, 1.0, t, gate)
    L.CpLite._merge_h(me, *args, **kw)
    ((grid, _, opts),) = launch.calls
    return grid, opts


def test_the_per_head_merge_takes_hilo(monkeypatch):
    grid, opts = _merge(monkeypatch, early=True, plain_wait=True, hilo=True)
    assert opts["HILO"] is True and opts["PLAIN_WAIT"] is True and grid == (14, 4)
    _, opts = _merge(monkeypatch, early=True, plain_wait=True)
    assert "HILO" not in opts


@pytest.mark.parametrize(
    "kw",
    [
        dict(plain=True),
        dict(rows=L._MAX_ROWS // 2 + 1),  # no room for the residual rows
    ],
)
def test_hilo_is_the_per_head_merge_with_room_for_the_residual_rows(monkeypatch, kw):
    with pytest.raises(AssertionError):
        _merge(monkeypatch, hilo=True, **kw)


def test_the_partial_gets_the_precise_build_and_the_cache_key_carries_it(monkeypatch):
    calls = []
    monkeypatch.setattr(F, "q35_cp_fused_partial", lambda *a, **k: calls.append(k))
    cp = _cp(
        po_x=torch.zeros(4, 56, 4, 256, dtype=torch.bfloat16),
        pl_x=torch.zeros(4, 56, 4),
        q_all=torch.zeros(28, 16, 256, dtype=torch.float8_e4m3fn),
    )
    args = ("k", "v", "table", "lens", 7, 4, 28, 1.0, 1.0)
    cp._exact_partial(*args, pdl=True, len_sub=7)
    cp._exact_partial(*args, pdl=True, len_sub=7, o_hilo=True, p_hilo=True)
    plain, precise = calls
    assert "o_hilo" not in plain and "p_hilo" not in plain
    assert precise["o_hilo"] is True and precise["p_hilo"] is True
    params = inspect.signature(F._compiled).parameters
    assert params["o_hilo"].default is False and params["p_hilo"].default is False


class _CpLite:
    def __init__(self):
        self.calls = []
        self._gate_epilogue = None

    def attend(self, q, caches, page_table, seq_lens, q_len, *args, **kwargs):
        self.calls.append((q.shape[0], kwargs))
        out = kwargs.get("out")
        o = torch.ones(q.shape[0], q.shape[1], q.shape[2], dtype=torch.bfloat16)
        if out is not None:
            out.copy_(o)
            return out
        return o


def _cp_core(requests, q_len, rows, **kw):
    heads, dim = 4, 256
    attn = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=heads,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=dim,
        scaling=1.0,
    )
    pool = torch.zeros(4 * split_kv.PAGE_SIZE, 1, dim, dtype=torch.float8_e4m3fn)
    kv = SimpleNamespace(
        layer_ids=[3], k=[pool], v=[pool], k_scale=[1.0], v_scale=[1.0]
    )
    meta = SimpleNamespace(
        page_table=torch.zeros(requests, 2, dtype=torch.int32),
        cache_seqlens=torch.full((requests,), 100, dtype=torch.int32),
    )
    q = torch.zeros(rows, heads * dim, dtype=torch.float8_e4m3fn)
    fake = _CpLite()
    with (
        patch.object(split_kv, "cp_lite", return_value=fake),
        patch("sglang.srt.runtime_context.get_buffer", return_value=torch.zeros(1)),
    ):
        o = split_kv.cp_core(attn, q, kv, meta, "fb", q_len, min_bs=None, **kw)
    return o, fake.calls


def test_cp_core_serves_a_padded_calls_requests_and_zeroes_the_pad_rows():
    o, calls = _cp_core(3, 7, 24, pad_rows=True, precise=True, merge_h=True)
    ((rows, kw),) = calls
    assert rows == 21 and kw["precise"] is True and kw["merge_h"] is True
    assert tuple(o.shape) == (24, 4 * 256)
    assert torch.all(o[:21] == 1) and torch.all(o[21:] == 0)
    # without pad rows such a call is declined, and an unpadded call runs as before
    assert _cp_core(3, 7, 24) == (None, [])
    o, calls = _cp_core(3, 7, 21)
    ((rows, kw),) = calls
    assert rows == 21 and "out" not in kw and "precise" not in kw
    assert tuple(o.shape) == (21, 4 * 256)
