# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch
from test_qwen3_5_cp_o_push import _GATE, _cp, _Launch, _merge

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
NEEDS = {
    "skv2",
    "skv2_cp",
    "mka",
    "draft_loop",
    "draft_extend",
    "draft_multi",
    "draft_cp",
}
PRECISE = dict(early_merge=False, merge_h=True, plain_wait=True, hilo=True, p_hilo=True)


@pytest.mark.parametrize("slot", ["draft_gate_fold", "draft_o_push"])
def test_declared_as_a_bitwise_draft_slot_of_the_measured_plans(slot):
    assert slot in runtime.DRAFT_SLOTS and slot not in runtime.SHIPPED_SLOTS
    assert slot in runtime.MEASURED_DRAFT and slot in runtime.MEASURED_BITWISE
    assert slot not in runtime.NUMERICS_CHANGE and slot in runtime.CP_LITE_VARIANTS
    assert topology.SLOT_TOPOLOGIES[slot] == {"tp8", "tep8"}
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert slot in runtime.MEASURED_PLAN[key]


@pytest.mark.parametrize(
    "slot, drop",
    [
        ("draft_gate_fold", "draft_multi"),
        ("draft_gate_fold", "draft_cp"),
        ("draft_o_push", "draft_multi"),
        ("draft_o_push", "draft_cp"),
    ],
)
def test_the_slot_needs_the_precise_multi_request_call(monkeypatch, slot, drop):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {slot}))
    assert not [r for r in runtime.slot_reasons() if f"slot '{slot}'" in r]
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS - {drop} | {slot}))
    assert [r for r in runtime.slot_reasons() if f"slot '{slot}'" in r]


@pytest.mark.parametrize(
    "switch, slots, on", [("1", None, True), (None, None, False), ("1", "hop", False)]
)
def test_the_measured_plan_runs_it(monkeypatch, switch, slots, on):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for topo in ("tp8", "tep8"):
        for slot in ("draft_gate_fold", "draft_o_push"):
            assert runtime._measured_rows_on((topo, topology.GB300), slot) is on


def test_unsupported_cp_lite_topology_drops_it(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset(NEEDS | {"draft_gate_fold", "draft_o_push"})
    )
    dropped = runtime.apply_cp_lite_topology(2)
    assert {"draft_cp", "draft_gate_fold", "draft_o_push"} <= dropped


def test_defaults_off():
    assert (L.CpLite._gate_bf16, L.CpLite._gate_bf16_done) == (None, False)


B, QL = 3, 1
ROWS = B * QL


def _gate(rows=ROWS):
    return torch.zeros(rows, 4 * 256, dtype=torch.bfloat16)


def _ungated(monkeypatch, cp, ql=QL, **kw):
    monkeypatch.setattr(L, "_cp_gather_q_ll", _Launch())
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", _Launch())
    partials, merges = [], []
    cp._exact_partial = lambda *a, **k: partials.append(k)
    cp._merge_h = lambda *a, **k: merges.append(k)
    q = torch.zeros(B * ql, 4, 256, dtype=torch.int8)
    cp._ungated_main(q, "k", "v", "table", "lens", ql, B, B * ql, 1.0, 1.0, _GATE, **kw)
    return partials[0], merges[0]


def test_the_precise_calls_merge_takes_the_offered_gate(monkeypatch, caplog):
    cp = _cp()
    g = _gate()
    cp.offer_gate_bf16(g)
    with caplog.at_level(logging.INFO):
        _, merge = _ungated(monkeypatch, cp, **PRECISE)
    assert merge["gate_bf16"] is g
    assert "draft_gate_fold" in caplog.text
    assert cp.take_gate_bf16() is True
    assert cp.take_gate_bf16() is False and cp._gate_bf16 is None


@pytest.mark.parametrize(
    "kw, gate",
    [
        (dict(PRECISE, merge_h=False), _gate()),
        (dict(PRECISE, early_merge=None), _gate()),  # a one-request call's early merge
        (dict(PRECISE, hilo=False, p_hilo=False), _gate()),  # not precise
        (PRECISE, _gate(ROWS + 1)),  # another shape
        (PRECISE, _gate().float()),  # another dtype
    ],
)
def test_every_other_call_leaves_the_gate_to_its_launch(monkeypatch, kw, gate):
    cp = _cp()
    cp.offer_gate_bf16(gate)
    _, merge = _ungated(monkeypatch, cp, **kw)
    assert merge.get("gate_bf16") is None
    assert cp.take_gate_bf16() is False


def test_the_merge_gets_the_gate_option(monkeypatch):
    g = _gate(14)
    opts = _merge(monkeypatch, early=False, plain_wait=True, hilo=True, gate_bf16=g)
    assert opts["GATE_BF16"] is True and opts["gate"] is g
    assert opts["gate_stride"] == g.stride(0) and opts["HILO"] is True
    assert "GATE_BF16" not in _merge(
        monkeypatch, early=False, plain_wait=True, hilo=True
    )


@pytest.mark.parametrize("kw", [dict(early=True), dict(hilo=False)])
def test_the_gate_option_is_the_precise_merge_after_its_wait(monkeypatch, kw):
    args = dict(early=False, plain_wait=True, hilo=True, gate_bf16=_gate(14))
    args.update(kw)
    with pytest.raises(AssertionError):
        _merge(monkeypatch, **args)


class _CpLite:
    def __init__(self, take=True):
        self.offered, self.take = [], take

    def offer_gate_bf16(self, gate):
        self.offered.append(gate)

    def take_gate_bf16(self):
        return self.take and bool(self.offered)


def _attention(
    monkeypatch, *, gate_fold=True, precise=True, multi=True, takes=True, cp=None
):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.draft import attention as A

    cp = _CpLite() if cp is None else cp
    monkeypatch.setattr(split_kv, "cp_lite", lambda: cp)
    att = A.Attention.__new__(A.Attention)
    att.gate_fold = gate_fold
    att.layer = SimpleNamespace(attn_output_gate=True)
    att.cp_b1 = SimpleNamespace(
        precise=precise, multi=multi, takes=lambda q_len, rows: takes
    )
    att._noted, att._warned = set(), set()
    return att, cp


@pytest.mark.parametrize(
    "over, rows, rows_per_request, o_proj, offered",
    [
        ({}, 4, 1, None, True),
        ({}, 28, 7, None, True),
        ({}, 1, 1, None, False),  # one request
        ({}, 4, 1, "hop o_proj", False),  # the draft hop's o_proj stages the gate
        ({"gate_fold": False}, 4, 1, None, False),
        ({"precise": False}, 4, 1, None, False),  # no slot draft_cp: the plain merge
        ({"multi": False}, 4, 1, None, False),
        ({"takes": False}, 4, 1, None, False),  # past the cut-over
    ],
)
def test_the_draft_offers_the_gate_on_the_precise_multi_request_calls(
    monkeypatch, over, rows, rows_per_request, o_proj, offered
):
    att, cp = _attention(monkeypatch, **over)
    got = att._offer_gate(_gate(rows), rows, rows_per_request, o_proj)
    assert got is offered and bool(cp.offered) is offered


def test_a_gate_the_merge_did_not_take_runs_its_launch_with_a_warning(
    monkeypatch, caplog
):
    att, cp = _attention(monkeypatch, cp=_CpLite(take=False))
    assert att._offer_gate(_gate(4), 4, 1, None)
    with caplog.at_level(logging.WARNING):
        assert att._took_gate(4) is False
        assert att._took_gate(4) is False
    assert caplog.text.count("draft_gate_fold") == 1


def test_the_precise_calls_partial_pushes_and_its_merge_only_polls(monkeypatch, caplog):
    cp = _cp()
    with caplog.at_level(logging.INFO):
        partial, merge = _ungated(monkeypatch, cp, hilo_push=True, **PRECISE)
    assert partial["o_push"] is True and partial["o_hilo"] is True
    assert partial["late_trigger"] is False and partial["early_trigger"] is False
    assert merge["nopush"] is True and merge["early"] is False and merge["hilo"] is True
    assert "slot draft_o_push" in caplog.text


@pytest.mark.parametrize(
    "kw",
    [
        dict(PRECISE, merge_h=False),
        dict(PRECISE, early_merge=None),  # a one-request call's early merge
        dict(PRECISE, hilo=False, p_hilo=False),  # not precise
    ],
)
def test_every_other_call_keeps_the_merges_push(monkeypatch, caplog, kw):
    cp = _cp(o_push=False)
    with caplog.at_level(logging.WARNING):
        partial, merge = _ungated(monkeypatch, cp, hilo_push=True, **kw)
    assert partial["o_push"] is False and merge["nopush"] is False
    assert "slot draft_o_push" in caplog.text


def test_without_it_the_precise_call_is_unchanged(monkeypatch):
    partial, merge = _ungated(monkeypatch, _cp(), **PRECISE)
    assert partial["o_push"] is False and partial["late_trigger"] is True
    assert merge["nopush"] is False and merge["hilo"] is True


def test_the_polling_precise_merge_keeps_its_wait_at_entry(monkeypatch):
    opts = _merge(monkeypatch, early=False, plain_wait=True, hilo=True, nopush=True)
    assert opts["NOPUSH"] is True and opts["HILO"] is True and opts["EARLY"] is False
    assert "CG" not in opts and "MSLEEP" not in opts
    with pytest.raises(AssertionError):
        _merge(monkeypatch, early=False, plain_wait=True, nopush=True)


def test_the_partial_gets_the_push_with_its_residual_words(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

    calls = []
    monkeypatch.setattr(F, "q35_cp_fused_partial", lambda *a, **k: calls.append(k))
    cp = _cp(
        po_x=torch.zeros(4, 28, 4, 256, dtype=torch.bfloat16),
        pl_x=torch.zeros(4, 28, 4),
        q_all=torch.zeros(28, 16, 256, dtype=torch.float8_e4m3fn),
        sms=148,
    )
    args = ("k", "v", "table", "lens", 1, 4, 4, 1.0, 1.0)
    cp._exact_partial(
        *args,
        pdl=True,
        len_sub=1,
        o_push=True,
        o_hilo=True,
        p_hilo=True,
        late_trigger=False,
    )
    (k,) = calls
    assert k["o_push"] is True and k["o_hilo"] is True and k["late_trigger"] is False


def test_the_build_drops_the_early_epoch_read_with_residual_words(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

    monkeypatch.setattr(F, "_EP_EARLY", True)
    assert F._push_opts(True, 1, 4, False)["ep_early"] is True
    assert "ep_early" not in F._push_opts(True, 1, 4, False, o_hilo=True)


class _SplitKV:
    def __init__(self):
        self.calls = []

    def cp_core(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "o"


@pytest.mark.parametrize(
    "precise, push, rows, want",
    [
        (True, True, 4, True),
        (True, True, 1, False),
        (True, False, 4, False),
        (False, True, 4, False),
    ],
)
def test_draft_cp_asks_for_the_push_on_its_precise_multi_request_calls(
    monkeypatch, precise, push, rows, want
):
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    d = DraftCP(_SplitKV(), SimpleNamespace(), True, True, True, precise, push=push)
    attention = SimpleNamespace(layer=SimpleNamespace(attn=SimpleNamespace()))
    d.run(attention, torch.zeros(rows, 4), "meta", "kv", 1)
    assert d.split_kv.calls[0].get("hilo_push", False) is want


@pytest.mark.parametrize("name", ["draft_merge_early", "draft_hilo_trig"])
def test_merge_early_is_a_bitwise_option_of_the_measured_plans(name):
    assert name in runtime.MEASURED_OPTIONS and name in runtime.MEASURED_BITWISE
    assert name in runtime.OPTION_ROWS and name not in runtime.DRAFT_SLOTS
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert name in runtime.MEASURED_PLAN[key]


def test_the_early_polling_merge_is_released_after_the_partials_stream(
    monkeypatch, caplog
):
    cp = _cp()
    g = _gate()
    cp.offer_gate_bf16(g)
    with caplog.at_level(logging.INFO):
        partial, merge = _ungated(
            monkeypatch, cp, hilo_push=True, hilo_early=True, **PRECISE
        )
    assert partial["o_push"] is True and partial["late_trigger"] is True
    assert (
        merge["nopush"] is True and merge["early"] is True and merge["gate_bf16"] is g
    )
    assert "option draft_merge_early" in caplog.text
    # without the push it does nothing
    partial, merge = _ungated(monkeypatch, _cp(), hilo_early=True, **PRECISE)
    assert partial["o_push"] is False and merge["early"] is False


def test_the_early_polling_merge_options(monkeypatch):
    g = _gate(14)
    opts = _merge(
        monkeypatch, early=True, plain_wait=True, hilo=True, nopush=True, gate_bf16=g
    )
    assert opts["NOPUSH"] is True and opts["EARLY"] is True and opts["HILO"] is True
    assert opts["CG"] is True and opts["GATE_BF16"] is True


@pytest.mark.parametrize("on", [False, True])
def test_draft_cp_asks_for_the_early_merge_with_the_push(monkeypatch, on):
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    d = DraftCP(
        _SplitKV(), SimpleNamespace(), True, True, True, True, push=True, merge_early=on
    )
    attention = SimpleNamespace(layer=SimpleNamespace(attn=SimpleNamespace()))
    d.run(attention, torch.zeros(4, 4), "meta", "kv", 1)
    assert d.split_kv.calls[0].get("hilo_early", False) is on
    # never without the push
    assert (
        DraftCP(
            _SplitKV(), None, True, True, True, True, push=False, merge_early=True
        ).merge_early
        is False
    )


@pytest.mark.parametrize(
    "switch, slots, on", [("1", None, True), (None, None, False), ("1", "hop", False)]
)
def test_draft_cp_takes_the_option_from_the_measured_plan(
    monkeypatch, switch, slots, on
):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "MEASURED_KEY", ("tep8", topology.GB300))
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {"draft_o_push"}))
    monkeypatch.setattr(split_kv, "cp_reason", lambda: "")
    d = DraftCP.build(
        SimpleNamespace(
            slots=frozenset(NEEDS | {"draft_o_push"}) & set(runtime.DRAFT_SLOTS)
        )
    )
    assert d.push is True and d.merge_early is on and d.hilo_trig is on


def test_the_early_merge_releases_o_proj_at_entry(monkeypatch, caplog):
    cp = _cp()
    with caplog.at_level(logging.INFO):
        _, merge = _ungated(
            monkeypatch, cp, hilo_push=True, hilo_early=True, hilo_trig=True, **PRECISE
        )
    assert merge["trig"] is True and merge["early"] is True and merge["nopush"] is True
    assert "draft_hilo_trig" in caplog.text
    # not without the early merge (the merge after its wait releases nothing at entry)
    _, merge = _ungated(monkeypatch, _cp(), hilo_push=True, hilo_trig=True, **PRECISE)
    assert merge["trig"] is False and merge["early"] is False


def test_the_extends_early_merge_releases_nothing_at_entry(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        _, merge = _ungated(
            monkeypatch,
            _cp(),
            ql=7,
            hilo_push=True,
            hilo_early=True,
            hilo_trig=True,
            **PRECISE,
        )
    assert merge["trig"] is False
    assert merge["early"] is True and merge["nopush"] is True and merge["hilo"] is True
    assert "draft_hilo_trig" not in caplog.text


def test_the_trig_option(monkeypatch):
    opts = _merge(
        monkeypatch, early=True, plain_wait=True, hilo=True, nopush=True, trig=True
    )
    assert opts["TRIG"] is True and opts["EARLY"] is True
    assert "TRIG" not in _merge(
        monkeypatch, early=True, plain_wait=True, hilo=True, nopush=True
    )
    for kw in (dict(early=False), dict(nopush=False), dict(hilo=False)):
        args = dict(early=True, plain_wait=True, hilo=True, nopush=True, trig=True)
        args.update(kw)
        with pytest.raises(AssertionError):
            _merge(monkeypatch, **args)


def test_draft_cp_asks_for_trig_only_with_the_early_merge(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    d = DraftCP(
        _SplitKV(),
        None,
        True,
        True,
        True,
        True,
        push=True,
        merge_early=True,
        hilo_trig=True,
    )
    attention = SimpleNamespace(layer=SimpleNamespace(attn=SimpleNamespace()))
    d.run(attention, torch.zeros(4, 4), "meta", "kv", 1)
    assert d.split_kv.calls[0]["hilo_trig"] is True
    # the draft extend (7 rows a request) keeps its early merge without TRIG
    d.run(attention, torch.zeros(28, 4), "meta", "kv", 7)
    assert d.split_kv.calls[1]["hilo_early"] is True
    assert "hilo_trig" not in d.split_kv.calls[1]
    d = DraftCP(
        _SplitKV(),
        None,
        True,
        True,
        True,
        True,
        push=True,
        merge_early=False,
        hilo_trig=True,
    )
    assert d.hilo_trig is False
