# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

SLOT = "cp_o_push"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
NEEDS = {"skv2", "skv2_cp", "mka", "cp_q_push", "cp_merge_plain_wait"}


def test_declared_as_a_bitwise_cp_lite_variant_at_tp8_and_tep8():
    assert SLOT in runtime.OPT_IN_VARIANTS and SLOT in runtime.CP_LITE_VARIANTS
    assert SLOT in runtime.CP_PUSHED_CALL_SLOTS and SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_CP_LITE and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE


@pytest.mark.parametrize(
    "drop, needs",
    [
        (set(), None),
        ({"cp_q_push"}, "'cp_q_push'"),
        ({"cp_merge_plain_wait"}, "'cp_merge_plain_wait'"),
    ],
)
def test_the_slot_needs_cp_q_push_and_the_plain_wait(monkeypatch, drop, needs):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS - drop | {SLOT}))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    if needs is None:
        assert not reasons
    else:
        assert any(needs in r for r in reasons), reasons


def test_unsupported_cp_lite_topology_drops_the_slot(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {SLOT}))
    assert runtime.apply_cp_lite_topology(2) == {
        "skv2_cp",
        "cp_q_push",
        "cp_merge_plain_wait",
        SLOT,
    }


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("1", "hop", False)],
)
def test_the_measured_plan_runs_the_slot(monkeypatch, switch, slots, on):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for topo in ("tp8", "tep8"):
        assert runtime._measured_rows_on((topo, topology.GB300), SLOT) is on
    assert (SLOT in runtime._requested_slots()) is on


@pytest.mark.parametrize("slots, want", [(NEEDS | {SLOT}, True), (NEEDS, False)])
def test_the_build_sets_it_on_cp_lite(slots, want):
    instance = SimpleNamespace()
    lite = types.ModuleType("q35_cp_lite_stub")
    lite.CpLite = SimpleNamespace(_instance=instance)
    with (
        patch.object(runtime, "SLOTS", frozenset(slots)),
        patch.dict(
            sys.modules, {"sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite": lite}
        ),
    ):
        runtime.apply_cp_lite_build()
    assert instance.o_push is want


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
        o_push=True,
        early_merge=True,
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


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


_GATE = dict(nb=4, thr=0, GATE=False, NB=4, GMAX=False)


@pytest.mark.parametrize(
    "over, pushed, plain_wait, early_merge, b, want",
    [
        ({}, True, True, None, 4, True),
        ({}, True, True, None, 64, True),
        ({}, True, True, None, 65, False),
        ({}, False, True, None, 4, False),
        ({}, True, False, None, 4, False),
        ({}, True, True, False, 4, False),
        ({"early_merge": False}, True, True, None, 4, False),
        ({"o_push": False}, True, True, None, 4, False),
    ],
)
def test_the_ungated_exact_call_pushes_only_for_the_early_plain_wait_merge(
    monkeypatch, caplog, over, pushed, plain_wait, early_merge, b, want
):
    monkeypatch.setattr(L, "_cp_gather_q_ll", _Launch())
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", _Launch())
    cp = _cp(**over)
    partials, merges = [], []
    cp._exact_partial = lambda *a, **k: partials.append(k)
    cp._merge_h = lambda *a, **k: merges.append(k)
    rows = 7 * b
    if pushed:
        cp._q_pushed = rows
    q = torch.zeros(rows, 4, 256, dtype=torch.int8)
    with caplog.at_level(logging.INFO):
        cp._ungated_main(
            q,
            "k",
            "v",
            "table",
            "lens",
            7,
            b,
            rows,
            1.0,
            1.0,
            _GATE,
            early_merge=early_merge,
            plain_wait=plain_wait,
        )
    assert partials[0]["o_push"] is want and merges[0]["nopush"] is want
    early = cp.early_merge if early_merge is None else early_merge
    assert partials[0]["early_trigger"] is (early and not want)
    assert ("slot cp_o_push" in caplog.text) is want


def test_the_partial_gets_the_mailboxes_epochs_and_late_release(monkeypatch):
    calls = []
    monkeypatch.setattr(F, "q35_cp_fused_partial", lambda *a, **k: calls.append(k))
    cp = _cp(
        po_x=torch.zeros(4, 28, 4, 256, dtype=torch.bfloat16),
        pl_x=torch.zeros(4, 28, 4),
        q_all=torch.zeros(28, 16, 256, dtype=torch.float8_e4m3fn),
        sms=148,
    )
    args = ("k", "v", "table", "lens", 7, 4, 28, 1.0, 1.0)
    cp._exact_partial(*args, pdl=True, len_sub=7, early_trigger=True)
    cp._exact_partial(*args, pdl=True, len_sub=7, early_trigger=False, o_push=True)
    plain, pushed = calls
    assert "o_push" not in plain and plain["early_trigger"] is True
    extra = {k: pushed[k] for k in pushed if k not in plain}
    assert torch.equal(extra.pop("ll"), cp.peers + cp.llh_off)
    assert extra == dict(
        o_push=True, cnt_h="cnt_h", ql_real=7, hq_real=4, late_trigger=True
    )


def test_the_mailbox_table_holds_each_members_merge_mailbox():
    cp = _cp(cp=2, peers=torch.tensor([10, 20]), llh_off=5)
    assert cp._ll_tab().tolist() == [15, 25, 0, 0]
    assert cp._ll_tab() is cp._ll_tab()


def _merge(monkeypatch, **kw):
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
    args = (t, t, t, t, t, 7, 14, 1.0, 1.0, t, gate)
    L.CpLite._merge_h(me, *args, **kw)
    return launch.calls[0][2]


def test_the_pushed_merge_polls_through_l2_without_pushing(monkeypatch):
    opts = _merge(monkeypatch, early=True, plain_wait=True, push_rot=True, nopush=True)
    assert opts["NOPUSH"] is True and opts["CG"] is True and opts["EARLY"] is True
    assert opts["MSLEEP"] == L._PUSHED_MERGE_SLEEP_NS > 0
    assert opts["PLAIN_WAIT"] is True
    plain = _merge(monkeypatch, early=True, plain_wait=True, push_rot=True)
    assert "NOPUSH" not in plain and "MSLEEP" not in plain and "CG" not in plain


@pytest.mark.parametrize(
    "kw",
    [
        dict(early=False, plain_wait=True),
        dict(early=True, plain_wait=False),
        dict(early=True, plain_wait=True, plain=True),
        dict(early=True, plain_wait=True, pdl=False),
    ],
)
def test_the_pushed_merge_is_the_early_plain_wait_merge(monkeypatch, kw):
    with pytest.raises(AssertionError):
        _merge(monkeypatch, nopush=True, **kw)


def test_the_compile_cache_key_carries_the_push():
    params = inspect.signature(F._compiled.__wrapped__).parameters
    for name in ("o_push", "ql_real", "hq_real", "late_trigger"):
        assert name in params
    assert params["o_push"].default is False and params["late_trigger"].default is False
    wrapper = inspect.signature(F.q35_cp_fused_partial).parameters
    for name in ("o_push", "ll", "cnt_h", "late_trigger"):
        assert wrapper[name].kind is inspect.Parameter.KEYWORD_ONLY
