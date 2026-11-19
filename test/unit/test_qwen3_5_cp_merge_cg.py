# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

SLOT = "cp_merge_cg"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


def test_declared_as_a_bitwise_cp_lite_variant_at_tp8_and_tep8():
    assert SLOT in runtime.OPT_IN_VARIANTS and SLOT in runtime.CP_LITE_VARIANTS
    assert SLOT in runtime.CP_PUSHED_CALL_SLOTS and SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_CP_LITE and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE


@pytest.mark.parametrize("push", [True, False])
def test_the_slot_needs_cp_q_push(monkeypatch, push):
    slots = {"skv2", "skv2_cp", "mka", SLOT} | ({"cp_q_push"} if push else set())
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    assert not reasons if push else reasons


def test_unsupported_cp_lite_topology_drops_the_slot(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    slots = frozenset({"skv2", "skv2_cp", "mka", "cp_q_push", SLOT})
    monkeypatch.setattr(runtime, "SLOTS", slots)
    assert runtime.apply_cp_lite_topology(2) == {"skv2_cp", "cp_q_push", SLOT}


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


@pytest.mark.parametrize(
    "slots, want",
    [
        ({"skv2_cp", "mka", "cp_q_push", SLOT}, True),
        ({"skv2_cp", "mka", "cp_q_push"}, False),
    ],
)
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
    assert instance.merge_cg is want


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append(kwargs)


@pytest.mark.parametrize(
    "merge_cg, early, pdl, plain, want",
    [
        (True, True, True, False, True),
        (False, True, True, False, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
        (True, True, True, True, False),
    ],
)
def test_only_an_early_dependent_merge_reads_through_l2(
    monkeypatch, merge_cg, early, pdl, plain, want
):
    launch = _Launch()
    monkeypatch.setattr(L, "_cp_merge_h_ll", launch)
    monkeypatch.setattr(L, "_cp_merge_h_ll_plain", launch)
    me = SimpleNamespace(
        cp_pdl=True,
        hq=4,
        cp=4,
        merge_cg=merge_cg,
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
    L.CpLite._merge_h(me, *args, pdl=pdl, early=early, plain=plain)
    assert launch.calls[0].get("CG", False) is want
