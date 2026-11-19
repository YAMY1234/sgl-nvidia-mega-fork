# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.attention import mka

SLOT = "cp_qw_entry"
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
    "slots, pushed, want",
    [
        ({"cp_q_push", SLOT}, True, True),
        ({"cp_q_push", SLOT}, False, False),
        ({"cp_q_push"}, True, False),
    ],
)
def test_the_front_releases_at_entry_only_on_a_pushed_call(
    monkeypatch, slots, pushed, want
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    assert mka._qw_entry({"words": 256} if pushed else None) is want
