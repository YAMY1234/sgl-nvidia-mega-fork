# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

NEW = ("cp_smv2", "cp_ptxas_o1")
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


@pytest.fixture(autouse=True)
def _default_build():
    F.set_partial_build()
    F._BUILD_LOGGED.clear()
    yield
    F.set_partial_build()
    F._BUILD_LOGGED.clear()


def test_declared_as_bitwise_cp_lite_variants_at_tp8_and_tep8():
    for name in NEW:
        assert name in runtime.OPT_IN_VARIANTS and name in runtime.CP_LITE_VARIANTS
        assert name not in runtime.SHIPPED_SLOTS
        assert topology.SLOT_TOPOLOGIES[name] == {"tp8", "tep8"}
        assert name in runtime.MEASURED_CP_LITE and name in runtime.MEASURED_BITWISE
        assert name not in runtime.NUMERICS_CHANGE


@pytest.mark.parametrize("name", NEW)
def test_a_variant_needs_a_cp_slot(monkeypatch, name):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({name}))
    reasons = runtime.slot_reasons()
    assert any(f"slot {name!r} builds the CP slots' partial" in r for r in reasons)
    for cp in sorted(runtime.CP_LITE_SLOTS):
        base = {"skv2_cp": "skv2", "draft_q1_cp": "draft_q1"}.get(cp, "draft_skv2")
        entry = {"draft_skv2": "draft_extend"}.get(base, "draft_loop")
        slots = frozenset({name, cp, base, entry, "draft_loop"})
        monkeypatch.setattr(runtime, "SLOTS", slots)
        assert not [r for r in runtime.slot_reasons() if name in r], cp


def test_unsupported_cp_lite_topology_drops_the_variants(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp", *NEW}))
    assert runtime.apply_cp_lite_topology(2) == {"skv2_cp", *NEW}
    assert runtime.SLOTS == {"skv2"}


@pytest.mark.parametrize(
    "tp, dp, ep, on",
    [(8, 1, 1, True), (8, 1, 4, True), (4, 1, 1, False), (2, 1, 1, False)],
)
def test_only_tp8_and_tep8_select_the_variants(tp, dp, ep, on):
    got = topology.select(frozenset(NEW), topology.Topology(tp, dp, ep))
    assert got == (frozenset(NEW) if on else frozenset())


def test_set_partial_build_validates_strips_and_logs(caplog):
    assert (F._SMV2, F._PTXAS) == (0, "")
    with pytest.raises(AssertionError):
        F.set_partial_build(softmax_v2=3)
    caplog.set_level(logging.INFO)
    F.set_partial_build(softmax_v2=2, ptxas=" -O1 ")
    assert (F._SMV2, F._PTXAS) == (2, "-O1")
    assert "softmax v2 = 2, ptxas options '-O1'" in caplog.text


def test_build_opts_follow_the_setting_and_log_once_per_q_len(caplog):
    caplog.set_level(logging.INFO)
    assert F._build_opts(7, 16) == (0, "")
    assert "build CP-lite's partial" not in caplog.text
    F.set_partial_build(softmax_v2=2, ptxas="-O1")
    assert F._build_opts(7, 16) == (2, "-O1")
    assert F._build_opts(7, 8) == (2, "-O1")
    assert F._build_opts(1, 8) == (2, "-O1")
    assert caplog.text.count("build CP-lite's partial with softmax v2 = 2") == 2


def test_the_compile_cache_key_carries_the_options():
    import inspect

    params = inspect.signature(F._compiled.__wrapped__).parameters
    names = list(params)
    i = names.index("two_slot_pack")
    assert names[i + 1 : i + 4] == ["min_ctas", "softmax_v2", "ptxas"]
    assert params["min_ctas"].default == 1
    assert params["softmax_v2"].default == 0 and params["ptxas"].default == ""


def _build_cp_lite(slots, created=True, parallel=None):

    class _Binding:
        page_size, tp_size = 64, 8

    lite = types.ModuleType("q35_cp_lite_stub")
    cp_lite = SimpleNamespace(_instance=None)

    def maybe_create(binding, device):
        if created:
            cp_lite._instance = SimpleNamespace()
        return cp_lite._instance

    cp_lite.maybe_create = maybe_create
    lite.CpLite = cp_lite
    agree = types.ModuleType("binding_stub")
    agree.megakernel_agree = lambda *a, **k: None
    with (
        patch.object(runtime, "SLOTS", slots),
        patch.dict(
            sys.modules,
            {
                "sglang.srt.kernel_plugins.nvidia.binding": agree,
                "sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite": lite,
            },
        ),
        patch("torch.cuda.current_device", lambda: 0),
    ):
        runtime.build_cp_lite(_Binding())
        if parallel is not None:
            runtime.apply_topology(parallel)
        runtime.apply_cp_lite_build()


@pytest.mark.parametrize(
    "slots, want",
    [
        (frozenset({"skv2_cp"}), (0, "")),
        (frozenset({"skv2_cp", "cp_smv2"}), (2, "")),
        (frozenset({"draft_q1_cp", "cp_ptxas_o1"}), (0, "-O1")),
        (frozenset({"skv2_cp", "cp_smv2", "cp_ptxas_o1"}), (2, "-O1")),
        (frozenset({"cp_smv2", "cp_ptxas_o1"}), (0, "")),
    ],
)
def test_the_build_sets_the_partial_build_from_the_slots(slots, want):
    _build_cp_lite(slots)
    assert (F._SMV2, F._PTXAS) == want


def test_no_build_option_without_cp_lite(caplog):
    with caplog.at_level("WARNING"):
        _build_cp_lite(frozenset({"skv2_cp", *NEW}), created=False)
    assert (F._SMV2, F._PTXAS) == (0, "")


@pytest.mark.parametrize("tp, ep, want", [(4, 1, (0, "")), (8, 4, (2, "-O1"))])
def test_the_options_follow_the_topology_filter(tp, ep, want):
    parallel = SimpleNamespace(tp_size=tp, attn_dp_size=1, moe_ep_size=ep)
    _build_cp_lite(frozenset({"skv2", "skv2_cp", *NEW}), parallel=parallel)
    assert (F._SMV2, F._PTXAS) == want


def _record(monkeypatch, calls):
    for name in ("build_cp_lite", "apply_topology", "apply_cp_lite_build"):
        real = getattr(runtime, name)
        monkeypatch.setattr(
            runtime,
            name,
            lambda *a, _n=name, _f=real: calls.append(_n) or _f(*a),
        )


def test_each_build_sets_the_options_after_the_topology_filter(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    from sgl_nvidia_mega.qwen3_5 import draft

    monkeypatch.setattr(plugin_binding, "megakernel_agree", lambda *a, **k: None)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=8, attn_tp_size=8, moe_ep_size=1),
    )
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    calls = []
    _record(monkeypatch, calls)
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    runtime.build(model, ctx)
    assert calls == ["build_cp_lite", "apply_topology", "apply_cp_lite_build"]
    calls.clear()
    ctx = SimpleNamespace(
        widths=(), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    draft.SPEC.build("lm", SimpleNamespace(context_len=4096), ctx)
    assert calls == ["apply_topology", "build_cp_lite", "apply_cp_lite_build"]


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("1", "hop", False)],
)
def test_the_measured_plan_runs_the_variants_at_tp8_and_tep8(
    monkeypatch, switch, slots, on
):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for topo in ("tp8", "tep8"):
        for name in NEW:
            assert runtime._measured_rows_on((topo, topology.GB300), name) is on
            assert not runtime._measured_rows_on((topo, topology.Sku(10, 3, 148)), name)
    assert (set(NEW) <= runtime._requested_slots()) is on
