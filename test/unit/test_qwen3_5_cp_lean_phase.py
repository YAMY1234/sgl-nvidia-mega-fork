# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import logging
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
_B300 = topology.Sku(10, 3, 148)


@pytest.fixture(autouse=True)
def _phase_off():
    F.set_cp_phase(False)
    yield
    F.set_cp_phase(False)


def _flags():
    return F._LEAN_PHASE, F._PLAN_CACHE, F._COMB_W


def test_off_by_default():
    # a separate copy of the module, so the imported one keeps its state
    spec = importlib.util.spec_from_file_location("q35_cp_fused_copy", F.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert (fresh._LEAN_PHASE, fresh._PLAN_CACHE, fresh._COMB_W) == (False,) * 3


@pytest.mark.parametrize(
    "args, want",
    [
        ((True,), (True, True, True)),
        ((True, False), (True, False, True)),
        ((True, True, False), (True, True, False)),
        ((False,), (False, False, False)),
        ((False, True, True), (False, False, False)),
    ],
)
def test_set_cp_phase(args, want, caplog):
    caplog.set_level(logging.INFO)
    F.set_cp_phase(*args)
    assert _flags() == want
    assert ("lean fixed phase on" in caplog.text) is want[0]


def test_the_compile_cache_key_carries_the_phase():
    import inspect

    params = inspect.signature(F._compiled.__wrapped__).parameters
    names = list(params)
    assert names[names.index("softmax_v2") :][:5] == [
        "softmax_v2",
        "ptxas",
        "lean_phase",
        "plan_cache",
        "comb_w",
    ]
    assert not any(params[n].default for n in ("lean_phase", "plan_cache", "comb_w"))


def test_the_host_row_width_mirrors_the_kernel():
    src = Path(F.__file__).with_name("q35_cp_partial_attn.py").read_text()
    assert re.search(rf"^PC_ROW_WORDS = {F._PC_ROW_WORDS}$", src, re.M)


@pytest.fixture
def cpu_state(monkeypatch):
    st = F._DeviceState(torch.device("cpu"), 8)
    monkeypatch.setattr(F, "_STATES", {0: st})
    return st


def test_the_pool_exists_only_while_the_plan_cache_is_on(cpu_state):
    assert cpu_state.plan_cache_pool is None
    F.set_cp_phase(True, plan_cache=False)
    assert cpu_state.plan_cache_pool is None
    F.set_cp_phase(True)
    pool = cpu_state.plan_cache_pool
    assert pool.shape == (F._PC_SLOTS, F._num_ctas(8), F._PC_ROW_WORDS)
    assert F._DeviceState(torch.device("cpu"), 8).plan_cache_pool.shape == pool.shape
    F.set_cp_phase(True)
    assert cpu_state.plan_cache_pool is pool


def test_without_a_pool_rows_allocate_eagerly_only(cpu_state):
    with patch("torch.cuda.is_current_stream_capturing", lambda: False):
        a = F._plan_cache_rows(cpu_state, ("v1", 0), 4)
    assert a.shape == (4, F._PC_ROW_WORDS) and not bool(a.any())
    with patch("torch.cuda.is_current_stream_capturing", lambda: True):
        with pytest.raises(RuntimeError, match="eager call"):
            F._plan_cache_rows(cpu_state, ("v2", 0), 4)


def _state(slots=2, ctas=4):
    return SimpleNamespace(
        plan_cache={},
        plan_cache_pool=torch.zeros(slots, ctas, F._PC_ROW_WORDS, dtype=torch.int64),
        workspace=torch.empty(1),
    )


def test_plan_cache_rows_come_from_the_pool_then_allocate_eagerly():
    st = _state()
    a = F._plan_cache_rows(st, ("v1", 0), 4)
    assert a.data_ptr() == st.plan_cache_pool[0].data_ptr() and a.shape == (4, 40)
    assert F._plan_cache_rows(st, ("v1", 0), 4) is a
    b = F._plan_cache_rows(st, ("v1", 1), 3)
    assert b.data_ptr() == st.plan_cache_pool[1].data_ptr() and b.shape == (3, 40)
    with patch("torch.cuda.is_current_stream_capturing", lambda: False):
        c = F._plan_cache_rows(st, ("v2", 0), 4)
    assert c.shape == (4, 40) and not bool(c.any())
    assert c.data_ptr() not in {t.data_ptr() for t in st.plan_cache_pool}


def test_plan_cache_rows_refuse_to_allocate_inside_a_capture():
    st = _state(slots=1)
    F._plan_cache_rows(st, ("v1", 0), 4)
    with patch("torch.cuda.is_current_stream_capturing", lambda: True):
        with pytest.raises(RuntimeError, match="eager call"):
            F._plan_cache_rows(st, ("v2", 0), 4)


def _env(monkeypatch, switch, slots):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "switch, slots, key, cp, on",
    [
        ("1", None, ("tep8", topology.GB300), {"skv2_cp"}, True),
        ("1", None, ("tp8", topology.GB300), {"draft_skv2_cp"}, True),
        (None, None, ("tep8", topology.GB300), {"skv2_cp"}, False),
        ("0", None, ("tep8", topology.GB300), {"skv2_cp"}, False),
        ("1", "skv2,skv2_cp", ("tep8", topology.GB300), {"skv2_cp"}, False),
        ("1", None, ("tep8", _B300), {"skv2_cp"}, False),
        ("1", None, ("tp4", topology.GB300), {"skv2_cp"}, False),
        ("1", None, ("tep8", topology.GB300), set(), False),
        ("1", None, None, {"skv2_cp"}, False),
    ],
)
def test_the_constants_follow_measured_rows_on(monkeypatch, switch, slots, key, cp, on):
    _env(monkeypatch, switch, slots)
    monkeypatch.setattr(runtime, "MEASURED_KEY", key)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "mka"} | cp))
    assert runtime._cp_lite_constants_on() is on


def _build_cp_lite(slots, created=True):
    """build_cp_lite, then apply_cp_lite_build, as a build runs them."""

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
        runtime.apply_cp_lite_build()


@pytest.mark.parametrize("on", [True, False])
def test_the_build_sets_the_phase_from_the_rule(monkeypatch, on):
    monkeypatch.setattr(runtime, "_cp_lite_constants_on", lambda: on)
    _build_cp_lite(frozenset({"skv2", "skv2_cp"}))
    assert _flags() == (on, on, on)


def test_no_phase_without_cp_lite(monkeypatch):
    monkeypatch.setattr(runtime, "_cp_lite_constants_on", lambda: True)
    _build_cp_lite(frozenset({"skv2", "skv2_cp"}), created=False)
    assert _flags() == (False, False, False)


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("1", "skv2,skv2_cp,mka", False)],
)
def test_the_measured_tep8_build_runs_the_constants(monkeypatch, switch, slots, on):
    _env(monkeypatch, switch, slots)
    monkeypatch.setattr(runtime, "MEASURED_KEY", None)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    monkeypatch.setattr(plugin_binding, "megakernel_agree", lambda *a, **k: None)
    parallel = SimpleNamespace(
        tp_size=8, attn_dp_size=1, moe_ep_size=4, attn_tp_size=8, pp_size=1
    )
    runtime.apply_measured_plan(parallel, topology.GB300)
    _build_cp_lite(runtime.SLOTS)
    assert _flags() == (on, on, on)
