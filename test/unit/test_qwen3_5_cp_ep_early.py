# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import inspect
import logging
from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.qwen3_5 import runtime
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F


@pytest.fixture(autouse=True)
def _phase_off():
    F.set_cp_phase(False)
    yield
    F.set_cp_phase(False)


def test_off_by_default():
    spec = importlib.util.spec_from_file_location("q35_cp_fused_copy", F.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh._EP_EARLY is False


@pytest.mark.parametrize(
    "args, kw, want",
    [
        ((True,), {}, True),
        ((True,), {"ep_early": False}, False),
        ((True, False, False), {}, True),
        ((False,), {}, False),
        ((False,), {"ep_early": True}, False),
    ],
)
def test_the_lean_phase_carries_it(caplog, args, kw, want):
    caplog.set_level(logging.INFO)
    F.set_cp_phase(*args, **kw)
    assert F._EP_EARLY is want
    if F._LEAN_PHASE:
        assert f"early push epochs {want}" in caplog.text


@pytest.mark.parametrize("phase", [True, False])
def test_only_a_pushing_call_builds_it(phase):
    F.set_cp_phase(phase)
    assert F._push_opts(False, 7, 4, True) == {}
    want = dict(o_push=True, ql_real=7, hq_real=4, late_trigger=True)
    if phase:
        want["ep_early"] = True
    assert F._push_opts(True, 7, 4, True) == want


def test_the_compile_cache_key_carries_it():
    params = inspect.signature(F._compiled.__wrapped__).parameters
    assert params["ep_early"].default is False
    names = list(params)
    assert names.index("ep_early") > names.index("o_push")


@pytest.mark.parametrize("on", [True, False])
def test_the_build_sets_it_with_the_phase(monkeypatch, on):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    monkeypatch.setattr(runtime, "_cp_lite_constants_on", lambda: on)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp"}))
    monkeypatch.setattr(L.CpLite, "_instance", SimpleNamespace())
    runtime.apply_cp_lite_build()
    assert F._EP_EARLY is on
