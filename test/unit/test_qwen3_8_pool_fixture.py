# SPDX-License-Identifier: Apache-2.0
"""Real canonical fixture IO/facade replay on CPU with explicit CUDA doubles."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import numeric as n
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from test_qwen3_8_pool import raw_spy

from sgl_nvidia_mega.qwen3_8 import providers

path = Path(__file__).parents[1] / "smoke/qwen3_8_pool.py"
spec = importlib.util.spec_from_file_location("qwen3_8_pool_smoke", path)
f = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = f
spec.loader.exec_module(f)


@pytest.fixture
def outdir(monkeypatch, request):
    root = (
        Path(__file__).resolve().parents[3] / "cpu-fixture-replay" / request.node.name
    )
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("M7_CANONICAL_OUT", str(root))
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "qsa_core")
    monkeypatch.delenv("FN_SMOKE_ONLY_OPERATION", raising=False)
    monkeypatch.delenv("SGLANG_QSA_POOL_DECODE", raising=False)
    return root


def test_fixture_plan_exact_56():
    cases = f.plan()
    assert len(cases) == 56
    assert {c.variant for c in cases} == {1, 2}
    assert {c.length for c in cases} == set(f.LENGTHS)
    assert all(providers.supported(f.signature(c)) for c in cases)
    assert sum(c.variant == 1 for c in cases) == 28


@pytest.mark.parametrize("case", f.plan())
def test_real_canonical_fixture_metadata(monkeypatch, case):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(multi_processor_count=200),
    )
    io = f.inputs(case, "meta")
    assert isinstance(io, n.DirectPoolAttentionIO)
    assert io.q.shape == io.output.shape == (case.rows, 24, 256)
    assert io.gate is None and io.query_width == case.width
    # A one-row view may canonicalize its unused leading stride to512.
    assert (
        io.new_k.stride()
        == io.new_v.stride()
        == ((512 if case.rows == 1 else 2304), 256, 1)
    )
    assert io.selection.selected_indices.dtype == torch.int32
    assert io.selection.full_slot_map.shape == (case.rows // case.width, 8193)
    assert io.scratch_out.dtype == torch.float32 and io.scratch_out.numel() > 0


@pytest.mark.parametrize("value", ["", "all", "GDN_CORE", None])
def test_scope_rejected(value):
    with pytest.raises(ValueError):
        f.plan(value)


@pytest.mark.parametrize(
    "bad",
    ["master", "source", "source_empty", "slot", "relative", "outside", "operation"],
)
def test_deployment_hard_gates(monkeypatch, outdir, bad):
    target = outdir / "negative.json"
    op = f.OPERATION
    if bad == "master":
        monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    elif bad == "source":
        monkeypatch.setenv("SGLANG_QSA_POOL_DECODE", "0")
    elif bad == "source_empty":
        monkeypatch.setenv("SGLANG_QSA_POOL_DECODE", "")
    elif bad == "slot":
        monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "gdn_mk")
    elif bad == "relative":
        target = Path("negative.json")
    elif bad == "outside":
        target = outdir.parent / "negative.json"
    else:
        op = "GDN_CORE"
    with pytest.raises((AssertionError, ValueError)):
        f.check_env(target, op)


def test_alias_rejected():
    one = {"q": dict(ptr=3, shape=[4, 24, 256], stride=[6144, 256, 1], dtype="bf16")}
    with pytest.raises(AssertionError, match="aliases"):
        f.assert_distinct((one, one, one))


def runtime(monkeypatch):
    raw, state = raw_spy(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 7))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: SimpleNamespace(multi_processor_count=200),
    )

    class Stream:
        def wait_stream(self, other):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: Stream())
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None)
    )

    @contextmanager
    def graph(*a, **kw):
        state.capturing = True
        try:
            yield
        finally:
            state.capturing = False

    monkeypatch.setattr(torch.cuda, "graph", graph)
    inputs = f.inputs
    monkeypatch.setattr(f, "inputs", lambda c, d: inputs(c, "cpu"))
    # Use the actual registry/facade/adapter. The final raw merge is a CPU spy.
    original = raw._qsa_pool_merge

    class Merge:
        def __getitem__(self, grid):
            launch = original[grid]

            def run(*args, **kw):
                result = launch(*args, **kw)
                args[3].zero_()
                return result

            return run

    raw._qsa_pool_merge = Merge()
    registry = n.NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(n, "numeric_registry", registry)
    providers.make_offer.cache_clear()
    return raw, state


def test_full_fixture_control_flow_real_registry_facade_raw_spies(monkeypatch, outdir):
    _, state = runtime(monkeypatch)
    cases = (
        f.Case(4, 1, 4, 499),
        f.Case(32, 1, 4, 8193),
        f.Case(1, 2, 1, 513),
        f.Case(8, 2, 1, 8192),
    )
    monkeypatch.setattr(f, "plan", lambda operation=f.OPERATION: cases)
    out = outdir / "RESULT.json"
    try:
        f.run(out, operation=f.OPERATION)
    finally:
        providers.make_offer.cache_clear()
    result = json.loads(out.read_text())
    assert result["status"] == "PASS_TP1_POOL_BUILD_LAUNCH_SCOPE"
    assert len(result["records"]) == 4
    assert result["successful_offer_fixture_calls"] == {f.OPERATION: 8}
    assert all(
        r["gate_applied"] is False and r["target_signature"]["variant"] in (1, 2)
        for r in result["records"]
    )
    assert len(state.calls) == 4 * 3 * 3
    assert state.props == 4


def test_failure_archives_phase_and_keeps_HOLD(monkeypatch, outdir):
    _, state = runtime(monkeypatch)
    state.fail = "split"
    monkeypatch.setattr(
        f, "plan", lambda operation=f.OPERATION: (f.Case(4, 1, 4, 499),)
    )
    out = outdir / "FAILED.json"
    try:
        with pytest.raises(RuntimeError, match="raw launch failed"):
            f.run(out)
    finally:
        providers.make_offer.cache_clear()
    result = json.loads(out.read_text())
    assert result["status"] == "HOLD"
    assert result["current"]["phase"] == "eager_A"
    assert "pointers_ABC" in result["current"]
    assert result["records"][-1]["status"] == "FAILED"
    assert [c[0] for c in state.calls] == ["store", "split"]
