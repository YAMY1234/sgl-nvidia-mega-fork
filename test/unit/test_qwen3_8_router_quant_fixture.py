# SPDX-License-Identifier: Apache-2.0
"""Facade fixture control flow with explicit CUDA/preparation doubles.

Until the framework producer lands, its one IO_TYPES entry is a test double.
These tests do not qualify a model producer, CUDA arithmetic or engagement.
"""

import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import numeric
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from test_qwen3_8_router_quant import make_io, setup  # noqa: F401

from sgl_nvidia_mega.qwen3_8 import providers


@pytest.fixture
def helper(monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "smoke/qwen3_8_router_quant.py"
    spec = importlib.util.spec_from_file_location("router_quant_fixture", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("M7_CANONICAL_OUT", str(tmp_path))
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "router_quant")
    monkeypatch.setenv("FN_SMOKE_ONLY_OPERATION", "ROUTER_QUANT")
    return module


def test_finite_plan_and_signature_identity(helper):
    assert [asdict(case) for case in helper.plan()] == [
        dict(rows=m, variant=v)
        for m, v in ((1, 0), (4, 0), (1, 1), (4, 1), (8, 2), (16, 2))
    ]
    for case in helper.plan():
        assert asdict(helper.signature(case)) == dict(
            operation=numeric.NumericOperation.ROUTER_QUANT,
            rows=case.rows,
            hidden_size=2560,
            expert_count=512,
            scalar_type=numeric.ScalarType.BF16,
            variant=case.variant,
        )


@pytest.mark.parametrize("index", range(6))
def test_actual_fixture_input_layouts(helper, index):
    case = helper.plan()[index]
    io = helper.inputs(case, torch.device("cpu"))
    assert io.hidden_states.shape == (case.rows, 2560)
    assert io.router_weight.shape == (512, 2560)
    assert io.global_scale.dtype == torch.float32 and io.global_scale.item() == 1
    assert io.logits_out.dtype == torch.bfloat16
    assert io.packed_out.dtype == torch.uint8
    assert io.scale_out.dtype == torch.float8_e4m3fn
    assert all(getattr(io, name).is_contiguous() for name in io.__slots__)


@pytest.mark.parametrize(
    "field",
    (
        "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION",
        "FN_SMOKE_ONLY_OPERATION",
        "SGLANG_NVIDIA_MEGAKERNEL_SLOTS",
    ),
)
def test_disabled_or_wrong_child_rejected(helper, monkeypatch, tmp_path, field):
    monkeypatch.setenv(field, "wrong")
    with pytest.raises(RuntimeError):
        helper.check_env(tmp_path / "OUT.json")


def test_absolute_own_out_required(helper, tmp_path):
    with pytest.raises(ValueError):
        helper.check_env("relative.json")
    with pytest.raises(ValueError):
        helper.check_env(tmp_path)
    with pytest.raises(ValueError):
        helper.check_env(tmp_path.parent / "foreign.json")


def install_runtime_doubles(setup, helper, monkeypatch):
    monkeypatch.setattr(numeric, "numeric_registry", providers.numeric_registry)
    monkeypatch.setitem(
        numeric_binding._IO_TYPES,
        numeric.NumericOperation.ROUTER_QUANT,
        numeric.RouterQuantIO,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    stream = SimpleNamespace(cuda_stream=71, wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None)
    )

    @contextmanager
    def graph(*args, **kwargs):
        setup.capturing = True
        try:
            yield
        finally:
            setup.capturing = False

    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(helper, "inputs", lambda case, device: make_io(case.rows))


def test_six_canonical_facade_abc_cases_and_original_identities(
    setup, helper, monkeypatch, tmp_path
):
    install_runtime_doubles(setup, helper, monkeypatch)
    before = helper.method_identities()
    out = tmp_path / "OUT.json"
    helper.run(out)
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "PASS_ROUTER_QUANT_BUILD_LAUNCH_SCOPE"
    assert len(receipt["records"]) == 6 and len(setup.calls) == 18
    assert receipt["successful_offer_fixture_calls"] == {"ROUTER_QUANT": 12}
    assert receipt["CPU_fixture"] and receipt["helper_method_identities_preserved"]
    assert all(a is b for a, b in zip(before, helper.method_identities()))
    for record in receipt["records"]:
        assert (
            record["canonical_facade"]
            and record["eager_AB"]
            and record["capture_replay_C"]
        )
        assert (
            len(
                {
                    entry["ptr"]
                    for inv in record["pointers_ABC"]
                    for entry in inv.values()
                }
            )
            == 18
        )
    assert receipt["model_engagement"] == "NOT_PROVEN"
    assert receipt["quantizer_skipped_in_model"] == "NOT_PROVEN"


def test_missing_facade_map_fails_and_persists_own_receipt(
    setup, helper, monkeypatch, tmp_path
):
    install_runtime_doubles(setup, helper, monkeypatch)
    monkeypatch.delitem(
        numeric_binding._IO_TYPES, numeric.NumericOperation.ROUTER_QUANT
    )
    out = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="mapping is missing"):
        helper.run(out)
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "HOLD" and receipt["helper_method_identities_preserved"]
    assert receipt["current"]["phase"] == "prepare_views"
    assert receipt["records"][-1]["status"] == "FAILED" and not setup.calls


def test_provider_exception_persists_before_authorization(
    setup, helper, monkeypatch, tmp_path
):
    install_runtime_doubles(setup, helper, monkeypatch)
    setup.error = RuntimeError("deliberate raw launch failure")
    out = tmp_path / "failure.json"
    with pytest.raises(RuntimeError, match="deliberate raw launch failure"):
        helper.run(out)
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "HOLD" and receipt["helper_method_identities_preserved"]
    assert receipt["current"]["phase"] == "eager_A"
    assert receipt["successful_offer_fixture_calls"] == {"ROUTER_QUANT": 0}
    assert not setup.calls


def test_alias_abc_rejected(helper):
    io = make_io(4)
    inventory = helper.addresses(io)
    with pytest.raises(AssertionError, match="alias"):
        helper.assert_distinct((inventory,) * 3)
