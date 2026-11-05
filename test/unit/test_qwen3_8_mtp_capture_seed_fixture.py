# SPDX-License-Identifier: Apache-2.0
"""Synthetic fixture plan/real integer storage comparison; no CUDA arithmetic."""

import importlib.util
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpCaptureSeedIO,
    MtpCaptureSeedSignature,
    NumericOperation,
)

path = Path(__file__).parents[1] / "smoke/qwen3_8_mtp_capture_seed.py"
spec = importlib.util.spec_from_file_location("mtp_capture_smoke", path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_plan_covers_all_integer_dtype_and_pdl_variants_without_widening_domain():
    cases = helper.plan()
    assert len(cases) == 32
    assert {
        (
            c["extend_scalar_type"],
            c["accept_scalar_type"],
            c["request_scalar_type"],
            c["length_scalar_type"],
            c["pdl"],
        )
        for c in cases
    } == {
        (a, b, c, d, p)
        for a in (4, 5)
        for b in (4, 5)
        for c in (4, 5)
        for d in (4, 5)
        for p in (0, 1)
    }
    assert all(
        c["rows"] == 4
        and c["front_tokens"] == 0
        and c["columns"] == 2051
        and c["seed_columns"] == 2055
        for c in cases
    )
    assert all(type(v) is int for c in cases for v in c.values())


@pytest.mark.parametrize("case_index", range(32))
def test_real_Torch_builder_singleton_stride_exact_comparison_and_no_mutation(
    monkeypatch, case_index
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    for name in ("empty_strided", "arange", "full"):
        original = getattr(torch, name)

        def cpu(*a, _fn=original, **kw):
            kw["device"] = "cpu"
            return _fn(*a, **kw)

        monkeypatch.setattr(torch, name, cpu)
    s = MtpCaptureSeedSignature(**helper.plan()[case_index])
    io = helper.make_io(s, case_index)
    other = helper.clone_io(io)
    original = [getattr(io, f.name) for f in fields(io)]
    before = [
        (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.data_ptr(),
            t.untyped_storage().data_ptr(),
            t.clone(),
        )
        for t in original
    ]
    helper.compare_exact(io, other)
    for t, values in zip(original, before):
        assert (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.data_ptr(),
            t.untyped_storage().data_ptr(),
        ) == values[:5]
        assert torch.equal(t, values[5])
    assert io.requests.stride() == (3,) and io.seed_out.stride() == (2080, 1)
    other.seed_out[7, 2054] += 1
    with pytest.raises(AssertionError, match="byte difference: seed_out"):
        helper.compare_exact(io, other)


def test_reference_original_wrapper_flag_restore_and_all_borrowed_pointers():
    seen = []

    def capture(*args):
        seen.append(args)
        return True

    module = SimpleNamespace(_PDL_SMALL="original", try_capture_metadata=capture)
    io = MtpCaptureSeedIO(
        *(torch.zeros(1, dtype=torch.int32) for _ in fields(MtpCaptureSeedIO))
    )
    s = MtpCaptureSeedSignature(**helper.plan()[0])
    helper.reference_call(module, s, io)
    assert module._PDL_SMALL == "original"
    assert all(a is getattr(io, f.name) for a, f in zip(seen[0][:7], fields(io)))
    assert seen[0][-2:] == (0, 7)


def test_reference_fullSHA_required_before_import(tmp_path):
    wrong = tmp_path / "wrong.py"
    wrong.write_text("raise AssertionError('not imported')")
    with pytest.raises(ValueError, match="frozen original source"):
        helper.load_reference(wrong)


def test_master_failure_finally_persists_failure_receipt(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    out = tmp_path / "own.json"
    with pytest.raises(ValueError, match="master"):
        helper.run(str(out))
    assert json.loads(out.read_text())["status"] == "FAIL"


def test_run_constructs_all32_enum_signatures_before_any_CUDA_IO(monkeypatch, tmp_path):
    from sgl_nvidia_mega.qwen3_8 import spec as spec_module

    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    seen = []

    def build(model, config, context):
        for s in context.numeric_signatures:
            assert s.operation is NumericOperation.MTP_CAPTURE_SEED
            assert s.operation.name == "MTP_CAPTURE_SEED"
            seen.append(s)
        raise RuntimeError("CPU stop before any GPU tensor/reference")

    monkeypatch.setattr(spec_module, "SPEC", SimpleNamespace(build=build))
    with pytest.raises(RuntimeError, match="CPU stop"):
        helper.run(str(tmp_path / "out.json"))
    assert len(seen) == 32


def test_reference_private_dependency_import_without_production_module_alias(
    monkeypatch, tmp_path
):
    import hashlib
    import sys

    path = tmp_path / "source.py"
    dep = tmp_path / "original_pdl_small.py"
    path.write_text("from sglang.kernels.ops.pdl_small import ENABLED as _PDL_SMALL\n")
    dep.write_text("ENABLED = True\n")
    monkeypatch.setattr(
        helper, "REFERENCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        helper, "PDL_REFERENCE_SHA256", hashlib.sha256(dep.read_bytes()).hexdigest()
    )
    previous = sys.modules.get("sglang.kernels.ops.pdl_small")
    ref = helper.load_reference(path)
    assert ref._PDL_SMALL is True
    assert sys.modules.get("sglang.kernels.ops.pdl_small") is previous
    dep.write_text("ENABLED = False\n")
    with pytest.raises(ValueError, match="PDL dependency differs"):
        helper.load_reference(path)
