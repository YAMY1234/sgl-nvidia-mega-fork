# SPDX-License-Identifier: Apache-2.0
"""Diagnostic plan/reference/receipt checks; no CUDA or kernel arithmetic."""

import importlib.util
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NativeGdnPdlIO,
    NativeGdnPdlSignature,
)

PATH = Path(__file__).parents[1] / "smoke/qwen3_8_native_gdn_pdl.py"
spec = importlib.util.spec_from_file_location("native_pdl_fixture", PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def tiny_io():
    # Comparator/clone tests only, deliberately not a valid CUDA launch view.
    values = [
        torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
        for _ in fields(NativeGdnPdlIO)
    ]
    values[12:15] = [None, None, None]
    return NativeGdnPdlIO(*values)


def test_plan_covers_source_storage_flags_and_independent_heads():
    cases = helper.plan()
    assert len(cases) == 12
    for name, expected in (
        ("storage_mode", {0, 1, 2}),
        ("state_scalar_type", {1, 3}),
        ("dt_bias_scalar_type", {1, 3}),
        ("wait_for_gate_producer", {0, 1}),
        ("state_prefetch", {0, 1}),
    ):
        assert {c[name] for c in cases} == expected
    assert {(c["key_heads"], c["value_heads"]) for c in cases} == {(4, 12), (16, 48)}
    assert any(
        c["wait_for_gate_producer"]
        == c["state_prefetch"]
        == c["state_scalar_type"]
        == 1
        for c in cases
    )
    assert all(type(v) is int for c in cases for v in c.values())
    assert {c["dot_precision"] for c in helper.plan(2)} == {2}


def test_clone_retains_strides_and_distinct_storage():
    io = tiny_io()
    clone = helper.clone_io(io)
    helper.compare_exact(io, clone)
    for f in fields(io):
        t, other = getattr(io, f.name), getattr(clone, f.name)
        if t is not None:
            assert t.data_ptr() != other.data_ptr()
            assert t.stride() == other.stride()


@pytest.mark.parametrize("case_index", range(12))
def test_real_fixture_builder_metadata_and_strides_with_cpu_device_double(
    monkeypatch, case_index
):
    # Allocation functions are real Torch CPU ops; only requested CUDA device
    # selection is redirected. This does not qualify a CUDA launch or context.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    for name in ("zeros", "randn", "full", "tensor", "arange"):
        original = getattr(torch, name)

        def cpu(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu)
    s = NativeGdnPdlSignature(**helper.plan()[case_index])
    mode = s.storage_mode
    io = helper.make_io(s)
    assert (
        io.q.stride(0)
        == io.k.stride(0)
        == io.v.stride(0)
        == (2 * s.key_heads + s.value_heads) * 128
    )
    assert io.a.stride(0) == io.b.stride(0) == 2 * s.value_heads
    assert (
        io.query_start_loc.stride(0),
        io.ssm_state_indices.stride(0),
        io.replay_indices.stride(0),
    ) == (2, 3, 4)
    assert io.query_start_loc.tolist() == list(range(0, s.rows + 1, 4))
    assert io.ssm_state_indices.tolist() == list(range(1, s.rows // 4 + 1))
    assert io.checkpoint_state.shape[0] != io.d_cache.shape[0]
    assert (
        tuple(t is not None for t in (io.rawv_cache, io.rawk_cache, io.beta_cache))
        == ((False, False, False), (True, True, False), (True, True, True))[mode]
    )
    assert (
        io.is_flush.dtype == torch.int8 and torch.count_nonzero(io.checkpoint_state) > 0
    )
    expected = helper.clone_io(io)
    originals = [
        getattr(view, f.name)
        for view in (io, expected)
        for f in fields(view)
        if getattr(view, f.name) is not None
    ]
    metadata = [
        (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.data_ptr(),
            t.untyped_storage().data_ptr(),
            t.untyped_storage().nbytes(),
        )
        for t in originals
    ]
    contents = [t.clone() for t in originals]
    helper.compare_exact(io, expected)
    for t, before, values in zip(originals, metadata, contents):
        assert (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.data_ptr(),
            t.untyped_storage().data_ptr(),
            t.untyped_storage().nbytes(),
        ) == before
        assert torch.equal(t, values)
    if s.rows == 4:
        assert io.ssm_state_indices.shape == io.replay_indices.shape == (1,)
        assert io.ssm_state_indices.stride() == (3,) and io.replay_indices.stride() == (
            4,
        )
        assert (
            io.ssm_state_indices.is_contiguous() and io.replay_indices.is_contiguous()
        )
    # Preserve byte-zero strictness for integer metadata too.
    expected.ssm_state_indices[0] += 1
    with pytest.raises(AssertionError, match="byte difference: ssm_state_indices"):
        helper.compare_exact(io, expected)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_identical_nonfinite_is_not_source_equality(value):
    io = tiny_io()
    io.out.fill_(value)
    with pytest.raises(AssertionError, match="nonfinite"):
        helper.compare_exact(io, helper.clone_io(io))


def test_comparison_detects_ordered_bytes_signedzero_and_layout():
    io = tiny_io()
    clone = helper.clone_io(io)
    clone.out[0, 0] = -0.0
    with pytest.raises(AssertionError, match="byte difference"):
        helper.compare_exact(io, clone)
    with pytest.raises(AssertionError, match="stride"):
        helper.compare_exact(io, replace(clone, out=clone.out.contiguous()))


def test_reference_exact_file_hash_is_required_before_module_import(tmp_path):
    path = tmp_path / "wrong-reference.py"
    path.write_text("raise AssertionError('must never be imported')\n")
    with pytest.raises(ValueError, match="frozen original source"):
        helper.load_reference(path)


def test_reference_wrapper_gets_real21_pointers_and_source_flags(monkeypatch):
    from sglang.kernels.ops.attention.fla import utils

    s = NativeGdnPdlSignature(**helper.plan(0 if utils.is_tf32_supported else 2)[0])
    io = tiny_io()
    calls = []

    def verify(*args, **kw):
        calls.append((args, kw))
        return io.out

    monkeypatch.setenv("Q38_GDN_STATE_PREFETCH", "unchanged-before-reference")
    assert helper.reference_call(SimpleNamespace(verify=verify), s, io) is io.out
    assert all(a is getattr(io, f.name) for a, f in zip(calls[0][0], fields(io)))
    assert len(calls[0][0]) == 21 and calls[0][1]["pdl"] is True
    import os

    assert os.environ["Q38_GDN_STATE_PREFETCH"] == "unchanged-before-reference"


def test_master_failure_still_persists_own_receipt(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    path = tmp_path / "own-failure.json"
    with pytest.raises(ValueError, match="master"):
        helper.run(str(path))
    result = json.loads(path.read_text())
    assert result["status"] == "FAIL" and result["records"] == []
    assert "master must be explicitly enabled" in result["traceback"]


def test_run_uses_canonical_operation_enum_before_registration(monkeypatch, tmp_path):
    from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation

    from sgl_nvidia_mega.qwen3_8 import spec as spec_module

    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setenv("FN_SMOKE_ONLY_OPERATION", "GDN_NATIVE_PDL")
    seen = []

    def register(model, config, context):
        for signature in context.numeric_signatures:
            # Real observer86 and driver summaries consume .operation.name.
            assert signature.operation is NumericOperation.GDN_NATIVE_PDL
            assert signature.operation.name == "GDN_NATIVE_PDL"
            seen.append(signature)
        raise RuntimeError("CPU stop after all real signatures; before CUDA IO")

    monkeypatch.setattr(spec_module, "SPEC", SimpleNamespace(build=register))
    path = tmp_path / "enum-registration.json"
    with pytest.raises(RuntimeError, match="CPU stop after all real signatures"):
        helper.run(str(path))
    assert len(seen) == 12
    for actual, case in zip(seen, helper.plan(seen[0].dot_precision)):
        got = {f.name: int(getattr(actual, f.name)) for f in fields(actual)}
        # SGLang with the verify stage count declares the default two stages.
        assert got.pop("num_stages", 2) == 2
        assert got == case
    assert json.loads(path.read_text())["status"] == "FAIL"


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_singleton_integer_nonunit_stride_and_offset_compare_without_mutation(dtype):
    io = tiny_io()
    index = torch.arange(12, dtype=dtype)[2::3][:1]
    assert (
        index.shape == (1,) and index.stride() == (3,) and index.storage_offset() == 2
    )
    assert index.is_contiguous() and index.contiguous().data_ptr() == index.data_ptr()
    with pytest.raises(RuntimeError, match="stride"):
        index.contiguous().view(torch.uint8)
    io = replace(io, ssm_state_indices=index)
    expected = helper.clone_io(io)
    before = (
        index.shape,
        index.stride(),
        index.storage_offset(),
        index.data_ptr(),
        index.untyped_storage().data_ptr(),
        index.clone(),
    )
    helper.compare_exact(io, expected)
    assert (
        index.shape,
        index.stride(),
        index.storage_offset(),
        index.data_ptr(),
        index.untyped_storage().data_ptr(),
    ) == before[:5]
    assert torch.equal(index, before[5])
    expected.ssm_state_indices[0] += 1
    with pytest.raises(AssertionError, match="byte difference: ssm_state_indices"):
        helper.compare_exact(io, expected)
