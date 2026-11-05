# SPDX-License-Identifier: Apache-2.0
"""Synthetic C03 metadata, real BF16 storage/bytes, independent source dependency."""

import hashlib
import importlib.util
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpDeferredKvSignature,
    NumericOperation,
)

path = Path(__file__).parents[1] / "smoke/qwen3_8_mtp_deferred_kv.py"
spec = importlib.util.spec_from_file_location("mtp_deferred_smoke", path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_plan_all64_dtype_masks_no_TP1_FP8_or_fullfront_widening():
    cases = helper.plan()
    assert len(cases) == 64
    assert {c["metadata_dtype_mask"] for c in cases} == set(range(64))
    assert all(
        (c["rows"], c["kv_heads"], c["head_dim"], c["scalar_type"]) == (1, 1, 256, 1)
        for c in cases
    )
    assert all(
        (c["index_columns"], c["packed_columns"], c["tail_width"]) == (2055, 2112, 4)
        for c in cases
    )
    assert all(c["new_k_row_stride"] == c["new_v_row_stride"] == 256 for c in cases)
    assert all(type(v) is int for c in cases for v in c.values())


@pytest.mark.parametrize("case_index", range(64))
def test_actual_builder_shape_strides_all15_bytes_and_originals_unchanged(
    monkeypatch, case_index
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    for name in ("randn", "arange", "empty_strided", "full"):
        original = getattr(torch, name)

        def cpu(*a, _fn=original, **kw):
            kw["device"] = "cpu"
            return _fn(*a, **kw)

        monkeypatch.setattr(torch, name, cpu)
    s = MtpDeferredKvSignature(**helper.plan()[case_index])
    io = helper.make_io(s, case_index)
    assert io.new_k.stride() == io.new_v.stride() == (256, 256, 1)
    assert io.new_v.storage_offset() == 3328 and io.indices_pool.shape == (8, 2055)
    assert io.locations.shape == (1,) and io.counts_out.dtype == torch.int32
    expected = helper.clone_io(io)
    tensors = [getattr(view, f.name) for view in (io, expected) for f in fields(view)]
    before = [
        (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.data_ptr(),
            t.untyped_storage().data_ptr(),
            t.clone(),
        )
        for t in tensors
    ]
    helper.compare_exact(io, expected)
    for t, values in zip(tensors, before):
        assert (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.data_ptr(),
            t.untyped_storage().data_ptr(),
        ) == values[:5]
        assert torch.equal(t, values[5])
    expected.counts_out[0] += 1
    with pytest.raises(AssertionError, match="byte difference: counts_out"):
        helper.compare_exact(io, expected)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_matching_nonfinite_is_rejected(monkeypatch, value):
    # Explicit meta device is not involved; construct tiny real storage instead.
    from sglang.kernels.plugin_kernels.nvidia.numeric import MtpDeferredKvIO

    io = MtpDeferredKvIO(*(torch.zeros(1) for _ in fields(MtpDeferredKvIO)))
    io.packed_k_out.fill_(value)
    with pytest.raises(AssertionError, match="nonfinite"):
        helper.compare_exact(io, helper.clone_io(io))


def test_original_reference_dependency_is_private_and_fullSHA_checked(
    monkeypatch, tmp_path
):
    import sys

    path = tmp_path / "source.py"
    dep = tmp_path / "original_mtp_lookup.py"
    path.write_text(
        "from sglang.srt.layers.attention.qsa.mtp_lookup import DeferredMtpSparseIndices\n"
    )
    dep.write_text("class DeferredMtpSparseIndices: pass\n")
    monkeypatch.setattr(
        helper, "REFERENCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        helper, "LOOKUP_REFERENCE_SHA256", hashlib.sha256(dep.read_bytes()).hexdigest()
    )
    previous = sys.modules.get("sglang.srt.layers.attention.qsa.mtp_lookup")
    reference = helper.load_reference(path)
    assert (
        reference.DeferredMtpSparseIndices.__module__
        == "original_mtp_lookup_dependency"
    )
    assert sys.modules.get("sglang.srt.layers.attention.qsa.mtp_lookup") is previous
    dep.write_text('raise AssertionError("must not import")\n')
    with pytest.raises(ValueError, match="lookup dependency differs"):
        helper.load_reference(path)


def test_helper_enum64_registration_precedes_all_CUDA_IO_and_failure_receipt(
    monkeypatch, tmp_path
):
    from sgl_nvidia_mega.qwen3_8 import spec as spec_module

    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    seen = []

    def build(model, config, context):
        for s in context.numeric_signatures:
            assert s.operation is NumericOperation.MTP_DEFERRED_KV_PREPARE
            assert s.operation.name == "MTP_DEFERRED_KV_PREPARE"
            seen.append(s)
        raise RuntimeError("CPU stop after enum64 registration")

    monkeypatch.setattr(spec_module, "SPEC", SimpleNamespace(build=build))
    out = tmp_path / "own.json"
    with pytest.raises(RuntimeError, match="CPU stop"):
        helper.run(str(out))
    assert len(seen) == 64 and json.loads(out.read_text())["status"] == "FAIL"
