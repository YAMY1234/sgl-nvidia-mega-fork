# SPDX-License-Identifier: Apache-2.0
"""Capture uses graph-private activation addresses, not eager allocation keys."""

import ast
import gc
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_qwen3_8_experts import CASES, Tensor, gpu_doubles, signature, view

from sgl_nvidia_mega.qwen3_8._providers import experts


@pytest.mark.parametrize("rows,variant", CASES)
def test_new_activation_address_during_capture_is_host_rebound(
    monkeypatch, rows, variant
):
    calls, state, _, _ = gpu_doubles(monkeypatch)
    run = experts.build(signature(rows, variant))
    io = view(rows)
    run(io)
    first = io.packed_x.data_ptr()
    io2 = replace(io, packed_x=Tensor((rows, 1280), 2001))
    run(io2)
    assert len(run._resources) == 1
    assert len([1 for name, _ in calls if name.endswith("-maps")]) == 1
    assert len([1 for name, _ in calls if name in ("base", "sf", "wide")]) == 1
    key = run._resource_key(io)
    assert key == run._resource_key(io2)
    assert key[-2:] == (rows, variant)
    assert key[:3] == (2, io.packed_w13.data_ptr(), io.packed_w2.data_ptr())
    assert [args[0] for name, args in calls if name == "host-x-map"] == [first, 2001]
    calls.clear()
    state.capturing = True
    io3 = replace(io, packed_x=Tensor((rows, 1280), 3002))
    assert run(io3) is None
    assert len(run._resources) == 1
    # Any load, maps/allocation, compile or sync hook would be an unexpected event.
    assert [name for name, _ in calls] == [
        "host-x-map",
        "narrow-run" if rows <= 7 else "wide-run",
        "finalize",
    ]
    host = calls[0][1]
    assert host == (
        3002,
        dict(
            rows=rows,
            row_bytes=1280,
            box_bytes=128,
            box_rows=8 if rows <= 7 else 16,
            swizzle=128,
        ),
    )
    launch = calls[1][1]
    assert launch[1][-1] == ("x-map", 3002)


@pytest.mark.parametrize("rows,variant", [(4, 0), (4, 1), (12, 2), (16, 2)])
def test_new_weight_key_during_capture_still_hard_fails(monkeypatch, rows, variant):
    calls, state, _, _ = gpu_doubles(monkeypatch)
    io = view(rows)
    run = experts.build(signature(rows, variant))
    run(io)
    calls.clear()
    state.capturing = True
    changed = replace(io, packed_w13=Tensor(io.packed_w13.shape, 4003))
    with pytest.raises(RuntimeError, match="prepared before capture"):
        run(changed)
    assert calls == []


@pytest.mark.parametrize("rows,variant", [(4, 0), (4, 1), (12, 2)])
def test_resource_does_not_retain_activation_owner(monkeypatch, rows, variant):
    calls, _, _, _ = gpu_doubles(monkeypatch)
    run = experts.build(signature(rows, variant))
    io = view(rows)
    activation = weakref.ref(io.packed_x)
    run(io)
    calls.clear()  # The spy's call log deliberately owns all recorded arguments.
    del io
    gc.collect()
    assert activation() is None
    assert len(run._resources) == 1


def test_original_host_encoder_has_no_cuda_alloc_compile_or_sync():
    path = Path(experts.__file__).parents[1] / "kernels/ops/moe/q38_mk_moe.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "tensor_map"
    )
    calls = []
    enum = SimpleNamespace
    driver = SimpleNamespace(
        cuuint64_t=int,
        cuuint32_t=int,
        CUtensorMapDataType=enum(CU_TENSOR_MAP_DATA_TYPE_UINT8=1),
        CUtensorMapInterleave=enum(CU_TENSOR_MAP_INTERLEAVE_NONE=0),
        CUtensorMapL2promotion=enum(CU_TENSOR_MAP_L2_PROMOTION_L2_256B=256),
        CUtensorMapFloatOOBfill=enum(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0),
        cuTensorMapEncodeTiled=lambda *a: calls.append(a) or ("descriptor",),
    )
    ns = {
        "jit": SimpleNamespace(
            driver=lambda: driver, check=lambda result, label: result
        )
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    assert (
        ns["tensor_map"](Tensor((4, 1280), 3002), 4, 1280, 128, 8, 128) == "descriptor"
    )
    assert len(calls) == 1
    assert calls[0][2:6] == (3002, [1280, 4], [1280], [128, 8])
    # These are the only calls in the exact unchanged host encoder.
    names = {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
    assert names == {
        "jit.driver",
        "jit.check",
        "cu.cuTensorMapEncodeTiled",
        "tensor.data_ptr",
        "u64",
        "u32",
    }
