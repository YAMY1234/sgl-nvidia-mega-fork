# SPDX-License-Identifier: Apache-2.0
"""Execute unchanged raw Python selector/cache with CPU-only CUDA API spies."""

import ast
import ctypes
import os
import sys
from dataclasses import fields, replace
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_qwen3_8_gdn_core_norm import signature, view

from sgl_nvidia_mega.qwen3_8._providers import gdn


class PointerView:
    next_ptr = 0x1000

    def __init__(self, meta):
        self.meta = meta
        PointerView.next_ptr += 0x100
        self.pointer = PointerView.next_ptr
        self.device = torch.device("cuda", 2)

    def __getattr__(self, name):
        return getattr(self.meta, name)

    def data_ptr(self):
        return self.pointer


def pointer_io(io, hk):
    ring = replace(
        io.ring,
        d=torch.empty((2, io.raw_out.shape[1], 16, 128), device="meta"),
        k=torch.empty((2, hk, 16, 128), device="meta"),
    )
    ring = replace(
        ring,
        **{
            f.name: PointerView(getattr(ring, f.name))
            for f in fields(ring)
            if getattr(ring, f.name) is not None
        },
    )
    return replace(
        io,
        ring=ring,
        **{
            f.name: PointerView(getattr(io, f.name))
            for f in fields(io)
            if isinstance(getattr(io, f.name), torch.Tensor)
        },
    )


def raw_module(monkeypatch, flat):
    package = "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_mkgdn"
    path = Path(gdn.__file__).parents[1] / "kernels/ops/attention/q38_mkgdn.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Params"]
    nodes += [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_check", "_kernel", "core")
    ]
    state = SimpleNamespace(capturing=False, compiled=[], launches=[])
    nvrtc = ModuleType("cuda.bindings.nvrtc")

    def create(source, *args):
        assert not state.capturing
        state.compiled.append(source.decode())
        return (0, len(state.compiled))

    nvrtc.nvrtcCreateProgram = create
    nvrtc.nvrtcCompileProgram = lambda *a: (0,)
    nvrtc.nvrtcGetCUBINSize = lambda *a: (0, 1)
    nvrtc.nvrtcGetCUBIN = lambda *a: (0,)
    nvrtc.nvrtcDestroyProgram = lambda *a: (0,)
    cu = ModuleType("cuda.bindings.driver")
    cu.cuModuleLoadData = lambda *a: (0, "module")
    cu.cuModuleGetFunction = lambda *a: (0, "function")
    cu.CUlaunchConfig = SimpleNamespace
    cu.CUlaunchAttribute = lambda: SimpleNamespace(value=SimpleNamespace())
    cu.CUlaunchAttributeID = SimpleNamespace(
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION=1
    )
    cu.CUstream = lambda x: x
    cu.cuLaunchKernelEx = lambda *a: state.launches.append(a) or (0,)
    cuda = ModuleType("cuda")
    cuda.__path__ = []
    bindings = ModuleType("cuda.bindings")
    bindings.__path__ = []
    bindings.driver, bindings.nvrtc = cu, nvrtc
    cuda.bindings = bindings
    for name, module in [
        ("cuda", cuda),
        ("cuda.bindings", bindings),
        (cu.__name__, cu),
        (nvrtc.__name__, nvrtc),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    fake_torch = SimpleNamespace(
        bfloat16=torch.bfloat16,
        float32=torch.float32,
        int64=torch.int64,
        cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: state.capturing,
            get_device_capability=lambda d: (10, 7),
            current_stream=lambda d: SimpleNamespace(cuda_stream=77),
        ),
    )
    ns = dict(
        ctypes=ctypes,
        cache=cache,
        torch=fake_torch,
        os=os,
        SOURCE=SimpleNamespace(
            read_text=lambda: "// CPU selector test, no GPU compile"
        ),
        _BS1_VARIANT=flat,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    module = ModuleType(package)
    module.core = ns["core"]
    monkeypatch.setitem(sys.modules, package, module)
    return ns, state


@pytest.mark.parametrize("rows", [4, 8, 12])
@pytest.mark.parametrize("hk,hv", [(4, 12), (16, 48)])
@pytest.mark.parametrize("dt_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("flat", [False, True])
def test_exact_raw_key_eager_then_capture_and_cold_dtype_refusal(
    monkeypatch, rows, hk, hv, dt_dtype, flat
):
    ns, state = raw_module(monkeypatch, flat)
    s = signature(rows, hk, hv, 1)
    io = view(s)
    io = replace(io, dt_bias=io.dt_bias.to(dt_dtype))
    io = pointer_io(io, hk)
    run = gdn.build_core(s)
    assert run(io) is None
    assert len(state.compiled) == len(state.launches) == 1
    source = state.compiled[0]
    assert f"#define DT_BF16 {int(dt_dtype == torch.bfloat16)}" in source
    assert "#define FUSED_NORM 1" in source
    assert "#define NORM_F32 0" in source
    assert ("#define FLAT_STORES 1" in source) == (flat and rows == 4 and hk == 4)
    state.capturing = True
    assert run(io) is None
    assert len(state.compiled) == 1 and len(state.launches) == 2
    # Numeric variant1 is not the raw FLAT_STORES flag; dt_bias dtype independently keys NVRTC.
    other_dtype = torch.float32 if dt_dtype == torch.bfloat16 else torch.bfloat16
    changed = replace(
        io, dt_bias=PointerView(torch.empty((hv,), dtype=other_dtype, device="meta"))
    )
    with pytest.raises(RuntimeError, match="prewarm MK_GDN"):
        run(changed)
    assert len(state.compiled) == 1 and len(state.launches) == 2
    assert ns["_kernel"].cache_info().hits == 1
