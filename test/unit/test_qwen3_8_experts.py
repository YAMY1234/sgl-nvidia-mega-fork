# SPDX-License-Identifier: Apache-2.0
"""Canonical DTO/startup and original launch ABI checks without a CUDA device."""

import ast
import ctypes
import sys
from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizeSharedIO,
    LocalExpertsIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import experts

CASES = [(m, 0) for m in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)] + [
    (1, 1),
    (4, 1),
    (12, 2),
    (16, 2),
]


def signature(rows=4, variant=0, **kw):
    values = dict(
        operation=O.LOCAL_EXPERTS,
        rows=rows,
        hidden_size=2560,
        intermediate_size=192,
        experts=512,
        topk=10,
        scalar_type=D.PACKED_FP4,
        variant=variant,
    )
    values.update(kw)
    return ExpertSignature(**values)


class Tensor:
    def __init__(self, shape, pointer):
        self.shape = shape
        self.pointer = pointer
        self.device = SimpleNamespace(index=2)

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True


def view(rows=4, final=True):
    tensors = [Tensor((1,), 100 + i) for i in range(16)]
    tensors[0].shape = (512, 384, 1280)
    tensors[1].shape = (512, 2560, 96)
    tensors[7].shape = (rows, 1280)
    tensors[9].shape = (rows, 512)
    tensors[9].dtype = torch.bfloat16
    return LocalExpertsIO(
        *tensors[:10],
        experts=512,
        intermediate_size=192,
        hidden_size=2560,
        topk=10,
        activation=3,
        local_expert_start=0,
        local_expert_count=512,
        routed_scale=1.0,
        route_weights_out=tensors[10],
        route_ids_out=tensors[11],
        expert_rows_out=tensors[12],
        final_out=tensors[13] if final else None,
        scratch_out=tensors[14],
    )


def install(monkeypatch, name, **fields):
    module = ModuleType(name)
    module.__dict__.update(fields)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def gpu_doubles(monkeypatch):
    calls = []
    state = SimpleNamespace(capturing=False)
    install(
        monkeypatch,
        "torch",
        bfloat16=torch.bfloat16,
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: state.capturing),
    )

    class ActivationType(IntEnum):
        Swiglu = 3

    parent = install(monkeypatch, "flashinfer")
    parent.__path__ = []
    install(monkeypatch, "flashinfer.tllm_enums", ActivationType=ActivationType)

    def load(kind, *args):
        calls.append((kind, args))
        return (kind + "-kernel", "finalize-handle", {"shared": 32, "params": 3})

    def maps(kind, *args):
        calls.append((kind + "-maps", args))
        return (kind + "-w13", kind + "-w2", kind + "-x")

    def tensor_map(tensor, **layout):
        calls.append(("host-x-map", (tensor.data_ptr(), layout)))
        return ("x-map", tensor.data_ptr())

    driver = SimpleNamespace(
        CUtensorMapSwizzle=SimpleNamespace(CU_TENSOR_MAP_SWIZZLE_128B=128)
    )
    narrow = install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mk_q35",
        MAX_M=7,
        tensor_map=tensor_map,
        jit=SimpleNamespace(driver=lambda: driver),
        _bundle=lambda d: load("base", d),
        _sf_pack_bundle=lambda d: load("sf", d),
        maps=lambda *a: maps("narrow", *a),
        run=lambda *a: calls.append(("narrow-run", a)),
        _finalize=lambda *a: calls.append(("finalize", a)),
    )
    wide = install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mkm_wide",
        bundle=lambda *a: load("wide", *a),
        tensor_map=tensor_map,
        jit=SimpleNamespace(driver=lambda: driver),
        maps=lambda *a: maps("wide", *a),
        run=lambda *a: calls.append(("wide-run", a)),
    )
    return calls, state, narrow, wide


@pytest.mark.parametrize("rows,variant", CASES)
def test_supported_exact_variants(rows, variant):
    s = signature(rows, variant)
    assert experts.supported(s) and providers.supported(s)


@pytest.mark.parametrize(
    "changes",
    [
        dict(rows=0),
        dict(rows=9),
        dict(rows=32),
        dict(rows=3, variant=1),
        dict(rows=8, variant=1),
        dict(rows=4, variant=2),
        dict(rows=8, variant=2),
        dict(variant=3),
        dict(variant=4),
        dict(variant=-1),
        dict(hidden_size=2048),
        dict(intermediate_size=640),
        dict(experts=256),
        dict(topk=8),
        dict(scalar_type=D.BF16),
    ],
)
def test_unsupported_declares_no_offer(changes):
    s = signature(**changes)
    assert not experts.supported(s) and providers.make_offer(s) is None


@pytest.mark.parametrize("rows,variant", CASES)
@pytest.mark.parametrize("final", [False, True])
def test_exact_raw_launch_and_capture_reuse(monkeypatch, rows, variant, final):
    calls, state, _, _ = gpu_doubles(monkeypatch)
    io = view(rows, final)
    run = experts.build(signature(rows, variant))
    # The signature, not process env, chooses a prepared launch variant.
    monkeypatch.setenv("SGLANG_Q38_MK_MOE_SF_PACK", "0" if variant == 1 else "1")
    assert run(io) is None
    loader = "sf" if variant == 1 else "base" if rows <= 7 else "wide"
    assert calls[0] == (loader, (2,) if rows <= 7 else (2, rows == 8))
    assert calls[1][1] == (io.packed_w13, io.packed_w2, io.packed_x)
    args = next(args for name, args in calls if name in ("narrow-run", "wide-run"))
    assert args[0] == loader + "-kernel"
    if rows <= 7:
        assert args[2:6] == (
            io.routing_logits,
            io.route_weights_out,
            io.route_ids_out,
            io.x_scale,
        )
        layer = args[6]
        assert args[7] is io.expert_rows_out
    else:
        layer = args[2]
        assert layer["hidden_states"] is io.packed_x
        assert layer["hidden_states_scale"] is io.x_scale
        assert layer["routing_logits"] is io.routing_logits
        assert args[3:] == (io.expert_rows_out, io.route_weights_out, io.route_ids_out)
    for key, value in [
        ("gemm1_weights_scale", io.scale13),
        ("gemm2_weights_scale", io.scale2),
        ("output1_scale_gate_scalar", io.global_scale1),
        ("output1_scale_scalar", io.global_scale1_c),
        ("output2_scale_scalar", io.global_scale2),
    ]:
        assert layer[key] is value
    final_calls = [a for name, a in calls if name == "finalize"]
    assert len(final_calls) == int(final)
    if final:
        assert final_calls[0][2:] == (
            io.expert_rows_out,
            io.route_weights_out,
            io.final_out,
        )
    key = run._resource_key(io)
    assert run._resources[key][4] == (io.packed_w13, io.packed_w2)
    assert run._resources[key][3].keywords == dict(
        rows=rows,
        row_bytes=1280,
        box_bytes=128,
        box_rows=8 if rows <= 7 else 16,
        swizzle=128,
    )
    calls.clear()
    state.capturing = True
    assert run(io) is None
    assert all(
        name in ("host-x-map", "narrow-run", "wide-run", "finalize")
        for name, args in calls
    )


@pytest.mark.parametrize("rows,variant", [(4, 0), (4, 1), (12, 2)])
def test_capture_miss_fails_before_any_resource_or_output(monkeypatch, rows, variant):
    calls, state, _, _ = gpu_doubles(monkeypatch)
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        experts.build(signature(rows, variant))(view(rows))
    assert calls == []


@pytest.mark.parametrize(
    "changes",
    [
        dict(local_expert_start=1),
        dict(local_expert_count=256),
        dict(routed_scale=2.0),
        dict(activation=999),
        dict(hidden_size=2048),
        dict(packed_x=Tensor((8, 1280), 999)),
    ],
)
def test_invalid_view_rejected_before_any_write(monkeypatch, changes):
    calls, _, _, _ = gpu_doubles(monkeypatch)
    with pytest.raises(ValueError, match="offered layout"):
        experts.build(signature())(replace(view(), **changes))
    assert calls == []


def test_registry_uses_new_adapters_and_stock_for_absent_signatures(monkeypatch):
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    providers.make_offer.cache_clear()
    try:
        signatures = tuple(signature(m, v) for m, v in CASES)
        signatures += (signature(operation=O.FINALIZE_SHARED, scalar_type=D.BF16),)
        offers = providers.register_signatures(signatures)
        assert len(offers) == 15 and offers == providers.register_signatures(signatures)
        for s, offer in zip(signatures, offers):
            assert registry.get(s) is offer
            assert (
                type(offer.run) is experts.LocalExperts
                if s.operation == O.LOCAL_EXPERTS
                else offer.run is experts.finalize_shared
            )
        assert registry.get(signature(8, 2)) is None
    finally:
        providers.make_offer.cache_clear()


@pytest.mark.parametrize("cols", [256, 512, 1024, 4096])
@pytest.mark.parametrize("pdl", [False, True])
def test_finalize_shared_keeps_caller_output_and_launch_options(monkeypatch, cols, pdl):
    calls = []
    install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_finalize_shared",
        finalize_shared=lambda *a, **kw: calls.append((a, kw)),
    )
    values = [object() for _ in range(6)]
    io = FinalizeSharedIO(*values[:5], cols, pdl, values[5])
    assert (
        experts.build(signature(operation=O.FINALIZE_SHARED, scalar_type=D.BF16))(io)
        is None
    )
    assert calls == [(tuple(values), dict(cols=cols, use_pdl=pdl))]


@pytest.mark.parametrize("wide", [False, True])
def test_original_raw_wrapper_launch_abi(wide):
    # Execute the actual relocated wrappers, with only the CUDA driver recorded.
    root = Path(experts.__file__).parents[1] / "kernels/ops/moe"
    narrow_tree = ast.parse((root / "q38_mk_q35.py").read_text())
    calls = []
    namespace = dict(
        ctypes=ctypes,
        MAX_M=7,
        GRID=140,
        CLUSTER=2,
        TOPK=10,
        jit=SimpleNamespace(launch=lambda *a, **kw: calls.append((a, kw))),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=77))
        ),
    )
    nodes = [
        n
        for n in narrow_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Params"
    ]
    tree = ast.parse(
        (root / ("q38_mkm_wide.py" if wide else "q38_mk_q35.py")).read_text()
    )
    nodes += [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "original-raw-wrapper", "exec"
        ),
        namespace,
    )
    io = view(12 if wide else 4)
    layer = dict(
        gemm1_weights_scale=io.scale13,
        gemm2_weights_scale=io.scale2,
        output1_scale_gate_scalar=io.global_scale1,
        output1_scale_scalar=io.global_scale1_c,
        output2_scale_scalar=io.global_scale2,
    )
    io.routing_logits.shape = (12 if wide else 4, 512)
    if wide:
        layer.update(
            hidden_states=io.packed_x,
            hidden_states_scale=io.x_scale,
            routing_logits=io.routing_logits,
        )
        namespace["run"](
            "kernel",
            ("w13", "w2", "x"),
            layer,
            io.expert_rows_out,
            io.route_weights_out,
            io.route_ids_out,
        )
    else:
        namespace["run"](
            "kernel",
            ("w13", "w2", "x"),
            io.routing_logits,
            io.route_weights_out,
            io.route_ids_out,
            io.x_scale,
            layer,
            io.expert_rows_out,
        )
    args, kwargs = calls[0]
    assert args[0] == "kernel" and args[3] == 77
    assert args[1] == ((120, 1, 1) if wide else (140, 1, 1))
    assert kwargs == ({} if wide else dict(cluster=(2, 1, 1)))
    params = args[2][-1]
    assert params.num_tokens == (12 if wide else 4)
    assert (
        params.logits == io.routing_logits.data_ptr()
        and params.out == io.expert_rows_out.data_ptr()
    )
    assert (
        params.expert_weights == io.route_weights_out.data_ptr()
        and params.topk_ids == io.route_ids_out.data_ptr()
    )
