# SPDX-License-Identifier: Apache-2.0
"""Compare caller-buffer scaffold with original allocating wrapper ABI on CPU."""

import ast
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnActivation,
    GdnFrontIO,
    GdnLaunchConfig,
    GdnSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import gdn_front


def raw_double(monkeypatch, family, stages, *, wide=False):
    name = "q38_mkgdn_front" if family == "mk" else "gdn_inproj_conv"
    path = Path(gdn_front.__file__).parents[1] / "gdn" / (name + ".py")
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    state = SimpleNamespace(capturing=False, calls=[])
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *a: SimpleNamespace(cuda_stream=77)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 7))

    class Op:
        def __init__(self, **kw):
            self.options = kw

    def as_view(t, align):
        base = t if t._base is None else t._base
        return (
            id(base),
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            t.storage_offset(),
            align,
        )

    def compile_op(op, *args):
        assert not state.capturing
        state.calls.append(("compile", vars(op).copy(), args))

        def launch(*runtime):
            state.calls.append(("launch", runtime))

        return launch

    module = ModuleType("sgl_nvidia_mega.qwen3_8.gdn." + name)
    module.torch = torch
    module.os = SimpleNamespace(
        environ={
            "Q38_GDNPROJ_FRONT_STAGES": str(stages),
            "SGLANG_Q38_MK_GDN_WIDE": "1" if wide else "0",
        }
    )
    module._FRONT_STAGES = stages
    module._CACHE = {}
    module.cuda = SimpleNamespace(CUstream=lambda x: x)
    module.cutlass = SimpleNamespace(BFloat16="BF16")
    module.cute_ext = SimpleNamespace(compile=compile_op)
    module.MkGdnFront = module.TgvGemmCuteExtKernel = Op
    module._view = as_view
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module, state


def inputs(rows, hk, hv, bias):
    channels = (2 * hk + hv) * 128
    packed = channels + hv * 128 + 2 * hv
    bf = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
    return dict(
        x=bf((rows, 2560)),
        packed_weight=bf((packed, 2560)),
        conv_weight=bf((channels, 4)),
        conv_bias=bf((channels,)) if bias else None,
        conv_state=bf((4, channels, 3)),
        state_indices=torch.empty((rows // 4,), dtype=torch.int32, device="meta"),
        intermediate_indices=torch.empty(
            (rows // 4,), dtype=torch.int64, device="meta"
        ),
        window=bf((4, 4, channels, 3)),
        pad_slot=-1,
    )


def caller_view():
    args = inputs(4, 4, 12, False)
    bf = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
    return GdnFrontIO(
        **{k: v for k, v in args.items() if k != "window"},
        intermediate_conv_window_out=args["window"],
        packed_out=bf((4, 4120)),
        convolved_out=bf((4, 2560)),
        activation=GdnActivation.SILU,
        token_major_out=True,
        window_mode=0,
        launch=GdnLaunchConfig(0, 0, 6, False, 0, False, 0, True, False),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 1),
        ("rows", 32),
        ("key_heads", 8),
        ("value_heads", 24),
        ("key_dim", 64),
        ("value_dim", 64),
        ("tokens_per_request", 1),
        ("scalar_type", D.FP32),
        ("gate_scalar_type", D.BF16),
        ("variant", 4),
    ],
)
def test_unsupported_signature_has_no_offer(field, value):
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    bad = replace(sig, **{field: value})
    assert not providers.supported(bad)
    assert providers.make_offer(bad) is None
    with pytest.raises(ValueError, match="unsupported"):
        gdn_front.build(bad)


@pytest.mark.parametrize(
    "variant,rows,hk,hv",
    [
        (1, 8, 4, 12),
        (1, 4, 16, 48),
        (2, 8, 16, 48),
        (2, 4, 4, 12),
        (3, 4, 4, 12),
    ],
)
def test_stages_cannot_widen_source_family(variant, rows, hk, hv):
    sig = GdnSignature(O.GDN_FRONT, rows, hk, hv, 128, 128, 4, D.BF16, variant=variant)
    assert not providers.supported(sig)
    assert providers.make_offer(sig) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_n", 64),
        ("num_warps", 8),
        ("num_stages", 12),
        ("hoist", True),
        ("dot_precision", 1),
        ("prefetch", True),
        ("historical_branch", 1),
        ("use_pdl", False),
        ("trigger_early", True),
    ],
)
def test_non_neutral_launch_is_rejected_before_compile(monkeypatch, field, value):
    _, state = raw_double(monkeypatch, "mk", 6)
    io = caller_view()
    io = replace(io, launch=replace(io.launch, **{field: value}))
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    with pytest.raises(ValueError, match="prepared source layout"):
        gdn_front.build(sig)(io)
    assert state.calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"packed_out": None},
        {"packed_weight": None},
        {"activation": 0},
        {"token_major_out": False},
        {"window_mode": 1},
    ],
)
def test_missing_or_changed_source_metadata_has_no_launch(monkeypatch, change):
    _, state = raw_double(monkeypatch, "mk", 6)
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    with pytest.raises(ValueError, match="prepared source layout"):
        gdn_front.build(sig)(replace(caller_view(), **change))
    assert state.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "x",
        "packed_weight",
        "conv_weight",
        "conv_state",
        "packed_out",
        "convolved_out",
        "intermediate_conv_window_out",
    ],
)
def test_operand_dtype_rejected_without_cast(monkeypatch, field):
    _, state = raw_double(monkeypatch, "mk", 6)
    io = caller_view()
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    with pytest.raises(ValueError, match="prepared source layout"):
        gdn_front.build(sig)(replace(io, **{field: getattr(io, field).float()}))
    assert state.calls == []


def test_adapter_has_no_allocation_cast_copy_or_sync_calls():
    tree = ast.parse(Path(gdn_front.__file__).read_text())
    prohibited = {
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "clone",
        "to",
        "contiguous",
        "copy_",
        "synchronize",
        "sort",
    }
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in prohibited
    ]


def test_canonical_registry_facade_and_source_preparation(monkeypatch):
    from sglang.kernels.plugin_kernels.nvidia.numeric import NumericRegistry
    from sglang.srt.environ import envs
    from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    _, state = raw_double(monkeypatch, "mk", 6)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    providers.make_offer.cache_clear()
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    try:
        built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=(sig,)))
        assert len(built.offers) == 1 and registry.get(sig) is built.offers[0]
        assert numeric_binding.numeric_operator(sig, prepared=False) is None
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False):
            assert numeric_binding.numeric_operator(sig, prepared=True) is None
        assert state.calls == []
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
            run = numeric_binding.numeric_operator(sig, prepared=True)
            assert run is not None and run(caller_view()) is None
            monkeypatch.setattr(binding, "megakernel_enabled", lambda: False)
            assert numeric_binding.numeric_operator(sig, prepared=True) is None
        assert [c[0] for c in state.calls] == ["compile", "launch"]
    finally:
        providers.make_offer.cache_clear()


def test_front_cache_keeps_layout_not_activation_owner(monkeypatch):
    _, state = raw_double(monkeypatch, "mk", 6)
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    run = gdn_front.build(sig)
    io = caller_view()
    owner = weakref.ref(io.x)
    run(io)
    del io
    assert owner() is None
    state.capturing = True
    state.calls.clear()
    run(caller_view())
    assert [c[0] for c in state.calls] == ["launch"]


def test_initial_cold_capture_has_no_compile_or_state_writing_launch(monkeypatch):
    _, state = raw_double(monkeypatch, "mk", 6)
    sig = GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16)
    run = gdn_front.build(sig)
    state.capturing = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(caller_view())
    assert state.calls == [] and run._resources == {}


CASES = [
    ("mk", m, hk, hv, 6, bias)
    for m in (4, 8, 12)
    for hk, hv in ((4, 12), (16, 48))
    for bias in (False, True)
]
CASES += [("mk", 16, 4, 12, 6, bias) for bias in (False, True)]
CASES += [("mk", 4, 4, 12, 12, bias) for bias in (False, True)]
CASES += [
    ("legacy", 4, 16, 48, stages, bias) for stages in (8, 12) for bias in (False, True)
]


@pytest.mark.parametrize("family,rows,hk,hv,stages,bias", CASES)
def test_original_factory_parameters_layouts_and_caller_outputs(
    monkeypatch, family, rows, hk, hv, stages, bias
):
    module, state = raw_double(monkeypatch, family, stages, wide=rows == 16)
    args = inputs(rows, hk, hv, bias)
    packed, conv = module.run(**args)
    original = list(state.calls)
    assert [c[0] for c in original] == ["compile", "launch"]
    io = GdnFrontIO(
        **{k: v for k, v in args.items() if k != "window"},
        intermediate_conv_window_out=args["window"],
        packed_out=packed,
        convolved_out=conv,
        activation=GdnActivation.SILU,
        token_major_out=True,
        window_mode=0,
        launch=GdnLaunchConfig(0, 0, stages, False, 0, False, 0, True, False),
    )
    variant = (0 if stages == 6 else 1) if family == "mk" else (2 if stages == 8 else 3)
    sig = GdnSignature(O.GDN_FRONT, rows, hk, hv, 128, 128, 4, D.BF16, variant=variant)
    run = gdn_front.build(sig)
    state.calls.clear()
    assert run(io) is None
    assert state.calls == original
    # Capture uses a different activation Tensor object with the same runtime ABI.
    io = replace(io, x=torch.empty_like(io.x))
    state.calls.clear()
    state.capturing = True
    assert run(io) is None
    assert [c[0] for c in state.calls] == ["launch"]
    runtime = state.calls[0][1]
    assert runtime[1][0] == id(io.x)
    assert runtime[2][0] == id(packed) and runtime[-2][0] == id(conv)
    assert len(run._resources) == 1
    # A pad-slot/layout specialization not eagerly prepared cannot compile in capture.
    io = replace(io, pad_slot=0)
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(io)
    assert [c[0] for c in state.calls] == ["launch"]
