# SPDX-License-Identifier: Apache-2.0
"""F-only canonical DTO / caller-output launch CPU spies; no GPU evidence."""

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import get_args

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    FinalizePermutedSharedIO,
    FinalizePermutedSignature,
    NumericIO,
    NumericOffer,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
    OperatorSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers.finalize_permuted import build


class CudaTensor(torch.Tensor):
    """Real CPU storage/dtype/strides; only device metadata is a CUDA double."""

    @property
    def device(self):
        return getattr(self, "test_device", torch.device("cuda:0"))


def fake_cuda(tensor):
    return tensor.as_subclass(CudaTensor)


class Launch:
    def __init__(self):
        self.calls = []
        self.error = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self.error:
                raise self.error
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.fixture
def setup(monkeypatch):
    state = SimpleNamespace(capturing=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    raw = ModuleType("sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.elementwise")
    raw._fused_finalize_gate_sigmoid_mul_add_kernel = launch = Launch()
    monkeypatch.setitem(sys.modules, raw.__name__, raw)
    monkeypatch.setattr(providers, "numeric_registry", NumericRegistry())
    providers.make_offer.cache_clear()
    yield SimpleNamespace(state=state, launch=launch)
    providers.make_offer.cache_clear()


def signature(rows=4, weight_dtype=torch.bfloat16, warps=16, pdl=True):
    return FinalizePermutedSignature(
        O.FINALIZE_PERMUTED_SHARED,
        rows,
        2560,
        10,
        D.BF16,
        D.BF16 if weight_dtype == torch.bfloat16 else D.FP32,
        num_warps=warps,
        use_pdl=int(pdl),
    )


def make_io(rows=4, dtype=torch.bfloat16, padding=0):
    def empty(shape, dtype=torch.bfloat16):
        return fake_cuda(torch.empty(shape, dtype=dtype))

    return FinalizePermutedSharedIO(
        empty((rows, 2560)),
        empty((2560,)),
        empty((rows, 2560)),
        empty((rows * 10, 2560 + padding))[:, :2560],
        empty((rows * 10,), torch.int32),
        empty((rows, 10), dtype),
        empty((rows, 2560)),
    )


def test_canonical_unions_and_seven_field_identity():
    assert FinalizePermutedSharedIO in get_args(NumericIO)
    assert FinalizePermutedSignature in get_args(OperatorSignature)
    assert O.FINALIZE_PERMUTED_SHARED.value == 42
    assert tuple(f.name for f in fields(FinalizePermutedSharedIO)) == (
        "hidden",
        "gate_weight",
        "shared",
        "gemm2_out",
        "expanded_idx_to_permuted_idx",
        "expert_weights",
        "output",
    )


@pytest.mark.parametrize("rows", range(1, 65))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_finite_both_weight_capabilities(setup, rows, dtype):
    s = signature(rows, dtype)
    assert providers.supported(s)
    offers = providers.register_signatures((s,))
    assert offers == providers.register_signatures((s,))
    assert len(offers) == 1 and isinstance(offers[0], NumericOffer)
    assert providers.numeric_registry.get(s) is offers[0]
    assert not setup.launch.calls


@pytest.mark.parametrize(
    "changes",
    (
        {"operation": O.FINALIZE_SHARED},
        {"rows": 0},
        {"rows": 65},
        {"hidden_size": 1280},
        {"topk": 8},
        {"scalar_type": D.FP32},
        {"weight_scalar_type": D.FP16},
        {"num_warps": 4},
        {"num_warps": 32},
        {"use_pdl": 3},
        {"variant": 1},
    ),
)
def test_unsupported_never_binds_or_offers(setup, changes):
    s = replace(signature(), **changes)
    assert not providers.supported(s)
    assert providers.make_offer(s) is None
    with pytest.raises(ValueError, match="unsupported"):
        build(s)
    assert not setup.launch.calls


@pytest.mark.parametrize("rows", (1, 4, 8, 16, 32, 64))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("padding", (0, 64))
@pytest.mark.parametrize("warps", (8, 16))
@pytest.mark.parametrize("pdl", (False, True))
def test_original_raw_launch_caller_identity(setup, rows, dtype, padding, warps, pdl):
    s = signature(rows, dtype, warps, pdl)
    io = make_io(rows, dtype, padding)
    offer = providers.register_signatures((s,))[0]
    assert offer.run(io) is None
    setup.state.capturing = True
    assert offer.run(io) is None
    assert len(setup.launch.calls) == 2
    for grid, args, kwargs in setup.launch.calls:
        assert grid == (rows,)
        expected = (
            io.hidden,
            io.gate_weight,
            io.shared,
            io.output,
            io.gemm2_out,
            io.expanded_idx_to_permuted_idx,
            io.expert_weights,
        )
        assert all(a is b for a, b in zip(args[:7], expected))
        assert args[7] == 2560 + padding
        assert kwargs == dict(
            hidden_dim=2560,
            TOP_K=10,
            BLOCK_SIZE=4096,
            USE_PDL=pdl,
            num_warps=warps,
            **({"launch_pdl": True} if pdl else {}),
        )


@pytest.mark.parametrize(
    "field", tuple(f.name for f in fields(FinalizePermutedSharedIO))
)
def test_all_seven_tensor_guards_before_metadata(setup, field):
    class Poison:
        def __getattribute__(self, name):
            if name == "__class__":
                return object.__getattribute__(self, name)
            raise AssertionError("must not inspect nonTensor metadata: " + name)

    io = replace(make_io(), **{field: Poison()})
    with pytest.raises(ValueError, match="seven tensors"):
        build(signature())(io)
    assert not setup.launch.calls


@pytest.mark.parametrize(
    "field", ("hidden", "gate_weight", "shared", "gemm2_out", "output")
)
def test_main_dtype_rejected_without_cast(setup, field):
    io = make_io()
    bad = fake_cuda(torch.empty(tuple(getattr(io, field).shape), dtype=torch.float32))
    with pytest.raises(ValueError, match="layout"):
        build(signature())(replace(io, **{field: bad}))
    assert not setup.launch.calls


@pytest.mark.parametrize(
    "field", tuple(f.name for f in fields(FinalizePermutedSharedIO))
)
def test_mixed_devices_rejected(setup, field):
    io = make_io()
    getattr(io, field).test_device = torch.device("cuda:1")
    with pytest.raises(ValueError, match="layout"):
        build(signature())(io)
    assert not setup.launch.calls


@pytest.mark.parametrize(
    "mutation",
    (
        "cpu",
        "hidden_shape",
        "gate_shape",
        "shared_shape",
        "output_shape",
        "weight_shape",
        "weight_dtype",
        "perm_dtype",
        "perm_count",
        "perm_stride",
        "gemm2_rank",
        "gemm2_width",
        "gemm2_innerstride",
        "gemm2_zerostride",
        "hidden_stride",
        "output_stride",
        "shared_stride",
        "weight_stride",
        "gate_stride",
    ),
)
def test_layout_fail_before_raw(setup, mutation):
    io = make_io()
    changes = {
        "cpu": ("hidden", torch.empty((4, 2560), dtype=torch.bfloat16)),
        "hidden_shape": (
            "hidden",
            fake_cuda(torch.empty((3, 2560), dtype=torch.bfloat16)),
        ),
        "gate_shape": (
            "gate_weight",
            fake_cuda(torch.empty((1, 2560), dtype=torch.bfloat16)),
        ),
        "shared_shape": (
            "shared",
            fake_cuda(torch.empty((3, 2560), dtype=torch.bfloat16)),
        ),
        "output_shape": (
            "output",
            fake_cuda(torch.empty((3, 2560), dtype=torch.bfloat16)),
        ),
        "weight_shape": (
            "expert_weights",
            fake_cuda(torch.empty((4, 9), dtype=torch.bfloat16)),
        ),
        "weight_dtype": (
            "expert_weights",
            fake_cuda(torch.empty((4, 10), dtype=torch.float32)),
        ),
        "perm_dtype": (
            "expanded_idx_to_permuted_idx",
            fake_cuda(torch.empty((40,), dtype=torch.int64)),
        ),
        "perm_count": (
            "expanded_idx_to_permuted_idx",
            fake_cuda(torch.empty((39,), dtype=torch.int32)),
        ),
        "perm_stride": (
            "expanded_idx_to_permuted_idx",
            fake_cuda(torch.empty((80,), dtype=torch.int32))[::2],
        ),
        "gemm2_rank": (
            "gemm2_out",
            fake_cuda(torch.empty((2560,), dtype=torch.bfloat16)),
        ),
        "gemm2_width": (
            "gemm2_out",
            fake_cuda(torch.empty((40, 2559), dtype=torch.bfloat16)),
        ),
        "gemm2_innerstride": (
            "gemm2_out",
            fake_cuda(torch.empty((40, 5120), dtype=torch.bfloat16))[:, ::2],
        ),
        "gemm2_zerostride": (
            "gemm2_out",
            fake_cuda(torch.empty((1, 2560), dtype=torch.bfloat16)).expand(40, -1),
        ),
        "hidden_stride": (
            "hidden",
            fake_cuda(torch.empty((2560, 4), dtype=torch.bfloat16)).T,
        ),
        "output_stride": (
            "output",
            fake_cuda(torch.empty((2560, 4), dtype=torch.bfloat16)).T,
        ),
        "shared_stride": (
            "shared",
            fake_cuda(torch.empty((2560, 4), dtype=torch.bfloat16)).T,
        ),
        "weight_stride": (
            "expert_weights",
            fake_cuda(torch.empty((10, 4), dtype=torch.bfloat16)).T,
        ),
        "gate_stride": (
            "gate_weight",
            fake_cuda(torch.empty((5120,), dtype=torch.bfloat16))[::2],
        ),
    }
    field, bad = changes[mutation]
    with pytest.raises(ValueError, match="layout"):
        build(signature())(replace(io, **{field: bad}))
    assert not setup.launch.calls


def test_cold_bind_and_capture_refuse(setup):
    setup.state.capturing = True
    with pytest.raises(RuntimeError, match="bind before capture"):
        build(signature())
    setup.state.capturing = False
    run = build(signature())
    setup.state.capturing = True
    with pytest.raises(RuntimeError, match="warm eagerly"):
        run(make_io())
    assert not setup.launch.calls


def test_changed_raw_stride_needs_eager_warmup(setup):
    run = build(signature())
    run(make_io())
    setup.state.capturing = True
    with pytest.raises(RuntimeError, match="warm eagerly"):
        run(make_io(padding=64))
    assert len(setup.launch.calls) == 1


def test_raw_failure_propagates_and_does_not_warm(setup):
    run = build(signature())
    io = make_io()
    setup.launch.error = RuntimeError("raw failure")
    with pytest.raises(RuntimeError, match="raw failure"):
        run(io)
    setup.launch.error = None
    setup.state.capturing = True
    with pytest.raises(RuntimeError, match="warm eagerly"):
        run(io)
    assert not setup.launch.calls


def test_independent_fixture_layout_on_cpu():
    path = Path(__file__).parents[1] / "smoke/qwen3_8_finalize_permuted_int.py"
    spec = importlib.util.spec_from_file_location("f_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for rows in (1, 4, 8, 16, 32, 64):
        for dtype in (torch.bfloat16, torch.float32):
            io = module.make_io(rows, dtype, "cpu", row_padding=64)
            assert io.output.shape == io.hidden.shape == (rows, 2560)
            assert io.gemm2_out.stride() == (2624, 1)
            assert io.expert_weights.dtype == dtype
            assert io.expanded_idx_to_permuted_idx.dtype == torch.int32
            assert (io.expanded_idx_to_permuted_idx[::7] == -1).all()
            assert io.output.data_ptr() != io.shared.data_ptr()
            assert module.signature(rows, dtype) == signature(rows, dtype)


def test_pointer_alignment_change_needs_own_warmup(setup):
    io = make_io()
    run = build(signature())
    run(io)
    # Contiguous source view, same shape/dtype, but a different Triton pointer
    # alignment specialization. Retain the backing storage via the tensor view.
    hidden = fake_cuda(torch.empty((4 * 2560 + 1,), dtype=torch.bfloat16))[1:].view(
        4, 2560
    )
    assert hidden.data_ptr() % 16 != io.hidden.data_ptr() % 16
    setup.state.capturing = True
    with pytest.raises(RuntimeError, match="warm eagerly"):
        run(replace(io, hidden=hidden))
    assert len(setup.launch.calls) == 1


def test_wrong_io_type_never_launches(setup):
    with pytest.raises(TypeError, match="bound tensor view"):
        build(signature())(SimpleNamespace())
    assert not setup.launch.calls


def test_provider_contains_no_allocator_conversion_or_stream_owner():
    import ast

    from sgl_nvidia_mega.qwen3_8._providers import finalize_permuted

    tree = ast.parse(Path(finalize_permuted.__file__).read_text())
    forbidden = {
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "clone",
        "copy_",
        "to",
        "float",
        "bfloat16",
        "contiguous",
        "synchronize",
        "wait_stream",
        "record_stream",
        "current_stream",
    }
    # dtype constants (torch.bfloat16) are permitted; conversion calls are not.
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not calls & forbidden


@pytest.mark.parametrize("value", (False, True, -1, 2, 0.0, 1.0, "1", None))
def test_noninteger_or_out_of_domain_pdl_declares_no_offer(setup, value):
    s = replace(signature(), use_pdl=value)
    assert not providers.supported(s)
    assert providers.make_offer(s) is None
    with pytest.raises(ValueError, match="unsupported"):
        build(s)
    assert not setup.launch.calls


@pytest.mark.parametrize("value", (0, 1))
def test_integer_signature_decodes_only_at_raw_bool_boundary(setup, value):
    s = replace(signature(), use_pdl=value)
    assert type(s.use_pdl) is int
    run = build(s)
    run(make_io())
    kwargs = setup.launch.calls[-1][2]
    assert type(kwargs["USE_PDL"]) is bool
    assert kwargs["USE_PDL"] is bool(value)


@pytest.mark.parametrize("value", (0, 1))
def test_bool_must_not_alias_cached_integer_offer(setup, value):
    good = replace(signature(), use_pdl=value)
    offer = providers.make_offer(good)
    assert offer is not None
    bad = replace(good, use_pdl=bool(value))
    # Python's dataclass equality aliases bool/int. Validate before cache lookup.
    assert bad == good and not providers.supported(bad)
    assert providers.make_offer(bad) is None
    assert providers.make_offer(good) is offer
