# SPDX-License-Identifier: Apache-2.0
"""Real canonical/provider and original verify ABI; CUDA/JIT are explicit spies."""

import ast
import gc
import sys
import weakref
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NativeGdnPdlIO,
    NativeGdnPdlSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers, slots
from sgl_nvidia_mega.qwen3_8._providers import gdn_native_pdl as native
from sgl_nvidia_mega.qwen3_8.spec import SPEC


def signature(**kwargs):
    return replace(
        NativeGdnPdlSignature(O.GDN_NATIVE_PDL, 32, 16, 48, 128, 128, 4, D.BF16),
        **kwargs,
    )


class CudaTensor(torch.Tensor):
    """Meta storage for large state; only CUDA device/pointer metadata doubled."""

    @property
    def device(self):
        return getattr(self, "test_device", torch.device("cuda:2"))

    def data_ptr(self):
        return getattr(self, "test_pointer", id(self) * 16)


def view(s, *, strided=True, indices_dtype=torch.int64):
    def tensor(shape, dtype=torch.bfloat16, pad=0, vector_stride=1):
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        strides.reverse()
        if len(shape) > 1:
            strides[0] += pad
        else:
            strides[0] = vector_stride
        return torch.empty_strided(
            shape, strides, dtype=dtype, device="meta"
        ).as_subclass(CudaTensor)

    m, h, hv, bsz = s.rows, s.key_heads, s.value_heads, s.rows // 4
    # Persistent and replay pools deliberately differ in capacity.
    checkpoint_slots, replay_slots = bsz + 3, bsz + 1
    state = torch.float32 if s.state_scalar_type == D.FP32 else torch.bfloat16
    dt = torch.float32 if s.dt_bias_scalar_type == D.FP32 else torch.bfloat16
    pad = 128 if strided else 0
    d = tensor((replay_slots, hv, 16, 128))
    k = tensor((replay_slots, h, 16, 128))
    g = tensor((replay_slots, hv, 16), torch.float32)
    return NativeGdnPdlIO(
        tensor((m, h, 128), pad=pad),
        tensor((m, h, 128), pad=pad * 2),
        tensor((m, hv, 128), pad=pad * 3),
        tensor((m, hv), pad=24 if strided else 0),
        tensor((m, hv), pad=48 if strided else 0),
        tensor((hv,), torch.float32),
        tensor((hv,), dt),
        tensor((m, hv, 128), pad=pad),
        tensor((checkpoint_slots, hv, 128, 128), state),
        d,
        k,
        g,
        tensor(d.shape) if s.storage_mode else None,
        tensor(k.shape) if s.storage_mode else None,
        tensor(g.shape, torch.float32) if s.storage_mode == 2 else None,
        tensor((bsz + 1,), indices_dtype, vector_stride=2 if strided else 1),
        tensor((bsz,), indices_dtype, vector_stride=3 if strided else 1),
        tensor((bsz,), indices_dtype, vector_stride=4 if strided else 1),
        tensor((replay_slots,), torch.int32),
        tensor((replay_slots,), torch.int32),
        tensor((replay_slots,), torch.int8),
    )


@pytest.fixture
def raw(monkeypatch):
    path = (
        Path(native.__file__).parents[1]
        / "kernels/ops/attention/q38_gdn_front/replayssm_pdl.py"
    )
    wrapper = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "verify"
    )
    state = SimpleNamespace(capturing=False, calls=[], fail=False)

    def freeze(x):
        if isinstance(x, torch.Tensor):
            return id(x), tuple(x.shape), tuple(x.stride()), x.dtype, x.data_ptr()
        return x

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                state.calls.append((grid, tuple(map(freeze, args)), kw))
                if state.fail:
                    raise RuntimeError("selected native verify failed")

            return launch

    name = "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_gdn_front.replayssm_pdl"
    module = ModuleType(name)
    module.torch = torch
    module.triton = SimpleNamespace(
        cdiv=lambda a, b: (a + b - 1) // b,
        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
    )
    module.os = SimpleNamespace(environ={})
    module.gdn_replayssm_spec_circular_kernel = Kernel()
    exec(
        compile(ast.Module(body=[wrapper], type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, name, module)
    utils = ModuleType("sglang.kernels.ops.attention.fla.utils")
    utils.is_tf32_supported = True
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS)
    providers.make_offer.cache_clear()
    yield module, state, utils
    providers.make_offer.cache_clear()


def original(raw, s, io):
    module, _, utils = raw
    module.os.environ = {"Q38_GDN_STATE_PREFETCH": str(s.state_prefetch)}
    utils.is_tf32_supported = s.dot_precision == 0
    return module.verify(
        *(getattr(io, f.name) for f in fields(io)),
        max_cache_len=16,
        max_spec_len=4,
        scale=128**-0.5,
        null_block_id=-1,
        block_v=64,
        num_warps=1,
        num_stages=getattr(s, "num_stages", 2),
        nk=2,
        bs_min=4,
        pdl=True,
        wait_for_gate_producer=bool(s.wait_for_gate_producer),
    )


CASES = list(
    product(
        ((4, 12), (16, 48)),
        (D.BF16, D.FP32),
        (D.BF16, D.FP32),
        (0, 1, 2),
        (0, 1),
        (0, 1),
        (0, 2),
    )
)


def staged(s, n):
    """The signature at n verify stages; SGLang without the field gets a test subclass."""
    if "num_stages" in {f.name for f in fields(NativeGdnPdlSignature)}:
        return replace(s, num_stages=n)

    @dataclass(frozen=True)
    class Staged(NativeGdnPdlSignature):
        num_stages: int = 2

    return Staged(**{f.name: getattr(s, f.name) for f in fields(s)}, num_stages=n)


@pytest.mark.parametrize("stages", (2, 3))
def test_verify_stage_count_reaches_the_launch(raw, stages):
    s = staged(signature(rows=32), stages)
    assert native.supported(s)
    io = view(s)
    assert original(raw, s, io) is io.out
    expected = list(raw[1].calls)
    raw[1].calls.clear()
    assert native.build(s)(io) is None
    assert raw[1].calls == expected


def test_unknown_verify_stage_count_has_no_offer():
    assert not native.supported(staged(signature(), 4))


@pytest.mark.parametrize("case", CASES)
def test_original_all_source_arguments_and_caller_identity(raw, case):
    heads, state_dtype, dt_dtype, storage, wait, prefetch, dot = case
    index = CASES.index(case)
    s = signature(
        rows=(4, 8, 12, 16, 32, 64)[index % 6],
        key_heads=heads[0],
        value_heads=heads[1],
        state_scalar_type=state_dtype,
        dt_bias_scalar_type=dt_dtype,
        storage_mode=storage,
        wait_for_gate_producer=wait,
        state_prefetch=prefetch,
        dot_precision=dot,
    )
    io = view(
        s,
        strided=index % 2 == 0,
        indices_dtype=torch.int32 if index % 2 else torch.int64,
    )
    assert original(raw, s, io) is io.out
    expected = list(raw[1].calls)
    raw[1].calls.clear()
    run = native.build(s)
    assert run(io) is None
    assert raw[1].calls == expected
    assert len(run._warmed) == 1


@pytest.mark.parametrize(
    "field,value",
    (
        ("operation", O.GDN_CORE),
        ("rows", 0),
        ("rows", 31),
        ("key_heads", 8),
        ("value_heads", 24),
        ("key_dim", 64),
        ("value_dim", 64),
        ("tokens_per_request", 1),
        ("scalar_type", D.FP32),
        ("gate_scalar_type", D.FP32),
        ("state_scalar_type", D.FP16),
        ("dt_bias_scalar_type", D.FP16),
        ("cache_len", 32),
        ("storage_mode", 3),
        ("wait_for_gate_producer", 2),
        ("state_prefetch", 2),
        ("dot_precision", 1),
    ),
)
def test_unsupported_source_metadata_no_offer(field, value):
    bad = replace(signature(), **{field: value})
    assert not providers.supported(bad) and providers.make_offer(bad) is None


@pytest.mark.parametrize("field", [f.name for f in fields(NativeGdnPdlSignature)])
@pytest.mark.parametrize("bad_type", (bool, float))
def test_signature_cache_cannot_admit_bool_or_float(raw, field, bad_type):
    s = signature()
    providers.make_offer(s)
    bad = replace(s, **{field: bad_type(getattr(s, field))})
    assert providers.make_offer(bad) is None
    assert not native.supported(bad)


@pytest.mark.parametrize("field", [f.name for f in fields(NativeGdnPdlIO)])
def test_wrong_operand_dtype_rejected_before_launch(raw, field):
    s = signature(storage_mode=2)
    io = view(s)
    t = getattr(io, field)
    bad = t.to(torch.int16 if t.dtype != torch.int16 else torch.float32)
    with pytest.raises(ValueError):
        native.build(s)(replace(io, **{field: bad}))
    assert not raw[1].calls


@pytest.mark.parametrize("field", [f.name for f in fields(NativeGdnPdlIO)])
def test_wrong_shape_rejected_before_launch(raw, field):
    s = signature(storage_mode=2)
    io = view(s)
    with pytest.raises(ValueError):
        native.build(s)(replace(io, **{field: getattr(io, field).unsqueeze(0)}))
    assert not raw[1].calls


@pytest.mark.parametrize(
    "storage,presence", tuple(product((0, 1, 2), product((False, True), repeat=3)))
)
def test_optional_ring_combination_is_explicit(raw, storage, presence):
    s = signature(storage_mode=storage)
    complete = view(replace(s, storage_mode=2))
    names = ("rawv_cache", "rawk_cache", "beta_cache")
    io = replace(
        complete,
        **{
            name: getattr(complete, name) if present else None
            for name, present in zip(names, presence)
        },
    )
    if (
        presence
        == ((False, False, False), (True, True, False), (True, True, True))[storage]
    ):
        native.build(s)(io)
        assert len(raw[1].calls) == 1
    else:
        with pytest.raises(ValueError, match="storage mode"):
            native.build(s)(io)
        assert not raw[1].calls


def test_cold_capture_then_three_addresses_no_allocation_compile_or_owner_retention(
    raw, monkeypatch
):
    s = signature()
    ios = [view(s) for _ in range(3)]
    assert len({io.q.data_ptr() for io in ios}) == 3
    run = native.build(s)
    state = raw[1]
    state.capturing = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(ios[0])
    assert not state.calls
    state.capturing = False
    run(ios[0])
    run(ios[1])

    def forbidden(*a, **kw):
        raise AssertionError("allocation/cast/compile/sync in prepared invocation")

    for name in ("empty", "empty_like", "zeros", "full", "tensor", "empty_strided"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    state.capturing = True
    run(ios[2])
    assert len(state.calls) == 3 and len(run._warmed) == 1
    assert [call[1][0][-1] for call in state.calls] == [io.q.data_ptr() for io in ios]
    refs = [
        weakref.ref(getattr(io, f.name))
        for io in ios
        for f in fields(io)
        if getattr(io, f.name) is not None
    ]
    del ios
    gc.collect()
    assert all(ref() is None for ref in refs)
    tree = ast.parse(Path(native.__file__).read_text())
    forbidden_names = {
        "empty",
        "empty_like",
        "new_empty",
        "to",
        "contiguous",
        "copy_",
        "synchronize",
        "compile",
        "compile_source",
    }
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in forbidden_names
    ]


@pytest.mark.parametrize(
    "fault", ("stride", "alignment", "metadata_dtype", "pool_capacity")
)
def test_cold_specialization_refuses_before_raw(raw, fault):
    s = signature()
    io = view(s)
    run = native.build(s)
    run(io)
    other = (
        view(s, strided=False, indices_dtype=torch.int32)
        if fault == "metadata_dtype"
        else view(s)
    )
    if fault == "alignment":
        other.q.test_pointer = other.q.data_ptr() + 2
    elif fault == "stride":
        other = replace(
            other,
            q=other.q.as_strided(other.q.shape, (other.q.stride(0) + 128, 128, 1)),
        )
    elif fault == "pool_capacity":
        other = replace(other, checkpoint_state=other.checkpoint_state[:1])
    raw[1].capturing = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(other)
    assert len(raw[1].calls) == 1


def test_failure_never_publishes_warm_key_or_retries(raw):
    run = native.build(signature())
    raw[1].fail = True
    with pytest.raises(RuntimeError, match="selected native verify failed"):
        run(view(signature()))
    assert len(raw[1].calls) == 1 and not run._warmed


def test_spec_registers_exact_finite_offer_without_warming_or_framework_access(
    raw, monkeypatch
):
    from sglang.kernels.plugin_kernels.nvidia.numeric import NumericRegistry

    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    s = signature()
    built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,)))
    assert len(built.offers) == 1 and registry.get(s) is built.offers[0]
    assert slots.OPERATION_SLOTS[s.operation] == "gdn_native_pdl"
    assert not built.offers[0].run._warmed and not raw[1].calls
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS - {"gdn_native_pdl"})
    providers.make_offer.cache_clear()
    assert not providers.supported(s) and providers.make_offer(s) is None
