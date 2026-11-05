# SPDX-License-Identifier: Apache-2.0
"""Canonical router-quant offers and source driver ABI, without CUDA execution."""

import gc
import hashlib
import json
import weakref
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import get_args

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
    OperatorSignature,
    RouterQuantIO,
    RouterQuantSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers, slots
from sgl_nvidia_mega.qwen3_8._providers import router_quant
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_front as raw
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_jit as jit
from sgl_nvidia_mega.qwen3_8.spec import SPEC

CASES = ((1, 0), (4, 0), (1, 1), (4, 1), (8, 2), (16, 2))
IO_FIELDS = (
    "hidden_states",
    "router_weight",
    "global_scale",
    "logits_out",
    "packed_out",
    "scale_out",
)


def signature(rows=4, variant=0):
    return RouterQuantSignature(O.ROUTER_QUANT, rows, 2560, 512, D.BF16, variant)


class CudaTensor(torch.Tensor):
    """Real CPU tensors; only device metadata is doubled for driver ABI checks."""

    @property
    def device(self):
        return getattr(self, "test_device", torch.device("cuda:2"))


def make_io(rows):
    def empty(shape, dtype):
        return torch.zeros(shape, dtype=dtype).as_subclass(CudaTensor)

    return RouterQuantIO(
        empty((rows, 2560), torch.bfloat16),
        empty((512, 2560), torch.bfloat16),
        empty((1,), torch.float32),
        empty((rows, 512), torch.bfloat16),
        empty((rows, 1280), torch.uint8),
        empty((rows, 160), torch.float8_e4m3fn),
    )


@pytest.fixture
def setup(monkeypatch, tmp_path):
    state = SimpleNamespace(
        capturing=False, loads=[], attributes=[], calls=[], error=None
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 7)
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=71),
    )
    roots = []
    for variant, (variable, split, capacities) in enumerate(
        (
            ("SGLANG_Q38_ROUTER_AOT", 2, (1, 8)),
            ("SGLANG_Q38_ROUTER_K4_AOT", 4, (1, 8)),
            ("SGLANG_Q38_ROUTER_WIDE_AOT", 2, (8, 16)),
        )
    ):
        root = tmp_path / str(variant)
        root.mkdir()
        source = (
            raw._SOURCE if variant < 2 else raw._SOURCE.with_name("q38_router_wide.cu")
        )
        hashes = {}
        for capacity in capacities:
            name = f"route-m{capacity}.cubin"
            data = f"CPU_CUBIN_SPY_ONLY_{variant}_{capacity}".encode()
            (root / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (root / "manifest.json").write_text(
            json.dumps(
                dict(
                    arch="sm_107a",
                    shape=[2560, 512, split],
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    sha256=hashes,
                )
            )
        )
        monkeypatch.setenv(variable, str(root))
        roots.append(root)

    def load(data, name, block, shared):
        state.loads.append((data, name, block, shared))
        return SimpleNamespace(
            function=len(state.loads), block=block, shared_memory_bytes=shared
        )

    def record_attribute(function, attribute, shared):
        state.attributes.append((function, attribute, shared))
        return (0,)

    def launch(kernel, grid, arguments, stream):
        if state.error:
            raise state.error
        # Record the original by-value parameter bytes, not activation owners.
        state.calls.append(
            (kernel, grid, raw.Params.from_buffer_copy(bytes(arguments[0])), stream)
        )

    monkeypatch.setattr(jit, "load", load)
    monkeypatch.setattr(jit, "launch", launch)
    monkeypatch.setattr(
        jit,
        "driver",
        lambda: SimpleNamespace(
            cuFuncSetAttribute=record_attribute,
            CUfunction_attribute=SimpleNamespace(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=8
            ),
        ),
    )
    monkeypatch.setattr(slots, "SLOTS", frozenset(slots.KNOWN_SLOTS))
    monkeypatch.setattr(providers, "numeric_registry", NumericRegistry())
    router_quant._bundle.cache_clear()
    providers.make_offer.cache_clear()
    state.roots = roots
    yield state
    router_quant._bundle.cache_clear()
    providers.make_offer.cache_clear()


def test_canonical_identity():
    assert O.ROUTER_QUANT == 43
    assert RouterQuantSignature in get_args(OperatorSignature)
    assert RouterQuantIO in get_args(NumericIO)
    assert tuple(field.name for field in fields(RouterQuantIO)) == IO_FIELDS
    assert slots.OPERATION_SLOTS[O.ROUTER_QUANT] == "router_quant"


@pytest.mark.parametrize("rows,variant", CASES)
def test_actual_spec_finite_registration_and_source_abi(setup, rows, variant):
    s = signature(rows, variant)
    ctx = SimpleNamespace(numeric_signatures=(s,))
    built = SPEC.build(None, None, ctx)
    offer = providers.numeric_registry.get(s)
    assert built.offers == (offer,) and offer is providers.make_offer(s)
    assert built.offers == SPEC.build(None, None, ctx).offers
    assert len(setup.loads) == 2 and len(setup.attributes) == 2
    split = 4 if variant == 1 else 2
    capacities = (8, 16) if variant == 2 else (1, 8)
    for call, capacity in zip(setup.loads, capacities):
        assert call[1:3] == ("q38_router_gemm_quant", (256, 1, 1))
        assert call[3] == (8 + capacity) * (2560 * 2 + 16) + split * 16 * 8 * 4 + 16
    assert [x[2] for x in setup.attributes] == [x[3] for x in setup.loads]
    io = make_io(rows)
    assert offer.run(io) is None
    _, grid, params, stream = setup.calls[-1]
    assert grid == (64, 1, 1) and stream == 71
    for parameter, field in zip(
        ("x", "weight", "global_scale", "logits", "x_fp4", "x_sf"), IO_FIELDS
    ):
        assert getattr(params, parameter) == getattr(io, field).data_ptr()
    assert params.num_tokens == rows and params.mode == 1
    assert not params.topk_weights and not params.topk_ids and not params.trace
    assert not params.sh_weight and not params.arrival


@pytest.mark.parametrize("rows,variant", CASES)
def test_primary_context_before_eager_driver_module_load(
    setup, monkeypatch, rows, variant
):
    """Queries/lazy init can succeed with no current CUDA context (8699).

    Execute the real shared jit.load/check; only the CUDA driver is doubled.
    A direct cold load reproduces 201; real SPEC.build must bind the selected
    device before loading either cubin, without preparing activation tensors.
    """
    from sgl_nvidia_mega.common import jit as common_jit

    events = []
    current_context = None

    def select_device(device):
        nonlocal current_context
        events.append(("set_device", device))
        current_context = device

    def current_device():
        events.append(("current_device", 2))
        return 2

    def module_load(cubin):
        events.append(("cuModuleLoadData", current_context))
        return (201,) if current_context != 2 else (0, len(events))

    cu = SimpleNamespace(
        cuModuleLoadData=module_load,
        cuModuleGetFunction=lambda module, symbol: (0, module),
        cuFuncSetAttribute=lambda *args: (0,),
        CUfunction_attribute=SimpleNamespace(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=8
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "set_device", select_device)
    monkeypatch.setattr(common_jit, "driver", lambda: cu)
    monkeypatch.setattr(jit, "driver", lambda: cu)
    monkeypatch.setattr(jit, "load", common_jit.load)
    # Metadata and lazy initialization alone are not a current primary context.
    monkeypatch.setattr(torch.cuda, "init", lambda: events.append(("lazy_init", None)))
    torch.cuda.init()
    assert torch.cuda.current_device() == 2
    with pytest.raises(RuntimeError, match="cuModuleLoadData failed: 201"):
        router_quant._bundle(2, variant)
    assert current_context is None
    events.clear()
    built = SPEC.build(
        None, None, SimpleNamespace(numeric_signatures=(signature(rows, variant),))
    )
    assert len(built.offers) == 1
    assert events[:2] == [("current_device", 2), ("set_device", 2)]
    assert len([e for e in events if e[0] == "cuModuleLoadData"]) == 2
    assert all(e[1] == 2 for e in events if e[0] == "cuModuleLoadData")
    assert not setup.calls  # registration itself does not launch the kernel


def test_capture_refuses_before_context_binding(setup, monkeypatch):
    setup.capturing = True
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda device: pytest.fail("context bind in capture")
    )
    with pytest.raises(RuntimeError, match="before capture"):
        providers.make_offer(signature())
    assert not setup.loads and not setup.calls


@pytest.mark.parametrize("field", tuple(f.name for f in fields(RouterQuantSignature)))
@pytest.mark.parametrize("invalid", (True, False, 1.0))
def test_noninteger_signatures_never_alias_cached_offer(setup, field, invalid):
    valid = signature(1)
    assert providers.make_offer(valid) is not None
    invalid_signature = replace(valid, **{field: invalid})
    assert not providers.supported(invalid_signature)
    assert providers.make_offer(invalid_signature) is None


@pytest.mark.parametrize(
    "changes",
    (
        dict(operation=O.LOCAL_EXPERTS),
        dict(rows=0),
        dict(rows=2),
        dict(rows=8),
        dict(rows=16),
        dict(rows=32),
        dict(hidden_size=4096),
        dict(expert_count=256),
        dict(scalar_type=D.FP32),
        dict(variant=3),
        dict(variant=-1),
        dict(variant=2, rows=4),
        dict(variant=2, rows=12),
        dict(variant=1, rows=16),
    ),
)
def test_unsupported_exact_domains_do_not_load(setup, changes):
    assert providers.make_offer(replace(signature(), **changes)) is None
    assert not setup.loads and not setup.calls


@pytest.mark.parametrize("rows,variant", CASES)
def test_eager_abc_capture_host_only_and_no_activation_owners(
    setup, monkeypatch, rows, variant
):
    offer = providers.make_offer(signature(rows, variant))
    ios = [make_io(rows) for _ in range(3)]
    assert len({io.hidden_states.data_ptr() for io in ios}) == 3
    offer.run(ios[0])
    offer.run(ios[1])
    setup.capturing = True

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation/compile/sync during a prepared call")

    for name in ("empty", "empty_like", "zeros", "full", "tensor"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(jit, "compile_source", forbidden)
    monkeypatch.setattr(jit, "load", forbidden)
    offer.run(ios[2])
    assert len(setup.calls) == 3
    assert [c[2].x for c in setup.calls] == [io.hidden_states.data_ptr() for io in ios]
    refs = [weakref.ref(io.hidden_states) for io in ios]
    del ios
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("variant", (0, 1, 2))
def test_cold_capture_refuses_before_aot_load(setup, variant):
    setup.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        providers.make_offer(signature(8 if variant == 2 else 4, variant))
    assert not setup.loads and not setup.calls


@pytest.mark.parametrize("variant", (0, 1, 2))
@pytest.mark.parametrize(
    "fault", ("arch", "source_sha256", "shape", "cubin", "missing")
)
def test_aot_manifest_hard_gates(setup, variant, fault):
    root = setup.roots[variant]
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "cubin":
        (root / next(iter(manifest["sha256"]))).write_bytes(b"corrupt")
    elif fault == "missing":
        (root / next(iter(manifest["sha256"]))).unlink()
    else:
        manifest[fault] = dict(
            arch="sm_103a", source_sha256="wrong", shape=[2560, 512, 8]
        )[fault]
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        providers.make_offer(signature(8 if variant == 2 else 4, variant))
    assert not setup.loads and not setup.calls


@pytest.mark.parametrize(
    "field,fault",
    [
        (field, fault)
        for field in IO_FIELDS
        for fault in ("shape", "dtype", "device", "noncontiguous")
        if (field, fault) != ("global_scale", "noncontiguous")
    ],
)
def test_all_views_validated_before_any_write(setup, field, fault):
    offer = providers.make_offer(signature())
    io = make_io(4)
    t = getattr(io, field)
    if fault == "shape":
        bad = t.unsqueeze(0)
    elif fault == "dtype":
        bad = t.to(torch.float32 if t.dtype != torch.float32 else torch.bfloat16)
    elif fault == "device":
        bad = t.clone()
        bad.test_device = torch.device("cuda:1")
    else:
        bad = (
            torch.empty(t.numel() * 2, dtype=t.dtype)
            .as_subclass(CudaTensor)[::2]
            .view(t.shape)
        )
    with pytest.raises(ValueError, match="exact source ABI"):
        offer.run(replace(io, **{field: bad}))
    assert not setup.calls


@pytest.mark.parametrize("field", ("hidden_states", "router_weight", "packed_out"))
def test_offset_alignment_before_write(setup, field):
    offer = providers.make_offer(signature())
    io = make_io(4)
    t = getattr(io, field)
    bad = (
        torch.empty(t.numel() + 1, dtype=t.dtype)
        .as_subclass(CudaTensor)[1:]
        .view(t.shape)
    )
    with pytest.raises(ValueError, match="exact source ABI"):
        offer.run(replace(io, **{field: bad}))
    assert not setup.calls


def test_output_alias_rejected_and_launch_error_propagates_once(setup):
    offer = providers.make_offer(signature())
    io = make_io(4)
    aliased = io.hidden_states.view(-1)[: 4 * 512].view(4, 512)
    with pytest.raises(ValueError, match="alias"):
        offer.run(replace(io, logits_out=aliased))
    assert not setup.calls
    setup.error = RuntimeError("raw failure after selection")
    with pytest.raises(RuntimeError, match="raw failure"):
        offer.run(io)
    assert not setup.calls
