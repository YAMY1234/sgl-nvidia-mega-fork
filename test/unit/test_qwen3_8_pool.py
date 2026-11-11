# SPDX-License-Identifier: Apache-2.0
"""Exact original pool-wrapper call ABI, with real CPU tensors and launch spies."""

import ast
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    DirectPoolAttentionIO,
    DirectPoolSelection,
    KvTensors,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
    NumericSignature,
    PoolUnionAttentionIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import pool, pool_union


def signature(rows=4, variant=1):
    return NumericSignature(
        O.DIRECT_POOL_ATTENTION, rows, 24, 2, 256, 512, 4, D.BF16, variant=variant
    )


def view(rows=4, width=4, sms=200, strided=False, store=True):
    bf = lambda shape: torch.empty(shape, dtype=torch.bfloat16)
    packed = bf((rows, 2304 if strided else 1024))
    nk, nv = packed[:, :512].view(rows, 2, 256), packed[:, 512:1024].view(rows, 2, 256)
    # Keep the original hardware formula independent of the provider under test.
    rh = rows * 2
    bn, waves = (32, 3) if rh >= 32 else (64, 1)
    splits = max(1, round(waves * sms / rh))
    ceil = lambda x, y: (x + y - 1) // y
    chunk = ceil(ceil(2051, splits), bn) * bn
    splits = ceil(2051, chunk)
    return DirectPoolAttentionIO(
        bf((rows, 24, 256)),
        None,
        nk,
        nv,
        torch.ones(rows, dtype=torch.int64),
        KvTensors(bf((96, 2, 256)), bf((96, 2, 256)), None, None, 1),
        DirectPoolSelection(
            torch.zeros((16, 1024), dtype=torch.int64),
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros((rows, 512), dtype=torch.int32),
            torch.full((rows,), 8192, dtype=torch.int32),
            torch.full((rows,), 8191, dtype=torch.int64),
        ),
        0.0625,
        width,
        store,
        bf((rows, 24, 256)),
        torch.empty(rows * 2 * splits * 16 * 258, dtype=torch.float32),
    )


def raw_spy(monkeypatch, sms=200):
    name = "sgl_nvidia_mega.qwen3_8.qsa.pool_decode"
    raw = ModuleType(name)
    state = SimpleNamespace(capturing=False, calls=[], props=0, fail=None)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def run(*args, **kw):
                state.calls.append((self.name, grid, args, kw))
                if state.fail == self.name:
                    raise RuntimeError("raw launch failed")
                return self.name

            return run

    raw.torch = torch
    raw.LOG2E = 1.4426950408889634
    raw.triton = SimpleNamespace(
        cdiv=lambda x, y: (x + y - 1) // y,
        next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
    )
    raw.LAST_KERNEL = {}
    raw._MODES = {"separate_write": 0, "fresh_select": 1, "fresh_dual": 2}
    raw._num_sms = lambda device: hardware()

    def hardware():
        assert not state.capturing, "hardware discovery during capture"
        state.props += 1
        return sms

    raw._qsa_write_new_kv = Kernel("store")
    raw._qsa_pool_splitk = Kernel("split")
    raw._qsa_pool_merge = Kernel("merge")
    path = Path(pool.__file__).parents[1] / "qsa/pool_decode.py"
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("choose_config", "qsa_pool_sparse_decode")
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        raw.__dict__,
    )
    monkeypatch.setitem(sys.modules, name, raw)
    from sgl_nvidia_mega.qwen3_8 import qsa

    monkeypatch.setattr(qsa, "pool_decode", raw, raising=False)
    return raw, state


def normalize(x):
    if isinstance(x, torch.Tensor):
        return (
            x.untyped_storage()._cdata,
            x.storage_offset(),
            tuple(x.shape),
            tuple(x.stride()),
            x.dtype,
        )
    if isinstance(x, tuple):
        return tuple(normalize(y) for y in x)
    if isinstance(x, list):
        return [normalize(y) for y in x]
    if isinstance(x, dict):
        return {k: normalize(v) for k, v in x.items()}
    return x


@pytest.mark.parametrize(
    "rows,variant", [(m, 1) for m in range(4, 65, 4)] + [(m, 2) for m in range(1, 65)]
)
def test_finite_exact_signatures(rows, variant):
    s = signature(rows, variant)
    assert providers.supported(s)
    assert providers.make_offer(s).signature == s
    assert isinstance(providers.make_offer(s).run, pool.PoolAttention)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 0),
        ("rows", 65),
        ("query_heads", 6),
        ("kv_heads", 1),
        ("head_dim", 128),
        ("selected_count", 256),
        ("compress_ratio", 8),
        ("scalar_type", D.FP16),
        ("rotary_dim", 64),
        ("context_bound", 8192),
        ("variant", 0),
        ("variant", 3),
    ],
)
def test_bad_shape_has_no_offer(field, value):
    s = replace(signature(), **{field: value})
    assert not providers.supported(s)
    assert providers.make_offer(s) is None
    with pytest.raises(ValueError):
        pool.build(s)


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 6, 7, 63])
def test_verify_requires_source_width4(rows):
    assert not providers.supported(signature(rows, 1))


def test_old_tp4_only_variant0():
    s = NumericSignature(O.DIRECT_POOL_ATTENTION, 4, 6, 1, 256, 512, 4, D.BF16)
    assert providers.supported(s)
    for v in (1, 2, 3):
        assert not providers.supported(replace(s, variant=v))


@pytest.mark.parametrize(
    "rows,variant", [(4, 1), (8, 1), (32, 1), (64, 1), (1, 2), (8, 2), (32, 2), (64, 2)]
)
@pytest.mark.parametrize("sms", [132, 200])
@pytest.mark.parametrize("strided,store", [(False, True), (True, True), (True, False)])
def test_original_wrapper_exact_launch_abi(
    monkeypatch, rows, variant, sms, strided, store
):
    raw, state = raw_spy(monkeypatch, sms)
    io = view(rows, 4 if variant == 1 else 1, sms, strided, store)
    expected = raw.qsa_pool_sparse_decode(
        io.q,
        io.kv.k,
        io.kv.v,
        io.new_k,
        io.new_v,
        io.out_cache_loc,
        io.selection.full_slot_map,
        io.selection.row_requests,
        io.selection.selected_indices,
        io.selection.sequence_lengths,
        io.selection.query_positions,
        io.query_width,
        io.scale,
        out=io.output.view(rows, 6144),
        workspace=io.scratch_out,
        write_new_kv=store,
    )
    calls = normalize(state.calls)
    state.calls.clear()
    run = pool.build(signature(rows, variant))
    assert run(io) is None
    assert normalize(state.calls) == calls
    assert expected.data_ptr() == io.output.data_ptr()
    assert all(call[-1]["launch_pdl"] is True for call in state.calls)


@pytest.mark.parametrize("rows,variant", [(4, 5), (64, 5), (1, 6), (64, 6)])
def test_merge_gate_variants_are_supported(rows, variant):
    s = signature(rows, variant)
    assert providers.supported(s)
    assert isinstance(providers.make_offer(s).run, pool.PoolAttention)


@pytest.mark.parametrize("rows,variant", [(4, 4), (4, 7), (4, 8), (4, 13), (2, 5)])
def test_merge_gate_needs_a_route_bit(rows, variant):
    assert not providers.supported(signature(rows, variant))


def gated(io, rows):
    """The model's gate: a [rows, 24, 256] view of the interleaved Q+gate rows."""
    q_gate = torch.empty((rows, 24, 512), dtype=torch.bfloat16)
    return replace(io, gate=q_gate[:, :, 256:])


@pytest.mark.parametrize("rows,variant", [(4, 1), (64, 1), (1, 2), (32, 2)])
def test_merge_gate_launch_is_the_wrappers(monkeypatch, rows, variant):
    raw, state = raw_spy(monkeypatch)
    io = gated(view(rows, 4 if variant == 1 else 1), rows)
    expected = raw.qsa_pool_sparse_decode(
        io.q,
        io.kv.k,
        io.kv.v,
        io.new_k,
        io.new_v,
        io.out_cache_loc,
        io.selection.full_slot_map,
        io.selection.row_requests,
        io.selection.selected_indices,
        io.selection.sequence_lengths,
        io.selection.query_positions,
        io.query_width,
        io.scale,
        out=io.output.view(rows, 6144),
        workspace=io.scratch_out,
        gate=io.gate,
    )
    calls = normalize(state.calls)
    state.calls.clear()
    assert pool.build(signature(rows, variant | pool.MERGE_GATE))(io) is None
    assert normalize(state.calls) == calls
    merge = state.calls[-1]
    assert merge[0] == "merge" and merge[3]["HAS_GATE"] is True
    assert merge[3]["GATE"] is io.gate
    assert (merge[3]["GATE_STRIDE_ROW"], merge[3]["GATE_STRIDE_HEAD"]) == (
        24 * 512,
        512,
    )
    assert expected.data_ptr() == io.output.data_ptr()


@pytest.mark.parametrize("case", ["missing", "unexpected", "shape", "dtype", "inner"])
def test_merge_gate_refusal_before_any_write(monkeypatch, case):
    _, state = raw_spy(monkeypatch)
    io, variant = gated(view(), 4), 1 | pool.MERGE_GATE
    if case == "missing":
        io = replace(io, gate=None)
    elif case == "unexpected":
        variant = 1
    elif case == "shape":
        io = replace(io, gate=io.gate[:, :12])
    elif case == "dtype":
        io = replace(io, gate=io.gate.float())
    else:
        io = replace(io, gate=io.gate.transpose(1, 2).contiguous().transpose(1, 2))
    with pytest.raises(ValueError):
        pool.build(signature(4, variant))(io)
    assert state.calls == []


def test_changed_activation_capture_uses_warm_layout_no_owners(monkeypatch):
    _, state = raw_spy(monkeypatch)
    run = pool.build(signature())
    a, b, c = view(), view(), view()
    owners = [weakref.ref(x) for x in (a.q, a.new_k, a.output, a.scratch_out)]
    run(a)
    run(b)
    props = state.props
    state.capturing = True
    run(c)
    assert state.props == props == 1
    assert a.q.data_ptr() != b.q.data_ptr() != c.q.data_ptr()
    state.calls.clear()
    del a
    assert all(ref() is None for ref in owners)


def test_cold_capture_rejects_before_properties_or_write(monkeypatch):
    _, state = raw_spy(monkeypatch)
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        pool.build(signature())(view())
    assert state.props == 0 and state.calls == []


def test_cold_layout_capture_rejects_without_write(monkeypatch):
    _, state = raw_spy(monkeypatch)
    run = pool.build(signature())
    run(view())
    state.calls.clear()
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        run(view(strided=True))
    assert state.calls == [] and state.props == 1


@pytest.mark.parametrize(
    "field",
    [
        "gate",
        "query_width",
        "output",
        "scratch_out",
        "selected_indices",
        "row_requests",
        "full_slot_map",
        "sequence_lengths",
        "query_positions",
        "new_k",
        "new_v",
        "q",
        "out_cache_loc",
    ],
)
def test_refusal_before_any_write(monkeypatch, field):
    _, state = raw_spy(monkeypatch)
    io = view()
    if field == "gate":
        io = replace(io, gate=io.q)
    elif field == "query_width":
        io = replace(io, query_width=1)
    elif field == "output":
        io = replace(io, output=io.output.float())
    elif field == "scratch_out":
        io = replace(io, scratch_out=io.scratch_out[:1])
    elif field in (
        "selected_indices",
        "row_requests",
        "full_slot_map",
        "sequence_lengths",
        "query_positions",
    ):
        t = getattr(io.selection, field)
        io = replace(io, selection=replace(io.selection, **{field: t.float()}))
    elif field in ("new_k", "new_v", "q"):
        io = replace(io, **{field: getattr(io, field).float()})
    else:
        io = replace(io, out_cache_loc=io.out_cache_loc.float())
    with pytest.raises(ValueError):
        pool.build(signature())(io)
    assert state.calls == []


def test_cpu_tensor_rejected(monkeypatch):
    raw, state = raw_spy(monkeypatch)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: False))
    with pytest.raises(ValueError):
        pool.build(signature())(view())
    assert not state.calls


def test_failure_after_write_propagates_without_warming_or_retry(monkeypatch):
    _, state = raw_spy(monkeypatch)
    run = pool.build(signature())
    io = view()
    state.fail = "split"
    with pytest.raises(RuntimeError, match="raw launch failed"):
        run(io)
    assert [c[0] for c in state.calls] == ["store", "split"]
    assert not run._prepared
    state.fail = None
    state.calls.clear()
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        run(io)
    assert not state.calls


def test_adapter_has_no_allocate_cast_copy_compile_sync_or_sort():
    tree = ast.parse(Path(pool.__file__).read_text())
    bad = {
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
        "compile",
    }
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in bad
    ]


def test_canonical_registry_facade_and_master_gates(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    _, state = raw_spy(monkeypatch)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    providers.make_offer.cache_clear()
    sig = signature()
    try:
        built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=(sig,)))
        assert len(built.offers) == 1 and registry.get(sig) is built.offers[0]
        assert numeric_binding.numeric_operator(sig, prepared=False) is None
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False):
            assert numeric_binding.numeric_operator(sig, prepared=True) is None
        assert not state.calls
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
            run = numeric_binding.numeric_operator(sig, prepared=True)
            assert run(view()) is None
    finally:
        providers.make_offer.cache_clear()


@pytest.mark.parametrize(
    "slots,expected", [(None, 3), ("", 0), ("qsa_core", 3), ("hc_gate", 0)]
)
def test_pool_variants_follow_same_slot_in_fresh_process(slots, expected):
    import json
    import os
    import subprocess

    child = """
import json
from types import SimpleNamespace
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericSignature,NumericOperation as O,ScalarType as D,NumericRegistry
from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8.spec import SPEC
registry=NumericRegistry()
providers.numeric_registry=registry
signatures=(NumericSignature(O.DIRECT_POOL_ATTENTION,4,6,1,256,512,4,D.BF16),
 NumericSignature(O.DIRECT_POOL_ATTENTION,4,24,2,256,512,4,D.BF16,variant=1),
 NumericSignature(O.DIRECT_POOL_ATTENTION,4,24,2,256,512,4,D.BF16,variant=2))
built=SPEC.build(None,None,SimpleNamespace(numeric_signatures=signatures))
assert built.offers==SPEC.build(None,None,SimpleNamespace(numeric_signatures=signatures)).offers
print(json.dumps({'offers':len(built.offers),'supported':[providers.supported(s) for s in signatures],
 'registered':[registry.get(s) is not None for s in signatures]}))
"""
    env = dict(os.environ)
    if slots is None:
        env.pop("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", None)
    else:
        env["SGLANG_NVIDIA_MEGAKERNEL_SLOTS"] = slots
    result = subprocess.run(
        [sys.executable, "-c", child],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(result.stdout)
    assert record["offers"] == expected
    assert record["supported"] == record["registered"] == [bool(expected)] * 3


def test_cold_pointer_alignment_capture_rejects_before_write(monkeypatch):
    _, state = raw_spy(monkeypatch)
    run = pool.build(signature())
    run(view())
    state.calls.clear()
    io = view()
    shifted = torch.ones(5, dtype=torch.int64)[1:]
    assert shifted.data_ptr() % 16 == 8
    io = replace(io, out_cache_loc=shifted)
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        run(io)
    assert not state.calls


def union_signature(rows=64):
    return NumericSignature(O.POOL_UNION_ATTENTION, rows, 24, 2, 256, 512, 4, D.BF16)


@pytest.mark.parametrize("rows", [64, 68, 128, 416, 576])
def test_union_supported_rows(rows):
    assert pool_union.supported(union_signature(rows))


@pytest.mark.parametrize(
    "change",
    [
        dict(rows=0),
        dict(rows=63),
        dict(rows=578),
        dict(query_heads=6),
        dict(kv_heads=1),
        dict(scalar_type=D.FP8_E4M3),
        dict(variant=3),
        dict(variant=1 << 7),
        dict(context_bound=1),
        dict(rotary_dim=1),
        dict(rows=True),
        dict(rows=64.0),
    ],
)
def test_union_refusal(change):
    assert not pool_union.supported(replace(union_signature(), **change))


@pytest.mark.parametrize("variant", [16, 32, 64, 64 | 16, 8, 4])
def test_union_declared_options(variant):
    # Bits 0-5: kernel U warps (0 = 8); bit 6: the warp-specialized kernel.
    assert pool_union.supported(replace(union_signature(), variant=variant))


class CudaMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def union_view():
    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype).as_subclass(CudaMetadata)

    q = empty((64, 24, 256), torch.bfloat16)
    k = empty((80, 2, 256), torch.float8_e4m3fn)
    nk = empty((64, 2, 256), torch.float8_e4m3fn)
    loc = empty((64,), torch.int64)
    sel = DirectPoolSelection(
        empty((2, 80), torch.int32), loc, empty((64, 512), torch.int32), loc, loc
    )
    return PoolUnionAttentionIO(
        q,
        None,
        nk,
        nk,
        loc,
        KvTensors(k, k, 1.0, 1.0, 1),
        sel,
        0.0625,
        4,
        True,
        torch.empty_like(q),
        empty((16 * 2 * 2 * 64 * 258,), torch.float32),
        empty((16 * (2 * 8208 + 2048 + 1) + 64,), torch.int32),
        2,
    )


def raw_union(monkeypatch, launch, ws_launch=None):
    import sgl_nvidia_mega.qwen3_8.qsa as package

    module = ModuleType(package.__name__ + ".pool_union_decode")
    module.qsa_pool_union_decode = launch
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(package, "pool_union_decode", module, raising=False)
    ws = ModuleType(package.__name__ + ".pool_union_ws")
    ws.qsa_pool_union_decode_ws = ws_launch or launch
    monkeypatch.setitem(sys.modules, ws.__name__, ws)
    monkeypatch.setattr(package, "pool_union_ws", ws, raising=False)


def test_union_launch_caller_output_and_capture(monkeypatch):
    calls = []
    capture = [False]
    raw_union(monkeypatch, lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture[0])
    run = pool_union.build(union_signature())
    io = union_view()
    capture[0] = True
    with pytest.raises(RuntimeError, match="warmed"):
        run(io)
    assert not calls
    capture[0] = False
    run(io)
    assert calls[-1][1]["out"].data_ptr() == io.output.data_ptr()
    workspace = calls[-1][1]["workspace"]
    capture[0] = True
    run(replace(io, output=torch.empty_like(io.output)))
    assert calls[-1][1]["workspace"] is not workspace
    assert calls[-1][1]["workspace"][0].data_ptr() == workspace[0].data_ptr()
    with pytest.raises(ValueError):
        run(replace(io, query_width=1))
    assert len(calls) == 2


@pytest.mark.parametrize(
    "variant,warps,ws", [(0, 8, False), (16, 16, False), (64, 8, True), (80, 16, True)]
)
def test_union_launch_takes_the_declared_options(monkeypatch, variant, warps, ws):
    calls = []
    raw_union(
        monkeypatch,
        lambda *a, **kw: calls.append(("union", kw)),
        lambda *a, **kw: calls.append(("ws", kw)),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    io = union_view()
    pool_union.build(replace(union_signature(), variant=variant))(io)
    ((name, kw),) = calls
    assert name == ("ws" if ws else "union") and kw["union_warps"] == warps
    assert kw["splits"] == io.split_count and len(kw["workspace"]) == 8


def test_union_failed_launch_does_not_publish(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("launch failure")

    raw_union(monkeypatch, fail)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    run = pool_union.build(union_signature())
    with pytest.raises(RuntimeError, match="launch failure"):
        run(union_view())
    assert not run._resources


@pytest.mark.parametrize("bad", ["zero", "nan", "large_error"])
def test_union_numeric_gate_rejects_bad_output(bad):
    path = Path(__file__).parents[1] / "kernels/test_qwen3_8_pool_union.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "check_tolerance"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    reference = torch.linspace(0.001, 0.01, 128).reshape(8, 16)
    out = reference.to(torch.bfloat16)
    scope["check_tolerance"](out, reference)
    if bad == "zero":
        out.zero_()
    elif bad == "nan":
        out[0, 0] = float("nan")
    else:
        out += 0.002
    with pytest.raises(AssertionError):
        scope["check_tolerance"](out, reference)


def packed_signature(**kw):
    return replace(
        NumericSignature(
            O.PACKED_PREFILL_ATTENTION, 0, 24, 2, 256, 2051, 4, D.BF16, variant=3
        ),
        **kw,
    )


@pytest.mark.parametrize("qh,kh", [(24, 2), (6, 1), (16, 1), (32, 2)])
def test_packed_prefill_head_groups(qh, kh):
    assert providers.supported(packed_signature(query_heads=qh, kv_heads=kh))


@pytest.mark.parametrize(
    "change",
    [
        dict(rows=513),
        dict(kv_heads=0),
        dict(query_heads=25),
        dict(query_heads=34),
        dict(head_dim=128),
        dict(selected_count=512),
        dict(context_bound=257),
        dict(ctas_per_group=128),
        dict(variant=0),
        dict(variant=9),
        dict(scalar_type=D.FP8_E4M3),
    ],
)
def test_packed_prefill_refusal(change):
    assert providers.make_offer(packed_signature(**change)) is None


def packed_view(rows=513, max_q=257):
    from sglang.kernels.plugin_kernels.nvidia.numeric import PackedPrefillAttentionIO

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype).as_subclass(CudaMetadata)

    q = empty((rows, 24, 256), torch.bfloat16)
    k = empty((1024, 2, 256), torch.bfloat16)
    return PackedPrefillAttentionIO(
        q,
        k,
        k,
        empty((rows, 2051), torch.int32),
        empty((3,), torch.int64),
        empty((3,), torch.int64),
        empty((2,), torch.int32),
        0.0625,
        max_q,
        torch.empty_like(q),
    )


def test_packed_prefill_launch_grid_stages_and_capture(monkeypatch):
    import sgl_nvidia_mega.qwen3_8.qsa as package
    from sgl_nvidia_mega.qwen3_8._providers import packed_prefill

    raw = ModuleType(package.__name__ + ".tma_prefill")
    calls = []
    raw.launch = lambda io, *, max_q, stages: calls.append((io, max_q, stages))
    monkeypatch.setitem(sys.modules, raw.__name__, raw)
    monkeypatch.setattr(package, "tma_prefill", raw, raising=False)
    capturing = SimpleNamespace(value=True)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing.value
    )
    run = packed_prefill.build(packed_signature(variant=2))
    io = packed_view()
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(io)
    capturing.value = False
    run(io)
    capturing.value = True
    longer = packed_view(rows=600, max_q=600)
    run(longer)
    assert calls == [(io, 257, 2), (longer, 600, 2)]
    for bad in (
        replace(io, max_query_rows=0),
        replace(io, max_query_rows=514),
        replace(io, selected_indices=io.selected_indices[:, :512]),
        replace(io, cu_key=io.cu_key[:2]),
    ):
        with pytest.raises(ValueError, match="declared layout"):
            run(bad)
    assert len(calls) == 2
