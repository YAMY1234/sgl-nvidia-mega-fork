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
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import pool


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
