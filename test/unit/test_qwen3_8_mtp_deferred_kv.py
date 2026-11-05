# SPDX-License-Identifier: Apache-2.0
"""Exact LOOKUP source call, caller tensors, and refusal-before-write boundaries."""

import gc
import weakref
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpDeferredKvIO,
    MtpDeferredKvSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers, slots
from sgl_nvidia_mega.qwen3_8._providers import mtp_deferred_kv as impl
from sgl_nvidia_mega.qwen3_8.qsa import fused_kv as raw
from sgl_nvidia_mega.qwen3_8.spec import SPEC


class CudaTensor(torch.Tensor):
    @property
    def device(self):
        return getattr(self, "test_device", torch.device("cuda:2"))


def signature(mask=0, outer=512):
    return MtpDeferredKvSignature(
        O.MTP_DEFERRED_KV_PREPARE,
        1,
        1,
        256,
        D.BF16,
        2055,
        2112,
        4,
        8,
        32,
        7,
        4096,
        4128,
        outer,
        outer + 256,
        mask,
    )


def view(s):
    def t(shape, dtype, stride=None):
        x = (
            torch.zeros(shape, dtype=dtype)
            if stride is None
            else torch.empty_strided(shape, stride, dtype=dtype)
        )
        return x.as_subclass(CudaTensor)

    integer = lambda bit: (
        torch.int64 if s.metadata_dtype_mask & (1 << bit) else torch.int32
    )
    return MtpDeferredKvIO(
        t((s.cache_slots, 1, 256), torch.bfloat16),
        t((s.cache_slots, 1, 256), torch.bfloat16),
        t((1, 1, 256), torch.bfloat16, (s.new_k_row_stride, 256, 1)),
        t((1, 1, 256), torch.bfloat16, (s.new_v_row_stride, 256, 1)),
        t((1,), integer(0), (3,)),
        t((s.request_pool_rows, s.map_columns), integer(3), (s.map_row_stride, 1)),
        t((1,), integer(1)),
        t((1,), integer(2)),
        t((s.index_pool_rows, 2055), torch.int32),
        t((1,), integer(4)),
        t((s.index_pool_rows,), torch.int32),
        t((1,), integer(5)),
        t((1,), torch.int32),
        t((2112, 1, 256), torch.bfloat16),
        t((2112, 1, 256), torch.bfloat16),
    )


@pytest.fixture
def launch(monkeypatch):
    state = SimpleNamespace(calls=[], capturing=False, error=False)

    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kw):
                state.calls.append(
                    (
                        grid,
                        tuple(
                            (id(a), a.dtype, a.shape, a.stride(), a.data_ptr())
                            if isinstance(a, torch.Tensor)
                            else a
                            for a in args
                        ),
                        kw,
                    )
                )
                if state.error:
                    raise RuntimeError("selected deferred KV failed")

            return call

    monkeypatch.setattr(raw, "_qsa_fused_kv_prepare", Kernel())
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS)
    monkeypatch.setattr(providers, "numeric_registry", NumericRegistry())
    providers.make_offer.cache_clear()
    yield state
    providers.make_offer.cache_clear()


@pytest.mark.parametrize("mask", range(64))
def test_all_metadata_masks_original_wrapper_15caller_pointers_and_capture(
    launch, mask
):
    s = signature(mask)
    io = view(s)
    built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,)))
    (offer,) = built.offers
    assert (
        offer is providers.numeric_registry.get(s) and providers.make_offer(s) is offer
    )
    offer.run(io)
    actual = launch.calls[-1]
    lookup = raw.DeferredMtpSparseIndices(
        io.indices_pool,
        io.captured_lengths,
        io.lookup_requests,
        io.lookup_positions,
        torch.arange(4),
    )
    raw.fused_kv_prepare(
        io.k_cache,
        io.v_cache,
        io.new_k,
        io.new_v,
        io.locations,
        io.req_to_token,
        io.row_requests,
        lookup,
        io.sequence_lengths,
        io.counts_out,
        io.packed_k_out,
        io.packed_v_out,
        1,
        compress_ratio=1,
        query_positions=None,
        chain_positions=True,
        visible_page_size=64,
    )
    assert launch.calls[-1] == actual
    assert (
        actual[0] == (134,) and actual[1][9] is None and actual[2] == dict(num_warps=4)
    )
    assert actual[1][16:] == (
        1,
        256,
        4128,
        512,
        768,
        2055,
        2112,
        1,
        16,
        4096,
        1,
        1,
        False,
        1,
        2055,
        True,
        0,
        True,
        4,
    )
    launch.capturing = True
    offer.run(view(s))
    assert len(offer.run._warmed) == 1


@pytest.mark.parametrize("outer", (1, 17, 256, 512))
def test_real_singleton_outer_stride_not_repacked(launch, outer):
    s = signature(outer=outer)
    io = view(s)
    run = providers.make_offer(s).run
    run(io)
    assert launch.calls[-1][1][19:21] == (outer, outer + 256)


@pytest.mark.parametrize("field", [f.name for f in fields(MtpDeferredKvSignature)])
@pytest.mark.parametrize("bad", (True, 1.0))
def test_bool_float_rejected_before_cached_signature_key(launch, field, bad):
    s = signature()
    assert providers.make_offer(s)
    wrong = replace(s, **{field: bad})
    assert not impl.supported(wrong) and providers.make_offer(wrong) is None


@pytest.mark.parametrize(
    "field,bad",
    (
        ("operation", O.DIRECT_POOL_ATTENTION),
        ("rows", 4),
        ("kv_heads", 2),
        ("head_dim", 128),
        ("scalar_type", D.FP8_E4M3),
        ("index_columns", 2051),
        ("packed_columns", 2055),
        ("tail_width", 3),
        ("index_pool_rows", 0),
        ("cache_slots", 0),
        ("request_pool_rows", 0),
        ("map_columns", 0),
        ("map_row_stride", 4095),
        ("new_k_row_stride", 0),
        ("metadata_dtype_mask", 64),
    ),
)
def test_firstcut_unsupported_and_TP1_FP8_stock(launch, field, bad):
    s = replace(signature(), **{field: bad})
    assert not impl.supported(s)
    assert providers.make_offer(s) is None
    assert SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,))).offers == ()


@pytest.mark.parametrize("field", [f.name for f in fields(MtpDeferredKvIO)])
def test_each_borrowed_tensor_type_device_dtype_rejects_before_cache_write(
    launch, field
):
    s = signature()
    io = view(s)
    run = providers.make_offer(s).run
    with pytest.raises(ValueError, match="fifteen tensors"):
        run(replace(io, **{field: None}))
    wrong = getattr(io, field).to(torch.float32)
    with pytest.raises(ValueError):
        run(replace(io, **{field: wrong}))
    getattr(io, field).test_device = torch.device("cuda:3")
    with pytest.raises(ValueError, match="share the CUDA"):
        run(io)
    assert not launch.calls and not run._warmed


@pytest.mark.parametrize("field", [f.name for f in fields(MtpDeferredKvIO)])
def test_every_field_shape_rejected_before_any_write(launch, field):
    s = signature()
    io = view(s)
    run = providers.make_offer(s).run
    with pytest.raises(ValueError):
        run(replace(io, **{field: getattr(io, field).flatten()[:0]}))
    assert not launch.calls


def test_mask_and_inner_stride_cache_layout_exactness(launch):
    s = signature()
    io = view(s)
    run = providers.make_offer(s).run
    wrongs = [
        replace(io, indices_pool=io.indices_pool[:, ::2]),
        replace(io, req_to_token=io.req_to_token[:, ::2]),
        replace(io, new_k=io.new_k.transpose(1, 2)),
        replace(io, packed_v_out=io.packed_v_out[::2]),
        replace(io, lookup_positions=io.lookup_positions.to(torch.int64)),
    ]
    for wrong in wrongs:
        with pytest.raises(ValueError):
            run(wrong)
    assert not launch.calls


def test_cold_capture_exception_boundary_no_retry_no_activation_owners(launch):
    s = signature()
    io = view(s)
    run = providers.make_offer(s).run
    launch.capturing = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(io)
    assert not launch.calls
    launch.capturing = False
    launch.error = True
    with pytest.raises(RuntimeError, match="selected deferred KV failed"):
        run(io)
    assert len(launch.calls) == 1 and not run._warmed
    launch.error = False
    run(io)
    refs = [weakref.ref(getattr(io, f.name)) for f in fields(io)]
    del io
    gc.collect()
    assert all(ref() is None for ref in refs) and len(run._warmed) == 1


def test_slot_off_leaves_stock(launch, monkeypatch):
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS - {"mtp_deferred_kv"})
    s = signature()
    assert providers.make_offer(s) is None
    assert SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,))).offers == ()
