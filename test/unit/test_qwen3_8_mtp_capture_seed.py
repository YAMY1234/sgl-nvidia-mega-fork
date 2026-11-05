# SPDX-License-Identifier: Apache-2.0
"""Real DTO/registry/adapter calls; CUDA metadata and Triton launch are doubles."""

import gc
import weakref
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpCaptureSeedIO,
    MtpCaptureSeedSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers, slots
from sgl_nvidia_mega.qwen3_8._providers import mtp_capture_seed as impl
from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import mtp_capture_metadata as raw
from sgl_nvidia_mega.qwen3_8.spec import SPEC


class CudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return getattr(self, "test_device", torch.device("cuda:2"))


def signature(mask=0, pdl=1, padding=0):
    types = [D.INT64 if mask & (1 << i) else D.INT32 for i in range(4)]
    return MtpCaptureSeedSignature(
        O.MTP_CAPTURE_SEED,
        4,
        2051,
        2051 + padding,
        8,
        2055,
        2055 + padding,
        0,
        7,
        *types,
        pdl,
    )


def view(s):
    def t(shape, dtype, stride=None):
        x = (
            torch.zeros(shape, dtype=dtype)
            if stride is None
            else torch.empty_strided(shape, stride, dtype=dtype)
        )
        return x.as_subclass(CudaTensor)

    dt = lambda n: torch.int64 if n == D.INT64 else torch.int32
    return MtpCaptureSeedIO(
        t((4, 2051), torch.int32, (s.topk_row_stride, 1)),
        t((1,), dt(s.extend_scalar_type), (3,)),
        t((), dt(s.accept_scalar_type)),
        t((1,), dt(s.request_scalar_type), (4,)),
        t((6,), dt(s.length_scalar_type)),
        t((s.seed_rows, 2055), torch.int32, (s.seed_row_stride, 1)),
        t((s.seed_rows + 2,), torch.int32),
    )


@pytest.fixture
def launch(monkeypatch):
    state = SimpleNamespace(calls=[], capturing=False, error=None)

    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kw):
                state.calls.append(
                    (
                        grid,
                        tuple(
                            (id(t), t.dtype, t.shape, t.stride(), t.data_ptr())
                            for t in args
                        ),
                        kw,
                    )
                )
                if state.error:
                    raise RuntimeError("selected capture seed failure")

            return call

    monkeypatch.setattr(raw, "_mtp_capture_metadata", Kernel())
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS)
    monkeypatch.setattr(providers, "numeric_registry", NumericRegistry())
    providers.make_offer.cache_clear()
    yield state
    providers.make_offer.cache_clear()


@pytest.mark.parametrize("mask", range(16))
@pytest.mark.parametrize("pdl", (0, 1))
@pytest.mark.parametrize("padding", (0, 29))
def test_original_wrapper_exact_call_registry_and_capture_key(
    launch, monkeypatch, mask, pdl, padding
):
    s = signature(mask, pdl, padding)
    io = view(s)
    built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,)))
    (offer,) = built.offers
    assert offer is providers.numeric_registry.get(s) and offer is providers.make_offer(
        s
    )
    assert (
        SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,))).offers
        == built.offers
    )
    offer.run(io)
    actual = launch.calls[-1]
    monkeypatch.setattr(raw, "_PDL_SMALL", bool(pdl))
    assert raw.try_capture_metadata(
        *(getattr(io, f.name) for f in fields(io)), s.front_tokens, s.trash_row
    )
    assert launch.calls[-1] == actual
    assert (
        actual[0] == (9,) and actual[2]["BLOCK"] == 256 and actual[2]["num_warps"] == 4
    )
    assert actual[2]["FRONT"] == 0 and actual[2]["TRASH"] == 7
    assert len(offer.run._warmed) == 1
    launch.capturing = True
    offer.run(view(s))  # Different pointers, same exact metadata/alignment.
    assert len(offer.run._warmed) == 1


@pytest.mark.parametrize("field", [f.name for f in fields(MtpCaptureSeedSignature)])
@pytest.mark.parametrize("bad", (True, 1.0))
def test_reject_noninteger_before_cached_signature_collision(launch, field, bad):
    s = signature()
    assert providers.make_offer(s)
    wrong = replace(s, **{field: bad})
    assert not impl.supported(wrong) and providers.make_offer(wrong) is None


@pytest.mark.parametrize(
    "field,bad",
    (
        ("operation", 1),
        ("rows", 8),
        ("columns", 2055),
        ("topk_row_stride", 2050),
        ("seed_rows", 7),
        ("seed_columns", 2051),
        ("seed_row_stride", 2054),
        ("front_tokens", 1),
        ("trash_row", -1),
        ("extend_scalar_type", D.BF16),
        ("pdl", 2),
    ),
)
def test_unsupported_exact_domains_no_registration(launch, field, bad):
    s = replace(signature(), **{field: bad})
    assert not impl.supported(s) and providers.make_offer(s) is None
    assert SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,))).offers == ()


@pytest.mark.parametrize("field", [f.name for f in fields(MtpCaptureSeedIO)])
def test_all_fields_type_and_device_preflight_before_write(launch, field):
    s = signature()
    run = providers.make_offer(s).run
    io = view(s)
    with pytest.raises(ValueError, match="seven tensors"):
        run(replace(io, **{field: None}))
    t = getattr(io, field)
    t.test_device = torch.device("cuda:3")
    with pytest.raises(ValueError, match="share the CUDA"):
        run(io)
    assert not launch.calls and not run._warmed


@pytest.mark.parametrize("field", [f.name for f in fields(MtpCaptureSeedIO)])
def test_all_field_dtypes_reject_before_write(launch, field):
    s = signature()
    run = providers.make_offer(s).run
    io = view(s)
    wrong = getattr(io, field).to(torch.float32)
    with pytest.raises(ValueError):
        run(replace(io, **{field: wrong}))
    assert not launch.calls and not run._warmed


def test_stride_capacity_and_scalar_validation(launch):
    s = signature()
    run = providers.make_offer(s).run
    io = view(s)
    wrongs = [
        replace(io, topk_indices=io.topk_indices[:, ::2]),
        replace(io, seed_out=io.seed_out[:-1]),
        replace(io, captured_lengths_out=io.captured_lengths_out[:2]),
        replace(io, expanded_lengths=io.expanded_lengths[:3]),
        replace(io, expanded_lengths=io.expanded_lengths[::2]),
        replace(io, requests=io.requests.expand(2)),
    ]
    for wrong in wrongs:
        with pytest.raises(ValueError):
            run(wrong)
    assert not launch.calls


def test_cold_exact_key_and_selected_exception_no_retry_or_owner_retention(launch):
    s = signature()
    run = providers.make_offer(s).run
    io = view(s)
    launch.capturing = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(io)
    assert not launch.calls
    launch.capturing = False
    launch.error = True
    with pytest.raises(RuntimeError, match="selected capture seed failure"):
        run(io)
    assert len(launch.calls) == 1 and not run._warmed
    launch.error = False
    run(io)
    refs = [weakref.ref(getattr(io, f.name)) for f in fields(io)]
    del io
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert len(run._warmed) == 1


def test_slot_off_is_stock(launch, monkeypatch):
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS - {"mtp_capture_seed"})
    s = signature()
    assert providers.make_offer(s) is None
    assert SPEC.build(None, None, SimpleNamespace(numeric_signatures=(s,))).offers == ()
