# SPDX-License-Identifier: Apache-2.0
"""Core norm ABI/caller ownership tests; no GPU numeric qualification."""

import sys
from dataclasses import replace
from types import ModuleType

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnCoreIO,
    GdnRingIO,
    GdnSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import gdn


def signature(rows=4, hk=4, hv=12, variant=1):
    return GdnSignature(
        O.GDN_CORE,
        rows,
        hk,
        hv,
        128,
        128,
        4,
        D.BF16,
        variant=variant,
        gate_scalar_type=D.BF16,
    )


def view(s, fused=True):
    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    # Packed projection stores a/b as strided row views, never cast/contiguous.
    ab = tensor((s.rows, s.value_heads * 2))
    a, b = ab[:, s.value_heads :], ab[:, : s.value_heads]
    fields = {name: tensor((1,)) for name in GdnRingIO.__dataclass_fields__}
    return GdnCoreIO(
        tensor((s.rows, (2 * s.key_heads + s.value_heads) * 128)),
        a,
        b,
        a.stride(0),
        b.stride(0),
        tensor((s.value_heads,), torch.float32),
        tensor((s.value_heads,)),
        tensor((2, s.value_heads, 128, 128)),
        GdnRingIO(**fields),
        tensor((s.rows, s.value_heads, 128)) if fused else None,
        tensor((128,)) if fused else None,
        1e-6,
        128,
        tensor((s.rows, s.value_heads, 128)),
        tensor((s.rows, s.value_heads, 128)) if fused else None,
    )


def install_core(monkeypatch):
    calls = []
    module = ModuleType("sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_mkgdn")
    module.core = lambda *a, **kw: calls.append((a, kw))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize("rows", [4, 8, 12])
@pytest.mark.parametrize("hk,hv", [(4, 12), (16, 48)])
@pytest.mark.parametrize("fused", [False, True])
def test_exact_views_and_raw_optional_fields_preserved(
    monkeypatch, rows, hk, hv, fused
):
    s = signature(rows, hk, hv, int(fused))
    io = view(s, fused)
    calls = install_core(monkeypatch)
    assert gdn.build_core(s)(io) is None
    args, kw = calls[0]
    assert args[0] is io.qkv and args[1] is io.a and args[2] is io.b
    assert args[-1] is io.raw_out
    assert args[5] is io.checkpoint
    assert kw["z"] is io.z and kw["norm_weight"] is io.norm_weight
    assert kw["normalized"] is io.normalized_out
    assert kw["eps"] == io.epsilon and kw["vrows"] == 128
    assert io.a.stride(0) == io.b.stride(0) == hv * 2


@pytest.mark.parametrize("field", ["z", "norm_weight", "normalized_out"])
def test_incomplete_fused_view_refuses_before_launch(monkeypatch, field):
    s = signature()
    calls = install_core(monkeypatch)
    with pytest.raises(ValueError, match="complete caller"):
        gdn.build_core(s)(replace(view(s), **{field: None}))
    assert calls == []


def test_raw_offer_cannot_silently_change_to_norm(monkeypatch):
    calls = install_core(monkeypatch)
    s = signature(variant=0)
    with pytest.raises(ValueError, match="cannot change"):
        gdn.build_core(s)(view(s, fused=True))
    assert calls == []


@pytest.mark.parametrize("field", ["z", "norm_weight", "normalized_out"])
def test_fp32_norm_view_has_no_implicit_cast(monkeypatch, field):
    calls = install_core(monkeypatch)
    s = signature()
    io = view(s)
    with pytest.raises(ValueError, match="bound BF16"):
        gdn.build_core(s)(replace(io, **{field: getattr(io, field).float()}))
    assert calls == []


def test_fused_half_head_slice_is_refused(monkeypatch):
    calls = install_core(monkeypatch)
    s = signature()
    with pytest.raises(ValueError, match="bound BF16"):
        gdn.build_core(s)(replace(view(s), value_rows=64))
    assert calls == []


@pytest.mark.parametrize("rows", [4, 8, 12])
@pytest.mark.parametrize("hk,hv", [(4, 12), (16, 48)])
def test_norm_capability_exact_same_slot_and_no_extra_shape(rows, hk, hv):
    assert providers.supported(signature(rows, hk, hv, 0))
    assert providers.supported(signature(rows, hk, hv, 1))
    assert not providers.supported(signature(rows, hk, hv, 2))


@pytest.mark.parametrize("rows", [1, 2, 64])
def test_unapproved_rows_stay_stock(rows):
    assert not providers.supported(signature(rows=rows))


@pytest.mark.parametrize("field", ["qkv", "a", "b", "checkpoint", "raw_out"])
def test_common_activation_checkpoint_output_abi_rejects_fp32(monkeypatch, field):
    calls = install_core(monkeypatch)
    s = signature()
    io = view(s)
    with pytest.raises(ValueError, match="bound BF16"):
        gdn.build_core(s)(replace(io, **{field: getattr(io, field).float()}))
    assert calls == []
