# SPDX-License-Identifier: Apache-2.0
"""Original TP4 M16/front and M32/norm domains; CPU CUDA/raw ABI spies only."""

from dataclasses import replace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D
from test_qwen3_8_gdn_core_cache import PointerView, pointer_io, raw_module
from test_qwen3_8_gdn_core_norm import signature, view
from test_qwen3_8_gdn_front_abi import inputs, raw_double

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import gdn, gdn_verify_glue


@pytest.mark.parametrize("rows,variant", [(16, 0), (16, 1), (32, 1)])
def test_only_original_TP4_core_domain_has_an_offer(rows, variant):
    sig = signature(rows, 4, 12, variant)
    assert providers.supported(sig)
    assert providers.make_offer(sig).signature is sig


@pytest.mark.parametrize(
    "rows,hk,hv,variant",
    [
        (16, 16, 48, 0),
        (16, 16, 48, 1),
        (32, 16, 48, 0),
        (32, 16, 48, 1),
        (32, 4, 12, 0),
        (16, 4, 12, 2),
        (32, 4, 12, 2),
        (20, 4, 12, 1),
        (64, 4, 12, 1),
    ],
)
def test_unapproved_raw_or_TP1_geometry_has_no_offer(rows, hk, hv, variant):
    sig = signature(rows, hk, hv, variant)
    assert not providers.supported(sig) and providers.make_offer(sig) is None


@pytest.mark.parametrize(
    "operation,rows,hk,hv,variant",
    [
        (O.GDN_FRONT, 16, 16, 48, 0),
        (O.GDN_FRONT, 32, 4, 12, 0),
        (O.GDN_FRONT, 32, 16, 48, 0),
        (O.GDN_FRONT, 16, 4, 12, 1),
        (O.GDN_FRONT, 16, 4, 12, 2),
        (O.GDN_FRONT, 16, 4, 12, 3),
        (O.GDN_VERIFY_GLUE, 36, 4, 12, 0),
        (O.GDN_VERIFY_GLUE, 36, 16, 48, 0),
    ],
)
def test_no_M32_front_glue_or_changed_front_family(operation, rows, hk, hv, variant):
    sig = GdnSignature(
        operation,
        rows,
        hk,
        hv,
        128,
        128,
        4,
        D.BF16,
        gate_scalar_type=1 if operation == O.GDN_VERIFY_GLUE else 3,
        variant=variant,
    )
    assert not providers.supported(sig) and providers.make_offer(sig) is None


def test_glue_M16_remains_original_capability_without_a_new_kernel():
    for hk, hv in ((4, 12), (16, 48)):
        sig = GdnSignature(
            O.GDN_VERIFY_GLUE, 16, hk, hv, 128, 128, 4, D.BF16, gate_scalar_type=1
        )
        assert gdn_verify_glue.supported(sig)


@pytest.mark.parametrize("rows,variant", [(16, 0), (16, 1), (32, 1)])
@pytest.mark.parametrize("dt_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("flat", [False, True])
def test_source_raw_flags_grid_cache_and_current_activation(
    monkeypatch, rows, variant, dt_dtype, flat
):
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1")
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_M32_NORM", "1")
    ns, state = raw_module(monkeypatch, flat)
    sig = signature(rows, 4, 12, variant)
    io = pointer_io(
        replace(
            view(sig, bool(variant)),
            dt_bias=torch.empty((12,), dtype=dt_dtype, device="meta"),
        ),
        4,
    )
    run = gdn.build_core(sig)
    assert run(io) is None
    assert len(state.compiled) == len(state.launches) == 1
    code = state.compiled[0]
    assert f"#define FUSED_NORM {variant}" in code
    assert f"#define DT_BF16 {int(dt_dtype == torch.bfloat16)}" in code
    assert "#define NORM_F32 0" in code
    assert "#define FLAT_STORES" not in code  # numeric variant1 is not raw FLAT_STORES.
    assert state.launches[0][0].gridDimX == (rows // 4) * 12
    assert state.launches[0][0].blockDimX == 128
    assert state.launches[0][0].numAttrs == 1
    # New per-call Params encode the live activation addresses; no owner cache.
    other = pointer_io(
        replace(
            view(sig, bool(variant)),
            dt_bias=torch.empty((12,), dtype=dt_dtype, device="meta"),
        ),
        4,
    )
    assert other.qkv.data_ptr() != io.qkv.data_ptr()
    state.capturing = True
    assert run(other) is None
    assert len(state.compiled) == 1 and len(state.launches) == 2
    changed = replace(
        other,
        dt_bias=PointerView(
            torch.empty(
                (12,),
                dtype=torch.float32 if dt_dtype == torch.bfloat16 else torch.bfloat16,
                device="meta",
            )
        ),
    )
    with pytest.raises(RuntimeError, match="prewarm MK_GDN"):
        run(changed)
    assert len(state.compiled) == 1 and len(state.launches) == 2
    assert ns["_kernel"].cache_info().hits == 1


@pytest.mark.parametrize(
    "rows,variant,flag",
    [
        (16, 0, "SGLANG_Q38_MK_GDN_WIDE"),
        (16, 1, "SGLANG_Q38_MK_GDN_WIDE"),
        (32, 1, "SGLANG_Q38_MK_GDN_M32_NORM"),
    ],
)
def test_original_source_raw_flag_still_rejects_before_compile(
    monkeypatch, rows, variant, flag
):
    monkeypatch.setenv(flag, "0")
    _, state = raw_module(monkeypatch, False)
    sig = signature(rows, 4, 12, variant)
    io = pointer_io(view(sig, bool(variant)), 4)
    with pytest.raises(ValueError, match="MK_GDN supports"):
        gdn.build_core(sig)(io)
    assert not state.compiled and not state.launches


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("field", ["z", "norm_weight", "normalized_out"])
def test_BF16_norm_is_separate_from_WIDE_admission(monkeypatch, rows, field):
    _, state = raw_module(monkeypatch, False)
    sig = signature(rows, 4, 12, 1)
    io = view(sig)
    with pytest.raises(ValueError):
        gdn.build_core(sig)(replace(io, **{field: None}))
    with pytest.raises(ValueError):
        gdn.build_core(sig)(replace(io, **{field: getattr(io, field).float()}))
    assert not state.compiled and not state.launches


def test_original_M16_front_wrapper_requires_WIDE_without_numeric_widening(monkeypatch):
    module, state = raw_double(monkeypatch, "mk", 6)
    with pytest.raises(ValueError, match="MK_GDN_WIDE"):
        module.run(**inputs(16, 4, 12, False))
    assert not state.calls
