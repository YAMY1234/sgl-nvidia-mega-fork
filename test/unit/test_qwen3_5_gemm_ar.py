# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import plan
from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop
from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar

H, K = 4096, 1024


@pytest.fixture
def kernel(monkeypatch):
    import sgl_nvidia_mega.qwen3_5.sm103.hop as pkg

    calls = []
    fake = types.ModuleType("gemm_ar_kernel")
    fake.DEFAULT_OPTS = 7
    fake.workspace_rows = lambda m: (m + 7) // 8 * 8
    fake.unsupported = lambda m, k, h, w, sms, opts: "" if m <= 128 else f"{m} rows"
    fake.max_rows = lambda hidden, nranks, opts: min(128, 16 * nranks)
    fake.gemm_ar_norm = lambda *a, **kw: calls.append((a, kw)) or ("norm", "res")
    monkeypatch.setitem(sys.modules, f"{pkg.__name__}.gemm_ar_kernel", fake)
    monkeypatch.setattr(pkg, "gemm_ar_kernel", fake, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(multi_processor_count=152),
    )
    monkeypatch.setattr(gemm_ar, "_LOGGED", set())
    return calls


def _gar(max_rows=128):
    ws = SimpleNamespace(
        max_rows=max_rows, device="cpu", world=8, rank=3, table="the table"
    )
    return gemm_ar.GemmAllReduce(ws, (plan.GEMM_AR_MIN_ROWS, 128), 7)


def _aligned(shape, dtype):
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    buf = torch.zeros(nbytes + 128, dtype=torch.uint8)
    start = -buf.data_ptr() % 128
    return buf[start : start + nbytes].view(dtype).view(shape)


def _linear():
    return SimpleNamespace(
        weight=_aligned((H, K), torch.float8_e4m3fn).t(),
        input_scale=torch.ones(1),
        weight_scale=torch.ones(1),
    )


def _x(m, dtype=torch.float8_e4m3fn):
    x = _aligned((m, K), dtype)
    assert x.data_ptr() % 128 == 0
    return x


def test_a_deferred_projection_runs_in_the_next_all_reduce(kernel):
    gar = _gar()
    out = torch.empty((56, H), dtype=torch.bfloat16)
    x, lin = _x(56), _linear()
    assert gar.defer(out, x, lin, "o_proj")
    assert gar.pending_for(out) and not gar.pending_for(torch.empty((56, H)))
    with pytest.raises(RuntimeError, match="never consumed"):
        gar.defer(out, x, lin, "o_proj")
    with pytest.raises(RuntimeError, match="the deferred o_proj was never consumed"):
        gar.check_consumed()
    residual = torch.zeros((56, H), dtype=torch.bfloat16)
    assert gar.run(residual, torch.ones(H), 1e-6, "scale") == ("norm", "res")
    ((args, kw),) = kernel
    assert args[0] is x and args[7:10] == ("the table", 3, 8)
    assert kw == {"scale": "scale", "opts": 7}
    assert not gar.pending_for(out)
    gar.check_consumed()


@pytest.mark.parametrize(
    "x, why",
    [
        (lambda: _x(56, torch.bfloat16), "input torch.bfloat16"),
        (lambda: _x(200), "200 rows"),
    ],
)
def test_a_projection_the_kernel_refuses_stays_on_cublas_loudly(kernel, caplog, x, why):
    gar = _gar(max_rows=256)
    x = x()
    out = torch.empty((x.shape[0], H), dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger=gemm_ar.__name__):
        assert not gar.defer(out, x, _linear(), "out_proj")
        assert not gar.defer(out, x, _linear(), "out_proj")  # once per reason
    assert len(caplog.records) == 1
    assert "slot gemm_ar" in caplog.text and why in caplog.text
    gar.check_consumed()


def test_settle_runs_a_deferred_projection_on_cublas(kernel, monkeypatch, caplog):
    from sglang.srt.layers.quantization import fp8_utils

    seen = []
    monkeypatch.setattr(
        fp8_utils,
        "flashinfer_bmm_fp8",
        lambda x, w, sx, sw, dtype: seen.append((x, w.shape)) or torch.ones((56, H)),
        raising=False,
    )
    gar = _gar()
    out = torch.zeros((56, H), dtype=torch.bfloat16)
    x = _x(56)
    gar.settle()  # nothing deferred: nothing to do
    assert gar.defer(out, x, _linear(), "out_proj")
    with caplog.at_level(logging.WARNING, logger=gemm_ar.__name__):
        gar.settle()
    assert seen == [(x, (K, H))] and bool((out == 1).all())
    assert "runs on cuBLAS after all" in caplog.text
    gar.check_consumed()


def test_build_needs_a_width_in_the_slots_rows(kernel):
    """Outside its rows the slot builds nothing (no workspace, no collective)."""
    ws, gar, why = gemm_ar.build(1e-6, None, "cpu", (7, 14, 28, 224))
    assert ws is gar is None and "within 33-128 rows" in why


@pytest.mark.parametrize("world, hi", [(4, 64), (8, 128)])
def test_build_covers_only_the_rows_the_owner_warps_take(
    kernel, monkeypatch, world, hi
):
    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5.sm103.hop import twoshot_ar

    monkeypatch.setattr(dist, "get_world_size", lambda group: world)
    monkeypatch.setattr(
        twoshot_ar,
        "TwoShotWorkspace",
        lambda **kw: SimpleNamespace(**kw, world=world),
    )
    ws, gar, why = gemm_ar.build(1e-6, None, "cpu", (7, 35, 56, 63, 70, 98, 126))
    assert why == "" and (gar.lo, gar.hi) == (plan.GEMM_AR_MIN_ROWS, hi)
    assert ws.max_rows == (64 if world == 4 else 128)
    assert gar.covers(63) and gar.covers(70) == (world == 8)
    # no width at or below the bound: nothing is built
    ws, gar, why = gemm_ar.build(1e-6, None, "cpu", (70, 98))
    assert gar is None if world == 4 else gar is not None
    if world == 4:
        assert "within 33-64 rows at world 4" in why


def test_the_mixer_defers_only_where_the_plan_runs_gemm_ar_and_the_tail_takes_it(
    kernel, monkeypatch
):
    gar = _gar()
    tail = SimpleNamespace(gemm_ar=gar)
    takes = []
    monkeypatch.setattr(
        boundary_hop,
        "takes_moe_input",
        lambda engine, layer, st, rows: takes.append((engine, rows)) or rows != 63,
    )
    st = SimpleNamespace(
        slots=plan.Slots(bt_tail=True, gemm_ar=True),
        bound={"boundary": SimpleNamespace(bt=tail)},
    )
    assert boundary_hop.gemm_ar("layer", st, 56) is gar
    assert boundary_hop.gemm_ar("layer", st, 63) is None  # the tail declines these rows
    assert boundary_hop.gemm_ar("layer", st, 200) is None  # past its rows
    st.slots = plan.Slots(bt_tail=True)
    assert boundary_hop.gemm_ar("layer", st, 56) is None
    assert takes == [(tail, 56), (tail, 63), (tail, 200)]


def test_a_width_the_build_does_not_cover_warns_once(kernel, monkeypatch, caplog):
    ws = SimpleNamespace(max_rows=64, device="cpu", world=4, rank=1, table="the table")
    gar = gemm_ar.GemmAllReduce(ws, (plan.GEMM_AR_MIN_ROWS, 64), 7)
    monkeypatch.setattr(boundary_hop, "_ENGAGED", set())
    monkeypatch.setattr(
        boundary_hop, "takes_moe_input", lambda engine, layer, st, rows: rows != 98
    )
    st = SimpleNamespace(
        slots=plan.Slots(bt_tail=True, gemm_ar=True),
        bound={"boundary": SimpleNamespace(bt=SimpleNamespace(gemm_ar=gar))},
    )
    with caplog.at_level(logging.WARNING, logger=boundary_hop.__name__):
        assert boundary_hop.gemm_ar("layer", st, 70) is None
        assert boundary_hop.gemm_ar("layer", st, 70) is None
        assert boundary_hop.gemm_ar("layer", st, 98) is None  # not taken: no WARNING
        assert boundary_hop.gemm_ar("layer", st, 63) is gar
    warned = [
        r.getMessage() for r in caplog.records if "stays on cuBLAS" in r.getMessage()
    ]
    assert warned == [
        "NVIDIA megakernel: Qwen3.5 slot gemm_ar covers 33-64 rows; the projection "
        "at 70 rows stays on cuBLAS"
    ]


def _bt_tail():
    from sgl_nvidia_mega.qwen3_5.boundary.bt_tail import BTTail
    from sgl_nvidia_mega.qwen3_5.sm103.hop import bt_tail as kernels

    with patch.object(kernels, "kernels", lambda *a: {}):
        return BTTail(
            service=SimpleNamespace(_workspace=None),
            eps=1e-6,
            device="cpu",
            min_rows=plan.BT_TAIL_MIN_ROWS,
            max_rows=plan.BT_TAIL_MAX_ROWS,
        )


def test_the_bt_tails_moe_input_runs_the_deferred_projection(kernel):
    tail = _bt_tail()
    tail.gemm_ar = _gar()
    out = torch.empty((56, H), dtype=torch.bfloat16)
    assert tail.gemm_ar.defer(out, _x(56), _linear(), "o_proj")
    got = tail.all_reduce_residual_rms_norm(
        local_contribution=out,
        residual=torch.zeros((56, H), dtype=torch.bfloat16),
        gamma=torch.ones(H),
        fp8_scale="s",
    )
    assert got == ("norm", "res") and len(kernel) == 1
    # a deferral that reaches another all-reduce, or the finalize point, raises
    assert tail.gemm_ar.defer(out, _x(56), _linear(), "o_proj")
    with pytest.raises(RuntimeError, match="never consumed"):
        tail.all_reduce_residual_rms_norm(
            local_contribution=torch.empty((56, H)), residual="r", gamma="g"
        )
    with pytest.raises(RuntimeError, match="never consumed"):
        tail.moe_finalize_all_reduce_rms_norm(
            routed_output="rows",
            expert_weights="w",
            permuted_indices="p",
            gated_shared_output="s",
            residual="r",
            gamma="g",
        )


def test_the_moe_input_settles_a_deferral_its_all_reduce_declines(kernel, monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import boundary

    settled = []
    tail = SimpleNamespace(gemm_ar=SimpleNamespace(settle=lambda: settled.append(1)))
    monkeypatch.setattr(boundary_hop, "moe_input", lambda engine, *a: None)
    layer = SimpleNamespace(
        attn_boundary=SimpleNamespace(finish=lambda part, fb: "handle"),
        ffn_boundary=SimpleNamespace(prepare=lambda h, fb: "sglang MoE input"),
    )
    st = SimpleNamespace(
        fb="fb",
        slots=plan.Slots(bt_tail=True, gemm_ar=True),
        bound={"boundary": boundary_hop.Bound(bt=tail)},
    )
    assert boundary.post_attn(layer, "part", st) == "sglang MoE input"
    assert settled == [1]
    st.slots = plan.Slots(bt_tail=True)
    assert boundary.post_attn(layer, "part", st) == "sglang MoE input"
    assert settled == [1]
