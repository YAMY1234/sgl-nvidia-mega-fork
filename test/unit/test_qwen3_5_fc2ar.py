# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import plan
from sgl_nvidia_mega.qwen3_5.moe import body, fc2ar


@pytest.fixture
def clean(monkeypatch):
    prev = fc2ar._ACTIVE[0]
    monkeypatch.setattr(fc2ar, "_FELL", set())
    monkeypatch.setattr(fc2ar, "_LOGGED", set())
    # the CPU device's buffer, as a build makes it
    monkeypatch.setattr(fc2ar, "_BUFFERS", {None: _sentinel_rows(64)})
    yield monkeypatch
    fc2ar._ACTIVE[0] = prev
    fc2ar._ARMED[0] = 0
    fc2ar._HIT[0] = False


def _sentinel_rows(rows):
    buf = torch.empty((rows, fc2ar.HIDDEN), dtype=torch.bfloat16)
    buf.view(torch.int32).fill_(fc2ar.SENTINEL)
    return buf


class _Tail:
    def __init__(self, lam, routes=range(29, 129)):
        self.fc2ar, self.routes = lam, routes

    def supports(self, m):
        return m in self.routes


def _st(tail, **slots):
    return SimpleNamespace(
        slots=plan.Slots(**{"bt_tail": True, "fc2ar": True, **slots}),
        bound={"boundary": SimpleNamespace(bt=tail)},
    )


def _lam(lo=plan.FC2AR_MIN_ROWS, hi=plan.FC2AR_MAX_ROWS):
    lam = fc2ar.LamportAR.__new__(fc2ar.LamportAR)
    lam.lo, lam.hi = lo, hi
    return lam


def _draft():
    return SimpleNamespace(
        slots=plan.Slots(route=True), bound={"moe": object()}, scratch={}
    )


def test_scope_arms_only_the_steps_the_plan_gives_slot_fc2ar(clean):
    lam = _lam()
    fc2ar._ACTIVE[0] = lam
    tail = _Tail(lam)
    assert fc2ar.scope(98, _st(tail)).rows == 98
    assert fc2ar.scope(98, _st(tail, fc2ar=False)).rows == 0
    assert fc2ar.scope(98, _draft()).rows == 0
    with fc2ar.scope(98, _st(tail)):
        assert fc2ar.armed() and fc2ar.armed_at(98) and not fc2ar.armed_at(91)
        with fc2ar.scope(98, _draft()):
            assert not fc2ar.armed()
        assert fc2ar.armed()
    assert not fc2ar.armed()


def test_engine_options_of_an_armed_call(clean):
    lam = _lam(29, 128)
    fc2ar._ACTIVE[0] = lam
    tail = _Tail(lam)
    base = {"early_unit": 1}
    assert fc2ar.engine_opts(base, 98) is base  # not armed
    with fc2ar.scope(98, _st(tail)):
        assert fc2ar.engine_opts(base, 98) == {
            "early_unit": 1,
            "early_trigger": 2,
            "grid_ctas": fc2ar.ENGINE_GRID,
        }
        assert fc2ar.engine_opts(base, 56) is base  # another call's rows
        # an explicit engine grid wins
        assert fc2ar.engine_opts({"grid_ctas": 144}, 98)["grid_ctas"] == 144


def test_a_lamport_handoff_outside_its_all_reduce_raises(clean):
    fc2ar._ACTIVE[0] = _lam()
    clean.setattr(fc2ar, "pending", lambda st: fc2ar._ACTIVE[0])
    with pytest.raises(RuntimeError, match="Lamport FC2 rows"):
        fc2ar.refuse_unconsumed(SimpleNamespace(), "final norm")
    clean.setattr(fc2ar, "pending", lambda st: None)
    fc2ar.refuse_unconsumed(SimpleNamespace(), "final norm")


def test_a_step_of_slot_fc2ar_that_stays_on_bt_says_so_once(clean, caplog):
    lam = _lam()
    fc2ar._ACTIVE[0] = lam
    tail = _Tail(lam)
    with caplog.at_level(logging.WARNING, logger=fc2ar.__name__):
        fc2ar.scope(84, _st(_Tail(lam, routes=())))
        fc2ar.scope(84, _st(_Tail(lam, routes=())))  # once per width
        fc2ar.scope(112, _draft())  # not a step of the slot: silent
        fc2ar.scope(98, _st(_Tail(None)))  # another tail's all-reduce
        with fc2ar.scope(70, _st(tail)):
            pass  # armed, but no engine call wrote the Lamport rows
        with fc2ar.scope(56, _st(tail)):
            out = torch.empty((32, fc2ar.HIDDEN), dtype=torch.bfloat16)
            got = fc2ar.engine_out(out, 56, 8)  # armed and written: silent
            assert fc2ar.is_lamport(got) and not fc2ar.is_lamport(out)
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 3, lines
    assert all("slot fc2ar at" in s and "stays on BT" in s for s in lines)
    assert "84 rows" in lines[0] and "no BT route at these rows" in lines[0]
    assert "98 rows" in lines[1] and "holds no all-reduce" in lines[1]
    assert "70 rows" in lines[2] and "no MoE engine call" in lines[2]


def test_build_needs_a_width_in_the_slots_rows(clean):
    """Outside its rows the slot builds nothing (no workspace, no collective)."""
    got, why = fc2ar.build(1e-6, None, "cpu", (7, 14, 28, 224), _experts())
    assert got is None and "within 35-128 rows" in why


def _built(clean, widths):
    import sgl_nvidia_mega.qwen3_5.sm103.hop as pkg

    fake = types.ModuleType("slot_ar")
    fake.SlotAllReduce = lambda **kw: SimpleNamespace(world=8, **kw)
    clean.setitem(sys.modules, f"{pkg.__name__}.slot_ar", fake)
    clean.setattr(pkg, "slot_ar", fake, raising=False)
    clean.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(multi_processor_count=152),
    )
    clean.setattr(fc2ar, "_BUFFERS", {})
    return fc2ar.build(1e-6, None, "cpu", widths, _experts())


def test_build_makes_the_lamport_buffer_every_tile_n_shares(clean):
    lam, why = _built(clean, (7, 35, 70, 98, 224))
    assert why == "" and lam.hi == 98
    (buf,) = fc2ar._BUFFERS.values()
    assert buf.shape == (17152, fc2ar.HIDDEN) and buf.dtype == torch.bfloat16
    assert bool((buf.view(torch.int32) == fc2ar.SENTINEL).all())
    assert max(body.TILES) == 128
    tail = _Tail(lam)
    clean.setattr(torch, "empty", None)
    for m, tile in ((98, 8), (98, 64), (35, 16)):
        out = torch.zeros((body.engine_rows(m, 10, 128, tile), fc2ar.HIDDEN))
        with fc2ar.scope(m, _st(tail)):
            got = fc2ar.engine_out(out, m, tile)
        assert got.data_ptr() == buf.data_ptr() and got.shape == out.shape
    assert list(fc2ar._BUFFERS.values()) == [buf]


def test_an_armed_call_the_build_buffer_cannot_hold_raises(clean):
    lam = _lam()
    fc2ar._ACTIVE[0] = lam
    with fc2ar.scope(56, _st(_Tail(lam))):
        with pytest.raises(RuntimeError, match="made at build"):
            fc2ar.engine_out(torch.empty((65, fc2ar.HIDDEN)), 56, 8)
        clean.setattr(fc2ar, "_BUFFERS", {})
        with pytest.raises(RuntimeError, match="made at build"):
            fc2ar.engine_out(torch.empty((32, fc2ar.HIDDEN)), 56, 8)


def _experts():
    u8 = torch.zeros(8, dtype=torch.uint8)
    f32 = torch.ones(2)
    return SimpleNamespace(
        top_k=10,
        num_local_experts=128,
        layer_id=3,
        w13_weight=SimpleNamespace(data=u8),
        w13_weight_scale=SimpleNamespace(data=u8),
        w2_weight=SimpleNamespace(data=u8),
        w2_weight_scale=SimpleNamespace(data=u8),
        g1_scale_c=SimpleNamespace(data=f32),
        g1_alphas=SimpleNamespace(data=f32),
        g2_alphas=SimpleNamespace(data=f32),
    )


def _tables():
    i32 = torch.zeros(8, dtype=torch.int32)
    return SimpleNamespace(
        tile=16,
        permuted_to_token=i32,
        cta_expert=i32,
        cta_row_limit=i32,
        live_ctas=i32,
        ids=i32,
        weights=torch.zeros(8, dtype=torch.bfloat16),
        expanded_to_permuted=i32,
        total_padded=i32,
        local_offset=0,
    )


def _engine_call(clean, module, run, m, armed_rows=0):
    import sgl_nvidia_mega.qwen3_5.sm103.moe as pkg

    seen = {}
    fake = types.ModuleType(module)
    fake.q35_moe_engine = lambda *args, **kw: seen.update(out=args[15], kw=kw)
    clean.setitem(sys.modules, f"{pkg.__name__}.{module}", fake)
    clean.setattr(pkg, module, fake, raising=False)
    body.configure(True)
    plain = torch.empty((64, fc2ar.HIDDEN), dtype=torch.bfloat16)
    clean.setattr(
        body, "_engine_buffers", lambda *a: (torch.empty(1), torch.empty(1), plain)
    )
    clean.setattr(body, "_engine_state", lambda d: torch.zeros(4, dtype=torch.int32))
    clean.setattr(body, "_route_p2tp", lambda t, d: torch.empty(1))
    clean.setattr(body, "ring_opts", lambda e, m, opts, d: (opts, None))
    x_fp4 = torch.zeros((m, 2048), dtype=torch.uint8)
    x_sf = torch.zeros((m, 256), dtype=torch.uint8)
    args = (SimpleNamespace(tables=_tables()), _experts(), x_fp4, x_sf)
    if run is body.run_engine_a:
        args += (torch.zeros(8, dtype=torch.int32),)
    lam = _lam(29, 128)
    fc2ar._ACTIVE[0] = lam
    if armed_rows:
        with fc2ar.scope(armed_rows, _st(_Tail(lam))):
            res = run(*args)
    else:
        res = run(*args)
    assert res[0] is seen["out"]
    return seen["kw"], seen["out"]


def test_the_engine_writes_lamport_rows_and_releases_early_only_when_armed(clean):
    kw, out = _engine_call(clean, "q35_moe_engine", body.run_engine, 56)
    assert "early_trigger" not in kw and "grid_ctas" not in kw
    assert "cand" not in kw  # the route front's tables (prerouted 0)
    assert not fc2ar.is_lamport(out)
    kw, out = _engine_call(clean, "q35_moe_engine", body.run_engine, 56, armed_rows=63)
    assert "early_trigger" not in kw and not fc2ar.is_lamport(out)
    kw, out = _engine_call(clean, "q35_moe_engine", body.run_engine, 56, armed_rows=56)
    assert kw["early_trigger"] == 2 and kw["grid_ctas"] == fc2ar.ENGINE_GRID
    assert kw["early_unit"] == body.ENGINE_OPTIONS["early_unit"]
    assert fc2ar.is_lamport(out)


def test_the_routing_engine_releases_the_armed_all_reduce_after_its_routing(
    clean, caplog
):
    assert body.ROUTED_TRIGGER == 3
    kw, out = _engine_call(clean, "q35_moe_engine", body.run_engine_a, 56)
    assert "early_trigger" not in kw and "grid_ctas" not in kw
    assert kw["cand"] is not None  # the engine routes the call (prerouted 1)
    assert not fc2ar.is_lamport(out)
    with caplog.at_level(logging.INFO):
        kw, out = _engine_call(
            clean, "q35_moe_engine", body.run_engine_a, 56, armed_rows=56
        )
    assert kw["early_trigger"] == 3 and kw["grid_ctas"] == fc2ar.ENGINE_GRID
    assert fc2ar.is_lamport(out)
    assert "no MoE engine call" not in caplog.text
    assert (
        "writes Lamport FC2 rows at 56 rows (tile_n 16, early_trigger 3)" in caplog.text
    )
    # at moe_engine_route's first rows the armed call runs 3 as well
    kw, _ = _engine_call(clean, "q35_moe_engine", body.run_engine_a, 33, armed_rows=33)
    assert kw["early_trigger"] == 3


def test_the_routing_engines_route_delay_is_a_test_hook_only(clean):
    assert body.TEST_ROUTE_DELAY_NS == 0
    kw, _ = _engine_call(clean, "q35_moe_engine", body.run_engine_a, 56, armed_rows=56)
    assert "test_route_delay_ns" not in kw
    clean.setattr(body, "TEST_ROUTE_DELAY_NS", 20_000)
    kw, _ = _engine_call(clean, "q35_moe_engine", body.run_engine_a, 56, armed_rows=56)
    assert kw["test_route_delay_ns"] == 20_000 and kw["early_trigger"] == 3


# ------------------------------------------------ the boundary's points


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


def test_the_bt_tails_finalize_point_gives_lamport_rows_to_the_slot_all_reduce(clean):
    tail = _bt_tail()
    calls = []

    class _AR:
        def moe_finalize_all_reduce_rms_norm(self, **kw):
            calls.append(kw)
            return ["norm", "residual"]

    tail.fc2ar = _AR()
    clean.setattr(fc2ar, "is_lamport", lambda t: t == "lamport rows")
    kw = dict(
        expert_weights="w",
        permuted_indices="p",
        gated_shared_output="s",
        residual="r",
        gamma="g",
        fp8_scale="f",
    )
    got = tail.moe_finalize_all_reduce_rms_norm(routed_output="lamport rows", **kw)
    assert got == ("norm", "residual")
    assert calls == [dict(routed_output="lamport rows", **kw)]
    # plain rows take FlashInfer's BT route
    stubs = {
        "cutlass": SimpleNamespace(Int32=int),
        "flashinfer.comm.mnnvl_cutedsl.runtime": SimpleNamespace(
            current_cu_stream=None, to_cute=None, to_cute_dynamic=None
        ),
    }
    with (
        patch.dict(sys.modules, stubs),
        patch.object(tail, "_kernel", side_effect=LookupError("BT")),
    ):
        with pytest.raises(LookupError, match="BT"):
            tail.moe_finalize_all_reduce_rms_norm(
                routed_output="plain", **{**kw, "permuted_indices": torch.zeros(2, 10)}
            )
    assert len(calls) == 1


def test_the_final_norm_gives_a_lamport_handoff_to_its_all_reduce(clean):
    from sgl_nvidia_mega.qwen3_5 import boundary
    from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop

    lam = _lam()
    fc2ar._ACTIVE[0] = lam
    clean.setattr(fc2ar, "pending", lambda st: lam)
    clean.setattr(boundary_hop, "final_norm", lambda engine, h, st: ("final", engine))
    st = SimpleNamespace(fb="fb", slots=plan.Slots(bt_tail=True, fc2ar=True))
    assert boundary.close_stream(st, "h") == ("final", lam)
    # declined by the all-reduce: no other reader may take it
    clean.setattr(boundary_hop, "final_norm", lambda engine, h, st: None)
    with pytest.raises(RuntimeError, match="final norm holds Lamport FC2 rows"):
        boundary.close_stream(st, "h")


def test_a_layer_input_off_the_bt_tail_refuses_a_lamport_handoff(clean):
    from sgl_nvidia_mega.qwen3_5 import boundary

    fc2ar._ACTIVE[0] = _lam()
    clean.setattr(fc2ar, "pending", lambda st: fc2ar._ACTIVE[0])
    layer = SimpleNamespace(
        layer_id=5,
        attn_boundary=SimpleNamespace(prepare=lambda h, fb, capture_gathered: "sglang"),
    )
    st = SimpleNamespace(fb="fb", slots=plan.Slots(), bound={})
    with pytest.raises(RuntimeError, match="layer 5's input holds Lamport FC2 rows"):
        boundary.pre(layer, "h", st)
    clean.setattr(fc2ar, "pending", lambda st: None)
    assert boundary.pre(layer, "h", st) == "sglang"
