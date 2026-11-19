# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import logging
import sys
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import lowm
from sgl_nvidia_mega.qwen3_5 import moe as moe_pkg
from sgl_nvidia_mega.qwen3_5 import plan, runtime, topology
from sgl_nvidia_mega.qwen3_5.moe import body, route

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
TEP8_ROWS = frozenset(runtime.MEASURED_TEP8)
MOE_ROWS = frozenset(moe_pkg.TEP8_ROWS)
SE_ROWS = frozenset({"lowm_se_tep8", "lowm_se_wide", "lowm_se_coresident"})


@pytest.fixture(autouse=True)
def rows_off(monkeypatch):
    monkeypatch.setattr(body, "_ENGINE_STATE", {})
    body.configure(False)
    route.configure()
    lowm._configure_tep8_se((), False)
    yield
    body.configure(False)
    route.configure()
    lowm._configure_tep8_se((), False)


def _experts(local=128, layer_id=5):
    return SimpleNamespace(num_local_experts=local, top_k=10, layer_id=layer_id)


def test_the_rows_are_tep8_rows_of_the_measured_plan_only():
    assert TEP8_ROWS == MOE_ROWS | SE_ROWS
    for key, rows in runtime.MEASURED_PLAN.items():
        assert (TEP8_ROWS <= rows) is (key == ("tep8", topology.GB300)), key
    # each is a variant (no width of its own), labelled once
    assert TEP8_ROWS <= set(runtime.OPT_IN_VARIANTS)
    labelled = set(runtime.MEASURED_BITWISE) | set(runtime.NUMERICS_CHANGE)
    assert TEP8_ROWS <= labelled
    numeric = {"route_big_gemm", "route_big_mid", "lowm_se_tep8", "lowm_se_wide"}
    assert TEP8_ROWS & set(runtime.NUMERICS_CHANGE) == numeric
    from sgl_nvidia_mega.qwen3_5 import spec

    assert numeric <= set(spec.NUMERIC_TOLERANCES)


@pytest.mark.parametrize("name", sorted(topology.TOPOLOGIES))
def test_the_rows_run_at_tep8_only(name):
    shape = {
        "tp2": (2, 1, 1),
        "tp4": (4, 1, 1),
        "tp8": (8, 1, 1),
        "tep8": (8, 1, 4),
        "dep8": (8, 8, 8),
    }[name]
    current = topology.Topology(*shape)
    assert current.name == name
    kept = topology.select(frozenset(TEP8_ROWS), current)
    assert kept == (TEP8_ROWS if name == "tep8" else frozenset())


@pytest.mark.parametrize("switch, slots", [(None, None), ("0", None), ("1", "hop")])
def test_without_the_measured_stack_no_row_is_requested(monkeypatch, switch, slots):
    """The runtime switch off, or an explicit slot list: none of the rows is requested."""
    for env, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)
    assert not runtime._requested_slots() & TEP8_ROWS


def test_the_measured_stack_requests_them_and_other_keys_drop_them(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV, "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    assert TEP8_ROWS <= runtime._requested_slots()
    for key in (("tep8", topology.Sku(10, 3, 148)), ("tp8", topology.GB300)):
        assert not any(runtime._measured_rows_on(key, row) for row in TEP8_ROWS)
    assert all(
        runtime._measured_rows_on(("tep8", topology.GB300), r) for r in TEP8_ROWS
    )


@pytest.mark.parametrize(
    "drop, needs",
    [
        ("moe_engine", "moe_ring"),
        ("route_lean", "moe_ring"),
        ("moe_engine", "moe_engine_route"),
        ("route_big_gemm", "moe_engine_route"),
        ("route_topk", "moe_engine"),
        ("route", "route_big_mid"),
        ("lowm_se", "lowm_se_tep8"),
        ("lowm_se_tep8", "lowm_se_wide"),
        ("lowm_se_tep8", "lowm_se_coresident"),
    ],
)
def test_each_row_needs_the_rows_it_runs_on(monkeypatch, drop, needs):
    full = (
        frozenset(runtime.MEASURED_PLAN[("tep8", topology.GB300)]) - runtime.OPTION_ROWS
    )
    monkeypatch.setattr(runtime, "SLOTS", full)
    assert runtime.slot_reasons() == []
    monkeypatch.setattr(runtime, "SLOTS", full - {drop})
    assert any(f"'{needs}'" in r for r in runtime.slot_reasons()), (
        runtime.slot_reasons()
    )


# ------------------------------------------------------------------ the MoE build


def test_every_row_is_off_until_a_build_sets_it():
    x = torch.zeros((56, 4096), dtype=torch.bfloat16)
    w = torch.zeros((512, 4096), dtype=torch.bfloat16)
    assert not body.engine_serves(56)
    assert route.big_router_gemm(x, w) is None
    assert route.route_lean(112, 128, 0) is None
    assert route.engine_route(SimpleNamespace(), x, None, None) is None
    assert lowm.tep8_se_config()["rows"] == ()


@pytest.mark.parametrize(
    "slots, engine, ring, gemm, lean, engine_route",
    [
        (MOE_ROWS, True, True, True, True, True),
        ({"moe_engine"}, True, False, False, False, False),
        ({"moe_engine", "moe_ring"}, True, False, False, False, False),
        ({"moe_engine", "moe_ring", "route_lean"}, True, True, False, True, False),
        ({"moe_engine", "moe_engine_route"}, True, False, False, False, False),
        ({"route_big_gemm", "moe_engine_route"}, False, False, True, False, False),
        ({"route_big_gemm", "route_lean"}, False, False, True, True, False),
        (frozenset(), False, False, False, False, False),
    ],
)
def test_the_build_sets_each_row_from_its_slots(
    monkeypatch, slots, engine, ring, gemm, lean, engine_route
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots) | {"route", "route_topk"})
    monkeypatch.setattr(body, "engine_unsupported", lambda experts: "")
    bound = moe_pkg._build_tep8_rows(
        SimpleNamespace(experts=_experts()), route.Bound(pre=True)
    )
    assert body._ENGINE["on"] is engine and body._ENGINE["ring"] is ring
    assert route._BIG_GEMM["on"] is gemm
    assert route._ROUTE_LEAN["on"] is lean
    assert route._ENGINE_ROUTE["on"] is engine_route
    assert bound.big_mid is ("route_big_mid" in slots)


def test_an_engine_refusal_turns_its_rows_off_and_says_why(monkeypatch, caplog):
    monkeypatch.setattr(runtime, "SLOTS", MOE_ROWS | {"route", "route_topk"})
    monkeypatch.setattr(
        body, "engine_unsupported", lambda experts: "SwiGLU experts only"
    )
    with caplog.at_level(logging.INFO):
        moe_pkg._build_tep8_rows(SimpleNamespace(experts=_experts()), route.Bound())
    assert not body._ENGINE["on"] and not body._ENGINE["ring"]
    assert not route._ENGINE_ROUTE["on"]
    # route-big's own rows do not need the engine
    assert route._BIG_GEMM["on"] and route._ROUTE_LEAN["on"]
    assert (
        "['moe_engine', 'moe_engine_route', 'moe_ring'] use stock (SwiGLU"
        in caplog.text
    )


def test_a_later_build_keeps_the_engine_state_and_records(monkeypatch):
    """Captured graphs keep the addresses of the call state and ring records."""
    from sgl_nvidia_mega.qwen3_5.runtime import persistent

    made = []

    def new(device, *rest):
        made.append(device)
        return torch.zeros(4, dtype=torch.int32)

    _fake_module(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_5.sm103.moe.q35_moe_engine",
        new_state=new,
        new_record=new,
    )
    monkeypatch.setattr(
        persistent, "persistent", lambda device: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    dev = torch.device("cpu")
    body.configure(True, ring=True)
    state, rec = body._engine_state(dev), body._ring_records(dev)
    body.configure(False)
    body.configure(True, ring=True)
    assert body._engine_state(dev) is state and body._ring_records(dev) is rec
    assert len(made) == 2


def test_a_build_without_the_tep8_body_turns_every_row_off(monkeypatch):
    """A later build (or one without route / route_topk) leaves nothing of an earlier one on."""
    body.configure(True, ring=True)
    route.configure(gemm=True, lean=True, engine_route=True)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(MOE_ROWS))
    moe_pkg.build(SimpleNamespace(), SimpleNamespace(widths=(7, 14)))
    assert not body._ENGINE["on"] and not body._ENGINE["ring"]
    assert not any(
        d["on"] for d in (route._BIG_GEMM, route._ROUTE_LEAN, route._ENGINE_ROUTE)
    )


def test_engine_unsupported_names_the_shape_it_refuses():
    def experts(local=128, hidden=4096, inter=512, act="silu"):
        return SimpleNamespace(
            moe_runner_config=SimpleNamespace(activation=act, is_gated=True),
            w13_weight=torch.empty((local, 2 * inter, hidden // 2), dtype=torch.uint8),
            w2_weight=torch.empty((local, hidden, inter // 2), dtype=torch.uint8),
            intermediate_size_per_partition=inter,
        )

    assert "128 local experts" in body.engine_unsupported(experts(local=129))
    assert "SwiGLU" in body.engine_unsupported(experts(act="gelu"))
    assert "multiples of 256" in body.engine_unsupported(experts(inter=384))


def test_the_engine_runs_the_plans_options():
    assert body.ENGINE_OPTIONS == {"early_unit": 1}
    assert body.RING_ROWS == (64, 128) and body.RING_STAGES == 3
    body.configure(True)
    assert body.engine_serves(1) and body.engine_serves(896)
    body.configure(False, ring=True)
    assert not body.engine_serves(64) and not body._ENGINE["ring"]


@pytest.mark.parametrize("m", [63, 64, 98, 128, 129])
def test_the_ring_fills_at_its_rows_behind_the_early_release(monkeypatch, m):
    records = {5: "record of layer 5"}
    monkeypatch.setattr(body, "_ring_records", lambda device: records)
    monkeypatch.setattr(body, "front_releases_early", lambda experts, m, device: True)
    body.configure(True, ring=True)
    opts, rec = body.ring_opts(_experts(), m, dict(body.ENGINE_OPTIONS), None)
    if 64 <= m <= 128:
        assert opts == {"early_unit": 1, "ring": 3, "q0": 1} and rec == records[5]
    else:
        assert opts == {"early_unit": 1} and rec is None
    body.configure(True, ring=False)
    assert body.ring_opts(_experts(), m, {}, None) == ({}, None)


@pytest.mark.parametrize(
    "layer_id, early, why",
    [(200, True, "layer id 200 outside"), (5, False, "releases the engine at its end")],
)
def test_the_ring_says_why_it_is_off(monkeypatch, caplog, layer_id, early, why):
    monkeypatch.setattr(body, "_ring_records", lambda device: {5: "rec"})
    monkeypatch.setattr(body, "front_releases_early", lambda experts, m, device: early)
    body.configure(True, ring=True)
    with caplog.at_level(logging.INFO):
        assert body.ring_opts(_experts(layer_id=layer_id), 98, {}, None) == ({}, None)
        body.ring_opts(_experts(layer_id=layer_id), 98, {}, None)
    lines = [
        r.getMessage() for r in caplog.records if "moe_ring is off" in r.getMessage()
    ]
    assert len(lines) == 1 and why in lines[0], lines


def _fake_module(monkeypatch, name, **attrs):
    import importlib

    package, _, leaf = name.rpartition(".")
    module = SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.import_module(package), leaf, module, raising=False)


def test_the_engine_state_and_records_come_from_the_private_pool(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.runtime import persistent

    entered, made = [], []

    @contextlib.contextmanager
    def scope(device):
        entered.append(device)
        yield

    def new(device, *rest):
        made.append(list(entered))
        return torch.zeros(4, dtype=torch.int32)

    _fake_module(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_5.sm103.moe.q35_moe_engine",
        new_state=new,
        new_record=new,
    )
    monkeypatch.setattr(persistent, "persistent", scope)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    dev = torch.device("cpu")
    s1, s2 = body._engine_state(dev), body._engine_state(dev)
    r1, r2 = body._ring_records(dev), body._ring_records(dev)
    assert s1 is s2 and r1 is r2
    assert made == [[dev], [dev, dev]] and entered == [dev, dev]


def test_the_engine_state_is_not_created_under_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA-graph capture"):
        body._engine_state(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="before CUDA-graph capture"):
        body._ring_records(torch.device("cpu"))


def test_route_lean_takes_only_the_variants_its_build_compiled(monkeypatch, caplog):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    compiled = []
    monkeypatch.setattr(
        router, "build_big", lambda dev, local, variants=(): compiled.append(variants)
    )
    route.configure(lean=True)
    with caplog.at_level(logging.INFO):
        assert route.route_lean(112, 128, 0) is None
    assert "not compiled at build" in caplog.text
    route.route_lean_build(0, 128, range(17, 449))
    assert compiled and all(len(v) == 4 for v in compiled[0])
    want = dict(lean=True, et=False, threads=512, ctas=None)
    assert route.route_lean(112, 128, 0) == want
    assert route.route_lean(112, 64, 0) is None
    assert route.route_lean(112, 128, 1) is None
    body.configure(True)  # the engine serves the width: released at route-big's start
    want = dict(lean=True, et=True, threads=512, ctas=None)
    assert route.route_lean(112, 128, 0) == want
    want = dict(lean=True, et=True, threads=512, ctas=64)
    assert route.route_lean(259, 128, 0) == want
    assert route.route_lean(449, 128, 0) is None  # past its rows: silent
    dev = torch.device("cuda", 0)
    assert body.front_releases_early(_experts(), 112, dev)
    assert not body.front_releases_early(_experts(local=64), 112, dev)
    route.configure()
    assert route.route_lean(112, 128, 0) is None
    assert not body.front_releases_early(_experts(), 112, dev)


def test_route_lean_builds_nothing_when_off(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    compiled = []
    monkeypatch.setattr(
        router,
        "build_big",
        lambda dev, local, variants=(): compiled.append(set(variants)),
    )
    route.route_lean_build(0, 128, range(17, 449))
    assert compiled == [set()]


@pytest.mark.parametrize("rows, min_rows", [(16, 17), (32, 33), (193, 33)])
def test_the_router_gemm_stays_out_of_rows_it_does_not_serve(
    monkeypatch, rows, min_rows
):
    def no_compile(*a, **k):
        raise AssertionError("compiled outside the slot's rows")

    monkeypatch.setattr(route, "_big_gemm_compile", no_compile)
    route.configure(gemm=True)
    x = torch.zeros((rows, 4096), dtype=torch.bfloat16)
    w = torch.zeros((512, 4096), dtype=torch.bfloat16)
    assert route.big_router_gemm(x, w, min_rows=min_rows) is None
    assert route.big_router_gemm(x.float(), w) is None


def test_the_router_gemm_does_not_compile_under_capture(monkeypatch, caplog):
    monkeypatch.setattr(route, "_big_gemm_compile", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    route.configure(gemm=True)
    x = torch.zeros((56, 4096), dtype=torch.bfloat16)
    w = torch.zeros((512, 4096), dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING):
        assert route.big_router_gemm(x, w) is None
        assert route.big_router_gemm(x, w) is None
    lines = [
        r.getMessage() for r in caplog.records if "route_big_gemm" in r.getMessage()
    ]
    assert len(lines) == 1 and "not compiled before CUDA-graph capture" in lines[0]


def test_the_engine_route_falls_back_to_route_big_loudly(caplog):
    route.configure(engine_route=True)
    experts = SimpleNamespace(top_k=10, num_local_experts=128)
    mlp = SimpleNamespace(
        experts=experts,
        gate=SimpleNamespace(weight=torch.zeros((512, 4096), dtype=torch.bfloat16)),
    )
    x = torch.zeros((56, 4096), dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING):
        assert (
            route.engine_route(mlp, x[:32], None, torch.ones(1)) is None
        )  # outside its rows
        assert not caplog.records
        assert (
            route.engine_route(mlp, x, None, torch.ones(1, dtype=torch.float64)) is None
        )
    assert "falls back to route-big at 56 rows (router GEMM operands" in caplog.text
    experts.num_local_experts = 256
    assert route.engine_route(mlp, x, None, torch.ones(1)) is None
    assert "at most 128 local experts" in caplog.text


def _tep8_facts(**kw):
    base = dict(
        route=True, route_topk=True, tep8_moe=True, lowm_se=True, lowm_se_pre=True
    )
    return plan.Facts(**{**base, **kw})


@pytest.mark.parametrize("rows, on", [(14, False), (21, True), (28, True), (35, False)])
def test_route_big_mid_runs_at_tier_m(rows, on):
    assert plan.slots(rows, _tep8_facts(route_big_mid=True)).route_big_mid is on
    assert not plan.slots(rows, _tep8_facts()).route_big_mid
    assert not plan.slots(
        rows, _tep8_facts(route_big_mid=True, tep8_moe=False)
    ).route_big_mid


@pytest.mark.parametrize(
    "rows, on", [(21, True), (35, False), (56, False), (147, False)]
)
def test_lowm_se_keeps_its_rows_without_lowm_se_tep8(rows, on):
    assert plan.slots(rows, _tep8_facts()).lowm_se is on


@pytest.mark.parametrize("rows", [35, 56, 98, 128, 147, 160])
def test_lowm_se_tep8_widens_lowm_se_at_tep8(rows):
    assert plan.slots(rows, _tep8_facts(lowm_se_tep8=(33, 160))).lowm_se
    assert not plan.slots(
        rows, _tep8_facts(lowm_se_tep8=(33, 160), tep8_moe=False)
    ).lowm_se
    assert not plan.slots(168, _tep8_facts(lowm_se_tep8=(33, 160))).lowm_se


@pytest.mark.parametrize("rows", [21, 33, 56, 84, 98, 112, 126, 128])
@pytest.mark.parametrize("coresident", [False, True])
def test_up_to_128_rows_one_pair(rows, coresident):
    lowm._configure_tep8_se((33, lowm.TEP8_SE_WIDE_HI), coresident)
    se = lowm.tep8_se_config()["rows"]
    late = coresident and 84 <= rows <= 128
    assert lowm._tep8_se_trigger(se, rows) is (not late)
    assert lowm.se_launches(se, rows) == ((0, rows, not late, True),)


F, T = False, True


@pytest.mark.parametrize(
    "coresident, rows, want",
    [
        (True, 133, ((0, 128, F, F), (128, 133, T, T))),
        (True, 160, ((0, 128, F, F), (128, 160, T, T))),
        (False, 147, ((0, 128, T, F), (128, 147, T, T))),
    ],
)
def test_wide_rows_are_a_128_row_pair_then_a_32_row_pair(coresident, rows, want):
    lowm._configure_tep8_se((33, lowm.TEP8_SE_WIDE_HI), coresident)
    cfg = lowm.tep8_se_config()
    assert lowm.se_launches(cfg["rows"], rows) == want
    assert lowm._tep8_se_tactics(cfg["rows"], rows, 8, 1) == cfg["t1"]
    assert lowm._tep8_se_tactics(cfg["rows"], rows, 0, 2) == cfg["t2"]
    assert lowm._tep8_se_tactics(cfg["rows"], rows, 0, 0) is None


def test_outside_the_rows_one_as_built_pair_and_no_tactic():
    lowm._configure_tep8_se(lowm.TEP8_SE_ROWS, True)
    cfg = lowm.tep8_se_config()
    assert lowm.se_launches(cfg["rows"], 147) == ((0, 147, True, True),)
    assert lowm._tep8_se_tactics(cfg["rows"], 147, 8, 1) is None
    assert lowm._tep8_se_tactics(cfg["rows"], 21, 8, 1) is None


def test_the_down_launch_is_within_the_coresident_budget():
    lowm._configure_tep8_se(lowm.TEP8_SE_ROWS, True)
    t2 = lowm.tep8_se_config()["t2"]
    assert (t2.kw, t2.nt8, t2.ksplit) == (2, 1, 1)
    assert lowm.tep8_se_down_smem(t2, 128) <= lowm.TEP8_SE_CORESIDENT_SMEM
