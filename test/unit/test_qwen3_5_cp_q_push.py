# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

SLOT = "cp_q_push"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


def test_declared_as_a_bitwise_cp_lite_variant_at_tp8_and_tep8():
    assert SLOT in runtime.OPT_IN_VARIANTS and SLOT in runtime.CP_LITE_VARIANTS
    assert SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_CP_LITE and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE


@pytest.mark.parametrize(
    "slots, ok",
    [({"skv2_cp", "mka"}, True), ({"skv2_cp"}, False), ({"mka"}, False)],
)
def test_the_slot_needs_skv2_cp_and_mka(monkeypatch, slots, ok):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots | {SLOT, "skv2"}))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    assert not reasons if ok else reasons


def test_unsupported_cp_lite_topology_drops_the_slot(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp", "mka", SLOT}))
    assert runtime.apply_cp_lite_topology(2) == {"skv2_cp", SLOT}


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("1", "hop", False)],
)
def test_the_measured_plan_runs_the_slot(monkeypatch, switch, slots, on):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for topo in ("tp8", "tep8"):
        assert runtime._measured_rows_on((topo, topology.GB300), SLOT) is on
    assert (SLOT in runtime._requested_slots()) is on


def _cp(**over):
    cp = L.CpLite.__new__(L.CpLite)
    fields = dict(
        ll=True,
        cp_ungated=True,
        exact=True,
        min_kv={},
        min_bs=2,
        peers="peers",
        me=1,
        llq_off=4096,
        cnt=["cnt0"],
        cp=4,
        hq=4,
        _exact_supported=lambda q_len, b: True,
    )
    fields.update(over)
    for k, v in fields.items():
        setattr(cp, k, v)
    return cp


def test_q_push_args_for_the_ungated_exact_call():
    cp = _cp()
    assert cp.q_push_args(28, 4, 7) == {
        "peers": "peers",
        "me": 1,
        "off": 4096,
        "cnt": "cnt0",
        "cp": 4,
        "words": 4 * 256 // 4,
    }
    assert cp._q_pushed == 28


@pytest.mark.parametrize(
    "over, rows, b",
    [
        ({"ll": False}, 28, 4),
        ({"cp_ungated": False}, 28, 4),
        ({"exact": False}, 28, 4),
        ({"min_kv": {2: 1000}}, 28, 4),
        ({"min_bs": 8}, 28, 4),
        ({"_exact_supported": lambda q_len, b: False}, 28, 4),
        ({}, 27, 4),
        ({}, 7 * 129, 129),
    ],
)
def test_no_push_where_the_ungated_exact_call_does_not_run(over, rows, b):
    cp = _cp(**over)
    assert cp.q_push_args(rows, b, 7) is None
    assert cp._q_pushed is None


def test_a_pushed_call_takes_the_wait_half_once():
    cp = _cp()
    assert L._cp_gather_q_ll_or_wait(cp, 28) is L._cp_gather_q_ll
    cp.q_push_args(28, 4, 7)
    assert L._cp_gather_q_ll_or_wait(cp, 28) is L._WAIT_ONLY_GATHER
    assert cp._q_pushed is None
    assert L._cp_gather_q_ll_or_wait(cp, 28) is L._cp_gather_q_ll
    cp.q_push_args(28, 4, 7)
    with pytest.raises(AssertionError):
        L._cp_gather_q_ll_or_wait(cp, 35)


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


_GATHER = dict(
    nb=4,
    thr=0,
    GATE=False,
    NB=4,
    GMAX=False,
    CP=4,
    WORDS=256,
    QL=7,
    FOLD=True,
    GWRITE=False,
    TRIG=True,
    WAIT=True,
    num_warps=4,
    launch_pdl=True,
)


def test_the_wait_half_takes_the_gathers_launch(monkeypatch):
    wait = _Launch()
    monkeypatch.setattr(L, "_cp_gather_q_wait", wait)
    args = ("q", 256, "peers", 1, 4096, "q_all", "cnt", "lens", "pre", "stock")
    L._WAIT_ONLY_GATHER[(28,)](*args, **_GATHER)
    grid, got, kwargs = wait.calls[0]
    assert grid == (28,)
    assert got == ("peers", 1, 4096, "q_all", "cnt", "lens", "pre")
    assert kwargs == dict(
        CP=4, WORDS=256, QL=7, TRIG=True, num_warps=4, launch_pdl=True
    )
    for bad in ({"GATE": True}, {"FOLD": False}, {"WAIT": False}, {"GWRITE": True}):
        with pytest.raises(AssertionError):
            L._WAIT_ONLY_GATHER[(28,)](*args, **{**_GATHER, **bad})


@pytest.mark.parametrize("pushed", [False, True])
def test_the_ungated_exact_call_follows_the_push(monkeypatch, pushed):
    full, wait = _Launch(), _Launch()
    monkeypatch.setattr(L, "_cp_gather_q_ll", full)
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", wait)
    cp = _cp(
        early_merge=True,
        q_all=torch.zeros(4, dtype=torch.int8),
        pre_lens="pre",
        stock_lens="stock",
        gather_pdl=True,
        device="cpu",
        _gate_epilogue=None,
        _logged=set(),
    )
    merges = []
    cp._exact_partial = lambda *a, **k: None
    cp._merge_h = lambda *a, **k: merges.append(k)
    if pushed:
        cp.q_push_args(28, 4, 7)
    q = torch.zeros(28, 4, 256, dtype=torch.int8)
    gate = {k: _GATHER[k] for k in ("nb", "thr", "GATE", "NB", "GMAX")}
    cp._ungated_main(q, "k", "v", "table", "lens", 7, 4, 28, 1.0, 1.0, gate)
    assert (len(wait.calls), len(full.calls)) == ((1, 0) if pushed else (0, 1))
    assert merges[0]["push_rot"] is pushed
    assert cp._q_pushed is None


@pytest.mark.parametrize(
    "plain, plain_wait, push_rot, want",
    [
        (False, True, True, True),
        (False, False, True, False),
        (True, True, True, False),
        (False, True, False, False),
    ],
)
def test_the_rotated_push_rides_the_plain_wait_merge(
    monkeypatch, plain, plain_wait, push_rot, want
):
    launch = _Launch()
    monkeypatch.setattr(L, "_cp_merge_h_ll", launch)
    monkeypatch.setattr(L, "_cp_merge_h_ll_plain", launch)
    me = SimpleNamespace(
        cp_pdl=True,
        merge_cg=False,
        hq=4,
        cp=4,
        _logged=set(),
        po_x=torch.zeros(2, 8, dtype=torch.int32),
        pl_x=torch.zeros(2, 4),
        peers=None,
        me=0,
        llh_off=0,
        cnt_h=None,
    )
    t = torch.zeros(1, 1, 1, 1)
    gate = {"nb": 0, "thr": 0, "NB": 1, "GMAX": 1}
    args = (t, t, t, t, t, 7, 14, 1.0, 1.0, t, gate)
    L.CpLite._merge_h(
        me, *args, early=True, plain=plain, plain_wait=plain_wait, push_rot=push_rot
    )
    assert launch.calls[0][2].get("PUSH_ROT", False) is want


def _front_stub(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import front

    kernel = _Launch()
    monkeypatch.setattr(front, "_fused_qk_rmsnorm_rope_gate_kernel", kernel)
    return front, kernel


def _front_args(rows=14, heads=4, dim=256):
    q_gate = torch.zeros(rows, heads * 2 * dim, dtype=torch.bfloat16)
    k = torch.zeros(rows, dim, dtype=torch.bfloat16)
    w = torch.zeros(dim, dtype=torch.bfloat16)
    cos_sin = torch.zeros(16, 64)
    pos = torch.zeros(rows, dtype=torch.int64)
    return (q_gate, k, w, w, cos_sin, pos, 1e-6, heads, 1, dim, 64)


def test_the_front_hands_the_push_to_its_q_writer(monkeypatch):
    front, kernel = _front_stub(monkeypatch)
    args = _front_args()
    pools = tuple(torch.zeros(64, 1, 256, dtype=torch.float8_e4m3fn) for _ in range(2))
    cache = dict(
        v=torch.zeros(14, 256, dtype=torch.bfloat16),
        kv_cache=pools,
        cache_loc=torch.zeros(14, dtype=torch.int64),
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
    )
    push = dict(peers="peers", me=2, off=64, cnt="cnt", cp=4, words=256)
    front.fused_qk_gemma_rmsnorm_rope_gate(*args, **cache)
    front.fused_qk_gemma_rmsnorm_rope_gate(*args, **cache, q_push=push)
    plain, pushed = kernel.calls[0][2], kernel.calls[1][2]
    assert "PUSH_Q" not in plain
    assert {k: pushed[k] for k in pushed if k not in plain} == dict(
        PUSH_Q=True,
        push_peers_ptr="peers",
        push_me=2,
        push_off=64,
        push_cnt_ptr="cnt",
        PUSH_CP=4,
        PUSH_WORDS=256,
    )
    with pytest.raises(AssertionError):
        front.fused_qk_gemma_rmsnorm_rope_gate(*args, q_push=push)


class _CpStub:
    def __init__(self, args):
        self.args, self.calls = args, []

    def q_push_args(self, rows, b, q_len):
        self.calls.append((rows, b, q_len))
        return self.args


def _layer(window=None, kv_heads=1):
    attn = SimpleNamespace(tp_k_head_num=kv_heads, sliding_window_size=window)
    return SimpleNamespace(attn=attn)


def _st(requests):
    lens = torch.zeros(requests, dtype=torch.int32)
    return SimpleNamespace(
        view=SimpleNamespace(attn=SimpleNamespace(cache_seqlens=lens))
    )


@pytest.fixture
def mka(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.attention import mka, split_kv

    monkeypatch.setattr(mka, "_ENGAGED", set())
    monkeypatch.setattr(split_kv, "cp_multi", lambda rows: rows > 7)
    return mka, split_kv


@pytest.mark.parametrize(
    "slots, layer, rows, requests, asked",
    [
        ({"skv2_cp", "mka", SLOT}, _layer(), 28, 4, True),
        ({"skv2_cp", "mka"}, _layer(), 28, 4, False),
        ({"skv2_cp", "mka", SLOT}, _layer(), 7, 1, False),
        ({"skv2_cp", "mka", SLOT}, _layer(kv_heads=2), 28, 4, False),
        ({"skv2_cp", "mka", SLOT}, _layer(window=128), 28, 4, False),
        ({"skv2_cp", "mka", SLOT}, _layer(), 28, 3, False),
    ],
)
def test_the_mka_front_asks_for_the_push_only_where_it_can_run(
    monkeypatch, mka, caplog, slots, layer, rows, requests, asked
):
    mod, split_kv = mka
    stub = _CpStub({"words": 256})
    monkeypatch.setattr(split_kv, "cp_lite", lambda: stub)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            got = mod._q_push(layer, _st(requests), rows)
    assert (got == {"words": 256}) is asked
    assert stub.calls == ([(rows, rows // 7, 7)] * 2 if asked else [])
    assert caplog.text.count("slot cp_q_push runs") == (1 if asked else 0)
