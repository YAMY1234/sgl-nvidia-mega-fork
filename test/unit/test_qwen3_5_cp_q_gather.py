# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

SLOT = "cp_q_gather"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
NEEDS = {"skv2", "skv2_cp", "mka", "cp_q_push", "cp_lam_q"}


def test_declared_as_a_bitwise_cp_lite_variant_at_tp8_and_tep8():
    assert SLOT in runtime.OPT_IN_VARIANTS and SLOT in runtime.CP_LITE_VARIANTS
    assert SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_CP_LITE and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE


@pytest.mark.parametrize(
    "drop, ok",
    [(set(), True), ({"cp_lam_q"}, False), ({"cp_q_push", "cp_lam_q"}, False)],
)
def test_the_slot_needs_cp_q_push_and_cp_lam_q(monkeypatch, drop, ok):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS - drop | {SLOT}))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    assert not reasons if ok else reasons


def test_unsupported_cp_lite_topology_drops_the_slot(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {SLOT}))
    assert runtime.apply_cp_lite_topology(2) == {
        "skv2_cp",
        "cp_q_push",
        "cp_lam_q",
        SLOT,
    }


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
        lamq_off=1 << 20,
        lam_q_end=(1 << 20) + 4096,
        lam_q=True,
        cnt=["cnt0"],
        cp=4,
        hq=4,
        q_all=torch.zeros(16, dtype=torch.int8),
        pre_lens="pre",
        _exact_supported=lambda q_len, b: True,
    )
    fields.update(over)
    for k, v in fields.items():
        setattr(cp, k, v)
    return cp


def test_q_push_args_carry_the_gather_with_lamport_words():
    cp = _cp()
    lens = torch.zeros(4, dtype=torch.int32)
    args = cp.q_push_args(28, 4, 7, gather=True, seq_lens=lens)
    gq = args["gather"]
    assert gq["q_all"].dtype == torch.int32 and gq["seq_lens"] is lens
    assert (gq["pre_lens"], gq["ql"]) == ("pre", 7)
    assert (cp._q_pushed, cp._q_gathered) == (28, 28)
    assert L._cp_gather_q_ll_or_wait(cp, 28) is None
    assert (cp._q_pushed, cp._q_gathered) == (None, None)
    assert L._cp_gather_q_ll_or_wait(cp, 28) is L._cp_gather_q_ll


@pytest.mark.parametrize(
    "over, lens, gather",
    [
        ({"lam_q": False}, torch.zeros(4, dtype=torch.int32), True),
        ({}, torch.zeros(4, dtype=torch.int64), True),
        ({}, torch.zeros(3, dtype=torch.int32), True),
        ({}, None, True),
        ({}, torch.zeros(4, dtype=torch.int32), False),
    ],
)
def test_no_gather_without_its_inputs(over, lens, gather):
    cp = _cp(**over)
    args = cp.q_push_args(28, 4, 7, gather=gather, seq_lens=lens)
    assert args is not None and "gather" not in args
    assert cp._q_gathered is None
    assert L._cp_gather_q_ll_or_wait(cp, 28) is L._WAIT_ONLY_GATHER


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


_GATE = dict(nb=4, thr=0, GATE=False, NB=4, GMAX=False)


@pytest.mark.parametrize("gathered", [True, False])
def test_a_gathered_call_launches_no_gather(monkeypatch, gathered):
    full, wait = _Launch(), _Launch()
    monkeypatch.setattr(L, "_cp_gather_q_ll", full)
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", wait)
    cp = _cp(
        early_merge=True,
        stock_lens="stock",
        gather_pdl=True,
        device="cpu",
        _gate_epilogue=None,
        _logged=set(),
    )
    partials = []
    cp._exact_partial = lambda *a, **k: partials.append(k)
    cp._merge_h = lambda *a, **k: None
    cp.q_push_args(
        28, 4, 7, gather=gathered, seq_lens=torch.zeros(4, dtype=torch.int32)
    )
    q = torch.zeros(28, 4, 256, dtype=torch.int8)
    cp._ungated_main(q, "k", "v", "table", "lens", 7, 4, 28, 1.0, 1.0, _GATE)
    assert (len(full.calls), len(wait.calls)) == ((0, 0) if gathered else (0, 1))
    assert partials[0]["pdl"] is True
    assert (cp._q_pushed, cp._q_gathered) == (None, None)


def _front_call(q_push, early_trigger=False):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import front

    rows, heads, dim = 28, 4, 256
    q_gate = torch.zeros(rows, heads * 2 * dim, dtype=torch.bfloat16)
    k = torch.zeros(rows, dim, dtype=torch.bfloat16)
    w = torch.zeros(dim, dtype=torch.bfloat16)
    args = (q_gate, k, w, w, torch.zeros(16, 64), torch.zeros(rows, dtype=torch.int64))
    args += (1e-6, heads, 1, dim, 64)
    pools = tuple(torch.zeros(64, 1, dim, dtype=torch.float8_e4m3fn) for _ in range(2))
    front.fused_qk_gemma_rmsnorm_rope_gate(
        *args,
        v=torch.zeros(rows, dim, dtype=torch.bfloat16),
        kv_cache=pools,
        cache_loc=torch.zeros(rows, dtype=torch.int64),
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
        q_push=q_push,
        early_trigger=early_trigger,
    )


def test_the_front_gathers_and_releases_at_entry(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import front

    kernel = _Launch()
    monkeypatch.setattr(front, "_fused_qk_rmsnorm_rope_gate_kernel", kernel)
    push = dict(peers="peers", me=2, off=1 << 20, cnt="cnt", cp=4, words=256, lam=True)
    gq = dict(
        q_all=torch.zeros(8, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        pre_lens="pre",
        ql=7,
    )
    _front_call(push)
    _front_call({**push, "gather": gq})
    lam, gathered = kernel.calls[0][2], kernel.calls[1][2]
    assert "GATHER_Q" not in lam and lam["EARLY_TRIGGER"] is False
    assert gathered["EARLY_TRIGGER"] is True
    assert {k: gathered[k] for k in gathered if k not in lam} == dict(
        GATHER_Q=True,
        gq_all_ptr=gq["q_all"],
        gq_seq_lens_ptr=gq["seq_lens"],
        gq_pre_lens_ptr="pre",
        GQ_QL=7,
    )
    with pytest.raises(AssertionError):
        _front_call({**push, "lam": False, "gather": gq})


class _CpStub:
    def __init__(self, gather_ok=True):
        self.gather_ok, self.calls = gather_ok, []

    def q_push_args(self, rows, b, q_len, **kw):
        self.calls.append((rows, b, q_len, kw))
        return {"words": 256, **({"gather": {}} if kw and self.gather_ok else {})}


def _layer():
    attn = SimpleNamespace(tp_k_head_num=1, sliding_window_size=None)
    return SimpleNamespace(attn=attn)


def _st(requests):
    lens = torch.zeros(requests, dtype=torch.int32)
    return SimpleNamespace(
        view=SimpleNamespace(attn=SimpleNamespace(cache_seqlens=lens))
    )


@pytest.mark.parametrize(
    "slots, rows, gather_ok, asked, line",
    [
        (NEEDS | {SLOT}, 63, True, True, "slot cp_q_gather runs at 63 rows"),
        (NEEDS | {SLOT}, 21, False, True, "keeps the gather's wait half at 21 rows"),
        (NEEDS | {SLOT}, 70, True, False, None),
        (NEEDS, 28, True, False, None),
    ],
)
def test_the_mka_front_asks_for_the_gather_through_its_rows(
    monkeypatch, caplog, slots, rows, gather_ok, asked, line
):
    from sgl_nvidia_mega.qwen3_5.attention import mka, split_kv

    monkeypatch.setattr(mka, "_ENGAGED", set())
    monkeypatch.setattr(split_kv, "cp_multi", lambda m: m > 7)
    stub = _CpStub(gather_ok)
    monkeypatch.setattr(split_kv, "cp_lite", lambda: stub)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    st = _st(rows // 7)
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            mka._q_push(_layer(), st, rows)
    kw = stub.calls[0][3]
    assert (kw.get("gather", False), kw.get("seq_lens")) == (
        (True, st.view.attn.cache_seqlens) if asked else (False, None)
    )
    assert caplog.text.count("cp_q_gather") == (1 if line else 0)
    if line:
        assert line in caplog.text
