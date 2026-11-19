# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

SLOT = "cp_lam_q"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


def test_declared_as_a_bitwise_cp_lite_variant_at_tp8_and_tep8():
    assert SLOT in runtime.OPT_IN_VARIANTS and SLOT in runtime.CP_LITE_VARIANTS
    assert SLOT in runtime.CP_PUSHED_CALL_SLOTS and SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_CP_LITE and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE


@pytest.mark.parametrize("push", [True, False])
def test_the_slot_needs_cp_q_push(monkeypatch, push):
    slots = {"skv2", "skv2_cp", "mka", SLOT} | ({"cp_q_push"} if push else set())
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    assert not reasons if push else reasons


def test_unsupported_cp_lite_topology_drops_the_slot(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    slots = frozenset({"skv2", "skv2_cp", "mka", "cp_q_push", SLOT})
    monkeypatch.setattr(runtime, "SLOTS", slots)
    assert runtime.apply_cp_lite_topology(2) == {"skv2_cp", "cp_q_push", SLOT}


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


def _build(slots, parallel=None):

    class _Binding:
        page_size, tp_size = 64, 8

    seen = {}
    lite = types.ModuleType("q35_cp_lite_stub")
    cp_lite = SimpleNamespace(_instance=None, lam_q=False)

    def maybe_create(binding, device):
        seen["lam_q"] = cp_lite.lam_q
        cp_lite._instance = SimpleNamespace()
        return cp_lite._instance

    cp_lite.maybe_create = maybe_create
    lite.CpLite = cp_lite
    agree = types.ModuleType("binding_stub")
    agree.megakernel_agree = lambda *a, **k: None
    with (
        patch.object(runtime, "SLOTS", frozenset(slots)),
        patch.dict(
            sys.modules,
            {
                "sglang.srt.kernel_plugins.nvidia.binding": agree,
                "sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite": lite,
            },
        ),
        patch("torch.cuda.current_device", lambda: 0),
    ):
        runtime.build_cp_lite(_Binding())
        if parallel is not None:
            runtime.apply_topology(parallel)
        runtime.apply_cp_lite_build()
    return seen["lam_q"], cp_lite._instance


@pytest.mark.parametrize(
    "slots, want",
    [
        ({"skv2", "skv2_cp", "mka", "cp_q_push", SLOT}, True),
        ({"skv2", "skv2_cp", "mka", "cp_q_push"}, False),
    ],
)
def test_the_build_allocates_and_sets_it_from_the_slots(slots, want):
    allocated, instance = _build(slots)
    assert allocated is want and instance.lam_q is want


@pytest.mark.parametrize("tp, ep, want", [(4, 1, False), (8, 4, True)])
def test_the_instance_follows_the_topology_filter(tp, ep, want):
    parallel = SimpleNamespace(tp_size=tp, attn_dp_size=1, moe_ep_size=ep)
    slots = {"skv2", "skv2_cp", "mka", "cp_q_push", SLOT}
    allocated, instance = _build(slots, parallel)
    assert allocated is True and instance.lam_q is want


def test_the_lamport_q_region_layout():
    off, end = L._lam_q_layout(1001, 4, 4)
    assert off == 1008 and off % 16 == 0
    assert end - off == L._MAX_ROWS * 4 * (4 * L.HEAD_DIM // 4) * 4
    assert L._lam_q_layout(1024, 4, 4)[0] == 1024


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
        cnt=["cnt0"],
        cp=4,
        hq=4,
        _exact_supported=lambda q_len, b: True,
    )
    fields.update(over)
    for k, v in fields.items():
        setattr(cp, k, v)
    return cp


@pytest.mark.parametrize("lam", [True, False])
def test_q_push_args_point_the_writer_at_the_lamport_region(lam):
    cp = _cp(lam_q=lam)
    args = cp.q_push_args(28, 4, 7)
    assert args["off"] == (1 << 20 if lam else 4096)
    assert args.get("lam", False) is lam
    assert (cp._q_pushed, cp._q_pushed_lam) == (28, lam)
    assert L._cp_gather_q_ll_or_wait(cp, 28) is L._WAIT_ONLY_GATHER
    assert (cp._q_pushed, cp._q_pushed_lam) == (None, False)


def test_q_push_args_refuse_lamport_words_without_their_region():
    cp = _cp(lam_q=True, lam_q_end=0)
    with pytest.raises(RuntimeError, match="without its Lamport regions"):
        cp.q_push_args(28, 4, 7)
    assert _cp(lam_q=False, lam_q_end=0).q_push_args(28, 4, 7)["off"] == 4096


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


_GATE = dict(nb=4, thr=0, GATE=False, NB=4, GMAX=False)


@pytest.mark.parametrize("lam, pushed", [(True, True), (False, True), (True, False)])
def test_the_ungated_exact_call_polls_the_pushed_region(
    monkeypatch, caplog, lam, pushed
):
    full, wait = _Launch(), _Launch()
    monkeypatch.setattr(L, "_cp_gather_q_ll", full)
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", wait)
    cp = _cp(
        lam_q=lam,
        early_merge=True,
        q_all=torch.zeros(4, dtype=torch.int8),
        pre_lens="pre",
        stock_lens="stock",
        gather_pdl=True,
        device="cpu",
        _gate_epilogue=None,
        _logged=set(),
    )
    cp._exact_partial = lambda *a, **k: None
    cp._merge_h = lambda *a, **k: None
    q = torch.zeros(28, 4, 256, dtype=torch.int8)
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            if pushed:
                cp.q_push_args(28, 4, 7)
            cp._ungated_main(q, "k", "v", "table", "lens", 7, 4, 28, 1.0, 1.0, _GATE)
    calls = wait.calls if pushed else full.calls
    _, args, kwargs = calls[0]
    want_lam = lam and pushed
    assert args[4] == (1 << 20 if want_lam else 4096)
    assert kwargs.get("LAM", False) is want_lam
    assert caplog.text.count("slot cp_lam_q moves") == (1 if want_lam else 0)


def test_the_wait_half_takes_lam(monkeypatch):
    kernel = _Launch()
    monkeypatch.setattr(L, "_cp_gather_q_wait", kernel)
    args = ("q", 256, "peers", 1, 1 << 20, "q_all", "cnt", "lens", "pre", "stock")
    opts = dict(
        **_GATE,
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
    L._WAIT_ONLY_GATHER[(28,)](*args, **opts, LAM=True)
    L._WAIT_ONLY_GATHER[(28,)](*args, **opts)
    assert kernel.calls[0][2]["LAM"] is True
    assert "LAM" not in kernel.calls[1][2]
    assert kernel.calls[0][1][2] == 1 << 20


def test_the_front_hands_the_lamport_push_to_its_q_writer(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import front

    kernel = _Launch()
    monkeypatch.setattr(front, "_fused_qk_rmsnorm_rope_gate_kernel", kernel)
    rows, heads, dim = 14, 4, 256
    q_gate = torch.zeros(rows, heads * 2 * dim, dtype=torch.bfloat16)
    k = torch.zeros(rows, dim, dtype=torch.bfloat16)
    w = torch.zeros(dim, dtype=torch.bfloat16)
    args = (q_gate, k, w, w, torch.zeros(16, 64), torch.zeros(rows, dtype=torch.int64))
    args += (1e-6, heads, 1, dim, 64)
    pools = tuple(torch.zeros(64, 1, dim, dtype=torch.float8_e4m3fn) for _ in range(2))
    cache = dict(
        v=torch.zeros(rows, dim, dtype=torch.bfloat16),
        kv_cache=pools,
        cache_loc=torch.zeros(rows, dtype=torch.int64),
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
    )
    push = dict(peers="peers", me=2, off=1 << 20, cnt="cnt", cp=4, words=256)
    front.fused_qk_gemma_rmsnorm_rope_gate(*args, **cache, q_push=push)
    front.fused_qk_gemma_rmsnorm_rope_gate(*args, **cache, q_push={**push, "lam": True})
    ll, lam = kernel.calls[0][2], kernel.calls[1][2]
    assert "PUSH_LAM" not in ll
    assert {k: lam[k] for k in lam if k not in ll} == dict(PUSH_LAM=True, num_warps=4)
    assert lam["push_off"] == 1 << 20
