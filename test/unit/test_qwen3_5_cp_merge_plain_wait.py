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
from sgl_nvidia_mega.qwen3_5.attention import split_kv

SLOT = "cp_merge_plain_wait"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


def test_declared_as_a_bitwise_cp_lite_variant_at_tp8_and_tep8():
    assert SLOT in runtime.OPT_IN_VARIANTS and SLOT in runtime.CP_LITE_VARIANTS
    assert SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_CP_LITE and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE


def test_it_needs_skv2_cp(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT, "skv2"}))
    assert any(
        f"slot '{SLOT}' polls slot 'skv2_cp'" in r for r in runtime.slot_reasons()
    )
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT, "skv2", "skv2_cp"}))
    assert not [r for r in runtime.slot_reasons() if SLOT in r]


def test_unsupported_cp_lite_topology_drops_it(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp", SLOT}))
    assert runtime.apply_cp_lite_topology(2) == {"skv2_cp", SLOT}


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("1", "skv2,skv2_cp", False)],
)
def test_the_measured_plan_runs_it_at_tp8_and_tep8(monkeypatch, switch, slots, on):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for topo in ("tp8", "tep8"):
        assert runtime._measured_rows_on((topo, topology.GB300), SLOT) is on
    assert (SLOT in runtime._requested_slots()) is on


class _CpLite:
    def __init__(self):
        self.calls = []

    def attend(self, q, caches, page_table, seq_lens, q_len, *args, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.bfloat16)


def _step(requests):
    rows = requests * split_kv.ROWS
    heads, dim = 4, 256
    attn = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=heads,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=dim,
        scaling=dim**-0.5,
    )
    kv = SimpleNamespace(
        layer_ids=[3],
        k=[torch.zeros(4 * 64, 1, dim)],
        v=[torch.zeros(4 * 64, 1, dim)],
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
    )
    meta = SimpleNamespace(
        page_table=torch.zeros(requests, 4, dtype=torch.int32),
        cache_seqlens=torch.full((requests,), 100, dtype=torch.int32),
    )
    st = SimpleNamespace(view=SimpleNamespace(kv=kv, attn=meta), fb=None)
    q = torch.zeros(rows, heads * dim)
    return attn, q, st


@pytest.fixture
def cp(monkeypatch):
    stub = _CpLite()
    monkeypatch.setattr(split_kv, "cp_lite", lambda: stub)
    monkeypatch.setattr(split_kv, "_ENGAGED", set())
    utils = types.ModuleType("utils_stub")
    utils.canonicalize_stride = lambda t: t
    mha = types.ModuleType("trtllm_mha_stub")
    mha.DEFAULT_WORKSPACE_SIZE_MB = 1
    ctx = types.ModuleType("runtime_context_stub")
    ctx.get_buffer = lambda name, make: torch.zeros(1)
    with patch.dict(
        sys.modules,
        {
            "sglang.kernels.ops.attention.utils": utils,
            "sglang.srt.layers.attention.trtllm_mha_backend": mha,
            "sglang.srt.runtime_context": ctx,
        },
    ):
        yield stub


@pytest.mark.parametrize("on", [True, False])
def test_the_multi_request_verify_asks_for_the_plain_wait(monkeypatch, cp, on):
    slots = {"skv2", "skv2_cp"} | ({SLOT} if on else set())
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    assert split_kv.merge_options() == ({"plain_wait": True} if on else {})
    attn, q, st = _step(8)
    for _ in range(2):
        split_kv.core(attn, q, st)
    assert len(cp.calls) == 2
    for kwargs in cp.calls:
        assert kwargs.get("plain_wait", False) is on
        assert kwargs["min_bs"] is None


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append(kwargs)


def test_only_the_exact_merge_takes_and_logs_the_plain_wait(monkeypatch, caplog):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

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
    cases = ((False, True, 14), (False, True, 14), (False, True, 21))
    cases += ((True, True, 14), (False, False, 14))
    with caplog.at_level(logging.INFO):
        for plain, plain_wait, rows in cases:
            args = (t, t, t, t, t, 7, rows, 1.0, 1.0, t, gate)
            L.CpLite._merge_h(me, *args, plain=plain, plain_wait=plain_wait)
    got = [kwargs.get("PLAIN_WAIT", False) for kwargs in launch.calls]
    assert got == [True, True, True, False, False]
    assert caplog.text.count("slot cp_merge_plain_wait polls") == 2


def test_cp_core_passes_the_plain_wait_only_when_asked(cp):
    attn, q, st = _step(4)
    for plain_wait in (False, True):
        split_kv.cp_core(
            attn,
            q,
            st.view.kv,
            st.view.attn,
            None,
            split_kv.ROWS,
            plain_wait=plain_wait,
        )
    assert "plain_wait" not in cp.calls[0]
    assert cp.calls[1]["plain_wait"] is True
