# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import attention as A

SLOT = "draft_q_gather"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
NEEDS = {
    "draft_loop",
    "draft_front",
    "draft_multi",
    "multi_request",
    "cp_q_push",
    "cp_lam_q",
    "draft_q_push",
}


def test_declared_as_a_bitwise_draft_slot_at_tp8_and_tep8():
    assert SLOT in runtime.DRAFT_SLOTS and SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_DRAFT and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert SLOT in runtime.MEASURED_PLAN[key]


@pytest.mark.parametrize(
    "drop, ok",
    [(set(), True), ({"cp_lam_q"}, False), ({"draft_q_push"}, False)],
)
def test_the_slot_needs_draft_q_push_and_cp_lam_q(monkeypatch, drop, ok):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS - drop | {SLOT}))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    assert not reasons if ok else reasons


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


class _CpLite:
    def __init__(self, gather_ok=True):
        self.gather_ok, self.calls = gather_ok, []

    def q_push_args(self, rows, b, q_len, **kw):
        self.calls.append((rows, b, q_len, kw))
        gathered = kw.get("gather") and self.gather_ok
        return {"words": 256, **({"gather": {}} if gathered else {})}


class _CpB1:
    multi = True

    def takes(self, rows_per_request, rows):
        return True


def _attention(slots):
    att = A.Attention.__new__(A.Attention)
    att.layer = SimpleNamespace(
        attn=SimpleNamespace(tp_k_head_num=1, sliding_window_size=None)
    )
    att.front = True
    att.q_push = "draft_q_push" in slots
    att.q_gather = att.q_push and SLOT in slots
    att.cp_b1 = _CpB1()
    att._noted, att._warned = set(), set()
    return att


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {SLOT}))


def _ask(monkeypatch, att, stub, rows, per_request):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    monkeypatch.setattr(split_kv, "cp_lite", lambda: stub)
    lengths = torch.zeros(rows // per_request, dtype=torch.int32)
    meta = SimpleNamespace(cache_seqlens=lengths)
    return att._q_push_args(rows, per_request, meta), lengths


@pytest.mark.parametrize("rows, per_request", [(16, 1), (63, 7)])
def test_the_front_asks_for_the_gather_with_the_calls_lengths(
    monkeypatch, capture, caplog, rows, per_request
):
    att, stub = _attention(NEEDS | {SLOT}), _CpLite()
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        args, lengths = _ask(monkeypatch, att, stub, rows, per_request)
    assert "gather" in args
    ((_, b, q_len, kw),) = stub.calls
    assert (b, q_len) == (rows // per_request, per_request)
    assert kw["gather"] is True and kw["seq_lens"] is lengths
    assert caplog.text.count("slot draft_q_gather lands the group's q of") == 1
    assert "WARNING" not in caplog.text


def test_without_the_slot_the_push_carries_no_gather(monkeypatch, capture, caplog):
    att, stub = _attention(NEEDS), _CpLite()
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        args, _ = _ask(monkeypatch, att, stub, 16, 1)
    assert args == {"words": 256} and stub.calls[0][3] == {}
    assert "draft_q_gather" not in caplog.text


def test_a_push_without_the_gather_keeps_the_wait_half_and_warns_once(
    monkeypatch, capture, caplog
):
    att, stub = _attention(NEEDS | {SLOT}), _CpLite(gather_ok=False)
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        for _ in range(2):
            args, _ = _ask(monkeypatch, att, stub, 16, 1)
    assert "gather" not in args
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "keeps the gather's wait half" in warnings[0].getMessage()


def _built(monkeypatch, slots):
    """The draft attention's own __init__ on a CPU stand-in of its parts."""
    import sglang.srt.layers.attention.trtllm_mla_backend as mla
    import sglang.srt.runtime_context as context

    monkeypatch.setattr(context, "get_buffer", lambda name, make: "workspace")
    monkeypatch.setattr(
        mla, "make_persistent_multi_ctas_kv_counter_buffer", lambda *a, **k: "counters"
    )
    attn = SimpleNamespace(tp_q_head_num=4, head_dim=256, k_scale=None, v_scale=None)
    parts = SimpleNamespace(
        layer=SimpleNamespace(attn=attn),
        device=torch.device("cpu"),
        binding=SimpleNamespace(page_size=64, tokens_per_request=7),
        draft=SimpleNamespace(context_len=4096),
        lm=SimpleNamespace(config=SimpleNamespace(num_attention_heads=32)),
        slots=frozenset(slots),
    )
    return A.Attention(parts)


@pytest.mark.parametrize(
    "slots, push, gather",
    [
        ({"draft_front", "draft_q_push", SLOT}, True, True),
        ({"draft_front", "draft_q_push"}, True, False),
        ({"draft_front", SLOT}, False, False),
        ({"draft_q_push", SLOT}, False, False),
    ],
)
def test_the_attention_reads_the_slot_with_draft_q_push_only(
    monkeypatch, slots, push, gather
):
    att = _built(monkeypatch, slots)
    assert (att.q_push, att.q_gather) == (push, gather)
