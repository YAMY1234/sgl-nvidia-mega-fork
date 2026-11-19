# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import attention as A

SLOT = "draft_q_push"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
NEEDS = {"draft_loop", "draft_front", "draft_multi", "multi_request", "cp_q_push"}


def test_declared_as_a_bitwise_draft_slot_at_tp8_and_tep8():
    assert SLOT in runtime.DRAFT_SLOTS and SLOT not in runtime.SHIPPED_SLOTS
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    assert SLOT in runtime.MEASURED_DRAFT and SLOT in runtime.MEASURED_BITWISE
    assert SLOT not in runtime.NUMERICS_CHANGE
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert SLOT in runtime.MEASURED_PLAN[key]
    assert SLOT not in runtime.MEASURED_PLAN.get(("dep4", topology.GB300), ())


@pytest.mark.parametrize(
    "drop, ok",
    [
        (set(), True),
        ({"cp_q_push"}, False),
        ({"draft_front"}, False),
        ({"draft_multi"}, False),
    ],
)
def test_the_slot_needs_draft_front_draft_multi_and_cp_q_push(monkeypatch, drop, ok):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS - drop | {SLOT}))
    reasons = [r for r in runtime.slot_reasons() if f"slot '{SLOT}'" in r]
    assert not reasons if ok else reasons


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("0", None, False), ("1", "hop", False)],
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
    def __init__(self, give=True):
        self.give, self.calls, self._q_pushed = give, [], None

    def q_push_args(self, rows, b, q_len, **kw):
        self.calls.append((rows, b, q_len, kw))
        if not self.give:
            return None
        self._q_pushed = rows
        return {"words": 256}


class _CpB1:
    def __init__(self, multi=True, takes=True):
        self.multi, self._takes, self.asked = multi, takes, []

    def takes(self, rows_per_request, rows):
        self.asked.append((rows_per_request, rows))
        return self._takes


def _attention(
    *, slots=frozenset({SLOT}), front=True, cp_b1=None, heads=1, window=None
):
    att = A.Attention.__new__(A.Attention)
    att.layer = SimpleNamespace(
        attn=SimpleNamespace(tp_k_head_num=heads, sliding_window_size=window),
        attn_output_gate=False,
    )
    att.front = front
    att.q_push = front and SLOT in slots
    att.q_gather = False
    att.cp_b1 = _CpB1() if cp_b1 is None else cp_b1
    att._noted, att._warned = set(), set()
    return att


def _meta(requests):
    return SimpleNamespace(cache_seqlens=torch.zeros(requests, dtype=torch.int32))


@pytest.fixture
def cp(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    stub = _CpLite()
    monkeypatch.setattr(split_kv, "cp_lite", lambda: stub)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {SLOT}))
    # the engagement lines are logged under capture only
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return stub


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
    "slots, on",
    [
        ({"draft_front", SLOT}, True),
        ({SLOT}, False),
        ({"draft_front"}, False),
    ],
)
def test_the_attention_pushes_only_with_draft_front(monkeypatch, slots, on):
    assert _built(monkeypatch, slots).q_push is on


@pytest.mark.parametrize(
    "rows, per_request, requests",
    [(2, 1, 2), (16, 1, 16), (28, 7, 4), (63, 7, 9)],
)
def test_a_multi_request_call_cp_lite_takes_gets_push_arguments(
    monkeypatch, cp, rows, per_request, requests
):
    att = _attention()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert att._q_push_args(rows, per_request, _meta(requests)) == {"words": 256}
    assert cp.calls == [(rows, requests, per_request, {})]
    assert att.cp_b1.asked == [(per_request, rows)]
    assert cp._q_pushed == rows
    line = (
        f"{rows} rows"
        if per_request == 1
        else f"the extend's {rows} rows ({requests} x {per_request})"
    )
    assert att._noted == {
        f"slot draft_q_push pushes the FP8 q of {line} from the front"
    }


@pytest.mark.parametrize(
    "case",
    [
        "no cp_q_push",
        "no CP-lite",
        "no multi",
        "one request",
        "ragged rows",
        "not taken",
        "two KV heads",
        "a window",
        "lengths",
        "no lengths",
    ],
)
def test_no_push_where_the_exact_call_cannot_take_it(monkeypatch, cp, case):
    att, rows, per_request, meta = _attention(), 16, 1, _meta(16)
    if case == "no cp_q_push":
        monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS - {"cp_q_push"}))
    elif case == "no CP-lite":
        from sgl_nvidia_mega.qwen3_5.attention import split_kv

        monkeypatch.setattr(split_kv, "cp_lite", lambda: None)
    elif case == "no multi":
        att.cp_b1 = _CpB1(multi=False)
    elif case == "one request":
        rows, per_request, meta = 7, 7, _meta(1)
    elif case == "ragged rows":
        rows, per_request = 15, 7
    elif case == "not taken":
        att.cp_b1 = _CpB1(takes=False)
    elif case == "two KV heads":
        att = _attention(heads=2)
    elif case == "a window":
        att = _attention(window=128)
    elif case == "lengths":
        meta = _meta(8)
    elif case == "no lengths":
        meta = SimpleNamespace()
    assert att._q_push_args(rows, per_request, meta) is None
    assert cp.calls == [] and cp._q_pushed is None


def test_cp_lite_declining_the_shapes_pushes_nothing(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    stub = _CpLite(give=False)
    monkeypatch.setattr(split_kv, "cp_lite", lambda: stub)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(NEEDS | {SLOT}))
    att = _attention()
    assert att._q_push_args(16, 1, _meta(16)) is None
    assert att._noted == set()


def _call(att, monkeypatch, rows, per_request, take=True):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    fronts = []

    def front(x, positions, cache_loc, kv, q_push=None):
        fronts.append(q_push)
        return torch.zeros(rows, 8), None

    def decode(q8, meta, kv, rows_per_request):
        if take:
            split_kv.cp_lite()._q_pushed = None
        return torch.zeros(rows, 8)

    monkeypatch.setattr(att, "_front", front)
    monkeypatch.setattr(att, "decode", decode)
    att.layer.o_proj = lambda o: (o, None)
    x = torch.zeros(rows, 8)
    att(x, None, None, _meta(rows // per_request), None, rows_per_request=per_request)
    return fronts


def test_the_front_gets_the_push_and_cp_lite_takes_it(monkeypatch, cp):
    att = _attention()
    assert _call(att, monkeypatch, 8, 1) == [{"words": 256}]
    assert cp._q_pushed is None


def test_without_the_slot_the_front_pushes_nothing(monkeypatch, cp):
    att = _attention(slots=frozenset())
    assert _call(att, monkeypatch, 8, 1) == [None]
    assert cp.calls == []


def test_an_untaken_push_fails_loudly(monkeypatch, cp):
    att = _attention()
    with pytest.raises(RuntimeError, match="did not take the pushed q rows"):
        _call(att, monkeypatch, 8, 1, take=False)


def test_the_engagement_line_is_logged_once_under_capture(monkeypatch, cp, caplog):
    att = _attention()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        for _ in range(2):
            att._q_push_args(8, 1, _meta(8))
            cp._q_pushed = None
    assert caplog.text.count("slot draft_q_push pushes the FP8 q of 8 rows") == 1
