# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import oproj_push


def test_the_option_is_a_numeric_row_of_the_tp8_and_tep8_plans():
    assert oproj_push.OPTION in runtime.MEASURED_OPTIONS
    assert oproj_push.OPTION in runtime.NUMERICS_CHANGE
    assert oproj_push.OPTION not in topology.SLOT_TOPOLOGIES
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert oproj_push.OPTION in runtime.MEASURED_PLAN[key]


def _parts(*, hop=True, weight=None, bias=None):
    o_proj = SimpleNamespace(
        weight=torch.zeros(4096, 1024, dtype=torch.bfloat16)
        if weight is None
        else weight,
        bias=bias,
    )
    return SimpleNamespace(
        binding=SimpleNamespace(tp_size=8, tp_rank=0, tp_cpu_group="cpu"),
        hop=SimpleNamespace(hop=SimpleNamespace(supports=lambda m: m <= 28))
        if hop
        else None,
        layer=SimpleNamespace(o_proj=o_proj),
    )


def test_off_the_measured_plan_nothing_is_built(monkeypatch):
    monkeypatch.setattr(runtime, "measured_option", lambda name: False)

    def agree(*_, **__):
        raise AssertionError("no agreement off the plan")

    monkeypatch.setattr(runtime, "slot_supported", agree)
    assert oproj_push.maybe_build(_parts()) is None


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"hop": False}, "slot draft_hop is off"),
        (
            {"weight": torch.zeros(4096, 1000, dtype=torch.bfloat16)},
            "not a BF16 [4096, K] slice",
        ),
        ({"bias": "b"}, "has a bias"),
    ],
)
def test_the_build_refuses_what_it_cannot_push(monkeypatch, kwargs, reason):
    monkeypatch.setattr(runtime, "measured_option", lambda name: True)
    seen = []
    monkeypatch.setattr(
        runtime,
        "slot_supported",
        lambda name, why, **_: seen.append((name, why)) or not why,
    )
    assert oproj_push.maybe_build(_parts(**kwargs)) is None
    ((name, why),) = seen
    assert name == oproj_push.OPTION and reason in why


def test_the_push_launch_marks_its_partial_pushed(monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import sm103

    runs = []
    module = types.ModuleType("sgl_nvidia_mega.qwen3_5.sm103.trt_lowm")
    module.run = lambda s, out, x16, w_bf16, push: runs.append((s, out.shape, push))
    monkeypatch.setitem(sys.modules, "sgl_nvidia_mega.qwen3_5.sm103.trt_lowm", module)
    monkeypatch.setattr(sm103, "trt_lowm", module, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    marked = []
    hop = SimpleNamespace(mark_pushed=lambda out: marked.append(out))
    push = oproj_push.OprojPush(
        torch.zeros(8, 4, dtype=torch.bfloat16), hop, {2: "spec2", 4: "spec4"}, "desc"
    )
    assert push.takes(2) and not push.takes(3)
    out = push(torch.ones(2, 4, dtype=torch.bfloat16))
    assert runs == [("spec2", (2, 8), "desc")] and marked == [out]


@pytest.mark.parametrize(
    "rows_per_request, rows, want",
    [(1, 2, True), (1, 1, False), (7, 14, False), (1, 3, False)],
)
def test_the_layer_hands_the_multi_request_steps_on_the_hop_to_it(
    rows_per_request, rows, want
):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    seen = []

    def attention(hidden, *args, rows_per_request, o_proj, **kwargs):
        seen.append(kwargs.get("o_push"))
        raise StopIteration

    o_push = SimpleNamespace(takes=lambda m: m == 2)
    parts = SimpleNamespace(
        layer=SimpleNamespace(input_layernorm=lambda x: x),
        attention=attention,
        hop=SimpleNamespace(
            hop=SimpleNamespace(supports=lambda m: m <= 28), gemv=False
        ),
        slots=frozenset(),
        oproj_push=o_push,
    )
    with pytest.raises(StopIteration):
        layer.layer(
            parts,
            torch.ones(rows, 4),
            None,
            None,
            None,
            None,
            rows_per_request=rows_per_request,
        )
    assert seen == [o_push if want else None]
