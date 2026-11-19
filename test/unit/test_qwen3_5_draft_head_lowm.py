# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import head_lowm


def test_draft_head_lowm_is_a_numeric_draft_row_of_the_tp8_and_tep8_plans():
    from sgl_nvidia_mega.qwen3_5 import draft
    from sgl_nvidia_mega.qwen3_5.spec import NumericsKind

    assert "draft_head_lowm" in runtime.DRAFT_SLOTS
    assert topology.SLOT_TOPOLOGIES["draft_head_lowm"] == {"tp4", "tp8", "tep8"}
    assert "draft_head_lowm" in runtime.MEASURED_DRAFT
    assert "draft_head_lowm" in runtime.NUMERICS_CHANGE
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert "draft_head_lowm" in runtime.MEASURED_PLAN[key]
    got = draft.numerics(frozenset({"draft_loop", "draft_head_lowm"}))
    assert got.kind == NumericsKind.TOLERANCE


def test_draft_head_lowm_needs_draft_multi():
    def reasons(slots):
        return [
            r
            for r in runtime.slot_reasons(frozenset(slots))
            if "'draft_head_lowm' needs" in r
        ]

    assert reasons({"draft_loop", "draft_multi", "draft_head_lowm"}) == []
    assert len(reasons({"draft_loop", "draft_head_lowm"})) == 1


def _parts(*, top1x=True, dtype=torch.bfloat16, logit_scale=None):
    lm = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.zeros(16, 8, dtype=dtype)),
        logits_processor=SimpleNamespace(logit_scale=logit_scale),
    )
    binding = SimpleNamespace(tp_cpu_group="cpu", tp_rank=0)
    return SimpleNamespace(lm=lm, binding=binding, top1x="x" if top1x else None)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"top1x": False}, "it serves slot draft_multi's steps"),
        ({"dtype": torch.float32}, "needs a contiguous BF16 shard"),
        ({"logit_scale": 0.5}, "logit scaling or softcapping"),
    ],
)
def test_the_build_refuses_a_head_it_cannot_serve(monkeypatch, kwargs, reason):
    seen = []
    monkeypatch.setattr(
        runtime,
        "slot_supported",
        lambda name, why, **_: seen.append((name, why)) or not why,
    )

    def race(*_, **__):
        raise AssertionError("a refused head is never raced")

    monkeypatch.setattr(head_lowm, "HeadLowM", race)
    assert head_lowm.maybe_build(_parts(**kwargs)) is None
    ((name, why),) = seen
    assert name == "draft_head_lowm" and reason in why


@pytest.fixture
def lowm(monkeypatch):
    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm import lowm_linear

    state = SimpleNamespace(times={}, cublas={}, prepared=[], runs=[], broadcast=None)
    module = types.ModuleType("sgl_nvidia_mega.qwen3_5.sm103.trt_lowm")

    def spec(*, m, k, n_bf16, fp8_act, tactic_bf16, out_stride=None, **pdl):
        assert out_stride is None
        assert fp8_act == "bf16" and pdl == head_lowm.PDL
        kw, nt, ks, swap, mb = tactic_bf16
        return SimpleNamespace(
            m=m,
            k_warps=kw,
            n_tiles_bf16=nt,
            ksplit=ks,
            swap_ab=bool(swap),
            min_blocks=mb,
        )

    module.spec = spec
    module.prepare = lambda s: state.prepared.append((s.m, head_lowm._tactic(s)))
    module.run = lambda s, out, x16, w_bf16: state.runs.append(
        ("run", s.m, head_lowm._tactic(s))
    )
    from sgl_nvidia_mega.qwen3_5 import sm103

    monkeypatch.setitem(sys.modules, "sgl_nvidia_mega.qwen3_5.sm103.trt_lowm", module)
    monkeypatch.setattr(sm103, "trt_lowm", module, raising=False)
    monkeypatch.setattr(
        lowm_linear, "default_tactic", lambda m, n, k, fp8: (16, 2, 1, 0, 1)
    )

    def timed(fn, reps=50):
        state.runs.clear()
        fn()
        if state.runs:
            _, m, tac = state.runs[0]
            return state.times.get((m, tac), 100.0)
        return state.cublas[state.m]

    monkeypatch.setattr(head_lowm, "_timed", timed)
    real_linear = torch.nn.functional.linear

    def linear(x, w):
        state.m = int(x.shape[0])
        return real_linear(x, w)

    monkeypatch.setattr(torch.nn.functional, "linear", linear)

    def broadcast(objs, src, group):
        if state.broadcast is not None:
            objs[0] = state.broadcast

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    return state


def test_the_race_takes_a_width_only_past_the_margin(lowm):
    lowm.cublas = {1: 10.0, 4: 10.0, 8: 10.0}
    lowm.times = {
        (1, (8, 1, 1, 0, 1)): 9.0,
        (4, (8, 1, 1, 1, 1)): 9.7,
        (8, (16, 2, 1, 0, 1)): 8.0,
    }
    weight = torch.zeros(16, 8, dtype=torch.bfloat16)
    head = head_lowm.HeadLowM(weight, (1, 4, 8, 9), cpu_group="cpu", rank=0)
    assert sorted(head.specs) == [1, 8]
    assert head_lowm._tactic(head.specs[1]) == (8, 1, 1, 0, 1)
    assert head_lowm._tactic(head.specs[8]) == (16, 2, 1, 0, 1)
    assert head.takes(1) and not head.takes(4) and not head.takes(9)


def test_every_rank_runs_rank_zeros_choice(lowm):
    lowm.cublas = {1: 10.0, 2: 10.0}
    lowm.times = {(1, (8, 1, 1, 0, 1)): 1.0, (2, (8, 1, 1, 0, 1)): 1.0}
    lowm.broadcast = [(2, (8, 1, 1, 1, 1))]
    head = head_lowm.HeadLowM(
        torch.zeros(16, 8, dtype=torch.bfloat16), (1, 2), cpu_group="cpu", rank=3
    )
    assert sorted(head.specs) == [2]
    assert head_lowm._tactic(head.specs[2]) == (8, 1, 1, 1, 1)
    assert lowm.prepared[-1] == (2, (8, 1, 1, 1, 1))


@pytest.mark.parametrize("rows, low_m", [(2, True), (3, False)])
def test_the_top1_takes_the_low_m_head_at_its_widths(monkeypatch, rows, low_m):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    calls = []

    class Head:
        def takes(self, m):
            return m == 2

        def __call__(self, h):
            calls.append("low-M")
            return h

    head = Head()
    lm = SimpleNamespace(
        logits_processor=SimpleNamespace(
            _compute_lm_head=lambda h, lm_head, bias: calls.append("cuBLAS") or h
        ),
        lm_head="lm_head",
    )
    parts = SimpleNamespace(
        lm=lm,
        head_lowm=head,
        head_keys=None,
        top1x=lambda logits, tokens, column, nan_key, **outputs: calls.append("top1"),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    layer.top1_into(parts, torch.ones(rows, 4), None, 1, nan_key=0)
    assert calls == ["low-M" if low_m else "cuBLAS", "top1"]
