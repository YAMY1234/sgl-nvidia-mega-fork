# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import draftmoe

H, I = 4096, 512
TEP8 = ("tep8", topology.GB300)


def test_the_options_are_numeric_rows_of_the_tep8_plan_only():
    for name in (draftmoe.ENGINE, draftmoe.WIDE, draftmoe.SHARED):
        assert name in runtime.MEASURED_TEP8_OPTIONS and name in runtime.OPTION_ROWS
        assert name in runtime.NUMERICS_CHANGE
        assert name not in topology.SLOT_TOPOLOGIES
        assert name in runtime.MEASURED_PLAN[TEP8]
        assert name not in runtime.MEASURED_PLAN[("tp8", topology.GB300)]
    assert draftmoe.ROUTER in runtime.DRAFT_SLOTS
    assert draftmoe.ROUTER in runtime.MEASURED_DRAFT
    assert draftmoe.ROUTER in runtime.NUMERICS_CHANGE
    assert topology.SLOT_TOPOLOGIES[draftmoe.ROUTER] == {"tp4", "tp8", "tep8"}


def _experts(**over):
    from sglang.srt.layers.moe.utils import RoutingMethodType

    e = SimpleNamespace(
        moe_runner_config=SimpleNamespace(
            routing_method_type=RoutingMethodType.Renormalize,
            routed_scaling_factor=None,
        ),
        num_experts=512,
        num_local_experts=128,
        w13_weight=torch.empty(
            128, H // 64, 2 * I, 64, dtype=torch.bfloat16, device="meta"
        ),
        w2_weight=torch.empty(128, I // 64, H, 64, dtype=torch.bfloat16, device="meta"),
    )
    topk = SimpleNamespace(
        topk_config=SimpleNamespace(
            top_k=10, correction_bias=None, num_expert_group=None, topk_group=None
        )
    )
    for key, value in over.items():
        target = topk.topk_config if hasattr(topk.topk_config, key) else e
        if hasattr(e.moe_runner_config, key):
            target = e.moe_runner_config
        setattr(target, key, value)
    return e, topk


def test_the_engine_takes_the_tep8_draft_experts():
    assert draftmoe.engine_unsupported(*_experts()) == ""


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"routing_method_type": "TopK"}, "routing"),
        ({"correction_bias": torch.zeros(512)}, "a routing bias"),
        ({"num_expert_group": 8}, "grouped routing"),
        ({"routed_scaling_factor": 2.5}, "routed scaling"),
        ({"top_k": 8}, "top-8 of 512 experts"),
        ({"num_local_experts": 64}, "64 local"),
        (
            {
                "w13_weight": torch.empty(
                    128, 64, 1024, 64, dtype=torch.float32, device="meta"
                )
            },
            "needs BlockMajorK BF16",
        ),
        (
            {
                "w2_weight": torch.empty(
                    128, 8, 2048, 64, dtype=torch.bfloat16, device="meta"
                )
            },
            "expert weights",
        ),
    ],
)
def test_the_engine_refuses_other_experts(over, reason):
    assert reason in draftmoe.engine_unsupported(*_experts(**over))


def _linear(rows, cols, **extra):
    return SimpleNamespace(
        weight=torch.zeros(rows, cols, dtype=torch.bfloat16), bias=None, **extra
    )


def _mlp_shared(**over):
    from sglang.srt.layers.activation import SiluAndMul

    se = SimpleNamespace(
        gate_up_proj=_linear(2 * 128, 4096),
        down_proj=_linear(4096, 128, reduce_results=False),
        act_fn=SiluAndMul(),
    )
    for key, value in over.items():
        setattr(se, key, value)
    return SimpleNamespace(shared_expert=se)


def test_the_shared_launches_take_the_tep8_shared_expert():
    assert draftmoe.shared_unsupported(_mlp_shared()) == ""


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"down_proj": _linear(4096, 128, reduce_results=True)}, "all-reduces"),
        ({"gate_up_proj": _linear(2 * 96, 4096)}, "shapes"),
        ({"act_fn": object()}, "not SiluAndMul"),
        ({"down_proj": None}, "not a gate_up / down MLP"),
    ],
)
def test_the_shared_launches_refuse_other_experts(over, reason):
    assert reason in draftmoe.shared_unsupported(_mlp_shared(**over))


def test_paired_gate_up_interleaves_eight_gate_and_eight_up_rows():
    w = torch.arange(32 * 2).view(32, 2).to(torch.bfloat16)
    got = draftmoe.paired_gate_up(w)
    gate, up = w[:16], w[16:]
    assert torch.equal(got[:8], gate[:8]) and torch.equal(got[8:16], up[:8])
    assert torch.equal(got[16:24], gate[8:]) and torch.equal(got[24:], up[8:])


# ------------------------------------------------------------------- the block


class _Mlp:
    def __init__(self, log):
        self.log, self.alt_stream, self.supports_deferred_finalize = log, None, True

        def topk(x, logits):
            log.append("topk")
            return "topk output"

        self.topk = topk
        self.experts = SimpleNamespace(
            forward_deferred_finalize=lambda x, t: (
                log.append(("flashinfer", t)) or _flashinfer(x.shape[0])
            )
        )

    def gate(self, x):
        self.log.append("gate")
        return torch.zeros(x.shape[0], 512, dtype=torch.bfloat16), None

    def _forward_shared_experts(self, x, apply_gate=True):
        assert not apply_gate
        self.log.append("shared")
        return torch.ones_like(x)

    def _gate_shared_output_out_of_place(self, x, shared):
        self.log.append("shared gate")
        return shared * 2


def _flashinfer(m):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        _make_deferred_finalize_output,
    )

    return _make_deferred_finalize_output(
        [
            torch.zeros(m * 10, 8, dtype=torch.bfloat16),
            torch.zeros(m, 10, dtype=torch.bfloat16),
            torch.zeros(m * 10, dtype=torch.int32),
        ],
        top_k=10,
    )


class _Piece:
    def __init__(self, log, name, rows, out):
        self.log, self.name, self.rows, self.out = log, name, rows, out

    def takes(self, *args):
        x = args[0]
        rows = x if isinstance(x, int) else int(x.shape[0])
        return rows in self.rows

    def __call__(self, x):
        self.log.append(self.name)
        return self.out(x)

    def run(self, x, logits):
        self.log.append(self.name)
        return self.out(x)


@pytest.fixture
def stock(monkeypatch):
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(TopKOutputChecker, "format_is_bypassed", lambda t: True)


@pytest.mark.parametrize("rows, pieces", [(2, True), (3, False)])
def test_the_block_takes_each_piece_at_its_widths(stock, rows, pieces):
    log = []
    mlp = _Mlp(log)
    engine = _Piece(
        log,
        "engine",
        {2},
        lambda x: [
            torch.zeros(x.shape[0] * 10, 8, dtype=torch.bfloat16),
            torch.zeros(x.shape[0], 10, dtype=torch.bfloat16),
            torch.zeros(x.shape[0] * 10, dtype=torch.int32),
        ],
    )
    shared = _Piece(log, "shared low-M", {2}, lambda x: torch.full_like(x, 3.0))
    router = _Piece(
        log,
        "router low-M",
        {2},
        lambda x: torch.zeros(x.shape[0], 512, dtype=torch.bfloat16),
    )
    block = draftmoe.DraftMoeBlock(mlp, engine=engine, shared=shared, router=router)
    x = torch.ones(rows, 8, dtype=torch.bfloat16)
    owed = block.deferred(x)
    if pieces:
        assert log == ["shared low-M", "shared gate", "router low-M", "engine"]
        assert torch.equal(owed.gated_shared_output, torch.full_like(x, 6.0))
    else:
        assert log == [
            "shared",
            "shared gate",
            "gate",
            "topk",
            ("flashinfer", "topk output"),
        ]
        assert torch.equal(owed.gated_shared_output, torch.full_like(x, 2.0))
    assert owed.m == rows
    assert owed.routed_output.shape == (rows * 10, 8)


def test_without_pieces_the_block_runs_sglangs_ops(stock):
    log = []
    block = draftmoe.DraftMoeBlock(_Mlp(log))
    block.deferred(torch.ones(2, 8, dtype=torch.bfloat16))
    assert log == [
        "shared",
        "shared gate",
        "gate",
        "topk",
        ("flashinfer", "topk output"),
    ]


def test_the_block_refuses_an_empty_step():
    with pytest.raises(RuntimeError, match="M=0"):
        draftmoe.DraftMoeBlock(_Mlp([])).deferred(torch.ones(0, 8))


def test_the_draft_layer_hands_its_deferred_finalize_to_the_block():
    from sgl_nvidia_mega.qwen3_5.draft import layer

    calls = []
    block = SimpleNamespace(deferred=lambda x: calls.append("block") or "owed")

    def mlp(x, defer_finalize=False):
        calls.append(("sglang", defer_finalize))
        return "sglang owed"

    parts = SimpleNamespace(moe_block=block, layer=SimpleNamespace(mlp=mlp))
    assert layer._deferred(parts, None) == "owed"
    parts.moe_block = None
    assert layer._deferred(parts, None) == "sglang owed"
    assert calls == ["block", ("sglang", True)]


# ------------------------------------------------------------------- the build


@pytest.fixture
def plan(monkeypatch):

    def at(key, *, slots=()):
        monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
        monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
        monkeypatch.setattr(runtime, "MEASURED_KEY", key)
        seen = []
        monkeypatch.setattr(
            runtime,
            "slot_supported",
            lambda name, why, **_: seen.append((name, why)) or not why,
        )
        return seen

    return at


def _parts(slots=frozenset()):
    experts, topk = _experts()
    mlp = SimpleNamespace(
        experts=experts,
        topk=topk,
        supports_deferred_finalize=True,
        gate=SimpleNamespace(
            weight=torch.zeros(512, 64, dtype=torch.bfloat16), bias=None
        ),
        **vars(_mlp_shared()),
    )
    return SimpleNamespace(
        layer=SimpleNamespace(mlp=mlp),
        binding=SimpleNamespace(tp_cpu_group="cpu", tp_rank=0),
        slots=frozenset(slots),
    )


@pytest.mark.parametrize("tep8, want", [(True, True), (False, False)])
def test_the_engine_and_shared_launches_build_at_tep8_only(
    monkeypatch, plan, tep8, want
):
    seen = plan(TEP8)
    monkeypatch.setattr(draftmoe, "_tep8", lambda: tep8)
    built = []
    engine = SimpleNamespace(wide=True)
    monkeypatch.setattr(
        draftmoe,
        "DraftMoeEngine",
        lambda experts, wide: built.append(("engine", wide)) or engine,
    )
    monkeypatch.setattr(
        draftmoe, "DraftSharedLowM", lambda mlp: built.append("shared") or "s"
    )
    parts = _parts()
    assert draftmoe.wanted(parts.slots)
    block = draftmoe.build(parts)
    if want:
        assert built == [("engine", True), "shared"]
        assert block.engine is engine and block.shared == "s" and block.router is None
        assert seen == [(draftmoe.ENGINE, ""), (draftmoe.SHARED, "")]
    else:
        assert built == [] and block is None
        assert seen == [(draftmoe.ENGINE, "not TEP8"), (draftmoe.SHARED, "not TEP8")]


def test_off_the_measured_stack_nothing_is_wanted(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", raising=False)
    monkeypatch.setattr(runtime, "MEASURED_KEY", TEP8)
    assert not draftmoe.wanted(frozenset({"draft_loop"}))
    assert draftmoe.wanted(frozenset({"draft_loop", draftmoe.ROUTER}))


@pytest.mark.parametrize("taken", [True, False])
def test_the_router_race_keeps_the_slot_only_where_it_took_a_width(
    monkeypatch, plan, taken
):
    from sgl_nvidia_mega.qwen3_5.draft import head_lowm

    plan(("tp8", topology.GB300))
    raced = []

    class Race:
        def __init__(self, weight, rows, **kwargs):
            raced.append((tuple(rows), kwargs["max_rows"], kwargs["candidates"]))
            self.specs = {1: "spec"} if taken else {}

    monkeypatch.setattr(head_lowm, "HeadLowM", Race)
    parts = _parts({"draft_loop", draftmoe.ROUTER})
    block = draftmoe.build(parts)
    assert raced == [
        (tuple(range(1, 17)), draftmoe.MAX_ROWS, draftmoe.ROUTER_CANDIDATES)
    ]
    if taken:
        assert block is not None and block.router is not None
        assert draftmoe.ROUTER in parts.slots
    else:
        assert block is None
        assert parts.slots == {"draft_loop"}
