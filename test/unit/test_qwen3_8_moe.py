# SPDX-License-Identifier: Apache-2.0
"""FN MoE CPU contracts: HF sizing, row coverage and the native FFN exit."""

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from contextlib import contextmanager

import pytest
import torch
from sgl_nvidia_mega.qwen3_8 import plan
from sgl_nvidia_mega.qwen3_8.moe.config import Shape, config_disable_reasons
from sgl_nvidia_mega.qwen3_8.moe.step import ffn
from sglang.srt.layers.layer_boundary.residual.stream import Contribution, OwedOutput


def config(**kw):
    return NS(
        text_config=NS(
            hidden_size=2560,
            num_experts=512,
            num_experts_per_tok=10,
            moe_intermediate_size=640,
            shared_expert_intermediate_size=640,
            **kw,
        )
    )


def test_hf_logical_and_native_storage_are_distinct():
    shape = Shape.from_config(config(), 4, 4)
    assert shape.local_intermediate == 160
    assert shape.storage_intermediate == 192
    assert not config_disable_reasons(config(), 4, 4)
    assert config_disable_reasons(config(), 1, 4)  # TP1 engine is not qualified
    other = config()
    other.text_config.hidden_size = 4096
    assert "H4096" in " ".join(config_disable_reasons(other, 4, 4))


def test_route_and_mkm_coverage_has_native_wide_fallback(monkeypatch):
    facts = plan.Facts(route=True, mkm=True)
    assert plan.slots(4, facts).mkm
    # The shared multi-request plan leaves all unqualified wider slots native.
    assert plan.slots(8, facts) == plan.Slots()
    assert plan.slots(16, facts) == plan.Slots()
    # Qualifying the router alone must not implicitly enable wide experts.
    monkeypatch.setattr(plan, "ROUTE_MAX_ROWS", 16)
    assert plan.slots(8, facts).route and not plan.slots(8, facts).mkm
    assert not plan.slots(4, plan.Facts()).route


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("communicator", [False, True])
def test_ffn_exit_keeps_prepare_finalize_and_finish_order(defer, opaque, communicator):
    order = []
    prepared, routed, done = object(), object(), object()
    tensor = torch.empty(4, 2560, dtype=torch.bfloat16)
    hidden = OwedOutput(Contribution(tensor, None)) if opaque else tensor

    def prepare(x, fb):
        assert x is hidden
        order.append("prepare")
        return prepared

    @contextmanager
    def exit_(fb):
        order.append("exit")
        yield NS(defer_moe_finalize=defer, finish=finish)
        order.append("join")

    def finish(x):
        assert x is routed
        order.append("finish")
        return done

    layer = NS(
        _decode_moe_comm=object() if communicator else None,
        ffn_boundary=NS(prepare=prepare, exit=exit_),
        _run_ffn_stage=Mock(),
    )
    st = NS(
        slots=plan.Slots(route=True, mkm=True),
        fb=object(),
        input_ids=torch.empty(4, dtype=torch.int64),
    )

    def moe(layer_, x, state, *, defer):
        assert x is prepared and state is st
        order.append("moe")
        return routed

    with patch("sgl_nvidia_mega.qwen3_8.moe.step.step", side_effect=moe) as run:
        assert ffn(layer, hidden, st) is done
    assert run.call_args.kwargs["defer"] is defer
    assert order == ["prepare", "exit", "moe", "join", "finish"]
    layer._run_ffn_stage.assert_not_called()


def test_native_single_row_decode_communicator_is_preserved():
    layer = NS(_decode_moe_comm=object(), _run_ffn_stage=Mock(return_value="native"))
    st = NS(
        slots=plan.Slots(route=True, mkm=True),
        fb=object(),
        input_ids=torch.empty(1, dtype=torch.int64),
    )
    hidden = OwedOutput(Contribution(torch.empty(1, 2560, dtype=torch.bfloat16), None))
    assert ffn(layer, hidden, st) == "native"
    layer._run_ffn_stage.assert_called_once_with(hidden, st.fb)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shared_slot", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_engaged_marks_only_dispatched_moe_slots(fused, shared_slot, defer):
    from sgl_nvidia_mega.qwen3_8.moe import step as module
    from sgl_nvidia_mega.qwen3_8.runtime import engaged
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )

    x = torch.empty(4, 2560, dtype=torch.bfloat16)
    shared = torch.empty_like(x)
    routed = torch.empty_like(x)
    raw = object()
    events = []
    mlp = NS(
        alt_stream=None,
        shared_expert=object(),
        shared_expert_gate=NS(weight=torch.empty(1, 2560, dtype=torch.bfloat16)),
        _forward_shared_experts=Mock(return_value=shared),
        _gate_shared_output_out_of_place=Mock(return_value=shared),
    )
    st = NS(
        slots=plan.Slots(route=True, mkm=fused, route_shared=shared_slot),
        bound={"moe": NS(mkm={0: object()})},
        scratch={},
    )

    def producer(*args):
        events.append("producer")
        return raw

    def shared_producer(*args):
        return producer(), object()

    def experts(*args, **kwargs):
        events.append("experts")
        assert kwargs["fused"] is fused
        return routed

    def mark(state, slot):
        assert state is st
        assert "producer" in events
        if slot == "mkm":
            assert "experts" in events
        events.append(slot)

    with (
        patch.object(module.route, "produce", side_effect=producer),
        patch.object(module.route, "produce_shared", side_effect=shared_producer),
        patch.object(module.route, "shared_expert_tail", return_value=shared),
        patch.object(module.route, "routed", side_effect=experts),
        patch.object(engaged, "mark", side_effect=mark),
        patch("sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add"),
        patch.object(MoeDeferredFinalize, "from_flashinfer", return_value="deferred"),
    ):
        result = module.step(NS(mlp=mlp), x, st, defer=defer)
    assert events.count("route") == 1
    assert events.count("route_shared") == int(shared_slot)
    assert events.count("mkm") == int(fused)
    if defer:
        assert result == "deferred"
    else:
        assert result.shape == (4, 2560)
