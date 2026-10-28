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


def test_route_and_mkm_coverage_has_native_wide_fallback():
    facts = plan.Facts(route=True, mkm=True)
    assert plan.slots(4, facts).mkm
    assert plan.slots(8, facts).route and not plan.slots(8, facts).mkm
    assert not plan.slots(4, plan.Facts()).route


@pytest.mark.parametrize("defer", [False, True])
def test_ffn_exit_keeps_prepare_finalize_and_finish_order(defer):
    order = []
    prepared, routed, done = object(), object(), object()
    hidden = torch.empty(4, 2560, dtype=torch.bfloat16)

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
        _decode_moe_comm=None,
        ffn_boundary=NS(prepare=prepare, exit=exit_),
        _run_ffn_stage=Mock(),
    )
    st = NS(slots=plan.Slots(route=True, mkm=True), fb=object())

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
    st = NS(slots=plan.Slots(route=True, mkm=True), fb=object())
    hidden = torch.empty(1, 2560, dtype=torch.bfloat16)
    assert ffn(layer, hidden, st) == "native"
    layer._run_ffn_stage.assert_called_once_with(hidden, st.fb)
