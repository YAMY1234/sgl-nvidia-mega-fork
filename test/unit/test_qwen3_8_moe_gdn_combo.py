# SPDX-License-Identifier: Apache-2.0
"""Merged GDN+MoE runtime assembly on CPU, with prebuilt MoE GPU state stubbed."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from sgl_nvidia_mega.qwen3_8 import runtime
from sgl_nvidia_mega.qwen3_8.moe.route import Bound
from sgl_nvidia_mega.qwen3_5.runtime import parity


@pytest.mark.parametrize("gdn_slot", ["gdn_verify", "gdn_mk"])
def test_requested_gdn_and_moe_survive_build_and_plan(gdn_slot):
    config = NS(
        text_config=NS(
            hidden_size=2560,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            num_experts=512,
            num_experts_per_tok=10,
            moe_intermediate_size=640,
            shared_expert_intermediate_size=640,
        )
    )
    linear = NS(
        norm=NS(activation="sigmoid", group_size=None, norm_before_gate=True),
        out_proj=NS(reduce_results=False, input_is_parallel=True, bias=None),
        attn_tp_size=4,
        _fused_in_proj_weight=torch.empty(4120, 2560, dtype=torch.bfloat16),
        attn=NS(layer_id=0),
    )
    model = NS(config=config, model=NS(layers=[NS(linear_attn=linear)]))
    ctx = NS(widths=(4, 8, 12), tokens_per_request=4, deadline_s=1.0, binding=object())
    moe = Bound(route=True, mkm={0: object()})
    requested = frozenset({gdn_slot, "gdn_pdl", "route", "mkm"})
    with (
        patch.object(runtime, "SLOTS", requested),
        patch.object(parity, "ENABLED", False),
        patch("sglang.srt.runtime_context.get_parallel", return_value=NS(tp_size=4)),
        patch("sgl_nvidia_mega.qwen3_8.moe.route.build", return_value=moe) as moe_build,
        patch("sgl_nvidia_mega.qwen3_8.runtime.executor.Qwen38Verify") as executor,
    ):
        assert not runtime.slot_reasons()
        runtime.build(model, ctx, hf_config=config)
    moe_build.assert_called_once_with(
        model, 4, fused=True, shared=False, hf_config=config, tp_size=4
    )
    args, kwargs = executor.call_args
    slots = args[1][4]
    assert slots.route and slots.mkm and slots.gdn_pdl
    assert getattr(slots, gdn_slot)
    assert slots.gdn_mk == (gdn_slot == "gdn_mk")
    assert kwargs["widths"] == (4,)
    assert kwargs["bound"]["moe"] is moe
    assert kwargs["bound"]["gdn"].shape.tokens_per_request == 4
