# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft stages on SGLang's fused posts, on CPU: the seam probe, the layer
call the routed layer builds, and which stages the draft forward routes."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sglang.srt.models import deepseek_v4_boundary as hc_boundary

from sgl_nvidia_mega.deepseek_v41 import forward

needs_h3 = pytest.mark.skipif(
    "mhc_fusions" not in hc_boundary.LayerCall.__struct_fields__,
    reason="this SGLang lacks LayerCall.mhc_fusions",
)


class _Stop(Exception):
    pass


def _call_of(**kw):
    """The LayerCall run_layer builds, stopped at the attention read."""
    layer = SimpleNamespace(
        attn_boundary=SimpleNamespace(prepare=Mock(side_effect=_Stop))
    )
    with pytest.raises(_Stop):
        forward.run_layer(
            layer, None, "x", "ids", "fb", "ids_g", cross_layer=False, runner=None,
            hop_a=False, moe=None, fast_gate=False, **kw,
        )  # fmt: skip
    return layer.attn_boundary.prepare.call_args.kwargs["read_context"]


@needs_h3
def test_the_seam_probe_follows_the_layer_call():
    assert forward.draft_post_seam() == ""
    fields = tuple(
        f for f in hc_boundary.LayerCall.__struct_fields__ if f != "mhc_fusions"
    )
    older = SimpleNamespace(__struct_fields__=fields)
    with patch.object(hc_boundary, "LayerCall", older):
        assert "LayerCall.mhc_fusions" in forward.draft_post_seam()


@needs_h3
def test_run_layer_sets_the_fusions_only_when_asked():
    unset = _call_of()
    assert unset == hc_boundary.LayerCall("ids", "ids_g", False)
    assert unset.mhc_fusions is None
    assert _call_of(mhc_fusions=True).mhc_fusions is True


@pytest.mark.parametrize("post, seam", [(False, False), (True, False), (True, True)])
def test_the_draft_routes_its_stages_by_the_post_slot(post, seam):
    stages = [Mock(return_value=f"s{i}") for i in range(3)]
    lm = SimpleNamespace(forward_embed=lambda ids: "embed", stages=stages, hc_mult=4)
    slots = {5: SimpleNamespace(routed=False)}
    draft = forward.Draft(lm, slots, runner="runner", post=post, seam=seam)
    state = SimpleNamespace(residual=Mock(), pre="pre")
    recorder = SimpleNamespace(disable_this_region=nullcontext)
    with (
        patch.object(forward, "run_layer", side_effect=lambda s, *a, **k: "r") as run,
        patch("sglang.srt.layers.layer_boundary.residual.batch.start"),
        patch(
            "sglang.srt.layers.layer_boundary.residual.batch.read_output",
            return_value=state,
        ),
        patch("sglang.srt.layers.layer_boundary.residual.batch.clear"),
        patch(
            "sglang.srt.eplb.expert_distribution."
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ),
        patch("sglang.kernels.ops.layernorm.mhc.hc_combine", return_value="x"),
    ):
        plan = SimpleNamespace(key=SimpleNamespace(rows=5))
        draft._forward(plan, "ids", "pos", "fb", None)
    if not post:
        run.assert_not_called()
        assert all(s.call_count == 1 for s in stages)
        return
    assert all(s.call_count == 0 for s in stages)
    assert run.call_count == 3
    for c in run.call_args_list:
        assert c.kwargs["mhc_fusions"] is True
        assert c.kwargs["cross_layer"] is seam
        assert c.kwargs["moe"] is None and not c.kwargs["hop_a"]


@pytest.mark.parametrize("tail_bf16", [False, True])
def test_the_draft_tail_combines_bf16_only_with_its_slot(tail_bf16):
    import torch

    lm = SimpleNamespace(forward_embed=lambda ids: "embed", stages=[], hc_mult=4)
    slots = {5: SimpleNamespace(routed=False)}
    draft = forward.Draft(lm, slots, tail_bf16=tail_bf16)
    residual = torch.ones(5, 4, 8, dtype=torch.bfloat16)
    state = SimpleNamespace(residual=residual, pre="pre")
    recorder = SimpleNamespace(disable_this_region=nullcontext)
    with (
        patch("sglang.srt.layers.layer_boundary.residual.batch.start"),
        patch(
            "sglang.srt.layers.layer_boundary.residual.batch.read_output",
            return_value=state,
        ),
        patch("sglang.srt.layers.layer_boundary.residual.batch.clear"),
        patch(
            "sglang.srt.eplb.expert_distribution."
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ),
        patch("sglang.kernels.ops.layernorm.mhc.hc_combine") as combine,
    ):
        plan = SimpleNamespace(key=SimpleNamespace(rows=5))
        draft._forward(plan, "ids", "pos", "fb", None)
    streams, pre, hc, out_dtype = combine.call_args.args
    assert streams.shape == (5, 32) and (pre, hc, out_dtype) == (
        "pre",
        4,
        torch.bfloat16,
    )
    assert streams.dtype == (torch.bfloat16 if tail_bf16 else torch.float32)
