# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The runtime: the executor of a covered step (``executor``), its per-step state
(``state``), the embedding (``embedding``), and ``build``, which the spec calls
once the weights are loaded and before the KV cache is sized."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel", '"tp_group"', '"attn_dp_size"'),
    "srt/models/deepseek_v4.py": (
        "DeepseekV4CudaDecoderLayer",
        "hc_pre_from_prev_sublayer",
        "model",
        "start_layer",
        "end_layer",
        "layers",
    ),
}


def model_disable_reasons(model, hf_config) -> list[str]:
    """What the framework does not check: the layer loop the executor follows."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4CudaDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    m = model.model
    reasons = []
    if not m.hc_pre_from_prev_sublayer:
        reasons.append("needs the mHC pre-mix carried from the previous sublayer")
    if get_parallel().attn_dp_size > 1:
        reasons.append("attention DP: the covered step reads replicated token ids")
    other = [
        i
        for i in range(m.start_layer, m.end_layer)
        if not isinstance(m.layers[i], DeepseekV4CudaDecoderLayer)
    ]
    if other:
        reasons.append(f"layers {other}: not SGLang's CUDA decoder layer")
    return reasons


def build(model, hf_config, ctx):
    """Each package's build (weights bound, buffers allocated, kernels compiled),
    the slot plan per width, and the executor."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
    from sgl_nvidia_mega.deepseek_v41.runtime.executor import PACKAGES, Executor

    group = get_parallel().tp_group
    bound = {name: pkg.build(model, ctx, group) for name, pkg in PACKAGES.items()}
    slots = slot_plan.plan(ctx.widths, slot_plan.Facts())
    logger.info(
        "sgl-nvidia-mega deepseek_v41: widths %s; every step function stock",
        list(ctx.widths),
    )
    return Executor(model, slots, bound, streams={})
