# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention step over an ``MQALayer``'s weights, SGLang's attention backend
and its caches: projections, KV / compressor / indexer stores, sparse decode,
inverse RoPE + ``wo_a``, ``wo_b`` (``step.py``).

``step`` returns the attention output in the form ``boundary.post(layer, "attn")``
takes: SGLang's ``AttnOutput``, ``wo_b``'s all-reduce deferred to the post. A layer
the step does not cover runs SGLang's own attention, and build logs why.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sgl_nvidia_mega.deepseek_v41.attention import chain
from sgl_nvidia_mega.deepseek_v41.attention.step import AttentionStep, disable_reason

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "self_attn",
        "maybe_use_decode_attn_tp",
        "_run_boundary_attention",
        "layer_id",
    ),
}


def build(model, ctx, group) -> dict[int, AttentionStep]:
    """Each covered layer's ``AttentionStep`` by layer id; the result is
    ``st.bound["attention"]``."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    m = model.model
    chain_on = "attn" in SLOTS
    steps, skipped = {}, []
    kinds: Counter = Counter()
    for i in range(m.start_layer, m.end_layer):
        attn = m.layers[i].self_attn
        reason = disable_reason(attn)
        if reason:
            skipped.append(reason)
            continue
        steps[attn.layer_id] = step_ = AttentionStep(attn, chain_ops=chain_on)
        if chain_on:
            chain.precompile(attn, ctx.widths)
            kinds[_chain_kind(attn, step_)] += 1
    logger.info(
        "NVIDIA megakernel: attention step in %d layers (%s)%s",
        len(steps),
        "; ".join(f"{n} x {k}" for k, n in sorted(kinds.items())) or "stock",
        f"; SGLang's in the rest ({'; '.join(skipped)})" if skipped else "",
    )
    return steps


def _chain_kind(attn, step_: AttentionStep) -> str:
    """One layer's chain decisions, as build logs them."""
    parts = [f"ratio {attn.compress_ratio}"]
    parts.append("wq_b low-latency 1-48" if step_.chain[0].wq_b_ll else "wq_b SGLang's")
    parts.append("swap-AB 1-24" if step_.swapab else "no swap-AB")
    parts.append("fused wo_a 1-48" if attn.use_fused_wo_a else "wo_a SGLang's")
    return ", ".join(parts)


def step(layer, x, st) -> Any:
    """``layer``'s attention on the normed input ``x`` [rows, dim]; reads
    ``st.x_quant``, ``st.positions``, ``st.fb``."""
    attention = st.bound["attention"].get(layer.self_attn.layer_id)
    if attention is None:
        with layer.self_attn.maybe_use_decode_attn_tp(st.fb):
            return layer._run_boundary_attention(x, st.positions, st.fb, st.x_quant)
    # hop A: wo_b publishes into the mailbox the attention post collects
    publish = st.bound["boundary"].hop_a.publish_wo_b if st.slots.hop_a else None
    return attention.run(x, st.positions, st.fb, st.x_quant, publish=publish)
