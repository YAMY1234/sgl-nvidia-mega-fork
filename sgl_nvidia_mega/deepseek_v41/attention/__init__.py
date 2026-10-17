# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention step over an ``MQALayer``'s weights, SGLang's attention backend
and its caches (owner: attention lane): projections, KV / compressor / indexer
stores, sparse decode, inverse RoPE + ``wo_a``, ``wo_b``.

``step`` returns the attention output in the form ``boundary.post(layer, "attn")``
takes (today SGLang's ``AttnOutput`` / unreduced output: ``wo_b`` is deferred).

SKELETON: ``step`` calls ``layer._run_boundary_attention``, as
``DeepseekV4CudaDecoderLayer.forward`` does. SGLang also keeps the attention post
inside ``maybe_use_decode_attn_tp``; that context only acts under
context-parallel decode, which the spec refuses.
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "self_attn",
        "maybe_use_decode_attn_tp",
        "_run_boundary_attention",
    ),
}


def build(model, ctx, group):
    """Bind weights, allocate buffers, compile; the result is ``st.bound["attention"]``."""
    return None


def step(layer, x, st):
    """``layer``'s attention on the normed input ``x`` [rows, dim]; reads
    ``st.x_quant``, ``st.positions``, ``st.fb``."""
    with layer.self_attn.maybe_use_decode_attn_tp(st.fb):
        return layer._run_boundary_attention(x, st.positions, st.fb, st.x_quant)
