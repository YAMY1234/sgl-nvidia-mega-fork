# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5AttentionDecoderLayer.self_attention).
"""The gated full attention of a covered target-verify step, over the layer's loaded
weights, SGLang's attention backend and its KV cache.

Stock: the CUDA arm of ``Qwen3_5AttentionDecoderLayer.self_attention`` with SGLang's own
ops in its order: the qkv projection with the fused q / k Gemma RMSNorm + RoPE + gate
split, the attention through the backend (which writes this step's KV rows), the
sigmoid output gate and ``o_proj``. ``step`` returns the TP-partial output, whose
all-reduce the boundary runs.
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "_prepare_qkv_gate",
        "_prepare_qkv_gate(positions)",
        "_prepare_qkv_gate(hidden_states)",
        "_prepare_qkv_gate(forward_batch)",
        "attn",
        "attn_output_gate",
        "o_proj",
    ),
    "kernels/ops/elementwise/elementwise.py": (
        "fused_sigmoid_mul",
        "fused_sigmoid_mul(inplace)",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.self_attention": "db0f4d98e0fb1abf",
}


def step(layer, x, st):
    """``layer``'s attention on the normed input ``x`` [rows, hidden]: the TP-partial
    output [rows, hidden]."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    q, k, v, gate = layer._prepare_qkv_gate(
        positions=st.positions, hidden_states=x, forward_batch=st.fb
    )
    o = layer.attn(q, k, v, st.fb)
    if layer.attn_output_gate:
        o = fused_sigmoid_mul(o, gate, inplace=True)
    out, _ = layer.o_proj(o)
    return out
