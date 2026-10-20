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
all-reduce the boundary runs. Slot ``skv2`` (``split_kv.py``) runs the attention's core
through our 2.2 branch's split-KV attention.
"""

from __future__ import annotations

import msgspec

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "Qwen3_5AttentionDecoderLayer",
        "layers",
        "start_layer",
        "end_layer",
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

ATTENTION_SLOTS = ("skv2", "mka", "mka_push")


class Bound(msgspec.Struct, frozen=True):
    """The attention package's build result (``st.bound["attention"]``)."""

    skv2: bool = False  # slot skv2 serves every full-attention layer
    mka: bool = False
    mka_push: bool = False
    reason: str = ""  # why a requested slot is not built


def build(model, ctx) -> Bound:
    """The requested attention slots for every full-attention layer, or why a layer
    cannot take them."""
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS

    if not SLOTS & set(ATTENTION_SLOTS):
        return Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if isinstance(layer, Qwen3_5AttentionDecoderLayer) and "skv2" in SLOTS:
            reason = split_kv.unsupported(layer.attn, ctx.tokens_per_request)
            if reason:
                return Bound(reason=f"layer {i}: {reason}")
    return Bound(skv2="skv2" in SLOTS)


def pool_reasons(view, binding) -> list[str]:
    """Why the attention slots cannot read this forward's KV pools ([] when they can)."""
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    return split_kv.pool_reasons(view.kv, binding.page_size)


def step(layer, x, st):
    """``layer``'s attention on the normed input ``x`` [rows, hidden]: the TP-partial
    output [rows, hidden]."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    q, k, v, gate = layer._prepare_qkv_gate(
        positions=st.positions, hidden_states=x, forward_batch=st.fb
    )
    if st.slots.skv2:
        from sgl_nvidia_mega.qwen3_5.attention import split_kv

        o = split_kv.attend(layer.attn, q, k, v, st)
    else:
        o = layer.attn(q, k, v, st.fb)
    if layer.attn_output_gate:
        o = fused_sigmoid_mul(o, gate, inplace=True)
    out, _ = layer.o_proj(o)
    return out
