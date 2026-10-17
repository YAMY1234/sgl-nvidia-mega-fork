# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Engram layers: table lookup, WKV GEMM and gate, before the layer's attention.

``hashes`` computes the step's token hashes once; ``apply`` runs one Engram layer
on the materialized streams and writes its output back as the residual. Stock:
``Engram.forward`` (lookup -> WKV -> ``engram_gate``) from its own parts, and the
DeepSeek-V4.1 image-token rows keep their residual, as SGLang's layer loop does.
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.deepseek_v41 import boundary

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/engram.py": (
        "engram_gate",
        "embed",
        "forward(cp_all_tokens)",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "layer_hash_index",
    ),
    "srt/models/deepseek_v4.py": ("engram_hasher", "engram", "config"),
    "srt/configs/deepseek_v41.py": ("vision_n_layers",),
}


def build(model, ctx, group):
    """The Engram prefetch's buffers and plane, when it applies; ``st.bound["engram"]``."""
    return None


def hashes(st) -> None:
    """The step's token hashes [rows, Engram layers, hash columns], or None."""
    hasher = st.model.engram_hasher
    st.hash_ids = hasher(st.input_ids, st.fb) if hasher is not None else None


def apply(layer, st) -> None:
    """``layer.engram`` on ``st.state.residual`` (materialized by the runtime)."""
    from sglang.srt.layers.engram import engram_gate

    engram, config = layer.engram, st.model.config
    x = st.state.residual
    ids = st.hash_ids[:, engram.layer_hash_index]
    emb = engram.embed(ids, st.fb, cp_all_tokens=False)
    kv, _ = engram.wkv(emb.flatten(-2))
    out = engram_gate(
        x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value
    )
    if config.model_type == "deepseek_v41" and config.vision_n_layers > 0:
        image = st.input_ids == config.image_token_id
        out = torch.where(image[:, None, None], x, out)
    boundary.set_residual(st, out)
