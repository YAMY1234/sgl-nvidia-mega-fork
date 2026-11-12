# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpAttentionDecoderLayer.self_attention).
"""The QSA attention layer step of a covered forward: SGLang's ``self_attention``."""

from __future__ import annotations

STOCK = "stock"

SGLANG_READS = {"srt/models/qwen4_exp.py": ("self_attention",)}


def step(executor, layer, x, st):
    """The layer's attention output [rows, hidden] (before the stage's reduction)."""
    st.ran("qsa", STOCK)
    return layer.self_attention(
        positions=st.positions, hidden_states=x, forward_batch=st.fb
    )
