# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen2_moe.py (Qwen2MoeSparseMoeBlock.forward).
"""The MoE step of a covered forward: SGLang's ``Qwen2MoeSparseMoeBlock``."""

from __future__ import annotations

STOCK = "stock"

SGLANG_READS = {"srt/models/qwen2_moe.py": ("Qwen2MoeSparseMoeBlock",)}


def step(executor, layer, x, st):
    """The MoE output [rows, hidden] (TP1: nothing left to reduce)."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    st.ran("moe", STOCK)
    if isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return mlp(x, st.fb, defer_finalize=False)
    return mlp(x)
