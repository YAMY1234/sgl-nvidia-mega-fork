# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GDN layer step of a covered forward: SGLang's ``GatedDeltaNet.forward``."""

from __future__ import annotations

STOCK = "stock"

SGLANG_READS = {"srt/models/qwen3_5.py": ("linear_attn",)}


def step(executor, layer, x, st):
    """The layer's mixer output ([rows, hidden], before the stage's reduction)."""
    st.ran("gdn", STOCK)
    return layer.linear_attn(x, st.fb)
