# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The token embedding of a covered step. Stock: SGLang's vocabulary-parallel
lookup (each rank's shard, then the all-reduce)."""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"srt/models/deepseek_v4.py": ("embed_tokens",)}


def embed(st):
    """[rows, dim] for ``st.input_ids``."""
    return st.model.embed_tokens(st.input_ids)
