# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang's python/sglang/kernels/ops/attention/dsv4/kv_layout.py (Apache-2.0).
"""The paged V4 fp8 KV-cache layout the swap-AB decode attention reads.

A page block stores ``page_size`` data rows followed by ``page_size`` scale rows;
the reader identifies the layout by the bytes per token (the last dim of the
``(num_pages, page_size, 1, bytes_per_token)`` view).
"""

from __future__ import annotations

import enum


class KVLayout(str, enum.Enum):
    # 448 fp8 nope + 64 bf16 rope, 7 ue8m0 scales (+1 pad) per 64 values.
    V4 = "v4"

    @property
    def data_bytes(self) -> int:
        return 576

    @property
    def scale_bytes(self) -> int:
        return 8

    @property
    def tile_size(self) -> int:
        """Values sharing one scale."""
        return 64

    @property
    def bytes_per_token(self) -> int:
        return self.data_bytes + self.scale_bytes
