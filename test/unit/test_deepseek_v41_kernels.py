# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 SM103 kernels' host-side bounds on CPU: the routed-expert record's
tile bound and the served row bound."""

import math

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg


@pytest.mark.parametrize(
    "rows,top_k,experts", [(12, 6, 384), (48, 6, 384), (64, 8, 32), (40, 3, 128)]
)
def test_max_ctas_bounds_any_routing(rows, top_k, experts):
    generator = torch.Generator().manual_seed(rows * top_k)
    for distinct in (top_k, top_k + 1, 2 * top_k, 64, experts):
        distinct = min(distinct, experts)
        pool = torch.randperm(experts, generator=generator)[:distinct]
        ids = torch.stack(
            [
                pool[torch.randperm(distinct, generator=generator)[:top_k]]
                for _ in range(rows)
            ]
        )
        counts = torch.bincount(ids.flatten()).tolist()
        tiles = sum(math.ceil(c / reg.TILE) for c in counts if c)
        assert tiles <= reg.max_ctas(rows, top_k, experts)


def test_the_served_rows_fit_the_wide_route():
    assert reg.MAX_ROWS < reg.SERVED_MAX_ROWS <= reg.WIDE_MAX_ROWS
