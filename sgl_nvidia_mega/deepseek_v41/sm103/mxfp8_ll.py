# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer's ``cutedsl_low_latency`` MXFP8 GEMM as the publishing wo_b uses it:
whether this GPU has it, and the GB300 tactics pinned for DeepSeek-V4.1-Flash's TP4
attention projections."""

from __future__ import annotations

import contextlib
import importlib
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

BACKEND = "cutedsl_low_latency"
# FlashInfer's cutedsl_low_latency mm_mxfp8 takes at most 8 rows and K % 128 == 0.
MAX_ROWS = 8
# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (SM, N, K) for 5-8 rows: the TP4 wqkv_a,
# wq_b, wo_b of DSV4.1-Flash on GB300, picked jointly in a 40-layer chain.
PINNED_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (103, 1792, 5120): (128, 10, 1, 4),
    (103, 8192, 1280): (256, 6, 1, 1),
    (103, 5120, 2048): (256, 4, 1, 2),
}
_GEMM_MODULE = "flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm"


@lru_cache(maxsize=1)
def supported() -> bool:
    try:
        import torch
        from flashinfer import mm_mxfp8

        major, minor = torch.cuda.get_device_capability()
        return bool(mm_mxfp8.is_backend_supported(BACKEND, major * 10 + minor))
    except Exception:
        return False


def quantize(x):
    """BF16 rows as MXFP8 with swizzled 128x4 UE8M0 scales, as SGLang's MXFP8 linears
    quantize their input."""
    from flashinfer import SfLayout, mxfp8_quantize

    return mxfp8_quantize(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        backend="cute-dsl",
        sf_swizzle_layout=SfLayout.layout_128x4,
    )


def pinned_tactic(
    *, sm_version: int, n: int, k: int, rows: int
) -> Optional[tuple[int, int, int, int]]:
    # 5-8 rows is FlashInfer's tuning bucket 8 (buckets round up to a power of two).
    if not MAX_ROWS // 2 < rows <= MAX_ROWS:
        return None
    return PINNED_TACTICS.get((sm_version, n, k))


@contextlib.contextmanager
def pinned_tactics(sm_version: int):
    """While tuning, FlashInfer's low-latency tactic list is the pinned tactic alone where
    one applies, so the autotuner caches it under its usual key on every rank."""
    ll_gemm = importlib.import_module(_GEMM_MODULE)
    enumerate_tactics = ll_gemm.autotune_tactics

    def autotune_tactics(problem_sizes_mnkl, *args, **kwargs):
        tactics = enumerate_tactics(problem_sizes_mnkl, *args, **kwargs)
        # The runner computes out.T, so the problem is (N, M, K, L).
        n, rows, k, _ = problem_sizes_mnkl
        pinned = pinned_tactic(sm_version=sm_version, n=n, k=k, rows=rows)
        if pinned is None:
            return tactics
        if pinned not in tactics:
            logger.warning(
                f"NVIDIA megakernel: pinned low-latency MXFP8 tactic {pinned} is not "
                f"offered for N={n} K={k}; autotuning that shape instead."
            )
            return tactics
        return [pinned]

    ll_gemm.autotune_tactics = autotune_tactics
    try:
        yield
    finally:
        ll_gemm.autotune_tactics = enumerate_tactics
