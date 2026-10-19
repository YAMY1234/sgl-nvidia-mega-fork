# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The mHC coefficients at decode rows from one DeepGEMM TF32 prenorm (slot
``hc_prenorm``).

Below 128 rows SGLang predicts each sublayer's (pre, post, comb) from a Triton
tf32x3 split-K over 80 K slices, then a reduce + Sinkhorn row kernel over the same
80 partials. Here one DeepGEMM ``tf32_hc_prenorm_gemm`` over 16 K slices reads the
fp32 mixing weight as TF32 (no compensation term), and SGLang's own reduce +
Sinkhorn kernel sums the 16 partials. Both kernels stay on the stats stream SGLang
forks for them; the coefficients differ from SGLang's by the TF32 rounding of the
weight, within ``spec.NUMERICS`` (``hc_prenorm``).
"""

from __future__ import annotations

import torch

# The SGLang names this module reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/deep_gemm_wrapper/entrypoint.py": ("tf32_hc_prenorm_gemm",),
    "srt/layers/deep_gemm_wrapper/configurer.py": ("ENABLE_JIT_DEEPGEMM",),
    "kernels/ops/layernorm/mhc.py": ("_hc_mix_reduce_sinkhorn_kernel",),
    "srt/models/deepseek_v4_mhc.py": (
        "mix_stats",
        "fn",
        "scale",
        "base",
        "cfg",
        "mult",
        "sinkhorn_iters",
        "rms_eps",
        "eps",
    ),
    "srt/batch_invariant_ops/batch_invariant_ops.py": (
        "is_batch_invariant_mode_enabled",
    ),
    "srt/runtime_context.py": ("get_platform", '"is_blackwell"'),
    # the module alias SGLang's DSV4 model imports too
    "srt/models/deepseek_v4.py": ("mhc",),
}

# The K slices of the prenorm GEMM, and so of the reduce
SLICES = 16
# The widest step it takes; past it SGLang's path
MAX_ROWS = 48


def reason() -> str:
    """Why the prenorm cannot run on this setup ("" when it can)."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
    from sglang.srt.runtime_context import get_platform

    if not get_platform().is_blackwell:
        return "needs SM100-family GPUs"
    if not ENABLE_JIT_DEEPGEMM:
        return "needs DeepGEMM's JIT"
    if is_batch_invariant_mode_enabled():
        return "batch-invariant mode keeps SGLang's split-K"
    return ""


def takes(x: torch.Tensor) -> bool:
    """Whether the prenorm computes the coefficients of these streams (shape and
    dtype only, so fixed per graph key and the same on every rank)."""
    x_flat = x.flatten(1)
    return (
        0 < x_flat.shape[0] <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and x_flat.is_contiguous()
    )


def mix_stats(hc, x: torch.Tensor, stats_stream=None):
    """SGLang's ``mix_stats(hc, x, stats_stream)``, with the prenorm GEMM where it
    ``takes`` the streams: the same stream handling, so callers join it as they join
    SGLang's."""
    from sglang.srt.models import deepseek_v4_mhc as mhc

    if not takes(x):
        return mhc.mix_stats(hc, x, stats_stream)
    x_flat = x.flatten(1)
    if stats_stream is None:
        return _coefficients(hc, x_flat)
    main_stream = torch.cuda.current_stream()
    x.record_stream(stats_stream)
    with torch.cuda.stream(stats_stream):
        coefficients = _coefficients(hc, x_flat)
    for coefficient in coefficients:
        coefficient.record_stream(main_stream)
    return coefficients


def _coefficients(hc, x_flat: torch.Tensor):
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    cfg = hc.cfg
    rows, k = x_flat.shape
    mult = cfg.mult
    mix = (2 + mult) * mult
    device = x_flat.device
    part_mix = torch.empty((SLICES, rows, mix), dtype=torch.float32, device=device)
    part_sq = torch.empty((SLICES, rows), dtype=torch.float32, device=device)
    tf32_hc_prenorm_gemm(x_flat, hc.fn, part_mix, part_sq, SLICES)
    pre = torch.empty((rows, mult), dtype=torch.float32, device=device)
    post = torch.empty_like(pre)
    comb = torch.empty((rows, mult, mult), dtype=torch.float32, device=device)
    _hc_mix_reduce_sinkhorn_kernel[(rows,)](
        part_mix,
        part_sq,
        hc.scale.float().contiguous(),
        hc.base.float().contiguous(),
        pre,
        post,
        comb,
        rows,
        1.0 / k,
        cfg.rms_eps,
        MIX=mix,
        HC=mult,
        NUM_SLICES=SLICES,
        ITERS=cfg.sinkhorn_iters,
        EPS=cfg.eps,
        num_warps=1,
    )
    return pre, post, comb
