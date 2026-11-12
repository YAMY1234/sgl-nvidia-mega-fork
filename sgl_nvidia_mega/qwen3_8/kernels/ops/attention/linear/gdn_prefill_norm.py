# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# From SGLang python/sglang/kernels/ops/attention/fla/layernorm_gated.py.
"""The gated RMSNorm launch (``_layer_norm_fwd``) with the extend-sized tile: 32 rows
and 4 warps a program for single-group 128-wide inputs (the GDN output norm).

Byte-identical to the source: ``_norm_retile_cfg`` and
``_layer_norm_fwd``. Changed: SGLANG_OPT_GDN_PREFILL_NORM_RETILE is the ``retile``
argument of both; the unchanged kernel and row heuristic are SGLang's
(``_layer_norm_fwd_1pass_kernel``, ``calc_rows_per_block``)."""

import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional

import torch
import triton
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.fla.layernorm_gated import (
    _layer_norm_fwd_1pass_kernel,
    calc_rows_per_block,
)
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils import cdiv, device_context

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "_layer_norm_fwd_1pass_kernel",
        "calc_rows_per_block",
    ),
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/utils/common.py": ("cdiv",),
    "srt/utils/device.py": ("device_context",),
}

# Extend-sized single-group 128-wide inputs (the GDN output norm) take 32 rows / 4
# warps per program instead of 1 warp x <= 4 rows. Each row keeps the stock 16-lane x
# 8-element reduction, so only the number of programs changes.
NORM_RETILE_MIN_ROWS = 32768
NORM_RETILE_CFG = (32, 4)  # (ROWS_PER_BLOCK, num_warps)


def _norm_retile_cfg(
    M: int, block_n: int, ngroups: int, quant_heads: int, retile: bool = False
) -> Optional[tuple]:
    if (
        retile
        and not quant_heads
        and ngroups == 1
        and block_n == 128
        and M >= NORM_RETILE_MIN_ROWS
        and not is_batch_invariant_mode_enabled()
    ):
        _log_norm_retile_once()
        return NORM_RETILE_CFG
    return None


@lru_cache(maxsize=1)
def _log_norm_retile_once() -> None:
    logger.info(
        "gated RMSNorm: launch %s for >= %d rows of 128 (SGLANG_OPT_GDN_PREFILL_NORM_RETILE)",
        NORM_RETILE_CFG,
        NORM_RETILE_MIN_ROWS,
    )


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    quant_heads: int = 0,
    launch_cfg: Optional[tuple] = None,
    retile: bool = False,
):
    """launch_cfg: (rows per program, num_warps); None keeps the default launch
    (or, with retile, the extend-sized tile where _norm_retile_cfg takes it)."""
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    z_is_3d = z is not None and z.ndim == 3
    if z is not None:
        assert z.stride(-1) == 1
        if z_is_3d:
            assert z.shape[0] * z.shape[1] == M
            assert z.shape[2] == N
        else:
            assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count
    rows_per_block = calc_rows_per_block(M, x.device)
    quant = {}
    # quant_heads: one block per token, same per-warp row split (rows round identically), FP8 (q, scale) out
    if quant_heads:
        assert ngroups == 1 and M % quant_heads == 0
        # at most 1024 threads (16 wave64s) per block; each row still reduces inside one warp
        num_warps = min(num_warps * quant_heads // rows_per_block, 16)
        rows_per_block = quant_heads
        q = torch.empty(
            (M // quant_heads, quant_heads * N),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        s = torch.empty((M // quant_heads, 1), dtype=torch.float32, device=x.device)
        quant = {"Q": q, "S": s, "QUANT": True}
    if launch_cfg is None:
        launch_cfg = _norm_retile_cfg(M, BLOCK_N, ngroups, quant_heads, retile)
    if launch_cfg is not None:
        rows_per_block, num_warps = launch_cfg
    # Update grid to use rows_per_block
    grid = (cdiv(M, rows_per_block), ngroups)
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    # Workaround for PyTorch <= 2.12: torch.xpu.device is not Dynamo-compatible
    # in that release — it creates a DynamoConfigPatchProxy that
    # SourcelessBuilder cannot wrap, causing a hard error under
    # torch.compile(fullgraph=True).  The device context is a functional no-op
    # for Triton kernel launches (device is determined by the tensor, not the
    # surrounding context), so we simply skip it when Dynamo is tracing.
    # PyTorch main already has the proper fix (XPUDeviceVariable registered in
    # torch/_dynamo/variables/ctx_manager.py analogous to CUDADeviceVariable).
    # TODO: remove this branch once we upgrade from PyTorch 2.12.
    device_ctx = (
        nullcontext()
        if x.device.type == "xpu" and torch.compiler.is_compiling()
        else device_context(x.device)
    )
    with device_ctx:
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None and not z_is_3d else 0,
            z.stride(0) if z_is_3d else 0,
            z.stride(1) if z_is_3d else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            ROWS_PER_BLOCK=rows_per_block,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            Z_IS_3D=z_is_3d,
            Z_HEADS=z.shape[1] if z_is_3d else 1,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
            ACTIVATION=activation,
            **pdl_kwargs,
            **quant,
        )
    return ((q, s) if quant_heads else out), mean, rstd
