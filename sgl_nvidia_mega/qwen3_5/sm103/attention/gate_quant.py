# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/elementwise/elementwise.py (_fused_sigmoid_mul_kernel) and
# python/sglang/kernels/ops/quantization/fp8_kernel.py (_static_quant_fp8).
"""The attention output gate and ``o_proj``'s static FP8 input quantization in one Triton
launch: the bf16 ``attn * sigmoid(gate)`` of SGLang's ``fused_sigmoid_mul``, then
``static_quant_fp8``'s ``clamp(y * (1 / scale))``, bit for bit the two launches."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_quant_kernel(
    attn_ptr,
    gate_ptr,
    scale_ptr,
    out_ptr,
    seq_lens,
    num_requests,
    min_kv_sum,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
    STOCK_ONLY: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    tl.extra.cuda.gdc_wait()
    if STOCK_ONLY:
        r = tl.arange(0, NB)
        lengths = tl.load(seq_lens + r, r < num_requests, other=0)
        total = tl.max(lengths, 0) if GMAX else tl.sum(lengths, 0)
        if total >= min_kv_sum:
            tl.extra.cuda.gdc_launch_dependents()
            return
    attn = tl.load(attn_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    y = (attn * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    s_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    tl.extra.cuda.gdc_launch_dependents()
    q = tl.clamp(y * s_inv, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(out_ptr + row * hidden + cols, q, mask=mask)


def gate_quant(
    attn: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    min_kv_sum: int = 0,
    gate_max: bool = False,
) -> torch.Tensor:
    """FP8 [rows, hidden] of the gated attention rows ``attn`` [rows, hidden] (bf16, gate
    the same shape) at the static per-tensor ``scale``."""
    rows, hidden = attn.shape
    if out is None:
        out = torch.empty_like(attn, dtype=torch.float8_e4m3fn)
    requests = 0 if seq_lens is None else seq_lens.numel()
    block = 1024
    _gate_quant_kernel[(rows, triton.cdiv(hidden, block))](
        attn,
        gate,
        scale,
        out,
        seq_lens,
        requests,
        min_kv_sum,
        hidden,
        BLOCK=block,
        STOCK_ONLY=seq_lens is not None,
        NB=triton.next_power_of_2(max(requests, 1)),
        GMAX=gate_max,
        num_warps=4,
        launch_pdl=True,
    )
    return out
