# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft attention's op slots (``DSparkAttention.qkv_a_op``): wq_a and wkv
as one MXFP8 GEMM, through the draft linear's own SGLang dispatch on a concatenated
weight, so the input is quantized once and the two projections are one launch."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": ("DSparkAttention", "qkv_a_op"),
    "srt/models/deepseek_v4.py": ("wq_a", "wkv"),
    "srt/layers/linear.py": ("quant_method",),
    "srt/layers/quantization/fp8.py": (
        "weight_scale_inv_swizzled",
        "apply(layer)",
        "mxfp8_dense_backend",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "is_flashinfer_cutedsl",
        "is_flashinfer_cutlass",
    ),
    "kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py": (
        "is_unified_kv_triton",
    ),
}

# The SGLang seam the slot needs, probed at build
SGLANG_SEAMS = {"srt/models/deepseek_v4_dspark.py": ("qkv_a_op",)}

_SCALE_ROWS = 128  # the swizzled UE8M0 scales are tiled 128 weight rows at a time


def qkv_a_seam() -> str:
    """Why SGLang cannot take the fused wqkv_a, or "" when it can."""
    from sglang.srt.models.deepseek_v4_dspark import DSparkAttention

    if not hasattr(DSparkAttention, "qkv_a_op"):
        return "this SGLang lacks DSparkAttention.qkv_a_op"
    return ""


def _fusable(wq_a, wkv) -> str:
    """Why wq_a and wkv cannot be one GEMM on SGLang's MXFP8 dispatch, or ""."""
    for name, linear in (("wq_a", wq_a), ("wkv", wkv)):
        scale = getattr(linear, "weight_scale_inv_swizzled", None)
        if scale is None or linear.weight.dtype != torch.float8_e4m3fn:
            return f"{name} is not an MXFP8 linear with swizzled scales"
        if (
            linear.weight.shape[0] % _SCALE_ROWS
            or getattr(linear, "bias", None) is not None
        ):
            return f"{name} {tuple(linear.weight.shape)} does not tile by {_SCALE_ROWS} rows"
    if type(wq_a.quant_method) is not type(wkv.quant_method):
        return "wq_a and wkv use different quant methods"
    backend = getattr(wq_a.quant_method, "mxfp8_dense_backend", None)
    if backend is None or not (
        backend.is_flashinfer_cutedsl() or backend.is_flashinfer_cutlass()
    ):
        # the backends whose MXFP8 apply takes the post's pre-quantized rows
        return f"MXFP8 dense backend {backend} takes no pre-quantized input"
    if wq_a.weight.shape[1] != wkv.weight.shape[1]:
        return "wq_a and wkv read different inputs"
    return ""


class FusedQkvA:
    """``qkv_a_op(x, x_quant) -> (q_lora, kv)``: one GEMM over [wq_a; wkv]. The fused
    linear is a shallow clone of wq_a that holds the concatenated weight and scales, so
    SGLang's own backend choice, input quantization and tactic cache apply to it."""

    def __init__(self, wq_a, wkv, unified_kv: bool):
        fused = torch.nn.Module.__new__(type(wq_a))
        fused.__dict__.update(wq_a.__dict__)
        fused._parameters = dict(wq_a._parameters)
        fused._buffers = dict(wq_a._buffers)
        fused._modules = dict(wq_a._modules)
        weight = torch.cat([wq_a.weight.data, wkv.weight.data])
        fused._parameters["weight"] = torch.nn.Parameter(weight, requires_grad=False)
        # scales tile per 128 weight rows, so the row tiles of the two concatenate
        fused.weight_scale_inv_swizzled = torch.cat(
            [
                wq_a.weight_scale_inv_swizzled.reshape(-1),
                wkv.weight_scale_inv_swizzled.reshape(-1),
            ]
        ).view(
            *(
                (-1,)
                if wq_a.weight_scale_inv_swizzled.dim() == 1
                else (-1, wq_a.weight_scale_inv_swizzled.shape[-1])
            )
        )
        fused.output_size_per_partition = weight.shape[0]
        self.fused, self.split = fused, wq_a.weight.shape[0]
        self.unified_kv = unified_kv

    def __call__(self, x, x_quant):
        out = self.fused.quant_method.apply(
            self.fused, x_quant if x_quant is not None else x, None
        )
        q_lora, kv = out[:, : self.split], out[:, self.split :]
        # the unified-KV store norms + ropes kv in place and scatters contiguous rows only
        return q_lora, kv.contiguous() if self.unified_kv else kv


def install_qkv_a(stages) -> str:
    """Set every draft stage's ``qkv_a_op``; the reason none was set, or ""."""
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    reason = qkv_a_seam()
    for i, stage in enumerate(stages):
        attn = stage.self_attn
        reason = reason or (
            f"stage {i}: {r}" if (r := _fusable(attn.wq_a, attn.wkv)) else ""
        )
    if reason:
        return reason
    for stage in stages:
        attn = stage.self_attn
        attn.qkv_a_op = FusedQkvA(attn.wq_a, attn.wkv, is_unified_kv_triton())
    return ""
