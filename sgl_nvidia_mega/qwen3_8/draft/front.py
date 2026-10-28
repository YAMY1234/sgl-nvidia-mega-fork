# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FN's pure BF16 draft Q/K Gemma norm, NeoX RoPE and gate deinterleave.

Reuse Qwen3.5's parameterized front with FP8_CACHE=False: no KV write,
quantization, QSA lookup or attention, and no PDL release before an undefined
consumer. The slot is not registered until the FN draft executor can call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", hf_config)


def config_disable_reasons(hf_config: Any, tp_size: int, dtype="bfloat16") -> list[str]:
    """Report unsupported configurations before any kernel or weight allocation."""
    cfg = _text_config(hf_config)
    reasons = []
    if str(dtype).removeprefix("torch.") != "bfloat16":
        reasons.append("draft_front requires bfloat16 Q/K and norm weights")
    fields = ("hidden_size", "num_attention_heads", "num_key_value_heads")
    for name in fields:
        value = getattr(cfg, name, None)
        if not isinstance(value, int) or value <= 0:
            reasons.append(f"draft_front: text_config.{name} must be a positive integer")
    if not isinstance(tp_size, int) or tp_size <= 0:
        reasons.append("draft_front: tp_size must be a positive integer")
    if reasons:
        return reasons
    qh, kvh = cfg.num_attention_heads, cfg.num_key_value_heads
    if qh % tp_size:
        reasons.append("draft_front: num_attention_heads must divide TP evenly")
    if kvh >= tp_size and kvh % tp_size:
        reasons.append("draft_front: num_key_value_heads must divide TP evenly")
    if kvh < tp_size and tp_size % kvh:
        reasons.append("draft_front: TP must be divisible by replicated KV heads")
    dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // qh
    if not isinstance(dim, int) or not 1 <= dim <= 1024:
        reasons.append("draft_front: head_dim must be an integer in [1, 1024]")
        return reasons
    factor = getattr(cfg, "partial_rotary_factor", 1.0)
    if not isinstance(factor, (int, float)) or not 0 < factor <= 1:
        reasons.append("draft_front: partial_rotary_factor must be in (0, 1]")
    else:
        rotary = int(dim * factor)
        if rotary == 0 or rotary % 2:
            reasons.append("draft_front: rotary dimension must be positive and even")
    eps = getattr(cfg, "rms_norm_eps", None)
    if not isinstance(eps, (float, int)) or eps <= 0:
        reasons.append("draft_front: rms_norm_eps must be positive")
    return reasons


@dataclass(frozen=True)
class FrontShape:
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    eps: float


def build_front(hf_config: Any, tp_size: int, dtype="bfloat16"):
    """(primitive, reasons); no registration and no model or binding mutation."""
    reasons = config_disable_reasons(hf_config, tp_size, dtype)
    if reasons:
        return None, reasons
    cfg = _text_config(hf_config)
    dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    shape = FrontShape(
        hidden_size=cfg.hidden_size,
        q_heads=cfg.num_attention_heads // tp_size,
        kv_heads=max(1, cfg.num_key_value_heads // tp_size),
        head_dim=dim,
        rotary_dim=int(dim * getattr(cfg, "partial_rotary_factor", 1.0)),
        eps=float(cfg.rms_norm_eps),
    )
    return DraftFront(shape), []


def _kernel():
    from sgl_nvidia_mega.qwen3_5.sm103.draft.front import (
        _fused_qk_rmsnorm_rope_gate_kernel,
    )

    return _fused_qk_rmsnorm_rope_gate_kernel


class DraftFront:
    """The BF16 primitive only; normal stream dependency, no cache side effect."""

    def __init__(self, shape: FrontShape):
        self.shape = shape

    def __call__(self, q_gate, k, q_weight, k_weight, cos_sin, positions, axis_map=None):
        import torch
        import triton

        s = self.shape
        rows = q_gate.shape[0]
        mrope = positions.ndim == 2
        reasons = []
        if q_gate.ndim != 2 or q_gate.shape[1] != 2 * s.q_heads * s.head_dim:
            reasons.append("Q+gate shape differs from the configured interleaved heads")
        if k.shape != (rows, s.kv_heads * s.head_dim):
            reasons.append("K shape differs from the configured KV heads")
        if any(x.dtype != torch.bfloat16 for x in (q_gate, k, q_weight, k_weight)):
            reasons.append("Q/K and both raw Gemma norm weights must be BF16")
        if not q_weight.is_contiguous() or not k_weight.is_contiguous():
            reasons.append("norm weights must be contiguous")
        if positions.ndim == 1 and positions.stride(0) != 1:
            reasons.append("one-dimensional positions must be contiguous")
        if q_weight.shape != (s.head_dim,) or k_weight.shape != (s.head_dim,):
            reasons.append("each norm weight must span one head")
        if q_gate.stride(-1) != 1 or k.stride(-1) != 1:
            reasons.append("Q/K must be contiguous within a row")
        if positions.ndim not in (1, 2) or positions.shape[-1] != rows:
            reasons.append("positions must be [rows] or [3, rows]")
        if mrope and (positions.shape[0] != 3 or positions.stride(1) != 1):
            reasons.append("mrope positions must be [3, rows], contiguous within each axis")
        if mrope != (axis_map is not None):
            reasons.append("axis_map is required exactly for mrope positions")
        if axis_map is not None and (
            axis_map.shape != (s.rotary_dim // 2,) or axis_map.stride(0) != 1
        ):
            reasons.append("axis_map must have one entry per rotary pair")
        if cos_sin.ndim != 2 or cos_sin.shape[1] < s.rotary_dim or cos_sin.stride(1) != 1:
            reasons.append("cos/sin cache must hold [cos..., sin...] rotary rows")
        if reasons:
            raise ValueError("draft_front: " + "; ".join(reasons))
        q_out = torch.empty((rows, s.q_heads * s.head_dim), dtype=q_gate.dtype, device=q_gate.device)
        k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
        gate = torch.empty((rows, s.q_heads, s.head_dim), dtype=q_gate.dtype, device=q_gate.device)
        _kernel()[(rows, s.q_heads + s.kv_heads)](
            q_gate, k, q_out, k_out, gate, q_weight, k_weight, cos_sin, positions, axis_map,
            q_gate.stride(0), k.stride(0), q_out.stride(0), k_out.stride(0), gate.stride(0),
            cos_sin.stride(0), positions.stride(0),
            NUM_Q_HEADS=s.q_heads, NUM_KV_HEADS=s.kv_heads, HEAD_DIM=s.head_dim,
            ROTARY_DIM=s.rotary_dim, HALF_ROTARY=s.rotary_dim // 2,
            HEAD_BLOCK=triton.next_power_of_2(s.head_dim),
            ROT_HALF_BLOCK=triton.next_power_of_2(s.rotary_dim // 2), EPS=s.eps,
            FP16=False, HAS_PASS=s.rotary_dim < s.head_dim, HAS_GATE=True, MROPE=mrope,
            ENABLE_PDL=False, WEIGHT_SHIFT=1.0, EARLY_TRIGGER=False,
            EXPLICIT_ROPE_FMA=False, FP8_CACHE=False,
            num_warps=4, launch_pdl=False,
        )
        return q_out, k_out, gate
