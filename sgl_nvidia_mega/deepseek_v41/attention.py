# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention slots of a routed step at 1-48 rows, filled through SGLang's
``MQAOpSlots`` (``MQALayer.forward_before_wo_b``) and the backend's ``sparse_decode``.

Each arrow below is a programmatic dependent launch (PDL):
q-LoRA norm + MXFP8 quant (triggers at entry, 1-48 rows) -> wq_b -> swap-AB sparse
attention (1-24 rows: triggers at entry; on the low-ratio layers it gathers the KV
before its wait and ropes the query as it loads it) -> the narrow combine (triggers
at entry) -> wo_a. At 25-48 rows only the norm + quant is ours; FlashMLA, which
needs the 64-head query, stays SGLang's.
"""

from __future__ import annotations

import functools
import logging

import torch

from sgl_nvidia_mega.deepseek_v41.plan import (
    ATTN_NORM_QUANT_MAX_ROWS,
    SWAPAB_MAX_ROWS,
)
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

logger = logging.getLogger(__name__)

# (slot, rows, variant) already logged: engagement is logged from the callables SGLang's
# dispatch runs, once per width on each rank
_ENGAGED: set = set()


def _log_engaged(slot: str, rows: int, variant: str = "") -> None:
    key = (slot, rows, variant)
    if key not in _ENGAGED:
        _ENGAGED.add(key)
        logger.info(
            "sgl-nvidia-mega attention: %s runs at %d rows%s", slot, rows, variant
        )


# MQALayer takes these compress ratios on its low-ratio multi-stream prepare, whose
# KV and index producers all run on side streams joined before attention
_LOW_RATIOS = (1, 2)

SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "MQAOpSlots",
        "MQALayer",
        "forward_before_wo_b(op_slots)",
        "wq_b",
        "q_norm",
        "freqs_cis",
        "compress_ratio",
        "q_head_norm",
        "n_local_heads",
        "head_dim",
        "q_lora_rank",
        "is_dsv41",
        "alt_streams",
    ),
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
    "srt/layers/quantization/fp8_utils.py": (
        "Mxfp8DenseGemmBackend",
        "FLASHINFER_CUTEDSL",
    ),
    "srt/layers/quantization/fp8.py": (
        "mxfp8_dense_backend",
        "use_mxfp8",
        "block_fp8_mxfp8_ready",
    ),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DeepseekV4AttnBackend",
        "DSV4AttnMetadata",
        "forward_metadata",
        "core_attn_metadata",
        "_forward_attention(sparse_decode)",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "request_window",
        "get_swa_key_bytes_per_token",
    ),
    "srt/batch_invariant_ops/*.py": ("is_batch_invariant_mode_enabled",),
    "srt/runtime_context.py": ("get_exec",),
}

# The SGLang seams the attention slots need, probed at build (``attention_seams``)
SGLANG_SEAMS = {
    "srt/models/deepseek_v4.py": (
        "MQAOpSlots",
        "forward_before_wo_b(op_slots)",
    ),
    "srt/layers/attention/deepseek_v4_backend.py": (
        "_forward_attention(sparse_decode)",
    ),
}


def attention_seams() -> str:
    """Why SGLang cannot run the attention slots, or "" when it can."""
    import inspect

    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
    )
    from sglang.srt.models.deepseek_v4 import MQALayer

    missing = []
    try:
        from sglang.srt.models.deepseek_v4 import MQAOpSlots  # noqa: F401
    except ImportError:
        missing.append("MQAOpSlots")
    if "op_slots" not in inspect.signature(MQALayer.forward_before_wo_b).parameters:
        missing.append("MQALayer.forward_before_wo_b(op_slots)")
    forward = DeepseekV4AttnBackend._forward_attention
    if "sparse_decode" not in inspect.signature(forward).parameters:
        missing.append("DeepseekV4AttnBackend._forward_attention(sparse_decode)")
    return f"this SGLang lacks {', '.join(missing)}" if missing else ""


def _norm_quant_applies(attn) -> bool:
    """wq_b reads FlashInfer-swizzled MXFP8, as SGLang's own q-LoRA quant requires."""
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

    method = attn.wq_b.quant_method
    return (
        attn.is_dsv41
        and attn.q_lora_rank == 1280
        and attn.q_norm.weight.dtype == torch.bfloat16
        and getattr(method, "mxfp8_dense_backend", None)
        == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        and (
            getattr(method, "use_mxfp8", False)
            or getattr(attn.wq_b, "block_fp8_mxfp8_ready", False)
        )
    )


def _swapab_applies(attn) -> bool:
    return (
        attn.is_dsv41
        and not attn.q_head_norm
        and attn.n_local_heads == swapab.NUM_HEADS
        and attn.head_dim == swapab.HEAD_DIM
    )


def _deterministic() -> bool:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_exec

    return (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    )


def _sparse_decode(
    *,
    q,
    swa_k_cache,
    swa_page_indices,
    swa_topk_lengths,
    attn_sink,
    extra_k_cache,
    extra_indices,
    extra_topk_lengths,
    q_rope=None,
    **_,
):
    """The backend's swap-AB / FlashMLA choice, as swap-AB on the 16 local heads."""
    _log_engaged(
        "swap-AB sparse decode",
        q.shape[0],
        " (query RoPE + KV before the wait)" if q_rope is not None else "",
    )
    return swapab.swapab_attention(
        q[..., : swapab.NUM_HEADS, :],
        swa_k_cache,
        swa_page_indices,
        swa_topk_lengths,
        attn_sink,
        extra_k_cache,
        extra_indices,
        extra_topk_lengths,
        max_rows=SWAPAB_MAX_ROWS,
        pdl=True,
        narrow_combine=True,
        q_rope=q_rope,
        # the query is the only input still in flight: see _LOW_RATIOS
        kv_before_wait=q_rope is not None,
        combine_entry_trigger=True,
    )


class AttentionSlots:
    """One target layer's attention slots; ``op_slots`` picks them per step."""

    def __init__(self, attn):
        self.attn = attn
        self.norm_quant = _norm_quant_applies(attn)
        self.swapab = _swapab_applies(attn)
        self.rope_in_attention = (
            self.swapab
            and attn.compress_ratio in _LOW_RATIOS
            and attn.alt_streams is not None
        )
        # fp32 [positions, 64]: cos, sin interleaved, as the query RoPE reads them
        self.freqs = (
            torch.view_as_real(attn.freqs_cis).flatten(-2).contiguous()
            if self.rope_in_attention
            else None
        )

    def _q_norm_quant(self, q):
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        attn = self.attn
        _log_engaged("q-LoRA norm + quant", q.shape[0])
        y, quant, scale = rmsnorm_mxfp8(
            q, attn.q_norm.weight, attn.q_norm.variance_epsilon, pdl=True
        )
        return y, Mxfp8SwizzledInput(quant, scale)

    def _q_b_unroped(self, q, positions, q_out):
        attn = self.attn
        return attn.wq_b(q)[0].view(-1, attn.n_local_heads, attn.head_dim)

    def op_slots(self, rows: int, positions, fb, *, swapab_rows: bool):
        """The slots this step fills, or None when it runs SGLang's ops only."""
        from sglang.srt.models.deepseek_v4 import MQAOpSlots

        norm_quant = self.norm_quant and rows <= ATTN_NORM_QUANT_MAX_ROWS
        take_swapab = self.swapab and swapab_rows and rows <= SWAPAB_MAX_ROWS
        if not (norm_quant or take_swapab):
            return None
        kwargs = {}
        if norm_quant:
            kwargs["q_norm_quant"] = self._q_norm_quant
        if take_swapab and self.rope_in_attention:
            kwargs.update(
                q_b=self._q_b_unroped,
                unpadded_q=True,
                sparse_decode=functools.partial(
                    _sparse_decode, q_rope=(self.freqs, positions)
                ),
            )
        elif take_swapab:
            kwargs["sparse_decode"] = _sparse_decode
        return MQAOpSlots(**kwargs)


def swapab_takes_step(rows: int, fb) -> bool:
    """SGLang's swap-AB conditions that only the step knows: a decode or verify step
    on the paged V4-layout cache, with no request window."""
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    mode = fb.forward_mode
    pool = get_token_to_kv_pool()
    meta = getattr(get_attn_backend(), "forward_metadata", None)
    return (
        0 < rows <= SWAPAB_MAX_ROWS
        and (mode.is_decode() or mode.is_target_verify())
        and isinstance(getattr(meta, "core_attn_metadata", None), DSV4AttnMetadata)
        and pool.request_window is None
        and pool.get_swa_key_bytes_per_token() == swapab.LAYOUT.bytes_per_token
    )


def build(model) -> tuple[dict, str]:
    """Every target layer's attention slots by layer id, and the reason there are
    none ("" when built)."""
    reason = attention_seams()
    if reason:
        return {}, reason
    if _deterministic():
        return {}, "deterministic inference keeps SGLang's q-LoRA quant"
    m = model.model
    slots = {
        m.layers[i].layer_id: AttentionSlots(m.layers[i].self_attn)
        for i in range(m.start_layer, m.end_layer)
    }
    return slots, ""
