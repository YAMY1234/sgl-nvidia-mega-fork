# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen2_moe.py (Qwen2MoeSparseMoeBlock.forward).
"""The MoE step of a covered forward.

Slot ``moe_finalize`` (TP1, no EP, up to 1024 rows): the router and the routed experts
run SGLang's FlashInfer TRT-LLM MoE with its finalize deferred, the shared expert runs
ungated, and one plugin kernel (``_fused_finalize_gate_sigmoid_mul_add_kernel``) writes
``sigmoid(x . g) * shared + sum_k w_k * expert_k`` from the un-finalized expert rows
(``gemm2_out`` and its permutation). Stock: FlashInfer's finalize, then
``fused_gate_sigmoid_mul_add``. Every other case runs SGLang's ``Qwen2MoeSparseMoeBlock``."""

from __future__ import annotations

import contextlib

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins, moe_routing

STOCK = "stock"
MAX_ROWS = 1024
HIDDEN, TOP_K = 2560, 10

SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "_forward_shared_experts",
        "shared_expert",
        "_forward_router_experts",
        "supports_deferred_finalize",
        "shared_expert_gate",
        "alt_stream",
        "TopKOutputChecker",
        "gate",
        "topk",
        "experts",
    ),
    "srt/layers/moe/topk.py": ("format_is_bypassed",),
    "srt/layers/moe/fused_moe_triton/layer.py": ("forward_deferred_finalize",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "gemm2_out",
        "expanded_idx_to_permuted_idx",
        "expert_weights",
        "top_k",
    ),
}


# With BF16 pins on, _router_experts and _shared_experts mirror these
SGLANG_MIRRORS = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": "251e3d343aec92fe",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_shared_experts": "3b55a78638831ffa",
}


def layer_reasons(layer) -> str:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return "not a sparse MoE block"
    if not mlp.supports_deferred_finalize:
        return "the MoE runner cannot defer its finalize"
    if mlp.shared_expert_gate is None:
        return "no shared-expert gate"
    if mlp.shared_expert_gate.weight.dtype != torch.bfloat16:
        return "shared-expert gate is not BF16"
    return ""


def build(parallel) -> str:
    if parallel.tp_size != 1 or getattr(parallel, "moe_ep_size", 1) != 1:
        return "the finalize kernel takes TP1 without expert parallelism"
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import moe_finalize  # noqa: F401
    except Exception as e:
        return f"finalize kernel does not import: {type(e).__name__}: {e}"
    return ""


def _side_stream(mlp):
    """The block's side stream while a graph is captured, as SGLang uses it; else None."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return mlp.alt_stream if mlp.alt_stream is not None and get_is_capture_mode() else None


def step(executor, layer, x, st):
    """The MoE output [rows, hidden] (TP1: nothing left to reduce)."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    rows = x.shape[0]
    if not (executor.moe_finalize_ready and 0 < rows <= MAX_ROWS and x.shape[1] == HIDDEN):
        st.ran("moe", STOCK)
        if isinstance(mlp, Qwen2MoeSparseMoeBlock):
            return mlp(x, st.fb, defer_finalize=False)
        return mlp(x)
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.moe_finalize import (
        _fused_finalize_gate_sigmoid_mul_add_kernel as kernel,
    )
    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    # SGLang's order (Qwen2MoeSparseMoeBlock.forward_normal_dual_stream): the routed
    # experts on the current stream; while capturing, the shared expert on the block's
    # side stream, so it runs beside the routed GEMMs.
    side = _side_stream(mlp)
    if side is not None:
        current = torch.cuda.current_stream()
        side.wait_stream(current)
    # slot moe_routing: None when it does not take these rows
    deferred = moe_routing.routed(executor, mlp, x, st)
    if deferred is None:
        deferred = _router_experts(mlp, x)
    with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
        shared = smallm_gemm.shared_expert(executor, mlp.shared_expert, x, st)
        if shared is None:
            shared = _shared_experts(mlp, x)
        shared = shared.contiguous()
    if side is not None:
        shared.record_stream(current)
        current.wait_stream(side)
    top_k = int(deferred.top_k)
    gemm2 = deferred.gemm2_out.view(-1, deferred.gemm2_out.shape[-1])
    perm = deferred.expanded_idx_to_permuted_idx.view(-1, top_k)[:rows].contiguous()
    weights = deferred.expert_weights.view(-1, top_k)[:rows].contiguous()
    if top_k != TOP_K or perm.dtype != torch.int32:
        raise RuntimeError(f"moe_finalize: deferred output layout differs (top_k {top_k})")
    out = torch.empty_like(x)
    kernel[(rows,)](
        x.contiguous(),
        mlp.shared_expert_gate.weight.view(-1),
        shared,
        out,
        gemm2,
        perm,
        weights,
        gemm2.stride(0),
        hidden_dim=HIDDEN,
        TOP_K=TOP_K,
        BLOCK_SIZE=4096,
        USE_PDL=False,
        num_warps=8,
    )
    st.ran("moe", "finalize_permuted")
    return out


def _router_experts(mlp, x):
    """``mlp._forward_router_experts(x, defer_finalize=True)``; with BF16 pins on, here
    with the router GEMM through ``bf16_gemm_pins``."""
    if not bf16_gemm_pins.routed():
        return mlp._forward_router_experts(x, defer_finalize=True)
    from sglang.srt.models.qwen2_moe import TopKOutputChecker

    router_logits = bf16_gemm_pins.linear(mlp.gate, x)
    topk_output = mlp.topk(x, router_logits)
    if not mlp.supports_deferred_finalize:
        raise RuntimeError(
            "Qwen deferred finalize requires a compatible FlashInfer "
            "TRTLLM MoE producer and a separate shared expert"
        )
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise RuntimeError("Qwen deferred finalize requires logits-based bypassed TopK")
    return mlp.experts.forward_deferred_finalize(x, topk_output)


def _shared_experts(mlp, x):
    """``mlp._forward_shared_experts(x, apply_gate=False)``; with BF16 pins on, the
    shared expert's projections through ``bf16_gemm_pins``."""
    if not bf16_gemm_pins.routed():
        return mlp._forward_shared_experts(x, apply_gate=False)
    if mlp.shared_expert is None:
        return None
    return bf16_gemm_pins.mlp(mlp.shared_expert, x)
