# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/srt/layers/moe/single_token_draft_moe.py
# (SingleTokenDraftMoe: its eligibility checks and single-launch forward), on branch
# nvidia/qwen35-agentx-gb300.
"""The MTP draft layer's MoE as one launch per token (``sm103/draft/moe.py``): router
GEMV + softmax top-k, the routed experts and the gated shared expert, from the bf16
weights SGLang loaded for the draft's ``Qwen2MoeSparseMoeBlock``.

``DraftMoe(x)`` returns this rank's unreduced output (routed and gated shared summed),
whose TP sum the draft boundary completes. Against SGLang's stock draft MoE only the
fp32 dot-product order differs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_5.moe.mkm import shared_topk_slots
from sgl_nvidia_mega.qwen3_5.sm103.draft import moe as kernels

logger = logging.getLogger(__name__)
DIRECT_ENV = "SGLANG_Q35FLY_DRAFT_MOE_DIRECT"


def _direct_requested() -> bool:
    """Prepare both capture choices when any arm requests the candidate."""
    variants = json.loads(os.environ.get("Q35MF_AB_VARIANTS") or "{}")
    return os.environ.get(DIRECT_ENV, "0") == "1" or any(
        arm.get(DIRECT_ENV) == "1" for arm in variants.values()
    )

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "gate",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "topk",
        "gate_up_proj",
        "down_proj",
        "layer_id",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("quant_method",),
    "srt/layers/quantization/unquant.py": (
        "UnquantizedFusedMoEMethod",
        "use_flashinfer_trtllm_moe",
        "w13_swiglu_interleaved",
        '"w13_weight"',
        '"w2_weight"',
    ),
    "srt/layers/moe/topk.py": (
        "topk_config",
        "top_k",
        "enable_waterfill",
        "scoring_func",
        "renormalize",
        "use_grouped_topk",
        "correction_bias",
        "custom_routing_function",
        "routed_scaling_factor",
    ),
    "srt/runtime_context.py": ("get_parallel", '"moe_ep_size"'),
}


def _router_reason(mlp) -> Optional[str]:
    cfg = mlp.topk.topk_config
    if (
        cfg.scoring_func != "softmax"
        or not cfg.renormalize
        or cfg.use_grouped_topk
        or cfg.correction_bias is not None
        or cfg.custom_routing_function is not None
        or shared_topk_slots(cfg) != 0
        or cfg.routed_scaling_factor not in (None, 1.0)
        or mlp.topk.enable_waterfill
    ):
        return "router is not plain softmax top-k with renormalization"
    w = mlp.gate.weight
    if w.dtype != torch.bfloat16 or w.shape[0] % 32 != 0 or not w.is_contiguous():
        return f"router weight {w.dtype} {tuple(w.shape)}"
    return None


def perflp_bf16_twin(experts) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The BF16 experts (checkpoint order, this rank's slice) SGLang keeps beside NVFP4 draft experts under
    SGLANG_PERFLP_DRAFT_MOE_KEEP_BF16 (PERF-LP), or None: the single launch serves the steps of <= 2 tokens in BF16."""
    w13 = getattr(experts, "perflp_bf16_w13", None)
    w2 = getattr(experts, "perflp_bf16_w2", None)
    return (w13, w2) if w13 is not None and w2 is not None else None


def unsupported_reason(mlp) -> Optional[str]:
    """Why the single launch cannot serve this block (None when it can): the checks of
    SingleTokenDraftMoe._unsupported_reason and _router_unsupported_reason."""
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.runtime_context import get_parallel

    experts, shared = mlp.experts, mlp.shared_expert
    if shared is None or mlp.shared_expert_gate is None:
        return "no separate shared expert"
    # The kernel reads W13 as [gate rows | up rows] in checkpoint order (flashinfer_trtllm's
    # layout is put back in that order, in a copy).
    method = experts.quant_method
    twin = perflp_bf16_twin(experts)
    if twin is None and not isinstance(method, UnquantizedFusedMoEMethod):
        return f"expert weights handled by {type(method).__name__}"
    if twin is None and method.w13_swiglu_interleaved:
        return "expert weights in a kernel-specific layout"
    if get_parallel().moe_ep_size != 1:
        return "expert parallelism"
    w13, w2 = twin or (experts.w13_weight, experts.w2_weight)
    ws13, ws2 = shared.gate_up_proj.weight, shared.down_proj.weight
    s13, s2 = (
        (tuple(w13.shape), tuple(w2.shape)) if twin else _checkpoint_shapes(experts)
    )
    hidden, inter = s13[-1], s2[-1]
    tensors = (w13, w2, ws13, ws2, mlp.shared_expert_gate.weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        return "non-bf16 operands"
    if len(s13) != 3 or s13[1:] != (2 * inter, hidden):
        return f"w13 shape {tuple(w13.shape)}"
    if tuple(ws13.shape) != (2 * inter, hidden) or tuple(ws2.shape) != (hidden, inter):
        return f"shared expert shapes {tuple(ws13.shape)} {tuple(ws2.shape)}"
    if hidden % 1024 != 0 or inter % 32 != 0:
        return f"hidden {hidden} / intermediate {inter} not tileable"
    if not all(t.is_contiguous() for t in tensors):
        return "non-contiguous operands"
    reason = _router_reason(mlp)
    if reason is not None:
        return reason
    # the single launch's own tiling (static_asserts of src/draft_moe.cu)
    num_experts, topk = mlp.gate.weight.shape[0], mlp.topk.topk_config.top_k
    per_lane = num_experts // 32
    if inter % 64 != 0 or per_lane & (per_lane - 1) or not 1 <= topk <= kernels.STAGES:
        return f"{num_experts} experts / top-{topk} / intermediate {inter} not tileable"
    return None


# flashinfer_trtllm's BF16 expert layout (UnquantizedFusedMoEMethod.
# process_weights_after_loading): W13 loaded up rows first, each expert's rows permuted for
# the gated GEMM's epilogue tiles of 128, both matrices in blocks of 128 bytes of K
TRTLLM_EPILOGUE_TILE_M = 128


def _checkpoint_shapes(experts) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The W13 and W2 shapes in checkpoint order, [E, 2I, H] and [E, H, I]."""
    w13, w2 = experts.w13_weight, experts.w2_weight
    if not experts.quant_method.use_flashinfer_trtllm_moe:
        return tuple(w13.shape), tuple(w2.shape)
    if w13.dim() != 4 or w2.dim() != 4:
        return tuple(w13.shape), tuple(w2.shape)
    # [E, K / block, rows, block]
    return (
        (w13.shape[0], w13.shape[2], w13.shape[1] * w13.shape[3]),
        (w2.shape[0], w2.shape[2], w2.shape[1] * w2.shape[3]),
    )


def checkpoint_weights(experts) -> tuple[torch.Tensor, torch.Tensor]:
    """W13 ([gate rows | up rows]) and W2 put back in checkpoint order from
    flashinfer_trtllm's layout: a copy of the block's expert weights."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    (e, rows13, hidden), (_, rows2, inter) = _checkpoint_shapes(experts)
    w13 = torch.empty(
        e, rows13, hidden, dtype=torch.bfloat16, device=experts.w13_weight.device
    )
    w2 = torch.empty(e, rows2, inter, dtype=torch.bfloat16, device=w13.device)
    cache: dict = {}
    # the single launch serves gated experts only (W13 = gate and up rows)
    perm13 = _maybe_get_cached_w3_w1_permute_indices(
        cache, w13[0].view(torch.uint8), TRTLLM_EPILOGUE_TILE_M, is_gated_act_gemm=True
    )
    perm2 = get_w2_permute_indices_with_cache(
        cache, w2[0].view(torch.uint8), TRTLLM_EPILOGUE_TILE_M
    )
    half = rows13 // 2
    w31 = torch.empty(rows13, 2 * hidden, dtype=torch.uint8, device=w13.device)
    for x in range(e):
        # undo convert_to_block_layout, then the row gather (row j came from row perm[j])
        w31[perm13] = (
            experts.w13_weight[x].view(torch.uint8).transpose(0, 1).reshape(rows13, -1)
        )
        up_gate = w31.view(torch.bfloat16)
        w13[x, :half] = up_gate[half:]
        w13[x, half:] = up_gate[:half]
        w2[x].view(torch.uint8)[perm2] = (
            experts.w2_weight[x].view(torch.uint8).transpose(0, 1).reshape(rows2, -1)
        )
    return w13, w2


class DraftMoe:
    """The single-launch draft MoE of one block: its scratch and the per-token launch."""

    MAX_TOKENS = 2

    def __init__(self, mlp, device: torch.device, weights=None):
        """weights: (W13, W2) in checkpoint order when SGLang keeps another layout."""
        self.mlp = mlp
        self.w13, self.w2 = weights or (mlp.experts.w13_weight, mlp.experts.w2_weight)
        num_experts = mlp.gate.weight.shape[0]
        topk = mlp.topk.topk_config.top_k
        inter = self.w2.shape[-1]
        self._direct_eligible = (
            self.w13.shape[-1], inter, num_experts, topk
        ) == (4096, 128, 512, 10) and kernels.columns(4096, device.index) == 27
        self._direct_kernel = (
            kernels.prepare_direct(device.index)
            if self._direct_eligible and _direct_requested()
            else None
        )
        self.scratch = kernels.draft_moe_fused_scratch(num_experts, topk, inter, device)
        self.topk_weights = torch.empty(1, topk, dtype=torch.float32, device=device)
        self.topk_ids = torch.empty(1, topk, dtype=torch.int32, device=device)

    @classmethod
    def build(cls, mlp, device_index: int) -> Optional[DraftMoe]:
        """The block's single-launch MoE, compiled and checked to fit this device, or
        None (logged) when it cannot serve the block."""
        reason = unsupported_reason(mlp)
        weights = perflp_bf16_twin(mlp.experts)
        if reason is None:
            if weights is None and mlp.experts.quant_method.use_flashinfer_trtllm_moe:
                weights = checkpoint_weights(mlp.experts)
            w13, w2 = weights or (mlp.experts.w13_weight, mlp.experts.w2_weight)
            reason = "; ".join(
                kernels.build(
                    device_index,
                    hidden=w13.shape[-1],
                    inter=w2.shape[-1],
                    experts=mlp.gate.weight.shape[0],
                    topk=mlp.topk.topk_config.top_k,
                )
            )
        if reason:
            logger.warning(
                "Single-launch draft MoE off for layer %d (%s); SGLang's MoE runs.",
                mlp.layer_id,
                reason,
            )
            return None
        logger.info(
            "Single-launch draft MoE active for layer %d%s.",
            mlp.layer_id,
            " (PERF-LP BF16 twin of the NVFP4 experts)"
            if perflp_bf16_twin(mlp.experts) is not None
            else " (checkpoint-order copy of flashinfer_trtllm's experts)"
            if weights
            else "",
        )
        return cls(mlp, torch.device("cuda", device_index), weights)

    def ready(self, num_tokens: int) -> bool:
        """Whether it runs a batch of this many tokens."""
        return 0 < num_tokens <= self.MAX_TOKENS

    def single_launch(self, num_tokens: int) -> bool:
        """Whether that batch runs as single launches (always, once ready)."""
        return self.ready(num_tokens)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[m, H] -> this rank's MoE output [m, H] (m <= MAX_TOKENS), unreduced; the
        tokens run one launch each, in order."""
        mlp = self.mlp
        direct = None
        if self._direct_eligible and hidden_states.shape[0] == 1 and os.environ.get(DIRECT_ENV, "0") == "1":
            direct = self._direct_kernel
            if direct is None:
                raise RuntimeError("direct was not prepared before capture")
        out = torch.empty_like(hidden_states)
        for m in range(hidden_states.shape[0]):
            kernels.draft_moe_fused(
                hidden_states[m : m + 1],
                mlp.gate.weight,
                self.w13,
                self.w2,
                mlp.shared_expert.gate_up_proj.weight,
                mlp.shared_expert.down_proj.weight,
                mlp.shared_expert_gate.weight.view(-1),
                self.scratch,
                self.topk_weights,
                self.topk_ids,
                out[m : m + 1],
                direct_kernel=direct,
            )
        return out
