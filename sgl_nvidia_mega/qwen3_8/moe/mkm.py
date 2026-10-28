# SPDX-License-Identifier: Apache-2.0
"""FN SF MKM over native ModelOpt trtllm-gen storage and dispatcher boundaries."""

from __future__ import annotations
import torch
from sgl_nvidia_mega.qwen3_8.moe.config import Shape

SGLANG_READS = {
    "srt/models/qwen2_moe.py": ("topk", "experts", "layer_id"),
    "srt/layers/moe/topk.py": (
        "topk_config",
        "top_k",
        "renormalize",
        "scoring_func",
        "correction_bias",
        "use_grouped_topk",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "num_experts",
        "num_local_experts",
        "moe_ep_rank",
        "top_k",
        "routing_method_type",
        "moe_runner_config",
        "dispatcher",
    ),
    "srt/layers/moe/moe_runner/base.py": (
        "activation",
        "is_gated",
        "routed_scaling_factor",
        "gemm1_alpha",
        "gemm1_clamp_limit",
    ),
    "srt/layers/moe/utils.py": ("RoutingMethodType", "RenormalizeNaive"),
    "srt/layers/quantization/modelopt_quant.py": (
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
        '"g1_alphas"',
        '"g2_alphas"',
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        '"g1_scale_c"',
        "FlashInferTrtllmDeferredFinalizeOutput",
        "gemm2_out",
        "expert_weights",
        "expanded_idx_to_permuted_idx",
        "top_k",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "StandardCombineInput",
        "combine(combine_input)",
    ),
}


def unsupported(mlp, shape: Shape):
    from sglang.srt.layers.moe.utils import RoutingMethodType

    experts, cfg = mlp.experts, mlp.topk.topk_config
    e, h, i = shape.experts, shape.hidden, shape.storage_intermediate
    for name, dtype, dims in [
        ("w13_weight", torch.uint8, (e, 2 * i, h // 2)),
        ("w2_weight", torch.uint8, (e, h, i // 2)),
        ("w13_weight_scale", torch.float8_e4m3fn, (e, 2 * i, h // 16)),
        ("w2_weight_scale", torch.float8_e4m3fn, (e, h, i // 16)),
        ("g1_alphas", torch.float32, (e,)),
        ("g1_scale_c", torch.float32, (e,)),
        ("g2_alphas", torch.float32, (e,)),
    ]:
        value = getattr(experts, name, None)
        if value is None or value.dtype != dtype or tuple(value.shape) != dims:
            return f"{name}: expected {dtype} {dims}, got {None if value is None else (value.dtype, tuple(value.shape))}"
    shared = getattr(cfg, "num_shared_topk_slots", None)
    if shared is None:
        shared = cfg.num_fused_shared_experts
    if not (
        experts.num_experts == experts.num_local_experts == e
        and experts.moe_ep_rank == 0
        and experts.top_k == cfg.top_k == shape.top_k
        and cfg.renormalize
        and cfg.scoring_func == "softmax"
        and cfg.correction_bias is None
        and not cfg.use_grouped_topk
        and shared == 0
        and experts.routing_method_type == RoutingMethodType.RenormalizeNaive
    ):
        return "routing requires local softmax top10, renormalized, no fused shared/EP"
    run = experts.moe_runner_config
    if not (
        run.activation == "silu"
        and run.is_gated
        and run.routed_scaling_factor in (None, 1.0)
        and run.gemm1_alpha is None
        and run.gemm1_clamp_limit is None
        and getattr(run, "swiglu_limit", None) is None
    ):
        return "MKM requires plain SwiGLU and original scalar scales"
    return ""


def routed(mlp, maps, identity, logits, x_fp4, x_sf, *, shape: Shape, defer=True):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sgl_nvidia_mega.qwen3_8.sm103.moe import mkm, finalize

    experts = mlp.experts
    rows, weights = mkm.mkm_moe(
        maps,
        logits,
        x_fp4,
        x_sf,
        experts.w13_weight_scale.view(torch.uint8),
        experts.w2_weight_scale.view(torch.uint8),
        experts.g1_alphas,
        experts.g1_scale_c,
        experts.g2_alphas,
        shape=shape,
    )
    if not defer:
        return finalize.run(rows, weights, shape=shape)
    deferred = FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=rows,
        expert_weights=weights,
        expanded_idx_to_permuted_idx=identity[: logits.shape[0] * shape.top_k],
        top_k=shape.top_k,
    )
    return experts.dispatcher.combine(
        combine_input=StandardCombineInput(hidden_states=deferred)
    )
