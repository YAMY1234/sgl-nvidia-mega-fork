# SPDX-License-Identifier: Apache-2.0
"""FN router/MKM binding, independent of ForwardView and EAGLE metadata."""

from __future__ import annotations
import logging
from contextlib import nullcontext
import msgspec
import torch
from .config import Shape, config_disable_reasons
from . import mkm

logger = logging.getLogger(__name__)
SGLANG_READS = {
    "srt/layers/activation.py": ("SiluAndMul",),
    "srt/layers/quantization/unquant.py": ("UnquantizedLinearMethod",),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "topk",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "supports_deferred_finalize",
        "layer_id",
        "reduce_results",
        "gate_up_proj",
        "down_proj",
        "act_fn",
        "_enable_silu_fp4_quant_fusion",
    ),
    "srt/layers/moe/topk.py": ("TopKOutputChecker", "format_is_bypassed"),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "dispatcher",
        "run_moe_core",
        "run_moe_core(dispatch_output)",
        "quant_method",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "dispatch(hidden_states)",
        "dispatch(topk_output)",
        "combine(combine_input)",
        "hidden_states_scale",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "flashinfer_trtllm_deferred_finalize_context",
        "trtllm_padded_hidden_size",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        '"w13_input_scale_quant"',
        "quant_config",
        "use_per_token_activation",
    ),
}


class Bound(msgspec.Struct, frozen=True):
    route: bool = False
    mkm: dict = msgspec.field(default_factory=dict)
    identity: object = None
    shared: bool = False
    shape: object = None
    reason: str = ""


def unsupported(mlp, shape):
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return f"{type(mlp).__name__}: not sparse MoE"
    w = mlp.gate.weight
    h, e = shape.hidden, shape.experts
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (e, h):
        return f"router gate {w.dtype} {tuple(w.shape)}"
    experts = mlp.experts
    scale = experts.w13_input_scale_quant
    if scale.dtype != torch.float32 or scale.numel() != 1:
        return f"router global scale {scale.dtype} {tuple(scale.shape)}"
    if experts.trtllm_padded_hidden_size is not None:
        return "padded expert hidden dimension"
    if experts.quant_method.quant_config.use_per_token_activation:
        return "per-token activation scale"
    if not mlp.supports_deferred_finalize:
        return "no native deferred-finalize producer"
    if mlp.reduce_results:
        return "MoE reduction must remain at the FN layer boundary"
    return ""


def build(
    model, max_rows, fused, shared=False, wide=False, *, hf_config=None, tp_size=None
):
    from sgl_nvidia_mega.qwen3_8.sm103.moe import router, mkm as kernel, finalize

    m = model.model
    layers = [m.layers[i].mlp for i in range(m.start_layer, m.end_layer)]
    hf_config = hf_config if hf_config is not None else model.config
    tp_size = int(tp_size if tp_size is not None else layers[0].tp_size)
    shape = Shape.from_config(hf_config, tp_size, max_rows)
    reasons = config_disable_reasons(
        hf_config, tp_size, max_rows, "mkm" if fused else "route"
    )
    if wide:
        reasons.append("FN mkm_wide is not implemented; retain native at wider rows")
    if shared:
        reasons.extend(
            config_disable_reasons(hf_config, tp_size, max_rows, "route_shared")
        )
    for i, mlp in enumerate(layers, m.start_layer):
        why = unsupported(mlp, shape)
        if why:
            reasons.append(f"layer{i}: {why}")
            break
        if shared:
            why = shared_unsupported(mlp, shape)
            if why:
                reasons.append(f"layer{i} shared: {why}")
                break
        if fused:
            why = mkm.unsupported(mlp, shape)
            if why:
                reasons.append(f"layer{i} mkm: {why}")
                break
    if reasons:
        return Bound(shape=shape, reason="; ".join(reasons))
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (10, 3):
        return Bound(shape=shape, reason="FN MoE currently requires SM103")
    router.build(device, shape)
    if shared:
        from sgl_nvidia_mega.qwen3_8.sm103.moe import shared as shared_kernel

        shared_kernel.build(device, shape)
    maps = {}
    if fused:
        kernel.build(device, shape)
        finalize.build(device, shape)
        maps = {
            mlp.layer_id: kernel.weight_maps(
                mlp.experts.w13_weight, mlp.experts.w2_weight, shape
            )
            for mlp in layers
        }
    identity = (
        torch.arange(max_rows * shape.top_k, dtype=torch.int32, device=f"cuda:{device}")
        if fused
        else None
    )
    return Bound(route=True, mkm=maps, identity=identity, shape=shape, shared=shared)


def produce(mlp, x, bound, shared=False):
    from sgl_nvidia_mega.qwen3_8.sm103.moe import router

    if shared:
        raise RuntimeError("FN BF16 route_shared is not yet qualified")
    return router.router_gemm_quant(
        x,
        mlp.gate.weight,
        mlp.experts.w13_input_scale_quant.reshape(1),
        shape=bound.shape,
    )


def routed(mlp, x, bound, *, defer=True, produced=None, fused=True):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    logits, x_fp4, x_sf = produce(mlp, x, bound) if produced is None else produced
    if fused and bound.mkm:
        return mkm.routed(
            mlp,
            bound.mkm[mlp.layer_id],
            bound.identity,
            logits,
            x_fp4,
            x_sf,
            shape=bound.shape,
            defer=defer,
        )
    experts = mlp.experts
    topk = mlp.topk(x, logits)
    if not TopKOutputChecker.format_is_bypassed(topk):
        raise RuntimeError("FN route requires logits-based bypassed TopK")
    dispatch = experts.dispatcher.dispatch(hidden_states=x, topk_output=topk)._replace(
        hidden_states=x_fp4, hidden_states_scale=x_sf.view(torch.float8_e4m3fn)
    )
    with flashinfer_trtllm_deferred_finalize_context() if defer else nullcontext():
        combine = experts.run_moe_core(dispatch_output=dispatch)
    return experts.dispatcher.combine(combine_input=combine)


def shared_unsupported(mlp, shape):
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    expert = mlp.shared_expert
    if expert is None:
        return "route_shared requires the separate native BF16 expert"
    local = shape.shared_intermediate // shape.tp_size
    if local <= 0 or 2 * local > shape.experts or local % 4:
        return f"BF16 shared grid requires 8-aligned gate/up rows <= E; got I{local}"
    for layer, dims in [
        (expert.gate_up_proj, (2 * local, shape.hidden)),
        (expert.down_proj, (shape.hidden, local)),
    ]:
        if not isinstance(layer.quant_method, UnquantizedLinearMethod):
            return "route_shared requires original unquantized BF16 weights"
        if (
            layer.weight.dtype != torch.bfloat16
            or tuple(layer.weight.shape) != dims
            or not layer.weight.is_contiguous()
        ):
            return f"BF16 shared projection expected contiguous {dims}"
    if (
        not isinstance(expert.act_fn, SiluAndMul)
        or expert._enable_silu_fp4_quant_fusion
    ):
        return "route_shared requires native BF16 SwiGLU without FP4 activation quantization"
    return ""


def produce_shared(mlp, x, bound):
    from sgl_nvidia_mega.qwen3_8.sm103.moe import shared

    return shared.router_shared(
        x,
        mlp.gate.weight,
        mlp.experts.w13_input_scale_quant.reshape(1),
        mlp.shared_expert.gate_up_proj.weight,
        shape=bound.shape,
    )


def shared_expert_tail(expert, gate_up):
    """Keep native BF16 activation, down projection and its original boundary."""
    act = expert.act_fn(gate_up)
    output, _ = expert.down_proj(act)
    return output
