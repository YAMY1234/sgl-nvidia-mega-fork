# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/moe/fused_moe_triton/layer.py (FusedMoE.forward_deferred_finalize).
"""Slot ``route``: a MoE layer's router GEMM and the NVFP4 quantization of its routed
experts' input in one launch (``sm103/moe/router.py``), then SGLang's trtllm routed
experts on that input, with the deferred finalize.

The logits equal the gate's cuBLAS GEMM bit for bit and the quantized input equals the
trtllm runner's own quantizer, so the step's output is the stock one's. The runner
takes the quantized input through its pre-quantized (``hidden_states_scale``) path,
or slot ``mkm`` (``mkm.py``) takes the router front's outputs in its place.
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "topk",
        "experts",
        "supports_deferred_finalize",
        "layer_id",
    ),
    "srt/models/qwen3_5.py": ("layers", "start_layer", "end_layer", "mlp"),
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

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE.forward_deferred_finalize": "af4b474cbaf33493",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE._dispatch_with_pre_quant": "dcdefc28cf643081",
}


# the (slot, rows) that have run (each logged once, as engagement evidence)
_ENGAGED: set = set()


class Bound(msgspec.Struct, frozen=True):
    """The MoE package's build result (``st.bound["moe"]``)."""

    route: bool = False  # the router front stands in for every layer's gate
    mkm: dict = msgspec.field(default_factory=dict)  # layer id -> its weight TMA maps
    identity: object = None  # [70] int32 0, 1, ...: slot mkm's rows are in expert order
    reason: str = ""  # why a requested slot is not built


def build(model, max_rows: int, fused: bool) -> Bound:
    """The router front for every MoE layer, compiled for 1..max_rows rows, and with
    ``fused`` slot mkm's kernel and weight maps; or why a layer cannot take them."""
    from sgl_nvidia_mega.qwen3_5.moe import mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as mkm_kernel
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    m = model.model
    layers = [m.layers[i].mlp for i in range(m.start_layer, m.end_layer)]
    for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
        reason = unsupported(mlp)
        if reason:
            return Bound(reason=f"layer {i}: {reason}")
    device = torch.cuda.current_device()
    router.build(device, max_rows)
    if not fused:
        return Bound(route=True)
    for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
        reason = mkm.unsupported(mlp)
        if reason:
            return Bound(route=True, reason=f"mkm: layer {i}: {reason}")
    mkm_kernel.build(device)
    return Bound(
        route=True,
        mkm={
            mlp.layer_id: mkm_kernel.weight_maps(
                mlp.experts.w13_weight, mlp.experts.w2_weight
            )
            for mlp in layers
        },
        identity=torch.arange(
            mkm_kernel.MAX_TOKENS * mkm_kernel.TOP_K, dtype=torch.int32, device="cuda"
        ),
    )


def unsupported(mlp) -> str:
    """Why the router front cannot stand in for ``mlp``'s gate and the runner's input
    quantizer ("" when it can)."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return f"{type(mlp).__name__} is not a Qwen2MoeSparseMoeBlock"
    w = mlp.gate.weight
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (router.EXPERTS, router.HIDDEN):
        return f"router gate {w.dtype} {tuple(w.shape)}"
    experts = mlp.experts
    scale = experts.w13_input_scale_quant
    if scale.dtype != torch.float32 or scale.numel() != 1:
        return f"routed input scale {scale.dtype} {tuple(scale.shape)}"
    if experts.trtllm_padded_hidden_size is not None:
        return "the routed experts' hidden dim is padded"
    if experts.quant_method.quant_config.use_per_token_activation:
        return "per-token activation scales"
    if not mlp.supports_deferred_finalize:
        return "no deferred finalize"
    return ""


def routed(mlp, x: torch.Tensor, fused: Bound | None = None):
    """``FusedMoE.forward_deferred_finalize`` of ``mlp``'s routed experts on ``x``
    [rows, hidden], with the router front's logits and quantized input (``fused``: on
    slot mkm's kernel)."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    from sgl_nvidia_mega.qwen3_5.moe import mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    experts = mlp.experts
    slot = "route" if fused is None else "mkm"
    if (slot, x.shape[0]) not in _ENGAGED:
        _ENGAGED.add((slot, x.shape[0]))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows", slot, x.shape[0]
        )
    logits, x_fp4, x_sf = router.router_gemm_quant(
        x, mlp.gate.weight, experts.w13_input_scale_quant.reshape(1)
    )
    if fused is not None:
        maps = fused.mkm[mlp.layer_id]
        return mkm.routed(mlp, maps, fused.identity, logits, x_fp4, x_sf)
    topk_output = mlp.topk(x, logits)
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise RuntimeError("the router front requires logits-based bypassed TopK")
    dispatch_output = experts.dispatcher.dispatch(
        hidden_states=x, topk_output=topk_output
    )._replace(hidden_states=x_fp4, hidden_states_scale=x_sf.view(torch.float8_e4m3fn))
    with flashinfer_trtllm_deferred_finalize_context():
        combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
    return experts.dispatcher.combine(combine_input=combine_input)
