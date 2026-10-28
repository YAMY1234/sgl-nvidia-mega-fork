"""FN attention slots, using generic ForwardView and native HC/AR boundaries.

All slots off calls the model's own self_attention. The admitted qsa_core ports
only selected-KV + merge/gate. New front/TMA routes are not registered here.
"""

from dataclasses import dataclass
import inspect
import logging
import os

from .config import Dimensions, config_disable_reasons

logger = logging.getLogger(__name__)
ATTENTION_SLOTS = ("qsa_core",)
_ENGAGED = set()

SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/models/qwen4_exp.py": (
        "self_attention", "is_qsa", "indexer",
    ),
    "srt/models/qwen3_5.py": (
        "layers", "start_layer", "end_layer", "num_heads", "num_kv_heads",
        "head_dim", "hidden_size", "o_proj", "_prepare_qkv_gate", "alt_stream",
    ),
    "srt/model_executor/runner_utils/capture_mode.py": (
        "get_is_capture_mode", "is_in_breakable_cuda_graph",
    ),
}


@dataclass(frozen=True)
class Bound:
    qsa_core: bool = False
    reason: str = ""
    projected_qk: bool = False


def build(model, ctx, hf_config, requested):
    requested = set(requested) & set(ATTENTION_SLOTS)
    if not requested:
        return Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    from sglang.srt.runtime_context import get_parallel

    reasons = config_disable_reasons(
        hf_config, get_parallel().tp_size, ctx.tokens_per_request,
    )
    dims = Dimensions.from_config(hf_config)
    layers = [model.model.layers[i] for i in range(
        model.model.start_layer, model.model.end_layer,
    ) if getattr(model.model.layers[i], "is_qsa", False)]
    if not layers:
        reasons.append("no FN QSA attention layers found")
    if not reasons:
        import torch

        qh, kvh = dims.local_heads(get_parallel().tp_size)
        for layer in layers:
            if (layer.num_heads, layer.num_kv_heads, layer.head_dim) != (qh, kvh, dims.head_dim):
                reasons.append(f"layer {layer.layer_id}: HF/local attention dimensions differ")
            if any(proj.weight.dtype != torch.bfloat16 for proj in (layer.qkv_proj, layer.o_proj)):
                reasons.append(f"layer {layer.layer_id}: FN attention requires BF16 qkv/o_proj weights")
    if reasons:
        return Bound(reason="; ".join(reasons))
    accepts_projection = all("projected_qk" in inspect.signature(
        layer.indexer.project_qk,
    ).parameters for layer in layers)
    return Bound(
        qsa_core="qsa_core" in requested, projected_qk=accepts_projection,
    )


def step(layer, x, st):
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode, is_in_breakable_cuda_graph,
    )
    from . import indexer, qsa_core

    built = st.bound["attention"]
    if not st.slots.qsa_core:
        return layer.self_attention(
            positions=st.positions, hidden_states=x, forward_batch=st.fb,
        )
    sparse = qsa_core.backend()
    if (x.shape[0] != 4 or not x.is_cuda or is_in_breakable_cuda_graph()
            or qsa_core.metadata_reasons(layer, st, sparse)
            or qsa_core.pool_reasons(st.view, layer.attn.layer_id)):
        return layer.self_attention(
            positions=st.positions, hidden_states=x, forward_batch=st.fb,
        )
    import torch

    if x.dtype != torch.bfloat16:
        return layer.self_attention(
            positions=st.positions, hidden_states=x, forward_batch=st.fb,
        )
    overlap = layer.alt_stream is not None and get_is_capture_mode()
    current = torch.cuda.current_stream()
    projected = None
    frontier = None
    if overlap and built.projected_qk and os.environ.get("Q38_QSA_PDL_INDEX", "").lower() in ("1", "true"):
        projected = indexer.projection(layer, x)
        if projected is not None:
            from .capture_fork import projection_frontier

            frontier = projection_frontier(current)
    if overlap and projected is None:
        layer.alt_stream.wait_stream(current)
    q, k, v, gate = layer._prepare_qkv_gate(
        positions=st.positions, hidden_states=x, forward_batch=st.fb,
    )
    if overlap:
        if projected is not None:
            layer.alt_stream.wait_stream(current)
            if frontier is not None:
                from .capture_fork import fork_after_projection

                fork_after_projection(layer.alt_stream, frontier)
        with torch.cuda.stream(layer.alt_stream):
            indices = indexer.select(layer, x, st.positions, sparse, projected_qk=projected)
        if projected is not None:
            projected.record_stream(layer.alt_stream)
        current.wait_stream(layer.alt_stream)
        indices.record_stream(current)
    else:
        indices = indexer.select(layer, x, st.positions, sparse)
    output = qsa_core.attend(layer, q, k, v, gate, indices, st, sparse)
    if x.shape[0] not in _ENGAGED:
        _ENGAGED.add(x.shape[0])
        logger.info("FN qsa_core route: rows=%s Q=%s KV=%s", x.shape[0], layer.num_heads, layer.num_kv_heads)
    # RowParallelLinear returns the TP-local BF16 output. The FN boundary owns
    # native FI/CA dispatch, real communication completion, and HC combine.
    out, _ = layer.o_proj(output)
    return out
