# SPDX-License-Identifier: Apache-2.0
"""FN bitwise Triton/ReplaySSM verify, distinct from numeric slot gdn_mk."""
import logging

SGLANG_READS = {
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "qwen3_5_gdn_prefill_projection_views",),
    "srt/models/qwen3_5.py": (
        "_forward_input_proj", "attn_tp_size", "num_k_heads", "num_v_heads",
        "head_k_dim", "head_v_dim", "norm", "out_proj",
        "_fused_in_proj_weight"),
    "srt/layers/attention/linear/gdn_backend.py": (
        "verify_intermediate_state_indices",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend", "forward_metadata", "state_layer_cache", "state_pool"),
    "srt/layers/radix_linear_attention.py": (
        "layer_id", "q_dim", "k_dim", "v_dim", "conv_weights", "bias",
        "activation", "A_log", "dt_bias"),
}
_ENGAGED = set()
logger = logging.getLogger(__name__)


def backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend
    return get_attn_backend().linear_attn_backend


def context(attn, st):
    """Read the active native GDN backend; never construct an Eagle view."""
    impl = backend()
    meta = impl.forward_metadata
    pool = impl.state_pool
    cache = impl.state_layer_cache(attn.layer_id)
    return impl, meta, pool, cache


def unsupported(gdn):
    if gdn.norm.activation != "sigmoid":
        return "FN GDN requires sigmoid output gating"
    if gdn.norm.group_size is not None or not gdn.norm.norm_before_gate:
        return "GDN requires ungrouped norm-before-gate"
    if (gdn.out_proj.reduce_results or not gdn.out_proj.input_is_parallel
            or gdn.out_proj.bias is not None):
        return "GDN out_proj must preserve the TP-partial boundary"
    return ""


def step(layer, x, st):
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        qwen3_5_gdn_prefill_projection_views,
    )
    from .conv import conv_update
    from . import replayssm

    gdn = layer.linear_attn
    bound = st.bound["gdn"]
    shape = bound.shape
    rows = x.shape[0]
    draft = shape.tokens_per_request
    requests = rows // draft
    key = (rows, st.slots.gdn_pdl, st.slots.gdn_mk)
    if key not in _ENGAGED:
        logger.info("NVIDIA megakernel: FN GDN rows=%d T=%d pdl=%d mk=%d",
                    rows, draft, st.slots.gdn_pdl, st.slots.gdn_mk)
        _ENGAGED.add(key)
    attn = gdn.attn
    impl, meta, pool, cache = context(attn, st)
    states = meta.mamba_cache_indices
    if st.slots.gdn_mk:
        from sgl_nvidia_mega.qwen3_8.sm103.gdn.front import run as mature_front
        packed, convolved = mature_front(
            x, gdn._fused_in_proj_weight, attn.conv_weights, attn.bias,
            cache.conv[0], states[:requests], cache.intermediate_conv_window[0],
            impl.verify_intermediate_state_indices[:requests], shape=shape)
        qkvz_width = shape.packed - 2*shape.hv
        qkvz, ba = packed[:, :qkvz_width], packed[:, qkvz_width:]
        _, z, b, a = qwen3_5_gdn_prefill_projection_views(
            qkvz, ba, shape.hk, shape.hv, shape.key_dim, shape.value_dim)
    else:
        qkvz, ba = gdn._forward_input_proj(x)
        mixed, z, b, a = qwen3_5_gdn_prefill_projection_views(
            qkvz, ba, shape.hk, shape.hv, shape.key_dim, shape.value_dim)
        convolved = conv_update(
            mixed.view(requests, draft, -1).transpose(1, 2),
            cache.conv[0], attn.conv_weights, attn.bias, attn.activation,
            states[:requests], cache.intermediate_conv_window[0],
            impl.verify_intermediate_state_indices[:requests],
            block_n=128, token_major_out=True)
        convolved = convolved.transpose(1, 2).view(rows, -1)
    q, k, v = convolved.split((attn.q_dim, attn.k_dim, attn.v_dim), dim=-1)
    q = q.view(rows, shape.hk, shape.key_dim)
    k = k.view(rows, shape.hk, shape.key_dim)
    v = v.view(rows, shape.hv, shape.value_dim)
    out = v.new_empty(v.shape)
    if st.slots.gdn_mk:
        from sgl_nvidia_mega.qwen3_8.sm103.gdn.core import core
        normalized = out.new_empty((rows, shape.hv*shape.value_dim))
        core(convolved, a.reshape(rows, shape.hv), b.reshape(rows, shape.hv),
             attn.A_log, attn.dt_bias, cache.temporal,
             cache.replayssm_d, cache.replayssm_k, cache.replayssm_g,
             cache.replayssm_rawv, cache.replayssm_rawk,
             states, st.fb.req_pool_indices, pool.replayssm_spec_write_pos,
             pool.replayssm_cache_base, pool.replayssm_is_flush, out,
             z=z.reshape(rows, shape.hv*shape.value_dim), norm_weight=gdn.norm.weight,
             normalized=normalized, eps=gdn.norm.eps, tokens_per_request=draft)
        norm = normalized
    else:
        replayssm.verify(
            q, k, v, a.reshape(rows, shape.hv), b.reshape(rows, shape.hv),
            attn.A_log, attn.dt_bias, out, cache.temporal,
            cache.replayssm_d, cache.replayssm_k, cache.replayssm_g,
            cache.replayssm_rawv, cache.replayssm_rawk, cache.replayssm_beta,
            meta.query_start_loc, states, st.fb.req_pool_indices,
            pool.replayssm_spec_write_pos, pool.replayssm_cache_base,
            pool.replayssm_is_flush,
            max_cache_len=cache.replayssm_d.shape[-2], max_spec_len=draft,
            scale=shape.key_dim**-0.5, null_block_id=-1,
            block_v=64, num_warps=1, pdl=st.slots.gdn_pdl,
            wait_for_gate_producer=True)
        norm = gdn.norm(out.reshape(-1, shape.value_dim),
                        z.reshape(-1, shape.value_dim))
    from sgl_nvidia_mega.qwen3_8.runtime import engaged
    engaged.mark(st, "gdn_mk" if st.slots.gdn_mk else "gdn_verify")
    if st.slots.gdn_pdl:
        engaged.mark(st, "gdn_pdl")
    return gdn.out_proj(norm.reshape(rows, shape.hv*shape.value_dim))[0]
