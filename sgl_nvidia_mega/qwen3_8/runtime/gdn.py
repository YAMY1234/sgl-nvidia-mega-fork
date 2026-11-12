# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet.forward) and
# python/sglang/srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_extend,
# its circular ReplaySSM target-verify branch, and _replayssm_target_verify).
"""The GDN layer step of a covered forward.

Target verify with the circular ReplaySSM spec rings (``--enable-linear-replayssm-spec``)
runs here, slot ``gdn_native_pdl``: the input projection, the q/k/v/z/b/a split, the
causal conv update into the verify's intermediate windows, the q/k/v split, the ReplaySSM
verify core (the plugin's native kernel: SGLang's circular verify kernel launched with
PDL), the gated RMSNorm and the output projection. Every other mode, and verify without
the slot, runs SGLang's ``GatedDeltaNet.forward``.

What the verify reads and writes (the pools are SGLang's): the conv states by state
slot (the conv update also writes them; SGLang's commit restores them), the
intermediate conv windows by verify row, the checkpoint (``temporal``) read-only, the
ring entries of this verify by request slot, and the cursors read-only (SGLang's commit
advances them)."""

from __future__ import annotations

import torch

STOCK = "stock"

SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "_forward_input_proj",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "attn_tp_size",
        "norm",
        "out_proj",
        "attn",
    ),
    "srt/layers/attention/linear/gdn_backend.py": (
        "forward_metadata",
        "verify_intermediate_state_indices",
        "_use_strided_target_verify_qkv",
        "MAX_FUSED_QKV_SPLIT_DIM",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "state_layer_cache",
        "state_pool",
        "req_to_token_pool",
    ),
    "srt/models/qwen4_exp.py": ("Qwen4ExpLinearDecoderLayer",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/radix_linear_attention.py": (
        "conv_weights",
        "bias",
        "activation",
        "layer_id",
        "A_log",
        "dt_bias",
        "q_dim",
        "k_dim",
        "v_dim",
        "num_q_heads",
        "head_q_dim",
    ),
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "fused_qkvzba_split_reshape_cat_contiguous",
        "fused_qkv_split_gdn_prefill",
    ),
    "kernels/ops/mamba/causal_conv1d_triton.py": ("causal_conv1d_update",),
    "srt/layers/attention/linear/utils.py": ("select_verify_intermediate_state_indices",),
}


def linear_backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    return getattr(backend, "linear_attn_backend", backend)


def verify_reasons(model) -> str:
    """'' when every GDN layer's verify can run the native core, else why not."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if not isinstance(layer, Qwen4ExpLinearDecoderLayer):
            continue
        attn = layer.linear_attn
        h, hv = attn.num_k_heads // attn.attn_tp_size, attn.num_v_heads // attn.attn_tp_size
        if (h, hv) not in ((4, 12), (16, 48)):
            return f"GDN heads per rank {(h, hv)}, the kernel takes (4, 12) or (16, 48)"
        if attn.head_k_dim != 128 or attn.head_v_dim != 128:
            return "GDN head dims are not 128"
        if attn.num_v_heads // attn.num_k_heads != 3:
            return "the fused q/k/v/z/b/a split needs 3 value heads per key head"
    return ""


def pool_reasons(backend, layer_id: int, tokens_per_request: int) -> str:
    """'' when the pool holds the circular ReplaySSM spec rings the core takes."""
    pool = backend.state_pool
    cache = backend.state_layer_cache(layer_id)
    if cache.replayssm_d is None or pool.replayssm_cache_base is None:
        return "the pool has no circular ReplaySSM spec rings (--enable-linear-replayssm-spec)"
    if pool.replayssm_spec_fold or pool.replayssm_is_kda:
        return "the pool's ReplaySSM rings fold every commit or are KDA's"
    if cache.replayssm_d.shape[-2] != 16:
        return f"ReplaySSM ring length {cache.replayssm_d.shape[-2]}, the kernel takes 16"
    if cache.temporal.dtype not in (torch.bfloat16, torch.float32):
        return f"GDN checkpoint dtype {cache.temporal.dtype}"
    if tokens_per_request != 4:
        return f"{tokens_per_request} verify rows a request, the kernel takes 4"
    return ""


def step(executor, layer, x, st):
    """The layer's mixer output ([rows, hidden], before the stage's reduction)."""
    fb = st.fb
    if not (executor.gdn_pdl_ready and fb.forward_mode.is_target_verify()):
        st.ran("gdn", STOCK)
        return layer.linear_attn(x, fb)
    return _verify(executor, layer.linear_attn, x, st)


def _verify(executor, gdn, x, st):
    """``Qwen3_5GatedDeltaNet.forward`` for a target verify without packed verify."""
    import triton
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    fb = st.fb
    qkvz, ba = gdn._forward_input_proj(x)
    nk = triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size)
    nv = triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size)
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz, ba, nk, nv, gdn.head_k_dim, gdn.head_v_dim
    )
    core = _verify_core(executor, gdn.attn, fb, mixed_qkv, a, b)
    st.ran("gdn", "native_pdl")
    z_shape = z.shape
    core = core.reshape(-1, core.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    if core.shape != z.shape:  # DP-attention padding
        padded = z.new_zeros(z.shape)
        padded[: core.shape[0], :] = core
        core = padded
    core = gdn.norm(core, z)
    core = core.reshape(z_shape)
    core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
    out, _ = gdn.out_proj(core)
    return out


def _verify_core(executor, layer, fb, mixed_qkv, a, b):
    """``GDNAttnBackend.forward_extend``'s circular-ReplaySSM target-verify branch
    with the native PDL core; returns [rows, HV, V]."""
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )

    backend = linear_backend()
    seq_len = mixed_qkv.shape[0]
    md = backend.forward_metadata
    query_start_loc = md.query_start_loc
    cache_indices = md.mamba_cache_indices
    cache = backend.state_layer_cache(layer.layer_id)
    window = cache.intermediate_conv_window[0]
    rows_state = select_verify_intermediate_state_indices(
        backend.verify_intermediate_state_indices,
        fb.req_pool_indices,
        cache_indices[: query_start_loc.shape[0] - 1] >= 0,
        backend.req_to_token_pool.size,
    )
    tokens = fb.spec_info.draft_token_num
    batch = seq_len // tokens
    conv_in = mixed_qkv.view(batch, tokens, -1).transpose(1, 2)
    conv_out = causal_conv1d_update(
        conv_in,
        cache.conv[0],
        layer.conv_weights,
        layer.bias,
        layer.activation,
        conv_state_indices=cache_indices[:batch],
        intermediate_conv_window=window,
        intermediate_state_indices=rows_state[:batch],
        retrieve_next_token=md.retrieve_next_token,
        retrieve_next_sibling=md.retrieve_next_sibling,
        retrieve_parent_token=md.retrieve_parent_token,
    )
    mixed_qkv = conv_out.transpose(1, 2).view(seq_len, -1)
    query, key, value = fused_qkv_split_gdn_prefill(
        mixed_qkv,
        layer.num_q_heads,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_q_dim,
        layer.head_k_dim,
        layer.head_v_dim,
    )
    return _native_pdl_core(
        executor,
        layer=layer,
        backend=backend,
        cache=cache,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        cache_indices=cache_indices,
        replay_indices=fb.req_pool_indices,
        query_start_loc=query_start_loc,
    )


def _native_pdl_core(
    executor, *, layer, backend, cache, query, key, value, a, b, cache_indices,
    replay_indices, query_start_loc
):
    """``_replayssm_target_verify`` with the plugin's PDL launch of the same circular
    verify kernel (``gdn_front/replayssm_pdl.py``)."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front.replayssm_pdl import (
        gdn_replayssm_spec_circular_kernel,
    )

    pool = backend.state_pool
    h, k_dim = layer.num_k_heads, layer.head_k_dim
    hv, v_dim = layer.num_v_heads, layer.head_v_dim
    seq_len = query.numel() // (h * k_dim)
    q = query.reshape(seq_len, h, k_dim)
    k = key.reshape(seq_len, h, k_dim)
    v = value.reshape(seq_len, hv, v_dim)
    a = a.reshape(seq_len, hv)
    b = b.reshape(seq_len, hv)
    out = q.new_empty(seq_len, hv, v_dim)
    bsz = query_start_loc.shape[0] - 1
    d, kc, g = cache.replayssm_d, cache.replayssm_k, cache.replayssm_g
    rawv = cache.replayssm_rawv if cache.replayssm_rawv is not None else d
    rawk = cache.replayssm_rawk if cache.replayssm_rawk is not None else kc
    beta = cache.replayssm_beta if cache.replayssm_beta is not None else g
    storage = 2 if cache.replayssm_beta is not None else (1 if cache.replayssm_rawv is not None else 0)
    gdn_replayssm_spec_circular_kernel[(2, bsz, hv)](
        q, k, v, a, b, layer.A_log, layer.dt_bias, out,
        cache.temporal, d, kc, g, rawv, rawk, beta,
        query_start_loc, cache_indices, replay_indices,
        pool.replayssm_spec_write_pos, pool.replayssm_cache_base, pool.replayssm_is_flush,
        k_dim**-0.5,
        q.stride(0), k.stride(0), v.stride(0), a.stride(0), b.stride(0), out.stride(0),
        cache.temporal.stride(0), d.stride(0), kc.stride(0), g.stride(0),
        rawv.stride(0), rawk.stride(0), beta.stride(0),
        query_start_loc.stride(0), cache_indices.stride(0), replay_indices.stride(0),
        H=h, HV=hv, K=k_dim, V=v_dim, BK=128, BV=64, BS=4, BC=16, NK=2, BKT=64,
        MAX_CACHE_LEN=16,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        STATE_FP32=cache.temporal.dtype == torch.float32,
        STORE_EXACT_RAW=storage == 2,
        STORE_D_RESIDUAL=storage == 1,
        STORE_K_RESIDUAL=storage == 1,
        IS_FLUSH=False,
        NULL_BLOCK_ID=-1,
        DOT_PRECISION="tf32",
        num_warps=1,
        num_stages=2,
        USE_GDC=True,
        launch_pdl=True,
        WAIT_FOR_GATE_PRODUCER=False,
        PREFETCH_CHECKPOINT=False,
    )
    return out.reshape(value.shape)
