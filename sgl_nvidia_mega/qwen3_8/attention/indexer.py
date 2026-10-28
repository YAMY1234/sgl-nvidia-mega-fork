"""Native indexer seam: native Q/K preparation and cache compression, B score,
and the admitted b76 exact selector. No M1 seed or draft-extend capture writes.
"""

SGLANG_READS = {
    "srt/layers/attention/qsa/qsa_indexer.py": (
        "project_qk", "update_key_state_and_compress", "_pending_ring_slots",
        "index_qk_proj", "_use_fused_prep",
    ),
    "srt/layers/attention/qsa/metadata.py": (
        "pending_ring_slots", "compress_member_rows", "token_to_kv_pool",
        "get_decode_mqa_inputs", "req_pool_indices", "decode_logical_positions",
    ),
    "srt/layers/quantization/unquant.py": (
        "UnquantizedLinearMethod", "get_bf16_gemm_backend",
        "should_enable_bf16_splitk_gemm", "use_bf16_splitk_gemm",
    ),
}


def select(layer, x, positions, sparse, *, projected_qk=None):
    """One target-verify request, after the caller validated graph metadata.

    Logical token positions come from QSA metadata, independently of RoPE.
    The native methods retain ring-buffer and compression ownership.
    """
    from sgl_nvidia_mega.qwen3_8.sm103.attention.score import score_mk
    from sgl_nvidia_mega.qwen3_8.sm103.attention.topk import topk_mk

    indexer = layer.indexer
    meta = sparse.forward_metadata.indexer_metadata
    logical = meta.decode_logical_positions
    state_slots = meta.pending_ring_slots
    if state_slots is None:
        state_slots = indexer._pending_ring_slots(
            meta, logical, meta.compress_member_rows is not None,
        )
    kwargs = {} if projected_qk is None else {"projected_qk": projected_qk}
    q, token_k, stored = indexer.project_qk(
        x, positions, pool=meta.token_to_kv_pool, cache_loc=state_slots,
        q_heads_padded=8, **kwargs,
    )
    indexer.update_key_state_and_compress(
        token_k, logical, positions, meta, state_slots=state_slots,
        state_stored=stored,
    )
    cache, table, lengths, capacity = meta.get_decode_mqa_inputs(indexer.layer_id)
    logits = score_mk(q, cache, table, lengths, capacity, meta.req_pool_indices)
    return topk_mk(logits, lengths, topk=indexer.block_topk, row_starts=None)


def projection(layer, x):
    """The mature BF16 projection before QKV, when its existing PDL route applies.

    The caller only invokes this when the native indexer accepts projected_qk.
    An unsupported projection returns None and retains the full native front.
    """
    import torch
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedLinearMethod,
        get_bf16_gemm_backend,
        should_enable_bf16_splitk_gemm,
        use_bf16_splitk_gemm,
    )

    proj = layer.indexer.index_qk_proj
    if (not isinstance(proj.quant_method, UnquantizedLinearMethod)
            or proj.weight.dtype != torch.bfloat16
            or not layer.indexer._use_fused_prep(x)
            or not should_enable_bf16_splitk_gemm(get_bf16_gemm_backend())
            or not use_bf16_splitk_gemm(x.shape[0], *proj.weight.shape)):
        return None
    output, _ = proj(x)
    return output
