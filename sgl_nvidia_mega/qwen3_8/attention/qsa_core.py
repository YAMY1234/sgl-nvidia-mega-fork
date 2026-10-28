"""Selected BF16 KV core, over the actual QSA backend's metadata.

The generic ForwardView's metadata is GDN metadata. It is deliberately never
used for attention lengths or logical positions here.
"""

SGLANG_READS = {
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/qsa/glue.py": ("resolve_qsa_sparse_backend",),
    "srt/layers/attention/qwen_sparse_attn_backend.py": (
        "forward_metadata", "sequence_lengths", "row_req_pool_indices",
        "indexer_metadata", "decode_logical_positions", "is_cuda_graph",
        "req_to_token_pool", "req_to_token", "compress_ratio",
    ),
    "srt/layers/radix_attention.py": (
        "layer_id", "tp_q_head_num", "tp_k_head_num", "head_dim", "scaling",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc", "req_pool_indices", "spec_info", "forward_mode",
    ),
    "srt/kernel_plugins/nvidia/binding.py": ("kv_pool", "pool"),
}


def backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend

    return resolve_qsa_sparse_backend(get_attn_backend())


def metadata_reasons(layer, st, sparse):
    """Host-only shape/mode check, before writing any state or launching kernels."""
    fb = st.fb
    if not fb.forward_mode.is_target_verify():
        return ["qsa_core covers target verify only"]
    if fb.req_pool_indices.numel() != 1 or fb.out_cache_loc.numel() != 4:
        return ["qsa_core covers one four-row request"]
    if int(getattr(fb.spec_info, "topk", 1) or 1) != 1:
        return ["qsa_core covers chain verify only"]
    meta = sparse.forward_metadata
    if meta is None or not meta.is_cuda_graph:
        return ["qsa_core requires the QSA graph metadata"]
    indexer = meta.indexer_metadata
    if indexer is None or indexer.decode_logical_positions is None:
        return ["QSA logical query positions are unavailable"]
    if not indexer.defer_block_expansion:
        return ["QSA relative block indices are unavailable"]
    if meta.sequence_lengths.numel() != 4 or meta.row_req_pool_indices is None:
        return ["QSA per-row lengths/request map are unavailable"]
    if meta.row_req_pool_indices.numel() != 4:
        return ["QSA request map must have one entry per query row"]
    if sparse.compress_ratio != 4:
        return ["QSA block ratio must be four"]
    return []


def pool_reasons(view, layer_id):
    import torch

    pool = view.kv_pool.pool
    k, v = pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id)
    if any(t.dtype != torch.bfloat16 or t.ndim != 3 or t.shape[1:] != (1, 256)
           or not t.is_contiguous() for t in (k, v)):
        return ["qsa_core requires contiguous BF16 KV pools [slots,1,256]"]
    return []


def attend(layer, q, k, v, gate, indices, st, sparse):
    from sgl_nvidia_mega.qwen3_8.sm103.attention.core import run_gated

    meta = sparse.forward_metadata
    pool = st.view.kv_pool.pool
    return run_gated(
        q.reshape(4, layer.num_heads, layer.head_dim),
        pool.get_key_buffer(layer.attn.layer_id),
        pool.get_value_buffer(layer.attn.layer_id),
        k.reshape(4, layer.num_kv_heads, layer.head_dim),
        v.reshape(4, layer.num_kv_heads, layer.head_dim), st.fb.out_cache_loc,
        sparse.req_to_token_pool.req_to_token, meta.row_req_pool_indices,
        indices, meta.sequence_lengths, meta.indexer_metadata.decode_logical_positions,
        layer.attn.scaling, gate, width=4,
    ).reshape(4, -1)
