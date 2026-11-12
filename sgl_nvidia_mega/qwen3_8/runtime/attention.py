# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpAttentionDecoderLayer.self_attention).
"""The QSA attention layer step of a covered forward.

Slot ``qsa_pool``: a TP1 target verify of up to 64 rows (16 requests x 4) on a BF16 KV
pool runs the sparse attention as the plugin's split-K pool kernels
(``qsa/pool_decode.py``): the new K/V write at the step's slots, the split-K partials
over the selected blocks (and the request's own uncompressed tail) and the merge, which
also applies the sigmoid output gate. The QKV / gate projection with its norms and
RoPE, the indexer (selection) and o_proj stay SGLang's. Every other step runs SGLang's
``self_attention``. Numerics: split-K online softmax against TRT-LLM's sparse decode, a
tolerance slot.

Reads: the layer's K/V pool, ``req_to_token``, the backend's step metadata (row
requests, logical positions). Writes: the new K/V at ``out_cache_loc``."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

STOCK = "stock"
# Test only: on eager steps, also run SGLang's attention on the same q/k/v and selected
# indices and log how far the pool kernel's output is from it, per row.
OPCHECK = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_OPCHECK") == "1"
MAX_ROWS = 64
QUERY_HEADS, KV_HEADS, HEAD_DIM, BLOCKS, RATIO = 24, 2, 256, 512, 4

SGLANG_READS = {
    "srt/models/qwen4_exp.py": (
        "_compute_qsa_topk_indices",
        "self_attention",
        "is_qsa",
    ),
    "srt/models/qwen3_5.py": ("_prepare_qkv_gate", "o_proj"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "kernels/ops/elementwise/elementwise.py": ("fused_sigmoid_mul",),
    "srt/layers/attention/qsa/glue.py": ("resolve_qsa_sparse_backend",),
    "srt/layers/attention/qwen_sparse_attn_backend.py": (
        "_resolve_metadata",
        "_uses_block_indices",
        "row_req_pool_indices",
        "indexer_metadata",
        "decode_logical_positions",
        "req_to_token_pool",
        "token_to_kv_pool",
    ),
    "srt/mem_cache/kv/router.py": ("read", "key", "value"),
}


def layer_reasons(layer) -> str:
    attn = layer.attn
    if not layer.is_qsa:
        return "not a QSA layer"
    if (attn.tp_q_head_num, attn.tp_k_head_num, attn.head_dim) != (
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
    ):
        return (
            f"QSA heads {(attn.tp_q_head_num, attn.tp_k_head_num, attn.head_dim)}, "
            f"the kernel takes {(QUERY_HEADS, KV_HEADS, HEAD_DIM)} (TP1)"
        )
    return ""


def build() -> str:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the pool kernels need an SM10x GPU"
    try:
        from sgl_nvidia_mega.qwen3_8.qsa import pool_decode  # noqa: F401
    except Exception as e:
        return f"pool kernels do not import: {type(e).__name__}: {e}"
    return ""


def step(executor, layer, x, st):
    """The layer's attention output [rows, hidden] (before the stage's reduction).

    With slot ``qsa_indexer`` or ``qsa_pool``, a decode / verify / draft-extend step
    runs the attention split into its pieces here: SGLang's QKV + gate projection, the
    indexer (``qsa_indexer.select``: the plugin's fold, scorers and selects where they
    serve the step), the sparse attention (the pool kernels where they serve it) and
    the gate and o_proj. Otherwise SGLang's ``self_attention`` runs whole."""
    fb = st.fb
    mode = fb.forward_mode
    if (
        (executor.qsa_indexer is not None or executor.qsa_pool_ready)
        and layer.is_qsa
        and (mode.is_target_verify() or mode.is_decode() or mode.is_draft_extend_v2())
    ):
        return _decomposed(executor, layer, x, st)
    if executor.qsa_prefill is not None and layer.is_qsa and st.step.mode == "prefill":
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_prefill

        out = qsa_prefill.extend(executor, layer, x, st)
        if out is not None:
            return out
    st.ran("qsa", STOCK)
    return layer.self_attention(positions=st.positions, hidden_states=x, forward_batch=fb)


def _pool_eligible(executor, fb, rows) -> bool:
    return (
        executor.qsa_pool_ready
        and fb.forward_mode.is_target_verify()
        and rows <= MAX_ROWS
        and int(fb.spec_info.draft_token_num) == 4
    )


def _decomposed(executor, layer, x, st):
    """``self_attention`` from its pieces. The indexer runs on the main stream here
    (SGLang overlaps it on a side stream under capture); both orders compute the
    same values."""
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer

    fb, positions = st.fb, st.positions
    q, k, v, gate = layer._prepare_qkv_gate(
        positions=positions, hidden_states=x, forward_batch=fb
    )
    if executor.qsa_indexer is not None:
        indices = qsa_indexer.select(executor, layer, x, positions, st)
    else:
        indices = layer._compute_qsa_topk_indices(x, positions, fb)
        st.ran("qsa.indexer", STOCK)
    rows = q.shape[0]
    if _pool_eligible(executor, fb, rows):
        backend = resolve_qsa_sparse_backend(get_attn_backend())
        out = _verify_pool(executor, layer, backend, q, k, v, gate, indices, st)
        if out is not None:
            return out
    attn_out = layer.attn(q, k, v, fb, topk_indices=indices)
    return _finish(layer, attn_out, gate, st, STOCK)


def _verify_pool(executor, layer, backend, q, k, v, gate, indices, st):
    """The pool kernels on a TP1 verify of up to 64 rows, or None (nothing launched)
    when this step's layout is not theirs."""
    fb = st.fb
    pool = backend.token_to_kv_pool
    if pool.layer_router.layer(layer.attn.layer_id).key().dtype != torch.bfloat16:
        return None
    md = backend._resolve_metadata(fb)
    rows = q.shape[0]
    indexer = md.indexer_metadata
    query_positions = None if indexer is None else indexer.decode_logical_positions
    row_requests = md.row_req_pool_indices
    if (
        not backend._uses_block_indices(indices)
        or tuple(indices.shape) != (rows, BLOCKS)
        or query_positions is None
        or row_requests is None
        or rows != fb.req_pool_indices.numel() * 4
    ):
        return None
    kv = pool.layer_router.read(layer.attn.layer_id)
    out = executor.qsa_pool.run(
        q=q.reshape(rows, QUERY_HEADS * HEAD_DIM),
        key_cache=kv.key(),
        value_cache=kv.value(),
        new_k=k.reshape(rows, KV_HEADS, HEAD_DIM),
        new_v=v.reshape(rows, KV_HEADS, HEAD_DIM),
        locations=fb.out_cache_loc,
        slot_map=backend.req_to_token_pool.req_to_token,
        row_requests=row_requests,
        indices=indices.to(torch.int32).contiguous(),
        sequence_lengths=md.sequence_lengths,
        query_positions=query_positions,
        gate=gate,
        scale=layer.attn.scaling,
    )
    out = out.view(rows, QUERY_HEADS * HEAD_DIM)
    if OPCHECK and not torch.cuda.is_current_stream_capturing():
        _opcheck(layer, q, k, v, fb, indices, gate, out, md, query_positions)
    return _finish(layer, out, None, st, "pool_splitk")


def _opcheck(layer, q, k, v, fb, indices, gate, got, md, query_positions):
    """Same selection, SGLang's attention (it rewrites the same K/V at the same slots),
    then the gate as SGLang applies it: the remaining difference is the kernel's."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    want = layer.attn(q, k, v, fb, topk_indices=indices)
    want = fused_sigmoid_mul(want.reshape(got.shape).clone(), gate, inplace=True)
    diff = (want.float() - got.float()).abs()
    per_row = diff.amax(dim=1)
    worst = int(per_row.argmax())
    rel = float((want.float() - got.float()).norm() / want.float().norm().clamp_min(1e-30))
    valid = int((indices[worst] >= 0).sum())
    logger.info(
        "NVIDIA megakernel opcheck qsa_pool layer=%d rows=%d max_abs=%.4g rel_l2=%.3g "
        "worst_row=%d worst_row_max=%.4g worst_row_valid_blocks=%d worst_row_seq_len=%d "
        "worst_row_pos=%d rows_over_0.05=%d",
        layer.attn.layer_id,
        got.shape[0],
        float(diff.max()),
        rel,
        worst,
        float(per_row[worst]),
        valid,
        int(md.sequence_lengths[worst]),
        int(query_positions[worst]),
        int((per_row > 0.05).sum()),
    )


def _finish(layer, attn_out, gate, st, impl):
    """The output gate (unless the merge applied it) and o_proj."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    if gate is not None:
        attn_out = fused_sigmoid_mul(attn_out, gate, inplace=True)
    st.ran("qsa", impl)
    output, _ = layer.o_proj(attn_out)
    return output


class PoolKernels:
    """The plugin's TP1 split-K pool attention for 4-row verify requests, with the
    sigmoid output gate in its merge. Hardware config is resolved eagerly (a capture
    warmup runs every width before its graph is captured)."""

    def __init__(self):
        from sgl_nvidia_mega.qwen3_8.qsa import pool_decode as raw

        self.raw = raw
        self._configs: dict = {}

    def run(
        self, *, q, key_cache, value_cache, new_k, new_v, locations, slot_map,
        row_requests, indices, sequence_lengths, query_positions, gate, scale
    ):
        raw = self.raw
        rows, dim = q.shape[0], HEAD_DIM
        device = q.device
        config_key = (device, rows)
        if config_key not in self._configs:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("qsa_pool: config must be resolved before capture")
            self._configs[config_key] = raw.choose_config(
                rows, KV_HEADS, BLOCKS * RATIO + RATIO - 1, raw._num_sms(device)
            )
        splits, chunk, bn = self._configs[config_key]
        gp = 16
        stats = rows * KV_HEADS * splits * gp
        scratch = torch.empty(stats * (dim + 2), dtype=torch.float32, device=device)
        maximum, sums = scratch[:stats], scratch[stats : 2 * stats]
        acc = scratch[2 * stats :]
        out = torch.empty(rows, QUERY_HEADS, dim, dtype=q.dtype, device=device)
        out2 = out.view(rows, QUERY_HEADS * dim)
        nk2, nv2 = new_k.reshape(rows, KV_HEADS * dim), new_v.reshape(rows, KV_HEADS * dim)
        scale = float(scale) * raw.LOG2E
        raw._qsa_write_new_kv[(rows, KV_HEADS)](
            key_cache, value_cache, nk2, nv2, locations, nk2.stride(0), nv2.stride(0),
            KV_HEADS, dim, True, num_warps=1, launch_pdl=True,
        )
        raw._qsa_pool_splitk[(rows, KV_HEADS, splits)](
            q, key_cache, value_cache, nk2, nv2, locations, slot_map, row_requests,
            indices, sequence_lengths, query_positions, maximum, sums, acc, maximum,
            out2, out2.stride(0), scale, q.stride(0), slot_map.stride(0),
            nk2.stride(0), nv2.stride(0), indices.stride(0),
            KV_HEADS, dim, QUERY_HEADS // KV_HEADS, gp, BLOCKS, RATIO, 4,
            splits, chunk, bn, 0, False, False, True,
            num_warps=4, num_stages=4, launch_pdl=True,
        )
        gate3 = gate.reshape(rows, QUERY_HEADS, dim) if gate.ndim == 2 else gate
        raw._qsa_pool_merge[(rows * QUERY_HEADS,)](
            maximum, sums, acc, out2, out2.stride(0), KV_HEADS, dim,
            QUERY_HEADS // KV_HEADS, gp, splits, min(16, 1 << (splits - 1).bit_length()),
            True, num_warps=2,
            GATE=gate3, GATE_STRIDE_ROW=gate3.stride(0), GATE_STRIDE_HEAD=gate3.stride(1),
            HAS_GATE=True, launch_pdl=True,
        )
        return out
