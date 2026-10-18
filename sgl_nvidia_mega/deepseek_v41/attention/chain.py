# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention chain at 1-48 rows: the stock step with the megakernel's kernels in
place of single ops, each launched as a programmatic dependent (PDL) of the last.

- q-LoRA RMSNorm + MXFP8 quant (1-48 rows), then wq_b on the low-latency MXFP8 GEMM;
- at 1-24 rows the query keeps the 16 local heads, unroped, and the swap-AB sparse
  attention ropes it as it loads it; its narrow combine triggers wo_a at entry;
- the fused inverse RoPE + wo_a + MXFP8, which triggers wo_b at entry (1-48 rows).
At 25-48 rows the attention itself is SGLang's FlashMLA on the 64-head query."""

from __future__ import annotations

from typing import NamedTuple

import torch

from sgl_nvidia_mega.deepseek_v41.attention import ops
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab
from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a as wo_a_kernel
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

NORM_QUANT_MAX_ROWS = 48
SWAPAB_MAX_ROWS = 24  # swap-AB served slower than FlashMLA at 48 rows
WO_A_MAX_ROWS = wo_a_kernel.MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "wq_b",
        "q_norm",
        "wo_a",
        "freqs_cis",
        "n_local_heads",
        "n_local_groups",
        "o_lora_rank",
        "head_dim",
        "q_lora_rank",
        "layer_id",
        "compress_ratio",
        "use_fused_wo_a",
        "use_flashinfer_mxfp8_wo_b",
        "_local_attn_sink",
        "q_head_norm",
        "softmax_scale",
        "wqkv_a",
        "fuse_wqa_wkv",
    ),
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
    "kernels/ops/attention/dsv4/__init__.py": ("fused_q_norm_rope",),
    "srt/layers/quantization/fp8.py": ("weight_scale_inv_swizzled",),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DSV4AttnMetadata",
        "forward_metadata",
        "core_attn_metadata",
        "token_to_kv_pool",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
        "_pad_tensor_to_size",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "request_window",
        "get_swa_key_buffer_radix",
        "get_swa_key_page_size",
        "get_swa_key_bytes_per_token",
        "get_extra_key_buffer",
        "get_extra_key_page_size",
        "get_extra_key_bytes_per_token",
    ),
    "srt/runtime_context.py": ("get_forward", '"sp_active"'),
}


def _ll_linear_applies(linear) -> bool:
    return (
        mxfp8_ll.supported()
        and getattr(linear, "weight_scale_inv_swizzled", None) is not None
        and linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight.is_contiguous()
    )


def swapab_applies(attn) -> bool:
    """The layer's query fits the swap-AB kernel: 16 local heads of 512, no head norm."""
    return (
        not attn.q_head_norm
        and attn.n_local_heads == swapab.NUM_HEADS
        and attn.head_dim == swapab.HEAD_DIM
        and attn.softmax_scale == swapab.SOFTMAX_SCALE
    )


def swapab_takes(rows: int, fb, attn_backend) -> bool:
    """This step's swap-AB conditions, from host state every rank shares: a decode or
    verify step of 1-24 rows on the paged V4-layout window cache."""
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata

    mode = fb.forward_mode
    pool = attn_backend.token_to_kv_pool
    return (
        0 < rows <= SWAPAB_MAX_ROWS
        and (mode.is_decode() or mode.is_target_verify())
        and isinstance(
            attn_backend.forward_metadata.core_attn_metadata, DSV4AttnMetadata
        )
        and pool.request_window is None
        and pool.get_swa_key_bytes_per_token() == swapab.LAYOUT.bytes_per_token
    )


class Mxfp8Rows(NamedTuple):
    """wq_b's input from the chain's norm: e4m3 codes [rows, K] and UE8M0 scales, in
    FlashInfer's 128x4 swizzle up to 8 rows and per-8-row tiles beyond, the two
    layouts the low-latency GEMM reads. Only ``ChainOps.compute_q_b`` consumes it."""

    quant: torch.Tensor
    scale: torch.Tensor


def wq_b_ll_applies(attn) -> bool:
    """wq_b runs on the low-latency GEMM, fed by the chain's PDL norm + quant."""
    return _ll_linear_applies(attn.wq_b) and attn.q_norm.weight.dtype == torch.bfloat16


class ChainOps:
    """``ops`` with the chain's query kernels; the prepare paths call either. Which
    kernels run is fixed at build (``wq_b_ll``, ``rope_in_attention``)."""

    def __init__(self, attn, *, rope_in_attention: bool):
        self.rope_in_attention = rope_in_attention
        self.wq_b_ll = wq_b_ll_applies(attn)
        self.kv_to_cache = ops.kv_to_cache

    def normalize_q_lora(self, attn, q):
        if not self.wq_b_ll:
            return ops.normalize_q_lora(attn, q)
        y, quant, scale = rmsnorm_mxfp8(
            q,
            attn.q_norm.weight,
            attn.q_norm.variance_epsilon,
            pdl=True,
            row_tiles=q.shape[0] > mxfp8_ll.MAX_ROWS,
        )
        return y, Mxfp8Rows(quant, scale)

    def compute_q_a(self, attn, x, qkv_a):
        q = qkv_a[..., : attn.q_lora_rank] if qkv_a is not None else attn.wq_a(x)[0]
        return self.normalize_q_lora(attn, q)

    def compute_q_b(self, attn, q, positions, q_out):
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope

        if self.wq_b_ll:
            out = mxfp8_ll.linear(
                q.quant, q.scale, attn.wq_b.weight, attn.wq_b.weight_scale_inv_swizzled
            )
        elif self.rope_in_attention:
            out = attn.wq_b(q)[0]
        else:
            return ops.compute_q_b(attn, q, positions, q_out)
        out = out.view(-1, attn.n_local_heads, attn.head_dim)
        if self.rope_in_attention:
            return out
        if q_out is None:
            q_out = torch.empty_like(out)
        fused_q_norm_rope(out, q_out, None, attn.freqs_cis, positions)
        return q_out


def resolve_caches(attn, attn_backend, rows: int):
    """The window and compressed KV-cache views and their sparse indices, as the
    backend's ``_forward_attention`` resolves them for a decode or verify step."""
    from sglang.srt.layers.attention.deepseek_v4_backend import _pad_tensor_to_size

    meta = attn_backend.forward_metadata.core_attn_metadata
    pool = attn_backend.token_to_kv_pool
    layer_id, ratio = attn.layer_id, attn.compress_ratio
    swa = pool.get_swa_key_buffer_radix(layer_id)
    page, width = pool.get_swa_key_page_size(), pool.get_swa_key_bytes_per_token()
    swa = swa[:, : page * width].view(swa.shape[0], page, 1, width)
    extra, extra_indices, extra_lengths = None, None, None
    if ratio != 0:
        extra = pool.get_extra_key_buffer(layer_id)
        page = pool.get_extra_key_page_size(layer_id)
        width = pool.get_extra_key_bytes_per_token(layer_id)
        extra = extra[:, : page * width].view(extra.shape[0], page, 1, width)
        extra_indices = meta.sparse_page_indices(ratio)
        extra_lengths = meta.sparse_topk_lengths(ratio)

    def match(x, value):
        if x is None or x.shape[0] == rows:
            return x
        if x.shape[0] > rows:
            return x[:rows]
        return _pad_tensor_to_size(x, rows, value=value)

    swa_indices = match(meta.swa_page_indices, 0)
    if swa_indices.ndim == 2:
        swa_indices = swa_indices.unsqueeze(1)
    extra_indices = match(extra_indices, -1)
    if extra_indices is not None and extra_indices.ndim == 2:
        extra_indices = extra_indices.unsqueeze(1)
    return (
        swa,
        swa_indices,
        match(meta.swa_topk_lengths, 1),
        extra,
        extra_indices,
        match(extra_lengths, 1),
    )


def sparse_decode(attn, attn_backend, q, positions, freqs):
    """Swap-AB on the 16 local heads in place of the backend's swap-AB / FlashMLA
    choice; ropes ``q`` (unroped, [rows, 16, 512]) as it loads it."""
    swa, swa_indices, swa_lengths, extra, extra_indices, extra_lengths = resolve_caches(
        attn, attn_backend, q.shape[0]
    )
    return swapab.swapab_attention(
        q.unsqueeze(1),
        swa,
        swa_indices,
        swa_lengths,
        attn._local_attn_sink(attn.n_local_heads),
        extra,
        extra_indices,
        extra_lengths,
        max_rows=SWAPAB_MAX_ROWS,
        pdl=True,
        narrow_combine=True,
        q_rope=(freqs, positions),
        # the KV is read after the PDL wait: gathering it before needs proof that
        # the captured side-stream joins are full graph edges
        kv_before_wait=False,
        combine_entry_trigger=True,
    )


def wo_a(attn, o, positions, fb):
    """The fused inverse RoPE + wo_a + MXFP8 with the early trigger, where SGLang's
    own step fuses them; SGLang's arm otherwise."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.runtime_context import get_forward

    if not (
        attn.use_fused_wo_a
        and attn.use_flashinfer_mxfp8_wo_b
        and not get_forward().sp_active
        and 0 < o.shape[0] <= WO_A_MAX_ROWS
    ):
        return ops.wo_a(attn, o, positions, fb)
    o = o.view(o.shape[0], attn.n_local_groups, -1)
    codes, scales = wo_a_kernel.fused_rope_wo_a(
        o,
        attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1),
        torch.view_as_real(attn.freqs_cis).flatten(1),
        positions,
        early_trigger=True,
    )
    return Mxfp8SwizzledInput(codes, scales)


def precompile(attn, widths) -> None:
    """Compile the chain's kernels that do not depend on the KV pool, for these widths:
    the low-latency wq_b GEMM, the fused wo_a, and the norm + quant (its row count and
    input row stride are compile-time constants). Called at build, before capture.
    The swap-AB kernels specialize on the pool's page geometry and the step's index
    widths, which exist only after KV sizing: they compile in SGLang's warm-up runs
    before graph capture."""
    rows = [w for w in widths if w <= NORM_QUANT_MAX_ROWS]
    if not rows:
        return
    if wq_b_ll_applies(attn):
        mxfp8_ll.precompile(attn.wq_b.weight, rows)
        k = attn.q_lora_rank
        # q_lora as the prepare paths slice it: a view of wqkv_a's output when fused
        width = attn.wqkv_a.weight.shape[0] if attn.fuse_wqa_wkv else k
        x = torch.zeros(
            max(rows), width, dtype=torch.bfloat16, device=attn.q_norm.weight.device
        )
        for r in rows:
            rmsnorm_mxfp8(
                x[:r, :k],
                attn.q_norm.weight,
                attn.q_norm.variance_epsilon,
                pdl=True,
                row_tiles=r > mxfp8_ll.MAX_ROWS,
            )
    if attn.use_fused_wo_a:
        wo_a_kernel.precompile(min(max(rows), WO_A_MAX_ROWS))
