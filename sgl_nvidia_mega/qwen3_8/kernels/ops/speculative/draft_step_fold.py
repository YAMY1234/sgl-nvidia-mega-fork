# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/speculative/draft_step_fold.py (FoldSlot, make_fold_slots,
# _finalize_and_embed, finalize_next_input). Changed: no switch is read; the shape gate
# is ``fold_reason``; the kernel is unchanged.
"""An MTP draft step's top-1 finalize with the next step's input folded in.

After a draft decode step whose output feeds another one, SGLang's topk-1 chain runs
the split argmax and its finalize (top-1, the draft token column, positions + 1),
then the M-RoPE positions' + 1, and the next step embeds the token. Here the finalize
also advances the M-RoPE positions and gathers the token's embedding row into a slot
the capture owns, which the next draft step takes as its input embedding: the split
argmax is SGLang's, the reduction and tie break its finalize's, so the token, the
positions and the embedding row are bit-identical.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.topk1 import (
    _DRAFT_TOPK1_BLOCK,
    _draft_topk1_partial_argmax_kernel,
)

SGLANG_READS = {
    "kernels/ops/speculative/topk1.py": (
        "_DRAFT_TOPK1_BLOCK",
        "_draft_topk1_partial_argmax_kernel",
    ),
    "srt/models/qwen3_5.py": ("embed_tokens",),
    "srt/models/qwen4_exp_mtp.py": ("hc_count",),
    "srt/layers/vocab_parallel_embedding.py": (
        "UnquantizedEmbeddingMethod",
        "weight",
        "tp_size",
        "quant_method",
        "output_dtype",
        "num_embeddings",
        "org_vocab_size",
    ),
}

HIDDEN = 2560
MAX_ROWS = 4


class FoldSlot(NamedTuple):
    weight: torch.Tensor
    embedding: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    partial_values: torch.Tensor
    partial_indices: torch.Tensor


def fold_reason(model, batch_size, num_steps) -> str:
    """'' when a draft graph of ``batch_size`` requests folds its first step's input
    (the source's gate: 4 HC streams, a 3-step chain, 1-4 requests, an unquantized
    TP1 BF16 [vocab, 2560] embedding)."""
    from sglang.srt.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    embed = model.model.embed_tokens
    weight = embed.weight
    if model.hc_count != 4 or num_steps != 3 or not 1 <= batch_size <= MAX_ROWS:
        return f"{model.hc_count} HC streams, {num_steps} steps, {batch_size} requests"
    if not (
        embed.tp_size == 1
        and isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
        and embed.output_dtype in (None, torch.bfloat16)
        and embed.num_embeddings == embed.org_vocab_size
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and weight.shape[1] == HIDDEN
        and weight.is_contiguous()
    ):
        return "not an unquantized TP1 BF16 [vocab, 2560] embedding"
    return ""


def make_slot(weight, batch_size) -> FoldSlot:
    """A capture's own scratch; nothing is shared across requests or graphs."""
    splits = triton.cdiv(weight.shape[0], _DRAFT_TOPK1_BLOCK)
    device = weight.device
    return FoldSlot(
        weight,
        torch.empty((batch_size, HIDDEN), dtype=torch.bfloat16, device=device),
        torch.empty((batch_size, 1), dtype=torch.float32, device=device),
        torch.empty((batch_size, 1), dtype=torch.int64, device=device),
        torch.empty((batch_size, splits), dtype=torch.float32, device=device),
        torch.empty((batch_size, splits), dtype=torch.int32, device=device),
    )


@triton.jit
def _finalize_and_embed(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    weight,
    embedding,
    weight_stride,
    mrope,
    mrope_axis_stride,
    mrope_row_stride,
    num_splits: tl.constexpr,
    WRITE_MROPE: tl.constexpr,
    HIDDEN: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    EMBED_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_BLOCK)
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=offsets < num_splits,
        other=-float("inf"),
    )
    # Keep the mature finalizer's exact reduction and first-split tie break.
    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)
    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)
    if WRITE_MROPE:
        for axis in tl.static_range(3):
            ptr = mrope + axis * mrope_axis_stride + row * mrope_row_stride
            tl.store(ptr, tl.load(ptr) + 1)

    col = tl.arange(0, EMBED_BLOCK)
    value = tl.load(weight + index * weight_stride + col, col < HIDDEN, other=0)
    tl.store(embedding + row * HIDDEN + col, value, col < HIDDEN)


def finalize_next_input(
    logits, positions, draft_tokens, column, slot, mrope_positions=None
):
    """SGLang's split argmax, then the finalize with the gather into ``slot``."""
    bs, vocab = logits.shape
    splits = slot.partial_values.shape[1]
    assert bs == slot.embedding.shape[0]
    assert vocab <= slot.weight.shape[0]
    assert splits == triton.cdiv(vocab, _DRAFT_TOPK1_BLOCK)
    _draft_topk1_partial_argmax_kernel[(bs, splits)](
        logits,
        slot.partial_values,
        slot.partial_indices,
        logits.stride(0),
        vocab,
        splits,
        BLOCK=_DRAFT_TOPK1_BLOCK,
        num_warps=8,
    )
    _finalize_and_embed[(bs,)](
        slot.partial_values,
        slot.partial_indices,
        slot.topk_p,
        slot.topk_index,
        positions,
        draft_tokens,
        draft_tokens.stride(0),
        column,
        slot.weight,
        slot.embedding,
        slot.weight.stride(0),
        mrope_positions,
        mrope_positions.stride(0) if mrope_positions is not None else 0,
        mrope_positions.stride(1) if mrope_positions is not None else 0,
        splits,
        WRITE_MROPE=mrope_positions is not None,
        HIDDEN=HIDDEN,
        REDUCE_BLOCK=triton.next_power_of_2(splits),
        EMBED_BLOCK=4096,
        num_warps=4,
    )
    return slot.topk_p, slot.topk_index
