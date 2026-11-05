# SPDX-License-Identifier: Apache-2.0
"""FN draft top1 finalization -> the next captured BF16 embedding slot.

The vocabulary projection and native split argmax are unchanged. Each capture
owns its scratch; nothing is memoized across requests or graph variants.
"""

import triton
import triton.language as tl


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
