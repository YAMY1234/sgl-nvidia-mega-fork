# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The row argmax of vocab-sharded logits as one 64-bit key per row and rank, so the
TP ranks exchange keys instead of all-gathering the logits.

key = order-preserving bits of the fp32 value << 32 | (0xFFFFFFFF - global index):
the largest key is the largest value, ties going to the smallest global index as
SGLang's row argmax does. -0.0 counts as +0.0 (the argmax compares values, and the
two are equal), and columns at or past ``vocab`` never win. Keys are stored with the
sign bit flipped, so a signed max over ranks is the unsigned max over keys."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_INDEX_MASK = 0xFFFFFFFF
_SIGN = -(2**63)


@triton.jit
def _ordered_bits(x):
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, ~bits, bits | (-2147483648)).to(tl.uint32, bitcast=True)


@triton.jit
def _argmax_keys_kernel(
    logits_ptr,
    keys_ptr,
    stride_row,
    keys_stride,
    vocab_local,
    vocab_offset,
    vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best = tl.zeros([BLOCK], dtype=tl.uint64)
    for start in range(0, vocab_local, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        index = (cols + vocab_offset).to(tl.int64)
        mask = (cols < vocab_local) & (index < vocab)
        x = tl.load(logits_ptr + row * stride_row + cols, mask=mask, other=0.0)
        # + 0.0 turns -0.0 into +0.0
        x = x.to(tl.float32) + 0.0
        key = (_ordered_bits(x).to(tl.uint64) << 32) | (
            0xFFFFFFFF - index.to(tl.uint64)
        )
        best = tl.maximum(best, tl.where(mask, key, 0))
    key = tl.max(best, axis=0) ^ 0x8000000000000000
    tl.store(keys_ptr + row * keys_stride, key.to(tl.int64, bitcast=True))


def argmax_keys(
    logits: torch.Tensor,
    *,
    vocab_offset: int,
    vocab: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[rows, vocab_local] fp32 / bf16 logits (this rank's shard, first column at global
    index ``vocab_offset``) -> [rows] int64 sign-flipped keys (into ``out``, any stride)."""
    assert logits.is_cuda and logits.dim() == 2 and logits.stride(1) == 1
    rows, vocab_local = logits.shape
    keys = (
        torch.empty(rows, dtype=torch.int64, device=logits.device)
        if out is None
        else out
    )
    # any row stride: the key exchange writes keys into a column of (key, 0) pairs
    assert keys.shape == (rows,) and keys.dtype == torch.int64
    if rows:
        _argmax_keys_kernel[(rows,)](
            logits,
            keys,
            logits.stride(0),
            keys.stride(0),
            vocab_local,
            vocab_offset,
            vocab,
            BLOCK=4096,
            num_warps=16,
        )
    return keys


def keys_to_ids(rank_keys: torch.Tensor) -> torch.Tensor:
    """[ranks, rows] sign-flipped keys -> [rows] int64 global argmax ids."""
    best = rank_keys.max(dim=0).values ^ _SIGN
    return _INDEX_MASK - (best & _INDEX_MASK)
