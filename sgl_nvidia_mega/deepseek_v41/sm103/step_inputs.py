# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A DSpark decode step's inputs in one launch, one program per request: the verify
window's positions and cache locations (read from ``req_to_token``), and the draft
block's anchor column (the bonus token).

SGLang builds these with four launches between the target and draft graphs: the
positions add, the end offsets, its cache-location kernel, and the anchor copy. The
values and dtypes are the same."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl


class StepInputs(NamedTuple):
    positions_2d: torch.Tensor  # [bs, verify_len] int64: seq_len + column
    verify_cache_loc: torch.Tensor  # [bs * verify_len] int64: req_to_token[req, pos]


@triton.jit
def _step_inputs_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    positions_ptr,
    cache_loc_ptr,
    bonus_ptr,
    anchor_ptr,
    anchor_stride,
    pool_len: tl.constexpr,
    VERIFY_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ANCHOR: tl.constexpr,
):
    r = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + r).to(tl.int64)
    req = tl.load(req_pool_indices_ptr + r).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < VERIFY_LEN
    pos = seq_len + col
    loc = tl.load(req_to_token_ptr + req * pool_len + pos, mask=mask, other=0)
    tl.store(positions_ptr + r * VERIFY_LEN + col, pos, mask=mask)
    tl.store(cache_loc_ptr + r * VERIFY_LEN + col, loc.to(tl.int64), mask=mask)
    if HAS_ANCHOR:
        tl.store(anchor_ptr + r * anchor_stride, tl.load(bonus_ptr + r))


def step_inputs(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    verify_len: int,
    bonus: Optional[torch.Tensor] = None,
    anchor_ids: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> StepInputs:
    """The verify window of ``seq_lens.shape[0]`` requests; with ``anchor_ids``, also
    its column 0 = ``bonus``. With ``out`` (positions, cache locations: contiguous int64,
    ``bs * verify_len`` each) the window is written there instead of new tensors."""
    bs, device = seq_lens.shape[0], seq_lens.device
    if out is None:
        positions_2d = torch.empty((bs, verify_len), dtype=torch.int64, device=device)
        cache_loc = torch.empty((bs * verify_len,), dtype=torch.int64, device=device)
    else:
        positions, cache_loc = out
        assert positions.dtype == cache_loc.dtype == torch.int64
        assert positions.is_contiguous() and cache_loc.is_contiguous()
        assert positions.numel() == cache_loc.numel() == bs * verify_len
        positions_2d = positions.view(bs, verify_len)
    has_anchor = anchor_ids is not None
    if bs > 0:
        _step_inputs_kernel[(bs,)](
            seq_lens,
            req_pool_indices,
            req_to_token,
            positions_2d,
            cache_loc,
            bonus if has_anchor else seq_lens,
            anchor_ids if has_anchor else positions_2d,
            anchor_ids.stride(0) if has_anchor else 0,
            pool_len=req_to_token.shape[1],
            VERIFY_LEN=verify_len,
            BLOCK=triton.next_power_of_2(verify_len),
            HAS_ANCHOR=has_anchor,
        )
    return StepInputs(positions_2d, cache_loc)
