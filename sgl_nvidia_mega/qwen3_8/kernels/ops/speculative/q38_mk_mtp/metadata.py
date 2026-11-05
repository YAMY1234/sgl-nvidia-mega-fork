# SPDX-License-Identifier: Apache-2.0
"""Integer-only M4 verify layout and N-gram history snapshot.

The history is read into graph-owned output, never aliased to the mutable pool.
This kernel does not write pool state or replace hash/gather/commit consumers.
"""

import logging
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.qwen3_8.kernels.ops import pdl_small

_PDL_SMALL = pdl_small.ENABLED

logger = logging.getLogger(__name__)
_logged = False


class VerifyLayout(NamedTuple):
    lengths: torch.Tensor
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: torch.Tensor


@triton.jit
def _q38_mk_mtp_ple_prepare(
    Tokens,
    Lengths,
    Requests,
    Locations,
    SlotMap,
    History,
    OutLengths,
    OutRequests,
    OutOffsets,
    OutValid,
    OutSlots,
    OutContext,
    EOS: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL_SMALL: tl.constexpr = False,
):

    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    col = tl.arange(0, BLOCK)
    length = tl.load(Lengths) if HAS_LENGTHS else 4
    req = tl.load(Requests).to(tl.int64)
    slot = tl.load(SlotMap + req).to(tl.int64)
    live_loc = tl.load(Locations + col, col < 4, other=0) != 0
    live_row = (length != 0) & (tl.sum(live_loc.to(tl.int32), 0) != 0)
    slot = tl.where(live_row, slot, 0)
    valid = col < length
    token = tl.load(Tokens + col, col < 4, other=EOS)
    tl.store(OutLengths, length)
    tl.store(OutRequests + col, 0, col < 4)
    tl.store(OutOffsets + col, col, col < 4)
    tl.store(OutValid + col, valid, col < 4)
    tl.store(OutSlots, slot)
    history = tl.load(History + slot * 2 + col, col < 2, other=EOS)
    tl.store(OutContext + col, history, col < 2)
    tl.store(OutContext + 2 + col, tl.where(valid, token, EOS), col < 4)
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


def prepare_verify_layout(tokens, lengths, requests, locations, slot_map, history, eos):
    """Low-level entry; caller guarantees one M4 row and contiguous pool inputs."""
    device = tokens.device
    result = VerifyLayout(
        torch.empty((1,), dtype=torch.int64, device=device),
        torch.empty((4,), dtype=torch.int64, device=device),
        torch.empty((4,), dtype=torch.int64, device=device),
        torch.empty((4,), dtype=torch.bool, device=device),
        torch.empty((1,), dtype=torch.int64, device=device),
        torch.empty((1, 6), dtype=torch.int64, device=device),
    )
    _q38_mk_mtp_ple_prepare[(1,)](
        tokens,
        lengths if lengths is not None else requests,
        requests,
        locations,
        slot_map,
        history,
        *result,
        EOS=eos,
        HAS_LENGTHS=lengths is not None,
        BLOCK=32,
        num_warps=4,
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    return result
