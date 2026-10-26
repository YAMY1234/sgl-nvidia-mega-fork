# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Q35FLY one-row head plus hidden-half FC, using the existing BF16/MXFP8 arithmetic."""

from __future__ import annotations

import ctypes
from functools import cache

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8 import HeadMx8Params, smem_bytes
from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import (
    HIDDEN,
    WORLDS,
    MkdHeadParams,
    MkdHiddenPartialParams,
    _check,
    _compile,
    _launch,
)


class HeadPartialBf16Params(ctypes.Structure):
    _fields_ = [("head", MkdHeadParams), ("partial", MkdHiddenPartialParams)]


class HeadPartialMx8Params(ctypes.Structure):
    _fields_ = [("head", HeadMx8Params), ("partial", MkdHiddenPartialParams)]


@cache
def _cubin(device_index: int, world: int):
    return _compile(
        device_index,
        "mkd_head_partial",
        (f"MKD_WORLD={world}",),
        includes=("mkd_fc", "mkd_head", "head_mx8"),
    )


@cache
def prepare(device_index: int, world: int, smem_pad: int, mxfp8: bool):
    """Compile/load outside CUDA graph capture, when the boundary is constructed."""
    if world not in WORLDS:
        raise ValueError(f"head+partial requires TP {WORLDS}, got {world}")
    shared = max(smem_pad, smem_bytes(1)) if mxfp8 else smem_pad
    name = "mkd_head_partial_mx8" if mxfp8 else "mkd_head_partial_bf16"
    return jit.load(_cubin(device_index, world), name, (512, 1, 1), shared)


def head_with_partial(
    x, weight, logits, part, shard, norm_weight, eps, col, out, epoch_done, sync,
    world, *, reverse=False, smem_pad=0, head_mx8=None,
):
    """Same logits and FC partial, all head CTAs, no side stream or reserved SMs."""
    n = weight.shape[0]
    width = 2 * HIDDEN // world
    _check(x, "x", (1, HIDDEN), torch.bfloat16, aligned=True)
    _check(weight, "weight", (n, HIDDEN), torch.bfloat16, aligned=True)
    _check(logits, "logits", (1, n), torch.bfloat16)
    _check(part, "part", (part.numel(),), torch.int32)
    _check(shard, "shard", (HIDDEN, width), torch.bfloat16, aligned=True)
    _check(norm_weight, "norm_weight", (HIDDEN,), torch.bfloat16, aligned=True)
    _check(out, "out", (1, HIDDEN), torch.float32)
    _check(epoch_done, "epoch_done", (2,), torch.int32)
    _check(sync, "sync", (2,), torch.int32)
    if not (0 < n <= 65536 and col >= 0 and col % width == 0 and col + width <= HIDDEN):
        raise ValueError("head+partial requires a supported vocabulary and hidden slice")
    partial = MkdHiddenPartialParams(
        h=x.data_ptr(), w=shard.data_ptr(), norm_w=norm_weight.data_ptr(),
        out=out.data_ptr(), epoch=epoch_done.data_ptr(), done=sync.data_ptr(),
        flag=sync.data_ptr() + 4, eps=eps, col=col,
    )
    if head_mx8 is None:
        p = HeadPartialBf16Params(
            MkdHeadParams(x=x.data_ptr(), w=weight.data_ptr(), logits=logits.data_ptr(),
                          part=part.data_ptr(), n=n, reverse=int(reverse)),
            partial,
        )
    else:
        p = HeadPartialMx8Params(
            HeadMx8Params(x=x.data_ptr(), w=head_mx8.values.data_ptr(),
                          s=head_mx8.scales.data_ptr(), logits=logits.data_ptr(),
                          part=part.data_ptr(), n=n, m=1, reverse=int(reverse)),
            partial,
        )
    kernel = prepare(x.device.index, world, smem_pad, head_mx8 is not None)
    _launch(kernel, part.numel(), p, x.device, pdl=True)
