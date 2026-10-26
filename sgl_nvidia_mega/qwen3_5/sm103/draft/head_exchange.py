# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HP2: last-completing head cluster owns the exact memo exchange and FC sum."""

from __future__ import annotations

import ctypes
import logging
from functools import cache

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8 import HeadMx8Params, smem_bytes
from sgl_nvidia_mega.qwen3_5.sm103.draft.head_partial import (
    HeadPartialBf16Params,
    HeadPartialMx8Params,
)
from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import (
    MkdHeadParams,
    MkdHiddenPartialParams,
    MkdXchgParams,
    _compile,
)

logger = logging.getLogger(__name__)
CLUSTER = 8


class HeadExchangeBf16Params(ctypes.Structure):
    _fields_ = [("compute", HeadPartialBf16Params), ("exchange", MkdXchgParams),
                ("completed", ctypes.c_void_p)]


class HeadExchangeMx8Params(ctypes.Structure):
    _fields_ = [("compute", HeadPartialMx8Params), ("exchange", MkdXchgParams),
                ("completed", ctypes.c_void_p)]


@cache
def _cubin(device_index, world):
    return _compile(device_index, "mkd_head_exchange", (f"MKD_WORLD={world}",),
                    includes=("mkd_fc", "mkd_head", "head_mx8", "mkd_head_partial", "mkd_xchg"))


@cache
def prepare(device_index, world, smem_pad, mxfp8, ctas):
    """Prepare both hidden and token-only paths, and reject unsupported clusters."""
    if world not in (4, 8) or not CLUSTER <= ctas <= 512 or ctas % CLUSTER:
        raise ValueError("HP2 requires TP4/8 and 8..512 CTAs in clusters of eight")
    cu = jit.driver()
    shared = max(smem_pad, smem_bytes(1)) if mxfp8 else smem_pad
    kernels = {}
    for partial in (False, True):
        name = "mkd_head_exchange_" + ("mx8" if mxfp8 else "bf16") + ("_partial" if partial else "")
        kernel = jit.load(_cubin(device_index, world), name, (512, 1, 1), shared)
        config = cu.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = ctas, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = kernel.block
        config.sharedMemBytes = shared
        config.hStream = cu.CUstream(0)
        dims = cu.CUlaunchAttribute()
        dims.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        dims.value.clusterDim.x, dims.value.clusterDim.y, dims.value.clusterDim.z = CLUSTER, 1, 1
        config.attrs, config.numAttrs = [dims], 1
        (active,) = jit.check(cu.cuOccupancyMaxActiveClusters(kernel.function, config),
                             "HP2 cuOccupancyMaxActiveClusters")
        if active < 1:
            raise RuntimeError(f"HP2 {name} cannot schedule an eight-CTA cluster")
        logger.info("Q35FLY HP2 %s: grid=%d, shared=%d, active clusters=%d",
                    name, ctas, shared, active)
        kernels[partial] = kernel
    return kernels


def launch(x, weight, shard, norm_weight, eps, col, own_partial, epoch_done,
           sync, completed, world, exchange, *, reverse=False, smem_pad=0,
           head_mx8=None, with_partial=False):
    """Exchange has already been checked/packed by xchg.mkd_xchg, with memo rows."""
    ctas = exchange.head_parts
    partial = MkdHiddenPartialParams(
        h=x.data_ptr(), w=shard.data_ptr(), norm_w=norm_weight.data_ptr(),
        out=own_partial.data_ptr(), epoch=epoch_done.data_ptr(), done=sync.data_ptr(),
        flag=sync.data_ptr() + 4, eps=eps, col=col,
    )
    if head_mx8 is None:
        compute = HeadPartialBf16Params(
            MkdHeadParams(x=x.data_ptr(), w=weight.data_ptr(), logits=exchange.logits,
                          part=exchange.head_part, n=exchange.vocab, reverse=int(reverse)), partial)
        p = HeadExchangeBf16Params(compute, exchange, completed.data_ptr())
    else:
        compute = HeadPartialMx8Params(
            HeadMx8Params(x=x.data_ptr(), w=head_mx8.values.data_ptr(), s=head_mx8.scales.data_ptr(),
                          logits=exchange.logits, part=exchange.head_part, n=exchange.vocab,
                          m=1, reverse=int(reverse)), partial)
        p = HeadExchangeMx8Params(compute, exchange, completed.data_ptr())
    kernel = prepare(x.device.index, world, smem_pad, head_mx8 is not None, ctas)[with_partial]
    jit.launch(kernel, (ctas, 1, 1), [p], torch.cuda.current_stream(x.device).cuda_stream,
               cluster=(CLUSTER, 1, 1))
