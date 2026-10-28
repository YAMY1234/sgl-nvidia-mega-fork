# SPDX-License-Identifier: Apache-2.0
"""FN TP4/TP1 BF16 projection and T4 verify convolution for B1/B2/B3.

Only target verify B1/B2/B3, T4, no tree and no accept-offset. Caller retains the
native path otherwise. This module owns no weights or state copies.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from .tgv_front import MkGdnFront

_CACHE = {}


def _view(t, align=2):
    return from_dlpack(t.detach(), assumed_align=align)


def run(x, packed_weight, conv_weight, conv_bias, conv_state, state_indices,
        window, intermediate_indices, *, shape, pad_slot=-1):
    """Return packed BF16[M,Npacked] and convolved BF16[M,Nqkv].

    State/window are mutated exactly at their supplied strides. Padded sequence
    (-1 by default) leaves conv output/state/windows undefined/unmodified, as
    the native conv; z/b/a projection still executes. All packed output
    columns currently contain the raw BF16 projection, including q/k/v.
    PDL consumers must issue their existing grid-dependency wait.
    """
    rows, width = x.shape
    channels, packed_width = shape.channels, shape.packed
    if (width != shape.hidden or rows % shape.tokens_per_request
            or shape.tokens_per_request != 4):
        raise ValueError('mature FN front is qualified for four rows/request')
    if packed_weight.shape != (packed_width, shape.hidden):
        raise ValueError('incorrect packed BF16 projection shape')
    if x.dtype != torch.bfloat16 or packed_weight.dtype != torch.bfloat16:
        raise ValueError('FN GDN projection requires native BF16 weights/activations')
    if conv_weight.shape != (channels, 4) or conv_state.shape[1] != channels:
        raise ValueError('native width-four convolution required')
    if window.shape[1:4] != (shape.tokens_per_request, channels, 3):
        raise ValueError('native verify window shape required')
    packed = torch.empty((rows, packed_width), device=x.device, dtype=x.dtype)
    conv_out = torch.empty((rows, channels), device=x.device, dtype=x.dtype)
    # GEMM kernel-M = output channel; kernel-N = token. No repack/copy.
    a = packed_weight.unsqueeze(-1)
    b = x.unsqueeze(-1)
    c = packed.t().unsqueeze(-1)
    bias = conv_weight[:, 0] if conv_bias is None else conv_bias
    tensors = (a, b, c, c, conv_weight, bias, conv_state,
               state_indices, window, intermediate_indices, conv_out)
    key = (x.device.index, conv_bias is not None, int(pad_slot),
           tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors))
    args = tuple(_view(t, 16 if i in (0, 1, 2, 3) else 2)
                 for i, t in enumerate(tensors))
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _CACHE.get(key)
    if compiled is None:
        op = MkGdnFront(
            cta_m=64, cta_n=8 if rows == 4 else 16, cta_k=128, num_ab_stage=6,
            use_2cta=False, use_pdl=True, has_bias=False,
            out_dtype=cutlass.BFloat16)
        op.qkv_channels = channels
        op.verify_rows = rows
        op.tokens_per_request = shape.tokens_per_request
        op.conv_has_bias = conv_bias is not None
        op.pad_slot = int(pad_slot)
        compiled = cute_ext.compile(op, *args, stream, options="--gpu-arch sm_103a")
        _CACHE[key] = compiled
    compiled(*args, stream)
    return packed, conv_out
