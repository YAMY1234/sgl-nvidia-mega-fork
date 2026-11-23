# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 draft LM head on an MXFP8 copy of this rank's shard, streamed
at 1-``max_rows`` rows (draft decode and draft extend: one row per request).

The weight format and the quantizer are the DeepSeek-V4.1 family's
(``deepseek_v41.sm103.lm_head_mxfp8``): E4M3 codes [N, K] and UE8M0 scales of 32
columns in FlashInfer's 128x4 swizzle; a block's scale is the smallest power of two that
keeps its largest magnitude within 448. The kernel (``lm_head_mxfp8.cu``) is that
family's stream kernel with x staged for ``max_rows`` rows only; for the same inputs
it returns the same bits (same fragments, MMAs and summation order).

Only the draft reads this copy: the target's verify head keeps the BF16 weight, so the
served tokens stay the target's; the draft's proposals (and so the acceptance) can
change. The copy is made once at the draft build, outside CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.common.jit import Kernel as _Kernel
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import pointer as _pointer
from sgl_nvidia_mega.deepseek_v41.sm103.lm_head_mxfp8 import (
    CHUNK,
    MAX_SCALE,
    TILE,
    WARPS,
    Mxfp8Weight,
    dequantize,
    quantize,
)

__all__ = [
    "Mxfp8Weight",
    "dequantize",
    "quantize",
    "unsupported_reason",
    "scale_reason",
    "precompile",
    "matmul",
    "matmul_bf16",
    "variant",
]

SOURCE = Path(__file__).resolve().parent / "lm_head_mxfp8.cu"
SYMBOL = "glm_draft_lm_head_stream"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
# SM100 family: GB300 (SM103, measured) and Rubin (SM107, compiled from the same source)
SUPPORTED_MAJOR = 10
MAX_ROWS = 8


class Variant(NamedTuple):
    k: int
    fp8: bool
    split: int
    depth: int
    max_rows: int


def unsupported_reason(weight: torch.Tensor, max_rows: int) -> str:
    """Why this shard cannot be served ("" when it can)."""
    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return f"the LM head weight is {weight.dtype} {tuple(weight.shape)}, not 2-D BF16"
    if weight.shape[1] % CHUNK:
        return f"K {weight.shape[1]} is not a multiple of {CHUNK}"
    if not 1 <= max_rows <= MAX_ROWS:
        return f"max rows {max_rows} outside 1-{MAX_ROWS}"
    if weight.is_cuda and torch.cuda.get_device_capability(weight.device)[0] != (
        SUPPORTED_MAJOR
    ):
        return "the stream kernel is built for the SM100 family"
    return ""


def scale_reason(q: Mxfp8Weight) -> str:
    peak = int(q.scales.max().item())
    if peak > MAX_SCALE:
        return f"a block scale of 2^{peak - 127} (the kernel takes up to 2^{MAX_SCALE - 127})"
    return ""


def split_for(n: int, sms: int) -> int:
    """The K split of the DeepSeek-V4.1 stream: two warps a tile; four when a CTA has
    fewer tiles than a quarter of its warps."""
    tiles = -(-n // TILE)
    per_cta = -(-tiles // min(sms, tiles))
    return 4 if per_cta <= WARPS // 4 else 2


def variant(
    n: int,
    k: int,
    *,
    fp8: bool,
    sms: int,
    max_rows: int,
    split: Optional[int] = None,
    depth: Optional[int] = None,
) -> Variant:
    split = split or split_for(n, sms)
    chunks = k // CHUNK // split
    if k % (CHUNK * split):
        raise ValueError(f"K {k} does not split into {split} parts of {CHUNK} columns")
    if depth is None:
        depths = (4, 5, 2, 1) if fp8 else (2, 1)
        depth = next(d for d in depths if chunks % d == 0)
    if chunks % depth:
        raise ValueError(f"depth {depth} does not divide {chunks} chunks")
    return Variant(k, fp8, split, depth, max_rows)


def shared_bytes(v: Variant) -> int:
    x = v.max_rows * (v.k + 8) * 2
    partials = (WARPS // v.split) * (v.split - 1) * 32 * 4 * 4
    return x + partials


@cache
def load_kernel(device_index: int, v: Variant) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        cubin = compile_cubin(v, f"sm_{major}{minor}a")
        return _load(cubin, SYMBOL, (WARPS * 32, 1, 1), shared_bytes(v))


def compile_cubin(v: Variant, arch: str) -> bytes:
    """NVRTC build of variant ``v`` for ``arch`` (e.g. ``sm_103a``; also the CPU compile
    check for ``sm_107a``)."""
    options = NVRTC_OPTIONS + (
        f"-DLH_K={v.k}",
        f"-DLH_FP8={int(v.fp8)}",
        f"-DLH_SPLIT={v.split}",
        f"-DLH_DEPTH={v.depth}",
        f"-DLH_WARPS={WARPS}",
        f"-DLH_MAXROWS={v.max_rows}",
    )
    name = (
        f"glm_draft_lm_head_{'mxfp8' if v.fp8 else 'bf16'}_k{v.k}_s{v.split}"
        f"_d{v.depth}_r{v.max_rows}"
    )
    return _compile_source(SOURCE.read_text(), name, arch, options)


@cache
def _sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def precompile(
    n: int, k: int, device: torch.device, *, max_rows: int, fp8: bool = True
) -> Variant:
    """Build the variant this shape runs; call before CUDA graph capture."""
    v = variant(n, k, fp8=fp8, sms=_sms(device.index), max_rows=max_rows)
    load_kernel(device.index, v)
    return v


def x_reason(x: torch.Tensor, k: int, max_rows: int) -> str:
    """Why ``x`` cannot be streamed ("" when it can): BF16 [1-max_rows, K], unit column
    stride, 16-byte aligned rows."""
    if not (
        x.dim() == 2
        and 1 <= x.shape[0] <= max_rows
        and x.shape[1] == k
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        return (
            f"x must be BF16 [1-{max_rows}, {k}] with unit column stride and 16-byte "
            f"aligned rows, got {tuple(x.shape)} {x.dtype} strides {tuple(x.stride())}"
        )
    return ""


def _run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: Optional[torch.Tensor],
    n: int,
    k: int,
    v: Variant,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    why = x_reason(x, k, v.max_rows)
    if why:
        raise ValueError(why)
    device = x.device
    rows = x.shape[0]
    if out is None:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
    assert out.shape == (rows, n) and out.stride(1) == 1 and out.dtype == torch.bfloat16
    sms = _sms(device.index)
    tiles = -(-n // TILE)
    _launch(
        load_kernel(device.index, v),
        (min(sms, tiles), 1, 1),
        [
            _pointer(weight),
            _pointer(scales) if scales is not None else ctypes.c_void_p(0),
            _pointer(x),
            _pointer(out),
            ctypes.c_int32(n),
            ctypes.c_int32(rows),
            ctypes.c_int32(x.stride(0)),
            ctypes.c_int32(out.stride(0)),
        ],
        torch.cuda.current_stream(device).cuda_stream,
    )
    return out


def matmul(
    x: torch.Tensor,
    q: Mxfp8Weight,
    *,
    max_rows: int,
    out: Optional[torch.Tensor] = None,
    split: Optional[int] = None,
    depth: Optional[int] = None,
) -> torch.Tensor:
    """``x @ dequantize(q).T`` in BF16 for BF16 ``x`` [1-max_rows, K]."""
    n, k = q.shape
    v = variant(
        n, k, fp8=True, sms=_sms(x.device.index), max_rows=max_rows, split=split,
        depth=depth,
    )
    return _run(x, q.codes, q.scales, n, k, v, out)


def matmul_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_rows: int,
    out: Optional[torch.Tensor] = None,
    split: Optional[int] = None,
    depth: Optional[int] = None,
) -> torch.Tensor:
    """The same kernel on a BF16 weight [N, K]: the order :func:`matmul` sums in."""
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    n, k = weight.shape
    v = variant(
        n, k, fp8=False, sms=_sms(x.device.index), max_rows=max_rows, split=split,
        depth=depth,
    )
    return _run(x, weight, None, n, k, v, out)
