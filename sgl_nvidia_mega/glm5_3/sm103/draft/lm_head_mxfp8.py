# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 draft LM head on an MXFP8 copy of this rank's shard, streamed
at 1-``max_rows`` rows (draft decode and draft extend: one row per request).

The weight format and the quantizer are the DeepSeek-V4.1 family's
(``deepseek_v41.sm103.lm_head_mxfp8``): E4M3 codes [N, K] and UE8M0 scales of 32
columns in FlashInfer's 128x4 swizzle; a block's scale is the smallest power of two that
keeps its largest magnitude within 448. The kernel (``lm_head_mxfp8.cu``) is that
family's stream kernel with x staged for ``max_rows`` rows only; for the same inputs
it returns the same bits (same fragments, MMAs and summation order). By default it builds
the BF16 bits of a code with byte permutes (``dq=2``: one integer op fewer a code word
than the shifts of ``dq=1``, the same bits), about 2 % faster on GB300. ``kernel="ring"``
selects a build that feeds the same per-warp chains through a shared-memory ring (tensor
copies by a producer warp, no registers held by the bytes in flight); also the same bits,
but not faster on GB300 (the per-warp dequantization and MMA chain, not the loads, set
the pace at the TP8 shard).

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
RING_SYMBOL = "glm_draft_lm_head_ring"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
# SM100 family: GB300 (SM103, measured) and Rubin (SM107, compiled from the same source)
SUPPORTED_MAJOR = 10
MAX_ROWS = 8
# the ring build: WARPS consumer warps (a round is WARPS / split tiles) and one producer
RING_BOX = TILE * CHUNK  # a chain's codes a stage
RING_SCALES = 256  # the half scale block of a tile a stage
RING_MAX_STAGES = 8
OPTIN_DEFAULT = 232448  # sharedMemPerBlockOptin of SM100-family parts (227 KB)


class Variant(NamedTuple):
    k: int
    fp8: bool
    split: int
    depth: int
    max_rows: int
    stages: int = 0  # 0: the stream build (registers, `depth` chunks ahead); else the ring's stage count
    trigger: int = 0  # ring: 1 signals PDL dependents early, 0 at the end
    group: int = 1  # ring: consecutive tiles a tensor copy (a CTA's work unit)
    unroll: int = 1  # ring: stages a consumer iteration
    diag: int = 0  # ring, timing only: 1 no math, 2 no copies (wrong outputs)
    dq: int = 2  # the dequantization's instructions (same bits): 1 shifts (DeepSeek-V4.1's), 2 byte permutes
    release: int = 0  # ring: 1 a consumer frees its stage once loaded (before the math), 0 after the math


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


def group_for(n: int, sms: int, split: int) -> int:
    """Tiles a ring work unit (one tensor copy a stage and part): the fewest tiles on the
    busiest CTA, then the largest group (fewest copies)."""
    tiles = -(-n // TILE)
    best = None
    for group in (8, 4, 2, 1):
        if (WARPS // split) % group:
            continue
        groups = -(-tiles // group)
        busiest = group * -(-groups // min(sms, groups))
        if best is None or busiest < best[0]:
            best = (busiest, group)
    return best[1]


def _ring_fixed_bytes(k: int, max_rows: int, split: int) -> int:
    """Shared memory of the ring build before its stages: mbarriers, x, the split partials."""
    partials = (WARPS // split) * (split - 1) * 32 * 4 * 4
    return -(-(128 + max_rows * (k + 8) * 2 + partials) // 1024) * 1024


def ring_stage_bytes(split: int, group: int) -> int:
    boxes = WARPS // split // group * split
    return WARPS * RING_BOX + boxes * (RING_SCALES if group == 1 else 2 * RING_SCALES)


def ring_stages(k: int, max_rows: int, split: int, group: int, optin: int) -> int:
    """The stages that fit next to x (0: the ring does not fit)."""
    stages = (optin - _ring_fixed_bytes(k, max_rows, split)) // ring_stage_bytes(split, group)
    stages = min(stages, RING_MAX_STAGES)
    return stages if stages >= 2 else 0


def variant(
    n: int,
    k: int,
    *,
    fp8: bool,
    sms: int,
    max_rows: int,
    split: Optional[int] = None,
    depth: Optional[int] = None,
    kernel: Optional[str] = None,
    stages: Optional[int] = None,
    trigger: int = 0,
    group: Optional[int] = None,
    unroll: int = 1,
    dq: int = 2,
    release: int = 0,
    optin: int = OPTIN_DEFAULT,
) -> Variant:
    """The build ``n`` x ``k`` runs: the stream (the default) or, with ``kernel="ring"``,
    the ring (MXFP8, N a multiple of 16, the stages fit; measured no faster on GB300).
    The K split is the stream's choice for the shape in both, and ``dq`` only changes the
    instructions, so every build gives the same bits."""
    if kernel not in (None, "ring", "stream"):
        raise ValueError(f"kernel {kernel!r}: ring or stream")
    split = split or split_for(n, sms)
    chunks = k // CHUNK // split
    if k % (CHUNK * split):
        raise ValueError(f"K {k} does not split into {split} parts of {CHUNK} columns")
    if kernel == "ring" and fp8 and n % TILE == 0 and WARPS % split == 0:
        group = group or group_for(n, sms, split)
        fit = ring_stages(k, max_rows, split, group, optin)
        stages = stages or fit
        if (
            group in (1, 2, 4, 8)
            and (WARPS // split) % group == 0
            and stages > unroll
            and chunks % unroll == 0
            and _ring_fixed_bytes(k, max_rows, split)
            + stages * ring_stage_bytes(split, group)
            <= optin
        ):
            return Variant(
                k, fp8, split, 1, max_rows, stages, trigger, group, unroll,
                dq=dq, release=release,
            )
    if kernel == "ring":
        raise ValueError(f"no ring build for n={n} k={k} fp8={fp8} max_rows={max_rows}")
    if depth is None:
        depths = (4, 5, 2, 1) if fp8 else (2, 1)
        depth = next(d for d in depths if chunks % d == 0)
    if chunks % depth:
        raise ValueError(f"depth {depth} does not divide {chunks} chunks")
    return Variant(k, fp8, split, depth, max_rows, dq=dq)


def shared_bytes(v: Variant) -> int:
    if v.stages:
        return _ring_fixed_bytes(v.k, v.max_rows, v.split) + v.stages * ring_stage_bytes(
            v.split, v.group
        )
    x = v.max_rows * (v.k + 8) * 2
    partials = (WARPS // v.split) * (v.split - 1) * 32 * 4 * 4
    return x + partials


def threads(v: Variant) -> int:
    return (WARPS + 1) * 32 if v.stages else WARPS * 32


@cache
def load_kernel(device_index: int, v: Variant) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        cubin = compile_cubin(v, f"sm_{major}{minor}a")
        symbol = RING_SYMBOL if v.stages else SYMBOL
        return _load(cubin, symbol, (threads(v), 1, 1), shared_bytes(v))


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
        f"-DLH_RING={int(bool(v.stages))}",
        f"-DLH_STAGES={v.stages or 2}",
        f"-DLH_TRIGGER={v.trigger}",
        f"-DLH_GROUP={v.group}",
        f"-DLH_UNROLL={v.unroll}",
        f"-DLH_DIAG={v.diag}",
        f"-DLH_DQ={v.dq}",
        f"-DLH_RELEASE={v.release}",
    )
    name = (
        f"glm_draft_lm_head_{'mxfp8' if v.fp8 else 'bf16'}_k{v.k}_s{v.split}"
        f"_d{v.depth}_r{v.max_rows}"
    )
    if v.stages:
        name += f"_ring{v.stages}_t{v.trigger}_g{v.group}_u{v.unroll}_x{v.diag}"
    name += f"_q{v.dq}_e{v.release}"
    return _compile_source(SOURCE.read_text(), name, arch, options)


@cache
def _sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cache
def _optin(device_index: int) -> int:
    props = torch.cuda.get_device_properties(device_index)
    return int(getattr(props, "shared_memory_per_block_optin", OPTIN_DEFAULT))


_MAPS: dict = {}


def tensor_map(codes: torch.Tensor, group: int):
    """The codes' TMA descriptor for the ring build: [N][K / 128][128] with a box of
    ``group`` tiles x 128 columns (host only, so also inside CUDA-graph capture); one per
    (address, shape, group), cached."""
    n, k = codes.shape
    key = (codes.data_ptr(), n, k, group)
    found = _MAPS.get(key)
    if found is not None:
        return found
    from cuda.bindings import driver as cu

    assert codes.is_contiguous() and codes.element_size() == 1 and n % TILE == 0
    dims = (CHUNK, k // CHUNK, n)
    status, encoded = cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        len(dims),
        codes.data_ptr(),
        [cu.cuuint64_t(d) for d in dims],
        [cu.cuuint64_t(s) for s in (CHUNK, k)],
        [cu.cuuint32_t(b) for b in (CHUNK, 1, TILE * group)],
        [cu.cuuint32_t(1)] * len(dims),
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(status) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {status!r}")
    if len(_MAPS) > 64:
        _MAPS.clear()
    _MAPS[key] = encoded
    return encoded


def _variant_on(device_index: int, n: int, k: int, **kw) -> Variant:
    return variant(n, k, sms=_sms(device_index), optin=_optin(device_index), **kw)


def precompile(
    n: int, k: int, device: torch.device, *, max_rows: int, fp8: bool = True
) -> Variant:
    """Build the variant this shape runs; call before CUDA graph capture."""
    v = _variant_on(device.index, n, k, fp8=fp8, max_rows=max_rows)
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
    units = -(-n // TILE)  # tiles; groups of tiles for the ring
    if v.stages:
        units = -(-units // v.group)
    _launch(
        load_kernel(device.index, v),
        (min(sms, units), 1, 1),
        [
            tensor_map(weight, v.group) if v.stages else _pointer(weight),
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
    kernel: Optional[str] = None,
    stages: Optional[int] = None,
    trigger: int = 0,
    group: Optional[int] = None,
    unroll: int = 1,
    dq: int = 2,
    release: int = 0,
) -> torch.Tensor:
    """``x @ dequantize(q).T`` in BF16 for BF16 ``x`` [1-max_rows, K]: the stream build
    (``kernel="ring"``: the ring build; the same bits)."""
    n, k = q.shape
    v = _variant_on(
        x.device.index, n, k, fp8=True, max_rows=max_rows, split=split, depth=depth,
        kernel=kernel, stages=stages, trigger=trigger, group=group,
        unroll=unroll, dq=dq, release=release,
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
    v = _variant_on(
        x.device.index, n, k, fp8=False, max_rows=max_rows, split=split, depth=depth
    )
    return _run(x, weight, None, n, k, v, out)
