# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/modules/q35_draft_head.py
# (release 4d09932ca7: quantize_mxfp8_rows and dequantize_mxfp8_rows verbatim,
# default_slice_blocks, and the GEMV's compile and launch from _compiled / mxfp8_head_gemv).
"""The MTP draft's LM head from a draft-only MXFP8 copy of this rank's vocabulary
shard, in TRT-LLM's ``TRTLLM_Q35_DRAFT_FP8_HEAD`` format: E4M3 values with one UE8M0 scale
per 32 K-elements (``quantize_mxfp8_rows``); the activation stays bf16, every product is
exact and the sum is fp32, so only the fp32 summation order differs from a dequantized
reference. The target's lm_head is never touched.

- ``DraftHeadMx8.logits``: bf16 shard logits [m, N] of m <= 16 rows (TRT-LLM's GEMV row
  limit), from TRT-LLM's CuTe DSL GEMV (``head_mx8_gemv``, ``gemv="cute"``) or its NVRTC
  translation (``src/head_mx8.cu``, ``gemv="nvrtc"``), which keeps its tiles, fragments and
  summation order;
- ``DraftHeadMx8.mkd_head``: the one-row head of the MKD draft step boundary
  (``mkd_head_mx8``): bf16 logits and per CTA the packed maximum of its rows, as
  ``xchg.mkd_draft_head`` writes them for the exchange.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
from functools import cache
from pathlib import Path
from typing import Tuple

import torch

from sgl_nvidia_mega.common import jit

logger = logging.getLogger(__name__)

SRC = Path(__file__).parent / "src"
HIDDEN = 4096
# Rows of x per GEMV: its activation lives in registers (q35_draft_head.MAX_ROWS)
MAX_ROWS = 16
THREADS = 16 * 32
# Rows of the head quantized per step of the one-time conversion (bounds its FP32 scratch).
_QUANT_ROWS = 4096


def quantize_mxfp8_rows(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``weight`` [N, K] as E4M3 values with one UE8M0 scale per 32 K-elements: each block is
    divided by the power of two that brings its absolute max to at most 448."""
    rows, k = weight.shape
    assert k % 32 == 0, k
    values = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((rows, k // 32), dtype=torch.uint8, device=weight.device)
    for start in range(0, rows, _QUANT_ROWS):
        end = min(start + _QUANT_ROWS, rows)
        blocks = weight[start:end].float().view(end - start, k // 32, 32)
        amax = blocks.abs().amax(-1).clamp_min(2.0**-126)
        exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
        values[start:end] = (
            (blocks / torch.exp2(exponent)[..., None])
            .view(end - start, k)
            .to(torch.float8_e4m3fn)
        )
        scales[start:end] = (exponent + 127).to(torch.uint8)
    return values, scales


def dequantize_mxfp8_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 ``[N, K]`` from ``quantize_mxfp8_rows`` output (reference / tests)."""
    rows, k = values.shape
    scale = torch.exp2(scales.float() - 127.0)
    return (values.float().view(rows, k // 32, 32) * scale[..., None]).view(rows, k)


def default_slice_blocks(k: int) -> int:
    # About 16-20 warps per CTA (GB300; K = 4096: 16 warps of 8 blocks).
    for slice_blocks in (8, 16, 4, 2):
        if k % (32 * slice_blocks) == 0 and 4 <= k // (32 * slice_blocks) <= 32:
            return slice_blocks
    raise ValueError(f"unsupported K={k} for the Qwen3.5 draft head GEMV")


# ---------------------------------------------------------------- CuTe DSL GEMV (TRT-LLM's)


class _GraphSafeDlpack:
    """DLPack export that does not synchronize streams (CUDA-graph capture safe)."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor.detach()

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _cute(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(_GraphSafeDlpack(t), assumed_align=align)


def _cute_stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def cute_compile(
    x: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    num_ctas: int,
    slice_blocks: int,
):
    """TRT-LLM's GEMV compiled for these shapes (bf16 logits; PDL with the early trigger)."""
    import cutlass
    import cutlass.cute as cute

    from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8_gemv import (
        Q35DraftHeadGemvKernel,
    )

    m, k = x.shape
    kernel = Q35DraftHeadGemvKernel(
        num_rows=m,
        k_extent=k,
        num_ctas=num_ctas,
        slice_blocks=slice_blocks,
        use_pdl=True,
        early_trigger=True,
        argmax=False,
    )
    return cute.compile(
        kernel,
        _cute(x),
        _cute(values.view(torch.uint8)),
        _cute(scales),
        _cute(out, 8),
        cutlass.Int32(0),
        cutlass.Float32(1.0),
        _cute_stream(),
    )


# ---------------------------------------------------------------- NVRTC kernels


class HeadMx8Params(ctypes.Structure):
    """sglang::HeadMx8Params."""

    _fields_ = [
        (name, ctypes.c_void_p) for name in ("x", "w", "s", "logits", "part")
    ] + [
        ("n", ctypes.c_int32),
        ("m", ctypes.c_int32),
        ("reverse", ctypes.c_int32),
        ("keys", ctypes.c_void_p),
        ("vocab_start", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("nan_key", ctypes.c_uint32),
    ]


def smem_bytes(token_tiles: int) -> int:
    """Dynamic shared memory of a head_mx8.cu kernel (mirrors head_mx8::Layout::kBytes)."""
    warps, row_stride, scale_row = 16, HIDDEN + 32, HIDDEN // 32
    return (
        2 * 16 * row_stride
        + 2 * 16 * scale_row
        + 2 * warps * 4 * token_tiles * 32 * 4
        + 16
        + 8 * token_tiles * 4
        + warps * 4
    )


@cache
def _cubin(device_index: int) -> bytes:
    major, minor = torch.cuda.get_device_capability(device_index)
    return jit.compile_source(
        (SRC / "head_mx8.cu").read_text(),
        "q35_head_mx8",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers={"mkd_common.cuh": (SRC / "mkd_common.cuh").read_text()},
    )


@cache
def _kernel(device_index: int, name: str, shared: int) -> jit.Kernel:
    return jit.load(_cubin(device_index), name, (THREADS, 1, 1), shared)


def source_digest() -> str:
    return hashlib.sha256((SRC / "head_mx8.cu").read_bytes()).hexdigest()[:16]


def _launch(
    kernel: jit.Kernel,
    ctas: int,
    x: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    logits: torch.Tensor,
    part,
    reverse: bool,
    keys=None,
    vocab_start: int = 0,
    vocab_size: int = 0,
    nan_key: int = 0,
) -> None:
    p = HeadMx8Params(
        x=x.data_ptr(),
        w=values.data_ptr(),
        s=scales.data_ptr(),
        logits=0 if logits is None else logits.data_ptr(),
        part=0 if part is None else part.data_ptr(),
        n=int(values.shape[0]),
        m=int(x.shape[0]),
        reverse=int(bool(reverse)),
        keys=0 if keys is None else keys.data_ptr(),
        vocab_start=vocab_start,
        vocab_size=vocab_size,
        nan_key=nan_key,
    )
    stream = torch.cuda.current_stream(x.device).cuda_stream
    jit.launch(kernel, (ctas, 1, 1), [p], stream)


def nvrtc_gemv(
    x: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    ctas: int,
) -> torch.Tensor:
    """``out`` [m, N] bf16 = x [m, 4096] bf16 @ dequant(values, scales).T, m <= 16, on
    ``ctas`` <= ceil(N / 8) CTAs (head_mx8_m<m>)."""
    m = int(x.shape[0])
    kernel = _kernel(x.device.index, f"head_mx8_m{m}", smem_bytes(1 if m <= 8 else 2))
    _launch(kernel, ctas, x, values, scales, out, None, False)
    return out


def nvrtc_keys(
    x: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    keys: torch.Tensor,
    ctas: int,
    *,
    vocab_start: int,
    vocab_size: int,
    nan_key: int,
) -> None:
    """Fold each row's top-1 key directly from the MXFP8 head into ``keys``."""
    m = int(x.shape[0])
    kernel = _kernel(
        x.device.index, f"head_mx8_keys_m{m}", smem_bytes(1 if m <= 8 else 2)
    )
    _launch(
        kernel,
        ctas,
        x,
        values,
        scales,
        None,
        None,
        False,
        keys,
        vocab_start,
        vocab_size,
        nan_key,
    )


def mkd_head_mx8(
    x: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    logits: torch.Tensor,
    part: torch.Tensor,
    reverse: bool = False,
    smem_pad: int = 0,
) -> None:
    """``xchg.mkd_draft_head`` on the MXFP8 copy: ``logits`` [1, N] bf16 and per CTA
    (``part.numel()`` <= ceil(N / 8) CTAs) the packed maximum of its rows; ``reverse`` walks
    the row groups from the end (same values). Dynamic shared memory is the larger of the
    kernel's and ``smem_pad``."""
    kernel = _kernel(x.device.index, "mkd_head_mx8", max(smem_bytes(1), smem_pad))
    _launch(kernel, part.numel(), x, values, scales, logits, part, reverse)


# ---------------------------------------------------------------- the draft head


class DraftHeadMx8:
    """The MXFP8 copy of one rank's lm_head shard [N, 4096] and its compiled kernels; build
    before any capture (every kernel compiles and runs once here)."""

    def __init__(self, weight: torch.Tensor, *, max_rows: int = MAX_ROWS, gemv="cute"):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the MXFP8 draft head must be built before graph capture"
            )
        if (
            weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.shape[1] != HIDDEN
        ):
            raise ValueError(
                f"MXFP8 draft head: lm_head shard {weight.dtype} {tuple(weight.shape)} "
                f"(want bf16 [N, {HIDDEN}])"
            )
        if gemv not in ("cute", "nvrtc"):
            raise ValueError(f"MXFP8 draft head: GEMV {gemv!r} (cute or nvrtc)")
        dev = weight.device
        self.n, self.k = int(weight.shape[0]), int(weight.shape[1])
        self.values, self.scales = quantize_mxfp8_rows(weight.detach().contiguous())
        self.max_rows = max(1, min(int(max_rows), MAX_ROWS))
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.ctas = min(sms, -(-self.n // 8))
        self.gemv = gemv
        self._noted: set = set()
        # every row count's GEMV compiled and run once on zeros
        x = torch.zeros(self.max_rows, self.k, dtype=torch.bfloat16, device=dev)
        out = torch.empty(self.max_rows, self.n, dtype=torch.bfloat16, device=dev)
        self._cute_fns: dict = {}
        if gemv == "cute":
            self._cute_w = _cute(self.values.view(torch.uint8))
            self._cute_s = _cute(self.scales)
            slice_blocks = default_slice_blocks(self.k)
            for m in range(1, self.max_rows + 1):
                self._cute_fns[m] = cute_compile(
                    x[:m],
                    self.values,
                    self.scales,
                    out[:m],
                    num_ctas=self.ctas,
                    slice_blocks=slice_blocks,
                )
        for m in range(1, self.max_rows + 1):
            self._gemv(x[:m], out[:m])
        torch.cuda.synchronize(dev)
        self._keys_ready = False

    def supports(self, rows: int) -> bool:
        return 0 < rows <= self.max_rows

    def describe(self) -> str:
        return (
            f"rows 1-{self.max_rows}, shard [{self.n}, {self.k}], "
            f"{self.values.numel() / 2**20:.1f} MiB values + "
            f"{self.scales.numel() / 2**20:.1f} MiB scales, GEMV {self.gemv} on "
            f"{self.ctas} CTAs, head_mx8.cu {source_digest()}"
        )

    def _gemv(self, x: torch.Tensor, out: torch.Tensor) -> None:
        if self.gemv == "cute":
            import cutlass

            self._cute_fns[int(x.shape[0])](
                _cute(x),
                self._cute_w,
                self._cute_s,
                _cute(out, 8),
                cutlass.Int32(0),
                cutlass.Float32(1.0),
                _cute_stream(),
            )
        else:
            nvrtc_gemv(x, self.values, self.scales, out, self.ctas)

    def note(self, site: str, rows: int) -> None:
        """Log once per (call site, row count) that the MXFP8 head ran there."""
        if (site, rows) not in self._noted:
            self._noted.add((site, rows))
            logger.info(
                "NVIDIA megakernel: MXFP8 draft head engaged: %s, %d row(s)%s",
                site,
                rows,
                " (graph capture)" if torch.cuda.is_current_stream_capturing() else "",
            )

    def logits(self, hidden: torch.Tensor, site: str) -> torch.Tensor:
        """Shard logits [m, N] bf16 of the final-norm rows ``hidden`` [m, 4096], m <=
        max_rows (where the BF16 head's ``_compute_lm_head`` would run)."""
        m = int(hidden.shape[0])
        if not self.supports(m) or hidden.shape[1] != self.k:
            raise ValueError(f"MXFP8 draft head: rows {tuple(hidden.shape)}")
        x = hidden.to(torch.bfloat16).contiguous()
        out = torch.empty((m, self.n), dtype=torch.bfloat16, device=x.device)
        self._gemv(x, out)
        self.note(site, m)
        return out

    def prepare_keys(self) -> None:
        """Compile and warm every supported row count's folded-key head."""
        dev = self.values.device
        x = torch.zeros(self.max_rows, self.k, dtype=torch.bfloat16, device=dev)
        keys = torch.zeros(self.max_rows, dtype=torch.int64, device=dev)
        for m in range(1, self.max_rows + 1):
            nvrtc_keys(
                x[:m],
                self.values,
                self.scales,
                keys,
                self.ctas,
                vocab_start=0,
                vocab_size=self.n,
                nan_key=0,
            )
        torch.cuda.synchronize(dev)
        self._keys_ready = True

    def fold_keys(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        *,
        vocab_start: int,
        vocab_size: int,
        nan_key: int,
        site: str,
    ) -> None:
        """Fold the keys of the shard logits that ``logits`` would have returned."""
        m = int(hidden.shape[0])
        if not self._keys_ready or not self.supports(m) or hidden.shape[1] != self.k:
            raise ValueError(
                f"MXFP8 draft head keys: rows {tuple(hidden.shape)} "
                f"(prepared {self._keys_ready})"
            )
        x = hidden.to(torch.bfloat16).contiguous()
        nvrtc_keys(
            x,
            self.values,
            self.scales,
            keys,
            self.ctas,
            vocab_start=vocab_start,
            vocab_size=vocab_size,
            nan_key=nan_key,
        )
        self.note(f"{site} (keys)", m)

    def prepare_mkd(self, smem_pad: int) -> None:
        """Load and run once the MKD head kernel for ``smem_pad`` (before capture)."""
        dev = self.values.device
        x = torch.zeros(1, self.k, dtype=torch.bfloat16, device=dev)
        logits = torch.empty(1, self.n, dtype=torch.bfloat16, device=dev)
        part = torch.empty(self.ctas, dtype=torch.int32, device=dev)
        mkd_head_mx8(x, self.values, self.scales, logits, part, smem_pad=smem_pad)
        torch.cuda.synchronize(dev)

    def mkd_head(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        part: torch.Tensor,
        *,
        reverse: bool,
        smem_pad: int,
    ) -> None:
        """``xchg.mkd_draft_head``'s outputs from the MXFP8 copy (``x`` [1, 4096] bf16)."""
        if not 0 < part.numel() <= -(-self.n // 8):
            raise ValueError(f"MXFP8 MKD head: {part.numel()} CTAs for {self.n} rows")
        mkd_head_mx8(
            x,
            self.values,
            self.scales,
            logits,
            part,
            reverse=reverse,
            smem_pad=smem_pad,
        )
        self.note("draft_loop MKD head", 1)
