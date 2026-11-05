# SPDX-License-Identifier: Apache-2.0
"""Original BS1 integer seed publication into borrowed persistent outputs."""

from dataclasses import fields
from enum import IntEnum

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpCaptureSeedIO,
    MtpCaptureSeedSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "MtpCaptureSeedIO",
        "MtpCaptureSeedSignature",
        "NumericOperation",
        "ScalarType",
    ),
}


def supported(s):
    if not isinstance(s, MtpCaptureSeedSignature):
        return False
    if any(
        type(v := getattr(s, f.name)) is not int and not isinstance(v, IntEnum)
        for f in fields(s)
    ):
        return False
    return (
        s.operation == NumericOperation.MTP_CAPTURE_SEED
        and s.rows == 4
        and s.front_tokens == 0
        and s.columns == 2051
        and s.seed_columns == 2055
        and s.topk_row_stride >= s.columns
        and s.seed_row_stride >= s.seed_columns
        and s.trash_row >= 0
        and s.seed_rows == s.trash_row + 1
        and all(
            dtype in (ScalarType.INT32, ScalarType.INT64)
            for dtype in (
                s.extend_scalar_type,
                s.accept_scalar_type,
                s.request_scalar_type,
                s.length_scalar_type,
            )
        )
        and s.pdl in (0, 1)
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported MTP seed capture signature")
    return MtpCaptureSeed(signature)


class MtpCaptureSeed:
    def __init__(self, signature):
        self.signature = signature
        self._warmed = set()

    def __call__(self, io):
        import torch

        if not isinstance(io, MtpCaptureSeedIO):
            raise TypeError("MTP seed capture requires its borrowed tensor view")
        s = self.signature
        tensors = tuple(getattr(io, f.name) for f in fields(io))
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("MTP seed capture requires seven tensors")
        device = io.topk_indices.device
        if device.type != "cuda" or any(t.device != device for t in tensors):
            raise ValueError("MTP seed tensors must share the CUDA device")
        topk, extend, accept, requests, lengths, seed, captured = tensors
        if (
            topk.dtype != torch.int32
            or topk.shape != (s.rows, s.columns)
            or topk.stride() != (s.topk_row_stride, 1)
            or seed.dtype != torch.int32
            or seed.shape != (s.seed_rows, s.seed_columns)
            or seed.stride() != (s.seed_row_stride, 1)
            or lengths.ndim != 1
            or lengths.numel() < s.rows
            or not lengths.is_contiguous()
            or captured.ndim != 1
            or captured.numel() < s.seed_rows
            or captured.dtype != torch.int32
            or not captured.is_contiguous()
        ):
            raise ValueError(
                "MTP seed capture layout differs from its prepared signature"
            )
        for t, dtype in (
            (extend, s.extend_scalar_type),
            (accept, s.accept_scalar_type),
            (requests, s.request_scalar_type),
            (lengths, s.length_scalar_type),
        ):
            expected = torch.int64 if dtype == ScalarType.INT64 else torch.int32
            if t.dtype != expected or (t is not lengths and t.numel() != 1):
                raise ValueError("MTP seed integer metadata differs from its signature")
        key = (
            device,
            tuple(
                (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() % 16)
                for t in tensors
            ),
        )
        if key not in self._warmed and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MTP seed exact key must be warmed before capture")
        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative.mtp_capture_metadata import (
            _mtp_capture_metadata,
        )

        _mtp_capture_metadata[((s.columns + 255) // 256,)](
            *tensors,
            ROWS=s.rows,
            COLS=s.columns,
            TOPK_STRIDE=s.topk_row_stride,
            SEED_STRIDE=s.seed_row_stride,
            FRONT=s.front_tokens,
            TRASH=s.trash_row,
            BLOCK=256,
            num_warps=4,
            launch_pdl=bool(s.pdl),
            PDL_SMALL=bool(s.pdl),
        )
        self._warmed.add(key)
