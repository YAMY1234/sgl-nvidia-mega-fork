# SPDX-License-Identifier: Apache-2.0
"""Original deferred lookup resolved inside the caller-buffer KV preparation."""

from dataclasses import fields
from enum import IntEnum

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpDeferredKvIO,
    MtpDeferredKvSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "MtpDeferredKvIO",
        "MtpDeferredKvSignature",
        "NumericOperation",
        "ScalarType",
    ),
}


def supported(s):
    if not isinstance(s, MtpDeferredKvSignature):
        return False
    if any(
        type(v := getattr(s, f.name)) is not int and not isinstance(v, IntEnum)
        for f in fields(s)
    ):
        return False
    return (
        s.operation == NumericOperation.MTP_DEFERRED_KV_PREPARE
        and s.rows == s.kv_heads == 1
        and s.head_dim == 256
        and s.scalar_type == ScalarType.BF16
        and (s.index_columns, s.packed_columns, s.tail_width) == (2055, 2112, 4)
        and min(s.index_pool_rows, s.cache_slots, s.request_pool_rows, s.map_columns)
        > 0
        and s.map_row_stride >= s.map_columns
        and s.new_k_row_stride > 0
        and s.new_v_row_stride > 0
        and 0 <= s.metadata_dtype_mask < 64
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported deferred MTP KV signature")
    return MtpDeferredKv(signature)


class MtpDeferredKv:
    def __init__(self, signature):
        self.signature = signature
        self._warmed = set()

    def __call__(self, io):
        import torch

        if not isinstance(io, MtpDeferredKvIO):
            raise TypeError("deferred MTP KV requires its borrowed tensor view")
        s = self.signature
        tensors = tuple(getattr(io, f.name) for f in fields(io))
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("deferred MTP KV requires fifteen tensors")
        device = io.k_cache.device
        if device.type != "cuda" or any(t.device != device for t in tensors):
            raise ValueError("deferred MTP KV tensors must share the CUDA device")
        for t in (io.k_cache, io.v_cache):
            if (
                t.shape != (s.cache_slots, 1, 256)
                or t.dtype != torch.bfloat16
                or not t.is_contiguous()
            ):
                raise ValueError("deferred MTP KV cache differs from its signature")
        for t, stride in (
            (io.new_k, s.new_k_row_stride),
            (io.new_v, s.new_v_row_stride),
        ):
            if (
                t.shape != (1, 1, 256)
                or t.dtype != torch.bfloat16
                or t.stride() != (stride, 256, 1)
            ):
                raise ValueError("deferred MTP new KV differs from its signature")
        for t in (io.packed_k_out, io.packed_v_out):
            if (
                t.shape != (2112, 1, 256)
                or t.dtype != torch.bfloat16
                or not t.is_contiguous()
            ):
                raise ValueError(
                    "deferred MTP packed output differs from its signature"
                )
        if (
            io.indices_pool.shape != (s.index_pool_rows, 2055)
            or io.indices_pool.dtype != torch.int32
            or not io.indices_pool.is_contiguous()
            or io.captured_lengths.shape != (s.index_pool_rows,)
            or io.captured_lengths.dtype != torch.int32
            or not io.captured_lengths.is_contiguous()
            or io.counts_out.shape != (1,)
            or io.counts_out.dtype != torch.int32
            or not io.counts_out.is_contiguous()
        ):
            raise ValueError("deferred MTP index pool/captured/counts layout mismatch")
        metadata = (
            io.locations,
            io.row_requests,
            io.sequence_lengths,
            io.req_to_token,
            io.lookup_requests,
            io.lookup_positions,
        )
        for bit, t in enumerate(metadata):
            dtype = torch.int64 if s.metadata_dtype_mask & (1 << bit) else torch.int32
            if t.dtype != dtype:
                raise ValueError("deferred MTP metadata dtype differs from its mask")
            if bit == 3:
                if t.shape != (s.request_pool_rows, s.map_columns) or t.stride() != (
                    s.map_row_stride,
                    1,
                ):
                    raise ValueError("deferred MTP map differs from its signature")
            elif t.shape != (1,) or not t.is_contiguous():
                raise ValueError("deferred MTP row metadata requires its scalar view")
        key = (
            device,
            tuple(
                (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() % 16)
                for t in tensors
            ),
        )
        if key not in self._warmed and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "deferred MTP KV exact key must be warmed before capture"
            )
        from sgl_nvidia_mega.qwen3_8.qsa.fused_kv import _qsa_fused_kv_prepare

        _qsa_fused_kv_prepare[(134,)](
            io.k_cache,
            io.v_cache,
            io.new_k,
            io.new_v,
            io.locations,
            io.req_to_token,
            io.row_requests,
            io.indices_pool,
            io.sequence_lengths,
            None,
            io.counts_out,
            io.packed_k_out,
            io.packed_v_out,
            io.lookup_requests,
            io.captured_lengths,
            io.lookup_positions,
            1,
            256,
            s.map_row_stride,
            s.new_k_row_stride,
            s.new_v_row_stride,
            2055,
            2112,
            1,
            16,
            4096,
            1,
            1,
            False,
            1,
            2055,
            True,
            0,
            True,
            4,
            num_warps=4,
        )
        self._warmed.add(key)
