# SPDX-License-Identifier: Apache-2.0
"""Prefix-chunk sparse prefill attention with TMA-gathered K/V.

The producer admits only passes on which stock runs this kernel's tile (16
tokens, 1 warp), so the output is the stock kernel's. The signature fixes the
head geometry and stage count; the packed rows and max query rows are per call.
"""

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericOperation,
    NumericSignature,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "NumericOperation",
        "NumericSignature",
        "ScalarType",
    ),
}


def supported(s):
    return (
        isinstance(s, NumericSignature)
        and s.operation == NumericOperation.PACKED_PREFILL_ATTENTION
        and s.rows == s.rotary_dim == s.context_bound == s.ctas_per_group == 0
        and 0 < s.kv_heads
        and s.query_heads % s.kv_heads == 0
        and s.query_heads // s.kv_heads <= 16
        and (s.head_dim, s.selected_count, s.compress_ratio, s.scalar_type)
        == (256, 2051, 4, ScalarType.BF16)
        and 1 <= s.variant <= 8
    )


def build(s):
    if not supported(s):
        raise ValueError("unsupported packed prefill signature")
    return PackedPrefill(s)


class PackedPrefill:
    def __init__(self, signature):
        self.signature = signature
        self._prepared = set()

    def __call__(self, io):
        import torch

        s = self.signature
        q, k, v, out, idx = (
            io.query,
            io.packed_k,
            io.packed_v,
            io.output,
            io.selected_indices,
        )
        cq, ck, lengths = io.cu_query, io.cu_key, io.sequence_lengths
        tensors = (q, k, v, out, idx, cq, ck, lengths)
        if (
            q.ndim != 3
            or q.shape[1:] != (s.query_heads, 256)
            or not q.is_cuda
            or q.dtype != torch.bfloat16
            or q.stride(2) != 1
            or k.ndim != 3
            or k.shape[1:] != (s.kv_heads, 256)
            or k.shape != v.shape
            or k.shape[0] >= 2**31 - 1
            or k.dtype != q.dtype
            or v.dtype != q.dtype
            or not k.is_contiguous()
            or not v.is_contiguous()
            or k.data_ptr() % 16
            or v.data_ptr() % 16
            or k.stride(0) * k.element_size() % 16
            or out.shape != q.shape
            or out.dtype != q.dtype
            or not out.is_contiguous()
            or idx.shape != (q.shape[0], 2051)
            or idx.dtype != torch.int32
            or idx.stride(1) != 1
            or cq.ndim != 1
            or cq.shape[0] < 2
            or ck.shape != cq.shape
            or lengths.shape != (cq.shape[0] - 1,)
            or any(
                t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()
                for t in (cq, ck, lengths)
            )
            or any(t.device != q.device for t in tensors)
            or type(io.max_query_rows) is not int
            or not 0 < io.max_query_rows <= q.shape[0]
        ):
            raise ValueError("packed prefill input differs from its declared layout")
        # One Triton specialization per device; shapes and the grid are runtime.
        if q.device not in self._prepared and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("packed prefill must be warmed before capture")
        from ..qsa import tma_prefill as raw

        raw.launch(io, max_q=io.max_query_rows, stages=s.variant)
        self._prepared.add(q.device)
