# SPDX-License-Identifier: Apache-2.0
"""Ordered fast prefill selection into framework-owned buffers.

The signature fixes the head geometry and bounds a row's compressed keys; the
packed rows and the call's width are per-call shapes the caller sizes. The
producer resolves the scale-fold and source gates. The variant carries the
source options the caller resolved: bits 0-3 the top-k
(SGLANG_QSA_PREFILL_FAST_SELECT_TOPK: 0 sampled, 1 onepass) and bit 4 the
tensor-core scorer (SGLANG_QSA_PREFILL_SCORER_TC).
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
_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024
_TOPK_IMPLS = ("sampled", "onepass")
_TOPK_MASK = 0xF
_SCORER_TC = 1 << 4


def _options(variant):
    """(top-k, tensor-core scorer) of a declared variant, or None for an unknown one."""
    topk = variant & _TOPK_MASK
    if variant & ~(_TOPK_MASK | _SCORER_TC) or topk >= len(_TOPK_IMPLS):
        return None
    return _TOPK_IMPLS[topk], bool(variant & _SCORER_TC)


def supported(s):
    return (
        isinstance(s, NumericSignature)
        and s.operation == NumericOperation.PREFILL_SELECT
        and (
            s.rows,
            s.query_heads,
            s.kv_heads,
            s.head_dim,
            s.selected_count,
            s.compress_ratio,
            s.scalar_type,
            s.rotary_dim,
            s.ctas_per_group,
        )
        == (0, 4, 1, 128, 512, 4, ScalarType.BF16, 0, 0)
        and 0 < s.context_bound <= 131072
        and _options(s.variant) is not None
    )


def build(s):
    if not supported(s):
        raise ValueError("unsupported fast prefill signature")
    return PrefillSelect(s)


class PrefillSelect:
    def __init__(self, signature):
        self.signature = signature
        self._topk_impl, self._scorer_tc = _options(signature.variant)
        self._prepared = set()

    def __call__(self, io):
        import torch

        from ..qsa import prefill_select as raw

        q, k, out = io.query, io.compressed_keys, io.selected_indices_out
        rows, width = q.shape[0], io.logits_out.shape[-1]
        metadata = (
            io.row_starts,
            io.row_ends,
            io.query_positions,
            io.sequence_lengths,
            io.row_lengths,
        )
        if (
            q.ndim != 3
            or q.shape[1:] != (4, 128)
            or rows == 0
            or q.dtype != torch.bfloat16
            or not q.is_cuda
            or k.ndim != 3
            or k.shape[1:] != (1, 128)
            or k.shape[0] == 0
            or k.dtype != q.dtype
            or out.shape != (rows, 2051)
            or out.dtype != torch.int32
            or any(t.shape != (rows,) or t.dtype != torch.int32 for t in metadata)
            or not 0 < width <= self.signature.context_bound
            or type(io.chunk_rows) is not int
            or type(io.scorer_sms) is not int
            or io.scorer_sms <= 0
        ):
            raise ValueError("fast prefill input differs from its declared layout")
        if io.chunk_rows != raw.plan_chunk_rows(
            rows, width, 2 * io.scorer_sms, _LOGITS_BUDGET_BYTES
        ):
            raise ValueError("caller chunk plan differs from source policy")
        padded = (io.chunk_rows + 31) // 32 * 32
        work = (
            (io.query_scratch_out, (padded, 4, 128), q.dtype),
            (io.starts_scratch_out, (padded,), torch.int32),
            (io.ends_scratch_out, (padded,), torch.int32),
            (io.zero_offsets, (io.chunk_rows,), torch.int32),
            (io.block_indices_out, (rows, 512), torch.int32),
            (io.logits_out, (padded, width), torch.float32),
            (io.split_tag, (4096,), torch.int32),
        )
        tensors = (q, k, out, *metadata, *(t for t, _, _ in work))
        if any(t.shape != shape or t.dtype != dtype for t, shape, dtype in work) or any(
            not t.is_contiguous() or t.device != q.device for t in tensors
        ):
            raise ValueError("fast prefill requires complete caller workspace")
        # The scorer compiles once; the top-k module depends on the row width.
        key = q.device, raw._topk_impl_for(width, self._topk_impl)
        if key not in self._prepared and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fast prefill layout must be warmed before capture")
        raw.select_into(io, width, scorer_tc=self._scorer_tc, topk_impl=self._topk_impl)
        self._prepared.add(key)
