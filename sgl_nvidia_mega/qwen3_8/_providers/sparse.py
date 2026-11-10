# SPDX-License-Identifier: Apache-2.0
"""Finite sparse scoring/selection into framework-owned buffers.

Metadata, layout conversions and paired source-policy admission stay in SGLang.
Only eager warmup may prepare a new compiled layout; capture misses fail closed.
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
    if not isinstance(s, NumericSignature):
        return False
    if (s.kv_heads, s.head_dim, s.selected_count, s.compress_ratio, s.rotary_dim) != (
        1,
        128,
        512,
        4,
        0,
    ) or not 0 < s.context_bound <= 131072:
        return False
    variant = getattr(s, "variant", 0)
    grouped4 = s.operation == NumericOperation.SPARSE_SCORE and variant in (6, 7)
    if (s.ctas_per_group > 0) != grouped4:
        return False
    if (
        s.operation == NumericOperation.SPARSE_SCORE
        and s.scalar_type == ScalarType.BF16
    ):
        if grouped4:
            # The source's grouped4 domain: 17..144 four-row verify requests.
            return 64 < s.rows <= 576 and s.rows % 4 == 0 and s.query_heads == 8
        if variant in (0, 3):
            return s.rows == 4 and s.query_heads == 8
        if variant == 4:
            return s.rows in range(4, 65, 4) and s.query_heads in (8, 16)
        if variant == 5:
            return s.rows in range(1, 65) and s.query_heads in (8, 16)
    if (
        s.operation == NumericOperation.SPARSE_SELECT
        and s.scalar_type == ScalarType.FP32
    ):
        if variant == 0:
            return s.rows == 4 and s.query_heads == 8
        if variant == 1:
            return s.rows in range(1, 65) and s.query_heads in (8, 16)
    return False


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported sparse score/select signature")
    return (
        SparseScore(signature)
        if signature.operation == NumericOperation.SPARSE_SCORE
        else SparseSelect(signature)
    )


class SparseScore:
    def __init__(self, signature):
        self.signature = signature
        self._resources = {}

    def __call__(self, io):
        import torch

        capacity = self.signature.context_bound
        rows, heads = self.signature.rows, self.signature.query_heads
        variant = getattr(self.signature, "variant", 0)
        mk = variant in (0, 3)
        q, keys = io.query, io.compressed_keys
        table, lengths, requests = (
            io.compressed_slot_map,
            io.compressed_lengths,
            io.row_requests,
        )
        output = io.scores_out
        tensors = (q, keys, table, lengths, output) + ((requests,) if mk else ())
        if (
            q.shape != (rows, heads, 128)
            or keys.ndim != 4
            or tuple(keys.shape[1:]) != (16, 1, 128)
            or table.ndim != 2
            or table.shape[0] != rows
            or table.shape[1] * 16 < capacity
            or lengths.shape != (rows,)
            or (mk and (requests is None or requests.shape != (rows,)))
            or output.shape != (rows, capacity)
            or q.dtype != torch.bfloat16
            or keys.dtype != torch.bfloat16
            or table.dtype != torch.int32
            or lengths.dtype != torch.int32
            or (mk and requests.dtype not in (torch.int32, torch.int64))
            or output.dtype != torch.float32
            or any(not t.is_contiguous() or t.device != q.device for t in tensors)
            or io.query_weights is not None
            or (io.page_size, io.compress_ratio) != (16, 4)
            or not io.scale > 0
        ):
            raise ValueError("sparse score view does not match its prepared capability")
        # Everything else is fixed by the signature and contiguous-view checks.
        key = keys.shape[0], table.shape[1], requests.dtype if mk else None, q.device
        kernel = self._resources.get(key)
        cold = kernel is None
        if cold:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "sparse score cold layout must be warmed before capture"
                )
            if mk:
                request_dtype = {torch.int32: "int32", torch.int64: "int64"}[
                    requests.dtype
                ]
                if variant == 3:
                    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_perf_score_short import (
                        _perf_score_kernel as factory,
                    )
                else:
                    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_mk_score import (
                        _mk_score_kernel as factory,
                    )
                kernel = factory(
                    min_scored_len=512, request_dtype=request_dtype, long_ctas=1024
                )
            else:
                from sgl_nvidia_mega.qwen3_8.qsa import mqa

                args = dict(
                    heads=heads,
                    head_dim=128,
                    page_size=16,
                    min_scored_len=512,
                    dtype="bfloat16",
                )
                if variant in (6, 7):
                    kernel = (
                        mqa._tilelang_qsa_mqa_decode_grouped4_pipe_kernel(
                            **args,
                            ctas_per_row=self.signature.ctas_per_group,
                            num_stages=2,
                        )
                        if variant == 7
                        else mqa._tilelang_qsa_mqa_decode_grouped4_chain_kernel(
                            **args, ctas_per_row=self.signature.ctas_per_group
                        )
                    )
                elif variant == 4:
                    ctas = max(
                        level
                        for level in (16, 32, 64, 128, 256, 512, 1024)
                        if level <= 4096 // (rows // 4)
                    )
                    expected = dict(W=4, C=ctas, ver=7, NT=1, guard=False, peel=False)
                    if mqa._v2_config(rows, 4, q.device) != expected:
                        raise ValueError("grouped scorer overrides require stock")
                    kernel = mqa._tilelang_qsa_mqa_decode_grouped_v7_kernel(
                        **args, rows_per_group=4, ctas_per_group=ctas
                    )
                else:
                    if mqa._v2_config(rows, 1, q.device) is not None:
                        raise ValueError("per-row scorer overrides require stock")
                    kernel = mqa._tilelang_qsa_mqa_decode_kernel(
                        **args, ctas_per_row=mqa._decode_ctas_per_row(rows, q.device)
                    )
        if mk:
            kernel(q, keys, table, lengths, requests, output, float(io.scale))
        else:
            # The original wrapper uses this same zero-allocation input view.
            kernel(q.unsqueeze(1), keys, table, lengths, output, float(io.scale))
        # TileLang can compile on its first tensor call, so publish only after
        # the eager launch succeeded. Replay reuses exactly this layout.
        if cold:
            self._resources[key] = kernel


class SparseSelect:
    """Only indices are written; counts_out is framework-prepared metadata."""

    def __init__(self, signature):
        self.signature = signature
        self._resources = {}

    def __call__(self, io):
        import torch

        capacity = self.signature.context_bound
        rows = self.signature.rows
        split = getattr(self.signature, "variant", 0) == 1
        scores, lengths, output = io.scores, io.lengths, io.selected_indices_out
        if (
            scores.shape != (rows, capacity)
            or scores.dtype != torch.float32
            or lengths.shape != (rows,)
            or lengths.dtype != torch.int32
            or output.shape != (rows, 512)
            or output.dtype != torch.int32
            or io.selected_count != 512
            or any(
                not t.is_contiguous() or t.device != scores.device
                for t in (scores, lengths, output)
            )
        ):
            raise ValueError(
                "sparse select view does not match its prepared capability"
            )
        vec = (
            scores.data_ptr() % 16 == 0 and scores.stride(0) % 4 == 0 if split else None
        )
        key = scores.device, vec
        module = self._resources.get(key)
        cold = module is None
        if cold:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "sparse select cold layout must be warmed before capture"
                )
            if split:
                from sgl_nvidia_mega.qwen3_8.qsa import split_topk

                if (
                    split_topk._env_int("SGLANG_QSA_TOPK_THREADS") not in (None, 1024)
                    or split_topk._env_int("SGLANG_QSA_TOPK_CLUSTER") is not None
                ):
                    raise ValueError("split selector overrides require stock")
                threads, cluster, iters = split_topk._config(rows, capacity)
                if threads != 1024 or not 1 <= cluster <= 16 or not 1 <= iters <= 8:
                    raise ValueError("unsupported default split selector capacity")
                module = split_topk._module(threads, cluster, iters, vec)
            else:
                from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import q38_mk_topk

                items = max(1, (capacity + 4095) // 4096)
                module = q38_mk_topk._module(items, True)
        if split:
            module.cluster_topk(scores, lengths, output)
        else:
            module.select_rows(scores, lengths, None, output)
        if cold:
            self._resources[key] = module
