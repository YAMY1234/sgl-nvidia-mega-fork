# SPDX-License-Identifier: Apache-2.0
"""Original TP1 pool split-K launches into framework-owned output and scratch."""

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
        and s.operation == NumericOperation.DIRECT_POOL_ATTENTION
        and (
            s.query_heads,
            s.kv_heads,
            s.head_dim,
            s.selected_count,
            s.compress_ratio,
            s.scalar_type,
            s.rotary_dim,
            s.context_bound,
        )
        == (24, 2, 256, 512, 4, ScalarType.BF16, 0, 0)
        and (
            (s.variant == 1 and 4 <= s.rows <= 64 and s.rows % 4 == 0)
            or (s.variant == 2 and 1 <= s.rows <= 64)
        )
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported TP1 pool signature")
    return PoolAttention(signature)


class PoolAttention:
    """Keep only eager-prepared hardware/layout keys, never activation owners."""

    def __init__(self, signature):
        if not supported(signature):
            raise ValueError("unsupported TP1 pool signature")
        self.signature = signature
        self.width = 4 if signature.variant == 1 else 1
        self._prepared = set()
        self._configs = {}

    def __call__(self, io):
        import torch

        from sgl_nvidia_mega.qwen3_8.qsa import pool_decode as raw

        s = self.signature
        q, k, v = io.q, io.kv.k, io.kv.v
        nk, nv, loc = io.new_k, io.new_v, io.out_cache_loc
        sel, out, scratch = io.selection, io.output, io.scratch_out
        integers = (
            loc,
            sel.full_slot_map,
            sel.row_requests,
            sel.sequence_lengths,
            sel.query_positions,
        )
        borrowed = (q, k, v, nk, nv, *integers, sel.selected_indices, out, scratch)
        if (
            io.gate is not None
            or io.query_width != self.width
            or not isinstance(io.store_new_kv, bool)
            or q.shape != (s.rows, 24, 256)
            or not q.is_cuda
            or q.stride(1) != 256
            or q.stride(2) != 1
            or q.stride(0) < 24 * 256
            or k.ndim != 3
            or k.shape[1:] != (2, 256)
            or v.shape != k.shape
            or not k.is_contiguous()
            or not v.is_contiguous()
            or any(
                t.shape != (s.rows, 2, 256)
                or t.stride(1) != 256
                or t.stride(2) != 1
                or t.stride(0) < 2 * 256
                for t in (nk, nv)
            )
            or any(t.dtype != torch.bfloat16 for t in (q, k, v, nk, nv, out))
            or out.shape != (s.rows, 24, 256)
            or not out.is_contiguous()
            or sel.full_slot_map.ndim != 2
            or sel.full_slot_map.stride(1) != 1
            or any(
                t.shape != (s.rows,) or not t.is_contiguous()
                for t in (
                    loc,
                    sel.row_requests,
                    sel.sequence_lengths,
                    sel.query_positions,
                )
            )
            or any(t.dtype not in (torch.int32, torch.int64) for t in integers)
            or sel.selected_indices.shape != (s.rows, 512)
            or sel.selected_indices.dtype != torch.int32
            or not sel.selected_indices.is_contiguous()
            or scratch.dtype != torch.float32
            or scratch.ndim != 1
            or not scratch.is_contiguous()
            or any(t.device != q.device for t in borrowed)
        ):
            raise ValueError("TP1 pool view differs from its prepared source layout")
        capturing = torch.cuda.is_current_stream_capturing()
        device = q.device
        if device not in self._configs:
            if capturing:
                raise RuntimeError(
                    "TP1 pool hardware config must be prepared before capture"
                )
            # The original cache reads actual hardware only during eager warmup.
            self._configs[device] = raw.choose_config(
                s.rows,
                2,
                s.selected_count * s.compress_ratio + s.compress_ratio - 1,
                raw._num_sms(device),
            )
        splits, chunk, bn = self._configs[device]
        nparts, gp, dim = s.rows * 2 * splits, 16, 256
        stats = nparts * gp
        if scratch.numel() < stats * (dim + 2):
            raise ValueError("TP1 pool requires caller FP32 split-K scratch")
        # Addresses are runtime parameters, not compile/resource keys or owners.
        key = (
            device,
            tuple(
                (tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() % 16)
                for t in borrowed
            ),
            io.store_new_kv,
        )
        if capturing and key not in self._prepared:
            raise RuntimeError("TP1 pool layout must be prepared before capture")
        maximum = scratch[:stats]
        sums = scratch[stats : 2 * stats]
        acc = scratch[2 * stats : 2 * stats + stats * dim]
        nk2, nv2 = nk.view(s.rows, 512), nv.view(s.rows, 512)
        out2 = out.view(s.rows, 24 * dim)
        scale = float(io.scale) * raw.LOG2E
        # Exact original separate-write / separate-merge, including PDL ordering.
        if io.store_new_kv:
            raw._qsa_write_new_kv[(s.rows, 2)](
                k,
                v,
                nk2,
                nv2,
                loc,
                nk2.stride(0),
                nv2.stride(0),
                2,
                dim,
                True,
                num_warps=1,
                launch_pdl=True,
            )
        raw.LAST_KERNEL["splitk"] = raw._qsa_pool_splitk[(s.rows, 2, splits)](
            q,
            k,
            v,
            nk2,
            nv2,
            loc,
            sel.full_slot_map,
            sel.row_requests,
            sel.selected_indices,
            sel.sequence_lengths,
            sel.query_positions,
            maximum,
            sums,
            acc,
            maximum,
            out2,
            out2.stride(0),
            scale,
            q.stride(0),
            sel.full_slot_map.stride(0),
            nk2.stride(0),
            nv2.stride(0),
            sel.selected_indices.stride(0),
            2,
            dim,
            12,
            gp,
            512,
            4,
            self.width,
            splits,
            chunk,
            bn,
            0,
            False,
            False,
            True,
            num_warps=4,
            num_stages=4,
            launch_pdl=True,
        )
        sb = min(16, 1 << (splits - 1).bit_length())
        raw._qsa_pool_merge[(s.rows * 24,)](
            maximum,
            sums,
            acc,
            out2,
            out2.stride(0),
            2,
            dim,
            12,
            gp,
            splits,
            sb,
            True,
            num_warps=2,
            launch_pdl=True,
        )
        # A failed launch never authorizes capture or a fallback after KV writes.
        self._prepared.add(key)
