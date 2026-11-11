# SPDX-License-Identifier: Apache-2.0
"""FP8 pool union target verify; its reordered FP32 reduction is not bitwise stock.

The variant carries the source options the caller resolved: bits 0-5 kernel U's warp
count (SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS; 0 is the source's 8; its output does not
depend on it), bit 6 the warp-specialized attention kernel
(SGLANG_QSA_FP8_POOL_VERIFY_WS), whose split count the caller resolves, bit 7 (with
bit 6 only) PDL into that kernel (SGLANG_QSA_FP8_POOL_VERIFY_WS_PDL; same math), and bit 8
BF16 new K/V cast to E4M3 in the pool write (SGLANG_QSA_FUSED_FP8_KV_WRITE; the cast is
checked against torch's on every BF16 pattern before first use, so the pool bytes match).
"""

from dataclasses import fields

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


_UNION_WARPS_MASK = 0x3F
_WS = 1 << 6
_WS_PDL = 1 << 7
_FP8_KV_WRITE = 1 << 8


def _options(variant):
    """(kernel U warps, warp-specialized, PDL into it, BF16 new K/V cast in the write)
    of a declared variant, or None for an unknown one."""
    warps = variant & _UNION_WARPS_MASK
    if (
        variant & ~(_UNION_WARPS_MASK | _WS | _WS_PDL | _FP8_KV_WRITE)
        or warps not in (0, 1, 2, 4, 8, 16, 32)
        or (variant & _WS_PDL and not variant & _WS)
    ):
        return None
    return (
        warps or 8,
        bool(variant & _WS),
        bool(variant & _WS_PDL),
        bool(variant & _FP8_KV_WRITE),
    )


def supported(s):
    return (
        isinstance(s, NumericSignature)
        and all(
            isinstance(getattr(s, f.name), int)
            and not isinstance(getattr(s, f.name), bool)
            for f in fields(s)
        )
        and s.operation == NumericOperation.POOL_UNION_ATTENTION
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
        and _options(s.variant) is not None
        and 64 <= s.rows <= 576
        and s.rows % 4 == 0
    )


def build(s):
    if not supported(s):
        raise ValueError("unsupported FP8 pool union signature")
    return PoolUnion(s)


class PoolUnion:
    def __init__(self, signature):
        self.signature = signature
        self._union_warps, self._ws, self._ws_pdl, self._cast_write = _options(
            signature.variant
        )
        self._resources = {}

    def __call__(self, io):
        import torch

        s = self.signature
        q, k, v, nk, nv = io.q, io.kv.k, io.kv.v, io.new_k, io.new_v
        sel, out, loc = io.selection, io.output, io.out_cache_loc
        integers = (
            loc,
            sel.full_slot_map,
            sel.row_requests,
            sel.sequence_lengths,
            sel.query_positions,
        )
        borrowed = (q, k, v, nk, nv, *integers, sel.selected_indices, out)
        if (
            io.gate is not None
            or io.query_width != 4
            or type(io.store_new_kv) is not bool
            or (io.kv.k_scale, io.kv.v_scale) != (1.0, 1.0)
            or q.shape != (s.rows, 24, 256)
            or q.dtype != torch.bfloat16
            or q.stride(1) != 256
            or q.stride(2) != 1
            or k.ndim != 3
            or k.shape[1:] != (2, 256)
            or v.shape != k.shape
            or any(
                t.dtype != torch.float8_e4m3fn or not t.is_contiguous() for t in (k, v)
            )
            or not all(
                (t.dtype == torch.float8_e4m3fn and t.is_contiguous())
                or (
                    self._cast_write
                    and t.dtype == torch.bfloat16
                    and t.stride(1) == 256
                    and t.stride(2) == 1
                )
                for t in (nk, nv)
            )
            or nk.dtype != nv.dtype
            or nk.shape != (s.rows, 2, 256)
            or nv.shape != nk.shape
            or out.shape != q.shape
            or out.dtype != q.dtype
            or not out.is_contiguous()
            or sel.full_slot_map.ndim != 2
            or sel.full_slot_map.stride(1) != 1
            or any(t.dtype not in (torch.int32, torch.int64) for t in integers)
            or any(
                t.shape != (s.rows,) or not t.is_contiguous()
                for t in (
                    loc,
                    sel.row_requests,
                    sel.sequence_lengths,
                    sel.query_positions,
                )
            )
            or sel.selected_indices.shape != (s.rows, 512)
            or sel.selected_indices.dtype != torch.int32
            or not sel.selected_indices.is_contiguous()
            or not q.is_cuda
            or any(t.device != q.device for t in borrowed)
        ):
            raise ValueError("FP8 union input differs from its prepared layout")
        if type(io.split_count) is not int or not 1 <= io.split_count <= 16:
            raise ValueError("invalid union split count")
        requests = s.rows // 4
        nb, nt = 2048, 8208
        stats = requests * 2 * io.split_count * 64
        integer = io.integer_scratch_out
        fp = io.scratch_out
        integer_count = requests * (2 * nt + nb + 1) + s.rows
        if (
            integer.dtype != torch.int32
            or fp.dtype != torch.float32
            or integer.ndim != 1
            or fp.ndim != 1
            or integer.numel() < integer_count
            or fp.numel() < stats * 258
            or not integer.is_contiguous()
            or not fp.is_contiguous()
            or integer.device != q.device
            or fp.device != q.device
        ):
            raise ValueError("union requires caller integer and FP32 scratch")
        key = (
            q.device,
            io.split_count,
            tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in borrowed),
        )
        if torch.cuda.is_current_stream_capturing() and key not in self._resources:
            raise RuntimeError("FP8 union layout must be warmed before capture")
        a = requests * nt
        b = a * 2
        c = b + requests * nb
        d = c + requests
        workspace = (
            integer[:a],
            integer[a:b],
            integer[b:c],
            integer[c:d],
            integer[d : d + s.rows],
            fp[:stats],
            fp[stats : 2 * stats],
            fp[2 * stats : 2 * stats + stats * 256],
        )
        if nk.dtype == torch.bfloat16:
            from ..qsa import pool_decode

            # Checked once per device, eagerly (an unchecked pool under capture refuses).
            if not pool_decode.write_new_kv_e4m3_ok(k, v):
                raise ValueError(
                    "the E4M3 cast write differs from torch's cast on this device "
                    "(or was not checked before capture): unset "
                    "SGLANG_QSA_FUSED_FP8_KV_WRITE"
                )
        from ..qsa import pool_union_decode as raw

        launch, options = raw.qsa_pool_union_decode, {}
        if self._ws:
            from ..qsa import pool_union_ws

            # Shapes it does not serve run kernel 2 with the same workspace and splits.
            launch = pool_union_ws.qsa_pool_union_decode_ws
            options = {"ws_pdl": self._ws_pdl}
        launch(
            q,
            k,
            v,
            nk,
            nv,
            loc,
            sel.full_slot_map,
            sel.row_requests,
            sel.selected_indices,
            sel.sequence_lengths,
            sel.query_positions,
            4,
            io.scale,
            ratio=4,
            out=out.view(s.rows, 24 * 256),
            workspace=workspace,
            splits=io.split_count,
            waves=None,
            bn=128,
            num_warps=8,
            num_stages=3,
            pdl=True,
            write_new_kv=io.store_new_kv,
            union_warps=self._union_warps,
            **options,
        )
        self._resources[key] = True
