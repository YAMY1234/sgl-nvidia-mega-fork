# SPDX-License-Identifier: Apache-2.0
"""Original native ReplaySSM PDL verify into borrowed output/ring buffers."""

from dataclasses import fields
from enum import IntEnum

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NativeGdnPdlIO,
    NativeGdnPdlSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "NativeGdnPdlIO",
        "NativeGdnPdlSignature",
        "NumericOperation",
        "ScalarType",
    ),
}


def supported(s):
    if not isinstance(s, NativeGdnPdlSignature):
        return False
    if any(
        type(v := getattr(s, f.name)) is not int and not isinstance(v, IntEnum)
        for f in fields(s)
    ):
        return False
    return (
        s.operation == NumericOperation.GDN_NATIVE_PDL
        and s.rows > 0
        and s.rows % 4 == 0
        and (s.key_heads, s.value_heads) in ((4, 12), (16, 48))
        and s.key_dim == s.value_dim == 128
        and s.tokens_per_request == 4
        and s.scalar_type == s.gate_scalar_type == ScalarType.BF16
        and s.state_scalar_type in (ScalarType.BF16, ScalarType.FP32)
        and s.dt_bias_scalar_type in (ScalarType.BF16, ScalarType.FP32)
        and s.cache_len == 16
        and s.storage_mode in (0, 1, 2)
        and s.wait_for_gate_producer in (0, 1)
        and s.state_prefetch in (0, 1)
        and s.dot_precision in (0, 2)
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported native GDN PDL signature")
    return NativeGdnPdl(signature)


def _inner_dense(tensor, tail):
    """The raw ABI carries only outer strides; inner dimensions are dense."""
    if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
        return False
    stride = 1
    for size, actual in zip(reversed(tail), reversed(tensor.stride()[1:])):
        if actual != stride:
            return False
        stride *= size
    return tensor.stride(0) >= stride


class NativeGdnPdl:
    def __init__(self, signature):
        self.signature = signature
        self._warmed = set()

    def __call__(self, io):
        import torch

        if not isinstance(io, NativeGdnPdlIO):
            raise TypeError("native GDN PDL requires its borrowed tensor view")
        s = self.signature
        m, h, hv, bsz = s.rows, s.key_heads, s.value_heads, s.rows // 4
        optional = (io.rawv_cache, io.rawk_cache, io.beta_cache)
        if (
            tuple(t is not None for t in optional)
            != ((False, False, False), (True, True, False), (True, True, True))[
                s.storage_mode
            ]
        ):
            raise ValueError("native GDN PDL storage mode differs from its signature")
        tensors = tuple(
            getattr(io, f.name) for f in fields(io) if getattr(io, f.name) is not None
        )
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("native GDN PDL requires borrowed tensors")
        device = io.q.device
        if device.type != "cuda" or any(t.device != device for t in tensors):
            raise ValueError("native GDN PDL tensors must share the CUDA device")
        state_dtype = (
            torch.float32 if s.state_scalar_type == ScalarType.FP32 else torch.bfloat16
        )
        bias_dtype = (
            torch.float32
            if s.dt_bias_scalar_type == ScalarType.FP32
            else torch.bfloat16
        )
        # Checkpoint slots and replay slots are independent physical pools.
        slots = io.d_cache.shape[0] if io.d_cache.ndim == 4 else 0
        views = (
            (io.q, (h, 128), torch.bfloat16, m),
            (io.k, (h, 128), torch.bfloat16, m),
            (io.v, (hv, 128), torch.bfloat16, m),
            (io.a, (hv,), torch.bfloat16, m),
            (io.b, (hv,), torch.bfloat16, m),
            (io.out, (hv, 128), torch.bfloat16, m),
            (io.checkpoint_state, (hv, 128, 128), state_dtype, None),
            (io.d_cache, (hv, 16, 128), torch.bfloat16, slots),
            (io.k_cache, (h, 16, 128), torch.bfloat16, slots),
            (io.g_cache, (hv, 16), torch.float32, slots),
        )
        if (
            slots < 1
            or io.checkpoint_state.ndim != 4
            or io.checkpoint_state.shape[0] < 1
            or any(
                not _inner_dense(t, tail)
                or t.dtype != dtype
                or (rows is not None and t.shape[0] != rows)
                for t, tail, dtype, rows in views
            )
            or io.a_log.shape != (hv,)
            or io.a_log.dtype != torch.float32
            or not io.a_log.is_contiguous()
            or io.dt_bias.shape != (hv,)
            or io.dt_bias.dtype != bias_dtype
            or not io.dt_bias.is_contiguous()
        ):
            raise ValueError("native GDN PDL views differ from the prepared layout")
        for t, size in (
            (io.query_start_loc, bsz + 1),
            (io.ssm_state_indices, bsz),
            (io.replay_indices, bsz),
        ):
            if (
                t.shape != (size,)
                or t.dtype not in (torch.int32, torch.int64)
                or t.stride(0) < 1
            ):
                raise ValueError(
                    "native GDN PDL requires actual query/state/replay metadata"
                )
        for t, dtype in (
            (io.write_pos, torch.int32),
            (io.cache_base, torch.int32),
            (io.is_flush, torch.int8),
        ):
            if (
                t.ndim != 1
                or t.numel() < slots
                or t.dtype != dtype
                or not t.is_contiguous()
            ):
                raise ValueError("native GDN PDL requires prepared cursors/int8 flush")
        for t, tail, dtype in (
            (io.rawv_cache, (hv, 16, 128), torch.bfloat16),
            (io.rawk_cache, (h, 16, 128), torch.bfloat16),
            (io.beta_cache, (hv, 16), torch.float32),
        ):
            if t is not None and (
                not _inner_dense(t, tail) or t.shape[0] != slots or t.dtype != dtype
            ):
                raise ValueError(
                    "native GDN PDL raw ring differs from the declared storage"
                )
        # Pointer alignment affects Triton's specialization, but addresses and
        # activation owners are not persistent resources. Each call passes its
        # own real views; only a successful exact eager key becomes capture-safe.
        key = (
            device,
            tuple(
                (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() % 16)
                for t in tensors
            ),
        )
        if key not in self._warmed and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("native GDN PDL exact key must be warmed before capture")
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_gdn_front.replayssm_pdl import (
            gdn_replayssm_spec_circular_kernel,
        )

        # These are the original verify wrapper's dummy pointers for unused
        # optional storage. They neither manufacture raw buffers nor enable it.
        rawv = io.rawv_cache if io.rawv_cache is not None else io.d_cache
        rawk = io.rawk_cache if io.rawk_cache is not None else io.k_cache
        beta = io.beta_cache if io.beta_cache is not None else io.g_cache
        gdn_replayssm_spec_circular_kernel[(2, bsz, hv)](
            io.q,
            io.k,
            io.v,
            io.a,
            io.b,
            io.a_log,
            io.dt_bias,
            io.out,
            io.checkpoint_state,
            io.d_cache,
            io.k_cache,
            io.g_cache,
            rawv,
            rawk,
            beta,
            io.query_start_loc,
            io.ssm_state_indices,
            io.replay_indices,
            io.write_pos,
            io.cache_base,
            io.is_flush,
            128**-0.5,
            io.q.stride(0),
            io.k.stride(0),
            io.v.stride(0),
            io.a.stride(0),
            io.b.stride(0),
            io.out.stride(0),
            io.checkpoint_state.stride(0),
            io.d_cache.stride(0),
            io.k_cache.stride(0),
            io.g_cache.stride(0),
            rawv.stride(0),
            rawk.stride(0),
            beta.stride(0),
            io.query_start_loc.stride(0),
            io.ssm_state_indices.stride(0),
            io.replay_indices.stride(0),
            H=h,
            HV=hv,
            K=128,
            V=128,
            BK=128,
            BV=64,
            BS=4,
            BC=16,
            NK=2,
            BKT=64,
            MAX_CACHE_LEN=16,
            SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=True,
            STATE_FP32=s.state_scalar_type == ScalarType.FP32,
            STORE_EXACT_RAW=s.storage_mode == 2,
            STORE_D_RESIDUAL=s.storage_mode == 1,
            STORE_K_RESIDUAL=s.storage_mode == 1,
            IS_FLUSH=False,
            NULL_BLOCK_ID=-1,
            DOT_PRECISION="tf32" if s.dot_precision == 0 else "ieee",
            num_warps=1,
            num_stages=2,
            USE_GDC=True,
            launch_pdl=True,
            WAIT_FOR_GATE_PRODUCER=bool(s.wait_for_gate_producer),
            PREFETCH_CHECKPOINT=bool(s.state_prefetch),
        )
        self._warmed.add(key)
