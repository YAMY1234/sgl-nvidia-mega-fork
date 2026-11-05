# SPDX-License-Identifier: Apache-2.0
"""Original T4 ReplaySSM glue launch with framework-owned output and metadata."""

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "GdnSignature",
        "NumericOperation",
        "ScalarType",
        "GdnVerifyGlueIO",
    ),
}


def supported(s):
    return isinstance(s, GdnSignature) and (
        s.operation == NumericOperation.GDN_VERIFY_GLUE
        and s.variant == 0
        and s.rows in (8, 12, 16)
        and (s.key_heads, s.value_heads) in ((4, 12), (16, 48))
        and s.key_dim == s.value_dim == 128
        and s.tokens_per_request == 4
        and s.scalar_type == s.gate_scalar_type == ScalarType.BF16
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported GDN verify glue signature")
    return GdnVerifyGlue(signature)


class GdnVerifyGlue:
    def __init__(self, signature):
        self.signature = signature
        self._warmed = set()

    def __call__(self, io):
        import torch

        s, r, launch = self.signature, io.ring, io.launch
        m, h, hv, dim = s.rows, s.key_heads, s.value_heads, s.key_dim
        channels = (2 * h + hv) * dim
        capacity = r.d.shape[-2]
        if (
            (io.key_heads, io.key_dim, io.tokens_per_request) != (h, dim, 4)
            or (
                launch.block_n,
                launch.num_warps,
                launch.num_stages,
                launch.hoist,
                launch.dot_precision,
                launch.prefetch,
                launch.historical_branch,
                launch.use_pdl,
                launch.trigger_early,
            )
            != (0, 4, 1, False, 0, True, 1, True, True)
            or io.conv_activation != 1
            or io.norm_activation not in (1, 2)
            or io.mixed_qkvz.ndim != 2
            or io.mixed_qkvz.shape[0] != m
            or io.mixed_qkvz.shape[1] < channels + hv * dim
            or io.mixed_ba.ndim != 2
            or io.mixed_ba.shape[0] != m
            or io.mixed_ba.shape[1] < 2 * hv
            or io.mixed_qkvz.stride(-1) != 1
            or io.mixed_ba.stride(-1) != 1
            or io.conv_state.ndim != 3
            or io.conv_state.shape[1:] != (channels, 3)
            or io.conv_weight.shape != (channels, 4)
            or not io.conv_weight.is_contiguous()
            # Original framework source admission requires no convolution bias.
            or io.conv_bias is not None
            or io.intermediate_conv_window_out.ndim != 4
            or io.intermediate_conv_window_out.shape[1:] != (4, channels, 3)
            or io.checkpoint.ndim != 4
            or io.checkpoint.shape[1:] != (hv, dim, dim)
            or io.norm_weight.shape != (dim,)
            or not io.norm_weight.is_contiguous()
            or io.a_log.shape != (hv,)
            or io.dt_bias.shape != (hv,)
            or io.output.shape != (m, hv * dim)
            or not io.output.is_contiguous()
            or any(
                t.dtype != torch.bfloat16
                for t in (
                    io.mixed_qkvz,
                    io.mixed_ba,
                    io.conv_state,
                    io.conv_weight,
                    io.intermediate_conv_window_out,
                    io.checkpoint,
                    io.norm_weight,
                    io.output,
                )
            )
            or io.dt_bias.dtype not in (torch.bfloat16, torch.float32)
            or io.a_log.dtype != torch.float32
            or r.is_flush.dtype != torch.int8
            or capacity < 1
            or capacity & (capacity - 1)
            or r.d.shape[1:] != (hv, capacity, dim)
            or r.k.shape[1:] != (h, capacity, dim)
            or r.g.shape[1:] != (hv, capacity)
            or any(not t.is_contiguous() for t in (r.d, r.k, r.g))
            or io.intermediate_indices.numel() < m // 4
            or r.state_indices.numel() < m // 4
            or r.replay_indices.numel() < m // 4
        ):
            raise ValueError("GDN verify glue differs from prepared source metadata")
        # Preserve raw/exact/residual mode and alias choices of the original wrapper.
        store_exact = r.raw_v is not None and r.raw_k is not None and r.beta is not None
        store_residual = r.raw_v is not None and r.raw_k is not None and r.beta is None
        rawv = r.raw_v if r.raw_v is not None else r.d
        rawk = r.raw_k if r.raw_k is not None else r.k
        beta = r.beta if r.beta is not None else r.g
        if (
            rawv.shape[1:] != (hv, capacity, dim)
            or rawk.shape[1:] != (h, capacity, dim)
            or beta.shape[1:] != (hv, capacity)
            or any(not t.is_contiguous() for t in (rawv, rawk, beta))
        ):
            raise ValueError("GDN verify glue raw ring differs from source layout")
        w2 = io.conv_weight.view(channels, 4)
        tensors = (
            io.mixed_qkvz,
            io.mixed_ba,
            io.conv_state,
            w2,
            w2,
            io.intermediate_conv_window_out,
            io.intermediate_indices,
            io.a_log,
            io.dt_bias,
            io.norm_weight,
            io.output,
            io.checkpoint,
            r.d,
            r.k,
            r.g,
            rawv,
            rawk,
            beta,
            r.state_indices,
            r.replay_indices,
            r.write_pos,
            r.cache_base,
            r.is_flush,
        )
        key = (
            io.mixed_qkvz.device.index,
            int(io.norm_activation),
            int(io.null_slot),
            store_exact,
            store_residual,
            # Triton specializes pointer divisibility, not the activation address.
            tuple(
                (tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() % 16)
                for t in tensors
            ),
        )
        if key not in self._warmed and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GDN verify glue exact key must be warmed before capture"
            )
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_fused_verify_glue import (
            gdn_fused_verify_glue_kernel,
        )

        gdn_fused_verify_glue_kernel[(m // 4, hv)](
            *tensors,
            dim**-0.5,
            io.norm_epsilon,
            io.mixed_qkvz.stride(0),
            io.mixed_ba.stride(0),
            io.output.stride(0),
            *io.conv_state.stride(),
            *w2.stride(),
            *io.intermediate_conv_window_out.stride(),
            io.checkpoint.stride(0),
            r.d.stride(0),
            r.k.stride(0),
            r.g.stride(0),
            rawv.stride(0),
            rawk.stride(0),
            beta.stride(0),
            T=4,
            H=h,
            HV=hv,
            K=dim,
            V=dim,
            BS=4,
            BC=max(16, capacity),
            MAX_CACHE_LEN=capacity,
            STATE_LEN=3,
            SOFTPLUS_THRESHOLD=20.0,
            STATE_FP32=False,
            STORE_EXACT_RAW=store_exact,
            STORE_D_RESIDUAL=store_residual,
            STORE_K_RESIDUAL=store_residual,
            HAS_CONV_BIAS=False,
            CONV_SILU=True,
            NORM_ACT=io.norm_activation,
            NULL_BLOCK_ID=io.null_slot,
            DOT_PRECISION="tf32",
            PREFETCH=True,
            HIST_BRANCH=True,
            PDL_TRIGGER_EARLY=True,
            num_warps=4,
            num_stages=1,
            USE_GDC=True,
            launch_pdl=True,
        )
        self._warmed.add(key)
