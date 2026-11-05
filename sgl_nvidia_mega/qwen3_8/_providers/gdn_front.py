# SPDX-License-Identifier: Apache-2.0
"""Caller-buffer launches of the original finite GDN projection/conv families."""

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnActivation,
    GdnFrontVariant,
    GdnSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "GdnActivation",
        "GdnFrontVariant",
        "GdnSignature",
        "GdnFrontIO",
        "NumericOperation",
        "ScalarType",
    ),
}

_FAMILIES = {
    GdnFrontVariant.MK_STAGES_6: ("mk", 6),
    GdnFrontVariant.MK_STAGES_12: ("mk", 12),
    GdnFrontVariant.TGV_STAGES_8: ("legacy", 8),
    GdnFrontVariant.TGV_STAGES_12: ("legacy", 12),
}


def supported(s):
    if not isinstance(s, GdnSignature) or (
        s.operation != NumericOperation.GDN_FRONT
        or s.key_dim != 128
        or s.value_dim != 128
        or s.tokens_per_request != 4
        or s.scalar_type != ScalarType.BF16
        # FRONT has no gate tensor: canonical3 is a neutral unused numeric field.
        or s.gate_scalar_type != 3
    ):
        return False
    if s.variant == GdnFrontVariant.MK_STAGES_6:
        return (
            s.rows in (4, 8, 12) and (s.key_heads, s.value_heads) in ((4, 12), (16, 48))
        ) or (s.rows, s.key_heads, s.value_heads) == (16, 4, 12)
    if s.variant == GdnFrontVariant.MK_STAGES_12:
        return (s.rows, s.key_heads, s.value_heads) == (4, 4, 12)
    if s.variant in (GdnFrontVariant.TGV_STAGES_8, GdnFrontVariant.TGV_STAGES_12):
        return (s.rows, s.key_heads, s.value_heads) == (4, 16, 48)
    return False


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported GDN front signature")
    return GdnFront(signature)


class GdnFront:
    def __init__(self, signature):
        if not supported(signature):
            raise ValueError("unsupported GDN front signature")
        self.signature = signature
        self.family, self.stages = _FAMILIES[signature.variant]
        self._resources = {}

    def __call__(self, io):
        import torch

        s = self.signature
        rows, channels = s.rows, (2 * s.key_heads + s.value_heads) * s.key_dim
        packed_width = channels + s.value_heads * s.value_dim + 2 * s.value_heads
        if (
            io.packed_out is None
            or io.packed_weight is None
            or io.activation != GdnActivation.SILU
            or io.window_mode != 0
            or io.token_major_out is not True
            or io.launch.num_stages != self.stages
            # These are canonical neutral fields, not CuTe launch tuning knobs.
            # Original factory threads and internal PDL signaling stay unchanged.
            or (
                io.launch.block_n,
                io.launch.num_warps,
                io.launch.hoist,
                io.launch.dot_precision,
                io.launch.prefetch,
                io.launch.historical_branch,
                io.launch.use_pdl,
                io.launch.trigger_early,
            )
            != (0, 0, False, 0, False, 0, True, False)
            or s.key_dim != 128
            or s.value_dim != 128
            or s.tokens_per_request != 4
            or (s.key_heads, s.value_heads) not in ((4, 12), (16, 48))
            or (
                rows not in (4, 8, 12)
                and not (self.family == "mk" and rows == 16 and channels == 2560)
            )
            or (self.family == "legacy" and (rows, channels) != (4, 10240))
            or io.x.shape != (rows, 2560)
            or io.packed_weight.shape != (packed_width, 2560)
            or io.conv_weight.shape != (channels, 4)
            or io.conv_state.ndim != 3
            or io.conv_state.shape[1] != channels
            or io.conv_state.shape[2] < 3
            or io.intermediate_conv_window_out.shape[1:] != (4, channels, 3)
            or io.packed_out.shape != (rows, packed_width)
            or io.convolved_out.shape != (rows, channels)
            or not io.packed_out.is_contiguous()
            or not io.convolved_out.is_contiguous()
            or (
                io.conv_bias is not None
                and (
                    io.conv_bias.shape != (channels,)
                    or io.conv_bias.dtype != torch.bfloat16
                )
            )
            or io.state_indices.numel() < rows // 4
            or io.intermediate_indices.numel() < rows // 4
            or any(
                t.dtype != torch.bfloat16
                for t in (
                    io.x,
                    io.packed_weight,
                    io.conv_weight,
                    io.conv_state,
                    io.intermediate_conv_window_out,
                    io.packed_out,
                    io.convolved_out,
                )
            )
        ):
            raise ValueError("GDN front view differs from its prepared source layout")
        # Only metadata views: no CUDA allocation/cast/copy/contiguous or sync.
        a = io.packed_weight.unsqueeze(-1)
        b = io.x.unsqueeze(-1)
        c = io.packed_out.t().unsqueeze(-1)
        bias = io.conv_weight[:, 0] if io.conv_bias is None else io.conv_bias
        tensors = (
            a,
            b,
            c,
            c,
            io.conv_weight,
            bias,
            io.conv_state,
            io.state_indices,
            io.intermediate_conv_window_out,
            io.intermediate_indices,
            io.convolved_out,
        )
        key = (
            io.x.device.index,
            self.family,
            self.stages,
            io.conv_bias is not None,
            int(io.pad_slot),
            tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
        )
        resource = self._resources.get(key)
        cold = resource is None
        if cold:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GDN front exact layout must be warmed before capture"
                )
            if self.family == "mk":
                from sgl_nvidia_mega.qwen3_8.gdn import q38_mkgdn_front as module

                factory = module.MkGdnFront
            else:
                from sgl_nvidia_mega.qwen3_8.gdn import gdn_inproj_conv as module

                factory = module.TgvGemmCuteExtKernel
            op = factory(
                cta_m=64,
                cta_n=8 if rows == 4 else 16,
                cta_k=128,
                num_ab_stage=self.stages,
                use_2cta=False,
                use_pdl=True,
                has_bias=False,
                out_dtype=module.cutlass.BFloat16,
            )
            if self.family == "mk":
                op.qkv_channels = channels
                op.verify_rows = rows
            op.conv_has_bias = io.conv_bias is not None
            op.pad_slot = int(io.pad_slot)
            args = tuple(
                module._view(t, 16 if i < 4 else 2) for i, t in enumerate(tensors)
            )
            stream = module.cuda.CUstream(
                torch.cuda.current_stream(io.x.device).cuda_stream
            )
            compiled = module.cute_ext.compile(op, *args, stream)
            resource = (compiled, module._view, module.cuda.CUstream)
        else:
            compiled, as_view, as_stream = resource
            args = tuple(as_view(t, 16 if i < 4 else 2) for i, t in enumerate(tensors))
            stream = as_stream(torch.cuda.current_stream(io.x.device).cuda_stream)
        compiled(*args, stream)
        if cold:
            # A successful eager launch establishes this exact replay ABI.
            self._resources[key] = resource
