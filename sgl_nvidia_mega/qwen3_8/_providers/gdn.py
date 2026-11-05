# SPDX-License-Identifier: Apache-2.0
"""Local GDN math into caller buffers; source admission stays in the framework."""

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": ("GdnSignature", "GdnCoreIO"),
}


def build_core(signature):
    """Variant0 raw; variant1 BF16 fused RMS with the source sigmoid(z)."""
    if signature.variant not in (0, 1):
        raise ValueError("unsupported GDN core variant")
    return GdnCore(signature)


class GdnCore:
    def __init__(self, signature):
        self.signature = signature

    def __call__(self, io):
        import torch

        from .local import gdn_core

        signature = self.signature
        fused = signature.variant == 1
        rows, heads, dim = signature.rows, signature.value_heads, signature.value_dim
        if (
            io.value_rows != 128
            or io.qkv.shape != (rows, (2 * signature.key_heads + heads) * dim)
            or io.a.shape != (rows, heads)
            or io.b.shape != (rows, heads)
            or io.raw_out.shape != (rows, heads, dim)
            or io.checkpoint.ndim != 4
            or io.checkpoint.shape[1:] != (heads, dim, signature.key_dim)
            or any(
                t.dtype != torch.bfloat16
                for t in (io.qkv, io.a, io.b, io.checkpoint, io.raw_out)
            )
            or not io.raw_out.is_contiguous()
        ):
            raise ValueError("GDN core views differ from the bound BF16 layout")
        norm_fields = (io.z, io.norm_weight, io.normalized_out)
        if not fused:
            if any(t is not None for t in norm_fields):
                raise ValueError("raw GDN capability cannot change to fused norm")
        else:
            if any(t is None for t in norm_fields):
                raise ValueError("fused GDN requires complete caller norm buffers")
            if (
                io.z.shape != (rows, heads, dim)
                or io.norm_weight.shape != (dim,)
                or io.normalized_out.shape != (rows, heads, dim)
                or any(t.dtype != torch.bfloat16 for t in norm_fields)
                or any(t.device != io.qkv.device for t in norm_fields)
                or io.z.stride(-1) != 1
                or not io.norm_weight.is_contiguous()
                or not io.normalized_out.is_contiguous()
            ):
                raise ValueError(
                    "fused GDN norm views differ from the bound BF16 layout"
                )
        # The original raw helper owns exact eager compilation/capture cache hits.
        # No new state, weights, scratch, dtype conversion or synchronization.
        gdn_core(io)
