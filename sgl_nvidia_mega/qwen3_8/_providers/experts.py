# SPDX-License-Identifier: Apache-2.0
"""Local expert math with caller outputs and signature-selected launch variants.

The framework resolves eligibility and prepares the finite signatures. Bit 0
selects narrow scale-factor packing; bit 1 selects the prepared batch3/4 wide
path. No batch metadata, stream joins, or communication cross this boundary.
"""

from functools import partial

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "ExpertSignature",
        "NumericOperation",
        "ScalarType",
    )
}


def supported(signature):
    if not isinstance(signature, ExpertSignature) or signature.rows <= 0:
        return False
    if signature.operation == NumericOperation.FINALIZE_SHARED:
        return (
            signature.hidden_size == 2560
            and signature.topk == 10
            and signature.scalar_type == ScalarType.BF16
            and signature.variant == 0
        )
    if signature.operation != NumericOperation.LOCAL_EXPERTS:
        return False
    if (
        signature.hidden_size != 2560
        or signature.intermediate_size != 192
        or signature.experts != 512
        or signature.topk != 10
        or signature.scalar_type != ScalarType.PACKED_FP4
    ):
        return False
    if signature.variant == 0:
        return signature.rows in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)
    if signature.variant == 1:
        return signature.rows in (1, 4)
    if signature.variant == 2:
        return signature.rows in (12, 16)
    return False


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported local expert signature")
    return (
        LocalExperts(signature)
        if signature.operation == NumericOperation.LOCAL_EXPERTS
        else finalize_shared
    )


def finalize_shared(io):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_finalize_shared import (
        finalize_shared as launch,
    )

    launch(
        io.hidden,
        io.gate_weight,
        io.shared,
        io.expert_rows,
        io.route_weights,
        io.output,
        cols=io.cols,
        use_pdl=io.use_pdl,
    )


class LocalExperts:
    """Prepare bundles/weight maps eagerly; bind activation maps on the host."""

    def __init__(self, signature):
        self.signature = signature
        self._resources = {}

    def _resource_key(self, io):
        return (
            io.packed_w13.device.index,
            io.packed_w13.data_ptr(),
            io.packed_w2.data_ptr(),
            self.signature.rows,
            self.signature.variant,
        )

    def prepare(self, io):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "local expert descriptors must be prepared before capture"
            )
        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_mk_q35 as narrow
        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_mkm_wide as wide

        module = narrow if self.signature.rows <= narrow.MAX_M else wide
        if module is narrow:
            load = (
                narrow._sf_pack_bundle
                if self.signature.variant == 1
                else narrow._bundle
            )
            bundle = load(io.packed_x.device.index)
        else:
            bundle = wide.bundle(io.packed_x.device.index, self.signature.rows == 8)
        maps = module.maps(io.packed_w13, io.packed_w2, io.packed_x)
        # Validate the original complete layout above during eager preparation.
        # Capture's activation lives in a different allocation from eager warmup.
        # Keep only weights and a numeric X-map template, never packed_x itself.
        swizzle = module.jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        bind_x = partial(
            module.tensor_map,
            rows=self.signature.rows,
            row_bytes=self.signature.hidden_size // 2,
            box_bytes=128,
            box_rows=8 if module is narrow else 16,
            swizzle=swizzle,
        )
        self._resources[self._resource_key(io)] = (
            module,
            bundle,
            maps[:2],
            bind_x,
            (io.packed_w13, io.packed_w2),
        )

    def __call__(self, io):
        import torch

        signature = self.signature
        # All three raw CUDA families read this pointer as uint16_t BF16.
        # Reject a mismatched view before descriptors, launches or output writes.
        if (
            io.routing_logits.dtype != torch.bfloat16
            or io.routing_logits.shape != (signature.rows, signature.experts)
            or not io.routing_logits.is_contiguous()
        ):
            raise ValueError(
                "local expert raw ABI requires contiguous BF16 [M, E] routing logits"
            )

        from flashinfer.tllm_enums import ActivationType

        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_mk_q35 as narrow

        if (
            (io.hidden_size, io.intermediate_size, io.experts, io.topk)
            != (
                signature.hidden_size,
                signature.intermediate_size,
                signature.experts,
                signature.topk,
            )
            or io.local_expert_start != 0
            or io.local_expert_count != signature.experts
            or io.routed_scale != 1.0
            or io.activation != int(ActivationType.Swiglu)
            or io.packed_x.shape != (signature.rows, signature.hidden_size // 2)
        ):
            raise ValueError("local expert view does not match its offered layout")
        key = self._resource_key(io)
        if key not in self._resources:
            self.prepare(io)
        module, (kernel, finalize, metadata), weight_maps, bind_x, owners = (
            self._resources[key]
        )
        # Original host-only cuTensorMapEncodeTiled, including during capture:
        # no CUDA allocation, module loading, compilation or synchronization.
        # CUDA graph capture copies this call's address into the kernel arguments.
        maps = (*weight_maps, bind_x(io.packed_x))
        layer = dict(
            gemm1_weights_scale=io.scale13,
            gemm2_weights_scale=io.scale2,
            output1_scale_gate_scalar=io.global_scale1,
            output1_scale_scalar=io.global_scale1_c,
            output2_scale_scalar=io.global_scale2,
        )
        if module is narrow:
            narrow.run(
                kernel,
                maps,
                io.routing_logits,
                io.route_weights_out,
                io.route_ids_out,
                io.x_scale,
                layer,
                io.expert_rows_out,
            )
        else:
            kwargs = dict(
                layer,
                hidden_states=io.packed_x,
                hidden_states_scale=io.x_scale,
                routing_logits=io.routing_logits,
            )
            module.run(
                kernel,
                maps,
                kwargs,
                io.expert_rows_out,
                io.route_weights_out,
                io.route_ids_out,
            )
        if io.final_out is not None:
            narrow._finalize(
                finalize,
                metadata,
                io.expert_rows_out,
                io.route_weights_out,
                io.final_out,
            )
