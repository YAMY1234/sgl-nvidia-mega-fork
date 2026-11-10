# SPDX-License-Identifier: Apache-2.0
"""Native FI permutation finalize and ungated shared add into a caller output."""

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    FinalizePermutedSharedIO,
    FinalizePermutedSignature,
    NumericOperation,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "FinalizePermutedSharedIO",
        "FinalizePermutedSignature",
        "NumericOperation",
        "ScalarType",
    )
}


def supported(signature):
    return (
        isinstance(signature, FinalizePermutedSignature)
        and signature.operation == NumericOperation.FINALIZE_PERMUTED_SHARED
        and 0 <= signature.rows <= 1024
        and signature.hidden_size == 2560
        and signature.topk == 10
        and signature.scalar_type == ScalarType.BF16
        and signature.weight_scalar_type in (ScalarType.BF16, ScalarType.FP32)
        and signature.num_warps in (8, 16)
        and type(signature.use_pdl) is int
        and signature.use_pdl in (0, 1)
        and signature.variant == 0
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported native permuted finalize signature")
    return _FinalizePermuted(signature)


class _FinalizePermuted:
    def __init__(self, signature):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("native permuted finalize must bind before capture")
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.elementwise import (
            _fused_finalize_gate_sigmoid_mul_add_kernel,
        )

        self.signature = signature
        self._kernel = _fused_finalize_gate_sigmoid_mul_add_kernel
        self._weight_dtype = (
            torch.bfloat16
            if signature.weight_scalar_type == ScalarType.BF16
            else torch.float32
        )
        self._warmed_abis = set()

    def __call__(self, io):
        import torch

        if not isinstance(io, FinalizePermutedSharedIO):
            raise TypeError("native permuted finalize requires its bound tensor view")
        tensors = (
            io.hidden,
            io.gate_weight,
            io.shared,
            io.gemm2_out,
            io.expanded_idx_to_permuted_idx,
            io.expert_weights,
            io.output,
        )
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("native permuted finalize requires seven tensors")
        s = self.signature
        # Rows 0 is declared for every call above the graph widths.
        m, h, k = s.rows or io.hidden.shape[0], s.hidden_size, s.topk
        device = io.hidden.device
        if (
            device.type != "cuda"
            or any(tensor.device != device for tensor in tensors)
            or any(
                tensor.dtype != torch.bfloat16
                for tensor in (
                    io.hidden,
                    io.gate_weight,
                    io.shared,
                    io.gemm2_out,
                    io.output,
                )
            )
            or io.expert_weights.dtype != self._weight_dtype
            or io.expanded_idx_to_permuted_idx.dtype != torch.int32
            or io.hidden.shape != (m, h)
            or io.gate_weight.shape != (h,)
            or io.shared.shape != (m, h)
            or io.output.shape != (m, h)
            or io.expert_weights.shape != (m, k)
            or io.expanded_idx_to_permuted_idx.numel() != m * k
            or any(
                not tensor.is_contiguous()
                for tensor in (
                    io.hidden,
                    io.gate_weight,
                    io.shared,
                    io.expanded_idx_to_permuted_idx,
                    io.expert_weights,
                    io.output,
                )
            )
            or io.gemm2_out.ndim != 2
            or io.gemm2_out.shape[1] < h
            or io.gemm2_out.stride(1) != 1
            or io.gemm2_out.stride(0) <= 0
        ):
            raise ValueError(
                "native permuted finalize view does not match its offered layout"
            )
        # Only the raw launch ABI participates: no FI allocation/cache policy.
        # Triton specializes pointer alignment and the scalar row stride. A new
        # ABI must be warmed eagerly even if another signature/layout is cached.
        abi = (
            device,
            io.gemm2_out.stride(0),
            tuple(
                (tensor.data_ptr() % 16, tensor.data_ptr() == 0) for tensor in tensors
            ),
        )
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and abi not in self._warmed_abis:
            raise RuntimeError(
                "native permuted finalize ABI must warm eagerly before capture"
            )
        pdl = {"launch_pdl": True} if s.use_pdl else {}
        self._kernel[(m,)](
            io.hidden,
            io.gate_weight,
            io.shared,
            io.output,
            io.gemm2_out,
            io.expanded_idx_to_permuted_idx,
            io.expert_weights,
            io.gemm2_out.stride(0),
            hidden_dim=h,
            TOP_K=k,
            BLOCK_SIZE=4096,
            USE_PDL=bool(s.use_pdl),
            num_warps=s.num_warps,
            **pdl,
        )
        if not capturing:
            self._warmed_abis.add(abi)
