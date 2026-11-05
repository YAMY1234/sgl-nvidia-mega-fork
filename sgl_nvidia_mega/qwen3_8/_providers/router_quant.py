# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang's q38_router_fused.py (Apache-2.0), caller-buffer ABI only.
"""Source router GEMM and NVFP4 quantization into framework-owned buffers."""

import hashlib
from enum import IntEnum
from functools import cache

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericOperation,
    RouterQuantIO,
    RouterQuantSignature,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "NumericOperation",
        "RouterQuantIO",
        "RouterQuantSignature",
        "ScalarType",
    ),
}


def supported(signature):
    if not isinstance(signature, RouterQuantSignature):
        return False
    values = (
        signature.operation,
        signature.rows,
        signature.hidden_size,
        signature.expert_count,
        signature.scalar_type,
        signature.variant,
    )
    if any(
        type(value) is not int and not isinstance(value, IntEnum) for value in values
    ):
        return False
    return (
        signature.operation == NumericOperation.ROUTER_QUANT
        and signature.hidden_size == 2560
        and signature.expert_count == 512
        and signature.scalar_type == ScalarType.BF16
        and (
            (signature.variant in (0, 1) and signature.rows in (1, 4))
            or (signature.variant == 2 and signature.rows in (8, 16))
        )
    )


@cache
def _bundle(device_index, variant):
    import torch

    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_jit as jit
    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_moe_aot import bundle_manifest
    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_router_front import _SOURCE

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("router quant AOT must load before capture")
    variable, split, capacities, source = (
        ("SGLANG_Q38_ROUTER_AOT", 2, (1, 8), _SOURCE),
        ("SGLANG_Q38_ROUTER_K4_AOT", 4, (1, 8), _SOURCE),
        (
            "SGLANG_Q38_ROUTER_WIDE_AOT",
            2,
            (8, 16),
            _SOURCE.with_name("q38_router_wide.cu"),
        ),
    )[variant]
    root, manifest = bundle_manifest(
        variable,
        torch.cuda.get_device_capability(device_index),
    )
    if manifest["source_sha256"] != hashlib.sha256(
        source.read_bytes()
    ).hexdigest() or manifest["shape"] != [2560, 512, split]:
        raise ValueError("router quant AOT source/shape differs from exact variant")
    kernels = {}
    for capacity in capacities:
        name = f"route-m{capacity}.cubin"
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["sha256"][name]:
            raise ValueError("router quant AOT cubin SHA differs from manifest")
        shared = (8 + capacity) * (2560 * 2 + 16) + split * 16 * 8 * 4 + 16
        kernel = jit.load(data, "q38_router_gemm_quant", (256, 1, 1), shared)
        cu = jit.driver()
        # M1 also needs the source's static+dynamic shared-memory opt-in.
        jit.check(
            cu.cuFuncSetAttribute(
                kernel.function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ),
            "router quant dynamic shared opt-in",
        )
        kernels[capacity] = kernel
    return kernels


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported router quant signature")
    return _RouterQuant(signature)


class _RouterQuant:
    def __init__(self, signature):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("router quant must bind before capture")
        self.signature = signature
        self.device_index = torch.cuda.current_device()
        # Device queries/lazy init do not guarantee a current primary context.
        # Bind the already selected rank device before direct driver module load.
        torch.cuda.set_device(self.device_index)
        kernels = _bundle(self.device_index, signature.variant)
        capacity = 1 if signature.rows == 1 else 8 if signature.rows <= 8 else 16
        self.kernel = kernels[capacity]

    def __call__(self, io):
        import torch

        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_jit as jit
        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_router_front import Params

        if not isinstance(io, RouterQuantIO):
            raise TypeError("router quant requires its exact tensor view")
        tensors = (
            io.hidden_states,
            io.router_weight,
            io.global_scale,
            io.logits_out,
            io.packed_out,
            io.scale_out,
        )
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("router quant requires six tensors")
        m = self.signature.rows
        shapes = ((m, 2560), (512, 2560), (1,), (m, 512), (m, 1280), (m, 160))
        dtypes = (
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        alignments = (16, 16, 4, 2, 8, 1)
        pointers = tuple(t.data_ptr() for t in tensors)
        device = io.hidden_states.device
        if (
            device.type != "cuda"
            or device.index != self.device_index
            or any(
                t.device != device
                or t.shape != shape
                or t.dtype != dtype
                or not t.is_contiguous()
                or pointer % alignment
                for t, shape, dtype, alignment, pointer in zip(
                    tensors, shapes, dtypes, alignments, pointers
                )
            )
        ):
            raise ValueError("router quant view differs from its exact source ABI")
        # The original producer allocated distinct outputs. Preserve borrowed
        # input bytes and avoid two output stores aliasing the same allocation.
        ranges = tuple(
            (p, p + t.numel() * t.element_size()) for t, p in zip(tensors, pointers)
        )
        for i in range(3, 6):
            for lo, hi in ranges[:i]:
                if ranges[i][0] < hi and lo < ranges[i][1]:
                    raise ValueError(
                        "router quant outputs must not alias borrowed buffers"
                    )
        params = Params(
            x=pointers[0],
            weight=pointers[1],
            global_scale=pointers[2],
            logits=pointers[3],
            x_fp4=pointers[4],
            x_sf=pointers[5],
            num_tokens=m,
            mode=1,
        )
        jit.launch(
            self.kernel,
            (64, 1, 1),
            [params],
            torch.cuda.current_stream(device).cuda_stream,
        )
