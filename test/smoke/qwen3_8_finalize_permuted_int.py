# SPDX-License-Identifier: Apache-2.0
"""Integer-F raw ABI smoke via the real registry; not facade/model qualification.

Separate successor; the original bool-F smoke file/commit is unchanged.

Run after installing the exact F canonical binding, with the source default
EXTEND-off policy. The caller owns every buffer, including the output. No
allocating framework wrapper is called by the provider.
"""

import argparse
import json
from itertools import product

import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    FinalizePermutedSharedIO,
    FinalizePermutedSignature,
    NumericOperation,
    ScalarType,
)


def signature(rows, weight_dtype, num_warps=16, use_pdl=1):
    return FinalizePermutedSignature(
        NumericOperation.FINALIZE_PERMUTED_SHARED,
        rows,
        2560,
        10,
        ScalarType.BF16,
        ScalarType.BF16 if weight_dtype == torch.bfloat16 else ScalarType.FP32,
        num_warps=num_warps,
        use_pdl=use_pdl,
    )


def make_io(rows, weight_dtype, device, row_padding=0):
    h, k = 2560, 10
    hidden = torch.randn((rows, h), dtype=torch.bfloat16, device=device)
    gate = torch.randn((h,), dtype=torch.bfloat16, device=device)
    shared = torch.randn_like(hidden)
    # Preserve actual native row pitch; the first H columns are read.
    gemm2 = torch.randn(
        (rows * k, h + row_padding), dtype=torch.bfloat16, device=device
    )[:, :h]
    permutation = torch.arange(rows * k, dtype=torch.int32, device=device)
    permutation[::7] = -1  # source negative-map skip path
    weights = torch.randn((rows, k), dtype=weight_dtype, device=device)
    output = torch.empty_like(hidden)
    return FinalizePermutedSharedIO(
        hidden, gate, shared, gemm2, permutation, weights, output
    )


def run(device=0):
    from sgl_nvidia_mega.qwen3_8 import providers

    torch.cuda.set_device(device)
    records = []
    for rows, dtype, warps, pdl in product(
        (1, 4, 8, 16, 32, 64), (torch.bfloat16, torch.float32), (8, 16), (0, 1)
    ):
        s = signature(rows, dtype, warps, pdl)
        io = make_io(rows, dtype, f"cuda:{device}")
        offers = providers.register_signatures((s,))
        assert len(offers) == 1
        assert providers.numeric_registry.get(s) is offers[0]
        offers[0].run(io)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            offers[0].run(io)
        graph.replay()
        torch.cuda.synchronize()
        records.append(
            {
                "operation": s.operation.name,
                "rows": rows,
                "weight_dtype": str(dtype),
                "num_warps": warps,
                "use_pdl": pdl,
                "gemm2_stride": list(io.gemm2_out.stride()),
                "output_ptr": io.output.data_ptr(),
                "eager_capture_replay": True,
            }
        )
    assert len(records) == 48
    return {
        "records": records,
        "model_qualification": False,
        "math_qualification": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.device), sort_keys=True))
