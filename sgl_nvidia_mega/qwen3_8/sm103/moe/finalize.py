# SPDX-License-Identifier: Apache-2.0
"""The original FN ordered BF16 finalize, with no PDL wait or early trigger."""

import ctypes
from functools import cache
from pathlib import Path
import torch
from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src/finalize.cu"


def options(shape):
    return ("-std=c++17", "-default-device", *shape.compile_options())


@cache
def _kernel(device, shape):
    major, minor = torch.cuda.get_device_capability(device)
    cubin = jit.compile_source(
        SOURCE.read_text(), "q38_moe_finalize", f"sm_{major}{minor}a", options(shape)
    )
    return jit.load(cubin, "q38_moe_finalize", (128, 1, 1), 0)


def build(device, shape):
    _kernel(device, shape)


def run(rows, weights, *, shape):
    m = weights.shape[0]
    out = torch.empty((m, shape.hidden), dtype=torch.bfloat16, device=rows.device)
    kernel = _kernel(rows.device.index, shape)
    cu = jit.driver()
    values = [ctypes.c_void_p(t.data_ptr()) for t in (rows, weights, out)]
    pointers = (ctypes.c_void_p * 3)(*(ctypes.addressof(v) for v in values))
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = (
        (shape.hidden + 255) // 256,
        m,
        1,
    )
    config.blockDimX, config.blockDimY, config.blockDimZ = 128, 1, 1
    config.sharedMemBytes = 0
    config.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    config.attrs, config.numAttrs = [], 0
    jit.check(
        cu.cuLaunchKernelEx(config, kernel.function, ctypes.addressof(pointers), 0),
        "FN ordered finalize",
    )
    return out
