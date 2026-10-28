# SPDX-License-Identifier: Apache-2.0
"""FN ae62 router/quant producer. Logit parity and final consumer parity are tested separately."""

from __future__ import annotations
import ctypes
from functools import cache
from pathlib import Path
import torch
from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_8.moe.config import Shape

SOURCE = Path(__file__).parent / "src/router_front.cu"
THREADS = 256


class Params(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "x",
            "weight",
            "global_scale",
            "logits",
            "topk_weights",
            "topk_ids",
            "x_fp4",
            "x_sf",
            "tile_idx_to_expert_idx",
            "tile_idx_to_mn_limit",
            "expanded_idx_to_permuted_idx",
            "permuted_idx_to_expanded_idx",
            "total_num_padded_tokens",
            "num_non_exiting_tiles",
            "arrival",
            "trace",
        )
    ] + [
        ("num_tokens", ctypes.c_uint32),
        ("tile_log2", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("sh_weight", ctypes.c_void_p),
        ("sh_in_scale", ctypes.c_void_p),
        ("sh_w_scale", ctypes.c_void_p),
        ("sh_out", ctypes.c_void_p),
        ("sh_n", ctypes.c_uint32),
    ]


def max_m_bucket(m):
    return 1 if m == 1 else 8 if m <= 8 else 16


def shared_bytes(shape: Shape, capacity):
    return (8 + capacity) * (shape.hidden * 2 + 16) + 2 * 16 * 8 * 4 + 16


def options(shape: Shape, capacity):
    return (
        "-std=c++17",
        "-default-device",
        f"-DQ35_ROUTER_MAX_M={capacity}",
        "-DQ35_ROUTER_KSPLIT=2",
        *shape.compile_options(),
    )


@cache
def _kernel(device_index, shape: Shape, capacity):
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        f"q38_router_m{capacity}",
        f"sm_{major}{minor}a",
        options(shape, capacity),
    )
    shared = shared_bytes(shape, capacity)
    loaded = jit.load(cubin, "q38_router_gemm_quant", (THREADS, 1, 1), shared)
    # M1 dynamic<48KiB but static+dynamic exceeds the default budget.
    cu = jit.driver()
    jit.check(
        cu.cuFuncSetAttribute(
            loaded.function,
            cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared,
        ),
        "FN router shared opt-in",
    )
    return loaded


def build(device_index, shape: Shape):
    for capacity in sorted({max_m_bucket(m) for m in range(1, shape.max_rows + 1)}):
        _kernel(device_index, shape, capacity)


def router_gemm_quant(x, weight, global_scale, *, shape: Shape):
    m = x.shape[0]
    h, e = shape.hidden, shape.experts
    if not 1 <= m <= 16:
        raise ValueError(f"FN router: {m} rows outside1..16")
    logits = torch.empty((m, e), dtype=torch.bfloat16, device=x.device)
    packed = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m, h // 16), dtype=torch.uint8, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=packed.data_ptr(),
        x_sf=sf.data_ptr(),
        num_tokens=m,
        mode=1,
    )
    jit.launch(
        _kernel(x.device.index, shape, max_m_bucket(m)),
        (e // 8, 1, 1),
        [p],
        torch.cuda.current_stream().cuda_stream,
    )
    return logits, packed, sf
