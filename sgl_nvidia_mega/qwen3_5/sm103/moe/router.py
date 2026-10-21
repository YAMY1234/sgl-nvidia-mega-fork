# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The router GEMM and the NVFP4 quantization of its input in one launch, for the
trtllm MoE runner, which routes from the logits itself (``src/router_front.cu``).

The logits equal SGLang's cuBLAS router GEMM bit for bit, and the quantization equals
the runner's own NVFP4 input quantizer (16-element blocks, e4m3 scales, linear layout).
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "router_front.cu"
HIDDEN, EXPERTS, MAX_TOKENS = 4096, 512, 16
THREADS, CTAS = 256, 64  # 8 experts a CTA
_ROW_BYTES = HIDDEN * 2 + 16
_PART_BYTES = 2 * 16 * 8 * 4
MODE_QUANT = 1


def max_m_bucket(m: int) -> int:
    """The compiled row capacity that serves m rows."""
    return 1 if m <= 1 else 8 if m <= 8 else 16


def shared_bytes(max_m: int) -> int:
    """GemvSmem<max_m>::kBytes: 8 weight rows, max_m input rows, the fp32 partials and
    two mbarriers."""
    return 8 * _ROW_BYTES + max_m * _ROW_BYTES + _PART_BYTES + 16


class Params(ctypes.Structure):
    """qwen_router_front::Params."""

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


@cache
def _kernel(device_index: int, max_m: int) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        f"q35_router_front_m{max_m}",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_ROUTER_MAX_M={max_m}"),
    )
    return jit.load(
        cubin, "q35_router_gemm_quant", (THREADS, 1, 1), shared_bytes(max_m)
    )


def build(device_index: int, max_rows: int) -> None:
    """Compile every row capacity up to max_rows (no compile inside a step)."""
    for m in sorted({max_m_bucket(r) for r in range(1, max_rows + 1)}):
        _kernel(device_index, m)


def router_gemm_quant(
    x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, 4096] bf16 (M <= 16), weight [512, 4096] bf16, global_scale [1] fp32:
    (logits bf16 [M, 512], x_fp4 uint8 [M, 2048], x_sf uint8 [M, 256])."""
    m = x.shape[0]
    # the kernel reads x as packed rows (bulk copies) and serves at most 16
    if x.dtype != torch.bfloat16 or x.shape[1:] != (HIDDEN,) or not x.is_contiguous():
        raise ValueError(
            f"router front: x {x.dtype} {tuple(x.stride())} {tuple(x.shape)}"
        )
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"router front: {m} rows (1-{MAX_TOKENS})")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    p = Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=global_scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=x_fp4.data_ptr(),
        x_sf=x_sf.data_ptr(),
        num_tokens=m,
        mode=MODE_QUANT,
    )
    kernel = _kernel(x.device.index, max_m_bucket(m))
    jit.launch(kernel, (CTAS, 1, 1), [p], torch.cuda.current_stream().cuda_stream)
    return logits, x_fp4, x_sf
