# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed experts of up to 7 rows in one launch (``src/mkm_moe.cu``), on the router
front's outputs: the top-10 of the logits, FC1, SwiGLU, the NVFP4 requantization of the
intermediate and FC2, reading the trtllm-gen weight layout in place. It writes the
deferred-finalize rows (row ``t * 10 + k`` is expert ``k`` of token ``t``) and the
expert weights, as the trtllm runner's deferred finalize takes them.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "mkm_moe.cu"
HIDDEN, EXPERTS, TOP_K, MAX_TOKENS = 4096, 512, 10, 7
W13_ROWS = 256  # an expert's gate and up rows at TP8 (intermediate 128 per rank)
W13_BOX_ROWS = 128  # kW13BoxRows: rows of one W13 TMA box
THREADS = 256
CTAS = (
    2 * MAX_TOKENS * TOP_K
)  # a 2-CTA cluster per selectable expert; the unused ones exit
SMEM_BYTES = 211968  # sizeof(Smem) + 1024, checked when the source compiles


class Params(ctypes.Structure):
    """qwen_mkm_moe::Params."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "logits",
            "expert_weights",
            "topk_ids",
            "x_sf",
            "w13_sf",
            "w2_sf",
            "g1_alpha",
            "g1_scale_c",
            "g2_alpha",
            "out",
            "trace",
        )
    ] + [("num_tokens", ctypes.c_int)]


@cache
def _kernel(device_index: int) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_mkm_moe",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", f"-DQ35_MKM_SMEM_BYTES={SMEM_BYTES}"),
    )
    return jit.load(cubin, "q35_mkm_moe", (THREADS, 1, 1), SMEM_BYTES)


def build(device_index: int) -> None:
    """Compile the kernel (no compile inside a step)."""
    _kernel(device_index)


def _map(
    tensor: torch.Tensor,
    rows: int,
    row_bytes: int,
    box_bytes: int,
    box_rows: int,
    swizzle,
):
    """A 2-D uint8 TMA map of ``tensor`` as [rows, row_bytes]."""
    cu = jit.driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = jit.check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            tensor.data_ptr(),
            [u64(row_bytes), u64(rows)],
            [u64(row_bytes)],
            [u32(box_bytes), u32(box_rows)],
            [u32(1), u32(1)],
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled",
    )
    return tensor_map


def weight_maps(w13: torch.Tensor, w2: torch.Tensor) -> tuple:
    """The TMA maps of a layer's W13 [512, 256, 2048] and W2 [512, 4096, 64] (uint8, the
    trtllm-gen layout); they stay valid as long as the weights."""
    swizzle = jit.driver().CUtensorMapSwizzle
    return (
        _map(
            w13,
            EXPERTS * W13_ROWS,
            HIDDEN // 2,
            128,
            W13_BOX_ROWS,
            swizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        ),
        _map(w2, EXPERTS * HIDDEN, 64, 64, 256, swizzle.CU_TENSOR_MAP_SWIZZLE_64B),
    )


def mkm_moe(
    maps: tuple,
    logits: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    w13_sf: torch.Tensor,
    w2_sf: torch.Tensor,
    g1_alpha: torch.Tensor,
    g1_scale_c: torch.Tensor,
    g2_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(gemm2_out bf16 [M * 10, 4096], expert_weights bf16 [M, 10]) for logits [M, 512]
    bf16 (M <= 7) and the router front's x_fp4 [M, 2048] / x_sf [M, 256] (uint8, linear
    e4m3); w13_sf [512, 256, 256] / w2_sf [512, 4096, 8] e4m3 viewed as uint8; g1_alpha,
    g1_scale_c, g2_alpha [512] fp32."""
    m = logits.shape[0]
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"mkm: {m} rows (1-{MAX_TOKENS})")
    device = logits.device
    gemm2_out = torch.empty((m * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device)
    expert_weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=device)
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=device)
    swizzle = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    x_map = _map(x_fp4, m, HIDDEN // 2, 128, 8, swizzle)
    p = Params(
        logits=logits.data_ptr(),
        expert_weights=expert_weights.data_ptr(),
        topk_ids=topk_ids.data_ptr(),
        x_sf=x_sf.data_ptr(),
        w13_sf=w13_sf.data_ptr(),
        w2_sf=w2_sf.data_ptr(),
        g1_alpha=g1_alpha.data_ptr(),
        g1_scale_c=g1_scale_c.data_ptr(),
        g2_alpha=g2_alpha.data_ptr(),
        out=gemm2_out.data_ptr(),
        num_tokens=m,
    )
    jit.launch(
        _kernel(device.index),
        (CTAS, 1, 1),
        [*maps, x_map, p],
        torch.cuda.current_stream().cuda_stream,
    )
    return gemm2_out, expert_weights
