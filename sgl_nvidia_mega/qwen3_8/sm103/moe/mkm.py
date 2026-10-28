# SPDX-License-Identifier: Apache-2.0
"""FN SF-pack NVFP4 MKM: qualified I192 storage, HF logical I160 at TP4."""

from __future__ import annotations
import ctypes
from functools import cache
from pathlib import Path
import torch
from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_8.moe.config import Shape
from sgl_nvidia_mega.qwen3_5.sm103.moe.mkm import _map

THREADS, SHARED, CLUSTER = 256, 216064, 2
SOURCE = Path(__file__).parent / "src/mkm_moe_tp4.cu"


class Params(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_void_p)
        for n in (
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


def options(shape: Shape):
    return (
        "-std=c++17",
        "-default-device",
        f"-DQ35_MKM_SMEM_BYTES={SHARED}",
        "-DPDL_TRIGGER=0",
        *shape.compile_options(),
    )


@cache
def _kernel(device_index: int, shape: Shape):
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(), "q38_mkm_moe_tp4", f"sm_{major}{minor}a", options(shape)
    )
    return jit.load(cubin, "q38_mkm_moe_tp4", (THREADS, 1, 1), SHARED)


def build(device_index: int, shape: Shape):
    _kernel(device_index, shape)


def weight_maps(w13, w2, shape: Shape):
    h, e, i = shape.hidden, shape.experts, shape.storage_intermediate
    sw = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    return (
        _map(w13, e * 2 * i, h // 2, 128, 128, sw),
        # Storage I192 is96B per row; the128B SW128 box zero-fills its K tail.
        _map(w2, e * h, i // 2, 128, 256, sw),
    )


def mkm_moe(
    maps,
    logits,
    x_fp4,
    x_sf,
    w13_sf,
    w2_sf,
    g1_alpha,
    g1_scale_c,
    g2_alpha,
    *,
    shape: Shape,
    return_ids=False,
):
    m = logits.shape[0]
    if not 1 <= m <= shape.mkm_capacity:
        raise ValueError(
            f"FN mkm: {m} rows outside current1..{shape.mkm_capacity} engine"
        )
    e, h, k = shape.experts, shape.hidden, shape.top_k
    out = torch.empty((m * k, h), dtype=torch.bfloat16, device=logits.device)
    weights = torch.empty((m, k), dtype=torch.bfloat16, device=logits.device)
    ids = torch.empty((m, k), dtype=torch.int32, device=logits.device)
    sw = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    xmap = _map(x_fp4, m, h // 2, 128, 8, sw)
    p = Params(
        logits=logits.data_ptr(),
        expert_weights=weights.data_ptr(),
        topk_ids=ids.data_ptr(),
        x_sf=x_sf.data_ptr(),
        w13_sf=w13_sf.data_ptr(),
        w2_sf=w2_sf.data_ptr(),
        g1_alpha=g1_alpha.data_ptr(),
        g1_scale_c=g1_scale_c.data_ptr(),
        g2_alpha=g2_alpha.data_ptr(),
        out=out.data_ptr(),
        num_tokens=m,
    )
    jit.launch(
        _kernel(logits.device.index, shape),
        (2 * shape.mkm_capacity * k, 1, 1),
        [*maps, xmap, p],
        torch.cuda.current_stream().cuda_stream,
        cluster=(CLUSTER, 1, 1),
    )
    return (out, weights, ids) if return_ids else (out, weights)
