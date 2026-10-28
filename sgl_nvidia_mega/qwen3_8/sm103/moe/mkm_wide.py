# SPDX-License-Identifier: Apache-2.0
"""FN 16-row NVFP4 expert engine; slot limits remain unchanged until qualified."""
from __future__ import annotations
from functools import cache
from pathlib import Path
import torch
from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_8.moe.config import Shape
from sgl_nvidia_mega.qwen3_5.sm103.moe.mkm import _map
from .mkm import Params, weight_maps as weight_maps

THREADS, SHARED, MAX_ROWS = 256, 215040, 16
SOURCE = Path(__file__).parent / 'src/mkm_moe_m16.cu'


def options(shape: Shape):
    return ('-std=c++17', '-default-device',
            f'-DQ35_MKM_SMEM_BYTES={SHARED}', *shape.compile_options())


@cache
def _kernel(device_index: int, shape: Shape):
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(SOURCE.read_text(), 'q38_mkm_moe_m16',
                               f'sm_{major}{minor}a', options(shape))
    return jit.load(cubin, 'q38_mkm_moe_m16', (THREADS, 1, 1), SHARED)


def build(device_index: int, shape: Shape):
    _kernel(device_index, shape)


def mkm_moe(maps, logits, x_fp4, x_sf, w13_sf, w2_sf,
            g1_alpha, g1_scale_c, g2_alpha, *, shape: Shape, return_ids=False):
    m = logits.shape[0]
    if not 1 <= m <= MAX_ROWS:
        raise ValueError(f'FN wide MKM: {m} rows outside1..{MAX_ROWS}')
    h, k = shape.hidden, shape.top_k
    rows = torch.empty((m * k, h), dtype=torch.bfloat16, device=logits.device)
    weights = torch.empty((m, k), dtype=torch.bfloat16, device=logits.device)
    ids = torch.empty((m, k), dtype=torch.int32, device=logits.device)
    sw = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    xmap = _map(x_fp4, m, h // 2, 128, MAX_ROWS, sw)
    p = Params(logits=logits.data_ptr(), expert_weights=weights.data_ptr(),
               topk_ids=ids.data_ptr(), x_sf=x_sf.data_ptr(),
               w13_sf=w13_sf.data_ptr(), w2_sf=w2_sf.data_ptr(),
               g1_alpha=g1_alpha.data_ptr(), g1_scale_c=g1_scale_c.data_ptr(),
               g2_alpha=g2_alpha.data_ptr(), out=rows.data_ptr(), num_tokens=m)
    jit.launch(_kernel(logits.device.index, shape), (m * k, 1, 1),
               [*maps, xmap, p], torch.cuda.current_stream().cuda_stream)
    return (rows, weights, ids) if return_ids else (rows, weights)
