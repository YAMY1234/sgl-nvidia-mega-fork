# SPDX-License-Identifier: Apache-2.0

"""AOT-only Rubin routed engine for M8/M12/M16; physical intermediate192."""

import hashlib
import logging
import os
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_jit as jit
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mk_moe import tensor_map
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mk_q35 import (
    Params,
    _bundle,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_moe_aot import bundle_manifest

H, I, E, TOPK = 2560, 192, 512, 10
SHARED, THREADS = 215040, 256
logger = logging.getLogger(__name__)
_seen = set()


@cache
def bundle(device_index, m8=False):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("wide MoE AOT must load during eager warmup")
    variable = (
        "SGLANG_Q38_MKM_WIDE_M8_AOT"
        if m8 and os.environ.get("SGLANG_Q38_MKM_WIDE_M8_AOT")
        else "SGLANG_Q38_MKM_WIDE_AOT"
    )
    root, manifest = bundle_manifest(
        variable, torch.cuda.get_device_capability(device_index)
    )
    source = Path(__file__).resolve().parents[2] / "jit/csrc/moe/q38_mkm_wide.cu"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["shape"] == [H, I, E, TOPK, 16]
    assert manifest["shared"] == SHARED and manifest["threads"] == THREADS
    cubin = root / "wide.cubin"
    assert manifest["cubin_sha256"] == hashlib.sha256(cubin.read_bytes()).hexdigest()
    props = torch.cuda.get_device_properties(device_index)
    assert SHARED <= props.shared_memory_per_block_optin
    expert = jit.load(cubin.read_bytes(), "q38_mkm_wide", (THREADS, 1, 1), SHARED)
    _, finalize, metadata = _bundle(device_index)
    return (
        expert,
        finalize,
        dict(metadata, expert_spec_stages=manifest.get("spec_stages", 4)),
    )


def maps(w13, w2, xq):
    swizzle = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    assert tuple(w13.shape) == (E, 2 * I, H // 2)
    assert tuple(w2.shape) == (E, H, I // 2)
    return (
        tensor_map(w13, E * 2 * I, H // 2, 128, 128, swizzle),
        tensor_map(w2, E * H, I // 2, 128, 256, swizzle),
        tensor_map(xq, xq.shape[0], H // 2, 128, 16, swizzle),
    )


def run(kernel, tensor_maps, kwargs, rows, weights, ids):
    m = kwargs["hidden_states"].shape[0]
    assert m in (8, 12, 16)
    tensors = (
        kwargs["routing_logits"],
        weights,
        ids,
        kwargs["hidden_states_scale"],
        kwargs["gemm1_weights_scale"],
        kwargs["gemm2_weights_scale"],
        kwargs["output1_scale_gate_scalar"],
        kwargs["output1_scale_scalar"],
        kwargs["output2_scale_scalar"],
        rows,
    )
    params = Params(*(tensor.data_ptr() for tensor in tensors), 0, m)
    jit.launch(
        kernel,
        (m * TOPK, 1, 1),
        [*tensor_maps, params],
        torch.cuda.current_stream().cuda_stream,
    )
    return rows
