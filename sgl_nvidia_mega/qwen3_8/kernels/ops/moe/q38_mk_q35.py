# SPDX-License-Identifier: Apache-2.0

"""Original Q35 TP4 scheduling, FN shape adaptation; precompiled cubin only."""

import ctypes
import hashlib
import logging
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_jit as jit
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mk_moe import tensor_map
from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_moe_aot import bundle_manifest

H, I, E, TOPK, MAX_M = 2560, 192, 512, 10, 7
GRID, CLUSTER, THREADS, SHARED = 140, 2, 256, 216064
logger = logging.getLogger(__name__)
_seen = set()


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


def load(cubin):
    return jit.load(
        Path(cubin).read_bytes(), "q35_mkm_moe_tp4", (THREADS, 1, 1), SHARED
    )


def maps(w13, w2, xq):
    sw = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    assert tuple(w13.shape) == (E, 2 * I, H // 2)
    assert tuple(w2.shape) == (E, H, I // 2)

    return (
        tensor_map(w13, E * 2 * I, H // 2, 128, 128, sw),
        tensor_map(w2, E * H, I // 2, 128, 256, sw),
        tensor_map(xq, xq.shape[0], H // 2, 128, 8, sw),
    )


def run(kernel, tensor_maps, logits, weights, ids, xsf, layer, out):
    m = logits.shape[0]
    assert 1 <= m <= MAX_M
    tensors = (
        logits,
        weights,
        ids,
        xsf,
        layer["gemm1_weights_scale"],
        layer["gemm2_weights_scale"],
        layer["output1_scale_gate_scalar"],
        layer["output1_scale_scalar"],
        layer["output2_scale_scalar"],
        out,
    )
    params = Params(*(t.data_ptr() for t in tensors), 0, m)
    jit.launch(
        kernel,
        (GRID, 1, 1),
        [*tensor_maps, params],
        torch.cuda.current_stream().cuda_stream,
        cluster=(CLUSTER, 1, 1),
    )
    return out


@cache
def _bundle(device_index):
    """Load CPU-built artifacts once; content hashes reject stale source/binaries."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q38 MK MoE AOT bundle must load during eager graph warmup")
    root, manifest = bundle_manifest(
        "SGLANG_Q38_MK_MOE_AOT", torch.cuda.get_device_capability(device_index)
    )
    source = Path(__file__).resolve().parents[2] / "jit/csrc/moe/q38_mk_moe_q35_tp4.cu"
    finalize_source = Path(__file__).with_name("q38_mk_finalize.py")
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        manifest["finalize_source_sha256"]
        == hashlib.sha256(finalize_source.read_bytes()).hexdigest()
    )
    assert manifest["shape"] == [H, I, E, TOPK, GRID, CLUSTER, SHARED]
    for name in ("q35-port.cubin", "finalize.cubin"):
        assert (
            manifest["sha256"][name]
            == hashlib.sha256((root / name).read_bytes()).hexdigest()
        )
    expert = load(root / "q35-port.cubin")
    final = manifest["finalize"]
    finalize_kernel = jit.load(
        (root / "finalize.cubin").read_bytes(),
        final["symbol"],
        (128, 1, 1),
        final["shared"],
    )
    return expert, finalize_kernel, final


@cache
def _sf_pack_bundle(device_index):
    """Load the independently built SF-pack expert; retain qualified finalize."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q38 SF-pack AOT bundle must load during eager graph warmup")
    _, finalize_kernel, final = _bundle(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    root, manifest = bundle_manifest("SGLANG_Q38_MK_MOE_SF_PACK_AOT", capability)
    source = (
        Path(__file__).resolve().parents[2]
        / "jit/csrc/moe/q38_mk_moe_q35_sf_pack_tp4.cu"
    )
    baseline_root, _ = bundle_manifest("SGLANG_Q38_MK_MOE_AOT", capability)
    baseline = baseline_root / "manifest.json"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        manifest["base_manifest_sha256"]
        == hashlib.sha256(baseline.read_bytes()).hexdigest()
    )
    assert manifest["shape"] == [H, I, E, TOPK, GRID, CLUSTER, SHARED]
    assert manifest["pdl_trigger"] == 0
    cubin = root / "q35-sf-pack.cubin"
    assert manifest["cubin_sha256"] == hashlib.sha256(cubin.read_bytes()).hexdigest()
    return load(cubin), finalize_kernel, final


def _finalize(kernel, metadata, rows, weights, output):

    cu = jit.driver()
    values = [ctypes.c_void_p(t.data_ptr()) for t in (rows, weights, output)]
    values += [ctypes.c_void_p(0) for _ in range(metadata["params"] - 3)]
    pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = 10, rows.shape[0], 1
    config.blockDimX, config.blockDimY, config.blockDimZ = 128, 1, 1
    config.sharedMemBytes = metadata["shared"]
    config.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    config.attrs, config.numAttrs = [], 0
    jit.check(
        cu.cuLaunchKernelEx(config, kernel.function, ctypes.addressof(pointers), 0),
        "Q38 ordered local MoE finalize",
    )


@cache
def _identity(device_index, m):

    return torch.arange(m * TOPK, dtype=torch.int32, device=f"cuda:{device_index}")
