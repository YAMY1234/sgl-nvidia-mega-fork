# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): the source's
# python/sglang/srt/layers/bf16_gemm_pins.py (the file format, _parse, _load, the
# runners, _build and gemm) and its call sites' conditions in
# python/sglang/srt/layers/quantization/unquant.py (UnquantizedLinearMethod.apply into
# _bf16_gemm_dispatch_impl) and python/sglang/srt/models/qwen3_5.py
# (Qwen3_5GatedDeltaNet._forward_input_proj's packed GEMM through bf16_gemm_dispatch).
# Changed: the file comes from the environment directly, with an optional SHA-256 check.
# The source serves the pins inside SGLang's BF16 dispatch; here the executor's own
# projection calls take them (``linear``, ``in_proj``), where that dispatch would have
# run, at TP1. GEMMs that SGLang's modules launch themselves keep SGLang's dispatch.
"""Pinned kernels for unquantized BF16 GEMMs at listed (M, N, K) (default off).

``SGLANG_BF16_GEMM_PIN_FILE`` names a JSON file (``.json.py`` so a source overlay can
carry it)::

    {"_metadata": {"flashinfer_version": "<flashinfer.__version__>", "cuda_version": "13.4",
                   "cublas_version": "13.8.1", "gpu": "NVIDIA VR NVL72", "sm": "10.7"},
     "bf16_gemm_pins": [
        {"n": 16480, "k": 2560, "m": 256, "kind": "cublaslt", "algo": "<128 hex>", "workspace": 0},
        {"n": 640, "k": 2560, "m": 256, "kind": "tgv", "tactic": 21},
     ]}

``SGLANG_BF16_GEMM_PIN_FILE_SHA256``, when set, is the file's SHA-256; a file with
another digest is not applied.

Each entry names, for one exact M (a CUDA-graph batch size), a kernel measured faster
than the stock dispatch and bit-equal to it:

* ``"cublaslt"``: a serialized ``cublasLtMatmulAlgo_t`` (opaque, valid for the cuBLAS
  build the metadata names) run through FlashInfer's mm_bf16_cublaslt module (TRANSA=T,
  TRANSB=N, FP32 compute, BF16 output, beta 0). A pin keeps stock's K reduction order.
* ``"tgv"``: SGLang's CuTe DSL TGV kernel at a fixed tactic (it never splits K).

The file applies only under its exact ``_metadata``
(``flashinfer_tactic_pins.tuning_file_applies``). Before a key serves, its first eager
call checks it once (seeded random activations, the layer's weight) against the stock
dispatch (``stock``): a pin that fails to launch or differs in a single bit is dropped
for good and the stock dispatch kept. A first call under CUDA graph capture cannot
check, so that call keeps the stock dispatch. Any other shape or M, a bias, a
non-contiguous or non-BF16 operand, batch-invariant mode or torch.compile tracing takes
the stock dispatch.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch

from sgl_nvidia_mega.qwen3_8.runtime import flashinfer_tactic_pins

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/layers/quantization/unquant.py": (
        "UnquantizedLinearMethod",
        "get_bf16_gemm_backend",
        "is_cutedsl",
        "is_gemv",
        "_bf16_gemm_dispatch_impl",
    ),
    "kernels/ops/gemm/cutedsl_bf16_gemm.py": (
        "cutedsl_bf16_gemm",
        "_resolve_tactic",
        "_run_tgv",
    ),
    "srt/models/qwen3_5.py": (
        "_forward_input_proj",
        "_fused_in_proj_weight",
        "_fused_in_proj_qkvz_width",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

# ``in_proj`` takes this branch's GEMM
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet._forward_input_proj": "5269f43d28e6f303",
}

FILE_ENV = "SGLANG_BF16_GEMM_PIN_FILE"
SHA256_ENV = "SGLANG_BF16_GEMM_PIN_FILE_SHA256"
_SECTION = "bf16_gemm_pins"
_ALGO_BYTES = 64
_KINDS = ("cublaslt", "tgv")
_CHECK_SEED = 0x4442
# _forward_input_proj takes the packed GEMM up to this many rows
_PACKED_IN_PROJ_MAX_ROWS = 1024

_state: Optional[dict] = None  # None: not read yet; {} when off
_runners: dict = {}  # (device, m, n, k) -> callable(x, w) or None (stock)
_lt_module = None
_workspaces: dict = {}
# set by build: the executor's projection calls take the pins (TP1)
_routed = False


def _parse(document: dict, path: str) -> dict:
    section = document.get(_SECTION)
    if not isinstance(section, list):
        raise ValueError(f"{path}: needs a list {_SECTION!r}")
    entries = {}
    for i, entry in enumerate(section):
        where = f"{path}: {_SECTION}[{i}]"
        try:
            m, n, k = int(entry["m"]), int(entry["n"]), int(entry["k"])
            kind = entry["kind"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{where} needs integer m, n, k and a kind") from None
        if kind not in _KINDS or min(m, n, k) <= 0:
            raise ValueError(f"{where}: kind must be one of {_KINDS}, m, n, k positive")
        if kind == "cublaslt":
            algo = bytes.fromhex(entry["algo"])
            if len(algo) != _ALGO_BYTES:
                raise ValueError(f"{where}: algo must be {_ALGO_BYTES} bytes")
            spec = ("cublaslt", algo, int(entry.get("workspace", 32 << 20)))
        else:
            spec = ("tgv", int(entry["tactic"]))
        if (m, n, k) in entries:
            raise ValueError(f"{where}: duplicate key {(m, n, k)}")
        entries[(m, n, k)] = spec
    return entries


def _load() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    path = os.environ.get(FILE_ENV)
    if not path:
        return _state
    entries = _parse(flashinfer_tactic_pins.read_tuning_file(path), path)
    reason = flashinfer_tactic_pins.digest_reason(path, SHA256_ENV)
    if reason:
        logger.warning("BF16 GEMM pins off: %s", reason)
        return _state
    if not flashinfer_tactic_pins.tuning_file_applies(path):
        logger.warning("BF16 GEMM pins off: %s is for another stack (see above)", path)
        return _state
    _state = {
        "path": path,
        "entries": entries,
        "shapes": {(n, k) for (_, n, k) in entries},
    }
    logger.info(
        "BF16 GEMM pins on: %d keys over %d shapes from %s",
        len(entries),
        len(_state["shapes"]),
        os.path.basename(path),
    )
    return _state


def enabled() -> bool:
    return bool(_load())


def has_shape(n: int, k: int) -> bool:
    state = _load()
    return bool(state) and (n, k) in state["shapes"]


def _cublaslt_runner(device: torch.device, n: int, k: int, algo: bytes, workspace: int):
    global _lt_module
    if _lt_module is None:
        from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

        _lt_module = gen_mm_bf16_cublaslt_module().build_and_load()
    module = _lt_module
    key = (device.index, n, k)
    ws = _workspaces.get(key)
    if ws is None or ws.numel() < max(workspace, 1):
        # One per shape: two GEMMs of one shape never run concurrently on two streams here.
        ws = torch.empty(max(workspace, 1), dtype=torch.uint8, device=device)
        _workspaces[key] = ws
    algo_buf = torch.frombuffer(bytearray(algo), dtype=torch.uint8).clone()

    def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
        )
        with torch.cuda.device(x.device):
            handle = torch.cuda.current_blas_handle()
        module.mm_bf16_cublaslt_run_with_algo(
            x, weight, None, out, ws, handle, algo_buf, 0
        )
        return out

    return run


def _tgv_runner(tactic: int):
    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm

    cutedsl_bf16_gemm._resolve_tactic(tactic)  # range check

    def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
        )
        cutedsl_bf16_gemm._run_tgv(x, weight.t(), None, out, True, tactic)
        return out

    return run


def _build(key, spec, weight: torch.Tensor, stock: Callable) -> Optional[Callable]:
    """The pinned runner for ``key`` if it launches and matches ``stock`` bit for bit, else None (for good)."""
    m, n, k = key[1:]
    try:
        if spec[0] == "cublaslt":
            run = _cublaslt_runner(weight.device, n, k, spec[1], spec[2])
        else:
            run = _tgv_runner(spec[1])
        gen = torch.Generator(device=weight.device).manual_seed(_CHECK_SEED + m)
        x = torch.randn((m, k), device=weight.device, generator=gen).to(torch.bfloat16)
        got, ref = run(x, weight), stock(x, weight)
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
    except Exception as e:  # noqa: BLE001 - no module, bad algorithm or tactic: stock for good
        logger.warning(
            "BF16 GEMM pin M=%d N=%d K=%d %s refused, stock kept: %r",
            m,
            n,
            k,
            spec[0],
            e,
        )
        return None
    if not same:
        logger.warning(
            "BF16 GEMM pin M=%d N=%d K=%d %s differs from stock, stock kept",
            m,
            n,
            k,
            spec[0],
        )
        return None
    logger.info(
        "BF16 GEMM pin engaged: M=%d N=%d K=%d %s%s",
        m,
        n,
        k,
        spec[0],
        f" tactic {spec[1]}" if spec[0] == "tgv" else "",
    )
    return run


def gemm(
    x: torch.Tensor, weight: torch.Tensor, stock: Callable
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` on the pinned kernel for this (M, N, K), or None to run the stock dispatch.

    ``stock(x, weight)`` is the dispatch without pins; it is called only once per key, by the check."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    state = _load()
    if not state or x.ndim != 2 or weight.ndim != 2:
        return None
    n, k = weight.shape
    if (n, k) not in state["shapes"]:
        return None
    m = x.shape[0]
    spec = state["entries"].get((m, n, k))
    if (
        spec is None
        or x.shape[1] != k
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.compiler.is_compiling()
        or is_batch_invariant_mode_enabled()
    ):
        return None
    key = (x.device.index, m, n, k)
    if key in _runners:
        run = _runners[key]
    elif torch.cuda.is_current_stream_capturing():
        logger.warning(
            "BF16 GEMM pin M=%d N=%d K=%d first seen under graph capture: stock in this graph",
            m,
            n,
            k,
        )
        return None
    else:
        run = _runners[key] = _build(key, spec, weight, stock)
    return None if run is None else run(x, weight)


# -- the executor's projection calls -------------------------------------------------


def _stock(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

    return _bf16_gemm_dispatch_impl(x, weight, None)


def _reaches_dispatch(module, x: torch.Tensor) -> bool:
    """Whether ``UnquantizedLinearMethod.apply`` hands this call to
    ``_bf16_gemm_dispatch_impl`` without a bias (its ROCm-only branches before that
    never run on CUDA)."""
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedLinearMethod,
        get_bf16_gemm_backend,
    )

    backend = get_bf16_gemm_backend()
    return (
        type(module.quant_method) is UnquantizedLinearMethod
        and module.bias is None
        and (backend.is_cutedsl() or backend.is_gemv())
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and module.weight.dtype == torch.bfloat16
        and not module.weight.requires_grad
    )


def routed() -> bool:
    return _routed


def enable_routes() -> None:
    """Build agreed: the executor's projection calls take the pins."""
    global _routed
    _routed = True


def linear(module, x: torch.Tensor) -> torch.Tensor:
    """``module(x)[0]`` for an executor's TP1 projection call, its GEMM on the pinned
    kernel where one serves this (M, N, K) and SGLang's linear method would have run
    its BF16 dispatch."""
    if _routed and _reaches_dispatch(module, x):
        out = gemm(x, module.weight, _stock)
        if out is not None:
            return out
    return module(x)[0]


def in_proj(gdn, x: torch.Tensor):
    """``gdn._forward_input_proj(x)``, its packed (qkvz | ba) GEMM on the pinned kernel
    where one serves (the CUDA branch that runs it through ``bf16_gemm_dispatch``)."""
    if (
        _routed
        and gdn._fused_in_proj_weight is not None
        and x.dtype == torch.bfloat16
        and x.shape[0] <= _PACKED_IN_PROJ_MAX_ROWS
    ):
        out = gemm(x, gdn._fused_in_proj_weight, _stock)
        if out is not None:
            split = gdn._fused_in_proj_qkvz_width
            return out[:, :split], out[:, split:]
    return gdn._forward_input_proj(x)


def build(model, call_sites: dict) -> str:
    """'' when the file's pins reach at least one of the executor's projection calls
    (``call_sites``: (N, K) -> name), else why not (a malformed file raises). Logs the
    file's shapes no executor call takes."""
    from sglang.srt.runtime_context import get_parallel

    if not os.environ.get(FILE_ENV):
        return f"{FILE_ENV} unset"
    state = _load()
    if not state:
        return f"{os.environ[FILE_ENV]} does not apply here (see the warning above)"
    if get_parallel().tp_size != 1:
        return "TP1 only: a pinned GEMM replaces the whole projection"
    served = sorted(call_sites[s] for s in state["shapes"] if s in call_sites)
    stranded = sorted(state["shapes"] - set(call_sites))
    if stranded:
        logger.warning(
            "BF16 GEMM pins: no executor call takes the shapes (N, K) %s; SGLang's "
            "modules launch them on its own dispatch",
            stranded,
        )
    if not served:
        return f"no executor projection has a pinned shape (file: {sorted(state['shapes'])})"
    logger.info(
        "BF16 GEMM pins: %s (SHA-256 %s) serve the executor's %s, each key checked "
        "bit for bit against SGLang's dispatch at its first eager call",
        state["path"],
        flashinfer_tactic_pins.file_digest(state["path"])[:16],
        ", ".join(served),
    )
    return ""
