# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measurement knobs for static A/Bs of changes that move MoE routing (default off):

- ``SGLANG_PERFLL_UHIST=<dir>``: per layer, the histogram of distinct experts (U) of every
  mkm step of up to 16 rows, counted on the GPU by a one-warp launch on a side stream
  forked after the engine (joined after the last layer, off the layer chain). When the
  file ``<dir>/dump-now`` appears (the bench script touches it once the server is idle),
  each rank writes ``<dir>/uhist-r<rank>-<pid>.json``: {layer: {U: steps}}.
- ``SGLANG_PERFLL_PIN_LOGITS=<ids.npy>`` (int32 [layers, rows, 10], e.g. a census routing
  table): mkm reads crafted bf16 logits whose top-10 is that table's row instead of the
  step's own logits, so both arms of an A/B route alike; the step still computes its logits.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
import time

import torch

from sgl_nvidia_mega.common import jit

logger = logging.getLogger(__name__)

DIR = os.environ.get("SGLANG_PERFLL_UHIST", "")
PIN = os.environ.get("SGLANG_PERFLL_PIN_LOGITS", "")
_MAX_LAYERS, _ROW = 128, 168  # U of up to 16 rows x top-10
_STATE: dict = {}

_SRC = r"""
extern "C" __global__ void q35_uhist(const int* ids, int n, int layer, unsigned long long* hist) {
  __shared__ unsigned seen[16];
  if (threadIdx.x < 16) seen[threadIdx.x] = 0u;
  __syncwarp();
  for (int i = threadIdx.x; i < n; i += 32) {
    const int e = ids[i];
    if (e >= 0 && e < 512) atomicOr(&seen[e >> 5], 1u << (e & 31));
  }
  __syncwarp();
  if (threadIdx.x == 0) {
    int u = 0;
    for (int w = 0; w < 16; ++w) u += __popc(seen[w]);
    atomicAdd(&hist[layer * 168 + u], 1ull);
  }
}
"""


def build() -> None:
    """Compile the counter and load the pinned logits (before any capture)."""
    if DIR and "k" not in _STATE:
        dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(dev)
        cubin = jit.compile_source(_SRC, "q35_uhist", f"sm_{major}{minor}a", ("-std=c++17", "-default-device"))
        _STATE["k"] = jit.load(cubin, "q35_uhist", (32, 1, 1), 0)
        _STATE["hist"] = torch.zeros(_MAX_LAYERS * _ROW, dtype=torch.int64, device="cuda")
        _STATE["side"] = torch.cuda.Stream()
        _STATE["pending"] = False
        threading.Thread(target=_dumper, args=(dev,), daemon=True).start()
        logger.info("NVIDIA megakernel: Qwen3.5 U histogram on (%s)", DIR)
    if PIN and "pin" not in _STATE:
        import numpy as np

        ids = torch.from_numpy(np.load(PIN).astype(np.int64)).cuda()  # [layers, rows, 10]
        layers, rows = ids.shape[0], min(ids.shape[1], 16)
        lg = torch.full((layers, rows, 512), -4.0, dtype=torch.float32, device="cuda")
        val = (4.0 - 0.1 * torch.arange(10, dtype=torch.float32, device="cuda")).expand(layers, rows, 10)
        lg.scatter_(2, ids[:, :rows], val.contiguous())
        _STATE["pin"] = lg.to(torch.bfloat16).contiguous()
        logger.info("NVIDIA megakernel: Qwen3.5 mkm logits pinned to %s %s", PIN, tuple(ids.shape))


def pinned_logits(layer_id: int, logits: torch.Tensor) -> torch.Tensor:
    """The pinned logits of ``layer_id`` for ``logits``' rows, or ``logits``."""
    table = _STATE.get("pin")
    if table is None or logits.shape[0] > table.shape[1]:
        return logits
    return table[layer_id % table.shape[0], : logits.shape[0]]


def record(layer_id: int, topk_ids: torch.Tensor) -> None:
    """Count this step's U at ``layer_id`` on the side stream, after the engine."""
    if "k" not in _STATE or topk_ids is None or topk_ids.numel() > _ROW - 8:
        return
    cu = jit.driver()
    side, cur = _STATE["side"], torch.cuda.current_stream()
    side.wait_stream(cur)
    args = [ctypes.c_void_p(topk_ids.data_ptr()), ctypes.c_int(topk_ids.numel()),
            ctypes.c_int(layer_id % _MAX_LAYERS), ctypes.c_void_p(_STATE["hist"].data_ptr())]
    ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = 1, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
    config.sharedMemBytes = 0
    config.hStream = cu.CUstream(side.cuda_stream)
    config.numAttrs = 0  # a plain launch: it reads the engine's ids after it completes
    jit.check(cu.cuLaunchKernelEx(config, _STATE["k"].function, ctypes.addressof(ptrs), 0), "q35_uhist")
    _STATE["pending"] = True


def join() -> None:
    """The step's stream waits for the counters (after the last layer)."""
    if _STATE.get("pending"):
        torch.cuda.current_stream().wait_stream(_STATE["side"])
        _STATE["pending"] = False


def _dumper(dev: int) -> None:
    flag, last = os.path.join(DIR, "dump-now"), None
    while True:
        time.sleep(1.0)
        try:
            if not os.path.exists(flag):
                continue
            stamp = os.path.getmtime(flag)
            if stamp == last:
                continue
            last = stamp
            import torch.distributed as dist

            torch.cuda.set_device(dev)
            rank = dist.get_rank() if dist.is_initialized() else 0
            s = torch.cuda.Stream(device=dev)
            with torch.cuda.stream(s):
                host = _STATE["hist"].to("cpu", non_blocking=True)
            s.synchronize()
            h = host.view(_MAX_LAYERS, _ROW).tolist()
            out = {str(i): {str(u): c for u, c in enumerate(row) if c} for i, row in enumerate(h) if any(row)}
            path = os.path.join(DIR, f"uhist-r{rank}-{os.getpid()}.json")
            with open(path + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(path + ".tmp", path)
        except Exception as e:  # measurement only: never take the server down
            logger.warning("NVIDIA megakernel: U histogram dump failed: %s", e)
