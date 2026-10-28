# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Isolated HC hop transport; no replacement of SGLang's boundary dispatch.

The producer publishes its final local BF16 block. The consumer preserves an
additional BF16 cut after the TP sum, injects into four residual branches, and
normalizes each branch. This is a separate multicast mailbox, never an alias of
a FlashInfer/CA workspace. CPU-built cubins are mandatory for the first-stage test.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "hop.cu"
HEADER = Path(__file__).parent / "src" / "fn_hop.cuh"
STAGES, ROW_CTAS, ROW_THREADS = 3, 4, 128
SENTINEL = -32768


@dataclass(frozen=True)
class Config:
    hidden: int
    branches: int
    lowrank: int
    eps: float

    @classmethod
    def from_text_config(cls, text_config):
        return cls(
            hidden=int(text_config.hidden_size),
            branches=int(text_config.hc_count),
            lowrank=int(text_config.hc_lowrank),
            eps=float(text_config.rms_norm_eps),
        )


def config_disable_reasons(config: Config, *, world: int, max_m: int) -> list[str]:
    reasons = []
    if world != 4:
        reasons.append(f"HC hop first stage serves TP4, not TP{world}")
    if config.branches != 4:
        reasons.append(f"HC hop serves four branches, not {config.branches}")
    if config.hidden <= 0 or config.hidden % (ROW_CTAS * 8):
        reasons.append(f"hidden {config.hidden} must be positive and divisible by 32")
    elif config.hidden // (ROW_CTAS * 8) > ROW_THREADS:
        reasons.append(f"hidden {config.hidden} exceeds the four-CTA fragment capacity")
    if config.lowrank <= 0 or config.eps <= 0:
        reasons.append("HC lowrank and RMS epsilon must be positive")
    if max_m <= 0:
        reasons.append(f"mailbox row capacity must be positive, got {max_m}")
    return reasons


class Site(ctypes.Structure):
    _fields_ = [
        ("mc", ctypes.c_void_p),
        ("uc", ctypes.c_void_p),
        ("ctrl", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("world", ctypes.c_int),
        ("m_cap", ctypes.c_int),
    ]


class Args(ctypes.Structure):
    _fields_ = [
        ("site", Site),
        ("partial", ctypes.c_void_p),
        ("residual", ctypes.c_void_p),
        ("gates", ctypes.c_void_p),
        ("gamma", ctypes.c_void_p),
        ("reduced", ctypes.c_void_p),
        ("combined", ctypes.c_void_p),
        ("normalized", ctypes.c_void_p),
        ("rows", ctypes.c_int),
        ("eps", ctypes.c_float),
    ]


class Hop:
    """Collectively allocated mailbox, with explicit publish and consume calls.

    Allocate on all ranks before graph capture. All ranks must execute the same
    sequence of hops; each published generation is consumed exactly once.
    """

    def __init__(self, *, config: Config, max_m: int, group, device, cubin: bytes):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        reasons = config_disable_reasons(config, world=self.world, max_m=max_m)
        if reasons:
            raise ValueError("; ".join(reasons))
        self.config, self.max_m = config, max_m
        self.device = torch.device(device)
        self.publish_kernel = jit.load(cubin, "fn_hop_publish", (ROW_THREADS, 1, 1), 0)
        self.consume_kernel = jit.load(cubin, "fn_hop_consume_w4", (ROW_THREADS, 1, 1), 0)
        self.evict_kernel = jit.load(cubin, "fn_hop_evict", (256, 1, 1), 0)
        if symm_mem.get_backend(self.device) is None:
            symm_mem.set_backend("NCCL")
        self.mailbox = symm_mem.empty(
            (STAGES, self.world, max_m, config.hidden),
            dtype=torch.int16, device=self.device,
        )
        self.handle = symm_mem.rendezvous(self.mailbox, group)
        multicast = int(self.handle.multicast_ptr or 0)
        if not multicast or multicast % 16:
            raise RuntimeError("HC hop requires an aligned NVLink multicast mapping")
        self.mailbox.fill_(SENTINEL)
        self.ctrl = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.site = Site(
            multicast, self.mailbox.data_ptr(), self.ctrl.data_ptr(),
            self.rank, self.world, max_m,
        )
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def args(self, *, partial, residual, gates, gamma, reduced, combined, normalized):
        """Bind already allocated contiguous tensors; no activation allocation at launch."""
        rows = int(partial.shape[0])
        if not 1 <= rows <= self.max_m:
            raise ValueError(f"HC hop rows {rows} exceed mailbox capacity {self.max_m}")
        return Args(
            self.site,
            *(tensor.data_ptr() for tensor in
              (partial, residual, gates, gamma, reduced, combined, normalized)),
            rows, self.config.eps,
        )

    def publish(self, args: Args, *, stream: int):
        jit.launch(self.publish_kernel, (args.rows, ROW_CTAS, 1), [args], stream)

    def consume(self, args: Args, *, stream: int):
        jit.launch(
            self.consume_kernel, (args.rows, ROW_CTAS, 1), [args], stream,
            cluster=(1, ROW_CTAS, 1),
        )

    def evict(self, storage, *, stream: int):
        jit.launch(
            self.evict_kernel, (256, 1, 1),
            [ctypes.c_void_p(storage.data_ptr()), ctypes.c_uint(storage.numel())],
            stream,
        )
