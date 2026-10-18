# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-shot TP all-reduce fused with DeepSeek-V4.1's mHC hops, for 49-384 rows.

The attention hop (all-reduce of the attention output) and the MoE hop (deferred
top-k finalize + shared-expert add + all-reduce) both end in the HC=4 post-mix of the
residual streams and, unless the seam is closed, the next layer's pre-collapse and
RMSNorm, optionally with the MXFP8 quant of that normalized input. One kernel
(src/two_shot_mhc.cu, compiled with NVRTC) does each hop: every rank pushes row t to
its owner rank t % world, which reduces in rank order and pushes the BF16 row back.
Each instance owns a symmetric-memory mailbox; construction is collective over the TP
group and must happen before CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from sgl_nvidia_mega.common.jit import compile_source, launch, load

SOURCE = Path(__file__).resolve().parent / "src" / "two_shot_mhc.cu"
# The rows this hop serves (fewer stay on the one-shot Lamport hop); 384 = three
# 128-row scale tiles of the quant epilogue.
MIN_ROWS = 49
MAX_ROWS = 384
HIDDEN = 5120
HC = 4
CLUSTER = 5  # blocks per row: 640 16-byte vectors over 128 threads each
THREADS = 128
MAX_WORLD = 8  # kMaxWorldSize in two_shot_mhc.cu
# No --use_fast_math: the arithmetic must round as SGLang's nvcc build does.
NVRTC_OPTIONS = ("-std=c++17",)


def scale_bytes(rows: int, hidden: int = HIDDEN) -> int:
    """UE8M0 scales of [rows, hidden] in 128x4 swizzled tiles, rows padded to 128."""
    return (hidden // 32) * (-(-rows // 128) * 128)


class _Params(ctypes.Structure):
    # TwoShotParams in two_shot_mhc.cu, field for field.
    _fields_ = [
        ("out", ctypes.c_void_p),
        ("gemm2", ctypes.c_void_p),
        ("idx", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("shared", ctypes.c_void_p),
        ("norm_weight", ctypes.c_void_p),
        ("mhc_out", ctypes.c_void_p),
        ("residual", ctypes.c_void_p),
        ("post", ctypes.c_void_p),
        ("comb", ctypes.c_void_p),
        ("pre", ctypes.c_void_p),
        ("normalized", ctypes.c_void_p),
        ("quantized", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("counter", ctypes.c_void_p),
        ("workspaces", ctypes.c_void_p * MAX_WORLD),
        ("deadline_ns", ctypes.c_uint64),
        ("norm_eps", ctypes.c_float),
        ("rank", ctypes.c_uint32),
        ("num_tokens", ctypes.c_uint32),
        ("num_push_counters", ctypes.c_uint32),
        ("slot_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def _ptr(tensor: Optional[torch.Tensor]) -> Optional[int]:
    return None if tensor is None else tensor.data_ptr()


class TwoShotMHC:
    """``attn_post`` / ``moe_post`` return ``(post-mixed residual streams, normalized
    next-layer input)``; ``moe_post_only`` returns the streams for a closed seam.
    Inputs: residual [M, hc, hidden] BF16, post [M, hc] FP32, comb [M, hc, hc] FP32
    ([source, target]), pre [M, hc] FP32, norm_weight [hidden] BF16; MoE: gemm2_out
    [P, hidden] BF16 permuted rows, expanded_idx_to_permuted_idx [M, top_k] int32 (-1:
    dropped), expert_weights [M, top_k] FP32 (scale folded in), shared [M, hidden] BF16
    or None. All contiguous."""

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        max_num_tokens: int,
        top_k: int,
        device,
        group: dist.ProcessGroup,
        deadline_s: float,
    ) -> None:
        if hidden_size != HIDDEN or hc_mult != HC:
            raise ValueError(f"built for hidden {HIDDEN} and hc {HC}")
        self.hidden_size, self.hc_mult, self.top_k = hidden_size, hc_mult, top_k
        self.capacity = capacity = min(max_num_tokens, MAX_ROWS)
        self.tp = tp = dist.get_world_size(group)
        self.rank = rank = dist.get_rank(group)
        if not 2 <= tp <= MAX_WORLD:
            raise ValueError(f"TP {tp} is outside [2, {MAX_WORLD}]")
        # every poll faults at address 0x0 past it (0: none), as the Lamport hop does
        self.deadline_ns = int(deadline_s * 1e9)
        device = torch.device(device)
        with torch.cuda.device(device):
            torch.zeros(1, device=device)  # make the primary context current
            major, minor = torch.cuda.get_device_capability(device)
            arch = f"sm_{major}{minor}a"
            source = SOURCE.read_text()

            def build(k: int, shared: bool, norm: bool, quant: bool):
                options = NVRTC_OPTIONS + (
                    f"-DTS_WORLD={tp}",
                    f"-DTS_TOP_K={k}",
                    f"-DTS_HAS_SHARED={int(shared)}",
                    f"-DTS_NORM={int(norm)}",
                    f"-DTS_QUANT={int(quant)}",
                )
                cubin = compile_source(source, "two_shot_mhc", arch, options)
                return load(cubin, "dsv41_two_shot_mhc", (THREADS, 1, 1), 0)

            # Identity routing (top_k 1, unit weights) is the attention hop's plain all-reduce.
            self._attn = build(1, False, True, False)
            self._moe = {
                (shared, quant): build(top_k, shared, True, quant)
                for shared in (True, False)
                for quant in (False, True)
            }
            self._moe_post = {
                shared: build(top_k, shared, False, False) for shared in (True, False)
            }

            # Two epochs x tp producer slots of one [capacity, hidden] BF16 row view, zero
            # (the empty marker) before first use; every round restores what it reads.
            self.slot_bytes = capacity * hidden_size * 2
            self._mailbox = symm_mem.empty(
                2 * tp * self.slot_bytes, dtype=torch.uint8, device=device
            )
            self._mailbox.zero_()
            handle = symm_mem.rendezvous(self._mailbox, group)
            self._workspaces = (ctypes.c_void_p * MAX_WORLD)(
                *[int(p) for p in handle.buffer_ptrs[:tp]]
            )
            self._mailbox_handle = handle
            # One epoch counter per row cluster plus the bumper's; rank-local.
            self.num_counters = capacity + 1
            self._counter = torch.zeros(
                self.num_counters, dtype=torch.int32, device=device
            )
            # The kernel always writes the reduced row; nothing reads it.
            self._reduced = torch.empty(
                capacity, hidden_size, dtype=torch.bfloat16, device=device
            )
            self._identity_idx = torch.arange(
                capacity, dtype=torch.int32, device=device
            )
            self._identity_weights = torch.ones(
                capacity, 1, dtype=torch.float32, device=device
            )
            torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def fits(self, rows: int, hidden: int) -> bool:
        return hidden == self.hidden_size and MIN_ROWS <= rows <= self.capacity

    def scale_bytes(self, rows: int) -> int:
        """Bytes of the swizzled UE8M0 scales ``moe_post(mxfp8_out=...)`` writes."""
        return scale_bytes(rows, self.hidden_size)

    def attn_post(self, partial, residual, post, comb, pre, norm_weight, eps):
        m = partial.shape[0]
        if partial.shape != (m, self.hidden_size):
            raise ValueError("partial must be [M, hidden]")
        updated, normalized = self._outputs(m, residual, post, comb, pre, norm_weight)
        self._launch(
            self._attn,
            m,
            partial,
            self._identity_idx[:m],
            self._identity_weights[:m],
            None,
            residual,
            post,
            comb,
            updated,
            pre,
            norm_weight,
            eps,
            normalized,
        )
        return updated, normalized

    def moe_post(
        self,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        top_k,
        shared,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
        mxfp8_out=None,
    ):
        m = self._check_moe(
            gemm2_out, expanded_idx_to_permuted_idx, expert_weights, top_k, shared
        )
        updated, normalized = self._outputs(m, residual, post, comb, pre, norm_weight)
        quantized = scales = None
        if mxfp8_out is not None:
            quantized, scales = mxfp8_out
            if (
                quantized.shape != (m, self.hidden_size)
                or quantized.element_size() != 1
            ):
                raise ValueError("mxfp8_out values must be [M, hidden] FP8")
            if scales.numel() != self.scale_bytes(m) or scales.element_size() != 1:
                raise ValueError("mxfp8_out scales must hold scale_bytes(M) bytes")
        kernel = self._moe[(shared is not None, mxfp8_out is not None)]
        self._launch(
            kernel,
            m,
            gemm2_out,
            expanded_idx_to_permuted_idx,
            expert_weights,
            shared,
            residual,
            post,
            comb,
            updated,
            pre,
            norm_weight,
            eps,
            normalized,
            quantized,
            scales,
        )
        return updated, normalized

    def moe_post_only(
        self,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        top_k,
        shared,
        residual,
        post,
        comb,
    ):
        m = self._check_moe(
            gemm2_out, expanded_idx_to_permuted_idx, expert_weights, top_k, shared
        )
        self._check_mix(m, residual, post, comb, None)
        updated = torch.empty_like(residual)
        self._launch(
            self._moe_post[shared is not None],
            m,
            gemm2_out,
            expanded_idx_to_permuted_idx,
            expert_weights,
            shared,
            residual,
            post,
            comb,
            updated,
            None,
            None,
            0.0,
            None,
        )
        return updated

    def _check_moe(self, gemm2_out, idx, weights, top_k, shared) -> int:
        m = weights.shape[0]
        if top_k != self.top_k:
            raise ValueError(f"built for top_k {self.top_k}, got {top_k}")
        if weights.dtype != torch.float32 or weights.numel() != m * top_k:
            raise ValueError("expert_weights must be FP32 [M, top_k]")
        if idx.dtype != torch.int32 or idx.numel() != m * top_k:
            raise ValueError("expanded_idx_to_permuted_idx must be int32 [M, top_k]")
        if gemm2_out.dim() != 2 or gemm2_out.shape[1] != self.hidden_size:
            raise ValueError("gemm2_out must be [rows, hidden]")
        if shared is not None and shared.shape != (m, self.hidden_size):
            raise ValueError("shared must be [M, hidden]")
        return m

    def _check_mix(self, m, residual, post, comb, pre) -> None:
        hc = self.hc_mult
        if not 1 <= m <= self.capacity:
            raise ValueError(f"M={m} is outside [1, {self.capacity}]")
        if residual.shape != (m, hc, self.hidden_size):
            raise ValueError("residual must be [M, hc, hidden]")
        if post.numel() != m * hc or comb.numel() != m * hc * hc:
            raise ValueError("unexpected post / comb shapes")
        if pre is not None and pre.numel() != m * hc:
            raise ValueError("unexpected pre shape")

    def _outputs(self, m, residual, post, comb, pre, norm_weight):
        self._check_mix(m, residual, post, comb, pre)
        if norm_weight.shape != (self.hidden_size,):
            raise ValueError("norm_weight must be [hidden]")
        normalized = torch.empty(
            (m, self.hidden_size), dtype=torch.bfloat16, device=residual.device
        )
        return torch.empty_like(residual), normalized

    def _launch(
        self,
        kernel,
        m,
        gemm2,
        idx,
        weights,
        shared,
        residual,
        post,
        comb,
        mhc_out,
        pre,
        norm_weight,
        eps,
        normalized,
        quantized=None,
        scales=None,
    ) -> None:
        params = _Params(
            out=self._reduced.data_ptr(),
            gemm2=gemm2.data_ptr(),
            idx=idx.data_ptr(),
            weights=weights.data_ptr(),
            shared=_ptr(shared),
            norm_weight=_ptr(norm_weight),
            mhc_out=mhc_out.data_ptr(),
            residual=residual.data_ptr(),
            post=post.data_ptr(),
            comb=comb.data_ptr(),
            pre=_ptr(pre),
            normalized=_ptr(normalized),
            quantized=_ptr(quantized),
            scales=_ptr(scales),
            counter=self._counter.data_ptr(),
            workspaces=self._workspaces,
            deadline_ns=self.deadline_ns,
            norm_eps=float(eps),
            rank=self.rank,
            num_tokens=m,
            num_push_counters=self.num_counters,
            slot_bytes=self.slot_bytes,
            reserved=0,
        )
        launch(
            kernel,
            (m + 1, CLUSTER, 1),
            [params],
            torch.cuda.current_stream().cuda_stream,
            cluster=(1, CLUSTER, 1),
        )
