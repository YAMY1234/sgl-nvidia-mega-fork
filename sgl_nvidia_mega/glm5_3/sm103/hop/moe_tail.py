# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The NextN draft MoE's tail in one hop: the finalize of the trtllm-gen deferred output
plus the shared-expert add, folded into the all-reduce + residual + RMSNorm.

Inputs (SGLang's ``MoeDeferredFinalize``, up to 8 rows): ``gemm2`` the routed FC2 rows
(bf16, row stride ``gemm2_ld`` elements), ``perm`` ``[rows][TOPK]`` int32 (the expanded
index's permuted row; ``< 0``: dropped slot), ``weights`` ``[rows][TOPK]`` (fp32 or bf16,
the routing weights with the routed scaling folded in) and ``shared`` ``[rows][HIDDEN]``
bf16 (the shared expert's output). Per element this rank's partial is

    s = fma(w_k, x_k, s) over k = 0..TOPK-1 ascending from +0 (dropped slots skipped)
    partial = bf16(s + shared)                      (one rounding)

and the hop then all-reduces it as ``AllreduceNormHop(residual=True, norm=True)``:
``out_res = bf16(bf16(sum) + residual)``, ``out_norm = RMSNorm(out_res) * gamma`` in the
one-shot fused order (``transport.Consumer``). Compared with SGLang's draft tail (the
TRT-LLM finalize to bf16, a bf16 shared add, the plain all-reduce, then SGLang's fused
add + RMSNorm) the partial is rounded once instead of twice and the norm is the hop's,
so only the draft's proposals can change.

The routing tensors and the GEMM output are read after the grid-dependency wait (the
FC2 GEMM is the previous kernel on this stream; the shared expert joins through the
stream's event edge). Grid, protocol and launch are ``AllreduceNormHop``'s.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as C
from ..primitives import ptx_sys as sys_ops
from . import _ptx_local as P
from .constants import BF16_NAN, CONSUMER_THREADS, HIDDEN, POLL_THREADS, TAG_MASK, TOPK
from .transport import ConsSmem, Consumer, Mbox


class MoeTailNormHop(Consumer):
    """Draft MoE finalize + shared add + all-reduce + residual + RMSNorm (up to 8 rows)."""

    def __init__(
        self,
        *,
        weights_f32: bool,
        has_shared: bool = True,
        max_rows: int = 8,
        trigger_late: bool = False,
    ):
        if max_rows != 8:
            raise ValueError("this hop is provided for up to 8 rows")
        self.weights_f32 = bool(weights_f32)
        self.has_shared = bool(has_shared)
        self.max_rows = max_rows
        # no early griddepcontrol.launch_dependents: the dependent grid launches when this
        # grid completes (as after SGLang's final RMSNorm kernel)
        self.trigger_late = bool(trigger_late)

    @cute.jit
    def __call__(
        self,
        gemm2: Int64,
        perm: Int64,
        weights: Int64,
        shared: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        gemm2_ld: Int32,
        eps: Float32,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        """``p0 .. p7``: mailbox bases rotated so that ``p0`` is this rank's own."""
        self.kernel(
            gemm2,
            perm,
            weights,
            shared,
            residual,
            gamma,
            out_res,
            out_norm,
            gemm2_ld,
            eps,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            rank,
            err_host,
            site,
        ).launch(
            grid=[rows * 8 + 8, 1, 1],
            block=[CONSUMER_THREADS, 1, 1],
            max_number_threads=[CONSUMER_THREADS, 1, 1],
            cluster=[8, 1, 1],
            cooperative=True,
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gemm2: Int64,
        perm: Int64,
        weights: Int64,
        shared: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        gemm2_ld: Int32,
        eps: Float32,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
    ):
        mb = Mbox((p0, p1, p2, p3, p4, p5, p6, p7), rank, self.max_rows, err_host, site)
        t = P.thread_index()
        bx = P.block_index()
        smem = cutlass.utils.SmemAllocator()
        cs = ConsSmem.allocate(smem, self.max_rows)
        nmain = rows * 8
        if bx < nmain:
            self.cons_prologue(cs)
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.trigger_late):
            cute.arch.griddepcontrol_launch_dependents()
        sys_ops.fence_proxy_async_global()  # inputs may come from an async-proxy writer
        if bx >= nmain:
            e = self.epoch_begin(mb, cs.es, False)
            if t == 0:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
                if bx == nmain:
                    self.ack_cta(mb, e)
            sys_ops.exit_thread()
        m = bx // 8
        c = bx % 8
        h0 = c * 768 + 6 * t
        w = cute.make_rmem_tensor((3,), Uint32)
        ru = cute.make_rmem_tensor((3,), Uint32)
        gu = cute.make_rmem_tensor((3,), Uint32)
        w.fill(0)
        ru.fill(0)
        gu.fill(0)
        # every input load is issued before the epoch / acknowledgement check
        if t < 128:
            off = Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                ru[k] = sys_ops.ld_cg_u32(residual + off + Int64(4 * k))
            for k in cutlass.range_constexpr(3):
                gu[k] = sys_ops.ld_cg_u32(gamma + Int64(2) * Int64(h0) + Int64(4 * k))
            # the row's slots: permuted rows and weights
            # int32 permuted rows kept as their bits: a dropped slot (< 0) is >= 2^31
            pid = cute.make_rmem_tensor((TOPK,), Uint32)
            wt = cute.make_rmem_tensor((TOPK,), Float32)
            for kk in cutlass.range_constexpr(TOPK):
                pid[kk] = sys_ops.ld_cg_u32(
                    perm + Int64(4) * (Int64(m) * TOPK + Int64(kk))
                )
            if cutlass.const_expr(self.weights_f32):
                for kk in cutlass.range_constexpr(TOPK):
                    wt[kk] = P.bits_f32(
                        sys_ops.ld_cg_u32(
                            weights + Int64(4) * (Int64(m) * TOPK + Int64(kk))
                        )
                    )
            else:
                for j in cutlass.range_constexpr(TOPK // 2):
                    u = sys_ops.ld_cg_u32(
                        weights + Int64(2) * (Int64(m) * TOPK) + Int64(4 * j)
                    )
                    wt[2 * j] = P.bf16_lo(u)
                    wt[2 * j + 1] = P.bf16_hi(u)
            # every slot's 6 elements (a dropped slot reads row 0 and is skipped)
            xs = cute.make_rmem_tensor((TOPK * 3,), Uint32)
            for kk in cutlass.range_constexpr(TOPK):
                prow = Uint32(
                    cutlass.select_(pid[kk] >= Uint32(0x80000000), Uint32(0), pid[kk])
                )
                base = gemm2 + Int64(2) * (Int64(prow) * Int64(gemm2_ld) + Int64(h0))
                for k in cutlass.range_constexpr(3):
                    xs[kk * 3 + k] = sys_ops.ld_cg_u32(base + Int64(4 * k))
            su = cute.make_rmem_tensor((3,), Uint32)
            su.fill(0)
            if cutlass.const_expr(self.has_shared):
                for k in cutlass.range_constexpr(3):
                    su[k] = sys_ops.ld_cg_u32(shared + off + Int64(4 * k))
            acc = cute.make_rmem_tensor((6,), Float32)
            acc.fill(0.0)
            for kk in cutlass.range_constexpr(TOPK):
                live = pid[kk] < Uint32(0x80000000)
                for k in cutlass.range_constexpr(3):
                    lo = C.fma_rn(wt[kk], P.bf16_lo(xs[kk * 3 + k]), acc[2 * k])
                    hi = C.fma_rn(wt[kk], P.bf16_hi(xs[kk * 3 + k]), acc[2 * k + 1])
                    acc[2 * k] = Float32(cutlass.select_(live, lo, acc[2 * k]))
                    acc[2 * k + 1] = Float32(cutlass.select_(live, hi, acc[2 * k + 1]))
            for k in cutlass.range_constexpr(3):
                a0 = acc[2 * k]
                a1 = acc[2 * k + 1]
                if cutlass.const_expr(self.has_shared):
                    a0 = C.add_rn(a0, P.bf16_lo(su[k]))
                    a1 = C.add_rn(a1, P.bf16_hi(su[k]))
                w[k] = C.cvt_rn_bf16x2_f32(a0, a1)
        self.epoch_begin(mb, cs.es, True)
        pz = cs.es.pz[0]
        em = cs.es.erow[m]  # row epoch: buffer slot and tag of row m
        buf = self.buffer_of(em)
        tag = em & Uint32(TAG_MASK)
        if t < 128:
            for k in cutlass.range_constexpr(3):
                if pz != 0:
                    w[k] = Uint32(BF16_NAN | (BF16_NAN << 16))
                else:
                    w[k] = P.no_neg_zero(w[k])
            r0, r1 = self.pack_records(w[0], w[1], w[2], tag)
            self.delay(mb, 1)
            self.publish_records(mb, buf, m, c * 256 + 2 * t, r0, r1)
        if t >= POLL_THREADS:  # the epoch warp arrives, then leaves the CTA
            if t == POLL_THREADS:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
            sys_ops.exit_thread()
        self.consume_cta(
            mb,
            cs,
            m,
            c,
            em,
            pz,
            ru,
            gu,
            out_res,
            out_norm,
            eps,
            True,
            True,
        )
