# SPDX-License-Identifier: Apache-2.0


"""FN TP1/TP4 front: mature TGV arithmetic, parameterized verify convolution.

Derived from gdn_inproj_conv_core; dynamic head width and B1/B2/B3.
Weight TMA remains before activation PDL wait; no cross-CTA conv state sharing.

CuTe DSL TGV BF16 GEMM (low-latency SM10x GEMM).

Computes ``out[M, N] = x[M, K] @ weight[N, K].T (+ bias[N])`` for bf16 inputs,
fp32 accumulation, bf16 output. The kernel writes M-contiguous output, so the
runner swaps A and B (``gemm_fn(b.t(), a.t(), ...)``) to land row-major (M, N).

Toggled by ``use_2cta`` in the constructor:
  use_2cta=False -> 1x1 cluster, 1-CTA tcgen05.mma, cta_n in [8, 256] step 8
  use_2cta=True  -> 2x1 cluster, 2-CTA tcgen05.mma, cta_n in [16, 256] step 16

Warp specialization (8 warps, 256 threads/CTA; warp 3 idle):
  Warp 0    DMA_A   TMA-loads A tiles
  Warp 1    DMA_B   TMA-loads B tiles; PDL griddepcontrol.wait
  Warp 2    MMA     tcgen05.mma into TMEM; owns alloc/dealloc
  Warps 4-7 EPILOG  TMEM -> RMEM -> bf16 cast -> st.global
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(), cutlass.Float32(b).ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.experimental.jit
def _add_product(acc, a, b):
    product = cutlass.BFloat16(_mul_rn(a, b))
    return _add_rn(acc, cutlass.Float32(product))


@dsl_user_op
def _silu(a, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value()],
            "{ .reg .f32 e, d; mul.rn.f32 e, $1, -1.4426950408889634; "
            "ex2.approx.f32 e, e; add.rn.f32 d, e, 1.0; div.full.f32 $0, $1, d; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


_TGV_CUTE_EXT_CTA_K: int = 128
_TGV_CUTE_EXT_DEFAULT_TACTIC: int = 1
_TGV_CUTE_EXT_TACTIC_CONFIGS: List[Tuple[int, int, int, bool]] = [
    (64, 8, 6, False),
    (64, 8, 8, False),
    (64, 8, 10, False),
    (64, 8, 12, False),
    (64, 16, 6, False),
    (64, 16, 8, False),
    (64, 16, 11, False),
    (64, 32, 6, False),
    (64, 32, 9, False),
    (64, 64, 7, False),
    (64, 128, 4, False),
    (128, 8, 6, False),
    (128, 16, 6, False),
    (128, 32, 5, False),
    (128, 64, 4, False),
    (128, 128, 3, False),
    (64, 16, 6, True),
    (64, 16, 8, True),
    (64, 16, 12, True),
    (64, 32, 6, True),
    (64, 32, 8, True),
    (64, 32, 11, True),
    (64, 64, 6, True),
    (64, 64, 9, True),
    (64, 128, 7, True),
    (128, 16, 6, True),
    (128, 32, 6, True),
    (128, 64, 5, True),
    (128, 128, 4, True),
]


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes. Mirrors the original WorkTileInfo."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


class MkGdnFront:
    """
    Low-latency Blackwell GEMM kernel rewritten with cute_ext primitives,
    keeping the raw-mbarrier 7-warp specialization from the C++ kernel
    in ``include/flashinfer/gemm/tgv_gemm.cuh``.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = _TGV_CUTE_EXT_CTA_K,
        num_ab_stage: int = 8,
        use_2cta: bool = False,
        use_pdl: bool = False,
        pdl_launch: Optional[bool] = None,
        pdl_count: int = -1,
        has_bias: bool = False,
        out_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_2cta = use_2cta
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count

        self.has_bias = has_bias

        min_n, step_n = (16, 16) if use_2cta else (8, 8)
        if cta_n < min_n or cta_n > 256 or cta_n % step_n != 0:
            raise ValueError(
                f"cta_n={cta_n} invalid for use_2cta={use_2cta}: "
                f"bf16 K-major mma requires N ∈ [{min_n}, 256] step {step_n}"
            )

        self.threads_per_cta = 256
        if use_2cta:
            self.cluster_shape = (2, 1, 1)
            self.mma_tiler_mn = (cta_m * 2, cta_n)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
        else:
            self.cluster_shape = (1, 1, 1)
            self.mma_tiler_mn = (cta_m, cta_n)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    def __repr__(self) -> str:
        return (
            f"TgvGemmCuteExtKernel_cta{self.cta_m}x{self.cta_n}x{self.cta_k}"
            f"_2cta{int(self.use_2cta)}_pdl{int(self.use_pdl)}"
            f"_bias{int(self.has_bias)}"
            f"_out{self.out_dtype.__name__.lower()}"
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        conv_weight: cute.Tensor,
        conv_bias: cute.Tensor,
        conv_state: cute.Tensor,
        state_indices: cute.Tensor,
        window: cute.Tensor,
        intermediate_indices: cute.Tensor,
        conv_out: cute.Tensor,
        stream: cuda.CUstream,
    ):

        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(
            a,
            b,
            c,
            bias,
            conv_weight,
            conv_bias,
            conv_state,
            state_indices,
            window,
            intermediate_indices,
            conv_out,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=None,
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mBias: cute.Tensor,
        conv_weight: cute.Tensor,
        conv_bias: cute.Tensor,
        conv_state: cute.Tensor,
        state_indices: cute.Tensor,
        window: cute.Tensor,
        intermediate_indices: cute.Tensor,
        conv_out: cute.Tensor,
    ):
        """
        Device-side dispatcher: build MMA descriptor, allocate SMEM/TMEM/
        mbarriers, init barriers, dispatch into dma_a/dma_b/mma/epilog warps.
        """
        DMA_Stage = self.num_ab_stage

        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type
        c_dtype = mC.element_type
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)

        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        sE = cute_ext.allocate(
            cutlass.BFloat16,
            cute.AddressSpace.smem,
            cute.make_layout((64, self.cta_n), stride=(1, 64)),
            alignment=128,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=1,
        )

        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )

        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator
        bar_tma_epilog = bar_tma_epilog_arr.iterator
        bar_mma_epilog = bar_mma_epilog_arr.iterator
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator
        tmem_base_ptr = tmem_base_arr.iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        gA_tile = cute.local_tile(
            mA,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        gBias_tile = cute.local_tile(
            mBias,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_tile,
                sA,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                gBias_tile,
                epi_tid,
                c_dtype,
                d_layout,
                conv_weight,
                conv_bias,
                conv_state,
                state_indices,
                window,
                intermediate_indices,
                conv_out,
                sE,
            )

        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,
        bar_empty,
        leader_rank: cutlass.Int32,
        gA_tile: cute.Tensor,
        sA: cute.Tensor,
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        sA.element_type,
                        cute.slice_(sA.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],
                sA[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

            if cutlass.const_expr(self.use_pdl):
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,
        sB: cute.Tensor,
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],
                sB[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)

        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc_view = tAcc[None, None, None, 0]

        if is_leader:
            mma_atom = cute.make_mma_atom(tiled_mma.op)

            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)

                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    a_frag = cute.append_ones(
                        sA[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)

                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1

            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        conv_weight: cute.Tensor,
        conv_bias: cute.Tensor,
        conv_state: cute.Tensor,
        state_indices: cute.Tensor,
        window: cute.Tensor,
        intermediate_indices: cute.Tensor,
        conv_out: cute.Tensor,
        sE: cute.Tensor,
    ):

        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc_view = tCtAcc[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_view)

        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            gBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tiled_copy_g2r = cute.make_tiled_copy_D(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                tiled_copy_t2r,
            )
            thr_g2r = tiled_copy_g2r.get_slice(epi_tid)
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )

        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_wait(bar_tma_epilog, 0)

        if cutlass.const_expr(self.has_bias):
            cute_ext.partition_and_copy(thr_g2r, gBias_epi[None, None, 0, 0], rBias)
            rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)

        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)

        cute.arch.fence_view_async_tmem_load()

        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.has_bias):
            rAcc.store(rAcc.load() + rBiasAcc.load())

        rD.store(rAcc.load().to(c_dtype))

        cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        feature_base = cute.arch.block_idx()[0] * 64
        if feature_base < self.qkv_channels:
            cute_ext.partition_and_copy(thr_t2r, rD, sE)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            for local_request in cutlass.range_constexpr(self.cta_n // 4):
                request = cute.arch.block_idx()[1] * (self.cta_n // 4) + local_request
                if request < self.verify_rows // 4:
                    if epi_tid < 64:
                        channel = feature_base + epi_tid
                        slot = cutlass.Int64(state_indices[request])
                        if slot != self.pad_slot:
                            inter = cutlass.Int64(intermediate_indices[request])
                            c0 = conv_state[slot, channel, 0]
                            c1 = conv_state[slot, channel, 1]
                            c2 = conv_state[slot, channel, 2]
                            w0 = conv_weight[channel, 0]
                            w1 = conv_weight[channel, 1]
                            w2 = conv_weight[channel, 2]
                            w3 = conv_weight[channel, 3]
                            bias_value = cutlass.Float32(0.0)
                            if cutlass.const_expr(self.conv_has_bias):
                                bias_value = cutlass.Float32(conv_bias[channel])
                            for token in cutlass.range_constexpr(4):
                                xt = sE[epi_tid, local_request * 4 + token]

                                acc = _add_product(bias_value, c0, w0)
                                acc = _add_product(acc, c1, w1)
                                acc = _add_product(acc, c2, w2)
                                acc = _add_product(acc, xt, w3)
                                c0 = c1
                                c1 = c2
                                c2 = xt
                                window[inter, token, channel, 0] = c0
                                window[inter, token, channel, 1] = c1
                                window[inter, token, channel, 2] = c2
                                conv_out[request * 4 + token, channel] = (
                                    cutlass.BFloat16(_silu(acc))
                                )

                            conv_state[slot, channel, 0] = c0
                            conv_state[slot, channel, 1] = c1
                            conv_state[slot, channel, 2] = c2
