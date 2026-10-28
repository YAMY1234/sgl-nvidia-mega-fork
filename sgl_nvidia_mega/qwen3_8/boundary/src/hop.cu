// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// HC adaptation of Qwen3.5's multicast decode hop. The mailbox protocol is
// independent of FI/CA. Keep the BF16 TP-sum cut before HC's gated FMA.
#include "fn_hop.cuh"

extern "C" __device__ float __nv_expf(float);

namespace fnhop_kernels {
using namespace fnhop;

struct Args {
  Site site;
  const uint16_t* partial;
  const uint16_t* residual;
  const float* gates;
  const uint16_t* gamma;
  uint16_t* reduced;
  uint16_t* combined;
  uint16_t* normalized;
  int rows;
  float eps;
};

struct BranchSmem {
  float xs[4][kRowCtas][kRowWarps];
  uint64_t mbar;
};

__device__ __forceinline__ uint4 ld4(const void* p) {
  return *static_cast<const uint4*>(p);
}

__device__ __forceinline__ float inject_gate(const float* partials, int row, int branch) {
  float sum = 0.f;
#pragma unroll
  for (int split = 0; split < 8; ++split)
    sum = __fadd_rn(sum, partials[(row * 8 + split) * 4 + branch]);
  return __fdiv_rn(2.f, __fadd_rn(1.f, __nv_expf(__fmul_rn(-sum, .25f))));
}

// Exactly one cluster initialization for all four branches, before the input wait.
__device__ __forceinline__ void branch_init(BranchSmem& rs) {
  if (threadIdx.x == 0) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(static_cast<uint32_t>(sizeof(rs.xs))) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\t"
               "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Q35's warp butterfly and asynchronous DSM exchange, extended to four RMS rows.
// Masked fragment lanes participate with zero, including the final partial warp.
__device__ __forceinline__ void branch_fold(float (&sums)[4], BranchSmem& rs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t me = cluster_ctarank();
#pragma unroll
  for (int b = 0; b < 4; ++b) {
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
      sums[b] = __fadd_rn(sums[b], __shfl_xor_sync(0xffffffffu, sums[b], off));
    if (lane < kRowCtas) {
      const uint32_t slot_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.xs[b][me][warp]));
      const uint32_t mbar_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
      uint32_t slot_r, mbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                   : "=r"(slot_r) : "r"(slot_l), "r"(lane));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                   : "=r"(mbar_r) : "r"(mbar_l), "r"(lane));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                   :: "r"(slot_r), "f"(sums[b]), "r"(mbar_r) : "memory");
    }
  }
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
  asm volatile("{\n\t.reg .pred P;\n\tFN_WAIT_BRANCH:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n\t"
               "@!P bra FN_WAIT_BRANCH;\n\t}" :: "r"(a) : "memory");
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    float total = 0.f;
#pragma unroll
    for (int c = 0; c < kRowCtas; ++c)
      total = __fadd_rn(total, __fadd_rn(__fadd_rn(rs.xs[b][c][0], rs.xs[b][c][2]),
                                       __fadd_rn(rs.xs[b][c][1], rs.xs[b][c][3])));
    sums[b] = total;
  }
}

__device__ __forceinline__ void publish(const Args& a) {
  const int row = blockIdx.x;
  pdl_wait();
  if (threadIdx.x < kFragsPerCta) {
    const int frag = blockIdx.y * kFragsPerCta + threadIdx.x;
    const size_t offset = static_cast<size_t>(row) * kHidden + frag * kVec;
    push_frag(a.site, stage_peek(a.site), row, frag, ld4(a.partial + offset));
  }
  pdl_trigger();
}

__device__ __forceinline__ void consume(const Args& a) {
  __shared__ BranchSmem rs;
  const int row = blockIdx.x;
  const int frag = blockIdx.y * kFragsPerCta + threadIdx.x;
  const bool active = threadIdx.x < kFragsPerCta;
  const unsigned fragment_mask = __ballot_sync(0xffffffffu, active);
  const size_t h = static_cast<size_t>(frag) * kVec;
  pdl_trigger();
  float gam[4][8];
  if (active) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      unpack8(ld4(a.gamma + b * kHidden + h), gam[b]);
#pragma unroll
      for (int e = 0; e < 8; ++e) gam[b][e] = __fadd_rn(1.f, gam[b][e]);
    }
  }
  branch_init(rs);
  pdl_wait();
  const Hop state = hop_enter(a.site);
  float pre[4][8];
  float sums[4] = {};
  if (active) {
    float block[8];
    pull_frag<4>(a.site, state.stage, row, frag, block);
    // Polling is lane-divergent. The communication mask must describe all
    // fragment owners, rather than whichever lanes happen to have returned.
    __syncwarp(fragment_mask);
    const uint4 block_bf16 = pack8(block);
    *reinterpret_cast<uint4*>(a.reduced + static_cast<size_t>(row) * kHidden + h) = block_bf16;
    unpack8(block_bf16, block);  // FN requires this BF16 materialization before injection.
    const int lane = threadIdx.x & 31;
    const float own_gate = lane < 4 ? inject_gate(a.gates, row, lane) : 0.f;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const size_t offset = (static_cast<size_t>(row) * 4 + b) * kHidden + h;
      float residual[8];
      unpack8(ld4(a.residual + offset), residual);
      const float gate = __shfl_sync(fragment_mask, own_gate, b);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        pre[b][e] = __fmaf_rn(gate, block[e], residual[e]);
      const uint4 combined = pack8(pre[b]);
      *reinterpret_cast<uint4*>(a.combined + offset) = combined;
      unpack8(combined, pre[b]);  // RMS reads BF16 combined, not the FP32 FMA.
#pragma unroll
      for (int e = 0; e < 8; ++e)
        sums[b] = __fadd_rn(sums[b], __fmul_rn(pre[b][e], pre[b][e]));
    }
  }
  branch_fold(sums, rs);
  if (active) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float inv = rsqrtf(__fadd_rn(sums[b] / static_cast<float>(kHidden), a.eps));
      float norm[8];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        norm[e] = __fmul_rn(__fmul_rn(pre[b][e], inv), gam[b][e]);
      const size_t offset = (static_cast<size_t>(row) * 4 + b) * kHidden + h;
      *reinterpret_cast<uint4*>(a.normalized + offset) = pack8(norm);
    }
    rearm_frag<4>(a.site, state.stage, row, frag);
  }
  hop_leave(a.site, state, gridDim.x * gridDim.y);
}
}  // namespace fnhop_kernels

extern "C" __global__ __launch_bounds__(128)
void fn_hop_publish(fnhop_kernels::Args args) { fnhop_kernels::publish(args); }

extern "C" __global__ __launch_bounds__(128)
void fn_hop_consume_w4(fnhop_kernels::Args args) { fnhop_kernels::consume(args); }

extern "C" __global__ void fn_hop_evict(uint32_t* p, unsigned n) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(v) : "l"(p+i) : "memory");
    asm volatile("st.global.wb.u32 [%0],%1;" :: "l"(p+i), "r"(v^1u) : "memory");
  }
}
