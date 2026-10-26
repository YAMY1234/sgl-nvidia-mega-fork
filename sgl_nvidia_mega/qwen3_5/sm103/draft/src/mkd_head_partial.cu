// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Q35FLY: the hidden-half FC partial inside the one-row head's prefetch window.
// Every head CTA first computes its disjoint FC rows, then resumes its head rows.
// The norm retains mkd_fc's 256-thread statistics and every dot retains slice_dot's
// FMA and warp reduction order. Neither the head's arithmetic nor its keys change.

#include "mkd_fc.cuh"
#include "mkd_head.cuh"
#include "head_mx8.cuh"

namespace sglang {

struct MkdHeadPartialInputWork {
  MkdHiddenPartialParams p;

  SGL_DEVICE void operator()() const {
    using namespace device;
    constexpr int kH = 4096, kThreads = 512, kNormThreads = 256;
    constexpr int kW = 2 * kH / MKD_WORLD;
    constexpr int kWarps = kThreads / 32, kNormWarps = kNormThreads / 32;
    // All SMs participate now: one row in flight avoids retaining the old
    // side-stream four-row register footprint beside BF16 head prefetches.
    constexpr int kDotVecs = kW / (8 * 32), kRowsPerIter = 1;
    using vec_t = AlignedVector<bf16_t, 8>;
    __shared__ alignas(16) bf16_t s_x[kH];
    __shared__ float s_red[kNormWarps];
    const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;

    // The first eight warps own exactly the original norm's vectors. All sixteen
    // warps reach both CTA barriers; inactive norm warps must not index s_red.
    if (tid < kNormThreads) {
      float ss = 0.0f;
#pragma unroll
      for (int k = 0; k < kH / (8 * kNormThreads); ++k) {
        vec_t v;
        v.load(p.h, k * kNormThreads + tid);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float f = cast<fp32_t>(v[j]);
          ss = fmaf(f, f, ss);
        }
      }
      ss = warp::reduce_sum(ss);
      if (lane == 0) s_red[warp] = ss;
    }
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < kNormWarps; ++w) tot += s_red[w];
    const float inv = rsqrtf(tot / static_cast<float>(kH) + p.eps);
    for (int k = tid; k < kH / 8; k += kThreads) {
      vec_t v, g, o;
      v.load(p.h, k);
      g.load(p.norm_w, k);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = cast<bf16_t>(cast<fp32_t>(v[j]) * inv * (1.0f + cast<fp32_t>(g[j])));
      o.store(s_x, k);
    }
    __syncthreads();
    const int stride = static_cast<int>(gridDim.x) * kWarps;
    for (int j0 = static_cast<int>(blockIdx.x) * kWarps + warp;
         j0 < kH; j0 += stride * kRowsPerIter) {
      vec_t wv[kRowsPerIter][kDotVecs];
#pragma unroll
      for (int r = 0; r < kRowsPerIter; ++r) {
        const int j = j0 + r * stride;
        if (j < kH) {
#pragma unroll
          for (int k = 0; k < kDotVecs; ++k)
            wv[r][k].load(p.w + static_cast<int64_t>(j) * kW, k * 32 + lane);
        }
      }
#pragma unroll
      for (int r = 0; r < kRowsPerIter; ++r) {
        const int j = j0 + r * stride;
        if (j < kH) {
          const float acc = mkd_fc::slice_dot_regs<kW>(wv[r], s_x + p.col);
          if (lane == 0) p.out[j] = acc;
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      if (atomicAdd(p.done, 1u) == gridDim.x - 1) {
        *p.done = 0u;
        uint32_t epoch;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(epoch) : "l"(p.epoch) : "memory");
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p.flag), "r"(epoch) : "memory");
      }
    }
  }
};

struct MkdHeadPartialBf16Params {
  MkdHeadParams head;
  MkdHiddenPartialParams partial;
};
struct MkdHeadPartialMx8Params {
  HeadMx8Params head;
  MkdHiddenPartialParams partial;
};
static_assert(sizeof(MkdHeadPartialBf16Params) == 104, "Python/CUDA BF16 ABI");
static_assert(sizeof(MkdHeadPartialMx8Params) == 144, "Python/CUDA MXFP8 ABI");

}  // namespace sglang

extern "C" __global__ void __launch_bounds__(512, 1) mkd_head_partial_bf16(
    const sglang::MkdHeadPartialBf16Params __grid_constant__ p) {
  sglang::mkd_head_kernel<4096, 16, true>(p.head, sglang::MkdHeadPartialInputWork{p.partial});
}

extern "C" __global__ void __launch_bounds__(512, 1) mkd_head_partial_mx8(
    const sglang::MkdHeadPartialMx8Params __grid_constant__ p) {
  sglang::head_mx8_kernel<4096, 8, 1, true>(p.head, sglang::MkdHeadPartialInputWork{p.partial});
}
