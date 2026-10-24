// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/cute_dsl_kernels/blackwell/
// q35_draft_head_gemv.py (release 4d09932ca7, Q35DraftHeadGemvKernel with MXFP8 weights and
// argmax off), translated from CuTe DSL to CUDA C++ with the same tiles, MMA fragments, rounding
// points and fp32 summation order (its logits bit for bit); mkd_head.cu's per-CTA packed maxima
// and row walk for the one-row head of the MKD draft step boundary.
//
// PERF-LP draft lm_head from a draft-only MXFP8 copy of this rank's vocabulary shard (E4M3
// values, one UE8M0 scale per 32 K-elements; TRT-LLM's TRTLLM_Q35_DRAFT_FP8_HEAD format):
// logits[t, r] = bf16(sum_k x[t, k] * w[r, k] * 2^(s[r, k / 32] - 127)) for at most 16 rows of x.
//
// Persistent CTAs stream the weights in groups of 8 rows (CTA c takes groups c, c + C, ...), two
// groups per 16-row tile, bulk-copied (one TMA bulk copy per weight row and per scale row) into
// two shared-memory slots; the first two tiles are issued before the PDL wait. The weights are
// the M side of mma.sync.m16n8k16 (f16 x f16 -> f32) and the activation the N side (8 rows per
// MMA tile). K is split over 16 warps of 8 scale blocks; the activation is converted to f16 once,
// row t times 2^e_t so its absolute max lands in [2^14, 2^15), so every product is exact
// (E4M3 converts exactly to f16). Per 32-element block two MMAs from zero, scaled by the block's
// power of two into an fp32 accumulator; the warps' partials are summed in warp order and scaled
// back by 2^-e_t: only the fp32 summation order differs from a dequantized reference.
//
// Built with NVRTC for hidden 4096: mkd_head_mx8 (one row; bf16 logits and per CTA the packed
// maximum of its rows, as mkd_head) and head_mx8_m<M> for M = 1..16 rows (bf16 logits).
// HEAD_MX8_TMA_WARPS 1: every warp issues one weight row and its scale row of a tile (one copy
// pair per warp) instead of 16 lanes of warp 0 (TRT-LLM's; the copies of a warp serialize).

#include "mkd_common.cuh"

#ifndef HEAD_MX8_TMA_WARPS
#define HEAD_MX8_TMA_WARPS 1
#endif

namespace sglang {

namespace head_mx8 {

inline constexpr int kBlock = 32;        // K elements per UE8M0 scale: two m16n8k16 steps
inline constexpr int kRows = 16;         // weight rows per tile (the MMA's M)
inline constexpr int kGroupRows = 8;     // weight rows per group: a tile is two groups
inline constexpr int kTokens = 8;        // activation rows per MMA tile (the MMA's N)
inline constexpr int kHeadroomExp = 14;  // per-row activation max scaled into [2^14, 2^15)
inline constexpr int kRowPad = 32;       // rows 4 apart start 128 bytes apart: no bank conflicts
// Packed partial of a CTA whose rows hold a NaN logit (mkd_head::kNanPart).
inline constexpr uint32_t kNanPart = 0xffffffffu;

template <int kK, int kSliceBlocks, int kTokenTiles>
struct Layout {
  static constexpr int kWarps = kK / (kBlock * kSliceBlocks);
  static constexpr int kRowStride = kK + kRowPad;  // shared bytes per weight row
  static constexpr int kScaleRow = kK / kBlock;    // scale bytes per weight row
  static constexpr int kTokensPad = kTokens * kTokenTiles;
  static constexpr int kPartFloats = 2 * kWarps * 4 * kTokenTiles * 32;
  // byte offsets in dynamic shared memory
  static constexpr int kW = 0;                                // [2][kRows][kRowStride]
  static constexpr int kS = kW + 2 * kRows * kRowStride;      // [2][kRows][kScaleRow]
  static constexpr int kPart = kS + 2 * kRows * kScaleRow;    // f32 [2][kWarps][4 * tiles][32]
  static constexpr int kMbar = kPart + kPartFloats * 4;       // u64 [2]
  static constexpr int kAmax = kMbar + 16;                    // u32 [kTokensPad]
  static constexpr int kBest = kAmax + kTokensPad * 4;        // u32 [kWarps]
  static constexpr int kBytes = kBest + kWarps * 4;
  static_assert(kWarps >= 4 && kWarps <= 32 && kK % (kBlock * kSliceBlocks) == 0);
  static_assert(kRowStride % 16 == 0 && kS % 128 == 0 && kPart % 16 == 0 && kMbar % 8 == 0);
};

SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// The tile's phase, with CuTe DSL's 10 ms suspend hint (the warp sleeps until the phase flips
// instead of spinning) and a memory clobber: the tile's shared-memory reads stay after it.
SGL_DEVICE void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1, 10000000;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(device::ptx::to_shared(bar)),
      "r"(parity)
      : "memory");
}

SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// bytes (a multiple of 16, both addresses 16-byte aligned) from global src to shared dst,
// completing as transaction bytes on the shared mbarrier bar
SGL_DEVICE void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(dst)),
      "l"(src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

SGL_DEVICE void red_max_shared(uint32_t* ptr, uint32_t value) {
  asm volatile("red.shared.max.u32 [%0], %1;" ::"r"(device::ptx::to_shared(ptr)), "r"(value) : "memory");
}

// four packed E4M3 values -> four f16 values as two packed words (exact)
SGL_DEVICE void e4m3x4_to_f16x4(uint32_t x, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t"
      ".reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t"
      "}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(x));
}

// f16x2 {lo, hi} of two fp32 values, each rounded to nearest even
SGL_DEVICE uint32_t f16x2_rn(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32: a 8 f16 (4 words), b 4 f16 (2 words)
SGL_DEVICE void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, const float (&c)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

SGL_DEVICE float bf16_lo(uint32_t w) {
  return __uint_as_float(w << 16);
}

SGL_DEVICE float bf16_hi(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

// 2^e as an fp32 bit pattern (|e| <= 126)
SGL_DEVICE float exp2i(int e) {
  return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

/// \brief 16-bit order key of one bf16 value (-0.0 as +0.0); sets nan for a NaN (mkd_head::key16).
SGL_DEVICE uint32_t key16(bf16_t v, bool& nan) {
  uint32_t b = static_cast<uint32_t>(__bfloat16_as_ushort(v));
  nan |= (b & 0x7fffu) > 0x7f80u;
  b = b == 0x8000u ? 0u : b;
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

}  // namespace head_mx8

struct HeadMx8Params {
  const bf16_t* x;                // [m, K] rows of the final-norm hidden state, written by the previous grid
  const uint8_t* __restrict__ w;  // [n, K] E4M3 values of the shard copy
  const uint8_t* __restrict__ s;  // [n, K / 32] UE8M0 block scales
  bf16_t* logits;                 // [m, n]
  uint32_t* part;                 // kMkd: [gridDim.x] packed (key16 << 16) | (0xffff - row), or kNanPart
  int32_t n;
  int32_t m;        // rows of x (the kernel's kM)
  int32_t reverse;  // 1: walk the groups from the end (alternate draft steps, as mkd_head)
};

/**
 * \brief kM (<= 16) rows of x against the n weight rows on gridDim.x persistent CTAs of
 * kK / (32 * kSliceBlocks) warps (gridDim.x <= ceil(n / 8)). kMkd (kM == 1): mkd_head's PDL
 * points and per-CTA packed maxima; else TRT-LLM's (trigger once the activation is staged).
 */
template <int kK, int kSliceBlocks, int kM, bool kMkd>
SGL_DEVICE void head_mx8_kernel(const HeadMx8Params& p) {
  using namespace device;
  using namespace head_mx8;
  constexpr int kTokenTiles = (kM + kTokens - 1) / kTokens;
  using L = Layout<kK, kSliceBlocks, kTokenTiles>;
  static_assert(kM >= 1 && kM <= 2 * kTokens && (!kMkd || kM == 1));
  static_assert(kSliceBlocks == 8, "one 8-byte scale load per weight row and warp");
  static_assert(!HEAD_MX8_TMA_WARPS || L::kWarps >= kRows, "a warp per tile row");
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_w = smem + L::kW;
  uint8_t* s_s = smem + L::kS;
  float* s_part = reinterpret_cast<float*>(smem + L::kPart);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + L::kMbar);
  uint32_t* s_amax = reinterpret_cast<uint32_t*>(smem + L::kAmax);
  uint32_t* s_best = reinterpret_cast<uint32_t*>(smem + L::kBest);

  const int tidx = static_cast<int>(threadIdx.x);
  const int warp = tidx / 32;
  const int lane = tidx % 32;
  const int g = lane / 4;
  const int q = lane % 4;
  const int n = p.n;
  const int ctas = static_cast<int>(gridDim.x);
  const int bidx = static_cast<int>(blockIdx.x);
  const int num_groups = (n + kGroupRows - 1) / kGroupRows;
  const int my_groups = bidx < num_groups ? (num_groups - bidx + ctas - 1) / ctas : 0;
  const int my_tiles = (my_groups + 1) / 2;
  // group of this CTA's local group j; reversed calls walk the groups from the end
  auto group_of = [&](int j) {
    const int gg = bidx + j * ctas;
    return p.reverse ? num_groups - 1 - gg : gg;
  };
  // the issuing threads: lanes 0..15 of warp 0 (TRT-LLM's), or lane 0 of warps 0..15
  const bool issuer = HEAD_MX8_TMA_WARPS ? (lane == 0 && warp < kRows) : warp == 0;
  // tile `tile` (local groups 2 tile, 2 tile + 1) into `slot`: one weight row (row r of the
  // tile: the issuing lane or warp) and its scale row each; rows past n re-read row n - 1
  // (their outputs are not stored)
  auto issue_tile = [&](int slot, int tile) {
    constexpr uint32_t kRowBytes = kK + L::kScaleRow;
    const int r = HEAD_MX8_TMA_WARPS ? warp : lane;
    if (r == 0 && (HEAD_MX8_TMA_WARPS || lane == 0)) {
      const int groups = min(my_groups - 2 * tile, 2);
      ptx::mbar_arrive_expect_tx(&mbar[slot], static_cast<uint32_t>(groups * kGroupRows) * kRowBytes);
    }
    if (r < kRows) {
      const int local = 2 * tile + r / kGroupRows;
      if (local < my_groups) {
        const int row = min(group_of(local) * kGroupRows + r % kGroupRows, n - 1);
        bulk_g2s(s_w + (slot * kRows + r) * L::kRowStride, p.w + static_cast<int64_t>(row) * kK, kK, &mbar[slot]);
        bulk_g2s(
            s_s + (slot * kRows + r) * L::kScaleRow,
            p.s + static_cast<int64_t>(row) * L::kScaleRow,
            L::kScaleRow,
            &mbar[slot]);
      }
    }
  };

  if (tidx == 0) {
    ptx::mbar_init(&mbar[0], 1);
    ptx::mbar_init(&mbar[1], 1);
    fence_mbarrier_init();
  }
  if (tidx < L::kTokensPad) s_amax[tidx] = 0u;
  __syncthreads();

  // ---- the first two tiles (static weights: before the PDL wait) ----
  if (issuer && my_tiles > 0) {
    issue_tile(0, 0);
    if (my_tiles > 1) issue_tile(1, 1);
  }
  PDLWaitPrimary<true>();
  if constexpr (kMkd) PDLTriggerSecondary<true>();

  // ---- this warp's activation slice as f16 B fragments ----
  // act[t][b][j]: row 8t + g, K elements 32 (warp * kSliceBlocks + b) + 8q + 2j, + 1; loaded as
  // bf16 pairs (zeros past kM) and converted in place
  uint32_t act[kTokenTiles][kSliceBlocks][4];
#pragma unroll
  for (int t = 0; t < kTokenTiles; ++t) {
    const int row = kTokens * t + g;
#pragma unroll
    for (int b = 0; b < kSliceBlocks; ++b) {
      const uint4 v = row < kM ? reinterpret_cast<const uint4*>(
                                     p.x + static_cast<int64_t>(row) * kK)[(warp * kSliceBlocks + b) * 4 + q]
                               : make_uint4(0u, 0u, 0u, 0u);
      act[t][b][0] = v.x;
      act[t][b][1] = v.y;
      act[t][b][2] = v.z;
      act[t][b][3] = v.w;
    }
  }
#pragma unroll
  for (int t = 0; t < kTokenTiles; ++t) {
    float amax = 0.0f;
#pragma unroll
    for (int b = 0; b < kSliceBlocks; ++b) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        amax = fmaxf(amax, fabsf(bf16_lo(act[t][b][j])));
        amax = fmaxf(amax, fabsf(bf16_hi(act[t][b][j])));
      }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (q == 0 && kTokens * t + g < kM) red_max_shared(&s_amax[kTokens * t + g], __float_as_uint(amax));
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < kTokenTiles; ++t) {
    const int shift = max(min(kHeadroomExp + 127 - static_cast<int>(s_amax[kTokens * t + g] >> 23), 126), -126);
    const float factor = exp2i(shift);
#pragma unroll
    for (int b = 0; b < kSliceBlocks; ++b) {
#pragma unroll
      for (int j = 0; j < 4; ++j) act[t][b][j] = f16x2_rn(bf16_lo(act[t][b][j]) * factor, bf16_hi(act[t][b][j]) * factor);
    }
  }
  // the output (token, MMA row half) this thread reduces when tidx < 128 * kTokenTiles, and 2^-e
  // of its token
  const int out_e = (tidx / 32) % 4;
  const int out_token = kTokens * (tidx / 128) + 2 * q + out_e % 2;
  float inv;
  {
    const int shift = max(min(kHeadroomExp + 127 - static_cast<int>(s_amax[out_token % L::kTokensPad] >> 23), 126), -126);
    inv = exp2i(-shift);
  }
  const bool reducer = tidx < 128 * kTokenTiles && out_token < kM;
  if constexpr (!kMkd) PDLTriggerSecondary<true>();

  // ---- stream the tiles ----
  const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  uint32_t best = 0u;
  bool nan = false;
  for (int i = 0; i < my_tiles; ++i) {
    const int slot = i % 2;
    mbar_wait(&mbar[slot], static_cast<uint32_t>((i / 2) % 2));
    // weight rows g and g + 8 of the tile, this warp's slice: 8 bytes of block b at 32 b + 8q
    const uint8_t* w_lo = s_w + (slot * kRows + g) * L::kRowStride + warp * kSliceBlocks * kBlock + 8 * q;
    const uint8_t* w_hi = w_lo + kGroupRows * L::kRowStride;
    // the 8 scale bytes of this warp's slice, rows g and g + 8: one 8-byte load each
    const uint8_t* sc_row = s_s + (slot * kRows + g) * L::kScaleRow + warp * kSliceBlocks;
    const uint2 sc_lo = *reinterpret_cast<const uint2*>(sc_row);
    const uint2 sc_hi = *reinterpret_cast<const uint2*>(sc_row + kGroupRows * L::kScaleRow);
    float acc[kTokenTiles][4];
#pragma unroll
    for (int t = 0; t < kTokenTiles; ++t) {
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[t][e] = 0.0f;
    }
#pragma unroll
    for (int b = 0; b < kSliceBlocks; ++b) {
      const uint2 wl = *reinterpret_cast<const uint2*>(w_lo + b * kBlock);
      const uint2 wh = *reinterpret_cast<const uint2*>(w_hi + b * kBlock);
      uint32_t lo01[2], lo23[2], hi01[2], hi23[2];
      e4m3x4_to_f16x4(wl.x, lo01[0], lo01[1]);
      e4m3x4_to_f16x4(wl.y, lo23[0], lo23[1]);
      e4m3x4_to_f16x4(wh.x, hi01[0], hi01[1]);
      e4m3x4_to_f16x4(wh.y, hi23[0], hi23[1]);
      // step j: (row g, 4j..4j+1), (row g+8, same), (row g, 4j+2..4j+3), (row g+8, same)
      const uint32_t a0[4] = {lo01[0], hi01[0], lo01[1], hi01[1]};
      const uint32_t a1[4] = {lo23[0], hi23[0], lo23[1], hi23[1]};
      // UE8M0 -> the bit pattern of 2^(s - 127)
      const float scale_lo = __uint_as_float((((b < 4 ? sc_lo.x : sc_lo.y) >> (8 * (b % 4))) & 0xffu) << 23);
      const float scale_hi = __uint_as_float((((b < 4 ? sc_hi.x : sc_hi.y) >> (8 * (b % 4))) & 0xffu) << 23);
#pragma unroll
      for (int t = 0; t < kTokenTiles; ++t) {
        float part[4];
        mma_16816(part, a0, act[t][b][0], act[t][b][1], zero);
        mma_16816(part, a1, act[t][b][2], act[t][b][3], part);
        acc[t][0] = acc[t][0] + scale_lo * part[0];
        acc[t][1] = acc[t][1] + scale_lo * part[1];
        acc[t][2] = acc[t][2] + scale_hi * part[2];
        acc[t][3] = acc[t][3] + scale_hi * part[3];
      }
    }
#pragma unroll
    for (int t = 0; t < kTokenTiles; ++t) {
#pragma unroll
      for (int e = 0; e < 4; ++e) s_part[((slot * L::kWarps + warp) * 4 * kTokenTiles + 4 * t + e) * 32 + lane] = acc[t][e];
    }
    // every warp is done with this slot's weights and has written its partial tile
    __syncthreads();
    if (issuer && i + 2 < my_tiles) {
      fence_proxy_async_shared();
      issue_tile(slot, i + 2);
    }
    const int local = 2 * i + out_e / 2;  // MMA rows 0..7: the tile's first group, 8..15: its second
    if (reducer && local < my_groups) {
      const int row = group_of(local) * kGroupRows + g;
      if (row < n) {
        // element out_e of the lane's MMA tile, summed over the warps in warp order
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < L::kWarps; ++w) total = total + s_part[((slot * L::kWarps + w) * 4 * kTokenTiles + tidx / 32) * 32 + lane];
        const bf16_t v = __float2bfloat16_rn(total * inv);
        p.logits[static_cast<int64_t>(out_token) * n + row] = v;
        if constexpr (kMkd) {
          const uint32_t k = key16(v, nan);
          best = max(best, (k << 16) | (0xffffu - static_cast<uint32_t>(row)));
        }
      }
    }
  }

  if constexpr (kMkd) {
    best = __reduce_max_sync(0xffffffffu, best);
    const uint32_t nan_any = __reduce_or_sync(0xffffffffu, nan ? 1u : 0u);
    if (lane == 0) s_best[warp] = nan_any != 0u ? kNanPart : best;
    __syncthreads();
    if (tidx == 0) {
      uint32_t b = s_best[0];
#pragma unroll
      for (int w = 1; w < L::kWarps; ++w) b = max(b, s_best[w]);
      p.part[bidx] = b;
    }
  }
}

}  // namespace sglang

using sglang::HeadMx8Params;

extern "C" __global__ void __launch_bounds__(16 * 32, 1) mkd_head_mx8(const HeadMx8Params __grid_constant__ p) {
  sglang::head_mx8_kernel<4096, 8, 1, true>(p);
}

#define HEAD_MX8_GEMV(rows)                                                                       \
  extern "C" __global__ void __launch_bounds__(16 * 32, 1) head_mx8_m##rows(const HeadMx8Params __grid_constant__ p) { \
    sglang::head_mx8_kernel<4096, 8, rows, false>(p);                                           \
  }

HEAD_MX8_GEMV(1)
HEAD_MX8_GEMV(2)
HEAD_MX8_GEMV(3)
HEAD_MX8_GEMV(4)
HEAD_MX8_GEMV(5)
HEAD_MX8_GEMV(6)
HEAD_MX8_GEMV(7)
HEAD_MX8_GEMV(8)
HEAD_MX8_GEMV(9)
HEAD_MX8_GEMV(10)
HEAD_MX8_GEMV(11)
HEAD_MX8_GEMV(12)
HEAD_MX8_GEMV(13)
HEAD_MX8_GEMV(14)
HEAD_MX8_GEMV(15)
HEAD_MX8_GEMV(16)
