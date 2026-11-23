// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The GLM-5.3 / GLM-5.2 draft LM head shard at 1-LH_MAXROWS rows as a weight stream (lm_head_mxfp8.py):
// out[rows, N] = x[rows, K] @ W[N, K]^T, W read once from HBM as MXFP8 (E4M3 codes [N, K] + UE8M0 scales of
// 32 columns in FlashInfer's 128x4 swizzle), x BF16. Two kernels with the same arithmetic (LH_RING selects one).
//
// glm_draft_lm_head_stream (LH_RING 0, served): the DeepSeek-V4.1 family's stream kernel
// (deepseek_v41/sm103/src/lm_head_mxfp8.cu), with the same fragments, dequantization, MMAs and summation order, so
// the same inputs give the same bits. Changes for the draft's one or two rows:
// - x is staged for LH_MAXROWS rows only (shared memory and the staging loop scale with it), by 16-byte loads in
//   an unrolled loop, so the staging after the PDL wait is a few batched loads instead of a chain of dependent ones;
// - the B fragment of lanes whose column g is >= LH_MAXROWS reads row 0 (those columns are never stored);
// - LH_DQ 2 builds the same BF16 bits of a code word with byte permutes (sign replicated over a byte, times 16,
//   one mask: 4 integer-pipe ops a word instead of 5) and the scale pairs with a permute and a multiply-add.
// From the original: a warp owns a 16-row tile of W (mma.m16n8k16, W as A, the rows of x as the 8 columns of B)
// and LH_SPLIT warps of a CTA split its K; the CTA holds x in shared memory, its 4-column groups stored in the
// order (0, 2, 1, 3) so that a lane's B fragment is one 8-byte read. Per 128 columns of rows g and g + 8 a lane
// streams codes [16t, 16t + 16) and [64 + 16t, 80 + 16t) and the four scales as one word, LH_DEPTH chunks ahead;
// the first chunks load before the PDL wait. A code becomes BF16 bits exactly (sign, then exponent and mantissa
// shifted by 4: the value times 2^-120, subnormals included) and one BF16 multiply by 2^(scale - 7) gives the
// dequantized weight, which BF16 holds exactly. LH_FP8 0 reads a BF16 W through the same fragments and MMAs.
// Per tile: FP32 MMA accumulation in K order, the split parts added in part order, one BF16 rounding.
// Grid: CTAs <= SMs; tile t runs on CTA t % grid, slot t / grid. Block LH_WARPS * 32, PDL.
//
// glm_draft_lm_head_ring (LH_RING 1, MXFP8 only, N a multiple of 16; opt-in, not faster on GB300): the same chains
// fed through shared memory. A chain is (tile, split part): the LH_SPLIT parts of a tile are LH_SPLIT consecutive
// consumer warps, each running the stream kernel's per-warp sequence on its 128-column chunks (the same lane
// fragments in the same registers, the same MMAs in K order from zero), then the same part-order sum and rounding;
// so the outputs are the stream kernel's bit for bit at the same split (lanes whose B column g >= LH_MAXROWS take
// zeros instead of row 0: column g of an MMA's D depends on column g of B only, and those columns are never stored).
// What changes is who loads the bytes:
// - one producer warp copies, per stage, one 128-column chunk of every chain of the CTA's round (LH_WARPS / LH_SPLIT
//   tiles, LH_GROUP consecutive tiles a copy): the codes by tensor copies of the [N][K / 128][128] view
//   (cp.async.bulk.tensor, L2 promotion 256 B; a tile lands as 16 rows of 128 bytes, so the two rows of a quarter
//   warp's reads share banks: two-way conflicts) and the scale blocks by bulk copies, into a ring of LH_STAGES stages
//   with full / empty mbarriers (completion counted in bytes); the bytes in flight hold no registers and are pooled
//   over the CTA's chains, and the producer never waits for the PDL predecessor (weights and scales are static).
//   Copies of one tile each (LH_GROUP 1) are limited by the copy issue rate (about 4 TB/s on GB300); groups of 8
//   tiles reach the stream kernel's speed. (A conflict-free 5-D view with the row parity inside the 128-byte row
//   needs non-monotonic strides, which the driver refuses.)
// - consumers wait for the PDL predecessor, stage x as the stream kernel does, then per LH_UNROLL stages: wait full,
//   read their fragments from shared memory, release the stages (LH_RELEASE 1: right after the reads, the arrive's
//   release ordering them; 0: after the math), the stream kernel's dequantization and MMAs. Group g of a CTA's
//   groups runs in round g / (LH_WARPS / LH_SPLIT / LH_GROUP). LH_DIAG (timing only, wrong outputs): 1 no math,
//   2 no copies.
// LH_TRIGGER 1 signals dependents early (producer after its last copy, consumers after staging x); 0 at the end.
// Grid: CTAs <= SMs. Block (LH_WARPS + 1) * 32.

#include <cuda_bf16.h>

#ifndef LH_K
#define LH_K 5120
#endif
#ifndef LH_FP8
#define LH_FP8 1
#endif
#ifndef LH_SPLIT
#define LH_SPLIT 1
#endif
#ifndef LH_DEPTH
#define LH_DEPTH 4
#endif
#ifndef LH_WARPS
#define LH_WARPS 16
#endif
#ifndef LH_MAXROWS
#define LH_MAXROWS 8
#endif
#ifndef LH_RING
#define LH_RING 0
#endif
#ifndef LH_STAGES
#define LH_STAGES 5
#endif
#ifndef LH_TRIGGER
#define LH_TRIGGER 0
#endif
#ifndef LH_GROUP
#define LH_GROUP 1
#endif
#ifndef LH_UNROLL
#define LH_UNROLL 1
#endif
#ifndef LH_DIAG
#define LH_DIAG 0
#endif
#ifndef LH_DQ
#define LH_DQ 1
#endif
#ifndef LH_RELEASE
#define LH_RELEASE 0
#endif

namespace {

constexpr int kK = LH_K;
constexpr bool kFp8 = LH_FP8 != 0;
constexpr int kSplit = LH_SPLIT;
constexpr int kDepth = LH_DEPTH;
constexpr int kWarps = LH_WARPS;
constexpr int kSlots = kWarps / kSplit;
constexpr int kChunk = 128;  // columns per lane step: 4 lanes x 32
constexpr int kChunks = kK / kChunk / kSplit;  // per warp per tile
constexpr int kMaxRows = LH_MAXROWS;  // rows of x staged (<= 8: the MMA's columns)
// the instruction sequence building the same BF16 bits: 1 shifts and masks (the DeepSeek-V4.1 kernel's), 2 byte
// permutes with sign replication (one ALU op fewer a code word) and permuted scale pairs
constexpr int kDq = LH_DQ;
constexpr int kXStride = kK + 8;  // BF16 elements a shared row; +16 B spreads the banks
constexpr int kScaleCols = kK / 32;
// a stage: 32 codes (FP8) or 32 BF16 values of rows g and g + 8, as u32 words, in two
// segments of 16 columns 64 apart
constexpr int kWords = kFp8 ? 8 : 16;
constexpr int kSegmentBytes = kFp8 ? 16 : 32;

static_assert(kK % (kChunk * kSplit) == 0, "K splits into whole 128-column chunks");
static_assert(kChunks % kDepth == 0, "the prefetch depth divides the chunks");
static_assert(kWarps % kSplit == 0, "whole tiles a CTA");
static_assert(kMaxRows >= 1 && kMaxRows <= 8, "one MMA's 8 columns");
static_assert(kK % 8 == 0, "16-byte x loads");
static_assert(kSplit == 1 || kSlots < 16, "a named barrier per slot");

struct Stage {
  unsigned w[2][kWords];
  unsigned scales[2];  // the chunk's four UE8M0 bytes of each row
};

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint4 load_stream(const void* p) {
  uint4 v;
  // the 256-byte L2 prefetch brings the lane's next segments with this one
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned load_word(const unsigned char* p) {
  unsigned v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// FlashInfer's 128x4 swizzle of UE8M0 scales over rows padded to 128 (mxfp8_ll.quantize)
__device__ __forceinline__ long long scale_offset(int row, int block) {
  return ((long long)(row >> 7) * (kScaleCols >> 2) + (block >> 2)) * 512 + (row & 31) * 16 +
         ((row >> 5) & 3) * 4 + (block & 3);
}

// chunk columns [chunk, chunk + 128): this lane's segments start at chunk + 16t, + 64
__device__ __forceinline__ void load_stage(Stage& s, const unsigned char* w, const unsigned char* sf,
                                           int row0, int row1, int chunk, int t) {
  const int rows[2] = {row0, row1};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const unsigned char* p = w + ((long long)rows[h] * kK + chunk) * (kFp8 ? 1 : 2) + kSegmentBytes * t;
#pragma unroll
    for (int v = 0; v < kWords / 4; ++v) {
      // segment v / (kWords / 8): 64 columns on
      const int seg = v / (kWords / 8), part = v % (kWords / 8);
      uint4 q = load_stream(p + seg * 64 * (kFp8 ? 1 : 2) + 16 * part);
      s.w[h][4 * v] = q.x;
      s.w[h][4 * v + 1] = q.y;
      s.w[h][4 * v + 2] = q.z;
      s.w[h][4 * v + 3] = q.w;
    }
    if (kFp8) s.scales[h] = load_word(sf + scale_offset(rows[h], chunk >> 5));
  }
}

// 2^(scale - 7) as BF16 in both halves for scale byte i of the word: undoes the code's
// 2^-120 and applies the scale
__device__ __forceinline__ unsigned scale_pair(unsigned scales, int i) {
  unsigned bits = (((scales >> (8 * i)) & 0xFFu) + 120u) << 7;
  return bits | (bits << 16);
}

__device__ __forceinline__ unsigned bf16_mul(unsigned a, unsigned b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<unsigned*>(&r);
}

// codes (c0, c1, c2, c3) of a word -> BF16 pairs (c0, c2) and (c1, c3), dequantized
__device__ __forceinline__ void dequant(unsigned w, unsigned scale, unsigned& even, unsigned& odd) {
  const unsigned sign = 0x80008000u;
  unsigned e, o;
  if (kDq == 2) {
    // [s2 c2 s0 c0] and [s3 c3 s1 c1] (s: the code's sign over a byte), times 16: each half holds the sign at bit
    // 15, copies of it at 12-14, the code at 4-11 (its sign at 11) and the other half's spill at 0-3; the mask
    // keeps bit 15 and bits 4-10, the same bits as the shifts below
    e = (prmt(w, 0u, 0xA280u) * 16u) & 0x87F087F0u;
    o = (prmt(w, 0u, 0xB391u) * 16u) & 0x87F087F0u;
  } else {
    e = ((w << 8) & sign) | ((w & 0x007F007Fu) << 4);
    o = (w & sign) | ((w & 0x7F007F00u) >> 4);
  }
  even = bf16_mul(e, scale);
  odd = bf16_mul(o, scale);
}

// scale_pair's bits from a permute selector (scale byte i into bytes 0 and 2): ((byte + 120) << 7) in both halves,
// as byte * 128 + 120 * 128 per half (each half stays below 2^16)
__device__ __forceinline__ unsigned scale_pair_sel(unsigned scales, unsigned sel) {
  return prmt(scales, 0u, sel) * 128u + 0x3C003C00u;
}

__device__ __forceinline__ void mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// the chunk's 8 MMAs on a staged chunk; sc: the scale pairs [row half][segment] (FP8); xw: the 16 x words of
// this lane's B column (two segments)
__device__ __forceinline__ void mma_chunk(float (&acc)[4], const Stage& s, const unsigned (&sc)[2][2],
                                          const unsigned* xw) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    unsigned a0, a1, a2, a3;  // (row g: c0 c2), (row g+8: c0 c2), (row g: c1 c3), (row g+8: c1 c3)
    if (kFp8) {
      dequant(s.w[0][j], sc[0][j >> 2], a0, a2);
      dequant(s.w[1][j], sc[1][j >> 2], a1, a3);
    } else {
      a0 = prmt(s.w[0][2 * j], s.w[0][2 * j + 1], 0x5410);
      a2 = prmt(s.w[0][2 * j], s.w[0][2 * j + 1], 0x7632);
      a1 = prmt(s.w[1][2 * j], s.w[1][2 * j + 1], 0x5410);
      a3 = prmt(s.w[1][2 * j], s.w[1][2 * j + 1], 0x7632);
    }
    mma(acc, a0, a1, a2, a3, xw[2 * j], xw[2 * j + 1]);
  }
}

// xs: this lane's x (permuted) at chunk + 16t, for its column g of B
__device__ __forceinline__ void consume(float (&acc)[4], const Stage& s, const unsigned* xs, int t) {
  // the 32 x values of the two segments: words 0-7 at xs, 8-15 at xs + 32 (64 columns on)
  uint4 xv[4];
#pragma unroll
  for (int v = 0; v < 4; ++v) xv[v] = reinterpret_cast<const uint4*>(xs + (v >> 1) * 32)[v & 1];
  unsigned sc[2][2] = {{0, 0}, {0, 0}};  // [row half][segment]
  if (kFp8) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (kDq == 2) {
        sc[h][0] = scale_pair_sel(s.scales[h], 0x4040u | (0x0101u * (t >> 1)));
        sc[h][1] = scale_pair_sel(s.scales[h], 0x4040u | (0x0101u * (2 + (t >> 1))));
      } else {
        sc[h][0] = scale_pair(s.scales[h], t >> 1);
        sc[h][1] = scale_pair(s.scales[h], 2 + (t >> 1));
      }
    }
  }
  mma_chunk(acc, s, sc, reinterpret_cast<const unsigned*>(xv));
}

}  // namespace

#if !LH_RING

extern "C" __global__ void __launch_bounds__(kWarps * 32, 1)
    glm_draft_lm_head_stream(const unsigned char* __restrict__ weight, const unsigned char* __restrict__ scales,
                         const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int rows,
                         int x_stride, int out_stride) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* xs = reinterpret_cast<unsigned*>(smem);  // [kMaxRows][kXStride / 2] words
  float* partial = reinterpret_cast<float*>(smem + kMaxRows * kXStride * 2);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int slot = warp / kSplit, part = warp % kSplit;
  const int tiles = (n + 15) >> 4;
  const int col_base = part * kChunks * kChunk;

  int tile = blockIdx.x + gridDim.x * slot;
  Stage stage[kDepth];
  auto rows_of = [&](int tl, int& r0, int& r1) {
    // rows past N read row N - 1 and are not stored
    r0 = min(tl * 16 + g, n - 1);
    r1 = min(tl * 16 + g + 8, n - 1);
  };
  if (tile < tiles) {
    int r0, r1;
    rows_of(tile, r0, r1);
#pragma unroll
    for (int d = 0; d < kDepth; ++d) load_stage(stage[d], weight, scales, r0, r1, col_base + d * kChunk, t);
  }

  pdl_wait();
  // x into shared memory by 16-byte loads (two 4-column groups each), each group as (0, 2, 1, 3); rows past
  // `rows` zero. Every load of a thread is issued before its first store.
  constexpr int kXItems = kMaxRows * (kK / 8), kThreads = kWarps * 32;
  uint4 xv[(kXItems + kThreads - 1) / kThreads];
#pragma unroll
  for (int j = 0; j < (kXItems + kThreads - 1) / kThreads; ++j) {
    const int i = threadIdx.x + j * kThreads, r = i / (kK / 8), q = i % (kK / 8);
    xv[j] = make_uint4(0u, 0u, 0u, 0u);
    if (i < kXItems && r < rows) xv[j] = *reinterpret_cast<const uint4*>(x + (long long)r * x_stride + 8 * q);
  }
#pragma unroll
  for (int j = 0; j < (kXItems + kThreads - 1) / kThreads; ++j) {
    const int i = threadIdx.x + j * kThreads, r = i / (kK / 8), q = i % (kK / 8);
    if (i < kXItems) {
      unsigned* d = xs + r * (kXStride / 2) + 4 * q;
      d[0] = prmt(xv[j].x, xv[j].y, 0x5410);
      d[1] = prmt(xv[j].x, xv[j].y, 0x7632);
      d[2] = prmt(xv[j].z, xv[j].w, 0x5410);
      d[3] = prmt(xv[j].z, xv[j].w, 0x7632);
    }
  }
  __syncthreads();

  // columns g >= kMaxRows are never stored: their lanes read row 0
  const unsigned* xrow = xs + (g < kMaxRows ? g : 0) * (kXStride / 2) + (col_base + 16 * t) / 2;
  for (; tile < tiles; tile += gridDim.x * kSlots) {
    int r0, r1;
    rows_of(tile, r0, r1);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c0 = 0; c0 < kChunks; c0 += kDepth) {
#pragma unroll
      for (int d = 0; d < kDepth; ++d) {
        const int c = c0 + d;
        consume(acc, stage[d], xrow + c * (kChunk / 2), t);
        if (c + kDepth < kChunks) {
          load_stage(stage[d], weight, scales, r0, r1, col_base + (c + kDepth) * kChunk, t);
        }
      }
    }
    const int next = tile + gridDim.x * kSlots;
    if (next < tiles) {
      int n0, n1;
      rows_of(next, n0, n1);
#pragma unroll
      for (int d = 0; d < kDepth; ++d) load_stage(stage[d], weight, scales, n0, n1, col_base + d * kChunk, t);
    }
    if (kSplit > 1) {
      float* mine = partial + (slot * (kSplit - 1) * 32 + lane) * 4;
      if (part > 0) {
        *reinterpret_cast<float4*>(mine + (part - 1) * 128) = make_float4(acc[0], acc[1], acc[2], acc[3]);
      }
      asm volatile("bar.sync %0, %1;" ::"r"(slot + 1), "r"(kSplit * 32) : "memory");
      if (part == 0) {
#pragma unroll
        for (int p = 1; p < kSplit; ++p) {
          const float4 o = *reinterpret_cast<const float4*>(mine + (p - 1) * 128);
          acc[0] += o.x;
          acc[1] += o.y;
          acc[2] += o.z;
          acc[3] += o.w;
        }
      }
      asm volatile("bar.sync %0, %1;" ::"r"(slot + 1), "r"(kSplit * 32) : "memory");
    }
    if (part == 0) {
      const int row = tile * 16 + g;
      const int col = 2 * t;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = row + (i >> 1) * 8, c = col + (i & 1);
        if (r < n && c < rows) out[(long long)c * out_stride + r] = __float2bfloat16_rn(acc[i]);
      }
    }
  }
  pdl_trigger();
}

#else  // LH_RING

namespace {

// ring: kWarps consumer warps (kSlots tiles x kSplit parts a round) and one producer warp
constexpr int kStages = LH_STAGES;
constexpr int kGroup = LH_GROUP;  // consecutive tiles a tensor copy
constexpr int kUnroll = LH_UNROLL;  // stages a consumer iteration
constexpr int kDiag = LH_DIAG;  // timing only: 1 consumers skip the math, 2 the producer skips the copies
// 1: a consumer releases its stages once their fragments are loaded (the arrive's release orders the loads), before
// the math; 0: after the math
constexpr int kEarlyRelease = LH_RELEASE;
constexpr int kGroupsPerRound = kSlots / kGroup;
constexpr int kBoxes = kGroupsPerRound * kSplit;  // code copies (and scale copies) a stage
constexpr int kBoxBytes = 16 * kChunk;  // a tile's codes a stage: 16 rows x 128
// the scales of a group's chunk: one 128-row swizzled block, or for a single tile the half of it holding its rows
constexpr int kScaleCopy = kGroup == 1 ? 256 : 512;
constexpr int kStageBytes = kWarps * kBoxBytes + kBoxes * kScaleCopy;
constexpr int kXOffset = 128;  // after the mbarriers
constexpr int kPartialOffset = kXOffset + kMaxRows * kXStride * 2;
constexpr int kRingOffset = (kPartialOffset + kSlots * (kSplit - 1) * 32 * 16 + 1023) / 1024 * 1024;
constexpr unsigned kSpinLimit = 1u << 26;  // polls of one mbarrier wait before trapping

static_assert(kFp8, "the ring reads MXFP8");
static_assert(kGroup == 1 || kGroup == 2 || kGroup == 4 || kGroup == 8, "a group's rows lie in one 128-row block");
static_assert(kSlots % kGroup == 0, "whole groups a round");
static_assert(kBoxes <= 16, "one producer lane per code copy and one per scale copy");
static_assert(kStages >= kUnroll + 1 && 2 * kStages * 8 <= kXOffset, "stages: more than a consumer holds; "
              "mbarriers fit before x");
static_assert(kChunks % kUnroll == 0, "whole consumer iterations");
static_assert(kSlots + 2 <= 16, "named barriers 2.. per slot");

struct __align__(64) TensorMap {
  unsigned long long words[16];
};

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(unsigned bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(unsigned bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(unsigned bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
  unsigned ok = 0, spins = 0;
  while (true) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
    if (ok) return;
    if (++spins == kSpinLimit) __trap();
  }
}

// the consumers read stages through the generic proxy, the copies write them through the async proxy
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// the codes of tiles [tile, tile + kGroup) at 128-column chunk `chunk`: kGroup x 2 KB, tile by tile (box {128, 1,
// 16 kGroup} of the view {128, chunk, row})
__device__ __forceinline__ void tensor_copy(unsigned dst, const TensorMap* map, int chunk, int tile, unsigned bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<unsigned long long>(map)), "r"(0), "r"(chunk), "r"(tile * 16), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void bulk_copy(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

__device__ __forceinline__ uint4 lds128(unsigned a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}

__device__ __forceinline__ unsigned lds32(unsigned a) {
  unsigned v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}

}  // namespace

extern "C" __global__ void __launch_bounds__((kWarps + 1) * 32, 1)
    glm_draft_lm_head_ring(const __grid_constant__ TensorMap codes, const unsigned char* __restrict__ scales,
                           const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int rows,
                           int x_stride, int out_stride) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const unsigned bars = smem_u32(smem);  // full[s] at bars + 8 s, empty[s] at bars + 8 (kStages + s)
  unsigned* xs = reinterpret_cast<unsigned*>(smem + kXOffset);  // [kMaxRows][kXStride / 2] words
  float* partial = reinterpret_cast<float*>(smem + kPartialOffset);
  const unsigned ring = smem_u32(smem + kRingOffset);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tiles = n >> 4, groups = (tiles + kGroup - 1) / kGroup;
  const int cta = blockIdx.x, grid = gridDim.x;
  // this CTA's groups: cta + grid * m, m < count; round r holds m in [r * kGroupsPerRound, (r + 1) * kGroupsPerRound)
  const int count = cta < groups ? (groups - cta + grid - 1) / grid : 0;
  const int rounds = (count + kGroupsPerRound - 1) / kGroupsPerRound;
  const int total = rounds * kChunks;  // stages

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * (kStages + s), kWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == kWarps) {
    // ---- producer: lane b < kBoxes copies box b's codes, lane kBoxes + b its scales; box b = (part, group) ----
    if (lane == 0) asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<unsigned long long>(&codes)) : "memory");
    const int box = lane % kBoxes, q = box % kGroupsPerRound, part = box / kGroupsPerRound;
    for (int i = 0; i < total; ++i) {
      const int s = i % kStages, r = i / kChunks, c = i % kChunks;
      const unsigned full = bars + 8 * s;
      if (i >= kStages) mbar_wait(bars + 8 * (kStages + s), ((i / kStages) & 1) ^ 1);
      fence_proxy_async();
      const int active = min(kGroupsPerRound, count - r * kGroupsPerRound);
      if (kDiag == 2) {
        if (lane == 0) mbar_arrive(full);
        continue;
      }
      // a box partly past N is zero-filled and still counts its full bytes
      if (lane == 0) mbar_arrive_expect_tx(full, active * kSplit * (kGroup * kBoxBytes + kScaleCopy));
      __syncwarp();
      if (lane < 2 * kBoxes && q < active) {
        const int group = cta + grid * (r * kGroupsPerRound + q), tile = group * kGroup;
        const int chunk = part * kChunks + c;  // 128-column chunk of the row
        const unsigned stage = ring + s * kStageBytes;
        if (lane < kBoxes) {
          tensor_copy(stage + (part * kSlots + q * kGroup) * kBoxBytes, &codes, chunk, tile, full);
        } else {
          // rows 16 tile .. 16 (tile + kGroup) - 1 lie in one 128-row block; a single tile's in one half of it
          const unsigned char* src =
              scales + ((long long)(tile >> 3) * (kK / kChunk) + chunk) * 512 + (kGroup == 1 ? (tile & 1) * 256 : 0);
          bulk_copy(stage + kWarps * kBoxBytes + box * kScaleCopy, src, kScaleCopy, full);
        }
      }
    }
#if LH_TRIGGER
    pdl_trigger();
#endif
  } else {
    // ---- consumers ----
    pdl_wait();
    // x into shared memory as the stream kernel stages it
    constexpr int kXItems = kMaxRows * (kK / 8), kThreads = kWarps * 32;
    uint4 xv[(kXItems + kThreads - 1) / kThreads];
#pragma unroll
    for (int j = 0; j < (kXItems + kThreads - 1) / kThreads; ++j) {
      const int i = threadIdx.x + j * kThreads, r = i / (kK / 8), q = i % (kK / 8);
      xv[j] = make_uint4(0u, 0u, 0u, 0u);
      if (i < kXItems && r < rows) xv[j] = *reinterpret_cast<const uint4*>(x + (long long)r * x_stride + 8 * q);
    }
#pragma unroll
    for (int j = 0; j < (kXItems + kThreads - 1) / kThreads; ++j) {
      const int i = threadIdx.x + j * kThreads, r = i / (kK / 8), q = i % (kK / 8);
      if (i < kXItems) {
        unsigned* d = xs + r * (kXStride / 2) + 4 * q;
        d[0] = prmt(xv[j].x, xv[j].y, 0x5410);
        d[1] = prmt(xv[j].x, xv[j].y, 0x7632);
        d[2] = prmt(xv[j].z, xv[j].w, 0x5410);
        d[3] = prmt(xv[j].z, xv[j].w, 0x7632);
      }
    }
    asm volatile("bar.sync 1, %0;" ::"r"(kThreads) : "memory");
#if LH_TRIGGER
    pdl_trigger();
#endif

    const int g = lane >> 2, t = lane & 3;
    const int slot = warp / kSplit, part = warp % kSplit;
    const int q = slot / kGroup;  // the group of this slot's tile in a round
    // this lane's x words at its chunk 0 (column g of B; columns >= kMaxRows take zeros)
    const unsigned xbase = smem_u32(xs + (g < kMaxRows ? g : 0) * (kXStride / 2) + (part * kChunks * kChunk + 16 * t) / 2);
    // this lane's codes in its tile's 2 KB (row r at 128 r): (row half h, segment s) at code_lane + h * 1024 + s * 64
    constexpr unsigned kSegStep = 64;
    const unsigned code_lane = (part * kSlots + slot) * kBoxBytes + 128 * g + 16 * t;
    const unsigned sel0 = 0x4040u | (0x0101u * (t >> 1)), sel1 = 0x4040u | (0x0101u * (2 + (t >> 1)));
    int i = 0;  // stage sequence
    for (int r = 0; r < rounds; ++r) {
      const int tile = (cta + grid * (r * kGroupsPerRound + q)) * kGroup + slot % kGroup;
      const bool active = r * kGroupsPerRound + q < count && tile < tiles;
      // row 16 tile + g + 8 h in its 128-row block: (row & 31) * 16 + ((row >> 5) & 3) * 4 (minus the half not copied)
      const int row = 16 * tile + g;
      const unsigned scale_lane = kWarps * kBoxBytes + (part * kGroupsPerRound + q) * kScaleCopy + (row & 31) * 16 +
                                  ((row >> 5) & 3) * 4 - (kGroup == 1 ? (tile & 1) * 256 : 0);
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int c = 0; c < kChunks; c += kUnroll) {
        Stage st[kUnroll];
        uint4 xw4[kUnroll][4];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          const int s = (i + u) % kStages;
          mbar_wait(bars + 8 * s, ((i + u) / kStages) & 1);
          if (active && kDiag != 1) {
            const unsigned stage = ring + s * kStageBytes;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
              for (int seg = 0; seg < 2; ++seg) {
                const uint4 v = lds128(stage + code_lane + h * 1024 + seg * kSegStep);
                st[u].w[h][4 * seg] = v.x;
                st[u].w[h][4 * seg + 1] = v.y;
                st[u].w[h][4 * seg + 2] = v.z;
                st[u].w[h][4 * seg + 3] = v.w;
              }
              st[u].scales[h] = lds32(stage + scale_lane + h * 128);
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              xw4[u][v] = make_uint4(0u, 0u, 0u, 0u);
              if (g < kMaxRows) xw4[u][v] = lds128(xbase + 4 * ((c + u) * (kChunk / 2) + (v >> 1) * 32 + (v & 1) * 4));
            }
          }
        }
        if (kEarlyRelease) {
          __syncwarp();
          if (lane == 0) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) mbar_arrive(bars + 8 * (kStages + (i + u) % kStages));
          }
        }
        if (active && kDiag != 1) {
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            unsigned sc[2][2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              sc[h][0] = scale_pair_sel(st[u].scales[h], sel0);
              sc[h][1] = scale_pair_sel(st[u].scales[h], sel1);
            }
            mma_chunk(acc, st[u], sc, reinterpret_cast<const unsigned*>(xw4[u]));
          }
        }
        if (!kEarlyRelease) {
          __syncwarp();
          if (lane == 0) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) mbar_arrive(bars + 8 * (kStages + (i + u) % kStages));
          }
        }
        i += kUnroll;
      }
      if (active) {
        if (kSplit > 1) {
          float* mine = partial + (slot * (kSplit - 1) * 32 + lane) * 4;
          if (part > 0) {
            *reinterpret_cast<float4*>(mine + (part - 1) * 128) = make_float4(acc[0], acc[1], acc[2], acc[3]);
          }
          asm volatile("bar.sync %0, %1;" ::"r"(slot + 2), "r"(kSplit * 32) : "memory");
          if (part == 0) {
#pragma unroll
            for (int p = 1; p < kSplit; ++p) {
              const float4 o = *reinterpret_cast<const float4*>(mine + (p - 1) * 128);
              acc[0] += o.x;
              acc[1] += o.y;
              acc[2] += o.z;
              acc[3] += o.w;
            }
          }
          asm volatile("bar.sync %0, %1;" ::"r"(slot + 2), "r"(kSplit * 32) : "memory");
        }
        if (part == 0) {
          const int col = 2 * t;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int rr = row + (e >> 1) * 8, cc = col + (e & 1);
            if (cc < rows) out[(long long)cc * out_stride + rr] = __float2bfloat16_rn(acc[e]);
          }
        }
      }
    }
  }
#if !LH_TRIGGER
  pdl_trigger();
#endif
}

#endif  // LH_RING
