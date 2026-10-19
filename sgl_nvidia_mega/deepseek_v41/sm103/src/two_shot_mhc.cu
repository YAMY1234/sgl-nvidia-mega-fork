// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Two-shot MoE finalize / TP all-reduce fused with DeepSeek-V4.1's HC=4 mHC post
// [+ pre-collapse + RMSNorm [+ MXFP8 quant of the normalized row]], for 49-384 rows
// (two_shot_mhc.py). Row t is owned by rank o = t % world: every rank pushes its
// finalized row t to o only, o polls the world partials and reduces them in rank
// order, then pushes the BF16 row to every other rank, which runs the epilogue on it.
// Slots use the empty-marker (+0.0) exchange: a producer rewrites +0 words to -0, and
// every slot location a round writes is restored to +0 by its reader. Each row cluster
// owns one epoch counter that picks which half of the double-buffered mailbox the round
// uses; a trailing "bumper" cluster flips the counters of the rows this call leaves
// unused and zeroes the scale padding rows.
//
// Variant constants come in as -D options: TS_WORLD, TS_TOP_K, TS_HAS_SHARED, TS_NORM
// (combine + RMSNorm epilogue), TS_QUANT (MXFP8 of the normalized row). The arithmetic
// is the SGLang kernel's, so it must be built without --use_fast_math.
// Grid (num_tokens + 1, kClusterSize), block kBlockSize, cluster (1, kClusterSize, 1), PDL.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef TS_WORLD
#define TS_WORLD 4
#endif
#ifndef TS_TOP_K
#define TS_TOP_K 6
#endif
#ifndef TS_HAS_SHARED
#define TS_HAS_SHARED 1
#endif
#ifndef TS_NORM
#define TS_NORM 1
#endif
#ifndef TS_QUANT
#define TS_QUANT 0
#endif

namespace {

constexpr unsigned kWorldSize = TS_WORLD;
constexpr unsigned kTopK = TS_TOP_K;
constexpr bool kHasShared = TS_HAS_SHARED != 0;
constexpr bool kNorm = TS_NORM != 0;
constexpr bool kQuant = TS_QUANT != 0;
constexpr unsigned kMaxWorldSize = 8;  // must match MAX_WORLD in two_shot_mhc.py
constexpr unsigned kHiddenDim = 5120;
constexpr unsigned kHc = 4;
constexpr unsigned kClusterSize = 5;  // SGLang's default_cluster_size(5120)
constexpr unsigned kWarpThreads = 32;
constexpr unsigned kRowVecs = kHiddenDim / 8;  // 16B vectors per row
constexpr unsigned kBlockSize = kRowVecs / kClusterSize;
constexpr unsigned kNumWarps = kBlockSize / kWarpThreads;
constexpr unsigned kDeadlinePolls = 1024;  // polls between two reads of the global timer

static_assert(kWorldSize >= 2 && kWorldSize <= kMaxWorldSize, "world size");
static_assert(kRowVecs % kClusterSize == 0 && kBlockSize % kWarpThreads == 0, "row geometry");
static_assert(!kQuant || kNorm, "the quant epilogue quantizes the normalized row");

}  // namespace

// Field order and types must match _Params in two_shot_mhc.py.
struct TwoShotParams {
  __nv_bfloat16* out;                // [T, hidden] reduced row, output-only
  const __nv_bfloat16* gemm2;        // [P, hidden] permuted GEMM2 rows
  const int* idx;                    // [T * kTopK], -1 = dropped slot
  const float* weights;              // [T, kTopK], scaling folded in
  const __nv_bfloat16* shared;       // [T, hidden] (kHasShared only)
  const __nv_bfloat16* norm_weight;  // [hidden] (kNorm only)
  __nv_bfloat16* mhc_out;            // [T, 4, hidden]
  const __nv_bfloat16* residual;     // [T, 4, hidden]
  const float* post;                 // [T, 4]
  const float* comb;                 // [T, 4, 4], [source, target]
  const float* pre;                  // [T, 4] (kNorm only)
  __nv_bfloat16* normalized;         // [T, hidden] (kNorm only)
  unsigned char* quantized;          // [T, hidden] fp8 e4m3 (kQuant only)
  unsigned char* scales;             // ue8m0, 128x4 swizzled, rows padded to 128 (kQuant only)
  unsigned int* counter;             // [num_push_counters], rank-local epochs
  unsigned char* workspaces[kMaxWorldSize];  // per-rank mailbox base: 2 x world slots
  unsigned long long deadline_ns;    // 0: wait forever
  float norm_eps;
  unsigned int rank;
  unsigned int num_tokens;
  unsigned int num_push_counters;
  unsigned int slot_bytes;
  unsigned int reserved;
};

namespace {

/// 16B of staged data: 8 bf16 as 4 packed bf16x2 words.
struct __align__(16) StageVec {
  unsigned int w[4];
};

__device__ __forceinline__ float2 bf2_to_f2(unsigned int bits) {
  __nv_bfloat162 v;
  *reinterpret_cast<unsigned int*>(&v) = bits;
  return __bfloat1622float2(v);
}

__device__ __forceinline__ unsigned int f2_to_bf2(float2 v) {
  const __nv_bfloat162 r = __float22bfloat162_rn(v);
  return *reinterpret_cast<const unsigned int*>(&r);
}

__device__ __forceinline__ void load_vec(StageVec& x, const void* base, long long vec) {
  const uint4 val = reinterpret_cast<const uint4*>(base)[vec];
  x.w[0] = val.x, x.w[1] = val.y, x.w[2] = val.z, x.w[3] = val.w;
}

__device__ __forceinline__ void store_vec(const StageVec& x, void* base, long long vec) {
  reinterpret_cast<uint4*>(base)[vec] = make_uint4(x.w[0], x.w[1], x.w[2], x.w[3]);
}

__device__ __forceinline__ void st_global_16B(const StageVec& x, void* base, long long vec) {
  void* addr = static_cast<unsigned char*>(base) + vec * 16;
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(x.w[0]), "r"(x.w[1]), "r"(x.w[2]), "r"(x.w[3]),
               "l"(addr));
}

// Peer-visible accesses; ordering comes from the payload's own markers.
__device__ __forceinline__ void ld_relaxed_16B(StageVec& x, const void* base, long long vec) {
  const void* addr = static_cast<const unsigned char*>(base) + vec * 16;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.w[0]), "=r"(x.w[1]), "=r"(x.w[2]), "=r"(x.w[3])
               : "l"(addr));
}

__device__ __forceinline__ void st_relaxed_16B(const StageVec& x, void* base, long long vec) {
  void* addr = static_cast<unsigned char*>(base) + vec * 16;
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(x.w[0]), "r"(x.w[1]), "r"(x.w[2]),
               "r"(x.w[3]), "l"(addr));
}

// Empty-marker words: one u32 (a bf16 pair) is "not arrived" while it is +0.
__device__ __forceinline__ void clear_pos_zero(StageVec& x) {
#pragma unroll
  for (unsigned i = 0; i < 4; ++i)
    if (x.w[i] == 0) x.w[i] = 0x80008000u;
}

__device__ __forceinline__ bool has_pos_zero(const StageVec& x) {
  bool result = false;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) result |= x.w[i] == 0;
  return result;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void barrier_cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

// cooperative_groups' cluster.sync(): release arrive + acquire wait.
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive;" ::: "memory");
  asm volatile("barrier.cluster.wait;" ::: "memory");
}

__device__ __forceinline__ void st_shared_cluster(float* local, unsigned rank, float value) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(local));
  unsigned remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(addr), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(value) : "memory");
}

__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// A missed wait deadline stops the kernel: a store to address 0x0, then trap.
__device__ __forceinline__ void fault_at_null() {
  asm volatile("{\n.reg .u64 a;\nmov.u64 a, 0;\nst.volatile.global.u32 [a], 0;\ntrap;\n}" ::: "memory");
}

struct Deadline {
  unsigned polls = 0;
  unsigned long long started = 0;
  __device__ __forceinline__ void tick(unsigned long long deadline_ns) {
    if (deadline_ns == 0 || ++polls % kDeadlinePolls != 0) return;
    const unsigned long long now = global_timer_ns();
    if (started == 0) started = now;
    if (now - started > deadline_ns) fault_at_null();
  }
};

/// Picks which half of the mailbox's 2 x world slots this round owns.
struct PushEpoch {
  const TwoShotParams& p;
  unsigned epoch;
  __device__ __forceinline__ explicit PushEpoch(const TwoShotParams& params)
      : p(params), epoch(params.counter[blockIdx.x]) {}
  /// Rank `src`'s slot inside rank `dst`'s mailbox, for the current epoch.
  __device__ __forceinline__ void* slot_ptr(unsigned dst, unsigned src) const {
    const unsigned epoch_stride_bytes = (epoch & 1) * p.slot_bytes * kWorldSize;
    return p.workspaces[dst] + src * p.slot_bytes + epoch_stride_bytes;
  }
  __device__ __forceinline__ void flip() const {
    if (threadIdx.x == 0) p.counter[blockIdx.x] = epoch ^ 1;
  }
  __device__ __forceinline__ void unsafe_flip_range(unsigned start, unsigned finish) const {
    for (unsigned i = start + threadIdx.x; i < finish; i += blockDim.x) p.counter[i] = epoch ^ 1;
  }
};

__device__ __forceinline__ void mhc_quant_vec(const TwoShotParams& params, const StageVec& value, unsigned token,
                                              unsigned hvec) {
  float2 v[4];
  float amax = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    v[j] = bf2_to_f2(value.w[j]);
    amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
  const float normalized = amax * (1.0f / 448.0f);
  const unsigned bits = __float_as_uint(normalized);
  const unsigned exponent = (bits >> 23) & 255;
  const unsigned mantissa = bits & 0x7fffff;
  const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
  const unsigned sf = normalized <= 0 ? 0 : min(exponent + unsigned(bump), 254u);
  const float inv_scale = __uint_as_float(sf == 0 ? 0 : (254 - sf) << 23);
  __nv_fp8x2_storage_t q[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    q[j] = __nv_cvt_float2_to_fp8x2(
        make_float2(fminf(fmaxf(v[j].x * inv_scale, -448.0f), 448.0f), fminf(fmaxf(v[j].y * inv_scale, -448.0f), 448.0f)),
        __NV_SATFINITE, __NV_E4M3);
  }
  uint2 packed;
  packed.x = unsigned(q[0]) | (unsigned(q[1]) << 16);
  packed.y = unsigned(q[2]) | (unsigned(q[3]) << 16);
  reinterpret_cast<uint2*>(params.quantized + static_cast<long long>(token) * kHiddenDim)[hvec] = packed;
  if (hvec % 4 == 0) {
    const unsigned g = hvec / 4;
    const unsigned off = (token / 128) * (kHiddenDim / 128) * 512 + (g / 4) * 512 +
                         ((token % 32) * 4 + (token / 32) % 4) * 4 + g % 4;
    params.scales[off] = static_cast<unsigned char>(sf);
  }
}

/// HC=4 post mixing of an already BF16-rounded all-reduce vector: round
/// comb[0]*residual[0], FMA post*x, then the remaining three residual streams.
template <bool kCollapse>
__device__ __forceinline__ StageVec mhc_post_vec(const TwoShotParams& params, const StageVec& red, unsigned token,
                                                 unsigned hvec) {
  StageVec residual[kHc];
  float2 collapsed[4] = {};
#pragma unroll
  for (unsigned c = 0; c < kHc; ++c) {
    load_vec(residual[c], params.residual + (static_cast<long long>(token) * kHc + c) * kHiddenDim, hvec);
  }
#pragma unroll
  for (unsigned c = 0; c < kHc; ++c) {
    const float post = params.post[token * kHc + c];
    float comb[kHc];
#pragma unroll
    for (unsigned r = 0; r < kHc; ++r) comb[r] = params.comb[token * kHc * kHc + r * kHc + c];
    StageVec out;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const float2 x = bf2_to_f2(red.w[j]);
      const float2 r0 = bf2_to_f2(residual[0].w[j]);
      float2 acc = make_float2(fmaf(post, x.x, __fmul_rn(comb[0], r0.x)), fmaf(post, x.y, __fmul_rn(comb[0], r0.y)));
#pragma unroll
      for (unsigned r = 1; r < kHc; ++r) {
        const float2 v = bf2_to_f2(residual[r].w[j]);
        acc.x = fmaf(comb[r], v.x, acc.x);
        acc.y = fmaf(comb[r], v.y, acc.y);
      }
      out.w[j] = f2_to_bf2(acc);
      if constexpr (kCollapse) {
        const float2 rounded = bf2_to_f2(out.w[j]);
        const float pre = params.pre[token * kHc + c];
        collapsed[j].x = fmaf(rounded.x, pre, collapsed[j].x);
        collapsed[j].y = fmaf(rounded.y, pre, collapsed[j].y);
      }
    }
    store_vec(out, params.mhc_out + (static_cast<long long>(token) * kHc + c) * kHiddenDim, hvec);
  }
  StageVec result = {};
  if constexpr (kCollapse) {
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) result.w[j] = f2_to_bf2(collapsed[j]);
  }
  return result;
}

/// The deferred finalize of one 16B vector: FP32 FMAs of the routed rows in slot
/// order, rounded to BF16, then the BF16 + BF16 shared-expert add (two roundings).
__device__ __forceinline__ StageVec finalize_vec(const TwoShotParams& params, unsigned token, unsigned hvec) {
  const int* idx = params.idx + static_cast<long long>(token) * kTopK;
  const float* weights = params.weights + static_cast<long long>(token) * kTopK;
  int rows[kTopK];
  float w[kTopK];
#pragma unroll
  for (unsigned k = 0; k < kTopK; ++k) {
    rows[k] = idx[k];
    w[k] = weights[k];
  }

  StageVec shared_in;
  if constexpr (kHasShared) load_vec(shared_in, params.shared + static_cast<long long>(token) * kHiddenDim, hvec);

  StageVec in[kTopK];
#pragma unroll
  for (unsigned k = 0; k < kTopK; ++k) {
    if (rows[k] >= 0) load_vec(in[k], params.gemm2 + static_cast<long long>(rows[k]) * kHiddenDim, hvec);
  }

  float2 acc[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) acc[j] = make_float2(0.0f, 0.0f);

#pragma unroll
  for (unsigned k = 0; k < kTopK; ++k) {
    if (rows[k] < 0) continue;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const float2 x = bf2_to_f2(in[k].w[j]);
      acc[j].x = fmaf(x.x, w[k], acc[j].x);
      acc[j].y = fmaf(x.y, w[k], acc[j].y);
    }
  }
  StageVec out;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    if constexpr (kHasShared) {
      const float2 routed = bf2_to_f2(f2_to_bf2(acc[j]));
      const float2 sh = bf2_to_f2(shared_in.w[j]);
      out.w[j] = f2_to_bf2(make_float2(routed.x + sh.x, routed.y + sh.y));
    } else {
      out.w[j] = f2_to_bf2(acc[j]);
    }
  }
  return out;
}

/// FP32 sum of the world's partials in rank order, rounded to BF16.
__device__ __forceinline__ StageVec reduce_vec(const StageVec (&vec)[kWorldSize]) {
  float2 acc[4];
#pragma unroll
  for (unsigned i = 0; i < kWorldSize; ++i) {
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const float2 v = bf2_to_f2(vec[i].w[j]);
      acc[j].x = i == 0 ? v.x : acc[j].x + v.x;
      acc[j].y = i == 0 ? v.y : acc[j].y + v.y;
    }
  }
  StageVec out;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) out.w[j] = f2_to_bf2(acc[j]);
  return out;
}

}  // namespace

// blockIdx.x is the token row (and its epoch counter), blockIdx.y the rank inside
// the row's cluster. The extra cluster (blockIdx.x == num_tokens) is the bumper.
extern "C" __global__ void __launch_bounds__(kBlockSize)
    dsv41_two_shot_mhc(const __grid_constant__ TwoShotParams params) {
  __shared__ float smem_sq[kClusterSize][kNumWarps];
  const unsigned tx = threadIdx.x;
  const unsigned row_idx = blockIdx.x;
  const unsigned cluster_rank = blockIdx.y;
  const unsigned hvec = cluster_rank * kBlockSize + tx;

  // The epoch must not be read before the predecessor's flip is visible.
  pdl_wait();

  if (row_idx == params.num_tokens) {
    if constexpr (kQuant) {
      // The SF buffer is padded to 128 rows; the norm epilogue writes the active rows.
      const unsigned padded_rows = (params.num_tokens + 127) / 128 * 128;
      for (unsigned off = hvec; off < (kHiddenDim / 32) * padded_rows; off += kRowVecs) {
        const unsigned swizzled_row = (off % 512) / 4;
        const unsigned row = off / ((kHiddenDim / 128) * 512) * 128 + swizzled_row / 4 + (swizzled_row % 4) * 32;
        if (row >= params.num_tokens) params.scales[off] = 0;
      }
    }
    if (cluster_rank == 0) {
      const PushEpoch epoch(params);
      __syncthreads();
      epoch.unsafe_flip_range(row_idx, params.num_push_counters);
    }
    pdl_trigger();
    return;
  }

  const PushEpoch epoch(params);
  const unsigned r = params.rank;
  const unsigned owner = row_idx % kWorldSize;

  // stage 1: finalize this row's vector in registers and push it to the owner only
  const long long vid = static_cast<long long>(row_idx) * kRowVecs + hvec;
  {
    StageVec vec = finalize_vec(params, row_idx, hvec);
    clear_pos_zero(vec);
    st_relaxed_16B(vec, epoch.slot_ptr(/*dst=*/owner, /*src=*/r), vid);
  }

  // every block of the cluster has read the epoch before rank 0 flips it
  if constexpr (!kNorm) barrier_cluster_arrive_relaxed();

  // stage 2: the owner polls its world slots and reduces in rank order, then sends the
  // BF16 sum (empty-marker words remapped) to the other ranks, which poll it
  Deadline deadline;
  StageVec red;
  if (r == owner) {
    StageVec vec[kWorldSize];
    do {
      deadline.tick(params.deadline_ns);
      bool has_zero = false;
#pragma unroll
      for (unsigned i = 0; i < kWorldSize; ++i) ld_relaxed_16B(vec[i], epoch.slot_ptr(/*dst=*/r, /*src=*/i), vid);
#pragma unroll
      for (unsigned i = 0; i < kWorldSize; ++i) has_zero |= has_pos_zero(vec[i]);
      if (!has_zero) break;
    } while (true);
    red = reduce_vec(vec);
    StageVec wire = red;
    clear_pos_zero(wire);
#pragma unroll
    for (unsigned i = 0; i < kWorldSize; ++i) {
      if (i != r) st_relaxed_16B(wire, epoch.slot_ptr(/*dst=*/i, /*src=*/r), vid);
    }
  } else {
    const void* gather_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/owner);
    do {
      deadline.tick(params.deadline_ns);
      ld_relaxed_16B(red, gather_ptr, vid);
    } while (has_pos_zero(red));
  }

  if constexpr (!kNorm) {
    st_global_16B(red, params.out, vid);
    mhc_post_vec<false>(params, red, row_idx, hvec);
    barrier_cluster_wait();
  } else {
    st_global_16B(red, params.out, vid);
    red = mhc_post_vec<true>(params, red, row_idx, hvec);
    StageVec w;
    load_vec(w, params.norm_weight, hvec);
    float2 acc[4];
    float sq = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      acc[j] = bf2_to_f2(red.w[j]);
      sq = fmaf(acc[j].x, acc[j].x, sq);
      sq = fmaf(acc[j].y, acc[j].y, sq);
    }
#pragma unroll
    for (unsigned mask = kWarpThreads / 2; mask >= 1; mask >>= 1) {
      sq = sq + __shfl_xor_sync(0xffffffff, sq, mask, kWarpThreads);
    }
    const unsigned lane = tx % kWarpThreads;
    const unsigned warp = tx / kWarpThreads;
    if (lane < kClusterSize) st_shared_cluster(&smem_sq[cluster_rank][warp], lane, sq);
    cluster_sync();
    float total = 0.0f;
#pragma unroll
    for (unsigned c = 0; c < kClusterSize; ++c) {
#pragma unroll
      for (unsigned wp = 0; wp < kNumWarps; ++wp) total += smem_sq[c][wp];
    }
    const float factor = rsqrtf(total / static_cast<float>(kHiddenDim) + params.norm_eps);
    StageVec out;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const float2 wf = bf2_to_f2(w.w[j]);
      out.w[j] = f2_to_bf2(make_float2(acc[j].x * factor * wf.x, acc[j].y * factor * wf.y));
    }
    st_global_16B(out, params.normalized, vid);
    if constexpr (kQuant) mhc_quant_vec(params, out, row_idx, hvec);
  }
  pdl_trigger();

  // re-establish the empty markers for the next same-epoch round
  StageVec zero_vec = {};
  if (r == owner) {
#pragma unroll
    for (unsigned i = 0; i < kWorldSize; ++i) st_global_16B(zero_vec, epoch.slot_ptr(/*dst=*/r, /*src=*/i), vid);
  } else {
    st_global_16B(zero_vec, epoch.slot_ptr(/*dst=*/r, /*src=*/owner), vid);
  }

  if (cluster_rank == 0) epoch.flip();
}
