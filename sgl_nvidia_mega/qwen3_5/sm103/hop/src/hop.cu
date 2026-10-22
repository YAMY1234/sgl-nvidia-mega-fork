// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The decode hop of a Qwen3.5 TP verify step: the TP sum of bf16 partials + residual + RMSNorm
// (+ static FP8) for hidden 4096, up to 8 ranks on one NVLink domain, bit-identical to
// FlashInfer's MNNVL CuTe DSL low-latency all-reduce + residual + RMSNorm.
//   publish      push this rank's bf16 partial [M, 4096]
//   publish_fin  finalize the routed MoE rows (top-k weighted, plus the gated shared output) and push
//   consume      pull + rank-order sum + residual + RMSNorm (+ static FP8), re-arm, advance
//
// Transport: every rank multicast-stores its partial into slot mailbox[stage][rank][token][...]
// of every rank; a reader polls its own copy until no bf16 -0.0 sentinel is left, sums the ranks
// in order and writes the sentinel back. Three stages rotate, one per hop. A fragment is 8
// consecutive bf16 of a token row (frag = 0..511); a token row is finished by one cluster of
// kRowCtas CTAs x kRowThreads threads (CTA c owns fragments [128c, 128c + 128)).
//
//   oproj_push   the GatedDeltaNet out_proj (static FP8, M <= 8) whose epilogue pushes its partial
//
// Built with NVRTC: C-linkage entries per instantiation; Q35_HOP_DEADLINE_NS (0: none) is the
// consumer's wait deadline, past which it faults at address 0x0 (fail-stop).

#include "q35_hop.cuh"

// a named namespace: the C entries below name Args in their signatures
namespace q35hop_kernels {


using namespace q35hop;

struct Args {
  Site site;
  const uint16_t* partial;      // [M, H] bf16 (publish)
  const uint16_t* routed;       // [M*K, H] bf16 (publish_fin), rows indexed by permuted
  const uint16_t* expert_w;     // [M, K] bf16
  const int32_t* permuted;      // [M, K]; -1 skips a slot
  const uint16_t* shared;       // [M, H] bf16 gated shared expert output, or null
  const uint16_t* residual_in;  // [M, H] bf16
  const uint16_t* gamma;        // [H] bf16, the applied multiplier
  const float* fp8_scale;       // [1], or null
  uint16_t* residual_out;       // [M, H] bf16
  uint16_t* norm_out;           // [M, H] bf16, or null
  uint8_t* fp8_out;             // [M, H] e4m3, or null
  int m, top_k;
  float eps;
};

constexpr int kMaxTopK = 16;

struct Site2;
__device__ __forceinline__ Hop hop_enter_two(const Site2& s);
__device__ __forceinline__ void hop_leave_two(const Site2& s, const Hop& h, uint32_t participants);

__device__ __forceinline__ uint4 ld_v4(const void* p) { return *static_cast<const uint4*>(p); }

__device__ __forceinline__ size_t row_elem(int token, int frag) {
  return static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
}

// Finalize-publish, 4 bf16 per thread (8 CTAs x 128 threads per token): per element
// sum_k w_k * routed[perm_k] in k order with separate multiply and add, then the shared output,
// as the reference finalize does. TOPK > 0 fixes top-k at compile time, so every routed row is
// loaded into registers before any accumulation (TOPK = 0 reads top_k at run time).
template <int TOPK>
__device__ __forceinline__ void publish_fin_quad_kernel(const Args& a) {
  constexpr int kSlots = TOPK > 0 ? TOPK : kMaxTopK;
  __shared__ int32_t rows_s[kSlots];
  __shared__ float wts_s[kSlots];
  const int top_k = TOPK > 0 ? TOPK : a.top_k;
  const int token = blockIdx.x;
  const int quad = blockIdx.y * kRowThreads + threadIdx.x;
  if (threadIdx.x < top_k) {
    const int i = token * top_k + threadIdx.x;
    const int r = a.permuted[i];
    rows_s[threadIdx.x] = r;
    wts_s[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(a.expert_w[i]) << 16);
  }
  __syncthreads();
  const size_t col = static_cast<size_t>(quad) * 4;
  pdl_wait();
  const uint32_t stage = stage_peek(a.site);
  // Read after the wait: the side-stream join that produces it may be captured as a programmatic
  // edge, which lets this launch start before the producer has finished.
  uint2 sh = make_uint2(0, 0);
  if (a.shared != nullptr) sh = *reinterpret_cast<const uint2*>(a.shared + static_cast<size_t>(token) * kHidden + col);
  int rows[kSlots];
#pragma unroll
  for (int k = 0; k < kSlots; ++k) rows[k] = k < top_k ? rows_s[k] : -1;
  uint2 src[kSlots];
#pragma unroll
  for (int k = 0; k < kSlots; ++k)
    src[k] = rows[k] >= 0 ? *reinterpret_cast<const uint2*>(a.routed + static_cast<size_t>(rows[k]) * kHidden + col)
                          : make_uint2(0, 0);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kSlots; ++k) {
    if (rows[k] >= 0) {
      const float w = wts_s[k];
      const float f[4] = {bf_lo(src[k].x), bf_hi(src[k].x), bf_lo(src[k].y), bf_hi(src[k].y)};
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], __fmul_rn(f[e], w));
    }
  }
  if (a.shared != nullptr) {
    const float f[4] = {bf_lo(sh.x), bf_hi(sh.x), bf_lo(sh.y), bf_hi(sh.y)};
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  push_quad(a.site, stage, token, quad, make_uint2(pack_bf16x2(acc[0], acc[1]), pack_bf16x2(acc[2], acc[3])));
  pdl_trigger();
}

__device__ __forceinline__ void publish_kernel(const Args& a) {
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  pdl_wait();
  const uint32_t stage = stage_peek(a.site);
  push_frag(a.site, stage, token, frag, ld_v4(a.partial + row_elem(token, frag)));
  pdl_trigger();
}

template <int W>
__device__ __forceinline__ void consume_kernel(const Args& a) {
  __shared__ RowSmem rs;
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  pdl_trigger();
  float gam[8];
  unpack8(ld_v4(a.gamma + static_cast<size_t>(frag) * kVec), gam);
  const float inv_s = a.fp8_out != nullptr ? inv_scale_of(*a.fp8_scale) : 1.f;
  row_init(rs);
  pdl_wait();
  const Hop h = hop_enter(a.site);
  const uint4 res = ld_v4(a.residual_in + row_elem(token, frag));
  float acc[8];
  pull_frag<W>(a.site, h.stage, token, frag, acc);
  float pre[8];
  const uint4 res_out = residual_add(acc, res, pre);
  const float total = row_sumsq_fold(sumsq8(pre), rs);
  const uint4 norm = norm_apply(pre, inv_rms(total, a.eps), gam);
  *reinterpret_cast<uint4*>(a.residual_out + row_elem(token, frag)) = res_out;
  if (a.norm_out != nullptr) *reinterpret_cast<uint4*>(a.norm_out + row_elem(token, frag)) = norm;
  if (a.fp8_out != nullptr) *reinterpret_cast<uint2*>(a.fp8_out + row_elem(token, frag)) = quant_fp8(norm, inv_s);
  rearm_frag<W>(a.site, h.stage, token, frag);
  hop_leave(a.site, h, gridDim.x * gridDim.y);
}

// Row-parallel FP8 out_proj (M <= 8, N = 4096, K = 8 * S * 32) whose epilogue pushes the bf16
// partial: CTA c owns output rows [32c, 32c + 32) over the full K, warp w takes k-steps
// [w*S, (w+1)*S), the weights are mma.m16n8k32 e4m3 A fragments loaded before the PDL wait, and
// the warp partials are summed in warp order.
struct OprojArgs {
  Site site;
  const uint4* w;    // packed [128 cta][8 warp][S step][2 block][32 lane] x 16 B
  const uint8_t* x;  // [M, K] e4m3 activation
  const float* sx;   // input scale
  const float* sw;   // weight scale
  uint16_t* out;     // [M, 4096] bf16 copy of the partial
  int m;
};

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// Weights are static for the life of the process.
__device__ __forceinline__ uint4 ldg_nc_v4(const uint4* q) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(q));
  return r;
}

template <int S>
__device__ __forceinline__ void oproj_push_kernel(const OprojArgs& o) {
  constexpr int kK = 8 * S * 32;
  __shared__ __align__(16) float red[8][32][8];
  const int cta = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  uint4 a[S][2];
  {
    const uint4* wb = o.w + (static_cast<size_t>(cta) * 8 + warp) * (S * 2 * 32) + lane;
#pragma unroll
    for (int st = 0; st < S; ++st)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[st][b] = ldg_nc_v4(wb + (st * 2 + b) * 32);
  }
  const float alpha = __fmul_rn(*o.sx, *o.sw);
  pdl_wait();
  const int m = o.m;
  const bool tok_ok = g < m;
  const uint8_t* xr = o.x + static_cast<size_t>(g) * kK + warp * S * 32 + 4 * t;
  float ch[2][2][4] = {};
#pragma unroll
  for (int st = 0; st < S; ++st) {
    const uint32_t b0 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32) : 0u;
    const uint32_t b1 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32 + 16) : 0u;
    mma_e4m3(ch[0][st & 1], a[st][0], b0, b1);
    mma_e4m3(ch[1][st & 1], a[st][1], b0, b1);
  }
  {
    float4* dst = reinterpret_cast<float4*>(&red[warp][lane][0]);
    dst[0] = make_float4(__fadd_rn(ch[0][0][0], ch[0][1][0]), __fadd_rn(ch[0][0][1], ch[0][1][1]),
                         __fadd_rn(ch[0][0][2], ch[0][1][2]), __fadd_rn(ch[0][0][3], ch[0][1][3]));
    dst[1] = make_float4(__fadd_rn(ch[1][0][0], ch[1][1][0]), __fadd_rn(ch[1][0][1], ch[1][1][1]),
                         __fadd_rn(ch[1][0][2], ch[1][1][2]), __fadd_rn(ch[1][0][3], ch[1][1][3]));
  }
  __syncthreads();
  // Thread (tok, j) owns rows [8j, 8j + 8) of token tok. Accumulator element (row r, token k) sits
  // at lane (r % 8) * 4 + k / 2, entry (r / 16) * 4 + ((r % 16) / 8) * 2 + k % 2 of the mma layout.
  const int tok = tid >> 2, j = tid & 3;
  if (tok < m) {
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int r = 8 * j + q;
      const int l = (r % 8) * 4 + tok / 2;
      const int e = (r / 16) * 4 + ((r % 16) / 8) * 2 + (tok % 2);
      float x = red[0][l][e];
#pragma unroll
      for (int w = 1; w < 8; ++w) x = __fadd_rn(x, red[w][l][e]);
      v[q] = __fmul_rn(x, alpha);
    }
    const uint4 frag = pack8(v);
    const int fr = cta * 4 + j;
    push_frag(o.site, stage_peek(o.site), tok, fr, frag);
    *reinterpret_cast<uint4*>(o.out + row_elem(tok, fr)) = frag;
  }
  pdl_trigger();
}


// ------------------------------------------------------------------ the two-shot hop (rows past the one-shot's)
// FlashInfer's MNNVL CuTe DSL BT protocol (kernel_bt/device_kernels.py: publish / finalize-unicast, owner reduce +
// multicast, materialize RMSNorm) in one launch, with its arithmetic: token t's owner is rank t % W. A CTA per token,
// one fragment (8 bf16) a thread:
//   1. publish: this rank's bf16 row (or its finalized MoE row, top-k weighted in k order, plus the gated shared
//      output) unicast into the owner's contribution mailbox [stage][rank][t / W];
//   2. on the owner: the W contributions summed in rank order (fp32 from 0) + residual, rounded to bf16 (prenorm),
//      multicast into every rank's prenorm mailbox [stage][t];
//   3. every rank: the prenorm row -> residual out, its sum of squares (thread: sequential over its 8 values; warps:
//      butterfly 16..1; block: warp 0 over the warp sums, butterfly 16..1), rsqrt, norm out (+ static FP8).
// Every wait is on a CTA of the same token on some rank, whose publish never waits, so CTAs in launch order progress.
struct Site2 {
  void* contrib_uc;               // this rank's contribution mailbox [kStages][W][local_cap][kHidden] bf16
  void* contrib_peer[8];          // every rank's (unicast, by rank)
  void* prenorm_mc;               // multicast prenorm mailbox [kStages][m_cap][kHidden] bf16
  void* prenorm_uc;               // this rank's view of it
  Ctrl* ctrl;
  int rank, world, m_cap, local_cap;
};

struct Args2 {
  Site2 site;
  const uint16_t* partial;      // [M, H] bf16 (all-reduce)
  const uint16_t* routed;       // [M*K, H] bf16 (finalize), rows indexed by permuted
  const uint16_t* expert_w;     // [M, K] bf16
  const int32_t* permuted;      // [M, K]; -1 skips a slot
  const uint16_t* shared;       // [M, H] bf16 gated shared expert output, or null
  const uint16_t* residual_in;  // [M, H] bf16
  const uint16_t* gamma;        // [H] bf16, the applied multiplier
  const float* fp8_scale;       // [1], or null
  uint16_t* residual_out;       // [M, H] bf16
  uint16_t* norm_out;           // [M, H] bf16, or null
  uint8_t* fp8_out;             // [M, H] e4m3, or null
  int m, top_k;
  float eps;
};

constexpr int kTwoThreads = kFrags;  // 512: a fragment a thread
constexpr int kTwoWarps = kTwoThreads / 32;

__device__ __forceinline__ size_t contrib_slot(const Site2& s, uint32_t stage, int src, int local_token, int frag) {
  return ((static_cast<size_t>(stage) * s.world + src) * s.local_cap + local_token) * kHidden +
         static_cast<size_t>(frag) * kVec;
}
__device__ __forceinline__ size_t prenorm_slot(const Site2& s, uint32_t stage, int token, int frag) {
  return (static_cast<size_t>(stage) * s.m_cap + token) * kHidden + static_cast<size_t>(frag) * kVec;
}
__device__ __forceinline__ uint4 sanitize4(uint4 v) {
  return make_uint4(sanitize_word(v.x), sanitize_word(v.y), sanitize_word(v.z), sanitize_word(v.w));
}
// A volatile 16-byte read, retried while any bf16 of it is the -0.0 sentinel (deadline as pull_frag).
__device__ __forceinline__ uint4 poll_frag(const uint16_t* p) {
  uint4 v = ld_volatile_v4(p);
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (frag_has_negzero(v)) {
    v = ld_volatile_v4(p);
    if (Q35_HOP_DEADLINE_NS != 0 && ((++spins & 255u) == 0)) {
      const unsigned long long now = gtimer();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > Q35_HOP_DEADLINE_NS) {
        fault_at_null();
      }
    }
  }
  return v;
}

template <int W, int TOPK, bool kFin>
__device__ __forceinline__ void two_shot_kernel(const Args2& a) {
  constexpr int kSlots = TOPK > 0 ? TOPK : kMaxTopK;
  __shared__ int32_t rows_s[kSlots];
  __shared__ float wts_s[kSlots];
  __shared__ float warp_sums[kTwoWarps];
  const int token = blockIdx.x, frag = threadIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int top_k = TOPK > 0 ? TOPK : a.top_k;
  if (kFin && threadIdx.x < top_k) {
    const int i = token * top_k + threadIdx.x;
    const int r = a.permuted[i];
    rows_s[threadIdx.x] = r;
    wts_s[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(a.expert_w[i]) << 16);
  }
  float gam[8];
  unpack8(ld_v4(a.gamma + static_cast<size_t>(frag) * kVec), gam);
  const float inv_s = a.fp8_out != nullptr ? inv_scale_of(*a.fp8_scale) : 1.f;
  if (kFin) __syncthreads();
  pdl_trigger();
  pdl_wait();
  const Hop h = hop_enter_two(a.site);
  // 1. publish
  uint4 mine;
  if constexpr (kFin) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int rows[kSlots];
#pragma unroll
    for (int k = 0; k < kSlots; ++k) rows[k] = k < top_k ? rows_s[k] : -1;
    uint4 src[kSlots];
#pragma unroll
    for (int k = 0; k < kSlots; ++k)
      src[k] = rows[k] >= 0 ? ld_v4(a.routed + static_cast<size_t>(rows[k]) * kHidden + static_cast<size_t>(frag) * kVec)
                            : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int k = 0; k < kSlots; ++k) {
      if (rows[k] >= 0) {
        const float w = wts_s[k];
        float f[8];
        unpack8(src[k], f);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], __fmul_rn(f[e], w));
      }
    }
    if (a.shared != nullptr) {
      float f[8];
      unpack8(ld_v4(a.shared + row_elem(token, frag)), f);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
    }
    mine = pack8(acc);
  } else {
    mine = ld_v4(a.partial + row_elem(token, frag));
  }
  const int owner = token % W, local_token = token / W;
  st_v4(static_cast<uint16_t*>(a.site.contrib_peer[owner]) + contrib_slot(a.site, h.stage, a.site.rank, local_token, frag),
        sanitize4(mine));
  // 2. the owner's reduce
  if (owner == a.site.rank) {
    const uint16_t* in = static_cast<const uint16_t*>(a.site.contrib_uc);
    uint4 v[W];
#pragma unroll
    for (int r = 0; r < W; ++r) v[r] = poll_frag(in + contrib_slot(a.site, h.stage, r, local_token, frag));
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < W; ++r) {
      float f[8];
      unpack8(v[r], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
    }
    float res[8];
    unpack8(ld_v4(a.residual_in + row_elem(token, frag)), res);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], res[e]);
    mc_st_v4(static_cast<uint16_t*>(a.site.prenorm_mc) + prenorm_slot(a.site, h.stage, token, frag), sanitize4(pack8(acc)));
    const uint4 z = make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair);
#pragma unroll
    for (int r = 0; r < W; ++r) st_v4(static_cast<uint16_t*>(a.site.contrib_uc) + contrib_slot(a.site, h.stage, r, local_token, frag), z);
  }
  // 3. materialize
  uint16_t* pre_uc = static_cast<uint16_t*>(a.site.prenorm_uc) + prenorm_slot(a.site, h.stage, token, frag);
  const uint4 packed = poll_frag(pre_uc);
  *reinterpret_cast<uint4*>(a.residual_out + row_elem(token, frag)) = packed;
  float x[8];
  unpack8(packed, x);
  float sum = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) sum = __fadd_rn(sum, __fmul_rn(x[e], x[e]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float t = lane < kTwoWarps ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, off));
    if (lane == 0) warp_sums[0] = t;
  }
  __syncthreads();
  const float total = warp_sums[0];
  float inv;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(__fadd_rn(total / static_cast<float>(kHidden), a.eps)));
  const uint4 norm = norm_apply(x, inv, gam);
  if (a.norm_out != nullptr) *reinterpret_cast<uint4*>(a.norm_out + row_elem(token, frag)) = norm;
  if (a.fp8_out != nullptr) *reinterpret_cast<uint2*>(a.fp8_out + row_elem(token, frag)) = quant_fp8(norm, inv_s);
  st_v4(pre_uc, make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair));
  hop_leave_two(a.site, h, gridDim.x);
}


__device__ __forceinline__ Hop hop_enter_two(const Site2& s) {
  Hop h;
  h.stage = ld_volatile_u32(&s.ctrl->stage);
  h.seq = ld_volatile_u32(&s.ctrl->seq);
  h.ticket = threadIdx.x == 0 ? atomicAdd(&s.ctrl->arrive, 1u) : 0u;
  return h;
}
__device__ __forceinline__ void hop_leave_two(const Site2& s, const Hop& h, uint32_t participants) {
  if (threadIdx.x == 0 && h.ticket == participants - 1) {
    s.ctrl->arrive = 0;
    s.ctrl->stage = (h.stage + 1) % kStages;
    s.ctrl->seq = h.seq + 1;
  }
}
}  // namespace q35hop_kernels

using q35hop_kernels::Args;
using q35hop_kernels::OprojArgs;

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_publish(Args a) {
  q35hop_kernels::publish_kernel(a);
}

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_publish_fin_top10(Args a) {
  q35hop_kernels::publish_fin_quad_kernel<10>(a);
}

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_publish_fin(Args a) {
  q35hop_kernels::publish_fin_quad_kernel<0>(a);
}

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_consume_w2(Args a) {
  q35hop_kernels::consume_kernel<2>(a);
}

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_consume_w4(Args a) {
  q35hop_kernels::consume_kernel<4>(a);
}

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_consume_w8(Args a) {
  q35hop_kernels::consume_kernel<8>(a);
}

extern "C" __global__ void __launch_bounds__(256, 1) q35hop_oproj_push_s4(OprojArgs o) {
  q35hop_kernels::oproj_push_kernel<4>(o);
}

extern "C" __global__ void __launch_bounds__(256, 1) q35hop_oproj_push_s8(OprojArgs o) {
  q35hop_kernels::oproj_push_kernel<8>(o);
}

using q35hop_kernels::Args2;

#define Q35HOP2_ENTRIES(W)                                                                                      \
  extern "C" __global__ void __launch_bounds__(q35hop_kernels::kTwoThreads, 2) q35hop2_ar_w##W(Args2 a) {        \
    q35hop_kernels::two_shot_kernel<W, 0, false>(a);                                                            \
  }                                                                                                             \
  extern "C" __global__ void __launch_bounds__(q35hop_kernels::kTwoThreads, 2) q35hop2_fin_top10_w##W(Args2 a) { \
    q35hop_kernels::two_shot_kernel<W, 10, true>(a);                                                            \
  }                                                                                                             \
  extern "C" __global__ void __launch_bounds__(q35hop_kernels::kTwoThreads, 2) q35hop2_fin_w##W(Args2 a) {       \
    q35hop_kernels::two_shot_kernel<W, 0, true>(a);                                                             \
  }
Q35HOP2_ENTRIES(2)
Q35HOP2_ENTRIES(4)
Q35HOP2_ENTRIES(8)
