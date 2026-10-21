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
// Built with NVRTC: C-linkage entries per instantiation; Q35_HOP_DEADLINE_NS (0: none) is the
// consumer's wait deadline, past which it faults at address 0x0 (fail-stop).

#include <cuda_bf16.h>

// NVRTC has no <stdint.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#ifndef Q35_HOP_DEADLINE_NS
#define Q35_HOP_DEADLINE_NS 0ull
#endif

namespace q35hop {


constexpr int kHidden = 4096;
constexpr int kVec = 8;
constexpr int kFrags = kHidden / kVec;
constexpr int kStages = 3;
constexpr int kRowCtas = 4;
constexpr int kRowThreads = 128;
constexpr int kRowWarps = kRowThreads / 32;
constexpr uint32_t kNegZeroPair = 0x80008000u;
static_assert(kRowCtas * kRowThreads == kFrags, "a row is one fragment per thread of the cluster");
static_assert(kRowWarps == 4, "row_sumsq_fold folds exactly four warp sums per CTA");

// Per-process control block in device memory.
struct Ctrl {
  uint32_t stage;   // stage of the next hop
  uint32_t arrive;  // CTA tickets of the running consumer launch
  uint32_t seq;     // hops completed
  uint32_t pad;
};

// Passed by value as a kernel parameter.
struct Site {
  void* mc;                     // multicast base of the mailbox [kStages][world][m_cap][kHidden] bf16
  void* uc;                     // this rank's unicast view of the same mailbox
  Ctrl* ctrl;
  int rank, world, m_cap;
};

// ---------------------------------------------------------------- PTX

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void mc_st_v4(void* mc, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void mc_st_v2(void* mc, uint2 v) {
  asm volatile("multimem.st.relaxed.sys.global.v2.bf16x2 [%0], {%1,%2};" ::"l"(mc), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// Two f32 to an e4m3 pair; lo lands in the low byte.
__device__ __forceinline__ uint16_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// ---------------------------------------------------------------- bf16

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void unpack8(uint4 v, float f[8]) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ uint4 pack8(const float f[8]) {
  return make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]), pack_bf16x2(f[4], f[5]),
                    pack_bf16x2(f[6], f[7]));
}
__device__ __forceinline__ bool word_has_negzero(uint32_t w) {
  return (w & 0xFFFFu) == 0x8000u || (w >> 16) == 0x8000u;
}
__device__ __forceinline__ bool frag_has_negzero(uint4 v) {
  return word_has_negzero(v.x) | word_has_negzero(v.y) | word_has_negzero(v.z) | word_has_negzero(v.w);
}
__device__ __forceinline__ uint32_t sanitize_word(uint32_t w) {
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000FFFFu;
  return w;
}

// ---------------------------------------------------------------- stage protocol

__device__ __forceinline__ size_t slot(const Site& s, uint32_t stage, int src, int token, int frag) {
  return ((static_cast<size_t>(stage) * s.world + src) * s.m_cap + token) * kHidden +
         static_cast<size_t>(frag) * kVec;
}

struct Hop {
  uint32_t stage;
  uint32_t seq;
  uint32_t ticket;  // thread 0 only
};

// Consumer launch, every thread of every participating CTA, after griddepcontrol.wait. Thread
// 0 takes a ticket whose value is consumed only in hop_leave, so the atomic stays off the
// critical path.
__device__ __forceinline__ Hop hop_enter(const Site& s) {
  Hop h;
  h.stage = ld_volatile_u32(&s.ctrl->stage);
  h.seq = ld_volatile_u32(&s.ctrl->seq);
  h.ticket = threadIdx.x == 0 ? atomicAdd(&s.ctrl->arrive, 1u) : 0u;
  return h;
}

// Thread 0 of every participating CTA; the last arrival advances the stage. No CTA of this
// launch can read the advanced value, since every CTA read the stage before taking its ticket.
__device__ __forceinline__ void hop_leave(const Site& s, const Hop& h, uint32_t participants) {
  if (threadIdx.x == 0 && h.ticket == participants - 1) {
    s.ctrl->arrive = 0;
    s.ctrl->stage = (h.stage + 1) % kStages;
    s.ctrl->seq = h.seq + 1;
  }
}

// Producer-only launch, after griddepcontrol.wait.
__device__ __forceinline__ uint32_t stage_peek(const Site& s) { return ld_volatile_u32(&s.ctrl->stage); }

// ---------------------------------------------------------------- producer

// This rank's final bf16 partial for one fragment; exactly once per (hop, rank, token, frag).
// A launch that also stores its own outputs should push first: the push starts the flight.
__device__ __forceinline__ void push_frag(const Site& s, uint32_t stage, int token, int frag, uint4 v) {
  v = make_uint4(sanitize_word(v.x), sanitize_word(v.y), sanitize_word(v.z), sanitize_word(v.w));
  mc_st_v4(static_cast<uint16_t*>(s.mc) + slot(s, stage, s.rank, token, frag), v);
}

// Same for 4 consecutive bf16 (half a fragment, columns [4q, 4q + 4)); a fragment may be pushed
// as two halves by different threads.
__device__ __forceinline__ void push_quad(const Site& s, uint32_t stage, int token, int quad, uint2 v) {
  v = make_uint2(sanitize_word(v.x), sanitize_word(v.y));
  mc_st_v2(static_cast<uint16_t*>(s.mc) + slot(s, stage, s.rank, token, quad >> 1) + (quad & 1) * 4, v);
}

// ---------------------------------------------------------------- consumer

// A peer that missed the wait deadline: store to address 0x0, then trap (fail-stop).
__device__ __forceinline__ void fault_at_null() {
  asm volatile("{\n.reg .u64 a;\nmov.u64 a, 0;\nst.volatile.global.u32 [a], 0;\ntrap;\n}" ::: "memory");
}

// Sum over ranks 0..W-1 (fp32, from 0) of one fragment; spins on the sentinel. A wait past
// Q35_HOP_DEADLINE_NS (if not 0) stops the kernel.
template <int W>
__device__ __forceinline__ bool pull_frag(const Site& s, uint32_t stage, int token, int frag, float acc[8]) {
  const uint16_t* uc = static_cast<const uint16_t*>(s.uc);
  uint4 v[W];
#pragma unroll
  for (int r = 0; r < W; ++r) v[r] = ld_volatile_v4(uc + slot(s, stage, r, token, frag));
  uint32_t pending = 0;
#pragma unroll
  for (int r = 0; r < W; ++r) pending |= frag_has_negzero(v[r]) ? (1u << r) : 0u;
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (pending) {
#pragma unroll
    for (int r = 0; r < W; ++r)
      if (pending & (1u << r)) v[r] = ld_volatile_v4(uc + slot(s, stage, r, token, frag));
    uint32_t still = 0;
#pragma unroll
    for (int r = 0; r < W; ++r)
      if ((pending & (1u << r)) && frag_has_negzero(v[r])) still |= 1u << r;
    pending = still;
    if (Q35_HOP_DEADLINE_NS != 0 && pending && ((++spins & 255u) == 0)) {
      const unsigned long long now = gtimer();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > Q35_HOP_DEADLINE_NS) {
        fault_at_null();
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int r = 0; r < W; ++r) {
    float f[8];
    unpack8(v[r], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  return true;
}

// Write the sentinel back into the W slots this thread consumed (any time after pull_frag).
template <int W>
__device__ __forceinline__ void rearm_frag(const Site& s, uint32_t stage, int token, int frag) {
  uint16_t* uc = static_cast<uint16_t*>(s.uc);
  const uint4 z = make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair);
#pragma unroll
  for (int r = 0; r < W; ++r) st_v4(uc + slot(s, stage, r, token, frag), z);
}

// ---------------------------------------------------------------- row finish

// prenorm = bf16(acc + residual); pre[] gets the bf16-rounded values the norm uses.
__device__ __forceinline__ uint4 residual_add(const float acc[8], uint4 residual, float pre[8]) {
  float r[8];
  unpack8(residual, r);
#pragma unroll
  for (int e = 0; e < 8; ++e) pre[e] = __fadd_rn(acc[e], r[e]);
  const uint4 out = pack8(pre);
  unpack8(out, pre);
  return out;
}

__device__ __forceinline__ float sumsq8(const float x[8]) {
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) s = __fadd_rn(s, __fmul_rn(x[e], x[e]));
  return s;
}

struct RowSmem {
  float xs[kRowCtas][kRowWarps];
  uint64_t mbar;
};

// Prologue of every CTA of the row cluster, all threads, before griddepcontrol.wait: arm the
// exchange barrier and make it visible to the peers before any of them can send.
__device__ __forceinline__ void row_init(RowSmem& rs) {
  if (threadIdx.x == 0) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(a) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a),
                 "r"(static_cast<uint32_t>(sizeof(rs.xs)))
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Row total of the per-thread values: lane butterfly per warp, then each warp sends its sum to
// every CTA of the cluster with st.async; each CTA waits on its own mbarrier (no release fence,
// so this thread's earlier remote stores are not waited for) and folds per CTA as
// (w0 + w2) + (w1 + w3), then the CTAs in rank order from 0.
__device__ __forceinline__ float row_sumsq_fold(float v, RowSmem& rs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  if (lane < kRowCtas) {
    const uint32_t me = cluster_ctarank();
    const uint32_t slot_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.xs[me][warp]));
    const uint32_t mbar_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    uint32_t slot_r, mbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(slot_r) : "r"(slot_l), "r"(lane));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mbar_r) : "r"(mbar_l), "r"(lane));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(slot_r),
                 "f"(v), "r"(mbar_r)
                 : "memory");
  }
  {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile(
        "{\n\t.reg .pred P;\n\tWAIT_ROW:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n\t"
        "@!P bra WAIT_ROW;\n\t}" ::"r"(a)
        : "memory");
  }
  float total = 0.f;
#pragma unroll
  for (int c = 0; c < kRowCtas; ++c)
    total = __fadd_rn(total, __fadd_rn(__fadd_rn(rs.xs[c][0], rs.xs[c][2]), __fadd_rn(rs.xs[c][1], rs.xs[c][3])));
  return total;
}

__device__ __forceinline__ float inv_rms(float row_sumsq, float eps) {
  return rsqrtf(__fadd_rn(row_sumsq / static_cast<float>(kHidden), eps));
}

// bf16((x * inv) * gamma), gamma being the applied multiplier.
__device__ __forceinline__ uint4 norm_apply(const float pre[8], float inv, const float gamma[8]) {
  float out[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) out[e] = __fmul_rn(__fmul_rn(pre[e], inv), gamma[e]);
  return pack8(out);
}

// The reciprocal the static per-tensor FP8 quant multiplies by.
__device__ __forceinline__ float inv_scale_of(float scale) { return div_full(1.f, scale); }

// e4m3(clamp(x * inv_scale, +-448)) of the bf16 norm output, 8 bytes.
__device__ __forceinline__ uint2 quant_fp8(uint4 norm_bf16, float inv_scale) {
  float q[8];
  unpack8(norm_bf16, q);
  uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float a = fminf(fmaxf(__fmul_rn(q[2 * j], inv_scale), -448.f), 448.f);
    const float b = fminf(fmaxf(__fmul_rn(q[2 * j + 1], inv_scale), -448.f), 448.f);
    w[j] = cvt_e4m3x2(a, b);
  }
  return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

}  // namespace q35hop

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

}  // namespace q35hop_kernels

using q35hop_kernels::Args;

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
