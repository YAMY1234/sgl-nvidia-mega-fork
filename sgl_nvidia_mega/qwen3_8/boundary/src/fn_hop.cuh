// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The decode hop's device protocol (namespace fnhop): the mailbox site, the stage protocol, the
// producer's push of a bf16 fragment and the consumer's rank-order pull and row finish. hop.cu
// builds the stand-alone hop launches on it; a kernel that pushes its own partial into the hop
// from its epilogue includes this protocol and pushes
// each fragment with push_frag(site, stage_peek(site), token, frag, v) after griddepcontrol.wait.
// Includers that typedef the fixed-width integers themselves must use these same types.

#pragma once

#include <cuda_bf16.h>

#include <stdint.h>
#include <stddef.h>

#ifndef FN_HOP_DEADLINE_NS
#define FN_HOP_DEADLINE_NS 0ull
#endif

namespace fnhop {


#ifndef FN_HOP_HIDDEN
#define FN_HOP_HIDDEN 2560
#endif
constexpr int kHidden = FN_HOP_HIDDEN;
constexpr int kVec = 8;
constexpr int kFrags = kHidden / kVec;
constexpr int kStages = 3;
constexpr int kRowCtas = 4;
constexpr int kRowThreads = 128;
constexpr int kRowWarps = kRowThreads / 32;
constexpr uint32_t kNegZeroPair = 0x80008000u;
constexpr int kFragsPerCta = kFrags / kRowCtas;
static_assert(kFrags % kRowCtas == 0 && kFragsPerCta <= kRowThreads);
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
// FN_HOP_DEADLINE_NS (if not 0) stops the kernel.
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
    if (FN_HOP_DEADLINE_NS != 0 && pending && ((++spins & 255u) == 0)) {
      const unsigned long long now = gtimer();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > FN_HOP_DEADLINE_NS) {
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


}  // namespace fnhop
