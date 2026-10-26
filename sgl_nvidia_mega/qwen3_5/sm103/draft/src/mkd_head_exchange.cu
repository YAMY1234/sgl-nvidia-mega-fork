// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Q35FLY HP2: the last completing eight-CTA head cluster performs the exchange.
// A non-last cluster never waits for another cluster; it retires. No grid-wide
// residency assumption or spin on an unscheduled head CTA is required.

#include "mkd_head_partial.cuh"
#include "mkd_xchg.cuh"

namespace sglang {
inline constexpr int kHeadExchangeCluster = 8;

SGL_DEVICE void head_exchange_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\t"
               "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

SGL_DEVICE bool head_exchange_last_cluster(uint32_t* completed) {
  __shared__ uint32_t s_last;
  const int c = static_cast<int>(blockIdx.x) % kHeadExchangeCluster;
  // The head's stores (including the NaN rescan's logits) precede the leader's
  // acq_rel ticket. The cluster barrier collects all eight CTA completions.
  __syncthreads();
  if (threadIdx.x == 0) __threadfence();
  head_exchange_cluster_sync();
  if (c == 0 && threadIdx.x == 0) {
    uint32_t before;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(before) : "l"(completed) : "memory");
    s_last = before == gridDim.x / kHeadExchangeCluster - 1;
    if (s_last) {
      // All clusters have taken their ticket; no later head writer can race
      // this reset. The next head depends on this whole kernel completing.
      asm volatile("st.release.gpu.global.u32 [%0], 0;" :: "l"(completed) : "memory");
    }
  }
  head_exchange_cluster_sync();
  uint32_t remote, last;
  const uint32_t local = device::ptx::to_shared(&s_last);
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(local));
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(last) : "r"(remote) : "memory");
  // No cluster CTA may retire while peers still read its shared memory.
  head_exchange_cluster_sync();
  return last != 0;
}

struct MkdHeadExchangeBf16Params {
  MkdHeadPartialBf16Params compute;
  MkdXchgParams exchange;
  uint32_t* completed;
};
struct MkdHeadExchangeMx8Params {
  MkdHeadPartialMx8Params compute;
  MkdXchgParams exchange;
  uint32_t* completed;
};
static_assert(sizeof(MkdHeadExchangeBf16Params) == 336, "Python/CUDA BF16 HP2 ABI");
static_assert(sizeof(MkdHeadExchangeMx8Params) == 376, "Python/CUDA MXFP8 HP2 ABI");

template <bool kPartial>
SGL_DEVICE void head_exchange_bf16(const MkdHeadExchangeBf16Params& p) {
  if constexpr (kPartial)
    mkd_head_kernel<4096, 16, true>(p.compute.head, MkdHeadPartialInputWork{p.compute.partial});
  else
    mkd_head_kernel<4096, 16, true>(p.compute.head);
  if (head_exchange_last_cluster(p.completed))
    mkd_xchg_kernel<4096, MKD_WORLD, true, 8, 512, false, true, true>(p.exchange);
}

template <bool kPartial>
SGL_DEVICE void head_exchange_mx8(const MkdHeadExchangeMx8Params& p) {
  if constexpr (kPartial)
    head_mx8_kernel<4096, 8, 1, true>(p.compute.head, MkdHeadPartialInputWork{p.compute.partial});
  else
    head_mx8_kernel<4096, 8, 1, true>(p.compute.head);
  if (head_exchange_last_cluster(p.completed))
    mkd_xchg_kernel<4096, MKD_WORLD, true, 8, 512, false, true, true>(p.exchange);
}
}  // namespace sglang

#define HEAD_EXCHANGE_ENTRY(format, partial, suffix) \
  extern "C" __global__ void __launch_bounds__(512, 1) mkd_head_exchange_##suffix( \
      const sglang::MkdHeadExchange##format##Params __grid_constant__ p)

HEAD_EXCHANGE_ENTRY(Bf16, false, bf16) { sglang::head_exchange_bf16<false>(p); }
HEAD_EXCHANGE_ENTRY(Bf16, true, bf16_partial) { sglang::head_exchange_bf16<true>(p); }
HEAD_EXCHANGE_ENTRY(Mx8, false, mx8) { sglang::head_exchange_mx8<false>(p); }
HEAD_EXCHANGE_ENTRY(Mx8, true, mx8_partial) { sglang::head_exchange_mx8<true>(p); }
