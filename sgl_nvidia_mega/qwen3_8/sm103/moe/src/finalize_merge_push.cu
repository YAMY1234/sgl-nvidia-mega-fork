// SPDX-License-Identifier: Apache-2.0
// Isolated ordinary-FFN producer prototype. No runtime registration or exit ownership.
// The CPU builder must provide the frozen boundary helper as fn_hop.cuh.
#include <cuda_bf16.h>
#include "fn_hop.cuh"

static_assert(sizeof(fnhop::Site) == 40 && alignof(fnhop::Site) == 8);
static_assert(FN_HIDDEN == fnhop::kHidden && FN_HIDDEN % 4 == 0);

// Launch grid=(ceil(H/(4*blockDim.x)), M), blockDim.x=128. The caller joins the
// real routed and shared streams before this launch. Gate is native GATE_ONLY's
// FP32 result; it is not a BF16 gated-shared intermediate. Deferred is excluded.
extern "C" __global__ void q38_moe_finalize_merge_push(
    const __nv_bfloat16* rows, const __nv_bfloat16* weights,
    const __nv_bfloat16* shared_down, const float* gate,
    __nv_bfloat16* local_out, fnhop::Site site) {
  fnhop::pdl_wait();
  const uint32_t stage = fnhop::stage_peek(site);
  const int token = blockIdx.y;
  const int h0 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (h0 >= FN_HIDDEN) return;
  __nv_bfloat16 result[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < FN_TOP_K; ++k) {
      const float r = __bfloat162float(rows[(token * FN_TOP_K + k) * FN_HIDDEN + h0 + c]);
      const float w = __bfloat162float(weights[token * FN_TOP_K + k]);
      total = __fadd_rn(total, __fmul_rn(r, w));
    }
    const float routed = __bfloat162float(__float2bfloat16_rn(total));
    const float shared = __bfloat162float(shared_down[token * FN_HIDDEN + h0 + c]);
    result[c] = __float2bfloat16_rn(__fmaf_rn(gate[token], shared, routed));
  }
  const uint2 packed = make_uint2(
      static_cast<uint32_t>(__bfloat16_as_ushort(result[0])) |
          (static_cast<uint32_t>(__bfloat16_as_ushort(result[1])) << 16),
      static_cast<uint32_t>(__bfloat16_as_ushort(result[2])) |
          (static_cast<uint32_t>(__bfloat16_as_ushort(result[3])) << 16));
  // Publish rounded local partial only. Helper canonicalizes -0 in mailbox;
  // the debug tensor preserves the original BF16 bits, including signed zero.
  fnhop::push_quad(site, stage, token, h0 / 4, packed);
#pragma unroll
  for (int c = 0; c < 4; ++c)
    local_out[token * FN_HIDDEN + h0 + c] = result[c];
  // Consumer's pdl_wait also requires completion of every CTA. No early-ready
  // flag, stage advance, rearm, AR, or HC operation belongs to this producer.
}
