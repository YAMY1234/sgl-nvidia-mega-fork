// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Router top-k of the DeepSeek-V4.1 target MoE for the low-batch routed-expert GEMMs
// (routed_expert_gemm.py, gate_topk_and_gemms): the Triton moe_fused_gate's routing (sqrtsoftplus via
// log1p, fp32 bias, top-6, renormalize, routed scale) with one CTA per row and one expert
// per thread, so a row's scoring and selection take a few warp reductions.
//
// Numerics follow moe_fused_gate bit for bit: the same libdevice exp / log1p / sqrt under
// Triton's reflection (flush to zero, approximate sqrt; see GATE_TOPK_NVRTC_OPTIONS), NaNs
// ranked above every finite score, the lowest expert id wins ties, the routed sum in the
// Triton reduction's order, the same approximate division (div.full.f32), then the scale.
//
// PDL: dependents (the route kernel) launch at once and read the ids after their own wait;
// the static bias is read before this kernel's grid-dependency wait, the logits after it.

namespace {
constexpr int kExperts = 384;
constexpr int kTopK = 6;
constexpr int kThreads = kExperts;
constexpr int kWarps = kThreads / 32;
constexpr int kCandidates = kWarps * kTopK;

// Order-preserving u32 key of a score (+0 for -0); 0 marks a taken expert.
__device__ __forceinline__ unsigned int Key(float v) {
  const unsigned int bits = __float_as_uint(v + 0.0f);
  return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

// The Triton kernel's fp32 division lowers to div.full.f32, not the IEEE div.rn.
__device__ __forceinline__ float DivFull(float a, float b) {
  float q;
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(a), "f"(b));
  return q;
}
}  // namespace

// grid = rows, block = 384. num_valid: rows >= *num_valid get weight 0 and id -1 (padded
// graph rows), as moe_fused_gate's num_token_non_padded. With input_ids (int32, or int64 when
// ids64), a row whose token is alt_token_id ranks with bias_alt instead of bias (DSV4.1 VL
// routing). eps > 0 normalizes by sum + eps, as renormalize_epsilon; else by the guarded sum.
extern "C" __global__ __launch_bounds__(kThreads) void dsv41_gate_topk(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ unsigned int cand_key[kCandidates];
  __shared__ int cand_id[kCandidates];
  __shared__ float cand_act[kCandidates];
  const int e = static_cast<int>(threadIdx.x);
  const int lane = e & 31;
  const int warp = e >> 5;
  const int row = static_cast<int>(blockIdx.x);
  const float b_main = bias[e];
  const float b_alt = bias_alt != nullptr ? bias_alt[e] : b_main;
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int valid = *num_valid;
  bool alt = false;
  if (input_ids != nullptr && row < valid) {
    const long long token = ids64 ? static_cast<const long long*>(input_ids)[row]
                                  : static_cast<long long>(static_cast<const int*>(input_ids)[row]);
    alt = token == alt_token_id;
  }
  const float b = alt ? b_alt : b_main;
  const float s = logits[row * kExperts + e];
  const float sp = s > 20.0f ? s : log1pf(expf(s));
  const float act = sqrtf(sp);
  const float biased = act + b;
  unsigned int key = Key(biased == biased ? biased : __int_as_float(0x7f800000));

  // This warp's top-6 (its ids are contiguous, so the lowest lane wins a tie).
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const unsigned int top = __reduce_max_sync(0xffffffffU, key);
    const unsigned int win = __reduce_min_sync(0xffffffffU, key == top ? static_cast<unsigned int>(e) : 0xffffffffU);
    if (static_cast<unsigned int>(e) == win) {
      cand_key[warp * kTopK + k] = key;
      cand_id[warp * kTopK + k] = e;
      cand_act[warp * kTopK + k] = act;
      key = 0U;
    }
  }
  __syncthreads();
  if (warp != 0) return;

  // Merge the 72 candidates: every global top-6 expert is in its warp's top-6.
  unsigned int ck[3];
  int ci[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int c = lane + 32 * j;
    ck[j] = c < kCandidates ? cand_key[c] : 0U;
    ci[j] = c < kCandidates ? cand_id[c] : kExperts;
  }
  float sel_w[kTopK];
  int sel_i[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    unsigned int best = max(max(ck[0], ck[1]), ck[2]);
    const unsigned int top = __reduce_max_sync(0xffffffffU, best);
    unsigned int mine = 0xffffffffU;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (ck[j] == top) mine = min(mine, static_cast<unsigned int>(ci[j]));
    }
    const int id = static_cast<int>(__reduce_min_sync(0xffffffffU, mine));
    int slot = -1;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (ck[j] == top && ci[j] == id) { slot = lane + 32 * j; ck[j] = 0U; }
    }
    const unsigned int owner = __ballot_sync(0xffffffffU, slot >= 0);
    const int src = __shfl_sync(0xffffffffU, slot, __ffs(owner) - 1);
    sel_w[k] = cand_act[src];
    sel_i[k] = id;
  }
  // Routed sum in the Triton reduction's order: two threads hold 4 of the 8 slots each.
  const float sum = ((sel_w[0] + sel_w[2]) + (sel_w[1] + sel_w[3])) + (sel_w[4] + sel_w[5]);
  const float norm = eps > 0.0f ? sum + eps : (sum > 0.0f ? sum : 1.0f);
  if (lane < kTopK) {
    float w = 0.0f;
    int id = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      if (lane == k) { w = sel_w[k]; id = sel_i[k]; }
    }
    w = DivFull(w, norm) * scale;
    if (row >= valid) { w = 0.0f; id = -1; }
    weights[row * kTopK + lane] = w;
    ids[row * kTopK + lane] = id;
  }
}
