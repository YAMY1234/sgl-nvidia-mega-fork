// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Router top-k of the DeepSeek-V4.1 target MoE: gate_topk.cuh's GateTopK for 384 experts,
// top-6; grid = rows, block = 384.

#include "gate_topk.cuh"

extern "C" __global__ __launch_bounds__(384) void dsv41_gate_topk(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps) {
  GateTopK<384, 6>(logits, bias, bias_alt, input_ids, ids64, alt_token_id, num_valid, weights, ids, scale, eps);
}
