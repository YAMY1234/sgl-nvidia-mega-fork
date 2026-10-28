// SPDX-License-Identifier: Apache-2.0
// Original FN/MTP slot0->9 FP32 multiply/add, one BF16 store. Ordinary stream launch.
#include <cuda_bf16.h>
extern "C" __global__ void q38_moe_finalize(const __nv_bfloat16* rows,
 const __nv_bfloat16* weights,__nv_bfloat16* out) {
 const int t=blockIdx.y;
#pragma unroll
 for(int q=0;q<2;++q) {
  const int h=blockIdx.x*256+threadIdx.x+q*128;
  if(h<FN_HIDDEN) {
   float total=0.f;
#pragma unroll
   for(int k=0;k<FN_TOP_K;++k)
    total=__fadd_rn(total,__fmul_rn(__bfloat162float(rows[(t*FN_TOP_K+k)*FN_HIDDEN+h]),__bfloat162float(weights[t*FN_TOP_K+k])));
   out[t*FN_HIDDEN+h]=__float2bfloat16_rn(total);
  }
 }
}
