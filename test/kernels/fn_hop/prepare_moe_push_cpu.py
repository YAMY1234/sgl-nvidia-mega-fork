"""Compile the unchanged native shared producer for a finite MoE-hop test.

Run inside the qualified image without GPUs. Inputs and expanded sources stay
on node scratch; the batch wrapper archives the resulting directory once.
"""
import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import shutil

import torch

p = argparse.ArgumentParser()
p.add_argument("--sglang", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
assert not torch.cuda.is_available()
import sglang
assert Path(sglang.__file__).is_relative_to(a.sglang), sglang.__file__
out = a.out
out.mkdir(exist_ok=False)
unquant_path = a.sglang / "sglang/srt/layers/quantization/unquant.py"
tree = ast.parse(unquant_path.read_text())
tables = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                "_BF16_SPLITK_TUNED_TACTICS", "_BF16_WARP_SPLITK_TUNED_TACTICS"
            ):
                tables[target.id] = ast.literal_eval(node.value)

from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv
from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as native
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import prefer_direct_bf16_gemm_sm100
import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as ext
from cutlass.cute.runtime import from_dlpack, make_fake_tensor, make_fake_stream

handles = [ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
           for ffi in (False, True)
           for path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=ffi)]
records = []
for rows in (1, 4):
    gu_shape, down_shape = (rows, 320, 2560), (rows, 2560, 160)
    assert gu_shape not in tables["_BF16_WARP_SPLITK_TUNED_TACTICS"]
    tactic = tables["_BF16_SPLITK_TUNED_TACTICS"][gu_shape]
    direct = prefer_direct_bf16_gemm_sm100(*gu_shape)
    print("NATIVE_SHARED_DISPATCH", gu_shape, "direct", direct, "splitk", tactic, flush=True)
    if direct:
        raise RuntimeError("native shared gate/up selects direct GEMM; compile that exact entry before GPU")
    assert down_shape not in tables["_BF16_SPLITK_TUNED_TACTICS"]
    assert down_shape not in tables["_BF16_WARP_SPLITK_TUNED_TACTICS"]
    assert not tgv.use_cutedsl_bf16_gemm(*down_shape)
    tensors = [make_fake_tensor(
        cutlass.BFloat16, tuple(cute.sym_int32() for _ in range(3)),
        tuple(1 if d == leading else cute.sym_int64() for d in range(3)),
        assumed_align=32,
    ) for leading in (2, 1, 1)]
    # Preserve the native runtime's i32 dynamic shapes and i64 strides.
    from cutlass._mlir import ir
    cpu = (torch.empty((1, 8, 128), dtype=torch.bfloat16),
           torch.empty((1, 8, 128), dtype=torch.bfloat16).transpose(-2, -1),
           torch.empty((1, 8, 128), dtype=torch.bfloat16).transpose(-2, -1))
    with ir.Context(), ir.Location.unknown():
        for fake, value, leading in zip(tensors, cpu, (2, 1, 1)):
            actual = from_dlpack(value, assumed_align=32).mark_layout_dynamic(leading_dim=leading)
            assert str(fake.__get_mlir_types__()[0]) == str(actual.__get_mlir_types__()[0]).replace("generic", "gmem")
    op = native.SplitKDenseGemmKernel(tactic=native.SplitKTactic(*tactic), use_pdl=True, has_bias=False)
    compiled = ext.compile(native._bmm_no_bias, op, *tensors, make_fake_stream(),
                           options="--gpu-arch sm_103a --enable-tvm-ffi --keep-cubin")
    compiled.export_to_c(object_file_path=str(out / f"shared-gu-m{rows}.o"), function_name="shared_gu")
    records.append(dict(rows=rows, gu_shape=gu_shape, tactic=tactic, gu_entry="shared_gu",
                        gu_source_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                        down_shape=down_shape, down="native F.linear fallback; no listed SplitK/TGV path"))

# The actual CUDA activation path is native ActivationKernel<BF16,true>, fast math.
# I160 is divisible by the SM103 16-BF16 vector, so the JIT path is applicable.
jit_root = a.sglang / "sglang/kernels/jit"
source = out / "native_silu.cu"
source.write_text('''#include "elementwise/activation.cuh"
#include "tvm_ffi_utils.h"
void silu_native(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  sglang::ActivationKernel<__nv_bfloat16,true>::run_activation(input,output,"silu");
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_native,silu_native);
''')
from flashinfer.jit.core import gen_jit_spec
spec = gen_jit_spec("fn_hop_native_shared_silu", [source],
    extra_cuda_cflags=["-std=c++20", "-DSGL_CUDA_ARCH=1030", "--use_fast_math",
                      "-gencode=arch=compute_103a,code=sm_103a"],
    extra_include_paths=[jit_root / "include", jit_root / "csrc"], use_fast_math=False)
spec.build()
shutil.copy2(spec.get_library_path(), out / "native_silu.so")
(out / "SHARED-NATIVE.json").write_text(json.dumps(dict(
    sglang_file=str(sglang.__file__), unquant_sha256=hashlib.sha256(unquant_path.read_bytes()).hexdigest(),
    activation_header_sha256=hashlib.sha256((jit_root / "csrc/elementwise/activation.cuh").read_bytes()).hexdigest(),
    dispatch=records, gpu_used=False,
    scope="CPU compile only; native BF16 gate/up + SiLU + F.linear down reference, not captured outputs",
), indent=2) + "\n")
print("MOE_HOP_NATIVE_SHARED_CPU_PASS", flush=True)
