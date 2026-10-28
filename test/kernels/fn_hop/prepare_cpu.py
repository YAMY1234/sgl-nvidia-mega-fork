"""Offline SM103 HC-hop build. Expanded source, outputs and caches stay on NVMe."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--image-digest", required=True)
p.add_argument("--sglang-jit", type=Path, required=True)
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
if torch.cuda.is_available():
    raise RuntimeError("prepare_cpu must have no GPU")
root = Path(__file__).resolve().parents[3]
src = root / "sgl_nvidia_mega/qwen3_8/boundary/src"
out = a.out
out.mkdir(parents=True, exist_ok=False)
cfg = json.loads(a.config.read_text())
cfg = cfg.get("text_config", cfg)
config = {k: cfg[k] for k in ("hidden_size", "hc_count", "hc_lowrank", "rms_norm_eps")}
h, branches, m = config["hidden_size"], config["hc_count"], 4
if h != 2560 or branches != 4:
    raise RuntimeError(f"this finite qualification requires FN TP4/M4/HC4, got {config}")
subprocess.run([
    "nvcc", "-std=c++17", "-O3", "-arch=sm_103a", "--cubin", "--fmad=false",
    "-Xptxas=-v", f"-DFN_HOP_HIDDEN={h}", "-DFN_HOP_DEADLINE_NS=2000000000ull",
    str(src / "hop.cu"), "-o", str(out / "hop.cubin"),
], check=True)
print("HC_HOP_CPU_CUBIN_PASS", flush=True)
native = out / "native_hc.cu"
native.write_text('''#include "elementwise/hc_combine_decode.cuh"
#include "tvm_ffi_utils.h"
void hc_native(tvm::ffi::TensorView block, tvm::ffi::TensorView residual,
               tvm::ffi::TensorView gates, tvm::ffi::TensorView gamma,
               tvm::ffi::TensorView combined, tvm::ffi::TensorView normed, double eps) {
  sglang::HcCombineNormDecode<2,true>::run(block,residual,gates,gamma,combined,normed,float(eps));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hc_native, hc_native);
''')
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.comm import gen_trtllm_mnnvl_comm_module
spec = gen_jit_spec(
    "fn_hop_native_hc_2a92",
    [native],
    extra_cuda_cflags=["-std=c++20", "-DSGL_CUDA_ARCH=1030", "-gencode=arch=compute_103a,code=sm_103a"],
    extra_include_paths=[a.sglang_jit / "include", a.sglang_jit / "csrc"],
    use_fast_math=False,
)
spec.build()
shutil.copy2(spec.get_library_path(), out / "native_hc.so")
fi_spec = gen_trtllm_mnnvl_comm_module()
fi_spec.build()
shutil.copy2(fi_spec.get_library_path(), out / "native_fi.so")
print("HC_HOP_NATIVE_REFERENCES_PASS", flush=True)
torch.manual_seed(20261028)
torch.set_num_threads(8)
count = 12
local = torch.randn(count, 4, m, h).bfloat16()
residual = torch.randn(count, m, branches * h).bfloat16()
gates = torch.randn(count, m, 8, branches)
local[1].zero_(); residual[1].zero_()
local[2] *= .01; residual[2] *= .01
local[3] *= 10; residual[3] *= 10
local[4, 1] = -local[4, 0]; local[4, 3] = -local[4, 2]
local[5].fill_(1); residual[5].fill_(-4)
local[6].zero_(); local[6].view(torch.int16)[::2] = -32768
local[7, 0, 0, 3] = float("inf")
local[8, 1, 0, 17] = float("nan")
local[9, 0, 0, 19] = float("inf"); local[9, 1, 0, 19] = -float("inf")
gates[10].fill_(100); gates[10, :, :, ::2] = -100
local[11] *= 2**-60; residual[11] *= 2**-60
gamma = (torch.randn(branches * h) * .1).bfloat16()
torch.save(dict(config=config, rows=m, local=local, residual=residual, gates=gates, gamma=gamma), out / "fixtures.pt")
files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
(out / "MANIFEST.json").write_text(json.dumps(dict(
    target="sm_103a", config=config, rows=m, cases=count, files=files,
    native_parent="2a92c89d22d660197949fc36e6f4fb6b66332e14", gpu_used=False,
    source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir()},
    image_digest=a.image_digest,
    flashinfer_version=importlib.metadata.version("flashinfer-python"),
    torch_version=torch.__version__,
    native_hc_header_sha256=hashlib.sha256(
        (a.sglang_jit / "csrc/elementwise/hc_combine_decode.cuh").read_bytes()).hexdigest(),
    note="native FI rebuilt from installed image package; native HC rebuilt against 2a92",
), indent=2) + "\n")
print("HC_HOP_CPU_ARTIFACTS_PASS", flush=True)
