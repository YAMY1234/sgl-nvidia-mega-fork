"""Load-only real-L23 MoE branches for the isolated hop producer test.

Native shared outputs are freshly computed, never taken from an FP64 fixture.
No model/executor registration is performed by this test helper.
"""
import ctypes
import io
import json
import os
from pathlib import Path
import tarfile

import torch
import tvm_ffi

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_8.moe.config import Shape
from sgl_nvidia_mega.qwen3_8.sm103.moe import router, mkm, finalize


class MoeFixture:
    def __init__(self, *, rank, rows, work, qc):
        import sglang
        assert Path(sglang.__file__).is_relative_to(Path(os.environ["FN_SGLANG"])), sglang.__file__
        self.work, self.rows = work, rows
        self.shape = Shape(2560, 512, 10, 640, 4, 4)
        saved = torch.load(qc / f"lanes/hc2/router-mkm-chain-v1/prepared/rank{rank}.pt",
                           map_location="cpu", weights_only=True)
        self.layer = {k: v.cuda() for k, v in saved["layer"].items()}
        self.router = {k: v.cuda() for k, v in saved["router"].items()}
        self.cases = [{k: v.cuda() for k, v in c.items() if isinstance(v, torch.Tensor)}
                      for c in saved["cases"] if c["meta"]["rows"] == rows]
        self.case_metadata = [c["meta"] for c in saved["cases"] if c["meta"]["rows"] == rows]
        with tarfile.open(qc / "lanes/mkmoe/shared-cpu-v4/prepared.tar") as t:
            raw = t.extractfile(f"prepared/shared-rank{rank}.pt").read()
        sh = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["shared"]
        self.wgu, self.wd, self.gate_weight = (sh[k].cuda() for k in ("wgu", "wd", "gate"))
        self.x = self.cases[0]["hidden"].clone()
        self.gu = torch.empty((rows, 320), device="cuda", dtype=torch.bfloat16)
        self.act = torch.empty((rows, 160), device="cuda", dtype=torch.bfloat16)
        self.gate = torch.empty(rows, device="cuda", dtype=torch.float32)
        self.local = torch.empty_like(self.x)
        self.routed_stream = torch.cuda.Stream()

        # Explicitly load the same image's FI AOT; a GPU compiler miss is fatal.
        import flashinfer.fused_moe.core as core
        meta = json.loads((work / "moe/prepared/native.json").read_text())
        native_path = Path(meta["image_path"])
        assert native_path.stat().st_size == meta["size"]
        factory = core.gen_trtllm_gen_fused_moe_sm100_module
        def aot_only(*args, **kwargs):
            spec = factory(*args, **kwargs)
            assert native_path in spec.get_library_paths()
            spec.build_and_load = lambda: spec.load(native_path)
            spec.get_library_paths = lambda: (native_path,)
            return spec
        core.gen_trtllm_gen_fused_moe_sm100_module = aot_only
        router.build(torch.cuda.current_device(), self.shape)
        mkm.build(torch.cuda.current_device(), self.shape)
        finalize.build(torch.cuda.current_device(), self.shape)
        self.maps = mkm.weight_maps(self.layer["gemm1_weights"], self.layer["gemm2_weights"], self.shape)

        from sglang.kernels.jit.cute_aot_cache import _load_object
        self.gu_native = _load_object(work / f"shared/prepared/shared-gu-m{rows}.o", "shared_gu", True)
        self.silu_native = tvm_ffi.load_module(str(work / "shared/prepared/native_silu.so"))
        raw = json.loads((work / "merge/prepared/CPU-PASS.json").read_text())
        self.merge_kernels = {}
        for record in raw["compiled"]:
            name = record["name"]
            self.merge_kernels[name] = jit.load((work / f"merge/prepared/{name}.cubin").read_bytes(),
                                                 record["symbol"], (512, 1, 1), record["shared"])
        self.push_kernel = jit.load((work / "merge/prepared/finalize_merge_push.cubin").read_bytes(),
                                    "q38_moe_finalize_merge_push", (128, 1, 1), 0)

    def native_experts(self, case, *, do_finalize):
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer.tllm_enums import SfLayout
        from sglang.srt.layers.moe.utils import RoutingMethodType
        w, s = self.layer, self.shape
        return trtllm_fp4_block_scale_moe(
            routing_logits=case["logits"], routing_bias=None,
            hidden_states=case["xq"], hidden_states_scale=case["sf"].view(torch.float8_e4m3fn),
            hidden_states_scale_layout=SfLayout.layout_linear,
            gemm1_weights=w["gemm1_weights"], gemm1_weights_scale=w["gemm1_weights_scale"],
            gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=w["gemm2_weights"], gemm2_weights_scale=w["gemm2_weights_scale"], gemm2_bias=None,
            output1_scale_scalar=w["output1_scale_scalar"],
            output1_scale_gate_scalar=w["output1_scale_gate_scalar"],
            output2_scale_scalar=w["output2_scale_scalar"], per_token_scale=None,
            num_experts=s.experts, top_k=s.top_k, n_group=None, topk_group=None,
            intermediate_size=s.storage_intermediate, local_expert_offset=0, local_num_experts=s.experts,
            routed_scaling_factor=None, routing_method_type=RoutingMethodType.RenormalizeNaive,
            do_finalize=do_finalize, activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=self.rows, enable_pdl=True)

    def shared(self):
        from cutlass.cute.runtime import from_dlpack
        views = (self.wgu.unsqueeze(0), self.x.unsqueeze(0).transpose(-2, -1),
                 self.gu.unsqueeze(0).transpose(-2, -1))
        tensors = [from_dlpack(v.detach(), assumed_align=32, enable_tvm_ffi=True)
                   .mark_layout_dynamic(leading_dim=ld) for v, ld in zip(views, (2, 1, 1))]
        self.gu_native(*tensors, jit.driver().CUstream(torch.cuda.current_stream().cuda_stream))
        self.silu_native.silu_native(self.gu, self.act)
        # Source-proven native fallback for (M,2560,160), with BF16 inputs/output.
        return torch.nn.functional.linear(self.act, self.wd)

    def merge(self, name, shared, out):
        pointers = [jit.pointer(t) for t in (self.x, self.gate_weight, shared, out)]
        jit.launch(self.merge_kernels[name], (self.rows, 1, 1),
                   pointers + [ctypes.c_void_p(0), ctypes.c_void_p(0)],
                   torch.cuda.current_stream().cuda_stream)

    def branches(self, *, fused):
        """Native ordinary order: route, fork routed/parallel shared, true join.

        Parent finalize runs on its routed stream; candidate finalize moves to
        the post-join merge. Its lost overlap is charged rather than hidden.
        """
        logits, q, sf = router.router_gemm_quant(
            self.x, self.router["weight"], self.router["global_scale"], shape=self.shape)
        current = torch.cuda.current_stream()
        self.routed_stream.wait_stream(current)
        with torch.cuda.stream(self.routed_stream):
            rows, weights = mkm.mkm_moe(
                self.maps, logits, q, sf,
                self.layer["gemm1_weights_scale"].view(torch.uint8),
                self.layer["gemm2_weights_scale"].view(torch.uint8),
                self.layer["output1_scale_gate_scalar"], self.layer["output1_scale_scalar"],
                self.layer["output2_scale_scalar"], shape=self.shape)
            routed = None if fused else finalize.run(rows, weights, shape=self.shape)
        shared = self.shared()
        current.wait_stream(self.routed_stream)
        rows.record_stream(current)
        weights.record_stream(current)
        if routed is not None:
            routed.record_stream(current)
        return rows, weights, shared, routed

    def parent(self):
        rows, weights, shared, routed = self.branches(fused=False)
        self.merge("native_merge", shared, routed)
        return routed

    def publish(self, site):
        rows, weights, shared, _ = self.branches(fused=True)
        self.merge("gate_only", shared, self.gate)
        jit.launch(self.push_kernel, (5, self.rows, 1),
                   [jit.pointer(t) for t in (rows, weights, shared, self.gate, self.local)] + [site],
                   torch.cuda.current_stream().cuda_stream)
        return self.local
