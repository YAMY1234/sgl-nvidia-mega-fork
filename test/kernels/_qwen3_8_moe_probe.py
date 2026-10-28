"""Load-only FN plugin router→MKM→ordered finalize vs native FlashInfer, real L23 inputs."""

import argparse, hashlib, json, os, statistics
from pathlib import Path
import torch
import sglang

assert Path(sglang.__file__).is_relative_to(Path(os.environ["FN_SGLANG"])), (
    sglang.__file__
)
from sgl_nvidia_mega.qwen3_8.moe.config import Shape
from sgl_nvidia_mega.qwen3_8.sm103.moe import router, mkm, finalize
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.core import ActivationType
from flashinfer.tllm_enums import SfLayout
from sglang.srt.layers.moe.utils import RoutingMethodType
from sgl_nvidia_mega.common import jit

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--sanity-only", action="store_true")
a = p.parse_args()
prepared = Path(os.environ["FN_PREPARED"])
out = Path(os.environ["Q38CC_ARM_OUTPUT"])
# Initialize this thread's actual CUDA context before any module load.
saved = torch.load(
    Path(os.environ["FN_MOE_FIXTURE"]), map_location="cpu", weights_only=True
)
layer = {k: v.cuda() for k, v in saved["layer"].items()}
rw = {k: v.cuda() for k, v in saved["router"].items()}
# Cache hit is mandatory: never call NVRTC on a GPU worker, even outside capture.
original_compile = jit.compile_source
from cuda.bindings import nvrtc


def forbidden(*args, **kwargs):
    raise RuntimeError("GPU_NVRTC_FORBIDDEN")


nvrtc.nvrtcCreateProgram = forbidden
import flashinfer.fused_moe.core as core

meta = json.loads((prepared / "native.json").read_text())
native_path = Path(meta["image_path"])
assert native_path.stat().st_size == meta["size"]
assert (
    hashlib.file_digest(native_path.open("rb"), "sha256").hexdigest() == meta["sha256"]
)
factory = core.gen_trtllm_gen_fused_moe_sm100_module


def aot_only(*args, **kwargs):
    spec = factory(*args, **kwargs)
    assert native_path in spec.get_library_paths()
    spec.build_and_load = lambda: spec.load(native_path)
    spec.get_library_paths = lambda: (native_path,)
    return spec


core.gen_trtllm_gen_fused_moe_sm100_module = aot_only
shape = Shape(2560, 512, 10, 640, 4, 4)
router.build(torch.cuda.current_device(), shape)
mkm.build(torch.cuda.current_device(), shape)
finalize.build(torch.cuda.current_device(), shape)
maps = mkm.weight_maps(layer["gemm1_weights"], layer["gemm2_weights"], shape)
report = dict(
    rank=a.rank,
    sglang=str(sglang.__file__),
    gpu_compile=False,
    scope="real old193 L23 hidden/checkpoint weights, fresh native FI comparison; isolated TP4 rank, not model smoke or TP4 service performance",
    cases=[],
)


def same(x, y):
    if x.dtype == torch.bfloat16:
        return bool(torch.equal(x.view(torch.int16), y.view(torch.int16)))
    return bool(torch.equal(x, y))


def error(x, y):
    d = x.double() - y.double()
    return dict(
        bitwise=same(x, y),
        max_abs=float(d.abs().max()),
        rel_l2=float(d.norm() / y.double().norm().clamp_min(1e-30)),
        finite=bool(x.isfinite().all()),
    )


def native(c, do_finalize):
    return trtllm_fp4_block_scale_moe(
        routing_logits=c["logits"],
        routing_bias=None,
        hidden_states=c["xq"],
        hidden_states_scale=c["sf"].view(torch.float8_e4m3fn),
        hidden_states_scale_layout=SfLayout.layout_linear,
        gemm1_weights=layer["gemm1_weights"],
        gemm1_weights_scale=layer["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer["gemm2_weights"],
        gemm2_weights_scale=layer["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=layer["output1_scale_scalar"],
        output1_scale_gate_scalar=layer["output1_scale_gate_scalar"],
        output2_scale_scalar=layer["output2_scale_scalar"],
        per_token_scale=None,
        num_experts=shape.experts,
        top_k=shape.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=shape.storage_intermediate,
        local_expert_offset=0,
        local_num_experts=shape.experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
        do_finalize=do_finalize,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=c["hidden"].shape[0],
        enable_pdl=True,
    )


def chain(x):
    logits, q, sf = router.router_gemm_quant(
        x, rw["weight"], rw["global_scale"], shape=shape
    )
    rows, w, ids = mkm.mkm_moe(
        maps,
        logits,
        q,
        sf,
        layer["gemm1_weights_scale"].view(torch.uint8),
        layer["gemm2_weights_scale"].view(torch.uint8),
        layer["output1_scale_gate_scalar"],
        layer["output1_scale_scalar"],
        layer["output2_scale_scalar"],
        shape=shape,
        return_ids=True,
    )
    return finalize.run(rows, w, shape=shape), rows, w, ids, logits, q, sf


for m in (1, 4):
    cases = [c for c in saved["cases"] if c["meta"]["rows"] == m]
    gpu = [
        {k: v.cuda() for k, v in c.items() if isinstance(v, torch.Tensor)}
        for c in cases
    ]
    refs = []
    for i, c in enumerate(gpu):
        nrows, nweights, permutation = native(c, False)[:3]
        expected = nrows[permutation.reshape(-1).long()[: m * 10]]
        nfinal = native(c, True)[0]
        result = chain(c["hidden"])
        torch.cuda.synchronize()
        final, rows, w, ids, logits, q, sf = result
        record = dict(
            m=m,
            sample=i,
            xq_bitwise=same(q, c["xq"]),
            sf_bitwise=same(sf, c["sf"]),
            logits=error(logits, c["logits"]),
            weights=error(w, nweights),
            expert_rows=error(rows, expected),
            final=error(final, nfinal),
            ids_match=bool(
                torch.equal(
                    ids,
                    torch.argsort(
                        c["logits"].float(), dim=-1, descending=True, stable=True
                    )[:, :10].int(),
                )
            ),
        )
        record["passed"] = (
            record["xq_bitwise"]
            and record["sf_bitwise"]
            and record["ids_match"]
            and all(record[k]["bitwise"] for k in ("weights", "expert_rows", "final"))
        )
        report["cases"].append(record)
        refs.append(final.clone())
        print("PLUGIN_NATIVE_CASE", json.dumps(record), flush=True)
    # Replay the whole producer/consumer with changed actual hidden input.
    x = gpu[0]["hidden"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = chain(x)
    checks = []
    for i in range(9):
        x.copy_(gpu[i % 3]["hidden"])
        graph.replay()
        torch.cuda.synchronize()
        checks.append(same(captured[0], refs[i % 3]))
    report.setdefault("replays", {})[str(m)] = checks
    if not a.sanity_only:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(64):
                chain(x)
        g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(7):
            b, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            b.record()
            g.replay()
            e.record()
            e.synchronize()
            times.append(b.elapsed_time(e) * 1000 / 64)
        report.setdefault("micro", {})[str(m)] = dict(
            chain_hot_us=statistics.median(times),
            samples=times,
            includes="router+MKM+ordinary ordered finalize; not shared/AR/HC",
        )
report["passed"] = all(c["passed"] for c in report["cases"]) and all(
    all(v) for v in report["replays"].values()
)
(out / ("memcheck.json" if a.sanity_only else "parity-micro.json")).write_text(
    json.dumps(report, indent=2) + "\n"
)
print("PLUGIN_NATIVE_RESULT", json.dumps(report), flush=True)
raise SystemExit(0 if report["passed"] else 2)
