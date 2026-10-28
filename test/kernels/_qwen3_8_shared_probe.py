# SPDX-License-Identifier: Apache-2.0
"""Load-only BF16 shared gate/up numeric probe; prepared real inputs and weights."""

import json
import os
from pathlib import Path
import statistics

import sglang
import torch
from cuda.bindings import nvrtc
from sgl_nvidia_mega.qwen3_8.moe.config import Shape
from sgl_nvidia_mega.qwen3_8.sm103.moe import router, shared

assert Path(sglang.__file__).is_relative_to(Path(os.environ["FN_SGLANG"])), (
    sglang.__file__
)
record = torch.load(os.environ["FN_MOE_FIXTURE"], map_location="cpu", weights_only=True)
weights = torch.load(
    os.environ["FN_SHARED_FIXTURE"], map_location="cpu", weights_only=True
)["shared"]
wgu = weights["wgu"].cuda()
rw = {k: v.cuda() for k, v in record["router"].items()}


def no_compile(*args, **kwargs):
    raise RuntimeError("GPU_NVRTC_FORBIDDEN")


nvrtc.nvrtcCreateProgram = no_compile
shape = Shape(2560, 512, 10, 640, 4, 4)
router.build(torch.cuda.current_device(), shape)
shared.build(torch.cuda.current_device(), shape)
results = []
for m in (1, 4):
    samples = []
    for case in (c for c in record["cases"] if c["meta"]["rows"] == m):
        x = case["hidden"].cuda()
        produced, gu = shared.router_shared(
            x, rw["weight"], rw["global_scale"], wgu, shape=shape
        )
        standard = router.router_gemm_quant(
            x, rw["weight"], rw["global_scale"], shape=shape
        )
        assert all(torch.equal(x, y) for x, y in zip(produced, standard))
        stock = torch.mm(x, wgu.T)
        exact = x.double() @ wgu.double().T
        errors = {}
        for name, value in [("candidate", gu), ("stock", stock)]:
            delta = value.double() - exact
            errors[name] = dict(
                rel_l2=float(delta.norm() / exact.norm()),
                max_abs=float(delta.abs().max()),
            )
        samples.append(errors)
    assert len(samples) == 3
    cand_rel = statistics.mean(c["candidate"]["rel_l2"] for c in samples)
    stock_rel = statistics.mean(c["stock"]["rel_l2"] for c in samples)
    cand_max = max(c["candidate"]["max_abs"] for c in samples)
    stock_max = max(c["stock"]["max_abs"] for c in samples)
    passed = cand_rel <= 1.05 * stock_rel and cand_max <= 2 * stock_max
    results.append(
        dict(
            m=m,
            passed=passed,
            rel_l2_ratio=cand_rel / stock_rel,
            max_abs_ratio=cand_max / stock_max,
            samples=samples,
        )
    )
report = dict(
    passed=all(r["passed"] for r in results),
    results=results,
    scope="shared gate/up projection only; original BF16 operands and native BF16 GEMM",
)
(Path(os.environ["Q38CC_ARM_OUTPUT"]) / "shared-numeric.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
assert report["passed"], report
