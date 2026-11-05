# SPDX-License-Identifier: Apache-2.0
"""C02 synthetic original-source comparison; external observer/driver authorize."""

import hashlib
import importlib.util
import json
import os
import traceback
from dataclasses import asdict, fields
from pathlib import Path

OPERATION = "MTP_CAPTURE_SEED"
REFERENCE_SHA256 = "67745a656691c036fafa7b1375b93e9d06bad4ab85b27637118aa4a1295d05a5"
PDL_REFERENCE_SHA256 = (
    "a6acac31b9dfe0302284469bf01d4338381fbfa8e824201b77d337eb21bec6a4"
)


def plan():
    return [
        dict(
            operation=7,
            rows=4,
            columns=2051,
            topk_row_stride=2080,
            seed_rows=8,
            seed_columns=2055,
            seed_row_stride=2080,
            front_tokens=0,
            trash_row=7,
            extend_scalar_type=4 if mask & 1 else 5,
            accept_scalar_type=4 if mask & 2 else 5,
            request_scalar_type=4 if mask & 4 else 5,
            length_scalar_type=4 if mask & 8 else 5,
            pdl=pdl,
        )
        for mask in range(16)
        for pdl in (0, 1)
    ]


def make_io(s, case_index):
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import MtpCaptureSeedIO

    device = torch.device("cuda", torch.cuda.current_device())

    def scalar(value, dtype):
        t = torch.empty_strided(
            (1,), (3,), dtype=torch.int64 if dtype == 4 else torch.int32, device=device
        )
        t.fill_(value)
        return t

    topk = torch.empty_strided(
        (4, 2051), (s.topk_row_stride, 1), dtype=torch.int32, device=device
    )
    topk.copy_(torch.arange(4 * 2051, dtype=torch.int32, device=device).view(4, 2051))
    seed = torch.empty_strided(
        (s.seed_rows, 2055), (s.seed_row_stride, 1), dtype=torch.int32, device=device
    )
    seed.fill_(-777)
    extend, accept, request = (
        (4, 1, 1),
        (0, 3, 2),
        (-1, 1, 2),
        (4, -2, 1),
        (4, 99, 1),
        (4, 3, 0),
        (4, 2, -1),
        (4, 4, s.trash_row - 1),
    )[case_index % 8]
    lengths = (
        torch.arange(
            6,
            dtype=torch.int64 if s.length_scalar_type == 4 else torch.int32,
            device=device,
        )
        + 100
    )
    captured = torch.full((s.seed_rows,), -999, dtype=torch.int32, device=device)
    return MtpCaptureSeedIO(
        topk,
        scalar(extend, s.extend_scalar_type),
        scalar(accept, s.accept_scalar_type),
        scalar(request, s.request_scalar_type),
        lengths,
        seed,
        captured,
    )


def clone_io(io):
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import MtpCaptureSeedIO

    tensors = []
    for f in fields(io):
        t = getattr(io, f.name)
        other = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        other.copy_(t)
        tensors.append(other)
    return MtpCaptureSeedIO(*tensors)


def compare_exact(got, expected):
    import torch

    for f in fields(got):
        a, b = getattr(got, f.name), getattr(expected, f.name)
        if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride():
            raise AssertionError("shape/dtype/stride mismatch: " + f.name)
        if a.is_floating_point() and (
            not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item()
        ):
            raise AssertionError("nonfinite source comparison: " + f.name)
        if not torch.equal(
            a.clone(memory_format=torch.contiguous_format).view(torch.uint8),
            b.clone(memory_format=torch.contiguous_format).view(torch.uint8),
        ):
            raise AssertionError("nonzero byte difference: " + f.name)


def load_reference(path, pdl_reference=None):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("MTP capture reference differs from frozen original source")
    dependency = (
        Path(pdl_reference)
        if pdl_reference is not None
        else path.with_name("original_pdl_small.py")
    )
    dependency = dependency.resolve(strict=True)
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != PDL_REFERENCE_SHA256:
        raise ValueError(
            "MTP capture PDL dependency differs from frozen original source"
        )
    dep_spec = importlib.util.spec_from_file_location(
        "original_mtp_capture_pdl_dependency", dependency
    )
    dep = importlib.util.module_from_spec(dep_spec)
    dep_spec.loader.exec_module(dep)
    spec = importlib.util.spec_from_file_location(
        "original_mtp_capture_source_reference", path
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    # Resolve ONLY this original dependency in the reference's private globals.
    # Never create or replace a production sglang.* module alias.
    import builtins

    def reference_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.kernels.ops.pdl_small" and level == 0:
            return dep
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=reference_import)
    spec.loader.exec_module(module)
    return module


def reference_call(module, signature, io):
    previous = module._PDL_SMALL
    module._PDL_SMALL = bool(signature.pdl)
    try:
        if not module.try_capture_metadata(
            *(getattr(io, f.name) for f in fields(io)),
            signature.front_tokens,
            signature.trash_row,
        ):
            raise RuntimeError("original source refused the declared MTP capture view")
    finally:
        module._PDL_SMALL = previous


def run(out, reference=None):
    from types import SimpleNamespace

    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        MtpCaptureSeedSignature,
        NumericOperation,
        numeric_registry,
    )
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import bind_numeric_operator

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    path = Path(out)
    if not path.is_absolute():
        raise ValueError("MTP capture receipt requires absolute ownOUT")
    receipt = dict(
        operation=OPERATION,
        status="FAIL",
        records=[],
        scope="synthetic integer seed ABI; no model/source-policy or external authorization claim",
    )
    try:
        if os.environ.get("FN_SMOKE_ONLY_OPERATION") not in (None, OPERATION):
            raise ValueError("MTP capture helper executes only its target operation")
        if os.environ.get("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION") != "1":
            raise ValueError("numeric master must be explicitly enabled")
        signatures = tuple(
            MtpCaptureSeedSignature(
                **{**case, "operation": NumericOperation.MTP_CAPTURE_SEED}
            )
            for case in plan()
        )
        built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=signatures))
        module = load_reference(
            reference or os.environ["FN_MTP_CAPTURE_SEED_REFERENCE"]
        )
        for i, s in enumerate(signatures):
            record = dict(signature=asdict(s), phase="lookup", status="FAIL")
            receipt["records"].append(record)
            offer = type(numeric_registry).get(numeric_registry, s)
            if offer is None or offer not in built.offers:
                raise RuntimeError("MTP capture exact offer missing before writes")
            selected = bind_numeric_operator(
                enabled=True,
                signature=s,
                prepared=(s,),
                offers=(offer,),
                stock=lambda io: None,
            )
            base = make_io(s, i)
            ios = [base, clone_io(base), clone_io(base)]
            originals = [clone_io(io) for io in ios]
            record["pointers"] = [
                {f.name: getattr(io, f.name).data_ptr() for f in fields(io)}
                for io in ios
            ]
            if any(
                len({p[f.name] for p in record["pointers"]}) != 3 for f in fields(base)
            ):
                raise AssertionError("MTP capture A/B/C alias")
            record["phase"] = "eager_A"
            selected(ios[0])
            record["phase"] = "eager_B"
            selected(ios[1])
            torch.cuda.synchronize()
            record["phase"] = "capture_C"
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                selected(ios[2])
            graph.replay()
            torch.cuda.synchronize()
            record["phase"] = "source_comparison"
            for actual, original in zip(ios, originals):
                reference_call(module, s, original)
                torch.cuda.synchronize()
                compare_exact(actual, original)
            record.update(
                status="PASS",
                phase="complete",
                source_byte_differences=0,
                finite_scope="all seven tensors are integers",
                eager_calls=2,
                capture_calls=1,
                replay_calls=1,
            )
        receipt["status"] = "PASS"
        return receipt
    except BaseException:
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--reference")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan(), indent=2))
    else:
        run(args.out, args.reference)
