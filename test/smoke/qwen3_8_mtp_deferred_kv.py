# SPDX-License-Identifier: Apache-2.0
"""C03 synthetic original-source comparison; external observer/driver authorize."""

import hashlib
import importlib.util
import json
import os
import traceback
from dataclasses import asdict, fields
from pathlib import Path

OPERATION = "MTP_DEFERRED_KV_PREPARE"
REFERENCE_SHA256 = "545e4cc937feaa1e4f8d1e387dfd74a44e1d24e4e6b554afb08d30a6e161d35d"
LOOKUP_REFERENCE_SHA256 = (
    "c38f5739289971158fc6625622cedf31986f1fc1b6da8e2dfb752714381266be"
)


def plan():
    return [
        dict(
            operation=8,
            rows=1,
            kv_heads=1,
            head_dim=256,
            scalar_type=1,
            index_columns=2055,
            packed_columns=2112,
            tail_width=4,
            index_pool_rows=8,
            cache_slots=64,
            request_pool_rows=8,
            map_columns=4096,
            map_row_stride=4128,
            new_k_row_stride=256,
            new_v_row_stride=256,
            metadata_dtype_mask=mask,
        )
        for mask in range(64)
    ]


def make_io(s, case_index):
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import MtpDeferredKvIO

    device = torch.device("cuda", torch.cuda.current_device())

    def integer(bit):
        return torch.int64 if s.metadata_dtype_mask & (1 << bit) else torch.int32

    def scalar(value, bit):
        return torch.full((1,), value, dtype=integer(bit), device=device)

    kc = (
        torch.randn((s.cache_slots, 1, 256), dtype=torch.bfloat16, device=device) * 0.05
    )
    vc = torch.randn(kc.shape, dtype=kc.dtype, device=device) * 0.05
    projected = torch.randn((1, 3584), dtype=torch.bfloat16, device=device) * 0.05
    nk = (torch.randn((1, 256), dtype=torch.bfloat16, device=device) * 0.05).reshape(
        1, 1, 256
    )
    nv = projected[:, 3328:].reshape(1, 1, 256)
    mapping = torch.empty_strided(
        (s.request_pool_rows, s.map_columns),
        (s.map_row_stride, 1),
        dtype=integer(3),
        device=device,
    )
    mapping.copy_(
        torch.arange(s.map_columns, dtype=integer(3), device=device)
        % (s.cache_slots - 1)
        + 1
    )
    indices = (
        torch.arange(2055, dtype=torch.int32, device=device).repeat(
            s.index_pool_rows, 1
        )
        % 128
    )
    indices[:, ::7] = -1
    captured = torch.full((s.index_pool_rows,), 100, dtype=torch.int32, device=device)
    return MtpDeferredKvIO(
        kc,
        vc,
        nk,
        nv,
        scalar(0 if case_index % 4 == 0 else 5, 0),
        mapping,
        scalar(2, 1),
        scalar(102, 2),
        indices,
        scalar(case_index % 3, 4),
        captured,
        scalar(100 + case_index % 4, 5),
        torch.full((1,), -33, dtype=torch.int32, device=device),
        torch.full((2112, 1, 256), -55, dtype=torch.bfloat16, device=device),
        torch.full((2112, 1, 256), -55, dtype=torch.bfloat16, device=device),
    )


def clone_io(io):
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import MtpDeferredKvIO

    tensors = []
    for f in fields(io):
        t = getattr(io, f.name)
        other = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        other.copy_(t)
        tensors.append(other)
    return MtpDeferredKvIO(*tensors)


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


def load_reference(path, lookup_reference=None):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError(
            "deferred MTP KV reference differs from frozen original source"
        )
    dependency = (
        Path(lookup_reference)
        if lookup_reference is not None
        else path.with_name("original_mtp_lookup.py")
    )
    dependency = dependency.resolve(strict=True)
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != LOOKUP_REFERENCE_SHA256:
        raise ValueError(
            "deferred MTP KV lookup dependency differs from frozen original source"
        )
    dep_spec = importlib.util.spec_from_file_location(
        "original_mtp_lookup_dependency", dependency
    )
    dep = importlib.util.module_from_spec(dep_spec)
    import sys

    sys.modules[dep_spec.name] = dep
    dep_spec.loader.exec_module(dep)
    spec = importlib.util.spec_from_file_location(
        "original_mtp_deferred_kv_source_reference", path
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    # Resolve ONLY this original dependency in the reference's private globals.
    # Never create or replace a production sglang.* module alias.
    import builtins

    def reference_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.layers.attention.qsa.mtp_lookup" and level == 0:
            return dep
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=reference_import)
    spec.loader.exec_module(module)
    return module


def reference_call(module, signature, io):
    import torch

    # The original wrapper consumes only tail_offsets.numel(); the independent
    # original dataclass remains a reference object, never a provider operand.
    indices = module.DeferredMtpSparseIndices(
        io.indices_pool,
        io.captured_lengths,
        io.lookup_requests,
        io.lookup_positions,
        torch.arange(signature.tail_width, device=io.indices_pool.device),
    )
    module.fused_kv_prepare(
        io.k_cache,
        io.v_cache,
        io.new_k,
        io.new_v,
        io.locations,
        io.req_to_token,
        io.row_requests,
        indices,
        io.sequence_lengths,
        io.counts_out,
        io.packed_k_out,
        io.packed_v_out,
        1,
        compress_ratio=1,
        query_positions=None,
        chain_positions=True,
        visible_page_size=64,
    )


def run(out, reference=None):
    from types import SimpleNamespace

    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        MtpDeferredKvSignature,
        NumericOperation,
        numeric_registry,
    )
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import bind_numeric_operator

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    path = Path(out)
    if not path.is_absolute():
        raise ValueError("deferred MTP KV receipt requires absolute ownOUT")
    receipt = dict(
        operation=OPERATION,
        status="FAIL",
        records=[],
        scope="synthetic deferred KV ABI; no model/source-policy or external authorization claim",
    )
    try:
        if os.environ.get("FN_SMOKE_ONLY_OPERATION") not in (None, OPERATION):
            raise ValueError(
                "deferred MTP KV helper executes only its target operation"
            )
        if os.environ.get("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION") != "1":
            raise ValueError("numeric master must be explicitly enabled")
        signatures = tuple(
            MtpDeferredKvSignature(
                **{**case, "operation": NumericOperation.MTP_DEFERRED_KV_PREPARE}
            )
            for case in plan()
        )
        built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=signatures))
        module = load_reference(reference or os.environ["FN_MTP_DEFERRED_KV_REFERENCE"])
        for i, s in enumerate(signatures):
            record = dict(signature=asdict(s), phase="lookup", status="FAIL")
            receipt["records"].append(record)
            offer = type(numeric_registry).get(numeric_registry, s)
            if offer is None or offer not in built.offers:
                raise RuntimeError("deferred MTP KV exact offer missing before writes")
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
                raise AssertionError("deferred MTP KV A/B/C alias")
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
                finite_scope="all floating fields finite on both sides",
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
