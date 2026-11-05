# SPDX-License-Identifier: Apache-2.0
"""Synthetic source comparison; external observer/driver retain authorization.

No native convolution or model source-policy execution is claimed. The frozen
original reference file is supplied by the harness, never fetched at runtime.
"""

import hashlib
import importlib.util
import json
import os
import traceback
from dataclasses import asdict, fields
from pathlib import Path

REFERENCE_SHA256 = "706b4284f5cc41f2aac15c6a3d2a96dad03cb8b2346b1293494e621de0f1664f"
OPERATION = "GDN_NATIVE_PDL"


def plan(dot_precision=0):
    """Twelve exact representatives, not all 192 dtype/launch combinations."""
    cases = []
    for h, hv in ((4, 12), (16, 48)):
        for storage in (0, 1, 2):
            for fp32 in (False, True):
                cases.append(
                    dict(
                        operation=33,
                        rows=(16 if fp32 else 32) if h == 16 else (4 if fp32 else 8),
                        key_heads=h,
                        value_heads=hv,
                        key_dim=128,
                        value_dim=128,
                        tokens_per_request=4,
                        scalar_type=1,
                        gate_scalar_type=1,
                        # Residual/B also exercises the BF16 checkpoint prefetch
                        # branch; other B cases cover FP32 checkpoint storage.
                        state_scalar_type=3 if fp32 and storage != 1 else 1,
                        dt_bias_scalar_type=1 if fp32 else 3,
                        cache_len=16,
                        storage_mode=storage,
                        wait_for_gate_producer=int(fp32),
                        state_prefetch=int(fp32),
                        dot_precision=dot_precision,
                    )
                )
    return cases


def make_io(s):
    """Framework-side synthetic buffers with actual packed projection views."""
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import NativeGdnPdlIO

    m, h, hv, bsz = s.rows, s.key_heads, s.value_heads, s.rows // 4
    device = torch.device("cuda", torch.cuda.current_device())

    def zeros(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=device)

    # Q/K/V share a genuine packed allocation and retain its padded row stride.
    packed = (
        torch.randn((m, (2 * h + hv) * 128), dtype=torch.bfloat16, device=device) * 0.05
    )
    q = packed[:, : h * 128].view(m, h, 128)
    k = packed[:, h * 128 : 2 * h * 128].view(m, h, 128)
    v = packed[:, 2 * h * 128 :].view(m, hv, 128)
    gates = zeros((m, 2 * hv))
    a, b = gates[:, :hv], gates[:, hv:]
    state_dtype = torch.float32 if s.state_scalar_type == 3 else torch.bfloat16
    dt_dtype = torch.float32 if s.dt_bias_scalar_type == 3 else torch.bfloat16
    slots = bsz + 1
    d = torch.randn((slots, hv, 16, 128), dtype=torch.bfloat16, device=device) * 0.02
    kc = torch.randn((slots, h, 16, 128), dtype=torch.bfloat16, device=device) * 0.02
    g = zeros((slots, hv, 16), torch.float32)
    g.copy_(-torch.arange(16, dtype=torch.float32, device=device) * 0.02)

    def vector(values, dtype, stride):
        t = zeros((len(values) * stride,), dtype)[::stride]
        t.copy_(torch.tensor(values, dtype=dtype, device=device))
        return t

    integer = torch.int64 if h == 16 else torch.int32
    return NativeGdnPdlIO(
        q,
        k,
        v,
        a,
        b,
        torch.full((hv,), -2.0, dtype=torch.float32, device=device),
        zeros((hv,), dt_dtype),
        zeros((m, hv, 128)),
        torch.randn((bsz + 3, hv, 128, 128), dtype=state_dtype, device=device) * 0.01,
        d,
        kc,
        g,
        zeros(d.shape) if s.storage_mode else None,
        zeros(kc.shape) if s.storage_mode else None,
        zeros(g.shape, torch.float32) if s.storage_mode == 2 else None,
        vector(list(range(0, m + 1, 4)), integer, 2),
        vector(list(range(1, bsz + 1)), integer, 3),
        vector(list(range(bsz)), integer, 4),
        torch.full((slots,), 4, dtype=torch.int32, device=device),
        zeros((slots,), torch.int32),
        zeros((slots,), torch.int8),
    )


def clone_io(io):
    """Preserve source strides without packing any view for the provider."""
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import NativeGdnPdlIO

    values = []
    for f in fields(io):
        t = getattr(io, f.name)
        if t is None:
            values.append(None)
        else:
            copied = torch.empty_strided(
                t.shape, t.stride(), dtype=t.dtype, device=t.device
            )
            copied.copy_(t)
            values.append(copied)
    return NativeGdnPdlIO(*values)


def load_reference(path):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("native PDL reference differs from frozen original source")
    spec = importlib.util.spec_from_file_location(
        "native_pdl_original_source_reference", path
    )
    module = importlib.util.module_from_spec(spec)
    # Triton resolves its defining module when compiling the original function.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_call(module, signature, io):
    """Use the unchanged original wrapper, with explicit source policy inputs."""
    from sglang.kernels.ops.attention.fla import utils

    if bool(utils.is_tf32_supported) != (signature.dot_precision == 0):
        # TF32 capability is resolved by the real framework, not overridden.
        raise ValueError("reference precision must match actual original device policy")
    previous = os.environ.get("Q38_GDN_STATE_PREFETCH")
    os.environ["Q38_GDN_STATE_PREFETCH"] = str(signature.state_prefetch)
    try:
        return module.verify(
            *(getattr(io, f.name) for f in fields(io)),
            max_cache_len=16,
            max_spec_len=4,
            scale=128**-0.5,
            null_block_id=-1,
            block_v=64,
            num_warps=1,
            num_stages=2,
            nk=2,
            bs_min=4,
            pdl=True,
            wait_for_gate_producer=bool(signature.wait_for_gate_producer),
        )
    finally:
        if previous is None:
            os.environ.pop("Q38_GDN_STATE_PREFETCH", None)
        else:
            os.environ["Q38_GDN_STATE_PREFETCH"] = previous


def compare_exact(got, expected):
    """Compare all outputs/rings and borrowed inputs; reject matching nonfinite."""
    import torch

    for f in fields(got):
        a, b = getattr(got, f.name), getattr(expected, f.name)
        if a is None or b is None:
            if a is not b:
                raise AssertionError("optional tensor mismatch: " + f.name)
            continue
        if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride():
            raise AssertionError("shape/dtype/stride mismatch: " + f.name)
        if a.is_floating_point() and (
            not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item()
        ):
            raise AssertionError("nonfinite source comparison: " + f.name)
        # A singleton can be contiguous with a nonunit final stride. Force only
        # diagnostic copies dense; preserve the borrowed ABI views above.
        if not torch.equal(
            a.clone(memory_format=torch.contiguous_format).view(torch.uint8),
            b.clone(memory_format=torch.contiguous_format).view(torch.uint8),
        ):
            raise AssertionError("nonzero byte difference: " + f.name)


def run(out, reference=None):
    """Only op33 runs; observer86/driver7b/validator remain external unchanged."""
    from types import SimpleNamespace

    import torch
    from sglang.kernels.ops.attention.fla import utils
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        NativeGdnPdlSignature,
        NumericOperation,
        numeric_registry,
    )
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import bind_numeric_operator

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    path = Path(out)
    if not path.is_absolute():
        raise ValueError("native PDL receipt requires absolute ownOUT")
    receipt = dict(
        operation=OPERATION,
        status="FAIL",
        records=[],
        scope="synthetic source comparison; no nativeconv/model route; external authorization required",
    )
    try:
        if os.environ.get("FN_SMOKE_ONLY_OPERATION") not in (None, OPERATION):
            raise ValueError("native PDL helper executes only its target operation")
        if os.environ.get("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION") != "1":
            raise ValueError("numeric master must be explicitly enabled")
        # Precision follows actual original device policy. Other signature modes
        # stay explicit; no claim that this synthetic domain is source-engaged.
        cases = plan(dot_precision=0 if utils.is_tf32_supported else 2)
        signatures = tuple(
            NativeGdnPdlSignature(
                **{**case, "operation": NumericOperation.GDN_NATIVE_PDL}
            )
            for case in cases
        )
        # Registration precedes tensor/reference creation, as in the framework.
        built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=signatures))
        module = load_reference(reference or os.environ["FN_NATIVE_GDN_PDL_REFERENCE"])
        for s in signatures:
            record = dict(signature=asdict(s), phase="lookup", status="FAIL")
            receipt["records"].append(record)
            offer = type(numeric_registry).get(numeric_registry, s)
            if offer is None or offer not in built.offers:
                raise RuntimeError(
                    "native PDL exact offer missing before any tensor writes"
                )
            selected = bind_numeric_operator(
                enabled=True,
                signature=s,
                prepared=(s,),
                offers=(offer,),
                stock=lambda io: None,
            )
            base = make_io(s)
            ios = [base, clone_io(base), clone_io(base)]
            # Compare each phase to original source using independent cloned rings.
            references = [clone_io(io) for io in ios]
            record["pointers"] = [
                {
                    f.name: getattr(io, f.name).data_ptr()
                    for f in fields(io)
                    if getattr(io, f.name) is not None
                }
                for io in ios
            ]
            for f in fields(base):
                if (
                    getattr(base, f.name) is not None
                    and len({p[f.name] for p in record["pointers"]}) != 3
                ):
                    raise AssertionError("A/B/C alias: " + f.name)
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
            for actual, ref in zip(ios, references):
                reference_call(module, s, ref)
                torch.cuda.synchronize()
                compare_exact(actual, ref)
            record.update(
                status="PASS",
                phase="complete",
                byte_differences=0,
                finite=True,
                state_capacity=base.checkpoint_state.shape[0],
                replay_capacity=base.d_cache.shape[0],
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
        print(
            json.dumps(
                dict(operation=OPERATION, records=plan(), expected_records=12), indent=2
            )
        )
    elif args.out:
        run(args.out, args.reference)
    else:
        parser.error("--out is required unless --plan-only")
