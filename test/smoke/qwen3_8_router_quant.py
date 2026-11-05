# SPDX-License-Identifier: Apache-2.0
"""Prepared router-quant facade smoke; synthetic buffers, not a model witness.

The owning harness supplies the original external observer/restoration driver.
This helper never patches registry or facade methods and never calls offer.run.
"""

import argparse
import json
import os
import traceback
from dataclasses import asdict, dataclass, fields
from pathlib import Path

OPERATION = "ROUTER_QUANT"
SLOT = "router_quant"


@dataclass(frozen=True)
class Case:
    rows: int
    variant: int


def plan():
    return tuple(
        Case(m, v) for m, v in ((1, 0), (4, 0), (1, 1), (4, 1), (8, 2), (16, 2))
    )


def signature(case):
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        NumericOperation,
        RouterQuantSignature,
        ScalarType,
    )

    return RouterQuantSignature(
        NumericOperation.ROUTER_QUANT,
        case.rows,
        2560,
        512,
        ScalarType.BF16,
        variant=case.variant,
    )


def selected(case):
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        NumericOperation,
        RouterQuantIO,
        numeric_registry,
    )
    from sglang.srt.kernel_plugins.nvidia import numeric_binding

    target = signature(case)
    # Preserve the external diagnostic driver's instance lookup adapter; the
    # original class get (including its observer) returns the actual offer.
    offer = type(numeric_registry).get(numeric_registry, target)
    if offer is None:
        raise RuntimeError("HOLD: exact router quant offer is missing")
    if (
        numeric_binding._IO_TYPES.get(NumericOperation.ROUTER_QUANT)
        is not RouterQuantIO
    ):
        raise RuntimeError("HOLD: canonical router quant facade mapping is missing")

    def stock(io):
        raise RuntimeError("HOLD: prepared router quant unexpectedly selected stock")

    run = numeric_binding.bind_numeric_operator(
        enabled=True,
        signature=target,
        prepared=(target,),
        offers=(offer,),
        stock=stock,
    )
    if run is stock:
        raise RuntimeError("HOLD: exact router quant facade was not selected")
    return run


def inputs(case, device):
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import RouterQuantIO

    m = case.rows
    # These are synthetic finite launch inputs, not a loaded model's scale or
    # proof that its runner consumes prequantized activations.
    return RouterQuantIO(
        torch.full((m, 2560), 0.125, dtype=torch.bfloat16, device=device),
        torch.full((512, 2560), 0.015625, dtype=torch.bfloat16, device=device),
        torch.ones((1,), dtype=torch.float32, device=device),
        torch.full((m, 512), float("nan"), dtype=torch.bfloat16, device=device),
        torch.empty((m, 1280), dtype=torch.uint8, device=device),
        torch.empty((m, 160), dtype=torch.float8_e4m3fn, device=device),
    )


def addresses(io):
    return {
        field.name: dict(
            ptr=getattr(io, field.name).data_ptr(),
            shape=list(getattr(io, field.name).shape),
            stride=list(getattr(io, field.name).stride()),
            dtype=str(getattr(io, field.name).dtype),
        )
        for field in fields(io)
    }


def assert_distinct(inventories):
    if len(inventories) != 3 or any(len(x) != 6 for x in inventories):
        raise AssertionError("three complete six-tensor inventories required")
    pointers = [
        entry["ptr"] for inventory in inventories for entry in inventory.values()
    ]
    if any(p <= 0 for p in pointers) or len(set(pointers)) != 18:
        raise AssertionError("A/B/C caller buffers alias or are null")
    for name in inventories[0]:
        for other in inventories[1:]:
            assert all(
                other[name][k] == inventories[0][name][k]
                for k in ("shape", "stride", "dtype")
            )


def check_env(out):
    path = Path(out)
    root = Path(os.environ["M7_CANONICAL_OUT"])
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError("absolute own OUT/root required")
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("router quant receipt must be inside own OUT")
    if os.environ.get("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION") != "1":
        raise RuntimeError("HOLD: numeric runtime master is not enabled")
    if os.environ.get("FN_SMOKE_ONLY_OPERATION") not in (None, OPERATION):
        raise RuntimeError("wrong operation child")
    slots = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_SLOTS")
    if slots is not None and SLOT not in {s.strip() for s in slots.split(",")}:
        raise RuntimeError("HOLD: router_quant slot is disabled")
    return path


def method_identities():
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        NumericRegistry,
        numeric_registry,
    )
    from sglang.srt.kernel_plugins.nvidia import numeric_binding

    return (
        NumericRegistry.register,
        NumericRegistry.get,
        numeric_registry.__dict__.get("get"),
        numeric_binding.bind_numeric_operator,
    )


def run(out):
    out = check_env(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records, current, status, original = [], {}, "HOLD", None
    identity_preserved, cuda_build = None, None

    def archive():
        out.write_text(
            json.dumps(
                dict(
                    status=status,
                    operation_filter=OPERATION,
                    expected_records=len(plan()),
                    records=records,
                    current=current,
                    successful_offer_fixture_calls={
                        OPERATION: 2 * sum(r.get("status") == "PASS" for r in records)
                    },
                    count_scope="lower bound from complete eager A/B calls; observer captures independent",
                    helper_method_identities_preserved=identity_preserved,
                    external_observer_restoration="REQUIRED: original external observer/driver receipt; not fabricated here",
                    CPU_fixture=cuda_build is None,
                    torch_cuda_build=cuda_build,
                    source_scope="synthetic six-tensor raw ABI, no model producer or typed runner witness",
                    numerical_comparison="NOT_RUN; finite sanity only",
                    model_engagement="NOT_PROVEN",
                    quantizer_skipped_in_model="NOT_PROVEN",
                    performance="NOT_RUN",
                ),
                indent=2,
            )
            + "\n"
        )

    archive()
    try:
        current = dict(phase="imports")
        archive()
        import torch

        from sgl_nvidia_mega.qwen3_8.providers import register_signatures

        original, cuda_build = method_identities(), torch.version.cuda
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            10,
            7,
        ):
            raise RuntimeError("router quant GPU fixture requires real SM107")
        current = dict(phase="register_eager_AOT")
        archive()
        register_signatures(tuple(signature(case) for case in plan()))
        device = torch.device("cuda", torch.cuda.current_device())
        for case in plan():
            current = dict(case=asdict(case), phase="prepare_views")
            archive()
            invoke = selected(case)
            frames = tuple(inputs(case, device) for _ in range(3))
            inventories = tuple(addresses(io) for io in frames)
            assert_distinct(inventories)
            current["pointers_ABC"] = inventories
            for phase, io in zip(("eager_A", "eager_B"), frames[:2]):
                current["phase"] = phase
                archive()
                assert invoke(io) is None
                torch.cuda.synchronize()
            current["phase"] = "capture_C"
            archive()
            stream = torch.cuda.Stream()
            main = torch.cuda.current_stream()
            stream.wait_stream(main)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                assert invoke(frames[2]) is None
            main.wait_stream(stream)
            current["phase"] = "replay_C"
            archive()
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.isfinite(io.logits_out).all().item() for io in frames)
            records.append(
                dict(
                    case=asdict(case),
                    status="PASS",
                    target_signature=asdict(signature(case)),
                    executed_operations=[OPERATION],
                    eager_AB=True,
                    capture_replay_C=True,
                    canonical_facade=True,
                    six_caller_tensors=True,
                    pointers_ABC=inventories,
                    three_allocations_alive_through_replay=True,
                )
            )
            current = {}
            archive()
        if len(records) != len(plan()):
            raise RuntimeError("incomplete router quant matrix")
        status = "PASS_ROUTER_QUANT_BUILD_LAUNCH_SCOPE"
    except Exception:
        records.append(dict(status="FAILED", error=traceback.format_exc()))
        raise
    finally:
        if original is not None:
            identity_preserved = all(
                a is b for a, b in zip(original, method_identities())
            )
            if not identity_preserved:
                status = "HOLD"
        archive()
        if identity_preserved is False:
            raise RuntimeError("HOLD: helper changed registry/facade identity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    check_env(args.out)
    if args.plan_only:
        print(json.dumps([asdict(case) for case in plan()], indent=2))
    else:
        run(args.out)
