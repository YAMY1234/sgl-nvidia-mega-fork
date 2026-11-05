"""TP1 pool variants: synthetic caller-buffer eager/capture/replay diagnostics.

Harness observer86/driver7b and authorization remain external and unchanged.
This fixture does not supply a model producer, quality or performance proof.
"""

import argparse
import json
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

OPERATION = "DIRECT_POOL_ATTENTION"
LENGTHS = (499, 512, 513, 4096, 4097, 8192, 8193)


@dataclass(frozen=True)
class Case:
    rows: int
    variant: int
    width: int
    length: int


def plan(operation=OPERATION):
    if operation != OPERATION:
        raise ValueError("one fresh TP1 pool child required")
    return tuple(
        Case(m, v, w, n)
        for v, w, ms in ((1, 4, (4, 8, 32, 64)), (2, 1, (1, 8, 32, 64)))
        for m in ms
        for n in LENGTHS
    )


def signature(case):
    from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        NumericSignature,
    )
    from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

    return NumericSignature(
        O.DIRECT_POOL_ATTENTION,
        case.rows,
        24,
        2,
        256,
        512,
        4,
        D.BF16,
        variant=case.variant,
    )


def selected(sig):
    from sglang.kernels.plugin_kernels.nvidia.numeric import numeric_registry
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import bind_numeric_operator

    offer = type(numeric_registry).get(numeric_registry, sig)
    if offer is None:
        raise RuntimeError("HOLD: exact TP1 pool variant not registered")
    return bind_numeric_operator(
        enabled=True,
        signature=sig,
        prepared=(sig,),
        offers=(offer,),
        stock=lambda io: None,
    )


def inputs(case, device):
    import torch
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        DirectPoolAttentionIO,
        DirectPoolSelection,
        KvTensors,
    )

    m, w, n = case.rows, case.width, case.length
    requests = m // w
    context = max(LENGTHS)
    # Common immutable history prefix, distinct new slots per row/request.
    slots = context + 1 + m
    row_requests = torch.arange(
        requests, device=device, dtype=torch.int32
    ).repeat_interleave(w)
    locations = context + 1 + torch.arange(m, device=device, dtype=torch.int64)
    slotmap = torch.arange(1, context + 1, device=device, dtype=torch.int64).repeat(
        requests, 1
    )
    slotmap[:, n - w : n] = locations.view(requests, w)
    lengths = torch.full((m,), n, dtype=torch.int32, device=device)
    positions = (
        n - w + torch.arange(w, device=device, dtype=torch.int64).repeat(requests)
    )
    selected_indices = torch.arange(512, device=device, dtype=torch.int32).repeat(m, 1)
    selected_indices = torch.where(
        selected_indices < (n - w) // 4, selected_indices, -1
    )
    # Source packed projection row stride; no adapter makes a contiguous copy.
    packed = torch.zeros((m, 2304), dtype=torch.bfloat16, device=device)
    nk = packed[:, :512].view(m, 2, 256)
    nv = packed[:, 512:1024].view(m, 2, 256)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rh = m * 2
    bn, waves = (32, 3) if rh >= 32 else (64, 1)
    div = lambda a, b: (a + b - 1) // b
    ns = max(1, round(waves * sms / rh))
    chunk = div(div(2051, ns), bn) * bn
    ns = div(2051, chunk)
    return DirectPoolAttentionIO(
        torch.ones((m, 24, 256), dtype=torch.bfloat16, device=device),
        None,
        nk,
        nv,
        locations,
        KvTensors(
            torch.zeros((slots, 2, 256), dtype=torch.bfloat16, device=device),
            torch.zeros((slots, 2, 256), dtype=torch.bfloat16, device=device),
            None,
            None,
            1,
        ),
        DirectPoolSelection(
            slotmap, row_requests, selected_indices, lengths, positions
        ),
        0.0625,
        w,
        True,
        torch.empty((m, 24, 256), dtype=torch.bfloat16, device=device),
        torch.empty(m * 2 * ns * 16 * 258, dtype=torch.float32, device=device),
    )


def addresses(io):
    tensors = {
        "q": io.q,
        "new_k": io.new_k,
        "new_v": io.new_v,
        "K": io.kv.k,
        "V": io.kv.v,
        "locations": io.out_cache_loc,
        "output": io.output,
        "scratch": io.scratch_out,
        "map": io.selection.full_slot_map,
        "requests": io.selection.row_requests,
        "indices": io.selection.selected_indices,
        "lengths": io.selection.sequence_lengths,
        "positions": io.selection.query_positions,
    }
    return {
        k: dict(
            ptr=v.data_ptr(),
            shape=list(v.shape),
            stride=list(v.stride()),
            dtype=str(v.dtype),
        )
        for k, v in tensors.items()
    }


def assert_distinct(frames):
    if len(frames) != 3 or any(set(f) != set(frames[0]) for f in frames):
        raise AssertionError("three exact caller-buffer inventories required")
    for name in frames[0]:
        if len({f[name]["ptr"] for f in frames}) != 3:
            raise AssertionError("A/B/C aliases: " + name)
        assert all(
            all(f[name][v] == frames[0][name][v] for v in ("shape", "stride", "dtype"))
            for f in frames[1:]
        )


def check_env(out, operation=OPERATION):
    plan(operation)
    out = Path(out)
    assert out.is_absolute(), "exact absolute own OUT required"
    out = out.resolve()
    assert out.is_relative_to(Path(os.environ["M7_CANONICAL_OUT"]).resolve())
    assert os.environ.get("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION") == "1"
    assert os.environ.get("SGLANG_QSA_POOL_DECODE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )
    assert os.environ.get("FN_SMOKE_ONLY_OPERATION") in (None, operation)
    slots = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_SLOTS")
    assert slots is None or "qsa_core" in slots.split(",")
    return out


def run(out, *, operation=OPERATION):
    import torch

    from sgl_nvidia_mega.qwen3_8.providers import register_signatures

    out = check_env(out, operation)
    out.parent.mkdir(parents=True, exist_ok=True)
    records, current, status = [], {}, "HOLD"

    def archive():
        out.write_text(
            json.dumps(
                dict(
                    status=status,
                    operation_filter=operation,
                    expected_records=len(plan()),
                    records=records,
                    current=current,
                    successful_offer_fixture_calls={
                        OPERATION: 2 * sum(r.get("status") == "PASS" for r in records)
                    },
                    source_scope="synthetic shared-history pool + distinct new slots; source-compatible TP1 default policy; not model metadata",
                    executed_variants=[1, 2],
                    old_TP4_variant0="NOT_RUN by this helper",
                    model_engagement="NOT_PROVEN",
                    numerical_comparison="NOT_RUN; finite sanity only",
                    race="NOT_QUALIFIED",
                    performance="NOT_RUN",
                    original_observer_and_authorization="harness observer86/driver7b unchanged",
                ),
                indent=2,
            )
            + "\n"
        )

    archive()
    try:
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (
            10,
            7,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        cases = plan()
        register_signatures(tuple(dict.fromkeys(signature(c) for c in cases)))
        for case in cases:
            current = dict(case=asdict(case), phase="preflight")
            archive()
            sig = signature(case)
            call = selected(sig)
            frames = tuple(inputs(case, device) for _ in range(3))
            ptrs = tuple(addresses(frame) for frame in frames)
            assert_distinct(ptrs)
            current["pointers_ABC"] = ptrs
            for phase, frame in zip(("eager_A", "eager_B"), frames[:2]):
                current["phase"] = phase
                archive()
                assert call(frame) is None
                torch.cuda.synchronize()
            current["phase"] = "capture_C"
            archive()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                assert call(frames[2]) is None
            current["phase"] = "replay_C"
            archive()
            graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            assert torch.isfinite(frames[2].output).all().item()
            records.append(
                dict(
                    status="PASS",
                    case=asdict(case),
                    target_signature=asdict(sig),
                    canonical_facade=True,
                    eager_AB=True,
                    capture_replay_C=True,
                    pointers_ABC=ptrs,
                    executed_operations=[OPERATION],
                    gate_applied=False,
                    store_new_kv=True,
                )
            )
            current = {}
            archive()
        assert len(records) == len(cases)
        status = "PASS_TP1_POOL_BUILD_LAUNCH_SCOPE"
    except Exception:
        records.append(dict(status="FAILED", error=traceback.format_exc()))
        raise
    finally:
        archive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--operation", choices=(OPERATION,), default=OPERATION)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    check_env(args.out, args.operation)
    if args.plan_only:
        print(json.dumps([asdict(c) for c in plan(args.operation)], indent=2))
    else:
        run(args.out, operation=args.operation)
