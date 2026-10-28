"""Finite TP4/M4 qualification: native FI bare AR+HC versus the isolated HC hop.

No SGLang Python imports or GPU JIT. Both candidates use exactly the same inputs.
Consume-only timing excludes that generation's immediately preceding publish;
publish+consume is separately charged. Neither is a service performance claim.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
import tvm_ffi
import flashinfer.comm as comm
import flashinfer.comm.trtllm_mnnvl_ar as mnnvl
from flashinfer.comm.mnnvl import TorchDistBackend

from sgl_nvidia_mega.qwen3_8.boundary.hop import Config, Hop


p = argparse.ArgumentParser()
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--parity-only", action="store_true")
p.add_argument("--path", choices=("compare", "native", "candidate"), default="compare")
a = p.parse_args()
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.empty(1, device="cuda")  # Establish the real context before loading libraries.
dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
group = dist.new_group(ranks=list(range(4)), backend="nccl")
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
torch.cuda.set_stream(stream)


class Backend(TorchDistBackend):
    """SGLang's same full-world mapping, with workspace bootstrap on GLOO."""
    def __init__(self):
        super().__init__(group=group)

    def bcast(self, data, root):
        objects = [data]
        dist.broadcast_object_list(objects, src=root, group=dist.group.WORLD)
        return objects[0]


modules = {}
if a.path != "candidate":
    modules = {name: tvm_ffi.load_module(str(a.prepared / (name + ".so")))
               for name in ("native_fi", "native_hc")}
    mnnvl.gen_trtllm_mnnvl_comm_module = lambda: SimpleNamespace(
        build_and_load=lambda: modules["native_fi"]
    )
data = torch.load(a.prepared / "fixtures.pt", map_location="cpu", weights_only=True)
cfg = Config.from_text_config(SimpleNamespace(**data["config"]))
m, h = data["rows"], cfg.hidden
workspace = None if a.path == "candidate" else comm.create_allreduce_fusion_workspace(
    backend="mnnvl", world_size=4, rank=rank, max_token_num=2048,
    hidden_dim=h, dtype=torch.bfloat16, gpus_per_node=4,
    comm_backend=Backend(), group=group,
)
hop = None if a.path == "native" else Hop(
    config=cfg, max_m=m, group=group, device=torch.device("cuda", rank),
    cubin=(a.prepared / "hop.cubin").read_bytes())
inputs = {key: value.cuda() for key, value in data.items() if isinstance(value, torch.Tensor)}
local = torch.empty(m, h, dtype=torch.bfloat16, device="cuda")
residual = torch.empty(m, 4*h, dtype=torch.bfloat16, device="cuda")
gates = torch.empty(m, 8, 4, dtype=torch.float32, device="cuda")
gamma = inputs["gamma"]
ref_reduced, reduced = (torch.empty_like(local) for _ in range(2))
ref_combined, combined, ref_normed, normed = (torch.empty_like(residual) for _ in range(4))
args = None if hop is None else hop.args(
    partial=local, residual=residual, gates=gates, gamma=gamma,
    reduced=reduced, combined=combined, normalized=normed)
side = torch.cuda.Stream()
ready = torch.cuda.Event(external=True)
eviction = torch.zeros(384*1024*1024//4, dtype=torch.int32, device="cuda")
a.out.mkdir(parents=True, exist_ok=True)
result = dict(rank=rank, config=data["config"], rows=m, checks=[], timing=[],
              path=a.path, native_executed=a.path!="candidate", candidate_executed=a.path!="native",
              native="same-image mnnvl ONESHOT kAllReduce PDL=true/trigger_end=false + 2a92 native HC parts2",
              scope="synthetic TP4 same-input/same-boot microbenchmark; no executor/service gain")


def save():
    (a.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")


def parent():
    comm.allreduce_fusion(
        input=local, workspace=workspace, pattern=comm.AllReduceFusionPattern.kAllReduce,
        launch_with_pdl=True, trigger_completion_at_end=False, output=ref_reduced,
    )
    modules["native_hc"].hc_native(ref_reduced, residual, gates, gamma,
                                 ref_combined, ref_normed, cfg.eps)


def publish():
    hop.publish(args, stream=stream.cuda_stream)


def consume():
    hop.consume(args, stream=stream.cuda_stream)


def candidate():
    publish()
    consume()


def copy_case(case, *, delay=False):
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        if delay:
            torch.cuda._sleep(100000)
        local.copy_(inputs["local"][case, rank])
        residual.copy_(inputs["residual"][case])
        gates.copy_(inputs["gates"][case])
        ready.record()
    stream.wait_event(ready)


def norm_errors(got, oracle):
    values = got.cpu().double().reshape(m, 4, h)
    special = (torch.equal(values.isnan(), oracle.isnan())
               and torch.equal(values.isposinf(), oracle.isposinf())
               and torch.equal(values.isneginf(), oracle.isneginf()))
    mask = oracle.isfinite()
    error = (values[mask] - oracle[mask]).abs()
    denom = float(oracle[mask].norm())
    norm = float(error.norm())
    rel = norm / denom if denom else (0. if norm == 0 else float("inf"))
    return dict(rel_l2=rel, max_abs=float(error.max()) if error.numel() else 0.,
                nonfinite_classes_equal=special)


def check(label):
    torch.cuda.synchronize()
    if a.path != "compare":
        ctrl = hop.ctrl.cpu().tolist() if hop is not None else None
        row = dict(label=label, scope=a.path+" sanitizer only; not native numerical parity",
                   ctrl=ctrl, passed=ctrl is None or (ctrl[0]==ctrl[2]%3 and ctrl[1]==0))
        result["checks"].append(row)
        save()
        print("HC_HOP_TOOL_CHECK", rank, a.path, label, row["passed"], flush=True)
        if not row["passed"]:
            raise RuntimeError("candidate protocol state invalid")
        return
    oracle_input = ref_combined.cpu().double().reshape(m, 4, h)
    oracle = (oracle_input * torch.rsqrt(oracle_input.square().mean(-1, keepdim=True) + cfg.eps)
              * (1 + data["gamma"].double().reshape(4, h)))
    native_error, candidate_error = norm_errors(ref_normed, oracle), norm_errors(normed, oracle)
    norm_ok = (candidate_error["nonfinite_classes_equal"]
               and native_error["nonfinite_classes_equal"]
               and candidate_error["rel_l2"] <= 1.05*native_error["rel_l2"]
               and candidate_error["max_abs"] <= 2.*native_error["max_abs"])
    ctrl = hop.ctrl.cpu().tolist()
    row = dict(label=label,
               reduced_bitwise=bool(torch.equal(reduced.view(torch.int16), ref_reduced.view(torch.int16))),
               combined_bitwise=bool(torch.equal(combined.view(torch.int16), ref_combined.view(torch.int16))),
               native_fp64_error=native_error, candidate_fp64_error=candidate_error,
               norm_gate=norm_ok, ctrl=ctrl, stage_valid=ctrl[0] == ctrl[2] % 3 and ctrl[1] == 0)
    row["passed"] = all(row[k] for k in ("reduced_bitwise", "combined_bitwise", "norm_gate", "stage_valid"))
    result["checks"].append(row)
    save()
    print("HC_HOP_CHECK", rank, label, row["passed"], flush=True)
    if not row["passed"]:
        raise RuntimeError(f"HC hop numeric/protocol stop gate: {row}")


for case in range(len(data["local"])):
    copy_case(case, delay=case % 2 == 1)
    if a.path != "native":
        candidate()
    if a.path != "candidate":
        parent()
    check(f"eager-case{case}")

# Captured true readiness edge, changed data and delayed rank3, through many ring wraps.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    stream.wait_event(ready)
    parent() if a.path == "native" else candidate()
for replay in range(15):
    copy_case(replay % len(data["local"]), delay=replay % 2 == 1)
    if rank == 3 and replay % 3 == 1:
        torch.cuda._sleep(1000000)
    graph.replay()
    if a.path == "compare":
        parent()
    check(f"graph-change-skew-{replay}")

# Let consume poll before this generation's publish; no cross-stream ready bypass.
for trial in (() if a.path == "native" else range(3)):
    copy_case(trial)
    torch.cuda.synchronize()
    dist.barrier()
    producer = torch.cuda.Stream()
    producer.wait_event(ready)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(1000000 if rank == 3 else 100000)
        hop.publish(args, stream=producer.cuda_stream)
    consume()
    torch.cuda.synchronize()
    if a.path == "compare":
        parent()
    check(f"late-publish-{trial}")

result["checks_completed"] = True
result["parity_passed"] = a.path == "compare"
save()


def capture(mode, cold, calls=96):
    # Each consume has its own real same-generation publish; it never reads a
    # mailbox filled once and reused without a protocol transition.
    for _ in range(3):
        parent() if mode == "parent" else candidate()
    torch.cuda.synchronize()
    dist.barrier()
    pairs = [(torch.cuda.Event(enable_timing=True, external=True),
              torch.cuda.Event(enable_timing=True, external=True)) for _ in range(calls)]
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured, stream=stream):
        for start, end in pairs:
            if cold:
                hop.evict(eviction, stream=stream.cuda_stream)
            if mode == "consume":
                publish()
            start.record()
            if mode == "parent":
                parent()
            elif mode == "consume":
                consume()
            else:
                candidate()
            end.record()
    return captured, pairs


if not a.parity_only and a.path == "compare":
    for kind, case in (("gdn", 0), ("qsa", 2), ("moe", 3)):
        copy_case(case)
        torch.cuda.synchronize()
        for cold in (False, True):
            modes = ("parent", "consume", "publish_consume")
            captures = {mode: capture(mode, cold) for mode in modes}
            samples = []
            for trial in range(11):
                record = dict(trial=trial, order=list(modes if trial % 2 == 0 else reversed(modes)))
                for mode in record["order"]:
                    dist.barrier()
                    captured, pairs = captures[mode]
                    captured.replay()
                    pairs[-1][1].synchronize()
                    times = [start.elapsed_time(end)*1000 for start, end in pairs]
                    record[mode] = dict(first_us=times[0], tail95_mean_us=statistics.mean(times[1:]),
                                        all96_us=times)
                samples.append(record)
            entry = dict(kind=kind, input_case=case, synthetic=True, cold=cold, samples=samples)
            result["timing"].append(entry)
            save()
            print("HC_HOP_TIMED", rank, kind, cold, flush=True)

all_results = [None]*4
dist.all_gather_object(all_results, result)
if rank == 0:
    summary = dict(parity=all(r["parity_passed"] for r in all_results),
                   all_checks_completed=all(r["checks_completed"] for r in all_results), path=a.path,
                   minimum_same_method_saving_us=.9,
                   gate="lead 09:40: consume at least 0.9us faster than native in both hot/cold modes",
                   weights=dict(gdn=36,qsa=12,moe=48),
                   consume_excludes_same_generation_publish=True, modes={})
    for cold in (() if a.parity_only or a.path!="compare" else (False, True)):
        kinds = {}
        for kind in ("gdn", "qsa", "moe"):
            trials = []
            for trial in range(11):
                row = {}
                for mode in ("parent", "consume", "publish_consume"):
                    row[mode] = max(next(t for t in r["timing"] if t["kind"]==kind and t["cold"]==cold)
                                    ["samples"][trial][mode]["tail95_mean_us"] for r in all_results)
                trials.append(row)
            if not a.parity_only:
                kinds[kind] = {mode:statistics.median(row[mode] for row in trials)
                               for mode in ("parent", "consume", "publish_consume")}
        if not a.parity_only:
            weighted = {mode:sum(summary["weights"][kind]*row[mode] for kind,row in kinds.items())/96
                        for mode in ("parent", "consume", "publish_consume")}
            summary["modes"]["cold" if cold else "hot"] = dict(kinds=kinds,weighted=weighted,
                saving_us=weighted["parent"]-weighted["consume"],
                passed=weighted["parent"]-weighted["consume"]>=summary["minimum_same_method_saving_us"])
    summary["stage1_passed"] = (summary["parity"] and not a.parity_only
                                and all(v["passed"] for v in summary["modes"].values()))
    (a.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+'\n')
if workspace is not None:
    workspace.destroy()
dist.barrier()
dist.destroy_process_group(group)
dist.destroy_process_group()
