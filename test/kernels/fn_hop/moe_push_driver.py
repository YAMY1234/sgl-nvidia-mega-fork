"""Finite four-rank MoE merge/push qualification; GPU compilation is forbidden."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
import tvm_ffi

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_8.boundary.hop import Config, Hop

p = argparse.ArgumentParser()
p.add_argument("--work", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--phase", choices=("record", "arithmetic", "chain"), required=True)
p.add_argument("--parity-only", action="store_true")
a = p.parse_args()
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.empty(1, device="cuda")
dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=240))
group = dist.new_group(ranks=list(range(4)), backend="nccl")
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
torch.cuda.set_stream(stream)
data = torch.load(a.work / "hop/prepared/fixtures.pt", map_location="cpu", weights_only=True)
cfg = Config.from_text_config(SimpleNamespace(**data["config"]))
M, H = 4, cfg.hidden
a.out.mkdir(parents=True, exist_ok=True)
ref_dir = a.work / "recorded"
ref_dir.mkdir(exist_ok=True)
result = dict(rank=rank, phase=a.phase, rows=M, checks=[], timing=[],
              scope="old193 real L23 inputs, fresh FI0.7.2/native shared; isolated producer/HC, not service",
              native_shared="FI BF16 SplitK GU -> native SiLU BF16 -> native F.linear down BF16",
              source_sglang=os.environ["FN_SGLANG"], compile_allowed=False)

def save():
    (a.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")

def same(x, y):
    dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    return bool(torch.equal(x.view(dtype), y.view(dtype)))

workspace = None
if a.phase != "arithmetic":
    import flashinfer.comm as comm
    import flashinfer.comm.trtllm_mnnvl_ar as mnnvl
    from flashinfer.comm.mnnvl import TorchDistBackend
    class Backend(TorchDistBackend):
        def __init__(self):
            super().__init__(group=group)
        def bcast(self, value, root):
            values = [value]
            dist.broadcast_object_list(values, src=root)
            return values[0]
    fi = tvm_ffi.load_module(str(a.work / "hop/prepared/native_fi.so"))
    hc = tvm_ffi.load_module(str(a.work / "hop/prepared/native_hc.so"))
    mnnvl.gen_trtllm_mnnvl_comm_module = lambda: SimpleNamespace(build_and_load=lambda: fi)
    workspace = comm.create_allreduce_fusion_workspace(
        backend="mnnvl", world_size=4, rank=rank, max_token_num=2048,
        hidden_dim=H, dtype=torch.bfloat16, gpus_per_node=4,
        comm_backend=Backend(), group=group)

reduced, combined, normalized = (torch.empty(shape, dtype=torch.bfloat16, device="cuda")
                                for shape in ((M,H), (M,4*H), (M,4*H)))
residual = data["residual"][0].cuda()
gates = data["gates"][0].cuda()
gamma = data["gamma"].cuda()

def native_hc(local):
    comm.allreduce_fusion(input=local, workspace=workspace,
        pattern=comm.AllReduceFusionPattern.kAllReduce, launch_with_pdl=True,
        trigger_completion_at_end=False, output=reduced)
    hc.hc_native(reduced, residual, gates, gamma, combined, normalized, cfg.eps)

if a.phase == "record":
    from moe_fixture import MoeFixture
    from sgl_nvidia_mega.qwen3_8.sm103.moe import finalize
    producer = MoeFixture(rank=rank, rows=M, work=a.work, qc=Path(os.environ["Q38CC_ROOT"]))
    recorded = []
    for index, case in enumerate(producer.cases):
        producer.x.copy_(case["hidden"])
        rows, weights, permutation = producer.native_experts(case, do_finalize=False)[:3]
        rows = rows[permutation.reshape(-1).long()[:M*10]].reshape(M,10,H)
        native_routed = producer.native_experts(case, do_finalize=True)[0]
        ordered = finalize.run(rows.reshape(M*10,H), weights, shape=producer.shape)
        torch.cuda.synchronize()
        assert same(ordered, native_routed), "native FI vs ordered finalize mismatch"
        shared = producer.shared()
        producer.merge("gate_only", shared, producer.gate)
        local = native_routed.clone()
        producer.merge("native_merge", shared, local)
        native_hc(local)
        torch.cuda.synchronize()
        recorded.append({k:v.cpu() for k,v in dict(
            x=producer.x, rows=rows, weights=weights, shared=shared, gate=producer.gate,
            gate_weight=producer.gate_weight, local=local, reduced=reduced,
            combined=combined, normalized=normalized).items()})
        result["checks"].append(dict(label=f"fresh-native-L23-{index}", passed=True,
                                     metadata=producer.case_metadata[index]))
    # Arithmetic edges are explicitly synthetic, not native shared captures.
    for label in ("zero", "negative-zero", "cancel", "tiny", "inf", "nan", "saturated-gate"):
        case = {k:v.cuda() for k,v in recorded[0].items()}
        if label == "zero":
            case["rows"].zero_(); case["shared"].zero_()
        elif label == "negative-zero":
            case["rows"].zero_(); case["shared"].zero_()
            case["rows"].view(torch.int16).fill_(-32768)
            case["shared"].view(torch.int16).fill_(-32768)
        elif label == "cancel":
            case["rows"][:,1::2] = -case["rows"][:,::2]
            case["weights"].fill_(.5)
        elif label == "tiny":
            case["rows"] *= 2**-60; case["shared"] *= 2**-60
        elif label == "inf":
            case["rows"][0,0,19] = float("inf")
        elif label == "nan":
            case["shared"][0,17] = float("nan")
        else:
            case["x"] *= 1000
        producer.x.copy_(case["x"])
        local = finalize.run(case["rows"].reshape(M*10,H),case["weights"],shape=producer.shape)
        producer.merge("gate_only",case["shared"],producer.gate)
        producer.merge("native_merge",case["shared"],local)
        native_hc(local)
        torch.cuda.synchronize()
        case.update(gate=producer.gate.clone(),local=local,reduced=reduced.clone(),
                    combined=combined.clone(),normalized=normalized.clone())
        recorded.append({k:v.cpu() for k,v in case.items()})
        result["checks"].append(dict(label="synthetic-"+label,passed=True))
    torch.save(dict(cases=recorded, residual=residual.cpu(), gates=gates.cpu(), gamma=gamma.cpu()),
               ref_dir / f"rank{rank}.pt")
else:
    samples = torch.load(ref_dir / f"rank{rank}.pt", map_location="cpu", weights_only=True)
    cases = [{k:v.cuda() for k,v in c.items()} for c in samples["cases"]]
    hop = Hop(config=cfg,max_m=M,group=group,device=torch.device("cuda",rank),
              cubin=(a.work/"hop/prepared/hop.cubin").read_bytes())
    local = torch.empty((M,H),dtype=torch.bfloat16,device="cuda")
    args = hop.args(partial=local,residual=residual,gates=gates,gamma=gamma,
                    reduced=reduced,combined=combined,normalized=normalized)

    def check(index,label,got_local):
        torch.cuda.synchronize()
        target=cases[index]
        oracle_x=target["combined"].double().reshape(M,4,H)
        oracle=oracle_x*torch.rsqrt(oracle_x.square().mean(-1,keepdim=True)+cfg.eps)*(1+gamma.double().reshape(4,H))
        def errors(value):
            value=value.double().reshape(M,4,H)
            special=all(torch.equal(fn(value),fn(oracle)) for fn in (torch.isnan,torch.isposinf,torch.isneginf))
            delta=(value[oracle.isfinite()]-oracle[oracle.isfinite()]).abs()
            den=float(oracle[oracle.isfinite()].norm()); num=float(delta.norm())
            return dict(rel_l2=num/den if den else (0. if num==0 else float("inf")),
                        max_abs=float(delta.max()) if delta.numel() else 0.,special=special)
        ne,ce=errors(target["normalized"]),errors(normalized)
        norm_gate=(ne["special"] and ce["special"] and ce["rel_l2"]<=1.05*ne["rel_l2"] and ce["max_abs"]<=2*ne["max_abs"])
        row=dict(label=label,local_bitwise=same(got_local,target["local"]),
                 reduced_bitwise=same(reduced,target["reduced"]),combined_bitwise=same(combined,target["combined"]),
                 native_error=ne,candidate_error=ce,norm_gate=norm_gate,ctrl=hop.ctrl.cpu().tolist())
        row["passed"]=all(row[k] for k in ("local_bitwise","reduced_bitwise","combined_bitwise","norm_gate"))
        result["checks"].append(row);save()
        if not row["passed"]:raise RuntimeError(str(row))

    if a.phase=="arithmetic":
        raw=json.loads((a.work/"merge/prepared/CPU-PASS.json").read_text())
        gate_meta=next(x for x in raw["compiled"] if x["name"]=="gate_only")
        gate_kernel=jit.load((a.work/"merge/prepared/gate_only.cubin").read_bytes(),gate_meta["symbol"],(512,1,1),gate_meta["shared"])
        push=jit.load((a.work/"merge/prepared/finalize_merge_push.cubin").read_bytes(),"q38_moe_finalize_merge_push",(128,1,1),0)
        current={k:torch.empty_like(v) for k,v in cases[0].items() if k in ("x","rows","weights","shared","gate_weight")}
        gate=torch.empty(M,dtype=torch.float32,device="cuda")
        def candidate():
            jit.launch(gate_kernel,(M,1,1),[jit.pointer(current[k]) for k in ("x","gate_weight","shared")]+[jit.pointer(gate),ctypes.c_void_p(0),ctypes.c_void_p(0)],stream.cuda_stream)
            jit.launch(push,(5,M,1),[jit.pointer(current[k]) for k in ("rows","weights","shared")]+[jit.pointer(gate),jit.pointer(local),hop.site],stream.cuda_stream)
            hop.consume(args,stream=stream.cuda_stream)
        def select(index):
            for k,v in current.items():v.copy_(cases[index][k])
        for index in range(len(cases)):
            select(index);candidate();check(index,f"arithmetic-{index}",local)
            assert same(gate,cases[index]["gate"]),"native GATE_ONLY FP32 changed"
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):candidate()
        for replay in range(15):
            index=replay%len(cases);select(index)
            if rank==3 and replay%3==1:torch.cuda._sleep(1000000)
            graph.replay();check(index,f"graph-change-skew-{replay}",local)
    else:
        from moe_fixture import MoeFixture
        producer=MoeFixture(rank=rank,rows=M,work=a.work,qc=Path(os.environ["Q38CC_ROOT"]))
        def parent():native_hc(producer.parent())
        def candidate():
            producer.publish(hop.site)
            hop.consume(args,stream=stream.cuda_stream)
        for index in range(3):
            producer.x.copy_(producer.cases[index]["hidden"])
            candidate();check(index,f"full-chain-{index}",producer.local)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):candidate()
        for replay in range(9):
            index=replay%3;producer.x.copy_(producer.cases[index]["hidden"])
            graph.replay();check(index,f"full-chain-graph-{replay}",producer.local)
        if not a.parity_only:
            producer.x.copy_(producer.cases[0]["hidden"])
            eviction=torch.zeros(384*1024*1024//4,dtype=torch.int32,device="cuda")
            for cold in (False,True):
                graphs={}
                for mode,fn in (("parent",parent),("candidate",candidate)):
                    for _ in range(3):fn()
                    torch.cuda.synchronize();dist.barrier()
                    events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(96)]
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g,stream=stream):
                        for start,end in events:
                            if cold:hop.evict(eviction,stream=stream.cuda_stream)
                            start.record();fn();end.record()
                    graphs[mode]=(g,events)
                trials=[]
                for trial in range(11):
                    row=dict(trial=trial,order=["parent","candidate"] if trial%2==0 else ["candidate","parent"])
                    for mode in row["order"]:
                        dist.barrier();g,events=graphs[mode];g.replay();events[-1][1].synchronize()
                        values=[start.elapsed_time(end)*1000 for start,end in events]
                        row[mode]=dict(first_us=values[0],tail95_mean_us=statistics.mean(values[1:]),all96_us=values)
                    trials.append(row)
                result["timing"].append(dict(cold=cold,samples=trials));save()

result["passed"]=all(c["passed"] for c in result["checks"])
save()
reports=[None]*4;dist.all_gather_object(reports,result)
if rank==0:
    summary=dict(phase=a.phase,passed=all(r["passed"] for r in reports),checks_per_rank=[len(r["checks"]) for r in reports],modes={})
    for cold in (() if not result["timing"] else (False,True)):
        trials=[{mode:max(next(t for t in r["timing"] if t["cold"]==cold)["samples"][i][mode]["tail95_mean_us"] for r in reports) for mode in ("parent","candidate")} for i in range(11)]
        values={mode:statistics.median(t[mode] for t in trials) for mode in ("parent","candidate")}
        summary["modes"]["cold" if cold else "hot"]=dict(**values,delta_us=values["candidate"]-values["parent"],matched_max_rank_trials=trials)
    (a.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+'\n')
if workspace is not None:workspace.destroy()
dist.barrier();dist.destroy_process_group(group);dist.destroy_process_group()
