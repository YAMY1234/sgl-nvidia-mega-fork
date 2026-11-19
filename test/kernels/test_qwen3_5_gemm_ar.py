# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

HIDDEN, TOP_K, EPS, K, MAX_M = 4096, 10, 1e-6, 1024, 128
CALLS = int(os.environ.get("GEMM_AR_BENCH_CALLS", "20"))
NEG0 = -(2**31)
TEP8 = tuple(7 * b for b in range(5, 17))

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield rank
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


@pytest.fixture(scope="module")
def tail(rank):
    from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
        FlashInferMNNVLCuteDSLARFusion,
    )
    from sglang.srt.runtime_context import get_spec

    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.boundary.bt_tail import BTTail
    from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar
    from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k

    group = ps.get_tp_group().device_group
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with get_spec().override(speculative_algorithm="EAGLE"):
        fusion = FlashInferMNNVLCuteDSLARFusion(
            hidden_size=HIDDEN,
            top_k=TOP_K,
            max_m=plan.BT_TAIL_MAX_ROWS,
            rms_epsilon=EPS,
            weight_bias=0.0,
            process_group=group,
            device=device,
        )
    service = SimpleNamespace(
        _workspace=fusion, hidden_size=HIDDEN, top_k=TOP_K, rms_epsilon=EPS
    )
    t = BTTail(
        service=service,
        eps=EPS,
        device=device,
        min_rows=plan.BT_TAIL_MIN_ROWS,
        max_rows=plan.BT_TAIL_MAX_ROWS,
        early=frozenset(("all_reduce", "finalize")),
    )
    assert t.bind() == []
    t.twoshot, t.gemm_ar, why = gemm_ar.build(EPS, group, device, ROWS_ALL)
    assert why == "" and t.gemm_ar is not None, why
    world = ps.get_tp_group().world_size
    assert (t.gemm_ar.lo, t.gemm_ar.hi) == (
        plan.GEMM_AR_MIN_ROWS,
        min(plan.GEMM_AR_MAX_ROWS, k.max_rows(HIDDEN, world, k.DEFAULT_OPTS)),
    )
    t.fusion = fusion
    return t


ROWS_ALL = tuple(sorted(set(TEP8) | {33, 64, 65, 128}))


def _rows():
    from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k

    world = ps.get_tp_group().world_size
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return tuple(m for m in ROWS_ALL if not k.unsupported(m, K, HIDDEN, world, sms))


def _operands(m, device, rank, seed):
    own = torch.Generator(device=device).manual_seed(seed * 100 + rank)
    common = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, gen=own):
        return torch.randn(*shape, device=device, generator=gen)

    rows = m * TOP_K
    perm = (
        torch.randperm(rows, device=device, generator=own)
        .to(torch.int32)
        .view(m, TOP_K)
    )
    drop = (
        torch.rand(m, TOP_K, device=device, generator=own) < 0.25
    )  # routed to other EP ranks, as at TEP8
    w = (randn(HIDDEN, K) * 0.5).to(torch.float8_e4m3fn)
    return dict(
        x=(randn(m, K) * 2).to(torch.float8_e4m3fn),
        lin=SimpleNamespace(
            weight=w.t(),  # [K, H]: ModelOpt's layout after loading
            input_scale=torch.tensor([0.1], device=device),
            weight_scale=torch.tensor([0.3], device=device),
        ),
        routed=randn(rows + 5, HIDDEN).bfloat16(),
        expert_weights=torch.rand(m, TOP_K, device=device, generator=own).bfloat16(),
        permuted=torch.where(drop, torch.full_like(perm, -1), perm),
        shared=randn(m, HIDDEN).bfloat16(),
        residual=randn(m, HIDDEN, gen=common).bfloat16(),
        gamma=(1 + 0.1 * randn(HIDDEN, gen=common)).bfloat16(),
        scale=torch.tensor([0.02], device=device),
    )


def _bmm(x):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    lin = x["lin"]
    return flashinfer_bmm_fp8(
        x["x"], lin.weight, lin.input_scale, lin.weight_scale, torch.bfloat16
    )


def _bt_ar(t, x, quant=True):
    return t.all_reduce_residual_rms_norm(
        local_contribution=_bmm(x),
        residual=x["residual"],
        gamma=x["gamma"],
        **({"fp8_scale": x["scale"]} if quant else {}),
    )


def _l5(t, x, quant=True):
    """Slot gemm_ar: the projection deferred into the BT tail's all-reduce of its partial."""
    out = torch.empty(
        (x["x"].shape[0], HIDDEN), dtype=torch.bfloat16, device=x["x"].device
    )
    assert t.gemm_ar.defer(out, x["x"], x["lin"], "test")
    return t.all_reduce_residual_rms_norm(
        local_contribution=out,
        residual=x["residual"],
        gamma=x["gamma"],
        **({"fp8_scale": x["scale"]} if quant else {}),
    )


def _fusion(t, op, x):
    if op == "ar":
        return t.fusion.all_reduce_residual_rms_norm(
            local_contribution=_bmm(x), residual=x["residual"], gamma=x["gamma"]
        )
    return t.fusion.moe_finalize_all_reduce_rms_norm(
        routed_output=x["routed"],
        expert_weights=x["expert_weights"],
        permuted_indices=x["permuted"],
        gated_shared_output=x["shared"],
        residual=x["residual"],
        gamma=x["gamma"],
    )


def _fin(t, x):
    """The BT tail's finalize point (the layer input) on the same workspace."""
    return t.moe_finalize_all_reduce_rms_norm(
        routed_output=x["routed"],
        expert_weights=x["expert_weights"],
        permuted_indices=x["permuted"],
        gated_shared_output=x["shared"],
        residual=x["residual"],
        gamma=x["gamma"],
        fp8_scale=x["scale"],
    )


def _mismatch(got, want):
    return [
        int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())
        for a, b in zip(got, want)
    ]


def _idle_buffers_poisoned(two):
    torch.cuda.synchronize()
    cur = two.flags()[0]
    last = (cur + 2) % 3
    words = two._buf.view(torch.int32)[: two.flags_off // 4].view(3, -1)
    for b in range(3):
        if b != last:
            bad = (words[b] != NEG0).sum().item()
            assert bad == 0, (b, last, bad)
    dist.barrier()


def test_gemm_ar_vs_the_bt_path(rank, tail):
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    rows = _rows()
    calls = [("ar", m) for m in rows]
    calls += calls[::-1]
    stats, bad = {}, []
    with torch.inference_mode():
        torch.cuda.synchronize()
        cur0 = tail.twoshot.flags()[0]
        n_ws = 0
        for i, (op, m) in enumerate(calls):
            x = _operands(m, device, rank, seed=i)
            want = _bt_ar(tail, x)
            if i % (world + 1) == rank:
                torch.cuda._sleep(2_000_000)  # this rank arrives late
            got = _l5(tail, x)
            n_ws += 1
            mism = _mismatch(got, want)
            fus = _mismatch(_fusion(tail, op, x), want[:2])
            if fus != [0, 0]:
                bad.append((op, m, "bt_tail vs fusion", fus))
            q = (
                (got[0].float() * (1.0 / x["scale"].float()))
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            if (
                q.view(torch.uint8).int() - got[2].view(torch.uint8).int()
            ).abs().max().item() > 1:
                bad.append((op, m, "fp8"))
            stats[(op, m)] = mism
            torch.cuda.synchronize()
            assert tail.twoshot.flags()[0] == (cur0 + n_ws) % 3, (
                i,
                tail.twoshot.flags(),
            )
            _idle_buffers_poisoned(tail.twoshot)
    tot = {"ar": [0, 0, 0]}
    for (op, m), mism in stats.items():
        tot[op] = [a + b for a, b in zip(tot[op], mism)]
    if rank == 0:
        for (op, m), mism in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            print(
                f"VS-BT world {world} {op:3s} m {m:3d}: mismatches norm/res/fp8 {mism}",
                flush=True,
            )
        print(
            f"VS-BT world {world} TOTAL gemm_ar {tot['ar']} "
            f"elements per output {sum(m for o, m in calls if o == 'ar') * HIDDEN}",
            flush=True,
        )
        for b in bad:
            print(f"VS-BT world {world} ENVELOPE MISS {b}", flush=True)
    assert tot["ar"] == [0, 0, 0], tot
    assert not bad, bad


def test_gemm_ar_positive_control_differs(rank, tail):
    from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    m = 56
    x = _operands(m, device, rank, seed=4242)
    with torch.inference_mode():
        want = _bt_ar(tail, x)
        opts, tail.gemm_ar.opts = tail.gemm_ar.opts, tail.gemm_ar.opts | k.OPT_LIBNUM
        try:
            got = _l5(tail, x)
        finally:
            tail.gemm_ar.opts = opts
        mism = _mismatch(got, want)
        same = _mismatch(_l5(tail, x), want)
    if rank == 0:
        print(
            f"POSCTL m {m}: LIBNUM mismatches norm/res/fp8 {mism}, default opts {same}",
            flush=True,
        )
    assert mism[1] > 0 and same == [0, 0, 0], (mism, same)
    _idle_buffers_poisoned(tail.twoshot)


def test_gemm_ar_refusal_settle_and_unconsumed(rank, tail):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    m = 56
    x = _operands(m, device, rank, seed=77)
    with torch.inference_mode():
        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
        assert not tail.gemm_ar.defer(
            out, x["x"].view(torch.uint8), x["lin"], "refused"
        )  # not E4M3
        assert tail.gemm_ar.defer(out, x["x"], x["lin"], "settled")
        tail.gemm_ar.settle()
        assert torch.equal(out, _bmm(x))
        # a deferred projection the next all-reduce is not about raises there
        assert tail.gemm_ar.defer(out, x["x"], x["lin"], "dangling")
        with pytest.raises(RuntimeError):
            _fin(tail, x)
        tail.gemm_ar._pending = None
        torch.cuda.synchronize()
    dist.barrier()


def test_gemm_ar_replays_from_a_cuda_graph(rank, tail):
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    big = 112 if world == 8 else 63
    plan = [("ar", big), ("fin", big), ("ar", 35), ("bt", 35), ("fin", 56), ("ar", 56)]
    xs = [
        _operands(m, device, rank, seed=500 + m + 7 * j)
        for j, (_, m) in enumerate(plan)
    ]

    def call(op, x):
        if op == "ar":
            return _l5(tail, x)
        if op == "bt":
            return _bt_ar(tail, x)
        return _fin(tail, x)

    with torch.inference_mode():
        eager = [[t.clone() for t in call(op, x)] for (op, _), x in zip(plan, xs)]
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [call(op, x) for (op, _), x in zip(plan, xs)]
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(outs, eager):
                assert _mismatch(got, want) == [0] * len(got)
        for j, ((op, m), x) in enumerate(zip(plan, xs)):
            fresh = _operands(m, device, rank, seed=900 + 13 * j)
            for key in (
                "x",
                "routed",
                "expert_weights",
                "permuted",
                "shared",
                "residual",
                "gamma",
            ):
                x[key].copy_(fresh[key])
            x["lin"].weight.copy_(fresh["lin"].weight)
        graph.replay()
        torch.cuda.synchronize()
        for (op, m), x, got in zip(plan, xs, outs):
            want = call(op, x)
            assert _mismatch(got, want) == [0] * len(got), (op, m)
        _idle_buffers_poisoned(tail.twoshot)


def test_gemm_ar_pdl_chain_is_deterministic(rank, tail):
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    widths = (112, 56, 84, 35, 42) if world == 8 else (63, 56, 35, 42, 49)
    xs = [
        _operands(widths[j % len(widths)], device, rank, seed=1700 + j)
        for j in range(30)
    ]

    def run():
        res = {}
        outs = []
        for j, x in enumerate(xs):
            m = x["x"].shape[0]
            y = dict(x, residual=res.get(m, x["residual"]))
            o = _fin(tail, y) if j % 3 == 2 else _l5(tail, y)
            res[m] = o[1]
            outs.append(o)
        return outs

    with torch.inference_mode():
        run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = run()
        graph.replay()
        torch.cuda.synchronize()
        first = [[t.clone() for t in o] for o in outs]
        for r in range(20):
            if r % 4 == rank % 4:
                torch.cuda._sleep(1_000_000)
            graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(outs, first):
            assert _mismatch(got, want) == [0] * len(got)
        _idle_buffers_poisoned(tail.twoshot)


def _time(graph, calls):
    ts = []
    for _ in range(9):
        dist.barrier()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / calls)
    return sorted(ts)[4]


def test_bench_gemm_ar_chain(rank, tail):
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for m in [w for w in (35, 56, 63, 84, 112, 128) if w in _rows()]:
        x0 = _operands(m, device, rank, seed=2300 + m)
        x0["residual"] = x0["residual"] * 1e-3
        line = f"CHAIN world {world} rank {rank} m {m:3d}:"
        for kind in ("bt_res", "l5_res", "bt_x", "l5_x"):

            def chain(kind=kind):
                res, prev = x0["residual"], None
                for _ in range(CALLS):
                    y = dict(x0, residual=res)
                    if kind.endswith("_x") and prev is not None:
                        y["x"] = (prev[:, :K] * 4).to(torch.float8_e4m3fn)
                    o = _bt_ar(tail, y) if kind.startswith("bt") else _l5(tail, y)
                    prev, res = o[0], o[1]

            with torch.inference_mode():
                chain()
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    chain()
                graph.replay()
                torch.cuda.synchronize()
                line += f" {kind} {_time(graph, CALLS):6.2f}"
        if rank == 0 or os.environ.get("GEMM_AR_ALL_RANKS"):
            print(line, flush=True)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
