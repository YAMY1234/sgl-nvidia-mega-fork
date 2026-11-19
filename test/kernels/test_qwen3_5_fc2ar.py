# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import json
import os
import random
import statistics as stat
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

sys.path.insert(0, str(Path(__file__).resolve().parent))

HIDDEN, TOP_K, EPS, LOCAL, INTER = 4096, 10, 1e-6, 128, 512
ROWS = tuple(
    int(r)
    for r in os.environ.get(
        "FC2AR_ROWS", "35,42,49,56,63,64,70,77,84,91,98,105,112,119,126,128"
    ).split(",")
)
Q = 7  # rows a request (the verify's q_len)
# slot moe_engine_route's widths at 7 rows a request (5-9 requests)
ENGINE_ROUTE_ROWS = (35, 42, 49, 56, 63)
FP8 = torch.float8_e4m3fn
SENT = -1

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
def kit(rank):
    from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
        FlashInferMNNVLCuteDSLARFusion,
    )
    from sglang.srt.runtime_context import get_spec

    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.boundary.bt_tail import BTTail
    from sgl_nvidia_mega.qwen3_5.moe import body, fc2ar, route
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    group = ps.get_tp_group().device_group
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    with get_spec().override(speculative_algorithm="EAGLE"):
        fusion = FlashInferMNNVLCuteDSLARFusion(
            hidden_size=HIDDEN,
            top_k=TOP_K,
            max_m=plan.BT_TAIL_MAX_ROWS,
            rms_epsilon=EPS,
            weight_bias=0.0,
            process_group=group,
            device=dev,
        )
    service = SimpleNamespace(
        _workspace=fusion, hidden_size=HIDDEN, top_k=TOP_K, rms_epsilon=EPS
    )
    tail = BTTail(
        service=service,
        eps=EPS,
        device=dev,
        min_rows=plan.BT_TAIL_MIN_ROWS,
        max_rows=plan.BT_TAIL_MAX_ROWS,
        early=frozenset(("all_reduce", "finalize")),
    )
    assert tail.bind() == []
    experts = SimpleNamespace(top_k=TOP_K, num_local_experts=LOCAL)
    tail.fc2ar, why = fc2ar.build(EPS, group, dev, ROWS, experts)
    assert why == "" and fc2ar.active() is tail.fc2ar, why
    router.build_big(dev.index, LOCAL)
    route.configure(gemm=True, lean=True, engine_route=True)
    body.configure(True)
    st = SimpleNamespace(
        slots=plan.Slots(bt_tail=True, fc2ar=True),
        bound={"boundary": SimpleNamespace(bt=tail)},
    )
    return SimpleNamespace(tail=tail, lam=tail.fc2ar, dev=dev, st=st, chains={})


def _weight8(g, n, k, std=0.05):
    w = torch.randn(n, k, device="cuda", generator=g) * std
    s = (w.abs().amax() / 448.0).float().reshape(1)
    return (w / s).to(FP8).contiguous(), s


def _kind():
    return os.environ.get("FC2AR_KIND", "skew:0.75")


class Chain:
    def __init__(self, kit, m: int, layers: int, seed: int = 0):
        from test_qwen3_5_moe import _ep_router_inputs, _experts_of, _local_layer

        from sgl_nvidia_mega.qwen3_5.moe import body, route
        from sgl_nvidia_mega.qwen3_5.sm103 import lowm
        from sgl_nvidia_mega.qwen3_5.sm103.moe import router

        self.kit, self.m, self.L = kit, m, layers
        r = dist.get_rank()
        ep = r % 4
        self.layers = [
            _experts_of(_local_layer(LOCAL, INTER, 100 + i + 1000 * r), ep)
            for i in range(layers)
        ]
        kind = _kind()
        base = kind.split(":")[0]
        x, w0, self.scale = _ep_router_inputs(
            m, 50 + m + seed, "random" if base == "skew" else base, (ep * LOCAL, LOCAL)
        )
        if base == "skew":
            xf = x.float()
            up = xf.mean(0) / xf.mean(0).pow(2).sum()
            gb = torch.Generator(device="cuda").manual_seed(9)
            bias = torch.randn(w0.shape[0], 1, device="cuda", generator=gb) * float(
                kind.split(":")[1]
            )
            w0 = (w0.float() + bias * up).to(torch.bfloat16)
        self.x = x
        gg = torch.Generator(device="cuda").manual_seed(7 + m + seed)
        self.ws = [w0] + [
            (
                w0.float() + 0.002 * torch.randn(w0.shape, device="cuda", generator=gg)
            ).to(torch.bfloat16)
            for _ in range(layers - 1)
        ]
        self.s_in = (x.float().abs().amax() / 448.0 * 1.1).reshape(1).float()
        body._WIDTHS.clear()
        self.width = body.width(self.layers[0], x)
        gs = torch.Generator(device="cuda").manual_seed(31 + r + seed)
        self.sh = []
        for _ in range(layers):
            gu, sgu = _weight8(gs, 256, HIDDEN)
            dn, sdn = _weight8(gs, HIDDEN, 128)
            gate8 = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
            gate8[:1].copy_(
                (torch.randn(1, HIDDEN, device="cuda", generator=gs) * 0.02).to(
                    torch.bfloat16
                )
            )
            self.sh.append(
                dict(pairs=lowm.swiglu_pairs(gu), sgu=sgu, dn=dn, sdn=sdn, gate8=gate8)
            )
        self.s_h = torch.tensor([0.02], device="cuda")
        self.pw, self.psw = _weight8(gs, HIDDEN, 1024)
        self.p_in = torch.tensor([0.05], device="cuda")
        self.p_x = (torch.randn(m, 1024, device="cuda", generator=gs) * 0.5).to(
            torch.bfloat16
        )
        self.p_out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.h8 = [torch.empty(m, 128, dtype=FP8, device="cuda") for _ in range(layers)]
        self.logits = [
            torch.empty(m, 8, dtype=torch.float32, device="cuda") for _ in range(layers)
        ]
        self.sh_out = [
            torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
            for _ in range(layers)
        ]
        gc = torch.Generator(device="cuda").manual_seed(
            77 + m + seed
        )  # alike on every rank
        self.res0 = torch.randn(m, HIDDEN, device="cuda", generator=gc).to(
            torch.bfloat16
        )
        self.gamma = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gc)).to(
            torch.bfloat16
        )
        self.s8 = torch.tensor([0.05], device="cuda")
        self.lean_kw = dict(  # route.route_lean's launch at m rows
            lean=True,
            et=True,
            threads=route.route_lean_threads(m),
            ctas=route.route_lean_ctas(m),
        )
        router.build_big(
            torch.cuda.current_device(),
            LOCAL,
            [(True, True, self.lean_kw["threads"], router.big_shape(m)[0])],
        )
        self.alt = torch.cuda.Stream()
        # the router GEMM's floor: route_big_mid's below route-big's own
        self.gemm_min = min(m, route.BIG_GEMM_MIN_ROWS)
        self.sleep = None
        self.poison = False
        self.poison_routing = False
        self.saved = None

    def moe(self, i, arm):
        from sgl_nvidia_mega.qwen3_5.moe import body, fc2ar, route
        from sgl_nvidia_mega.qwen3_5.sm103 import lowm
        from sgl_nvidia_mega.qwen3_5.sm103.moe import router

        m = self.m
        cur = torch.cuda.current_stream()
        if self.sleep is not None and self.sleep[0] == i:
            torch.cuda._sleep(self.sleep[1])
        if self.poison_routing:
            self.width.tables.expanded_to_permuted.fill_(-7)
            self.width.tables.weights.view(torch.int16).fill_(0x7F7F)
        lowm.gemm(
            x16=self.p_x,
            w8=self.pw,
            w_scale=self.psw,
            in_scale=self.p_in,
            out=self.p_out,
            trigger=True,
        )
        self.alt.wait_stream(cur)
        scope = (
            fc2ar.scope(m, self.kit.st)
            if arm in ("fc2ar", "engine_route")
            else contextlib.nullcontext()
        )
        with scope:
            if arm != "bt":
                assert fc2ar.armed()
            if arm == "engine_route":
                got = route.router_gemm_a(self.x, self.ws[i], self.scale, True)
                assert not isinstance(got, str), got
                _, cand, x_fp4, x_sf = got
                rows, wts, e2p = body.run_engine_a(
                    self.width, self.layers[i], x_fp4, x_sf, cand
                )
                assert fc2ar.is_lamport(rows)
            else:
                lg = route.big_router_gemm(self.x, self.ws[i], min_rows=self.gemm_min)
                _, _, x_fp4, x_sf = router.route_big_quant(
                    lg, self.x, self.scale, self.width.tables, **self.lean_kw
                )
                rows, wts, e2p = body.run_engine(
                    self.width, self.layers[i], x_fp4, x_sf
                )
            with torch.cuda.stream(self.alt):
                w = self.sh[i]
                if self.poison:
                    self.sh_out[i].view(torch.uint8).fill_(0xFF)
                lowm.gemm(
                    x16=self.x,
                    w8=w["pairs"],
                    w_scale=w["sgu"],
                    in_scale=self.s_in,
                    w16=w["gate8"],
                    out=self.h8[i],
                    out2=self.logits[i],
                    swiglu_scale=self.s_h,
                    tactic=lowm.Tactic(8, 2, 1, 1, 4),
                    trigger=True,
                )
                lowm.gemm(
                    x8=self.h8[i],
                    w8=w["dn"],
                    w_scale=w["sdn"],
                    in_scale=self.s_h,
                    out=self.sh_out[i],
                    gate=self.logits[i][:, 0],
                    tactic=lowm.Tactic(2, 1, 1, 1, 1),
                )
                fc2ar.aux_fence()
        cur.wait_stream(self.alt)
        return rows, wts.view(m, TOP_K), e2p.view(m, TOP_K), self.sh_out[i]

    def run(self, arm, keep=False):
        """The chain: per layer (norm, residual_out, fp8)."""
        res = self.res0
        outs, saved = [], []
        for i in range(self.L):
            rows, wts, e2p, sh = self.moe(i, arm)
            if keep:
                saved.append((rows.clone(), wts.clone(), e2p.clone(), sh.clone()))
            o = self.kit.tail.moe_finalize_all_reduce_rms_norm(
                routed_output=rows,
                expert_weights=wts,
                permuted_indices=e2p,
                gated_shared_output=sh,
                residual=res,
                gamma=self.gamma,
                fp8_scale=self.s8,
            )
            res = o[1]
            outs.append(o)
        if keep:
            self.saved = saved
        return outs


def _buffers_sentinel() -> bool:
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar

    return all(
        bool((b.view(torch.int32) == SENT).all()) for b in fc2ar._BUFFERS.values()
    )


def _refill():
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar

    for b in fc2ar._BUFFERS.values():
        b.view(torch.int32).fill_(SENT)
    torch.cuda.synchronize()


def _reference(rows, wts, e2p, sh, residual, gamma):
    """fp64 (norm, residual_out) of the sum over ranks of each rank's folded rows."""
    perm = e2p.long()
    r = rows.double()[perm.clamp_min(0)]
    w = wts.double().unsqueeze(-1) * (perm >= 0).unsqueeze(-1)
    mine = (w * r).sum(1) + sh.double()
    parts = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, mine, group=ps.get_tp_group().device_group)
    total = sum(parts) + residual.double()
    rstd = torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + EPS)
    return total * rstd * gamma.double(), total


def _per_request(got, want, ref):
    g = (got.double() - ref).pow(2).sum(-1)
    w = (want.double() - ref).pow(2).sum(-1)
    r = ref.pow(2).sum(-1)
    eg, ew = [], []
    for i in range(0, ref.shape[0], Q):
        rr = r[i : i + Q].sum().sqrt()
        eg.append((g[i : i + Q].sum().sqrt() / rr).item())
        ew.append((w[i : i + Q].sum().sqrt() / rr).item())
    worst = max(
        (a / b if b > 0 else (1.0 if a == 0 else float("inf"))) for a, b in zip(eg, ew)
    )
    batch = (g.sum().sqrt() / w.sum().sqrt()).item()
    return worst, batch, sum(eg) / sum(ew)


def test_lamport_rows_and_the_fold_hold_to_fp64(rank, kit):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.qwen3_5.moe import body, fc2ar

    with torch.inference_mode():
        for m in ROWS:
            ch = Chain(kit, m, layers=2, seed=m)
            want = ch.run("bt", keep=True)
            torch.cuda.synchronize()
            # the armed engine alone (layer 0): its Lamport rows vs the plain rows
            with fc2ar.scope(m, kit.st):
                from sgl_nvidia_mega.qwen3_5.moe import route
                from sgl_nvidia_mega.qwen3_5.sm103.moe import router

                lg = route.big_router_gemm(ch.x, ch.ws[0], min_rows=ch.gemm_min)
                _, _, x_fp4, x_sf = router.route_big_quant(
                    lg, ch.x, ch.scale, ch.width.tables, **ch.lean_kw
                )
                rows, wts, e2p = body.run_engine(ch.width, ch.layers[0], x_fp4, x_sf)
            torch.cuda.synchronize()
            assert fc2ar.is_lamport(rows), m
            plain, pw, pe, psh = ch.saved[0]
            valid = pe.reshape(-1)
            valid = valid[valid >= 0].long()
            assert torch.equal(
                rows[valid].view(torch.int16), plain[valid].view(torch.int16)
            ), m
            assert not bool((rows[valid].view(torch.int32) == SENT).any()), m
            words = rows.view(torch.int32).clone()
            words[valid] = SENT
            assert bool((words == SENT).all()), (
                f"m {m}: the engine wrote rows the fold does not read"
            )
            buf = next(
                b
                for b in fc2ar._BUFFERS.values()
                if b.data_ptr() <= rows.data_ptr() < b.data_ptr() + b.numel() * 2
            )
            tail_rows = buf.view(torch.int32)[rows.shape[0] :]
            assert bool((tail_rows == SENT).all()), m
            got = kit.tail.moe_finalize_all_reduce_rms_norm(
                routed_output=rows,
                expert_weights=wts.view(m, TOP_K),
                permuted_indices=e2p.view(m, TOP_K),
                gated_shared_output=psh,
                residual=ch.res0,
                gamma=ch.gamma,
                fp8_scale=ch.s8,
            )
            torch.cuda.synchronize()
            assert _buffers_sentinel(), f"m {m}: the fold left words written"
            for what, a, b in zip(("norm", "residual", "fp8"), got, want[0]):
                n_bad = int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                assert n_bad == 0, (m, what, n_bad)
            fp8, _ = static_quant_fp8(got[0], ch.s8, repeat_scale=False)
            assert torch.equal(got[2].view(torch.uint8), fp8.view(torch.uint8)), m
            n_ref, r_ref = _reference(plain, pw, pe, psh, ch.res0, ch.gamma)
            lam = ch.run("fc2ar")
            torch.cuda.synchronize()
            assert _buffers_sentinel(), f"m {m}: the overlapped fold left words written"
            for t0, t1 in zip(lam[0], got):
                assert torch.equal(t0.view(torch.uint8), t1.view(torch.uint8)), m
            for t0, t1 in zip(lam[1], want[1]):
                assert torch.equal(t0.view(torch.uint8), t1.view(torch.uint8)), m
            kit.lam.test_trtllm_numerics = True
            try:
                old = ch.run("fc2ar")
                torch.cuda.synchronize()
            finally:
                kit.lam.test_trtllm_numerics = False
            assert _buffers_sentinel(), f"m {m}: TensorRT-LLM's fold left words written"
            bad_n = int(
                (old[0][0].view(torch.int16) != want[0][0].view(torch.int16)).sum()
            )
            bad_r = int(
                (old[0][1].view(torch.int16) != want[0][1].view(torch.int16)).sum()
            )
            assert bad_n > 0 and bad_r > 0, (m, bad_n, bad_r)
            if rank == 0:
                hn = _per_request(old[0][0], want[0][0], n_ref)
                hr = _per_request(old[0][1], want[0][1], r_ref)
                print(
                    f"FP64 m {m:3d} world {dist.get_world_size()}: new == BT bit for bit (norm, residual, fp8; "
                    f"layer 1 too); TRT-LLM vs BT: norm worst / batch / summed {hn[0]:.4f} / {hn[1]:.4f} / "
                    f"{hn[2]:.4f}, residual {hr[0]:.4f} / {hr[1]:.4f} / {hr[2]:.4f}; TRT-LLM differs at {bad_n} "
                    f"norm / {bad_r} "
                    f"residual of {m * HIDDEN} values (ts {kit.lam.ts_ctas(m)})",
                    flush=True,
                )


def _graph(ch, arm, counters=None, refs=None):
    outs = ch.run(arm)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = ch.run(arm)
        if refs is not None:
            bad = torch.zeros((), dtype=torch.int64, device="cuda")
            for o, r in zip(outs, refs):
                for a, b in zip(o, r):
                    bad += (
                        (a.view(torch.uint8) != b.view(torch.uint8))
                        .any()
                        .to(torch.int64)
                    )
            counters += (bad > 0).to(torch.int64)
    return g, outs


def test_graph_replay_and_skewed_stress(rank, kit):
    n = int(os.environ.get("FC2AR_STRESS", "2000"))
    world = dist.get_world_size()
    rng = random.Random(1234 + rank)
    total = 0
    with torch.inference_mode():
        for m in tuple(
            int(r)
            for r in os.environ.get(
                "FC2AR_STRESS_ROWS", "64,98,112,128,56,35,91"
            ).split(",")
        ):
            ch = Chain(kit, m, layers=4, seed=3 * m)
            ch.poison = True
            ch.sleep = (rank % ch.L, 20_000 * (1 + rank))
            ref = [[t.clone() for t in o] for o in ch.run("fc2ar")]
            torch.cuda.synchronize()
            assert _buffers_sentinel()
            counters = torch.zeros((), dtype=torch.int64, device="cuda")
            g, outs = _graph(ch, "fc2ar", counters, ref)
            for i in range(3):
                g.replay()
            torch.cuda.synchronize()
            for o, r in zip(outs, ref):
                for a, b in zip(o, r):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), m
            counters.zero_()
            t0 = time.time()
            for i in range(n):
                if i % 64 == 0:
                    time.sleep(rng.random() * 2e-4)
                g.replay()
            torch.cuda.synchronize()
            bad = int(counters.item())
            assert _buffers_sentinel()
            total += n
            print(
                f"STRESS rank {rank} m {m:3d} world {world}: {n} replays of 4 layers, {bad} differ "
                f"({time.time() - t0:.1f} s)",
                flush=True,
            )
            assert bad == 0, (m, bad)
            del g
            if m in (98, 56):
                kit.lam.test_no_poll = True
                try:
                    counters.zero_()
                    gc, _ = _graph(ch, "fc2ar", counters, ref)
                    counters.zero_()
                    for i in range(100):
                        gc.replay()
                    torch.cuda.synchronize()
                    caught = int(counters.item())
                finally:
                    kit.lam.test_no_poll = False
                    _refill()
                print(
                    f"POSCTL rank {rank} m {m:3d}: rows read once, not polled: {caught} / 100 replays differ",
                    flush=True,
                )
                assert caught > 0, m
                del gc
            dist.barrier()


def _tile(default: int) -> int:
    return int(os.environ.get("FC2AR_TILE", str(default)))


@contextlib.contextmanager
def _forced_tile(tile: int):
    from sgl_nvidia_mega.qwen3_5.moe import tactics

    if not tile:
        yield
        return
    pick = tactics.pick
    tactics.pick = lambda *a, **k: [tile, 0]
    try:
        yield
    finally:
        tactics.pick = pick


def _engine_route_rows() -> list:
    raw = os.environ.get("ENGINE_ROUTE_ROWS", ",".join(map(str, ENGINE_ROUTE_ROWS)))
    return [int(r) for r in raw.split(",")]


def _same(got, want, what):
    for o, r in zip(got, want):
        for a, b in zip(o, r):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), what


def test_engine_route_folds_bit_for_bit_as_route_big(rank, kit):
    tile = _tile(16)
    with torch.inference_mode():
        for m in _engine_route_rows():
            with _forced_tile(tile):
                ch = Chain(kit, m, layers=3, seed=11 * m)
            assert not tile or ch.width.tables.tile == tile
            want = ch.run("fc2ar")
            torch.cuda.synchronize()
            assert _buffers_sentinel(), m
            got = ch.run("engine_route")
            torch.cuda.synchronize()
            assert _buffers_sentinel(), (
                f"m {m}: the routing engine's fold left words written"
            )
            _same(got, want, f"m {m}")
            ch.poison_routing = True
            again = ch.run("engine_route")
            torch.cuda.synchronize()
            assert _buffers_sentinel(), m
            _same(again, want, f"m {m}, routing poisoned")
            if rank == 0:
                print(
                    f"ENGINE_ROUTE m {m:3d} tile {ch.width.tables.tile}: 3 layers bit for bit the route-big chain's "
                    "(norm, residual, FP8), plain and routing-poisoned",
                    flush=True,
                )


def _engine_route_replays(ch, counters, ref, n, rng, trigger=None, delay_ns=0):
    from sgl_nvidia_mega.qwen3_5.moe import body

    saved = (body.ROUTED_TRIGGER, body.TEST_ROUTE_DELAY_NS)
    if trigger is not None:
        body.ROUTED_TRIGGER = trigger
    body.TEST_ROUTE_DELAY_NS = delay_ns
    try:
        counters.zero_()
        g, outs = _graph(ch, "engine_route", counters, ref)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        counters.zero_()
        for i in range(n):
            if i % 64 == 0:
                time.sleep(rng.random() * 2e-4)
            g.replay()
        torch.cuda.synchronize()
        bad = int(counters.item())
        del g
    finally:
        body.ROUTED_TRIGGER, body.TEST_ROUTE_DELAY_NS = saved
    return bad, outs


def test_engine_route_graph_replay_and_skewed_stress(rank, kit):
    n = int(os.environ.get("ENGINE_ROUTE_STRESS", "2000"))
    n_late = int(os.environ.get("ENGINE_ROUTE_LATE", str(max(100, n // 10))))
    late_ns = int(os.environ.get("ENGINE_ROUTE_LATE_NS", "20000"))
    pos_ns = int(os.environ.get("ENGINE_ROUTE_POSCTL_NS", "40000"))
    world = dist.get_world_size()
    rng = random.Random(4321 + rank)
    tile = _tile(16)
    rows = _engine_route_rows()
    with torch.inference_mode():
        for m in rows:
            with _forced_tile(tile):
                ch = Chain(kit, m, layers=4, seed=13 * m)
            ch.poison = True
            ch.sleep = (rank % ch.L, 20_000 * (1 + rank))
            ref = [[t.clone() for t in o] for o in ch.run("fc2ar")]
            torch.cuda.synchronize()
            assert _buffers_sentinel()
            ch.poison_routing = True
            counters = torch.zeros((), dtype=torch.int64, device="cuda")
            t0 = time.time()
            bad, outs = _engine_route_replays(ch, counters, ref, n, rng)
            assert _buffers_sentinel()
            _same(outs, ref, m)
            print(
                f"ENGINE_ROUTE STRESS rank {rank} m {m:3d} tile {ch.width.tables.tile} world {world}: {n} replays of 4 "
                f"layers, {bad} differ ({time.time() - t0:.1f} s)",
                flush=True,
            )
            assert bad == 0, (m, bad)
            if n_late:
                hold = late_ns if rank % 2 else 0
                t0 = time.time()
                late, _ = _engine_route_replays(
                    ch, counters, ref, n_late, rng, delay_ns=hold
                )
                assert _buffers_sentinel(), (
                    f"m {m}: the late release left words written"
                )
                print(
                    f"ENGINE_ROUTE LATE rank {rank} m {m:3d} world {world}: routing stores held {hold} ns, {n_late} "
                    f"replays of 4 layers, {late} differ ({time.time() - t0:.1f} s)",
                    flush=True,
                )
                assert late == 0, (m, late)
            if m in (rows[0], rows[-1]):
                try:
                    caught, _ = _engine_route_replays(
                        ch, counters, ref, 100, rng, trigger=2, delay_ns=pos_ns
                    )
                finally:
                    _refill()
                print(
                    f"ENGINE_ROUTE POSCTL rank {rank} m {m:3d}: released at the engine's wait (early_trigger 2), routing "
                    f"stores held {pos_ns} ns: {caught} / 100 replays differ",
                    flush=True,
                )
                assert caught > 0, m
            dist.barrier()


def _kineto(g):
    from torch.profiler import ProfilerActivity, profile

    g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    p.export_chrome_trace(path)
    ev = [
        e
        for e in json.load(open(path))["traceEvents"]
        if e.get("ph") == "X" and e.get("cat") == "kernel"
    ]
    os.unlink(path)
    ev.sort(key=lambda e: e["ts"])
    eng = [k for k, e in enumerate(ev) if "MoeEngine" in e["name"]]
    keys = ("Slot", "inaliz", "Owner", "q35bt", "bt_tail", "Publish", "Reduce")
    rows = []
    for j, k in enumerate(eng):
        e = ev[k]
        e0, e1 = e["ts"], e["ts"] + e["dur"]
        lim = ev[eng[j + 1]]["ts"] if j + 1 < len(eng) else float("inf")
        ar = [
            x
            for x in ev[k + 1 :]
            if e0 <= x["ts"] < lim and any(t in x["name"] for t in keys)
        ]
        if not ar:
            continue
        a0 = min(x["ts"] for x in ar)
        a1 = max(x["ts"] + x["dur"] for x in ar)
        rows.append(dict(eng=e1 - e0, ar0=a0 - e0, exp=a1 - e1))
    if not rows:
        return "kineto: no layers"
    med = {k: stat.median(r[k] for r in rows) for k in rows[0]}
    return (
        "engine {eng:5.1f} us, AR start {ar0:+6.1f} from the engine start, AR end {exp:+5.1f} after the engine end "
        "(medians)"
    ).format(**med)


@contextlib.contextmanager
def _arm(kit, arm):
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar

    saved = fc2ar.ENGINE_GRID
    for part in arm.split(":")[1:]:
        if part.startswith("p"):
            kit.lam.poll_ns = int(part[1:])
        elif part.startswith("g"):
            fc2ar.ENGINE_GRID = int(part[1:])
        elif part == "t":
            kit.lam.test_trtllm_numerics = True
    try:
        yield
    finally:
        kit.lam.test_trtllm_numerics = False
        kit.lam.poll_ns = fc2ar.POLL_NS
        fc2ar.ENGINE_GRID = saved


def test_bench_chain_bt_vs_fc2ar(rank, kit):
    rows = [
        int(r) for r in os.environ.get("FC2AR_BENCH_ROWS", "56,70,84,98,112").split(",")
    ]
    L = int(os.environ.get("FC2AR_BENCH_LAYERS", "8"))
    rounds = int(os.environ.get("FC2AR_BENCH_ROUNDS", "3"))
    arms = os.environ.get("FC2AR_BENCH_ARMS", "bt,fc2ar,fc2ar:p200").split(",")
    with torch.inference_mode():
        for m in rows:
            with _forced_tile(_tile(0)):
                ch = Chain(kit, m, layers=L, seed=5 * m)
            graphs = {}
            for arm in arms:
                with _arm(kit, arm):
                    graphs[arm] = _graph(ch, arm.split(":")[0])[0]
            res = {a: [] for a in arms}
            for _ in range(rounds):
                for arm in arms:
                    g = graphs[arm]
                    for _ in range(3):
                        g.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    ts = []
                    for _ in range(30):
                        a.record()
                        g.replay()
                        b.record()
                        torch.cuda.synchronize()
                        ts.append(a.elapsed_time(b) * 1000.0 / L)
                    res[arm].append(sorted(ts)[len(ts) // 2])
            line = []
            for arm in arms:
                t = torch.tensor([stat.median(res[arm])], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                line.append((arm, t.item(), stat.median(res[arm])))
            if rank == 0:
                base = line[0][1]
                print(
                    f"BENCH m {m:3d} kind {_kind()} tile {ch.width.tables.tile} layers {L} world "
                    f"{dist.get_world_size()}: "
                    + "; ".join(
                        f"{a} {t:6.2f} us/layer (max rank; r0 {r0:6.2f}) {t - base:+5.2f}"
                        for a, t, r0 in line
                    ),
                    flush=True,
                )
            for arm in arms:
                k = _kineto(graphs[arm])
                if rank == 0:
                    print(f"  KINETO m {m:3d} {arm:10s} rank 0: {k}", flush=True)
            graphs.clear()
            torch.cuda.synchronize()
            dist.barrier()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8), timeout=3000)
