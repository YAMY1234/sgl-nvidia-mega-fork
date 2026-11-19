# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import logging
from types import SimpleNamespace

import pytest
import torch
from test_qwen3_5_moe import (  # noqa: E402  (pytest puts this directory on sys.path)
    _ep_router_inputs,
    _experts_of,
    _gpu_ready,
    _local_layer,
    _same_moe,
)

pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")

TOP_K = 10
LOCAL, INTER = 128, 512
ROWS = [
    17,
    21,
    28,
    32,
    33,
    56,
    63,
    64,
    70,
    77,
    84,
    91,
    98,
    105,
    112,
    113,
    119,
    126,
    128,
    168,
    224,
    256,
    257,
    259,
    280,
    336,
    392,
    448,
]
TABLES = (
    "ids",
    "weights",
    "counter",
    "total_padded",
    "expanded_to_permuted",
    "permuted_to_token",
    "tokens_per_expert",
    "cta_expert",
    "cta_row_limit",
    "live_ctas",
)


@pytest.fixture(scope="module")
def tep8_layer():
    return _local_layer(LOCAL, INTER, 17)


@pytest.fixture
def switches(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.moe import body, route

    body.configure(True)
    route.configure(gemm=True, lean=True)
    route.route_lean_build(torch.cuda.current_device(), LOCAL, ROWS)
    yield monkeypatch
    body.configure(False)
    route.configure()


def _inputs(m, kind, rank, seed):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    x, w, scale = _ep_router_inputs(m, seed, kind, (rank * LOCAL, LOCAL))
    logits = torch.nn.functional.linear(x, w)
    router.build_big(x.device.index, LOCAL)
    return logits, x, w, scale


def _poison(t, value=-7):
    for name in TABLES:
        if name == "weights":
            t.weights.view(torch.int16).fill_(0x7F7F)
        else:
            getattr(t, name).fill_(value)


def _snap(t, out):
    """route_big_quant's outputs and every table tensor, cloned (weights as bits)."""
    ids, weights, x_fp4, x_sf = out
    got = dict(x_fp4=x_fp4.clone(), x_sf=x_sf.clone())
    for name in TABLES:
        v = getattr(t, name)
        got[name] = (v.view(torch.int16) if name == "weights" else v).clone()
    return got


def _same(got, want):
    for key in want:
        assert torch.equal(got[key], want[key]), key


def _run(router, logits, x, scale, tables, **kw):
    _poison(tables)
    out = router.route_big_quant(logits, x, scale, tables, **kw)
    torch.cuda.synchronize()
    return _snap(tables, out)


def _variants(m):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    out = []
    for lean, et, threads, ctas in (
        (True, False, 1024, None),
        (True, True, 1024, None),
        (False, True, 1024, None),
        (True, True, 512, None),
        (True, True, 256, None),
        (True, True, 512, 64),
        (True, True, 256, 64),
        (True, True, 1024, 64),
    ):
        if not router.big_variant_unsupported(m, threads):
            out.append(dict(lean=lean, et=et, threads=threads, ctas=ctas))
    return out


@pytest.mark.parametrize("kind", ["random", "local", "same", "ties"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("m", ROWS)
def test_lean_route_big_is_route_big(m, rank, kind):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits, x, _, scale = _inputs(m, kind, rank, 12100 + 31 * m + rank)
    tables = router.RouteTables(m, 8 if m <= 64 else 16, x.device, rank * LOCAL, LOCAL)
    want = _run(router, logits, x, scale, tables)
    for kw in _variants(m):
        _same(_run(router, logits, x, scale, tables, **kw), want)
        _same(
            _run(router, logits, x, scale, tables, **kw), want
        )  # a second launch on the same tables


@pytest.mark.parametrize("tile", [8, 32, 64, 128])
@pytest.mark.parametrize("m", [64, 112, 256, 448])
def test_lean_route_big_other_tiles(m, tile):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits, x, _, scale = _inputs(m, "local", 0, 12500 + m + tile)
    tables = router.RouteTables(m, tile, x.device, 0, LOCAL)
    want = _run(router, logits, x, scale, tables)
    for kw in _variants(m):
        _same(_run(router, logits, x, scale, tables, **kw), want)


@pytest.mark.parametrize("m", [64, 112, 336, 448])
def test_lean_exchange_positive_control(m):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits, x, _, scale = _inputs(m, "local", 0, 12700 + m)
    tables = router.RouteTables(m, 16, x.device, 0, LOCAL)
    want = _run(router, logits, x, scale, tables)
    bad = _run(router, logits, x, scale, tables, lean=True, et=True, bad_slot=True)
    assert any(
        not torch.equal(bad[k], want[k])
        for k in ("expanded_to_permuted", "permuted_to_token")
    )


def test_lean_variant_refuses_other_shapes():
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits, x, _, scale = _inputs(449, "random", 0, 12800)
    tables = router.RouteTables(449, 16, x.device, 0, LOCAL)
    with pytest.raises(ValueError, match="clusters of 16"):
        router.route_big_quant(logits, x, scale, tables, lean=True)
    logits, x, _, scale = _inputs(168, "random", 0, 12801)
    tables = router.RouteTables(168, 16, x.device, 0, LOCAL)
    with pytest.raises(ValueError, match="two a warp"):
        router.route_big_quant(logits, x, scale, tables, lean=True, threads=256)


@pytest.mark.parametrize("m", [64, 112, 126, 280, 448])
def test_lean_route_big_replays_from_a_cuda_graph(m):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits, x, _, scale = _inputs(m, "same", 3, 12900 + m)
    tables = router.RouteTables(m, 16, x.device, 3 * LOCAL, LOCAL)
    want = _run(router, logits, x, scale, tables)
    kw = dict(
        lean=True, et=True, threads=512, ctas=None if m <= 128 else 64
    )  # as route.route_lean serves it
    _run(router, logits, x, scale, tables, **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.route_big_quant(logits, x, scale, tables, **kw)
    for _ in range(3):
        _poison(tables)
        captured[2].zero_()
        captured[3].zero_()
        graph.replay()
        torch.cuda.synchronize()
        _same(_snap(tables, captured), want)


def _width(tep8_layer, rank, x):
    from sgl_nvidia_mega.qwen3_5.moe import body

    experts = _experts_of(tep8_layer, rank)
    body._WIDTHS.clear()
    return body, experts, body.width(experts, x)


@pytest.mark.parametrize(
    "opts", [dict(), dict(early_trigger=2)], ids=["default", "et2"]
)
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("m", [21, 28, 64, 84, 112, 126, 168, 256, 280, 448])
def test_engine_on_lean_tables_is_the_route_big_chain(
    tep8_layer, switches, m, rank, opts
):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits, x, _, scale = _inputs(m, "random", rank, 13100 + m + rank)
    body, experts, width = _width(tep8_layer, rank, x)
    switches.setattr(body, "ENGINE_OPTIONS", opts)
    t = width.tables

    def chain(**kw):
        _poison(t)
        _, _, x_fp4, x_sf = router.route_big_quant(logits, x, scale, t, **kw)
        out = [v.clone() for v in body.run_engine(width, experts, x_fp4, x_sf)[:3]]
        torch.cuda.synchronize()
        return out

    want = chain()
    for kw in _variants(m):
        _same_moe(chain(**kw), want, m)
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [56, 84, 112, 280, 448])
def test_routed_topk_with_route_lean_and_the_engine_route(
    tep8_layer, switches, m, caplog
):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    from sgl_nvidia_mega.qwen3_5.moe import route

    caplog.set_level(logging.INFO)
    logits, x, w, scale = _inputs(m, "random", 0, 13300 + m)
    body, experts, width = _width(tep8_layer, 0, x)
    experts.w13_input_scale_quant = scale
    experts.dispatcher = SimpleNamespace(
        combine=lambda combine_input: combine_input.hidden_states
    )

    class _Gate:
        weight = w

        def __call__(self, h):
            return torch.nn.functional.linear(h, w), None

    mlp = SimpleNamespace(experts=experts, gate=_Gate())

    def fields(d):
        assert isinstance(d, FlashInferTrtllmDeferredFinalizeOutput)
        return [
            d.gemm2_out.clone(),
            d.expert_weights.clone(),
            d.expanded_idx_to_permuted_idx.clone(),
        ]

    switches.setitem(route._ROUTE_LEAN, "on", False)
    want = fields(route.routed_topk(mlp, x, pre=True))
    torch.cuda.synchronize()
    switches.setitem(route._ROUTE_LEAN, "on", True)
    switches.setitem(route._ENGINE_ROUTE, "on", True)
    for _ in range(2):
        _poison(width.tables)
        got = fields(route.routed_topk(mlp, x, pre=True))
        torch.cuda.synchronize()
        _same_moe(got, want, m)
    lines = [
        r.getMessage()
        for r in caplog.records
        if "slot route_lean runs" in r.getMessage()
    ]
    if m <= 63:
        assert not lines
    else:
        assert (
            len(lines) == 1
            and f"at {m} rows" in lines[0]
            and "released at the start" in lines[0]
        ), lines
        want = (
            "512 threads x 32 CTAs in clusters of 8"
            if m <= 128
            else "512 threads x 64 CTAs in clusters of 16"
        )
        assert want in lines[0], lines
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [17, 21, 28, 32])
def test_routed_topk_at_tier_m_on_route_big(tep8_layer, switches, m, caplog):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    from sgl_nvidia_mega.qwen3_5.moe import route

    caplog.set_level(logging.INFO)
    logits, x, w, scale = _inputs(m, "random", 0, 13500 + m)
    body, experts, width = _width(tep8_layer, 0, x)
    experts.w13_input_scale_quant = scale
    experts.dispatcher = SimpleNamespace(
        combine=lambda combine_input: combine_input.hidden_states
    )

    class _Gate:
        weight = w

        def __call__(self, h):
            return torch.nn.functional.linear(h, w), None

    mlp = SimpleNamespace(experts=experts, gate=_Gate())

    def fields(d):
        assert isinstance(d, FlashInferTrtllmDeferredFinalizeOutput)
        return [
            d.gemm2_out.clone(),
            d.expert_weights.clone(),
            d.expanded_idx_to_permuted_idx.clone(),
        ]

    switches.setitem(route._ROUTE_LEAN, "on", True)
    switches.setitem(route._BIG_GEMM, "on", False)
    want = fields(route.routed_topk(mlp, x, pre=True))
    torch.cuda.synchronize()
    assert not [r for r in caplog.records if "slot route_lean runs" in r.getMessage()]
    for _ in range(2):
        _poison(width.tables)
        got = fields(route.routed_topk(mlp, x, pre=True, big=True))
        torch.cuda.synchronize()
        _same_moe(got, want, m)
    lines = [
        r.getMessage()
        for r in caplog.records
        if "slot route_lean runs" in r.getMessage()
    ]
    assert len(lines) == 1 and f"at {m} rows" in lines[0], lines
    # the router GEMM serves these rows only from route_big_mid's floor
    switches.setitem(route._BIG_GEMM, "on", True)
    assert route.big_router_gemm(x, w) is None
    assert route.big_router_gemm(x, w, min_rows=17) is not None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    body._WIDTHS.clear()


def test_route_lean_falls_back_loudly(tep8_layer, switches, caplog):
    from sgl_nvidia_mega.qwen3_5.moe import body, route

    caplog.set_level(logging.INFO)
    dev = torch.cuda.current_device()
    assert route.route_lean(449, LOCAL, dev) is None
    assert not caplog.records
    want = dict(lean=True, et=True, threads=512, ctas=64)
    assert route.route_lean(259, LOCAL, dev) == want
    body.configure(False)
    kw = route.route_lean(112, LOCAL, dev)
    assert kw == dict(lean=True, et=False, threads=512, ctas=None), kw
    caplog.clear()
    assert route.route_lean(112, LOCAL // 2, dev) is None
    assert "not compiled at build" in caplog.text
    switches.setitem(route._ROUTE_LEAN, "built", set())
    assert route.route_lean(112, LOCAL, dev) is None
    switches.setitem(route._ROUTE_LEAN, "on", False)
    assert route.route_lean(112, LOCAL, dev) is None


PROBE = r"""
// FlashInfer's BT publish (VectorFinalizeUnicast) as it reads the routing: threads < 10 of each of a token's CTAs
// load the token's e2p and bf16 weights with plain loads BEFORE griddepcontrol.wait; after the wait this probe
// reloads them from L2 and counts the slots whose early load differs.
extern "C" __global__ void __launch_bounds__(128) q35_lean_probe(
    const int* e2p, const unsigned short* w, int ctas_per_token, int* bad) {
  const int token = blockIdx.x / ctas_per_token, t = threadIdx.x;
  int pe = 0;
  unsigned pw = 0;
  if (t < 10) {
    pe = e2p[token * 10 + t];
    pw = w[token * 10 + t];
  }
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (t < 10) {
    const int qe = __ldcg(e2p + token * 10 + t);
    const unsigned qw = __ldcg(w + token * 10 + t);
    if (qe != pe || qw != pw) atomicAdd(bad, 1);
  }
  asm volatile("griddepcontrol.launch_dependents;");
}
"""


def _probe(device_index):
    from sgl_nvidia_mega.common import jit

    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        PROBE, "q35_lean_probe", f"sm_{major}{minor}a", ("-std=c++17",)
    )
    return jit.load(cubin, "q35_lean_probe", (128, 1, 1), 0)


def _graph_rounds(one, n_inputs, rounds, per_graph=16):
    for i in range(n_inputs):  # eager first: compile, buffers, call state
        one(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(per_graph):
            one(i % n_inputs)
    for _ in range(rounds // per_graph):
        graph.replay()
    torch.cuda.synchronize()


STRESS_ROUNDS = 120_000


@pytest.mark.parametrize("threads", [512, 1024])
@pytest.mark.parametrize("m", [64, 113, 126, 280, 448])
def test_lean_exchange_under_stress(m, threads):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    kinds = ("random", "local", "same", "ties")
    cases = [_inputs(m, k, i % 2 * 3, 13500 + 7 * m + i) for i, k in enumerate(kinds)]
    tabs = [
        router.RouteTables(m, 16, cases[0][1].device, i % 2 * 3 * LOCAL, LOCAL)
        for i in range(len(kinds))
    ]
    want = [_run(router, c[0], c[1], c[3], t) for c, t in zip(cases, tabs)]
    keys = (
        "expanded_to_permuted",
        "permuted_to_token",
        "cta_expert",
        "cta_row_limit",
        "live_ctas",
    )
    bad = torch.zeros(1, dtype=torch.int64, device="cuda")

    def one(i):
        logits, x, _, scale = cases[i]
        t = tabs[i]
        _poison(t)
        router.route_big_quant(
            logits,
            x,
            scale,
            t,
            lean=True,
            et=True,
            threads=threads,
            ctas=None if m <= 128 else 64,
        )
        for k in keys:
            bad.add_((getattr(t, k) != want[i][k]).sum())

    _graph_rounds(one, len(cases), STRESS_ROUNDS)
    print(
        f"m {m} threads {threads}: lean launches {STRESS_ROUNDS}, differing table words {int(bad.item())}"
    )
    assert int(bad.item()) == 0


@pytest.mark.parametrize(
    "opts", [dict(), dict(early_trigger=2)], ids=["default", "et2"]
)
@pytest.mark.parametrize("m", [64, 112, 126, 280, 448])
def test_early_release_hand_off_under_stress(tep8_layer, switches, m, opts):
    from sgl_nvidia_mega.common import jit
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    xs = [_inputs(m, k, 0, 13700 + m + i) for i, k in enumerate(("random", "local"))]
    body, experts, width = _width(tep8_layer, 0, xs[0][1])
    switches.setattr(body, "ENGINE_OPTIONS", opts)
    t = width.tables
    probe = _probe(xs[0][1].device.index)
    bad_probe = torch.zeros(1, dtype=torch.int32, device="cuda")
    bad_rows = torch.zeros(1, dtype=torch.int64, device="cuda")
    refs = []
    for (
        logits,
        x,
        _,
        scale,
    ) in xs:  # route-big's chain: the local slots' FC2 rows and their permuted indices
        _poison(t)
        _, _, x_fp4, x_sf = router.route_big_quant(logits, x, scale, t)
        out = body.run_engine(width, experts, x_fp4, x_sf)
        torch.cuda.synchronize()
        perm = out[2].reshape(-1)[: m * TOP_K].long()
        idx = perm[perm >= 0]
        refs.append(
            (idx.clone(), out[0].index_select(0, idx).view(torch.int16).clone())
        )

    def launch_probe():
        jit.launch(
            probe,
            (m * 4, 1, 1),
            [
                ctypes.c_void_p(t.expanded_to_permuted.data_ptr()),
                ctypes.c_void_p(t.weights.data_ptr()),
                ctypes.c_int(4),
                ctypes.c_void_p(bad_probe.data_ptr()),
            ],
            torch.cuda.current_stream().cuda_stream,
        )

    def one(i, engine=True):
        logits, x, _, scale = xs[i]
        _poison(t)
        _, _, x_fp4, x_sf = router.route_big_quant(  # as route.route_lean serves it
            logits,
            x,
            scale,
            t,
            lean=True,
            et=True,
            threads=512,
            ctas=None if m <= 128 else 64,
        )
        if engine:
            out = body.run_engine(width, experts, x_fp4, x_sf)
            launch_probe()
            idx, rows = refs[i]
            bad_rows.add_(
                (out[0].index_select(0, idx).view(torch.int16) != rows).any(dim=1).sum()
            )
            out[0].fill_(0x7F7F)  # the next round's rows must be rewritten
        else:
            launch_probe()

    bad_probe.zero_()
    _graph_rounds(one, 2, STRESS_ROUNDS)
    shipped_probe, shipped_rows = int(bad_probe.item()), int(bad_rows.item())
    bad_probe.zero_()
    _graph_rounds(lambda i: one(i, engine=False), 2, 4000)
    control = int(bad_probe.item())
    loads = m * TOP_K * 4
    print(
        f"m {m} engine opts {opts or 'default'}: rounds {STRESS_ROUNDS}, FC2 rows that differ {shipped_rows}, "
        f"early loads that differ {shipped_probe} / {STRESS_ROUNDS * loads}; control (probe behind the "
        f"early-released route-big) {control} / {4000 * loads}"
    )
    assert shipped_rows == 0 and shipped_probe == 0
    assert control > 0, (
        "the probe did not see the control's race: it is not sensitive enough"
    )
    body._WIDTHS.clear()
