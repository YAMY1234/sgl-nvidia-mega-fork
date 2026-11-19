# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
ROWS = [33, 35, 42, 49, 56, 63, 64]  # 5-9 requests of 7 rows, and the ends


@pytest.fixture(scope="module")
def tep8_layer():
    return _local_layer(LOCAL, INTER, 17)


@pytest.fixture
def switches(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.moe import body, route

    body.configure(True)
    route.configure(gemm=True, engine_route=True)
    monkeypatch.setattr(route, "ENGINE_ROUTE_MAX_ROWS", 64)
    yield monkeypatch
    body.configure(False)
    route.configure()


def _case(layer, m, kind, rank, seed):
    from sgl_nvidia_mega.qwen3_5.moe import body
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    experts = _experts_of(layer, rank)
    body._WIDTHS.clear()
    x, w, scale = _ep_router_inputs(m, seed, kind, (rank * LOCAL, LOCAL))
    width = body.width(experts, x)
    router.build_big(x.device.index, LOCAL)
    mlp = SimpleNamespace(experts=experts, gate=SimpleNamespace(weight=w))
    return body, experts, width, mlp, x, w, scale


def _tables(t):
    live = int(t.live_ctas.item())
    lim = t.cta_row_limit[:live].long().cpu()
    rows = (
        torch.cat([torch.arange(c * t.tile, int(lim[c])) for c in range(live)])
        if live
        else torch.zeros(0)
    )
    return dict(
        ids=t.ids.clone(),
        weights=t.weights.view(torch.int16).clone(),
        e2p=t.expanded_to_permuted.clone(),
        live=live,
        total_padded=int(t.total_padded.item()),
        cta_expert=t.cta_expert[:live].clone(),
        cta_row_limit=t.cta_row_limit[:live].clone(),
        p2t=t.permuted_to_token[rows.long().to(t.permuted_to_token.device)].clone(),
    )


def _poison(t):
    for name in (
        "ids",
        "expanded_to_permuted",
        "permuted_to_token",
        "cta_expert",
        "cta_row_limit",
        "live_ctas",
        "total_padded",
    ):
        getattr(t, name).fill_(-7)
    t.weights.view(torch.int16).fill_(0x7F7F)


def _reference(body, route, experts, width, x, w, scale):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    logits = route.big_router_gemm(x, w)
    assert logits is not None
    ids, weights, x_fp4, x_sf = router.route_big_quant(logits, x, scale, width.tables)
    out = [t.clone() for t in body.run_engine(width, experts, x_fp4, x_sf)[:3]]
    torch.cuda.synchronize()
    return (
        logits.clone(),
        x_fp4.clone(),
        x_sf.view(torch.uint8).clone(),
        _tables(width.tables),
        out,
    )


def _same_tables(got, want):
    for key in want:
        if isinstance(want[key], torch.Tensor):
            assert torch.equal(got[key], want[key]), key
        else:
            assert got[key] == want[key], key


@pytest.mark.parametrize("kind", ["random", "local", "same", "ties"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("m", ROWS)
def test_engine_route_is_the_route_big_chain(tep8_layer, switches, m, rank, kind):
    from sgl_nvidia_mega.qwen3_5.moe import route

    body, experts, width, mlp, x, w, scale = _case(
        tep8_layer, m, kind, rank, 9100 + 17 * m + rank
    )
    logits, x_fp4, x_sf, tables, want = _reference(
        body, route, experts, width, x, w, scale
    )
    got_logits, cand, a_fp4, a_sf = route.router_gemm_a(x, w, scale, True)
    torch.cuda.synchronize()
    assert torch.equal(got_logits.view(torch.int16), logits.view(torch.int16))
    assert torch.equal(a_fp4, x_fp4) and torch.equal(a_sf, x_sf)
    # the whole front + body, on poisoned tables
    _poison(width.tables)
    got = route.engine_route(mlp, x, width, scale)
    torch.cuda.synchronize()
    assert got is not None
    _same_tables(_tables(width.tables), tables)
    _same_moe(got, want, m)
    body._WIDTHS.clear()


@pytest.mark.parametrize("et", [False, True])
@pytest.mark.parametrize(
    "opts",
    [dict()],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default",
)
@pytest.mark.parametrize("m", [35, 56])
def test_engine_route_options_keep_the_bits(tep8_layer, switches, m, opts, et):
    from sgl_nvidia_mega.qwen3_5.moe import route

    body, experts, width, mlp, x, w, scale = _case(tep8_layer, m, "random", 0, 9300 + m)
    switches.setattr(body, "ENGINE_OPTIONS", opts)
    switches.setattr(route, "ENGINE_ROUTE_EARLY_TRIGGER", et)
    _, _, _, tables, want = _reference(body, route, experts, width, x, w, scale)
    _poison(width.tables)
    got = route.engine_route(mlp, x, width, scale)
    torch.cuda.synchronize()
    _same_tables(_tables(width.tables), tables)
    _same_moe(got, want, m)
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [35, 63])
def test_engine_route_replays_from_a_cuda_graph(tep8_layer, switches, m):
    from sgl_nvidia_mega.qwen3_5.moe import route

    body, experts, width, mlp, x, w, scale = _case(tep8_layer, m, "local", 3, 9500 + m)
    _, _, _, tables, want = _reference(body, route, experts, width, x, w, scale)
    route.engine_route(
        mlp, x, width, scale
    )  # eager first: compile, buffers, call state
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = route.engine_route(mlp, x, width, scale)
    for _ in range(3):
        captured[0].zero_()
        _poison(width.tables)
        graph.replay()
        torch.cuda.synchronize()
        _same_tables(_tables(width.tables), tables)
        _same_moe(captured, want, m)
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [35, 63])
def test_routed_topk_returns_the_deferred_output_with_the_engine_route(
    tep8_layer, switches, m
):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    from sgl_nvidia_mega.qwen3_5.moe import route
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    body, experts, width, mlp, x, w, scale = _case(tep8_layer, m, "random", 0, 9800 + m)
    experts.w13_input_scale_quant = scale
    experts.dispatcher = SimpleNamespace(
        combine=lambda combine_input: combine_input.hidden_states
    )

    def fields(d):
        assert isinstance(d, FlashInferTrtllmDeferredFinalizeOutput)
        assert d.top_k == TOP_K
        return [
            d.gemm2_out.clone(),
            d.expert_weights.clone(),
            d.expanded_idx_to_permuted_idx.clone(),
        ]

    switches.setitem(route._ENGINE_ROUTE, "on", False)
    want = fields(route.routed_topk(mlp, x, pre=True))
    torch.cuda.synchronize()
    switches.setitem(route._ENGINE_ROUTE, "on", True)

    def no_route_big(*a, **k):
        raise AssertionError("route-big ran with slot moe_engine_route on")

    switches.setattr(router, "route_big_quant", no_route_big)
    _poison(width.tables)
    got = fields(route.routed_topk(mlp, x, pre=True))
    torch.cuda.synchronize()
    _same_moe(got, want, m)
    body._WIDTHS.clear()


def test_engine_route_p2tp_keeps_smaller_buffers_alive(switches):
    """A larger width's per-CTA copy does not free the smaller one a CUDA graph may hold."""
    from sgl_nvidia_mega.qwen3_5.moe import body

    switches.setattr(body, "_ROUTE_P2TP", [])
    dev = torch.device("cuda")
    grid = torch.cuda.get_device_properties(dev).multi_processor_count
    small = body._route_p2tp(SimpleNamespace(cta_expert=torch.empty(150), tile=8), dev)
    big = body._route_p2tp(SimpleNamespace(cta_expert=torch.empty(159), tile=16), dev)
    again = body._route_p2tp(SimpleNamespace(cta_expert=torch.empty(150), tile=8), dev)
    assert big.data_ptr() != small.data_ptr() and again.data_ptr() == big.data_ptr()
    assert [t.numel() for t in body._ROUTE_P2TP] == [grid * 151 * 8, grid * 160 * 16]


def test_engine_route_alternates_with_route_big_on_one_engine_state(
    tep8_layer, switches
):
    from sgl_nvidia_mega.qwen3_5.moe import route

    body, experts, width, mlp, x, w, scale = _case(tep8_layer, 56, "random", 0, 9600)
    _, _, _, tables, want = _reference(body, route, experts, width, x, w, scale)
    for _ in range(3):
        _poison(width.tables)
        got = route.engine_route(mlp, x, width, scale)
        torch.cuda.synchronize()
        _same_tables(_tables(width.tables), tables)
        _same_moe(got, want, 56)
        _, _, _, _, again = _reference(body, route, experts, width, x, w, scale)
        _same_moe(again, want, 56)
    body._WIDTHS.clear()


def test_engine_route_falls_back_loudly(tep8_layer, switches, caplog):
    from sgl_nvidia_mega.qwen3_5.moe import route

    body, experts, width, mlp, x, w, scale = _case(tep8_layer, 56, "random", 0, 9700)
    caplog.set_level(logging.WARNING)
    caplog.clear()
    switches.setitem(route._ENGINE_ROUTE, "on", False)
    assert route.engine_route(mlp, x, width, scale) is None  # off: silent
    switches.setitem(route._ENGINE_ROUTE, "on", True)
    assert (
        route.engine_route(mlp, x[:32].contiguous(), width, scale) is None
    )  # outside its rows
    assert not caplog.records
    assert route.engine_route(mlp, x, width, scale.double()) is None
    assert (
        "falls back to route-big at 56 rows" in caplog.text
        and "operands" in caplog.text
    )
    body._WIDTHS.clear()


PROBE = r"""
// FlashInfer's BT publish (VectorFinalizeUnicast) as it reads the routing: threads < 10 of each of a token's CTAs
// load the token's e2p and bf16 weights with plain loads BEFORE griddepcontrol.wait; after the wait this probe
// reloads them from L2 and counts the slots whose early load differs.
extern "C" __global__ void __launch_bounds__(128) q35_engine_route_probe(
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
        PROBE, "q35_engine_route_probe", f"sm_{major}{minor}a", ("-std=c++17",)
    )
    return jit.load(cubin, "q35_engine_route_probe", (128, 1, 1), 0)


def _stress(
    route, body, mlp, width, experts, xs, scale, probe, bad, rounds, engine_opts=None
):
    import ctypes

    from sgl_nvidia_mega.common import jit

    t = width.tables
    m = xs[0].shape[0]

    def one(x):
        _poison(t)
        if engine_opts is None:
            out = route.engine_route(mlp, x, width, scale)
        else:
            _, cand, x_fp4, x_sf = route.router_gemm_a(x, mlp.gate.weight, scale, True)
            out = body.run_engine_a(width, experts, x_fp4, x_sf, cand)
        jit.launch(
            probe,
            (m * 4, 1, 1),
            [
                ctypes.c_void_p(t.expanded_to_permuted.data_ptr()),
                ctypes.c_void_p(t.weights.data_ptr()),
                ctypes.c_int(4),
                ctypes.c_void_p(bad.data_ptr()),
            ],
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    for x in xs:  # eager first: compile, buffers, call state
        one(x)
    torch.cuda.synchronize()
    bad.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(8):
            one(xs[i % 2])
    for _ in range(rounds // 8):
        graph.replay()
    torch.cuda.synchronize()
    return int(bad.item())


@pytest.mark.parametrize("m", [35, 56, 63])
def test_engine_route_finalize_hand_off_under_stress(tep8_layer, switches, m):
    from sgl_nvidia_mega.qwen3_5.moe import route

    body, experts, width, mlp, x, w, scale = _case(tep8_layer, m, "random", 0, 9900 + m)
    x2, _, _ = _ep_router_inputs(m, 9950 + m, "local", (0, LOCAL))
    probe = _probe(x.device.index)
    bad = torch.zeros(1, dtype=torch.int32, device="cuda")
    shipped = _stress(
        route, body, mlp, width, experts, (x, x2), scale, probe, bad, 4000
    )
    switches.setattr(body, "ENGINE_OPTIONS", dict(early_trigger=2))
    control = _stress(
        route,
        body,
        mlp,
        width,
        experts,
        (x, x2),
        scale,
        probe,
        bad,
        400,
        engine_opts={},
    )
    loads = m * 10 * 4
    print(
        f"m {m}: early loads that differ: shipped {shipped} / {4000 * loads}, control (early_trigger 2) "
        f"{control} / {400 * loads}"
    )
    assert shipped == 0
    assert control > 0, (
        "the probe did not see the control's race: it is not sensitive enough"
    )
    body._WIDTHS.clear()
