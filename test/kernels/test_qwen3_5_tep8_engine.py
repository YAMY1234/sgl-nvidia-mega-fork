# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

E, TOP_K = 512, 10
LOCAL, INTER = 128, 512
ROWS = [
    7,
    14,
    28,
    35,
    56,
    84,
    112,
    224,
]


@pytest.fixture(scope="module")
def tep8_layer():
    return _local_layer(LOCAL, INTER, 17)


def _route(m, kind, rank, tables, seed):
    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    window = (rank * LOCAL, LOCAL)
    x, w, scale = _ep_router_inputs(m, seed, kind, window)
    logits = torch.nn.functional.linear(x, w)
    router.build_topk(x.device.index)
    router.build_big(x.device.index, LOCAL)

    def front():
        if m >= plan.ROUTE_BIG_MIN_ROWS:
            got = router.route_big_quant(logits, x, scale, tables)
        else:
            got = router.route_topk_quant(logits, x, scale, tables=tables)
        return got[2], got[3].view(torch.float8_e4m3fn)

    return x, front


def _case(layer, m, kind, rank, seed):
    from sgl_nvidia_mega.qwen3_5.moe import body

    experts = _experts_of(layer, rank)
    body._WIDTHS.clear()
    body.configure(False)  # body.run is FlashInfer's body, the reference
    probe = torch.empty((m, 4096), dtype=torch.bfloat16, device="cuda")
    width = body.width(experts, probe)
    x, front = _route(m, kind, rank, width.tables, seed)
    return body, experts, width, front


@pytest.mark.parametrize("kind", ["random", "local", "same"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("m", ROWS)
def test_engine_is_the_flashinfer_body(tep8_layer, m, rank, kind):
    body, experts, width, front = _case(tep8_layer, m, kind, rank, 7100 + 13 * m + rank)
    x_fp4, x_sf = front()
    body.configure(False)
    assert not body.engine_serves(m)  # slot off: FlashInfer's body
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]
    got = body.run_engine(width, experts, x_fp4, x_sf)
    torch.cuda.synchronize()
    assert got[1].shape == want[1].shape and got[2].shape == want[2].shape
    _same_moe(got, want, m)
    body._WIDTHS.clear()


OPTS = [dict(early_unit=0), dict(early_unit=1)]


@pytest.mark.parametrize(
    "opts", OPTS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items())
)
@pytest.mark.parametrize("m", [28, 112])
def test_engine_options_keep_the_rows(tep8_layer, m, opts, monkeypatch):
    body, experts, width, front = _case(tep8_layer, m, "random", 0, 7300 + m)
    x_fp4, x_sf = front()
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]
    monkeypatch.setattr(body, "ENGINE_OPTIONS", opts)
    got = body.run_engine(width, experts, x_fp4, x_sf)
    torch.cuda.synchronize()
    _same_moe(got, want, m)
    body._WIDTHS.clear()


RING_LAYER = 5
RING_CALLS = [(1, "random"), (1, "random"), (2, "random"), (3, "same"), (4, "local")]


def _ring_on(monkeypatch, stages, **extra):
    from sgl_nvidia_mega.qwen3_5.moe import body

    # body.run stays FlashInfer's body (the reference); run_engine takes the ring
    body.configure(False)
    monkeypatch.setitem(body._ENGINE, "ring", True)
    monkeypatch.setattr(body, "ENGINE_OPTIONS", dict(extra))
    monkeypatch.setattr(body, "RING_STAGES", stages)
    monkeypatch.setattr(body, "RING_ROWS", (1, 1 << 30))
    monkeypatch.setattr(body, "front_releases_early", lambda experts, m, device: True)
    return body


def _ranked(width, local):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine as eng

    rows = width.tables.tokens_per_expert[:local].tolist()
    tier = lambda r: 0 if r >= 4 else 1 if r >= 2 else 2
    got = sorted((tier(r), e) for e, r in enumerate(rows) if r > 0)
    want = [e for _, e in got][: eng.PRED_MAX]
    return want + [-1] * (eng.PRED_MAX - len(want))


EARLY = dict(early_trigger=2, grid_ctas=136)


@pytest.mark.parametrize(
    "stages,extra",
    [(2, {}), (3, {}), (4, {}), (3, EARLY)],
    ids=["s2", "s3", "s4", "s3-early"],
)
@pytest.mark.parametrize("m", [56, 64, 91, 98, 112, 128])
def test_ring_keeps_the_rows_and_records_the_experts(
    tep8_layer, m, stages, extra, monkeypatch
):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine as eng

    body, experts, width, _ = _case(tep8_layer, m, "random", 0, 1)
    experts.layer_id = RING_LAYER
    _ring_on(monkeypatch, stages, **extra)
    rec = body._ring_records(torch.device("cuda", torch.cuda.current_device()))[
        RING_LAYER
    ]
    rec[: eng.REC_HDR] = 0
    rec[eng.REC_HDR :] = -1
    g0 = 0
    for i, (seed, kind) in enumerate(RING_CALLS):
        _, front = _route(m, kind, 0, width.tables, 9100 + 7 * seed + m)
        x_fp4, x_sf = front()
        want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]
        opts, r = body.ring_opts(experts, m, dict(body.ENGINE_OPTIONS), x_fp4.device)
        assert r is not None and opts == dict(extra, ring=stages, q0=1)
        got = body.run_engine(width, experts, x_fp4, x_sf)
        torch.cuda.synchronize()
        _same_moe(got, want, m)
        g = int(rec[0].item())
        assert g == g0 + i + 1, (i, g)
        lst = rec[eng.REC_HDR + (g & 1) * eng.PRED_MAX :][: eng.PRED_MAX].tolist()
        assert lst == _ranked(width, LOCAL), (i, kind)
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [91, 112])
def test_ring_drops_duplicate_and_out_of_range_predictions(tep8_layer, m, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine as eng

    body, experts, width, front = _case(tep8_layer, m, "random", 0, 9300 + m)
    experts.layer_id = RING_LAYER + 1
    _ring_on(monkeypatch, 2)
    x_fp4, x_sf = front()
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]
    rec = body._ring_records(torch.device("cuda", torch.cuda.current_device()))[
        RING_LAYER + 1
    ]
    bad = [5, 5, 300, -7, 0, 128, 1, 0] + list(range(127, 103, -1))
    for g in (6, 7):  # either parity's list
        rec[0] = g
        rec[eng.REC_HDR + (g & 1) * eng.PRED_MAX :][: eng.PRED_MAX] = torch.tensor(
            bad, dtype=torch.int32
        )
        got = body.run_engine(width, experts, x_fp4, x_sf)
        torch.cuda.synchronize()
        _same_moe(got, want, m)
        assert int(rec[0].item()) == g + 1
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [56, 112])
def test_ring_replays_from_a_cuda_graph(tep8_layer, m, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine as eng

    body, experts, width, front = _case(tep8_layer, m, "random", 3, 9500 + m)
    experts.layer_id = RING_LAYER + 2
    _ring_on(monkeypatch, 2)
    x_fp4, x_sf = front()
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]

    def step():
        a, b = front()
        return body.run_engine(width, experts, a, b)

    step()  # eager first: the engine's buffers, call state and records
    torch.cuda.synchronize()
    rec = body._ring_records(torch.device("cuda", torch.cuda.current_device()))[
        RING_LAYER + 2
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    g = int(rec[0].item())
    for i in range(3):
        captured[0].zero_()
        graph.replay()
        torch.cuda.synchronize()
        _same_moe(captured, want, m)
        assert int(rec[0].item()) == g + i + 1
    assert rec[eng.REC_HDR :][: eng.PRED_MAX].ne(-1).any()
    body._WIDTHS.clear()


@pytest.mark.parametrize("m", [14, 56, 112])
def test_engine_replays_from_a_cuda_graph(tep8_layer, m):
    body, experts, width, front = _case(tep8_layer, m, "random", 3, 7500 + m)
    x_fp4, x_sf = front()
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]

    def step():
        a, b = front()
        return body.run_engine(width, experts, a, b)

    step()  # eager first: the engine's buffers and call state
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(3):
        captured[0].zero_()
        graph.replay()
        torch.cuda.synchronize()
        _same_moe(captured, want, m)
    body._WIDTHS.clear()


def test_the_slot_routes_the_body_through_the_engine(tep8_layer, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.moe import body

    m = 56
    _, experts, width, front = _case(tep8_layer, m, "random", 0, 7700)
    x_fp4, x_sf = front()
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]
    calls = []
    real = body.run_engine
    monkeypatch.setattr(body, "run_engine", lambda *a: calls.append(1) or real(*a))
    body.configure(True)
    got = body.run(width, experts, x_fp4, x_sf)
    torch.cuda.synchronize()
    assert calls == [1]
    _same_moe(got, want, m)
    body.configure(False)
    body.run(width, experts, x_fp4, x_sf)
    assert calls == [1]
    body._WIDTHS.clear()


def test_engine_refuses_more_than_128_local_experts():
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5.moe import body

    w13 = torch.empty((E, 2 * 128, 2048), dtype=torch.uint8, device="cuda")
    w2 = torch.empty((E, 4096, 64), dtype=torch.uint8, device="cuda")
    experts = SimpleNamespace(
        moe_runner_config=SimpleNamespace(activation="silu", is_gated=True),
        w13_weight=w13,
        w2_weight=w2,
        intermediate_size_per_partition=128,
        top_k=TOP_K,
    )
    assert "128 local experts" in body.engine_unsupported(experts)


# ------------------------------------------- slot route_big_gemm's router GEMM


def _router_case(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, 4096, device="cuda", generator=g)).to(torch.bfloat16)
    w = (torch.randn(E, 4096, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize(
    "m", [17, 21, 28, 32, 33, 35, 56, 84, 96, 112, 128, 133, 147, 168, 189, 192]
)
def test_big_router_gemm_holds_its_logits_to_fp64(m, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.moe import route

    route.configure(gemm=True)
    x, w = _router_case(m, 8100 + m)
    got = route.big_router_gemm(x, w, min_rows=17)
    assert got is not None
    stock = torch.nn.functional.linear(x, w)
    ref = x.double() @ w.double().T
    rounded = ref.to(torch.bfloat16)
    got_exact = (got == rounded).float().mean().item()
    stock_exact = (stock == rounded).float().mean().item()
    assert got_exact >= stock_exact - 1e-4, (got_exact, stock_exact)
    got_max = (got.double() - ref).abs().max().item()
    stock_max = (stock.double() - ref).abs().max().item()
    assert got_max <= stock_max + ref.abs().max().item() * 2.0**-8, (
        got_max,
        stock_max,
    )
    head = route.big_router_gemm(x[:17].contiguous(), w, min_rows=17)
    assert torch.equal(head, got[:17])
    got_row = (got.double() - ref).abs().amax(dim=1)
    stock_row = (stock.double() - ref).abs().amax(dim=1)
    rel_l2 = ((got.double() - ref).norm() / ref.norm()).item()
    rel_l2_stock = ((stock.double() - ref).norm() / ref.norm()).item()
    print(
        f"m {m}: correctly rounded {got_exact:.5f} (cuBLAS {stock_exact:.5f}); max abs "
        f"{got_max:.3e} (cuBLAS {stock_max:.3e}); worst row {got_row.max().item():.3e} (cuBLAS "
        f"{stock_row.max().item():.3e}); rel-L2 {rel_l2:.3e} (cuBLAS {rel_l2_stock:.3e}, "
        f"ratio {rel_l2 / rel_l2_stock:.3f})"
    )


ROW_SEEDS = 32


@pytest.mark.parametrize("m", [112, 192])
def test_big_router_gemm_replays_from_a_cuda_graph(m, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.moe import route

    route.configure(gemm=True)
    x, w = _router_case(m, 8300 + m)
    want = route.big_router_gemm(x, w).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = route.big_router_gemm(x, w)
    for _ in range(2):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want)


def test_big_router_gemm_stays_off_by_default_and_outside_its_rows():
    from sgl_nvidia_mega.qwen3_5.moe import route

    x, w = _router_case(56, 8400)
    route.configure()
    assert route.big_router_gemm(x, w) is None
    route.configure(gemm=True)
    assert route.big_router_gemm(x[:32].contiguous(), w) is None
    assert route.big_router_gemm(x[:32].contiguous(), w, min_rows=17) is not None
    assert route.big_router_gemm(x[:16].contiguous(), w, min_rows=17) is None
    x2, _ = _router_case(193, 8401)
    assert route.big_router_gemm(x2, w) is None  # past the wide tiles
    x3, _ = _router_case(129, 8402)
    assert route.big_router_gemm(x3, w) is not None
    route.configure()
