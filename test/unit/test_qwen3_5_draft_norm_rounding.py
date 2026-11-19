# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology

HIDDEN = 4096
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


# ------------------------------------------------------------------- DraftHop


class _Norm:
    """A GemmaRMSNorm whose folded gamma no draft hop norm may read."""

    def __init__(self, weight):
        self.weight = weight
        self.variance_epsilon = 1e-6

    @property
    def gemma_weight(self):
        raise AssertionError("a draft norm read the folded bf16 gemma_weight")


def _draft_hop(monkeypatch, seen):
    from sgl_nvidia_mega.qwen3_5.draft import hop as hop_mod

    def fake_ar(hop, **kw):
        seen.append(("ar", kw))
        return kw["residual"], kw["residual"]

    def fake_fin(hop, **kw):
        seen.append(("fin", kw))
        return kw["residual"], kw["residual"]

    monkeypatch.setattr(hop_mod.kernels, "draft_all_reduce_rms_norm", fake_ar)
    monkeypatch.setattr(hop_mod.kernels, "draft_finalize_all_reduce_rms_norm", fake_fin)
    return hop_mod.DraftHop(SimpleNamespace())


def _owed(x):
    return SimpleNamespace(
        routed_output=x, expert_weights=x, permuted_indices=x, gated_shared_output=None
    )


def test_every_draft_hop_norm_takes_the_raw_weight(monkeypatch):
    seen = []
    dh = _draft_hop(monkeypatch, seen)
    norm = _Norm(torch.zeros(HIDDEN, dtype=torch.bfloat16))
    x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    dh.ffn_input(x, x, norm, moe_single_launch=False)
    dh.ffn_input(x, x, norm, moe_single_launch=True)
    dh.final_norm(x, x, norm)
    dh.fin_norm(_owed(x), x, norm)
    assert [k for k, _ in seen] == ["ar"] * 3 + ["fin"]
    for _, kw in seen:
        assert kw["gamma"] is norm.weight
        assert kw["weight_bias"] == 1.0
        assert kw["eps"] == norm.variance_epsilon
        assert "add_norm" not in kw
    # the FFN input's releases are unchanged: early when a single-launch MoE is next
    assert [kw["early_release"] for _, kw in seen[:2]] == [False, True]
    fin = seen[-1][1]
    assert fin["routed_output"] is x and fin["gated_shared_output"] is None
    # the decode hop's consume, which fin_norm stands in for, releases at entry too
    assert fin["early_release"] is True


@pytest.mark.parametrize(
    "switch, slots, key",
    [
        ("1", None, ("tep8", topology.GB300)),
        ("1", None, ("tp8", topology.GB300)),
        (None, None, ("tep8", topology.GB300)),
        ("0", None, ("tp8", topology.GB300)),
        ("1", "draft_hop,draft_fin", ("tep8", topology.GB300)),
        ("1", None, ("tp4", topology.GB300)),
        ("1", None, ("tep8", topology.Sku(10, 3, 148))),
        ("1", None, None),
    ],
)
def test_every_setup_builds_the_draft_hop_in_the_stock_rounding(
    monkeypatch, switch, slots, key
):
    from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop
    from sgl_nvidia_mega.qwen3_5.draft import hop as hop_mod
    from sgl_nvidia_mega.qwen3_5.draft.parts import Parts

    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "MEASURED_KEY", key)
    target = SimpleNamespace(
        world=4, max_m=7, supports=lambda m: m <= 7, device=torch.device("cpu")
    )
    monkeypatch.setattr(boundary_hop, "active", lambda: target)
    monkeypatch.setattr(hop_mod.kernels, "build", lambda *args, **kwargs: None)
    seen = []
    _draft_hop(monkeypatch, seen)  # records the kernels' launches
    parts = SimpleNamespace(
        binding=SimpleNamespace(tp_size=4),
        draft=SimpleNamespace(deadline_s=30.0),
        slots=frozenset({"draft_hop"}),
    )
    hop = Parts._build_hop(parts)
    assert hop.hop is target
    norm = _Norm(torch.zeros(HIDDEN, dtype=torch.bfloat16))
    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    hop.ffn_input(x, x, norm, moe_single_launch=True)
    ((_, kw),) = seen
    assert kw["gamma"] is norm.weight and kw["weight_bias"] == 1.0


def test_fin_norm_refuses_an_unconsumed_push(monkeypatch):
    seen = []
    dh = _draft_hop(monkeypatch, seen)
    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    dh.mark_pushed(x)
    with pytest.raises(RuntimeError, match="never consumed"):
        dh.fin_norm(_owed(x), x, _Norm(x[0]))
    assert seen == []


# ------------------------------------------------------------------- the launches


class _Hop:
    """The decode hop's surface the draft kernels read, on the CPU."""

    def __init__(self, world=8, max_m=7):
        from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop

        self.world, self.max_m = world, max_m
        self.device = torch.device("cpu")
        self.site = decode_hop.Site(0, 0, 0, 0, world, max_m)
        self.kernels = {
            n: f"decode:{n}"
            for n in (
                "q35hop_publish",
                "q35hop_publish_fin",
                "q35hop_publish_fin_top10",
            )
        }
        self.consumed_checks = 0

    def supports(self, m):
        return 1 <= m <= self.max_m

    def _check_consumed(self):
        self.consumed_checks += 1


@pytest.fixture
def launches(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    seen = []

    def launch(kernel, grid, arguments, stream, cluster=None):
        (args,) = arguments
        seen.append((kernel, tuple(grid), args, cluster))

    monkeypatch.setattr(kernels.jit, "launch", launch)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=0),
    )
    names = [
        kernels.consume_name(w, early) for w in (2, 4, 8) for early in (False, True)
    ]
    monkeypatch.setitem(kernels._CONSUME, None, {n: f"draft:{n}" for n in names})
    return seen


def _finalize_operands(m, top_k, *, shared=True, rows=None):
    g = torch.Generator().manual_seed(m * 100 + top_k)
    if rows is None:
        rows = torch.randperm(m * top_k + 1, generator=g)[: m * top_k].int()
        rows = rows.view(m, top_k)
    return dict(
        routed_output=torch.randn(m * top_k + 1, HIDDEN, generator=g).bfloat16(),
        expert_weights=torch.rand(m, top_k, generator=g).bfloat16(),
        permuted_indices=rows,
        gated_shared_output=(
            torch.randn(m, HIDDEN, generator=g).bfloat16() if shared else None
        ),
    )


@pytest.mark.parametrize(
    "m, top_k, shared, early",
    [(1, 10, True, True), (7, 8, False, False), (2, 10, False, True)],
)
def test_the_finalize_publishes_with_the_decode_hop_then_consumes_add_norm(
    launches, m, top_k, shared, early
):
    from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    hop = _Hop()
    owed = _finalize_operands(m, top_k, shared=shared)
    residual = torch.randn(m, HIDDEN).bfloat16()
    weight = torch.randn(HIDDEN).bfloat16()
    norm, res_out = kernels.draft_finalize_all_reduce_rms_norm(
        hop,
        **owed,
        residual=residual,
        gamma=weight,
        eps=1e-6,
        weight_bias=1.0,
        early_release=early,
    )
    assert hop.consumed_checks == 1
    assert norm.shape == res_out.shape == (m, HIDDEN) and norm.dtype == torch.bfloat16
    (pk, pgrid, pargs, pcluster), (ck, cgrid, cargs, ccluster) = launches
    assert pk == f"decode:q35hop_publish_fin{'_top10' if top_k == 10 else ''}"
    assert pgrid == (m, decode_hop.QUAD_CTAS, 1) and pcluster is None
    assert isinstance(pargs, decode_hop.Args)
    assert pargs.routed == owed["routed_output"].data_ptr()
    assert pargs.expert_w == owed["expert_weights"].data_ptr()
    assert pargs.permuted == owed["permuted_indices"].data_ptr()
    want_shared = owed["gated_shared_output"].data_ptr() if shared else None
    assert (pargs.shared or None) == want_shared
    assert (pargs.m, pargs.top_k) == (m, top_k)
    assert ck == f"draft:{kernels.consume_name(8, early)}"
    assert cgrid == (m, decode_hop.ROW_CTAS, 1)
    assert ccluster == (1, decode_hop.ROW_CTAS, 1)
    assert isinstance(cargs, kernels.Args)
    assert cargs.gamma == weight.data_ptr() and cargs.weight_bias == 1.0
    assert cargs.residual_in == residual.data_ptr()
    assert (cargs.norm_out, cargs.residual_out) == (norm.data_ptr(), res_out.data_ptr())
    assert cargs.m == m and abs(cargs.eps - 1e-6) < 1e-12


def test_the_finalize_elides_a_registered_identity_row_map(launches, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    rows = torch.arange(64, dtype=torch.int32)
    monkeypatch.setitem(decode_hop._IDENTITY_ROWS, rows.data_ptr(), rows.numel())
    owed = _finalize_operands(3, 10, rows=rows[:30].view(3, 10))
    kernels.draft_finalize_all_reduce_rms_norm(
        _Hop(),
        **owed,
        residual=torch.zeros(3, HIDDEN, dtype=torch.bfloat16),
        gamma=torch.zeros(HIDDEN, dtype=torch.bfloat16),
        eps=1e-6,
        weight_bias=1.0,
    )
    assert launches[0][2].permuted is None


@pytest.mark.parametrize(
    "bad",
    [
        dict(routed_output=torch.zeros(9, HIDDEN)),
        dict(permuted_indices=torch.zeros(1, 8, dtype=torch.int64)),
        dict(gated_shared_output=torch.zeros(1, HIDDEN)),
        dict(permuted_indices=torch.zeros(1, 17, dtype=torch.int32)),
        dict(permuted_indices=torch.zeros(8, 8, dtype=torch.int32)),
    ],
)
def test_the_finalize_refuses_before_any_launch(launches, bad):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    owed = _finalize_operands(1, 8)
    owed["expert_weights"] = torch.zeros(
        owed["permuted_indices"].shape, dtype=torch.bfloat16
    )
    owed.update(bad)
    m = int(owed["permuted_indices"].shape[0])
    with pytest.raises(ValueError):
        kernels.draft_finalize_all_reduce_rms_norm(
            _Hop(),
            **owed,
            residual=torch.zeros(m, HIDDEN, dtype=torch.bfloat16),
            gamma=torch.zeros(HIDDEN, dtype=torch.bfloat16),
            eps=1e-6,
            weight_bias=1.0,
        )
    assert launches == []


def test_the_all_reduce_launches_the_add_norm_consume(launches):
    from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    hop = _Hop(world=4)
    part = torch.randn(2, HIDDEN).bfloat16()
    res = torch.randn(2, HIDDEN).bfloat16()
    gam = torch.randn(HIDDEN).bfloat16()
    kernels.draft_all_reduce_rms_norm(
        hop, local_contribution=part, residual=res, gamma=gam, eps=1e-5, weight_bias=1.0
    )
    kernels.draft_all_reduce_rms_norm(
        hop,
        local_contribution=part,
        residual=res,
        gamma=gam,
        eps=1e-5,
        weight_bias=1.0,
        early_release=True,
        delay_ns=7,
        pushed=True,
    )
    cluster = (1, decode_hop.ROW_CTAS, 1)
    assert [(k, g, c) for k, g, _, c in launches] == [
        ("decode:q35hop_publish", (2, decode_hop.ROW_CTAS, 1), None),
        ("draft:q35hop_draft_add_norm_w4", (2, decode_hop.ROW_CTAS, 1), cluster),
        ("draft:q35hop_draft_add_norm_early_w4", (2, decode_hop.ROW_CTAS, 1), cluster),
    ]
    a0, a1 = launches[1][2], launches[2][2]
    assert launches[0][2].partial == part.data_ptr() and launches[0][2].m == 2
    assert (a0.weight_bias, a0.delay_ns) == (1.0, 0)
    assert (a0.gamma, a0.residual_in) == (gam.data_ptr(), res.data_ptr())
    assert (a1.weight_bias, a1.delay_ns) == (1.0, 7)


def test_the_draft_hop_builds_only_add_norm_consumes():
    from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    source = kernels.SOURCE.read_text()
    entries = re.findall(r"^Q35HOP_DRAFT_(?:COLS_)?ENTRY\((\w+),", source, re.M)
    assert sorted(entries) == sorted(
        kernels.consume_name(w, early, cols)
        for w in decode_hop.WORLDS
        for cols in (False, True)
        for early in (False, True)
    )
    assert all(name.startswith("q35hop_draft_add_norm") for name in entries)
    assert "kDraftMnnvl" not in source


class _GemmaNorm:
    """A GemmaRMSNorm stub: (h + residual) * scale, and its sum as the residual."""

    def __init__(self, scale, log, name):
        self.scale, self.log, self.name = scale, log, name

    def __call__(self, h, residual):
        self.log.append(self.name)
        return (h + residual) * self.scale, h + residual


@pytest.mark.parametrize(
    "rows, takes", [(1, True), (703, True), (704, False), (1025, False)]
)
def test_the_workspace_serves_only_its_rank_order_widths(monkeypatch, rows, takes):
    import sglang.srt.layers.flashinfer_mnnvl_cutedsl as cutedsl
    import sglang.srt.runtime_context as runtime_context

    from sgl_nvidia_mega.qwen3_5.draft import layer

    assert layer.STOCK_SUM_MAX_ROWS == 703
    workspace = SimpleNamespace(supports=lambda m: m <= 2048)
    monkeypatch.setattr(cutedsl, "_WORKSPACE", workspace, raising=False)
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=8, moe_ep_size=4),
    )
    parts = SimpleNamespace(slots=frozenset({"draft_hop", "draft_fin"}))
    assert (layer._mnnvl_workspace(parts, rows) is workspace) is takes


def test_past_the_rank_order_widths_the_draft_runs_sglangs_all_reduce(
    monkeypatch,
):
    import sglang.srt.layers.flashinfer_mnnvl_cutedsl as cutedsl
    import sglang.srt.runtime_context as runtime_context

    from sgl_nvidia_mega.qwen3_5.draft import layer

    rows, log = layer.STOCK_SUM_MAX_ROWS + 1, []

    def workspace_op(**_):
        raise AssertionError("the HT route's widths take SGLang's all-reduce")

    monkeypatch.setattr(
        cutedsl,
        "_WORKSPACE",
        SimpleNamespace(
            supports=lambda m: True,
            all_reduce_residual_rms_norm=workspace_op,
            moe_finalize_all_reduce_rms_norm=workspace_op,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=8, moe_ep_size=4),
    )

    def all_reduce(t):
        log.append("all_reduce")
        return t * 2

    monkeypatch.setattr(
        "sglang.srt.distributed.tensor_model_parallel_all_reduce", all_reduce
    )

    def mlp(hidden, defer_finalize=False):
        assert not defer_finalize
        return 3 * hidden

    mlp.supports_deferred_finalize = True
    x = torch.ones(rows, 4)
    parts = SimpleNamespace(
        layer=SimpleNamespace(
            input_layernorm=lambda hidden: hidden,
            post_attention_layernorm=_GemmaNorm(10.0, log, "post norm"),
            mlp=mlp,
        ),
        attention=lambda hidden, *args, rows_per_request, o_proj: hidden,
        hop=None,
        moe=None,
        moe_mkm=None,
        moe_block=None,
        oproj_push=None,
        slots=frozenset({"draft_hop", "draft_fin"}),
        model=SimpleNamespace(norm=_GemmaNorm(100.0, log, "final norm")),
    )
    out = layer.layer(parts, x, None, None, None, None, rows_per_request=1)
    assert log == ["all_reduce", "post norm", "all_reduce", "final norm"]
    moe_in, residual = (2 * x + x) * 10, 2 * x + x
    assert torch.equal(out, (2 * (3 * moe_in) + residual) * 100)
