# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fast gate top-k for the DSpark draft router (128 experts, top-3) on CPU: the
routing it reproduces, which kernel variant each router loads, and the draft gate's
own off switch (``draft_gate`` in SGLANG_NVIDIA_MEGA_SLOTS)."""

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import forward
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

TARGET, DRAFT = (384, 6), (128, 3)


def _reference(logits, bias, top_k, scale, num_valid=None):
    """moe_fused_gate's routing: sqrtsoftplus via log1p (identity above 20), NaN ranked
    first, the lowest id on ties, renormalized in the Triton sum order, then scaled."""
    act = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt()
    biased = act + bias
    biased = torch.where(biased.isnan(), torch.full_like(biased, float("inf")), biased)
    ids = biased.sort(dim=-1, descending=True, stable=True).indices[:, :top_k]
    w = act.gather(1, ids)
    if top_k == 3:
        total = (w[:, 0] + w[:, 2]) + w[:, 1]
    else:
        total = ((w[:, 0] + w[:, 2]) + (w[:, 1] + w[:, 3])) + (w[:, 4] + w[:, 5])
    total = torch.where(total > 0, total, torch.ones_like(total))
    weights = w / total[:, None] * scale
    ids = ids.to(torch.int32)
    if num_valid is not None:
        weights[num_valid:], ids[num_valid:] = 0.0, -1
    return weights, ids


def test_the_draft_routing_ranks_by_the_biased_score_and_weighs_the_unbiased():
    gen = torch.Generator().manual_seed(5)
    for rows in [*range(1, 17), 48]:
        logits = torch.randn(rows, 128, generator=gen) * 2
        bias = torch.randn(128, generator=gen) * 0.05
        weights, ids = _reference(logits, bias, 3, 1.5)
        act = torch.nn.functional.softplus(logits).sqrt()
        want = torch.topk(act + bias, 3, dim=-1).indices
        assert torch.equal(ids, want.to(torch.int32))
        picked = act.gather(1, want)
        torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True) * 1.5)


def test_the_draft_routing_breaks_ties_low_ranks_nans_first_and_pads():
    logits = torch.zeros(4, 128)
    logits[1, 77] = float("nan")
    logits[2, 100] = 25.0  # above 20 softplus passes the logit through
    weights, ids = _reference(logits, torch.zeros(128), 3, 1.0, num_valid=3)
    assert ids[0].tolist() == [0, 1, 2]
    assert ids[1].tolist() == [77, 0, 1] and weights[1, 0].isnan()
    assert ids[2].tolist() == [100, 0, 1]
    assert ids[3].tolist() == [-1, -1, -1] and weights[3].eq(0).all()


def test_each_router_has_its_own_program_and_symbol():
    assert set(reg.GATE_TOPK_VARIANTS) == {TARGET, DRAFT}
    for (experts, top_k), (name, symbol) in reg.GATE_TOPK_VARIANTS.items():
        source = (reg.SOURCE_DIR / f"{name}.cu").read_text()
        assert '#include "gate_topk.cuh"' in source
        assert f"__launch_bounds__({experts}) void {symbol}(" in source
        assert f"GateTopK<{experts}, {top_k}>(" in source
    for header in reg.GATE_TOPK_HEADERS:
        assert (reg.SOURCE_DIR / header).is_file()


@pytest.mark.parametrize(
    "args,program,symbol,block",
    [
        ((), "gate_topk", "dsv41_gate_topk", 384),
        (DRAFT, "gate_topk_draft", "dsv41_gate_topk_e128_k3", 128),
    ],
)
def test_the_loader_compiles_the_routers_variant(monkeypatch, args, program, symbol, block):  # fmt: skip
    seen = {}
    cuda = SimpleNamespace(
        device=lambda device: contextlib.nullcontext(),
        get_device_capability=lambda device: (10, 3),
    )
    stub = SimpleNamespace(device=lambda *a: a, zeros=lambda *a, **k: None, cuda=cuda)
    monkeypatch.setattr(reg, "torch", stub)

    def compile_source(source, name, arch, options, headers=None):
        seen.update(name=name, arch=arch, options=options, headers=sorted(headers))
        return b"cubin"

    monkeypatch.setattr(reg, "compile_source", compile_source)
    monkeypatch.setattr(reg, "_load", lambda cubin, sym, blk, smem: (sym, blk))
    assert reg.load_gate_topk.__wrapped__(0, *args) == (symbol, (block, 1, 1))
    assert seen == dict(
        name=program,
        arch="sm_103a",
        options=reg.GATE_TOPK_NVRTC_OPTIONS,
        headers=["gate_topk.cuh"],
    )
    with pytest.raises(KeyError):
        reg.load_gate_topk.__wrapped__(0, 128, 6)


@pytest.fixture
def loaded(monkeypatch):
    """RoutedExpertGemm on CPU stand-ins; records the (experts, top_k) gates loaded."""
    seen = []
    monkeypatch.setattr(reg, "check_weights", lambda *args: None)
    monkeypatch.setattr(reg, "load_kernels", lambda index: {})
    monkeypatch.setattr(reg, "load_route_wide", lambda index: None)
    monkeypatch.setattr(reg, "_weight_map", lambda *args: None)
    monkeypatch.setattr(reg, "_scale_map", lambda *args: None)

    def load_gate_topk(index, experts, top_k):
        seen.append((experts, top_k))
        return object()

    monkeypatch.setattr(reg, "load_gate_topk", load_gate_topk)
    return seen


def _gemm(experts, top_k, gate_topk):
    w = torch.zeros((experts, 1), dtype=torch.uint8)
    return reg.RoutedExpertGemm(
        w, w, w, w, None, top_k=top_k, max_rows=48, gate_topk=gate_topk
    )


def test_the_gemm_loads_the_variant_of_its_router(loaded):
    assert _gemm(*DRAFT, gate_topk=True).gate_topk_kernel is not None
    assert _gemm(*TARGET, gate_topk=True).gate_topk_kernel is not None
    assert loaded == [DRAFT, TARGET]
    assert _gemm(*DRAFT, gate_topk=False).gate_topk_kernel is None
    assert _gemm(128, 6, gate_topk=True).gate_topk_kernel is None
    assert loaded == [DRAFT, TARGET]


def test_the_gate_takes_logits_of_its_routers_width(loaded):
    draft, target = _gemm(*DRAFT, gate_topk=True), _gemm(*TARGET, gate_topk=True)
    bias = torch.zeros(128)
    for rows in (1, 12, 16, 48):
        assert draft.supports_gate_topk(torch.zeros(rows, 128), bias)
        assert not draft.supports_gate_topk(torch.zeros(rows, 384), bias)
        assert not target.supports_gate_topk(torch.zeros(rows, 128), bias)
    assert not draft.supports_gate_topk(torch.zeros(6, 128, dtype=torch.bfloat16), bias)


def _mlp(**config):
    """A DSpark draft stage's MoE as the slot reads it (the target's routing, 128 x 3)."""
    topk_config = SimpleNamespace(
        scoring_func="sqrtsoftplus",
        sqrtsoftplus_log1p=True,
        renormalize=True,
        num_fused_shared_experts=0,
        use_grouped_topk=False,
        custom_routing_function=None,
        correction_bias=torch.zeros(128),
    )
    vars(topk_config).update(config)
    return SimpleNamespace(
        alt_stream=object(),
        num_fused_shared_experts=0,
        is_hash=False,
        gate=SimpleNamespace(e_score_correction_bias_vl=None),
        topk=SimpleNamespace(topk_config=topk_config),
    )


def test_the_draft_router_is_one_the_gate_reproduces():
    assert forward._gate_applies(_mlp())
    assert not forward._gate_applies(_mlp(use_grouped_topk=True))
    assert not forward._gate_applies(_mlp(renormalize=False))
    assert not forward._gate_applies(_mlp(sqrtsoftplus_log1p=False))


@pytest.mark.parametrize(
    "slots,target,draft",
    [
        ({"moe", "gate", "draft_gate"}, True, True),
        ({"moe", "gate"}, True, False),
        ({"moe", "draft_gate"}, False, True),
    ],
)
def test_the_draft_gate_has_its_own_off_switch(monkeypatch, slots, target, draft):
    monkeypatch.setattr(forward, "SLOTS", frozenset(slots))
    monkeypatch.setattr(forward, "MoeSlot", lambda mlp, gemm, *, gate: gate)
    runner = SimpleNamespace(moe={0: SimpleNamespace(gate_topk_kernel=object())})
    layers = [(0, SimpleNamespace(layer_id=0, mlp=_mlp()))]
    assert forward._moe_slots(runner, layers) == {0: target}
    assert forward._moe_slots(runner, layers, gate_slot="draft_gate") == {0: draft}


def test_the_draft_gate_is_on_by_default():
    assert "draft_gate" in forward.SLOTS


@pytest.mark.parametrize("on", [True, False])
def test_the_draft_build_compiles_its_gate_only_when_on(monkeypatch, on):
    seen = {}
    group = SimpleNamespace(rank_in_group=0)
    parallel = SimpleNamespace(get_parallel=lambda: SimpleNamespace(tp_group=group))
    runner = SimpleNamespace(Runner=lambda **kw: SimpleNamespace(moe={}))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", parallel)
    monkeypatch.setitem(
        sys.modules, "sgl_nvidia_mega.deepseek_v41.sm103.runner", runner
    )
    slots = {"moe", "gate"} | ({"draft_gate"} if on else set())
    monkeypatch.setattr(forward, "SLOTS", frozenset(slots))
    monkeypatch.setattr(forward, "moe_seam", lambda: "")

    def build_moe(runner, layers, *, gate_topk, group):
        seen["gate_topk"] = gate_topk
        return ""

    def moe_slots(runner, layers, key, gate_slot):
        seen["gate_slot"] = gate_slot
        return {}

    monkeypatch.setattr(forward, "_build_moe", build_moe)
    monkeypatch.setattr(forward, "_moe_slots", moe_slots)
    model = SimpleNamespace(stages=[None], parameters=lambda: iter([torch.zeros(1)]))
    draft = forward.build_draft(model, None, SimpleNamespace(widths=(1, 12, 24, 48)))
    assert seen == {"gate_topk": on, "gate_slot": "draft_gate"}
    assert [w for w, s in draft.slots.items() if s.fast_gate] == [1, 12, 48]
