# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology

FC_INPUT_MODULE = "sgl_nvidia_mega.qwen3_5.draft.fc_input"
SLOTS = ("draft_fc_input",)


def test_the_slot_is_a_numeric_draft_row_of_the_tp8_and_tep8_plans():
    from sgl_nvidia_mega.qwen3_5 import draft
    from sgl_nvidia_mega.qwen3_5.spec import NumericsKind

    for name in SLOTS:
        assert name in runtime.DRAFT_SLOTS
        assert topology.SLOT_TOPOLOGIES[name] == {"tp4", "tp8", "tep8"}
        assert name in runtime.MEASURED_DRAFT
        assert name in runtime.NUMERICS_CHANGE
        assert name not in runtime.MEASURED_BITWISE
        for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
            assert name in runtime.MEASURED_PLAN[key]
        got = draft.numerics(frozenset({"draft_loop", name}))
        assert got.kind == NumericsKind.TOLERANCE


@pytest.mark.parametrize(
    "slots, reason",
    [
        ({"draft_loop", "draft_embed_local", "draft_fc_input"}, None),
        (
            {"draft_loop", "draft_fc_input"},
            "'draft_fc_input' needs slot 'draft_embed_local'",
        ),
    ],
)
def test_each_slot_names_the_slot_it_reads(slots, reason):
    reasons = [r for r in runtime.slot_reasons(frozenset(slots)) if "needs slot" in r]
    if reason is None:
        assert reasons == []
    else:
        assert len(reasons) == 1 and reason in reasons[0]


@pytest.mark.parametrize("table", [True, False])
def test_the_build_drops_a_slot_whose_input_is_missing(monkeypatch, table):
    from sgl_nvidia_mega.qwen3_5.draft import attention
    from sgl_nvidia_mega.qwen3_5.draft.parts import Parts

    model = SimpleNamespace(layers=["decoder"], start_layer=0, end_layer=1)
    draft = SimpleNamespace(
        lm=SimpleNamespace(model=model),
        binding="binding",
        slots=frozenset({"draft_loop", "draft_embed_local", "draft_fc_input"}),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(attention, "Attention", lambda _: None)
    local = SimpleNamespace(table="table")
    monkeypatch.setattr(
        Parts, "_build_embed_local", lambda self: local if table else None
    )
    parts = Parts.build(draft)
    if table:
        assert parts.slots == draft.slots
        assert parts.fc_input_table == "table"
    else:
        assert parts.slots == {"draft_loop"}
        assert parts.fc_input_table is None


@pytest.fixture
def fused(monkeypatch):
    """A stand-in for the fused input module (its Triton kernel needs a GPU)."""
    seen = []

    def mtp_fused_input(ids, table, hidden, w_emb, w_hid, eps_emb, eps_hid, *tp):
        seen.append((ids, table, hidden, w_emb, w_hid, eps_emb, eps_hid, tp))
        return torch.full((hidden.shape[0], 2 * hidden.shape[1]), 3.0)

    module = types.ModuleType(FC_INPUT_MODULE)
    module.mtp_fused_input = mtp_fused_input
    monkeypatch.setitem(sys.modules, FC_INPUT_MODULE, module)
    return seen


def _lm():
    def norm(scale):
        return SimpleNamespace(
            weight=SimpleNamespace(data=scale), variance_epsilon=1e-6
        )

    return SimpleNamespace(
        fc=lambda x: x * 2,
        pre_fc_norm_embedding=norm("w_emb"),
        pre_fc_norm_hidden=norm("w_hid"),
    )


def test_fc_input_takes_the_fused_input_from_the_replicated_table(fused):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    noted = []
    parts = SimpleNamespace(
        lm=_lm(), fc_input_table="table", note_fc_input=noted.append
    )
    tokens = torch.arange(6).view(3, 2)[:, 1]
    hidden = torch.ones(3, 4)
    out = layer.fc_input(parts, tokens, hidden)
    ((ids, table, h, w_emb, w_hid, eps_emb, eps_hid, tp),) = fused
    assert torch.equal(ids, tokens) and table == "table" and h is hidden
    assert (w_emb, w_hid, eps_emb, eps_hid, tp) == ("w_emb", "w_hid", 1e-6, 1e-6, ())
    assert torch.equal(out, torch.full((3, 8), 6.0))
    assert noted == [3]


@pytest.mark.parametrize("table, rows", [(None, 3), ("table", 2)])
def test_fc_input_keeps_the_norms_without_the_table_or_for_other_rows(
    monkeypatch, fused, table, rows
):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    calls = []
    monkeypatch.setattr(
        layer,
        "norms_into_cat",
        lambda embeds, hidden, *_: calls.append(tuple(embeds.shape)) or hidden,
    )
    parts = SimpleNamespace(
        lm=_lm(), fc_input_table=table, embed=lambda ids: torch.zeros(ids.shape[0], 4)
    )
    out = layer.fc_input(parts, torch.arange(3), torch.ones(rows, 4))
    assert calls == [(3, 4)] and fused == []
    assert torch.equal(out, torch.full((rows, 4), 2.0))


def test_fc_input_takes_the_output_of_a_replicated_linear_fc_on_the_fused_input(fused):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    lm = _lm()
    lm.fc = lambda x: (x * 2, None)
    parts = SimpleNamespace(lm=lm, fc_input_table="table", note_fc_input=lambda r: None)
    out = layer.fc_input(parts, torch.arange(3), torch.ones(3, 4))
    assert isinstance(out, torch.Tensor) and len(fused) == 1
    assert torch.equal(out, torch.full((3, 8), 6.0))


class _Mkd:
    def __init__(self, calls):
        self.calls = calls

    def split_fc(self, embeds, hidden):
        self.calls.append("split_fc")
        return hidden


@pytest.mark.parametrize(
    "table, slots, want",
    [
        (None, {"draft_xchg"}, "split_fc"),
        ("table", {"draft_xchg", "draft_fc_input"}, "split_fc"),
        ("table", {"draft_fc_input"}, "fc_input"),
    ],
)
def test_a_draft_step_takes_its_fc_by_slot(monkeypatch, table, slots, want):
    from sgl_nvidia_mega.qwen3_5.draft import layer
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    monkeypatch.setattr(
        layer, "fc_input", lambda parts, token, hidden: calls.append("fc_input")
    )
    parts = SimpleNamespace(
        mkd=_Mkd(calls) if "draft_xchg" in slots else None,
        fc_tp=None,
        fc_input_table=table,
        slots=frozenset(slots),
        embed=lambda ids: ids,
        note_fc_input=lambda rows: None,
        note_engaged=lambda entry, bs: calls.append(entry),
    )
    loop = DraftLoop.__new__(DraftLoop)
    loop.parts = parts
    loop._fc(torch.arange(2), torch.ones(2, 4))
    assert calls == [want]
