# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology

ROWS_MODULE = "sgl_nvidia_mega.qwen3_5.sm103.draft.rows"


def test_draft_select_is_a_bitwise_draft_slot_at_tp4_tp8_and_tep8():
    assert "draft_select" in runtime.DRAFT_SLOTS
    assert topology.SLOT_TOPOLOGIES["draft_select"] == {"tp4", "tp8", "tep8"}
    assert "draft_select" in runtime.MEASURED_DRAFT
    assert "draft_select" in runtime.MEASURED_BITWISE
    assert "draft_select" not in runtime.NUMERICS_CHANGE
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert "draft_select" in runtime.MEASURED_PLAN[key]


def test_draft_select_adds_no_numerics_claim():
    from sgl_nvidia_mega.qwen3_5 import draft

    base = frozenset({"draft_loop", "draft_extend"})
    assert draft.numerics(base | {"draft_select"}) == draft.numerics(base)


@pytest.mark.parametrize(
    "slots, reason",
    [
        ({"draft_extend", "draft_select"}, None),
        (
            {"draft_loop", "draft_select"},
            "'draft_select' runs inside entry 'draft_extend'",
        ),
    ],
)
def test_draft_select_runs_inside_the_draft_extend(slots, reason):
    reasons = [r for r in runtime.slot_reasons(frozenset(slots)) if "draft_select" in r]
    if reason is None:
        assert reasons == []
    else:
        assert len(reasons) == 1 and reason in reasons[0]


@pytest.mark.parametrize("switch, measured", [("1", True), (None, False)])
def test_the_measured_stack_requests_draft_select(monkeypatch, switch, measured):
    if switch is None:
        monkeypatch.delenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", switch)
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    assert ("draft_select" in runtime._requested_slots()) is measured


# ------------------------------------------------------------------- the extend


@pytest.fixture
def copies(monkeypatch):
    """A stand-in for the row-copy module (its Triton kernel needs a GPU)."""
    seen = []

    def copy_rows(src, index, out):
        seen.append((src, index, out))
        return out.copy_(src.index_select(0, index))

    module = types.ModuleType(ROWS_MODULE)
    module.copy_rows = copy_rows
    monkeypatch.setitem(sys.modules, ROWS_MODULE, module)
    return seen


def _parts(slots):
    engaged = []
    parts = SimpleNamespace(
        slots=frozenset(slots),
        note_engaged=lambda entry, bs: engaged.append((entry, bs)),
    )
    return parts, engaged


def _io(rows=14, hidden=8, selected=2, out=True, dtype=torch.bfloat16):
    return SimpleNamespace(
        select_index=torch.tensor([6, 13][:selected]),
        hidden_out=(torch.empty(selected, hidden, dtype=dtype) if out else None),
    ), torch.randn(rows, hidden).to(dtype)


def test_the_extend_copies_its_rows_with_draft_select(copies):
    from sgl_nvidia_mega.qwen3_5.draft import extend

    parts, engaged = _parts({"draft_extend", "draft_select"})
    io, hidden = _io()
    got = extend._select(parts, hidden, io)
    assert got is io.hidden_out
    assert torch.equal(got, hidden[[6, 13]])
    assert len(copies) == 1 and copies[0][0] is hidden
    assert engaged == [("draft_select", 2)]


def test_a_strided_hidden_takes_the_copy(copies):
    from sgl_nvidia_mega.qwen3_5.draft import extend

    parts, _ = _parts({"draft_extend", "draft_select"})
    io, hidden = _io(hidden=16)
    view = hidden[:, :8]
    assert view.stride(0) == 16
    io.hidden_out = torch.empty(2, 8, dtype=torch.bfloat16)
    assert torch.equal(extend._select(parts, view, io), view[[6, 13]])
    assert len(copies) == 1


@pytest.mark.parametrize(
    "slots, change",
    [
        ({"draft_extend"}, None),
        ({"draft_extend", "draft_select"}, "fp32"),
        ({"draft_extend", "draft_select"}, "no out"),
        ({"draft_extend", "draft_select"}, "transposed"),
    ],
)
def test_otherwise_the_extend_keeps_index_select(copies, slots, change):
    from sgl_nvidia_mega.qwen3_5.draft import extend

    parts, engaged = _parts(slots)
    io, hidden = _io(dtype=torch.float32 if change == "fp32" else torch.bfloat16)
    if change == "no out":
        io.hidden_out = None
    elif change == "transposed":
        hidden = hidden.t().contiguous().t()
    got = extend._select(parts, hidden, io)
    assert torch.equal(got, hidden[[6, 13]])
    assert copies == [] and engaged == []


def test_copy_rows_refuses_what_it_cannot_copy():
    pytest.importorskip("triton")
    rows = pytest.importorskip(ROWS_MODULE)
    src = torch.zeros(4, 8, dtype=torch.bfloat16)
    index = torch.tensor([1, 2])
    out = torch.empty(2, 8, dtype=torch.bfloat16)
    for bad in (
        dict(src=src.float()),
        dict(src=src.t()),
        dict(out=out.float()),
        dict(out=torch.empty(8, 2, dtype=torch.bfloat16).t()),
        dict(index=index.view(2, 1)),
        dict(out=torch.empty(3, 8, dtype=torch.bfloat16)),
    ):
        args = dict(src=src, index=index, out=out) | bad
        with pytest.raises(ValueError, match="copy_rows"):
            rows.copy_rows(**args)
