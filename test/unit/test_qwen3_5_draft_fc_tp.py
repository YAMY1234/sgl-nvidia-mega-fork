# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import fc_tp

TEP8 = ("tep8", topology.GB300)
FC_INPUT_MODULE = "sgl_nvidia_mega.qwen3_5.draft.fc_input"


def test_the_slot_and_option_are_numeric_rows_of_the_tep8_plan_only():
    assert fc_tp.SLOT in runtime.DRAFT_SLOTS
    assert topology.SLOT_TOPOLOGIES[fc_tp.SLOT] == {"tep8"}
    assert fc_tp.SLOT in runtime.MEASURED_DRAFT_TEP8
    assert fc_tp.PUSH in runtime.MEASURED_TEP8_OPTIONS
    for name in (fc_tp.SLOT, fc_tp.PUSH):
        assert name in runtime.NUMERICS_CHANGE
        assert name in runtime.MEASURED_PLAN[TEP8]
        assert name not in runtime.MEASURED_PLAN[("tp8", topology.GB300)]
    reasons = runtime.slot_reasons(frozenset({"draft_loop", fc_tp.SLOT}))
    assert any("'draft_fc_tp' needs slot 'draft_fc_input'" in r for r in reasons)


def test_a_column_window_is_its_own_spec_and_contiguous_is_not():
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    base = dict(m=4, k=1024, n_bf16=64, fp8_act="bf16", tactic_bf16=(8, 1, 1, 0, 1))
    plain = trt_lowm.spec(**base)
    assert trt_lowm.spec(**base, out_stride=64) == plain
    wide = 64 * 64
    window = trt_lowm.spec(**base, out_stride=wide)
    assert window.out_stride == wide and window != plain
    with pytest.raises(ValueError, match="below the 64 output columns"):
        trt_lowm.spec(**base, out_stride=32)
    assert trt_lowm.spec(**base, push=1).push == 1 and plain.push == 0


def test_the_push_descriptor_addresses_this_ranks_columns():
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    desc = trt_lowm.push_descriptor(
        mailbox=0x7F00_0000_1000,
        ctrl=0x10,
        rank=2,
        world=4,
        m_cap=8,
        row_elems=4096,
        col0=2048,
        device="cpu",
    )
    words = desc.tolist()
    assert words[4] == 2 * 8 * 4096 + 2048
    assert words[5] == 4 * 8 * 4096 and words[6] == 4096
    assert (words[1] << 32) | (words[0] & 0xFFFFFFFF) == 0x7F00_0000_1000
    for bad in (dict(col0=4), dict(row_elems=12), dict(rank=4), dict(mailbox=0x1008)):
        args = dict(
            mailbox=0x1000, ctrl=0, rank=0, world=4, m_cap=8, row_elems=4096, col0=0
        )
        args.update(bad)
        with pytest.raises(ValueError):
            trt_lowm.push_descriptor(device="cpu", **args)


def test_the_owner_slot_consume_is_an_add_norm_consume_of_a_pushed_partial():
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    assert kernels.consume_name(8, False, cols=True) == "q35hop_draft_add_norm_cols_w8"
    assert (
        kernels.consume_name(4, True, cols=True)
        == "q35hop_draft_add_norm_cols_early_w4"
    )
    with pytest.raises(ValueError, match="column-owned consume"):
        kernels.draft_all_reduce_rms_norm(
            None,
            local_contribution=None,
            residual=None,
            gamma=None,
            eps=1e-6,
            weight_bias=1.0,
            pushed=False,
            cols=True,
        )


# ------------------------------------------------------------------- the front


@pytest.fixture
def fused(monkeypatch):
    """The fused input stood in: the concat of the hidden rows, twice."""
    module = types.ModuleType(FC_INPUT_MODULE)
    module.mtp_fused_input = lambda ids, table, hidden, *a: torch.cat(
        [hidden, hidden], 1
    )
    monkeypatch.setitem(sys.modules, FC_INPUT_MODULE, module)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


class _Hop:
    def __init__(self, log, max_m=8):
        self.log = log
        self.hop = SimpleNamespace(supports=lambda m: 1 <= m <= max_m, max_m=max_m)

    def layer_input(self, partial, residual, norm, *, cols=False):
        self.log.append(("hop", partial.clone(), bool(residual.any()), cols))
        return partial * 3, partial.clone()


def _front(log, *, hidden=16, world=4, rank=1, hop=True, mnnvl=None):
    weight = (torch.arange(hidden // world * 2 * hidden) % 7).to(torch.bfloat16)
    weight = weight.view(hidden // world, 2 * hidden).contiguous()
    norm = SimpleNamespace(weight=torch.ones(hidden), variance_epsilon=1e-6)
    return fc_tp.FcTp(
        weight_slice=weight,
        table=None,
        norm_embedding=norm,
        norm_hidden=norm,
        input_norm=SimpleNamespace(gemma_weight="g"),
        rank=rank,
        world=world,
        hop=_Hop(log) if hop else None,
        mnnvl=mnnvl,
        max_rows=32,
    )


def test_the_front_writes_only_its_window_and_takes_the_hop(fused):
    log = []
    front = _front(log)
    hidden = torch.ones(3, 16, dtype=torch.bfloat16)
    x, normed = front(torch.arange(3), hidden)
    ((_, block, residual_nonzero, cols),) = log
    assert not residual_nonzero and cols is False
    want = torch.cat([hidden, hidden], 1) @ front.weight.t()
    assert torch.equal(block[:, 4:8], want)
    assert not block[:, :4].any() and not block[:, 8:].any()
    assert torch.equal(normed, block * 3) and torch.equal(x, block)


def test_past_the_hop_the_workspace_sums_and_the_layer_norms(fused):
    calls = []

    def all_reduce(*, local_contribution, residual, gamma):
        calls.append(gamma)
        return "norm", local_contribution + residual

    workspace = SimpleNamespace(all_reduce_residual_rms_norm=all_reduce)
    front = _front([], mnnvl=lambda rows: workspace if rows <= 16 else None)
    x, normed = front(torch.arange(12), torch.ones(12, 16, dtype=torch.bfloat16))
    assert normed is None and calls == ["g"] and x.shape == (12, 16)


def test_the_front_declines_widths_without_an_all_reduce(fused):
    front = _front([], mnnvl=lambda rows: None)
    assert front.ar_for(8) == "hop" and front.ar_for(9) == "" and front.ar_for(0) == ""
    assert front(torch.arange(9), torch.ones(9, 16, dtype=torch.bfloat16)) is None
    # hidden rows that differ from the tokens
    assert front(torch.arange(2), torch.ones(3, 16, dtype=torch.bfloat16)) is None


def test_the_front_refuses_a_weight_window_of_the_wrong_shape():
    with pytest.raises(ValueError, match="weight window"):
        fc_tp.FcTp(
            weight_slice=torch.zeros(4, 16, dtype=torch.bfloat16),
            table=None,
            norm_embedding=None,
            norm_hidden=None,
            input_norm=None,
            rank=0,
            world=4,
        )


def _parts(**over):
    hidden = 64
    fc = SimpleNamespace(
        weight=torch.zeros(hidden, 2 * hidden, dtype=torch.bfloat16), bias=None
    )
    parts = SimpleNamespace(
        lm=SimpleNamespace(fc=fc, pre_fc_norm_embedding=None, pre_fc_norm_hidden=None),
        binding=SimpleNamespace(tp_size=4, tp_rank=0, tp_cpu_group="cpu"),
        fc_input_table="table",
        hop="hop",
        layer=SimpleNamespace(input_layernorm=SimpleNamespace(gemma_weight="g")),
    )
    for key, value in over.items():
        if key == "fc":
            parts.lm.fc = value
        else:
            setattr(parts, key, value)
    return parts


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"fc_input_table": None}, "replicated table"),
        (
            {"fc": SimpleNamespace(weight=torch.zeros(64, 64, dtype=torch.bfloat16))},
            "not a replicated BF16 [H, 2H] linear",
        ),
        (
            {
                "fc": SimpleNamespace(
                    weight=torch.zeros(64, 128, dtype=torch.bfloat16), bias="b"
                )
            },
            "has a bias",
        ),
        ({"binding": SimpleNamespace(tp_size=3, tp_rank=0)}, "8-column tiles"),
        ({"hop": None}, "no fused all-reduce"),
        ({"layer": SimpleNamespace(input_layernorm=object())}, "not a GemmaRMSNorm"),
    ],
)
def test_the_build_refuses_what_it_cannot_split(monkeypatch, over, reason):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    seen = []
    monkeypatch.setattr(
        runtime, "slot_supported", lambda name, why, **_: seen.append(why) or not why
    )
    monkeypatch.setattr(layer, "_mnnvl_workspace", lambda parts, rows: None)
    assert fc_tp.maybe_build(_parts(**over)) is None
    (why,) = seen
    assert reason in why


# ------------------------------------------------------------------- the callers


def test_a_draft_step_takes_the_fronts_output_and_norm(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.draft import layer
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    monkeypatch.setattr(
        layer, "fc_input", lambda parts, token, hidden: calls.append("fc") or "x"
    )
    loop = DraftLoop.__new__(DraftLoop)
    loop.parts = SimpleNamespace(fc_tp=lambda t, h: ("sum", "normed"), mkd=None)
    assert loop._fc(torch.arange(2), None) == ("sum", "normed")
    loop.parts.fc_tp = lambda t, h: None
    assert loop._fc(torch.arange(2), None) == ("x", None) and calls == ["fc"]


@pytest.mark.parametrize("normed", [None, "normed"])
def test_the_layer_takes_the_fronts_norm_as_the_attention_input(monkeypatch, normed):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    seen = []

    def attention(hidden, *args, rows_per_request, o_proj):
        seen.append(hidden)
        raise StopIteration

    parts = SimpleNamespace(
        layer=SimpleNamespace(input_layernorm=lambda x: "own norm"),
        attention=attention,
        hop=None,
        slots=frozenset(),
        oproj_push=None,
    )
    with pytest.raises(StopIteration):
        layer.layer(
            parts,
            torch.ones(2, 4),
            None,
            None,
            None,
            None,
            rows_per_request=1,
            **({} if normed is None else {"normed": normed}),
        )
    assert seen == ["own norm" if normed is None else normed]
