# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import random
import struct
import types

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import head_keys


@pytest.fixture(autouse=True)
def _no_capture(monkeypatch):
    # CPU: no CUDA context to ask
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_the_option_is_a_numeric_row_of_the_tp8_and_tep8_plans():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    assert head_keys.OPTION in runtime.MEASURED_OPTIONS
    assert head_keys.OPTION in runtime.NUMERICS_CHANGE
    assert head_keys.OPTION not in runtime.MEASURED_BITWISE
    assert head_keys.OPTION not in topology.SLOT_TOPOLOGIES
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert head_keys.OPTION in runtime.MEASURED_PLAN[key]
    dep4 = runtime.MEASURED_PLAN.get(("dep4", topology.GB300), ())
    assert head_keys.OPTION not in dep4
    numerics = draft.numerics(frozenset({head_keys.OPTION}))
    assert numerics.kind is NumericsKind.TOLERANCE


def test_the_widths_are_nine_to_sixteen():
    from sgl_nvidia_mega.qwen3_5.draft import MULTI_LOOP_BS, head_lowm

    assert head_keys.head_rows(MULTI_LOOP_BS) == tuple(range(9, 17))
    assert head_keys.MIN_ROWS == head_lowm.MAX_ROWS + 1


def test_nan_word_is_the_signed_form_of_the_sharded_top1_key():
    from sgl_nvidia_mega.qwen3_5.sm103.draft import xchg
    from sgl_nvidia_mega.qwen3_5.sm103.head import keys

    assert head_keys.nan_hi(xchg.NAN_KEY_TORCH_ARGMAX) == keys.NAN_TORCH
    assert head_keys.nan_hi(xchg.NAN_KEY_SPLIT_ARGMAX) == keys.NAN_SPLIT
    rng = random.Random(7)
    values = [float("inf"), float("-inf"), 0.0, -0.0, 1.0, -1.0, -1e30, 1e30]
    values += [3.4e38, -3.4e38, 1e-45, -1e-45]
    for _ in range(2000):
        bits = rng.getrandbits(32) & 0x7F7FFFFF | rng.choice((0, 1 << 31))
        values.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    unsigned = [xchg._order_key(0.0 if v == 0.0 else v) for v in values]
    signed = []
    for v, u in zip(values, unsigned):
        s = head_keys.nan_hi(u)
        assert s == keys.ordered_hi(v), v
        signed.append(s)
    # the same order: sorting by the value sorts both key forms
    order = sorted(range(len(values)), key=lambda i: values[i])
    assert [signed[i] for i in order] == sorted(signed)
    assert [unsigned[i] for i in order] == sorted(unsigned)


class _Keys:
    def __init__(self, rows):
        self.rows, self.calls = set(rows), []

    def takes(self, rows):
        return rows in self.rows

    def serves(self, hidden, tokens, outputs):
        return head_keys.DraftHeadKeys.serves(self, hidden, tokens, outputs)

    def top1_into(self, hidden, tokens, column, *, nan_key, **outputs):
        self.calls.append((int(hidden.shape[0]), column, nan_key, sorted(outputs)))


def _parts(keys, head_lowm=None):
    stock = []

    def compute(hidden, lm_head, _):
        stock.append(("cublas", int(hidden.shape[0])))
        return torch.zeros(int(hidden.shape[0]), 8, dtype=torch.bfloat16)

    def top1x(logits, tokens, column, *, nan_key, **outputs):
        stock.append(("top1x", int(logits.shape[0]), column))

    lm = types.SimpleNamespace(
        logits_processor=types.SimpleNamespace(_compute_lm_head=compute), lm_head=None
    )
    parts = types.SimpleNamespace(
        lm=lm, head_keys=keys, head_lowm=head_lowm, top1x=top1x
    )
    return parts, stock


@pytest.mark.parametrize("rows", [9, 13, 16])
def test_the_top1_sends_its_widths_to_the_keys_head(rows):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    keys = _Keys(range(9, 17))
    parts, stock = _parts(keys)
    tokens = torch.zeros(rows, 6, dtype=torch.int64)
    pos = torch.zeros(rows, dtype=torch.int64)
    x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
    layer.top1_into(parts, x, tokens, 3, nan_key=5, positions=pos)
    assert keys.calls == [(rows, 3, 5, ["positions"])] and stock == []


@pytest.mark.parametrize("rows", [8, 17, 32])
def test_another_width_keeps_the_stock_head(rows):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    keys = _Keys(range(9, 17))
    parts, stock = _parts(keys)
    x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
    layer.top1_into(parts, x, torch.zeros(rows, 6, dtype=torch.int64), 1, nan_key=5)
    assert keys.calls == [] and stock == [("cublas", rows), ("top1x", rows, 1)]


def test_outputs_the_launches_do_not_take_keep_the_stock_head():
    from sgl_nvidia_mega.qwen3_5.draft import layer

    keys = _Keys(range(9, 17))
    parts, stock = _parts(keys)
    x = torch.zeros(12, 4096, dtype=torch.bfloat16)
    tokens = torch.zeros(12, 1, dtype=torch.int64)
    bf16_p = torch.zeros(12, 1, dtype=torch.bfloat16)
    layer.top1_into(parts, x, tokens, 0, nan_key=5, topk_p=bf16_p)
    assert keys.calls == [] and stock == [("cublas", 12), ("top1x", 12, 0)]
    fp32_p = torch.zeros(12, 1, dtype=torch.float32)
    layer.top1_into(parts, x, tokens, 0, nan_key=5, topk_p=fp32_p)
    assert keys.calls == [(12, 0, 5, ["topk_p"])]
    stock.clear()
    narrow = torch.zeros(3, 11, dtype=torch.int64)
    layer.top1_into(parts, x, tokens, 0, nan_key=5, mrope_positions=narrow)
    int32_pos = torch.zeros(12, dtype=torch.int32)
    layer.top1_into(parts, x, tokens, 0, nan_key=5, positions=int32_pos)
    assert len(keys.calls) == 1 and stock == [("cublas", 12), ("top1x", 12, 0)] * 2
    layer.top1_into(
        parts,
        x,
        tokens,
        0,
        nan_key=5,
        positions=torch.zeros(12, dtype=torch.int64),
        mrope_positions=torch.zeros(3, 16, dtype=torch.int64),
    )
    assert keys.calls[-1] == (12, 0, 5, ["mrope_positions", "positions"])
    stock.clear()
    layer.top1_into(parts, x.float(), tokens, 0, nan_key=5)
    assert len(keys.calls) == 2 and stock == [("cublas", 12), ("top1x", 12, 0)]
    parts, stock = _parts(None)
    layer.top1_into(parts, x, tokens, 0, nan_key=5)
    assert stock == [("cublas", 12), ("top1x", 12, 0)]


def test_the_low_m_head_keeps_its_widths_beside_the_keys_head():
    from sgl_nvidia_mega.qwen3_5.draft import layer

    class _LowM:
        def takes(self, rows):
            return rows <= 8

        def __call__(self, hidden):
            stock.append(("low-M", int(hidden.shape[0])))
            return hidden

    keys = _Keys(range(9, 17))
    parts, stock = _parts(keys, _LowM())
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    layer.top1_into(parts, x, torch.zeros(4, 1, dtype=torch.int64), 0, nan_key=5)
    assert keys.calls == [] and stock == [("low-M", 4), ("top1x", 4, 0)]


def test_off_the_measured_plan_nothing_is_built(monkeypatch):
    monkeypatch.setattr(runtime, "measured_option", lambda name: False)

    def agree(*_, **__):
        raise AssertionError("no collective off the plan")

    monkeypatch.setattr(runtime, "slot_supported", agree)
    assert head_keys.maybe_build(types.SimpleNamespace()) is None


def test_the_build_asks_for_its_own_option(monkeypatch):
    asked = []
    monkeypatch.setattr(runtime, "measured_option", lambda name: asked.append(name))
    head_keys.maybe_build(types.SimpleNamespace())
    assert asked == [head_keys.OPTION]


def test_a_shard_the_kernels_cannot_take_is_refused_on_every_rank(monkeypatch):
    monkeypatch.setattr(runtime, "measured_option", lambda name: True)
    agreed = []

    def agree(slot, reason, *, cpu_group=None):
        agreed.append((slot, reason, cpu_group))
        return not reason

    monkeypatch.setattr(runtime, "slot_supported", agree)
    proc = types.SimpleNamespace(
        logit_scale=None, final_logit_softcapping=None, vocab_size=512
    )
    lm_head = types.SimpleNamespace(
        weight=torch.zeros(64, 4096, dtype=torch.float32),
        shard_indices=types.SimpleNamespace(org_vocab_start_index=64),
    )
    parts = types.SimpleNamespace(
        lm=types.SimpleNamespace(lm_head=lm_head, logits_processor=proc),
        binding=types.SimpleNamespace(tp_rank=1, tp_size=8, tp_cpu_group="cpu"),
    )
    assert head_keys.maybe_build(parts) is None
    ((slot, reason, group),) = agreed
    assert (slot, group) == ("draft_head_lowm", "cpu")
    assert "argmax-key head at 9-16 rows" in reason and "contiguous BF16" in reason


def test_unsupported_shards_are_refused_with_their_reason():
    proc = types.SimpleNamespace(logit_scale=None, final_logit_softcapping=None)
    w = torch.zeros(64, 4096, dtype=torch.bfloat16)
    kw = dict(vocab_start=64, vocab_size=512, rank=1, world=8, processor=proc)
    why = head_keys.unsupported_reason
    assert "not rank 1's slice" in why(weight=w, **{**kw, "vocab_size": 500})
    assert "not rank 1's slice" in why(weight=w, **{**kw, "vocab_start": 0})
    assert "contiguous BF16" in why(weight=w.float(), **kw)
    narrow = torch.zeros(64, 2048, dtype=torch.bfloat16)
    assert "hidden size" in why(weight=narrow, **kw)
    scaled = types.SimpleNamespace(logit_scale=2.0, final_logit_softcapping=None)
    assert "scaling" in why(weight=w, **{**kw, "processor": scaled})
    capped = types.SimpleNamespace(logit_scale=None, final_logit_softcapping=30.0)
    assert "softcapping" in why(weight=w, **{**kw, "processor": capped})


def test_the_exchange_launch_fills_one_ranks_parameter_block(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.head import keys as K
    from sgl_nvidia_mega.qwen3_5.sm103.head import xchg

    launched = []
    monkeypatch.setattr(
        xchg, "_launch", lambda p, grid, device: launched.append((p, grid))
    )
    ex = xchg.HeadTop1Exchange()
    ex.world, ex.rank, ex.device = 8, 3, torch.device("cpu")
    ex.peers = [0x1000 * (r + 1) for r in range(8)]
    lines = torch.full((xchg.MAX_ROWS, xchg.KEY_LINE), K.KEY_FLOOR, dtype=torch.int64)
    ex.keys = lines[:, 0]
    ex.epoch = xchg._epoch(1, "cpu")
    out = torch.zeros(12, 4, dtype=torch.int64)
    pos = torch.zeros(12, dtype=torch.int64)
    ex(12, out, column=2, positions=pos)
    ((p, grid),) = launched
    assert grid == (2, 1)
    assert list(p.peers)[:8] == ex.peers and list(p.peers)[8:] == [0] * 24
    assert (p.m, p.world, p.rank, p.column) == (12, 8, 3, 2)
    assert (p.keys_stride, p.key_row_stride) == (0, xchg.KEY_LINE)
    assert (p.done, p.head_parts, p.win_keys) == (None, 0, None)
    assert (p.out_stride, p.out_inst_stride) == (4, 0)
    assert p.positions == pos.data_ptr() and p.topk_p is None and p.mrope is None
    assert xchg.EPOCH_WORDS == 2 + xchg.MAX_ROWS // xchg.ROWS_PER_CTA
    assert (
        ex.epoch[0, 0] == 1
        and ex.epoch[0, 1] == 0
        and bool((ex.epoch[0, 2:] == 1).all())
    )
    assert ctypes.sizeof(xchg.HeadXchgParams) == 8 * 32 + 8 * 8 + 8 * 5 + 4 * 5 + 4


@pytest.mark.parametrize(
    "call, match",
    [
        (dict(m=0), "0 rows"),
        (dict(m=513), "513 rows"),
        (dict(column=4), "column 4"),
        (dict(positions=torch.zeros(12, dtype=torch.int32)), "positions"),
        (dict(topk_p=torch.zeros(12, dtype=torch.bfloat16)), "topk_p"),
        (dict(mrope_positions=torch.zeros(3, 11, dtype=torch.int64)), "M-RoPE"),
    ],
)
def test_the_exchange_refuses_what_its_kernel_cannot_take(monkeypatch, call, match):
    from sgl_nvidia_mega.qwen3_5.sm103.head import xchg

    monkeypatch.setattr(xchg, "_launch", lambda *a: pytest.fail("no launch"))
    ex = xchg.HeadTop1Exchange()
    ex.world, ex.rank, ex.device, ex.peers = 8, 0, torch.device("cpu"), [0] * 8
    ex.keys, ex.epoch = torch.zeros(xchg.MAX_ROWS, dtype=torch.int64), None
    kwargs = dict(call)
    m = kwargs.pop("m", 12)
    with pytest.raises(ValueError, match=match):
        ex(m, torch.zeros(16, 4, dtype=torch.int64), **kwargs)
