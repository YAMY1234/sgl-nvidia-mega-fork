# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_qwen3_5_cp_precise_fp64 as H  # noqa: E402

pytestmark = pytest.mark.skipif(not H._gpu_ready(), reason="needs an SM100-family GPU")

SCALES = (1.0, 1.0)
CASES = [(2, 1), (8, 1), (13, 1), (16, 1), (64, 1), (2, 7), (9, 7), (16, 7), (64, 7)]


def glue_call(grp, q8, kv, table, lens, q_len, gates=None, **over):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    attn, kvv = H.layer(), H.kv_view(kv, SCALES)
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    rows = q8.shape[0]
    fb = H._Fb()
    took: list = []

    def call(r, q_r):
        cp = split_kv.cp_lite()
        if gates is not None:
            cp.offer_gate_bf16(gates[r])
        o = split_kv.cp_core(
            attn,
            q_r.reshape(rows, H.HQ * H.DIM),
            kvv,
            meta,
            fb,
            q_len,
            min_bs=None,
            early_merge=False,
            merge_h=True,
            precise=True,
            **over,
        )
        if gates is not None:
            took.append(cp.take_gate_bf16())
        return o.clone()

    outs = grp.run(call, H.per_rank(q8, grp.cp), rows)
    return outs, took


def _inputs(b, q_len, seed):
    lengths = [max(L, q_len) for L in H.mix(b, seed, "split")]
    return H.make_inputs(lengths, q_len, "flat", seed=seed)


@pytest.mark.parametrize("b, q_len", CASES)
def test_gate_fold_is_the_merge_then_fused_sigmoid_mul(b, q_len):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    grp = H.group()
    q8, kv, table, lens = _inputs(b, q_len, seed=b + q_len)
    rows = q8.shape[0]
    g = torch.Generator(device="cuda").manual_seed(rows)
    gates = [
        (torch.randn(rows, H.HQ * H.DIM, device="cuda", generator=g) * 3).to(
            torch.bfloat16
        )
        for _ in range(grp.cp)
    ]
    base, _ = glue_call(grp, q8, kv, table, lens, q_len)
    got, took = glue_call(grp, q8, kv, table, lens, q_len, gates=gates)
    assert took and all(took)
    for r in range(grp.cp):
        want = fused_sigmoid_mul(base[r].clone(), gates[r], inplace=True)
        assert torch.equal(got[r].view(torch.int16), want.view(torch.int16)), r


def _gates(grp, rows):
    g = torch.Generator(device="cuda").manual_seed(rows + 1)
    return [
        (torch.randn(rows, H.HQ * H.DIM, device="cuda", generator=g) * 3).to(
            torch.bfloat16
        )
        for _ in range(grp.cp)
    ]


PUSH_CASES = [
    (b, q_len, early)
    for b, q_len in CASES
    for early in (False, True, "trig")
    if early != "trig" or q_len == 1
]


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("b, q_len, early", PUSH_CASES)
def test_the_partials_push_is_the_merges_own(b, q_len, early, gate):
    grp = H.group()
    q8, kv, table, lens = _inputs(b, q_len, seed=3 * b + q_len)
    gates = _gates(grp, q8.shape[0]) if gate else None
    want, _ = glue_call(grp, q8, kv, table, lens, q_len, gates=gates)
    got, took = glue_call(
        grp,
        q8,
        kv,
        table,
        lens,
        q_len,
        gates=gates,
        hilo_push=True,
        hilo_early=bool(early),
        hilo_trig=early == "trig",
    )
    assert all(grp.o_pushed), "the partial pushed the merge's words"
    if gate:
        assert took and all(took)
    for r in range(grp.cp):
        assert torch.equal(got[r].view(torch.int16), want[r].view(torch.int16)), r
    # the next call without the push still lands (its epochs follow the pushed call's)
    again, _ = glue_call(grp, q8, kv, table, lens, q_len, gates=gates)
    for r in range(grp.cp):
        assert torch.equal(again[r].view(torch.int16), want[r].view(torch.int16)), r
