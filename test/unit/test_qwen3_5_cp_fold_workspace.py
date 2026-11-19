# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from test_qwen3_5_cp_p2f16 import _built

from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

CONTEXT = 262144


def owned_pages(context: int, cp: int) -> int:
    pages = -(-context // L.PAGE)
    return -(-pages // cp)


def softmax_stats_bytes(rows: int, heads: int) -> int:
    return 8 * heads * rows * 256 + (1 << 20)


def counter_slots(rows: int, heads: int, sms: int) -> int:
    """The multi-CTA KV counters FlashInfer 0.7.4 checks the counter buffer for."""
    return -(-max(rows * heads, sms) // 8) * 8


def fold_calls(cp, ncols: int, q_len: int, b: int) -> list[tuple[int, int]]:
    heads = q_len * cp.cp * cp.hq
    spec = cp.fold_chunk_q1 if q_len == 1 else cp.fold_chunk
    keys = [k for k in spec if k <= b]
    calls = [(b, heads)]
    if keys:
        c = spec[max(keys)]
        if ncols > c and ncols % c == 0:
            calls.append((b * (ncols // c), heads))
    return calls


def _fresh(tp: int):
    """A copy of the built CpLite with no own workspace yet, its buffers on the meta device."""
    cp = copy.copy(_built(tp))
    cp.device = torch.device("meta")
    cp._ws, cp._ws_bytes, cp._ws_captured, cp._ws_in_graph = None, {}, [], False
    return cp


def test_the_carve_out_is_flashinfers():
    assert softmax_stats_bytes(2048, 112) == 470810624
    assert L._lse_workspace_bytes(2048, 112, 0) == 470810624
    assert L._lse_workspace_bytes(2048, 112, 152) == 470810624 + 152 * 128 * 1032


@pytest.mark.parametrize("context", [CONTEXT, 131072, 100000, 4096])
@pytest.mark.parametrize("q_len", [7, 1])
@pytest.mark.parametrize("tp", [4, 8, 16])
def test_the_own_workspace_holds_every_call_the_table_allows(tp, q_len, context):
    cp = _built(tp)
    ncols = owned_pages(context, cp.cp)
    size = cp._fold_ws_bytes(ncols, q_len)
    widest = 0
    for b in range(1, L._MAX_ROWS // q_len + 1):
        for rows, heads in fold_calls(cp, ncols, q_len, b):
            assert L._lse_workspace_bytes(rows, heads, cp.sms) <= size, (b, rows)
            assert counter_slots(rows, heads, cp.sms) <= cp.counter.numel(), (b, rows)
            widest = max(widest, rows * heads)
    # sized from the widest call, not a constant
    assert size == L._lse_workspace_bytes(widest, 1, cp.sms)


def test_tp4_at_256k_takes_449_mib_of_stats_at_both_q_lens():
    cp = _built(4)
    ncols = owned_pages(CONTEXT, cp.cp)
    assert ncols == 2048
    want = L._lse_workspace_bytes(2048, 112, cp.sms)
    assert cp._fold_ws_bytes(ncols, 7) == cp._fold_ws_bytes(ncols, 1) == want
    assert want > 384 << 20  # the former fixed size
    tp8 = _built(8)
    assert tp8._fold_ws_bytes(owned_pages(CONTEXT, tp8.cp), 7) < 384 << 20


@pytest.mark.parametrize("b", [55, 64, 110, 128])
def test_tp4s_fold_partial_hands_flashinfer_a_workspace_it_accepts(monkeypatch, b):
    cp, q_len = _fresh(4), 7
    ncols, heads = owned_pages(CONTEXT, cp.cp), q_len * cp.cp * cp.hq
    seen = []

    def decode(**kw):
        rows, h, _ = kw["query"].shape
        ws = kw["workspace_buffer"]
        assert softmax_stats_bytes(rows, h) <= ws.numel(), (rows, h, ws.numel())
        assert (
            counter_slots(rows, h, cp.sms) <= kw["multi_ctas_kv_counter_buffer"].numel()
        )
        seen.append((rows, ws.numel()))

    fi = SimpleNamespace(
        decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache=decode)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)

    def meta(*shape, dtype=torch.int32):
        return torch.empty(*shape, dtype=dtype, device="meta")

    (_, _), (nv, _) = fold_calls(cp, ncols, q_len, b)
    chunk = (meta(b), meta(nv, ncols // (nv // b)), meta(nv), meta(b), meta(b))
    fused = (
        chunk,
        meta(nv, heads, L.HEAD_DIM, dtype=torch.float8_e4m3fn),
        meta(nv, heads, L.HEAD_DIM, dtype=torch.bfloat16),
        meta(nv, heads, dtype=torch.float32),
        (meta(b, ncols), meta(b), ncols * L.PAGE),
    )
    stock_ws = meta(512 << 20, dtype=torch.uint8)
    cp._fold_partial(
        None,
        None,
        None,
        None,
        q_len,
        b,
        b * q_len,
        1.0,
        1.0,
        None,
        3,
        {},
        stock_ws,
        fused,
    )
    assert seen == [(b, 512 << 20), (nv, cp._ws.numel())]
    assert softmax_stats_bytes(nv, heads) > 384 << 20


def test_a_wider_table_grows_the_workspace_and_keeps_a_captured_one():
    cp = _fresh(4)
    a = cp._own_workspace(1024, 7)
    assert cp._own_workspace(1024, 7) is a and cp._own_workspace(512, 1) is a
    grown = cp._own_workspace(2048, 7)
    assert grown is not a and grown.numel() > a.numel()
    assert cp._ws_captured == []
    cp._ws_in_graph = True  # as if a capture took this one
    wider = cp._own_workspace(4096, 7)
    assert wider.numel() > grown.numel()
    assert len(cp._ws_captured) == 1 and cp._ws_captured[0] is grown
    assert cp._ws_in_graph is False
