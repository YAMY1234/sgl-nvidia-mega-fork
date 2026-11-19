# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

import test_qwen3_5_attention as A  # noqa: E402

LENGTHS = (32768, 65536, 131072, 184320, 245759)

pytestmark = pytest.mark.skipif(not A._gpu_ready(), reason="needs an SM100-family GPU")


def _attention(context: int):
    from sglang.srt.layers.attention.trtllm_mla_backend import (
        make_persistent_multi_ctas_kv_counter_buffer,
    )

    from sgl_nvidia_mega.qwen3_5.attention import route, split_kv
    from sgl_nvidia_mega.qwen3_5.draft import attention as D

    att = D.Attention.__new__(D.Attention)
    att.layer = SimpleNamespace(
        attn=SimpleNamespace(
            layer_id=0,
            scaling=A.SCALE,
            sliding_window_size=-1,
            tp_q_head_num=A.HEADS,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=A.DIM,
        )
    )
    att.heads, att.head_dim, att.page_size = A.HEADS, A.DIM, A.PAGE
    att.max_context_len = context
    att.workspace = torch.zeros(512 << 20, dtype=torch.uint8, device="cuda")
    att.counters = make_persistent_multi_ctas_kv_counter_buffer(
        torch.device("cuda"), num_q_heads=32, max_batch_size=A.ROWS
    )
    att.q1 = None
    att._noted = set()
    att.cp_b1 = D.DraftCP(split_kv, route, steps=True, extend=True)
    return att


def _stock(att, q, pools, table, lengths, v_scale):
    import flashinfer
    from sglang.srt.environ import envs

    kv = att._paged(pools[0], pools[1])
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv,
        workspace_buffer=att.workspace,
        block_tables=table,
        seq_lens=lengths,
        max_seq_len=att.max_context_len,
        bmm1_scale=1.0 * att.layer.attn.scaling,
        bmm2_scale=v_scale,
        window_left=-1,
        sinks=None,
        skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        out=None,
        out_dtype=torch.bfloat16,
        kv_cache_sf=None,
        multi_ctas_kv_counter_buffer=att.counters,
        q_len_per_req=A.ROWS,
    )


def _case(length: int, seed: int):
    from sgl_nvidia_mega.qwen3_5.attention import route
    from sgl_nvidia_mega.qwen3_5.draft import attention as D

    g = torch.Generator(device="cuda").manual_seed(seed)
    q, pools, table, lengths = A._inputs(length, g)
    att = _attention(table.shape[1] * A.PAGE)
    v_scale = 1.0
    kv = SimpleNamespace(
        layer_ids=(0,), k=(pools[0],), v=(pools[1],), k_scale=(1.0,), v_scale=(v_scale,)
    )
    meta = SimpleNamespace(page_table=table, cache_seqlens=lengths)
    with mock.patch.object(D, "_q1_label", lambda: route.SPLIT_KV_LONG):
        got = att.decode(q.reshape(A.ROWS, -1), meta, kv, A.ROWS)
    stock = _stock(att, q, pools, table, lengths, v_scale)
    return q, pools, table, lengths, got.view(A.ROWS, A.HEADS, A.DIM), stock


@pytest.mark.parametrize("length", LENGTHS)
def test_the_one_request_draft_extend_is_the_stock_decode(length):
    *_, got, stock = _case(length, 1000 + length)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), stock.view(torch.int16)), length
