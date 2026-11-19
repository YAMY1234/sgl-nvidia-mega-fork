# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(autouse=True)
def _fresh_refusals():
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    saved = set(split_kv._REFUSED)
    split_kv._REFUSED.clear()
    yield
    split_kv._REFUSED.clear()
    split_kv._REFUSED.update(saved)


def _meta(requests):
    return SimpleNamespace(
        cache_seqlens=torch.full((requests,), 4096, dtype=torch.int32),
        page_table=torch.zeros(requests, 1, dtype=torch.int32),
    )


def _call(requests, q_len, **kw):
    """cp_core on a fake call; returns (output, whether it asked for the CP state)."""
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    asked = []

    def cp_lite():
        asked.append(1)
        return None

    q = torch.zeros(requests * q_len, 4 * 8)
    with patch.object(split_kv, "cp_lite", cp_lite):
        try:
            o = split_kv.cp_core(None, q, None, _meta(requests), None, q_len, **kw)
        except ImportError:
            o = "went-on"
            asked.append(1)
    return o, bool(asked)


def test_limit_is_the_q1_plans_limit():
    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused

    assert split_kv.Q1_MAX_REQUESTS == q35_cp_fused.MAX_BATCH_Q1 == 64


@pytest.mark.parametrize("requests", [65, 96, 128])
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"min_bs": None},
        {"min_bs": None, "early_merge": False},
        {"min_bs": 1, "plain_wait": True},
    ],
)
def test_refuses_wide_q1(requests, kw):
    o, asked = _call(requests, 1, **kw)
    assert o is None and not asked


@pytest.mark.parametrize("requests, q_len", [(2, 1), (64, 1), (65, 7), (128, 7)])
def test_other_calls_go_on(requests, q_len):
    _, asked = _call(requests, q_len, min_bs=None)
    assert asked


def test_refusal_warns_once_a_width(caplog):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    with caplog.at_level(logging.WARNING, logger=split_kv.__name__):
        for requests in (96, 96, 128, 96):
            assert _call(requests, 1)[0] is None
        assert not split_kv.refuses_q1(64, 1) and not split_kv.refuses_q1(128, 7)
    lines = [r for r in caplog.records if "refuses" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.WARNING] * 2
    assert (
        "96 requests x 1 row" in lines[0].getMessage()
        and "128 requests x 1 row" in lines[1].getMessage()
    )


def test_mega_draft_run_is_refused_too():
    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP

    cp = DraftCP(split_kv=split_kv, route=None, steps=True, extend=False, multi=True)
    assert cp.takes(1, 96)
    attention = SimpleNamespace(layer=SimpleNamespace(attn=None))
    with patch.object(split_kv, "cp_lite", side_effect=AssertionError("CP-lite asked")):
        assert cp.run(attention, torch.zeros(96, 4, 256), _meta(96), None, 1) is None
