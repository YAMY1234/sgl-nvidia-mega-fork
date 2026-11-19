# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.draft import early, fc_tp

FC_INPUT_MODULE = "sgl_nvidia_mega.qwen3_5.draft.fc_input"


def test_the_option_is_a_bitwise_row_of_the_tp8_and_tep8_plans():
    assert early.OPTION in runtime.MEASURED_OPTIONS
    assert early.OPTION in runtime.MEASURED_BITWISE
    assert early.OPTION not in runtime.NUMERICS_CHANGE
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert early.OPTION in runtime.MEASURED_PLAN[key]


def _qkv(**over):
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    mod = SimpleNamespace(
        weight=torch.zeros(64, 32, dtype=torch.bfloat16),
        bias=None,
        quant_method=UnquantizedLinearMethod.__new__(UnquantizedLinearMethod),
    )
    for key, value in over.items():
        setattr(mod, key, value)
    return SimpleNamespace(qkv_proj=mod)


@pytest.mark.parametrize(
    "over, tgv, want",
    [
        ({}, True, True),
        ({}, False, False),
        ({"bias": torch.zeros(64)}, True, False),
        ({"weight": torch.zeros(64, 32)}, True, False),
        ({"quant_method": object()}, True, False),
    ],
)
def test_the_qkv_gemm_releases_only_on_sglangs_tgv_route(monkeypatch, over, tgv, want):
    seen = []
    monkeypatch.setattr(
        early, "_tgv_takes", lambda rows, n, k: seen.append((rows, n, k)) or tgv
    )
    assert early.qkv_takes_pdl(_qkv(**over), 4) is want
    if not over:
        assert seen == [(4, 64, 32)]


@pytest.mark.parametrize(
    "on, rows, takes, want",
    [
        (True, 4, True, True),
        (False, 4, True, False),
        (True, 1, True, False),
        (True, 17, True, False),
        (True, 4, False, False),
    ],
)
def test_the_ffn_input_releases_the_low_m_router(monkeypatch, on, rows, takes, want):
    monkeypatch.setattr(early, "on", lambda: on)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    router = SimpleNamespace(takes=lambda m: takes)
    parts = SimpleNamespace(moe_block=SimpleNamespace(router=router))
    assert early.releases_router(parts, rows) is want
    parts.moe_block = None
    assert early.releases_router(parts, rows) is False


@pytest.mark.parametrize("on", [True, False])
def test_the_fc_front_releases_its_window_gemm_and_qkv(monkeypatch, on):
    monkeypatch.setattr(early, "on", lambda: on)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    released = []
    module = types.ModuleType(FC_INPUT_MODULE)

    def fused(ids, table, hidden, *args, release_next=False):
        released.append(("input", release_next))
        return torch.cat([hidden, hidden], 1)

    module.mtp_fused_input = fused
    monkeypatch.setitem(sys.modules, FC_INPUT_MODULE, module)

    class Hop:
        hop = SimpleNamespace(supports=lambda m: m <= 8, max_m=8)

        def layer_input(
            self, partial, residual, norm, *, cols=False, release_next=False
        ):
            released.append(("consume", release_next))
            return partial, partial

    lowm_calls = []

    class LowM:
        def takes(self, m):
            return True

        def __call__(self, full, out):
            lowm_calls.append(tuple(out.shape))
            out.zero_()

    front = fc_tp.FcTp(
        weight_slice=torch.zeros(4, 32, dtype=torch.bfloat16),
        table=None,
        norm_embedding=SimpleNamespace(weight=torch.ones(16), variance_epsilon=1e-6),
        norm_hidden=SimpleNamespace(weight=torch.ones(16), variance_epsilon=1e-6),
        input_norm=SimpleNamespace(gemma_weight="g"),
        rank=1,
        world=4,
        hop=Hop(),
        max_rows=32,
    )
    front.lowm = LowM()
    front.qkv_pdl = lambda rows: True
    front(torch.arange(2), torch.ones(2, 16, dtype=torch.bfloat16))
    assert released == [("input", on), ("consume", on)]
    assert lowm_calls == [(2, 4)]
