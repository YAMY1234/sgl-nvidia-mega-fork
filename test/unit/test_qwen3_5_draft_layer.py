# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch


def _fc(returns_bias):

    def fc(x):
        out = x * 2
        return (out, None) if returns_bias else out

    return fc


@pytest.mark.parametrize("returns_bias", [False, True])
def test_fc_input_returns_the_fc_rows_of_a_plain_or_replicated_linear(
    monkeypatch, returns_bias
):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    seen = []
    monkeypatch.setattr(
        layer,
        "norms_into_cat",
        lambda embeds, hidden, *_: seen.append(tuple(embeds.shape)) or hidden,
    )
    lm = SimpleNamespace(
        fc=_fc(returns_bias), pre_fc_norm_embedding=None, pre_fc_norm_hidden=None
    )
    parts = SimpleNamespace(
        lm=lm, fc_input_table=None, embed=lambda ids: torch.zeros(ids.shape[0], 4)
    )
    out = layer.fc_input(parts, torch.arange(3), torch.ones(3, 4))
    assert seen == [(3, 4)]
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, torch.full((3, 4), 2.0))
