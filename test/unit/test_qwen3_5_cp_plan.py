# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import topology
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as fused


@pytest.mark.parametrize(
    "layout",
    [
        topology.Topology(8),
        topology.Topology(8, moe_ep_size=8),
        topology.Topology(4),
        topology.Topology(8, attn_dp_size=8, moe_ep_size=8),
    ],
)
def test_only_tp8_small_batches_change_split_mode(layout):
    state = fused._DeviceState.__new__(fused._DeviceState)
    state.unbalanced_batches = topology.cp_unbalanced_batches(layout)
    for batch in range(1, 129):
        assert state.balanced(batch) == (layout.name != "tp8" or not 2 <= batch <= 4)
    assert (fused.MAX_BATCH, fused.MAX_BATCH_Q1) == (128, 64)


def test_prepare_keeps_the_shared_counter_policy_fixed(monkeypatch):
    made = []
    monkeypatch.setattr(fused, "_STATES", {})
    monkeypatch.setattr(
        fused,
        "_DeviceState",
        lambda device, sms, batches: (
            made.append((device, sms, batches))
            or SimpleNamespace(unbalanced_batches=batches)
        ),
    )
    device = torch.device("cuda", 0)
    fused.q35_cp_fused_prepare(device, 152, unbalanced_batches=(2, 3, 4))
    # CpLite and all query lengths reuse this state, without resetting counters.
    fused.q35_cp_fused_prepare(device, 152)
    fused.q35_cp_fused_prepare(device, 152, unbalanced_batches=(2, 3, 4))
    assert made == [(device, 152, (2, 3, 4))]
    with pytest.raises(AssertionError, match="policy must be fixed"):
        fused.q35_cp_fused_prepare(device, 152, unbalanced_batches=())
