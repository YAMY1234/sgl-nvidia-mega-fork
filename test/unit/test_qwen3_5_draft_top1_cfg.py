# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes

import pytest

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.draft import top1

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


def test_the_option_is_a_bitwise_row_of_the_tp8_and_tep8_plans():
    assert top1.OPTION == "draft_top1_cfg"
    assert top1.OPTION in runtime.MEASURED_OPTIONS
    assert top1.OPTION in runtime.MEASURED_BITWISE
    assert top1.OPTION not in runtime.NUMERICS_CHANGE
    assert top1.OPTION not in topology.SLOT_TOPOLOGIES
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert top1.OPTION in runtime.MEASURED_PLAN[key]


@pytest.mark.parametrize(
    "switch, slots, key, want",
    [
        ("1", None, ("tep8", topology.GB300), top1.DEFAULT_CFG),
        ("1", None, ("tp8", topology.GB300), top1.DEFAULT_CFG),
        (None, None, ("tep8", topology.GB300), top1.SHIPPED_CFG),
        ("1", "draft_loop,draft_multi", ("tep8", topology.GB300), top1.SHIPPED_CFG),
        ("1", None, ("tp4", topology.GB300), top1.SHIPPED_CFG),
        ("1", None, ("tep8", topology.Sku(10, 3, 148)), top1.SHIPPED_CFG),
        ("1", None, None, top1.SHIPPED_CFG),
    ],
)
def test_the_variant_follows_the_measured_rows(monkeypatch, switch, slots, key, want):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "MEASURED_KEY", key)
    assert top1.config() == want


def test_the_configs():
    assert top1.SHIPPED_CFG == (256, 8, False, False)
    assert top1.DEFAULT_CFG == (256, 8, True, True)


def test_p2p_params_extend_the_shipped_layout():
    base = top1.DraftTop1Params
    p2p = top1.DraftTop1P2PParams
    for name, _ in base._fields_:
        assert getattr(p2p, name).offset == getattr(base, name).offset
    assert p2p.peers.offset % 8 == 0 and p2p.peers.offset >= ctypes.sizeof(base) - 8


def test_peer_table_maps_the_buffer_offset_on_every_rank():
    ptrs = [0x7F0000000000 + r * (1 << 32) for r in range(8)]
    peers, why = top1.peer_table(ptrs, ptrs[3] + 4096, 3, 8)
    assert why == "" and peers == [p + 4096 for p in ptrs]


@pytest.mark.parametrize(
    "ptrs, uc",
    [
        ([1 << 40, 2 << 40, 3 << 40, 4 << 40], 2 << 40),  # 4 addresses for world 8
        ([(r + 1) << 40 for r in range(8)], 1 << 20),
        ([(r + 1) << 40 if r != 5 else 0 for r in range(8)], 4 << 40),  # a null peer
        ([1 << 40] * 8, 1 << 40),  # repeated addresses
    ],
)
def test_peer_table_refuses_bad_handles(ptrs, uc):
    peers, why = top1.peer_table(ptrs, uc, 3, 8)
    assert peers is None and why
