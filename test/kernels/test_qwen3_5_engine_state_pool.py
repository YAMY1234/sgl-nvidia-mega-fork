# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _in_pool(pool, t) -> bool:
    p = t.data_ptr()
    return any(
        seg["address"] <= p < seg["address"] + seg["total_size"]
        for seg in pool.snapshot()
    )


def _check(make):
    dev = torch.device("cuda", torch.cuda.current_device())
    outer = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(outer, device=dev):
        freed = torch.empty(1 << 20, dtype=torch.uint8, device=dev)
        freed.fill_(0x5A)
        del freed
        control = torch.zeros(16 + 2 * 512, dtype=torch.int32, device=dev)
        state = make(dev)
    torch.cuda.synchronize()
    assert _in_pool(outer, control), (
        "the outer routing did not take the control allocation"
    )
    assert not _in_pool(outer, state), (
        "the call state came from the outer (shared) pool"
    )
    return state


def test_target_engine_state_pool():
    from sgl_nvidia_mega.qwen3_5.moe import body

    with patch.dict(body._ENGINE_STATE, clear=True):
        s = _check(body._engine_state)
        assert body._engine_state(s.device) is s
    assert int(s.abs().sum()) == 0  # zero-initialized


def test_target_ring_records_pool():
    from sgl_nvidia_mega.qwen3_5.moe import body
    from sgl_nvidia_mega.qwen3_5.sm103.moe.q35_moe_engine import REC_HDR

    with patch.dict(body._ENGINE_STATE, clear=True):
        r = _check(body._ring_records)
        assert body._ring_records(r.device) is r
    # each layer's header zeros (generation 0), its two prediction lists empty (-1)
    assert bool((r[:, :REC_HDR] == 0).all()) and bool((r[:, REC_HDR:] == -1).all())
