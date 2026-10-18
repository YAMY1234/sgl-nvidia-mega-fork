# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hop B at 9-48 rows: it writes the next input's scales per 8-row tile only for a next
layer whose chain runs wqkv_a on the wide low-latency GEMM; SGLang's wqkv_a never
receives them."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import plan
from sgl_nvidia_mega.deepseek_v41.attention import chain, ops
from sgl_nvidia_mega.deepseek_v41.boundary import boundaries
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

HIDDEN = 5120


def _rows(rows):
    return mxfp8_ll.RowTileMxfp8(
        torch.zeros(rows, HIDDEN, dtype=torch.uint8),
        torch.zeros(mxfp8_ll.scale_bytes(rows, HIDDEN), dtype=torch.uint8),
    )


def test_hop_b_spans_batch_2_to_8_below_the_two_shot():
    assert plan.HOP_A_MAX_ROWS < plan.HOP_B_MAX_ROWS < plan.TWO_SHOT_MIN_ROWS
    assert plan.HOP_B_MAX_ROWS == plan.ROUTED_MAX_ROWS


def test_row_tiles_only_past_8_rows_for_a_reading_chain():
    b = boundaries.Boundaries.__new__(boundaries.Boundaries)
    b._next_layer, b.row_tile_layers = {3: 4, 4: 5}, frozenset({4})
    assert b._row_tiles(3, 12) and b._row_tiles(3, 48)
    assert not b._row_tiles(3, 8)
    assert not b._row_tiles(4, 12)  # layer 5's chain does not read them
    assert not b._row_tiles(39, 12)  # the last layer has no next


def test_the_quantized_rows_carry_their_layout():
    q, sf = torch.zeros(12, 4), torch.zeros(8)
    with mock.patch("sglang.srt.models.deepseek_v4_mhc.HcQuantized", lambda r, s: s):
        assert isinstance(
            boundaries._quantized(None, q, sf, True), mxfp8_ll.RowTileMxfp8
        )
        assert not isinstance(boundaries._quantized(None, q, sf), mxfp8_ll.RowTileMxfp8)


@pytest.mark.parametrize("rows", [12, 48])
def test_the_chain_runs_row_tile_rows_on_the_wide_gemm(rows):
    attn = SimpleNamespace(wqkv_a=mock.Mock(return_value=("sglang", None)), layer_id=0)
    attn.wqkv_a.weight, attn.wqkv_a.weight_scale_inv_swizzled = "w", "w_sf"
    cops = chain.ChainOps.__new__(chain.ChainOps)
    cops.qkv_a_ll, cops.qkv_a_bf16 = True, False
    x = _rows(rows)
    with mock.patch.object(chain.mxfp8_ll, "linear", return_value="ll") as ll:
        assert cops.qkv_a(attn, x) == "ll"
    assert ll.call_args.args == (x.quant, x.scale, "w", "w_sf")
    attn.wqkv_a.assert_not_called()


def test_sglangs_wqkv_a_never_takes_row_tile_scales():
    attn = SimpleNamespace(wqkv_a=mock.Mock(return_value=("sglang", None)))
    with pytest.raises(RuntimeError, match="row-tile MXFP8 scales"):
        ops.qkv_a(attn, _rows(12))
    attn.wqkv_a.assert_not_called()
