# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention slots on CPU: which widths and layers take them, and what the
sparse-decode slot hands the swap-AB kernel."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import attention
from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan

deepseek_v4 = pytest.importorskip("sglang.srt.models.deepseek_v4")
if not hasattr(deepseek_v4, "MQAOpSlots"):
    pytest.skip("this SGLang lacks the attention op slots", allow_module_level=True)


def test_attention_widths():
    """Norm + quant to 48 rows, swap-AB to 24 (it served slower at 48); none without the seams."""
    on = slot_plan.plan((1, 8, 24, 25, 48), slot_plan.Facts(hop_a=False, attn=True))
    assert [w for w, s in on.items() if s.attn_norm_quant] == [1, 8, 24, 25, 48]
    assert [w for w, s in on.items() if s.swapab] == [1, 8, 24]
    off = slot_plan.plan((1, 48), slot_plan.Facts(hop_a=False))
    assert not any(s.attn_norm_quant or s.swapab for s in off.values())


def _attn(compress_ratio):
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

    method = SimpleNamespace(
        mxfp8_dense_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL, use_mxfp8=True
    )
    return SimpleNamespace(
        is_dsv41=True,
        q_lora_rank=1280,
        q_norm=SimpleNamespace(
            weight=torch.ones(1280, dtype=torch.bfloat16), variance_epsilon=1e-6
        ),
        wq_b=SimpleNamespace(quant_method=method),
        q_head_norm=False,
        n_local_heads=16,
        head_dim=512,
        compress_ratio=compress_ratio,
        alt_streams=[object()] * 3,
        freqs_cis=torch.polar(torch.ones(4, 32), torch.zeros(4, 32)),
    )


def test_low_ratio_layers_rope_the_query_in_attention():
    """Only a low-ratio layer hands attention an unroped, unpadded query."""
    positions = torch.arange(12)
    low = attention.AttentionSlots(_attn(compress_ratio=2))
    slots = low.op_slots(12, positions, None, swapab_rows=True)
    assert slots.unpadded_q and slots.q_b is not None
    assert slots.sparse_decode.keywords["q_rope"][1] is positions
    assert tuple(low.freqs.shape) == (4, 64) and low.freqs.dtype == torch.float32

    swa_only = attention.AttentionSlots(_attn(compress_ratio=0))
    slots = swa_only.op_slots(12, positions, None, swapab_rows=True)
    assert not slots.unpadded_q and slots.q_b is None
    assert slots.sparse_decode is attention._sparse_decode

    wide = low.op_slots(30, torch.arange(30), None, swapab_rows=False)
    assert wide.q_norm_quant is not None
    assert wide.sparse_decode is None and not wide.unpadded_q
    assert low.op_slots(64, torch.arange(64), None, swapab_rows=False) is None


def test_sparse_decode_gathers_kv_before_its_wait_only_with_the_query_rope():
    """KV before the wait is safe only where every KV producer is a joined side stream."""
    q = torch.zeros(3, 1, 64, 512)
    inputs = dict(
        q=q,
        swa_k_cache="swa",
        swa_page_indices="swa idx",
        swa_topk_lengths="swa len",
        attn_sink="sink",
        extra_k_cache="extra",
        extra_indices="extra idx",
        extra_topk_lengths="extra len",
        flashmla_metadata="unused",
    )
    with mock.patch.object(attention.swapab, "swapab_attention") as kernel:
        attention._sparse_decode(**inputs)
        args, kwargs = kernel.call_args
        assert tuple(args[0].shape) == (3, 1, 16, 512)
        assert args[1:] == (
            "swa",
            "swa idx",
            "swa len",
            "sink",
            "extra",
            "extra idx",
            "extra len",
        )
        assert kwargs["pdl"] and kwargs["narrow_combine"]
        assert kwargs["combine_entry_trigger"]
        assert kwargs["max_rows"] == slot_plan.SWAPAB_MAX_ROWS
        assert not kwargs["kv_before_wait"] and kwargs["q_rope"] is None
        attention._sparse_decode(**inputs, q_rope=("freqs", "positions"))
        assert kernel.call_args.kwargs["kv_before_wait"]
