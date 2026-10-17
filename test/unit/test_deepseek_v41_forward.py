# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 routed forward's glue on CPU: what the fast gate hands the
kernel, and where the MoE slot joins the quantized input."""

from types import SimpleNamespace

import torch

from sgl_nvidia_mega.deepseek_v41 import forward


def _gate(num_valid=None):
    return forward._GateTopK(
        torch.zeros(12, 384), torch.zeros(384), 1.5, num_valid, None, 0.0
    )


def test_the_fast_gate_passes_no_padded_row_count_through():
    """Without num_token_non_padded the kernel gets None, not a fresh tensor: a fill
    kernel per MoE layer in the decode graph broke the PDL chain."""
    slot = object.__new__(forward.MoeSlot)
    slot.gemm = SimpleNamespace(supports_gate_topk=lambda logits, bias: True)
    slot.bias, slot.scale, slot.eps = torch.zeros(384), 1.5, 1e-20
    kw = dict(
        hidden_states=None, router_logits=torch.zeros(12, 384), input_ids_global=None
    )
    got = slot.select(num_token_non_padded=None, use_vision_topk=False, **kw)
    assert got.num_valid is None
    count = torch.tensor([10], dtype=torch.int32)
    got = slot.select(num_token_non_padded=count, use_vision_topk=False, **kw)
    assert got.num_valid is count


def test_the_quantized_input_is_joined_before_the_top_k(monkeypatch):
    """The join precedes the top-k launch, as in SGLang's MoE: between the route kernel
    and GEMM1 it would cost GEMM1 its PDL early launch."""
    order = []

    class _Stream:
        def wait_event(self, event):
            order.append(("join", event))

    class _Gemm:
        top_k = 6

        def gate_topk_and_gemms(self, x_q, x_s, logits, bias, scale, num_valid, **kw):
            assert "input_ready" not in kw
            order.append(("top-k", num_valid))
            return (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(12, 6, dtype=torch.int32),
                None,
            )

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: _Stream())
    slot = object.__new__(forward.MoeSlot)
    slot.mlp, slot.gemm = SimpleNamespace(experts=None), _Gemm()
    slot.run(
        hidden_states=None, topk_output=_gate(), pre_quant_input=(None, None, "ready")
    )
    assert order == [("join", "ready"), ("top-k", None)]

