# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""What the draft entries share, built once by the first entry factory SGLang calls
(before it captures the draft graphs, every rank in the same order): the draft layer,
the attention's buffers, and each requested draft slot's build."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": (
        "model",
        "fc",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "lm_head",
        "logits_processor",
    ),
    "srt/models/qwen3_5.py": (
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "attn",
    ),
    "srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_device_group",
        "tp_cpu_group",
        "page_size",
    ),
}


class Parts:
    """The draft model's modules the entries run, and the slot builds."""

    def __init__(self, draft, layer):
        self.draft = draft
        self.lm = draft.lm  # Qwen3_5ForCausalLMMTP
        self.model = draft.lm.model  # its one-layer Qwen3_5ForCausalLM
        self.layer = layer  # the MTP decoder layer (full attention + MoE)
        self.binding = draft.binding
        self.slots = draft.slots
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.attention = None  # draft.attention.Attention
        self.moe: Optional[Any] = None  # draft.moe.DraftMoe (slot draft_moe)
        # (entry, bs) pairs whose capture logged the engagement line
        self.engaged: set = set()

    @classmethod
    def build(cls, draft) -> Parts:
        from sgl_nvidia_mega.qwen3_5.draft.attention import Attention

        m = draft.lm.model
        layers = [m.layers[i] for i in range(m.start_layer, m.end_layer)]
        if len(layers) != 1:
            raise RuntimeError(
                f"NVIDIA megakernel: the MTP draft has {len(layers)} layers, not 1"
            )
        parts = cls(draft, layers[0])
        parts.attention = Attention(parts)
        slots = parts.slots
        if "draft_moe" in slots:
            parts.moe = parts._build_moe()
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft parts built (slots %s; MoE %s)",
            sorted(slots),
            "single launch" if parts.moe is not None else "SGLang's",
        )
        return parts

    def _build_moe(self):
        from sgl_nvidia_mega.qwen3_5.draft.moe import DraftMoe

        moe = DraftMoe.build(self.layer.mlp, self.device.index)
        if moe is None:
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_moe cannot run: "
                f"{getattr(DraftMoe, 'last_reason', 'see the log')}"
            )
        return moe

    def note_engaged(self, entry: str, bs: int) -> None:
        """Log once per (entry, bs) captured: the engagement line every rank prints."""
        if torch.cuda.is_current_stream_capturing() and (entry, bs) not in self.engaged:
            self.engaged.add((entry, bs))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MTP draft %s captured for batch size %d "
                "(slots %s)",
                entry,
                bs,
                sorted(self.slots),
            )
