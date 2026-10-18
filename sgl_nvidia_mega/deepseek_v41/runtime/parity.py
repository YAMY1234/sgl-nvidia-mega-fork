# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test-only switches of the composition check (``SGLANG_NVIDIA_MEGAKERNEL_PARITY``).

``SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK=1`` sorts each row's sparse attention indices
before FlashMLA in both forwards, so the indexer top-k's emission order (atomic
arrival order) does not change the attention's accumulation order between them.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DeepseekV4AttnBackend",
        "DSV4AttnMetadata",
        "forward_metadata",
        "core_attn_metadata",
        "sparse_page_indices",
        "sparse_topk_lengths",
    ),
}

SORT_TOPK = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK", "0") == "1"


def install_sorted_topk() -> None:
    """Test only: sort each decode or verify row's valid sparse indices in place
    before SGLang's DSV4 attention reads them (``SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK``)."""
    import functools

    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
        DSV4AttnMetadata,
    )

    original = DeepseekV4AttnBackend.forward
    if getattr(original, "_sgl_mega_sorted", False):
        return

    @functools.wraps(original)
    def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        core = getattr(self.forward_metadata, "core_attn_metadata", None)
        ratio = getattr(layer, "compress_ratio", 0)
        if (
            ratio
            and isinstance(core, DSV4AttnMetadata)
            and (mode.is_decode() or mode.is_target_verify())
        ):
            table = core.sparse_page_indices(ratio)
            lengths = core.sparse_topk_lengths(ratio)
            if table is not None and lengths is not None:
                rows = table.view(-1, table.shape[-1])
                n = lengths.view(-1)[: rows.shape[0]].to(torch.int64)
                col = torch.arange(rows.shape[1], device=rows.device)
                valid = col[None, :] < n[:, None]
                big = torch.iinfo(rows.dtype).max
                key = torch.where(valid, rows, torch.full_like(rows, big))
                key = key.sort(dim=-1).values
                rows.copy_(torch.where(valid, key, rows))
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)

    forward._sgl_mega_sorted = True
    DeepseekV4AttnBackend.forward = forward
    logger.warning("NVIDIA megakernel: test-only sorted sparse indices are on")
