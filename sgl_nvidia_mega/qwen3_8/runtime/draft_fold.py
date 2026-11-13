# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (the fold slots made
# at each capture), python/sglang/srt/speculative/eagle_worker_v2.py (draft_forward's
# finalize of a step that feeds another) and python/sglang/srt/models/qwen4_exp_mtp.py
# (_prepare_input_embeds taking the folded embedding).
"""Slot ``draft_fold``: an MTP draft step's top-1 finalize with the next step's input
folded in (``kernels/ops/speculative/draft_step_fold.py``).

It runs behind an SGLang hook binding v2 does not have (its draft_loop entry refuses
a QSA draft and gets no forward batch for the executor's layer loop):

- before each draft graph is captured, the draft runner calls the draft executor's
  ``draft_step_fold(batch_size, num_steps)`` (None: SGLang's own top-1);
- in a topk-1 chain without draft probabilities, the draft loop calls the fold's
  ``finalize`` in place of ``draft_topk1_postprocess`` and the M-RoPE + 1 for each
  step whose output feeds another step (the first of a 3-step chain);
- the next draft step's forward then takes the folded embedding instead of embedding
  the token (``Qwen3_8Executor._draft``).

Until SGLang calls the entry, nothing here runs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build() -> str:
    """'' when the fold's kernels import here, else why not."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the draft fold needs an SM10x GPU"
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import (  # noqa: F401
            draft_step_fold,
        )
    except Exception as e:
        return f"the draft fold kernel does not import: {type(e).__name__}: {e}"
    return ""


class StepFold:
    """One captured draft graph's fold: its slot, and the executor whose next draft
    step takes the folded embedding."""

    def __init__(self, executor, slot):
        self.executor = executor
        self.slot = slot

    def finalize(self, logits, positions, draft_tokens, column, mrope_positions=None):
        """``draft_topk1_postprocess(logits, positions, draft_tokens, column)`` and the
        M-RoPE positions' + 1 (when given), with the token's embedding row gathered
        for the next draft step; returns (topk_p, topk_index) in the slot."""
        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold

        out = draft_step_fold.finalize_next_input(
            logits, positions, draft_tokens, column, self.slot, mrope_positions
        )
        self.executor.step_embedding = self.slot.embedding
        return out


def make(executor, batch_size: int, num_steps: int):
    """The entry ``draft_step_fold``: a StepFold for a draft graph of ``batch_size``
    requests, or None where the source's gate refuses it."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold

    reason = draft_step_fold.fold_reason(executor.lm, batch_size, num_steps)
    if reason:
        logger.info("Qwen3.8 slot draft_fold: bs %d stock (%s)", batch_size, reason)
        return None
    weight = executor.lm.model.embed_tokens.weight
    return StepFold(executor, draft_step_fold.make_slot(weight, batch_size))
