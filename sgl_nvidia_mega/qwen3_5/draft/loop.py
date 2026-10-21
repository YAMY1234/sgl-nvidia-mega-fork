# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/speculative/eagle_worker_v2.py (EagleDraftWorker.draft_forward, the
# topk-1 chain with draft_topk1_postprocess).
"""Entry ``draft_loop``: a served draft graph's whole draft loop for one request.

SGLang calls the factory once before it captures its draft graphs; the step it returns
for bs 1 runs inside the capture in place of ``draft_forward``. Draft step i (of
``num_steps - 1``) runs the model step at ``positions + i`` on the KV slot
``out_cache_loc[i]`` and writes its top-1 into ``draft_tokens_out[:, i + 1]``
(column 0 is the round's first token); the step advances the positions in place and
restores them before it returns, as the binding asks.

Stock: SGLang's model step and top-1 (``layer``).
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.qwen3_5.draft import SERVED_BS, layer

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/megakernels/eagle_binding.py": (
        "EagleDraftLoopIO",
        "num_requests",
        "num_steps",
        "topk_index",
        "hidden_states",
        "positions",
        "mrope_positions",
        "out_cache_loc",
        "steps",
        "kv",
        "draft_tokens_out",
    ),
}


class DraftLoop:
    """The draft_loop factory's result: bs -> the step."""

    def __init__(self, parts, *, num_steps: int):
        self.parts = parts
        self.num_steps = num_steps
        if "draft_q1" in parts.slots and parts.attention.q1 is None:
            from sgl_nvidia_mega.qwen3_5.draft.attention import build_q1

            parts.attention.q1 = build_q1(parts)

    def step_for(self, bs: int):
        return self.step if bs == SERVED_BS else None

    def begin_round(self, io) -> None:
        """On the host before a served draft graph replays: nothing to prepare while
        every op is SGLang's."""

    @torch.no_grad()
    def step(self, io) -> None:
        parts = self.parts
        parts.note_engaged("draft_loop", io.num_requests)
        steps = io.num_steps - 1
        tokens = io.draft_tokens_out
        tokens[:, :1].copy_(io.topk_index)
        positions, mrope = io.positions, io.mrope_positions
        token, hidden = io.topk_index.view(-1), io.hidden_states
        for i in range(steps):
            hidden = self._model_step(io, i, layer.fc_input(parts, token, hidden))
            token = layer.top1(parts, hidden)
            tokens[:, i + 1].copy_(token)
            positions.add_(1)
            if mrope is not None:
                mrope.add_(1)
        positions.sub_(steps)
        if mrope is not None:
            mrope.sub_(steps)

    def _model_step(self, io, i: int, x):
        """Draft step i's decoder layer on its fc output ``x``: the final-norm rows."""
        return layer.layer(
            self.parts,
            x,
            io.positions,
            io.out_cache_loc[i :: io.num_steps],
            io.steps[i],
            io.kv,
            rows_per_request=1,
        )
