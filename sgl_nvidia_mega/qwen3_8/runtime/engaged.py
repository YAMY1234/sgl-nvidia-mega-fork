# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: which slots ran, counted on the device in eager steps and inside the
decode CUDA graphs (``SGLANG_NVIDIA_MEGAKERNEL_ENGAGED=1``).

Per covered width, one counter for the step and one per slot. The executor adds to
``step`` once per covered step; a slot's step function calls ``mark(st, slot)`` once
per layer after it has launched its own kernels, never on a path that falls back to
SGLang's op. Forwards SGLang runs before capturing a graph are not counted; a captured
step adds on every replay. The next plan of a forward that is not a target verify (a
new request's prefill) logs what was added and resets the counters, a host sync, so
the switch stays off in performance runs.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}

ENABLED = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_ENGAGED") == "1"
# the key under which the executor hands a step's counters to its step functions
SCRATCH_KEY = "runtime.engaged"


class Engaged:
    def __init__(self, widths: tuple[int, ...], slots: tuple[str, ...], device):
        self.names = ("step", *slots)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.counts = {
            w: torch.zeros(len(self.names), dtype=torch.int64, device=device)
            for w in widths
        }

    def add(self, rows: int, name: str) -> None:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode() and not torch.cuda.is_current_stream_capturing():
            return  # a forward SGLang runs before it captures a graph
        i = self.index[name]
        self.counts[rows][i : i + 1].add_(1)

    def flush(self) -> None:
        """Log and reset what the covered steps added since the last flush."""
        if torch.cuda.is_current_stream_capturing():
            return
        for rows, counts in self.counts.items():
            now = counts.tolist()
            if any(now):
                counts.zero_()
                logger.info(
                    "NVIDIA megakernel engaged rows=%d %s",
                    rows,
                    " ".join(f"{n}={c}" for n, c in zip(self.names, now)),
                )


def mark(st, slot: str) -> None:
    """Count one layer of ``slot`` in this step (a no-op unless the switch is on)."""
    held = st.scratch.get(SCRATCH_KEY)
    if held is not None:
        engaged, rows = held
        engaged.add(rows, slot)
