# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a routed step, per width (rows), fixed at build.

The row bounds are the measured ones: the wo_b publish + Lamport hop wins at 1-8
rows (FlashInfer's low-latency GEMM takes at most 8), the routed-expert pair at
1-48 rows, and the fast gate top-k only at 1-8 and 48 rows (a leave-one-out
measured it slower at 12 and 24). Everything else is SGLang's own op.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 384  # the widest covered step: the two-shot hops' mailbox
ROUTED_MAX_ROWS = 48  # the largest width of the routed-expert pair
TWO_SHOT_MIN_ROWS = 49  # measured slower than SGLang's fused kernel at 37 rows
HOP_A_MAX_ROWS = 8
FAST_GATE_ROWS = frozenset(range(1, HOP_A_MAX_ROWS + 1)) | {ROUTED_MAX_ROWS}


class Slots(msgspec.Struct, frozen=True):
    hop_a: bool  # wo_b publishes into the Lamport all-reduce + mHC
    fast_gate: bool  # the router's top-k runs in gate_topk.cu
    routed: bool  # the routed experts run on the GEMM pair
    two_shot: bool = False  # both posts run the two-shot all-reduce + mHC


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    hop_a: bool = False  # the wo_b publish and the Lamport mailbox built
    hop_a_reason: str = ""
    routed: bool = False  # the routed-expert GEMM pair (and its top-k) built
    two_shot: bool = False  # the two-shot mailbox and kernels built


def max_rows(two_shot: bool) -> int:
    """The widest covered step: past the routed pair only the two-shot hops serve."""
    return MAX_ROWS if two_shot else ROUTED_MAX_ROWS


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= max_rows(facts.two_shot):
        raise ValueError(
            f"{rows} rows are outside the covered widths 1-{max_rows(facts.two_shot)}"
        )
    return Slots(
        hop_a=facts.hop_a and rows <= HOP_A_MAX_ROWS,
        fast_gate=facts.routed and rows in FAST_GATE_ROWS,
        routed=facts.routed and rows <= ROUTED_MAX_ROWS,
        two_shot=facts.two_shot and rows >= TWO_SHOT_MIN_ROWS,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
