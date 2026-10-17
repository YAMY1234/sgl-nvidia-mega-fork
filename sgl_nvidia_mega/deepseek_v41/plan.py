# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a routed step, per width (rows), fixed at build.

The row bounds are the measured ones: the wo_b publish + Lamport hop wins at 1-8
rows (FlashInfer's low-latency GEMM takes at most 8), the routed-expert pair at
1-48 rows, the fast gate top-k only at 1-8 and 48 rows (a leave-one-out
measured it slower at 12 and 24), and the two-shot hops from 49 rows (37 rows
measured slower at batch 8). Everything else is SGLang's own op.
"""

from __future__ import annotations

import os

import msgspec

MAX_ROWS = 384  # the widest step: the two-shot hops' mailbox
ROUTED_MAX_ROWS = 48  # the largest width of the routed-expert pair
HOP_A_MAX_ROWS = 8
TWO_SHOT_MIN_ROWS = 49
FAST_GATE_ROWS = frozenset(range(1, HOP_A_MAX_ROWS + 1)) | {ROUTED_MAX_ROWS}

# The slots build may fill (hop_a, moe, gate, two_shot, hop_b, early_trigger; test
# only: "dual" runs SGLang's own MoE on the dual-stream path); hop_b and
# early_trigger stay off by default until their A/B books them
SLOTS = frozenset(
    os.environ.get("SGLANG_NVIDIA_MEGA_SLOTS", "hop_a,moe,gate,two_shot").split(",")
) - {""}


class Slots(msgspec.Struct, frozen=True):
    hop_a: bool  # wo_b publishes into the Lamport all-reduce + mHC
    fast_gate: bool  # the router's top-k runs in gate_topk.cu
    routed: bool  # the routed experts run on the GEMM pair
    two_shot: bool = False  # both posts run the two-shot all-reduce + mHC
    hop_b: bool = False  # the MoE post runs the Lamport finalize + all-reduce + mHC


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on."""

    # SGLang returns the attention output before wo_b, and the mailbox built
    hop_a: bool
    hop_a_reason: str = ""
    # the two-shot mailbox built (SGLang defers the sums the call asks it to)
    two_shot: bool = False
    # the Lamport mailbox takes the routed pair's deferred finalize
    hop_b: bool = False


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the widths 1-{MAX_ROWS}")
    return Slots(
        hop_a=facts.hop_a and rows <= HOP_A_MAX_ROWS,
        fast_gate=rows in FAST_GATE_ROWS,
        routed=rows <= ROUTED_MAX_ROWS,
        two_shot=facts.two_shot and rows >= TWO_SHOT_MIN_ROWS,
        hop_b=facts.hop_b and rows <= HOP_A_MAX_ROWS,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
