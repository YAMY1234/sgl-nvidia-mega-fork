# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a routed step, per width (rows), fixed at build.

The row bounds are the measured ones: the wo_b publish + Lamport hop wins at 1-8
rows (FlashInfer's low-latency GEMM takes at most 8), the routed-expert pair at
1-48 rows, and the fast gate top-k only at 1-16 and 48 rows (serving measured it
slower at 24). Everything else is SGLang's own op.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 48  # the largest routed width
HOP_A_MAX_ROWS = 8
FAST_GATE_MAX_SMALL_ROWS = 16
FAST_GATE_ROWS = frozenset(range(1, FAST_GATE_MAX_SMALL_ROWS + 1)) | {MAX_ROWS}


class Slots(msgspec.Struct, frozen=True):
    hop_a: bool  # wo_b publishes into the Lamport all-reduce + mHC
    fast_gate: bool  # the router's top-k runs in gate_topk.cu
    routed: bool  # the routed experts run on the GEMM pair
    embed: bool = False  # the embedding looks up this rank's whole table


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    hop_a: bool = False  # the wo_b publish and the Lamport mailbox built
    hop_a_reason: str = ""
    routed: bool = False  # the routed-expert GEMM pair (and its top-k) built
    embed: bool = False  # the whole embedding table replicated on every rank


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the routed widths 1-{MAX_ROWS}")
    return Slots(
        hop_a=facts.hop_a and rows <= HOP_A_MAX_ROWS,
        fast_gate=facts.routed and rows in FAST_GATE_ROWS,
        routed=facts.routed,
        embed=facts.embed,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
