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

MAX_ROWS = 384  # the widest covered step: the two-shot hops' mailbox
ROUTED_MAX_ROWS = 48  # the largest width of the routed-expert pair
TWO_SHOT_MIN_ROWS = 49  # measured slower than SGLang's fused kernel at 37 rows
# hop A: the publishing wo_b GEMM takes 1-8 rows on 128x4 scales and, with slot
# hop_a_batch2, 9-16 on the fused wo_a's row-tile scales; the mailbox holds this many rows
HOP_A_MAX_ROWS = 16
HOP_A_BATCH1_ROWS = 8
FAST_GATE_MAX_SMALL_ROWS = 16
FAST_GATE_ROWS = frozenset(range(1, FAST_GATE_MAX_SMALL_ROWS + 1)) | {ROUTED_MAX_ROWS}
# slot gate_wide: every routed width, to re-measure the 17-47-row cap (one launch per arm)
FAST_GATE_WIDE_ROWS = frozenset(range(1, ROUTED_MAX_ROWS + 1))
# the Engram prefetch serves batch 1 only (6 rows): at batch 2 (12 rows) the A/B measured
# +78 us, the quarters' HBM reads overlapping a MoE that is no longer latency-bound
ENGRAM_PREFETCH_MAX_ROWS = 8


class Slots(msgspec.Struct, frozen=True):
    hop_a: bool  # wo_b publishes into the Lamport all-reduce + mHC
    fast_gate: bool  # the router's top-k runs in gate_topk.cu
    routed: bool  # the routed experts run on the GEMM pair
    embed: bool = False  # the embedding looks up this rank's whole table
    engram_prefetch: bool = False  # Engram lookups + WKVs run on the prefetch stream
    two_shot: bool = False  # both posts run the two-shot all-reduce + mHC


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    hop_a: bool = False  # the wo_b publish and the Lamport mailbox built
    hop_a_reason: str = ""
    routed: bool = False  # the routed-expert GEMM pair (and its top-k) built
    embed: bool = False  # the whole embedding table replicated on every rank
    engram_prefetch: bool = False  # every Engram layer prefetches, checked at build
    two_shot: bool = False  # the two-shot mailbox and kernels built
    fast_gate_wide: bool = False  # slot gate_wide: the fast top-k at every routed width


def max_rows(two_shot: bool) -> int:
    """The widest covered step: past the routed pair only the two-shot hops serve."""
    return MAX_ROWS if two_shot else ROUTED_MAX_ROWS


def hop_a_max_rows(batch2: bool) -> int:
    return HOP_A_MAX_ROWS if batch2 else HOP_A_BATCH1_ROWS


def fast_gate_rows(wide: bool) -> frozenset:
    return FAST_GATE_WIDE_ROWS if wide else FAST_GATE_ROWS


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= max_rows(facts.two_shot):
        raise ValueError(
            f"{rows} rows are outside the covered widths 1-{max_rows(facts.two_shot)}"
        )
    return Slots(
        hop_a=facts.hop_a and rows <= HOP_A_MAX_ROWS,
        fast_gate=facts.routed and rows in fast_gate_rows(facts.fast_gate_wide),
        routed=facts.routed and rows <= ROUTED_MAX_ROWS,
        embed=facts.embed,
        engram_prefetch=facts.engram_prefetch and rows <= ENGRAM_PREFETCH_MAX_ROWS,
        two_shot=facts.two_shot and rows >= TWO_SHOT_MIN_ROWS,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
