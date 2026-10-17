# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a routed step, per width (rows), fixed at build.

The row bounds are the measured ones: the wo_b publish + Lamport hop wins at 1-8
rows (FlashInfer's low-latency GEMM takes at most 8), the routed-expert pair at
1-48 rows, and the fast gate top-k only at 1-8 and 48 rows (a leave-one-out
measured it slower at 12 and 24). The attention chain's q-LoRA norm + quant runs at
1-48 rows and its swap-AB sparse attention at 1-24 (at 48 rows swap-AB served slower
than FlashMLA). Everything else is SGLang's own op.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 48  # the largest routed width
HOP_A_MAX_ROWS = 8
FAST_GATE_ROWS = frozenset(range(1, HOP_A_MAX_ROWS + 1)) | {MAX_ROWS}
ATTN_NORM_QUANT_MAX_ROWS = 48
SWAPAB_MAX_ROWS = 24


class Slots(msgspec.Struct, frozen=True):
    hop_a: bool  # wo_b publishes into the Lamport all-reduce + mHC
    fast_gate: bool  # the router's top-k runs in gate_topk.cu
    routed: bool  # the routed experts run on the GEMM pair
    attn_norm_quant: bool = False  # the q-LoRA norm + MXFP8 quant, with PDL
    swapab: bool = False  # the swap-AB sparse attention + combine, with PDL


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on."""

    # SGLang returns the attention output before wo_b, and the mailbox built
    hop_a: bool
    hop_a_reason: str = ""
    # SGLang has the attention op slots and the sparse-decode slot
    attn: bool = False


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the routed widths 1-{MAX_ROWS}")
    return Slots(
        hop_a=facts.hop_a and rows <= HOP_A_MAX_ROWS,
        fast_gate=rows in FAST_GATE_ROWS,
        routed=True,
        attn_norm_quant=facts.attn and rows <= ATTN_NORM_QUANT_MAX_ROWS,
        swapab=facts.attn and rows <= SWAPAB_MAX_ROWS,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
