# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of FN's covered step, per width (rows), fixed at build.

A covered step is the target verify of one request (``tokens_per_request`` rows, 4 for
the served three-step MTP chain). A slot runs only where it is requested
(``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``), built and inside its row bounds; every other op is
SGLang's own. With no slot the executor runs SGLang's own ops in the model's order.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 8  # the widest covered step: one request of up to 8 verify rows


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step."""


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    return Slots()


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}


def candidates(offered: tuple[int, ...], tokens_per_request: int) -> tuple[int, ...]:
    """The widths SGLang offers (``BuildContext.widths``) of one request."""
    return tuple(w for w in offered if w == tokens_per_request)
