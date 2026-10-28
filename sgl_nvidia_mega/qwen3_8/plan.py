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

    gdn_verify: bool = False
    gdn_pdl: bool = False
    gdn_mk: bool = False  # numeric MK core, qualified separately
    gdn_front: bool = False  # numeric N32 projection/conv
    route: bool = False
    mkm: bool = False
    route_shared: bool = False


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    gdn_verify: bool = False
    gdn_pdl: bool = False
    gdn_mk: bool = False
    gdn_front: bool = False
    route: bool = False
    mkm: bool = False
    route_shared: bool = False


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    return Slots(
        gdn_verify=facts.gdn_verify,
        gdn_pdl=(facts.gdn_verify or facts.gdn_mk) and facts.gdn_pdl,
        gdn_mk=facts.gdn_mk,
        gdn_front=facts.gdn_front,
        route=facts.route,
        mkm=facts.route and facts.mkm and rows <= 7,
        route_shared=facts.route and facts.route_shared and rows <= 8,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}


def candidates(offered: tuple[int, ...], tokens_per_request: int) -> tuple[int, ...]:
    """The widths SGLang offers (``BuildContext.widths``) of one request."""
    return tuple(w for w in offered if w == tokens_per_request)
