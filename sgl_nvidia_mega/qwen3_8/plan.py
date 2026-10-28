# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of FN's covered step, per width (rows), fixed at build.

A covered step is the target verify of one request (``tokens_per_request`` rows, 4 for
the served three-step MTP chain), and with ``multi_request`` requested also of 2 or 4
requests (SGLang's decode graph sizes; 3 requests replay the graph of 4). A slot runs only
where it is requested (``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``), built and inside its row bound;
every other op is SGLang's own. A width wider than one request is covered only where a slot
runs. With no slot the executor runs SGLang's own ops in the model's order.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 16  # the widest covered step: four requests of 4 verify rows
REQUESTS = (1, 2, 4)  # the requests of a covered step
# the name that requests steps of more than one request in the slot list
MULTI_REQUEST = "multi_request"
# Each slot's widest step: the rows its kernels are qualified for, raised by each slot
# as it qualifies wider steps
GDN_MAX_ROWS = 4
ROUTE_MAX_ROWS = 4
MKM_MAX_ROWS = 7
ROUTE_SHARED_MAX_ROWS = 8
QSA_CORE_MAX_ROWS = 4


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step."""

    gdn_verify: bool = False
    gdn_pdl: bool = False
    gdn_mk: bool = False  # numeric MK core, qualified separately
    route: bool = False
    mkm: bool = False
    route_shared: bool = False
    qsa_core: bool = False  # numeric selected-KV attention + gate


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    gdn_verify: bool = False
    gdn_pdl: bool = False
    gdn_mk: bool = False
    route: bool = False
    mkm: bool = False
    route_shared: bool = False
    qsa_core: bool = False  # numeric selected-KV attention + gate


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    gdn = rows <= GDN_MAX_ROWS
    route = facts.route and rows <= ROUTE_MAX_ROWS
    return Slots(
        gdn_verify=facts.gdn_verify and gdn,
        gdn_pdl=(facts.gdn_verify or facts.gdn_mk) and facts.gdn_pdl and gdn,
        gdn_mk=facts.gdn_mk and gdn,
        route=route,
        mkm=route and facts.mkm and rows <= MKM_MAX_ROWS,
        route_shared=route and facts.route_shared and rows <= ROUTE_SHARED_MAX_ROWS,
        qsa_core=facts.qsa_core and rows <= QSA_CORE_MAX_ROWS,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}


def candidates(
    offered: tuple[int, ...], tokens_per_request: int, multi: bool = False
) -> tuple[int, ...]:
    """The widths SGLang offers (``BuildContext.widths``) of ``REQUESTS`` requests, or of
    one request without ``multi``."""
    requests = REQUESTS if multi else (1,)
    return tuple(
        w
        for w in offered
        if w % tokens_per_request == 0 and w // tokens_per_request in requests
    )


def covered(
    widths: tuple[int, ...], facts: Facts, tokens_per_request: int
) -> tuple[int, ...]:
    """The widths a build covers: one request always (with no slot it runs the stock
    step), more requests only where a slot runs."""
    return tuple(
        w for w in widths if w == tokens_per_request or slots(w, facts) != Slots()
    )


def widest(widths: tuple[int, ...], max_rows: int) -> int:
    """The widest of ``widths`` a slot bounded by ``max_rows`` rows serves (0: none)."""
    return max((w for w in widths if w <= max_rows), default=0)
