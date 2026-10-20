# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a covered step, per width (rows), fixed at build.

A slot runs only where it is requested (``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``), built and
inside its row bounds; every other op is SGLang's own. ``route`` (the router
GEMM and the routed experts' NVFP4 input in one launch) and ``mkm`` (on its outputs, the
top-10 and both expert GEMMs in one launch, up to 7 rows) cover every width, as do
``gdn_verify`` (the GatedDeltaNet verify chain with its launch choices changed) and
``hop`` (the boundary's fused all-reduce points on the decode hop). ``skv2`` runs the
full attention's core through our 2.2 branch's split-KV attention, ``mka`` the fused
attention prologue and epilogue around it, and ``mka_push`` (with ``hop``) has the
epilogue push its rows into the hop.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 7  # the widest covered step: one request of 7 verify rows (TP8 bs1)


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step."""

    route: bool = False  # router GEMM + NVFP4 input quantization (sm103/moe/router.py)
    mkm: bool = (
        False  # top-10 + FC1 + SwiGLU + FC2 on route's outputs (sm103/moe/mkm.py)
    )
    gdn_verify: bool = False  # the GDN verify chain's launch choices (gdn/verify.py)
    gdn_pdl: bool = False  # its ReplaySSM verify launched dependent on the conv
    hop: bool = False  # the boundary's fused all-reduce points (boundary/hop.py)
    skv2: bool = False  # the full attention's core through SKV2 (attention/split_kv.py)
    mka: bool = False  # the fused attention prologue and epilogue (attention/mka.py)
    mka_push: bool = False  # the epilogue pushes its rows into the hop


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    route: bool = False  # the router front built for every MoE layer
    mkm: bool = False  # the fused routed experts built for every MoE layer
    gdn_verify: bool = False  # the GDN verify slot built for every GDN layer
    gdn_pdl: bool = False  # with its dependent verify launch
    hop: bool = False  # the decode hop, built for the widest covered step
    skv2: bool = False  # SKV2 for every full-attention layer
    mka: bool = False  # the fused prologue and epilogue for every full-attention layer
    mka_push: bool = False  # with the epilogue's push into the hop


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    return Slots(
        route=facts.route,
        mkm=facts.route and facts.mkm,
        gdn_verify=facts.gdn_verify,
        gdn_pdl=facts.gdn_verify and facts.gdn_pdl,
        hop=facts.hop,
        skv2=facts.skv2,
        mka=facts.mka,
        mka_push=facts.mka and facts.mka_push and facts.hop,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
