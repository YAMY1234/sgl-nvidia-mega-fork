# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a covered step, per width (rows), fixed at build.

A slot runs only where it is requested (``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``), built and
inside its row bounds; every other op is SGLang's own. The covered steps are the
target verify of 1 to 32 requests (``REQUESTS``: SGLang's decode graphs of batch sizes
1, 2, 4, 8, 16 and 32); a width of more than one request is covered only where a slot
runs there. ``route`` (the router GEMM and the routed experts' NVFP4 input in one launch)
runs up to 16 rows, ``route_topk`` (behind SGLang's router GEMM: the top-10 and the NVFP4
input in one launch, then the routed experts on that routing) wherever ``route`` does
not, up to 224, ``mkm`` (on its outputs, the top-10 and both expert GEMMs in one launch)
up to 7, and with ``mkm_wide`` (its 16-token engine) up to 16; ``mkm_m32`` (its 32-token
engine, on route_topk's widths: the top-10 and both expert GEMMs in one launch on the
router GEMM's logits and the NVFP4 input) runs 17 to 32 rows, and ``mkm_m64`` (its 64-token
engine on route_topk's routing and NVFP4 input) 33 to 64, ``mkm_m128`` (its 128-token engine,
likewise) 65 to 128. ``gdn_verify`` (the
GatedDeltaNet verify chain with its launch choices changed) covers every width, with
``gdn_core`` (numeric: its ReplaySSM verify, gated norm and FP8 output in one launch), and ``hop`` (the boundary's fused all-reduce points on the decode hop) the
widths where FlashInfer's low-latency kernels, which it equals bit for bit, run, and with
``hop_wide`` up to 224 rows (past them its two-shot protocol, equal to FlashInfer's BT
routes), or with ``hop_mid`` (numeric: its one-shot protocol past FlashInfer's) up to 28.
``gdn_push`` (numeric: out_proj as the hop's push GEMM, on gdn_verify's FP8 norm) runs up
to 8 rows. ``gdn_front`` (numeric: the GDN projections and conv in one launch, then
gdn_pdl's verify) covers one or two requests of 7 rows, ``route_shared`` (numeric: the shared
expert's gate/up in route's launch) up to 8 rows. ``skv2`` runs the full attention's core
of one request through the split-KV attention of SGLang branch
nvidia/qwen35-agentx-gb300, ``mka`` the fused attention prologue and epilogue around it
(up to 8 rows; with four query heads a rank, its wide entries up to 14 rows and past
them its front path, bit for bit SGLang's, at every width), and ``mka_push`` (with
``hop``) has the epilogue push its rows into the hop. ``lowm_in`` / ``lowm_out`` (numeric) run the static FP8 input / output projections on a
low-M warp-MMA launch with the weights loaded before the dependency wait, up to each one's
measured rows (``LOWM_*_MAX_ROWS``), and with ``lowm_push`` (with ``hop``) push their rows into the hop;
``lowm_se`` (numeric) the shared expert in two such launches. ``hop_fp8`` (with ``hop``) has the hop
also write the next projection's static FP8 input, which it then reads without a quantize kernel.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 224  # the widest covered step: 32 requests of 7 verify rows (TP8 bs32)
# the requests of a covered step (SGLang's decode graph sizes)
REQUESTS = (1, 2, 4, 8, 16, 32)
# the TP sizes whose steps of more than one request are covered when requested (checked at
# TP8); at the others a covered step is one request
MULTI_REQUEST_TP = (8,)
# the name that requests them in the slot list
MULTI_REQUEST = "multi_request"
GDN_FRONT_ROWS = 7  # the rows of one request slot gdn_front's front is built for
GDN_FRONT_MAX_ROWS = 14  # its widest step: two requests (a cluster set a request pays past two)
# Each slot's widest step: its launcher's bound (sm103/moe/router.py MAX_TOKENS,
# sm103/moe/mkm.py MAX_TOKENS, sm103/hop OPROJ_MAX_M, sm103/attention/mka.py), and for
# hop the rows up to which FlashInfer's MNNVL CuTe DSL fusion runs the one-shot
# all-reduce the hop equals bit for bit (its finalize runs one-shot up to 23 rows)
ROUTE_MAX_ROWS = 16
ROUTE_TOPK_MAX_ROWS = 224
# slot route_gemm (numeric): the router front's 16-row tiles in place of the GEMM, up to here
ROUTE_GEMM_MAX_ROWS = 32
GDN_VERIFY_MAX_ROWS = 224  # the widths its kernels are checked at: up to 32 requests
GDN_CORE_MAX_ROWS = 224  # and gdn_core's launch
MKM_MAX_ROWS = 7
MKM_WIDE_MAX_ROWS = 16
# slot mkm_m32: the 32-token engine past the 16-token one, on route_topk's widths
MKM_M32_MAX_ROWS = 32
# slot mkm_m64: the 64-token engine on route_topk's routing, past the 32-token one
MKM_M64_MAX_ROWS = 64
# slot mkm_m128: the 128-token engine on route_topk's routing, past the 64-token one
MKM_M128_MAX_ROWS = 128
HOP_MAX_ROWS = 15
# with hop_wide: the two-shot hop past the one-shot rows (FlashInfer's BT routes, bit for bit)
HOP_WIDE_MAX_ROWS = 224
# with hop_mid (numeric): the one-shot hop up to four requests (in place of the BT routes)
HOP_MID_MAX_ROWS = 28
GDN_PUSH_MAX_ROWS = 8
# the rows slot route_shared's shared stage fits shared memory for
ROUTE_SHARED_ROWS = 8
MKA_MAX_ROWS = 8
# with four query heads a rank: the wide entries, then the front path (attention/mka.py)
MKA_WIDE_MAX_ROWS = MAX_ROWS
# slots lowm / lowm_se: the rows up to which each low-M launch measured faster than SGLang's quantize +
# cuBLAS path on GB300 (results/trt-1022/l6): the input projections (in_proj_qkvz + in_proj_ba, qkv_proj),
# the output projections (out_proj, o_proj) and the shared expert
LOWM_IN_MAX_ROWS = 14
# Up to two requests: on batch 3 (TP8 static) the launches at four requests (out_proj / o_proj with their push,
# the hop's FP8 copy) cost the step 0.14 ms a round, the shared expert's 0.21 ms more
LOWM_OUTPROJ_MAX_ROWS = 14
LOWM_OPROJ_MAX_ROWS = 14
LOWM_SE_MAX_ROWS = 14


class Slots(msgspec.Struct, frozen=True):
    """The kernels that fill one width's step."""

    route: bool = False  # router GEMM + NVFP4 input quantization (sm103/moe/router.py)
    # behind SGLang's router GEMM: top-10 + NVFP4 input in one launch (moe/route.py)
    route_topk: bool = False
    route_gemm: bool = False  # with the router front's tiles in place of that GEMM
    mkm: bool = (
        False  # top-10 + FC1 + SwiGLU + FC2 on route's outputs (sm103/moe/mkm.py)
    )
    # in route_topk's place: the 32-token engine on the router GEMM's logits (moe/route.py)
    mkm_m32: bool = False
    # in route_topk's runner: the 64-token engine on route_topk's routing (moe/route.py)
    mkm_m64: bool = False
    mkm_m128: bool = False  # likewise, its 128-token engine
    gdn_verify: bool = False  # the GDN verify chain's launch choices (gdn/verify.py)
    gdn_pdl: bool = False  # its ReplaySSM verify launched dependent on the conv
    # its ReplaySSM verify, gated norm and FP8 output in one launch (sm103/gdn/core.py)
    gdn_core: bool = False
    hop: bool = False  # the boundary's fused all-reduce points (boundary/hop.py)
    # out_proj pushes its partial into the hop from its GEMM (gdn/verify.py; numeric)
    gdn_push: bool = False
    # the projections and the conv in one launch (gdn/front.py; numeric)
    gdn_front: bool = False
    # the shared expert's gate/up in the router launch (moe/route.py; numeric)
    route_shared: bool = False
    skv2: bool = False  # the full attention's core through SKV2 (attention/split_kv.py)
    mka: bool = False  # the fused attention prologue and epilogue (attention/mka.py)
    mka_push: bool = False  # the epilogue pushes its rows into the hop
    # the input projections on the low-M launch (lowm.py; numeric)
    lowm_in: bool = False
    lowm_outproj: bool = False  # GDN out_proj on the low-M launch (lowm.py; numeric)
    lowm_oproj: bool = False  # attention o_proj on the low-M launch (lowm.py; numeric)
    lowm_se: bool = False  # the shared expert in two low-M launches (lowm.py; numeric)
    # the low-M output projections push their rows into the hop (lowm.py; numeric)
    lowm_push: bool = False
    # the hop also writes the next projection's static FP8 input (lowm.py; bitwise)
    hop_fp8: bool = False


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    route: bool = False  # the router front built for every MoE layer
    route_topk: bool = False  # the top-10 + NVFP4 input launch built for every MoE layer
    route_gemm: bool = False  # and the router front's tiles
    mkm: bool = False  # the fused routed experts built for every MoE layer
    mkm_wide: bool = False  # with their 16-token engine
    mkm_m32: bool = False  # the 32-token engine built for every MoE layer
    mkm_m64: bool = False  # the 64-token engine built for every MoE layer
    mkm_m128: bool = False  # the 128-token engine built for every MoE layer
    gdn_verify: bool = False  # the GDN verify slot built for every GDN layer
    gdn_pdl: bool = False  # with its dependent verify launch
    gdn_core: bool = False  # with the one-launch verify core
    hop: bool = False  # the decode hop, built for the widest step it covers
    hop_wide: bool = False  # with its two-shot protocol past the one-shot rows
    hop_mid: bool = False  # with its one-shot protocol up to HOP_MID_MAX_ROWS
    gdn_push: bool = False  # out_proj packed for the push GEMM on every GDN layer
    gdn_front: bool = False  # in_proj_qkvz packed for the front on every GDN layer
    route_shared: bool = (
        False  # the router front with the shared stage, every MoE layer
    )
    skv2: bool = False  # SKV2 for every full-attention layer
    mka: bool = False  # the fused prologue and epilogue for every full-attention layer
    mka_wide: bool = False  # with its wide entries (four query heads a rank)
    mka_push: bool = False  # with the epilogue's push into the hop
    lowm_in: bool = False  # the low-M input projections built for every layer
    lowm_out: bool = False  # the low-M output projections built for every layer
    lowm_se: bool = False  # the low-M shared expert built for every MoE layer
    lowm_push: bool = False  # with the output projections' push into the hop
    hop_fp8: bool = False  # the hop's FP8 copy of the norm for the next projection


def slots(rows: int, facts: Facts, tokens_per_request: int = GDN_FRONT_ROWS) -> Slots:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{rows} rows are outside the covered widths 1-{MAX_ROWS}")
    route = facts.route and rows <= ROUTE_MAX_ROWS
    hop = facts.hop and (
        rows <= (HOP_MID_MAX_ROWS if facts.hop_mid else HOP_MAX_ROWS)
        or (facts.hop_wide and rows <= HOP_WIDE_MAX_ROWS)
    )
    mka = facts.mka and rows <= (MKA_WIDE_MAX_ROWS if facts.mka_wide else MKA_MAX_ROWS)
    gdn_verify = facts.gdn_verify and rows <= GDN_VERIFY_MAX_ROWS
    route_topk = facts.route_topk and not route and rows <= ROUTE_TOPK_MAX_ROWS
    return Slots(
        route=route,
        route_topk=route_topk,
        route_gemm=route_topk and facts.route_gemm and rows <= ROUTE_GEMM_MAX_ROWS,
        mkm=route
        and facts.mkm
        and rows <= (MKM_WIDE_MAX_ROWS if facts.mkm_wide else MKM_MAX_ROWS),
        mkm_m32=route_topk
        and facts.mkm_m32
        and MKM_WIDE_MAX_ROWS < rows <= MKM_M32_MAX_ROWS,
        mkm_m64=route_topk
        and facts.mkm_m64
        and MKM_M32_MAX_ROWS < rows <= MKM_M64_MAX_ROWS,
        mkm_m128=route_topk
        and facts.mkm_m128
        and MKM_M64_MAX_ROWS < rows <= MKM_M128_MAX_ROWS,
        gdn_verify=gdn_verify,
        gdn_pdl=gdn_verify and facts.gdn_pdl,
        gdn_core=gdn_verify and facts.gdn_core and rows <= GDN_CORE_MAX_ROWS,
        hop=hop,
        gdn_push=gdn_verify
        and hop
        and facts.gdn_push
        and rows <= GDN_PUSH_MAX_ROWS,
        gdn_front=gdn_verify
        and facts.gdn_pdl
        and facts.gdn_front
        and rows % GDN_FRONT_ROWS == 0
        and rows <= GDN_FRONT_MAX_ROWS,
        route_shared=route and facts.route_shared and rows <= ROUTE_SHARED_ROWS,
        # SKV2 attends one sequence
        skv2=facts.skv2 and rows <= tokens_per_request,
        mka=mka,
        mka_push=mka and facts.mka_push and hop,
        lowm_in=facts.lowm_in and rows <= LOWM_IN_MAX_ROWS,
        lowm_outproj=facts.lowm_out and rows <= LOWM_OUTPROJ_MAX_ROWS,
        lowm_oproj=facts.lowm_out and rows <= LOWM_OPROJ_MAX_ROWS,
        lowm_se=facts.lowm_se and rows <= LOWM_SE_MAX_ROWS,
        lowm_push=facts.lowm_out
        and facts.lowm_push
        and hop
        and rows <= max(LOWM_OUTPROJ_MAX_ROWS, LOWM_OPROJ_MAX_ROWS),
        hop_fp8=hop and facts.hop_fp8 and rows <= HOP_MAX_ROWS,
    )


def plan(
    widths: tuple[int, ...], facts: Facts, tokens_per_request: int = GDN_FRONT_ROWS
) -> dict[int, Slots]:
    return {w: slots(w, facts, tokens_per_request) for w in widths}


def multi_request(tp_size: int, requested: frozenset) -> bool:
    """Whether steps of more than one request are covered: ``MULTI_REQUEST`` requested
    at a TP size in ``MULTI_REQUEST_TP``."""
    return MULTI_REQUEST in requested and tp_size in MULTI_REQUEST_TP


def candidates(
    offered: tuple[int, ...], tokens_per_request: int, multi: bool = True
) -> tuple[int, ...]:
    """The widths SGLang offers (``BuildContext.widths``) of ``REQUESTS`` requests, or of
    one request without ``multi``."""
    requests = REQUESTS if multi else (1,)
    return tuple(
        w
        for w in offered
        if w % tokens_per_request == 0 and w // tokens_per_request in requests
    )


def widest(widths: tuple[int, ...], max_rows: int) -> int:
    """The widest of ``widths`` a slot bounded by ``max_rows`` rows serves (0: none)."""
    return max((w for w in widths if w <= max_rows), default=0)


def covered(
    widths: tuple[int, ...], facts: Facts, tokens_per_request: int
) -> tuple[int, ...]:
    """The widths a build covers: one request always (with no slot it runs the stock
    step), more requests only where a slot runs."""
    return tuple(
        w
        for w in widths
        if w == tokens_per_request or slots(w, facts, tokens_per_request) != Slots()
    )
