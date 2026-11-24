# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Device side of the engines' die-local form (``localize``; host side: ``locality.py``).

Slot claim, per launch: crank 0 of each cluster reads its SM's locality domain d from
the layer's locality block and claims a domain-local rank r with one atomic on the
block's claim word (the launch's last claimer resets the word); the virtual slot is
(d, r) while r < n_d, else (1 - d, N - 1 - r) (n_0 from the block, n_1 = N - n_0), which
takes every slot exactly once whatever the SM placement. Crank 0 pushes the slot to its
peers' shared memory (one 16-byte ``st.async`` each).

Slot list (``slot_entry``, the Python model ``locality.slot_list``): slots 0, 1 of domain
d run shared quarters 2d, 2d + 1 first (schedule weight ``sw``); domain d's routed units
(its experts' four quarters, the domain's list order) go round-robin over its other
slots up to a per-slot cap, the rest to its two shared slots; the units of the busier
domain beyond its slots' capacity (its list tail) are dealt after the other domain's
own units."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64

from ..primitives import ptx_core as C
from . import _core_ptx as P
from .constants import LOC_CTR, LOC_DIE, LOC_N0

L_BCLAIM = 0  # mbarrier, in the (unused) schedule-table space
L_MCLAIM = 16  # the 16-byte slot word


@cute.jit
def claim_issue(loc, nclusters):
    """(claim word, domain, n_0) of the calling thread's cluster (one thread per
    cluster): the domain load, then the atomic."""
    die = P.ld_nc_u8(loc + LOC_DIE + Int64(P.smid())) & 1
    n0 = P.ld_nc_u8(loc + LOC_N0 + Int64(nclusters))
    claim = P.atom_add_u64(
        loc + LOC_CTR, (Uint64(1) << Uint64(32)) | (Uint64(1) << Uint64(16 * die))
    )
    return claim, die, n0


@cute.jit
def claim_publish(loc, nclusters, claim, die, n0, stab, cluster: cutlass.Constexpr[int]):
    """Crank 0, one lane, after the cluster barrier: the virtual slot word (vd | vr << 8
    | n_0 << 16) into this CTA's and the peers' slot words (``stab``: the schedule-table
    space). A claim word that already counted N claims (a launch overlapping another
    launch of the same layer) or a block without this grid size traps."""
    rk = Int32((claim >> Uint64(16 * die)) & Uint64(0xFFFF))
    before = Int32((claim >> Uint64(32)) & Uint64(0xFFFF))
    P.trap_if(Int32((before >= nclusters) | (n0 < 3)))
    P.trap_if(Int32(nclusters - n0 < 3))
    if before == nclusters - 1:
        P.atom_exch_u64(loc + LOC_CTR, Uint64(0))
    nd = n0
    if die != 0:
        nd = nclusters - n0
    cvd = die
    cvr = rk
    if rk >= nd:
        cvd = 1 - die
        cvr = nclusters - 1 - rk
    cword = cvd | (cvr << 8) | (n0 << 16)
    C.st_shared_u32(stab + L_MCLAIM, cword)
    for pr in cutlass.range_constexpr(1, cluster):
        P.st_async_v4_b32(
            C.mapa_shared_cluster(stab + L_MCLAIM, pr),
            C.mapa_shared_cluster(stab + L_BCLAIM, pr),
            cword,
            0,
            0,
            0,
        )
    C.mbarrier_arrive(stab + L_BCLAIM)


@cute.jit
def slot_entry(vd, vr, n0, ncl, sw, u0, u1, k):
    """Entry k of virtual slot (vd, vr)'s list for u0 / u1 routed units in domain 0 /
    1: (list length, entry domain, entry's unit index in that domain's list); for a
    shared slot entry 0 is its shared quarter (the returned unit is then meaningless)."""
    m0 = n0
    m1 = ncl - n0
    cap = (u0 + u1 + 4 * sw + ncl - 1) // ncl
    x0 = u0 + 2 * sw - m0 * cap
    if x0 < 0:
        x0 = Int32(0)
    x1 = u1 + 2 * sw - m1 * cap
    if x1 < 0:
        x1 = Int32(0)
    md = m0
    ud = u0
    xd = x0
    uo = u1
    xo = x1
    if vd != 0:
        md = m1
        ud = u1
        xd = x1
        uo = u0
        xo = x0
    own = ud - xd
    tot = own + xo
    nr = md - 2
    lim = nr * cap
    if tot < lim:
        lim = tot
    idx = (vr - 2) + nr * k
    nraw = Int32(0)
    if vr >= 2:
        if lim > vr - 2:
            nraw = (lim - (vr - 2) + nr - 1) // nr
    else:
        idx = lim + vr + 2 * (k - 1)
        nraw = Int32(1)
        if tot - lim - vr > 0:
            nraw = 1 + (tot - lim - vr + 1) // 2
    cdie = vd
    u = idx
    if idx >= own:
        cdie = 1 - vd
        u = (uo - xo) + (idx - own)
    if u < 0:
        u = Int32(0)
    return nraw, cdie, u


@cute.jit
def wait_slot(stab):
    """The slot word (every thread of the CTA, once published)."""
    C.mbarrier_wait_parity(stab + L_BCLAIM, 0)
    w = Int32(C.ld_shared_u32(stab + L_MCLAIM))
    return w & 1, (w >> 8) & 0xFF, (w >> 16) & 0xFF


__all__ = ["L_BCLAIM", "L_MCLAIM", "claim_issue", "claim_publish", "slot_entry", "wait_slot"]
