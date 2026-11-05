# SPDX-License-Identifier: Apache-2.0
"""Capture-owned FN HC4 bs1 inputs, published by the preceding extend graph.

Native argmax runs before publish. STEP2 remains the authority for whether its
length/slot staging can be used; a changed batch loads current inputs instead.
"""

import triton
import triton.language as tl


@triton.jit
def _publish(
    Token,
    Hidden,
    Select,
    Seed,
    Seq,
    Pos,
    Req,
    Loc,
    MPos,
    Weight,
    DToken,
    DP,
    DHidden,
    DSeed,
    DSeq,
    DPos,
    DReq,
    DLoc,
    DMPos,
    Embedding,
    Chain,
    H: tl.constexpr,
    HS: tl.constexpr,
    SEED: tl.constexpr,
    SEED_STRIDE: tl.constexpr,
    SELECTED: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MSTRIDE: tl.constexpr,
    DMSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    x = tile * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(Token).to(tl.int64)
    row = tl.load(Select).to(tl.int64) if SELECTED else 0
    hidden = tl.load(Hidden + row * HS + x, x < HS, other=0)
    tl.store(DHidden + x, hidden, x < HS)
    emb = tl.load(Weight + token * H + x, x < H, other=0)
    tl.store(Embedding + x, emb, x < H)
    if SEED:
        seed = tl.load(Seed + row * SEED_STRIDE + x, x < SEED, other=0)
        tl.store(DSeed + x, seed, x < SEED)
    if tile == 0:
        tl.store(DToken, token)
        tl.store(DP, 1.0)
        tl.store(Chain, token)
        tl.store(DSeq, tl.load(Seq))
        tl.store(DPos, tl.load(Pos))
        tl.store(DReq, tl.load(Req))
        loc = tl.load(Loc + x, x < 3, other=0)
        tl.store(DLoc + x, loc, x < 3)
        if HAS_MROPE:
            mpos = tl.load(MPos + x * MSTRIDE, x < 3, other=0)
        else:
            mpos = tl.full((BLOCK,), 0, tl.int64)
        tl.store(DMPos + x * DMSTRIDE, mpos, x < 3)
