# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 decode attention kernels on one SM100-family GPU (GB300):

- swap-AB sparse attention at 1-24 rows as ``attention._sparse_decode`` runs it (PDL,
  narrow combine, combine entry trigger; plus the query RoPE and KV-before-wait on the
  low-ratio layers): bit for bit against the same kernel launched without PDL, against
  SGLang's own 8-row kernel at 1-8 rows (no extras), and within a BF16 bound of a
  float64 reference; eager and replayed from a CUDA graph with fresh metadata;
- the query RoPE inside attention against SGLang's q RoPE kernel followed by the stock
  call, bit for bit;
- the q-LoRA RMSNorm + MXFP8 quant with PDL at 1-48 rows against the launch without PDL
  and SGLang's own kernel at 1-8 rows (BF16 row, codes and swizzled scales, padding
  included) bit for bit, and FlashInfer's rmsnorm (1 ulp) + mxfp8_quantize (bitwise).
"""

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

SWAPAB_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24)
# The plugin's launch (attention._sparse_decode) without and with the low-ratio extras
PLUGIN = dict(max_rows=24, pdl=True, narrow_combine=True, combine_entry_trigger=True)
# The same arithmetic launched as a plain grid: PDL and the triggers only move waits
PLAIN = dict(max_rows=24, narrow_combine=True)
Q_LORA = 1280


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _make_cache(page, pages=11):
    """A padded V4 cache with poisoned unused bytes, and its dequantized values."""
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device="cuda")
    fp8 = torch.randn(pages, page, 448, device="cuda").to(torch.float8_e4m3fn)
    exps = torch.randint(124, 130, (pages, page, 7), device="cuda", dtype=torch.uint8)
    rope = torch.randn(pages, page, 64, device="cuda").to(torch.bfloat16)
    tokens = raw[:, : page * 576].view(pages, page, 576)
    tokens[..., :448].copy_(fp8.view(torch.uint8))
    tokens[..., 448:].copy_(rope.view(torch.uint8))
    raw[:, page * 576 : page * 584].view(pages, page, 8)[..., :7].copy_(exps)
    nope = (
        fp8.float().view(pages, page, 7, 64) * torch.exp2(exps.float() - 127)[..., None]
    )
    values = torch.cat((nope.flatten(-2), rope.float()), -1).reshape(-1, 512)
    return raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1)), values


def _inputs(b, nk=192, ne=512, page=64, seed=0):
    """BEST's swap-AB case: strided query and index rows, out-of-range and negative
    slots, per-row edge lengths, and -inf / +inf / large sinks."""
    torch.manual_seed(20260915 + b + nk + ne + seed)
    storage = torch.full(
        (b, 1, 64, 512), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    q = storage[:, :, :16]
    q.copy_(torch.randn_like(q) * 0.5)
    kv, values = _make_cache(page)
    ekv, evalues = _make_cache(16)
    ids = torch.randint(
        -3, values.shape[0] + 4, (b, 1, nk + 19), device="cuda", dtype=torch.int32
    )[..., :nk]
    ei = torch.randint(
        -3, evalues.shape[0] + 4, (b, 1, ne + 17), device="cuda", dtype=torch.int32
    )[..., :ne]
    lengths = torch.tensor(
        [[0, 1, 31, 63, 64, nk - 1, nk, nk + 3][i % 8] for i in range(b)],
        device="cuda",
        dtype=torch.int32,
    )
    el = torch.tensor(
        [[ne, ne // 2, 1, 0, 63, ne - 1, 65, ne][i % 8] for i in range(b)],
        device="cuda",
        dtype=torch.int32,
    )
    sink = torch.randn(64, device="cuda")
    sink[0], sink[1], sink[2] = -torch.inf, torch.inf, 20.0
    extra = (ekv, ei, el) if ne else (None, None, None)
    return (q, kv, ids, lengths, sink, *extra), (values, evalues)


def _reference(args, values):
    q, _, ids, lengths, sink, _, ei, el = args

    def gather(vals, idx, lens):
        idx = idx[:, 0]
        valid = (idx >= 0) & (idx < vals.shape[0])
        valid &= torch.arange(idx.shape[-1], device=idx.device)[None, :] < lens[:, None]
        data = vals[idx.clamp(0, vals.shape[0] - 1).long()].double()
        return torch.where(valid[..., None], data, 0), valid

    kv, valid = gather(values[0], ids, lengths)
    if ei is not None:
        ev, em = gather(values[1], ei, el)
        kv, valid = torch.cat((kv, ev), 1), torch.cat((valid, em), 1)
    scores = q[:, 0].double() @ kv.transpose(1, 2) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    sinks = sink[None, :16, None].double().expand(q.shape[0], -1, -1)
    prob = torch.nan_to_num(torch.softmax(torch.cat((scores, sinks), -1), -1)[..., :-1])
    return prob @ kv


def _check_reference(out, args, values):
    gold = _reference(args, values)
    floor = (gold.bfloat16().double() - gold).square().mean().sqrt()
    error = (out.double() - gold).square().mean().sqrt()
    assert error.item() <= floor.item() * 1.01 + 1e-6
    torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)


def _bitwise(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
        f"{(a != b).sum().item()} of {a.numel()} differ, "
        f"max |diff| {(a.float() - b.float()).abs().max().item():.3e}"
    )


def _main_swapab():
    try:
        from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
            swapab_attention,
        )
    except ImportError:
        return None
    return swapab_attention


@pytest.mark.parametrize("ne", [512, 0])
@pytest.mark.parametrize("b", SWAPAB_ROWS)
def test_swapab_plugin_launch(b, ne):
    args, values = _inputs(b, ne=ne)
    originals = [x.clone() for x in args if x is not None]
    out = swapab.swapab_attention(*args, **PLUGIN)
    _bitwise(out, swapab.swapab_attention(*args, **PLAIN))
    _check_reference(out, args, values)
    wide = swapab.swapab_attention(*args, max_rows=24)  # main's combine shape
    _check_reference(wide, args, values)
    for before, after in zip(originals, (x for x in args if x is not None)):
        torch.testing.assert_close(before, after, rtol=0, atol=0, equal_nan=True)
    main = _main_swapab()
    if b <= 8 and main is not None:
        _bitwise(wide, main(*args))
        if not ne:  # no extra keys: the narrow combine is not taken
            _bitwise(out, main(*args))


@pytest.mark.parametrize("b", [1, 2, 6, 8, 12, 24])
def test_swapab_graph_replay(b):
    args, values = _inputs(b)
    q, _, ids, lengths, _, _, ei, el = args
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swapab.swapab_attention(*args, **PLUGIN)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = swapab.swapab_attention(*args, **PLUGIN)
    for i in range(8):
        # Device metadata and the query change under the graph: nothing is host-baked
        q.copy_(torch.randn_like(q) * (0.2 + i * 0.05))
        ids.copy_(
            torch.randint(
                -1, values[0].shape[0], ids.shape, device="cuda", dtype=torch.int32
            )
        )
        lengths.copy_(
            torch.randint(
                0, ids.shape[-1] + 1, lengths.shape, device="cuda", dtype=torch.int32
            )
        )
        ei.copy_(
            torch.randint(
                -1, values[1].shape[0], ei.shape, device="cuda", dtype=torch.int32
            )
        )
        el.copy_(
            torch.randint(
                0, ei.shape[-1] + 1, el.shape, device="cuda", dtype=torch.int32
            )
        )
        replayed.zero_()
        graph.replay()
        _bitwise(replayed, swapab.swapab_attention(*args, **PLAIN))
        _check_reference(replayed, args, values)


# 64 rows (65K rotated values) is what catches a reordered RoPE expression: rounding
# order flips about 1 in 30K values; the plugin serves 1-24.
@pytest.mark.parametrize("b", [1, 2, 4, 5, 6, 8, 12, 16, 24, 64])
def test_swapab_q_rope_and_kv_before_wait(b):
    try:
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope
    except ImportError:
        pytest.skip("this SGLang has no fused_q_norm_rope")
    kv, _ = _make_cache(64)
    extra, _ = _make_cache(16)
    angles = torch.rand(70000, 32, device="cuda") * 6.3
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs = torch.view_as_real(freqs_cis).flatten(-2)
    torch.manual_seed(77 + b)
    q_raw = (torch.randn(b, 16, 512, device="cuda") * 3).to(torch.bfloat16)
    positions = torch.randint(70000, (b,), device="cuda", dtype=torch.int64)
    ids = torch.randint(0, 64 * 11, (b, 1, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((b,), 128, device="cuda", dtype=torch.int32)
    ei = torch.randint(0, 16 * 11, (b, 1, 512), device="cuda", dtype=torch.int32)
    el = torch.full((b,), 500, device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda")
    q_pad = torch.zeros(b, 64, 512, device="cuda", dtype=torch.bfloat16)
    fused_q_norm_rope(q_raw, q_pad[:, :16], None, freqs_cis, positions)
    tail = (kv, ids, lengths, sink, extra, ei, el)
    rows = dict(max_rows=64) if b > 24 else {}
    expected = swapab.swapab_attention(q_pad[:, :16], *tail, **{**PLAIN, **rows})
    _bitwise(
        swapab.swapab_attention(q_pad[:, :16], *tail, **{**PLUGIN, **rows}), expected
    )

    def fused(**kwargs):
        return swapab.swapab_attention(
            q_raw, *tail, **{**PLUGIN, **rows}, q_rope=(freqs, positions), **kwargs
        )

    for kv_before_wait in (False, True):
        _bitwise(fused(kv_before_wait=kv_before_wait), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(kv_before_wait=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fused(kv_before_wait=True)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, expected)


def _check_flashinfer(x, w, got):
    """FlashInfer's rmsnorm within 1 BF16 ulp (it reduces in another order: 1-3 of
    10-60K elements differ on GB300), and its swizzled mxfp8_quantize of our row bit
    for bit, padding scales zero."""
    import flashinfer

    y, q, sf = got
    ulps = y.view(torch.int16).int() - flashinfer.norm.rmsnorm(x, w, 1e-6).view(
        torch.int16
    )
    assert ulps.abs().max().item() <= 1
    eq, esf = flashinfer.mxfp8_quantize(y, is_sf_swizzled_layout=True)
    assert torch.equal(q.view(torch.uint8), eq.view(torch.uint8))
    g = torch.arange(Q_LORA // 32, device=x.device)
    row = torch.arange(x.shape[0], device=x.device)[:, None]
    offsets = (g // 4) * 512 + ((row % 32) * 4 + row // 32) * 4 + g % 4
    assert torch.equal(sf[offsets], esf.flatten()[offsets])
    pad = torch.ones_like(sf, dtype=torch.bool)
    pad[offsets] = False
    assert torch.count_nonzero(sf[pad]) == 0


def _same(got, want):
    for a, b in zip(got, want):
        assert a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("stride", [Q_LORA, 1792])
@pytest.mark.parametrize("m", range(1, 49))
def test_rmsnorm_mxfp8_pdl(m, stride):
    torch.manual_seed(941 + m)
    x = torch.randn((m, stride), device="cuda", dtype=torch.bfloat16)[:, :Q_LORA]
    w = torch.randn(Q_LORA, device="cuda", dtype=torch.bfloat16)
    got = rmsnorm_mxfp8(x, w, 1e-6, pdl=True)
    _same(got, rmsnorm_mxfp8(x, w, 1e-6))
    try:
        from sglang.kernels.ops.layernorm.mxfp8_epilogue import (
            rmsnorm_mxfp8 as main_rmsnorm_mxfp8,
        )
    except ImportError:
        main_rmsnorm_mxfp8 = None
    if m <= 8 and main_rmsnorm_mxfp8 is not None:
        _same(got, main_rmsnorm_mxfp8(x, w, 1e-6))
    _check_flashinfer(x, w, got)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = rmsnorm_mxfp8(x, w, 1e-6, pdl=True)
    for scale in (0, 1e-4, 1, 100):
        x.copy_(torch.randn_like(x) * scale)
        w.copy_(torch.randn_like(w))
        for t in replayed:
            t.view(torch.uint8).fill_(0xFF)
        graph.replay()
        _same(replayed, rmsnorm_mxfp8(x, w, 1e-6))
        _check_flashinfer(x, w, replayed)


@pytest.mark.parametrize("m", range(9, 49))
def test_rmsnorm_mxfp8_row_tiles(m):
    """FlashInfer has no row-tile quantize, so the reference is its swizzled
    mxfp8_quantize of each 8-row tile of our BF16 row alone: one 128-row atom per tile
    with the tile's rows at its top, which is the row-tile layout. Codes and the live
    scale bytes are bit for bit; rows past m in the last tile carry zero scales."""
    import flashinfer

    torch.manual_seed(977 + m)
    x = torch.randn((m, Q_LORA), device="cuda", dtype=torch.bfloat16)
    w = torch.randn(Q_LORA, device="cuda", dtype=torch.bfloat16)
    y, q, sf = rmsnorm_mxfp8(x, w, 1e-6, pdl=True, row_tiles=True)
    plain = rmsnorm_mxfp8(x, w, 1e-6)
    assert torch.equal(y, plain[0])
    assert torch.equal(q.view(torch.uint8), plain[1].view(torch.uint8))
    atom = (Q_LORA // 128) * 512
    assert sf.numel() == -(-m // 8) * atom
    eq, _ = flashinfer.mxfp8_quantize(y, is_sf_swizzled_layout=True)
    assert torch.equal(q.view(torch.uint8), eq.view(torch.uint8))
    g = torch.arange(Q_LORA // 32, device=x.device)
    within = (g // 4) * 512 + torch.arange(8, device=x.device)[:, None] * 16 + g % 4
    for t in range(0, m, 8):
        rows = min(8, m - t)
        _, tile_sf = flashinfer.mxfp8_quantize(y[t : t + 8], is_sf_swizzled_layout=True)
        got = sf[(t // 8) * atom + within]
        assert torch.equal(got[:rows], tile_sf.flatten()[within[:rows]]), t
        assert torch.count_nonzero(got[rows:]) == 0, t
