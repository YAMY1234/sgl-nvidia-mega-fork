# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 TP4 attention projection wq_b (N 8192, K 1280) on FlashInfer's
low-latency MXFP8 GEMM launched directly (``mxfp8_ll.linear``), on one SM100-family
GPU (GB300):

- against SGLang's MXFP8 path for block-FP8-as-MXFP8 linears (FlashInfer ``mm_mxfp8``
  ``cute-dsl`` with tuning off) on the same MXFP8 operands at 1-8 rows (plain swizzled
  scales) and 9-48 rows (row-tile scales): relative error of the BF16 output; past 8
  rows also bit for bit against the same tactic run one 8-row tile at a time;
- eager against a CUDA graph replay, bit for bit; every pinned tactic is one FlashInfer
  offers.

The measured tolerances print as ``GEMM_REL`` lines (``pytest -s``).
"""

import json
import math

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

SHAPES = {"wq_b": (8192, 1280)}
A03_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)
A26_ROWS = (9, 12, 16, 17, 24, 32, 40, 48)
# The BF16 outputs differ only by FP32 summation order (split-K, K tile)
GEMM_NORM_REL = 5e-4


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return mxfp8_ll.supported()


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _synthetic_weight(n, k, gen):
    w = torch.randn(n, k, device="cuda", generator=gen) / math.sqrt(k)
    w_q, w_sf = mxfp8_ll.quantize(w.bfloat16())
    return w_q.contiguous(), w_sf.reshape(-1).contiguous()


def _activations(rows, k, gen, norm_weight=None):
    """Rows as a pre-norm feeds them to a projection: heavy-tailed channels (a few
    massive ones), RMS-normalized, times the norm weight; BF16."""
    gain = torch.exp(0.75 * torch.randn(k, device="cuda", generator=gen))
    gain[torch.randperm(k, device="cuda", generator=gen)[:8]] *= 30.0
    h = torch.randn(rows, k, device="cuda", generator=gen) * gain
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    if norm_weight is not None:
        y = y * norm_weight.float()
    return y.bfloat16()


def _cute_dsl(a, a_sf, w, w_sf):
    """SGLang's MXFP8 linear for block-FP8-as-MXFP8 weights (fp8_utils
    ``flashinfer_mxfp8_blockscaled_linear(backend="cute-dsl", pin_tactic=True)``)."""
    from flashinfer import mm_mxfp8
    from flashinfer.autotuner import autotune

    with autotune(False, skip_ops={"mxfp8_gemm"}):
        return mm_mxfp8(
            a,
            w.t(),
            a_sf.reshape(-1),
            w_sf,
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
        )


def quantize_row_tiles(x):
    """FlashInfer's swizzled ``mxfp8_quantize`` of each 8-row tile on its own: codes
    [M, K] and the row-tile scale layout (one 128-row atom per tile, rows at its top)."""
    tiles = [mxfp8_ll.quantize(x[t : t + 8]) for t in range(0, x.shape[0], 8)]
    codes = torch.cat([q for q, _ in tiles])
    return codes, torch.cat([s.reshape(-1) for _, s in tiles])


def _ll(x, w, w_sf, tactic=None):
    """The plugin's arm: MXFP8 rows as the layer input arrives (plain swizzled scales
    up to 8 rows, row-tile scales past 8)."""
    if x.shape[0] <= mxfp8_ll.MAX_ROWS:
        a, a_sf = mxfp8_ll.quantize(x)
    else:
        a, a_sf = quantize_row_tiles(x)
    return mxfp8_ll.linear(a, a_sf.reshape(-1), w, w_sf, tactic=tactic), a


def _rel(out, ref):
    d = (out.float() - ref.float()).abs()
    return (d.norm() / ref.float().norm()).item(), d.max().item()


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(7)
    return {name: _synthetic_weight(n, k, gen) for name, (n, k) in SHAPES.items()}


def test_pinned_tactics_are_offered():
    sm = _sm()
    for (pin_sm, n, k, bucket), tactic in mxfp8_ll.BUCKET_TACTICS.items():
        if pin_sm != sm:
            continue
        offered = mxfp8_ll.offered_tactics(n, k, min(bucket, mxfp8_ll.MAX_ROWS))
        assert tactic in offered, (n, k, bucket, tactic)
        assert mxfp8_ll.select_tactic(sm, n, k, bucket) == tactic
    for n, k in SHAPES.values():
        assert (
            mxfp8_ll.select_tactic(sm, n, k, 8) == mxfp8_ll.PINNED_TACTICS[(sm, n, k)]
        )


@pytest.mark.parametrize("name", list(SHAPES))
def test_low_latency_matches_cute_dsl(weights, name):
    n, k = SHAPES[name]
    w, w_sf = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(11)
    worst = {}
    for rows in A03_ROWS + A26_ROWS:
        x = _activations(rows, k, gen)
        a, a_sf = mxfp8_ll.quantize(x)
        ref = _cute_dsl(a, a_sf, w, w_sf)
        out, a_ll = _ll(x, w, w_sf)
        torch.testing.assert_close(
            a_ll.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0
        )
        rel, max_abs = _rel(out, ref)
        print(f"GEMM_REL {name} rows={rows} norm_rel={rel:.3e} max_abs={max_abs:.3e}")
        assert rel <= GEMM_NORM_REL, (rows, rel)
        key = "1-8" if rows <= 8 else "9-48"
        worst[key] = max(worst.get(key, 0.0), rel)
        if rows > mxfp8_ll.MAX_ROWS:
            # one CTA per 8-row tile: the same tactic on each tile alone gives the same bits
            tactic = mxfp8_ll.select_tactic(_sm(), n, k, mxfp8_ll.row_bucket(rows))
            for t in range(0, rows, 8):
                tile, _ = _ll(x[t : t + 8], w, w_sf, tactic=tactic)
                assert torch.equal(tile, out[t : t + 8]), (rows, t)
    print(f"GEMM_REL_WORST {name} {json.dumps(worst)}")


@pytest.mark.parametrize("name", list(SHAPES))
def test_cuda_graph_matches_eager(weights, name):
    n, k = SHAPES[name]
    w, w_sf = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(13)
    rows_list = (1, 2, 4, 6, 8, 12, 24, 48)
    mxfp8_ll.precompile(w, list(rows_list))
    for rows in rows_list:
        x = _activations(rows, k, gen)
        if rows <= mxfp8_ll.MAX_ROWS:
            a, a_sf = mxfp8_ll.quantize(x)
        else:
            a, a_sf = quantize_row_tiles(x)
        a_sf = a_sf.reshape(-1)
        eager = mxfp8_ll.linear(a, a_sf, w, w_sf)
        out = torch.zeros_like(eager)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mxfp8_ll.linear(a, a_sf, w, w_sf, out=out)  # warm-up off the capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mxfp8_ll.linear(a, a_sf, w, w_sf, out=out)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), rows
