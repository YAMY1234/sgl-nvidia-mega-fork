# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_conv_core_cases", Path(__file__).with_name("test_qwen3_5_gdn_conv_core.py")
)
_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cases)
_Case, _same, _gpu_ready = _cases._Case, _cases._same, _cases._gpu_ready

gpu = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")
BS = [1, 2, 3, 4, 5, 6, 8, 12, 16]


def _plain(c, out, qkvz=None, **kw):
    """Slot gdn_core's launch on rows convolved by slot gdn_conv's (run on ``out``'s state)."""
    from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import core

    qkv, z = c.views(c.qkvz if qkvz is None else qkvz)
    y = conv_update(
        qkv.view(c.bs, c.T, -1).transpose(1, 2),
        out["state"],
        c.weight,
        c.bias,
        "silu",
        c.states,
        c.windows(out["shared"]),
        c.windex,
        block_n=128,
        token_major_out=True,
        hoist=True,
        window_mode=1,
    )
    core(
        *c._core_args(y.transpose(1, 2).view(c.tokens, -1), z, out),
        rows_per_request=c.T,
        num_k_heads=c.H,
        o_out=out["o"],
        **kw,
    )


def _fused(c, out, arrive, split, qkvz=None, **kw):
    """Slot gdn_conv_core's launch with the value split's counters ``split``."""
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import Conv, core

    qkv, z = c.views(c.qkvz if qkvz is None else qkvz)
    core(
        *c._core_args(qkv, z, out),
        rows_per_request=c.T,
        num_k_heads=c.H,
        o_out=out.get("o"),
        conv=Conv(
            c.weight,
            c.bias,
            out["state"],
            c.windows(out["shared"]),
            c.windex,
            arrive,
            split,
        ),
        **kw,
    )


@gpu
@pytest.mark.parametrize("bs", BS)
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("conv", [False, True])
def test_vsplit_writes_the_one_cta_bytes(conv, pad, history, bs):
    if pad and bs == 1:
        pytest.skip("a padded single request leaves nothing to compare")
    c = _Case(
        8,
        bs,
        7,
        7000 + 10 * bs + 4 * conv + 2 * history + pad,
        history=history,
        pad=pad,
    )
    want, got = c.outputs(), c.outputs()
    if conv:
        a1, a2 = c.arrive(), c.arrive()
        c.fused(want, a1, vsplit=1)
        _fused(c, got, c.arrive(), a2, vsplit=2)
        assert torch.equal(2 * a1, a2)
    else:
        _plain(c, want, vsplit=1)
        _plain(c, got, vsplit=2)
    _same(got, want)
    assert got["y"].view(torch.uint8).any()


@gpu
@pytest.mark.parametrize("conv", [False, True])
def test_vsplit_flush_request(conv):
    c = _Case(8, 4, 7, 4343 + conv, flush=True)
    want, got = c.outputs(), c.outputs()
    if conv:
        c.fused(want, c.arrive(), vsplit=1)
        _fused(c, got, c.arrive(), c.arrive(), vsplit=2)
    else:
        _plain(c, want, vsplit=1)
        _plain(c, got, vsplit=2)
    _same(got, want)
    assert not got["y"][:7].float().any()


@gpu
@pytest.mark.parametrize(
    "variant",
    [
        {"early_dependents": True},
        {"min_blocks": 3},
        {"flat": False},
        {"early_hist": True},
        {"early_dependents": True, "min_blocks": 3},
    ],
)
@pytest.mark.parametrize("conv", [False, True])
@pytest.mark.parametrize("bs", [2, 8])
def test_vsplit_variants(bs, conv, variant):
    if conv and variant.get("early_hist"):
        pytest.skip("the conv-first launch keeps its history loads after the wait")
    c = _Case(8, bs, 7, 8100 + bs + 3 * conv + len(variant), pad=bs > 4)
    want, got = c.outputs(), c.outputs()
    if conv:
        c.fused(want, c.arrive(), vsplit=1, **variant)
        _fused(c, got, c.arrive(), c.arrive(), vsplit=2, **variant)
    else:
        _plain(c, want, vsplit=1, **variant)
        _plain(c, got, vsplit=2, **variant)
    _same(got, want)


@gpu
def test_vsplit_and_one_cta_launches_alternate_on_the_same_pools():
    for bs in (3, 8, 16):
        c = _Case(8, bs, 7, 9100 + bs, pad=bs > 4, flush=True)
        want, got = c.outputs(), c.outputs()
        arrive, split = c.arrive(), c.arrive()
        for i in range(6):
            qkvz = c.rand(c.tokens, c.D + c.HV * 128)
            c.stock(want, qkvz=qkvz)
            _fused(
                c,
                got,
                arrive,
                split,
                qkvz=qkvz,
                vsplit=1 + (i % 2),
                early_dependents=i >= 3,
            )
            _same(got, want)
        grp = c.HV // c.H
        live = c.states.long() >= 0
        assert (arrive.view(c.bs, c.H)[live] == 3 * grp).all()
        assert (split.view(c.bs, c.H)[live] == 3 * 2 * grp).all()
        assert (
            not arrive.view(c.bs, c.H)[~live].any()
            and not split.view(c.bs, c.H)[~live].any()
        )


@gpu
def test_vsplit_conv_needs_its_counters():
    c = _Case(8, 2, 7, 31)
    with pytest.raises(ValueError, match="arrive_split"):
        c.fused(c.outputs(), c.arrive(), vsplit=2)


@gpu
@pytest.mark.parametrize("conv", [False, True])
def test_vsplit_replays_from_a_cuda_graph(conv):
    c = _Case(8, 6, 7, 5150 + conv, pad=True)
    qkvz = c.qkvz.clone()
    got = c.outputs()
    arrive = c.arrive()

    def launch(out):
        if conv:
            _fused(c, out, c.arrive(), arrive, qkvz=qkvz, vsplit=2)
        else:
            _plain(c, out, qkvz=qkvz, vsplit=2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch(c.outputs())
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(got)
    for _ in range(3):
        qkvz.copy_(c.rand(*qkvz.shape))
        got["state"].copy_(c.rand(*c.state.shape))
        got["shared"].copy_(c.rand(*c.shared.shape))
        want = dict(
            state=got["state"].clone(),
            shared=got["shared"].clone(),
            ring=[t.clone() for t in got["ring"]],
            y=torch.zeros_like(got["y"]),
            o=torch.zeros_like(got["o"]),
        )
        if conv:
            c.stock(want, qkvz=qkvz)
        else:
            _plain(c, want, qkvz=qkvz, vsplit=1)
        graph.replay()
        torch.cuda.synchronize()
        _same(got, want)


def test_vsplit_widths():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import core as k

    lo, hi = k.VSPLIT_ROWS
    assert k.vsplit_for(lo, 8) == 2 and k.vsplit_for(hi, 8) == 2
    assert k.vsplit_for(lo - 1, 8) == 1 and k.vsplit_for(hi + 1, 8) == 1
    assert k.vsplit_for(lo, 16) == 1 and k.vsplit_for(lo, 4) == 1


@gpu
@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("i64", [False, True])
@pytest.mark.parametrize("conv", [False, True])
def test_vsplit_rows_a_request_and_index_widths(conv, i64, T):
    """4 and 8 rows a request (the conv's bounds), int32 and int64 indices."""
    c = _Case(8, 5, T, 6100 + 10 * T + 2 * i64 + conv, i64=i64)
    want, got = c.outputs(), c.outputs()
    if conv:
        c.fused(want, c.arrive(), vsplit=1)
        _fused(c, got, c.arrive(), c.arrive(), vsplit=2)
    else:
        _plain(c, want, vsplit=1)
        _plain(c, got, vsplit=2)
    _same(got, want)


@gpu
@pytest.mark.parametrize("bs", [3, 8])
def test_vsplit_conv_padded_windows_and_strided_states(bs):
    c = _Case(8, bs, 7, 6600 + bs, pad=True)
    padded = c.rand(c.slots, c.D, c.T + 6)
    c.shared = padded[:, :, : c.T + 2]
    wide = c.rand(c.slots, c.D, 4)
    c.state = wide[:, :, :3]
    want, got = c.outputs(), c.outputs()
    for out in (want, got):
        w, s = padded.clone(), wide.clone()
        out["shared"], out["state"] = w[:, :, : c.T + 2], s[:, :, :3]
        out["_keep"] = (w, s)
    c.fused(want, c.arrive(), vsplit=1)
    _fused(c, got, c.arrive(), c.arrive(), vsplit=2)
    _same(got, want)
    assert torch.equal(got["_keep"][0], want["_keep"][0])
    assert torch.equal(got["_keep"][1], want["_keep"][1])
