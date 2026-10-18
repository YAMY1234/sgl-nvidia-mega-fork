# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The stock mHC boundaries on CPU: they call SGLang's mHC functions with the
arguments SGLang's stage boundary passes, in its order."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

mhc = pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import Boundaries  # noqa: E402

ROWS, HC, HIDDEN = 6, 4, 8


def _layer(layer_id, *, engram=None, mxfp8=True):
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    return SimpleNamespace(
        layer_id=layer_id,
        engram=engram,
        boundary_mhc_fusions=True,
        boundary_pre_from_prev=True,
        hc_stats_stream=object(),
        hc_cfg=SimpleNamespace(hidden=HIDDEN),
        attn_hc=SimpleNamespace(name=f"attn{layer_id}"),
        ffn_hc=SimpleNamespace(name=f"ffn{layer_id}"),
        input_layernorm=SimpleNamespace(norm=norm, name=f"in{layer_id}"),
        post_attention_layernorm=SimpleNamespace(norm=norm, name=f"post{layer_id}"),
        self_attn=SimpleNamespace(
            attn_tp_size=4, accepts_mxfp8_swizzled_input=lambda: mxfp8
        ),
        mlp=SimpleNamespace(tp_size=4),
    )


def _boundary(fake, layers, **kw):
    def make_boundary(norm, *, accepts_mxfp8, hc=None):
        return ("next", norm.name, accepts_mxfp8, hc.name)

    fake.make_boundary.side_effect = make_boundary
    with patch("sglang.srt.models.deepseek_v4_mhc", fake):
        return Boundaries(layers, **kw)


@pytest.fixture
def fake():
    calls = []
    m = Mock()
    m.HcState = mhc.HcState
    m.AttnOutput = mhc.AttnOutput
    m.use_stats_stream.return_value = True
    m.fork_stats_stream.side_effect = lambda s: calls.append(("fork", s))

    def combine(hc, state, quantized):
        calls.append(("combine", hc.name, quantized is not None))
        if quantized is not None:
            quantized.append("q")
        return "x"

    m.combine.side_effect = combine
    m.run_attn_post.side_effect = lambda *a, **k: mhc.HcState(torch.ones(1), "pre")
    m.run_moe_post.side_effect = lambda *a, **k: mhc.HcState(torch.zeros(1), "pre")
    m.calls = calls
    # the fused collective's widths unless a test says otherwise
    with patch.object(Boundaries, "defers_sum", return_value=True):
        yield m


def test_a_layer_runs_sglangs_mhc_calls_in_order(fake):
    layers = [_layer(0), _layer(1), _layer(2, engram=object())]
    b = _boundary(fake, layers)
    streams = torch.randn(ROWS, HC, HIDDEN)
    st = b.start(streams)
    batch = object()
    layer = layers[0]

    x, xq = b.pre(layer, "attn", st, batch)
    assert (x, xq) == ("x", "q")
    assert fake.calls == [("fork", layer.hc_stats_stream), ("combine", "attn0", True)]
    fake.use_stats_stream.assert_called_with(layer.hc_cfg, batch, streams)

    partial = torch.randn(ROWS, HIDDEN)
    b.post(layer, "attn", partial, st)
    args, kw = fake.run_attn_post.call_args
    assert args[0] is layer.attn_hc and args[2] is streams
    assert isinstance(args[1], mhc.AttnOutput) and args[1].partial is partial
    assert kw == dict(
        stats_stream=layer.hc_stats_stream,
        next=("next", "post0", False, "ffn0"),
        world_size=4,
        precomputed=None,
    )

    fake.calls.clear()
    x, xq = b.pre(layer, "ffn", st, batch)
    assert xq is None and fake.calls[1] == ("combine", "ffn0", False)
    moe_out = object()
    b.post(layer, "ffn", moe_out, st)
    args, kw = fake.run_moe_post.call_args
    assert args[0] is layer.ffn_hc and args[1] is moe_out
    assert kw["next"] == ("next", "in1", True, "attn1") and kw["world_size"] == 4

    # the MoE post before an Engram layer and the last one leave the streams only
    for layer in layers[1:]:
        b.pre(layer, "attn", st, batch)
        b.post(layer, "attn", partial, st)
        b.pre(layer, "ffn", st, batch)
        b.post(layer, "ffn", moe_out, st)
        assert fake.run_moe_post.call_args.kwargs["next"] is None


def test_the_attention_post_takes_sglangs_attention_output_as_is(fake):
    layers = [_layer(0)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    b.pre(layers[0], "attn", st, object())
    partial = torch.randn(ROWS, HIDDEN)
    b.post(layers[0], "attn", mhc.AttnOutput(partial), st)
    assert fake.run_attn_post.call_args.args[1].partial is partial


def test_without_cross_layer_fusion_no_moe_post_builds_the_next_input(fake):
    layers = [_layer(0), _layer(1)]
    b = _boundary(fake, layers, cross_layer=False)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", object(), st)
    assert fake.run_moe_post.call_args.kwargs["next"] is None


def test_an_attention_without_mxfp8_input_gets_no_quantized_rows(fake):
    layers = [_layer(0, mxfp8=False), _layer(1, mxfp8=False)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    assert b.pre(layers[0], "attn", st, object()) == ("x", None)
    b.post(layers[0], "attn", torch.randn(ROWS, HIDDEN), st)
    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", object(), st)
    assert fake.run_moe_post.call_args.kwargs["next"][2] is False


def test_a_post_needs_its_pre(fake):
    b = _boundary(fake, [_layer(0)])
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    with pytest.raises(RuntimeError):
        b.post(_layer(0), "attn", torch.randn(ROWS, HIDDEN), st)
    b.pre(_layer(0), "attn", st, object())
    with pytest.raises(RuntimeError):
        b.pre(_layer(0), "ffn", st, object())


def test_past_the_push_plane_the_attention_sum_runs_before_a_plain_post(fake):
    """Above the fused collective's widths SGLang's transport all-reduces the wo_b
    rows on the attention TP group; handing the post the partial would run the
    collective kernel past its push plane."""
    layers = [_layer(0)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(200, HC, HIDDEN))
    b.pre(layers[0], "attn", st, object())
    partial, summed = torch.randn(200, HIDDEN), torch.randn(200, HIDDEN)
    group = SimpleNamespace(all_reduce=Mock(return_value=summed))
    with (
        patch.object(Boundaries, "defers_sum", return_value=False) as defers,
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(attn_tp_group=group),
        ),
    ):
        b.post(layers[0], "attn", partial, st)
    defers.assert_called_once_with(layers[0], "attn", 200)
    group.all_reduce.assert_called_once_with(partial)
    assert fake.run_attn_post.call_args.args[1] is summed


def test_hop_a_collects_a_published_partial_with_sglangs_triplet_and_norm(fake):
    """A published wo_b partial is collected by the Lamport consumer with the
    triplet SGLang's post would compute and the FFN norm; anything else takes
    SGLang's post."""
    triplet = (torch.ones(ROWS, HC), torch.ones(ROWS, HC), torch.ones(ROWS, HC, HC))
    fake.mix_stats.return_value = triplet
    fake.HcNormed = mhc.HcNormed
    updated, normed = torch.zeros(ROWS, HC, HIDDEN), torch.ones(ROWS, HIDDEN)
    published = torch.randn(ROWS, HIDDEN)
    hop = SimpleNamespace(
        published=lambda t: t is published,
        collect=Mock(return_value=(updated, normed)),
    )
    layers = [_layer(0)]
    with patch("sglang.srt.models.deepseek_v4_mhc", fake):
        b = Boundaries(layers, hop_a=hop)
    fake.make_boundary.return_value = None
    b._ffn_next[0] = SimpleNamespace(
        norm=SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    )
    streams = torch.randn(ROWS, HC, HIDDEN)
    st = b.start(streams)
    b.pre(layers[0], "attn", st, object())
    with patch.object(torch.cuda, "current_stream") as current:
        b.post(layers[0], "attn", mhc.AttnOutput(published), st)
    current.return_value.wait_stream.assert_called_once_with(layers[0].hc_stats_stream)
    fake.mix_stats.assert_called_once_with(
        layers[0].attn_hc, streams, layers[0].hc_stats_stream
    )
    args = hop.collect.call_args.args
    assert args[0] is published and args[1] is streams
    assert args[2:5] == (triplet[1], triplet[2], triplet[0])
    assert args[6] == 1e-6
    assert st.hc.streams is updated and st.hc.input.rows is normed
    assert st.hc.pre is triplet[0]
    fake.run_attn_post.assert_not_called()

    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", object(), st)
    b.pre(layers[0], "attn", st, object())
    b.post(layers[0], "attn", torch.randn(ROWS, HIDDEN), st)
    fake.run_attn_post.assert_called_once()
