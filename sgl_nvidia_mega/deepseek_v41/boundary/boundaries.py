# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4_mhc.py and deepseek_v4_boundary.py (the decode stage boundaries).
"""The mHC boundaries of a covered step: SGLang's stock decode boundary, with
hop A's Lamport collect as the attention post where wo_b published.

Between two sublayers, SGLang's stage boundary for DeepSeek-V4.1 with the mHC
pre-mix carried from the previous sublayer:

1. ``pre``: forks the mHC stats stream, then collapses the residual streams into
   the sublayer's input (combine + RMSNorm, MXFP8 too where the attention takes
   it), or takes the input the previous post already built;
2. the sublayer runs; its TP sum is left to the post;
3. ``post``: predicts the next mix triplet on the stats stream, joins it, and runs
   the fused all-reduce + mHC post, which also builds the next sublayer's input
   where the seam allows (the FFN norm after attention; the next layer's input
   norm, MXFP8 at up to 128 rows, after the MoE).

Each step calls SGLang's own mHC functions with the arguments its stage boundary
passes, so with hop A off the streams, inputs and coefficients are SGLang's bit
for bit. Hop A (``hop_a.py``) replaces only the attention post's collective.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

# The SGLang names this module reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/models/deepseek_v4_mhc.py": (
        "HcState",
        "AttnOutput",
        "make_boundary",
        "use_stats_stream",
        "mix_stats",
        "HcNormed",
        "fork_stats_stream",
        "combine",
        "run_attn_post",
        "run_moe_post",
        "release",
        "residual",
        "stats",
        "materialized",
        "with_residual",
    ),
    "srt/models/deepseek_v4.py": (
        "boundary_mhc_fusions",
        "boundary_pre_from_prev",
        "hc_stats_stream",
        "hc_cfg",
        "attn_hc",
        "ffn_hc",
        "input_layernorm",
        "post_attention_layernorm",
        "accepts_mxfp8_swizzled_input",
        "attn_tp_size",
        "engram",
        "_can_fuse_attn_mhc",
        "_can_fuse_ffn_mhc",
        # the module alias SGLang's DSV4 model imports too
        "mhc",
    ),
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": ("can_fuse_all_reduce",),
    "srt/models/deepseek_v2.py": ("tp_size",),
    "srt/runtime_context.py": ("get_parallel", '"attn_tp_group"'),
}


class Open(msgspec.Struct):
    """A sublayer between its pre and its post: the residual streams its post
    mixes into, and the mix triplet's stream."""

    streams: torch.Tensor
    stats: Optional[tuple] = None
    stats_stream: Optional[torch.cuda.Stream] = None


class BoundaryState(msgspec.Struct):
    """The mHC state of one step: the streams between two hyper-connections
    (SGLang's ``HcState``), and the open sublayer, if any."""

    hc: object  # deepseek_v4_mhc.HcState
    open: Optional[Open] = None


class Boundaries:
    """The boundaries of the target model's layers, built once from the loaded
    model: each layer's next-sublayer boundaries are load-time facts."""

    def __init__(
        self, layers, *, cross_layer: bool = True, hop_a=None, hop_a_reason=""
    ):
        from sglang.srt.models import deepseek_v4_mhc as mhc

        self._mhc = mhc
        self.cross_layer = cross_layer
        self.hop_a = hop_a  # hop_a.HopA, or None
        self.hop_a_reason = hop_a_reason
        self._attn_mxfp8 = {}
        self._ffn_next = {}  # the attention post's next: the FFN norm
        self._attn_next = {}  # the next layer's attention boundary (None: closed)
        ids = [layer.layer_id for layer in layers]
        for layer in layers:
            self._attn_mxfp8[layer.layer_id] = (
                layer.self_attn.accepts_mxfp8_swizzled_input()
            )
            self._ffn_next[layer.layer_id] = (
                mhc.make_boundary(
                    layer.post_attention_layernorm,
                    accepts_mxfp8=False,
                    hc=layer.ffn_hc,
                )
                if layer.boundary_mhc_fusions
                else None
            )
        for prev, layer in zip(ids, layers[1:]):
            self._attn_next[prev] = self._attention_boundary(layer)
        self._attn_next[ids[-1]] = None

    def _attention_boundary(self, layer):
        # SGLang reads an Engram layer's attention input after the injection,
        # so the MoE post before it leaves the streams only
        if not layer.boundary_mhc_fusions or layer.engram is not None:
            return None
        return self._mhc.make_boundary(
            layer.input_layernorm,
            accepts_mxfp8=self._attn_mxfp8[layer.layer_id],
            hc=layer.attn_hc,
        )

    def start(self, streams: torch.Tensor) -> BoundaryState:
        """The state before layer 0: the embedding broadcast to the hc streams."""
        return BoundaryState(self._mhc.HcState(streams))

    def pre(self, layer, kind: str, st: BoundaryState, forward_batch):
        """The sublayer's input: (rows, MXFP8 rows or None)."""
        mhc = self._mhc
        attention = _attention(kind)
        if st.open is not None:
            raise RuntimeError("a sublayer is still open: its post did not run")
        state = st.hc
        streams = state.residual
        stream = (
            layer.hc_stats_stream
            if mhc.use_stats_stream(layer.hc_cfg, forward_batch, streams)
            else None
        )
        mhc.fork_stats_stream(stream)
        quantized = [] if attention and self._attn_mxfp8[layer.layer_id] else None
        hc = layer.attn_hc if attention else layer.ffn_hc
        x = mhc.combine(hc, state, quantized)
        stats = state.stats
        state.release()
        st.open = Open(streams, stats, stream)
        return x, (quantized[0] if quantized else None)

    def post(self, layer, kind: str, partial, st: BoundaryState) -> None:
        """Reduce the open sublayer's TP-partial output into the streams.

        ``partial`` is the attention's wo_b rows before their TP sum (bare or in
        SGLang's ``AttnOutput``), or the
        MoE's output as SGLang's MoE returns it with the sum left to the post
        (a ``MoEOutput`` from ``leave_output_sum=True``, its routed finalize
        deferred where ``defers_sum``); past the push plane SGLang's own post
        merges, sums and posts it unfused."""
        mhc = self._mhc
        opened, st.open = st.open, None
        if opened is None:
            raise RuntimeError("post without a pre")
        if _attention(kind):
            if isinstance(partial, mhc.AttnOutput):  # SGLang's attention defers on TP
                partial = partial.partial
            if self.hop_a is not None and self.hop_a.published(partial):
                st.hc = self._collect_hop_a(layer, partial, opened)
                return
            if self.defers_sum(layer, kind, partial.shape[0]):
                out = mhc.AttnOutput(partial)
            else:
                # past the push plane SGLang's transport sums before a plain post
                from sglang.srt.runtime_context import get_parallel

                out = get_parallel().attn_tp_group.all_reduce(partial)
            st.hc = mhc.run_attn_post(
                layer.attn_hc,
                out,
                opened.streams,
                stats_stream=opened.stats_stream,
                next=self._ffn_next[layer.layer_id],
                world_size=layer.self_attn.attn_tp_size,
                precomputed=opened.stats,
            )
            return
        nxt = self._attn_next[layer.layer_id] if self.cross_layer else None
        st.hc = mhc.run_moe_post(
            layer.ffn_hc,
            partial,
            opened.streams,
            stats_stream=opened.stats_stream,
            next=nxt,
            world_size=layer.mlp.tp_size,
            precomputed=opened.stats,
        )

    def publish_wo_b(self, wo_b, o):
        """For the attention step at a hop-A width: wo_b's partial rows, published
        into the mailbox; None when hop A is off or its GEMM cannot take them."""
        return None if self.hop_a is None else self.hop_a.publish_wo_b(wo_b, o)

    def _collect_hop_a(self, layer, partial, opened):
        """``run_attn_post`` with the Lamport collect in place of SGLang's fused
        collective: the same triplet on the stats stream and the same FFN norm."""
        mhc = self._mhc
        pre, post_mix, comb = mhc.mix_stats(
            layer.attn_hc, opened.streams, opened.stats_stream
        )
        if opened.stats_stream is not None:
            torch.cuda.current_stream().wait_stream(opened.stats_stream)
        norm = self._ffn_next[layer.layer_id].norm
        updated, normed = self.hop_a.collect(
            partial,
            opened.streams,
            post_mix,
            comb,
            pre,
            norm.weight,
            norm.variance_epsilon,
        )
        return mhc.HcState(updated, pre, mhc.HcNormed(normed))

    @staticmethod
    def defers_sum(layer, kind: str, rows: int) -> bool:
        """Whether SGLang's transport leaves this sublayer's TP sum to the fused
        post at ``rows`` (the MoE then returns its finalize deferred)."""
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            can_fuse_all_reduce,
        )

        fusable = (
            layer._can_fuse_attn_mhc if _attention(kind) else layer._can_fuse_ffn_mhc
        )
        return (
            layer.boundary_mhc_fusions
            and layer.boundary_pre_from_prev
            and fusable
            and can_fuse_all_reduce(rows, layer.hc_cfg.hidden)
        )

    @staticmethod
    def residual(st: BoundaryState) -> torch.Tensor:
        """The materialized streams [rows, hc, hidden] (Engram, capture, head)."""
        return st.hc.residual

    def with_residual(self, st: BoundaryState, streams: torch.Tensor) -> None:
        """Replace the streams (an Engram injection); the pre-mix survives."""
        st.hc = st.hc.with_residual(streams)


def _attention(kind: str) -> bool:
    if kind not in ("attn", "ffn"):
        raise ValueError(f"unknown sublayer kind {kind!r}")
    return kind == "attn"
