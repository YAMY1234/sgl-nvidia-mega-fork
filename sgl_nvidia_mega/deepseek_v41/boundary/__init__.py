# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The mHC between sublayers (owner: boundary lane).

``pre`` turns the residual streams into a sublayer's input (mHC pre: combine +
norm); ``post`` reduces a sublayer's output into the streams (all-reduce + mHC
post) and, after attention, returns the MoE input (it folds the FFN's pre, as
SGLang does). ``start`` / ``final`` open and close the streams around the layers;
``materialize`` / ``set_residual`` serve the readers between layers (Engram, the
draft's capture).

SKELETON: every function calls SGLang's stage boundaries in the order
``DeepseekV4CudaDecoderLayer.forward`` does. The boundary lane replaces them with
the ported stock code, then with the plugin's kernels.
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "set_written",
        "written_residual",
        "read_output",
        "clear",
    ),
    "srt/models/deepseek_v4_mhc.py": ("HcState", "with_residual", "materialized"),
    "srt/models/deepseek_v4_boundary.py": (
        "LayerCall",
        "HcRead",
        "bind",
        "next_boundary",
        "materialize",
        "materialize(next)",
        "FinalRead",
        "quantized",
    ),
    "srt/models/deepseek_v4.py": (
        "attn_boundary",
        "ffn_boundary",
        "hc_cfg",
        "end_layer",
        "layers",
        # the module aliases SGLang's DSV4 model imports too
        "hc_boundary",
        "mhc",
        "residual_batch",
    ),
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "finish",
        "transport_context",
        "exit",
    ),
}


def build(model, ctx, group):
    """Bind weights, allocate buffers, compile; the result is ``st.bound["boundary"]``."""
    return None


def start(st, x) -> None:
    """Open the residual streams from the broadcast embedding ``x`` [rows, hc, dim]."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.models import deepseek_v4_mhc as mhc

    st.state = mhc.HcState(x)
    residual_batch.start(st.fb)
    st.hidden = residual_batch.set_written(st.state, st.fb)


def materialize(layer, st):
    """The streams as an ``HcState`` before ``layer`` (for Engram and the capture)."""
    from sglang.srt.models import deepseek_v4_boundary as hc_boundary

    call = hc_boundary.LayerCall(st.input_ids, st.input_ids_global)
    nxt = hc_boundary.HcRead(layer, True).bind(st.fb, call).next_boundary
    st.hidden = st.state = hc_boundary.materialize(st.hidden, st.fb, next=nxt)
    return st.state


def set_residual(st, residual) -> None:
    """Replace the materialized streams' residual (Engram's output)."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    st.state = st.state.with_residual(residual)
    st.hidden = residual_batch.set_written(st.state, st.fb)


def pre(layer, sublayer: str, st):
    """The input of ``layer``'s ``sublayer`` ("attn"); sets ``st.call`` and
    ``st.x_quant``."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.models import deepseek_v4_boundary as hc_boundary

    assert sublayer == "attn"
    st.call = hc_boundary.LayerCall(st.input_ids, st.input_ids_global, True)
    x = layer.attn_boundary.prepare(st.hidden, st.fb, read_context=st.call)
    st.x_quant = residual_batch.written_residual(st.fb).quantized
    return x


def post(layer, sublayer: str, out, st):
    """Reduce ``out`` of ``layer``'s ``sublayer`` into the streams. After "attn"
    returns the MoE input (and sets ``st.transport``); after "ffn" returns None."""
    if sublayer == "attn":
        hidden = layer.attn_boundary.finish(out, st.fb)
        x = layer.ffn_boundary.prepare(hidden, st.fb, read_context=st.call)
        st.transport = layer.ffn_boundary.transport_context(st.fb)
        st.scratch["boundary.ffn_output"] = st.exit.enter_context(
            layer.ffn_boundary.exit(st.fb)
        )
        return x
    assert sublayer == "ffn"
    st.hidden = st.scratch.pop("boundary.ffn_output").finish(out)
    return None


def final(st):
    """Close the streams after the last layer: (residual [rows, hc, dim], pre)."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.models import deepseek_v4_boundary as hc_boundary

    m = st.model
    state = residual_batch.read_output(st.hidden, st.fb, hc_boundary.FinalRead(), None)
    residual_batch.clear(st.fb)
    state = state.materialized(m.layers[m.end_layer - 1].hc_cfg)
    return state.residual, state.pre


def clear(st) -> None:
    """Drop the step's stream state (also on an error)."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    residual_batch.clear(st.fb)
