# SPDX-License-Identifier: Apache-2.0
"""FN MoE step; the native FFN exit retains all reduction and HC ownership."""

from __future__ import annotations

import torch
from . import route

SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "mlp",
        "alt_stream",
        "shared_expert",
        "shared_expert_gate",
        "_forward_shared_experts",
        "_forward_shared_experts(apply_gate)",
        "_gate_shared_output_out_of_place",
    ),
    "srt/models/qwen4_exp.py": ("_run_ffn_stage", "_decode_moe_comm", "ffn_boundary"),
    "srt/layers/layer_boundary/stage.py": ("prepare", "exit", "finish"),
    "srt/layers/layer_boundary/fusions/moe_finalize.py": (
        "MoeDeferredFinalize",
        "from_flashinfer",
        "from_flashinfer(gated_shared_output)",
        "from_flashinfer(m)",
        "from_flashinfer(reduce)",
    ),
    "srt/layers/layer_boundary/exit.py": ("defer_moe_finalize",),
    "srt/layers/moe/utils.py": ("sum_post_experts_output",),
    "kernels/ops/elementwise/elementwise.py": ("fused_gate_sigmoid_mul_add",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}


def ffn(layer, hidden, st):
    """Split only the MoE body out of the native prepare/exit/finish stage.

    The special single-row decode communicator remains entirely native. No
    communication, HC write, or publish is performed inside the MoE slot.
    """
    # The mixer may return an opaque OwedOutput. Only boundary.prepare may
    # consume it; the step already owns the host-known token count.
    if not st.slots.route or (
        layer._decode_moe_comm is not None and st.input_ids.shape[0] == 1
    ):
        return layer._run_ffn_stage(hidden, st.fb)
    x = layer.ffn_boundary.prepare(hidden, st.fb)
    with layer.ffn_boundary.exit(st.fb) as finish:
        out = step(layer, x, st, defer=finish.defer_moe_finalize)
    return finish.finish(out)


def step(layer, x, st, *, defer):
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )
    from sglang.srt.layers.moe.utils import sum_post_experts_output
    from sglang.srt.model_executor.runner import get_is_capture_mode

    mlp = layer.mlp
    rows, hidden = x.shape
    bound = st.bound["moe"]
    # ae62 ordering: the main-stream producer completes before the alternate
    # stream's actual wait event. No early ready flag or speculative input read.
    gate_up = None
    if st.slots.route_shared:
        produced, gate_up = route.produce_shared(mlp, x, bound)
    else:
        produced = route.produce(mlp, x, bound)

    def shared_output(input_, *, apply_gate):
        if gate_up is None:
            return mlp._forward_shared_experts(input_, apply_gate=apply_gate)
        output = route.shared_expert_tail(mlp.shared_expert, gate_up)
        return mlp._gate_shared_output_out_of_place(x, output) if apply_gate else output

    use_gate = mlp.shared_expert_gate is not None
    if (
        mlp.alt_stream is not None
        and get_is_capture_mode()
        and not torch.compiler.is_compiling()
    ):
        current = torch.cuda.current_stream()
        mlp.alt_stream.wait_stream(current)
        if defer:
            routed = route.routed(
                mlp, x, bound, defer=True, produced=produced, fused=st.slots.mkm
            )
            with torch.cuda.stream(mlp.alt_stream):
                shared = shared_output(x, apply_gate=False)
                if shared is not None:
                    shared = mlp._gate_shared_output_out_of_place(x, shared)
        else:
            # Preserve the native clone until the complete stock consumer's
            # ownership contract has been audited independently.
            shared = (
                shared_output(
                    x if gate_up is not None else x.clone(), apply_gate=not use_gate
                )
                if mlp.shared_expert is not None
                else None
            )
            with torch.cuda.stream(mlp.alt_stream):
                routed = route.routed(
                    mlp, x, bound, defer=False, produced=produced, fused=st.slots.mkm
                )
        current.wait_stream(mlp.alt_stream)
    else:
        shared = shared_output(x, apply_gate=not use_gate and not defer)
        if defer and shared is not None:
            shared = mlp._gate_shared_output_out_of_place(x, shared)
        routed = route.routed(
            mlp, x, bound, defer=defer, produced=produced, fused=st.slots.mkm
        )
    if defer:
        if shared is None:
            raise RuntimeError(
                "FN deferred finalize requires the native gated shared output"
            )
        return MoeDeferredFinalize.from_flashinfer(
            routed, gated_shared_output=shared, m=rows, reduce=sum_post_experts_output
        )
    if shared is not None:
        if use_gate:
            fused_gate_sigmoid_mul_add(
                x, mlp.shared_expert_gate.weight.squeeze(), shared, routed
            )
        else:
            routed += shared
    return routed.view(rows, hidden)
