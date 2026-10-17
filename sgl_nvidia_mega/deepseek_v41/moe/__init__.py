# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MoE step over a layer's MoE block weights (owner: MoE lane): router gate,
top-k, the routed experts (the GEMM pair at 1-48 rows, else FlashInfer's MoE), the
shared expert on the side stream, and a deferred finalize.

``step`` returns the MoE output in the form ``boundary.post(layer, "ffn")`` takes
(today SGLang's deferred-finalize output, ``leave_output_sum=True``).

SKELETON: ``step`` calls the layer's ``mlp`` with the transport's operands, as
``DeepseekV4CudaDecoderLayer.forward`` does.
"""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": ("mlp",),
    "srt/models/deepseek_v2.py": (
        "forward(skip_shared_experts)",
        "forward(return_moe_output)",
        "forward(leave_output_sum)",
    ),
    "srt/models/deepseek_common/moe_transport.py": (
        "input_ids",
        "input_ids_global",
        "do_shared_local",
    ),
    "srt/models/deepseek_v4_boundary.py": ("return_moe_output",),
}


def build(model, ctx, group):
    """Bind weights, allocate buffers, compile; the result is ``st.bound["moe"]``."""
    return None


def step(layer, x, st):
    """``layer``'s MoE on the normed input ``x`` [rows, dim]; reads
    ``st.transport`` (set by ``boundary.post(layer, "attn")``)."""
    t = st.transport
    return layer.mlp(
        x,
        st.fb,
        input_ids=t.input_ids,
        input_ids_global=t.input_ids_global,
        skip_shared_experts=t.do_shared_local,
        return_moe_output=t.return_moe_output,
        leave_output_sum=True,
    )
