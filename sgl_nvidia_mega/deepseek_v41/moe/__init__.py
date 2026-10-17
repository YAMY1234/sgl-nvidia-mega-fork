# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MoE step over a layer's MoE block weights (owner: MoE lane): router gate,
top-k, the routed experts (the GEMM pair at 1-48 rows, else FlashInfer's MoE), the
shared expert on the side stream, and a deferred finalize.

``step`` returns the MoE output in the form ``boundary.post(layer, "ffn")`` takes
(SGLang's MoE pieces or deferred finalize, as with ``leave_output_sum=True``).
Today it is the stock step (``moe/step.py``).
"""

from __future__ import annotations

from sgl_nvidia_mega.deepseek_v41.moe.step import step


def build(model, ctx, group):
    """Bind weights, allocate buffers, compile; the result is ``st.bound["moe"]``."""
    return None


__all__ = ["build", "step"]
