# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the plugin's hooks into SGLang (``hooks``), the executor
of a covered decode step (``executor``), its per-step state (``state``), the
embedding (``embedding``), and ``build``, which runs once the weights are loaded and
before the KV cache is sized."""

from __future__ import annotations

import logging
import os
import time

import msgspec
import torch

logger = logging.getLogger(__name__)

# The slots build may fill (comma-separated: moe, gate); empty = every
# step function runs its stock code. The per-slot off switch, and the tests' baseline.
SLOTS = frozenset(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", "").split(",")) - {
    ""
}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel", '"tp_group"', '"attn_dp_size"'),
    "srt/models/deepseek_v4.py": (
        "DeepseekV4CudaDecoderLayer",
        "hc_pre_from_prev_sublayer",
        "model",
        "start_layer",
        "end_layer",
        "layers",
    ),
}


class BuildContext(msgspec.Struct, frozen=True):
    """What every package's ``build(model, ctx, group)`` gets."""

    widths: tuple[int, ...]  # the covered steps' rows (requests x rows a request)
    tokens_per_request: int
    deadline_s: float  # each cross-GPU wait's deadline
    copies: dict  # (kind, name) -> tensor: weight copies the packages keep


def model_disable_reasons(model) -> list[str]:
    """What the server arguments do not show: the layer loop the executor follows."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4CudaDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    m = model.model
    reasons = []
    if not m.hc_pre_from_prev_sublayer:
        reasons.append("needs the mHC pre-mix carried from the previous sublayer")
    if get_parallel().attn_dp_size > 1:
        reasons.append("attention DP: the covered step reads replicated token ids")
    other = [
        i
        for i in range(m.start_layer, m.end_layer)
        if not isinstance(m.layers[i], DeepseekV4CudaDecoderLayer)
    ]
    if other:
        reasons.append(f"layers {other}: not SGLang's CUDA decoder layer")
    return reasons


def build(model, server_args):
    """Each package's build (weights bound, buffers allocated, kernels compiled),
    the slot plan per width, and the executor."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
    from sgl_nvidia_mega.deepseek_v41 import requirements
    from sgl_nvidia_mega.deepseek_v41.runtime import parity
    from sgl_nvidia_mega.deepseek_v41.runtime.executor import (
        PACKAGES,
        MegakernelDecode,
    )

    if parity.SORT_TOPK:
        parity.install_sorted_topk()
    group = get_parallel().tp_group
    tokens = int(server_args.speculative_num_draft_tokens)
    ctx = BuildContext(
        widths=tuple(range(tokens, slot_plan.MAX_ROWS + 1, tokens)),
        tokens_per_request=tokens,
        deadline_s=requirements.deadline_s(server_args.watchdog_timeout),
        copies={},
    )
    start, memory = time.perf_counter(), torch.cuda.memory_allocated()
    bound = {name: pkg.build(model, ctx, group) for name, pkg in PACKAGES.items()}
    # the MoE step's build result names the layers whose GEMM pair built
    facts = slot_plan.Facts(
        routed=bool(getattr(bound["moe"], "slots", None)),
    )
    slots = slot_plan.plan(ctx.widths, facts)
    logger.info(
        "NVIDIA megakernel: DeepSeek-V4.1 verify steps of %s rows built in %.1f s, "
        "%.1f MiB; slots %s; wait "
        "deadline %.0f s; weight copies %s",
        list(ctx.widths),
        time.perf_counter() - start,
        (torch.cuda.memory_allocated() - memory) / 2**20,
        sorted(SLOTS) or "none (every step function stock)",
        ctx.deadline_s,
        {
            k: f"{v.numel() * v.element_size() / 2**20:.1f} MiB"
            for k, v in ctx.copies.items()
        },
    )
    return MegakernelDecode(model, slots, bound, streams={}, widths=ctx.widths)
