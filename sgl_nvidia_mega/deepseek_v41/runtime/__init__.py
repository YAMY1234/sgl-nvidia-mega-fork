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

# The kernel slots this package ships; with the megakernel on, all of them run
SHIPPED_SLOTS = ("hop_a", "moe", "gate")
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def _requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; for tests and bisecting: "" runs every step
    function's stock code), else every shipped slot."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (shipped: {', '.join(SHIPPED_SLOTS)})"
        for name in sorted(SLOTS - set(SHIPPED_SLOTS))
    ]
    if "gate" in SLOTS and "moe" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'gate' needs slot 'moe'")
    return reasons


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": (
        "get_parallel",
        "get_spec",
        '"tp_group"',
        '"attn_dp_size"',
    ),
    "srt/arg_groups/fields/spec.py": ("speculative_num_draft_tokens",),
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
    the slot plan per width, and the executor; ``server_args`` holds resolved
    values (``requirements.Resolved``)."""
    from sglang.srt.runtime_context import get_parallel, get_spec

    from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
    from sgl_nvidia_mega.deepseek_v41 import requirements
    from sgl_nvidia_mega.deepseek_v41.runtime.executor import (
        PACKAGES,
        MegakernelDecode,
    )

    group = get_parallel().tp_group
    # resolved at startup (DSpark: block size + 1); the server argument may be unset
    tokens = int(get_spec().speculative_num_draft_tokens)
    if tokens > slot_plan.MAX_ROWS:
        raise RuntimeError(
            f"NVIDIA megakernel: {tokens} rows a request leave no covered width"
        )
    ctx = BuildContext(
        widths=tuple(range(tokens, slot_plan.MAX_ROWS + 1, tokens)),
        tokens_per_request=tokens,
        deadline_s=requirements.deadline_s(server_args.watchdog_timeout),
        copies={},
    )
    # free device memory, not memory_allocated: symmetric memory bypasses the allocator
    start, free = time.perf_counter(), torch.cuda.mem_get_info()[0]
    bound = {name: pkg.build(model, ctx, group) for name, pkg in PACKAGES.items()}
    # the boundary's hop A (its compiled widths) and the MoE step's GEMM pair, as built
    hop = getattr(bound["boundary"], "hop_a", None)
    facts = slot_plan.Facts(
        hop_a=hop is not None,
        hop_a_reason=getattr(bound["boundary"], "hop_a_reason", ""),
        routed=bool(getattr(bound["moe"], "slots", None)),
    )
    slots = {
        w: msgspec.structs.replace(s, hop_a=s.hop_a and w in hop.rows)
        for w, s in slot_plan.plan(ctx.widths, facts).items()
    }
    built = {
        "hop_a": any(s.hop_a for s in slots.values()),
        "moe": facts.routed,
        # the MoE build's own result: a layer's fast top-k may fail to build alone
        "gate": any(s.fast_gate for s in slots.values())
        and any(slot.gate for slot in getattr(bound["moe"], "slots", {}).values()),
    }
    # every package's build agrees across ranks, so this raises on every rank alike
    missing = sorted(name for name in SLOTS if not built[name])
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built at no width "
            f"(hop A: {facts.hop_a_reason or 'built'}; MoE: "
            f"{getattr(bound['moe'], 'reason', '') or 'built'})"
        )
    logger.info(
        "NVIDIA megakernel: DeepSeek-V4.1 verify steps of %s rows built in %.1f s, "
        "%.1f MiB; slots %s; hop A at %s rows, fast gate at %s rows; wait deadline "
        "%.0f s; weight copies %s",
        list(ctx.widths),
        time.perf_counter() - start,
        (free - torch.cuda.mem_get_info()[0]) / 2**20,
        sorted(name for name, on in built.items() if on) or "none (stock)",
        [w for w, x in slots.items() if x.hop_a],
        [w for w, x in slots.items() if x.fast_gate],
        ctx.deadline_s,
        {
            k: f"{v.numel() * v.element_size() / 2**20:.1f} MiB"
            for k, v in ctx.copies.items()
        },
    )
    return MegakernelDecode(model, slots, bound, widths=ctx.widths)
