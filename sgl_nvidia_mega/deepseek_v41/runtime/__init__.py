# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the executor of a covered decode step (``executor``), its
per-step state (``state``), the embedding (``embedding``), and ``build``, which SGLang
runs through the spec once the model is prepared and before the KV cache is sized."""

from __future__ import annotations

import os
from typing import Any

import msgspec

# The kernel slots this package ships; with the megakernel on, all of them run
SHIPPED_SLOTS = ("hop_a", "moe", "gate", "attn")
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
        '"tp_group"',
        '"attn_dp_size"',
    ),
    "srt/megakernels/spec.py": ("widths", "tokens_per_request", "deadline_s"),
    "srt/models/deepseek_v4.py": (
        "DeepseekV4CudaDecoderLayer",
        "hc_pre_from_prev_sublayer",
        "model",
        "start_layer",
        "end_layer",
        "layers",
    ),
}


class PackageContext(msgspec.Struct, frozen=True):
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


def built_slots(slots: dict, bound: dict, facts: Any) -> dict[str, bool]:
    """Each shipped slot: whether its build serves at least one width."""
    return {
        "hop_a": any(s.hop_a for s in slots.values()),
        "moe": facts.routed,
        # the MoE build's own result: a layer's fast top-k may fail to build alone
        "gate": any(s.fast_gate for s in slots.values())
        and any(slot.gate for slot in getattr(bound["moe"], "slots", {}).values()),
        "attn": any(step.chain is not None for step in bound["attention"].values()),
    }


def build(model, ctx):
    """Each package's build (weights bound, buffers allocated, kernels compiled),
    the slot plan per width, and the executor; ``ctx`` is SGLang's BuildContext."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
    from sgl_nvidia_mega.deepseek_v41.runtime.executor import (
        PACKAGES,
        MegakernelDecode,
    )

    group = get_parallel().tp_group
    if not ctx.widths:
        raise RuntimeError(
            f"NVIDIA megakernel: {ctx.tokens_per_request} rows a request leave no "
            "covered width"
        )
    pctx = PackageContext(
        widths=ctx.widths,
        tokens_per_request=ctx.tokens_per_request,
        deadline_s=ctx.deadline_s,
        copies={},
    )
    bound = {name: pkg.build(model, pctx, group) for name, pkg in PACKAGES.items()}
    # the boundary's hop A (its compiled widths) and the MoE step's GEMM pair, as built
    hop = getattr(bound["boundary"], "hop_a", None)
    facts = slot_plan.Facts(
        hop_a=hop is not None,
        hop_a_reason=getattr(bound["boundary"], "hop_a_reason", ""),
        routed=bool(getattr(bound["moe"], "slots", None)),
    )
    slots = {
        w: msgspec.structs.replace(s, hop_a=s.hop_a and w in hop.rows)
        for w, s in slot_plan.plan(pctx.widths, facts).items()
    }
    built = built_slots(slots, bound, facts)
    # every package's build agrees across ranks, so this raises on every rank alike
    missing = sorted(name for name in SLOTS if not built[name])
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built at no width "
            f"(hop A: {facts.hop_a_reason or 'built'}; MoE: "
            f"{getattr(bound['moe'], 'reason', '') or 'built'})"
        )
    summary = (
        f"slots {sorted(name for name, on in built.items() if on) or 'none (stock)'}; "
        f"hop A at {[w for w, x in slots.items() if x.hop_a]} rows, fast gate at "
        f"{[w for w, x in slots.items() if x.fast_gate]} rows; weight copies "
        + str(
            {
                k: f"{v.numel() * v.element_size() / 2**20:.1f} MiB"
                for k, v in pctx.copies.items()
            }
        )
    )
    return MegakernelDecode(model, slots, bound, widths=pctx.widths, summary=summary)
