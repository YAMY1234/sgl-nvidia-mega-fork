# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel runtime: the executor of a covered target-verify step (``executor``),
its per-step state (``state``), the embedding (``embedding``), the test-only parity
check (``parity``), and ``build``, which SGLang runs through the spec once the model is
prepared and before the KV cache is sized."""

from __future__ import annotations

import os

import msgspec

# The kernel slots that run by default, and the ones that run only when requested
# (until step-level timing and accuracy runs move them to the default list)
SHIPPED_SLOTS: tuple[str, ...] = ()
OPT_IN_SLOTS: tuple[str, ...] = ("route", "mkm", "gdn_verify", "gdn_pdl", "hop")
# The EAGLE entries (spec.ENTRIES) offered only when requested in the same variable;
# verify_sample has no step at any size, so it turns on SGLang's post-verify graphs
# with SGLang's own sampling
OPT_IN_ENTRIES: tuple[str, ...] = ("verify_sample",)
# The MTP draft's EAGLE entries and kernel groups (qwen3_5/draft), requested in the same
# variable; the draft spec's build reads them, the target's ignores them
DRAFT_ENTRIES: tuple[str, ...] = ("draft_loop", "draft_extend")
DRAFT_SLOTS: tuple[str, ...] = ("draft_moe", "draft_q1", "draft_xchg", "draft_hop")
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def _requested_slots() -> frozenset:
    """``SLOTS_ENV`` (comma-separated; "" runs the stock step), else every shipped
    slot."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = (*SHIPPED_SLOTS, *OPT_IN_SLOTS, *OPT_IN_ENTRIES, *DRAFT_ENTRIES, *DRAFT_SLOTS)
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(known) or 'none'})"
        for name in sorted(SLOTS - set(known))
    ]
    if "mkm" in SLOTS and "route" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'mkm' runs on slot 'route''s outputs")
    if "gdn_pdl" in SLOTS and "gdn_verify" not in SLOTS:
        reasons.append(
            f"{SLOTS_ENV}: slot 'gdn_pdl' launches slot 'gdn_verify''s verify"
        )
    for name in sorted(SLOTS & set(DRAFT_SLOTS)):
        if "draft_loop" not in SLOTS:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} runs inside entry 'draft_loop'")
    if "draft_hop" in SLOTS and "hop" not in SLOTS:
        reasons.append(f"{SLOTS_ENV}: slot 'draft_hop' runs on slot 'hop''s mailbox")
    return reasons


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "deadline_s",
        "binding",
    ),
}


class PackageContext(msgspec.Struct, frozen=True):
    """What every package's ``build(model, ctx, group)`` gets."""

    widths: tuple[int, ...]  # the covered steps' rows (requests x rows a request)
    tokens_per_request: int
    deadline_s: float  # each cross-GPU wait's deadline


def build(model, ctx):
    """Each package's build, the slot plan per width and the executor; ``ctx`` is
    SGLang's BuildContext."""
    import torch

    from sgl_nvidia_mega.qwen3_5 import boundary, gdn, moe
    from sgl_nvidia_mega.qwen3_5 import plan as slot_plan
    from sgl_nvidia_mega.qwen3_5.runtime import parity
    from sgl_nvidia_mega.qwen3_5.runtime.executor import Qwen35Verify

    if not ctx.widths:
        raise RuntimeError(
            f"NVIDIA megakernel: {ctx.tokens_per_request} rows a request leave no "
            "covered width"
        )
    pctx = PackageContext(
        widths=ctx.widths,
        tokens_per_request=ctx.tokens_per_request,
        deadline_s=ctx.deadline_s,
    )
    bound = {
        "moe": moe.build(model, pctx),
        "gdn": gdn.build(model, pctx),
        "boundary": boundary.build(model, pctx),
    }
    facts = slot_plan.Facts(
        route=bound["moe"].route,
        mkm=bool(bound["moe"].mkm),
        gdn_verify=bound["gdn"].gdn_verify,
        gdn_pdl=bound["gdn"].gdn_pdl,
        hop=bound["boundary"].hop is not None,
    )
    slots = slot_plan.plan(pctx.widths, facts)
    built = msgspec.structs.asdict(facts)
    entries = tuple(name for name in OPT_IN_ENTRIES if name in SLOTS)
    # every package's build agrees across ranks, so this raises on every rank alike
    target_slots = SLOTS - set(entries) - set(DRAFT_ENTRIES) - set(DRAFT_SLOTS)
    missing = sorted(name for name in target_slots if not built[name])
    if missing:
        raise RuntimeError(
            f"NVIDIA megakernel: requested slots {missing} built no width (MoE: "
            f"{bound['moe'].reason or 'built'}; GDN: {bound['gdn'].reason or 'built'}; "
            f"boundary: {bound['boundary'].reason or 'built'})"
        )
    checker = (
        parity.Parity(pctx.widths, torch.cuda.current_device(), model)
        if parity.ENABLED
        else None
    )
    on = sorted(name for name, ok in built.items() if ok)
    offered = f"; entries {list(entries)}" if entries else ""
    return Qwen35Verify(
        model,
        slots,
        bound=bound,
        widths=pctx.widths,
        binding=ctx.binding,
        entries=entries,
        summary=f"slots {on or 'none (stock)'}{offered}",
        parity=checker,
    )
