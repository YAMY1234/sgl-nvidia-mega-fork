# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The FN megakernel runtime: the executor of a covered target-verify step
(``executor``) and ``build``, which SGLang runs through the spec once the model is
prepared and before the KV cache is sized. FN runs on SGLang's generic megakernel
binding: its QSA attention backend has none of the dense metadata the EAGLE binding's
forward view reads."""

from __future__ import annotations

import msgspec

from sgl_nvidia_mega.qwen3_5.runtime import SLOTS, SLOTS_ENV

# The kernel slots that run only when requested (in Qwen3.5's request variable)
OPT_IN_SLOTS: tuple[str, ...] = (
    "gdn_verify",
    "gdn_pdl",
    "gdn_mk",
    "route",
    "mkm",
    "route_shared",
    "qsa_core",
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "binding",
        "deadline_s",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_size"'),
}


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    reasons = [
        f"{SLOTS_ENV}: unknown FN slot {name!r} "
        f"(known: {', '.join(OPT_IN_SLOTS) or 'none'})"
        for name in sorted(SLOTS - set(OPT_IN_SLOTS))
    ]
    for name in ("mkm", "route_shared"):
        if name in SLOTS and "route" not in SLOTS:
            reasons.append(f"{SLOTS_ENV}: slot {name!r} requires slot 'route'")
    return reasons


def build(model, ctx, *, hf_config=None):
    """The slot plan per width and the executor; ``ctx`` is SGLang's BuildContext."""
    import torch

    from sgl_nvidia_mega.qwen3_5.runtime import PackageContext, parity
    from sgl_nvidia_mega.qwen3_8 import attention, gdn
    from sgl_nvidia_mega.qwen3_8 import plan as slot_plan
    from sgl_nvidia_mega.qwen3_8.runtime.executor import Qwen38Verify

    widths = slot_plan.candidates(ctx.widths, ctx.tokens_per_request)
    if not widths:
        raise RuntimeError(
            f"NVIDIA megakernel: {ctx.tokens_per_request} rows a request leave no "
            "covered width"
        )
    pctx = PackageContext(
        widths=widths,
        tokens_per_request=ctx.tokens_per_request,
        deadline_s=ctx.deadline_s,
    )
    gdn_bound = gdn.build(model, pctx)
    if gdn_bound.reason:
        raise RuntimeError(f"NVIDIA megakernel: FN GDN: {gdn_bound.reason}")
    attn_bound = attention.build(model, pctx, hf_config, SLOTS)
    if "qsa_core" in SLOTS and not attn_bound.qsa_core:
        raise RuntimeError(f"NVIDIA megakernel: FN attention: {attn_bound.reason}")
    bound = {"gdn": gdn_bound, "attention": attn_bound}
    attention_facts = dict(qsa_core=attn_bound.qsa_core)
    gdn_facts = dict(
        gdn_verify=gdn_bound.gdn_verify,
        gdn_pdl=gdn_bound.gdn_pdl,
        gdn_mk=gdn_bound.gdn_mk,
    )
    facts = slot_plan.Facts(**gdn_facts, **attention_facts)
    if "route" in SLOTS:
        from sglang.srt.runtime_context import get_parallel
        from sgl_nvidia_mega.qwen3_8.moe import route

        moe = route.build(
            model,
            max(widths),
            fused="mkm" in SLOTS,
            shared="route_shared" in SLOTS,
            hf_config=hf_config,
            tp_size=get_parallel().tp_size,
        )
        if (
            not moe.route
            or ("mkm" in SLOTS and not moe.mkm)
            or ("route_shared" in SLOTS and not moe.shared)
        ):
            raise RuntimeError(
                f"NVIDIA megakernel: FN MoE slots not built: {moe.reason}"
            )
        bound["moe"] = moe
        facts = slot_plan.Facts(
            **gdn_facts, **attention_facts, route=moe.route, mkm=bool(moe.mkm), route_shared=moe.shared
        )
    slots = slot_plan.plan(widths, facts)
    checker = (
        parity.Parity(widths, torch.cuda.current_device(), model)
        if parity.ENABLED
        else None
    )
    on = sorted(name for name, ok in msgspec.structs.asdict(facts).items() if ok)
    return Qwen38Verify(
        model,
        slots,
        bound=bound,
        widths=widths,
        binding=ctx.binding,
        summary=f"slots {on or 'none (stock)'}",
        parity=checker,
    )
