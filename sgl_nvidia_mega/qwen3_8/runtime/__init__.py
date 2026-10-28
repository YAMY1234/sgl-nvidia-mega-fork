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
OPT_IN_SLOTS: tuple[str, ...] = ("gdn_verify", "gdn_pdl", "gdn_front")

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "binding",
    ),
}


def slot_reasons() -> list[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    return [
        f"{SLOTS_ENV}: unknown FN slot {name!r} "
        f"(known: {', '.join(OPT_IN_SLOTS) or 'none'})"
        for name in sorted(SLOTS - set(OPT_IN_SLOTS))
    ]


def build(model, ctx):
    """The slot plan per width and the executor; ``ctx`` is SGLang's BuildContext."""
    import torch

    from sgl_nvidia_mega.qwen3_5.runtime import PackageContext, parity
    from sgl_nvidia_mega.qwen3_8 import gdn
    from sgl_nvidia_mega.qwen3_8 import plan as slot_plan
    from sgl_nvidia_mega.qwen3_8.runtime.executor import Qwen38Verify

    widths = slot_plan.candidates(ctx.widths, ctx.tokens_per_request)
    if not widths:
        raise RuntimeError(
            f"NVIDIA megakernel: {ctx.tokens_per_request} rows a request leave no "
            "covered width"
        )
    pctx = PackageContext(widths=widths, tokens_per_request=ctx.tokens_per_request,
                          deadline_s=ctx.deadline_s)
    gdn_bound = gdn.build(model, pctx)
    if gdn_bound.reason:
        raise RuntimeError(f"NVIDIA megakernel: FN GDN: {gdn_bound.reason}")
    facts = slot_plan.Facts(gdn_verify=gdn_bound.gdn_verify,
                           gdn_pdl=gdn_bound.gdn_pdl,
                           gdn_mk=gdn_bound.gdn_mk,
                           gdn_front=gdn_bound.gdn_front)
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
        bound={"gdn": gdn_bound},
        widths=widths,
        binding=ctx.binding,
        summary=f"slots {on or 'none (stock)'}",
        parity=checker,
    )
