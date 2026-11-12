# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.8 forward executor: it owns the target's and the MTP draft's forward
(``executor``), with per-step state (``state``) and the per-op engagement record
(``engagement``); the test-only parity check is ``tools/parity.py``, loaded only when
its switch is set. ``build`` runs through the spec once the model is prepared, before
the KV cache is sized."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The SGLang names this package's build and checks read (test/unit/test_contract.py)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
}

# Comma-separated kernel slots; unset or "" runs every op on its stock step
# function. No slot ships on by default yet.
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
SHIPPED_SLOTS: tuple[str, ...] = ()
KNOWN_SLOTS: tuple[str, ...] = (
)
# The widest covered step: 4 rows a request x 256 requests.
MAX_ROWS = 1024
# Test only: compare every covered step with SGLang's forward (tools/parity.py)
PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
# S1 (test switch until it is the default): run each layer's HC read / mixer /
# HC write / MoE from the executor instead of through the stage boundaries.
DECOMPOSE_ENV = "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_DECOMPOSE"


def decompose_supported(parallel) -> bool:
    """The stage boundaries do no collective and keep no rows elsewhere."""
    return (
        parallel.tp_size == 1
        and getattr(parallel, "attn_dp_size", 1) == 1
        and getattr(parallel, "attn_cp_size", 1) == 1
        and getattr(parallel, "moe_ep_size", 1) == 1
    )


def requested_slots() -> frozenset:
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


def config_reasons(cfg, hf_config) -> list[str]:
    """Settings the executor cannot serve, checked before the weights load."""
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r}"
        for name in sorted(requested_slots() - set(KNOWN_SLOTS))
    ]
    graph = getattr(cfg, "cuda_graph_config", None)
    prefill = getattr(getattr(graph, "prefill", None), "backend", None)
    if prefill is not None and str(prefill) not in ("disabled", "Backend.DISABLED"):
        # A covered prefill runs eagerly (spec.Step); breakable prefill graphs
        # would capture it.
        reasons.append(
            f"prefill CUDA graphs {str(prefill)!r} are not supported by the "
            "Qwen3.8 executor; use --cuda-graph-backend-prefill disabled"
        )
    return reasons


def model_reasons(model, hf_config) -> list[str]:
    """This rank's reasons the loaded model cannot run the executor."""
    reasons = []
    if getattr(model, "capture_aux_hidden_states", False):
        reasons.append("auxiliary hidden-state capture (EAGLE3) is not supported")
    pp = getattr(model, "pp_group", None)
    if pp is not None and not (pp.is_first_rank and pp.is_last_rank):
        reasons.append("pipeline parallelism is not supported")
    return reasons


def build(model, hf_config, ctx):
    """The executor of this model (target or MTP draft); ``ctx`` is SGLang's
    BuildContext."""
    from sgl_nvidia_mega.qwen3_8.runtime.executor import Qwen3_8Executor

    draft = type(model).__name__.endswith("MTP")
    slots = requested_slots()
    if draft:
        # A draft decode step has one row a request; its draft-extend rows a
        # request are the target's verify width.
        widths = tuple(range(1, MAX_ROWS + 1))
    else:
        widths = tuple(ctx.widths)
    from sglang.srt.runtime_context import get_parallel

    checker = None
    if os.environ.get(PARITY_ENV, "0") != "0":
        from sgl_nvidia_mega.qwen3_8.tools import parity

        checker = parity.Parity(model, draft=draft) if parity.ENABLED else None
    decompose = os.environ.get(DECOMPOSE_ENV) == "1" and decompose_supported(
        get_parallel()
    )
    executor = Qwen3_8Executor(
        model,
        widths=widths,
        draft=draft,
        slots=slots,
        binding=ctx.binding,
        parity=checker,
        decompose=decompose,
    )
    logger.info(
        "NVIDIA megakernel: Qwen3.8 %s executor: %s; slots %s; every other op stock",
        "draft" if draft else "target",
        "layers decomposed (S1)" if decompose else "layers through the stage boundaries (S0)",
        sorted(slots) or "none",
    )
    return executor
