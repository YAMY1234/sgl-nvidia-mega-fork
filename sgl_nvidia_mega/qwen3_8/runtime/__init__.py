# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.8 forward executor: it owns the target's and the MTP draft's forward
(``executor``), with per-step state (``state``), the per-op engagement record
(``engagement``) and the test-only parity check (``parity``). ``build`` runs through
the spec once the model is prepared, before the KV cache is sized."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Comma-separated kernel slots; unset runs every shipped slot, "" runs every op on
# its stock step function. S0 ships none: every op runs stock.
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
SHIPPED_SLOTS: tuple[str, ...] = ()
# hc: the HC write-back kernel for up to 32 rows (runtime/hc.py); needs the S1 layers
# gdn_native_pdl: the GDN target verify's ReplaySSM core as the plugin's PDL launch
# qsa_pool: the TP1 QSA verify's sparse attention as split-K pool kernels (<= 64 rows)
# moe_finalize: the TP1 MoE's finalize + gated shared expert as one ordered kernel
# hc_wide: HC reads / write-backs of more than 32 rows (runtime/hc_wide.py)
KNOWN_SLOTS: tuple[str, ...] = (
    "hc",
    "hc_wide",
    "gdn_native_pdl",
    "qsa_pool",
    "qsa_indexer",
    "moe_finalize",
)
# A node-local directory for the JIT sources some kernels write before compiling.
SOURCE_DIR_ENV = "SGLANG_NVIDIA_MEGAKERNEL_Q38_SOURCE_DIR"
# The widest covered step: 4 rows a request x 256 requests.
MAX_ROWS = 1024
# S1 (test switch until it is the default): run each layer's HC read / mixer /
# HC write / MoE from the executor instead of through the stage boundaries.
DECOMPOSE_ENV = "SGLANG_NVIDIA_MEGAKERNEL_Q38_DECOMPOSE"


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
    if getattr(cfg, "enable_gdn_replayssm_ucache", False):
        # SGLang has no u-cache verify of its own: the executor must cover every
        # target verify, so its widest one must fit the built widths.
        tokens = int(getattr(cfg, "speculative_num_draft_tokens", 0) or 0)
        running = int(getattr(cfg, "max_running_requests", 0) or 0)
        if tokens and running and tokens * running > MAX_ROWS:
            reasons.append(
                f"--enable-gdn-replayssm-ucache: a verify of {running} requests x "
                f"{tokens} rows exceeds the executor's {MAX_ROWS} rows"
            )
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
    from sgl_nvidia_mega.qwen3_8.runtime import parity
    from sgl_nvidia_mega.qwen3_8.runtime.executor import Qwen38Executor

    draft = type(model).__name__.endswith("MTP")
    slots = requested_slots()
    if draft:
        # A draft decode step has one row a request; its draft-extend rows a
        # request are the target's verify width.
        widths = tuple(range(1, MAX_ROWS + 1))
    else:
        widths = tuple(ctx.widths)
    from sglang.srt.runtime_context import get_parallel

    checker = parity.Parity(model, draft=draft) if parity.ENABLED else None
    decompose = os.environ.get(DECOMPOSE_ENV) == "1" and decompose_supported(
        get_parallel()
    )
    executor = Qwen38Executor(
        model,
        widths=widths,
        draft=draft,
        slots=slots,
        binding=ctx.binding,
        parity=checker,
        decompose=decompose,
    )
    if not draft:
        from sglang.srt.runtime_context import get_exec

        mamba = getattr(get_exec(), "mamba", None)
        executor.verify_required = bool(
            getattr(mamba, "enable_gdn_replayssm_ucache", False)
        )
    if "hc" in slots:
        executor.hc_ready = _build_hc(model, decompose)
    if "hc_wide" in slots:
        executor.hc_wide_ready = _build_hc_wide(model, decompose, executor.hc_limits)
    if "gdn_native_pdl" in slots and not draft:
        executor.gdn_pdl_ready = _build_gdn_pdl(model)
    if "moe_finalize" in slots:
        executor.moe_finalize_ready = _build_moe_finalize(model, decompose)
    if "qsa_indexer" in slots:
        executor.qsa_indexer = _build_qsa_indexer(model)
    if "qsa_pool" in slots and not draft:
        executor.qsa_pool_ready = _build_qsa_pool(model)
        if executor.qsa_pool_ready:
            from sgl_nvidia_mega.qwen3_8.runtime.attention import PoolKernels

            executor.qsa_pool = PoolKernels()
    logger.info(
        "NVIDIA megakernel: Qwen3.8 %s executor: %s; slots %s; every other op stock",
        "draft" if draft else "target",
        "layers decomposed (S1)" if decompose else "layers through the stage boundaries (S0)",
        sorted(slots) or "none",
    )
    return executor


def _build_hc(model, decompose) -> bool:
    """Whether slot hc runs, agreed on every rank."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import hc

    m = model.model
    reason = "" if decompose else f"needs the S1 layers ({DECOMPOSE_ENV}=1 at TP1)"
    for i in range(m.start_layer, m.end_layer):
        reason = reason or hc.layer_supported(m.layers[i])
    reason = reason or hc.build()
    megakernel_agree(reason, what="Qwen3.8 slot hc")
    if reason:
        unavailable("hc", reason)
    return not reason


def _build_gdn_pdl(model) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import gdn

    reason = gdn.verify_reasons(model)
    megakernel_agree(reason, what="Qwen3.8 slot gdn_native_pdl")
    if reason:
        unavailable("gdn_native_pdl", reason)
    return not reason


def _build_hc_wide(model, decompose, limits) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import hc, hc_wide

    m = model.model
    reason = "" if decompose else f"needs the S1 layers ({DECOMPOSE_ENV}=1 at TP1)"
    for i in range(m.start_layer, m.end_layer):
        reason = reason or hc.layer_supported(m.layers[i])
    reason = reason or hc_wide.build(limits=limits)
    megakernel_agree(reason, what="Qwen3.8 slot hc_wide")
    if reason:
        unavailable("hc_wide", reason)
    return not reason


def _build_moe_finalize(model, decompose) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_8.runtime import moe

    m = model.model
    reason = "" if decompose else f"needs the S1 layers ({DECOMPOSE_ENV}=1 at TP1)"
    for i in range(m.start_layer, m.end_layer):
        reason = reason or moe.layer_reasons(m.layers[i])
    reason = reason or moe.build(get_parallel())
    megakernel_agree(reason, what="Qwen3.8 slot moe_finalize")
    if reason:
        unavailable("moe_finalize", reason)
    return not reason


def _build_qsa_indexer(model):
    """The indexer kernels (qsa_indexer.IndexerKernels), or None (stock indexer)."""
    import tempfile

    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer

    source_dir = os.environ.get(SOURCE_DIR_ENV) or os.path.join(
        tempfile.gettempdir(), "q38-mk-sources"
    )
    os.makedirs(source_dir, exist_ok=True)
    kernels = qsa_indexer.IndexerKernels()
    reason = qsa_indexer.build(model, kernels, source_dir=os.path.abspath(source_dir))
    megakernel_agree(reason, what="Qwen3.8 slot qsa_indexer")
    if reason:
        unavailable("qsa_indexer", reason)
        return None
    return kernels


def _build_qsa_pool(model) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer

    from sgl_nvidia_mega.qwen3_8.runtime import attention

    m = model.model
    reason = ""
    for i in range(m.start_layer, m.end_layer):
        if isinstance(m.layers[i], Qwen4ExpAttentionDecoderLayer):
            reason = reason or attention.layer_reasons(m.layers[i])
    reason = reason or attention.build()
    megakernel_agree(reason, what="Qwen3.8 slot qsa_pool")
    if reason:
        unavailable("qsa_pool", reason)
    return not reason


def unavailable(slot: str, reason: str) -> None:
    logger.info("NVIDIA megakernel: Qwen3.8 slot %s uses stock (%s)", slot, reason)
