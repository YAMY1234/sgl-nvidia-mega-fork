# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hop A: wo_b publishes its rows into a Lamport mailbox on every rank, and the
attention post collects them with the all-reduce + mHC post + FFN norm fused.
Hop B reuses the mailbox for the MoE post (the Lamport MoE finalize)."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan

logger = logging.getLogger(__name__)


class HopA:
    """One rank's mailbox, publishing wo_b GEMM and Lamport consumer."""

    def __init__(self, runner, rows: tuple[int, ...]):
        self._runner = runner
        self.rows = rows  # the widths the publishing GEMM was compiled for

    def publish_wo_b(self, wo_b, o) -> Optional[torch.Tensor]:
        """wo_b's TP-partial rows for ``o`` (BF16 [rows, heads * dim], or its MXFP8
        pair), also stored into the mailbox; None when the GEMM cannot take them."""
        return self._runner.publish_wo_b(wo_b, o)

    def published(self, partial: torch.Tensor) -> bool:
        return self._runner.hop._published is partial

    def collect(self, partial, residual, post_mix, comb, pre, norm_weight, eps):
        """(updated streams, normed FFN input) of the published ``partial``."""
        return self._runner.hop_a(
            partial, residual, post_mix, comb, pre, norm_weight, eps
        )

    def finalize(self, *args, mxfp8_out=None):
        """Hop B: ``AllReduceMHC.finalize`` of a deferred MoE output on this mailbox."""
        return self._runner.hop.finalize(*args, mxfp8_out=mxfp8_out)

    def check_consumed(self) -> None:
        """Raise on the host if a publish was not collected: it would leave the
        mailbox generation it wrote dirty."""
        self._runner.check_consumed()


def build(
    model,
    ctx,
    group,
    *,
    hop_b: bool = False,
    early_trigger: bool = False,
    hop_b_start_trigger: bool = False,
    batch2: bool = False,
) -> tuple[Optional[HopA], str]:
    """(the hop, "") or (None, why not). Collective over the TP group: a failure on
    any rank raises on every rank; an unsupported wo_b leaves the hop off. ``hop_b``
    also compiles the MoE finalize's consumers; ``early_trigger`` launches the
    consumer's dependents right after its poll, ``hop_b_start_trigger`` the MoE
    finalize's at its start; ``batch2`` (slot hop_a_batch2) adds 9-16 rows."""
    from sgl_nvidia_mega.deepseek_v41.sm103.runner import Runner

    m = model.model
    cfg = model.config
    wo_b = m.layers[m.start_layer].self_attn.wo_b
    max_rows = slot_plan.hop_a_max_rows(batch2)
    rows = tuple(w for w in ctx.widths if w <= max_rows)
    if not rows:
        return None, f"no width up to {max_rows} rows"
    runner = Runner(device=wo_b.weight.device, rank=group.rank_in_group)
    error, taken = "", ()
    try:
        runner.build_hop(
            group=group.device_group,
            hidden=cfg.hidden_size,
            hc_mult=cfg.hc_mult,
            top_k=cfg.num_experts_per_tok,
            hop_a_rows=max_rows,
            deadline_s=ctx.deadline_s,
            hop_b=hop_b,
            early_trigger=early_trigger,
            hop_b_start_trigger=hop_b_start_trigger,
        )
        taken = tuple(runner.precompile_publish(wo_b, list(rows)))
    except Exception as e:  # gathered, then raised on every rank
        error = f"{type(e).__name__}: {e}"
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, (error, taken), group=group.cpu_group)
    failed = [f"rank {r}: {e}" for r, (e, _) in enumerate(gathered) if e]
    if failed:
        raise RuntimeError(
            "NVIDIA megakernel: the hop-A build failed: " + "; ".join(failed)
        )
    if any(t != taken for _, t in gathered):
        raise RuntimeError(
            f"NVIDIA megakernel: the ranks disagree on hop A: {gathered}"
        )
    if not taken:
        return None, "wo_b is not an MXFP8 linear the low-latency GEMM takes"
    return HopA(runner, taken), ""
