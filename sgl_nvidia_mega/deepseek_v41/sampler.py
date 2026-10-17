# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash's DSpark draft proposal on SM103 (GB300): the draft LM head and
the Markov chain of every draft step in one vocab-sharded kernel per rank.

SGLang's folded proposal (``DsparkDraftSampler``) asks the draft's built object for
``draft_sampler`` once, on every rank, and runs the step it returns for a batch size
in place of ``compute_base_logits`` + ``draft_markov_block``. The step projects the
draft hidden states through this rank's LM-head shard (FlashInfer's low-M BF16 kernel
at up to ``LM_HEAD_MAX_ROWS`` rows, SGLang's projection above), then runs the chain
(``sm103/markov.py``): ranks agree on each step's token inside the kernel, so the
proposal needs no per-step TP sync. The draft noise is SGLang's (same Philox key,
counters and offset), drawn on a side stream while the LM head runs.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/distributed/parallel_state.py": ("rank_in_group", "world_size", "cpu_group"),
    "srt/models/deepseek_v4_dspark.py": (
        "markov_head",
        "markov_w1",
        "markov_w2",
        "vocab_size",
        "step_logits_dtype",
        "lm_head",
        "stages",
        "norm",
        "_use_fp32_lm_head",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "num_embeddings_per_partition",
        "weight",
    ),
    "srt/models/dspark.py": ("project_through_lm_head",),
    "srt/speculative/dspark_components/dspark_draft.py": ("key", "offset"),
    "srt/speculative/dspark_components/dspark_draft_sampler.py": (
        "DsparkDraftSampler",
    ),
}

# The SGLang seam the slot needs (H5): without it SGLang never asks for the sampler
SGLANG_SEAMS = {
    "srt/monokernels/runtime.py": ("built_monokernels",),
    "srt/speculative/dspark_components/dspark_draft_sampler.py": (
        "__init__(monokernel_sampler)",
    ),
}

# GB300 TP4 shard (N 32320, K 5120), cold L2: the low-M kernel beats cuBLAS split-K up
# to 24 rows (66.7 vs 72.8 us at 20-24) but not at 32.
LM_HEAD_MAX_ROWS = 24
LM_HEAD_BACKEND = "cute-dsl"


def seam() -> str:
    """Why SGLang cannot run the draft sampler slot, or "" when it can."""
    import importlib
    import inspect

    from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
        DsparkDraftSampler,
    )

    # by name: built_monokernels is a seam, not a read every SGLang provides
    runtime = importlib.import_module("sglang.srt.monokernels.runtime")

    if not hasattr(runtime, "built_monokernels"):
        return "this SGLang lacks monokernels.runtime.built_monokernels"
    if "monokernel_sampler" not in inspect.signature(DsparkDraftSampler).parameters:
        return "this SGLang lacks DsparkDraftSampler's monokernel_sampler"
    return ""


def disable_reason(lm, *, gamma: int, world: int, rank: int) -> str:
    """Why the chain cannot serve this draft, or ""; the same on every rank."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import markov

    head, lm_head = lm.markov_head, lm.lm_head
    if lm_head is None:
        return "the draft has no LM head attached"
    if lm._use_fp32_lm_head:
        return "an FP32 draft LM head (SGLANG_DSPARK_FP32_LM_HEAD)"
    vocab = int(head.vocab_size)
    # the chain reads this rank's base logits, the LM head's even vocab split
    shard = lm_head.shard_indices
    if int(lm_head.num_embeddings_per_partition) != vocab // world or int(
        shard.org_vocab_start_index
    ) != rank * (vocab // world):
        return "an LM-head vocab partition that is not the even split"
    if head.markov_w2.weight.dtype != torch.bfloat16:
        # SGLang's FP32 bias path rounds differently from the kernel's BF16 table
        return "an FP32 markov_w2 (set SGLANG_DSPARK_OPT_MARKOV_W2_BF16=1)"
    return (
        markov.unsupported_reason(
            world=world,
            vocab_size=vocab,
            num_steps=gamma,
            embed_weight=head.markov_w1.weight,
            head_weight=head.markov_w2.weight,
            cache_dtype=head.step_logits_dtype(torch.bfloat16),
            capability=torch.cuda.get_device_capability(),
        )
        or ""
    )


def low_m_lm_head_supported() -> bool:
    major, minor = torch.cuda.get_device_capability()
    try:
        from flashinfer import mm_bf16

        return bool(mm_bf16.is_backend_supported(LM_HEAD_BACKEND, major * 10 + minor))
    except (ImportError, AttributeError):
        return False


class DraftLmHead:
    """This rank's LM-head shard for the draft's rows: FlashInfer's low-M BF16 kernel
    at 1-LM_HEAD_MAX_ROWS rows, SGLang's projection otherwise."""

    def __init__(self, lm_head, *, low_m: bool):
        self.lm_head = lm_head
        # the CuTe DSL kernels take operands over DLPack, which refuses grad tensors
        self.weight_t = lm_head.weight.detach().t()
        self.low_m = low_m and lm_head.weight.dtype == torch.bfloat16

    def takes(self, rows: int) -> bool:
        return self.low_m and rows <= LM_HEAD_MAX_ROWS

    def tune(self, rows: tuple[int, ...]) -> None:
        """Tune the low-M kernel at the row counts the draft runs; the same calls on
        every rank, before any graph is captured."""
        from flashinfer.autotuner import autotune

        weight = self.lm_head.weight
        with autotune(True):
            for r in rows:
                if self.takes(r):
                    self(
                        torch.zeros(
                            r, weight.shape[1], dtype=weight.dtype, device=weight.device
                        )
                    )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.takes(x.shape[0]):
            from sglang.srt.models.dspark import project_through_lm_head

            return project_through_lm_head(x, self.lm_head)
        from flashinfer import mm_bf16

        return mm_bf16(x.contiguous(), self.weight_t, pdl=True, backend=LM_HEAD_BACKEND)


class DraftSampler:
    """The factory SGLang calls (``BuiltMonokernels.draft_sampler``)."""

    def __init__(self, lm, *, deadline_s: float, slots: frozenset[str]):
        self.lm, self.deadline_s = lm, deadline_s
        self.low_m = "draft_lm_head" in slots
        self.chain = None

    def __call__(
        self,
        *,
        gamma: int,
        max_bs: int,
        greedy_draft: bool,
        noise,
        draft_tokens: torch.Tensor,
        step_logits: Optional[torch.Tensor],
        step_temperatures: torch.Tensor,
    ) -> Optional[Callable[[int], Optional[Callable]]]:
        from sglang.srt.runtime_context import get_parallel

        from sgl_nvidia_mega.deepseek_v41.sm103 import markov

        group = get_parallel().tp_group
        world, rank = group.world_size, group.rank_in_group
        lm, head = self.lm, self.lm.markov_head
        reason = disable_reason(lm, gamma=gamma, world=world, rank=rank)
        if not _all_ranks_true(group.cpu_group, not reason):
            if rank == 0:
                logger.info(
                    "sgl-nvidia-mega deepseek_v41 draft sampler: SGLang's (%s)",
                    reason or "another rank cannot run it",
                )
            return None
        # collective over the TP group: every rank builds it or none does
        self.chain = markov.ShardedMarkov(
            embed_weight=head.markov_w1.weight,
            head_weight=head.markov_w2.weight,
            vocab_size=int(head.vocab_size),
            num_steps=gamma,
            cache_dtype=head.step_logits_dtype(torch.bfloat16),
            group=group.cpu_group,
            rank=rank,
            world=world,
            max_bs=max_bs,
            deadline_s=self.deadline_s,
            keeps_logits=not greedy_draft,
        )
        lm_head = DraftLmHead(
            lm.lm_head, low_m=self.low_m and low_m_lm_head_supported()
        )
        lm_head.tune(tuple(bs * gamma for bs in range(1, self.chain.max_rows + 1)))
        steps = {
            bs: _Step(
                lm=lm,
                chain=self.chain,
                lm_head=lm_head,
                bs=bs,
                gamma=gamma,
                greedy_draft=greedy_draft,
                noise=noise,
                draft_tokens=draft_tokens,
                step_logits=step_logits,
                step_temperatures=step_temperatures,
            )
            for bs in range(1, self.chain.max_rows + 1)
        }
        logger.info(
            "sgl-nvidia-mega deepseek_v41 draft sampler on rank %d: Markov chain in "
            "one kernel (%d ranks, batch 1-%d, %d draft tokens), LM head %s at "
            "1-%d rows; wait deadline %.0f s",
            rank,
            world,
            self.chain.max_rows,
            gamma,
            "low-M" if lm_head.low_m else "SGLang's",
            LM_HEAD_MAX_ROWS,
            self.deadline_s,
        )
        return steps.get


class _Step:
    """One batch size's proposal: LM head, then the chain."""

    def __init__(
        self,
        *,
        lm,
        chain,
        lm_head,
        bs,
        gamma,
        greedy_draft,
        noise,
        draft_tokens,
        step_logits,
        step_temperatures,
    ):
        self.lm, self.chain, self.lm_head = lm, chain, lm_head
        self.bs, self.gamma, self.greedy_draft, self.noise = (
            bs,
            gamma,
            greedy_draft,
            noise,
        )
        self.tokens = draft_tokens[: bs * gamma].view(bs, gamma)
        self.step_logits = step_logits
        self.temperatures = step_temperatures
        self.is_logged = False

    def __call__(self, hidden_states: torch.Tensor, anchor_tokens: torch.Tensor):
        if not self.is_logged and torch.cuda.is_current_stream_capturing():
            # engagement, through the model's dispatch, once per captured graph
            logger.info(
                "sgl-nvidia-mega deepseek_v41 draft sampler engaged: batch %d, %d LM-head "
                "rows (%s)",
                self.bs,
                hidden_states.shape[0],
                "low-M" if self.lm_head.takes(hidden_states.shape[0]) else "SGLang's",
            )
            self.is_logged = True
        # the V4.1 draft leaves forward() with its hidden states collapsed
        if not self.greedy_draft:
            self.chain.fork_noise(
                rows=self.bs, key=self.noise.key, offset=self.noise.offset
            )
        x = self.lm.stages[-1].norm(hidden_states)
        base_logits = self.lm_head(x).view(self.bs, self.gamma, -1)
        if not self.greedy_draft:
            self.chain.join_noise()
        self.chain.sample(
            base_logits=base_logits,
            anchor_tokens=anchor_tokens,
            temperatures=self.temperatures,
            logits_cache=self.step_logits,
            draft_tokens=self.tokens,
        )
        if not self.greedy_draft:
            # one block advances the counter once, as draft_markov_block does
            self.noise.offset.add_(1)
        return self.tokens, hidden_states


def _all_ranks_true(cpu_group, value: bool) -> bool:
    import torch.distributed as dist

    flag = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())
