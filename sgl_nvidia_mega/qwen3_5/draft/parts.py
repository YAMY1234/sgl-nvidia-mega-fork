# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""What the draft entries share, built once by the first entry factory SGLang calls
(before it captures the draft graphs, every rank in the same order): the draft layer,
the attention's buffers, and each requested draft slot's build."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Slot draft_fc_rep: the multi-request draft steps of at least this many rows take SGLang's
# replicated fc
FC_REP_MIN_ROWS = int(os.environ.get("SGLANG_Q35_DRAFT_FC_REP_MIN_ROWS", "8"))
# PERF-LP (default off): the draft lm_head of passes of at most HEAD_MXFP8_MAX_ROWS rows from
# a draft-only MXFP8 copy of the shard (TRT-LLM's TRTLLM_Q35_DRAFT_FP8_HEAD format); its GEMV
# is TRT-LLM's CuTe DSL kernel ("cute") or the NVRTC translation ("nvrtc")
HEAD_MXFP8 = os.environ.get("SGLANG_PERFLP_DRAFT_HEAD_MXFP8", "0") == "1"
HEAD_MXFP8_MAX_ROWS = int(
    os.environ.get("SGLANG_PERFLP_DRAFT_HEAD_MXFP8_MAX_ROWS", "16")
)
HEAD_MXFP8_GEMV = os.environ.get("SGLANG_PERFLP_DRAFT_HEAD_MXFP8_GEMV", "cute")

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": (
        "model",
        "fc",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "lm_head",
        "logits_processor",
    ),
    "srt/layers/logits_processor.py": ("logit_scale", "vocab_size"),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "added_vocab_start_index",
        "added_vocab_end_index",
    ),
    "srt/models/qwen3_5.py": (
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "attn",
    ),
    "srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_device_group",
        "tp_cpu_group",
        "page_size",
    ),
}


class Parts:
    """The draft model's modules the entries run, and the slot builds."""

    def __init__(self, draft, layer):
        self.draft = draft
        self.lm = draft.lm  # Qwen3_5ForCausalLMMTP
        self.model = draft.lm.model  # its one-layer Qwen3_5ForCausalLM
        self.layer = layer  # the MTP decoder layer (full attention + MoE)
        self.binding = draft.binding
        self.slots = draft.slots
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.attention = None  # draft.attention.Attention
        self.moe: Optional[Any] = None  # draft.moe.DraftMoe (slot draft_moe)
        self.hop: Optional[Any] = None  # draft.hop.DraftHop (slot draft_hop)
        self.mkd: Optional[Any] = None  # draft.xchg.MkdBoundary (slot draft_xchg)
        # sm103.draft.top1.Top1Exchange: the multi-request steps' sharded top-1 (slot
        # draft_multi)
        self.top1x: Optional[Any] = None
        # sm103.draft.embed.EmbedExchange: those steps' embedding rows (slot draft_embed)
        self.embedx: Optional[Any] = None
        # the full embedding table [vocab, hidden] on this rank (slot draft_embed_full)
        self.embed_full: Optional[torch.Tensor] = None
        # (entry, bs) pairs whose capture logged the engagement line
        self.engaged: set = set()
        # sm103.draft.head_mx8.DraftHeadMx8 (SGLANG_PERFLP_DRAFT_HEAD_MXFP8)
        self.head_mx8: Optional[Any] = None
        # draft.moe_mkm.DraftMkm: the draft MoE on the target's MoE path (SGLANG_PERFDRAFT_MOE_MKM)
        self.moe_mkm: Optional[Any] = None

    @classmethod
    def build(cls, draft) -> Parts:
        from sgl_nvidia_mega.qwen3_5.draft.attention import Attention

        m = draft.lm.model
        layers = [m.layers[i] for i in range(m.start_layer, m.end_layer)]
        if len(layers) != 1:
            raise RuntimeError(
                f"NVIDIA megakernel: the MTP draft has {len(layers)} layers, not 1"
            )
        parts = cls(draft, layers[0])
        parts.attention = Attention(parts)
        slots = parts.slots
        if "draft_moe" in slots:
            parts.moe = parts._build_moe()
        if "draft_hop" in slots:
            parts.hop = parts._build_hop()
        if "draft_xchg" in slots:
            parts.mkd = parts._build_mkd()
        if "draft_multi" in slots:
            parts.top1x = parts._build_top1x()
        if "draft_embed" in slots:
            parts.embedx = parts._build_embedx()
        if "draft_embed_full" in slots:
            parts.embed_full = parts._build_embed_full()
        if HEAD_MXFP8:
            parts.head_mx8 = parts._build_head_mx8()
            from sgl_nvidia_mega.qwen3_5.draft import layer as draft_layer

            if draft_layer.TOP1_KEYS and parts.top1x is not None:
                parts.head_mx8.prepare_keys()
        from sgl_nvidia_mega.qwen3_5.draft import moe_mkm

        if moe_mkm.ENABLED:
            parts.moe_mkm, why = moe_mkm.DraftMkm.build(parts.layer.mlp)
            if parts.moe_mkm is None:
                raise RuntimeError(f"NVIDIA megakernel: SGLANG_PERFDRAFT_MOE_MKM cannot run: {why}")
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft parts built (slots %s; MoE %s, hop "
            "%s, exchange %s, embedding %s, multi-request fc %s)",
            sorted(slots),
            ("target path" + (f" from {moe_mkm.MIN_ROWS} rows" if moe_mkm.MIN_ROWS > 1 else "") + ", ")
            * (parts.moe_mkm is not None)
            + ("single launch" if parts.moe is not None else "SGLang's"),
            "draft hop" if parts.hop is not None else "SGLang's",
            "MKD" if parts.mkd is not None else "SGLang's",
            "full table" if parts.embed_full is not None
            else "exchange" if parts.embedx is not None else "SGLang's",
            f"replicated from {FC_REP_MIN_ROWS} rows" if "draft_fc_rep" in slots
            else "row-split up to 16 rows",
        )
        return parts

    def _build_moe(self):
        from sgl_nvidia_mega.qwen3_5.draft.moe import DraftMoe

        moe = DraftMoe.build(self.layer.mlp, self.device.index)
        if moe is None:
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_moe cannot run: "
                f"{getattr(DraftMoe, 'last_reason', 'see the log')}"
            )
        return moe

    def _build_hop(self):
        from sgl_nvidia_mega.qwen3_5.draft.hop import GEMV_O_PROJ, DraftHop

        tp = self.binding.tp_size
        hop, why = DraftHop.build(
            tp_size=tp,
            deadline_s=self.draft.deadline_s,
            o_proj_k=(
                int(self.layer.o_proj.weight.shape[1])
                if GEMV_O_PROJ.get(tp, False)
                else None
            ),
            early=True,
            early_head=True,
        )
        if hop is None:
            raise RuntimeError(f"NVIDIA megakernel: slot draft_hop cannot run: {why}")
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft hop releases at entry: the "
            "single-launch MoE %s, the draft head %s",
            hop.early,
            hop.early_head,
        )
        return hop

    def _build_mkd(self):
        from sgl_nvidia_mega.qwen3_5.draft.xchg import MkdBoundary

        lm, b = self.lm, self.binding
        embed = self.model.embed_tokens
        processor = lm.logits_processor
        # what 2.2's boundary assumed of the draft head and MkdBoundary cannot see
        refused = [
            why
            for bad, why in (
                (lm.config.tie_word_embeddings, "tied embeddings"),
                (processor.logit_scale is not None, "logit scaling"),
                (
                    getattr(processor, "final_logit_softcapping", None) is not None,
                    "logit softcapping",
                ),
            )
            if bad
        ]
        mkd = None
        if not refused:
            mkd = MkdBoundary.maybe_create(
                vocab_size=processor.vocab_size,
                fc_weight=lm.fc.weight,
                norm_embedding=lm.pre_fc_norm_embedding,
                norm_hidden=lm.pre_fc_norm_hidden,
                embed_shard=embed.weight,
                vocab_start=embed.shard_indices.org_vocab_start_index,
                lm_head_shard=lm.lm_head.weight,
                embed=embed,
                rank=b.tp_rank,
                world=b.tp_size,
                device_group=b.tp_device_group,
                cpu_group=b.tp_cpu_group,
                num_steps=b.num_steps,
            )
        if mkd is None:
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_xchg cannot run: "
                + ("; ".join(refused) or "see the MKD draft boundary warning")
            )
        return mkd

    def _build_top1x(self):
        from sgl_nvidia_mega.qwen3_5.sm103.draft.top1 import Top1Exchange

        lm, b = self.lm, self.binding
        rows = int(lm.lm_head.weight.shape[0])
        start = int(lm.lm_head.shard_indices.org_vocab_start_index)
        vocab = int(lm.logits_processor.vocab_size)
        # the shard layout the all-gathered logits have: rank-major, no padding rows
        if start != b.tp_rank * rows or rows * b.tp_size != vocab:
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_multi cannot run: the LM head's "
                f"{rows}-row shard at {start} is not rank {b.tp_rank}'s slice of "
                f"{vocab} ids"
            )
        return Top1Exchange(
            world=b.tp_size,
            rank=b.tp_rank,
            vocab_local=rows,
            vocab_start=start,
            vocab_size=vocab,
            group=b.tp_device_group,
            cpu_group=b.tp_cpu_group,
            device=self.device,
        )

    def _build_embedx(self):
        from sgl_nvidia_mega.qwen3_5.sm103.draft.embed import EmbedExchange

        b = self.binding
        embed = self.model.embed_tokens
        shard = embed.shard_indices
        if shard.added_vocab_end_index != shard.added_vocab_start_index:
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_embed cannot run: the embedding has "
                "added vocabulary"
            )
        return EmbedExchange(
            world=b.tp_size,
            table=embed.weight.data,
            vocab_start=int(shard.org_vocab_start_index),
            vocab_local=int(shard.org_vocab_end_index - shard.org_vocab_start_index),
            group=b.tp_device_group,
            cpu_group=b.tp_cpu_group,
            device=self.device,
        )

    def _build_embed_full(self) -> torch.Tensor:
        """Every rank's shard of the embedding gathered into one table, -0.0 written as
        +0.0 (what the vocabulary-parallel embedding's all-reduce of zero rows returns)."""
        import torch.distributed as dist

        b = self.binding
        embed = self.model.embed_tokens
        shard = embed.shard_indices
        start = int(shard.org_vocab_start_index)
        local = int(shard.org_vocab_end_index - shard.org_vocab_start_index)
        weight = embed.weight.data
        layout = torch.tensor([start, local], dtype=torch.int64, device=self.device)
        layouts = torch.empty(b.tp_size, 2, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(layouts, layout, group=b.tp_device_group)
        want = [[r * local, local] for r in range(b.tp_size)]
        if (
            shard.added_vocab_end_index != shard.added_vocab_start_index
            or weight.dtype != torch.bfloat16
            or layouts.tolist() != want
        ):
            raise RuntimeError(
                "NVIDIA megakernel: slot draft_embed_full cannot run: the embedding "
                f"shards {layouts.tolist()} ({weight.dtype}) are not equal rank-major "
                "slices without added vocabulary"
            )
        rows = weight[:local].clone()
        bits = rows.view(torch.int16)
        bits.masked_fill_(bits == -32768, 0)
        full = torch.empty(
            b.tp_size * local, rows.shape[1], dtype=rows.dtype, device=self.device
        )
        dist.all_gather_into_tensor(full, rows, group=b.tp_device_group)
        del rows, bits
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft embedding: the full table "
            "[%d, %d] on every rank (%.2f GB)",
            full.shape[0],
            full.shape[1],
            full.numel() * full.element_size() / 2**30,
        )
        return full

    def _build_head_mx8(self):
        """PERF-LP: the draft-only MXFP8 copy of this rank's lm_head shard (the target's
        bf16 shard is only read), its GEMV for 1..HEAD_MXFP8_MAX_ROWS rows and, with slot
        draft_xchg, the MKD one-row head on it."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8 import DraftHeadMx8

        head = DraftHeadMx8(
            self.lm.lm_head.weight.data,
            max_rows=HEAD_MXFP8_MAX_ROWS,
            gemv=HEAD_MXFP8_GEMV,
        )
        if self.mkd is not None:
            self.mkd.set_head_mx8(head)
        logger.info(
            "PERF-LP MXFP8 draft head active on TP rank %d: %s; MKD one-row head %s",
            self.binding.tp_rank,
            head.describe(),
            "MXFP8" if self.mkd is not None and self.mkd.head_mx8 is head else "none",
        )
        return head

    def fc_replicated(self, rows: int) -> bool:
        """Whether a multi-request draft step of ``rows`` rows takes SGLang's replicated
        fc (slot draft_fc_rep)."""
        return "draft_fc_rep" in self.slots and rows >= FC_REP_MIN_ROWS

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The embedding rows of ``token_ids``: slot draft_embed_full's table, else slot
        draft_embed's exchange where it is built and the rows fit it, else SGLang's
        vocabulary-parallel embedding."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.embed import MAX_ROWS

        if self.embed_full is not None:
            return self.embed_full.index_select(0, token_ids)
        if self.embedx is not None and token_ids.shape[0] <= MAX_ROWS:
            return self.embedx(token_ids)
        return self.model.embed_tokens(token_ids)

    def note_engaged(self, entry: str, bs: int, paths: str = "") -> None:
        """Log once per (entry, bs) captured: the engagement line every rank prints."""
        if torch.cuda.is_current_stream_capturing() and (entry, bs) not in self.engaged:
            self.engaged.add((entry, bs))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MTP draft %s captured for batch size %d "
                "(slots %s)%s",
                entry,
                bs,
                sorted(self.slots),
                f": {paths}" if paths else "",
            )

    def embed_path(self, rows: int) -> str:
        """Which lookup ``embed`` takes for ``rows`` rows (for the engagement line)."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.embed import MAX_ROWS

        if self.embed_full is not None:
            return "embedding full table"
        if self.embedx is not None and rows <= MAX_ROWS:
            return "embedding exchange"
        return "embedding SGLang's"
