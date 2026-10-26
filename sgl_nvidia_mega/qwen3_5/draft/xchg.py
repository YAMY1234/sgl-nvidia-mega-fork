# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/srt/speculative/mkd_draft_boundary.py
# (MkdDraftBoundary), draft_vocab_exchange.py (the one-launch boundary's path) and
# python/sglang/srt/models/qwen3_5_mtp.py (the row-split fc), on branch
# nvidia/qwen35-agentx-gb300.
"""The greedy one-row MTP draft step boundary (MKD) in two launches instead of seven.

A draft step's lm_head runs on this rank's vocabulary shard (``head``, which also leaves
each CTA's best row), and one exchange launch (``exchange``) takes the shard argmax,
multicasts each rank's candidate together with its hidden-half fc partial (computed
beside the lm_head on a side stream, ``hidden_partial``), selects the same winner on every
rank and writes the next step's fc output (``fc``). The candidates are memo rows (the
T-table): per vocabulary row of the shard, the embedding-half fc partials already summed
in rank order, built at construction with the split fc's arithmetic and checked against
it on every row. The draft extend's exchange (``exchange_extend``) also writes the next
draft's first fc (extend fc); ``fill_fc`` writes it when no extend did.

Every value equals the row-split fc path bit for bit: the same argmax keys and ties, the
same per-slice dot arithmetic, the same rank-order fp32 sum and bf16 rounding. The step
state (``armed``, ``pending``, ``last_step``, ``step``) follows 2.2's: ``begin_step``
arms a one-row step, the exchange leaves ``pending`` for the next step's ``fc``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Callable, Optional

import torch

from sgl_nvidia_mega.qwen3_5.sm103.draft import xchg as kernels

logger = logging.getLogger(__name__)

# SGLANG_PERFLL_DRAFT_FC_SUM=1: the split fc's gather and rank-order sum in one launch
# (kernels.FcAllsum; the same values as FcGather + ordered_partial_sum)
FC_SUM = os.environ.get("SGLANG_PERFLL_DRAFT_FC_SUM", "0") == "1"
# SGLANG_PERFLL_DRAFT_FC_GEMV=1: the draft extend's partial as the fused norm + slice GEMV
# (draft_fc_partial_ext) in place of the norm module and cuBLAS: the GEMV's dot order, as the
# draft steps' partials and the memo rows have, not cuBLAS's (tokens may differ)
FC_GEMV = os.environ.get("SGLANG_PERFLL_DRAFT_FC_GEMV", "0") == "1"

# SGLANG_PERFDRAFT_EXTEND_MKD_HEAD=1 (PERF-DRAFT, default off; bitwise): the one-request draft extend's head runs as a
# draft step's (extend_head): the one-row MKD head, whose per-CTA maxima spare the exchange its scan of the shard logits,
# with the next draft's hidden partial beside it on the side stream, in place of the shard GEMV and a serial partial
EXTEND_MKD_HEAD = os.environ.get("SGLANG_PERFDRAFT_EXTEND_MKD_HEAD", "0") == "1"

# Q35FLY (default off, equivalent arithmetic): head CTAs compute the hidden-half
# FC in their weight-prefetch window, then all CTAs continue the head.
HEAD_PARTIAL = os.environ.get("SGLANG_Q35FLY_DRAFT_HEAD_PARTIAL", "0") == "1"
HEAD_EXCHANGE = os.environ.get("SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE", "0") == "1"
# Independent HP2 grid experiment; the original head allocation stays unchanged.
HEAD_EXCHANGE_GRID120 = os.environ.get("SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE_GRID120", "0") == "1"

# Memo rows built per launch at construction
_MEMO_BLOCK = 4096
# CTAs of mkd_draft_head per SM (T1, TP8 shard: 1 -> 39.3 us, 2 -> 40.5 us, 3 -> 41.6 us)
_HEAD_CTAS_PER_SM = 1
# Unused dynamic shared memory per head CTA: an early-released head CTA then starts only on
# an SM with 128 KB of shared memory free (its registers already hold the head at one CTA
# per SM)
HEAD_SMEM_PAD = 120 * 1024
# SMs a hidden rank leaves to the side-stream hidden partial: the early-released head
# would fill every SM, and the partial (which the exchange waits for) would start only as
# head CTAs retire; the head then runs on the other SMs. SGLANG_PERFDRAFT_MKD_PARTIAL_SMS
# (PERF-DRAFT) sets it: with the four-row partial (SGLANG_PERFDRAFT_MKD_PARTIAL_ROWS4) fewer
# SMs still end it before the head, and the hidden ranks' head gets the rest back
PARTIAL_SMS = int(os.environ.get("SGLANG_PERFDRAFT_MKD_PARTIAL_SMS", "24"))
# Draft and draft-extend widths that take the row-split fc (2.2 _FC_SPLIT_MAX_TOKENS)
FC_SPLIT_MAX_TOKENS = kernels.GATHER_MAX_TOKENS
# Rows of a draft step the vocabulary exchange serves (2.2 DraftVocabExchange.MAX_ROWS)
MAX_ROWS = 2


def fc_split_slice(
    hidden: int, rank: int, world: int
) -> Optional[tuple[bool, int, int]]:
    """(uses the hidden half, start inside the half, width) of this rank's fc input
    columns, or None when TP ``world`` does not split them."""
    width = 2 * hidden // world
    if world == 1 or (2 * hidden) % world != 0 or hidden % width != 0:
        return None
    start = rank * width
    return (start >= hidden, start % hidden, width)


def unsupported_reason(
    *,
    hidden: int,
    world: int,
    rank: int,
    vocab_size: int,
    embed_shard: torch.Tensor,
    lm_head_shard: torch.Tensor,
) -> Optional[str]:
    """Why the boundary cannot serve this draft, or None (2.2's MKD and vocabulary-exchange
    checks that the tensors decide; the caller checks the logits processor: untied
    embeddings, an unquantized lm_head, no logit scaling or softcapping)."""
    if world not in kernels.WORLDS:
        return f"tp_size {world}"
    if hidden != kernels.HIDDEN:
        return f"hidden size {hidden}"
    if fc_split_slice(hidden, rank, world) is None:
        return f"the fc does not split at TP {world}"
    sms = torch.cuda.get_device_properties(embed_shard.device).multi_processor_count
    if kernels.draft_fc_partial_rows(hidden, 2 * hidden // world, sms) is None:
        return "the fused fc norm + slice GEMV does not fit in shared memory"
    for name, w in (("embedding", embed_shard), ("lm_head", lm_head_shard)):
        if w.dtype != torch.bfloat16:
            return f"{name} dtype {w.dtype}"
    if embed_shard.shape != lm_head_shard.shape:
        return "embedding and lm_head shards differ"
    if embed_shard.shape[0] % 8 != 0 or embed_shard.shape[1] % 8 != 0:
        return "shard shape not 16-byte tileable"
    if embed_shard.shape[0] * world != vocab_size:
        return "full-vocabulary shards with padding rows"
    return None


def _clear_negative_zero(rows: torch.Tensor) -> torch.Tensor:
    # The exchanged embedding row has -0.0 cleared (vocab_argmax_pack); memo from the same row.
    bits = rows.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(
        torch.bfloat16
    )


def _source_digests() -> str:
    return " ".join(
        f"{path.name}={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
        for path in sorted(kernels.SRC.iterdir())
        if path.suffix in (".cu", ".cuh") and path.stem.startswith(("mkd_", "draft_fc"))
    )


class MkdBoundary:
    """Exchange buffers, the memo table, the split fc and the per-step launches of one
    rank. Construct on every rank of the TP group (collectives) after the draft shares the
    target's embedding and lm_head, before any capture.

    ``norm_embedding`` / ``norm_hidden``: the MTP model's pre-fc Gemma RMSNorms (their
    ``weight``, ``variance_epsilon``, and the module itself for splits over 2 rows);
    ``embed``: the vocabulary-parallel embedding lookup ([n] ids -> [n, hidden], with its
    all-reduce), which ``fill_fc`` runs."""

    def __init__(
        self,
        *,
        fc_weight: torch.Tensor,
        norm_embedding,
        norm_hidden,
        embed_shard: torch.Tensor,
        vocab_start: int,
        lm_head_shard: torch.Tensor,
        embed: Callable[[torch.Tensor], torch.Tensor],
        rank: int,
        world: int,
        device_group,
        cpu_group,
        num_steps: int,
        memo: bool = True,
        extend_fc: bool = True,
        head: bool = True,
        head_smem_pad: int = HEAD_SMEM_PAD,
        partial_sms: int = PARTIAL_SMS,
    ) -> None:
        import torch.distributed as dist

        dev = fc_weight.device
        hidden = int(fc_weight.shape[0])
        self.hidden, self.world, self.rank, self.num_steps = (
            hidden,
            world,
            rank,
            num_steps,
        )
        self.fc_weight, self.norm_embedding, self.norm_hidden = (
            fc_weight,
            norm_embedding,
            norm_hidden,
        )
        self._embed = embed
        self._device_group = device_group
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # the MTP model's row-split fc: this rank's input-column slice and its GEMV rows
        self.fc_split_slice = fc_split_slice(hidden, rank, world)
        use_hidden, start, width = self.fc_split_slice
        col = start + (hidden if use_hidden else 0)
        self._fc_shard = fc_weight[:, col : col + width].contiguous()
        self._fc_gemv_rows = kernels.draft_fc_partial_rows(hidden, width, sms)
        self._fc_gather = kernels.FcGather(
            world=world, rank=rank, group=device_group, cpu_group=cpu_group, device=dev
        )
        # the draft extend's rows (the verify's: num_steps draft tokens + 1), which the
        # opt-in builds serve
        self.ext_rows = num_steps + 1
        self._fc_ext_rows = (
            kernels.draft_fc_partial_ext_rows(self.ext_rows, hidden, width, sms)
            if FC_GEMV
            else None
        )
        if self._fc_ext_rows is not None:
            kernels.build_fc_ext(dev.index, world, self._fc_ext_rows, self.ext_rows)
        self._fc_allsum = (
            kernels.FcAllsum(
                world=world,
                rank=rank,
                group=device_group,
                cpu_group=cpu_group,
                device=dev,
                ext_rows=self.ext_rows,
            )
            if FC_SUM
            else None
        )
        self._fc_noted: set = set()
        self.is_hidden_rank = use_hidden
        self.slice_start, self.width = start, width
        self.n_embed = hidden // width
        self.memo = memo
        rows = int(embed_shard.shape[0])
        self.embed_weight, self.local_vocab = embed_shard, rows
        if vocab_start != rank * rows:
            raise RuntimeError(
                "MKD draft boundary: vocabulary shard is not the rank-th equal slice"
            )
        # Full vocabulary: candidate position p (rank-major) is vocabulary id p.
        self.hot_ids = torch.arange(rows * world, dtype=torch.int64).to(dev)
        self.cand = self._build_memo() if memo else embed_shard

        words = kernels.mkd_xchg_words(hidden, world, memo)
        self._buf, self._handle, self.mc = kernels.symmetric_words(
            words, device_group, dev
        )
        self.uc = self._buf.data_ptr()
        torch.cuda.synchronize(dev)
        dist.barrier(group=cpu_group)

        self.epoch_done = torch.tensor([1, 0], dtype=torch.int32, device=dev)
        self.own_partial = torch.zeros(1, hidden, dtype=torch.float32, device=dev)
        # [CTA counter, epoch flag] of the side-stream hidden partial (mkd_hidden_partial).
        self.partial_sync = torch.zeros(2, dtype=torch.int32, device=dev)
        # Per-CTA shard-argmax partials for large (full-vocabulary) shards.
        self.scan_part = torch.zeros(
            kernels.SLOTS * kernels.CHUNKS, dtype=torch.int64, device=dev
        )
        self.out_partials = torch.zeros(
            world - self.n_embed, hidden, dtype=torch.float32, device=dev
        )
        self.out_cand = torch.empty(
            1, hidden, dtype=torch.float32 if memo else torch.bfloat16, device=dev
        )
        self.fc_out = torch.empty(1, hidden, dtype=torch.bfloat16, device=dev)
        # Memo mode: the exchange writes the next step's fc output here directly.
        # Zeros: graph-capture warmups read it before any exchange has written it.
        self.fc_buf = (
            torch.zeros(1, hidden, dtype=torch.bfloat16, device=dev) if memo else None
        )
        # The draft extend's exchange only needs the token; its payload lands here unused.
        self.ext_cand = torch.empty_like(self.out_cand)
        self.ext_partials = torch.empty_like(self.out_partials)
        self.side_stream = torch.cuda.Stream(device=dev)
        self.lm_head_weight = lm_head_shard
        self.head_pad = head_smem_pad
        self.partial_sms = partial_sms
        self._install_head(head, dev)
        # Capability is built once, even when the first same-boot A/B arm is off.
        # Capture-time dispatch below follows each arm's environment. This does
        # not depend on the optional SGLang ab_variants helper (absent in R11).
        head_partial_variants = json.loads(os.environ.get("Q35MF_AB_VARIANTS") or "{}")
        self._head_partial_ab = bool(head_partial_variants)
        self._head_partial_available = self.head_part is not None and (
            HEAD_PARTIAL
            or any(v.get("SGLANG_Q35FLY_DRAFT_HEAD_PARTIAL") == "1"
                   for v in head_partial_variants.values())
        )
        if self._head_partial_available and self.is_hidden_rank:
            from sgl_nvidia_mega.qwen3_5.sm103.draft.head_partial import prepare

            prepare(dev.index, self.world, self.head_pad, False)
            logger.info("Q35FLY draft head+partial prepared on hidden rank %d", self.rank)
        self._head_exchange_available = HEAD_EXCHANGE or any(
            v.get("SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE") == "1"
            for v in head_partial_variants.values()
        )
        self._head_exchange_input = None
        self._head_exchange_done = None
        if self._head_exchange_available:
            from sgl_nvidia_mega.qwen3_5.sm103.draft.head_exchange import CLUSTER, prepare

            if self.head_part is None or not memo:
                raise RuntimeError("HP2 requires an installed MKD head and memo candidates")
            self._head_exchange_ctas = self.head_part.numel() // CLUSTER * CLUSTER
            self._head_exchange_grid120_available = HEAD_EXCHANGE_GRID120 or any(
                v.get("SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE_GRID120") == "1"
                for v in head_partial_variants.values()
            )
            self._head_exchange_grids = (self._head_exchange_ctas,)
            if self._head_exchange_grid120_available:
                if 120 > min(self.head_part.numel(), (self.lm_head_weight.shape[0] + 7) // 8):
                    raise ValueError("HP2 grid120 requires at least 120 head parts and 960 vocabulary rows")
                if self._head_exchange_ctas != 120:
                    self._head_exchange_grids += (120,)
            self._head_exchange_done = torch.zeros(1, dtype=torch.int32, device=dev)
            for ctas in self._head_exchange_grids:
                prepare(dev.index, self.world, self.head_pad, False, ctas)
        # extend_fc: the draft extend's exchange writes fc_buf for the next draft's first
        # step. input_seq counts calls that produce a draft input; fc_buf_key is the
        # input_seq whose extend wrote fc_buf.
        self.extend_fc = memo and extend_fc
        self.input_seq = 0
        self.fc_buf_key: Optional[int] = None
        # armed: this draft step ends in the exchange; pending: the next step's fc reads it;
        # last_step: no draft step follows, so the exchange carries the token alone.
        self.armed = False
        self.pending = False
        self.step = 0
        self.last_step = False
        self.partial_pending = False
        self._partial_side_stream_pending = False
        self._extend_head_noted = False
        # the vocabulary exchange's per-step outputs (DraftVocabExchange)
        self.num_rows = 1
        self.topk_index = [
            torch.empty(MAX_ROWS, 1, dtype=torch.int64, device=dev)
            for _ in range(num_steps)
        ]
        self.topk_p = [
            torch.empty(MAX_ROWS, 1, dtype=torch.float32, device=dev)
            for _ in range(num_steps)
        ]
        logger.info(
            "MKD draft boundary active (tp=%d, %s candidates, %.1f MB exchange buffer; "
            "split fc of %d extend rows: %s partial, %s); sources %s.",
            world,
            "memo" if memo else "embedding",
            words * 8 / 2**20,
            self.ext_rows,
            "fused GEMV" if self._fc_ext_rows is not None else "cuBLAS",
            "one-launch all-sum"
            if self._fc_allsum is not None
            else "gather + ordered sum",
            _source_digests(),
        )

    @classmethod
    def maybe_create(cls, *, vocab_size: int, **kwargs) -> Optional[MkdBoundary]:
        """The boundary, or None (logged) when ``unsupported_reason`` refuses it; every
        rank decides alike."""
        fc_weight = kwargs["fc_weight"]
        reason = unsupported_reason(
            hidden=int(fc_weight.shape[0]),
            world=kwargs["world"],
            rank=kwargs["rank"],
            vocab_size=vocab_size,
            embed_shard=kwargs["embed_shard"],
            lm_head_shard=kwargs["lm_head_shard"],
        )
        if reason is not None:
            logger.warning(
                "MKD draft boundary disabled (%s); using the stock path.", reason
            )
            return None
        return cls(**kwargs)

    # ---------------------------------------------------------------- memo table

    def _build_memo(self) -> torch.Tensor:
        import torch.distributed as dist

        emb = self.embed_weight
        rows = emb.shape[0]
        memo = torch.empty(rows, self.hidden, dtype=torch.float32, device=emb.device)
        norm = self.norm_embedding
        for i in range(0, rows, _MEMO_BLOCK):
            block = _clear_negative_zero(emb[i : i + _MEMO_BLOCK].clone())
            kernels.mkd_fc_memo(
                block,
                self.fc_weight,
                norm.weight.data,
                norm.variance_epsilon,
                self.world,
                memo[i : i + _MEMO_BLOCK],
            )
        bad = self._check_memo(emb, memo)
        bad_t = torch.tensor([bad, rows], dtype=torch.int64, device=emb.device)
        dist.all_reduce(bad_t, group=self._device_group)
        if int(bad_t[0]) != 0:
            raise RuntimeError(
                f"MKD memo table differs from the split fc on {int(bad_t[0])} of "
                f"{int(bad_t[1])} rows"
            )
        logger.info(
            "MKD memo table: %d rows on this rank (%.1f MB), equal to the split fc on all %d rows.",
            rows,
            memo.numel() * 4 / 2**20,
            int(bad_t[1]),
        )
        return memo

    def _check_memo(self, emb: torch.Tensor, memo: torch.Tensor) -> int:
        """Rows whose memo differs from the split-fc kernels (every row, all slices)."""
        norm = self.norm_embedding
        sms = torch.cuda.get_device_properties(emb.device).multi_processor_count
        width = self.width
        gemv_rows = kernels.draft_fc_partial_rows(self.hidden, width, sms)
        shards = [
            self.fc_weight[:, s * width : (s + 1) * width].contiguous()
            for s in range(self.n_embed)
        ]
        part = torch.empty(1, self.hidden, dtype=torch.float32, device=emb.device)
        bad = torch.zeros((), dtype=torch.int64, device=emb.device)
        for v in range(emb.shape[0]):
            x = _clear_negative_zero(emb[v : v + 1].clone())
            acc = None
            for s in range(self.n_embed):
                kernels.draft_fc_partial(
                    x,
                    norm.weight.data,
                    norm.variance_epsilon,
                    shards[s],
                    s * width,
                    part,
                    gemv_rows,
                )
                acc = part.clone() if acc is None else acc + part
            bad += (acc.view(torch.int32) != memo[v : v + 1].view(torch.int32)).any()
        return int(bad)

    # ---------------------------------------------------------------- draft lm_head

    def _install_head(self, enabled: bool, dev: torch.device) -> None:
        self.head_part = None
        self.head_part_cur = None
        self.head_used = False
        # PERF-LP: the head reads the draft-only MXFP8 copy (set_head_mx8)
        self.head_mx8 = None
        if not enabled:
            return
        head = self.lm_head_weight
        if (
            head.dtype != torch.bfloat16
            or head.dim() != 2
            or head.shape[1] != self.hidden
            or head.shape[0] > 65536
        ):
            logger.warning(
                "MKD draft head disabled (lm_head %s %s).",
                head.dtype,
                tuple(head.shape),
            )
            return
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.head_part = torch.zeros(
            _HEAD_CTAS_PER_SM * sms, dtype=torch.int32, device=dev
        )
        logger.info(
            "MKD draft head active: %d x %d shard, %d CTAs, smem pad %d B, hidden ranks "
            "leave %d SMs to the fc partial.",
            head.shape[0],
            head.shape[1],
            self.head_part.numel(),
            self.head_pad,
            self.partial_sms,
        )

    @property
    def fuse_head_partial(self) -> bool:
        # This runs while building/capturing the draft step, not on graph replay.
        # An A arm must retain its original side stream and 24-SM reservation.
        enabled = (
            os.environ.get("SGLANG_Q35FLY_DRAFT_HEAD_PARTIAL", "0") == "1"
            if self._head_partial_ab else HEAD_PARTIAL
        )
        return self._head_partial_available and enabled

    @property
    def fuse_head_exchange(self) -> bool:
        enabled = (
            os.environ.get("SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE", "0") == "1"
            if self._head_partial_ab else HEAD_EXCHANGE
        )
        return self._head_exchange_available and enabled

    @property
    def _head_exchange_capture_ctas(self) -> int:
        # Called only by head/extend construction, never by captured graph replay.
        if not self._head_exchange_grid120_available:
            return self._head_exchange_ctas
        enabled = (
            os.environ.get("SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE_GRID120", "0") == "1"
            if self._head_partial_ab else HEAD_EXCHANGE_GRID120
        )
        return 120 if enabled else self._head_exchange_ctas

    def _partial_ctas(self) -> int:
        """SMs left to the side-stream hidden partial on a hidden rank's step that sends
        one; 0 keeps one partial CTA per SM, queued behind the head."""
        if self.fuse_head_partial or self.fuse_head_exchange or not self.is_hidden_rank or self.last_step or self.head_part is None:
            return 0
        return max(0, min(self.partial_sms, self.head_part.numel() - 1))

    def head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The one-row draft logits [1, N] bf16 on this rank's shard (the exchange then
        reads the per-CTA maxima). For an armed step with the head installed."""
        x = hidden_states.to(torch.bfloat16).contiguous()
        logits = torch.empty(
            1, self.lm_head_weight.shape[0], dtype=torch.bfloat16, device=x.device
        )
        # The head's grid is its part count: with SMs left to the hidden partial it runs on
        # the rest
        ctas = self.head_part.numel() - self._partial_ctas()
        self.head_part_cur = self.head_part[:ctas]
        if self.fuse_head_exchange:
            self.head_part_cur = self.head_part[:self._head_exchange_capture_ctas]
            self._head_exchange_input = (x, self.step % 2 == 1,
                                         self.is_hidden_rank and not self.last_step)
            self.head_used = True
            return logits
        # Alternate the row walk per draft step: the next step re-reads first what is still in L2
        # (T1: -1.9 to -2.4 us per call on the TP8 shard).
        if self.fuse_head_partial and self.is_hidden_rank and not self.last_step:
            self._head_with_partial(x, logits, reverse=self.step % 2 == 1)
        elif self.head_mx8 is not None:
            self.head_mx8.mkd_head(
                x,
                logits,
                self.head_part_cur,
                reverse=self.step % 2 == 1,
                smem_pad=self.head_pad,
            )
        else:
            kernels.mkd_draft_head(
                x,
                self.lm_head_weight,
                logits,
                self.head_part_cur,
                reverse=self.step % 2 == 1,
                smem_pad=self.head_pad,
            )
        self.head_used = True
        return logits

    def _head_with_partial(self, x, logits, *, reverse):
        from sgl_nvidia_mega.qwen3_5.sm103.draft.head_partial import head_with_partial

        norm = self.norm_hidden
        head_with_partial(
            x, self.lm_head_weight, logits, self.head_part_cur, self._fc_shard,
            norm.weight.data, norm.variance_epsilon, self.slice_start,
            self.own_partial, self.epoch_done, self.partial_sync, self.world,
            reverse=reverse, smem_pad=self.head_pad, head_mx8=self.head_mx8,
        )
        self.partial_pending = True

    def _launch_head_exchange(self, exchange):
        from sgl_nvidia_mega.qwen3_5.sm103.draft.head_exchange import launch

        x, reverse, with_partial = self._head_exchange_input
        norm = self.norm_hidden
        launch(x, self.lm_head_weight, self._fc_shard, norm.weight.data,
               norm.variance_epsilon, self.slice_start, self.own_partial,
               self.epoch_done, self.partial_sync, self._head_exchange_done,
               self.world, exchange, reverse=reverse, smem_pad=self.head_pad,
               head_mx8=self.head_mx8, with_partial=with_partial)
        self._head_exchange_input = None

    def set_head_mx8(self, head_mx8) -> None:
        """PERF-LP: the one-row head reads ``head_mx8``'s draft-only MXFP8 copy of the
        lm_head shard (``mkd_head_mx8``: the same logits rounding and per-CTA maxima)."""
        if self.head_part is None:
            return
        head_mx8.prepare_mkd(self.head_pad)
        self.head_mx8 = head_mx8
        if self._head_partial_available and self.is_hidden_rank:
            from sgl_nvidia_mega.qwen3_5.sm103.draft.head_partial import prepare

            prepare(self.lm_head_weight.device.index, self.world, self.head_pad, True)
        if self._head_exchange_available:
            from sgl_nvidia_mega.qwen3_5.sm103.draft.head_exchange import prepare

            for ctas in self._head_exchange_grids:
                prepare(self.lm_head_weight.device.index, self.world, self.head_pad,
                        True, ctas)

    def head_follows(self) -> bool:
        """Whether the draft head is the next launch after this step's final norm (an
        armed step with the head installed)."""
        return self.armed and self.head_part is not None

    # ---------------------------------------------------------------- per step

    def begin_step(self, step: int, num_rows: int) -> None:
        """A draft step of ``num_rows`` rows starts (the draft loop forwards steps
        0 .. num_steps - 2); with extend fc, step 0's fc is ``fc_buf``."""
        self.num_rows = num_rows
        self.armed = num_rows == 1
        self.step = step
        self.last_step = step == self.num_steps - 2
        if step == 0:
            # With extend_fc the draft caller guarantees fc_buf holds this step's fc.
            self.pending = self.extend_fc and num_rows == 1

    def hidden_partial(self, hidden_states: torch.Tensor) -> None:
        """This rank's hidden-slice fc partial for the next draft step, on a side stream;
        the exchange waits on its flag, not on a stream join."""
        if not self.is_hidden_rank or self.last_step or self.fuse_head_partial or self.fuse_head_exchange:
            return
        norm = self.norm_hidden
        self.side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side_stream):
            kernels.mkd_hidden_partial(
                hidden_states,
                self._fc_shard,
                norm.weight.data,
                norm.variance_epsilon,
                self.slice_start,
                self.own_partial,
                self.epoch_done,
                self.partial_sync,
                self.world,
                ctas=self._partial_ctas(),
            )
        self.partial_pending = True
        self._partial_side_stream_pending = True

    def _join_side_streams(self) -> None:
        # Joins only end graph capture cleanly: the exchange syncs on flags; a join before
        # the exchange would cost its early launch.
        if self._partial_side_stream_pending:
            torch.cuda.current_stream().wait_stream(self.side_stream)
            self._partial_side_stream_pending = False
        self.partial_pending = False

    def exchange(
        self,
        local_logits: torch.Tensor,
        topk_index: torch.Tensor,
        topk_p: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        positions: torch.Tensor,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor],
    ) -> None:
        """The step-boundary exchange: the token into ``topk_index`` / ``topk_p`` and
        ``draft_tokens[:, draft_column]``, positions (and M-RoPE positions) + 1, and unless
        this is the last step the next step's fc into ``fc_buf``."""
        partial_flag = (
            self.partial_sync[1:]
            if self.partial_pending and not self.last_step
            else None
        )
        kernels.mkd_xchg(
            local_logits,
            self.cand,
            self.hot_ids,
            self.own_partial,
            self.epoch_done,
            self.out_cand,
            self.out_partials,
            topk_index,
            topk_p,
            draft_tokens,
            positions,
            mrope_positions,
            self.mc,
            self.uc,
            self.rank,
            self.world,
            draft_column,
            kernels.NAN_KEY_SPLIT_ARGMAX,
            fc_out=self.fc_buf,
            token_only=self.last_step,
            partial_flag=partial_flag,
            scan_part=self.scan_part,
            head_part=self.head_part_cur if self.head_used else None,
            head_exchange=self._launch_head_exchange if self._head_exchange_input is not None else None,
        )
        self.head_used = False
        self.pending = not self.last_step
        if self.last_step:
            self._join_side_streams()

    def finish_step(
        self,
        local_logits: torch.Tensor,
        positions: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        step: int,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(topk_p, topk_index) of an armed step from its shard-local logits, through the
        exchange (DraftVocabExchange.finish_step on the one-launch boundary)."""
        m = self.num_rows
        topk_index, topk_p = self.topk_index[step][:m], self.topk_p[step][:m]
        if not self.armed:
            raise RuntimeError(
                "MKD draft boundary: only armed (one-row) steps take the exchange"
            )
        self.exchange(
            local_logits,
            topk_index,
            topk_p,
            draft_tokens,
            positions,
            draft_column,
            mrope_positions,
        )
        self.armed = False
        return topk_p, topk_index

    def end(self) -> None:
        """The draft loop's steps are over (DraftVocabExchange.end: the model's handoffs,
        which the caller owns here)."""

    def fc(self) -> torch.Tensor:
        """The draft fc output [1, H] of the step after an exchange."""
        self.pending = False
        if self.fc_buf is not None:
            return self.fc_buf
        norm = self.norm_embedding
        out = torch.empty_like(self.fc_out)
        kernels.mkd_fc_sum(
            self.out_cand,
            self.out_cand if self.memo else None,
            self.out_partials,
            self.fc_weight,
            norm.weight.data,
            norm.variance_epsilon,
            out,
            self.world,
        )
        return out

    # ---------------------------------------------------------------- the draft extend

    def is_local(self, logits: torch.Tensor) -> bool:
        """Whether ``logits`` are this rank's shard of at most MAX_ROWS rows."""
        return logits.shape[-1] == self.local_vocab and 0 < logits.shape[0] <= MAX_ROWS

    def select_extend(
        self, local_logits: torch.Tensor, hidden_states: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft-extend argmax (torch.argmax semantics) of one row's shard-local logits."""
        if local_logits.shape[0] != 1:
            raise RuntimeError("MKD draft boundary: the extend exchange serves one row")
        return self.exchange_extend(local_logits, hidden_states)

    def extend_head(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """SGLANG_PERFDRAFT_EXTEND_MKD_HEAD: the one-request draft extend's head as a draft
        step's: on a hidden rank the next draft's hidden partial on the side stream (the
        exchange waits on its flag), then the one-row MKD head on the SMs it leaves. The
        logits [1, N] for ``exchange_extend(..., mkd_head=True)``, or None (the caller keeps
        the shard GEMV and the scan) without the switch, the head or the extend fc."""
        if (
            not EXTEND_MKD_HEAD
            or self.head_part is None
            or not self.extend_fc
            or hidden_states.shape[0] != 1
        ):
            return None
        x = hidden_states.to(torch.bfloat16).contiguous()
        if self.fuse_head_exchange:
            logits = torch.empty(1, self.lm_head_weight.shape[0], dtype=torch.bfloat16, device=x.device)
            self.head_part_cur = self.head_part[:self._head_exchange_capture_ctas]
            self._head_exchange_input = (x, False, self.is_hidden_rank)
            self.head_used = True
            return logits
        partial_ctas = (
            max(0, min(self.partial_sms, self.head_part.numel() - 1))
            if self.is_hidden_rank and not self.fuse_head_partial
            else 0
        )
        if self.is_hidden_rank and not self.fuse_head_partial:
            norm = self.norm_hidden
            self.side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side_stream):
                kernels.mkd_hidden_partial(
                    x,
                    self._fc_shard,
                    norm.weight.data,
                    norm.variance_epsilon,
                    self.slice_start,
                    self.own_partial,
                    self.epoch_done,
                    self.partial_sync,
                    self.world,
                    ctas=partial_ctas,
                )
            self.partial_pending = True
            self._partial_side_stream_pending = True
        logits = torch.empty(
            1, self.lm_head_weight.shape[0], dtype=torch.bfloat16, device=x.device
        )
        self.head_part_cur = self.head_part[: self.head_part.numel() - partial_ctas]
        if self.fuse_head_partial and self.is_hidden_rank:
            self._head_with_partial(x, logits, reverse=False)
        elif self.head_mx8 is not None:
            self.head_mx8.mkd_head(
                x, logits, self.head_part_cur, reverse=False, smem_pad=self.head_pad
            )
        else:
            kernels.mkd_draft_head(
                x,
                self.lm_head_weight,
                logits,
                self.head_part_cur,
                reverse=False,
                smem_pad=self.head_pad,
            )
        self.head_used = True
        if torch.cuda.is_current_stream_capturing() and not self._extend_head_noted:
            self._extend_head_noted = True
            logger.info(
                "NVIDIA megakernel: MKD draft extend head as a draft step's (SGLANG_PERFDRAFT_EXTEND_MKD_HEAD): "
                "%s head on %d CTAs, %s",
                "MXFP8" if self.head_mx8 is not None else "BF16",
                self.head_part_cur.numel(),
                "hidden partial inside head"
                if self.is_hidden_rank and self.fuse_head_partial
                else f"hidden partial on {partial_ctas} SMs of the side stream"
                if self.is_hidden_rank
                else "no partial (embedding rank)",
            )
        return logits

    def exchange_extend(
        self,
        local_logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        topk_index: Optional[torch.Tensor] = None,
        topk_p: Optional[torch.Tensor] = None,
        mkd_head: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft-extend token (torch.argmax NaN rule) through the same one-launch exchange;
        with extend_fc it also writes the next draft's first fc into fc_buf. The token and
        its 1.0 land in ``topk_index`` / ``topk_p`` ([1, 1]) when given. ``mkd_head``: the
        logits come from ``extend_head`` (its per-CTA maxima and side-stream partial)."""
        if topk_index is None:
            topk_index = torch.empty(
                (1, 1), dtype=torch.int64, device=local_logits.device
            )
        if topk_p is None:
            topk_p = torch.empty(
                (1, 1), dtype=torch.float32, device=local_logits.device
            )
        payload = (
            self.extend_fc and hidden_states is not None and hidden_states.shape[0] == 1
        )
        if payload and self.is_hidden_rank and not mkd_head:
            norm = self.norm_hidden
            kernels.draft_fc_partial(
                hidden_states.contiguous(),
                norm.weight.data,
                norm.variance_epsilon,
                self._fc_shard,
                self.slice_start,
                self.own_partial,
                self._fc_gemv_rows,
            )
        kernels.mkd_xchg(
            local_logits,
            self.cand,
            self.hot_ids,
            self.own_partial,
            self.epoch_done,
            self.ext_cand,
            self.ext_partials,
            topk_index,
            topk_p,
            None,
            None,
            None,
            self.mc,
            self.uc,
            self.rank,
            self.world,
            0,
            kernels.NAN_KEY_TORCH_ARGMAX,
            fc_out=self.fc_buf if payload else None,
            token_only=not payload,
            partial_flag=self.partial_sync[1:]
            if mkd_head and payload and self.partial_pending
            else None,
            scan_part=self.scan_part,
            head_part=self.head_part_cur if mkd_head and self.head_used else None,
            head_exchange=self._launch_head_exchange if self._head_exchange_input is not None else None,
        )
        if mkd_head:
            self.head_used = False
            self._join_side_streams()
        self.fc_buf_key = self.input_seq if payload else None
        return topk_p, topk_index

    def note_draft_input(self) -> None:
        """A draft extend (after prefill or decode) is about to produce the next draft input."""
        self.input_seq += 1

    def take_fc_buf(self) -> bool:
        """True when fc_buf holds the first fc of the coming draft: its extend produced the
        current draft input. Consumes it (the draft's own exchanges overwrite fc_buf)."""
        holds = self.fc_buf_key is not None and self.fc_buf_key == self.input_seq
        self.fc_buf_key = None
        return holds

    def fill_fc(self, token_ids: torch.Tensor, hidden_states: torch.Tensor) -> None:
        """The fc of a draft's first step into fc_buf, when no extend exchange produced it."""
        self.pending = False
        embeds = self._embed(token_ids.flatten())
        self.fc_buf.copy_(self.split_fc(embeds, hidden_states))

    # ---------------------------------------------------------------- the row-split fc

    def split_fc(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The MTP fc [n, H] bf16 of n <= FC_SPLIT_MAX_TOKENS rows: the fp32 partial of
        this rank's input-column slice, all-gathered and summed in rank order, rounded once
        to bf16 (the replicated fc's rounding points)."""
        if not 0 < input_embeds.shape[0] <= FC_SPLIT_MAX_TOKENS:
            raise ValueError(
                f"split fc: {input_embeds.shape[0]} rows (1-{FC_SPLIT_MAX_TOKENS})"
            )
        use_hidden, start, width = self.fc_split_slice
        x = hidden_states if use_hidden else input_embeds
        norm = self.norm_hidden if use_hidden else self.norm_embedding
        if (
            self._fc_gemv_rows is not None
            and 0 < x.shape[0] <= kernels.MAX_TOKENS
            and x.shape[0] == input_embeds.shape[0]
        ):
            partial = kernels.draft_fc_partial(
                x.contiguous(),
                norm.weight.data,
                norm.variance_epsilon,
                self._fc_shard,
                start,
                x.new_empty((x.shape[0], self.hidden), dtype=torch.float32),
                self._fc_gemv_rows,
            )
            return self._gather_partials(partial)
        if (
            self._fc_ext_rows is not None
            and x.shape[0] == self.ext_rows
            and x.shape[0] == input_embeds.shape[0]
        ):
            self._note_fc(x.shape[0], "fused norm + GEMV partial")
            partial = kernels.draft_fc_partial_ext(
                x.contiguous(),
                norm.weight.data,
                norm.variance_epsilon,
                self._fc_shard,
                start,
                x.new_empty((x.shape[0], self.hidden), dtype=torch.float32),
                self._fc_ext_rows,
            )
            return self._gather_partials(partial)
        x = norm(x)
        if use_hidden and x.shape[0] != input_embeds.shape[0]:
            rows = min(x.shape[0], input_embeds.shape[0])
            slot = x.new_zeros((input_embeds.shape[0], x.shape[1]))
            slot[:rows] = x[:rows]
            x = slot
        partial = torch.mm(
            x[:, start : start + width], self._fc_shard.T, out_dtype=torch.float32
        )
        return self._gather_partials(partial)

    def _gather_partials(self, partial: torch.Tensor) -> torch.Tensor:
        out = partial.new_empty(partial.shape, dtype=torch.bfloat16)
        if self._fc_allsum is not None:
            self._note_fc(partial.shape[0], "one-launch all-sum")
            return self._fc_allsum(partial, out)
        gathered = self._fc_gather(partial)
        return kernels.ordered_partial_sum(gathered, self.world, out)

    def _note_fc(self, rows: int, path: str) -> None:
        """Log once per captured (rows, path) which opt-in split fc path a capture took."""
        if (
            torch.cuda.is_current_stream_capturing()
            and (rows, path) not in self._fc_noted
        ):
            self._fc_noted.add((rows, path))
            logger.info("MKD split fc of %d rows: %s", rows, path)
