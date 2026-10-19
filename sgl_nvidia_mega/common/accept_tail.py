# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSpark static greedy verify: the accept and its bookkeeping in one launch, one program
per request row.

Per row, from the target's argmax ids: the leading draft matches and the bonus, the
simulated correct length override, commit / new seq lens and a zero cap trim, the output
tokens, the gated commit lengths, the committed SWA write locations plus positions, and
optionally the Engram history commit of the armed live rows.
Every output equals SGLang's separate ops bit for bit.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _static_accept_tail_kernel(
    candidates_ptr,
    target_ptr,
    seq_lens_ptr,
    verify_lens_ptr,
    gate_ptr,
    sim_ptr,
    correct_out_ptr,
    bonus_out_ptr,
    cap_trim_out_ptr,
    commit_out_ptr,
    new_seq_out_ptr,
    out_tokens_ptr,
    gated_out_ptr,
    cache_loc_ptr,
    swa_map_ptr,
    positions_ptr,
    swa_out_ptr,
    positions_out_ptr,
    rpi_ptr,
    history_ptr,
    history_stride,
    bs,
    map_size,
    cand_stride,
    out_tokens_stride,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_SIM: tl.constexpr,
    HISTORY_WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KEY_RANKS: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    in_t = offs < T
    if b >= bs:
        # Staged rows past the batch: no SWA store.
        tl.store(swa_out_ptr + b * T + offs, tl.full([BLOCK_T], -1, tl.int32), in_t)
    else:
        row = b.to(tl.int64)
        gamma = T - 1
        in_draft = offs < gamma
        drafts = tl.load(
            candidates_ptr + row * cand_stride + 1 + offs, mask=in_draft, other=-1
        )
        if KEY_RANKS:
            # per-rank argmax keys (argmax_keys.py), [rows, ranks, 2] int64, sign
            # flipped: the max over ranks, its low 32 bits the inverted global index
            best = tl.full([BLOCK_T], -9223372036854775807 - 1, tl.int64)
            for r in tl.static_range(KEY_RANKS):
                key = tl.load(
                    target_ptr + ((row * T + offs) * KEY_RANKS + r) * 2,
                    mask=in_t,
                    other=-9223372036854775807 - 1,
                )
                best = tl.maximum(best, key)
            ids = 0xFFFFFFFF - (best & 0xFFFFFFFF)
        else:
            ids = tl.load(target_ptr + row * T + offs, mask=in_t, other=0).to(tl.int64)
        targets = tl.where(in_draft, ids, -2)
        eq = (drafts == targets) & in_draft
        num_correct = tl.min(tl.where(eq, BLOCK_T, offs), 0)
        bonus = tl.sum(tl.where(offs == num_correct, ids, 0), 0)
        if HAS_SIM:
            correct = tl.load(sim_ptr).to(tl.int32)
        else:
            correct = num_correct.to(tl.int32)
        commit = correct + 1
        new_seq = tl.load(seq_lens_ptr + row) + commit
        tl.store(correct_out_ptr + row, correct)
        tl.store(bonus_out_ptr + row, bonus)
        tl.store(cap_trim_out_ptr + row, 0)
        tl.store(commit_out_ptr + row, commit)
        tl.store(new_seq_out_ptr + row, new_seq)
        token = tl.where(
            offs == correct, bonus, tl.where(in_draft, drafts.to(tl.int64), 0)
        )
        tl.store(out_tokens_ptr + row * out_tokens_stride + offs, token, mask=in_t)
        verify_len = tl.load(verify_lens_ptr + row)
        gate = tl.load(gate_ptr)
        if HISTORY_WIDTH > 0:
            # SGLang's engram_commit_history for a live, armed row: the window of the
            # old history then this block's anchor + drafts, from commit on
            if (verify_len != 0) & (gate != 0):
                slot = tl.load(rpi_ptr + row).to(tl.int64)
                col = tl.arange(0, BLOCK_H)
                source_col = commit + col
                valid = col < HISTORY_WIDTH
                old = tl.load(
                    history_ptr + slot * history_stride + source_col,
                    mask=valid & (source_col < HISTORY_WIDTH),
                    other=0,
                )
                new = tl.load(
                    candidates_ptr + row * cand_stride + source_col - HISTORY_WIDTH,
                    mask=valid & (source_col >= HISTORY_WIDTH),
                    other=0,
                )
                values = tl.where(source_col < HISTORY_WIDTH, old, new)
                # every lane reads the old row before any lane overwrites it
                tl.debug_barrier()
                tl.store(
                    history_ptr + slot * history_stride + col,
                    values.to(history_ptr.dtype.element_ty),
                    mask=valid,
                )
        gated = tl.minimum(commit, verify_len.to(tl.int32)) * gate
        tl.store(gated_out_ptr + row, gated)
        committed = in_t & (offs < gated)
        loc = tl.load(cache_loc_ptr + row * T + offs, committed, other=0).to(tl.int64)
        # Preserve torch indexing for an unused negative location.
        loc = tl.where(loc < 0, loc + map_size, loc)
        swa = tl.load(swa_map_ptr + loc, committed, other=-1).to(tl.int32)
        tl.store(swa_out_ptr + row * T + offs, swa, in_t)
        positions = tl.load(positions_ptr + row * T + offs, in_t)
        tl.store(positions_out_ptr + row * T + offs, positions, in_t)


def static_accept_tail(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    gated_commit_lens_out: torch.Tensor,
    cache_loc: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    positions: torch.Tensor,
    swa_loc_out: torch.Tensor,
    positions_out: torch.Tensor,
    simulated_correct_len: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    history: Optional[torch.Tensor] = None,
    target_keys: Optional[torch.Tensor] = None,
) -> None:
    """``candidates`` [bs, T] (anchor + drafts), ``target_predict`` [bs * T] argmax ids,
    or ``target_keys`` [bs * T, ranks, 2] the gathered per-rank argmax keys (the ids
    are decoded in the kernel); outputs are written into rows [:bs] (``swa_loc_out``:
    every row, -1 past bs * T). With ``history`` (SGLang's Engram history [slots,
    width]) and ``req_pool_indices``, rows with verify_len != 0 and gate != 0 also
    commit their history."""
    assert (history is None) == (req_pool_indices is None)
    if target_keys is not None and not candidates.is_cuda:
        from sgl_nvidia_mega.common.argmax_keys import keys_to_ids

        target_predict = keys_to_ids(target_keys[..., 0].t())
    elif target_keys is not None:
        # the kernel addresses keys as a contiguous int64 [rows, ranks, 2]
        assert target_keys.dtype == torch.int64 and target_keys.is_contiguous()
        assert target_keys.dim() == 3 and target_keys.shape[2] == 2
        target_predict = target_keys
    fn = _static_accept_tail_triton if candidates.is_cuda else static_accept_tail_torch
    if candidates.is_cuda:
        fn = functools.partial(
            fn, key_ranks=0 if target_keys is None else target_keys.shape[1]
        )
    fn(
        candidates=candidates,
        target_predict=target_predict,
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        gate=gate,
        correct_len_out=correct_len_out,
        bonus_out=bonus_out,
        cap_trim_lens_out=cap_trim_lens_out,
        commit_lens_out=commit_lens_out,
        new_seq_lens_out=new_seq_lens_out,
        out_tokens_out=out_tokens_out,
        gated_commit_lens_out=gated_commit_lens_out,
        cache_loc=cache_loc,
        full_to_swa_mapping=full_to_swa_mapping,
        positions=positions,
        swa_loc_out=swa_loc_out,
        positions_out=positions_out,
        simulated_correct_len=simulated_correct_len,
        req_pool_indices=req_pool_indices,
        history=history,
    )


def _static_accept_tail_triton(*, key_ranks: int = 0, **kw) -> None:
    candidates = kw["candidates"]
    bs, T = candidates.shape
    assert candidates.stride(1) == 1 and kw["out_tokens_out"].stride(1) == 1
    sim = kw["simulated_correct_len"]
    mapping = kw["full_to_swa_mapping"]
    history = kw["history"]
    width = 0 if history is None else history.shape[1]
    if history is not None:
        assert history.stride(1) == 1
    _static_accept_tail_kernel[(kw["swa_loc_out"].numel() // T,)](
        candidates,
        kw["target_predict"],
        kw["seq_lens"],
        kw["verify_lens"],
        kw["gate"],
        sim if sim is not None else kw["seq_lens"],
        kw["correct_len_out"],
        kw["bonus_out"],
        kw["cap_trim_lens_out"],
        kw["commit_lens_out"],
        kw["new_seq_lens_out"],
        kw["out_tokens_out"],
        kw["gated_commit_lens_out"],
        kw["cache_loc"],
        mapping,
        kw["positions"],
        kw["swa_loc_out"],
        kw["positions_out"],
        kw["req_pool_indices"] if history is not None else kw["seq_lens"],
        history if history is not None else kw["seq_lens"],
        0 if history is None else history.stride(0),
        bs,
        mapping.numel(),
        candidates.stride(0),
        kw["out_tokens_out"].stride(0),
        T=T,
        BLOCK_T=triton.next_power_of_2(T),
        HAS_SIM=sim is not None,
        HISTORY_WIDTH=width,
        BLOCK_H=triton.next_power_of_2(max(width, 1)),
        KEY_RANKS=key_ranks,
        num_warps=4,
    )


def static_accept_tail_torch(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    gated_commit_lens_out: torch.Tensor,
    cache_loc: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    positions: torch.Tensor,
    swa_loc_out: torch.Tensor,
    positions_out: torch.Tensor,
    simulated_correct_len: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    history: Optional[torch.Tensor] = None,
) -> None:
    """Reference: the same outputs from torch ops (CPU or GPU)."""
    bs, T = candidates.shape
    gamma = T - 1
    target = target_predict.view(bs, T)
    matches = (candidates[:, 1:] == target[:, :gamma]).to(torch.int32)
    num_correct = matches.cumprod(dim=1).sum(dim=1).to(torch.int32)
    rows = torch.arange(bs, device=candidates.device)
    bonus = target[rows, num_correct.long()].to(torch.int64)
    if simulated_correct_len is not None:
        correct = simulated_correct_len.repeat(bs).to(torch.int32)
    else:
        correct = num_correct
    commit = correct + 1
    new_seq = seq_lens[:bs] + commit.to(seq_lens.dtype)
    correct_len_out[:bs].copy_(correct)
    bonus_out[:bs].copy_(bonus)
    cap_trim_lens_out[:bs].zero_()
    commit_lens_out[:bs].copy_(commit)
    new_seq_lens_out[:bs].copy_(new_seq)
    tokens = torch.zeros((bs, T), dtype=torch.int64, device=candidates.device)
    tokens[:, :gamma] = candidates[:, 1:]
    tokens.scatter_(1, correct.long()[:, None], bonus[:, None])
    out_tokens_out[:bs].copy_(tokens)
    if history is not None and history.shape[1] > 0:
        width = history.shape[1]
        live = (verify_lens[:bs] != 0) & (gate[0] != 0)
        for b in range(bs):
            if not bool(live[b]):
                continue
            slot = int(req_pool_indices[b])
            window = torch.cat([history[slot], candidates[b].to(history.dtype)])
            start = int(commit[b])
            history[slot] = window[start : start + width]
    gated = torch.minimum(commit, verify_lens[:bs].to(torch.int32)) * gate
    gated_commit_lens_out[:bs].copy_(gated)
    col = torch.arange(T, device=candidates.device)
    committed = col[None, :] < gated[:, None].long()
    loc = cache_loc[: bs * T].view(bs, T).long()
    loc = torch.where(loc < 0, loc + full_to_swa_mapping.numel(), loc)
    swa = torch.where(
        committed,
        full_to_swa_mapping[torch.where(committed, loc, 0)].to(torch.int32),
        -1,
    )
    swa_loc_out[: bs * T].copy_(swa.reshape(-1))
    swa_loc_out[bs * T :].fill_(-1)
    positions_out[: bs * T].copy_(positions[: bs * T])
