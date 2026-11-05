# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.qwen3_8._jit import is_arch_support_pdl
from sgl_nvidia_mega.qwen3_8.kernels.ops.extend_fastpath import (
    GDN_CONV_ENABLED as _Q38_EXTEND_PATH,
)

PAD_SLOT_ID = -1


CAUSAL_CONV1D_FWD_BLOCK_M = 8


if _Q38_EXTEND_PATH:

    @triton.jit()
    def _causal_conv1d_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        initial_states_ptr,
        cache_indices_ptr,
        has_initial_states_ptr,
        query_start_loc_ptr,
        o_ptr,
        block_table_ptr,
        dim: tl.constexpr,
        seqlen: tl.int32,
        num_cache_lines: tl.constexpr,
        stride_x_seq: tl.constexpr,
        stride_x_dim: tl.constexpr,
        stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr,
        stride_w_width: tl.constexpr,
        stride_istate_seq: tl.constexpr,
        stride_istate_dim: tl.constexpr,
        stride_istate_token: tl.constexpr,
        stride_o_seq: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,
        pad_slot_id: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        KERNEL_WIDTH: tl.constexpr,
        SILU_ACTIVATION: tl.constexpr,
        HAS_INITIAL_STATES: tl.constexpr,
        HAS_CACHE: tl.constexpr,
        IS_CONTINUOUS_BATCHING: tl.constexpr,
        USE_PAD_SLOT: tl.constexpr,
        NP2_STATELEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_BLOCK_TABLE: tl.constexpr = False,
        SPLIT_D0: tl.constexpr = 0,
        SPLIT_D1: tl.constexpr = 0,
        TILE_2D: tl.constexpr = False,
        FEAT_FASTEST: tl.constexpr = False,
        EVEN_FEATS: tl.constexpr = False,
    ):
        total_tokens = seqlen
        conv_states_ptr = initial_states_ptr
        conv_state_indices_ptr = cache_indices_ptr
        stride_conv_state_seq = stride_istate_seq
        stride_conv_state_dim = stride_istate_dim
        stride_conv_state_tok = stride_istate_token
        state_len = KERNEL_WIDTH - 1

        if FEAT_FASTEST:
            pid_feat = tl.program_id(0)
            pid_blk = tl.program_id(1)
            pid_seq = tl.program_id(2)
        else:
            pid_feat = tl.program_id(2)
            pid_blk = tl.program_id(0)
            pid_seq = tl.program_id(0)
        if USE_BLOCK_TABLE:
            idx_block = pid_blk
            idx_seq = tl.load(block_table_ptr + idx_block * 2)
            chunk_offset = tl.load(block_table_ptr + idx_block * 2 + 1)
        else:
            idx_seq = pid_seq
            if FEAT_FASTEST:
                chunk_offset = pid_blk
            else:
                chunk_offset = tl.program_id(1)

        idx_feats = pid_feat * BLOCK_N + tl.arange(0, BLOCK_N)

        if idx_seq == pad_slot_id:
            return

        sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
        sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)

        seqlen = sequence_end_index - sequence_start_index

        token_offset = BLOCK_M * chunk_offset
        segment_len = min(BLOCK_M, seqlen - token_offset)

        if segment_len <= 0:
            return

        x_base = (
            x_ptr
            + sequence_start_index.to(tl.int64) * stride_x_token
            + idx_feats * stride_x_dim
        )

        if IS_CONTINUOUS_BATCHING:
            conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(
                tl.int64
            )
        else:
            conv_state_batch_coord = idx_seq
        if USE_PAD_SLOT:
            if conv_state_batch_coord == pad_slot_id:
                return
        conv_states_base = (
            conv_states_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )

        w_base = w_ptr + (idx_feats * stride_w_dim)

        if chunk_offset == 0:
            load_init_state = False
            if HAS_INITIAL_STATES:
                load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
            if load_init_state:
                x_elem_ty = x_ptr.dtype.element_ty
                prior_tokens = (
                    conv_states_base + (state_len - 1) * stride_conv_state_tok
                )
                mask_w = idx_feats < dim
                if KERNEL_WIDTH == 2:
                    conv_states_ptrs = prior_tokens
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                if KERNEL_WIDTH == 3:
                    conv_states_ptrs = prior_tokens
                    col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                if KERNEL_WIDTH == 4:
                    conv_states_ptrs = prior_tokens
                    col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                    col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                if KERNEL_WIDTH == 5:
                    conv_states_ptrs = prior_tokens
                    col3 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                    col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                    col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            else:
                if KERNEL_WIDTH >= 2:
                    col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 3:
                    col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 4:
                    col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 5:
                    col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

            if state_len <= seqlen:
                idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
                x_ptrs = (
                    x_ptr
                    + (
                        (sequence_start_index + idx_tokens_last).to(tl.int64)
                        * stride_x_token
                    )[:, None]
                    + (idx_feats * stride_x_dim)[None, :]
                )
                mask_x = (
                    (idx_tokens_last >= 0)[:, None]
                    & (idx_tokens_last < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_target = (
                    conv_states_base[None, :]
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )

                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.debug_barrier()
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

            else:
                if load_init_state:
                    idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                    conv_states_ptrs_source = (
                        conv_states_ptr
                        + (conv_state_batch_coord * stride_conv_state_seq)
                        + (idx_feats * stride_conv_state_dim)[None, :]
                        + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                    )
                    mask = (
                        (conv_state_batch_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                    VAL = state_len - seqlen

                    x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                    )

                    mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                    tl.debug_barrier()
                    new_conv_state = tl.where(mask, conv_state, loaded_x)
                    conv_states_ptrs_target = (
                        conv_states_base
                        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                    )
                    mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                        None, :
                    ]
                    tl.store(conv_states_ptrs_target, new_conv_state, mask)
                else:
                    idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                    VAL = state_len - seqlen

                    x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                    )

                    mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                    conv_states_ptrs_target = (
                        conv_states_base
                        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                    )
                    mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                        None, :
                    ]
                    tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            load_init_state = True
            prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 1 * stride_x_token
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 1 * stride_x_token
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 2 * stride_x_token
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 1 * stride_x_token
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 2 * stride_x_token
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 3 * stride_x_token
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            mask_bias = idx_feats < dim
            acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
        else:
            acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

        x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token

        mask_w = idx_feats < dim
        if KERNEL_WIDTH >= 2:
            w_ptrs = w_base + (0 * stride_w_width)
            w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
            w_ptrs = w_base + (1 * stride_w_width)
            w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
        if KERNEL_WIDTH >= 3:
            w_ptrs = w_base + (2 * stride_w_width)
            w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
        if KERNEL_WIDTH >= 4:
            w_ptrs = w_base + (3 * stride_w_width)
            w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
        mask_x_1d = idx_feats < dim
        if SPLIT_D0 > 0:
            feat0 = pid_feat * BLOCK_N
            in_k = feat0 >= SPLIT_D0
            in_v = feat0 >= SPLIT_D1
            col_base = tl.where(in_v, SPLIT_D1, tl.where(in_k, SPLIT_D0, 0))
            o_width = tl.where(
                in_v, dim - SPLIT_D1, tl.where(in_k, SPLIT_D1 - SPLIT_D0, SPLIT_D0)
            ).to(tl.int64)
            o_base_split = (
                o_ptr + col_base.to(tl.int64) * total_tokens + (idx_feats - col_base)
            )
        if TILE_2D:
            t_loc = tl.arange(0, BLOCK_M)
            tok = token_offset + t_loc
            x_rows = (
                x_ptr
                + sequence_start_index.to(tl.int64) * stride_x_token
                + (idx_feats * stride_x_dim)[None, :]
            )
            o_ptrs2 = (
                o_base_split[None, :]
                + ((sequence_start_index + tok).to(tl.int64) * o_width)[:, None]
            )
            acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + acc_preload[None, :]
            interior = (chunk_offset > 0) & (segment_len == BLOCK_M) & EVEN_FEATS
            if interior:
                for j in tl.static_range(KERNEL_WIDTH):
                    xv = tl.load(
                        x_rows
                        + (
                            (tok - (KERNEL_WIDTH - 1 - j)).to(tl.int64) * stride_x_token
                        )[:, None]
                    )
                    if j == 0:
                        w2 = w_col0
                    elif j == 1:
                        w2 = w_col1
                    elif j == 2:
                        w2 = w_col2
                    else:
                        w2 = w_col3
                    acc2 += xv * w2[None, :]
                if SILU_ACTIVATION:
                    acc2 = acc2 / (1 + tl.exp(-acc2))
                tl.store(o_ptrs2, acc2)
            else:
                row_ok = t_loc < segment_len
                for j in tl.static_range(KERNEL_WIDTH):
                    src = tok - (KERNEL_WIDTH - 1 - j)
                    m2 = (row_ok & (src >= 0))[:, None] & mask_x_1d[None, :]
                    xv = tl.load(
                        x_rows + (src.to(tl.int64) * stride_x_token)[:, None],
                        mask=m2,
                        other=0.0,
                    )

                    if j <= 2:
                        xv = tl.where((src == -1)[:, None], col2[None, :], xv)
                    if j <= 1:
                        xv = tl.where((src == -2)[:, None], col1[None, :], xv)
                    if j == 0:
                        xv = tl.where((src == -3)[:, None], col0[None, :], xv)
                    if j == 0:
                        w2 = w_col0
                    elif j == 1:
                        w2 = w_col1
                    elif j == 2:
                        w2 = w_col2
                    else:
                        w2 = w_col3
                    acc2 += xv * w2[None, :]
                if SILU_ACTIVATION:
                    acc2 = acc2 / (1 + tl.exp(-acc2))
                tl.store(o_ptrs2, acc2, mask=row_ok[:, None] & mask_x_1d[None, :])
        else:
            for idx_token in range(segment_len):
                acc = acc_preload

                matrix_w = w_col0
                matrix_x = col0
                for j in tl.static_range(KERNEL_WIDTH):
                    if KERNEL_WIDTH == 2:
                        if j == 1:
                            matrix_w = w_col1
                            x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                    elif KERNEL_WIDTH == 3:
                        if j == 1:
                            matrix_w = w_col1
                            matrix_x = col1
                        elif j == 2:
                            matrix_w = w_col2
                            x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                    elif KERNEL_WIDTH == 4:
                        if j == 1:
                            matrix_w = w_col1
                            matrix_x = col1
                        elif j == 2:
                            matrix_w = w_col2
                            matrix_x = col2
                        elif j == 3:
                            matrix_w = w_col3
                            x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                    acc += matrix_x * matrix_w

                if KERNEL_WIDTH == 2:
                    col0 = matrix_x
                elif KERNEL_WIDTH == 3:
                    col0 = col1
                    col1 = matrix_x
                elif KERNEL_WIDTH == 4:
                    col0 = col1
                    col1 = col2
                    col2 = matrix_x

                if SILU_ACTIVATION:
                    acc = acc / (1 + tl.exp(-acc))
                mask_1d = (idx_token < segment_len) & (idx_feats < dim)
                if SPLIT_D0 > 0:
                    o_ptrs = (
                        o_base_split
                        + (sequence_start_index + token_offset + idx_token).to(tl.int64)
                        * o_width
                    )
                else:
                    o_ptrs = (
                        o_ptr
                        + (sequence_start_index + token_offset + idx_token).to(tl.int64)
                        * stride_o_token
                        + (idx_feats * stride_o_dim)
                    )

                tl.store(o_ptrs, acc, mask=mask_1d)
else:

    @triton.jit()
    def _causal_conv1d_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        initial_states_ptr,
        cache_indices_ptr,
        has_initial_states_ptr,
        query_start_loc_ptr,
        o_ptr,
        block_table_ptr,
        dim: tl.constexpr,
        seqlen: tl.int32,
        num_cache_lines: tl.constexpr,
        stride_x_seq: tl.constexpr,
        stride_x_dim: tl.constexpr,
        stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr,
        stride_w_width: tl.constexpr,
        stride_istate_seq: tl.constexpr,
        stride_istate_dim: tl.constexpr,
        stride_istate_token: tl.constexpr,
        stride_o_seq: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,
        pad_slot_id: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        KERNEL_WIDTH: tl.constexpr,
        SILU_ACTIVATION: tl.constexpr,
        HAS_INITIAL_STATES: tl.constexpr,
        HAS_CACHE: tl.constexpr,
        IS_CONTINUOUS_BATCHING: tl.constexpr,
        USE_PAD_SLOT: tl.constexpr,
        NP2_STATELEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_BLOCK_TABLE: tl.constexpr = False,
    ):
        conv_states_ptr = initial_states_ptr
        conv_state_indices_ptr = cache_indices_ptr
        stride_conv_state_seq = stride_istate_seq
        stride_conv_state_dim = stride_istate_dim
        stride_conv_state_tok = stride_istate_token
        state_len = KERNEL_WIDTH - 1

        if USE_BLOCK_TABLE:
            idx_block = tl.program_id(0)
            idx_seq = tl.load(block_table_ptr + idx_block * 2)
            chunk_offset = tl.load(block_table_ptr + idx_block * 2 + 1)
        else:
            idx_seq = tl.program_id(0)
            chunk_offset = tl.program_id(1)

        idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

        if idx_seq == pad_slot_id:
            return

        sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
        sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)

        seqlen = sequence_end_index - sequence_start_index

        token_offset = BLOCK_M * chunk_offset
        segment_len = min(BLOCK_M, seqlen - token_offset)

        if segment_len <= 0:
            return

        x_base = (
            x_ptr
            + sequence_start_index.to(tl.int64) * stride_x_token
            + idx_feats * stride_x_dim
        )

        if IS_CONTINUOUS_BATCHING:
            conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(
                tl.int64
            )
        else:
            conv_state_batch_coord = idx_seq
        if USE_PAD_SLOT:
            if conv_state_batch_coord == pad_slot_id:
                return
        conv_states_base = (
            conv_states_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )

        w_base = w_ptr + (idx_feats * stride_w_dim)

        if chunk_offset == 0:
            load_init_state = False
            if HAS_INITIAL_STATES:
                load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
            if load_init_state:
                x_elem_ty = x_ptr.dtype.element_ty
                prior_tokens = (
                    conv_states_base + (state_len - 1) * stride_conv_state_tok
                )
                mask_w = idx_feats < dim
                if KERNEL_WIDTH == 2:
                    conv_states_ptrs = prior_tokens
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                if KERNEL_WIDTH == 3:
                    conv_states_ptrs = prior_tokens
                    col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                if KERNEL_WIDTH == 4:
                    conv_states_ptrs = prior_tokens
                    col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                    col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                if KERNEL_WIDTH == 5:
                    conv_states_ptrs = prior_tokens
                    col3 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                    col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                    col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                    conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok
                    col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            else:
                if KERNEL_WIDTH >= 2:
                    col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 3:
                    col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 4:
                    col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 5:
                    col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

            if state_len <= seqlen:
                idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
                x_ptrs = (
                    x_ptr
                    + (
                        (sequence_start_index + idx_tokens_last).to(tl.int64)
                        * stride_x_token
                    )[:, None]
                    + (idx_feats * stride_x_dim)[None, :]
                )
                mask_x = (
                    (idx_tokens_last >= 0)[:, None]
                    & (idx_tokens_last < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_target = (
                    conv_states_base[None, :]
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )

                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.debug_barrier()
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

            else:
                if load_init_state:
                    idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                    conv_states_ptrs_source = (
                        conv_states_ptr
                        + (conv_state_batch_coord * stride_conv_state_seq)
                        + (idx_feats * stride_conv_state_dim)[None, :]
                        + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                    )
                    mask = (
                        (conv_state_batch_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                    VAL = state_len - seqlen

                    x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                    )

                    mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                    tl.debug_barrier()
                    new_conv_state = tl.where(mask, conv_state, loaded_x)
                    conv_states_ptrs_target = (
                        conv_states_base
                        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                    )
                    mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                        None, :
                    ]
                    tl.store(conv_states_ptrs_target, new_conv_state, mask)
                else:
                    idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                    VAL = state_len - seqlen

                    x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                    )

                    mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                    conv_states_ptrs_target = (
                        conv_states_base
                        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                    )
                    mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                        None, :
                    ]
                    tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            load_init_state = True
            prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 1 * stride_x_token
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 1 * stride_x_token
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 2 * stride_x_token
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 1 * stride_x_token
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 2 * stride_x_token
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                conv_states_ptrs = prior_tokens - 3 * stride_x_token
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            mask_bias = idx_feats < dim
            acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
        else:
            acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

        x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token

        mask_w = idx_feats < dim
        if KERNEL_WIDTH >= 2:
            w_ptrs = w_base + (0 * stride_w_width)
            w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
            w_ptrs = w_base + (1 * stride_w_width)
            w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
        if KERNEL_WIDTH >= 3:
            w_ptrs = w_base + (2 * stride_w_width)
            w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
        if KERNEL_WIDTH >= 4:
            w_ptrs = w_base + (3 * stride_w_width)
            w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
        mask_x_1d = idx_feats < dim
        for idx_token in range(segment_len):
            acc = acc_preload

            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                acc += matrix_x * matrix_w

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            mask_1d = (idx_token < segment_len) & (idx_feats < dim)
            o_ptrs = (
                o_ptr
                + (sequence_start_index + token_offset + idx_token).to(tl.int64)
                * stride_o_token
                + (idx_feats * stride_o_dim)
            )

            tl.store(o_ptrs, acc, mask=mask_1d)


if _Q38_EXTEND_PATH:

    def causal_conv1d_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, None],
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: List[int],
        cache_indices: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = "silu",
        pad_slot_id: int = PAD_SLOT_ID,
        validate_data=False,
        block_table: Optional[torch.Tensor] = None,
        split_dims: Optional[tuple] = None,
        launch_cfg: Optional[tuple] = None,
        **kwargs,
    ):
        """support varlen + continuous batching when x is 2D tensor

        x: (dim,cu_seq_len)
            cu_seq_len = total tokens of all seqs in that batch
            sequences are concatenated from left to right for varlen
        weight: (dim, width)
        conv_states: (...,dim,width - 1) itype
            updated inplace if provided
            [it use `cache_indices` to get the index to the cache of conv_state for that sequence

            conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
                 and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
            ]
        query_start_loc: (batch + 1) int32
            The cumulative sequence lengths of the sequences in
            the batch, used to index into sequence. prepended by 0.
            if
            x = [5, 1, 1, 1] <- continuous batching (batch=4)
            then
            query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
               the ending index of the last sequence
            [length(query_start_loc)-1 == batch]
            for example: query_start_loc = torch.Tensor([0,10,16,17]),
            x.shape=(dim,17)
        seq_lens_cpu: (batch) int32
            The sequence lengths of the sequences in the batch
        cache_indices: (batch)  int32
            indicates the corresponding state index,
            like so: conv_state = conv_states[cache_indices[batch_id]]
        has_initial_state: (batch) bool
            indicates whether should the kernel take the current state as initial
            state for the calculations
            [single boolean for each sequence in the batch: True or False]
        bias: (dim,)
        activation: either None or "silu" or "swish" or True
        pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3

        out: same shape as `x`
        """
        if isinstance(activation, bool) and activation:
            activation = "silu"

        cfg = tuple(launch_cfg) if launch_cfg is not None else (256, 4, 2)
        block_n, n_warps, n_stages = cfg[:3]
        block_m = (
            cfg[3]
            if len(cfg) > 3 and block_table is None
            else CAUSAL_CONV1D_FWD_BLOCK_M
        )
        tile_2d = bool(cfg[4]) if len(cfg) > 4 else False
        feat_fastest = bool(cfg[5]) if len(cfg) > 5 else False
        if feat_fastest and block_table is not None and block_table.shape[0] > 65535:
            feat_fastest = False
        if tile_2d:
            assert split_dims is not None and weight.shape[1] == 4
        if split_dims is not None:
            assert split_dims[0] % block_n == 0 and split_dims[1] % block_n == 0
            out = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
        else:
            out = torch.empty_like(x)

        is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
        dim, cu_seqlen = x.shape
        _, width = weight.shape
        state_len = width - 1
        np2_statelen = triton.next_power_of_2(state_len)

        stride_x_seq = 0
        stride_x_dim = x.stride(0)
        stride_x_token = x.stride(1)
        stride_w_dim = weight.stride(0)
        stride_w_width = weight.stride(1)
        stride_istate_seq = 0
        stride_istate_dim = 0
        stride_istate_token = 0
        num_cache_lines = 0
        if conv_states is not None:
            num_cache_lines = conv_states.size(0)
            assert (
                num_cache_lines == conv_states.shape[0]
                and dim == conv_states.shape[1]
                and width - 1 <= conv_states.shape[2]
            )
            stride_istate_seq = conv_states.stride(0)
            stride_istate_dim = conv_states.stride(1)
            stride_istate_token = conv_states.stride(2)

        if split_dims is not None:
            stride_o_seq = stride_o_dim = stride_o_token = 0
        elif out.dim() == 2:
            stride_o_seq = 0
            stride_o_dim = out.stride(0)
            stride_o_token = out.stride(1)
        else:
            stride_o_seq = out.stride(0)
            stride_o_dim = out.stride(1)
            stride_o_token = out.stride(2)

        if validate_data:
            assert x.dim() == 2
            assert query_start_loc is not None
            assert query_start_loc.dim() == 1
            assert x.stride(0) == 1 or x.stride(1) == 1
            padded_batch = query_start_loc.size(0) - 1
            if bias is not None:
                assert bias.dim() == 1
                assert dim == bias.size(0)
            if cache_indices is not None:
                assert cache_indices.dim() == 1
                assert padded_batch == cache_indices.size(0)
            if has_initial_state is not None:
                assert has_initial_state.size() == (padded_batch,)
                assert conv_states is not None, (
                    "ERROR: `has_initial_state` is used, which needs also `conv_states`"
                )
            assert weight.stride(1) == 1
            assert (dim, width) == weight.shape
            assert is_channel_last, "Need to run in channel-last layout"

        def grid(META):
            if feat_fastest:
                n_feat = triton.cdiv(dim, META["BLOCK_N"])
                if block_table is not None:
                    return (n_feat, block_table.shape[0], 1)
                max_seq_len = max(seq_lens_cpu)
                return (
                    n_feat,
                    (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
                    len(seq_lens_cpu),
                )
            if block_table is not None:
                return (block_table.shape[0], 1, triton.cdiv(dim, META["BLOCK_N"]))
            max_seq_len = max(seq_lens_cpu)
            return (
                len(seq_lens_cpu),
                (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
                triton.cdiv(dim, META["BLOCK_N"]),
            )

        _causal_conv1d_fwd_kernel[grid](
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            out,
            block_table,
            dim,
            cu_seqlen,
            num_cache_lines,
            stride_x_seq,
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_o_seq,
            stride_o_dim,
            stride_o_token,
            pad_slot_id,
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            HAS_INITIAL_STATES=has_initial_state is not None,
            HAS_CACHE=conv_states is not None,
            IS_CONTINUOUS_BATCHING=cache_indices is not None,
            USE_PAD_SLOT=pad_slot_id is not None,
            NP2_STATELEN=np2_statelen,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            USE_BLOCK_TABLE=block_table is not None,
            SPLIT_D0=split_dims[0] if split_dims is not None else 0,
            SPLIT_D1=split_dims[1] if split_dims is not None else 0,
            TILE_2D=tile_2d,
            FEAT_FASTEST=feat_fastest,
            EVEN_FEATS=dim % block_n == 0,
            num_warps=n_warps,
            num_stages=n_stages,
        )
        return out
else:

    def causal_conv1d_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, None],
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: List[int],
        cache_indices: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = "silu",
        pad_slot_id: int = PAD_SLOT_ID,
        validate_data=False,
        block_table: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """support varlen + continuous batching when x is 2D tensor

        x: (dim,cu_seq_len)
            cu_seq_len = total tokens of all seqs in that batch
            sequences are concatenated from left to right for varlen
        weight: (dim, width)
        conv_states: (...,dim,width - 1) itype
            updated inplace if provided
            [it use `cache_indices` to get the index to the cache of conv_state for that sequence

            conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
                 and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
            ]
        query_start_loc: (batch + 1) int32
            The cumulative sequence lengths of the sequences in
            the batch, used to index into sequence. prepended by 0.
            if
            x = [5, 1, 1, 1] <- continuous batching (batch=4)
            then
            query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
               the ending index of the last sequence
            [length(query_start_loc)-1 == batch]
            for example: query_start_loc = torch.Tensor([0,10,16,17]),
            x.shape=(dim,17)
        seq_lens_cpu: (batch) int32
            The sequence lengths of the sequences in the batch
        cache_indices: (batch)  int32
            indicates the corresponding state index,
            like so: conv_state = conv_states[cache_indices[batch_id]]
        has_initial_state: (batch) bool
            indicates whether should the kernel take the current state as initial
            state for the calculations
            [single boolean for each sequence in the batch: True or False]
        bias: (dim,)
        activation: either None or "silu" or "swish" or True
        pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3

        out: same shape as `x`
        """
        if isinstance(activation, bool) and activation:
            activation = "silu"

        out = torch.empty_like(x)

        is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
        dim, cu_seqlen = x.shape
        _, width = weight.shape
        state_len = width - 1
        np2_statelen = triton.next_power_of_2(state_len)

        stride_x_seq = 0
        stride_x_dim = x.stride(0)
        stride_x_token = x.stride(1)
        stride_w_dim = weight.stride(0)
        stride_w_width = weight.stride(1)
        stride_istate_seq = 0
        stride_istate_dim = 0
        stride_istate_token = 0
        num_cache_lines = 0
        if conv_states is not None:
            num_cache_lines = conv_states.size(0)
            assert (
                num_cache_lines == conv_states.shape[0]
                and dim == conv_states.shape[1]
                and width - 1 <= conv_states.shape[2]
            )
            stride_istate_seq = conv_states.stride(0)
            stride_istate_dim = conv_states.stride(1)
            stride_istate_token = conv_states.stride(2)

        if out.dim() == 2:
            stride_o_seq = 0
            stride_o_dim = out.stride(0)
            stride_o_token = out.stride(1)
        else:
            stride_o_seq = out.stride(0)
            stride_o_dim = out.stride(1)
            stride_o_token = out.stride(2)

        if validate_data:
            assert x.dim() == 2
            assert query_start_loc is not None
            assert query_start_loc.dim() == 1
            assert x.stride(0) == 1 or x.stride(1) == 1
            padded_batch = query_start_loc.size(0) - 1
            if bias is not None:
                assert bias.dim() == 1
                assert dim == bias.size(0)
            if cache_indices is not None:
                assert cache_indices.dim() == 1
                assert padded_batch == cache_indices.size(0)
            if has_initial_state is not None:
                assert has_initial_state.size() == (padded_batch,)
                assert conv_states is not None, (
                    "ERROR: `has_initial_state` is used, which needs also `conv_states`"
                )
            assert weight.stride(1) == 1
            assert (dim, width) == weight.shape
            assert is_channel_last, "Need to run in channel-last layout"

        def grid(META):
            if block_table is not None:
                return (block_table.shape[0], 1, triton.cdiv(dim, META["BLOCK_N"]))
            max_seq_len = max(seq_lens_cpu)
            return (
                len(seq_lens_cpu),
                (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
                triton.cdiv(dim, META["BLOCK_N"]),
            )

        _causal_conv1d_fwd_kernel[grid](
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            out,
            block_table,
            dim,
            cu_seqlen,
            num_cache_lines,
            stride_x_seq,
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_o_seq,
            stride_o_dim,
            stride_o_token,
            pad_slot_id,
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            HAS_INITIAL_STATES=has_initial_state is not None,
            HAS_CACHE=conv_states is not None,
            IS_CONTINUOUS_BATCHING=cache_indices is not None,
            USE_PAD_SLOT=pad_slot_id is not None,
            NP2_STATELEN=np2_statelen,
            BLOCK_M=CAUSAL_CONV1D_FWD_BLOCK_M,
            BLOCK_N=256,
            USE_BLOCK_TABLE=block_table is not None,
            num_stages=2,
        )
        return out


GDN_SPLIT_CONV_LAUNCH = (256, 4, 2, 8, True)


def causal_conv1d_fn_split_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: Optional[List[int]],
    q_dim: int,
    k_dim: int,
    launch_cfg: Optional[tuple] = None,
    **kwargs,
):
    """causal_conv1d_fn whose output is three contiguous token-major tensors
    q [T, q_dim], k [T, k_dim], v [T, dim - q_dim - k_dim] (bit-identical
    values to causal_conv1d_fn(...).transpose(0, 1) split along the channel
    axis). q and k are adjacent in one allocation, so a [2, T, q_dim] view
    covers both when q_dim == k_dim."""
    dim, total = x.shape
    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        split_dims=(q_dim, q_dim + k_dim),
        launch_cfg=launch_cfg if launch_cfg is not None else GDN_SPLIT_CONV_LAUNCH,
        **kwargs,
    )
    qk_end = total * (q_dim + k_dim)
    q = out[: total * q_dim].view(total, q_dim)
    k = out[total * q_dim : qk_end].view(total, k_dim)
    v = out[qk_end:].view(total, dim - q_dim - k_dim)
    return q, k, v


@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,
    conv_state_indices_ptr,
    num_accept_tokens_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    o_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices: tl.constexpr,
    stride_retrieve_next_token_seq: tl.constexpr,
    stride_retrieve_next_token_token: tl.constexpr,
    stride_retrieve_next_sibling_seq: tl.constexpr,
    stride_retrieve_next_sibling_token: tl.constexpr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):

    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
        if SAVE_INTERMEDIATE:
            intermediate_state_batch_coord = tl.load(
                intermediate_state_indices_ptr
                + idx_seq * stride_intermediate_state_indices
            ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return

    if IS_SPEC_DECODING:
        conv_state_token_offset = tl.load(num_accept_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    idx_tokens = tl.arange(0, NP2_STATELEN)

    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)

    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        idx_tokens = tl.arange(0, NP2_SEQLEN)

        mask_retrieve = idx_tokens < seqlen
        retrieve_next_token_base = (
            retrieve_next_token_ptr
            + (idx_seq * stride_retrieve_next_token_seq)
            + idx_tokens * stride_retrieve_next_token_token
        )
        retrieve_next_tokens = tl.load(retrieve_next_token_base, mask_retrieve)
        retrieve_next_sibling_base = (
            retrieve_next_sibling_ptr
            + (idx_seq * stride_retrieve_next_sibling_seq)
            + idx_tokens * stride_retrieve_next_sibling_token
        )
        retrieve_next_siblings = tl.load(retrieve_next_sibling_base, mask_retrieve)
        parent_idx_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            retrieve_next_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0)
            )
            if retrieve_next_token_idx != -1:
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_next_token_idx,
                    idx_token,
                    parent_idx_tokens,
                )

            retrieve_sibling_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0)
            )
            if retrieve_sibling_token_idx != -1:
                parent_idx_token = tl.sum(
                    tl.where(idx_tokens == idx_token, parent_idx_tokens, 0)
                )
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_sibling_token_idx,
                    parent_idx_token,
                    parent_idx_tokens,
                )

            _idx_token = idx_token
            x_ptrs_1d = x_base_1d + _idx_token * stride_x_token
            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 0:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0
                elif KERNEL_WIDTH == 3:
                    if j == 0:
                        matrix_w = w_col2
                    elif j == 1:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0
                elif KERNEL_WIDTH == 4:
                    if j == 0:
                        matrix_w = w_col3
                    elif j == 1:
                        matrix_w = w_col2
                    elif j == 2:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0

                if SAVE_INTERMEDIATE:
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + intermediate_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )

                    if KERNEL_WIDTH - j - 2 >= 0:
                        tl.store(
                            base_ptr + (KERNEL_WIDTH - j - 2) * stride_inter_win,
                            matrix_x,
                            mask=mask_w,
                        )

                acc += matrix_x * matrix_w

                if _idx_token > 0:
                    _idx_token = tl.sum(
                        tl.where(idx_tokens == _idx_token, parent_idx_tokens, 0)
                    )
                    x_ptrs_1d = x_base_1d + _idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                else:
                    if KERNEL_WIDTH == 2:
                        if _idx_token == 0:
                            matrix_x = col0
                    elif KERNEL_WIDTH == 3:
                        if _idx_token == 0:
                            matrix_x = col1
                        else:
                            matrix_x = col0
                    elif KERNEL_WIDTH == 4:
                        if _idx_token == 0:
                            matrix_x = col2
                        elif _idx_token == -1:
                            matrix_x = col1
                        else:
                            matrix_x = col0
                    _idx_token = _idx_token - 1
        else:
            matrix_w = w_col0
            matrix_x = col0

            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                acc += matrix_x * matrix_w

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SAVE_INTERMEDIATE:
                base_ptr = (
                    intermediate_conv_window_ptr
                    + intermediate_state_batch_coord * stride_inter_seq
                    + idx_token * stride_inter_step
                    + idx_feats * stride_inter_dim
                )
                if KERNEL_WIDTH >= 2:
                    tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
                if KERNEL_WIDTH >= 3:
                    tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
                if KERNEL_WIDTH >= 4:
                    tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            tl.store(
                retrieve_parent_token_ptr
                + idx_seq * stride_retrieve_parent_token_seq
                + idx_tokens * stride_retrieve_parent_token_token,
                parent_idx_tokens,
                mask=mask_retrieve,
            )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape

    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1

        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape
            assert intermediate_state_indices is not None
            assert (batch,) == intermediate_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1
        assert cache_seqlens is None

    out = torch.empty_like(x)
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    stride_intermediate_state_indices = (
        intermediate_state_indices.stride(0)
        if intermediate_state_indices is not None
        else 0
    )
    if num_accept_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    np2_seqlen = triton.next_power_of_2(seqlen)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    if retrieve_next_token is not None:
        stride_retrieve_next_token_seq, stride_retrieve_next_token_token = (
            retrieve_next_token.stride(0),
            retrieve_next_token.stride(1),
        )
    else:
        stride_retrieve_next_token_seq = stride_retrieve_next_token_token = 0

    if retrieve_next_sibling is not None:
        stride_retrieve_next_sibling_seq, stride_retrieve_next_sibling_token = (
            retrieve_next_sibling.stride(0),
            retrieve_next_sibling.stride(1),
        )
    else:
        stride_retrieve_next_sibling_seq = stride_retrieve_next_sibling_token = 0

    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
            retrieve_parent_token.stride(0),
            retrieve_parent_token.stride(1),
        )
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accept_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_intermediate_state_indices,
        stride_retrieve_next_token_seq,
        stride_retrieve_next_token_token,
        stride_retrieve_next_sibling_seq,
        stride_retrieve_next_sibling_token,
        stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accept_tokens is not None,
        NP2_STATELEN=np2_statelen,
        NP2_SEQLEN=np2_seqlen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_next_token is not None,
        **pdl_kwargs,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out
