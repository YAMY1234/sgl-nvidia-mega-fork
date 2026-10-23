# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5_mtp.py (Qwen3_5ForCausalLMMTP.forward),
# python/sglang/srt/models/qwen3_5.py (Qwen3_5ForCausalLM.forward and
# Qwen3_5AttentionDecoderLayer.forward) and
# python/sglang/srt/layers/logits_processor.py (LogitsProcessor._get_logits).
"""The MTP draft's model step inside a draft entry, over SGLang's loaded draft modules:

- ``fc_input``: the embedding, the two pre-fc Gemma RMSNorms and ``fc``;
- ``layer``: the decoder layer (input norm, attention, the attention output's
  all-reduce + residual add + post-attention norm, the MoE, the MoE output's
  all-reduce + residual add + final norm), returning the final-norm rows;
- ``top1``: the LM head on this rank's vocabulary shard, the all-gather of the logits
  and their arg-max (``torch.argmax``'s choice, as SGLang's top-1).

Stock: the ops SGLang's draft runs for these rows, in its order. Slot ``draft_moe``
replaces the MoE, slot ``draft_hop`` the two all-reduce points and ``o_proj``; slot
``draft_fin`` (numeric) hands the MoE's finalize to the hop's final norm launch, as
SGLang's CuTe DSL fusion takes a deferred finalize (the norm then scales by the bf16
``gemma_weight``, as there).
"""

from __future__ import annotations

import logging
from functools import partial

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": (
        "embed_tokens",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "fc",
        "logits_processor",
        "lm_head",
    ),
    "srt/layers/logits_processor.py": (
        "_compute_lm_head",
        "_compute_lm_head(embedding_bias)",
        "vocab_size",
    ),
    "srt/distributed/communication_op.py": (
        "tensor_model_parallel_all_reduce",
        "tensor_model_parallel_all_gather",
    ),
    "srt/models/qwen2_moe.py": ("supports_deferred_finalize",),
    "srt/layers/layer_boundary/fusions/cutedsl.py": (
        "routed_output",
        "expert_weights",
        "permuted_indices",
        "gated_shared_output",
        "complete",
    ),
    "srt/layers/layernorm.py": ('"gemma_weight"',),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_5_mtp.py:Qwen3_5ForCausalLMMTP.forward": "6d038e59d55980a6",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward": "6473f673a1a0796b",
}


def fc_input(parts, token_ids, hidden):
    """The fc output [rows, hidden] of the tokens ``token_ids`` [rows] and the hidden
    rows ``hidden`` [rows, hidden] they follow."""
    lm = parts.lm
    embeds = lm.pre_fc_norm_embedding(parts.embed(token_ids))
    hidden = lm.pre_fc_norm_hidden(hidden)
    return lm.fc(torch.cat([embeds, hidden], dim=-1))


# Slot draft_hop on the draft extend's rows too: their FFN input all-reduce + residual +
# norm in the hop's MNNVL rounding, which equals what 2.2 runs there (FlashInfer's MNNVL
# fusion under DRAFT_MNNVL_AR) bit for bit at 1-7 rows
EXTEND_MNNVL = True


def layer(parts, x, positions, cache_loc, meta, kv, *, rows_per_request: int):
    """The decoder layer on the fc output ``x`` [rows, hidden]: the final-norm rows."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    lay = parts.layer
    residual = x
    rows = x.shape[0]
    # the draft hop takes the steps whose rows fit the target hop's mailbox; the
    # single-launch MoE the steps of at most its token count
    hop = (
        parts.hop
        if rows_per_request == 1 and parts.hop is not None and parts.hop.hop.supports(rows)
        else None
    )
    # a draft step of one request pushes its o_proj into the hop and releases the next
    # launch early; of two requests it takes only the hop's all-reduce points
    push = hop if x.shape[0] == 1 else None
    extend_hop = (
        parts.hop
        if rows_per_request > 1
        and EXTEND_MNNVL
        and parts.hop is not None
        and parts.hop.hop.supports(rows)
        else None
    )
    attn_out = parts.attention(
        lay.input_layernorm(x),
        positions,
        cache_loc,
        meta,
        kv,
        rows_per_request=rows_per_request,
        o_proj=(
            partial(push.o_proj, weight=lay.o_proj.weight)
            if push is not None and push.gemv
            else None
        ),
    )
    moe = (
        parts.moe
        if rows_per_request == 1 and parts.moe is not None and parts.moe.ready(rows)
        else None
    )
    if hop is not None:
        moe_in, residual = hop.ffn_input(
            attn_out,
            residual,
            lay.post_attention_layernorm,
            moe_single_launch=push is not None
            and moe is not None
            and moe.single_launch(1),
        )
    elif extend_hop is not None:
        moe_in, residual = extend_hop.ffn_input(
            attn_out, residual, lay.post_attention_layernorm, moe_single_launch=False
        )
    else:
        moe_in, residual = lay.post_attention_layernorm(
            tensor_model_parallel_all_reduce(attn_out), residual
        )
    fin = hop is not None and moe is None and "draft_fin" in parts.slots
    if fin and not lay.mlp.supports_deferred_finalize:
        _note_fin(rows, "SGLang's: the experts do not defer their finalize")
        fin = False
    if fin:
        owed = lay.mlp(moe_in, defer_finalize=True)
        if _hop_takes(hop.hop, owed):
            _note_fin(rows, "in the final norm's hop (slot draft_fin)")
            out, _ = hop.hop.moe_finalize_all_reduce_rms_norm(
                routed_output=owed.routed_output,
                expert_weights=owed.expert_weights,
                permuted_indices=owed.permuted_indices,
                gated_shared_output=owed.gated_shared_output,
                residual=residual,
                gamma=parts.model.norm.gemma_weight,
            )
            return out
        _note_fin(rows, "SGLang's: the hop does not read this deferred output")
        moe_out = owed.complete()
    else:
        moe_out = moe(moe_in) if moe is not None else lay.mlp(moe_in)
    if hop is not None:
        out, _ = hop.final_norm(
            moe_out,
            residual,
            parts.model.norm,
            head_follows=push is not None
            and parts.mkd is not None
            and parts.mkd.head_follows(),
        )
        return out
    out, _ = parts.model.norm(tensor_model_parallel_all_reduce(moe_out), residual)
    return out


_FIN_NOTED: set = set()


def _note_fin(rows: int, where: str) -> None:
    """Log once per captured row count where slot draft_fin's MoE finalize ran."""
    if torch.cuda.is_current_stream_capturing() and rows not in _FIN_NOTED:
        _FIN_NOTED.add(rows)
        logger.info(
            "NVIDIA megakernel: Qwen3.5 MTP draft step of %d rows: MoE finalize %s",
            rows,
            where,
        )


def _hop_takes(hop, owed) -> bool:
    """Whether the hop's finalize reads this deferred MoE output as SGLang wrote it."""
    shared = owed.gated_shared_output
    return (
        hop.supports(owed.m)
        and owed.routed_output.dtype == torch.bfloat16
        and owed.expert_weights.dtype == torch.bfloat16
        and owed.permuted_indices.dtype == torch.int32
        and (shared is None or shared.dtype == torch.bfloat16)
    )


def top1(parts, hidden):
    """SGLang's draft top-1 [rows] int64 of the final-norm rows ``hidden``."""
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    lm = parts.lm
    logits = lm.logits_processor._compute_lm_head(hidden, lm.lm_head, None)
    logits = tensor_model_parallel_all_gather(logits)
    logits = logits[:, : lm.logits_processor.vocab_size]
    return torch.argmax(logits, dim=-1)


def top1_into(parts, hidden, tokens, column: int, *, nan_key: int, **outputs) -> None:
    """Slot draft_multi's top-1 of the final-norm rows ``hidden``: the LM head on this
    rank's vocabulary shard, then the sharded argmax exchange, which writes the tokens
    into ``tokens[:, column]`` (and ``outputs``: topk_p, positions, mrope_positions)."""
    lm = parts.lm
    logits = lm.logits_processor._compute_lm_head(hidden, lm.lm_head, None)
    parts.top1x(logits, tokens, column, nan_key=nan_key, **outputs)
