# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The token embedding of a covered step.

Stock: SGLang's vocabulary-parallel lookup (each rank's shard, then the
all-reduce). With the ``embed`` slot, build all-gathers the shards once (+1 GB a
rank at TP4, recorded in the weight copies) and the step looks tokens up in the
whole table with SGLang's fused lookup, without the all-reduce. The values are the
same: the all-reduce sums one row and zeros (a -0.0 entry reads +0.0 after it).
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": ("embed_tokens",),
    "srt/layers/vocab_parallel_embedding.py": (
        "VocabParallelEmbedding",
        "tp_size",
        "use_attn_tp_group",
        "quant_method",
        "num_added_embeddings",
        "num_embeddings_padded",
        "org_vocab_size",
        "output_dtype",
        "num_embeddings",
        "num_embeddings_per_partition",
        "shard_indices",
        "org_vocab_start_index",
        "_use_triton_embedding",
        "_use_triton_embedding(input_)",
    ),
    "srt/layers/quantization/unquant.py": ("UnquantizedEmbeddingMethod",),
    "kernels/ops/embeddings/vocab_parallel_embedding.py": ("vocab_parallel_embedding",),
    "srt/utils/async_probe.py": ("maybe_detect_oob",),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("input_scattered",),
    "srt/distributed/parallel_state.py": (
        "rank_in_group",
        "world_size",
        "cpu_group",
        "device_group",
    ),
}


def reason(embed, rank: int, world: int) -> str:
    """Why the embedding cannot be replicated, or "": SGLang's TP-sharded table with
    no padding or added vocabulary, read by its fused lookup, whose shard on
    ``rank`` is the ``rank``-th slice of the vocabulary (what the gather assumes)."""
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if type(embed) is not VocabParallelEmbedding:
        return f"the embedding is {type(embed).__name__}"
    weight = embed.weight
    if not (
        embed.tp_size > 1
        and not embed.use_attn_tp_group
        and isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
        and embed.num_added_embeddings == 0
        and embed.num_embeddings_padded == embed.org_vocab_size
        and embed.output_dtype is None
        and weight.is_cuda
        and weight.dim() == 2
        and weight.stride(1) == 1
    ):
        return "not a plain TP-sharded vocabulary"
    rows = embed.num_embeddings_per_partition
    if (
        embed.shard_indices.org_vocab_start_index != rank * rows
        or rows * world != embed.org_vocab_size
        or weight.size(0) != rows
    ):
        return "its shard is not the rank's slice of the vocabulary"
    return ""


# the gathered table per embedding module: the target and the DSpark draft share one
_TABLES: dict[int, torch.Tensor] = {}


def replicate(embed, copies: dict, group):
    """(the whole table on this rank, "") or (None, the reason it is not). Collective
    over the TP group: every rank decides the same. A module gathered before (the
    draft's build and the target's share the table) returns its table at once."""
    if id(embed) in _TABLES:
        return _TABLES[id(embed)], ""
    reasons = [None] * group.world_size
    why = reason(embed, group.rank_in_group, group.world_size)
    torch.distributed.all_gather_object(reasons, why, group=group.cpu_group)
    if any(reasons):
        return None, "; ".join(f"rank {r}: {e}" for r, e in enumerate(reasons) if e)
    shard = embed.weight.detach().contiguous()
    full = shard.new_empty((embed.org_vocab_size, shard.size(1)))
    torch.distributed.all_gather_into_tensor(full, shard, group=group.device_group)
    copies[("embedding", "model.embed_tokens.weight")] = full
    _TABLES[id(embed)] = full
    return full, ""


def gathered_table(embed):
    """The whole table ``replicate`` gathered for this module, or None."""
    return _TABLES.get(id(embed))


def embed(st):
    """[rows, dim] for ``st.input_ids``."""
    embed_tokens = st.model.embed_tokens
    table = st.bound.get("embedding") if st.slots.embed else None
    if table is not None:
        x = _local(embed_tokens, table, st.input_ids)
        if x is not None:
            return x
    return embed_tokens(st.input_ids)


def _local(embed_tokens, table, input_ids):
    """The rows from this rank's whole table, or None where SGLang's sharded lookup
    must run (scattered input, or input its fused lookup does not take)."""
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )
    from sglang.srt.layers.layer_boundary import get_attn_tp_context
    from sglang.srt.utils.async_probe import maybe_detect_oob

    if get_attn_tp_context().input_scattered or not embed_tokens._use_triton_embedding(
        input_ids
    ):
        return None
    maybe_detect_oob(
        input_ids, 0, embed_tokens.num_embeddings, "VocabParallelEmbedding input id"
    )
    vocab = table.size(0)
    return vocab_parallel_embedding(input_ids, table, 0, vocab, 0, vocab, vocab)
