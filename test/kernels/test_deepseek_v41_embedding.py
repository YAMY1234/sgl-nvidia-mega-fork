# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The replicated embedding on one GB300: ``runtime.embedding``'s lookup on the whole
table against SGLang's TP4 lookup (each rank's fused shard lookup, then the
all-reduce's sum) at every routed row count, for int32 and int64 token ids. The
values are equal; only a -0.0 entry reads +0.0 after the all-reduce."""

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.runtime import embedding

VOCAB, HIDDEN, TP = 129280, 5120, 4  # DeepSeek-V4.1-Flash's table, sharded at TP4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class _Embedding:
    num_embeddings = VOCAB

    def _use_triton_embedding(self, input_):
        return True


@pytest.fixture(scope="module")
def table():
    gen = torch.Generator(device="cuda").manual_seed(5)
    full = torch.randn((VOCAB, HIDDEN), device="cuda", generator=gen).bfloat16()
    full[torch.rand((VOCAB, HIDDEN), device="cuda", generator=gen) < 1e-3] = -0.0
    return full


def _sharded(ids, full):
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )

    part = VOCAB // TP
    outs = [
        vocab_parallel_embedding(
            ids,
            full[r * part : (r + 1) * part],
            r * part,
            (r + 1) * part,
            0,
            VOCAB,
            VOCAB,
        )
        for r in range(TP)
    ]
    return torch.stack(outs).float().sum(0).bfloat16()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_the_whole_table_lookup_equals_the_sharded_one(table, dtype, monkeypatch):
    from sglang.srt.layers import layer_boundary

    context = type("_Context", (), {"input_scattered": False})()
    monkeypatch.setattr(layer_boundary, "get_attn_tp_context", lambda: context)
    embed = _Embedding()
    gen = torch.Generator(device="cuda").manual_seed(11)
    for rows in range(1, 49):
        ids = torch.randint(0, VOCAB, (rows,), device="cuda", generator=gen).to(dtype)
        ids[0] = VOCAB - 1  # the last row of the last shard
        got = embedding._local(embed, table, ids)
        want = _sharded(ids, table)
        torch.cuda.synchronize()
        assert torch.equal(got, want), rows
        assert torch.equal(got.view(torch.int16), table[ids.long()].view(torch.int16))
    context.input_scattered = True  # scattered input: SGLang's own lookup
    assert embedding._local(embed, table, ids) is None
