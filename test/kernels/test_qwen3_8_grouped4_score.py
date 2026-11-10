"""Grouped4 verify scorer bitwise gate against the stock per-row scorer; run on SM107.

Valid logits (rows longer than 512 blocks, positions below each row's length) and
the ordered fast top-k must be bit-identical, eagerly and after graph replay.
"""

import math

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    SparseScoreIO,
)

from sgl_nvidia_mega.qwen3_8._providers.sparse import build


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")
@pytest.mark.parametrize(
    "rows,variant,ctas",
    [(68, 7, 128), (128, 7, 128), (416, 7, 512), (576, 7, 128), (88, 6, 512)],
)
def test_grouped4_score_bits(rows, variant, ctas):
    from sglang.kernels.ops.attention.fast_topk import fast_topk
    from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode

    torch.manual_seed(309)
    capacity = 32768
    pages = capacity // 16
    q = torch.randn(rows, 8, 128, device="cuda", dtype=torch.bfloat16)
    q[:, 4:] = 0
    cache = torch.randn(pages, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = (
        torch.arange(pages, device="cuda", dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    # Four verify rows per request: shared page prefix, lengths increasing by one.
    base = (torch.arange(rows // 4, device="cuda", dtype=torch.int32) * 607) % (
        capacity - 4
    )
    lengths = (
        base[:, None] + torch.arange(1, 5, device="cuda", dtype=torch.int32)
    ).reshape(-1)
    out = torch.empty(rows, capacity, device="cuda", dtype=torch.float32)
    io = SparseScoreIO(
        q,
        cache,
        table,
        lengths,
        None,
        None,
        16,
        4,
        math.sqrt(128),
        out,
        torch.empty(0, device="cuda", dtype=torch.uint8),
    )
    run = build(
        NumericSignature(
            O.SPARSE_SCORE,
            rows,
            8,
            1,
            128,
            512,
            4,
            D.BF16,
            context_bound=capacity,
            variant=variant,
            ctas_per_group=ctas,
        )
    )

    def check():
        ref = qsa_mqa_decode(q, cache, table, lengths, capacity, min_scored_len=512)
        valid = (torch.arange(capacity, device="cuda")[None, :] < lengths[:, None]) & (
            lengths[:, None] > 512
        )
        assert torch.equal(out[valid], ref[valid])
        assert torch.equal(
            fast_topk(out, lengths, topk=512, row_starts=None),
            fast_topk(ref, lengths, topk=512, row_starts=None),
        )

    run(io)
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(io)
    q[:, :4].normal_()
    cache.normal_()
    graph.replay()
    torch.cuda.synchronize()
    check()
