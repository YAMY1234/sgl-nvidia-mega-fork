"""Packed prefill (TMA-gathered K/V) bitwise gate against the stock chunk kernel.

Run on SM107. Only passes on which stock runs the (16, 1, 2) tile are admitted,
so the output must be bit-identical, eagerly and after graph replay.
"""

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericSignature,
    PackedPrefillAttentionIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8._providers.packed_prefill import build


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")
@pytest.mark.parametrize(
    "qlens,prefixes,qh,kh",
    [
        ([513], [0], 24, 2),
        ([257, 513], [1024, 2048], 24, 2),
        ([513, 1], [0, 2048], 6, 1),
    ],
)
def test_packed_prefill_bits(qlens, prefixes, qh, kh):
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        _get_best_config,
        sparse_gqa_fwd_interface_triton_ck,
    )

    torch.manual_seed(733)
    rows = sum(qlens)
    max_q = max(qlens)
    assert _get_best_config(rows) == (16, 1, 2)
    q = torch.randn(rows, qh, 256, device="cuda", dtype=torch.bfloat16)
    lens = [a + b for a, b in zip(qlens, prefixes)]
    k = torch.randn(sum(lens), kh, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cq = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int64
    )
    ck = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int64
    )
    lengths = torch.tensor(lens, device="cuda", dtype=torch.int32)
    indices = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)
    offset = 0
    for n, prefix in zip(qlens, prefixes):
        for row in range(n):
            count = min(2051, prefix + row + 1)
            indices[offset + row, :count] = torch.arange(
                count, device="cuda", dtype=torch.int32
            )
        offset += n

    def stock():
        return sparse_gqa_fwd_interface_triton_ck(
            q, k, v, indices, cq, ck, lengths, 0.0625, max_q=max_q
        )

    ref = stock()
    out = torch.empty_like(q)
    io = PackedPrefillAttentionIO(q, k, v, indices, cq, ck, lengths, 0.0625, max_q, out)
    run = build(
        NumericSignature(
            O.PACKED_PREFILL_ATTENTION, 0, qh, kh, 256, 2051, 4, D.BF16, variant=3
        )
    )
    run(io)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(io)
    k.normal_()
    v.normal_()
    ref = stock()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
