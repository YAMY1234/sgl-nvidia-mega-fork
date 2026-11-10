"""Fast prefill select ordered bitwise gate against the stock relative path.

Run on SM107. The selected token IDs must equal stock's in order, including ties
and ragged tails, eagerly and after graph replay; no set-only comparison.
"""

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericSignature,
    PrefillSelectIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8._providers.prefill_select import build


def caller_io(q, k, starts, ends, positions, seqlens, out, width):
    from sgl_nvidia_mega.qwen3_8.qsa.prefill_select import plan_chunk_rows

    rows = q.shape[0]
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    chunk = plan_chunk_rows(rows, width, 2 * sms, 128 * 1024 * 1024)
    padded = (chunk + 31) // 32 * 32
    i32 = dict(device=q.device, dtype=torch.int32)
    return PrefillSelectIO(
        q,
        k,
        starts,
        ends,
        positions,
        seqlens,
        out,
        torch.empty((padded, 4, 128), device=q.device, dtype=q.dtype),
        torch.empty(padded, **i32),
        torch.empty(padded, **i32),
        ends - starts,
        torch.zeros(chunk, **i32),
        torch.empty((rows, 512), **i32),
        torch.empty((padded, width), device=q.device, dtype=torch.float32),
        torch.empty(4096, **i32),
        chunk,
        sms,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")
@pytest.mark.parametrize(
    "rows,width,ties",
    [
        (1, 512, False),
        (33, 4096, False),
        (65, 8193, False),
        (33, 8193, True),
        (2049, 8193, False),
    ],
)
def test_prefill_select_ordered_bits(rows, width, ties):
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.qsa.kernel import (
        expand_qsa_block_indices,
        qsa_fast_topk,
    )
    from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_prefill_relative

    torch.manual_seed(911)
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(width * 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    if ties:
        q.zero_()
        k.zero_()
    starts = (torch.arange(rows, device="cuda", dtype=torch.int32) % 2) * width
    lengths = torch.linspace(1, width, rows, device="cuda").int()
    ends = starts + lengths
    positions = lengths * 4 - 1
    seqlens = positions + 1

    def stock():
        # The stock divide path; the scale fold stores the same bits.
        with envs.SGLANG_OPT_QSA_PREFILL_SCALE_FOLD.override(False):
            logits = qsa_mqa_prefill_relative(q, k, starts, ends, width)
        ids = qsa_fast_topk(logits, torch.zeros_like(starts), lengths, 512)
        return expand_qsa_block_indices(ids, positions, seqlens, 4, 2048)

    reference = stock()
    out = torch.empty_like(reference)
    io = caller_io(q, k, starts, ends, positions, seqlens, out, width)
    run = build(
        NumericSignature(
            O.PREFILL_SELECT, 0, 4, 1, 128, 512, 4, D.BF16, context_bound=width
        )
    )
    for _ in range(2):
        run(io)
    torch.cuda.synchronize()
    assert torch.equal(out, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(io)
    q.normal_()
    reference = stock()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, reference)
