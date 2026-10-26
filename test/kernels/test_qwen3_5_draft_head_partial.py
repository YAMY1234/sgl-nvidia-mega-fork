# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-GPU admission for Q35FLY head+partial; distributed lockstep remains required.

Run: python -m pytest -q test/kernels/test_qwen3_5_draft_head_partial.py
Compares the original 24-SM partial plus reduced-grid head against the fused full-grid
head, bitwise, and checks repeated graph replay with changed activations and epochs.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("mxfp8", [False, True])
def test_head_partial_exact_and_graph(world, mxfp8, monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import head_partial, xchg
    from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8 import DraftHeadMx8

    monkeypatch.setattr(xchg, "PARTIAL_ROWS4", True)
    h, n, pad, eps = 4096, 31040, 120 * 1024, 1e-6
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(3541)

    def rand(*shape):
        return (torch.randn(*shape, device=dev, generator=gen) * 0.1).bfloat16()

    x, weight, shard, norm = rand(1, h), rand(n, h), rand(h, 2*h//world), rand(h)
    mx8 = DraftHeadMx8(weight, max_rows=1, gemv="nvrtc") if mxfp8 else None
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    reserve = min(24, sms - 1)
    ref_logits, got_logits = torch.empty(1, n, device=dev, dtype=torch.bfloat16), torch.empty(1, n, device=dev, dtype=torch.bfloat16)
    ref_part = torch.empty(sms-reserve, device=dev, dtype=torch.int32)
    got_part = torch.empty(sms, device=dev, dtype=torch.int32)
    ref_fc, got_fc = torch.empty(1, h, device=dev), torch.empty(1, h, device=dev)
    epoch = torch.tensor([1, 0], device=dev, dtype=torch.int32)
    ref_sync, got_sync = torch.zeros_like(epoch), torch.zeros_like(epoch)
    col = h - 2*h//world

    def reference(reverse):
        xchg.mkd_hidden_partial(x, shard, norm, eps, col, ref_fc, epoch, ref_sync, world, ctas=reserve)
        if mx8 is None:
            xchg.mkd_draft_head(x, weight, ref_logits, ref_part, reverse=reverse, smem_pad=pad)
        else:
            mx8.mkd_head(x, ref_logits, ref_part, reverse=reverse, smem_pad=pad)

    def candidate(reverse):
        head_partial.head_with_partial(x, weight, got_logits, got_part, shard, norm,
            eps, col, got_fc, epoch, got_sync, world, reverse=reverse, smem_pad=pad, head_mx8=mx8)

    def same():
        assert torch.equal(got_logits.view(torch.int16), ref_logits.view(torch.int16))
        assert torch.equal(got_fc.view(torch.int32), ref_fc.view(torch.int32))
        assert int((got_part.long() & 0xffffffff).max()) == int((ref_part.long() & 0xffffffff).max())
        assert got_sync.tolist() == ref_sync.tolist() == [0, int(epoch[0])]

    with torch.inference_mode():
        for reverse in (False, True):
            reference(reverse)
            candidate(reverse)
            torch.cuda.synchronize()
            same()
            # Warm before capture; the captured output addresses remain stable.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate(reverse)
            for i in range(5):
                x.copy_(rand(1, h) if i else torch.zeros_like(x))
                epoch[0] = 10 + i
                reference(reverse)
                graph.replay()
                torch.cuda.synchronize()
                same()
