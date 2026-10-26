# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HP2 TP4/TP8 admission: exact arithmetic and A16/H8 epoch interoperability.

Run under torchrun on four or eight NVLink-connected GPUs, with one process/GPU:
  torchrun ... -m pytest -q test/kernels/test_qwen3_5_head_exchange_distributed.py

Each graph contains 17 exchanges with rank-dependent delay and token-only tails.
The mixed graph switches A16/H8/H8/A16 without a per-exchange host barrier. Fresh
inputs and 16 replays exercise every three-slot wrap offset. Compare every step's
logits, token, partial, FC and positions bitwise with the original A-only graph.
"""

import json
import os

import pytest
import torch
import torch.distributed as dist

import test_qwen3_5_draft_xchg as base

ranks = base.ranks
pytestmark = base.pytestmark
KEY = "SGLANG_Q35FLY_DRAFT_HEAD_EXCHANGE"
COUNT = 17  # prime to the three epoch slots and the four-arm pattern


@pytest.fixture(scope="module")
def hp2_boundary(ranks):
    """One process group and one multicast boundary for both arithmetic cases."""
    from sgl_nvidia_mega.qwen3_5.draft import xchg as boundary_mod
    from sgl_nvidia_mega.qwen3_5.sm103.draft import xchg as kernels

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(KEY, "0")
        patch.setenv("Q35MF_AB_VARIANTS", json.dumps({"A": {KEY: "0"}, "H": {KEY: "1"}}))
        patch.setattr(boundary_mod, "HEAD_PARTIAL", False)
        patch.setattr(boundary_mod, "HEAD_EXCHANGE", False)
        patch.setattr(kernels, "PARTIAL_ROWS4", True)
        patch.setattr(kernels, "XCHG_SPLIT_SEND", True)
        mkd, _ = base.boundary.__wrapped__(ranks)
        yield mkd


@pytest.mark.parametrize("mxfp8", [False, True])
def test_mixed_graph_exact_with_skew_and_epoch_wrap(ranks, hp2_boundary, mxfp8):
    from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8 import DraftHeadMx8

    rank, world, _, cpu_group = ranks
    assert world in (4, 8)
    mkd = hp2_boundary
    assert mkd._head_exchange_available and not mkd.fuse_head_exchange
    if mxfp8:
        mkd.set_head_mx8(DraftHeadMx8(mkd.lm_head_weight, max_rows=1, gemv="nvrtc"))
    else:
        mkd.head_mx8 = None

    inputs = base._randn(COUNT, base.H, g=base._gen(35100))
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")
    tokens = torch.zeros(1, base.NUM_STEPS, dtype=torch.int64, device="cuda")
    mrope = torch.zeros(3, 1, dtype=torch.int64, device="cuda")

    def run(mixed):
        positions.zero_()
        mrope.zero_()
        mkd.fc_buf.zero_()
        out = []
        for i in range(COUNT):
            os.environ[KEY] = "1" if mixed and i % 4 in (1, 2) else "0"
            # Rotate the late producer across all ranks, inside the captured graph.
            if i % world == rank:
                torch.cuda._sleep(2_000_000)
            step = i % (base.NUM_STEPS - 1)
            mkd.begin_step(step, 1)
            mkd.fc()
            hidden = inputs[i:i + 1]
            mkd.hidden_partial(hidden)
            logits = mkd.head(hidden)
            probability, token = mkd.finish_step(
                logits, positions, tokens, step, step + 1, mrope)
            out.append(tuple(t.clone() for t in (
                logits, token, probability, mkd.own_partial, mkd.fc_buf,
                positions, mrope, tokens[:, step + 1])))
        # An A arm near the graph end can leave a side-stream partial pending.
        mkd._join_side_streams()
        os.environ[KEY] = "0"
        return out

    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(False)
            run(True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        graphs, outputs = [], []
        for mixed in (False, True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run(mixed)
            graphs.append(graph)
            outputs.append(captured)
            torch.cuda.synchronize()
            dist.barrier(group=cpu_group)

        for replay in range(16):
            fresh = base._randn(COUNT, base.H, g=base._gen(35200 + replay))
            # Includes exact ties and the zero-input norm path on every rank.
            fresh[replay % COUNT].zero_()
            # Force the exact full-logit NaN rescan, still comparing raw bytes.
            fresh[(replay + 1) % COUNT, 17] = float("nan")
            inputs.copy_(fresh)
            before = int(mkd.epoch_done[0])
            graphs[0].replay()
            graphs[1].replay()
            torch.cuda.synchronize()
            assert mkd.epoch_done.tolist() == [before + 2 * COUNT, 0]
            assert int(mkd._head_exchange_done) == 0
            if mkd.is_hidden_rank:
                assert mkd.partial_sync.tolist() == [0, before + 2 * COUNT - 1]
            for step, (want, got) in enumerate(zip(*outputs)):
                for field, (a, b) in enumerate(zip(want, got)):
                    # Compare raw bytes even for FP32/BF16; no tolerance or NaN exception.
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                        rank, mxfp8, replay, step, field)
            assert int(positions) == COUNT
            dist.barrier(group=cpu_group)
