# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PERF-DRAFT's folded-key top-1 (SGLANG_PERFDRAFT_TOP1_KEYS) on 4 SM100-family GPUs (GB300) with NVLink multicast:
the MXFP8 draft head folding each row's key over its shard (``head_mx8_keys_m<m>``) and the one-CTA exchange of
those keys (``draft_top1_keys_xchg``) write the same tokens, 1.0 and positions + 1 as the head's logits
(``head_mx8_m<m>``) through the scanning exchange (``draft_top1_xchg``), for 1-16 rows, under both NaN rules, with
equal logits across ranks (ties to the lowest id), and from a CUDA graph.
Run with ``python test/kernels/test_qwen3_5_draft_top1_keys.py --num-gpu 4``.
"""

import os

import pytest
import torch
import torch.distributed as dist
from sglang.test.kernels.utils import multigpu_pytest_main

H, VOCAB, NUM_STEPS = 4096, 2048, 6  # VOCAB rows a rank

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


@pytest.fixture(scope="module")
def ranks():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    cpu_group = dist.new_group(backend="gloo")
    yield dist.get_rank(), dist.get_world_size(), dist.group.WORLD, cpu_group
    dist.destroy_process_group()


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


@pytest.fixture(scope="module")
def parts(ranks):
    """This rank's MXFP8 head shard and the sharded top-1; ranks 0 and 2 hold the same shard rows (equal logits
    across ranks: ties to the lowest id)."""
    from sgl_nvidia_mega.qwen3_5.sm103.draft.head_mx8 import DraftHeadMx8
    from sgl_nvidia_mega.qwen3_5.sm103.draft.top1 import Top1Exchange

    rank, world, group, cpu_group = ranks
    g = _gen(4242)
    full = (torch.randn(world * VOCAB, H, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    full[2 * VOCAB : 3 * VOCAB] = full[:VOCAB]
    head = DraftHeadMx8(full[rank * VOCAB : (rank + 1) * VOCAB].contiguous(), gemv="nvrtc")
    head.prepare_keys()
    x = Top1Exchange(
        world=world,
        rank=rank,
        vocab_local=VOCAB,
        vocab_start=rank * VOCAB,
        vocab_size=world * VOCAB,
        group=group,
        cpu_group=cpu_group,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    return head, x


def _hidden(m, case, seed):
    g = _gen(seed)  # the same rows on every rank, as the draft's final norm gives them
    h = torch.randn(m, H, device="cuda", generator=g).to(torch.bfloat16)
    if case == "nan":
        h[m - 1, 7] = float("nan")
    return h


def _run(head, x, h, nan_key, keys: bool):
    m = h.shape[0]
    tokens = torch.full((m, NUM_STEPS), -1, dtype=torch.int64, device="cuda")
    topk_p = torch.zeros(m, 1, dtype=torch.float32, device="cuda")
    positions = torch.arange(m, dtype=torch.int64, device="cuda") * 10
    mrope = torch.arange(3 * (m + 1), dtype=torch.int64, device="cuda").view(3, -1)
    out = dict(topk_p=topk_p, positions=positions, mrope_positions=mrope[:, :m])
    if keys:
        head.fold_keys(h, x.keys, vocab_start=x.vocab_start, vocab_size=x.vocab_size, nan_key=nan_key, site="test")
        x.exchange_keys(m, tokens, 3, nan_key=nan_key, **out)
    else:
        x(head.logits(h, "test"), tokens, 3, nan_key=nan_key, **out)
    torch.cuda.synchronize()
    return tokens, topk_p, positions, mrope


@pytest.mark.parametrize("case", ["random", "nan"])
@pytest.mark.parametrize("nan", ["split", "torch"])
@pytest.mark.parametrize("m", list(range(1, 17)))
def test_folded_keys_equal_the_logits_path(ranks, parts, m, nan, case):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import top1

    rank, world, group, cpu_group = ranks
    head, x = parts
    key = top1.NAN_KEY_SPLIT_ARGMAX if nan == "split" else top1.NAN_KEY_TORCH_ARGMAX
    with torch.inference_mode():
        for seed in range(3):
            h = _hidden(m, case, 100 * m + seed)
            want = _run(head, x, h, key, keys=False)
            got = _run(head, x, h, key, keys=True)
            for a, b in zip(got, want):
                assert torch.equal(a, b), (m, nan, case, seed)
            assert (x.keys == 0).all()  # the exchange leaves the fold zero
    dist.barrier(group=cpu_group)


def test_folded_keys_replay_from_a_cuda_graph(ranks, parts):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import top1

    rank, world, group, cpu_group = ranks
    head, x = parts
    m = 8
    with torch.inference_mode():
        h = _hidden(m, "random", 77)
        tokens = torch.zeros(m, NUM_STEPS, dtype=torch.int64, device="cuda")
        positions = torch.zeros(m, dtype=torch.int64, device="cuda")

        def run():
            for col in range(1, NUM_STEPS):
                head.fold_keys(h, x.keys, vocab_start=x.vocab_start, vocab_size=x.vocab_size,
                               nan_key=top1.NAN_KEY_SPLIT_ARGMAX, site="test")
                x.exchange_keys(m, tokens, col, nan_key=top1.NAN_KEY_SPLIT_ARGMAX, positions=positions)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()
        for seed in (1, 2, 3):
            h.copy_(_hidden(m, "random", 900 + seed))
            positions.zero_()
            graph.replay()
            torch.cuda.synchronize()
            want = _run(head, x, h, top1.NAN_KEY_SPLIT_ARGMAX, keys=False)[0]
            assert all(torch.equal(tokens[:, c], want[:, 3]) for c in range(1, NUM_STEPS))
            assert torch.equal(positions, torch.full_like(positions, NUM_STEPS - 1))
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
