# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import random
import time

import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from test_qwen3_5_gemm_ar import (  # noqa: F401  (pytest puts this directory on sys.path; rank / tail are fixtures)
    _fin,
    _idle_buffers_poisoned,
    _l5,
    _operands,
    _rows,
    pytestmark,
    rank,
    tail,
)

REPLAYS = int(os.environ.get("GEMM_AR_STRESS_REPLAYS", "100000"))
CALLS = 12


def test_gemm_ar_skewed_rank_stress(rank, tail):  # noqa: F811
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    taken = set(_rows())
    widths = tuple(
        m
        for m in (
            (112, 56, 84, 35, 98, 63, 42) if world == 8 else (63, 56, 35, 42, 49, 64)
        )
        if m in taken
    )
    xs = [
        _operands(widths[j % len(widths)], device, rank, seed=2600 + j)
        for j in range(CALLS)
    ]

    def run():
        res, outs = {}, []
        for j, x in enumerate(xs):
            m = x["x"].shape[0]
            y = dict(x, residual=res.get(m, x["residual"]))
            o = _fin(tail, y) if j % 3 == 2 else _l5(tail, y)
            res[m] = o[1]
            outs.append(o)
        return outs

    with torch.inference_mode():
        eager = [[t.clone() for t in o] for o in run()]
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = run()
        flat = [t for o in outs for t in o]
        want = [t.view(torch.uint8) for o in eager for t in o]
        got = [t.view(torch.uint8) for t in flat]
        total_bytes = sum(t.numel() for t in got)
        poison_visible = int(sum((w != 0xFF).sum().item() for w in want))
        acc = torch.zeros((), dtype=torch.int64, device=device)

        def poison():
            for t in got:
                t.fill_(0xFF)

        def count():
            for g, w in zip(got, want):
                acc.add_((g != w).sum())

        poison()
        count()
        torch.cuda.synchronize()
        control = int(acc.item())
        acc.zero_()
        assert control == poison_visible, (control, poison_visible, total_bytes)

        # the first replay must equal the eager chain
        dist.barrier()
        poison()
        graph.replay()
        count()
        torch.cuda.synchronize()
        assert int(acc.item()) == 0, (
            "first replay differs from eager",
            int(acc.item()),
        )
        acc.zero_()

        rng = random.Random(7919 * (rank + 1))
        dist.barrier()
        t0 = time.time()
        bad_at, seen = [], 0
        skewed = 0
        for r in range(REPLAYS):
            poison()
            if rng.random() < 0.3:
                torch.cuda._sleep(rng.randrange(0, 40_000))
                skewed += 1
            graph.replay()
            count()
            if (r + 1) % 2000 == 0 or r + 1 == REPLAYS:
                torch.cuda.synchronize()
                n = int(acc.item())
                if n != seen:
                    bad_at.append((r + 1, n - seen))
                    seen = n
        torch.cuda.synchronize()
        dt = time.time() - t0
        tot = torch.tensor([seen, skewed], dtype=torch.int64, device=device)
        dist.all_reduce(tot, group=ps.get_tp_group().device_group)
        print(
            f"STRESS gemm_ar rank {rank} world {world} widths {widths} calls/replay {CALLS} replays {REPLAYS} "
            f"skewed {skewed} mismatched-bytes {seen} (all ranks {int(tot[0].item())}) bad-windows {bad_at[:8]} "
            f"control {control}/{poison_visible} ({total_bytes} bytes) {dt:.1f} s",
            flush=True,
        )
        _idle_buffers_poisoned(tail.twoshot)
        assert int(tot[0].item()) == 0, bad_at


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=3600)
