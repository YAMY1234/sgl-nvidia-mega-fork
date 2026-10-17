# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-stop waits on 4 SM100-family GPUs (GB300): when one rank never joins a hop, the
ranks that wait on it fault within their deadline instead of hanging (the Lamport
consumer of the wo_b publish, 1-8 rows).

Run with ``python test/kernels/test_deepseek_v41_deadline.py``: each case runs its own
torchrun, since a faulted rank leaves its CUDA context unusable.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest
import torch

DEADLINE_S = 2.0
WORKER = textwrap.dedent(
    """
    import os, sys, time
    import torch
    from sglang.srt.distributed import parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

    case, deadline = sys.argv[1], float(sys.argv[2])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.set_custom_all_reduce(True)
    ps.init_distributed_environment(world_size=world, rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]), distributed_init_method="env://")
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    tp = ps.get_tp_group()
    dev = torch.device("cuda", torch.cuda.current_device())
    H, HC = 5120, 4
    if case == "lamport":
        from sgl_nvidia_mega.deepseek_v41.sm103 import all_reduce_mhc as m
        m.set_wait_deadline(deadline)
        hop = m.AllReduceMHC(hidden_size=H, hc_mult=HC, max_num_tokens=16, top_k=6, device=dev,
                             group=tp.device_group)
        rows = 6
        def run():
            x = torch.randn(rows, H, device=dev).bfloat16()
            return hop(x, torch.randn(rows, HC, H, device=dev).bfloat16(),
                       torch.rand(rows, HC, device=dev), torch.rand(rows, HC, HC, device=dev),
                       torch.rand(rows, HC, device=dev), torch.randn(H, device=dev).bfloat16(), 1e-6)
    run(); torch.cuda.synchronize()  # every rank joins once: compiled and healthy
    torch.distributed.barrier(group=tp.cpu_group)
    print(f"RANK{rank} HEALTHY", flush=True)
    if rank == world - 1:
        time.sleep(4 * deadline + 20)  # never joins the second hop
        os._exit(0)
    t0 = time.time()
    try:
        run(); torch.cuda.synchronize()
        print(f"RANK{rank} NO-FAULT", flush=True)
    except Exception as e:
        print(f"RANK{rank} FAULTED after {time.time() - t0:.1f} s: {type(e).__name__}", flush=True)
    os._exit(0)
    """
)


def _gpu4() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


@pytest.mark.skipif(not _gpu4(), reason="needs 4 SM100-family GPUs")
@pytest.mark.parametrize("case", ["lamport"])
def test_a_missing_rank_faults_its_peers_in_time(case, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    port = 29500 + os.getpid() % 2000
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4",
         f"--master_port={port}", str(script), case, str(DEADLINE_S)],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    log = out.stdout + out.stderr
    assert log.count("HEALTHY") == 4, log[-4000:]
    # the waiting ranks fault (the kernel's store to 0x0 surfaces as a CUDA error);
    # none returns as if the hop had completed
    assert "NO-FAULT" not in log, log[-4000:]
    faulted = [l for l in log.splitlines() if "FAULTED after" in l]
    assert len(faulted) == 3, log[-4000:]
    for line in faulted:
        assert float(line.split("after ")[1].split(" s")[0]) < 4 * DEADLINE_S + 10, line
    assert time.time() - t0 < 300


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
