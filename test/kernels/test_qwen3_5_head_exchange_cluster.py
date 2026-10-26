# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HP2 one-GPU scheduler/memory-order admission, before distributed exchange tests.

No multicast or model service. Deliberately oversubscribes the head grid with 200 KiB
shared/CTA; non-last clusters must retire so queued clusters can run. It tests the
production last-cluster handoff under graph replay, not just a Python simulation.
"""

import ctypes

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)

PROBE = r'''
struct HeadClusterProbe {
  uint32_t *completed, *words, *observed, *tail_count, *winner;
  const uint32_t* epoch;
  uint32_t salt;
};
extern "C" __global__ void __launch_bounds__(512, 1) hp2_cluster_probe(
    const HeadClusterProbe __grid_constant__ p) {
  sglang::device::PDLWaitPrimary<true>();
  sglang::device::PDLTriggerSecondary<true>();
  const uint32_t i = blockIdx.x * 512 + threadIdx.x;
  const uint32_t epoch = *p.epoch;
  // Make a different cluster slow on successive replays; no grid-order assumption.
  if (blockIdx.x / 8 == epoch % (gridDim.x / 8)) {
    const uint64_t until = clock64() + 4000;
    while (clock64() < until) {}
  }
  p.words[i] = epoch * 65537u ^ i ^ p.salt;
  if (sglang::head_exchange_last_cluster(p.completed)) {
    const uint32_t local = blockIdx.x % 8;
    if (local == 0 && threadIdx.x == 0) {
      atomicAdd(p.tail_count, 1u);
      *p.winner = blockIdx.x / 8;
    }
    for (uint32_t j = local * 512 + threadIdx.x; j < gridDim.x * 512; j += 8 * 512)
      p.observed[j] = p.words[j];
  }
}
struct HeadClusterFollower {
  const uint32_t* observed;
  uint32_t* checked;
  uint32_t count;
};
extern "C" __global__ void __launch_bounds__(256) hp2_cluster_follower(
    const HeadClusterFollower __grid_constant__ p) {
  sglang::device::PDLWaitPrimary<true>();
  sglang::device::PDLTriggerSecondary<true>();
  for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < p.count; i += gridDim.x * 256)
    p.checked[i] = p.observed[i];
}
'''


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ("completed", "words", "observed", "tail_count", "winner", "epoch")] + [
                    ("salt", ctypes.c_uint32)]


class FollowerParams(ctypes.Structure):
    _fields_ = [("observed", ctypes.c_void_p), ("checked", ctypes.c_void_p),
                ("count", ctypes.c_uint32)]


def test_last_cluster_progress_memory_order_and_graph():
    from sgl_nvidia_mega.common import jit
    from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import SRC

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    stems = ("mkd_fc", "mkd_head", "head_mx8", "mkd_head_partial", "mkd_xchg")
    headers = {stem + ".cuh": (SRC / (stem + ".cu")).read_text() for stem in stems}
    headers["mkd_common.cuh"] = (SRC / "mkd_common.cuh").read_text()
    cubin = jit.compile_source((SRC / "mkd_head_exchange.cu").read_text() + PROBE,
        "q35fly_hp2_cluster_probe", f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", "-DMKD_WORLD=8"), headers=headers)
    kernel = jit.load(cubin, "hp2_cluster_probe", (512, 1, 1), 200 * 1024)
    follower = jit.load(cubin, "hp2_cluster_follower", (256, 1, 1), 0)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rounded = sms // 8 * 8
    for blocks in (8, rounded, 2 * rounded):
        words = torch.empty(blocks * 512, dtype=torch.int32, device=device)
        observed = torch.empty_like(words)
        completed = torch.zeros(1, dtype=torch.int32, device=device)
        tail_count, winner, epoch = completed.clone(), completed.clone(), completed.clone()
        checked = torch.empty(4, blocks * 512, dtype=torch.int32, device=device)
        params, followers = [], []
        for salt in range(4):
            params.append(Params(*(t.data_ptr() for t in (
                completed, words, observed, tail_count, winner, epoch)), salt * 1_000_000))
            followers.append(FollowerParams(observed.data_ptr(), checked[salt].data_ptr(), words.numel()))

        def launch():
            stream = torch.cuda.current_stream().cuda_stream
            # No non-PDL copy or host wait separates these kernels. The next
            # producer must wait for the tail/follower and may safely reuse done.
            for p, f in zip(params, followers):
                jit.launch(kernel, (blocks, 1, 1), [p], stream, cluster=(8, 1, 1))
                jit.launch(follower, (8, 1, 1), [f], stream)

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        torch.cuda.synchronize()
        tail_count.zero_()
        indices = torch.arange(blocks * 512, dtype=torch.int32, device=device)
        for i in range(64):
            epoch.fill_(i + 1)
            graph.replay()
            torch.cuda.synchronize()
            assert int(completed) == 0
            assert int(tail_count) == (i + 1) * 4
            assert 0 <= int(winner) < blocks // 8
            for salt in range(4):
                assert torch.equal(checked[salt], ((i + 1) * 65537) ^ indices ^ (salt * 1_000_000))
