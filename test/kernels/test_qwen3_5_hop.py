# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 decode hop (``qwen3_5/sm103/hop``) on 4 SM100-family GPUs (GB300) with
NVLink multicast, against SGLang's FlashInfer MNNVL CuTe DSL fusion workspace as the
verify server builds it (hidden 4096, top-10, no early shared-expert load), bit for bit:

- the all-reduce + residual add + RMSNorm, and the MoE finalize (top-10 rows and the
  gated shared output) + all-reduce + residual add + RMSNorm, at 1-15 rows, over a
  sequence that cycles the three mailbox stages, mixes both operations and delays one
  rank in turn; the static FP8 output equals SGLang's static quantization of the norm;
- the same operations replayed from a CUDA graph with fresh inputs.

Run with ``python test/kernels/test_qwen3_5_hop.py --num-gpu 4``.
"""

import os

import pytest
import torch
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

HIDDEN, TOP_K, EPS, MAX_M = 4096, 10, 1e-6, 15

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield rank
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


@pytest.fixture(scope="module")
def pair(rank):
    """(SGLang's fusion workspace, the hop) on the TP group, built once: each one
    rendezvouses its own multicast region."""
    from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
        FlashInferMNNVLCuteDSLARFusion,
    )
    from sglang.srt.runtime_context import get_spec

    from sgl_nvidia_mega.qwen3_5.sm103.hop import Hop

    group = ps.get_tp_group().device_group
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # speculative decoding keeps the finalize on the ordering the verify server runs
    with get_spec().override(speculative_algorithm="EAGLE"):
        fusion = FlashInferMNNVLCuteDSLARFusion(
            hidden_size=HIDDEN,
            top_k=TOP_K,
            max_m=MAX_M + 1,
            rms_epsilon=EPS,
            weight_bias=0.0,
            process_group=group,
            device=device,
        )
    hop = Hop(max_m=MAX_M, eps=EPS, group=group, device=device, deadline_s=30.0)
    return fusion, hop


def _operands(m, device, rank, seed):
    """One hop's inputs: this rank's partial and MoE rows, and the residual and the
    norm multiplier every rank holds alike."""
    own = torch.Generator(device=device).manual_seed(seed * 100 + rank)
    common = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, gen=own):
        return torch.randn(*shape, device=device, generator=gen).bfloat16()

    rows = m * TOP_K
    return dict(
        partial=randn(m, HIDDEN),
        # gemm2's output, with rows past the routed ones as the runner pads it
        routed=randn(rows + 5, HIDDEN),
        expert_weights=torch.rand(m, TOP_K, device=device, generator=own).bfloat16(),
        permuted=torch.randperm(rows, device=device, generator=own)
        .to(torch.int32)
        .view(m, TOP_K),
        shared=randn(m, HIDDEN),
        residual=randn(m, HIDDEN, gen=common),
        gamma=(1 + 0.1 * randn(HIDDEN, gen=common).float()).bfloat16(),
    )


def _run(engine, op, x, **kw):
    if op == "all_reduce":
        return engine.all_reduce_residual_rms_norm(
            local_contribution=x["partial"],
            residual=x["residual"],
            gamma=x["gamma"],
            **kw,
        )
    return engine.moe_finalize_all_reduce_rms_norm(
        routed_output=x["routed"],
        expert_weights=x["expert_weights"],
        permuted_indices=x["permuted"],
        gated_shared_output=x["shared"],
        residual=x["residual"],
        gamma=x["gamma"],
        **kw,
    )


def _same(got, want):
    for a, b in zip(got, want):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_hop_matches_flashinfer_fusion(rank, pair):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    fusion, hop = pair
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    scale = torch.tensor([0.05], device=device)
    calls = [(op, m) for m in (1, 2, 3, 7, 8, 15) for op in ("all_reduce", "finalize")]
    with torch.inference_mode():
        for i, (op, m) in enumerate(calls):
            x = _operands(m, device, rank, seed=i)
            want = _run(fusion, op, x)
            if i % (world + 1) == rank:
                torch.cuda._sleep(2_000_000)  # this rank publishes late
            quantize = i % 3 == 0
            got = _run(hop, op, x, fp8_scale=scale if quantize else None)
            _same(got[:2], want)
            if quantize:
                fp8, _ = static_quant_fp8(want[0], scale, repeat_scale=False)
                assert torch.equal(got[2].view(torch.uint8), fp8.view(torch.uint8))
        torch.cuda.synchronize()


def test_hop_replays_from_a_cuda_graph(rank, pair):
    fusion, hop = pair
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    m = 7
    # four hops: one more than the mailbox has stages
    ops = ("all_reduce", "finalize", "all_reduce", "finalize")
    with torch.inference_mode():
        static = [_operands(m, device, rank, seed=50 + i) for i in range(len(ops))]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for op, x in zip(ops, static):
                _run(hop, op, x)  # eager warmup on the capture stream
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [_run(hop, op, x) for op, x in zip(ops, static)]
        for replay in range(3):
            for i, x in enumerate(static):
                fresh = _operands(m, device, rank, seed=100 + 10 * replay + i)
                for name, tensor in x.items():
                    tensor.copy_(fresh[name])
            graph.replay()
            for op, x, got in zip(ops, static, outputs):
                _same(got, _run(fusion, op, x))
        torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
