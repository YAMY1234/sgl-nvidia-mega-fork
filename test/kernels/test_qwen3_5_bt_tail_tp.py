# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

HIDDEN, TOP_K, EPS, MAX_M = 4096, 10, 1e-6, 703

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
def fusion(rank):
    """SGLang's fusion workspace on the TP group, as the verify server builds it."""
    from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
        FlashInferMNNVLCuteDSLARFusion,
    )
    from sglang.srt.runtime_context import get_spec

    # speculative decoding keeps the finalize on the ordering the verify server runs
    with get_spec().override(speculative_algorithm="EAGLE"):
        return FlashInferMNNVLCuteDSLARFusion(
            hidden_size=HIDDEN,
            top_k=TOP_K,
            max_m=MAX_M,
            rms_epsilon=EPS,
            weight_bias=0.0,
            process_group=ps.get_tp_group().device_group,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )


def _tail(fusion, early=()):
    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.boundary.bt_tail import BTTail

    # the model's CuteDSLWorkspace once SGLang prepared it
    service = SimpleNamespace(
        _workspace=fusion, hidden_size=HIDDEN, top_k=TOP_K, rms_epsilon=EPS
    )
    tail = BTTail(
        service=service,
        eps=EPS,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        min_rows=plan.BT_TAIL_MIN_ROWS,
        max_rows=plan.BT_TAIL_MAX_ROWS,
        early=frozenset(early),
    )
    assert tail.bind() == []
    return tail


def _operands(m, device, rank, seed):
    own = torch.Generator(device=device).manual_seed(seed * 100 + rank)
    common = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, gen=own):
        return torch.randn(*shape, device=device, generator=gen).bfloat16()

    rows = m * TOP_K
    return dict(
        partial=randn(m, HIDDEN),
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


def test_bt_tail_serves_the_widths_of_both_bt_routes(rank, fusion):
    from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

    from sgl_nvidia_mega.qwen3_5 import plan

    tail = _tail(fusion)
    profile = fusion.workspace.profile
    served = []
    for m in range(1, MAX_M + 1):
        ar = profile.all_reduce_routes.select(m)
        bt = (
            profile.finalize_routes.select(m).protocol is ProtocolKind.BT
            and ar.protocol is ProtocolKind.BT
            and not ar.preset.fused  # FlashInfer's one-launch BT all-reduce
        )
        want = bt and plan.BT_TAIL_MIN_ROWS <= m <= plan.BT_TAIL_MAX_ROWS
        assert tail.supports(m) == want, m
        if want:
            served.append(m)
    assert served[0] <= 29 and served[-1] == MAX_M
    assert not tail.supports(MAX_M + 1)


def test_bt_tail_matches_flashinfer_fusion(rank, fusion):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    tails = (_tail(fusion), _tail(fusion, early=("finalize", "all_reduce")))
    world = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    scale = torch.tensor([0.05], device=device)
    widths = (24, 28, 29, 35, 42, 48, 49, 56, 84, 112, 224, 448, MAX_M)
    calls = [(op, m) for m in widths for op in ("all_reduce", "finalize")]
    with torch.inference_mode():
        for i, (op, m) in enumerate(calls):
            x = _operands(m, device, rank, seed=i)
            want = _run(fusion, op, x)
            for j, tail in enumerate(tails):
                if (i + j) % (world + 1) == rank:
                    torch.cuda._sleep(2_000_000)  # this rank publishes late
                quantize = (i + j) % 2 == 0
                got = _run(tail, op, x, fp8_scale=scale if quantize else None)
                _same(got[:2], want)
                if quantize:
                    fp8, _ = static_quant_fp8(want[0], scale, repeat_scale=False)
                    assert torch.equal(got[2].view(torch.uint8), fp8.view(torch.uint8))
                else:
                    assert len(got) == 2
                # the workspace's own call right after, on the stage the tail left
                _same(_run(fusion, op, x), want)
        torch.cuda.synchronize()


def test_bt_tail_replays_from_a_cuda_graph(rank, fusion):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    tail = _tail(fusion, early=("all_reduce",))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    scale = torch.tensor([0.05], device=device)
    # five points, more than the protocol has stages, the tail's and the workspace's
    points = (
        (tail, "finalize", 56),
        (tail, "all_reduce", 56),
        (fusion, "finalize", 84),
        (tail, "all_reduce", 84),
        (tail, "finalize", 42),
    )
    kw = {id(tail): {"fp8_scale": scale}, id(fusion): {}}
    with torch.inference_mode():
        static = [
            _operands(m, device, rank, seed=50 + i)
            for i, (_, _, m) in enumerate(points)
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for (engine, op, _), x in zip(points, static):
                _run(
                    engine, op, x, **kw[id(engine)]
                )  # eager warmup on the capture stream
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [
                _run(engine, op, x, **kw[id(engine)])
                for (engine, op, _), x in zip(points, static)
            ]
        for replay in range(3):
            for i, x in enumerate(static):
                fresh = _operands(
                    points[i][2], device, rank, seed=100 + 10 * replay + i
                )
                for name, tensor in x.items():
                    tensor.copy_(fresh[name])
            graph.replay()
            for (engine, op, _), x, got in zip(points, static, outputs):
                want = _run(fusion, op, x)
                _same(got[:2], want)
                if engine is tail:
                    fp8, _ = static_quant_fp8(want[0], scale, repeat_scale=False)
                    assert torch.equal(got[2].view(torch.uint8), fp8.view(torch.uint8))
        torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
