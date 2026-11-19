# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB_SHARD, HIDDEN = 31040, 4096


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import HeadLowM

    if not dist.is_initialized():
        store = tmp_path_factory.mktemp("pg") / "store"
        dist.init_process_group(
            "gloo", init_method=f"file://{store}", rank=0, world_size=1
        )
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = (
        0.02 * torch.randn(VOCAB_SHARD, HIDDEN, generator=g, device="cuda")
    ).bfloat16()
    return HeadLowM(weight, tuple(range(1, 9)), cpu_group=None, rank=0)


def _within_one_rounding(got, x, weight):
    want = x.double() @ weight.double().t()
    err = (got.double() - want).abs()
    return bool((err <= want.abs() * 2**-8 + 1e-3).all())


@cuda
@pytest.mark.parametrize("rows", range(1, 9))
def test_each_taken_width_is_within_one_rounding_of_fp64(head, rows):
    if not head.takes(rows):
        pytest.skip(f"the race kept cuBLAS at {rows} rows")
    x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
    assert _within_one_rounding(head(x), x, head.weight)


@cuda
def test_a_taken_width_replays_in_a_cuda_graph(head):
    rows = next((m for m in range(1, 9) if head.takes(m)), None)
    if rows is None:
        pytest.skip("the race kept cuBLAS at every width")
    x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
    out = torch.empty(rows, VOCAB_SHARD, dtype=torch.bfloat16, device="cuda")
    head(x, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head(x, out)
    for _ in range(3):
        x.copy_(torch.randn(rows, HIDDEN, device="cuda").bfloat16())
        graph.replay()
        assert _within_one_rounding(out, x, head.weight)
