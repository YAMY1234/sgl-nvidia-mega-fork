# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fused inverse RoPE + wo_a + MXFP8 kernel (src/wo_a_fused.cu) on one SM100-family
GPU (GB300), against SGLang's JIT build of the same device code,
``fused_rope_wo_a_bf16(..., out_mxfp8=True)``, bit for bit (codes and scales) at every
row count SGLang serves (1-32 on main, 1-48 where it has the 48-row tile), for int32 and
int64 positions, with the attention output's 64-head row stride and a packed one; eager
and replayed from a CUDA graph.
"""

import inspect

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a

SEEDS = (0, 1, 2)
MAX_POSITION = 4096
HEADS = 64  # the attention output x views: [T, 64, 512]
HEAD_DIM = 512


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _reference():
    """SGLang's wrapper and the most rows it serves; skips without SGLang."""
    module = pytest.importorskip("sglang.kernels.ops.attention.dsv4.wo_a")
    # 48 rows exist only where SGLang has the shared-inbox tile.
    return module, getattr(module, "WIDE_MAX_M", module.MAX_M)


def _inputs(rows: int, seed: int, int64_positions: bool, padded_heads: bool):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if padded_heads:
        attention = torch.randn(
            (rows, HEADS, HEAD_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        x = attention.view(rows, HEADS * HEAD_DIM // wo_a.GROUP_K, wo_a.GROUP_K)[
            :, : wo_a.GROUPS
        ]
    else:
        x = torch.randn(
            (rows, wo_a.GROUPS, wo_a.GROUP_K),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
    if seed == 1:
        x[0].zero_()  # an all-zero row takes the zero-scale branch
    weight = (
        torch.randn(
            (wo_a.GROUPS, wo_a.RANK, wo_a.GROUP_K),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    angles = (
        torch.rand(
            (MAX_POSITION, wo_a.ROPE_DIM // 2),
            generator=generator,
            device="cuda",
        )
        * 6.2831855
    )
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).reshape(
        MAX_POSITION, wo_a.ROPE_DIM
    )
    positions = torch.randint(
        0, MAX_POSITION, (rows,), generator=generator, device="cuda"
    ).to(torch.int64 if int64_positions else torch.int32)
    return x, weight, freqs, positions


def _assert_bitwise(actual, expected) -> None:
    codes, scales = actual
    expected_codes, expected_scales = expected
    assert codes.shape == expected_codes.shape and scales.shape == expected_scales.shape
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.parametrize("padded_heads", [True, False])
@pytest.mark.parametrize("int64_positions", [False, True])
@pytest.mark.parametrize("rows", range(1, wo_a.MAX_ROWS + 1))
def test_wo_a_fused_matches_sglang(rows, int64_positions, padded_heads):
    reference, reference_max_rows = _reference()
    if rows > reference_max_rows:
        pytest.skip(f"SGLang serves at most {reference_max_rows} rows")
    parameters = inspect.signature(reference.fused_rope_wo_a_bf16).parameters
    for seed in SEEDS:
        x, weight, freqs, positions = _inputs(rows, seed, int64_positions, padded_heads)
        expected = reference.fused_rope_wo_a_bf16(x, weight, freqs, positions)
        for early_trigger in (True, False):
            _assert_bitwise(
                wo_a.fused_rope_wo_a(
                    x, weight, freqs, positions, early_trigger=early_trigger
                ),
                expected,
            )
        if "row_tile_scales" in parameters:
            _assert_bitwise(
                wo_a.fused_rope_wo_a(x, weight, freqs, positions, row_tile_scales=True),
                reference.fused_rope_wo_a_bf16(
                    x, weight, freqs, positions, row_tile_scales=True
                ),
            )
    torch.cuda.synchronize()


@pytest.mark.parametrize("rows", [1, 5, 8, 16, 17, 32, 33, 48])
def test_wo_a_fused_graph_replay_matches_eager(rows):
    wo_a.precompile(rows)
    x, weight, freqs, positions = _inputs(
        rows, seed=3, int64_positions=False, padded_heads=True
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = wo_a.fused_rope_wo_a(x, weight, freqs, positions)
    torch.cuda.current_stream().wait_stream(stream)
    # New inputs in the captured buffers: the replay must read them, not the capture's.
    for seed in (4, 5):
        fresh = _inputs(rows, seed=seed, int64_positions=False, padded_heads=True)
        for buffer, values in zip((x, weight, freqs, positions), fresh):
            buffer.copy_(values)
        graph.replay()
        expected = wo_a.fused_rope_wo_a(x, weight, freqs, positions)
        _assert_bitwise(captured, expected)
    torch.cuda.synchronize()
