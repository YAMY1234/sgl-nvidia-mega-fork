# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Engram prefetch's WKV on one GB300: each TP4 rank's quarter of the output
columns (views of the weight and of its 128-row swizzled scale tiles), zero-padded and
summed as the plane's all-reduce sums them, equals SGLang's whole WKV GEMM bit for
bit at every row count the prefetch serves. The serving build repeats the check on
the loaded model, with the real all-reduces."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

TP, N, K, HASH_COLS = 4, 25600, 6144, 12  # DeepSeek-V4.1-Flash's WKV


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")


class _Plane:
    """One rank's view of the plane: its own all-reduce input, kept for the sum."""

    def __init__(self):
        self.seen = []

    def custom_all_reduce(self, x):
        self.seen.append(x)
        return x


@pytest.fixture(scope="module")
def wkv():
    from flashinfer import block_scale_interleave

    gen = torch.Generator(device="cuda").manual_seed(3)
    weight = (torch.randn((N, K), device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(118, 126, (N, K // 32), device="cuda", generator=gen)
    swizzled = block_scale_interleave(scales.to(torch.uint8).contiguous())
    return SimpleNamespace(
        weight=weight, weight_scale_inv_swizzled=swizzled.contiguous()
    )


@pytest.mark.parametrize("rows", [1, 6, 12, 16])
def test_the_column_quarters_sum_to_sglangs_wkv(wkv, rows):
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    gen = torch.Generator(device="cuda").manual_seed(rows)
    emb = torch.randn((rows, HASH_COLS, K // HASH_COLS), device="cuda", generator=gen)
    emb = emb.to(torch.bfloat16)
    want = flashinfer_mxfp8_blockscaled_linear(
        input=emb.flatten(-2),
        weight=wkv.weight,
        weight_scale=wkv.weight_scale_inv_swizzled,
        backend="cute-dsl",
        pin_tactic=True,
    )
    engram = SimpleNamespace(
        wkv=wkv, embed=SimpleNamespace(_owned_rows=lambda ids: emb)
    )
    total = torch.zeros((rows, N), dtype=torch.float32, device="cuda")
    for rank in range(TP):
        layer = ep._Layer(engram, rank, TP)
        assert (
            layer.weight.data_ptr() == wkv.weight[rank * N // TP].data_ptr()
        )  # a view
        plane = _Plane()
        layer.kv(None, plane)
        lookup, gathered = plane.seen
        assert lookup is emb
        quarter = slice(rank * N // TP, (rank + 1) * N // TP)
        assert (
            not gathered[:, : quarter.start].any()
            and not gathered[:, quarter.stop :].any()
        )
        total += gathered.float()
    torch.cuda.synchronize()
    assert torch.equal(total.bfloat16(), want)
