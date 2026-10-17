# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed step's SGLang-facing glue on CPU: the two-shot posts hand SGLang the
next sublayer's input that SGLang's own post would build."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import forward

mhc = pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")
flashinfer_trtllm = pytest.importorskip(
    "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm"
)
Deferred = flashinfer_trtllm.FlashInferTrtllmDeferredFinalizeOutput
if "fused_post" not in Deferred.__dataclass_fields__:
    pytest.skip(
        "this SGLang lacks the deferred MoE output's fused post",
        allow_module_level=True,
    )

ROWS, HIDDEN = 64, 8


def _moe_output(routed):
    from sglang.srt.models.deepseek_v2 import MoEOutput

    return MoEOutput(routed, torch.ones(ROWS, HIDDEN), None, 1.0, False)


def _hop():
    updated, normalized = torch.zeros(ROWS, 2, HIDDEN), torch.ones(ROWS, HIDDEN)
    return SimpleNamespace(
        moe_post_only=Mock(return_value=updated),
        moe_post=Mock(return_value=(updated, normalized)),
        scale_bytes=Mock(return_value=96),
    )


def test_moe_post_builds_the_next_input_sglang_would():
    """Closed seam or unfusable norm: streams only; fusable norm: normed; MXFP8
    receiver: quantized with SGLang's swizzled marker, so it skips its own quant."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    hop = _hop()
    posts = forward.TwoShotPosts(hop)
    routed = Deferred(
        torch.ones(ROWS * 6, HIDDEN), torch.ones(ROWS, 6), torch.zeros(ROWS, 6), 6
    )
    out = posts.attach_moe_post(_moe_output(routed))
    post = out.routed.fused_post
    assert post is not None and out.routed.gemm2_out is routed.gemm2_out
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    args = (routed.gemm2_out, routed.expanded_idx_to_permuted_idx)
    args += (routed.expert_weights, 6, out.shared, torch.zeros(ROWS, 2, HIDDEN))
    args += (torch.ones(ROWS, 2), torch.ones(ROWS, 2, 2), torch.ones(ROWS, 2))

    for nxt in (None, SimpleNamespace(norm=norm, norm_fusable=False)):
        _, given = post(*args, nxt)
        assert given is None
    assert hop.moe_post_only.call_count == 2 and not hop.moe_post.called

    nxt = SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=False)
    _, given = post(*args, nxt)
    assert isinstance(given, mhc.HcNormed)
    assert hop.moe_post.call_args.args[-3:] == (args[-1], norm.weight, 1e-6)
    assert "mxfp8_out" not in hop.moe_post.call_args.kwargs

    nxt = SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=True)
    _, given = post(*args, nxt)
    assert isinstance(given, mhc.HcQuantized)
    assert isinstance(given.swizzled, Mxfp8SwizzledInput)
    q, sf = hop.moe_post.call_args.kwargs["mxfp8_out"]
    assert given.swizzled.data is q and given.swizzled.scales is sf
    assert q.shape == (ROWS, HIDDEN) and q.dtype == torch.float8_e4m3fn
    assert sf.numel() == 96 and sf.dtype == torch.uint8
    hop.scale_bytes.assert_called_with(ROWS)


def test_only_a_deferred_finalize_takes_the_moe_post():
    posts = forward.TwoShotPosts(_hop())
    plain = torch.ones(ROWS, HIDDEN)
    assert posts.attach_moe_post(plain) is plain
    merged = _moe_output(plain)
    assert posts.attach_moe_post(merged) is merged
