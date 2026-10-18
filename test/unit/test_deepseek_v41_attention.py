# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The stock attention step on CPU: the prepare path it picks per layer and step,
and wo_b's hop-A publish."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang.srt.layers.attention.deepseek_v4_backend")

from sglang.srt.layers.quantization.fp8_utils import (  # noqa: E402
    Mxfp8DenseGemmBackend,
)
from sglang.srt.models import deepseek_v4_mhc as mhc  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.attention import ops, prepare  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.attention.step import AttentionStep  # noqa: E402


def _mode(*, decode=False, verify=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_target_verify=lambda: verify
        )
    )


def _attn(
    ratio, *, streams=True, wq_b_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
):
    return SimpleNamespace(
        compress_ratio=ratio,
        alt_streams=[object()] * 3 if streams else None,
        _multi_stream_bs_limit=128,
        wq_b=SimpleNamespace(
            quant_method=SimpleNamespace(mxfp8_dense_backend=wq_b_backend)
        ),
    )


@pytest.mark.parametrize(
    "ratio, streams, capture, mode, backend_, want",
    [
        (2, True, True, _mode(verify=True), None, prepare.low_ratio),
        (1, True, False, _mode(decode=True), None, prepare.low_ratio),
        (1, True, False, _mode(verify=True), "other", prepare.eager),
        (0, True, True, _mode(verify=True), None, prepare.multi_stream),
        (0, True, False, _mode(verify=True), None, prepare.eager),
        (2, False, True, _mode(verify=True), None, prepare.eager),
    ],
)
def test_prepare_path_is_sglangs_choice(ratio, streams, capture, mode, backend_, want):
    """The step must take the prepare SGLang's MQALayer takes, or it stops being bitwise."""
    kwargs = {} if backend_ is None else {"wq_b_backend": backend_}
    step = AttentionStep.__new__(AttentionStep)
    step.attn, step.low_ratio = _attn(ratio, streams=streams, **kwargs), ratio in (1, 2)
    with mock.patch(
        "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=capture
    ):
        assert step._prepare_path(6, mode) is want


def test_wo_b_publishes_at_hop_a_widths_and_falls_back_otherwise():
    """A published partial skips SGLang's wo_b; a GEMM that declines leaves it to SGLang."""
    o = torch.randn(6, 2, 4096)
    stock = mock.Mock(return_value=(torch.ones(6, 8), None))
    attn = SimpleNamespace(wo_b=stock, attn_tp_size=4)
    published = torch.zeros(6, 8)
    publish = mock.Mock(return_value=published)
    with mock.patch(
        "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce", return_value=False
    ):
        out = ops.wo_b(attn, o, publish)
        assert isinstance(out, mhc.AttnOutput) and out[0] is published
        stock.assert_not_called()
        assert publish.call_args.args[1].shape == (6, 2 * 4096)
        publish.return_value = None
        out = ops.wo_b(attn, o, publish)
    stock.assert_called_once()
    assert stock.call_args.kwargs == {"skip_all_reduce": True}
