# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft's fused wqkv_a slot on CPU: the fused linear is a clone holding the
concatenated weight and scales, the split matches the two linears, the originals are
untouched, and each refusal names its reason."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import draft_layers

CUTEDSL = SimpleNamespace(
    is_flashinfer_cutedsl=lambda: True, is_flashinfer_cutlass=lambda: False
)


class _Method:
    mxfp8_dense_backend = CUTEDSL

    def apply(self, layer, x, bias=None):
        if isinstance(x, tuple):
            x = x[0]
        return x.float() @ layer.weight.float().t()


class _Linear(torch.nn.Module):
    def __init__(self, n, k, seed, method=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n, k, generator=g).to(torch.float8_e4m3fn)
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.weight_scale_inv_swizzled = torch.full(
            (n * k // 32,), seed, dtype=torch.uint8
        )
        self.quant_method = method or _Method()
        self.bias = None
        self.output_size_per_partition = n


def test_the_fused_gemm_splits_into_the_two_linears():
    wq_a, wkv = _Linear(256, 64, 1), _Linear(128, 64, 2)
    original = wq_a.weight
    op = draft_layers.FusedQkvA(wq_a, wkv, unified_kv=False)
    x = torch.randn(5, 64)
    q_lora, kv = op(x, None)
    torch.testing.assert_close(q_lora, wq_a.quant_method.apply(wq_a, x), rtol=0, atol=0)
    torch.testing.assert_close(kv, wkv.quant_method.apply(wkv, x), rtol=0, atol=0)
    assert wq_a.weight is original and wq_a.weight.shape == (256, 64)
    assert (
        op.fused.weight.shape == (384, 64) and op.fused.output_size_per_partition == 384
    )
    scales = op.fused.weight_scale_inv_swizzled
    assert torch.equal(scales[: 256 * 2], torch.full((512,), 1, dtype=torch.uint8))
    assert torch.equal(scales[256 * 2 :], torch.full((256,), 2, dtype=torch.uint8))
    # the post's MXFP8 rows replace x when given
    q2, _ = op(torch.zeros(5, 64), (x, None))
    torch.testing.assert_close(q2, q_lora, rtol=0, atol=0)
    _, kv_u = draft_layers.FusedQkvA(wq_a, wkv, unified_kv=True)(x, None)
    assert kv_u.is_contiguous() and not kv.is_contiguous()


@pytest.mark.parametrize(
    "wq_a, wkv, want",
    [
        (_Linear(200, 64, 1), _Linear(128, 64, 2), "does not tile by 128"),
        (_Linear(256, 64, 1), _Linear(128, 32, 2), "read different inputs"),
        (
            _Linear(256, 64, 1, SimpleNamespace(mxfp8_dense_backend=None)),
            _Linear(128, 64, 2, SimpleNamespace(mxfp8_dense_backend=None)),
            "takes no pre-quantized input",
        ),
    ],
)
def test_each_refusal_names_its_reason(wq_a, wkv, want):
    assert want in draft_layers._fusable(wq_a, wkv)


def test_install_sets_every_stage_or_none():
    from sglang.srt.models.deepseek_v4_dspark import DSparkAttention

    if not hasattr(DSparkAttention, "qkv_a_op"):
        pytest.skip("this SGLang lacks DSparkAttention.qkv_a_op")
    good = SimpleNamespace(
        self_attn=SimpleNamespace(wq_a=_Linear(256, 64, 1), wkv=_Linear(128, 64, 2))
    )
    bad = SimpleNamespace(
        self_attn=SimpleNamespace(wq_a=_Linear(200, 64, 1), wkv=_Linear(128, 64, 2))
    )
    assert "stage 1" in draft_layers.install_qkv_a([good, bad])
    assert not hasattr(good.self_attn, "qkv_a_op")
    assert draft_layers.install_qkv_a([good]) == ""
    assert isinstance(good.self_attn.qkv_a_op, draft_layers.FusedQkvA)
