# SPDX-License-Identifier: Apache-2.0
"""The fused norm's BMM boundary: byte codes, static scale and output dtype."""

import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.gdn import verify


def module(monkeypatch, name, **fields):
    obj = ModuleType(name)
    obj.__dict__.update(fields)
    monkeypatch.setitem(sys.modules, name, obj)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 128)])
def test_prequantized_codes_scales_and_output_dtype(monkeypatch, dtype, shape):
    x = torch.zeros(shape, dtype=torch.float8_e4m3fn)
    weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
    sx, sw = torch.tensor([0.5]), torch.tensor([0.25])
    seen = []
    out = torch.empty((x.numel() // shape[-1], 64), dtype=dtype)

    def bmm(q, w, a, b, d):
        seen.append((q, w, a, b, d))
        return out

    module(
        monkeypatch, "sglang.srt.layers.quantization.fp8_utils", flashinfer_bmm_fp8=bmm
    )
    module(
        monkeypatch,
        "sglang.srt.distributed.device_communicators.pynccl_allocator",
        use_symmetric_memory=lambda *a, **kw: contextlib.nullcontext(),
    )
    module(
        monkeypatch,
        "sglang.srt.layers.dp_attention",
        is_allocation_symmetric=lambda: True,
    )
    module(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_parallel=lambda: SimpleNamespace(tp_group="tp"),
    )
    # _out_proj receives flattened rows from step(); no re-quantization or bias.
    flat = x.view(-1, x.shape[-1])
    layer = SimpleNamespace(
        weight=weight, input_scale=sx, weight_scale=sw, tp_group="tp", orig_dtype=dtype
    )
    assert verify._out_proj(layer, flat, dtype) is out
    q, w, a, b, d = seen[0]
    assert q is flat and w is weight and a is sx and b is sw and d == dtype


@pytest.mark.parametrize(
    "input_dtype,out_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float32),
    ],
)
def test_invalid_prequantized_contract_fails_before_bmm(input_dtype, out_dtype):
    with pytest.raises(ValueError):
        verify._out_proj(
            SimpleNamespace(orig_dtype=out_dtype),
            torch.zeros(1, 16, dtype=input_dtype),
            out_dtype,
        )


def test_bias_and_static_scalar_scale_remain_eligibility_constraints(monkeypatch):
    class Method:
        use_marlin = False
        use_sm120_fp8 = False

    module(
        monkeypatch,
        "sglang.srt.layers.quantization.modelopt_quant",
        ModelOptFp8LinearMethod=Method,
    )
    out = SimpleNamespace(
        quant_method=Method(),
        use_flashinfer_bmm=True,
        input_scale=torch.ones(1),
        reduce_results=False,
        use_dp_attention_reduce=False,
        use_decode_attn_tp=False,
        input_is_parallel=True,
        bias=None,
    )
    gdn = SimpleNamespace(
        attn=SimpleNamespace(head_k_dim=128, head_v_dim=128),
        norm=SimpleNamespace(group_size=None, norm_before_gate=True),
        out_proj=out,
    )
    assert verify.unsupported(gdn) == ""
    out.bias = torch.zeros(64)
    assert "adds a bias" in verify.unsupported(gdn)
    out.bias = None
    out.input_scale = torch.ones(2)
    assert "static per-tensor" in verify.unsupported(gdn)
