# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec

from sgl_nvidia_mega.qwen3_5.sm103.head import mma
from sgl_nvidia_mega.qwen3_5.sm103.head import verify_head as vh

if "VerifyShardHeadView" not in vars(spec):
    pytest.skip("needs SGLang's verify-shard head seam", allow_module_level=True)
VerifyShardHeadView = spec.VerifyShardHeadView

ROWS, H = 256, vh.HIDDEN


@pytest.fixture
def launches(monkeypatch):
    """No compile; each launch writes the bf16 GEMM into ``logits`` and is recorded."""
    calls = []

    def head_mma_keys(x, w, keys, **kwargs):
        calls.append(kwargs)
        kwargs["logits"].copy_(torch.matmul(x, w.T))
        if kwargs["pre"] is not None:
            kwargs["epoch"][0] += 1
        return 1

    monkeypatch.setattr(mma, "_kernel", lambda *args, **kwargs: None)
    monkeypatch.setattr(mma, "head_mma_keys", head_mma_keys)
    monkeypatch.setattr(vh, "_STASH", {})
    return calls


def _weight(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(ROWS, H, generator=g) * 0.02).to(torch.bfloat16)


def _hidden(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, H, generator=g).to(torch.bfloat16)


def _view(hidden, weight, out):
    return VerifyShardHeadView(hidden_states=hidden, weight=weight, out=out)


def test_the_head_writes_the_borrowed_buffer_or_its_own(launches):
    w = _weight()
    head = vh.VerifyHeadMma(w, vocab_start=3 * ROWS, valid=ROWS)
    x = _hidden(vh.MIN_ROWS)
    out = torch.empty(vh.MIN_ROWS, ROWS, dtype=torch.bfloat16)
    assert head(_view(x, w, out)) is out
    assert torch.equal(out, torch.matmul(x, w.T))
    assert launches[-1]["vocab_start"] == 3 * ROWS and launches[-1]["pre"] is None
    own = head(_view(x, w, None))
    assert own.dtype == torch.bfloat16 and tuple(own.shape) == (vh.MIN_ROWS, ROWS)
    # no keys without slot verify_head_keys
    assert vh.lookup(out) is None


@pytest.mark.parametrize(
    "case",
    [
        "rows_low",
        "rows_high",
        "weight",
        "out_dtype",
        "out_shape",
        "out_layout",
        "hidden",
    ],
)
def test_every_decline_leaves_the_buffer_to_the_gemm(launches, case):
    w = _weight()
    head = vh.VerifyHeadMma(w, vocab_start=0, valid=ROWS)
    m = vh.MAX_ROWS + 1 if case == "rows_high" else vh.MIN_ROWS
    m = vh.MIN_ROWS - 1 if case == "rows_low" else m
    x = _hidden(m)
    out = torch.zeros(m, ROWS, dtype=torch.bfloat16)
    view_w = w.clone() if case == "weight" else w
    if case == "out_dtype":
        out = out.float()
    elif case == "out_shape":
        out = out[:, :-8]
    elif case == "out_layout":
        out = torch.zeros(ROWS, m, dtype=torch.bfloat16).T
    elif case == "hidden":
        x = x.float()
    assert head(_view(x, view_w, out)) is None
    assert launches == [] and int(out.abs().sum()) == 0


def test_seam_accepts_the_heads_logits(launches):
    """SGLang's seam checks a provider head's result (shape, dtype, device, buffer)."""
    from sglang.srt.layers.logits_processor import _provider_shard_logits

    w = _weight()
    head = vh.VerifyHeadMma(w, vocab_start=0, valid=ROWS)
    x = _hidden(vh.MAX_ROWS)
    out = torch.empty(vh.MAX_ROWS, ROWS, dtype=torch.bfloat16)
    got = _provider_shard_logits(head, hidden_states=x, weight=w, out=out)
    assert got is out and torch.equal(got, torch.matmul(x, w.T))
    assert _provider_shard_logits(head, hidden_states=x[:8], weight=w, out=None) is None


def test_the_keys_build_stashes_keys_per_buffer_until_a_decline(launches):
    w = _weight()
    head = vh.VerifyHeadMma(w, vocab_start=0, valid=ROWS, keys=True)
    m = vh.MIN_ROWS
    x = _hidden(m)
    out = torch.empty(m, ROWS, dtype=torch.bfloat16)
    assert head(_view(x, w, out)) is out
    pre, ctas, epoch = vh.lookup(out)
    assert ctas == -(-ROWS // vh.BOX_ROWS)
    assert pre.dtype == torch.int64 and pre.numel() == 2 * m * ctas + 1
    assert launches[-1]["pre"] is pre and launches[-1]["epoch"] is epoch
    assert head(_view(x, w, out)) is out and vh.lookup(out)[0] is pre
    assert vh.lookup(head(_view(x, w, None))) is None
    before = int(epoch[0])
    assert head(_view(x.float(), w, out)) is None
    assert vh.lookup(out) is None and int(epoch[0]) == before + 1


def _head(**changes):
    weight = changes.pop("weight", _weight())
    head = SimpleNamespace(
        weight=weight,
        quant_method=None,
        bias=None,
        shard_indices=SimpleNamespace(
            num_added_elements_padded=0, num_org_elements=int(weight.shape[0])
        ),
    )
    for name, value in changes.items():
        setattr(head, name, value)
    return head


class UnquantizedEmbeddingMethod:
    pass


class Fp8LinearMethod:
    pass


def _processor(**changes):
    return SimpleNamespace(
        **{
            "logit_scale": None,
            "use_fp32_lm_head": False,
            "use_fp32_lm_head_output": False,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "head, processor, why",
    [
        (_head(), _processor(), ""),
        (_head(quant_method=UnquantizedEmbeddingMethod()), _processor(), ""),
        (_head(set_lora=None), _processor(), "a LoRA LM head"),
        (_head(quant_method=Fp8LinearMethod()), _processor(), "quantized"),
        (_head(weight_scale=torch.ones(1)), _processor(), "with weight_scale"),
        (_head(bias=torch.zeros(ROWS)), _processor(), "bias"),
        (_head(weight=_weight().float()), _processor(), "not bf16"),
        (_head(weight=_weight().T.contiguous().T), _processor(), "contiguous"),
        (
            _head(
                shard_indices=SimpleNamespace(
                    num_added_elements_padded=8, num_org_elements=ROWS
                )
            ),
            _processor(),
            "padded",
        ),
        (_head(), _processor(logit_scale=0.5), "logit scale"),
        (_head(), _processor(use_fp32_lm_head=True), "fp32"),
    ],
)
def test_only_a_plain_bf16_shard_is_served(monkeypatch, head, processor, why):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    got = vh.head_refusal(processor, head)
    assert (got == "") if why == "" else (why in got), got
