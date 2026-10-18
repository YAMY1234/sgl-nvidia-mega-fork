# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 spec through SGLang's startup checks on CPU: the documented
setup passes, and every other setup is refused with each reason named."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.megakernels import guards
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG, plan, spec

DOCUMENTED = dict(
    model_path="dummy",
    tp_size=4,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="DSPARK",
    moe_a2a_backend="none",
    dsv4_attn_backend="flashmla",
    cuda_graph_config=default_cuda_graph_config(),
)


def _reasons(hf=None, **args):
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        **{**HF_CONFIG, "model_type": "deepseek_v41", **(hf or {})},
    )
    with (
        patch.object(guards, "find_spec", return_value=(spec.SPEC, "")),
        patch.object(spec, "flashinfer_ll_gemm_digest", return_value=None),
        patch("importlib.util.find_spec", return_value=object()),
    ):
        _, reasons = guards.megakernel_disable_reasons(
            cfg=ServerArgs(**{**DOCUMENTED, **args}), hf_config=hf_config, isa="sm_103"
        )
    return reasons


def test_the_documented_setup_is_accepted():
    assert _reasons() == []


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8"),
        (dict(pp_size=2), "pipeline parallelism is in use"),
        (dict(kv_cache_dtype="auto"), "kv_cache_dtype 'auto'"),
        (dict(speculative_algorithm="EAGLE"), "speculative_algorithm 'EAGLE'"),
        (dict(dsv4_attn_backend="flashmla_mega"), "dsv4_attn_backend 'flashmla_mega'"),
        (dict(moe_a2a_backend="deepep"), "moe_a2a_backend 'deepep'"),
        (dict(enable_encoder_swa_bounded_replay=True), "enable_encoder_swa_bounded"),
        (dict(hf=dict(hidden_size=7168)), "config.hidden_size 7168"),
        (dict(hf=dict(model_type="deepseek_v4")), "config.model_type"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    reasons = _reasons(**kw)
    assert any(want in r for r in reasons), (want, reasons)


def test_the_toolchain_reasons_name_what_is_missing():
    with patch("importlib.util.find_spec", return_value=None):
        assert spec.toolchain_reasons() == [
            "needs CuTe DSL (nvidia-cutlass-dsl)",
            "needs FlashInfer",
            "needs the CUDA Python bindings",
        ]


def test_another_flashinfer_low_latency_gemm_is_refused():
    with patch.object(spec, "flashinfer_ll_gemm_digest", return_value="0" * 64):
        reasons = spec.toolchain_reasons()
    assert any("is not the low-latency GEMM" in r for r in reasons), reasons


def test_the_slot_plan_follows_the_measured_bounds():
    widths = tuple(range(6, 49, 6))
    assert not any(
        s.hop_a or s.fast_gate or s.routed
        for s in plan.plan(widths, plan.Facts()).values()
    )
    on = plan.plan(tuple(range(1, 49)), plan.Facts(hop_a=True, routed=True))
    assert [w for w, s in on.items() if s.hop_a] == list(range(1, 9))
    assert [w for w, s in on.items() if s.fast_gate] == [*range(1, 9), 48]
    with pytest.raises(ValueError):
        plan.slots(49, plan.Facts())
