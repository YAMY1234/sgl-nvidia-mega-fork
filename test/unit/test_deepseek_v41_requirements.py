# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 megakernel's setup checks on CPU: the documented setup passes,
and every other setup is refused with each reason named."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG, plan, requirements

DOCUMENTED = dict(
    tp_size=4,
    pp_size=1,
    nnodes=1,
    ep_size=1,
    enable_dp_attention=False,
    enable_eplb=False,
    attn_cp_size=1,
    dcp_size=1,
    enable_prefill_cp=False,
    enable_two_batch_overlap=False,
    enable_single_batch_overlap=False,
    enable_lora=False,
    lora_paths=None,
    disaggregation_mode="null",
    enable_hierarchical_cache=False,
    enable_memory_saver=False,
    enable_torch_compile=False,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="DSPARK",
    speculative_num_draft_tokens=6,
    moe_a2a_backend="none",
    dsv4_attn_backend="flashmla",
    enable_encoder_swa_bounded_replay=False,
    startup_weight_load_mode="serial",
    watchdog_timeout=300,
)


def _reasons(dtype="torch.bfloat16", hf=None, **args):
    server_args = SimpleNamespace(**{**DOCUMENTED, **args})
    hf_config = SimpleNamespace(
        **{**HF_CONFIG, "model_type": "deepseek_v41", **(hf or {})}
    )
    model_config = SimpleNamespace(dtype=dtype, hf_config=hf_config)
    with (
        patch.object(requirements, "device_reasons", return_value=[]),
        patch.object(requirements, "toolchain_reasons", return_value=[]),
    ):
        return requirements.disable_reasons(server_args, model_config)


def test_the_documented_setup_is_accepted():
    assert _reasons() == []


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8"),
        (dict(ep_size=4), "expert parallelism is in use"),
        (dict(pp_size=2), "pipeline parallelism is in use"),
        (dict(enable_dp_attention=True), "DP attention is in use"),
        (dict(enable_two_batch_overlap=True), "two-batch overlap is in use"),
        (dict(lora_paths=["a"]), "LoRA is in use"),
        (dict(kv_cache_dtype="auto"), "kv_cache_dtype 'auto'"),
        (dict(speculative_algorithm="EAGLE"), "speculative_algorithm 'EAGLE'"),
        (dict(speculative_algorithm=None), "speculative_algorithm None"),
        (dict(dsv4_attn_backend="flashmla_mega"), "dsv4_attn_backend 'flashmla_mega'"),
        (dict(moe_a2a_backend="deepep"), "moe_a2a_backend 'deepep'"),
        (dict(enable_encoder_swa_bounded_replay=True), "enable_encoder_swa_bounded"),
        (dict(watchdog_timeout=4), "--watchdog-timeout 4 s"),
        (dict(dtype="torch.float16"), "dtype 'float16'"),
        (dict(hf=dict(hidden_size=7168)), "config.hidden_size 7168"),
        (dict(hf=dict(model_type="deepseek_v4")), "config.model_type"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    reasons = _reasons(**kw)
    assert any(want in r for r in reasons), (want, reasons)


def test_every_reason_is_listed_at_once():
    assert len(_reasons(tp_size=8, kv_cache_dtype="auto", pp_size=2)) == 3


def test_the_toolchain_reasons_name_what_is_missing():
    with patch("importlib.util.find_spec", return_value=None):
        assert requirements.toolchain_reasons() == [
            "needs CuTe DSL (nvidia-cutlass-dsl)",
            "needs FlashInfer",
            "needs the CUDA Python bindings",
        ]


def test_another_flashinfer_low_latency_gemm_is_refused():
    with patch.object(requirements, "flashinfer_ll_gemm_digest", return_value="0" * 64):
        reasons = requirements.toolchain_reasons()
    assert any("is not the low-latency GEMM" in r for r in reasons), reasons


def test_the_wait_deadline_is_half_the_watchdog_at_most_60_s():
    assert requirements.deadline_s(300) == 60.0
    assert requirements.deadline_s(20) == 10.0
    assert requirements.deadline_s(None) == 60.0


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
