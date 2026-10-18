# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 spec through SGLang's startup checks on CPU, as the plugin
registers it: the documented setup passes, and every other setup is refused at
argument resolution with each reason named."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.environ import envs
from sglang.srt.megakernels import guards
from sglang.srt.megakernels import spec as registry
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG, plan, runtime, spec

DOCUMENTED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=4,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="DSPARK",
    moe_a2a_backend="none",
    dsv4_attn_backend="flashmla",
    cuda_graph_config=default_cuda_graph_config(),
)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


def _refusal(hf=None, dtype="bfloat16", isa="sm_103", **args) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else
    its error message."""
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        **{**HF_CONFIG, "model_type": "deepseek_v41", **(hf or {})},
    )
    with (
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
        patch.object(spec, "flashinfer_ll_gemm_digest", return_value=None),
        patch.object(spec.importlib.util, "find_spec", return_value=object()),
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        try:
            guards.handle_nvidia_megakernel(ServerArgs(**{**DOCUMENTED, **args}))
        except ValueError as e:
            return str(e)
    return ""


def test_the_documented_setup_is_accepted():
    assert _refusal() == ""


BREAKABLE = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
)


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8"),
        (dict(pp_size=2), "pipeline parallelism is in use"),
        (dict(kv_cache_dtype="auto"), "kv_cache_dtype 'auto'"),
        (dict(speculative_algorithm="EAGLE"), "speculative_algorithm 'EAGLE'"),
        (dict(speculative_algorithm=None), "speculative_algorithm None"),
        (dict(dsv4_attn_backend="flashmla_mega"), "dsv4_attn_backend 'flashmla_mega'"),
        (dict(moe_a2a_backend="deepep"), "moe_a2a_backend 'deepep'"),
        (dict(enable_encoder_swa_bounded_replay=True), "enable_encoder_swa_bounded"),
        (
            dict(startup_weight_load_mode="overlap"),
            "startup_weight_load_mode 'overlap'",
        ),
        (dict(cuda_graph_config=BREAKABLE), "decode CUDA graphs"),
        (dict(hf=dict(hidden_size=7168)), "config.hidden_size 7168"),
        (dict(hf=dict(model_type="deepseek_v4")), "config.model_type"),
        (dict(dtype="float16"), "dtype 'float16'"),
        (dict(isa="sm_100"), "not sm_100"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    assert want in _refusal(**kw)


def test_ragged_verify_is_refused():
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("dynamic"):
        assert "SGLANG_RAGGED_VERIFY_MODE" in _refusal()


def test_a_bad_slot_list_is_refused_before_the_load(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"gate"}))
    assert "slot 'gate' needs slot 'moe'" in _refusal()


def test_every_reason_is_listed_at_once():
    message = _refusal(tp_size=8, kv_cache_dtype="auto", pp_size=2)
    assert message.count("\n  - ") == 3, message


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
