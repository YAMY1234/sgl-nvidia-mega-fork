# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 spec through SGLang's startup checks on CPU, as the plugin
registers it: the documented setup passes, and every other setup is refused at
argument resolution with each reason named."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
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
# the documented launch's environment (benchmark/README.md)
DOCUMENTED_ENV = {
    "SGLANG_SPEC_TP_SYNC": "rng,-dspark-accept-sample",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": True,
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
}


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


def _refusal(hf=None, dtype="bfloat16", isa="sm_103", env=None, **args) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else
    its error message."""
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        **{**HF_CONFIG, "model_type": "deepseek_v41", **(hf or {})},
    )
    with (
        contextlib.ExitStack() as stack,
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
        patch.object(spec, "flashinfer_ll_gemm_digest", return_value=None),
        patch.object(spec.importlib.util, "find_spec", return_value=object()),
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        for name, value in {**DOCUMENTED_ENV, **(env or {})}.items():
            stack.enter_context(getattr(envs, name).override(value))
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


DISABLED = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.DISABLED
)


def test_decode_without_cuda_graphs_is_refused_with_its_reason():
    """The eager forward loses long-context accuracy, so a launch without decode CUDA
    graphs is refused, saying why."""
    message = _refusal(cuda_graph_config=DISABLED)
    assert "decode CUDA graphs 'disabled' is not one of ['full']" in message, message
    assert (
        "the eager forward is not supported yet; enable decode CUDA graphs" in message
    )


@pytest.mark.parametrize(
    "switch, value",
    [
        ("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "1"),
        ("SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT", "1:hop_a:5000"),
    ],
)
def test_a_test_only_switch_runs_eager_steps(monkeypatch, switch, value):
    """The stock-parity and hop-fault serving tests step eagerly; only their switches
    let a launch without decode CUDA graphs through."""
    assert switch in spec.EAGER_TEST_SWITCHES
    monkeypatch.setenv(switch, value)
    assert spec.decode_graph_backends() == {"full", "disabled"}
    # SPEC's requirements are read at import, under the server's environment
    requires = msgspec.structs.replace(
        spec.SPEC.requires, decode_graph_backends=spec.decode_graph_backends()
    )
    monkeypatch.setattr(
        spec, "SPEC", msgspec.structs.replace(spec.SPEC, requires=requires)
    )
    assert _refusal(cuda_graph_config=DISABLED) == ""


def test_ragged_verify_is_refused():
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("dynamic"):
        assert "SGLANG_RAGGED_VERIFY_MODE" in _refusal()


def test_a_bad_slot_list_is_refused_before_the_load(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"gate"}))
    assert "slot 'gate' needs slot 'moe'" in _refusal()


def test_a_launch_without_the_measured_environment_is_refused():
    """Each variable of the measured launch is named; SGLang's defaults are refused."""
    defaults = {
        "SGLANG_SPEC_TP_SYNC": "all",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": False,
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
    }
    message = _refusal(env=defaults)
    assert message.count("\n  - ") == len(defaults), message
    for name in defaults:
        assert name in message


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
        s.hop_a or s.fast_gate or s.routed or s.embed or s.engram_prefetch
        for s in plan.plan(widths, plan.Facts()).values()
    )
    on = plan.plan(tuple(range(1, 49)), plan.Facts(hop_a=True, routed=True))
    assert [w for w, s in on.items() if s.hop_a] == list(range(1, 17))
    assert [w for w, s in on.items() if s.fast_gate] == [*range(1, 17), 48]
    # slot gate_wide moves only the fast top-k, to every routed width
    gate_wide = plan.Facts(hop_a=True, routed=True, fast_gate_wide=True)
    wide_gate = plan.plan(tuple(range(1, 49)), gate_wide)
    assert [w for w, s in wide_gate.items() if s.fast_gate] == list(range(1, 49))
    assert all(
        msgspec.structs.replace(s, fast_gate=on[w].fast_gate) == on[w]
        for w, s in wide_gate.items()
    )
    engram = plan.plan(tuple(range(1, 49)), plan.Facts(engram_prefetch=True))
    assert [w for w, s in engram.items() if s.engram_prefetch] == list(range(1, 9))
    with pytest.raises(ValueError):
        plan.slots(49, plan.Facts())
    # the two-shot slot alone covers the widths past the routed pair
    wide = plan.plan(tuple(range(6, 385, 6)), plan.Facts(routed=True, two_shot=True))
    assert [w for w, s in wide.items() if s.two_shot] == list(range(54, 385, 6))
    assert [w for w, s in wide.items() if s.routed] == list(range(6, 49, 6))
    with pytest.raises(ValueError):
        plan.slots(385, plan.Facts(two_shot=True))
    assert spec.SPEC.max_rows == plan.max_rows("two_shot" in runtime.SLOTS)
