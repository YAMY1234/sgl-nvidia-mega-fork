# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.8 Flash Next (FN) megakernel's spec on CPU: the setups SGLang's startup
checks accept and refuse with it, the slot requests it refuses before the load, and the
slot plan."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.kernel_plugins.nvidia import binding as guards
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.qwen3_8 import plan, runtime, spec


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


# The measured TP4 server (the speculative algorithm as resolved: NEXTN is EAGLE)
DOCUMENTED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=4,
    kv_cache_dtype="bfloat16",
    speculative_algorithm="EAGLE",
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
    moe_runner_backend="flashinfer_trtllm",
    attention_backend="trtllm_mha",
    linear_attn_decode_backend="flashinfer",
    linear_attn_verify_backend="triton",
    enable_linear_replayssm_spec=True,
    mamba_ssm_dtype="bfloat16",
    cuda_graph_config=default_cuda_graph_config(),
)


def _refusal(dtype="bfloat16", isa="sm_103", **args) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else its
    error message."""
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        model_type="qwen4_exp",
        text_config=SimpleNamespace(model_type="qwen4_exp_text"),
    )
    with (
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        try:
            guards.handle_nvidia_megakernel(ServerArgs(**{**DOCUMENTED, **args}))
        except ValueError as e:
            return str(e)
    return ""


def test_the_documented_setup_is_accepted():
    assert _refusal() == ""


def test_fn_runs_on_the_generic_binding():
    # the EAGLE binding's forward view reads dense attention metadata QSA lacks
    assert guards._binding_class([spec.ARCHITECTURE]) is guards.MegakernelBinding


BREAKABLE = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
)


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8"),
        (dict(pp_size=2), "pipeline parallelism is in use"),
        (dict(speculative_algorithm=None), "speculative_algorithm None"),
        (dict(speculative_eagle_topk=4), "speculative_eagle_topk 4"),
        (
            dict(speculative_use_rejection_sampling=True),
            "speculative_use_rejection_sampling True",
        ),
        (dict(cuda_graph_config=BREAKABLE), "decode CUDA graphs"),
        (dict(dtype="float16"), "dtype 'float16'"),
        (dict(isa="sm_100"), "not sm_100"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    assert want in _refusal(**kw)


def test_an_unknown_slot_is_refused_before_the_load(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"hop"}))
    assert "unknown FN slot 'hop'" in _refusal()


def test_the_slot_plan_covers_one_request():
    assert plan.candidates((4, 8, 12, 16), 4) == (4,)
    assert plan.plan((4,), plan.Facts()) == {4: plan.Slots()}
    with pytest.raises(ValueError):
        plan.slots(plan.MAX_ROWS + 1, plan.Facts())


def test_multi_request_widths_are_covered_only_where_a_slot_runs():
    assert plan.candidates((4, 8, 12, 16), 4, multi=True) == (4, 8, 16)
    every = plan.Facts(**dict.fromkeys(plan.Facts.__struct_fields__, True))
    # every slot is bounded to one request until it qualifies wider steps
    assert plan.covered((4, 8, 16), every, 4) == (4,)
    assert plan.covered((4, 8, 16), plan.Facts(), 4) == (4,)
    assert plan.widest((4, 8, 16), plan.ROUTE_MAX_ROWS) == 4


def test_multi_request_is_a_known_request(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"multi_request", "gdn_mk"}))
    assert "unknown FN slot" not in _refusal()
