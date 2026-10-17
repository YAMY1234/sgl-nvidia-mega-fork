# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 SM103 monokernel specs on CPU: their registry entries, API
version, widths and the reasons SGLang refuses a server with. SGLang checks the
declared requirements, the same way for every spec."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from msgspec.structs import replace
from sglang.srt.monokernels import guards
from sglang.srt.monokernels import spec as registry
from sglang.srt.server_args import ServerArgs

import sgl_nvidia_mega
from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG, draft, plan
from sgl_nvidia_mega.deepseek_v41 import spec as dsv41

TARGET, DRAFT = "DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMDSpark"


@pytest.fixture(autouse=True)
def registered():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    with patch.dict(registry._REGISTRY, clear=True):
        sgl_nvidia_mega.register_specs()
        yield


def test_the_plugin_registers_both_specs():
    for arch, spec in ((TARGET, dsv41.SPEC), (DRAFT, draft.SPEC)):
        found, why = registry.find_spec(
            backend=sgl_nvidia_mega.BACKEND, architectures=[arch], isa=dsv41.ISA
        )
        assert found is spec, why


def test_the_specs_use_this_sglangs_api_version():
    for spec in (dsv41.SPEC, draft.SPEC):
        assert spec.api_version == registry.MONOKERNEL_API_VERSION
    # a spec of another version is refused with a reason, not loaded
    old = replace(dsv41.SPEC, api_version=registry.MONOKERNEL_API_VERSION + 1)
    with patch.object(dsv41, "SPEC", old):
        found, why = registry.find_spec(
            backend=sgl_nvidia_mega.BACKEND, architectures=[TARGET], isa=dsv41.ISA
        )
    assert found is None and "monokernel API version" in why


def test_widths_are_the_routed_rows():
    assert dsv41.WIDTHS == tuple(range(1, 49))
    # aligning SGLang's speculative capture list (6 rows a request) to the widths
    # keeps batch sizes 1-8, the batches the widths hold whole
    covered = guards.covered_capture_sizes(
        widths=dsv41.WIDTHS, tokens_per_request=6, sizes=[1, 2, 3, 4, 8, 16, 32, 64]
    )
    assert covered == [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]


def test_the_slot_plan_follows_the_measured_bounds():
    on = plan.Facts(hop_a=True)
    table = plan.plan(dsv41.WIDTHS, on)
    assert [w for w, s in table.items() if s.hop_a] == list(range(1, 9))
    assert [w for w, s in table.items() if s.fast_gate] == [*range(1, 9), 48]
    assert all(s.routed for s in table.values())
    # without SGLang's attention output before wo_b, the hop-A slot stays off
    assert not any(
        s.hop_a for s in plan.plan(dsv41.WIDTHS, plan.Facts(hop_a=False)).values()
    )
    with pytest.raises(ValueError):
        plan.slots(49, on)


def _server_args(**kw):
    defaults = dict(
        model_path="dummy",
        tp_size=4,
        ep_size=1,
        kv_cache_dtype="fp8_e4m3",
        dsv4_attn_backend="flashmla",
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        monokernel_backend=sgl_nvidia_mega.BACKEND,
    )
    return ServerArgs(**{**defaults, **kw})


def _reasons(cfg, *, isa=dsv41.ISA, dtype="bfloat16", **hf):
    hf_config = SimpleNamespace(
        architectures=[TARGET], **{**HF_CONFIG, "model_type": "deepseek_v41", **hf}
    )
    model_config = SimpleNamespace(hf_config=hf_config, dtype=dtype)
    # the decode backend DeepSeek-V4's override resolves on a real server
    with patch.object(guards, "attention_backends_of", return_value=("dsv4", "dsv4")):
        return guards.monokernel_disable_reasons(
            cfg=cfg, model_config=model_config, isa=isa
        )


def test_the_documented_setup_is_accepted():
    spec, reasons = _reasons(_server_args())
    assert spec is dsv41.SPEC
    # the toolchain checks run on the GPU image; on CPU only they may refuse
    assert [r for r in reasons if not r.startswith("needs ")] == []


@pytest.mark.parametrize(
    "kw, isa, dtype, hf, want",
    [
        ({}, "sm_100", "bfloat16", {}, "has no spec for"),  # not SM103
        ({}, "sm_90", "bfloat16", {}, "has no spec for"),
        ({"tp_size": 8}, None, "bfloat16", {}, "tp_size 8"),  # not TP4
        ({"tp_size": 2}, None, "bfloat16", {}, "tp_size 2"),
        ({"ep_size": 4}, None, "bfloat16", {}, "expert parallelism"),  # EP != 1
        ({}, None, "float16", {}, "dtype 'float16'"),  # wrong dtype
        ({}, None, "bfloat16", {"hidden_size": 7168}, "config.hidden_size 7168"),
        ({}, None, "bfloat16", {"model_type": "deepseek_v4"}, "config.model_type"),
        ({"kv_cache_dtype": "auto"}, None, "bfloat16", {}, "kv_cache_dtype 'auto'"),
        (
            {"moe_a2a_backend": "deepep"},
            None,
            "bfloat16",
            {},
            "moe_a2a_backend 'deepep'",
        ),
    ],
)
def test_each_refusal_names_its_reason(kw, isa, dtype, hf, want):
    spec, reasons = _reasons(
        _server_args(**kw), isa=isa or dsv41.ISA, dtype=dtype, **hf
    )
    reasons = [r for r in reasons if not r.startswith("needs ")]
    assert any(want in r for r in reasons), (want, reasons)


def test_a_server_without_dspark_is_refused():
    cfg = _server_args(speculative_algorithm=None, speculative_num_draft_tokens=None)
    _, reasons = _reasons(cfg)
    assert any("speculative_algorithm None" in r for r in reasons), reasons
    cfg = _server_args(speculative_algorithm="EAGLE", speculative_num_draft_tokens=6)
    _, reasons = _reasons(cfg)
    assert any("speculative_algorithm 'EAGLE'" in r for r in reasons), reasons


def test_another_flashinfer_low_latency_gemm_is_refused():
    with patch.object(dsv41, "_flashinfer_ll_gemm_digest", return_value="0" * 64):
        reasons = dsv41._config_disable_reasons(None, None)
    assert any("is not the low-latency GEMM" in r for r in reasons), reasons
    with patch.object(
        dsv41,
        "_flashinfer_ll_gemm_digest",
        return_value=min(dsv41.FLASHINFER_LL_GEMM_SHA256),
    ):
        reasons = dsv41._config_disable_reasons(None, None)
    assert not any("low-latency GEMM" in r for r in reasons), reasons


def test_the_toolchain_reasons_name_what_is_missing():
    with patch("importlib.util.find_spec", return_value=None):
        reasons = dsv41._config_disable_reasons(None, None)
    assert reasons == [
        "needs CuTe DSL (nvidia-cutlass-dsl)",
        "needs FlashInfer",
        "needs the CUDA Python bindings",
    ]


FLAG_OFF = textwrap.dedent(
    """
    import sys
    from types import SimpleNamespace

    from sglang.srt.monokernels import runtime
    from sglang.srt.monokernels.guards import handle_monokernel
    from sglang.srt.server_args import ServerArgs

    import sgl_nvidia_mega

    sgl_nvidia_mega.register_specs()  # the entry point runs in every process
    weights = [("w", object())]
    # no published server config (tools, the loader in tests): the normal path
    assert runtime.monokernel_weights_iterator(model=None, weights=weights) is weights
    # a server without --monokernel-backend: no spec lookup, no copies
    runtime._backend = lambda: None
    assert runtime.monokernel_weights_iterator(model=None, weights=weights) is weights
    runtime.install_monokernel_forward(
        model=None, hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        capture_tail_hooks=[],
    )
    handle_monokernel(ServerArgs(model_path="dummy"))
    ours = sorted(m for m in sys.modules if m.startswith("sgl_nvidia_mega."))
    assert ours == [], ours
    print("FLAG-OFF OK")
    """
)


def test_nothing_of_ours_runs_without_the_backend():
    # a fresh interpreter: this test module imported the specs already
    out = subprocess.run(
        [sys.executable, "-c", FLAG_OFF], capture_output=True, text=True, timeout=300
    )
    assert "FLAG-OFF OK" in out.stdout, out.stdout[-2000:] + out.stderr[-4000:]
