# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 megakernel's spec on CPU: the setups SGLang's startup checks accept
and refuse with it, the slot requests it refuses before the load, and the slot plan."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
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
from sgl_nvidia_mega.qwen3_5 import (
    TEXT_CONFIG,
    plan,
    runtime,
    spec,
    topology,
)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(runtime, "SLOTS", runtime.SLOTS)
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


# The measured TP8 server (the speculative algorithm as resolved: NEXTN is EAGLE)
DOCUMENTED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=8,
    nnodes=2,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="EAGLE",
    speculative_num_steps=6,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=7,
    moe_runner_backend="flashinfer_trtllm",
    flashinfer_allreduce_fusion_backend="cutedsl",
    attention_backend="trtllm_mha",
    linear_attn_decode_backend="triton",
    enable_linear_replayssm_spec=True,
    mamba_ssm_dtype="bfloat16",
    enable_hierarchical_cache=True,
    cuda_graph_config=default_cuda_graph_config(),
)


PREFILL = dict(
    tp_size=1,
    nnodes=1,
    disaggregation_mode="prefill",
    enable_linear_replayssm_spec=False,
    moe_runner_backend="flashinfer_cutedsl",
    flashinfer_allreduce_fusion_backend=None,
    linear_attn_prefill_backend="flashinfer",
)


def _refusal(text=None, dtype="bfloat16", isa="sm_103", **args) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else its
    error message."""
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(**{**TEXT_CONFIG, **(text or {})}),
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


def test_tp4_on_one_tray_is_accepted():
    assert _refusal(tp_size=4, nnodes=1) == ""


def test_the_decode_side_of_pd_is_accepted():
    assert _refusal(disaggregation_mode="decode") == ""


def test_an_unsupported_prefill_topology_is_refused():
    message = _refusal(
        disaggregation_mode="prefill", enable_linear_replayssm_spec=False
    )
    assert "PD prefill requires TP1/PP4" in message, message


def test_prefill_tp1_pp4_is_admitted_with_the_data_only_binding():
    assert (
        _refusal(
            tp_size=1,
            nnodes=1,
            pp_size=4,
            disaggregation_mode="prefill",
            enable_linear_replayssm_spec=False,
            moe_runner_backend="flashinfer_cutedsl",
            flashinfer_allreduce_fusion_backend=None,
        )
        == ""
    )


@pytest.mark.parametrize(
    "switch, slots",
    [
        pytest.param(None, None, id="megakernel-alone"),
        pytest.param("0", None, id="switch-off"),
        pytest.param("1", "hop", id="slot-list"),
        pytest.param("1", "", id="empty-slot-list"),
    ],
)
def test_prefill_tp1_pp2_is_refused_without_the_measured_stack(
    monkeypatch, switch, slots
):
    for name, value in (
        ("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", switch),
        (runtime.SLOTS_ENV, slots),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    message = _refusal(**PREFILL, pp_size=2)
    assert "PD prefill requires TP1/PP4 without expert parallelism" in message, message
    # TP1/PP4 keeps its admission, the FlashInfer prefill route included
    assert _refusal(**PREFILL, pp_size=4) == ""


def test_prefill_tp1_pp2_is_admitted_under_the_measured_stack(measured):
    assert _refusal(**PREFILL, pp_size=2) == ""
    assert _refusal(**PREFILL, pp_size=4) == ""


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(dict(pp_size=2, ep_size=2), id="ep2"),
        pytest.param(dict(pp_size=2, tp_size=2), id="tp2"),
        pytest.param(dict(pp_size=3), id="pp3"),
        pytest.param(dict(pp_size=1), id="pp1"),
    ],
)
def test_the_measured_stack_admits_no_other_prefill_shape(measured, args):
    message = _refusal(**{**PREFILL, **args})
    assert (
        "PD prefill requires TP1/PP4 or TP1/PP2 without expert parallelism" in message
    ), message


BREAKABLE = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
)


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=1, nnodes=1), "tp_size 1"),
        (dict(pp_size=2), "pipeline parallelism is in use"),
        (dict(attn_dp_size=4), "DP attention requires DEP8"),
        (dict(kv_cache_dtype="auto"), "kv_cache_dtype 'auto'"),
        (dict(speculative_algorithm=None), "speculative_algorithm None"),
        (dict(speculative_eagle_topk=4), "speculative_eagle_topk 4"),
        (
            dict(speculative_use_rejection_sampling=True),
            "speculative_use_rejection_sampling True",
        ),
        (dict(moe_runner_backend="flashinfer_cutedsl"), "moe_runner_backend"),
        (dict(flashinfer_allreduce_fusion_backend=None), "flashinfer_allreduce_fusion"),
        (dict(attention_backend="triton"), "attention_backend 'triton'"),
        (dict(linear_attn_decode_backend="flashinfer"), "linear_attn_decode_backend"),
        (dict(enable_linear_replayssm_spec=False), "enable_linear_replayssm_spec"),
        (dict(mamba_ssm_dtype="float32"), "mamba_ssm_dtype 'float32'"),
        (dict(startup_weight_load_mode="overlap"), "startup_weight_load_mode"),
        (dict(cuda_graph_config=BREAKABLE), "decode CUDA graphs"),
        (dict(text=dict(num_experts=256)), "config.text_config.num_experts 256"),
        (dict(text=dict(hidden_size=5120)), "config.text_config.hidden_size 5120"),
        (dict(dtype="float16"), "dtype 'float16'"),
        (dict(isa="sm_100"), "not sm_100"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    assert want in _refusal(**kw)


def test_an_sglang_without_the_eagle_forward_view_is_refused(monkeypatch):
    # an import of this name now raises ImportError
    monkeypatch.setitem(
        sys.modules, "sglang.srt.kernel_plugins.nvidia.eagle_binding", None
    )
    assert "this SGLang has no EAGLE forward view" in _refusal()


@pytest.mark.parametrize(
    "slots, want",
    [
        ({"hop2"}, "unknown slot 'hop2'"),
        ({"verify_sample"}, "'verify_sample' is no longer an entry: drop it"),
        ({"mkm"}, "slot 'mkm' runs on slot 'route''s outputs"),
        ({"gdn_pdl"}, "slot 'gdn_pdl' launches slot 'gdn_verify''s verify"),
        (
            {"gdn_verify", "gdn_push"},
            "slot 'gdn_push' pushes slot 'gdn_verify''s out_proj into slot 'hop'",
        ),
        (
            {"gdn_verify", "gdn_front"},
            "slot 'gdn_front' runs slot 'gdn_pdl''s dependent verify",
        ),
        (
            {"gdn_trt_front"},
            "slot 'gdn_trt_front' replaces slot 'gdn_front' in front of slot "
            "'gdn_core'",
        ),
        (
            {"route_lin"},
            "slot 'route_lin' runs the router before the all-reduce",
        ),
        (
            {
                "route",
                "mkm",
                "hop",
                "gdn_verify",
                "gdn_push",
                "mka",
                "mka_push",
                "route_lin",
                "route_shared",
            },
            "slot 'route_lin' needs 'lowm_se' to replace 'route_shared'",
        ),
        ({"route_shared"}, "slot 'route_shared' runs in slot 'route''s launch"),
        ({"hop_mid"}, "slot 'hop_mid' widens slot 'hop'"),
        ({"route", "mkm_wide"}, "slot 'mkm_wide' is slot 'mkm''s 16-token engine"),
        (
            {"route", "mkm", "mkm_wide"},
            "slot 'mkm_wide' serves the steps of more than one request ('multi_request')",
        ),
        ({"mkm_m32"}, "slot 'mkm_m32' takes slot 'route_topk''s rows"),
        ({"route_pre"}, "unknown slot 'route_pre'"),
        ({"route_pre_front"}, "unknown slot 'route_pre_front'"),
        (
            {"route_topk", "mkm_m32"},
            "slot 'mkm_m32' serves the steps of more than two requests ('multi_request')",
        ),
        ({"mkm_m64"}, "slot 'mkm_m64' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m64"},
            "slot 'mkm_m64' serves the steps of more than four requests ('multi_request')",
        ),
        ({"mkm_m128"}, "slot 'mkm_m128' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m128"},
            "slot 'mkm_m128' serves the steps of more than nine requests ('multi_request')",
        ),
        ({"mkm_m256"}, "slot 'mkm_m256' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m256"},
            "slot 'mkm_m256' serves the steps of more than 18 requests ('multi_request')",
        ),
        ({"mkm_m512"}, "slot 'mkm_m512' runs on slot 'route_topk''s routing"),
        (
            {"route_topk", "mkm_m512"},
            "slot 'mkm_m512' serves the steps of more than 36 requests ('multi_request')",
        ),
        (
            {"mkm_wide_bal"},
            "slot 'mkm_wide_bal' changes the work split of slot 'mkm_m256' or "
            "'mkm_m512'",
        ),
    ],
)
def test_a_slot_set_that_cannot_run_is_refused_before_the_load(
    monkeypatch, slots, want
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    assert want in _refusal()


def test_every_reason_is_listed_at_once():
    message = _refusal(tp_size=1, nnodes=1, kv_cache_dtype="auto", pp_size=2)
    assert message.count("\n  - ") == 3, message


def test_the_slot_plan_covers_one_request_of_7_rows():
    assert plan.plan((7,), plan.Facts()) == {7: plan.Slots()}
    assert plan.plan((7,), plan.Facts(route=True)) == {7: plan.Slots(route=True)}
    assert plan.plan((7,), plan.Facts(route=True, mkm=True)) == {
        7: plan.Slots(route=True, mkm=True)
    }
    assert plan.plan((7,), plan.Facts(hop=True)) == {7: plan.Slots(hop=True)}
    # gdn_push pushes gdn_verify's out_proj into the hop: it needs both built
    both = dict(gdn_verify=True, hop=True)
    assert plan.plan((7,), plan.Facts(**both, gdn_push=True)) == {
        7: plan.Slots(**both, gdn_push=True)
    }
    assert not plan.slots(7, plan.Facts(gdn_verify=True, gdn_push=True)).gdn_push
    # gdn_front runs gdn_pdl's verify, for one request of 7 rows only
    front = dict(gdn_verify=True, gdn_pdl=True, gdn_front=True)
    assert plan.plan((7,), plan.Facts(**front)) == {7: plan.Slots(**front)}
    assert not plan.slots(4, plan.Facts(**front)).gdn_front
    assert not plan.slots(7, plan.Facts(gdn_verify=True, gdn_front=True)).gdn_front
    # route_shared rides route's launch, up to 8 rows
    shared = dict(route=True, route_shared=True)
    assert plan.plan((7,), plan.Facts(**shared)) == {7: plan.Slots(**shared)}
    assert not plan.slots(7, plan.Facts(route_shared=True)).route_shared
    # lowm_se replaces route_shared's shared stage where both could run.
    assert _on(plan.slots(7, plan.Facts(**shared, lowm_se=True))) == [
        "lowm_se",
        "route",
    ]
    with pytest.raises(ValueError):
        plan.slots(plan.MAX_ROWS + 1, plan.Facts())


def _on(slots):
    return sorted(name for name in plan.Slots.__struct_fields__ if getattr(slots, name))


def test_the_trt_gdn_front_widens_slot_gdn_front_to_four_requests():
    """The custom front serves one request; the explicit TRT front serves four."""
    front = plan.Facts(gdn_verify=True, gdn_pdl=True, gdn_front=True)
    assert plan.slots(7, front).gdn_front
    assert not plan.slots(14, front).gdn_front
    trt = plan.Facts(
        gdn_verify=True,
        gdn_pdl=True,
        gdn_front=True,
        gdn_trt_front=True,
    )
    for rows in (7, 14, 21, 28):
        slots = plan.slots(rows, trt)
        assert slots.gdn_front and slots.gdn_trt_front
    assert not plan.slots(35, front).gdn_front
    assert not plan.slots(35, trt).gdn_front


def test_the_slot_plan_gates_each_slot_by_the_rows_its_kernels_serve():
    # Every public slot is built except mkm's 16-token engine (mkm_wide, below).
    every = plan.Facts(
        **{
            **dict.fromkeys(plan.Facts.__struct_fields__, True),
            "mkm_wide": False,
            "tep8_moe": False,
        }
    )
    assert _on(plan.slots(7, every)) == sorted(
        set(plan.Slots.__struct_fields__)
        - {
            "bt_tail",
            "bt_tail_early",
            "bt_tail_early_fin",
            "gemm_ar",
            "fc2ar",
            "gdn_conv_core",
            "lowm_ba",
            "mkm_m32",
            "mkm_m64",
            "mkm_m128",
            "mkm_m256",
            "mkm_m512",
            "route_gemm",
            "route_pre",
            "route_pre_topk",
            "route_shared",
            "route_topk",
            "route_big_mid",
        }
    )
    # Two requests also take every measured low-M launch and the hop's FP8 copy.
    assert _on(plan.slots(14, every)) == [
        "gdn_conv",
        "gdn_core",
        "gdn_front",
        "gdn_pdl",
        "gdn_trt_front",
        "gdn_verify",
        "hop",
        "hop_fp8",
        "lowm_in",
        "lowm_oproj",
        "lowm_outproj",
        "lowm_push",
        "lowm_se",
        "mka",
        "mka_push",
        "route",
    ]
    # Three requests keep only the shared expert's validated low-M launch.
    three = plan.slots(21, every)
    assert three.lowm_se
    assert three.gdn_front and three.gdn_trt_front
    assert three.hop and three.hop_fp8 and three.mka_push
    assert not (three.lowm_in or three.lowm_outproj or three.lowm_oproj)
    # The 28/56-row low-M projection paths stay stock; hop_mid's one-shot path and
    # FP8 copy also serve 28 rows, while GDN core and MKA serve through 896.
    for rows in (28, 56, 112, 224):
        expected = [
            "gdn_conv",
            "gdn_core",
            "gdn_pdl",
            "gdn_verify",
            "mka",
            "route_topk",
        ]
        if rows <= plan.GDN_TRT_FRONT_MAX_ROWS:
            expected.extend(("gdn_front", "gdn_trt_front"))
        if plan.GDN_CONV_CORE_MIN_ROWS <= rows <= plan.GDN_CONV_CORE_MAX_ROWS:
            expected.append("gdn_conv_core")
        if plan.LOWM_BA_MIN_ROWS <= rows <= plan.LOWM_BA_MAX_ROWS:
            expected.append("lowm_ba")
        if rows <= plan.MKM_M32_MAX_ROWS:
            expected.append("mkm_m32")
        if plan.MKM_M32_MAX_ROWS < rows <= plan.MKM_M64_MAX_ROWS:
            expected.append("mkm_m64")
        if plan.MKM_M64_MAX_ROWS < rows <= plan.MKM_M128_MAX_ROWS:
            expected.append("mkm_m128")
        if plan.MKM_M128_MAX_ROWS < rows <= plan.MKM_M256_MAX_ROWS:
            expected.append("mkm_m256")
        if rows <= plan.ROUTE_GEMM_MAX_ROWS:
            expected.append("route_gemm")
        if rows <= plan.HOP_MID_MAX_ROWS:
            expected.extend(("hop", "hop_fp8", "mka_push"))
        else:
            expected.extend(("bt_tail", "bt_tail_early", "bt_tail_early_fin"))
        if plan.GEMM_AR_MIN_ROWS <= rows <= plan.GEMM_AR_MAX_ROWS:
            expected.append("gemm_ar")
        assert _on(plan.slots(rows, every)) == sorted(expected)
    for rows in (448, 896):
        expected = [
            "gdn_conv",
            "gdn_core",
            "gdn_pdl",
            "gdn_verify",
            "mka",
            "route_topk",
        ]
        if rows <= plan.MKM_M512_MAX_ROWS:
            expected.append("mkm_m512")
        if rows <= plan.BT_TAIL_MAX_ROWS:
            expected.extend(("bt_tail", "bt_tail_early", "bt_tail_early_fin"))
        assert _on(plan.slots(rows, every)) == sorted(expected)
    assert _on(plan.slots(28, plan.Facts(route_topk=True))) == ["route_topk"]
    assert _on(plan.slots(7, plan.Facts(route_topk=True))) == ["route_topk"]
    assert not plan.slots(28, plan.Facts(route_gemm=True)).route_gemm
    mid = plan.Facts(hop=True, hop_mid=True)
    assert _on(plan.slots(28, mid)) == ["hop"]
    assert _on(plan.slots(56, mid)) == []
    assert _on(plan.slots(28, plan.Facts(hop=True, hop_mid=True, hop_fp8=True))) == [
        "hop",
        "hop_fp8",
    ]
    # TEP8's routed-body plan is internal: S1/S2 use route plus a routing-only
    # top-k launch, M uses the low-M shared expert, and L uses route-big. It
    # supersedes the all-expert MKM engines and the numeric tiled router front.
    tep8 = dict(route=True, route_topk=True, tep8_moe=True)
    for rows in (7, 14):
        assert _on(plan.slots(rows, plan.Facts(**tep8))) == [
            "route",
            "route_pre_topk",
        ]
    for rows in (21, 28):
        assert _on(plan.slots(rows, plan.Facts(**tep8))) == ["route_topk"]
    tier_m = plan.Facts(
        **tep8,
        route_gemm=True,
        lowm_se=True,
        lowm_se_pre=True,
    )
    for rows in (21, 28):
        assert _on(plan.slots(rows, tier_m)) == [
            "lowm_se",
            "route_pre",
            "route_topk",
        ]
    tier_l = plan.Facts(
        **tep8,
        route_gemm=True,
        mkm=True,
        mkm_wide=True,
        mkm_m32=True,
        mkm_m64=True,
        mkm_m128=True,
        mkm_m256=True,
        mkm_m512=True,
    )
    for rows in (35, 112, 224, 448, 672, 896):
        assert _on(plan.slots(rows, tier_l)) == ["route_pre", "route_topk"]
    assert plan.lowm_se_pre_rows(8, tep8_m=True) == 28
    assert plan.lowm_se_pre_rows(4, tep8_m=True) == 0
    assert [plan.moe_front_tier(r) for r in (7, 14, 28, 35, 224, 231, 896)] == [
        "S1",
        "S2",
        "M",
        "L1",
        "L1",
        "L2",
        "L2",
    ]
    # The S tiers run beside the low-M shared expert and replace route_shared.
    s_tier = plan.Facts(
        **tep8,
        route_shared=True,
        lowm_se=True,
        mkm=True,
    )
    for rows in (7, 14):
        assert _on(plan.slots(rows, s_tier)) == ["lowm_se", "route", "route_pre_topk"]
    assert _on(plan.slots(21, s_tier)) == ["lowm_se", "route_topk"]
    # SGLang offers every multiple of 7 up to MAX_ROWS; every request count is a candidate
    offered = tuple(range(7, plan.MAX_ROWS + 1, 7))
    assert plan.candidates(offered, 7) == offered
    assert plan.candidates(offered + (903,), 7) == offered
    assert plan.candidates(offered, 7, multi=False) == (7,)
    # more than one request only when requested, and only at TP8; TP4 keeps the
    # one-request step it is checked at
    assert plan.multi_request(8, frozenset({"multi_request"}))
    assert not plan.multi_request(8, frozenset())
    assert plan.multi_request(4, frozenset({"multi_request"}))
    assert plan.covered((7, 14, 28), every, 7) == (7, 14, 28)
    assert plan.covered((7, 448, 896), every, 7) == (7, 448, 896)
    # one request is always covered, more only where a slot runs
    assert plan.covered((7, 14, 28), plan.Facts(), 7) == (7,)
    assert plan.covered((7, 14, 28), plan.Facts(route=True, mkm=True), 7) == (7, 14)
    attention = plan.Facts(skv2=True, mka=True, mka_push=True, hop=True)
    assert plan.covered((7, 14, 28), attention, 7) == (7, 14)
    assert _on(plan.slots(14, attention)) == ["hop"]
    wide_attention = plan.Facts(mka=True, mka_wide=True)
    assert plan.covered((7, 14, 28, 56, 112, 224), wide_attention, 7) == (
        7,
        14,
        28,
        56,
        112,
        224,
    )
    # mkm_wide: mkm's 16-token engine at two requests, not past the router front's rows
    wide = plan.Facts(route=True, mkm=True, mkm_wide=True)
    assert _on(plan.slots(14, wide)) == ["mkm", "route"]
    assert _on(plan.slots(28, wide)) == []
    assert _on(plan.slots(14, plan.Facts(route=True, mkm=True))) == ["route"]
    m32 = plan.Facts(route_topk=True, mkm_m32=True)
    assert _on(plan.slots(28, m32)) == ["mkm_m32", "route_topk"]
    assert _on(plan.slots(14, m32)) == ["route_topk"]
    assert _on(plan.slots(56, m32)) == ["route_topk"]
    assert _on(plan.slots(28, plan.Facts(mkm_m32=True))) == []
    m64 = plan.Facts(route_topk=True, mkm_m64=True)
    assert _on(plan.slots(56, m64)) == ["mkm_m64", "route_topk"]
    assert _on(plan.slots(28, m64)) == ["route_topk"]
    assert _on(plan.slots(112, m64)) == ["route_topk"]
    assert _on(plan.slots(56, plan.Facts(mkm_m64=True))) == []
    m128 = plan.Facts(route_topk=True, mkm_m128=True)
    assert _on(plan.slots(112, m128)) == ["mkm_m128", "route_topk"]
    assert _on(plan.slots(56, m128)) == ["route_topk"]
    assert _on(plan.slots(224, m128)) == ["route_topk"]
    assert _on(plan.slots(112, plan.Facts(mkm_m128=True))) == []
    wide = plan.Facts(route_topk=True, mkm_m256=True, mkm_m512=True)
    assert _on(plan.slots(224, wide)) == ["mkm_m256", "route_topk"]
    assert _on(plan.slots(448, wide)) == ["mkm_m512", "route_topk"]
    assert _on(plan.slots(112, wide)) == ["route_topk"]
    assert _on(plan.slots(896, wide)) == ["route_topk"]
    # the router front and the hop's mailbox are built for the widest width they serve
    assert plan.widest((7, 14, 28), plan.ROUTE_MAX_ROWS) == 14
    assert plan.widest((7, 14, 28), plan.HOP_MAX_ROWS) == 14
    assert plan.widest((28,), plan.HOP_MAX_ROWS) == 0


def test_slot_bt_tail_takes_the_widths_past_the_hop_through_the_bt_routes():
    tail = plan.Facts(bt_tail=True)
    assert [r for r in range(1, plan.MAX_ROWS + 1) if plan.slots(r, tail).bt_tail] == (
        list(range(plan.BT_TAIL_MIN_ROWS, plan.BT_TAIL_MAX_ROWS + 1))
    )
    for hop, first in ((dict(hop=True), 16), (dict(hop=True, hop_mid=True), 29)):
        facts = plan.Facts(
            **hop, bt_tail=True, bt_tail_early=True, bt_tail_early_fin=True
        )
        for rows in (7, 14, 15, 16, 21, 28, 29, 35, 56, 112, 224, 703, 704, 896):
            got = plan.slots(rows, facts)
            assert got.bt_tail == (first <= rows <= plan.BT_TAIL_MAX_ROWS), rows
            assert got.bt_tail_early == got.bt_tail == got.bt_tail_early_fin
            assert not (got.hop and got.bt_tail)
    assert not plan.slots(56, plan.Facts(bt_tail_early=True)).bt_tail_early
    assert plan.covered((7, 35, 56, 896), tail, 7) == (7, 35, 56)


def test_slot_bt_tail_runs_at_tp4_tp8_and_tep8_and_its_variants_need_it():
    slots = frozenset({"bt_tail", "bt_tail_early", "bt_tail_early_fin"})
    for tp, ep, name in ((4, 1, "tp4"), (8, 1, "tp8"), (8, 4, "tep8"), (8, 8, "tep8")):
        assert topology.select(slots, topology.Topology(tp, 1, ep)) == slots, name
    for shape in (topology.Topology(2), topology.Topology(8, 8, 8)):
        assert topology.select(slots, shape) == frozenset(), shape.name
    assert runtime.slot_reasons(frozenset({"bt_tail_early", "bt_tail_early_fin"})) == [
        f"{runtime.SLOTS_ENV}: slot 'bt_tail_early' changes slot 'bt_tail''s "
        "MoE-input launch",
        f"{runtime.SLOTS_ENV}: slot 'bt_tail_early_fin' changes slot 'bt_tail''s "
        "layer-input launch",
    ]
    assert runtime.slot_reasons(slots) == []
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    assert spec.numerics(slots).kind is NumericsKind.BITWISE


def test_slots_gemm_ar_and_fc2ar_take_their_rows_on_the_bt_tails_points():
    facts = plan.Facts(bt_tail=True, gemm_ar=True, fc2ar=True, tep8_moe=True)
    on = {
        name: [
            r
            for r in range(1, plan.MAX_ROWS + 1)
            if getattr(plan.slots(r, facts), name)
        ]
        for name in ("gemm_ar", "fc2ar")
    }
    assert on["gemm_ar"] == list(
        range(plan.GEMM_AR_MIN_ROWS, plan.GEMM_AR_MAX_ROWS + 1)
    )
    assert on["fc2ar"] == list(range(plan.FC2AR_MIN_ROWS, plan.FC2AR_MAX_ROWS + 1))
    assert (plan.GEMM_AR_MIN_ROWS, plan.FC2AR_MIN_ROWS) == (33, 35)
    assert plan.GEMM_AR_MAX_ROWS == plan.FC2AR_MAX_ROWS == 128
    # no BT tail at a width, or no build: no slot
    assert not plan.slots(
        56, plan.Facts(gemm_ar=True, fc2ar=True, tep8_moe=True)
    ).gemm_ar
    assert not plan.slots(56, plan.Facts(bt_tail=True, tep8_moe=True)).fc2ar
    assert not plan.slots(56, plan.Facts(bt_tail=True, fc2ar=True)).fc2ar
    # hop_mid's widths stay on the hop
    mid = plan.Facts(**{**msgspec.structs.asdict(facts), "hop": True, "hop_mid": True})
    assert not plan.slots(28, mid).gemm_ar and plan.slots(35, mid).fc2ar


def test_slots_gemm_ar_and_fc2ar_run_where_their_engines_run():
    for tp, ep, name in ((4, 1, "tp4"), (8, 1, "tp8"), (8, 4, "tep8")):
        assert topology.select(
            frozenset({"gemm_ar"}), topology.Topology(tp, 1, ep)
        ) == {"gemm_ar"}, name
    assert topology.select(frozenset({"fc2ar"}), topology.Topology(8, 1, 4)) == {
        "fc2ar"
    }
    for shape in (topology.Topology(4), topology.Topology(8), topology.Topology(2)):
        assert topology.select(frozenset({"fc2ar"}), shape) == frozenset(), shape.name
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    assert spec.numerics({"bt_tail", "gemm_ar", "fc2ar"}).kind is NumericsKind.BITWISE


def test_the_slot_plans_row_bounds_are_the_kernels():
    from sgl_nvidia_mega.qwen3_5.sm103 import hop
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm, router

    assert plan.ROUTE_MAX_ROWS == router.MAX_TOKENS
    assert plan.ROUTE_TOPK_MAX_ROWS == router.TOPK_MAX_ROWS
    assert plan.ROUTE_SHARED_ROWS == router.SHARED_MAX_M
    assert plan.MKM_MAX_ROWS == mkm.MAX_TOKENS
    assert plan.MKM_WIDE_MAX_ROWS == mkm.M16_MAX_TOKENS
    assert plan.MKM_M32_MAX_ROWS == mkm.M32_MAX_TOKENS
    assert plan.MKM_M64_MAX_ROWS == mkm.M64_MAX_TOKENS
    assert plan.MKM_M128_MAX_ROWS == mkm.M128_MAX_TOKENS
    assert plan.MKM_M256_MAX_ROWS == mkm.M256_MAX_TOKENS
    assert plan.MKM_M512_MAX_ROWS == mkm.M512_MAX_TOKENS
    assert plan.GDN_PUSH_MAX_ROWS == hop.OPROJ_MAX_M
    assert plan.MKA_MAX_ROWS == mka.MAX_ROWS
    from sgl_nvidia_mega.qwen3_5.attention import mka as mka_step

    assert plan.MKA_WIDE_MAX_ROWS == plan.MAX_ROWS
    assert mka_step.FUSED_MAX_ROWS <= mka.WIDE_MAX_ROWS


def test_the_validated_moe_schedules_are_shape_closed():
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    assert mkm.MKM7_DEFINES == (
        "-DSPEC_STAGES=3",
        "-DPACE_D=4",
        "-DFC2SF_PRE=0",
    )
    assert mkm.engine_options(32) == (
        "-DTHR_D=2",
        "-DTOPK_NOINLINE=1",
        "-DONEPASS=1",
        "-DMMA_SFONLY=1",
    )
    assert mkm.engine_options(64) == ("-DMMA_SFONLY=1", "-DONEPASS=1")
    assert mkm.engine_options(64, plain=True) == ("-DMMA_SFONLY=1",)
    assert [mkm.reserved_sms(32, rows) for rows in (21, 22)] == [0, 32]
    assert [mkm.reserved_sms(64, rows) for rows in (35, 42, 43)] == [32, 16, 0]
    assert not mkm.plain_schedule(64, 42)
    assert mkm.plain_schedule(64, 43)


@pytest.mark.parametrize(
    "rows", [56, 112, 126, 128, 129, 133, 140, 168, 224, 256, 280, 512]
)
def test_bal_changes_only_the_wide_engine_schedule(rows):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    engine = mkm.engine_rows(rows)
    off = mkm.engine_options(engine)
    on = mkm.engine_options(engine, wide_bal=True)
    assert on == off + (("-DBAL_SPLIT=1",) if rows > 128 else ())
    assert mkm.plain_schedule(engine, rows, wide_bal=True) == mkm.plain_schedule(
        engine, rows
    )
    assert mkm.engine_options(engine, plain=True, wide_bal=True) == (
        mkm.engine_options(engine, plain=True)
    )


@pytest.mark.parametrize(
    "rows,plain", [(129, False), (210, False), (211, True), (280, True)]
)
def test_bal_internal_row_cap_selects_the_static_split(monkeypatch, rows, plain):
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    monkeypatch.setattr(mkm, "WIDE_BAL_MAX_ROWS", 210)
    engine = mkm.engine_rows(rows)
    assert mkm.plain_schedule(engine, rows, wide_bal=True) is plain
    assert (
        "-DBAL_SPLIT=1" in mkm.engine_options(engine, plain, wide_bal=True)
    ) is not plain


def test_a_numeric_slot_turns_the_claim_into_its_tolerances():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    route = spec.numerics(frozenset({"route", "hop"}))
    assert route.kind is NumericsKind.TOLERANCE
    assert dict(route.tolerances) == spec.NUMERIC_TOLERANCES["route"]
    tiled_route = spec.numerics(frozenset({"route_topk", "route_gemm"}))
    assert tiled_route.kind is NumericsKind.TOLERANCE
    assert dict(tiled_route.tolerances) == spec.NUMERIC_TOLERANCES["route_gemm"]
    claim = spec.numerics(frozenset({"gdn_verify", "hop", "gdn_push"}))
    assert claim.kind is NumericsKind.TOLERANCE
    assert dict(claim.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_push"],
        **spec.NUMERIC_TOLERANCES["gdn_verify"],
    }
    shared = spec.numerics(frozenset({"route", "route_shared"}))
    assert dict(shared.tolerances) == {
        **spec.NUMERIC_TOLERANCES["route"],
        **spec.NUMERIC_TOLERANCES["route_shared"],
    }
    both = spec.numerics(frozenset({"gdn_verify", "gdn_pdl", "gdn_front", "gdn_push"}))
    assert dict(both.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_verify"],
        **spec.NUMERIC_TOLERANCES["gdn_push"],
        **spec.NUMERIC_TOLERANCES["gdn_front"],
    }
    core = spec.numerics(frozenset({"gdn_verify", "gdn_core"}))
    assert dict(core.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_core"],
        **spec.NUMERIC_TOLERANCES["gdn_verify"],
    }
    attention = spec.numerics(frozenset({"skv2", "mka", "mka_push", "hop"}))
    assert attention.kind is NumericsKind.TOLERANCE
    assert dict(attention.tolerances) == {
        **spec.NUMERIC_TOLERANCES["skv2"],
        **spec.NUMERIC_TOLERANCES["mka"],
    }
    small = spec.numerics(frozenset({"gdn_front", "gdn_trt_front", "route_lin"}))
    assert dict(small.tolerances) == {
        **spec.NUMERIC_TOLERANCES["gdn_front"],
        **spec.NUMERIC_TOLERANCES["gdn_trt_front"],
        **spec.NUMERIC_TOLERANCES["route_lin"],
    }


def test_plugin_runtime_slot_declares_its_actual_numerics():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    sampled = spec.numerics(frozenset({"sharded_gumbel"}))
    assert sampled.kind is NumericsKind.TOLERANCE
    assert dict(sampled.tolerances) == spec.NUMERIC_TOLERANCES["sharded_gumbel"]

    # The deferred producer changes the draft's finalize/reduction order.  It uses
    # the draft spec's existing final-logit envelope, not a target-model tolerance.
    finalize = draft.numerics(frozenset({"draft_fin"}))
    assert finalize.kind is NumericsKind.TOLERANCE
    assert dict(finalize.tolerances) == {"draft_logits.rel": 2**-7}


def test_the_shipped_policy_claims_bitwise_numerics():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    shipped = frozenset(runtime.SHIPPED_SLOTS)
    assert spec.numerics(shipped).kind is NumericsKind.BITWISE
    assert draft.numerics(shipped).kind is NumericsKind.BITWISE


@pytest.mark.parametrize("tp, pd", [(2, "decode"), (4, "null")])
def test_topology_admission_is_driven_by_plugin_capabilities(tp, pd):
    assert _refusal(tp_size=tp, nnodes=1, disaggregation_mode=pd) == ""


# The router-overlap slots the TP8 serving policy requests, without route_lin
ROUTE_LIN_SLOTS = frozenset(
    "route route_shared lowm_se mkm hop gdn_verify gdn_push mka mka_push".split()
)


@pytest.mark.parametrize("pd", ["null", "decode"])
def test_the_tp8_route_lin_plan_is_accepted_in_both_roles(monkeypatch, pd):
    monkeypatch.setattr(runtime, "SLOTS", ROUTE_LIN_SLOTS)
    assert _refusal(disaggregation_mode=pd) == ""


def test_the_numerics_claim_covers_the_slots_a_plan_can_add():
    claim = spec.numerics(topology.plannable(ROUTE_LIN_SLOTS))
    assert claim == spec.numerics(ROUTE_LIN_SLOTS | {"route_lin"})
    shipped = frozenset(runtime.SHIPPED_SLOTS)
    assert topology.plannable(shipped) == shipped | {"mkm_wide_bal"}


@pytest.mark.parametrize("a2a", ["flashinfer", "flashinfer_nvlink_one_sided"])
def test_dep8_accepts_the_recipe_stock_moe_and_dp_lm_head(a2a):
    assert (
        _refusal(
            disaggregation_mode="decode",
            attn_dp_size=8,
            ep_size=8,
            enable_dp_lm_head=True,
            moe_runner_backend="flashinfer_cutedsl",
            moe_a2a_backend=a2a,
            flashinfer_allreduce_fusion_backend=None,
        )
        == ""
    )


@pytest.mark.parametrize(
    "fields, name",
    [
        pytest.param(dict(tp_size=8, attn_dp_size=8, ep_size=8), "dep8", id="dep8"),
        pytest.param(
            dict(tp_size=8, attn_dp_size=1, dp_size=2, ep_size=8), "tep8", id="replicas"
        ),
    ],
)
def test_the_configured_topology_reads_only_the_resolved_attention_dp_width(
    fields, name
):
    assert topology.from_config(SimpleNamespace(**fields)).name == name


_DEP4_ARGS = dict(
    tp_size=4,
    nnodes=1,
    disaggregation_mode="decode",
    attn_dp_size=4,
    ep_size=4,
    moe_a2a_backend="flashinfer",
    flashinfer_allreduce_fusion_backend=None,
)


@pytest.fixture
def measured(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({}, id="tp8"),
        pytest.param(dict(disaggregation_mode="decode", ep_size=4), id="tep8"),
        pytest.param(dict(tp_size=4, nnodes=1), id="tp4"),
        pytest.param(dict(tp_size=2, nnodes=1, disaggregation_mode="decode"), id="tp2"),
        pytest.param(
            dict(
                disaggregation_mode="decode",
                attn_dp_size=8,
                ep_size=8,
                enable_dp_lm_head=True,
                moe_runner_backend="flashinfer_cutedsl",
                moe_a2a_backend="flashinfer_nvlink_one_sided",
                flashinfer_allreduce_fusion_backend=None,
            ),
            id="dep8",
        ),
        pytest.param(_DEP4_ARGS, id="dep4"),
    ],
)
def test_the_measured_request_is_accepted_at_every_decode_topology(measured, args):
    # the launcher checks the request before a process reads its GPU and key
    assert runtime.SLOTS > frozenset(runtime.SHIPPED_SLOTS)
    assert _refusal(**args) == ""


_KEY_SHAPES = {"tp8": (8, 1, 1), "tep8": (8, 1, 4), "dep4": (4, 4, 4)}


def _key_slots(name, slots) -> frozenset:
    if name not in _KEY_SHAPES:
        return frozenset(
            slot
            for slot in slots
            if name in topology.SLOT_TOPOLOGIES.get(slot, frozenset())
        )
    return topology.plan(slots, topology.Topology(*_KEY_SHAPES[name]))


def _turned_on() -> frozenset:
    return frozenset().union(
        *(
            _key_slots(name, slots) | (slots & runtime.OPTION_ROWS)
            for (name, _), slots in runtime.MEASURED_PLAN.items()
        )
    ) - frozenset(runtime.SHIPPED_SLOTS)


def _has_row(name: str) -> bool:
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    return name in spec.NUMERIC_TOLERANCES or (
        (name in runtime.DRAFT_SLOTS or name in runtime.OPTION_ROWS)
        and draft.numerics(frozenset({name})).kind is NumericsKind.TOLERANCE
    )


def test_each_slot_the_measured_stack_turns_on_is_labelled_bitwise_or_numeric():
    bitwise, numeric = runtime.MEASURED_BITWISE, runtime.NUMERICS_CHANGE
    assert len(set(bitwise)) == len(bitwise) and len(set(numeric)) == len(numeric)
    assert not set(bitwise) & set(numeric)
    assert set(bitwise) | set(numeric) == _turned_on()
    opt_in = (
        set(runtime.MEASURED_OPT_IN)
        | set(runtime.MEASURED_DEP4)
        | set(runtime.MEASURED_PREFILL)
    )
    assert len(_turned_on()) == (
        len(opt_in)
        + 2
        + len(runtime.MEASURED_TEP8)
        + len(runtime.MEASURED_BT_TAIL)
        + len(runtime.MEASURED_BT_AR)
        + len(runtime.MEASURED_CP_LITE)
        + len(runtime.MEASURED_OPTIONS)
        + len(runtime.MEASURED_TEP8_OPTIONS)
        + len(runtime.MEASURED_DRAFT)
        + len(runtime.MEASURED_DRAFT_TEP8)
        + len(runtime.MEASURED_HOOKS)
    )


def test_each_numeric_slot_has_a_spec_row_except_the_unbounded_ones():
    unbounded = {name for name in runtime.NUMERICS_CHANGE if not _has_row(name)}
    assert unbounded == set(runtime.NUMERICS_UNBOUNDED) == {"hop_mid"}
    assert not any(_has_row(name) for name in runtime.MEASURED_BITWISE)


def test_the_measured_numerics_claim_carries_each_numerics_change_row(measured):
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    from sgl_nvidia_mega.qwen3_5 import draft

    claim = spec.numerics(
        topology.plannable(runtime.SLOTS) | runtime.measured_options()
    )
    assert claim.kind is NumericsKind.TOLERANCE
    assert dict(claim.tolerances) == {
        key: bound
        for name in runtime.NUMERICS_CHANGE
        for key, bound in spec.NUMERIC_TOLERANCES.get(name, {}).items()
    }
    assert dict(draft.numerics(draft._requested()).tolerances) == {
        "draft_logits.rel": 2**-7
    }


def test_slot_gdn_conv_core_takes_gdn_cores_widths_from_29_through_224_rows():
    import msgspec

    every = plan.Facts(
        gdn_verify=True, gdn_pdl=True, gdn_core=True, gdn_conv=True, gdn_conv_core=True
    )
    covered = [
        w for w in range(7, plan.MAX_ROWS + 1, 7) if plan.slots(w, every).gdn_conv_core
    ]
    assert covered == list(range(35, plan.GDN_CONV_CORE_MAX_ROWS + 1, 7))
    assert plan.slots(plan.GDN_CONV_CORE_MIN_ROWS, every).gdn_conv_core
    assert not plan.slots(plan.GDN_CONV_CORE_MIN_ROWS - 1, every).gdn_conv_core
    assert not plan.slots(plan.GDN_CONV_CORE_MAX_ROWS + 1, every).gdn_conv_core
    # past gdn_trt_front's rows, whose front already runs the conv
    assert plan.GDN_CONV_CORE_MIN_ROWS == plan.GDN_TRT_FRONT_MAX_ROWS + 1
    assert plan.GDN_CONV_CORE_MAX_ROWS <= plan.GDN_CORE_MAX_ROWS
    for off in ("gdn_core", "gdn_conv", "gdn_verify"):
        facts = msgspec.structs.replace(every, **{off: False})
        assert not any(plan.slots(w, facts).gdn_conv_core for w in range(29, 225))


def test_slot_gdn_conv_core_claims_bitwise_numerics():
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    slots = frozenset({"gdn_verify", "gdn_core", "gdn_conv"})
    assert "gdn_conv_core" not in spec.NUMERIC_TOLERANCES
    assert spec.numerics(slots | {"gdn_conv_core"}) == spec.numerics(slots)
    assert spec.numerics(frozenset({"gdn_conv_core"})).kind is NumericsKind.BITWISE


@pytest.mark.parametrize(
    "shape, conv_core",
    [
        ((2,), True),
        ((4,), True),
        ((8,), True),
        ((8, 1, 4), True),
        ((8, 8, 8), False),
    ],
)
def test_the_gdn_core_variants_run_on_their_topologies(shape, conv_core):
    slots = frozenset({"gdn_verify", "gdn_core", "gdn_conv", "gdn_conv_core"})
    selected = topology.select(slots, topology.Topology(*shape))
    assert ("gdn_conv_core" in selected) is conv_core
    assert ("gdn_conv_core" in selected) == ("gdn_core" in selected)
    assert runtime.slot_reasons(slots) == []


def test_the_gdn_core_variants_need_gdn_core():
    conv_core = (
        f"{runtime.SLOTS_ENV}: slot 'gdn_conv_core' runs slot 'gdn_conv''s conv in "
        "slot 'gdn_core''s launch"
    )
    for slots in (
        {"gdn_verify", "gdn_core", "gdn_conv_core"},
        {"gdn_verify", "gdn_conv", "gdn_conv_core"},
    ):
        assert runtime.slot_reasons(frozenset(slots)) == [conv_core]


@pytest.mark.parametrize(
    "overlap", ["enable_two_batch_overlap", "enable_single_batch_overlap"]
)
def test_slot_gdn_conv_core_never_runs_under_batch_overlap(monkeypatch, overlap):
    monkeypatch.setattr(
        runtime,
        "SLOTS",
        frozenset({"gdn_verify", "gdn_core", "gdn_conv", "gdn_conv_core"}),
    )
    assert registry.Feature.TWO_BATCH_OVERLAP not in spec.SPEC.features
    message = _refusal(**{overlap: True})
    assert "overlap is in use and Qwen3.5 does not accept it" in message, message


def test_slot_lowm_ba_takes_84_through_112_rows_of_gdn_verify():
    every = plan.Facts(gdn_verify=True, lowm_ba=True)
    covered = [
        w for w in range(7, plan.MAX_ROWS + 1, 7) if plan.slots(w, every).lowm_ba
    ]
    assert covered == [84, 91, 98, 105, 112]
    assert not any(
        plan.slots(w, plan.Facts(lowm_ba=True)).lowm_ba for w in range(84, 113)
    )


def test_slot_lowm_ba_claims_tolerance_numerics_on_the_tp8_shard(monkeypatch):
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

    claim = spec.numerics(frozenset({"gdn_verify", "lowm_ba"}))
    assert claim.kind is NumericsKind.TOLERANCE
    slots = frozenset({"gdn_verify", "lowm_ba"})
    for shape, on in (((8,), True), ((8, 1, 4), True), ((4,), False), ((2,), False)):
        assert ("lowm_ba" in topology.select(slots, topology.Topology(*shape))) is on
    assert runtime.slot_reasons(slots) == []
    assert runtime.slot_reasons(frozenset({"lowm_ba"})) == [
        f"{runtime.SLOTS_ENV}: slot 'lowm_ba' runs slot 'gdn_verify''s in_proj_ba"
    ]


def test_dep4_alone_keeps_the_upstream_refusal(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", raising=False)
    message = _refusal(**_DEP4_ARGS)
    assert "Qwen3.5 DP attention requires DEP8" in message, message
    assert "moe_a2a_backend 'flashinfer' is not 'none'" in message, message
    assert "flashinfer_allreduce_fusion_backend None is not 'cutedsl'" in message


def test_a_slot_list_keeps_the_dep4_refusal(measured, monkeypatch):
    monkeypatch.setenv(runtime.SLOTS_ENV, "gdn_verify,gdn_core")
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    assert "Qwen3.5 DP attention requires DEP8" in _refusal(**_DEP4_ARGS)


@pytest.mark.parametrize(
    "change, reason",
    [
        pytest.param(
            dict(moe_runner_backend="flashinfer_cutedsl"),
            "moe_runner_backend 'flashinfer_cutedsl' is not 'flashinfer_trtllm'",
            id="runner",
        ),
        pytest.param(
            dict(tp_size=8, nnodes=2, attn_dp_size=4, ep_size=8),
            "Qwen3.5 DP attention requires DEP8",
            id="dp4-of-tp8",
        ),
        pytest.param(
            dict(attn_dp_size=2, ep_size=4),
            "Qwen3.5 DP attention requires DEP8",
            id="dp2",
        ),
    ],
)
def test_the_measured_stack_admits_dep4_only(measured, change, reason):
    """Only DEP4's own shape is admitted, with its MoE runner pin."""
    assert reason in _refusal(**{**_DEP4_ARGS, **change})


def test_dep4_runs_only_its_declared_slots():
    dep4, dep8 = topology.Topology(4, 4, 4), topology.Topology(8, 8, 8)
    every = frozenset(topology.SLOT_TOPOLOGIES)
    assert dep4.name == "dep4" and dep4.stock_widths
    assert topology.select(every, dep4) == topology.DEP4_SLOTS
    assert topology.plan(every, dep4) == topology.DEP4_SLOTS
    # DEP8 and the TP8 / TEP8 keys declare none of DEP4's new rows
    new = {"gdn_core_tc", "gdn_commit_fold", "gathered_cdf"}
    for other in (dep8, topology.Topology(8), topology.Topology(8, 1, 4)):
        assert not topology.select(every, other) & new
    assert topology.select(every, dep8) == {
        "gdn_verify",
        "gdn_pdl",
        "gdn_conv",
        "mka",
        "multi_request",
        "sharded_gumbel",
    }


def test_dep4_rows_need_the_slots_they_run_in(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"gdn_verify", "gdn_core_tc"}))
    assert "slot 'gdn_core_tc' runs slot 'gdn_core''s launch" in " ".join(
        runtime.slot_reasons()
    )
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"gdn_verify", "gdn_core", "gdn_commit_fold"})
    )
    assert "slot 'gdn_commit_fold' folds the commit into slot 'gdn_core_tc''s" in (
        " ".join(runtime.slot_reasons())
    )
    monkeypatch.setattr(runtime, "SLOTS", topology.DEP4_SLOTS)
    assert runtime.slot_reasons() == []
