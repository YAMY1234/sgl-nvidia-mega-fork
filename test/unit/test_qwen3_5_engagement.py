# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt import runtime_context
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.kernel_plugins.nvidia import binding as guards
from sglang.srt.runtime_context import derive_attn_tp_size
from sglang.srt.server_args import ServerArgs
from test_qwen3_5_spec import _DEP4_ARGS, DOCUMENTED, PREFILL

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.qwen3_5 import TEXT_CONFIG, plan, runtime, spec, topology

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"

_DEPLOYMENTS = {
    "dep4-decode": dict(_DEP4_ARGS, disable_radix_cache=True),
    "tep8-decode": dict(nnodes=2, disaggregation_mode="decode", ep_size=4),
    "tp8-decode": dict(nnodes=2, disaggregation_mode="decode"),
    "tp1-pp2-prefill": dict(PREFILL, pp_size=2),
}
_GDN_CORE = ("gdn_verify", "gdn_core")
_PIECES = {
    "gdn_core_slots": _GDN_CORE,
    "measured_slot_list": tuple(
        sorted(set(runtime.MEASURED_OPT_IN) - runtime.EP_STOCK_SLOTS - set(_GDN_CORE))
    ),
    "dense_moe_slots": tuple(
        sorted(set(runtime.MEASURED_OPT_IN) & runtime.EP_STOCK_SLOTS)
    ),
    "route_lin": ("route_lin",),
    "tep8_routed_body": runtime.MEASURED_TEP8,
    "bt_tail": runtime.MEASURED_BT_TAIL,
    "cp_lite_variants": runtime.MEASURED_CP_LITE,
    "bt_all_reduces": runtime.MEASURED_BT_AR,
    "gdn_options": ("gdn_value_split", "gemm_tactics"),
    "draft_slots": runtime.MEASURED_DRAFT,
    "draft_tep8_slots": runtime.MEASURED_DRAFT_TEP8,
    "draft_options": (
        "embed_local",
        "draft_top1_cfg",
        "draft_oproj_push",
        "draft_early_hops",
        "draft_head_keys",
        "draft_merge_early",
        "draft_hilo_trig",
    ),
    "draft_tep8_options": runtime.MEASURED_TEP8_OPTIONS,
    "verify_keys": runtime.MEASURED_HOOKS,
    "gdn_commit": ("gdn_commit",),
    "dep4_rows": ("gdn_core_tc", "gdn_commit_fold", "gathered_cdf"),
    # the TP1/PP2 prefill's changes, each its own slot
    "prefill_front": ("gdn_prefill_front",),
    "prefill_track_snapshot": ("gdn_prefill_track_snapshot",),
    "prefill_moe_input_nvfp4": ("moe_prefill_input_nvfp4",),
    "prefill_moe_add_norm": ("moe_prefill_add_norm",),
    "prefill_moe_noclone": ("moe_prefill_noclone",),
    "prefill_gate_fp8": ("prefill_gate_fp8",),
    "prefill_act_fp8": ("prefill_act_fp8",),
    "prefill_gdn_norm_fp8": ("gdn_prefill_norm_fp8",),
    "prefill_norm_fp8": ("prefill_norm_fp8",),
    "prefill_qk_rows": ("attn_prefill_qk_rows",),
}
_TP8_AND_TEP8 = {
    "gdn_core_slots",
    "measured_slot_list",
    "bt_tail",
    "cp_lite_variants",
    "cp_lite_constants",
    "gdn_options",
    "draft_slots",
    "draft_options",
    "verify_keys",
    "gdn_commit",
}
_TABLE = {
    "dep4-decode": ("dep4", "dep4", {"gdn_core_slots", "dep4_rows"}),
    "tep8-decode": (
        "tep8",
        "tep8",
        _TP8_AND_TEP8
        | {
            "tep8_routed_body",
            "bt_all_reduces",
            "draft_tep8_slots",
            "draft_tep8_options",
        },
    ),
    "tp8-decode": ("tp8", "tp8", _TP8_AND_TEP8 | {"dense_moe_slots", "route_lin"}),
    "tp1-pp2-prefill": (
        "tp1",
        "p_tp1_pp2",
        {name for name in _PIECES if name.startswith("prefill_")},
    ),
}


@pytest.fixture(autouse=True)
def measured_stack(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    monkeypatch.setattr(registry, "_REGISTRY", {})
    plugin.register()
    monkeypatch.setenv(RUNTIME_ENV, "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.delenv("SGLANG_ENABLE_EMBED_REPLICATION", raising=False)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    monkeypatch.setattr(runtime, "MEASURED_KEY", None)
    monkeypatch.setattr(plugin_binding, "megakernel_agree", lambda *a, **kw: None)
    monkeypatch.setattr(topology, "device_sku", _no_gpu)


def _no_gpu():
    raise AssertionError("read the GPU")


def _started(args: dict):
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(**TEXT_CONFIG),
    )
    with (
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value="sm_103"),
        patch.object(guards, "model_config_of") as model_config,
    ):
        model_config.return_value = SimpleNamespace(
            hf_config=hf_config, dtype="bfloat16"
        )
        server_args = ServerArgs(**{**DOCUMENTED, **args})
        guards.handle_nvidia_megakernel(server_args)
    return resolving_view(server_args)


def _built(monkeypatch, cfg) -> frozenset:
    """The build's slot steps at the parallel state these arguments start."""
    parallel = SimpleNamespace(
        tp_size=cfg.tp_size,
        attn_tp_size=derive_attn_tp_size(
            tp_size=cfg.tp_size,
            attn_cp_size=cfg.attn_cp_size,
            attn_dp_size=cfg.attn_dp_size or 1,
        ),
        attn_dp_size=cfg.attn_dp_size or 1,
        moe_ep_size=cfg.ep_size or 1,
        pp_size=cfg.pp_size,
        tp_group=SimpleNamespace(world_size=cfg.tp_size),
    )
    # the startup check narrows the request; a build process starts from it again
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    runtime.apply_measured_plan(parallel, topology.GB300)
    if parallel.pp_size > 1:
        # a PD prefill runs the prefill slots declared at its key
        monkeypatch.setattr(
            runtime,
            "SLOTS",
            runtime.SLOTS & set(runtime.declared_prefill_slots(parallel)),
        )
        return runtime.SLOTS
    runtime.apply_expert_parallel(parallel.moe_ep_size)
    runtime.apply_cp_lite_topology(parallel.attn_tp_size)
    runtime.apply_topology(parallel)
    tokens = cfg.speculative_num_draft_tokens
    offered = tuple(range(tokens, plan.MAX_ROWS + 1, tokens))
    current = topology.from_parallel(parallel)
    multi = current.stock_widths or plan.MULTI_REQUEST in runtime.SLOTS
    if multi and not current.stock_widths:
        multi = plan.multi_request(parallel.tp_size, runtime.SLOTS)
    memory = SimpleNamespace(
        disable_radix_cache=cfg.disable_radix_cache,
        enable_unified_memory=cfg.enable_unified_memory,
    )
    monkeypatch.setattr(runtime_context, "get_memory", lambda: memory)
    monkeypatch.setattr(runtime, "_offers_recurrent_commit", lambda: True)
    runtime.apply_commit_fold(offered, plan.candidates(offered, tokens, multi))
    return runtime.SLOTS


def _on(row: str) -> bool:
    if row in runtime.OPTION_ROWS:
        return runtime.measured_option(row)
    return row in runtime.SLOTS


def _engaged() -> set:
    """The changes engaged in this process; each turns all of its rows on, or none."""
    engaged = set()
    for name, rows in _PIECES.items():
        on = {row for row in rows if _on(row)}
        assert on in (set(), set(rows)), (name, sorted(on))
        if on:
            engaged.add(name)
    if runtime._cp_lite_constants_on():
        engaged.add("cp_lite_constants")
    return engaged


def test_the_table_places_every_row_a_measured_plan_turns_on():
    rows = [row for piece in _PIECES.values() for row in piece]
    assert all(_PIECES.values()) and len(rows) == len(set(rows))
    every = set().union(*runtime.MEASURED_PLAN.values()) - set(runtime.SHIPPED_SLOTS)
    assert set(rows) == every | {"route_lin"}


@pytest.mark.parametrize("deployment", sorted(_DEPLOYMENTS))
def test_each_deployment_engages_its_row_of_the_table(monkeypatch, caplog, deployment):
    configured, key, want = _TABLE[deployment]
    cfg = _started(_DEPLOYMENTS[deployment])
    assert topology.from_config(cfg).name == configured
    with caplog.at_level("INFO"):
        _built(monkeypatch, cfg)
    assert runtime.MEASURED_KEY == (key, topology.GB300)
    assert _engaged() == want
    (line,) = [r.message for r in caplog.records if "measured plan at" in r.message]
    assert line.startswith(f"NVIDIA megakernel: Qwen3.5 measured plan at {key} on")
    named = {row for name in want & set(_PIECES) for row in _PIECES[name]}
    assert named and all(f"'{row}'" in line for row in named), line


def test_the_dep4_decode_folds_the_commit_only_with_the_radix_cache_off(monkeypatch):
    _built(monkeypatch, _started(_DEP4_ARGS))
    assert not _on("gdn_commit_fold")
    assert _on("gdn_core_tc") and _on("gathered_cdf")
