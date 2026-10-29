# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 megakernel's runtime on CPU: the steps it covers, the binding and the
EAGLE entries at build, the MTP draft spec, the step state and the parity check."""

from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.qwen3_5 import (
    runtime,
)
from sgl_nvidia_mega.qwen3_5.runtime import executor, hooks, parity
from sgl_nvidia_mega.qwen3_5.runtime.state import StepState


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    monkeypatch.setattr(
        plugin_binding, "megakernel_agree", lambda *args, **kwargs: None
    )
    plugin.register()
    # the target and draft builds read moe_ep_size (expert parallelism); no process groups here
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=8, attn_tp_size=8, moe_ep_size=1),
    )


# ---------------------------------------------------------------- the covered steps


def _verify(widths=(7,), binding="binding"):
    """An executor over a model with no layers (plan reads only host state), built
    with SGLang's EAGLE binding."""
    lm = SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))
    return executor.Qwen35Verify(lm, {}, {}, widths, binding=binding)


def _batch(verify=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: verify)
    )


def test_the_default_policy_ships_only_bitwise_shape_bounded_paths(monkeypatch):
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    assert runtime._requested_slots() == frozenset(
        {
            "route_topk",
            "mkm_m32",
            "mkm_m64",
            "mkm_m128",
            "mkm_m256",
            "mkm_m512",
            "hop",
            "hop_fp8",
            "multi_request",
            "draft_loop",
            "draft_extend",
            "draft_multi",
            "draft_embed",
        }
    )
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    assert runtime.slot_reasons() == []


def test_an_explicit_empty_policy_keeps_every_slot_off(monkeypatch):
    monkeypatch.setenv(runtime.SLOTS_ENV, "")
    assert runtime._requested_slots() == frozenset()


def test_cp_slots_build_the_plugin_state_before_validation(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    built = object()
    calls = []
    agreements = []
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_q1_cp"}))
    monkeypatch.setattr(
        plugin_binding,
        "megakernel_agree",
        lambda value, *, what: agreements.append((value, what)),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        CpLite,
        "maybe_create",
        lambda binding, device: calls.append((binding, device)) or built,
    )

    binding = SimpleNamespace(page_size=64, tp_size=8)
    runtime.build_cp_lite(binding)

    assert agreements == [
        (("draft_q1_cp",), "Qwen3.5 CP slots requested"),
        ((64, 8), "Qwen3.5 CP page size and TP size"),
    ]
    assert calls == [(binding, torch.device("cuda", 3))]


def test_cp_slots_refuse_an_incompatible_page_or_topology(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2_cp"}))
    monkeypatch.setattr(
        plugin_binding, "megakernel_agree", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="page size 64"):
        runtime.build_cp_lite(SimpleNamespace(page_size=32, tp_size=8))

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(CpLite, "maybe_create", lambda binding, device: None)
    with pytest.raises(RuntimeError, match="Q/KV-head topology"):
        runtime.build_cp_lite(SimpleNamespace(page_size=64, tp_size=8))


@pytest.mark.parametrize("attention_tp_size", [1, 2, 3, 6, 32])
def test_unsupported_attention_topology_drops_only_cp_slots(
    monkeypatch, caplog, attention_tp_size
):
    requested = runtime.CP_LITE_SLOTS | {"skv2"}
    monkeypatch.setattr(runtime, "SLOTS", requested)

    with caplog.at_level("INFO"):
        dropped = runtime.apply_cp_lite_topology(attention_tp_size)

    assert dropped == runtime.CP_LITE_SLOTS
    assert runtime.SLOTS == frozenset({"skv2"})
    assert f"attention-TP {attention_tp_size}" in caplog.text


@pytest.mark.parametrize("attention_tp_size", [4, 8, 16])
def test_supported_attention_topology_keeps_cp_slots(monkeypatch, attention_tp_size):
    requested = runtime.CP_LITE_SLOTS | {"skv2"}
    monkeypatch.setattr(runtime, "SLOTS", requested)

    assert runtime.apply_cp_lite_topology(attention_tp_size) == frozenset()
    assert runtime.SLOTS == requested


def test_config_filter_derives_attention_tp_before_publication(monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import spec as target

    calls = []
    monkeypatch.setattr(
        runtime,
        "apply_expert_parallel",
        lambda ep_size: calls.append(("ep", ep_size)) or frozenset(),
    )
    monkeypatch.setattr(
        runtime,
        "apply_cp_lite_topology",
        lambda attention_tp_size: (
            calls.append(("attention_tp", attention_tp_size)) or frozenset()
        ),
    )
    monkeypatch.setattr(target, "text_config_reasons", lambda hf_config: [])
    monkeypatch.setattr(target, "binding_reasons", lambda: [])
    monkeypatch.setattr(runtime, "slot_reasons", lambda: [])
    monkeypatch.setattr(target, "pd_reasons", lambda cfg: [])
    cfg = SimpleNamespace(tp_size=8, attn_cp_size=1, attn_dp_size=8, ep_size=8)

    assert not hasattr(cfg, "attn_tp_size")
    assert target._config_disable_reasons(cfg, SimpleNamespace()) == []
    assert calls == [("ep", 8), ("attention_tp", 1)]


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 6, 8, 16, 32])
def test_cp_topology_predicate_matches_state_creation(monkeypatch, tp):
    from sgl_nvidia_mega.qwen3_5 import cp_lite_topology
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    made = []
    binding = SimpleNamespace(tp_size=tp, tp_rank=0, tp_device_group=object())
    monkeypatch.setattr(CpLite, "_instance", None)
    monkeypatch.setattr(CpLite, "__init__", lambda self, *args: made.append(args))

    state = CpLite.maybe_create(binding, torch.device("cuda", 0))

    assert (state is not None) == (cp_lite_topology(tp) is not None)
    assert bool(made) == (cp_lite_topology(tp) is not None)


@pytest.mark.parametrize(
    "tp, rank, want",
    [(4, 3, (3, 4, 2, 8)), (8, 7, (7, 8, 4, 4))],
)
def test_cp_state_uses_the_pinned_qwen35_head_topology(monkeypatch, tp, rank, want):
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    calls = []
    group = object()
    binding = SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        tp_device_group=group,
    )
    monkeypatch.setattr(CpLite, "_instance", None)
    monkeypatch.setattr(
        CpLite,
        "__init__",
        lambda self, got_group, got_rank, got_tp, cp, hq, device: calls.append(
            (got_group, got_rank, got_tp, cp, hq, device)
        ),
    )

    made = CpLite.maybe_create(binding, torch.device("cuda", 0))

    assert isinstance(made, CpLite)
    assert calls == [(group, *want, torch.device("cuda", 0))]


def test_requested_cp_state_does_not_restore_a_device_length_gate():
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    cp = CpLite.__new__(CpLite)
    cp.min_kv = {}
    cp.gate_max = False

    assert [cp._gate_thr(batch) for batch in (2, 4, 16)] == [None, None, None]


def test_cp_exact_selector_keeps_r11s_unbalanced_batch_cap():
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as fused
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    cp = CpLite.__new__(CpLite)
    cp.cp = 4
    cp.hq = 4
    cp.sms = 132

    assert (fused.MAX_BATCH, fused.MAX_BATCH_Q1) == (32, 32)
    assert [cp._exact_supported(7, batch) for batch in (32, 64, 128)] == [
        True,
        False,
        False,
    ]
    assert [cp._exact_supported(1, batch) for batch in (32, 64)] == [True, False]


def test_cp_gate_epilogue_offer_is_consumed_once():
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    cp = CpLite.__new__(CpLite)
    cp._gate_epilogue = None
    cp._gate_epilogue_done = False
    gate, scale, out = object(), object(), object()

    cp.offer_gate_epilogue(gate, scale, out)
    assert cp._gate_epilogue == (gate, scale, out)
    assert not cp.take_gate_epilogue()
    assert cp._gate_epilogue is None

    cp.offer_gate_epilogue(gate, scale, out)
    cp._gate_epilogue_done = True
    assert cp.take_gate_epilogue()
    assert not cp.take_gate_epilogue()


def test_hook_slots_are_accepted_but_build_no_verify_width(monkeypatch):
    requested = frozenset({"sharded_gumbel"})
    monkeypatch.setattr(runtime, "SLOTS", requested)
    assert runtime.slot_reasons() == []

    factory = object()
    calls = []
    monkeypatch.setattr(
        hooks,
        "build",
        lambda model, binding: calls.append((model, binding)) or factory,
    )

    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    built = runtime.build(model, ctx)
    assert built.summary() == "slots none (stock)"
    assert built.speculative_runtime_hooks is factory
    assert calls == [(model, "EagleBinding")]


def test_draft_fin_owns_the_exact_bf16_finalize_policy():
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    from sgl_nvidia_mega.qwen3_5.draft import finalize

    method = object.__new__(UnquantizedFusedMoEMethod)
    experts = SimpleNamespace(
        is_draft=True,
        quant_method=method,
        params_dtype=torch.bfloat16,
        hidden_size=4096,
        num_experts=512,
        top_k=10,
        runner_backend=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
        supports_deferred_finalize=False,
    )
    assert (
        finalize.enable(SimpleNamespace(experts=experts, shared_expert=object())) == ""
    )
    assert experts.supports_deferred_finalize


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"is_draft": False}, "not a draft MoE"),
        ({"quant_method": object()}, "quant method object"),
        ({"params_dtype": torch.float16}, "params dtype torch.float16"),
        ({"hidden_size": 2048}, "hidden size 2048"),
        ({"num_experts": 256}, "experts 256"),
        ({"top_k": 8}, "top-k 8"),
        ({"shared_expert": None}, "no separate shared expert"),
        (
            {"runner_backend": SimpleNamespace(is_flashinfer_trtllm=lambda: False)},
            "MoE runner",
        ),
    ],
)
def test_draft_fin_refuses_every_unvalidated_policy_dimension(change, reason):
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    from sgl_nvidia_mega.qwen3_5.draft import finalize

    change = dict(change)
    experts = SimpleNamespace(
        is_draft=True,
        quant_method=object.__new__(UnquantizedFusedMoEMethod),
        params_dtype=torch.bfloat16,
        hidden_size=4096,
        num_experts=512,
        top_k=10,
        runner_backend=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
        supports_deferred_finalize=False,
    )
    shared_expert = change.pop("shared_expert", object())
    for name, value in change.items():
        setattr(experts, name, value)
    got = finalize.enable(SimpleNamespace(experts=experts, shared_expert=shared_expert))
    assert reason in got
    assert not experts.supports_deferred_finalize


def test_multi_request_coverage_has_no_kernel_build_result(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"multi_request"}))
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=(7, 14), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )

    built = runtime.build(model, ctx)
    assert built.widths == (7,)


def test_only_target_verify_steps_of_built_widths_are_covered():
    built = _verify()
    ids = torch.zeros(7, dtype=torch.int64)
    assert built.plan(ids, _batch()) == Step(mode="target_verify", rows=7)
    assert built.plan(ids, _batch(), pp_proxy_tensors=None) == Step(
        mode="target_verify", rows=7
    )
    assert built.plan(torch.zeros(14, dtype=torch.int64), _batch()) is None
    assert built.plan(ids, _batch(verify=False)) is None
    assert built.plan(ids, _batch(), get_embedding=True) is None
    assert built.plan(ids, _batch(), pp_proxy_tensors=object()) is None
    # an argument the executor does not know leaves the step to SGLang
    assert built.plan(ids, _batch(), input_embeds=None) is None


def test_the_covered_widths_are_the_ones_the_build_returns():
    built = _verify(widths=(7, 14, 28))
    for rows in (7, 14, 28):
        ids = torch.zeros(rows, dtype=torch.int64)
        assert built.plan(ids, _batch()) == Step(mode="target_verify", rows=rows)
    assert built.plan(torch.zeros(21, dtype=torch.int64), _batch()) is None


@pytest.mark.parametrize(
    "tp_size, want", [(8, (7,)), (4, (7, 14, 21, 28)), (2, (7, 14, 21, 28))]
)
def test_the_build_covers_more_requests_only_where_a_slot_runs(
    monkeypatch, tp_size, want
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=tp_size, attn_tp_size=tp_size, moe_ep_size=1),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    # what SGLang offers for the spec's MAX_ROWS with 7 rows a request
    ctx = SimpleNamespace(
        widths=(7, 14, 21, 28),
        tokens_per_request=7,
        deadline_s=60.0,
        binding="EagleBinding",
    )
    built = runtime.build(model, ctx)
    assert built.widths == want and built.summary().startswith("slots none (stock)")


def test_the_build_hands_over_the_binding_and_ships_the_bitwise_draft_entries():
    from sglang.kernels.plugin_kernels.nvidia.spec import ENTRIES

    from sgl_nvidia_mega.qwen3_5 import draft

    built = _verify()
    assert built.binding == "binding"
    # covered from the first step on: the binding comes with the build
    ids = torch.zeros(7, dtype=torch.int64)
    assert built.plan(ids, _batch()) == Step(mode="target_verify", rows=7)
    target, mtp = built, draft.SPEC.build("lm", None, _draft_ctx())
    assert [n for n in ENTRIES if getattr(target, n, None) is not None] == []
    assert [n for n in ENTRIES if getattr(mtp, n, None) is not None] == [
        "draft_loop",
        "draft_extend",
        "begin_draft_round",
    ]


def test_verify_sample_is_refused_with_the_switch_that_replaces_it(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"route", "verify_sample"}))
    assert [r.split(": ", 1)[1] for r in runtime.slot_reasons()] == [
        "'verify_sample' is no longer an entry: drop it; an SGLang with post-verify "
        "graphs turns them on with SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1"
    ]


def test_expert_parallel_keeps_route_and_boundary_slots(monkeypatch):
    requested = frozenset(
        {
            "route",
            "route_topk",
            "mkm",
            "mkm_wide",
            "mkm_m32",
            "mkm_m64",
            "mkm_m128",
            "mkm_m256",
            "mkm_m512",
            "mkm_wide_bal",
            "hop",
            "gdn_push",
            "draft_moe",
            "draft_hop",
        }
    )
    monkeypatch.setattr(runtime, "SLOTS", requested)
    assert runtime.apply_expert_parallel(1) == frozenset()
    assert runtime.SLOTS == requested
    assert runtime.apply_expert_parallel(8) == frozenset(
        {
            "mkm",
            "mkm_wide",
            "mkm_m32",
            "mkm_m64",
            "mkm_m128",
            "mkm_m256",
            "mkm_m512",
            "mkm_wide_bal",
            "draft_moe",
        }
    )
    assert runtime.SLOTS == frozenset(
        {"route", "route_topk", "hop", "gdn_push", "draft_hop"}
    )


def test_the_runtime_gives_the_executor_the_build_contexts_binding(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    built = runtime.build(model, ctx)
    assert built.binding == "EagleBinding" and built.summary() == "slots none (stock)"


def test_slot_skv2_cp_builds_as_a_graph_variant_of_slot_skv2(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp"}))
    monkeypatch.setattr(runtime, "build_cp_lite", lambda binding: None)
    monkeypatch.setattr(split_kv, "cp_reason", lambda: "")
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    assert runtime.slot_reasons() == []
    built = runtime.build(model, ctx)
    variants = built.graph_variants["target_verify"]
    assert variants.capture_labels == ("split_kv_stock", "split_kv_long", "split_kv_cp")


def _draft_ctx():
    return SimpleNamespace(
        widths=(), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )


def test_the_mtp_draft_covers_no_step():
    from sglang.kernels.plugin_kernels.nvidia.spec import ENTRIES

    from sgl_nvidia_mega.qwen3_5 import draft

    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert (
        draft.SPEC.max_rows == 0 and built.widths == () and built.draft_sampler is None
    )
    assert built.plan(torch.zeros(7, dtype=torch.int64), _batch()) is None
    assert (draft.ARCHITECTURE, "sgl_nvidia_mega.qwen3_5.draft:SPEC") in plugin.SPECS
    # Entries replace SGLang's draft graphs, not a model-forward width.
    assert [n for n in ENTRIES if getattr(built, n, None) is not None] == [
        "draft_loop",
        "draft_extend",
        "begin_draft_round",
    ]


def test_an_explicit_policy_controls_the_draft_entries(monkeypatch):
    from sglang.kernels.plugin_kernels.nvidia.spec import ENTRIES

    from sgl_nvidia_mega.qwen3_5 import draft

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_loop", "draft_extend"}))
    assert runtime.slot_reasons() == []
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert [n for n in ENTRIES if getattr(built, n, None) is not None] == [
        "draft_loop",
        "draft_extend",
        "begin_draft_round",
    ]
    assert built.context_len == 4096 and built.binding == "EagleBinding"
    # the target's build ignores the draft's names
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    target_ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    assert runtime.build(model, target_ctx).summary() == "slots none (stock)"


def test_unavailable_draft_moe_is_dropped_and_runs_stock_mlp(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.draft import attention, layer
    from sgl_nvidia_mega.qwen3_5.draft.moe import DraftMoe
    from sgl_nvidia_mega.qwen3_5.draft.parts import Parts

    calls = []

    def mlp(hidden):
        calls.append("stock mlp")
        return hidden

    decoder = SimpleNamespace(
        input_layernorm=lambda hidden: hidden,
        post_attention_layernorm=lambda hidden, residual: (hidden, residual),
        mlp=mlp,
    )
    norm = lambda hidden, residual: (hidden, residual)
    model = SimpleNamespace(layers=[decoder], start_layer=0, end_layer=1, norm=norm)
    draft = SimpleNamespace(
        lm=SimpleNamespace(model=model),
        binding="binding",
        slots=frozenset({"draft_moe"}),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        attention, "Attention", lambda _: lambda hidden, *a, **k: hidden
    )
    monkeypatch.setattr(DraftMoe, "build", classmethod(lambda cls, mlp, device: None))
    monkeypatch.setattr(
        "sglang.srt.distributed.tensor_model_parallel_all_reduce", lambda hidden: hidden
    )

    parts = Parts.build(draft)
    assert parts.moe is None and "draft_moe" not in parts.slots
    x = torch.ones(1, 4)
    assert torch.equal(
        layer.layer(parts, x, None, None, None, None, rows_per_request=1), x
    )
    assert calls == ["stock mlp"]


@pytest.mark.parametrize(
    "batch_size, lower_bound, seq_lens, want",
    [
        (1, 16384, None, "q1_long"),
        (1, 16383, None, "q1_stock"),
        (1, None, [20000], "q1_long"),
        (1, None, [100], "q1_stock"),
        (1, None, None, "q1_stock"),
        (2, 50000, None, "q1_stock"),
    ],
)
def test_the_draft_loop_routes_q1_by_kv_length_as_2_2(
    monkeypatch, batch_size, lower_bound, seq_lens, want
):
    from sgl_nvidia_mega.qwen3_5 import draft

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_loop", "draft_q1"}))
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    variants = built.graph_variants["draft_loop"]
    assert variants.capture_labels == ("q1_stock", "q1_long")
    # past max_batch_size every batch takes q1_stock, so SGLang captures one graph there
    assert variants.max_batch_size == 1
    long = SimpleNamespace(attention_kv_len_lower_bound=10**6, seq_lens_cpu=None)
    for bs in range(variants.max_batch_size + 1, 129):
        long.batch_size = bs
        assert variants.select(long) == "q1_stock"
    batch = SimpleNamespace(
        batch_size=batch_size,
        attention_kv_len_lower_bound=lower_bound,
        seq_lens_cpu=None if seq_lens is None else torch.tensor(seq_lens),
    )
    assert variants.select(batch) == want


def test_the_draft_extend_takes_the_split_kv_variants_with_slot_draft_skv2(
    monkeypatch,
):
    from sgl_nvidia_mega.qwen3_5 import draft
    from sgl_nvidia_mega.qwen3_5.attention import route

    # inside entry draft_extend only: the draft loop is not needed
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_extend", "draft_skv2"}))
    assert runtime.slot_reasons() == []
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert built.graph_variants == {"draft_extend": route.EXTEND}
    assert route.EXTEND.min_length == 32768
    # with draft_q1 the draft offers each runner its own variants
    monkeypatch.setattr(
        runtime,
        "SLOTS",
        frozenset({"draft_loop", "draft_q1", "draft_extend", "draft_skv2"}),
    )
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert sorted(built.graph_variants) == ["draft_extend", "draft_loop"]
    assert built.graph_variants["draft_extend"] is route.EXTEND


def test_the_cp_slots_offer_the_draft_runners_a_cp_variant(monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import draft
    from sgl_nvidia_mega.qwen3_5.attention import route

    slots = {"draft_loop", "draft_q1", "draft_q1_cp", "draft_extend", "draft_skv2"}
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots | {"draft_skv2_cp"}))
    monkeypatch.setattr(runtime, "build_cp_lite", lambda binding: None)
    assert runtime.slot_reasons() == []
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    loop = built.graph_variants["draft_loop"]
    extend = built.graph_variants["draft_extend"]
    assert loop.capture_labels == ("q1_stock", "q1_long", "q1_cp")
    assert extend.capture_labels == (
        "split_kv_stock",
        "split_kv_long",
        "split_kv_cp",
    )

    def one(n):
        return SimpleNamespace(
            batch_size=1, attention_kv_len_lower_bound=n, seq_lens_cpu=None
        )

    draft_cross = route.CP_MIN_KV["draft"]
    extend_cross = route.CP_MIN_KV["extend"]
    assert [
        loop.select(one(n)) for n in (16383, 16384, draft_cross - 1, draft_cross)
    ] == [
        "q1_stock",
        "q1_long",
        "q1_long",
        "q1_cp",
    ]
    assert [
        extend.select(one(n)) for n in (32767, 32768, extend_cross - 1, extend_cross)
    ] == ["split_kv_stock", "split_kv_long", "split_kv_long", "split_kv_cp"]

    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"draft_loop", "draft_q1_cp", "skv2_cp"})
    )
    assert sorted(r.split(": ", 1)[1] for r in runtime.slot_reasons()) == [
        "slot 'draft_q1_cp' takes slot 'draft_q1''s long-KV graph variant's place "
        "from its length up",
        "slot 'skv2_cp' takes slot 'skv2''s long-KV graph variant's place from its "
        "length up",
    ]


@pytest.mark.parametrize(
    "rows,label,requests,min_bs",
    [
        (1, "q1_cp", 1, 1),
        (7, "split_kv_cp", 1, 1),
        (1, "q1_stock", 4, None),
        (7, "split_kv_stock", 4, None),
    ],
)
def test_the_draft_decode_keeps_cp_lite_for_cp_variants_and_multi_request(
    monkeypatch, rows, label, requests, min_bs
):
    from sgl_nvidia_mega.qwen3_5.attention import route, split_kv
    from sgl_nvidia_mega.qwen3_5.draft import attention as draft_attention

    heads, dim, page = 4, 256, 64
    calls = []

    class CpLite:
        def attend(
            self, q, kv, pages, lens, q_len, bmm1, bmm2, ws, layer, owner, min_bs=None
        ):
            calls.append((tuple(q.shape), q_len, bmm1, bmm2, min_bs))
            return torch.ones(q.shape[0], heads, dim, dtype=torch.bfloat16)

    monkeypatch.setattr(split_kv, "cp_lite", CpLite)
    monkeypatch.setattr(draft_attention, "_q1_label", lambda: label)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_buffer", lambda name, make: None
    )
    attn = draft_attention.Attention.__new__(draft_attention.Attention)
    attn.layer = SimpleNamespace(
        attn=SimpleNamespace(
            layer_id=5,
            scaling=0.5,
            sliding_window_size=-1,
            tp_q_head_num=heads,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=dim,
        )
    )
    attn.heads, attn.head_dim, attn.page_size = heads, dim, page
    attn.q1 = attn.skv2 = None
    attn._noted = set()
    attn.cp_b1 = draft_attention.DraftCP(
        split_kv, route, steps=True, extend=True, multi=requests > 1
    )
    cache = torch.zeros(4 * page, 1, dim, dtype=torch.float8_e4m3fn)
    kv = SimpleNamespace(
        layer_ids=(5,), k=(cache,), v=(cache,), k_scale=(2.0,), v_scale=(3.0,)
    )
    meta = SimpleNamespace(
        page_table=torch.zeros(requests, 4, dtype=torch.int32),
        cache_seqlens=torch.full((requests,), 100, dtype=torch.int32),
    )
    q8 = torch.zeros(requests * rows, heads * dim, dtype=torch.float8_e4m3fn)
    out = attn.decode(q8, meta, kv, rows_per_request=rows)
    assert out.shape == (requests * rows, heads * dim) and bool((out == 1).all())
    assert calls == [((requests * rows, heads, dim), rows, 1.0, 3.0, min_bs)]


@pytest.mark.parametrize(
    "slots, want",
    [
        ({"draft_moe"}, "slot 'draft_moe' runs inside entry 'draft_loop'"),
        ({"draft_loop", "draft_hop"}, "slot 'draft_hop' runs on slot 'hop''s mailbox"),
        (
            {"draft_loop", "draft_fin"},
            "slot 'draft_fin' extends slot 'draft_hop''s fused final norm",
        ),
        (
            {"draft_loop", "draft_skv2"},
            "slot 'draft_skv2' runs inside entry 'draft_extend'",
        ),
        (
            {"draft_loop", "draft_multi"},
            "slot 'draft_multi' serves the draft steps of two or more requests "
            "('multi_request')",
        ),
        (
            {"draft_loop", "multi_request", "draft_embed"},
            "slot 'draft_embed' runs in slot 'draft_multi''s steps",
        ),
    ],
)
def test_the_draft_slots_name_what_they_need(monkeypatch, slots, want):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    assert [r.split(": ", 1)[1] for r in runtime.slot_reasons()] == [want]


def test_the_draft_target_path_uses_its_validated_engine_ranges():
    from sgl_nvidia_mega.qwen3_5.draft import moe_mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    draft = moe_mkm.DraftMkm(None, wide=True)
    assert [
        draft.takes(rows, per_request)
        for rows, per_request in ((1, 1), (2, 1), (7, 7), (64, 1), (65, 1))
    ] == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert [mkm.engine_rows(rows, m16_max_rows=12) for rows in (7, 12, 13, 16, 17)] == [
        mkm.MAX_TOKENS,
        mkm.M16_MAX_TOKENS,
        mkm.M32_MAX_TOKENS,
        mkm.M32_MAX_TOKENS,
        mkm.M32_MAX_TOKENS,
    ]


@pytest.mark.parametrize(
    "fused, path, taken, want",
    [
        (False, "sglang", True, ["mlp", "all_reduce", "norm"]),
        (False, "mkm", True, ["mkm", "complete", "all_reduce", "norm"]),
        (True, "sglang", True, ["mlp deferred", "fin"]),
        (True, "mkm", True, ["mkm", "fin"]),
        (True, "sglang", False, ["mlp deferred", "complete", "norm"]),
        (True, "mkm", False, ["mkm", "complete", "all_reduce", "norm"]),
    ],
)
def test_the_draft_extends_moe_output_is_reduced_once(
    monkeypatch, fused, path, taken, want
):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    rows, log = 14, []
    x = torch.ones(rows, 4)
    routed = torch.bfloat16 if taken else torch.float32

    def owed(reduced):
        def complete():
            log.append("complete")
            return 8 * x if reduced else x

        return SimpleNamespace(
            m=rows,
            routed_output=torch.zeros(1, dtype=routed),
            expert_weights=torch.zeros(1, dtype=torch.bfloat16),
            permuted_indices=torch.zeros(1, dtype=torch.int32),
            gated_shared_output=torch.zeros(1, dtype=torch.bfloat16),
            complete=complete,
        )

    def mlp(h, defer_finalize=False):
        log.append("mlp deferred" if defer_finalize else "mlp")
        return owed(reduced=True) if defer_finalize else h

    def fin(*, residual, gamma, **_):
        log.append("fin")
        return residual + gamma, residual

    def norm(h, residual):
        log.append("norm")
        return h + residual, residual

    def mkm(lay, h):
        log.append("mkm")
        return owed(reduced=False)

    mlp.supports_deferred_finalize = True
    mkm.takes = lambda r, rows_per_request: True
    norm.gemma_weight = 100.0
    extend_hop = SimpleNamespace(
        hop=SimpleNamespace(
            supports=lambda m: m <= rows,
            moe_finalize_all_reduce_rms_norm=fin,
        ),
        ffn_input=lambda h, residual, norm, moe_single_launch: (h, residual),
    )
    parts = SimpleNamespace(
        layer=SimpleNamespace(
            input_layernorm=lambda h: h,
            post_attention_layernorm=None,
            mlp=mlp,
        ),
        attention=lambda h, *args, rows_per_request, o_proj: h,
        hop=extend_hop,
        moe=None,
        moe_mkm=mkm if path == "mkm" else None,
        slots=frozenset({"draft_hop", "draft_fin"} if fused else {"draft_hop"}),
        model=SimpleNamespace(norm=norm),
    )

    def all_reduce(h):
        log.append("all_reduce")
        return 8 * h

    monkeypatch.setattr(
        "sglang.srt.distributed.tensor_model_parallel_all_reduce", all_reduce
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    out = layer.layer(parts, x, None, None, None, None, rows_per_request=7)
    assert log == want
    assert torch.equal(out, x + 100.0 if "fin" in log else 8 * x + x)


def test_only_tep8_draft_fin_uses_the_target_mnnvl_workspace(monkeypatch):
    import sglang.srt.layers.flashinfer_mnnvl_cutedsl as flashinfer_mnnvl_cutedsl

    from sgl_nvidia_mega.qwen3_5.draft import layer

    workspace = SimpleNamespace(supports=lambda rows: rows <= 896)
    monkeypatch.setattr(flashinfer_mnnvl_cutedsl, "_WORKSPACE", workspace)
    parts = SimpleNamespace(slots=frozenset({"draft_fin"}))

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=8, moe_ep_size=4),
    )
    assert layer._mnnvl_workspace(parts, 896) is workspace
    assert layer._mnnvl_workspace(parts, 897) is None
    assert layer._mnnvl_workspace(SimpleNamespace(slots=frozenset()), 32) is None

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=8, moe_ep_size=1),
    )
    assert layer._mnnvl_workspace(parts, 32) is None


@pytest.mark.parametrize("deferred", [True, False])
def test_tep8_draft_fin_uses_the_target_workspace_past_the_hop(monkeypatch, deferred):
    from sgl_nvidia_mega.qwen3_5.draft import layer

    rows, calls = 32, []
    x = torch.ones(rows, 4)

    def fused_ar(*, local_contribution, residual, gamma):
        calls.append("fused ar")
        return local_contribution + gamma, residual

    def fused_fin(*, residual, gamma, **_):
        calls.append("fused finalize")
        return residual + gamma, residual

    workspace = SimpleNamespace(
        all_reduce_residual_rms_norm=fused_ar,
        moe_finalize_all_reduce_rms_norm=fused_fin,
    )
    monkeypatch.setattr(layer, "_mnnvl_workspace", lambda parts, rows: workspace)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def mlp(hidden, defer_finalize=False):
        calls.append("mlp deferred" if defer_finalize else "mlp")
        if not defer_finalize:
            return 2 * hidden
        return SimpleNamespace(
            routed_output=torch.zeros(1, dtype=torch.bfloat16),
            expert_weights=torch.zeros(1, dtype=torch.bfloat16),
            permuted_indices=torch.zeros(1, dtype=torch.int32),
            gated_shared_output=None,
        )

    mlp.supports_deferred_finalize = deferred
    post_norm = SimpleNamespace(gemma_weight=10.0)
    final_norm = SimpleNamespace(gemma_weight=100.0)
    parts = SimpleNamespace(
        layer=SimpleNamespace(
            input_layernorm=lambda hidden: hidden,
            post_attention_layernorm=post_norm,
            mlp=mlp,
        ),
        attention=lambda hidden, *args, rows_per_request, o_proj: hidden,
        hop=None,
        moe=None,
        moe_mkm=None,
        slots=frozenset({"draft_hop", "draft_fin"}),
        model=SimpleNamespace(norm=final_norm),
    )

    def stock_all_reduce(_):
        raise AssertionError("the fused workspace owns both all-reduce points")

    monkeypatch.setattr(
        "sglang.srt.distributed.tensor_model_parallel_all_reduce", stock_all_reduce
    )
    out = layer.layer(parts, x, None, None, None, None, rows_per_request=1)
    if deferred:
        assert calls == ["fused ar", "mlp deferred", "fused finalize"]
        assert torch.equal(out, x + 100.0)
    else:
        assert calls == ["fused ar", "mlp", "fused ar"]
        assert torch.equal(out, 2 * (x + 10.0) + 100.0)


class _DraftParts:
    """Parts with no slot built: the stock model step."""

    def __init__(self, slots=frozenset(), tp_size=8):
        self.slots, self.mkd, self.top1x, self.engaged = slots, None, None, []
        self.binding = SimpleNamespace(tp_size=tp_size)

    def note_engaged(self, entry, bs):
        self.engaged.append((entry, bs))

    def embed(self, token_ids):
        return self.model.embed_tokens(token_ids)


def _stub_model_step(monkeypatch, calls):
    """The draft's model step as a log: fc_input -> x, layer -> hidden, top1 -> token."""
    from sgl_nvidia_mega.qwen3_5.draft import layer

    def fc_input(parts, token, hidden):
        calls.append(("fc", token.tolist(), hidden.tolist()))
        return hidden + 1

    def run_layer(parts, x, positions, cache_loc, meta, kv, *, rows_per_request):
        calls.append(("layer", positions.tolist(), cache_loc.tolist(), meta, kv))
        return x * 10

    def top1(parts, hidden):
        calls.append(("top1", hidden.tolist()))
        return hidden[:, 0].to(torch.int64) % 1000

    monkeypatch.setattr(layer, "fc_input", fc_input)
    monkeypatch.setattr(layer, "layer", run_layer)
    monkeypatch.setattr(layer, "top1", top1)


def test_the_draft_loop_writes_each_steps_token_and_restores_the_positions(
    monkeypatch,
):
    from sgl_nvidia_mega.qwen3_5.draft import SERVED_BS
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    _stub_model_step(monkeypatch, calls)
    parts = _DraftParts()
    loop = DraftLoop(parts, num_steps=4)
    assert loop.step_for(SERVED_BS) == loop.step
    io = SimpleNamespace(
        num_requests=1,
        num_steps=4,
        topk_index=torch.tensor([[7]]),
        hidden_states=torch.tensor([[1.0, 2.0]]),
        positions=torch.tensor([100]),
        mrope_positions=torch.tensor([[100], [100], [100]]),
        # request-major: step i writes slot 40 + i
        out_cache_loc=torch.tensor([40, 41, 42, 43]),
        steps=("m0", "m1", "m2"),
        kv="kv",
        draft_tokens_out=torch.full((1, 4), -1, dtype=torch.int64),
    )
    loop.step(io)
    assert parts.engaged == [("draft_loop", 1)]
    # three draft steps at positions 100..102 on slots 40..42 with step i's metadata
    layers = [c for c in calls if c[0] == "layer"]
    assert [c[1:] for c in layers] == [
        ([100], [40], "m0", "kv"),
        ([101], [41], "m1", "kv"),
        ([102], [42], "m2", "kv"),
    ]
    # step i + 1 embeds step i's token after its final-norm hidden state
    fcs = [c for c in calls if c[0] == "fc"]
    assert [c[1] for c in fcs] == [[7], [20], [210]]
    assert io.draft_tokens_out.tolist() == [[7, 20, 210, 110]]
    assert io.positions.tolist() == [100] and io.mrope_positions.tolist() == [
        [100],
        [100],
        [100],
    ]


@pytest.mark.parametrize(
    "tp_size, slots, served",
    [
        (8, {"multi_request"}, (1, 2)),
        (8, set(), (1,)),
        (4, {"multi_request"}, (1, 2)),
        (8, {"multi_request", "draft_multi"}, tuple(range(1, 129))),
        (4, {"multi_request", "draft_multi"}, tuple(range(1, 129))),
    ],
)
def test_the_draft_loop_serves_two_requests_at_tp8_on_request(
    monkeypatch, tp_size, slots, served
):
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    loop = DraftLoop(_DraftParts(slots=frozenset(slots), tp_size=tp_size), num_steps=4)
    sizes = tuple(range(1, 130))
    assert tuple(bs for bs in sizes if loop.step_for(bs) == loop.step) == served


@pytest.mark.parametrize(
    "tp_size, slots, built, served",
    [
        (8, {"multi_request", "draft_multi"}, True, tuple(range(1, 129))),
        (8, {"multi_request", "draft_multi"}, False, (1,)),
        (4, {"multi_request", "draft_multi"}, True, tuple(range(1, 129))),
        (8, {"draft_multi"}, True, (1,)),
    ],
)
def test_the_draft_extend_serves_more_requests_with_draft_multi(
    monkeypatch, tp_size, slots, built, served
):
    from sgl_nvidia_mega.qwen3_5.draft.extend import DraftExtend

    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    parts = _DraftParts(slots=frozenset(slots), tp_size=tp_size)
    parts.top1x = object() if built else None
    extend = DraftExtend(parts, tokens_per_request=7)
    sizes = tuple(range(1, 130))
    assert tuple(bs for bs in sizes if extend.step_for(bs) == extend.step) == served


def test_a_multi_request_draft_loop_takes_the_sharded_top1(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.draft import layer
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    _stub_model_step(monkeypatch, calls)

    def top1_into(
        parts, hidden, tokens, column, *, nan_key, positions, mrope_positions
    ):
        calls.append(("top1x", column, nan_key))
        tokens[:, column] = hidden[:, 0].to(torch.int64) % 1000
        positions.add_(1)

    monkeypatch.setattr(layer, "top1_into", top1_into)
    parts = _DraftParts(slots=frozenset({"draft_multi"}))
    parts.top1x = object()
    loop = DraftLoop(parts, num_steps=3)
    io = SimpleNamespace(
        num_requests=4,
        num_steps=3,
        topk_index=torch.tensor([[1], [2], [3], [4]]),
        hidden_states=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        positions=torch.tensor([10, 20, 30, 40]),
        mrope_positions=None,
        out_cache_loc=torch.arange(12),
        steps=("m0", "m1"),
        kv="kv",
        draft_tokens_out=torch.full((4, 3), -1, dtype=torch.int64),
    )
    loop.step(io)
    assert parts.engaged == [("draft_loop", 4)]
    assert [c[0] for c in calls if c[0] in ("top1", "top1x")] == ["top1x", "top1x"]
    # each step embeds the column the previous exchange wrote
    assert [c[1] for c in calls if c[0] == "fc"] == [[1, 2, 3, 4], [20, 30, 40, 50]]
    assert io.draft_tokens_out.tolist() == [
        [1, 20, 210],
        [2, 30, 310],
        [3, 40, 410],
        [4, 50, 510],
    ]
    # the request-major slots of step i, made dense
    assert [c[2] for c in calls if c[0] == "layer"] == [[0, 3, 6, 9], [1, 4, 7, 10]]
    assert io.positions.tolist() == [10, 20, 30, 40]


def test_the_draft_embeds_through_the_exchange_where_it_fits():
    from sgl_nvidia_mega.qwen3_5.draft.parts import Parts
    from sgl_nvidia_mega.qwen3_5.sm103.draft.embed import MAX_ROWS

    calls = []
    parts = Parts.__new__(Parts)
    parts.model = SimpleNamespace(embed_tokens=lambda t: calls.append("sglang") or t)
    parts.embedx = None
    parts.embed(torch.zeros(4, dtype=torch.int64))
    parts.embedx = lambda t: calls.append("exchange") or t
    parts.embed(torch.zeros(MAX_ROWS, dtype=torch.int64))
    parts.embed(torch.zeros(MAX_ROWS + 1, dtype=torch.int64))
    assert calls == ["sglang", "exchange", "sglang"]


@pytest.mark.parametrize("requests, split", [(16, True), (32, False)])
def test_the_split_fc_takes_the_draft_steps_up_to_its_widest_batch(
    monkeypatch, requests, split
):
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    _stub_model_step(monkeypatch, calls)
    parts = _DraftParts(slots=frozenset({"draft_multi"}))
    parts.model = SimpleNamespace(embed_tokens=lambda t: t)
    parts.mkd = SimpleNamespace(split_fc=lambda e, h: calls.append(("split",)) or h)
    loop = DraftLoop(parts, num_steps=2)
    loop._fc(torch.zeros(requests, dtype=torch.int64), torch.zeros(requests, 1))
    assert [c[0] for c in calls] == (["split"] if split else ["fc"])


@pytest.mark.parametrize("split_fc", [False, True])
def test_a_two_request_draft_loop_runs_each_step_on_dense_rows(monkeypatch, split_fc):
    from sgl_nvidia_mega.qwen3_5.draft import layer
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    _stub_model_step(monkeypatch, calls)
    stub_layer = layer.layer

    def run_layer(parts, x, positions, cache_loc, meta, kv, *, rows_per_request):
        calls.append(("dense", cache_loc.is_contiguous()))
        return stub_layer(
            parts, x, positions, cache_loc, meta, kv, rows_per_request=rows_per_request
        )

    monkeypatch.setattr(layer, "layer", run_layer)
    parts = _DraftParts()
    if split_fc:
        # slot draft_xchg's row-split fc, never its one-request exchange steps
        parts.model = SimpleNamespace(embed_tokens=lambda t: t.to(torch.float32))
        parts.mkd = SimpleNamespace(
            split_fc=lambda e, h: (calls.append(("split", e.tolist())), h + 1)[1]
        )
    loop = DraftLoop(parts, num_steps=4)
    io = SimpleNamespace(
        num_requests=2,
        num_steps=4,
        topk_index=torch.tensor([[7], [8]]),
        hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        positions=torch.tensor([100, 200]),
        mrope_positions=None,
        # request-major: request r's step i writes slot 40 + 10 r + i
        out_cache_loc=torch.tensor([40, 41, 42, 43, 50, 51, 52, 53]),
        steps=("m0", "m1", "m2"),
        kv="kv",
        draft_tokens_out=torch.full((2, 4), -1, dtype=torch.int64),
    )
    loop.step(io)
    assert parts.engaged == [("draft_loop", 2)]
    assert [c[1:] for c in calls if c[0] == "layer"] == [
        ([100, 200], [40, 50], "m0", "kv"),
        ([101, 201], [41, 51], "m1", "kv"),
        ([102, 202], [42, 52], "m2", "kv"),
    ]
    assert [c[1] for c in calls if c[0] == "dense"] == [True] * 3
    fc = "split" if split_fc else "fc"
    assert [c[1] for c in calls if c[0] == fc] == [[7, 8], [20, 40], [210, 410]]
    assert io.draft_tokens_out.tolist() == [[7, 20, 210, 110], [8, 40, 410, 110]]
    assert io.positions.tolist() == [100, 200]


def test_the_draft_extend_writes_the_selected_rows_outputs(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.draft.extend import DraftExtend

    calls = []
    _stub_model_step(monkeypatch, calls)
    parts = _DraftParts()
    extend = DraftExtend(parts, tokens_per_request=3)
    assert extend.step_for(1) == extend.step and extend.step_for(4) is None
    io = SimpleNamespace(
        num_requests=1,
        tokens_per_request=3,
        input_ids=torch.tensor([5, 6, 7]),
        positions=torch.tensor([10, 11, 12]),
        hidden_states=torch.tensor([[1.0], [2.0], [3.0]]),
        out_cache_loc=torch.tensor([20, 21, 22]),
        select_index=torch.tensor([1]),
        attn="meta",
        kv="kv",
        topk_index_out=torch.zeros(1, 1, dtype=torch.int64),
        topk_p_out=torch.zeros(1, 1),
        hidden_out=torch.zeros(1, 1),
    )
    extend.step(io)
    assert parts.engaged == [("draft_extend", 1)]
    assert [c[1:] for c in calls if c[0] == "layer"] == [
        ([10, 11, 12], [20, 21, 22], "meta", "kv")
    ]
    # the selected row 1: (2 + 1) * 10
    assert io.hidden_out.tolist() == [[30.0]]
    assert io.topk_index_out.tolist() == [[30]] and io.topk_p_out.tolist() == [[1.0]]


# ------------------------------------------------------------------- the parity check


@pytest.mark.parametrize("world", [2, 4, 8])
def test_draft_fc_tiles_fit_shared_memory_at_each_tp(world):
    from sgl_nvidia_mega.qwen3_5.draft.xchg import fc_split_slice
    from sgl_nvidia_mega.qwen3_5.sm103.draft import xchg

    hidden, sms = 4096, 148
    width = 2 * hidden // world
    rows = xchg.draft_fc_partial_rows(hidden, width, sms)
    assert world in xchg.WORLDS
    assert 0 < rows <= -(-hidden // sms)
    assert (rows + xchg.MAX_TOKENS) * width * 2 <= xchg._MAX_SMEM_BYTES
    assert [fc_split_slice(hidden, r, world) for r in range(world)] == [
        (r >= world // 2, (r % (world // 2)) * width, width) for r in range(world)
    ]
    if world in (4, 8):
        assert rows == -(-hidden // sms)


@pytest.mark.parametrize("requests", [2, 6, 10])
def test_multi_draft_extend_uses_split_fc_only_within_its_bound(monkeypatch, requests):
    from sgl_nvidia_mega.qwen3_5.draft import layer
    from sgl_nvidia_mega.qwen3_5.draft.extend import DraftExtend

    parts = _DraftParts()
    calls = []
    parts.embed = lambda ids: calls.append("embed") or ids
    parts.mkd = SimpleNamespace(
        split_fc=lambda embeds, hidden: calls.append("split") or hidden
    )
    monkeypatch.setattr(layer, "fc_input", lambda *a: calls.append("stock") or a[2])
    monkeypatch.setattr(layer, "layer", lambda p, x, *a, **kw: x)
    monkeypatch.setattr(layer, "top1_into", lambda *a, **kw: None)
    rows = requests * 7
    io = SimpleNamespace(
        input_ids=torch.arange(rows),
        hidden_states=torch.ones(rows, 1),
        positions=None,
        out_cache_loc=None,
        attn=None,
        kv=None,
        tokens_per_request=7,
        select_index=torch.arange(requests),
        hidden_out=torch.empty(requests, 1),
        topk_index_out=None,
        topk_p_out=None,
    )
    DraftExtend(parts, tokens_per_request=7)._multi_step(io)
    assert calls == (["embed", "split"] if rows <= 16 else ["stock"])


@pytest.fixture
def eager(monkeypatch):
    """Outside CUDA graph capture (a CPU build of torch has no capture to query), with
    an attention backend that has no linear-attention states."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        lambda: SimpleNamespace(),
    )


def test_the_second_forward_sees_the_conv_windows_the_first_saw(eager, monkeypatch):
    conv = torch.zeros(2, 3, 4)  # [layers, slots, ...]
    cache = SimpleNamespace(conv=[conv])
    linear = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=cache)
        ),
        forward_metadata=SimpleNamespace(mamba_cache_indices=torch.tensor([1, -1])),
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        lambda: SimpleNamespace(linear_attn_backend=linear),
    )

    def forward():
        # reads slot 1, then rewrites it in place, as a verify forward's conv update does
        out = conv[:, 1].sum().reshape(1, 1).clone()
        conv[:, 1] += 1.0
        return SimpleNamespace(next_token_logits=out, hidden_states=None)

    got = parity.Parity((7,), "cpu").run(Step("target_verify", 7), forward, forward)
    assert got.next_token_logits.item() == 0.0  # the restored state, not 8.0
    assert conv[:, 1].eq(1.0).all() and conv[:, 0].eq(0.0).all()


def test_parity_compares_bits_not_values():
    a = torch.tensor([0.0, 1.0, float("nan")])
    assert int(parity.differing(a, a.clone())) == 0
    assert int(parity.differing(a, torch.tensor([-0.0, 1.0, float("nan")]))) == 1
    assert (
        int(parity.differing(a, torch.tensor([0.0, 1.0 + 2**-20, float("nan")]))) == 1
    )
    assert parity.differing(None, None) == 0
    with pytest.raises(RuntimeError):
        parity.differing(a, a[:2])
    with pytest.raises(RuntimeError):
        parity.differing(a, None)


def test_parity_counts_the_requests_own_rows_apart():
    # two requests of 2 rows; the second is a padded one (the pad slot)
    valid = parity.valid_rows(torch.tensor([3, -1]), 4)
    assert valid.tolist() == [True, True, False, False]
    a = torch.zeros(4, 3)
    b = a.clone()
    b[3] = 1.0  # a padded row
    assert int(parity.differing(a, b)) == 3 and int(parity.differing(a, b, valid)) == 0
    b[0, 1] = 1.0  # a request's row
    assert int(parity.differing(a, b, valid)) == 1


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("output_rows", [0, 7])
def test_parity_handles_an_empty_dp_rank(
    eager, monkeypatch, caplog, captured, output_rows
):
    monkeypatch.setattr(parity, "MODE", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: captured)
    monkeypatch.setattr(
        "sglang.srt.model_executor.runner.get_is_capture_mode", lambda: False
    )
    monkeypatch.setattr(
        parity, "conv_rows", lambda: (torch.empty(0, dtype=torch.long), [])
    )

    def output(value):
        return lambda: SimpleNamespace(
            next_token_logits=torch.full((output_rows, 3), value),
            hidden_states=torch.full((output_rows, 4), value),
        )

    checker = parity.Parity((7,), "cpu")
    with caplog.at_level("INFO"):
        got = checker.run(Step("target_verify", 7), output(0.0), output(1.0))
        if captured:
            assert checker.counters[7].tolist() == [
                1,
                output_rows * 3,
                output_rows * 4,
                output_rows * 3,
                output_rows * 4,
                0,
                0,
            ]
            monkeypatch.setattr(
                torch.cuda, "is_current_stream_capturing", lambda: False
            )
            checker.flush()
    assert got.hidden_states.shape == (output_rows, 4)
    assert "rows_compared=" + str(output_rows) in caplog.text
    assert "hidden_dist_max=" + ("1" if output_rows else "0") in caplog.text
    if captured:
        assert "valid_logits_differ=0 valid_hidden_differ=0" in caplog.text
    else:
        assert "valid_rows=0" in caplog.text
        assert "logits_max_abs=" + ("1" if output_rows else "0") in caplog.text


def test_empty_request_metadata_has_no_valid_rows():
    slots = torch.empty(0, dtype=torch.long)
    assert parity.valid_rows(slots, 0).tolist() == []
    assert parity.valid_rows(slots, 7).tolist() == [False] * 7


@pytest.mark.parametrize("mode", ["1", "control"])
def test_an_eager_parity_step_runs_sglangs_forward_first(
    eager, monkeypatch, caplog, mode
):
    monkeypatch.setattr(parity, "MODE", mode)
    order = []

    def output(name, value):
        def run():
            order.append(name)
            return SimpleNamespace(
                next_token_logits=torch.full((7, 3), value), hidden_states=None
            )

        return run

    checker = parity.Parity((7,), "cpu")
    with caplog.at_level("INFO"):
        got = checker.run(
            Step("target_verify", 7), output("sglang", 1.0), output("plugin", 2.0)
        )
    assert order == (["sglang", "plugin"] if mode == "1" else ["sglang", "sglang"])
    want = 0 if mode == "control" else 21
    assert (
        f"rows=7 logits_differ={want} hidden_differ=0 logits_compared=21 "
        "hidden_compared=0" in caplog.text
    )
    assert got.next_token_logits[0, 0] == (1.0 if mode == "control" else 2.0)


def test_a_capture_warmup_is_logged_apart_from_the_eager_steps(
    eager, monkeypatch, caplog
):
    monkeypatch.setattr(
        "sglang.srt.model_executor.runner.get_is_capture_mode", lambda: True
    )

    def output():
        return SimpleNamespace(next_token_logits=torch.zeros(7, 3), hidden_states=None)

    with caplog.at_level("INFO"):
        parity.Parity((7,), "cpu").run(Step("target_verify", 7), output, output)
    assert "parity capture-warmup target_verify rows=7 logits_differ=0" in caplog.text
    assert " eager " not in caplog.text


@torch.no_grad()
def test_trace_names_the_first_module_call_that_differs(eager, monkeypatch, caplog):
    monkeypatch.setattr(parity, "MODE", "trace")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    x = torch.ones(7, 4)

    def forward(shift):
        def run():
            h = model[0](x) + shift  # an op between two modules
            return SimpleNamespace(next_token_logits=model[1](h), hidden_states=None)

        return run

    checker = parity.Parity((7,), "cpu", model)
    with caplog.at_level("INFO"):
        checker.run(Step("target_verify", 7), forward(0.0), forward(0.0))
        checker.run(Step("target_verify", 7), forward(0.0), forward(1.0))
    traces = [r.message for r in caplog.records if "parity trace" in r.message]
    assert traces == [
        "NVIDIA megakernel parity trace target_verify rows=7: no diverging call "
        "among 2 common module calls",
        "NVIDIA megakernel parity trace target_verify rows=7: 1#0 input: 28 of 28 "
        "elements differ",
    ]


def test_graph_parity_counts_are_logged_once_each(eager, caplog):
    checker = parity.Parity((7,), "cpu")
    checker.counters[7] += torch.tensor([3, 0, 5, 63, 0, 0, 2])
    checker.distance[7] += torch.tensor(
        [21.0, 2.0, 0.42, 0.25, 0.5], dtype=torch.float64
    )
    with caplog.at_level("INFO"):
        checker.flush()
        checker.flush()
    lines = [r.message for r in caplog.records if "parity graph" in r.message]
    assert lines == [
        "NVIDIA megakernel parity graph target_verify rows=7 steps=3 "
        "logits_differ=0 hidden_differ=5 logits_compared=63 hidden_compared=0 "
        "valid_logits_differ=0 valid_hidden_differ=2 "
        "rows_compared=21 top1_differ=2 dlp_mean=0.02 dlp_max=0.25 hidden_dist_max=0.5"
    ]
    # the distance restarts with every flush
    assert checker.distance[7].tolist() == [0.0] * 5


def test_the_parity_distance_reads_top1_flips_and_log_prob_gaps():
    a = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    b = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 2.0]])  # row 1's top-1 moves
    rows, top1, total, worst, hidden = parity.distance(
        a, b, torch.zeros(2, 4), torch.full((2, 4), 0.25)
    ).tolist()
    gap = torch.log_softmax(a[1], -1)[1] - torch.log_softmax(b[1], -1)[1]
    assert (rows, top1, hidden) == (2.0, 1.0, 0.25)
    assert total == pytest.approx(gap.item()) and worst == pytest.approx(gap.item())
    assert parity.distance(None, None, None, None) is None


def test_empty_logits_preserve_nonempty_hidden_distance():
    empty_logits = torch.empty(0, 3)
    apart = parity.distance(
        empty_logits, empty_logits, torch.zeros(2, 4), torch.ones(2, 4)
    )
    assert apart.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_the_step_state_carries_the_binding():
    lm = SimpleNamespace(model="model")
    st = StepState(
        lm=lm,
        forward_batch="fb",
        positions="pos",
        input_ids="ids",
        slots="slots",
        bound={},
        binding="EagleBinding",
        view="EagleForwardView",
    )
    assert (st.model, st.binding, st.view) == (
        "model",
        "EagleBinding",
        "EagleForwardView",
    )


@pytest.mark.parametrize("tp, dp, ep, rows", [(2, 1, 1, (42, 70)), (8, 8, 8, (224,))])
def test_new_topologies_cover_actual_pd_stock_widths(monkeypatch, tp, dp, ep, rows):
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=tp, attn_dp_size=dp, moe_ep_size=ep),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=rows, tokens_per_request=7, deadline_s=60.0, binding="binding"
    )
    built = runtime.build(model, ctx)
    assert built.widths == rows
    for width in rows:
        assert built.plan(torch.zeros(width, dtype=torch.int64), _batch()).rows == width


def test_slot_topology_declarations_are_complete_and_fail_closed():
    from sgl_nvidia_mega.qwen3_5 import topology

    known = set(
        (
            *runtime.SHIPPED_SLOTS,
            *runtime.OPT_IN_SLOTS,
            *runtime.OPT_IN_VARIANTS,
            *runtime.DRAFT_ENTRIES,
            *runtime.DRAFT_SLOTS,
            *runtime.OPT_IN_COVERAGE,
            *runtime.HOOK_SLOTS,
        )
    )
    assert set(topology.SLOT_TOPOLOGIES) == known
    assert (
        topology.select(frozenset({"undeclared"}), topology.Topology(2)) == frozenset()
    )


@pytest.mark.parametrize("tp,rows", [(2, 7), (2, 42), (2, 70), (4, 7), (8, 7)])
def test_router_only_slots_cover_aggregate_and_pd_widths(tp, rows):
    from sgl_nvidia_mega.qwen3_5 import plan, topology

    requested = frozenset({"route", "route_topk", "route_gemm"})
    assert topology.select(requested, topology.Topology(tp)) == requested
    slots = plan.slots(rows, plan.Facts(route=True, route_topk=True, route_gemm=True))
    assert slots.route == (rows <= 16)
    assert slots.route_topk == (rows > 16)
    assert not slots.mkm and not slots.mkm_m32 and not slots.mkm_m64
    assert not topology.select(requested, topology.Topology(8, 8, 8))


def test_tp4_preserves_v2_shipped_capabilities():
    from sgl_nvidia_mega.qwen3_5 import topology

    shipped = frozenset(runtime.SHIPPED_SLOTS)
    # The wide expert engines serve TP8's I=128, while TP4 holds I=256.
    tp8_engines = frozenset({"mkm_m32", "mkm_m64", "mkm_m128", "mkm_m256", "mkm_m512"})
    assert topology.select(shipped, topology.Topology(4)) == shipped - tp8_engines
    assert (
        topology.select(topology.TP4_V2_SLOTS, topology.Topology(4))
        == topology.TP4_V2_SLOTS
    )
    # Keep each owner's separately declared TP2 capabilities; restoring TP4
    # must not enable any other unimplemented TP2 kernel.
    assert topology.select(topology.TP4_V2_SLOTS, topology.Topology(2)) == {
        "route",
        "route_topk",
        "route_gemm",
        "mka",
        "gdn_verify",
        "gdn_pdl",
        "gdn_conv",
        "gdn_core",
    }
    assert topology.select(topology.TP4_V2_SLOTS, topology.Topology(8, 8, 8)) == {
        "mka",
        "gdn_verify",
        "gdn_pdl",
        "gdn_conv",
    }
    assert topology.select({"mka"}, topology.Topology(2)) == {"mka"}
    assert topology.select({"mka"}, topology.Topology(8, 8, 8)) == {"mka"}
    assert runtime.slot_reasons(topology.select(shipped, topology.Topology(4))) == []


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_measured_topology_policy_selects_actual_draft_entries(monkeypatch, tp_size):
    from sgl_nvidia_mega.qwen3_5 import draft, topology

    # All explicit opt-ins, not just SHIPPED_SLOTS: the formal47 recipe must
    # leave every TP2 draft callback unset, as in the topo-adapted v2 baseline.
    requested = frozenset(topology.SLOT_TOPOLOGIES)
    # This policy test uses a CPU binding stand-in. CP collective setup has
    # its own v3 contract tests and cannot run against the string stand-in.
    monkeypatch.setattr(runtime, "build_cp_lite", lambda binding: None)
    monkeypatch.setattr(runtime, "SLOTS", requested)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=tp_size, moe_ep_size=1),
    )
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=8192), _draft_ctx())
    if tp_size == 2:
        assert not built.slots
        assert all(
            getattr(built, name) is None
            for name in (
                "draft_loop", "draft_extend", "begin_draft_round", "graph_variants",
            )
        )
        assert runtime.SLOTS == frozenset(
            {
                "multi_request",
                "sharded_gumbel",
                "gdn_verify",
                "gdn_pdl",
                "gdn_conv",
                "gdn_core",
                "route",
                "route_topk",
                "route_gemm",
                "mka",
            }
        )
    else:
        assert built.draft_loop is not None and built.draft_extend is not None
        assert built.slots == topology.select(
            frozenset(runtime.DRAFT_ENTRIES + runtime.DRAFT_SLOTS),
            topology.Topology(tp_size),
        )


def test_tp4_declarations_preserve_existing_kernel_shape_boundaries():
    from sgl_nvidia_mega.qwen3_5 import topology
    from sgl_nvidia_mega.qwen3_5.sm103 import hop
    from sgl_nvidia_mega.qwen3_5.sm103.draft import q1, xchg
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    assert 256 in mkm.INTERMEDIATES
    assert 256 not in mkm.WIDE_ENGINES and 256 not in mkm.M32_ENGINES
    assert 4 in hop.WORLDS and 4 in xchg.WORLDS and 8 in q1.QUERY_HEADS
    tp8_only = frozenset(
        {
            "mkm_wide",
            "mkm_m32",
            "mkm_m64",
            "mkm_m128",
            "mkm_m256",
            "mkm_m512",
            "mkm_wide_bal",
            "gdn_front",
            "gdn_trt_front",
            "route_lin",
        }
    )
    assert not topology.select(tp8_only, topology.Topology(4))
    assert topology.select(tp8_only, topology.Topology(8)) == tp8_only


def test_unsupported_specialization_does_not_disable_a_supported_slot(monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import topology

    monkeypatch.setitem(topology.SLOT_TOPOLOGIES, "gdn_verify", topology.TOPOLOGIES)
    requested = frozenset({"gdn_verify", "gdn_front", "gdn_pdl", "multi_request"})
    selected = topology.select(requested, topology.Topology(2))
    assert selected == frozenset({"gdn_verify", "gdn_pdl", "multi_request"})
    assert runtime.slot_reasons(selected) == []


def test_dep8_is_not_confused_with_tep8_and_keeps_the_draft_stock():
    from sgl_nvidia_mega.qwen3_5 import topology

    requested = frozenset(runtime.SHIPPED_SLOTS)
    assert topology.Topology(8, 8, 8).name == "dep8"
    assert topology.Topology(8, 1, 8).name == "tep8"
    dep = topology.select(requested, topology.Topology(8, 8, 8))
    assert not dep & set(runtime.DRAFT_ENTRIES + runtime.DRAFT_SLOTS)
    assert topology.select(requested, topology.Topology(8, 1, 8)) == requested


@pytest.mark.parametrize("topology_name", ["tp2", "tp4", "tp8", "tep8", "dep8"])
def test_gdn_slot_topologies_keep_generic_verify_and_bound_the_core(topology_name):
    from sgl_nvidia_mega.qwen3_5 import topology

    configs = {
        "tp2": topology.Topology(2),
        "tp4": topology.Topology(4),
        "tp8": topology.Topology(8),
        "tep8": topology.Topology(8, 1, 8),
        "dep8": topology.Topology(8, 8, 8),
    }
    generic = frozenset({"gdn_verify", "gdn_pdl", "gdn_conv"})
    requested = generic | {"gdn_core", "gdn_front", "gdn_push"}
    got = topology.select(requested, configs[topology_name])
    assert generic <= got
    assert ("gdn_core" in got) == (topology_name != "dep8")
    if topology_name in {"tp2", "dep8"}:
        assert not got & {"gdn_front", "gdn_push"}
    if topology_name == "tp4":
        assert "gdn_push" in got and "gdn_front" not in got


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mm_inputs", [None, [], [None], [None, None, None, None]])
def test_prefill_plan_uses_actual_chunk_rows_and_stock_when_slot_off(
    enabled, mm_inputs
):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    built = Qwen35Prefill(model, SimpleNamespace(tokens_per_request=7), enabled)
    fb = SimpleNamespace(mm_inputs=mm_inputs, forward_mode=ForwardMode.EXTEND)
    step = built.plan(torch.zeros(33, dtype=torch.int64), fb)
    if enabled:
        assert step.mode == "prefill" and step.rows == 33
    else:
        assert step is None
    fb.forward_mode = ForwardMode.TARGET_VERIFY
    assert built.plan(torch.zeros(7, dtype=torch.int64), fb) is None


@pytest.mark.parametrize("mm_inputs", [[object()], [None, object()]])
def test_prefill_plan_keeps_real_multimodal_batches_on_stock(mm_inputs):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    built = Qwen35Prefill(model, SimpleNamespace(tokens_per_request=7), True)
    fb = SimpleNamespace(mm_inputs=mm_inputs, forward_mode=ForwardMode.EXTEND)
    assert built.plan(torch.zeros(33, dtype=torch.int64), fb) is None


def test_prefill_unsupported_view_returns_the_original_before_state_writes():
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill
    from sglang.kernels.plugin_kernels.nvidia.spec import Step

    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    built = Qwen35Prefill(model, SimpleNamespace(tokens_per_request=7), True)
    marker = object()
    original = lambda *a, **k: marker
    step = Step(mode="prefill", rows=33, view=SimpleNamespace(kernel_id=0))
    assert built.forward(step, original, None, None, None) is marker


def _prefill_parity_fixture():
    view = SimpleNamespace(
        state=SimpleNamespace(
            conv=(torch.zeros(1, 5, 2, 2),), temporal=torch.zeros(1, 5, 2)
        ),
        state_indices=torch.tensor([1]),
        has_track=True,
        track_conv_dst=torch.tensor([2]),
        track_ssm_h_dst=torch.tensor([3]),
        track_ssm_final_dst=torch.tensor([4]),
        track_ssm_recompute_dst=None,
    )

    def forward():
        conv, ssm = view.state.conv[0], view.state.temporal
        conv[:, 1] += 1
        conv[:, 2] = conv[:, 1] + 2
        ssm[:, 1] += 2
        ssm[:, 3] = ssm[:, 1] + 3
        ssm[:, 4] = ssm[:, 1] + 4
        return SimpleNamespace(tensors={"hidden": ssm[:, 1]})

    return view, forward


def test_prefill_parity_restores_committed_and_track_state_between_forwards():
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import PrefillParity

    view, forward = _prefill_parity_fixture()
    result = PrefillParity().run(33, view, forward, forward)
    assert torch.equal(result.tensors["hidden"], torch.full((1, 2), 2.0))
    assert torch.equal(view.state.conv[0][:, 1], torch.ones(1, 2, 2))
    assert torch.equal(view.state.temporal[:, 4], torch.full((1, 2), 6.0))
    assert not view.state.temporal[:, 0].any()


def test_prefill_parity_catches_state_corruption_with_equal_output():
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import PrefillParity

    view, forward = _prefill_parity_fixture()

    def corrupt():
        result = forward()
        view.state.temporal[:, 4] += 1
        return result

    with pytest.raises(RuntimeError, match="output/state differs"):
        PrefillParity().run(33, view, forward, corrupt)


def test_prefill_all_slots_off_can_compare_native_twice_under_test_switch(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    monkeypatch.setattr(parity, "ENABLED", True)
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    built = Qwen35Prefill(model, SimpleNamespace(tokens_per_request=7), False)
    step = built.plan(
        torch.zeros(33, dtype=torch.int64),
        SimpleNamespace(mm_inputs=None, forward_mode=ForwardMode.EXTEND),
    )
    assert step.mode == "prefill" and not built.enabled


@pytest.mark.parametrize("tp_size", [4, 8])
def test_bf16_draft_entries_cover_pd_batches_without_quantized_experts(
    monkeypatch, tp_size
):
    from sgl_nvidia_mega.qwen3_5 import topology
    from sgl_nvidia_mega.qwen3_5.draft.extend import DraftExtend
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    requested = frozenset(
        {
            "multi_request",
            "draft_loop",
            "draft_extend",
            "draft_multi",
            "draft_embed",
            "draft_xchg",
            "draft_moe",
            "draft_q1",
        }
    )
    selected = topology.select(requested, topology.Topology(tp_size))
    assert runtime.slot_reasons(selected) == []
    monkeypatch.setattr(runtime, "SLOTS", selected)
    parts = _DraftParts(slots=selected, tp_size=tp_size)
    parts.top1x = object()
    parts.attention = SimpleNamespace(q1=object())
    loop = DraftLoop(parts, num_steps=6)
    extend = DraftExtend(parts, tokens_per_request=7)
    for batch in (1, 6, 10):
        assert loop.step_for(batch) == loop.step
        assert extend.step_for(batch) == extend.step
