# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The stock MoE step on CPU, against SGLang's own ``DeepseekV2MoE.forward`` with
``leave_output_sum`` on the same stubbed block: the same ops in the same order, on
the same streams, with the same arguments, and the same pieces out, in the eager and
the CUDA graph capture branches."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.models import deepseek_v2

from sgl_nvidia_mega.deepseek_v41 import moe
from sgl_nvidia_mega.deepseek_v41.moe.step import dual_stream


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))

    def record_event(self):
        self.log.append(("record", self.name))
        return f"event:{self.name}"


def _block(log, *, inplace, quant_once, quant_routed, vision=False, hash_=False):
    """A DeepseekV2MoE whose gate, top-k, experts and shared expert log their calls."""
    cur = {"stream": "main"}
    x = torch.arange(12.0).reshape(3, 4)

    def tag(v):
        return "x" if v is x else ("tensor" if isinstance(v, torch.Tensor) else v)

    def op(name, out):
        def run(*args, **kwargs):
            kw = tuple(sorted((k, tag(v)) for k, v in kwargs.items()))
            log.append((name, cur["stream"], tuple(map(tag, args)), kw))
            return out

        return run

    mlp = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    torch.nn.Module.__init__(mlp)
    topk_out = SimpleNamespace(format="standard")
    quant = SimpleNamespace(quantize_routed_input=op("quantize_routed", ("xq", "xsf")))
    experts = SimpleNamespace(
        moe_runner_config=SimpleNamespace(inplace=inplace),
        supports_deferred_finalize=True,
        quant_method=quant,
        forward_deferred_finalize=op("experts_deferred", "routed_d"),
        forward_impl=op("experts_impl", "routed_i"),
        w13_weight=torch.zeros(1, 8, 2, dtype=torch.uint8),
    )
    experts_call = op("experts", "routed")
    mlp.__dict__.update(
        alt_stream=_Stream("alt", log),
        routed_quant_stream=_Stream("rq", log),
        num_fused_shared_experts=0,
        _enable_a2a_moe=False,
        _fuse_shared_experts_inside_sbo=False,
        _shared_expert_tp1=False,
        _deferred_finalize_max_tokens=0,
        _moe_quant_once=None,
        routed_scaling_factor=1.5,
        reduce_results=False,
        layer_id=7,
        is_nextn=False,
        is_hash=hash_,
    )
    mlp.__dict__["gate"] = _Callable(
        op("gate", "logits"), e_score_correction_bias_vl=1 if vision else None
    )
    mlp.__dict__["topk"] = _Callable(
        op("topk", topk_out),
        topk_config="cfg",
        empty_topk_output=op("empty_topk", topk_out),
    )
    mlp.__dict__["experts"] = _Callable(experts_call, **experts.__dict__)
    mlp.__dict__["shared_experts"] = _Callable(
        op("shared", "shared_out"), gate_up_proj=SimpleNamespace()
    )
    mlp.__dict__["_maybe_quant_moe_input_once"] = op(
        "quant_once", ("q", "s") if quant_once else None
    )
    mlp.__dict__["_should_quant_routed_input_mxfp8"] = op(
        "should_quant_routed", quant_routed
    )
    return mlp, x, cur


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


@contextmanager
def _sglang(log, cur, *, capture, defer_finalize=False):
    main = _Stream("main", log)

    @contextmanager
    def on(stream):
        prev, cur["stream"] = cur["stream"], stream.name
        try:
            yield
        finally:
            cur["stream"] = prev

    fwd = SimpleNamespace(
        flashinfer_trtllm_bypass=False,
        defer_moe_finalize=defer_finalize,
        mlp_reduce_scatter=False,
    )
    exec_ = SimpleNamespace(moe=SimpleNamespace(enable_eplb=False))
    with (
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(torch.cuda, "stream", side_effect=on),
        patch.object(deepseek_v2, "get_forward", return_value=fwd),
        patch.object(deepseek_v2, "get_exec", return_value=exec_),
        patch.object(deepseek_v2, "_is_cuda", True),
        patch.object(deepseek_v2, "_is_hip", False),
        patch.object(deepseek_v2, "get_is_capture_mode", return_value=capture),
        patch.object(deepseek_v2, "is_in_breakable_cuda_graph", return_value=False),
        patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=capture
        ),
        patch("sglang.srt.runtime_context.get_forward", return_value=fwd),
        patch("sglang.srt.layers.moe.mega_moe.should_use_mega_moe", return_value=False),
        patch.object(
            deepseek_v2.DeepseekV2MoE, "_can_dual_stream_graph", return_value=False
        ),
    ):
        yield


def _run_both(capture, defers_sum, skip_shared=False, **block):
    runs = []
    for which in ("sglang", "plugin"):
        log = []
        mlp, x, cur = _block(log, **block)
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: "nnp")
        with _sglang(log, cur, capture=capture):
            if which == "sglang":
                out = mlp.forward(
                    x,
                    batch,
                    input_ids="ids",
                    input_ids_global="ids_g",
                    skip_shared_experts=skip_shared,
                    return_moe_output=defers_sum,
                    leave_output_sum=True,
                )
            else:
                transport = SimpleNamespace(
                    input_ids="ids",
                    input_ids_global="ids_g",
                    do_shared_local=skip_shared,
                    return_moe_output=defers_sum,
                )
                st = SimpleNamespace(
                    fb=batch,
                    transport=transport,
                    slots=SimpleNamespace(routed=False, fast_gate=False),
                    bound={"moe": None},
                )
                out = moe.step(SimpleNamespace(mlp=mlp), x, st)
        runs.append((log, out))
    return runs


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("defers_sum", [False, True])
@pytest.mark.parametrize(
    "block",
    [
        dict(inplace=False, quant_once=False, quant_routed=False),
        dict(inplace=True, quant_once=True, quant_routed=False),
        dict(inplace=False, quant_once=False, quant_routed=True),
        dict(inplace=False, quant_once=False, quant_routed=False, vision=True),
        dict(inplace=False, quant_once=False, quant_routed=False, hash_=True),
        dict(inplace=True, quant_once=True, quant_routed=False, skip_shared=True),
    ],
)
def test_the_stock_step_issues_what_sglang_issues(capture, defers_sum, block):
    if block.get("vision"):
        pytest.importorskip("sglang.srt.multimodal.dsv41.vl_routing")
    with (
        patch("sglang.srt.multimodal.dsv41.vl_routing.vision_topk") as vt,
        patch.object(deepseek_v2, "vision_topk", vt),
    ):
        vt.side_effect = lambda *a, **k: SimpleNamespace(format="standard")
        (want_log, want), (got_log, got) = _run_both(capture, defers_sum, **block)
    assert got_log == want_log
    assert type(got) is type(want)
    assert got._replace(experts=None) == want._replace(experts=None)


def test_the_branch_follows_capture_mode():
    log = []
    mlp, x, cur = _block(log, inplace=False, quant_once=False, quant_routed=False)
    for capture in (False, True):
        with _sglang(log, cur, capture=capture):
            assert dual_stream(mlp, 3) is capture
            assert not dual_stream(mlp, 0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("fast_gate", [False, True])
def test_the_routed_pair_replaces_only_the_top_k_and_the_routed_call(
    capture, fast_gate
):
    log = []
    mlp, x, cur = _block(log, inplace=False, quant_once=False, quant_routed=False)

    def logged(name, out):
        def run(*args):
            log.append((name, cur["stream"]))
            return out

        return run

    slot = SimpleNamespace(
        gate=True,
        select=logged("fast_gate", "gate_topk"),
        run=logged("pair", "deferred"),
    )
    st = SimpleNamespace(
        fb=SimpleNamespace(moe_num_token_non_padded=lambda: "nnp"),
        transport=SimpleNamespace(
            input_ids_global="ids_g", do_shared_local=False, return_moe_output=True
        ),
        slots=SimpleNamespace(routed=True, fast_gate=fast_gate),
        bound={"moe": SimpleNamespace(slots={7: slot})},
    )
    with _sglang(log, cur, capture=capture):
        out = moe.step(SimpleNamespace(mlp=mlp, layer_id=7), x, st)
    names = [e[:2] if e[0] in ("fast_gate", "pair") else e[0] for e in log]
    assert ("pair", "main") in names and "experts" not in names
    assert (("fast_gate", "main") in names) is fast_gate
    assert ("topk" in names) is not fast_gate
    assert ("shared", "alt", ("x",), (("gemm_output_zero_allocator", None),)) in log
    assert out.routed == "deferred" and out.shared == "shared_out"


@pytest.mark.parametrize("fast_gate", [False, True])
@pytest.mark.parametrize("ready", [None, "event"])
def test_the_slot_calls_the_pair_with_its_signature(fast_gate, ready):
    from unittest.mock import create_autospec

    from sgl_nvidia_mega.deepseek_v41.moe import routed
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        RoutedExpertGemm,
    )

    gemm = create_autospec(RoutedExpertGemm, instance=True)
    gemm.top_k = 6
    pair_out = (
        torch.zeros(3, 4),
        torch.zeros(3, 6),
        torch.zeros(18, dtype=torch.int32),
    )
    gemm.supports_gate_topk.return_value = True
    config = SimpleNamespace(
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=True,
        correction_bias_for_dtype=lambda dtype: torch.zeros(384, dtype=dtype),
    )
    mlp = SimpleNamespace(
        topk=SimpleNamespace(topk_config=config),
        gate=SimpleNamespace(e_score_correction_bias_vl=None),
        experts=SimpleNamespace(quant_method=None),
    )
    slot = routed.MoeSlot(mlp, gemm, gate=True)
    topk = (
        slot.select(torch.zeros(3, 384), None, None, False)
        if fast_gate
        else SimpleNamespace(topk_ids="ids", topk_weights="w")
    )
    order = []
    stream = SimpleNamespace(wait_event=lambda e: order.append(("join", e)))
    gemm.gate_topk_and_gemms.side_effect = lambda *a, **k: (
        order.append(("gate", k.get("input_ready"))) or (*pair_out, None)
    )
    gemm.side_effect = lambda *a, **k: order.append("pair") or pair_out
    with patch.object(torch.cuda, "current_stream", return_value=stream):
        out = slot.run(torch.zeros(3, 4), topk, ("xq", "xs", ready))
    # the slot joins the routed input before the top-k in both cases
    first = ("gate", None) if fast_gate else "pair"
    assert order == ([("join", ready)] if ready else []) + [first]
    if fast_gate:
        assert gemm.gate_topk_and_gemms.call_args.args[5] is None  # no fill kernel
    assert out.top_k == 6 and out.gemm2_out is pair_out[0]
