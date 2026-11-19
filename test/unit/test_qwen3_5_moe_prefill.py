# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.layer_boundary import prepare as boundary_prepare
from sglang.srt.layers.layer_boundary import stage as boundary_stage
from sglang.srt.layers.layer_boundary.contracts import BatchVariant, StageKind
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_READOUT,
    PLAIN_ADD,
    Fp8Input,
    NormQuantReadout,
    UnfusedNormReadout,
)
from sglang.srt.layers.layer_boundary.residual.stream import (
    DeclaredSum,
    ResidualStream,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
    StandardDispatchOutput,
)
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.models import qwen2_moe, qwen3_5

from sgl_nvidia_mega.qwen3_5.moe import prefill as moe_prefill

H = 32


def _gemma_add_norm(x, residual, weight, eps):
    """The stand-in for both norm kernels: in place, as gemma_fused_add_rmsnorm."""
    residual.add_(x)
    h = residual.float()
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1 + weight.float())
    x.copy_(y.to(x.dtype))


def _norm_forward(self, x, residual=None, post_residual_addition=None):
    assert residual is not None and post_residual_addition is None
    _gemma_add_norm(x, residual, self.weight.data, self.variance_epsilon)
    return x, residual


def _nvfp4_bytes(x, gs):
    """The stand-in NVFP4 of ``x`` at ``gs``: (packed E2M1, block scales) of its values."""
    q = (x.float() * gs).to(torch.int8).view(torch.uint8)[:, : x.shape[1] // 2]
    sf = torch.full((x.shape[0], x.shape[1] // 16), 7, dtype=torch.uint8)
    return q.contiguous(), sf


def _nvfp4_kernel(calls):

    def run(x, residual, weight, eps, gs):
        calls.append((x, residual, weight, eps, gs))
        _gemma_add_norm(x, residual, weight, eps)
        return _nvfp4_bytes(x, gs)

    return SimpleNamespace(
        gemma_fused_add_rmsnorm_nvfp4=run,
        supported=lambda x, r: x.dim() == 2 and x.shape == r.shape and x.shape[0] > 0,
    )


def _entry(step_kw=None, **kw):
    step = partial(
        boundary_prepare._update_read,
        **(step_kw or dict(pre_move=None, enters_stack=False, read=NORM_READOUT)),
    )
    keywords = dict(
        step=step,
        is_plain_add=True,
        carried_fusions=(),
        carried_read_fusions=(),
        read=NORM_READOUT,
        expected_sum=None,
        completed_step=None,
        written_step=None,
    )
    keywords.update(kw)
    return SimpleNamespace(
        prepare=partial(boundary_prepare._run_entry, **keywords),
        input_rows=None,
        input_move=None,
        attn_input_adapter=None,
        declared_sum=None,
    )


def _ffn_stage(norm, entry=None, variant=BatchVariant.ORDINARY, **plan):
    entry = entry or _entry()
    path = SimpleNamespace(entry=entry)
    fields = dict(
        norm=norm,
        paths={BatchVariant.ORDINARY: path},
        fusions=None,
        qkv_latent_func=None,
        _publish_lora_layout=False,
        variant_for=lambda fb: variant,
        path_for=lambda fb: path,
    )
    fields.update(plan)
    declaration = SimpleNamespace(kind=StageKind.FFN, row_transport=None)
    return boundary_stage.StageBoundary(
        SimpleNamespace(**fields), declaration=declaration
    )


def _experts(scale=None):
    method = modelopt_quant.ModelOptNvFp4FusedMoEMethod.__new__(
        modelopt_quant.ModelOptNvFp4FusedMoEMethod
    )
    method.quant_config = SimpleNamespace(use_per_token_activation=False)
    return SimpleNamespace(
        quant_method=method,
        _cutedsl_wrapper=SimpleNamespace(quant_mode="w4a4"),
        _cutedsl_input_scale=scale,
    )


def _layer(norm, **stage):
    return SimpleNamespace(
        post_attention_layernorm=norm,
        ffn_boundary=_ffn_stage(norm, **stage),
        mlp=SimpleNamespace(experts=_experts()),
    )


def _case(seed=0, rows=5):
    g = torch.Generator().manual_seed(seed)
    out = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    residual = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    norm = GemmaRMSNorm(H, eps=1e-6)
    norm.weight.data.copy_(torch.randn(H, generator=g) * 0.1)
    return out, residual, norm


def _stream(out, residual, update=PLAIN_ADD, owed=None):
    stream = ResidualStream(residual)
    out = stream.record(out, update, declared_sum=owed)
    return out, SimpleNamespace(residual_stream=stream)


@contextlib.contextmanager
def _cuda_scale(scale):
    """``_global_scale`` checks a CUDA FP32 scale; on CPU stand in for is_cuda."""
    with (
        patch.object(flashinfer_cutedsl, "ensure_cutedsl_wrapper", lambda layer: None),
        patch.object(moe_prefill.torch.Tensor, "is_cuda", True),
    ):
        yield


def test_the_ffn_input_leaves_what_sglangs_boundary_leaves(monkeypatch):
    monkeypatch.setattr(GemmaRMSNorm, "forward", _norm_forward)
    runs = []
    for which in ("sglang", "plugin"):
        out, residual, norm = _case()
        layer = _layer(norm)
        hidden, fb = _stream(out, residual)
        if which == "sglang":
            x = layer.ffn_boundary.prepare(hidden, fb)
            pre = None
        else:
            calls = []
            gs = torch.tensor([3.0])
            layer.mlp.experts._cutedsl_input_scale = gs
            layers = moe_prefill.Layers([4])
            with (
                _cuda_scale(gs),
                patch(
                    "sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_input_nvfp4",
                    _nvfp4_kernel(calls),
                    create=True,
                ),
                patch.dict(
                    "sys.modules",
                    {
                        "sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_input_nvfp4": (
                            _nvfp4_kernel(calls)
                        )
                    },
                ),
            ):
                x, pre = moe_prefill.ffn_input(layers, 4, layer, hidden, fb)
            assert layers.scales == {4: gs}
        stream = fb.residual_stream
        assert stream.pending is None
        runs.append((x, stream.residual, out, residual, pre))
    (x0, r0, out0, res0, _), (x1, r1, out1, res1, pre) = runs
    assert x0 is out0 and x1 is out1 and r0 is res0 and r1 is res1
    assert torch.equal(x0.view(torch.int16), x1.view(torch.int16))
    assert torch.equal(r0.view(torch.int16), r1.view(torch.int16))
    q, sf = pre
    assert q.shape == (5, H // 2) and sf.shape == (5, H // 16)


def _plugin_kernel():
    return patch.dict(
        "sys.modules",
        {"sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_input_nvfp4": _nvfp4_kernel([])},
    )


@pytest.mark.parametrize(
    "case",
    [
        "declared sum",
        "update",
        "variant",
        "empty",
        "other value",
        "no static scale",
    ],
)
def test_the_ffn_input_leaves_every_other_batch_to_the_boundary(monkeypatch, case):
    out, residual, norm = _case(rows=0 if case == "empty" else 5)
    variant = (
        BatchVariant.SEQUENCE_PARALLEL if case == "variant" else BatchVariant.ORDINARY
    )
    layer = _layer(norm, variant=variant)
    update = (
        SimpleNamespace(is_plain_add=False, bind=None)
        if case == "update"
        else PLAIN_ADD
    )
    owed = DeclaredSum(None) if case == "declared sum" else None
    hidden, fb = _stream(out, residual, update=update, owed=owed)
    if case == "other value":
        hidden = out.clone()
    gs = torch.tensor([3.0])
    layer.mlp.experts._cutedsl_input_scale = gs
    if case == "no static scale":
        layer.mlp.experts._cutedsl_wrapper.quant_mode = "w4a16"
    stream = fb.residual_stream
    before = (stream.pending, stream.residual.clone(), out.clone())
    layers = moe_prefill.Layers([4])
    with _cuda_scale(gs), _plugin_kernel():
        assert moe_prefill.ffn_input(layers, 4, layer, hidden, fb) is None
    assert stream.pending is before[0]
    assert torch.equal(stream.residual, before[1]) and torch.equal(out, before[2])


@pytest.fixture
def cuda_norm(monkeypatch):
    """GemmaRMSNorm dispatching to its CUDA path, as on a GPU."""
    monkeypatch.setattr(
        GemmaRMSNorm, "_resolve_forward_method", lambda self: self.forward_cuda
    )


@pytest.mark.parametrize(
    "change",
    [
        {},
        # tried by _run_entry only on an owed input, which ffn_input never runs on
        dict(carried_fusions=(object(),), carried_read_fusions=(object(),)),
    ],
)
def test_the_build_takes_the_mirrored_ffn_entry(cuda_norm, change):
    _, _, norm = _case()
    assert moe_prefill.ffn_input_refusal(_layer(norm, entry=_entry(**change))) == ""


@pytest.mark.parametrize("case", ["native", "compile", "hook", "forced"])
def test_the_build_refuses_a_norm_off_its_cuda_path(monkeypatch, cuda_norm, case):
    from sglang.kernels import fused_op

    _, _, norm = _case()
    if case == "native":
        norm._forward_method = norm.forward_native
    elif case == "compile":
        norm.is_torch_compile = True
    elif case == "hook":
        norm.register_forward_hook(lambda *a: None)
    else:
        monkeypatch.setattr(fused_op, "get_fused_op_backend", lambda: "torch")
    assert moe_prefill.ffn_input_refusal(_layer(norm)) != ""


@pytest.mark.parametrize(
    "change, why",
    [
        (dict(fusions=object()), "fuses or moves"),
        (dict(_publish_lora_layout=True), "fuses or moves"),
        (dict(entry=_entry(expected_sum=object())), "plain norm"),
        (dict(entry=_entry(read=UnfusedNormReadout())), "plain norm"),
        (dict(entry=_entry(is_plain_add=False)), "plain norm"),
        (
            dict(
                entry=_entry(
                    step_kw=dict(pre_move=len, enters_stack=False, read=NORM_READOUT)
                )
            ),
            "residual add and norm read",
        ),
        (
            dict(
                entry=_entry(
                    step_kw=dict(
                        pre_move=None,
                        enters_stack=False,
                        read=NORM_READOUT,
                        read_gathers=(),
                    )
                )
            ),
            "residual add and norm read",
        ),
        (
            dict(
                entry=_entry(
                    step_kw=dict(pre_move=None, enters_stack=True, read=NORM_READOUT)
                )
            ),
            "residual add and norm read",
        ),
    ],
)
def test_the_build_refuses_every_other_ffn_input(cuda_norm, change, why):
    _, _, norm = _case()
    layer = _layer(norm, **change)
    assert why in moe_prefill.ffn_input_refusal(layer)


def test_the_build_refuses_a_norm_that_is_not_the_layers():
    _, _, norm = _case()
    layer = _layer(norm)
    layer.post_attention_layernorm = GemmaRMSNorm(H)
    assert "not the layer's GemmaRMSNorm" in moe_prefill.ffn_input_refusal(layer)
    layer.ffn_boundary.plan.norm = torch.nn.LayerNorm(H)
    assert "not the layer's GemmaRMSNorm" in moe_prefill.ffn_input_refusal(layer)


class _Log(list):
    def tag(self, v):
        for name, known in self.known.items():
            if v is known:
                return name
        if isinstance(v, torch.Tensor):
            return ("tensor", tuple(v.shape))
        return v

    def op(self, name, out=None):
        def run(*args, **kwargs):
            kw = tuple(sorted((k, self.tag(v)) for k, v in kwargs.items()))
            self.append((name, self.stream, tuple(map(self.tag, args)), kw))
            return out() if callable(out) else out

        return run


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


_NO_A2A = {
    name: (lambda: False)
    for name in (
        "is_deepep",
        "is_deepep_v2",
        "is_mori",
        "is_ascend_fuseep",
        "is_ascend_ops",
    )
}


@contextlib.contextmanager
def _block_env(log, capture):
    main = _Stream("main", log)

    @contextlib.contextmanager
    def on(stream):
        prev, log.stream = log.stream, stream.name
        try:
            yield
        finally:
            log.stream = prev

    no_paged = SimpleNamespace(moe=SimpleNamespace(enable_paged_experts=False))
    with (
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(torch.cuda, "stream", side_effect=on),
        patch.object(qwen2_moe, "get_is_capture_mode", return_value=capture),
        patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=capture
        ),
        patch.object(qwen2_moe, "get_exec", return_value=no_paged),
        patch("sglang.srt.runtime_context.get_exec", return_value=no_paged),
        patch.object(qwen2_moe, "fused_gate_sigmoid_mul_add", log.op("gate_mul_add")),
        patch(
            "sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add",
            log.op("gate_mul_add"),
        ),
        patch.object(qwen2_moe, "is_npu", return_value=False),
        patch.object(qwen2_moe, "use_intel_amx_backend", return_value=False),
        patch.object(qwen2_moe, "_use_aiter", False),
        patch.object(
            qwen2_moe, "get_moe_a2a_backend", return_value=_Callable(None, **_NO_A2A)
        ),
    ):
        yield


def _block(log, x):
    mlp = qwen2_moe.Qwen2MoeSparseMoeBlock.__new__(qwen2_moe.Qwen2MoeSparseMoeBlock)
    torch.nn.Module.__init__(mlp)
    topk = SimpleNamespace(format="standard")
    routed = torch.zeros(x.shape[0], H)
    log.known.update(routed=routed, topk_output=topk)
    mlp.__dict__.update(
        alt_stream=_Stream("alt", log),
        reduce_results=False,
        enable_shared_expert_fusion=False,
        _use_mega_moe=False,
        layer_id=3,
        gate=_Callable(log.op("gate", ("logits", None))),
        topk=_Callable(log.op("topk", topk)),
        experts=_Callable(log.op("experts", routed)),
        shared_expert=_Callable(log.op("shared", "shared_out")),
        shared_expert_gate=_Callable(
            log.op("shared_gate", "gate_out"), weight=torch.ones(1, H)
        ),
    )
    return mlp


@pytest.mark.parametrize("capture", [False, True])
def test_the_block_issues_what_sglang_issues(capture):
    runs = []
    for which in ("sglang", "plugin"):
        x = torch.zeros(7, H)
        log = _Log()
        log.known, log.stream = {"x": x}, "main"
        mlp = _block(log, x)
        with _block_env(log, capture):
            if which == "sglang":
                out = mlp.forward(x, None)
            else:
                out = moe_prefill.forward(mlp, x, None, None)
        runs.append((log, out))
    (want_log, want), (got_log, got) = runs
    # a clone of x is a new tensor of x's shape on both sides
    assert got_log == want_log
    order = [(e[0], e[1]) for e in got_log]
    if capture:
        assert order[:2] == [("wait", "alt"), ("shared", "main")]
        assert ("experts", "alt") in order and order[-2] == ("wait", "main")
    else:
        assert order == [
            ("shared", "main"),
            ("gate", "main"),
            ("topk", "main"),
            ("experts", "main"),
            ("gate_mul_add", "main"),
        ]
    assert torch.equal(want, got)


def test_an_empty_batch_is_sglangs_block():
    x = torch.zeros(0, H)
    calls = []
    mlp = _Callable(lambda *a, **k: calls.append((a, k)) or "stock")
    assert moe_prefill.forward(mlp, x, "fb", None) == "stock"
    assert calls == [((x, "fb"), {})]


def _moe_layer_stub(log, gs, a2):
    """A FusedMoE on the CuTe DSL standard path whose wrapper and dispatcher log."""
    experts = SimpleNamespace()
    method = modelopt_quant.ModelOptNvFp4FusedMoEMethod.__new__(
        modelopt_quant.ModelOptNvFp4FusedMoEMethod
    )
    out = torch.full((5, H), 2.0)
    method.__dict__.update(
        moe_runner_config=SimpleNamespace(activation="silu"),
        _moe_runner_backend=SimpleNamespace(
            is_flashinfer_megamoe=lambda: False,
            is_marlin=lambda: False,
            is_flashinfer_trtllm=lambda: False,
            is_flashinfer_trtllm_routed=lambda: False,
            is_flashinfer_cutedsl=lambda: True,
        ),
        enable_flashinfer_cutedsl_moe=True,
        quant_config=SimpleNamespace(use_per_token_activation=False),
    )
    runner = SimpleNamespace(
        fused_func=flashinfer_cutedsl.fused_experts_none_to_flashinfer_cutedsl_fp4,
        lora_enabled=False,
        config=method.moe_runner_config,
        fused_op=None,
    )
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner

    runner.run = MethodType(MoeRunner.run, runner)
    method.runner = runner
    dispatcher = StandardDispatcher.__new__(StandardDispatcher)
    dispatcher.dispatch = lambda hidden_states, topk_output: StandardDispatchOutput(
        hidden_states=hidden_states, hidden_states_scale=None, topk_output=topk_output
    )
    dispatcher.combine = lambda combine_input: (
        log.append(("combine", combine_input.hidden_states is out))
        or (combine_input.hidden_states + 1)
    )
    experts.__dict__.update(
        quant_method=method,
        dispatcher=dispatcher,
        _use_ascend_fuseep=False,
        _dwdp_bound=False,
        _mega_moe_nvfp4=False,
        reduce_results=False,
        moe_tp_size=1,
        moe_ep_size=1,
        w13_weight="w13",
        w2_weight="w2",
        w13_blockscale_swizzled="w13_sf_swizzled",
        w13_blockscale_mma="w13_sf_mma",
        w2_blockscale_swizzled="w2_sf_swizzled",
        _cutedsl_scales=("alpha1", a2, "alpha2"),
        _cutedsl_input_scale=gs,
        _cutedsl_wrapper=SimpleNamespace(
            quant_mode="w4a4",
            run=lambda **kw: log.append(("run", kw)) or out,
        ),
    )
    experts.run_moe_core = MethodType(fused_moe_layer.FusedMoE.run_moe_core, experts)
    experts._dispatch_with_pre_quant = MethodType(
        fused_moe_layer.FusedMoE._dispatch_with_pre_quant, experts
    )
    experts.forward_impl = MethodType(fused_moe_layer.FusedMoE.forward_impl, experts)
    return experts


@contextlib.contextmanager
def _routed_env(log, pre):
    from sglang.srt.distributed.device_communicators import pynccl_allocator
    from sglang.srt.layers import dp_attention
    from sglang.srt.model_executor.runner_backend_utils import (
        tc_piecewise_cuda_graph,
    )

    def quantize(x, scale, sf_vec_size, is_sf_swizzled_layout):
        log.append(("fp4_quantize", scale, sf_vec_size, is_sf_swizzled_layout))
        return pre

    parallel = SimpleNamespace(tp_group="tp")
    with (
        patch.object(fused_moe_layer, "is_in_tc_piecewise_cuda_graph", lambda: False),
        patch.object(
            tc_piecewise_cuda_graph, "is_in_tc_piecewise_cuda_graph", lambda: False
        ),
        patch.object(fused_moe_layer, "is_allocation_symmetric", lambda: False),
        patch.object(dp_attention, "is_allocation_symmetric", lambda: False),
        patch.object(
            fused_moe_layer,
            "use_symmetric_memory",
            lambda *a, **k: contextlib.nullcontext(),
        ),
        patch.object(
            pynccl_allocator,
            "use_symmetric_memory",
            lambda *a, **k: contextlib.nullcontext(),
        ),
        patch.object(fused_moe_layer, "get_parallel", lambda: parallel, create=True),
        patch("sglang.srt.runtime_context.get_parallel", lambda: parallel),
        patch.object(flashinfer_cutedsl, "ensure_cutedsl_wrapper", lambda layer: None),
        patch.object(modelopt_quant, "is_flashinfer_cutedsl_v1_path", lambda: False),
        patch("sglang.srt.layers.quantization.fp4_utils.fp4_quantize", quantize),
    ):
        yield


def test_the_routed_experts_run_the_runner_on_the_norms_bytes():
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    runs = []
    for which in ("sglang", "plugin"):
        log = []
        gs, a2 = torch.tensor([3.0]), torch.tensor([5.0])
        g = torch.Generator().manual_seed(1)
        x = torch.randn(5, H, generator=g).to(torch.bfloat16)
        q = torch.randint(0, 255, (5, H // 2), dtype=torch.uint8, generator=g)
        sf = torch.randint(0, 255, (5, H // 16), dtype=torch.uint8, generator=g)
        topk = StandardTopKOutput(
            topk_weights=torch.rand(5, 2, generator=g),
            topk_ids=torch.randint(0, 8, (5, 2), dtype=torch.int64, generator=g),
            router_logits=None,
        )
        experts = _moe_layer_stub(log, gs, a2)
        with _routed_env(log, (q, sf)):
            if which == "sglang":
                out = fused_moe_layer.FusedMoE.forward(experts, x, topk)
            else:
                out = moe_prefill.routed(experts, x, topk, (q, sf))
        runs.append((log, out, gs, a2, q, sf, topk))
    (want_log, want, gs, a2, q, sf, topk), (got_log, got, *_) = runs
    assert want_log[0] == ("fp4_quantize", gs, 16, False)
    assert [e[0] for e in got_log] == ["run", "combine"]
    (_, want_kw), (_, got_kw) = want_log[1], got_log[0]
    assert want_kw.keys() == got_kw.keys()
    for name in want_kw:
        a, b = want_kw[name], got_kw[name]
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(
                a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a,
                b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b,
            ), name
        else:
            assert a == b, name
    assert got_kw["x_sf"].dtype == torch.float8_e4m3fn
    assert got_kw["token_selected_experts"].dtype == torch.int32
    assert torch.equal(got_kw["fc2_input_scale"], a2)
    assert got_kw["w1_weight_sf"] == "w13_sf_mma"
    assert want_log[2] == got_log[1] == ("combine", True)
    assert torch.equal(want, got)


def test_the_routed_experts_without_bytes_are_sglangs():
    calls = []
    experts = _Callable(lambda *a: calls.append(a) or "stock")
    assert moe_prefill.routed(experts, "x", "topk", None) == "stock"
    assert calls == [("x", "topk")]


def _taken_block():
    log = _Log()
    log.known, log.stream = {}, "main"
    mlp = _block(log, torch.zeros(1, H))
    mlp.experts = _moe_layer_stub([], torch.tensor([1.0]), torch.tensor([1.0]))
    return mlp


@contextlib.contextmanager
def _build_env():
    backend = SimpleNamespace(is_none=lambda: True, is_flashinfer_cutedsl=lambda: True)
    with (
        patch("sglang.srt.layers.moe.get_moe_a2a_backend", lambda: backend),
        patch("sglang.srt.layers.moe.get_moe_runner_backend", lambda: backend),
        patch.object(qwen2_moe, "_is_cuda", True),
        patch.object(qwen2_moe, "_use_aiter", False),
        patch.object(qwen2_moe, "_SGLANG_EXPERIMENTAL_LORA_OPTI", False),
        patch.object(qwen2_moe, "use_intel_amx_backend", lambda m: False),
        patch.object(modelopt_quant, "is_flashinfer_cutedsl_v1_path", lambda: False),
    ):
        yield


def test_the_build_takes_the_cutedsl_block():
    with _build_env():
        mlp = _taken_block()
        assert moe_prefill.block_refusal(mlp) == ""
        assert moe_prefill.experts_refusal(mlp.experts) == ""


@pytest.mark.parametrize(
    "change, why",
    [
        (dict(_use_mega_moe=True), "MegaMoE"),
        (dict(shared_expert=None), "gated shared expert"),
        (dict(shared_expert_gate=None), "gated shared expert"),
        (dict(enable_shared_expert_fusion=True), "fused into the routed"),
        (dict(reduce_results=True), "sums its own output"),
    ],
)
def test_the_build_refuses_every_other_block(change, why):
    with _build_env():
        mlp = _taken_block()
        mlp.__dict__.update(change)
        assert why in moe_prefill.block_refusal(mlp)


@pytest.mark.parametrize(
    "module_patch, why",
    [
        (dict(_is_cuda=False), "CUDA path"),
        (dict(_use_aiter=True), "CUDA path"),
        (dict(_SGLANG_EXPERIMENTAL_LORA_OPTI=True), "LoRA"),
    ],
)
def test_the_build_refuses_other_platforms_and_lora(module_patch, why):
    with _build_env(), patch.multiple(qwen2_moe, **module_patch):
        assert why in moe_prefill.block_refusal(_taken_block())


def test_the_build_refuses_a_decoder_layer_with_hooks():
    with _build_env():
        layer = torch.nn.Module()
        layer.mlp = _taken_block()
        layer.register_forward_pre_hook(lambda *a: None)
        assert "module hooks" in moe_prefill.layer_refusal(layer)


def test_the_build_refuses_hooks_and_all_to_all():
    with _build_env():
        mlp = _taken_block()
        mlp.register_forward_pre_hook(lambda *a: None)
        assert "hooks" in moe_prefill.block_refusal(mlp)
    a2a = SimpleNamespace(is_none=lambda: False)
    with _build_env(), patch("sglang.srt.layers.moe.get_moe_a2a_backend", lambda: a2a):
        assert "all-to-all" in moe_prefill.block_refusal(_taken_block())


@pytest.mark.parametrize(
    "change, why",
    [
        ("method", "ModelOpt NVFP4"),
        ("backend", "flashinfer_cutedsl"),
        ("fused_func", "CuTe DSL standard"),
        ("lora", "CuTe DSL standard"),
        ("v1", "CuTe DSL standard"),
        ("dwdp", "dispatch or reduce"),
        ("dispatcher", "dispatch or reduce"),
    ],
)
def test_the_build_refuses_every_other_runner(change, why):
    with _build_env(), contextlib.ExitStack() as stack:
        experts = _taken_block().experts
        method = experts.quant_method
        if change == "method":
            experts.quant_method = SimpleNamespace()
        elif change == "backend":
            method._moe_runner_backend = SimpleNamespace(
                is_flashinfer_cutedsl=lambda: False
            )
        elif change == "fused_func":
            method.runner.fused_func = len
        elif change == "lora":
            method.runner.lora_enabled = True
        elif change == "v1":
            stack.enter_context(
                patch.object(
                    modelopt_quant, "is_flashinfer_cutedsl_v1_path", lambda: True
                )
            )
        elif change == "dwdp":
            experts._dwdp_bound = True
        else:
            experts.dispatcher = SimpleNamespace()
        assert why in moe_prefill.experts_refusal(experts)


def test_the_build_keeps_sglangs_layers_without_the_kernel_when_none_is_taken(caplog):
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=None)] * 2)
    )
    built, why = moe_prefill.build(model, range(2), add_norm=True)
    assert built is None
    assert why == {
        moe_prefill.SLOT: "layer 0: not a sparse MoE block",
        moe_prefill.SLOT_ADD_NORM: "layer 0: not a sparse MoE block",
    }


class _Exit:
    def __init__(self, log, kind):
        self.log, self.kind = log, kind
        self.defer_moe_finalize = False

    def __enter__(self):
        self.log.append(("exit", self.kind))
        return self

    def __exit__(self, *exc):
        return False

    def finish(self, out):
        self.log.append(("exit.finish", self.log.tag(out)))
        return "layer_out"


def _decoder_layer(cls, log):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)

    def stage(kind):
        return SimpleNamespace(
            prepare=log.op(f"{kind}.prepare", lambda: torch.zeros(3, H)),
            finish=log.op(f"{kind}.finish", f"{kind}_done"),
            exit=lambda fb: _Exit(log, kind),
        )

    mlp = qwen2_moe.Qwen2MoeSparseMoeBlock.__new__(qwen2_moe.Qwen2MoeSparseMoeBlock)
    torch.nn.Module.__init__(mlp)
    layer.__dict__.update(
        attn_boundary=stage("attn"),
        ffn_boundary=stage("ffn"),
        self_attention=log.op("self_attention", "attn_out"),
        linear_attn=log.op("linear_attn", "attn_out"),
    )
    layer._modules["mlp"] = mlp
    return layer


@pytest.mark.parametrize("linear", [False, True])
def test_the_body_runs_the_moe_layer_as_sglangs_layer(monkeypatch, linear):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    cls = (
        qwen3_5.Qwen3_5LinearDecoderLayer
        if linear
        else qwen3_5.Qwen3_5AttentionDecoderLayer
    )
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    runs = []
    for which in ("sglang", "plugin"):
        log = _Log()
        log.known, log.stream = {"fb": fb}, "main"
        layer = _decoder_layer(cls, log)
        hidden = torch.zeros(3, H)
        log.known["hidden"] = hidden
        mlp_op = log.op("mlp", "moe_out")
        monkeypatch.setattr(
            qwen2_moe.Qwen2MoeSparseMoeBlock,
            "__call__",
            lambda self, *a, **k: mlp_op(*a, **k),
        )
        # the slot's block and FFN input left to SGLang's: the order is the layer's
        monkeypatch.setattr(
            moe_prefill,
            "forward",
            lambda m, x, b, pre, noclone: mlp_op(x, b, defer_finalize=False),
        )
        monkeypatch.setattr(moe_prefill, "ffn_input", lambda *a: None)
        if which == "sglang":
            out = cls.forward(
                layer,
                **({} if linear else {"positions": "pos"}),
                hidden_states=hidden,
                forward_batch=fb,
                captured_last_layer_outputs="cap",
            )
        else:
            body = Qwen35Prefill.__new__(Qwen35Prefill)
            body.fp8 = None
            body.qk_rows = False
            body._linear = {4} if linear else set()
            out, owing = body._moe_layer(
                moe_prefill.Layers([4]),
                4,
                layer,
                hidden,
                "pos",
                fb,
                "cap",
                lambda gdn, x: gdn(x, fb),
            )
            assert owing is None
        runs.append((log, out))
    (want_log, want), (got_log, got) = runs
    assert got_log == want_log and want == got == "layer_out"
    assert ("ffn.prepare", "main") in [e[:2] for e in got_log]


def test_the_body_hands_the_ffn_input_and_its_bytes_to_the_block(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    log = _Log()
    log.known, log.stream = {"fb": fb}, "main"
    layer = _decoder_layer(qwen3_5.Qwen3_5AttentionDecoderLayer, log)
    normed, pre = torch.ones(3, H), ("q", "sf")
    log.known.update(normed=normed, pre=pre)
    monkeypatch.setattr(
        moe_prefill,
        "ffn_input",
        lambda layers, i, lay, x, b: (
            log.append(("ffn_input", i, log.tag(b))) or (normed, pre)
        ),
    )
    monkeypatch.setattr(moe_prefill, "forward", log.op("block", "moe_out"))
    body = Qwen35Prefill.__new__(Qwen35Prefill)
    body.fp8 = None
    body.qk_rows = False
    body._linear = set()
    out, owing = body._moe_layer(
        moe_prefill.Layers([4]), 4, layer, torch.zeros(3, H), "pos", fb, None, None
    )
    assert out == "layer_out" and owing is None
    names = [e[0] for e in log]
    assert names == [
        "attn.prepare",
        "self_attention",
        "attn.finish",
        "ffn_input",
        "exit",
        "block",
        "exit.finish",
    ]
    block = log[names.index("block")]
    assert block[2] == (layer.mlp, "normed", "fb", "pre", False)


@pytest.mark.parametrize("qk_rows", [False, True], ids=["", "row-tiles"])
@pytest.mark.parametrize("taken", [True, False], ids=["gate-row", "stock-gate"])
def test_a_moe_layers_attention_runs_the_fp8_gate_row_where_it_took_the_layer(
    monkeypatch, taken, qk_rows
):
    from sgl_nvidia_mega.qwen3_5.attention import prefill_rows
    from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    log = _Log()
    log.known, log.stream = {"fb": fb}, "main"
    layer = _decoder_layer(qwen3_5.Qwen3_5AttentionDecoderLayer, log)
    monkeypatch.setattr(moe_prefill, "ffn_input", lambda *a: None)
    monkeypatch.setattr(moe_prefill, "forward", log.op("block", "moe_out"))
    body = Qwen35Prefill.__new__(Qwen35Prefill)
    body._linear = set()
    body.qk_rows = qk_rows
    body.fp8 = prefill_fp8.PrefillFp8()
    if taken:
        body.fp8.gate[4] = "scale"
    monkeypatch.setattr(
        body.fp8, "self_attention", log.op("gate_row_attention", "attn_out")
    )
    gates = []
    monkeypatch.setattr(
        prefill_rows,
        "self_attention",
        lambda lay, pos, x, b, gate=None: (
            gates.append(gate) or log.op("row_tiles_attention", "attn_out")(lay, pos)
        ),
    )
    out, owing = body._moe_layer(
        moe_prefill.Layers([4]), 4, layer, torch.zeros(3, H), "pos", fb, None, None
    )
    assert out == "layer_out" and owing is None
    if qk_rows:
        attention = "row_tiles_attention"
    else:
        attention = "gate_row_attention" if taken else "self_attention"
    assert [e[0] for e in log] == [
        "attn.prepare",
        attention,
        "attn.finish",
        "ffn.prepare",
        "exit",
        "block",
        "exit.finish",
    ]
    if attention == "gate_row_attention":
        assert log[1][2][:3] == (4, layer, "pos")
    if qk_rows:
        # the row tiles' attention gets the FP8 gate for the layer the row took
        assert log[1][2] == (layer, "pos")
        (gate,) = gates
        assert (gate is not None) is taken
        if taken:
            assert gate.func == body.fp8.gate_input and gate.args == (4,)


_NVFP4 = "sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_input_nvfp4"

_TP1_READS = [
    pytest.param(cls, fp8_input, id=f"{cls.__name__[7:-12]}-{fp8_input}")
    for cls in (qwen3_5.Qwen3_5LinearDecoderLayer, qwen3_5.Qwen3_5AttentionDecoderLayer)
    for fp8_input in (None, *Fp8Input)
]


@contextlib.contextmanager
def _tp1():
    """What layer planning and the boundaries read at TP1, no attention DP."""
    from sglang.test.boundary_fixtures import parallel_of, planning

    parallel = parallel_of(attn_dp=1, attn_tp=1)
    with (
        planning(parallel),
        patch("sglang.srt.runtime_context.get_parallel", lambda: parallel),
    ):
        yield parallel


def _tp1_layers(cls, fp8_input, n=1, first=False, terminal=False):
    from sglang.test.boundary_fixtures import build_stages

    config = SimpleNamespace(model_type="qwen3_5_moe_text")
    g = torch.Generator().manual_seed(7)
    layers, stages = [], []
    for i in range(n):
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = GemmaRMSNorm(H, eps=1e-6)
            norm.weight.data.copy_(torch.randn(H, generator=g) * 0.1)
            setattr(layer, name, norm)
        attn, ffn = cls.stage_facts(
            config, i + 1, attn_read=NormQuantReadout(fp8_input=fp8_input)
        )
        stages += [
            (attn, layer.input_layernorm, {"fusions": None}),
            (ffn, layer.post_attention_layernorm, {"fusions": None}),
        ]
        layers.append(layer)
    bound = build_stages(
        *stages,
        previous=None if first else cls.stage_facts(config, 0)[1],
        terminal=terminal,
    )
    for i, layer in enumerate(layers):
        layer.attn_boundary, layer.ffn_boundary = bound[2 * i : 2 * i + 2]
        layer.self_attention = layer.linear_attn = _mixer
        layer._is_layer_to_capture = False
    return layers


def _mixer(hidden_states, *args, **kwargs):
    """An attention or GDN mixer's stand-in: new rows of its input's values."""
    return (hidden_states.float() * 0.5 + 0.25).to(hidden_states.dtype)


class _Batch(SimpleNamespace):
    """A prefill batch: the fields the TP1 boundaries read; None for any other."""

    def __getattr__(self, name):
        return None


def _tp1_batch(rows=5):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    g = torch.Generator().manual_seed(11)
    out = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    stream = ResidualStream(torch.randn(rows, H, generator=g).to(torch.bfloat16))
    hidden = stream.record(out, PLAIN_ADD)
    fb = _Batch(
        residual_stream=stream,
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.zeros(rows),
    )
    return hidden, fb


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_build_takes_the_ffn_input_sglang_binds_at_tp1(cuda_norm, cls, fp8_input):
    with _tp1():
        (layer,) = _tp1_layers(cls, fp8_input)
    assert moe_prefill.ffn_input_refusal(layer) == ""


@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_ffn_input_leaves_what_sglangs_tp1_stages_leave(
    monkeypatch, cls, fp8_input
):
    monkeypatch.setattr(GemmaRMSNorm, "forward", _norm_forward)
    runs = []
    for which in ("sglang", "plugin"):
        calls = []
        with _tp1():
            (layer,) = _tp1_layers(cls, fp8_input)
            layer.mlp = SimpleNamespace(experts=_experts(torch.tensor([3.0])))
            hidden, fb = _tp1_batch()
            x = layer.attn_boundary.prepare(hidden, fb)
            hidden = layer.attn_boundary.finish(_mixer(x), fb)
            if which == "sglang":
                x, pre = layer.ffn_boundary.prepare(hidden, fb), None
            else:
                with (
                    _cuda_scale(None),
                    patch.dict("sys.modules", {_NVFP4: _nvfp4_kernel(calls)}),
                ):
                    taken = moe_prefill.ffn_input(
                        moe_prefill.Layers([1]), 1, layer, hidden, fb
                    )
                # the slot took the batch: its kernel ran, the boundary did not
                assert taken is not None and len(calls) == 1
                x, pre = taken
        stream = fb.residual_stream
        assert stream.pending is None
        runs.append((x, stream.residual, pre))
    (x0, r0, _), (x1, r1, pre) = runs
    assert _same(x0, x1) and _same(r0, r1)
    assert all(map(torch.equal, pre, _nvfp4_bytes(x1, 3.0)))


def _tp1_block(log, seed):
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    g = torch.Generator().manual_seed(seed)
    w_router = torch.randn(H, 4, generator=g)
    w_gate = (torch.randn(1, H, generator=g) * 0.1).to(torch.bfloat16)

    def router(h):
        log.append(("router", h.data_ptr()))
        return h.float() @ w_router, None

    def topk(h, logits):
        weights, ids = torch.softmax(logits, -1).topk(2, dim=-1)
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=logits
        )

    def shared(h):
        log.append(("shared", h.data_ptr()))
        return (h.float() * 0.75 - 0.125).to(h.dtype)

    def run(**kw):
        q, sf = kw["x"], kw["x_sf"].view(torch.uint8)
        log.append(("run", q, sf))
        rows = torch.cat([q, q], -1).float() / 128
        return (rows + sf.float().mean(-1, keepdim=True) / 64).to(torch.bfloat16)

    stub = _moe_layer_stub(log, torch.tensor([3.0]), torch.tensor([5.0]))
    stub._cutedsl_wrapper.run = run
    stub.dispatcher.combine = lambda combine_input: combine_input.hidden_states
    experts = _Callable(partial(fused_moe_layer.FusedMoE.forward, stub), **vars(stub))
    mlp = qwen2_moe.Qwen2MoeSparseMoeBlock.__new__(qwen2_moe.Qwen2MoeSparseMoeBlock)
    torch.nn.Module.__init__(mlp)
    mlp.__dict__.update(
        alt_stream=_Stream("alt", log),
        reduce_results=False,
        enable_shared_expert_fusion=False,
        _use_mega_moe=False,
        layer_id=seed,
        gate=_Callable(router),
        topk=_Callable(topk),
        experts=experts,
        shared_expert=_Callable(shared),
        shared_expert_gate=_Callable(None, weight=w_gate),
    )
    return mlp


def _gated_add(x, w, shared, routed):
    """The stand-in for SGLang's gate add on both sides: routed = bf16(g * s + f)."""
    g = torch.sigmoid((x.float() * w.float()).sum(-1, keepdim=True))
    routed.copy_((g * shared.float() + routed.float()).to(routed.dtype))


@contextlib.contextmanager
def _tp1_env(log, capture, kernels):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill  # noqa: F401

    def quantize(x, scale, sf_vec_size, is_sf_swizzled_layout):
        log.append(("fp4_quantize",))
        return _nvfp4_bytes(x, scale)

    built = dict(
        config_supported=lambda h, sm: True,
        get_sm_version=lambda device: 103,
        precompile=lambda device, size: log.append(("precompile", size)),
    )
    with (
        _tp1() as parallel,
        _build_env(),
        _block_env(log, capture),
        _routed_env(log, None),
        # the TP1 state, in place of _routed_env's
        patch("sglang.srt.runtime_context.get_parallel", lambda: parallel),
        _cuda_scale(None),
        patch("sglang.srt.layers.quantization.fp4_utils.fp4_quantize", quantize),
        patch.object(qwen2_moe, "fused_gate_sigmoid_mul_add", _gated_add),
        patch(
            "sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add",
            _gated_add,
        ),
        patch.object(GemmaRMSNorm, "forward", _norm_forward),
        patch.object(
            GemmaRMSNorm, "_resolve_forward_method", lambda self: self.forward_cuda
        ),
        patch.object(moe_prefill, "sglang_refusal", lambda device, kernel: ""),
        patch.object(torch.cuda, "current_device", lambda: 0),
        patch.dict(
            "sys.modules",
            {
                name: SimpleNamespace(**vars(kernel), **built)
                for name, kernel in kernels.items()
            },
        ),
        patch(
            "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
            lambda: SimpleNamespace(
                with_current_layer=lambda i: contextlib.nullcontext()
            ),
        ),
    ):
        yield


def _tp1_run(which, cls, fp8_input, capture=False, n=2, **slots):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    log = _Log()
    log.known, log.stream = {}, "main"
    calls = {"nvfp4": [], "add_norm": []}
    kernels = {
        _NVFP4: _nvfp4_kernel(calls["nvfp4"]),
        _ADD_NORM: _add_norm_kernel(calls["add_norm"]),
    }
    linear = cls is qwen3_5.Qwen3_5LinearDecoderLayer
    built = None
    with (
        _tp1_env(log, capture, kernels),
        patch.object(
            moe_prefill,
            "gate_fp32",
            lambda x, w: torch.sigmoid((x.float() * w.float()).sum(-1)),
        ),
    ):
        layers = _tp1_layers(cls, fp8_input, n)
        for i, layer in enumerate(layers):
            layer.mlp = _tp1_block(log, i)
        hidden, fb = _tp1_batch()
        if which == "sglang":
            for layer in layers:
                kw = {} if linear else {"positions": None}
                hidden = cls.forward(
                    layer, hidden_states=hidden, forward_batch=fb, **kw
                )
        else:
            model = SimpleNamespace(
                model=SimpleNamespace(layers=layers, start_layer=0, end_layer=n)
            )
            built, why = moe_prefill.build(model, range(n), **slots)
            assert why == {}
            body = Qwen35Prefill.__new__(Qwen35Prefill)
            body.fp8 = None
            body.qk_rows = False
            body.lm = model
            body._linear = set(range(n)) if linear else set()
            hidden, _ = body._layers(
                hidden, None, fb, lambda gdn, x: gdn(x, fb), moe=built
            )
    return hidden, fb.residual_stream, log, built, calls


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_body_takes_sglangs_tp1_moe_layers(
    monkeypatch, caplog, cls, fp8_input, capture
):
    monkeypatch.setattr(moe_prefill, "_LOGGED", set())
    caplog.set_level("INFO", logger=moe_prefill.logger.name)
    slots = dict(nvfp4=True, add_norm=True, noclone=True)
    want, want_stream, want_log, _, _ = _tp1_run("sglang", cls, fp8_input, capture)
    got, got_stream, got_log, built, calls = _tp1_run(
        "plugin", cls, fp8_input, capture, **slots
    )
    assert built.nvfp4 == built.noclone == {0, 1} and built.add_norm == {0}
    for log, clones in ((want_log, capture), (got_log, False)):
        shared = [e[1] for e in log if e[0] == "shared"]
        router = [e[1] for e in log if e[0] == "router"]
        assert len(shared) == len(router) == 2
        assert [a != b for a, b in zip(shared, router)] == [clones] * 2
    # the one layer a slot leaves: the stage's last has no next input to take its add
    assert [
        r.getMessage() for r in caplog.records if "uses stock" in r.getMessage()
    ] == [
        f"NVIDIA megakernel: Qwen3.5 slot {moe_prefill.SLOT_ADD_NORM} uses stock on "
        "layer 1: the stage's last layer"
    ]
    assert f"slot {moe_prefill.SLOT} runs" in caplog.text
    assert f"slot {moe_prefill.SLOT_ADD_NORM} runs" in caplog.text
    # the second layer's input norm did the first block's gated add
    assert len(calls["add_norm"]) == 1
    assert len(calls["nvfp4"]) == 2
    assert [e[0] for e in want_log].count("fp4_quantize") == 2
    assert [e[0] for e in got_log].count("fp4_quantize") == 0
    want_runs = [e[1:] for e in want_log if e[0] == "run"]
    got_runs = [e[1:] for e in got_log if e[0] == "run"]
    assert len(got_runs) == len(want_runs) == 2
    for (q0, sf0), (q1, sf1) in zip(want_runs, got_runs):
        assert torch.equal(q0, q1) and torch.equal(sf0, sf1)
    assert _same(want, got)
    assert _same(want_stream.residual, got_stream.residual)
    assert want_stream.pending.value is want and got_stream.pending.value is got


def _e4m3(x, scale):
    return (x.float() * (1 / scale.float())).clamp(-448, 448).to(torch.float8_e4m3fn)


def _add_norm_kernel(calls):
    def run(x, shared, gate, residual, weight, eps, fp8_scale=None):
        calls.append((x, shared, gate, residual, fp8_scale))
        x.copy_((gate[:, None] * shared.float() + x.float()).to(x.dtype))
        _gemma_add_norm(x, residual, weight, eps)
        return None if fp8_scale is None else _e4m3(x, fp8_scale)

    return SimpleNamespace(
        gated_add_rmsnorm=run,
        supported=lambda x, s, g, r: x.shape == s.shape == r.shape and x.shape[0] > 0,
    )


def _attn_stage(norm, entry=None, variant=BatchVariant.ORDINARY, **plan):
    entry = entry or _entry()
    entry.attn_input_adapter = boundary_prepare._attn_input_default
    path = SimpleNamespace(entry=entry)
    fields = dict(
        norm=norm,
        paths={BatchVariant.ORDINARY: path},
        fusions=None,
        qkv_latent_func=None,
        _publish_lora_layout=False,
        enters_stack=False,
        variant_for=lambda fb: variant,
        path_for=lambda fb: path,
    )
    fields.update(plan)
    declaration = SimpleNamespace(kind=StageKind.ATTENTION, row_transport=None)
    return boundary_stage.StageBoundary(
        SimpleNamespace(**fields), declaration=declaration
    )


def _owed_case(seed=3, rows=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    w = (torch.randn(H, generator=g) * 0.1).to(torch.bfloat16)
    shared = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    routed = torch.randn(rows, H, generator=g).to(torch.bfloat16)
    gate = torch.sigmoid((x.float() * w.float()).sum(-1))
    return moe_prefill.Owed(routed=routed, shared=shared, gate=gate, x=x, w=w)


def test_the_layer_input_leaves_what_the_gated_add_and_sglangs_boundary_leave(
    monkeypatch,
):
    from sglang.kernels.ops.elementwise import elementwise

    monkeypatch.setattr(GemmaRMSNorm, "forward", _norm_forward)
    monkeypatch.setattr(elementwise, "fused_gate_sigmoid_mul_add", _gated_add)
    runs = []
    for which in ("sglang", "plugin"):
        _, residual, norm = _case(seed=4)
        owed = _owed_case()
        stage = _attn_stage(norm)
        hidden, fb = _stream(owed.routed, residual)
        if which == "sglang":
            moe_prefill.complete(owed)
            x = stage.prepare(hidden, fb)
        else:
            calls = []
            kernel = _add_norm_kernel(calls)
            with patch.dict(
                "sys.modules",
                {"sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_output_add_norm": kernel},
            ):
                x = moe_prefill.layer_input(
                    SimpleNamespace(attn_boundary=stage), hidden, fb, owed
                )
            assert len(calls) == 1 and calls[0][2] is owed.gate
        assert fb.residual_stream.pending is None
        runs.append((x, fb.residual_stream.residual, owed.routed, residual))
    (x0, r0, f0, res0), (x1, r1, f1, res1) = runs
    assert x0 is f0 and x1 is f1 and r0 is res0 and r1 is res1
    assert torch.equal(x0.view(torch.int16), x1.view(torch.int16))
    assert torch.equal(r0.view(torch.int16), r1.view(torch.int16))


@pytest.mark.parametrize(
    "case", ["declared sum", "update", "variant", "other value", "not the owed rows"]
)
def test_the_layer_input_leaves_every_other_batch_to_the_boundary(case):
    _, residual, norm = _case(seed=4)
    owed = _owed_case()
    variant = (
        BatchVariant.SEQUENCE_PARALLEL if case == "variant" else BatchVariant.ORDINARY
    )
    stage = _attn_stage(norm, variant=variant)
    update = (
        SimpleNamespace(is_plain_add=False, bind=None)
        if case == "update"
        else PLAIN_ADD
    )
    owed_sum = DeclaredSum(None) if case == "declared sum" else None
    out = owed.routed if case != "not the owed rows" else owed.routed.clone()
    hidden, fb = _stream(out, residual, update=update, owed=owed_sum)
    if case == "other value":
        hidden = out.clone()
    stream = fb.residual_stream
    before = (stream.pending, residual.clone(), owed.routed.clone())
    with patch.dict(
        "sys.modules",
        {
            "sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_output_add_norm": _add_norm_kernel(
                []
            )
        },
    ):
        layer = SimpleNamespace(attn_boundary=stage)
        assert moe_prefill.layer_input(layer, hidden, fb, owed) is None
    assert stream.pending is before[0] and torch.equal(residual, before[1])
    assert torch.equal(owed.routed, before[2])


class _Launch:
    """A Triton kernel stand-in that records each launch's grid and arguments."""

    def __init__(self, log):
        self.log = log

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.log.append((grid, args, kwargs))

        return run


@pytest.mark.parametrize("rows", [1, 1023, 1024, 4681])
def test_the_gate_launches_as_the_stock_gate_add_launches(monkeypatch, rows):
    from sglang.kernels.ops.elementwise import elementwise

    launches = []
    monkeypatch.setattr(
        elementwise, "_fused_gate_sigmoid_mul_add_kernel", _Launch(launches)
    )
    x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
    w = torch.zeros(4096, dtype=torch.bfloat16)
    s, f = torch.zeros_like(x), torch.zeros_like(x)
    elementwise.fused_gate_sigmoid_mul_add(x, w, s, f)
    g = moe_prefill.gate_fp32(x, w)
    (grid0, args0, kw0), (grid1, args1, kw1) = launches
    assert grid0 == grid1 == (rows,)
    assert args0[:3] == (x, w, s) and args0[3] is f
    assert args1[:3] == (x, w, None) and args1[3] is g
    assert kw0.pop("DO_ADD") is True and kw1.pop("DO_ADD") is False
    assert kw0.pop("GATE_ONLY") is False and kw1.pop("GATE_ONLY") is True
    assert kw0 == kw1
    assert g.dtype == torch.float32 and g.shape == (rows,)
    assert g.untyped_storage().nbytes() == (rows + 15) // 16 * 16 * 4


@pytest.mark.parametrize("capture", [False, True])
def test_the_owing_block_issues_the_blocks_launches_but_the_gate_only_tail(
    monkeypatch, capture
):

    runs = []
    for which in ("sglang", "plugin"):
        x = torch.zeros(7, H, dtype=torch.bfloat16)
        log = _Log()
        log.known, log.stream = {"x": x}, "main"
        mlp = _block(log, x)
        mlp.shared_expert = _Callable(
            log.op("shared", lambda: torch.ones(7, H, dtype=torch.bfloat16))
        )
        mlp.experts = _Callable(
            log.op("experts", lambda: torch.ones(7, H, dtype=torch.bfloat16))
        )
        mlp.shared_expert_gate.weight = torch.ones(1, H, dtype=torch.bfloat16)
        monkeypatch.setattr(
            moe_prefill, "gate_fp32", lambda x, w: log.append(("gate_only",)) or "g"
        )
        with _block_env(log, capture):
            if which == "sglang":
                out = mlp.forward(x, None)
                owed = None
            else:
                out, owed = moe_prefill.forward_owing(mlp, x, None, None)
        runs.append((log, out, owed))
    (want_log, want, _), (got_log, got, owed) = runs
    assert want_log[-1][0] == "gate_mul_add" and got_log[-1] == ("gate_only",)
    assert [e[:2] for e in got_log[:-1]] == [e[:2] for e in want_log[:-1]]
    assert owed.gate == "g" and owed.routed is got and owed.w.shape == (H,)
    assert torch.equal(want, got)  # the stand-in gate add left the rows as they were


def _exit(stream, fb, steps, **change):
    from sglang.srt.layers.layer_boundary.exit import ExitPolicy, FfnExit
    from sglang.srt.layers.layer_boundary.ops import keep_output

    policy = ExitPolicy(SimpleNamespace(path_for=lambda fb: steps))
    kw = dict(
        forward_batch=fb,
        dp_step=None,
        steps=steps,
        output_move=keep_output,
        update=PLAIN_ADD,
        owes_sum=False,
    )
    kw.update(change.pop("complete", {}))
    ffn_exit = FfnExit.__new__(FfnExit)
    ffn_exit._stream, ffn_exit.boundary = stream, policy
    ffn_exit._update = change.pop("update", PLAIN_ADD)
    ffn_exit._complete = partial(policy._complete_now, **kw)
    ffn_exit._declared_sum = change.pop("declared_sum", None)
    ffn_exit.defer_moe_finalize = change.pop("defer", False)
    assert not change
    return ffn_exit


def _steps(**change):
    from sglang.srt.layers.layer_boundary.ops import keep_output

    output = SimpleNamespace(
        group=None, sum_impl=None, transform=None, update=PLAIN_ADD
    )
    output.__dict__.update(change.pop("output", {}))
    steps = SimpleNamespace(
        output=output,
        returns_over_dp=False,
        output_move=keep_output,
        writes_at_handoff=False,
    )
    steps.__dict__.update(change)
    return steps


def test_an_exit_that_keeps_the_output_records_the_owed_rows_unread():
    _, residual, _ = _case()
    stream = ResidualStream(residual)
    fb = SimpleNamespace(residual_stream=stream)
    ffn_exit = _exit(stream, fb, _steps())
    assert moe_prefill.exit_keeps_output(ffn_exit, fb)
    routed = torch.ones(5, H)
    assert ffn_exit.finish(routed) is routed
    assert stream.pending.value is routed and stream.pending.owed is None
    assert stream.residual is residual


@pytest.mark.parametrize(
    "change",
    [
        dict(defer=True),
        dict(declared_sum=object()),
        dict(update=SimpleNamespace(is_plain_add=True, applied_at_exit=True)),
        dict(update=SimpleNamespace(is_plain_add=False, applied_at_exit=False)),
        dict(complete=dict(dp_step=len)),
        dict(steps=dict(returns_over_dp=True)),
        dict(steps=dict(output_move=len)),
        dict(steps=dict(writes_at_handoff=True)),
        dict(steps=dict(output=dict(transform=object()))),
        dict(steps=dict(output=dict(sum_impl=object()))),
        dict(complete=dict(owes_sum=True), steps=dict(output=dict(group="other"))),
    ],
)
def test_every_other_exit_completes_the_gated_add_first(change):
    stream = ResidualStream(torch.zeros(5, H))
    fb = SimpleNamespace(residual_stream=stream)
    steps = _steps(**change.pop("steps", {}))
    assert not moe_prefill.exit_keeps_output(_exit(stream, fb, steps, **change), fb)


def test_an_exit_owing_a_one_rank_sum_keeps_the_output(monkeypatch):
    from sglang.srt.layers.layer_boundary.layout import SumGroup

    parallel = SimpleNamespace(tp_size=1, attn_tp_size=1, moe_ep_size=1, moe_tp_size=1)
    monkeypatch.setattr("sglang.srt.runtime_context.get_parallel", lambda: parallel)
    stream = ResidualStream(torch.zeros(5, H))
    fb = SimpleNamespace(residual_stream=stream)
    for group in (SumGroup.ATTN_TP, SumGroup.TP, SumGroup.MOE_OUTPUT):
        steps = _steps(output=dict(group=group))
        ffn_exit = _exit(stream, fb, steps, complete=dict(owes_sum=True))
        assert moe_prefill.exit_keeps_output(ffn_exit, fb)
    parallel.moe_ep_size = 2
    steps = _steps(output=dict(group=SumGroup.MOE_OUTPUT))
    ffn_exit = _exit(stream, fb, steps, complete=dict(owes_sum=True))
    assert not moe_prefill.exit_keeps_output(ffn_exit, fb)


def _body_layers(monkeypatch, log, n, owing_from=(), layer_input_takes=True):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    layers = [
        _decoder_layer(qwen3_5.Qwen3_5AttentionDecoderLayer, log) for _ in range(n)
    ]
    for i, layer in enumerate(layers):
        layer.ffn_boundary.exit = lambda fb, i=i: _Exit(log, f"ffn{i}")
    monkeypatch.setattr(moe_prefill, "ffn_input", lambda *a: None)
    monkeypatch.setattr(moe_prefill, "exit_keeps_output", lambda e, b: True)
    monkeypatch.setattr(
        qwen2_moe.Qwen2MoeSparseMoeBlock,
        "__call__",
        lambda self, *a, **k: log.append(("forward",)) or "out",
    )
    monkeypatch.setattr(
        moe_prefill,
        "forward",
        lambda m, x, b, pre, nc: log.append(("forward",)) or "out",
    )
    monkeypatch.setattr(
        moe_prefill,
        "forward_owing",
        lambda m, x, b, pre, nc: log.append(("forward_owing",)) or ("out", "owed"),
    )
    monkeypatch.setattr(
        moe_prefill,
        "layer_input",
        lambda layer, h, b, owed, fp8=None: (
            log.append(("layer_input", owed))
            or (torch.zeros(3, H) if layer_input_takes else None)
        ),
    )
    monkeypatch.setattr(moe_prefill, "complete", lambda owed: log.append(("complete",)))
    body = Qwen35Prefill.__new__(Qwen35Prefill)
    body.fp8 = None
    body.qk_rows = False
    body._linear = set()
    body.lm = SimpleNamespace(
        model=SimpleNamespace(layers=layers, start_layer=0, end_layer=n)
    )
    moe = moe_prefill.Layers(add_norm=owing_from)
    with patch(
        "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda i: contextlib.nullcontext()),
    ):
        return body._layers(torch.zeros(3, H), "pos", fb, None, moe=moe)


def test_the_body_hands_an_owed_output_to_the_next_layers_input(monkeypatch):
    log = _Log()
    log.known, log.stream = {}, "main"
    _body_layers(monkeypatch, log, 3, owing_from=(0, 1))
    names = [
        e[0]
        for e in log
        if e[0]
        in ("forward", "forward_owing", "layer_input", "complete", "attn.prepare")
    ]
    assert names == [
        "attn.prepare",
        "forward_owing",
        "layer_input",
        "forward_owing",
        "layer_input",
        "forward",
    ]


def test_the_body_completes_an_owed_output_its_next_input_does_not_take(monkeypatch):
    log = _Log()
    log.known, log.stream = {}, "main"
    _body_layers(monkeypatch, log, 2, owing_from=(0,), layer_input_takes=False)
    names = [
        e[0]
        for e in log
        if e[0]
        in ("forward", "forward_owing", "layer_input", "complete", "attn.prepare")
    ]
    assert names == [
        "attn.prepare",
        "forward_owing",
        "layer_input",
        "complete",
        "attn.prepare",
        "forward",
    ]


def _add_norm_model(n=3, capture=(), dense=()):
    _, _, norm = _case()
    layers = []
    for i in range(n):
        mlp = _taken_block()
        layer = SimpleNamespace(
            mlp=mlp if i not in dense else torch.nn.Linear(1, 1),
            input_layernorm=norm if i else GemmaRMSNorm(H),
            attn_boundary=_attn_stage(norm),
            _is_layer_to_capture=i in capture,
        )
        layers.append(layer)
    return layers


def test_the_build_takes_each_layer_but_the_stages_last(cuda_norm):
    with _build_env():
        layers = _add_norm_model()
        ids = [0, 1, 2]
        assert moe_prefill.add_norm_refusal(layers, 0, ids) == ""
        assert moe_prefill.add_norm_refusal(layers, 1, ids) == ""
        assert "last layer" in moe_prefill.add_norm_refusal(layers, 2, ids)


def test_the_build_refuses_layers_with_hooks(cuda_norm):
    with _build_env():
        for i in (0, 1):
            layers = _add_norm_model()
            layer = torch.nn.Module()
            layer.__dict__.update(vars(layers[i]))
            layer.register_forward_hook(lambda *a: None)
            layers[i] = layer
            assert "module hooks" in moe_prefill.add_norm_refusal(layers, 0, [0, 1, 2])


def test_a_transport_exit_completes_the_gated_add_first():
    from sglang.srt.layers.layer_boundary.stage import TransportExit

    exit_ = TransportExit.__new__(TransportExit)
    assert not moe_prefill.exit_keeps_output(exit_, SimpleNamespace())


@pytest.mark.parametrize(
    "model, why",
    [
        (dict(capture=(1,)), "captures"),
        (dict(dense=(1,)), "not a sparse MoE layer"),
        (dict(dense=(0,)), "not a sparse MoE block"),
    ],
)
def test_the_build_refuses_an_output_the_next_input_cannot_take(cuda_norm, model, why):
    with _build_env():
        layers = _add_norm_model(**model)
        assert why in moe_prefill.add_norm_refusal(layers, 0, [0, 1, 2])


@pytest.mark.parametrize(
    "plan, why",
    [
        (dict(fusions=object()), "fuses or moves"),
        (dict(paths_sp=True, enters_stack=True), "adapts more"),
        (dict(adapter=len), "adapts more"),
        (dict(entry=_entry(expected_sum=object())), "plain norm"),
        (dict(entry=_entry(read=UnfusedNormReadout())), "plain norm"),
    ],
)
def test_the_build_refuses_every_other_attention_input(cuda_norm, plan, why):
    _, _, norm = _case()
    adapter = plan.pop("adapter", None)
    sp = plan.pop("paths_sp", False)
    stage = _attn_stage(norm, **plan)
    if adapter is not None:
        stage.plan.paths[BatchVariant.ORDINARY].entry.attn_input_adapter = adapter
    if sp:
        stage.plan.paths[BatchVariant.SEQUENCE_PARALLEL] = stage.plan.paths[
            BatchVariant.ORDINARY
        ]
    layer = SimpleNamespace(attn_boundary=stage, input_layernorm=norm)
    assert why in moe_prefill.attn_input_refusal(layer)


@pytest.mark.parametrize(
    "change",
    [
        {},
        # tried by _run_entry only on an owed input, which layer_input never runs on
        dict(carried_fusions=(object(),), carried_read_fusions=(object(),)),
    ],
)
def test_the_build_takes_the_mirrored_attention_entry(cuda_norm, change):
    _, _, norm = _case()
    stage = _attn_stage(norm, entry=_entry(**change))
    layer = SimpleNamespace(attn_boundary=stage, input_layernorm=norm)
    assert moe_prefill.attn_input_refusal(layer) == ""


_ADD_NORM = "sgl_nvidia_mega.qwen3_5.sm103.moe.prefill_output_add_norm"


@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_build_takes_the_attention_input_sglang_binds_at_tp1(
    cuda_norm, cls, fp8_input
):
    with _tp1():
        layers = _tp1_layers(cls, fp8_input, n=2)
    entry = layers[1].attn_boundary.plan.paths[BatchVariant.ORDINARY].entry
    assert entry.prepare.keywords["carried_fusions"]
    for layer in layers:
        assert moe_prefill.attn_input_refusal(layer) == ""
        assert moe_prefill.ffn_input_refusal(layer) == ""
    with _build_env():
        for i, layer in enumerate(layers):
            layer.mlp = _tp1_block(_Log(), i)
        assert moe_prefill.add_norm_refusal(layers, 0, [0, 1]) == ""


@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_sglangs_tp1_ffn_exit_keeps_the_output(cls, fp8_input):
    with _tp1():
        (layer,) = _tp1_layers(cls, fp8_input)
        hidden, fb = _tp1_batch()
        x = layer.attn_boundary.prepare(hidden, fb)
        x = layer.attn_boundary.finish(_mixer(x), fb)
        x = layer.ffn_boundary.prepare(x, fb)
        out = torch.ones_like(x)
        with layer.ffn_boundary.exit(fb) as ffn_exit:
            assert moe_prefill.exit_keeps_output(ffn_exit, fb)
        assert ffn_exit.finish(out) is out
    stream = fb.residual_stream
    assert stream.pending.value is out and stream.pending.owed is None


@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_layer_input_leaves_what_sglangs_tp1_stages_leave(
    monkeypatch, cls, fp8_input
):
    monkeypatch.setattr(GemmaRMSNorm, "forward", _norm_forward)
    runs = []
    for which in ("sglang", "plugin"):
        calls = []
        owed = _owed_case()
        with _tp1():
            (layer,) = _tp1_layers(cls, fp8_input)
            _, fb = _tp1_batch()
            stream = fb.residual_stream
            stream.pending = None
            hidden = stream.record(owed.routed, PLAIN_ADD)
            if which == "sglang":
                _gated_add(owed.x, owed.w, owed.shared, owed.routed)
                x = layer.attn_boundary.prepare(hidden, fb)
            else:
                kernel = _add_norm_kernel(calls)
                with patch.dict("sys.modules", {_ADD_NORM: kernel}):
                    x = moe_prefill.layer_input(layer, hidden, fb, owed)
                # the slot took the batch: its kernel ran, the boundary did not
                assert x is not None and len(calls) == 1
        assert stream.pending is None
        runs.append((x, stream.residual))
    (x0, r0), (x1, r1) = runs
    assert _same(x0, x1) and _same(r0, r1)


_GEMMA_NORM_FP8 = "sgl_nvidia_mega.qwen3_5.sm103.gemma_norm_fp8"


class _ReadingMixer:
    def __init__(self, reader):
        self.in_proj_qkvz = reader

    def __call__(self, hidden_states, *args, **kwargs):
        self.in_proj_qkvz(hidden_states)
        return _mixer(hidden_states)


def _reader(calls, i, static_fp8=True):
    from sglang.srt.layers.linear import ColumnParallelLinear

    reader = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(reader)
    reader.bias, reader.skip_bias_add, reader.gather_output = None, False, False
    reader.tp_group = None
    method = SimpleNamespace()
    if static_fp8:
        method = modelopt_quant.ModelOptFp8LinearMethod.__new__(
            modelopt_quant.ModelOptFp8LinearMethod
        )
        method.use_marlin = method.use_sm120_fp8 = False
    method.apply = lambda layer, x, bias=None: (
        calls.append((i, x)) or torch.zeros(x.shape[0], 8)
    )
    reader.quant_method = method
    reader.input_scale = torch.tensor([0.25 * (i + 1)])
    return reader


def _dual_norm_kernel():

    def run(x, residual, weight, eps, scale):
        _gemma_add_norm(x, residual, weight, eps)
        return _e4m3(x, scale)

    return SimpleNamespace(
        gemma_fused_add_rmsnorm_fp8=run,
        supported=lambda x, r: True,
        cute_route=lambda device: True,
        build=lambda device, hidden: None,
    )


def _tp1_fp8_run(which, cls, fp8_input, taken):
    from sglang.srt.layers import linear as sglang_linear

    from sgl_nvidia_mega.qwen3_5 import sm103
    from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    log = _Log()
    log.known, log.stream = {}, "main"
    calls = {"nvfp4": [], "add_norm": [], "applied": []}
    dual = _dual_norm_kernel()
    kernels = {
        _NVFP4: _nvfp4_kernel(calls["nvfp4"]),
        _ADD_NORM: _add_norm_kernel(calls["add_norm"]),
        _GEMMA_NORM_FP8: dual,
    }
    linear = cls is qwen3_5.Qwen3_5LinearDecoderLayer
    fp8 = None
    with (
        _tp1_env(log, False, kernels),
        patch.object(sm103, "gemma_norm_fp8", dual, create=True),
        # TP1 with no sequence-parallel region: the forward goes to its quant method
        patch.object(
            sglang_linear, "get_forward", lambda: SimpleNamespace(sp_active=False)
        ),
        patch.object(
            moe_prefill,
            "gate_fp32",
            lambda x, w: torch.sigmoid((x.float() * w.float()).sum(-1)),
        ),
    ):
        layers = _tp1_layers(cls, fp8_input, 2)
        for i, layer in enumerate(layers):
            layer.mlp = _tp1_block(log, i)
            reader = _reader(calls["applied"], i, static_fp8=i in taken)
            layer.self_attention = layer.linear_attn = _ReadingMixer(reader)
            layer.qkv_proj = reader
        hidden, fb = _tp1_batch()
        if which == "sglang":
            for layer in layers:
                kw = {} if linear else {"positions": None}
                hidden = cls.forward(
                    layer, hidden_states=hidden, forward_batch=fb, **kw
                )
        else:
            model = SimpleNamespace(
                model=SimpleNamespace(layers=layers, start_layer=0, end_layer=2)
            )
            built, why = moe_prefill.build(model, range(2), add_norm=True)
            assert why == {} and built.add_norm == {0}
            ids = set(range(2)) if linear else set()
            fp8 = prefill_fp8.PrefillFp8.build(model, ("prefill_norm_fp8",), ids)
            body = Qwen35Prefill.__new__(Qwen35Prefill)
            body.fp8, body.lm, body._linear = fp8, model, ids
            body.qk_rows = False
            with fp8.body():
                hidden, _ = body._layers(
                    hidden, None, fb, lambda gdn, x: gdn(x, fb), moe=built
                )
    return hidden, fb.residual_stream, calls["applied"], calls["add_norm"], fp8


@pytest.mark.parametrize("taken", [(0, 1), (1,)], ids=["both", "second"])
@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_gated_add_in_the_next_norm_writes_its_linears_fp8_input(
    monkeypatch, caplog, cls, fp8_input, taken
):
    from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8

    monkeypatch.setattr(moe_prefill, "_LOGGED", set())
    caplog.set_level("INFO", logger=moe_prefill.logger.name)
    caplog.set_level("INFO", logger=prefill_fp8.logger.name)
    want, want_stream, want_in, _, _ = _tp1_fp8_run("sglang", cls, fp8_input, taken)
    got, got_stream, got_in, add_norm, fp8 = _tp1_fp8_run(
        "plugin", cls, fp8_input, taken
    )
    readers = list(fp8.norm_in.values())
    assert [r.input_scale.item() for r in readers] == [0.25 * (i + 1) for i in taken]
    # the second layer's input norm ran the gated add, at its linear's input scale
    ((*_, scale),) = add_norm
    assert scale is readers[-1].input_scale
    assert [i for i, _ in want_in] == [i for i, _ in got_in] == [0, 1]
    for (i, x), (_, y) in zip(want_in, got_in):
        assert x.dtype == torch.bfloat16
        if i in taken:
            # the E4M3 copy of SGLang's input to the linear, at the linear's scale
            assert y.dtype == torch.float8_e4m3fn
            q = _e4m3(x, torch.tensor(0.25 * (i + 1)))
            assert torch.equal(y.view(torch.uint8), q.view(torch.uint8))
        else:
            assert _same(x, y)
    assert "slot prefill_norm_fp8 runs at 5 rows" in caplog.text
    assert f"slot {moe_prefill.SLOT_ADD_NORM} runs" in caplog.text
    assert _same(want, got)
    assert _same(want_stream.residual, got_stream.residual)
    assert want_stream.pending.value is want and got_stream.pending.value is got


@pytest.mark.parametrize("inference", [False, True], ids=["tensor", "inference"])
def test_the_gated_adds_norm_scale_drops_a_pending_hand_off_and_skips_inference_tensors(
    inference,
):
    from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8

    norm = object()
    reader = _reader([], 0)
    fp8 = prefill_fp8.PrefillFp8.__new__(prefill_fp8.PrefillFp8)
    fp8.norm_in, fp8.active = {norm: reader}, True
    fp8._stash = (reader, None, None, 0)
    if inference:
        with torch.inference_mode():
            x = torch.zeros(5, H, dtype=torch.bfloat16)
    else:
        x = torch.zeros(5, H, dtype=torch.bfloat16)
    assert x.is_inference() is inference
    with patch.object(torch.Tensor, "is_cuda", True):
        scale = fp8.norm_scale(norm, x)
    assert fp8._stash is None
    if inference:
        assert scale is None
    else:
        assert scale is reader.input_scale


def _reads_x(log, x, out):
    """A shared expert that logs whether it reads x's storage itself."""
    return _Callable(
        lambda h: log.append(("shared", h.data_ptr() == x.data_ptr())) or out()
    )


@pytest.mark.parametrize("capture", [False, True])
def test_the_noclone_block_gives_the_shared_expert_the_input_itself(capture):
    runs = []
    for which in ("sglang", "plugin"):
        x = torch.zeros(7, H)
        log = _Log()
        log.known, log.stream = {"x": x}, "main"
        mlp = _block(log, x)
        mlp.shared_expert = _reads_x(log, x, lambda: "shared_out")
        with _block_env(log, capture):
            if which == "sglang":
                out = mlp.forward(x, None)
            else:
                out = moe_prefill.forward(mlp, x, None, None, noclone=True)
        runs.append((log, out))
    (want_log, want), (got_log, got) = runs
    # SGLang hands the dual-stream shared expert a clone of x; the slot hands it x
    assert ("shared", not capture) in want_log and ("shared", True) in got_log
    assert [e for e in got_log if e[0] != "shared"] == [
        e for e in want_log if e[0] != "shared"
    ]
    assert torch.equal(want, got)


def test_the_noclone_owing_block_gives_the_shared_expert_the_input(monkeypatch):
    x = torch.zeros(7, H, dtype=torch.bfloat16)
    log = _Log()
    log.known, log.stream = {"x": x}, "main"
    mlp = _block(log, x)
    mlp.shared_expert = _reads_x(log, x, lambda: torch.ones(7, H, dtype=torch.bfloat16))
    mlp.experts = _Callable(
        log.op("experts", lambda: torch.ones(7, H, dtype=torch.bfloat16))
    )
    mlp.shared_expert_gate.weight = torch.ones(1, H, dtype=torch.bfloat16)
    monkeypatch.setattr(moe_prefill, "gate_fp32", lambda x, w: "g")
    with _block_env(log, True):
        _, owed = moe_prefill.forward_owing(mlp, x, None, None, noclone=True)
    assert [e for e in log if e[0] == "shared"] == [("shared", True)]
    assert owed.x.data_ptr() == x.data_ptr()


def test_the_build_takes_the_cutedsl_block_without_the_input_clone():
    with _build_env():
        layer = SimpleNamespace(mlp=_taken_block())
        assert moe_prefill.noclone_refusal(layer) == ""


@pytest.mark.parametrize(
    "change, why",
    [
        ("dense", "not a sparse MoE block"),
        ("hooks", "module hooks"),
        ("block", "MegaMoE"),
        ("runner", "flashinfer_cutedsl"),
        ("fused_func", "CuTe DSL standard"),
    ],
)
def test_the_build_keeps_the_clone_for_a_routed_path_that_may_write(change, why):
    with _build_env():
        mlp = _taken_block()
        layer = SimpleNamespace(mlp=mlp)
        if change == "dense":
            layer.mlp = torch.nn.Linear(1, 1)
        elif change == "hooks":
            layer = torch.nn.Module()
            layer.mlp = mlp
            layer.register_forward_hook(lambda *a: None)
        elif change == "block":
            mlp._use_mega_moe = True
        elif change == "runner":
            mlp.experts.quant_method._moe_runner_backend = SimpleNamespace(
                is_flashinfer_cutedsl=lambda: False
            )
        else:
            mlp.experts.quant_method.runner.fused_func = len
        assert why in moe_prefill.noclone_refusal(layer)


def test_the_build_of_the_noclone_slot_alone_compiles_no_kernel(monkeypatch):
    import sys

    # neither norm kernel module may load
    for name in ("prefill_input_nvfp4", "prefill_output_add_norm"):
        monkeypatch.setitem(
            sys.modules, f"sgl_nvidia_mega.qwen3_5.sm103.moe.{name}", None
        )
    with _build_env():
        layers = [SimpleNamespace(mlp=_taken_block()) for _ in range(2)]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        built, why = moe_prefill.build(model, range(2), nvfp4=False, noclone=True)
    assert why == {} and built.noclone == built.blocks == built.ids == {0, 1}
    assert not built.nvfp4 and not built.add_norm


def test_the_body_runs_a_noclone_layers_block_without_the_clone(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import Qwen35Prefill

    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    log = _Log()
    log.known, log.stream = {"fb": fb}, "main"
    layer = _decoder_layer(qwen3_5.Qwen3_5AttentionDecoderLayer, log)
    monkeypatch.setattr(
        moe_prefill,
        "forward",
        lambda m, x, b, pre, noclone=False: log.append(("block", noclone)) or "out",
    )
    body = Qwen35Prefill.__new__(Qwen35Prefill)
    body.fp8 = None
    body.qk_rows = False
    body._linear = set()
    for moe, want in (
        (moe_prefill.Layers(noclone=[4]), True),
        (moe_prefill.Layers(nvfp4=[4]), False),
    ):
        monkeypatch.setattr(moe_prefill, "ffn_input", lambda *a: None)
        body._moe_layer(moe, 4, layer, torch.zeros(3, H), "pos", fb, None, None)
        assert log[-2] == ("block", want)


def _entry_norm_forward(self, x, residual=None, post_residual_addition=None):
    if residual is not None:
        return _norm_forward(self, x, residual, post_residual_addition)
    h = x.float()
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return (y * (1 + self.weight.float())).to(x.dtype)


def _tp1_stage(cls, fp8_input, first, log, n=2):
    layers = _tp1_layers(cls, fp8_input, n, first=first, terminal=not first)
    for i, layer in enumerate(layers):
        layer.mlp = _tp1_block(log, i)
    norm = GemmaRMSNorm(H, eps=1e-6)
    norm.weight.data.copy_(torch.linspace(-0.1, 0.1, H))
    return SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=n,
        pp_group=SimpleNamespace(is_first_rank=first, is_last_rank=not first),
        norm=norm,
        flashinfer_mnnvl_cutedsl_fusion=None,
        hidden_size=H,
    )


def _stage_inputs(first, rows):
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    g = torch.Generator().manual_seed(5)
    rand = lambda: torch.randn(rows, H, generator=g).to(torch.bfloat16)
    if first:
        return dict(input_embeds=rand(), pp_proxy_tensors=None)
    proxy = PPProxyTensors({"hidden_states": rand(), "residual": rand()})
    return dict(input_embeds=None, pp_proxy_tensors=proxy)


def _copied(inputs):
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    def copy(v):
        if isinstance(v, PPProxyTensors):
            return PPProxyTensors({k: t.clone() for k, t in v.tensors.items()})
        return None if v is None else v.clone()

    return {name: copy(v) for name, v in inputs.items()}


def _tensors(out):
    return out.tensors if hasattr(out, "tensors") else {"tensor": out}


@pytest.mark.parametrize("first", [True, False], ids=["first", "last"])
@pytest.mark.parametrize("cls, fp8_input", _TP1_READS)
def test_the_parity_step_of_a_tp1_pp2_stage_runs_the_slots_on_its_own_inputs(
    monkeypatch, caplog, cls, fp8_input, first
):
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    monkeypatch.setattr(moe_prefill, "_LOGGED", set())
    caplog.set_level("INFO", logger=moe_prefill.logger.name)
    caplog.set_level("INFO", logger=prefill.logger.name)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # a GDN layer's core is the mixer stand-in its SGLang layer calls
    monkeypatch.setattr(prefill_fi, "step", lambda gdn, x, core, norm=None: gdn(x))
    log = _Log()
    log.known, log.stream = {}, "main"
    calls = {"nvfp4": [], "add_norm": []}
    kernels = {
        _NVFP4: _nvfp4_kernel(calls["nvfp4"]),
        _ADD_NORM: _add_norm_kernel(calls["add_norm"]),
    }
    rows = 1  # a one-token prefill step
    ids = torch.zeros(rows, dtype=torch.long)

    def batch():
        return _Batch(forward_mode=ForwardMode.EXTEND, input_ids=ids)

    recorder = SimpleNamespace(with_current_layer=lambda i: contextlib.nullcontext())
    with (
        _tp1_env(log, False, kernels),
        patch.object(
            moe_prefill,
            "gate_fp32",
            lambda x, w: torch.sigmoid((x.float() * w.float()).sum(-1)),
        ),
        patch.object(GemmaRMSNorm, "forward", _entry_norm_forward),
        patch.object(
            qwen3_5, "get_global_expert_distribution_recorder", lambda: recorder
        ),
    ):
        model = _tp1_stage(cls, fp8_input, first, log)
        lm = SimpleNamespace(model=model)
        original = partial(qwen3_5.Qwen3_5ForCausalLM.forward, model)
        inputs = _stage_inputs(first, rows)
        pristine = _copied(inputs)
        # SGLang's forward writes into the inputs it runs on
        probe = _copied(inputs)
        original(ids, None, batch(), **probe)
        written = [
            n
            for name, v in probe.items()
            if v is not None
            for n, t in _tensors(v).items()
            if not _same(t, _tensors(pristine[name])[n])
        ]
        assert written == (["tensor"] if first else ["hidden_states", "residual"])
        want = original(ids, None, batch(), **_copied(pristine))
        built, why = moe_prefill.build(
            lm, range(2), nvfp4=True, add_norm=True, noclone=True
        )
        assert why == {} and built.nvfp4 == {0, 1} and built.add_norm == {0}
        body = prefill.Qwen35Prefill.__new__(prefill.Qwen35Prefill)
        body.lm, body.moe, body._core, body._engaged = lm, built, None, False
        body.fp8, body.qk_rows, body.front = None, False, True
        linear = cls is qwen3_5.Qwen3_5LinearDecoderLayer
        body._linear = set(range(2)) if linear else set()
        body.parity = prefill.PrefillParity()
        view = SimpleNamespace(
            state=SimpleNamespace(
                conv=(torch.zeros(1, 3, 2, 2),), temporal=torch.zeros(1, 3, 2, 2, 2)
            ),
            state_indices=torch.tensor([1], dtype=torch.int32),
            has_track=False,
        )
        got = body.forward_body(
            Step(mode="prefill", rows=rows, view=view),
            original,
            ids,
            None,
            batch(),
            **inputs,
        )
    assert "output_differ=0 state_differ=0" in caplog.text
    assert len(calls["nvfp4"]) == 2 and len(calls["add_norm"]) == 1
    got, want = _tensors(got), _tensors(want)
    assert got.keys() == want.keys()
    assert set(want) == ({"hidden_states", "residual"} if first else {"tensor"})
    for name in want:
        assert _same(got[name], want[name]), name
