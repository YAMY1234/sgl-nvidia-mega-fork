# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer as _REAL_LAYER

from sgl_nvidia_mega.qwen3_5.runtime import prefill_fp8


class _CudaScale(torch.Tensor):
    """A CPU tensor that reads as a CUDA one (the static scale's device check)."""

    @property
    def is_cuda(self):
        return True


def _scale(n=1, dtype=torch.float32):
    return torch.full((n,), 0.5, dtype=dtype).as_subclass(_CudaScale)


def _modelopt(**fields):
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
    method.use_marlin = False
    method.use_sm120_fp8 = False
    method.__dict__.update(fields)
    return method


def _linear(scale=None, method=None, **extra):
    return SimpleNamespace(
        quant_method=_modelopt() if method is None else method,
        input_scale=_scale() if scale is None else scale,
        **extra,
    )


@pytest.mark.parametrize(
    "linear, taken",
    [
        pytest.param(_linear(), True, id="static-fp8"),
        pytest.param(_linear(scale=torch.zeros(())), False, id="cpu-scale"),
        pytest.param(_linear(scale=_scale(2)), False, id="per-channel-scale"),
        pytest.param(
            _linear(scale=_scale(dtype=torch.bfloat16)), False, id="bf16-scale"
        ),
        pytest.param(_linear(method=_modelopt(use_marlin=True)), False, id="marlin"),
        pytest.param(_linear(method=_modelopt(use_sm120_fp8=True)), False, id="sm120"),
        pytest.param(_linear(base_layer=object()), False, id="lora"),
        pytest.param(_linear(method=object()), False, id="not-modelopt"),
        pytest.param(SimpleNamespace(), False, id="unquantized"),
        pytest.param(
            SimpleNamespace(quant_method=_modelopt()), False, id="dynamic-scale"
        ),
    ],
)
def test_a_linear_takes_a_prequantized_input_only_on_the_static_fp8_path(linear, taken):
    got = prefill_fp8.static_fp8_scale(linear)
    assert (got is linear.input_scale) if taken else got is None


class _Hidden(str):
    """Tagged hidden states with a row count."""

    rows = 3

    @property
    def shape(self):
        return (self.rows, 8)


class _Boundary:
    """attn_boundary / ffn_boundary: records each call and tags the hidden states."""

    def __init__(self, calls, name, rows=3):
        self.calls, self.name, self.rows = calls, name, rows

    def prepare(self, hidden, fb, **kw):
        self.calls.append((f"{self.name}.prepare", hidden, tuple(sorted(kw.items()))))
        out = _Hidden(f"{self.name}.prepared({hidden})")
        out.rows = self.rows
        return out

    def finish(self, hidden, fb):
        self.calls.append((f"{self.name}.finish", hidden))
        return f"{self.name}.finished({hidden})"

    def exit(self, fb):
        calls = self.calls

        class _Exit:
            defer_moe_finalize = "defer"

            def __enter__(self):
                calls.append(("exit.enter",))
                return self

            def __exit__(self, *exc):
                calls.append(("exit.leave",))

            def finish(self, out):
                calls.append(("exit.finish", out))
                return f"finished({out})"

        return _Exit()


class _Idle:
    def __init__(self, idle=False):
        self.idle = idle

    def is_idle(self):
        return self.idle


class _AttentionLayer:
    def __init__(self, calls, gated=True, o_proj=None, moe=False, rows=3):
        self.calls = calls
        self.attn_output_gate = gated
        self.o_proj = o_proj or _OProj(calls, _linear())
        self.attn_boundary = _Boundary(calls, "attn", rows)
        self.ffn_boundary = _Boundary(calls, "ffn")
        self.mlp = _Moe(calls) if moe else _Mlp(calls)

    def _prepare_qkv_gate(self, positions, hidden_states, forward_batch):
        self.calls.append(("prepare_qkv_gate", positions, hidden_states))
        gate = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
        return "q", "k", "v", gate

    def attn(self, q, k, v, fb):
        self.calls.append(("attn", q, k, v))
        return torch.ones(3, 8, dtype=torch.bfloat16)


class _OProj:
    def __init__(self, calls, linear):
        self.calls = calls
        self.quant_method = linear.quant_method
        self.input_scale = linear.input_scale

    def __call__(self, x):
        self.calls.append(("o_proj", x if isinstance(x, str) else "bf16"))
        return "o_proj_out", None


class _Mlp:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, hidden, *args, **kw):
        self.calls.append(("mlp", hidden, args, tuple(sorted(kw.items()))))
        return f"mlp({hidden})"


class _Moe(_Mlp):
    pass


@pytest.fixture
def sglang_layer(monkeypatch):
    from sglang.srt.models import qwen2_moe, qwen3_5

    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    calls = []

    def stock_gate(attn_output, gate, inplace=False, **kw):
        calls.append(("gate", "bf16", inplace, tuple(sorted(kw))))
        return attn_output

    def fp8_gate(attn_output, gate, scale):
        calls.append(("gate", "fp8", scale))
        return "e4m3"

    monkeypatch.setattr(qwen3_5, "fused_sigmoid_mul", stock_gate)
    monkeypatch.setattr(qwen3_5, "Qwen2MoeSparseMoeBlock", _Moe)
    monkeypatch.setattr(qwen2_moe, "Qwen2MoeSparseMoeBlock", _Moe)
    monkeypatch.setattr(qwen3_5, "Qwen3_5AttentionDecoderLayer", _AttentionLayer)
    monkeypatch.setattr(gate_quant, "supported", lambda attn_output, gate: True)
    monkeypatch.setattr(gate_quant, "fused_sigmoid_mul_fp8", fp8_gate)
    return SimpleNamespace(calls=calls)


def _stock_forward(layer, fb, capture=None):
    """SGLang's own forward and self_attention (not the stand-in's) on the layer."""
    layer.self_attention = lambda **kw: _REAL_LAYER.self_attention(layer, **kw)
    return _REAL_LAYER.forward(
        layer,
        positions="pos",
        hidden_states="h",
        forward_batch=fb,
        captured_last_layer_outputs=capture,
    )


@pytest.mark.parametrize("moe", [False, True], ids=["dense-mlp", "moe"])
@pytest.mark.parametrize("capture", [None, "aux"], ids=["", "captured"])
def test_the_attention_layer_runs_sglangs_forward_with_the_gate_writing_fp8(
    sglang_layer, moe, capture
):
    fb = SimpleNamespace(forward_mode=_Idle())
    stock = _AttentionLayer(sglang_layer.calls, moe=moe)
    want_out = _stock_forward(stock, fb, capture)
    want = list(sglang_layer.calls)
    sglang_layer.calls.clear()

    built = prefill_fp8.PrefillFp8()
    mirrored = _AttentionLayer(sglang_layer.calls, moe=moe)
    built.gate[3] = mirrored.o_proj.input_scale
    got_out = built.attention_layer(3, mirrored, "pos", "h", fb, capture)
    got = list(sglang_layer.calls)

    assert got_out == want_out
    # the same calls in the same order; the gate writes o_proj's E4M3 input instead
    i = next(n for n, c in enumerate(want) if c[0] == "gate")
    assert want[i] == ("gate", "bf16", True, ())
    assert got[i] == ("gate", "fp8", mirrored.o_proj.input_scale)
    assert want[i + 1] == ("o_proj", "bf16") and got[i + 1] == ("o_proj", "e4m3")
    assert got[:i] == want[:i] and got[i + 2 :] == want[i + 2 :]


class _Cuda(torch.Tensor):
    """A CPU tensor that reads as a CUDA one (the kernels' layout checks)."""

    @property
    def is_cuda(self):
        return True


def _cuda(t):
    return t.as_subclass(_Cuda)


@pytest.mark.parametrize(
    "gate, ok",
    [
        pytest.param(lambda: torch.zeros(3, 8, dtype=torch.bfloat16), True, id="flat"),
        pytest.param(
            lambda: torch.zeros(3, 2, 8, dtype=torch.bfloat16)[:, :, 4:],
            True,
            id="strided-heads",
        ),
        pytest.param(
            lambda: torch.zeros(3, 16, dtype=torch.bfloat16)[:, 8:],
            True,
            id="flat-slice",
        ),
        pytest.param(lambda: torch.zeros(3, 8), False, id="fp32"),
        pytest.param(
            lambda: torch.zeros(8, 3, dtype=torch.bfloat16).t(),
            False,
            id="inner-stride",
        ),
        pytest.param(
            lambda: torch.zeros(3, 4, dtype=torch.bfloat16), False, id="width"
        ),
    ],
)
def test_the_gate_kernel_takes_the_layers_gate_layouts(gate, ok):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    attn = _cuda(torch.zeros(3, 8, dtype=torch.bfloat16))
    assert gate_quant.supported(attn, gate()) is ok
    # the output must be a contiguous CUDA BF16 [tokens, hidden]
    assert not gate_quant.supported(attn.as_subclass(torch.Tensor), gate())


class _Launch:
    """Records a Triton launch, ``kernel[grid](*args, **kwargs)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def _defaults(kernel, *names):
    params = inspect.signature(getattr(kernel, "fn", kernel)).parameters
    return {name: params[name].default for name in names}


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no-pdl"])
@pytest.mark.parametrize(
    "tokens, layout, block, strides",
    [
        pytest.param(3, "heads", 1024, (16, 8, 4), id="strided-heads"),
        pytest.param(3, "flat-slice", 1024, (16, 8, 8), id="flat-slice"),
        pytest.param(1024, "heads", 2048, (16, 8, 4), id="from-1024-tokens"),
    ],
)
def test_the_gate_runs_sglangs_launch_with_its_strides_as_constants(
    monkeypatch, tokens, layout, block, strides, pdl
):
    from sglang.kernels.jit import utils as jit_utils

    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    launch = _Launch()
    monkeypatch.setattr(gate_quant, "_gate_quant_kernel", launch)
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", lambda: pdl)
    attn = _cuda(torch.zeros(tokens, 8, dtype=torch.bfloat16))
    if layout == "heads":
        gate = torch.zeros(tokens, 2, 8, dtype=torch.bfloat16)[:, :, 4:]
    else:
        gate = torch.zeros(tokens, 16, dtype=torch.bfloat16)[:, 8:]
    scale = torch.ones(1)
    q = gate_quant.fused_sigmoid_mul_fp8(attn, gate, scale)
    [(grid, args, kwargs)] = launch.calls
    assert grid == (tokens, 1) and args[4] == 8
    assert all(a is b for a, b in zip(args, (attn, gate, scale, q)))
    row, head, head_dim = strides
    # SGLang's block per token count and its wait-only PDL; the gate's strides
    assert kwargs == {
        "BLOCK": block,
        "STRIDED": True,
        "GATE_ROW": row,
        "GATE_HEAD": head,
        "HEAD_DIM": head_dim,
        "USE_PDL": pdl,
        "EARLY_TRIGGER": False,
        "num_warps": 4,
        **({"launch_pdl": True} if pdl else {}),
    }


def test_the_verify_gate_keeps_its_launch(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    kernel = gate_quant._gate_quant_kernel
    assert _defaults(kernel, "STRIDED", "USE_PDL", "EARLY_TRIGGER") == {
        "STRIDED": False,
        "USE_PDL": True,
        "EARLY_TRIGGER": True,
    }
    launch = _Launch()
    monkeypatch.setattr(gate_quant, "_gate_quant_kernel", launch)
    attn = torch.zeros(7, 8, dtype=torch.bfloat16)
    gate_quant.gate_quant(attn, torch.zeros_like(attn), torch.ones(1))
    [(grid, args, kwargs)] = launch.calls
    assert grid == (7, 1) and len(args) == 5 and args[4] == 8
    assert kwargs == {"BLOCK": 1024, "num_warps": 4, "launch_pdl": True}


def test_the_attention_layer_keeps_the_stock_gate_on_another_layout(
    sglang_layer, monkeypatch
):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    monkeypatch.setattr(gate_quant, "supported", lambda attn_output, gate: False)
    built = prefill_fp8.PrefillFp8()
    layer = _AttentionLayer(sglang_layer.calls)
    built.gate[0] = layer.o_proj.input_scale
    built.attention_layer(
        0, layer, "pos", "h", SimpleNamespace(forward_mode=_Idle()), None
    )
    assert ("gate", "bf16", True, ()) in sglang_layer.calls
    assert ("o_proj", "bf16") in sglang_layer.calls


@pytest.mark.parametrize("idle, rows", [(True, 3), (False, 0)], ids=["idle", "empty"])
def test_an_idle_or_empty_attention_layer_skips_the_attention_as_sglang_does(
    sglang_layer, idle, rows
):
    fb = SimpleNamespace(forward_mode=_Idle(idle=idle))
    built = prefill_fp8.PrefillFp8()
    layer = _AttentionLayer(sglang_layer.calls, rows=rows)
    built.gate[0] = layer.o_proj.input_scale
    got = built.attention_layer(0, layer, "pos", "h", fb, None)
    assert not any(c[0] in ("attn", "gate", "o_proj") for c in sglang_layer.calls)
    want_calls = list(sglang_layer.calls)
    sglang_layer.calls.clear()
    assert _stock_forward(_AttentionLayer(sglang_layer.calls, rows=rows), fb) == got
    assert sglang_layer.calls == want_calls


def _model(layers, start=0):
    return SimpleNamespace(
        model=SimpleNamespace(start_layer=start, end_layer=len(layers), layers=layers)
    )


def test_the_build_takes_each_gated_static_fp8_attention_layer(sglang_layer, caplog):
    calls = sglang_layer.calls
    layers = [
        "gdn",
        _AttentionLayer(calls),
        _AttentionLayer(calls, gated=False),
        _AttentionLayer(calls, o_proj=_OProj(calls, _linear(scale=_scale(2)))),
        _AttentionLayer(calls),
    ]
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(_model(layers), ("prefill_gate_fp8",), {0})
    assert sorted(built.gate) == [1, 4]
    assert built.gate[1] is layers[1].o_proj.input_scale
    assert built.rows() == ["prefill_gate_fp8"]
    assert caplog.text.count("slot prefill_gate_fp8 uses stock on some layers") == 1
    assert "(layer 2: no output gate)" in caplog.text


def test_the_build_runs_no_row_it_was_not_given_or_no_layer_takes(sglang_layer, caplog):
    calls = sglang_layer.calls
    layers = [_AttentionLayer(calls, o_proj=_OProj(calls, _linear(method=object())))]
    assert prefill_fp8.PrefillFp8.build(_model(layers), (), set()).rows() == []
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(
            _model(layers), ("prefill_gate_fp8",), set()
        )
    assert built.rows() == [] and not built.gate
    assert (
        "slot prefill_gate_fp8 uses stock on every layer (layer 0: o_proj is not a "
        "static per-tensor FP8 linear)"
    ) in caplog.text


def test_the_engagement_line_is_logged_once_a_row(caplog):
    built = prefill_fp8.PrefillFp8()
    with caplog.at_level("INFO"):
        for rows in (5, 9):
            built._engage("prefill_gate_fp8", rows)
    assert caplog.text.count("slot prefill_gate_fp8 runs at 5 rows") == 1
    assert "runs at 9 rows" not in caplog.text


def _prefill(monkeypatch, layers, rows, front=True):
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    class _Linear:
        pass

    monkeypatch.setattr("sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer", _Linear)
    monkeypatch.setattr(prefill.parity, "ENABLED", False)
    monkeypatch.setattr(prefill.Qwen35Prefill, "_build_front", lambda self, m: front)
    gdn = _Linear()
    gdn.linear_attn = SimpleNamespace(
        attn=SimpleNamespace(q_dim=4096, k_dim=4096, v_dim=8192), attn_tp_size=1
    )
    return prefill.Qwen35Prefill(
        _model([gdn, *layers]),
        SimpleNamespace(tokens_per_request=7, binding=None),
        False,
        label="TP1/PP2",
        split_conv=False,
        front=True,
        fp8_rows=rows,
    )


def test_the_prefill_body_builds_the_rows_and_names_them(sglang_layer, monkeypatch):
    built = _prefill(
        monkeypatch, [_AttentionLayer(sglang_layer.calls)], ("prefill_gate_fp8",)
    )
    assert sorted(built.fp8.gate) == [1]
    assert built.summary() == (
        "prefill TP1/PP2; gdn_prefill_front built (prefill body); "
        "prefill_gate_fp8 built"
    )


def test_the_rows_use_stock_without_the_fronts_body(sglang_layer, monkeypatch, caplog):
    with caplog.at_level("INFO"):
        built = _prefill(
            monkeypatch,
            [_AttentionLayer(sglang_layer.calls)],
            ("prefill_gate_fp8",),
            front=False,
        )
    assert built.fp8 is None
    assert (
        "slots ['prefill_gate_fp8'] use stock (they run in the prefill body, which "
        "gdn_prefill_front builds)"
    ) in caplog.text


def test_the_body_runs_a_taken_attention_layer_through_the_row(
    sglang_layer, monkeypatch
):
    taken = _AttentionLayer(sglang_layer.calls)
    other = _AttentionLayer(sglang_layer.calls, gated=False)
    built = _prefill(monkeypatch, [taken, other], ("prefill_gate_fp8",))
    ran = []
    monkeypatch.setattr(
        built.fp8,
        "attention_layer",
        lambda i, layer, *args: ran.append(("row", i, layer)) or "h",
    )
    other.__class__ = type(
        "_Stock",
        (_AttentionLayer,),
        {"__call__": lambda self, **kw: ran.append(("stock", self)) or "h"},
    )
    model = built.lm.model
    monkeypatch.setattr(model, "start_layer", 1)
    built._layers("h0", "pos", "fb", gdn_step=None)
    assert ran == [("row", 1, taken), ("stock", other)]


class _SiluAndMul:
    """Stands in for SGLang's SiluAndMul (the class is patched to it)."""

    def __init__(self, calls, method=None):
        self.calls = calls
        self._forward_method = method

    def forward_cuda(self, x):
        raise AssertionError("not called")

    def __call__(self, gate_up):
        self.calls.append(("act_fn", "bf16"))
        return "act"


class _Proj:
    def __init__(self, calls, name, linear=None, width=32):
        self.calls, self.name = calls, name
        linear = linear or _linear()
        self.quant_method = linear.quant_method
        self.input_scale = linear.input_scale
        self.input_size_per_partition = width

    def __call__(self, x):
        tag = x if isinstance(x, str) else "tensor"
        self.calls.append((self.name, tag))
        if self.name == "gate_up_proj":
            return _cuda(torch.zeros(3, 2 * 16, dtype=torch.bfloat16)), None
        return f"{self.name}({tag})", None


class _MLP:
    def __init__(self, calls, down=None, act=None, width=32):
        self.calls = calls
        self.gate_up_proj = _Proj(calls, "gate_up_proj")
        self.down_proj = _Proj(calls, "down_proj", down, width)
        self.act_fn = act or _SiluAndMul(calls)
        self._enable_silu_fp4_quant_fusion = False


class _MoeBlock:
    def __init__(self, shared):
        self.shared_expert = shared


@pytest.fixture
def mlps(monkeypatch):
    from sglang.srt.layers import activation
    from sglang.srt.models import qwen2_moe

    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_act_fp8

    calls = []
    monkeypatch.setattr(qwen2_moe, "Qwen2MoeMLP", _MLP)
    monkeypatch.setattr(qwen2_moe, "Qwen2MoeSparseMoeBlock", _MoeBlock)
    monkeypatch.setattr(activation, "SiluAndMul", _SiluAndMul)
    monkeypatch.setattr(
        prefill_act_fp8,
        "silu_and_mul_static_fp8",
        lambda gate_up, scale: calls.append(("act", "fp8", scale)) or "e4m3",
    )
    monkeypatch.setattr(
        prefill_act_fp8, "build", lambda device: calls.append(("build", device))
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return calls


_REAL_MLP_FORWARD = [None]


@pytest.fixture(autouse=True)
def _keep_real_mlp_forward():
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    if _REAL_MLP_FORWARD[0] is None:
        _REAL_MLP_FORWARD[0] = Qwen2MoeMLP.forward


@pytest.mark.parametrize("active", [True, False], ids=["in-body", "outside-body"])
def test_the_mlp_runs_sglangs_forward_with_the_act_writing_fp8_in_the_body(
    mlps, active
):
    stock = _MLP(mlps)
    want_out = _REAL_MLP_FORWARD[0](stock, "x")
    want = list(mlps)
    mlps.clear()

    built = prefill_fp8.PrefillFp8()
    mirrored = _MLP(mlps)
    built.act[mirrored] = mirrored.down_proj.input_scale
    built.active = active
    got_out = built.mlp_forward(mirrored, "x")
    got = list(mlps)
    if not active:
        # outside the body: SGLang's forward
        assert (got_out, got) == (want_out, want)
        return
    assert want == [
        ("gate_up_proj", "x"),
        ("act_fn", "bf16"),
        ("down_proj", "act"),
    ]
    assert got == [
        ("gate_up_proj", "x"),
        ("act", "fp8", mirrored.down_proj.input_scale),
        ("down_proj", "e4m3"),
    ]
    assert got_out == "down_proj(e4m3)"


def test_the_mlp_keeps_sglangs_fp4_fusion(mlps, monkeypatch):
    from sglang.srt.models import qwen2_moe

    monkeypatch.setattr(qwen2_moe, "can_run_silu_fp4_quant_fusion", lambda m, g: True)
    monkeypatch.setattr(
        qwen2_moe, "silu_and_mul_fp4_quant", lambda m, g: mlps.append("fp4") or "nvfp4"
    )
    built = prefill_fp8.PrefillFp8()
    mlp = _MLP(mlps)
    built.act[mlp], built.active = mlp.down_proj.input_scale, True
    assert built.mlp_forward(mlp, "x") == "down_proj(nvfp4)"
    assert mlps == [("gate_up_proj", "x"), "fp4", ("down_proj", "nvfp4")]


class _MoeLayer:
    def __init__(self, mlp):
        self.mlp = mlp


def test_the_build_gives_each_fusable_mlp_the_forward_and_builds_the_kernel_once(
    mlps, caplog
):
    shared = _MLP(mlps)
    dense = _MLP(mlps)
    layers = [
        _MoeLayer(_MoeBlock(shared)),
        _MoeLayer(dense),
        _MoeLayer(
            _MoeBlock(_MLP(mlps, down=_linear(method=_modelopt(use_marlin=True))))
        ),
        _MoeLayer(_MoeBlock(_MLP(mlps, width=24))),
        _MoeLayer(_MoeBlock(_MLP(mlps, act=_SiluAndMul(mlps, method=print)))),
        _MoeLayer(_MoeBlock(None)),
    ]
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(
            _model(layers), ("prefill_act_fp8",), set()
        )
    assert list(built.act) == [shared, dense]
    assert built.rows() == ["prefill_act_fp8"]
    assert mlps == [("build", 0)]
    # SGLang's MoE block calls the shared expert: the forward is the package's
    assert shared.forward.func == built.mlp_forward and shared.forward.args == (shared,)
    assert "forward" not in vars(layers[2].mlp.shared_expert)
    assert (
        "slot prefill_act_fp8 uses stock on some layers (layer 2: down_proj is not a "
        "static per-tensor FP8 linear)"
    ) in caplog.text


@pytest.mark.parametrize(
    "mlp, why",
    [
        pytest.param(
            {"down": _linear(scale=_scale(2))},
            "down_proj is not a static per-tensor FP8 linear",
            id="per-channel",
        ),
        pytest.param(
            {"width": 24},
            "down_proj's input width is not a multiple of the kernel's vector",
            id="width",
        ),
        pytest.param(
            {"act": "native"},
            "the MLP's activation is not SGLang's CUDA SiluAndMul",
            id="native-activation",
        ),
        pytest.param(
            {"forward": True}, "the MLP's forward is not SGLang's", id="own-forward"
        ),
        pytest.param({}, "", id="fusable"),
    ],
)
def test_an_mlp_takes_the_fused_act_only_on_sglangs_cuda_silu(mlps, mlp, why):
    act = None
    if mlp.pop("act", None):
        act = _SiluAndMul(mlps, method=_SiluAndMul(mlps).__call__)
    forward = mlp.pop("forward", False)
    built = _MLP(mlps, act=act, **mlp)
    if forward:
        built.forward = print
    assert prefill_fp8._act_unsupported(built) == why


def test_the_fused_act_runs_inside_the_body_only():
    built = prefill_fp8.PrefillFp8()
    with built.body():
        assert built.active
    assert not built.active
    with pytest.raises(RuntimeError):
        with built.body():
            raise RuntimeError("step failed")
    assert not built.active


def _cuda_function_digest(text, name):
    import hashlib
    import re

    found = re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\(", text)
    if found is None:
        return None
    start = text.rfind("template <", 0, found.start())
    if (
        start < 0
        or ";" in text[start : found.start()]
        or "}" in text[start : found.start()]
    ):
        start = found.start()
    depth, end = 0, text.index("{", found.end())
    for end in range(end, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            break
    body = re.sub(r"//[^\n]*", "", text[start : end + 1])
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return hashlib.sha256(" ".join(body.split()).encode()).hexdigest()[:16]


def test_the_cuda_code_the_act_kernel_follows_is_unchanged():
    from pathlib import Path

    import sglang

    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_act_fp8

    root = Path(sglang.__file__).resolve().parent
    for key, digest in prefill_act_fp8.SGLANG_CUDA_MIRRORS.items():
        path, name = key.split(":")
        got = _cuda_function_digest((root / path).read_text(), name)
        assert got is not None, f"SGLang no longer defines {key}"
        assert got in ((digest,) if isinstance(digest, str) else digest), (
            f"SGLang changed {key}, whose body prefill_act_fp8.cuh follows: re-read "
            f"it, update the kernel, then pin {got!r}"
        )


def test_the_body_runs_the_layers_inside_the_fused_act_scope(monkeypatch):
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    built = prefill.Qwen35Prefill.__new__(prefill.Qwen35Prefill)
    built.fp8 = prefill_fp8.PrefillFp8()
    built.moe = None
    built.front, built.qk_rows = True, False
    built._core = "core"
    stage = SimpleNamespace(attn_boundary=SimpleNamespace(from_pp=lambda t, fb: "h"))
    built.lm = SimpleNamespace(
        model=SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
            start_layer=0,
            layers=[stage],
        )
    )
    seen = []
    monkeypatch.setattr(
        built,
        "_layers",
        lambda hidden, *args, **kw: seen.append(built.fp8.active) or (hidden, []),
        raising=False,
    )
    monkeypatch.setattr(residual_batch, "to_pp", lambda hidden, fb: ("pp", hidden))
    assert built._body("ids", "pos", "fb", None, "proxy") == ("pp", "h")
    assert seen == [True] and not built.fp8.active


def _gated_norm(**kw):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    return RMSNorm(4, eps=1e-6, norm_before_gate=True, **kw)


@pytest.mark.parametrize(
    "norm, why",
    [
        pytest.param(_gated_norm(), "", id="one-group"),
        pytest.param(
            _gated_norm(group_size=2),
            "the gated norm has groups or a bias",
            id="groups",
        ),
        pytest.param(
            torch.nn.LayerNorm(4),
            "the gated norm is not SGLang's Triton gated RMSNorm",
            id="other-norm",
        ),
    ],
)
def test_the_gated_norm_kernel_takes_sglangs_one_group_norm(norm, why):
    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    assert norm_fp8.unsupported(norm) == why


def _norm_inputs(tokens=3, heads=2, n=4):
    x = torch.zeros(tokens * heads, n, dtype=torch.bfloat16)
    # the prefill projection's z: a [tokens, heads, n] view with a wider row stride
    z = torch.zeros(tokens, heads * n + 8, dtype=torch.bfloat16)[:, : heads * n]
    return x, z.view(tokens, heads, n)


@pytest.mark.parametrize(
    "change, ok",
    [
        pytest.param({}, True, id="strided-3d-gate"),
        pytest.param({"z2d": True}, True, id="2d-gate"),
        pytest.param({"cpu": True}, False, id="cpu"),
        pytest.param({"fp32": True}, False, id="fp32"),
        pytest.param({"x_stride": True}, False, id="x-inner-stride"),
        pytest.param({"z_rows": True}, False, id="gate-rows"),
    ],
)
def test_the_gated_norm_kernel_takes_the_prefill_layout(change, ok):
    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    x, z = _norm_inputs()
    if change.get("z2d"):
        z = torch.zeros_like(x)
    if change.get("fp32"):
        x = x.float()
    if change.get("x_stride"):
        x = torch.zeros(x.shape[1], x.shape[0], dtype=x.dtype).t()
    if change.get("z_rows"):
        z = z[:2]
    if not change.get("cpu"):
        x = _cuda(x)
    assert norm_fp8.supported(x, z) is ok


@pytest.fixture
def norm_launch(monkeypatch):
    """The gated norm's launch recorded, with SGLang's launch helpers stood in."""
    from sglang.kernels.jit import utils as jit_utils
    from sglang.kernels.ops.attention.fla import layernorm_gated
    from sglang.srt.utils import device as device_utils

    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    out = SimpleNamespace(launch=_Launch(), tiles=[], devices=[], pdl=True, take=True)
    out.kernel = norm_fp8._layer_norm_fwd_1pass_kernel

    def rows_per_block(M, device):
        return 4

    rows_per_block.__module__ = layernorm_gated.__name__

    @contextlib.contextmanager
    def device_context(device):
        out.devices.append(device)
        yield

    def use_tiles(**kwargs):
        out.tiles.append(kwargs)
        return out.take

    monkeypatch.setattr(norm_fp8, "_layer_norm_fwd_1pass_kernel", out.launch)
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", lambda: out.pdl)
    monkeypatch.setattr(layernorm_gated, "calc_rows_per_block", rows_per_block)
    monkeypatch.setattr(layernorm_gated, "_use_prefill_tiles", use_tiles)
    monkeypatch.setattr(device_utils, "device_context", device_context)
    return out


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no-pdl"])
@pytest.mark.parametrize(
    "tiles", [True, False, None], ids=["prefill-rows", "one-tile", "no-prefill-rows"]
)
def test_the_gated_norm_runs_sglangs_prefill_launch(
    norm_launch, monkeypatch, tiles, pdl
):
    from sglang.kernels.ops.attention.fla import layernorm_gated

    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    norm_launch.pdl, norm_launch.take = pdl, tiles
    if tiles is None:  # an SGLang without the prefill-row launch
        monkeypatch.delattr(layernorm_gated, "_use_prefill_tiles")
    norm = _gated_norm(activation="sigmoid")
    x, z = _norm_inputs()
    x = _cuda(x)
    scale = torch.ones(1)
    q = norm_fp8.prefill_rms_norm_gated_fp8(norm, x, z, scale)
    [(grid, args, kwargs)] = norm_launch.launch.calls
    assert norm_launch.devices == [x.device]
    assert all(a is b for a, b in zip(args, (x, q))) and args[3] is z
    assert torch.equal(args[2], norm.weight) and args[4] is scale
    assert args[6:] == (4, 4, 0, z.stride(0), z.stride(1), 6, 4, norm.eps)
    if tiles is not None:
        assert norm_launch.tiles == [
            {"M": 6, "num_warps": 1, "rows_per_block": 4, "quant_heads": 0}
        ]
    rows = 16 if tiles else 4
    assert grid == (-(-6 // rows), 1)
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True, "EARLY_DEPENDENTS": False}
    assert kwargs == {
        "BLOCK_N": 4,
        "ROWS_PER_BLOCK": rows,
        "Z_IS_3D": True,
        "Z_HEADS": 2,
        "NORM_BEFORE_GATE": True,
        "ACTIVATION": "sigmoid",
        "OUT_DTYPE": kwargs["OUT_DTYPE"],
        "num_warps": 4 if tiles else 1,
        **(pdl_kwargs if pdl else {}),
        **({"EARLY_Z": True} if tiles else {}),
        "STORE_8": True,
    }


def test_the_verify_norm_keeps_its_launch(norm_launch):
    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    assert _defaults(norm_launch.kernel, "EARLY_Z", "STORE_8") == {
        "EARLY_Z": False,
        "STORE_8": False,
    }
    x, z = _norm_inputs()
    norm_fp8.rms_norm_gated_fp8(x, torch.ones(4), 1e-6, z, torch.ones(1))
    [(grid, args, kwargs)] = norm_launch.launch.calls
    assert grid == (2, 1) and kwargs["ROWS_PER_BLOCK"] == 4
    assert "EARLY_Z" not in kwargs and "STORE_8" not in kwargs
    assert not norm_launch.tiles and not norm_launch.devices


def test_the_step_hands_out_proj_the_norm_hooks_output(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi

    tokens, heads, n = 3, 2, 4
    qkvz = torch.zeros(tokens, 8)
    z = torch.randn(tokens, heads, n)
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.triton_gdn_fused_proj."
        "qwen3_5_gdn_prefill_projection_views",
        lambda *args: (torch.zeros(tokens, 4), z, "b", "a"),
    )
    seen = {}
    gdn = SimpleNamespace(
        attn=SimpleNamespace(num_v_heads=heads, head_v_dim=n),
        num_k_heads=1,
        num_v_heads=heads,
        attn_tp_size=1,
        head_k_dim=n,
        head_v_dim=n,
        _forward_input_proj=lambda x: (qkvz, "ba"),
        norm=lambda core, z_: pytest.fail("ran the stock norm"),
        out_proj=lambda x: (seen.setdefault("out", x), None),
    )

    def norm(gdn_, core, z_):
        assert gdn_ is gdn and core.shape == (tokens * heads, n) and z_ is z
        return torch.full(core.shape, 7, dtype=torch.uint8)

    run_core = lambda attn, mixed, a, b, output: output.fill_(1)  # noqa: E731
    out = prefill_fi.step(gdn, torch.zeros(tokens, 8), run_core, norm)
    assert out is seen["out"] and out.shape == (tokens, heads * n)
    assert bool((out == 7).all())


class _Module(SimpleNamespace):
    """A stand-in module: hashed and compared by identity, as an nn.Module."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _gdn_layer(norm=None, out_proj=None):
    return SimpleNamespace(
        linear_attn=_Module(
            norm=_gated_norm() if norm is None else norm,
            out_proj=out_proj or _linear(),
        )
    )


def test_the_build_takes_each_gdn_layer_whose_norm_feeds_a_static_fp8_out_proj(
    sglang_layer, caplog
):
    layers = [
        _gdn_layer(),
        _gdn_layer(norm=_gated_norm(group_size=2)),
        _gdn_layer(out_proj=_linear(method=_modelopt(use_marlin=True))),
        _gdn_layer(),
        _AttentionLayer(sglang_layer.calls),
    ]
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(
            _model(layers), ("gdn_prefill_norm_fp8",), {0, 1, 2, 3}
        )
    assert built.norm == {
        layers[0].linear_attn: layers[0].linear_attn.out_proj.input_scale,
        layers[3].linear_attn: layers[3].linear_attn.out_proj.input_scale,
    }
    assert not built.gate and built.rows() == ["gdn_prefill_norm_fp8"]
    assert (
        "slot gdn_prefill_norm_fp8 uses stock on some layers (layer 1: the gated "
        "norm has groups or a bias)"
    ) in caplog.text


def test_the_gdn_norm_writes_fp8_where_the_build_took_the_layer(monkeypatch, caplog):
    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    calls = []
    monkeypatch.setattr(
        norm_fp8,
        "prefill_rms_norm_gated_fp8",
        lambda norm, x, z, s: calls.append(("fp8", norm, s)) or "e4m3",
    )
    taken = _Module(norm=lambda x, z: calls.append("stock") or "bf16")
    other = _Module(norm=lambda x, z: calls.append("stock") or "bf16")
    built = prefill_fp8.PrefillFp8()
    built.norm[taken] = "s"
    x, z = _norm_inputs()
    with caplog.at_level("INFO"):
        assert built.gdn_norm(taken, _cuda(x), z) == "e4m3"
        assert built.gdn_norm(other, _cuda(x), z) == "bf16"
        # another layout at a step: SGLang's norm
        assert built.gdn_norm(taken, x, z) == "bf16"
    assert calls == [("fp8", taken.norm, "s"), "stock", "stock"]
    assert caplog.text.count("slot gdn_prefill_norm_fp8 runs at 6 rows") == 1


@pytest.mark.parametrize("taken", [True, False], ids=["norm-row", "no-norm-row"])
def test_the_body_passes_the_norm_row_to_the_fronts_step(monkeypatch, taken):
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    built = prefill.Qwen35Prefill.__new__(prefill.Qwen35Prefill)
    built.fp8 = prefill_fp8.PrefillFp8()
    if taken:
        built.fp8.norm["gdn"] = "s"
    built.moe = None
    built.front, built.qk_rows = True, False
    built._core = "core"
    stage = SimpleNamespace(attn_boundary=SimpleNamespace(from_pp=lambda t, fb: "h"))
    built.lm = SimpleNamespace(
        model=SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
            start_layer=0,
            layers=[stage],
        )
    )
    steps = []
    monkeypatch.setattr(
        prefill_fi, "step", lambda gdn, x, core, norm=None: steps.append(norm) or x
    )
    monkeypatch.setattr(
        built,
        "_layers",
        lambda hidden, positions, fb, gdn_step, *rest, **kw: (
            gdn_step("gdn", hidden),
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(residual_batch, "to_pp", lambda hidden, fb: ("pp", hidden))
    assert built._body("ids", "pos", "fb", None, "proxy") == ("pp", "h")
    assert steps == [built.fp8.gdn_norm if taken else None]


class _GemmaNorm:
    """Stands in for SGLang's GemmaRMSNorm (the class is patched to it)."""

    def __init__(self, calls, hidden=16, method=None):
        self.calls = calls
        self.weight = torch.zeros(hidden)
        self.variance_epsilon = 1e-6
        self._forward_method = method

    def forward_cuda(self, *args):
        raise AssertionError("not called")

    def forward(self, *args, **kwargs):
        self.calls.append(("stock_norm", len(args), tuple(sorted(kwargs))))
        return args[0], args[1] if len(args) > 1 else None


class _Column:
    """Stands in for SGLang's ColumnParallelLinear (the class is patched to it)."""

    def __init__(self, calls, linear=None):
        self.calls = calls
        linear = linear or _linear()
        self.quant_method = linear.quant_method
        self.input_scale = linear.input_scale

    def forward(self, x):
        self.calls.append(("linear", x))
        return "out", None


@pytest.fixture
def norms(monkeypatch):
    import sys
    from types import ModuleType

    from sglang.srt.layers import layernorm, linear

    from sgl_nvidia_mega.qwen3_5 import sm103

    calls = []
    monkeypatch.setattr(layernorm, "GemmaRMSNorm", _GemmaNorm)
    monkeypatch.setattr(linear, "ColumnParallelLinear", _Column)

    def dual(x, residual, weight, eps, scale):
        calls.append(("dual", scale))
        return "e4m3"

    kernel = ModuleType("sgl_nvidia_mega.qwen3_5.sm103.gemma_norm_fp8")
    kernel.gemma_fused_add_rmsnorm_fp8 = dual
    kernel.supported = lambda x, r: True
    kernel.cute_route = lambda device: True
    kernel.build = lambda device, hidden: calls.append(("build", hidden))
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    monkeypatch.setattr(sm103, "gemma_norm_fp8", kernel, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return calls


def _hand_off(norms):
    built = prefill_fp8.PrefillFp8()
    norm, linear = _GemmaNorm(norms), _Column(norms)
    built.norm_in[norm] = linear
    built.active = True
    return built, norm, linear


@pytest.mark.parametrize(
    "call",
    ["x-residual", "boundary", "keyword-residual", "keywords"],
)
def test_the_norm_hands_its_linear_the_fp8_copy_of_its_own_output(norms, call):
    built, norm, linear = _hand_off(norms)
    x, residual = torch.zeros(3, 16), torch.zeros(3, 16)
    args, kwargs = {
        "x-residual": ((x, residual), {}),
        "boundary": ((x, residual, None), {}),
        "keyword-residual": ((x,), {"residual": residual}),
        "keywords": (
            (),
            {"x": x, "residual": residual, "post_residual_addition": None},
        ),
    }[call]
    assert built.norm_forward(norm, *args, **kwargs) == (x, residual)
    assert built.linear_forward(linear, x) == ("out", None)
    assert norms == [("dual", linear.input_scale), ("linear", "e4m3")]
    # the hand-off is used once
    norms.clear()
    built.linear_forward(linear, x)
    assert norms == [("linear", x)]


def test_the_norm_reads_its_linears_input_scale_at_the_call(norms):
    built, norm, linear = _hand_off(norms)
    x, residual = torch.zeros(3, 16), torch.zeros(3, 16)
    linear.input_scale = _scale()
    built.norm_forward(norm, x, residual)
    assert norms[-1][0] == "dual" and norms[-1][1] is linear.input_scale
    linear.input_scale = _scale(2)
    built.norm_forward(norm, x, residual)
    assert norms[-1] == ("stock_norm", 2, ())
    assert built._stash is None


@pytest.mark.parametrize(
    "change",
    ["other-tensor", "view", "modified", "other-linear-first"],
)
def test_the_linear_takes_its_own_input_unless_it_is_the_norms_output(norms, change):
    built, norm, linear = _hand_off(norms)
    x, residual = torch.zeros(3, 16), torch.zeros(3, 16)
    built.norm_forward(norm, x, residual)
    given = x
    if change == "other-tensor":
        given = torch.zeros(3, 16)
    elif change == "view":
        given = x[:2]
    elif change == "modified":
        x.add_(1)
    else:
        # another linear does not take (or drop) the hand-off
        other = _Column(norms)
        built.linear_forward(other, x)
        assert norms[-1] == ("linear", x)
        built.linear_forward(linear, x)
        assert norms[-1] == ("linear", "e4m3")
        return
    built.linear_forward(linear, given)
    assert norms[-1] == ("linear", given)


@pytest.mark.parametrize(
    "case",
    [
        "outside-body",
        "no-residual",
        "x-only",
        "post-residual-addition",
        "keyword-post-residual-addition",
        "backend",
        "unknown-keyword",
        "layout",
        "inference-tensor",
    ],
)
def test_the_norm_runs_sglangs_forward_otherwise(norms, monkeypatch, case):
    import sys

    gemma_norm_fp8 = sys.modules["sgl_nvidia_mega.qwen3_5.sm103.gemma_norm_fp8"]
    built, norm, linear = _hand_off(norms)
    x, residual = torch.zeros(3, 16), torch.zeros(3, 16)
    args, kwargs = (x, residual), {}
    if case == "outside-body":
        built.active = False
    elif case == "no-residual":
        args = (x, None, None)
    elif case == "x-only":
        args = (x,)
    elif case == "post-residual-addition":
        args = (x, residual, torch.ones(3, 16))
    elif case == "keyword-post-residual-addition":
        args, kwargs = (x, residual), {"post_residual_addition": torch.ones(3, 16)}
    elif case == "backend":
        # BaseFusedOp's explicit backend selection
        kwargs = {"backend": "native"}
    elif case == "unknown-keyword":
        args, kwargs = (x,), {"residuals": residual}
    elif case == "layout":
        monkeypatch.setattr(gemma_norm_fp8, "supported", lambda x, r: False)
    else:
        with torch.inference_mode():
            x = torch.zeros(3, 16)
        args = (x, residual)
    built.norm_forward(norm, *args, **kwargs)
    assert norms == [("stock_norm", len(args), tuple(sorted(kwargs)))]
    assert built._stash is None
    built.linear_forward(linear, x)
    assert norms[-1] == ("linear", x)


class _GdnLayer:
    def __init__(self, norms, linear=None):
        self.input_layernorm = _GemmaNorm(norms)
        self.linear_attn = SimpleNamespace(in_proj_qkvz=linear or _Column(norms))


class _AttnLayer:
    def __init__(self, norms, linear=None, norm=None):
        self.input_layernorm = norm or _GemmaNorm(norms)
        self.qkv_proj = linear or _Column(norms)


def test_the_build_gives_each_layers_norm_and_reader_the_forwards(norms, caplog):
    layers = [
        _GdnLayer(norms),
        _AttnLayer(norms),
        _AttnLayer(norms, linear=_Column(norms, _linear(scale=_scale(2)))),
        _AttnLayer(norms, norm=_GemmaNorm(norms, hidden=12)),
    ]
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(_model(layers), ("prefill_norm_fp8",), {0})
    assert list(built.norm_in) == [layers[0].input_layernorm, layers[1].input_layernorm]
    assert (
        built.norm_in[layers[0].input_layernorm] is layers[0].linear_attn.in_proj_qkvz
    )
    assert built.rows() == ["prefill_norm_fp8"]
    # the kernel is built once a hidden size, at load
    assert norms == [("build", 16)]
    norm, linear = layers[1].input_layernorm, layers[1].qkv_proj
    assert norm.forward.func == built.norm_forward and norm.forward.args == (norm,)
    assert linear.forward.func == built.linear_forward
    assert "forward" not in vars(layers[2].qkv_proj)
    assert (
        "slot prefill_norm_fp8 uses stock on some layers (layer 2: the norm's reader "
        "is not a static per-tensor FP8 linear)"
    ) in caplog.text


def test_the_build_keeps_sglangs_norm_where_flashinfer_runs_its_cuda_kernel(
    norms, monkeypatch, caplog
):
    import sys

    gemma_norm_fp8 = sys.modules["sgl_nvidia_mega.qwen3_5.sm103.gemma_norm_fp8"]
    monkeypatch.setattr(gemma_norm_fp8, "cute_route", lambda device: False)
    layers = [_AttnLayer(norms)]
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(
            _model(layers), ("prefill_norm_fp8",), set()
        )
    assert not built.norm_in and norms == []
    assert "forward" not in vars(layers[0].input_layernorm)
    assert (
        "slot prefill_norm_fp8 uses stock on every layer (FlashInfer runs its CUDA "
        "norm kernel here)"
    ) in caplog.text


@pytest.mark.parametrize(
    "norm, linear, why",
    [
        pytest.param({}, {}, "", id="takes"),
        pytest.param(
            {"method": print},
            {},
            "the input norm is not SGLang's CUDA GemmaRMSNorm",
            id="native-norm",
        ),
        pytest.param(
            {"hidden": 12}, {}, "the hidden size is not a multiple of 8", id="hidden"
        ),
        pytest.param(
            {},
            {"linear": _linear(method=object())},
            "the norm's reader is not a static per-tensor FP8 linear",
            id="bf16-reader",
        ),
    ],
)
def test_a_layer_takes_the_dual_norm_only_on_sglangs_cuda_norm_and_fp8_reader(
    norms, norm, linear, why
):
    assert (
        prefill_fp8._norm_in_unsupported(
            _GemmaNorm(norms, **norm), _Column(norms, **linear)
        )
        == why
    )
    assert prefill_fp8._norm_in_unsupported(object(), _Column(norms)) == (
        "the input norm is not SGLang's CUDA GemmaRMSNorm"
    )
    assert prefill_fp8._norm_in_unsupported(_GemmaNorm(norms), object()) == (
        "the norm's reader is not a ColumnParallelLinear"
    )


def test_the_body_scope_drops_a_pending_hand_off(norms):
    built, norm, linear = _hand_off(norms)
    built.active = False
    with built.body():
        built.norm_forward(norm, torch.zeros(3, 16), torch.zeros(3, 16))
        assert built._stash is not None
    assert built._stash is None


@pytest.fixture
def dual_kernel(monkeypatch):
    import sys
    from types import ModuleType

    from sgl_nvidia_mega.qwen3_5 import sm103

    calls = []

    def dual(x, residual, weight, eps, scale):
        q = torch.empty(x.shape, dtype=torch.float8_e4m3fn)
        calls.append(("dual", x, residual, weight, eps, scale, q))
        return q

    kernel = ModuleType("sgl_nvidia_mega.qwen3_5.sm103.gemma_norm_fp8")
    kernel.gemma_fused_add_rmsnorm_fp8 = dual
    kernel.supported = lambda x, r: True
    kernel.cute_route = lambda device: True
    kernel.build = lambda device, hidden: calls.append(("build", hidden))
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    monkeypatch.setattr(sm103, "gemma_norm_fp8", kernel, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return calls


def _sglang_norm_and_reader(calls, monkeypatch, hidden=16):
    from sglang.srt.layers import layernorm
    from sglang.srt.layers import linear as sglang_linear
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.linear import ColumnParallelLinear

    monkeypatch.setattr(
        layernorm,
        "gemma_fused_add_rmsnorm",
        lambda x, residual, weight, eps: calls.append(("stock_norm", x, residual)),
        raising=False,
    )
    # TP1 with no sequence-parallel region: the forward goes to its quant method
    monkeypatch.setattr(
        sglang_linear, "get_forward", lambda: SimpleNamespace(sp_active=False)
    )
    norm = GemmaRMSNorm(hidden)
    norm._forward_method = norm.forward_cuda
    reader = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(reader)
    reader.bias, reader.skip_bias_add, reader.gather_output = None, False, False
    reader.tp_group = None
    reader.quant_method = _modelopt()
    reader.quant_method.apply = lambda layer, x, bias=None: (
        calls.append(("apply", layer, x)) or torch.zeros(x.shape[0], 8)
    )
    reader.input_scale = _scale()
    return norm, reader


def _sglang_layer(kind, norm, reader):
    from sglang.srt.layers.layer_boundary.residual.add_norm import NormQuantReadout
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.models import qwen3_5
    from sglang.test.boundary_fixtures import build_stages, parallel_of, planning

    assert not qwen3_5._linear_accepts_fp8_tuple(reader)
    cls = (
        qwen3_5.Qwen3_5LinearDecoderLayer
        if kind == "gdn"
        else qwen3_5.Qwen3_5AttentionDecoderLayer
    )
    config = SimpleNamespace(model_type="qwen3_5_moe_text")
    attn, ffn = cls.stage_facts(config, 1, attn_read=NormQuantReadout(fp8_input=None))
    with planning(parallel_of(attn_dp=1, attn_tp=1)):
        attn_boundary, _ = build_stages(
            (attn, norm, {"fusions": None}),
            (ffn, GemmaRMSNorm(norm.weight.shape[0]), {"fusions": None}),
            previous=cls.stage_facts(config, 0)[1],
        )
    if kind == "gdn":
        mixer = SimpleNamespace(
            in_proj_qkvz=reader,
            in_proj_ba=lambda x: (torch.zeros(x.shape[0], 4), None),
            _fused_in_proj_weight=None,
            alt_stream=None,
            _fused_input_proj_cpu_enabled=SimpleNamespace(value=False),
        )
        return SimpleNamespace(
            input_layernorm=norm, linear_attn=mixer, attn_boundary=attn_boundary
        )
    return SimpleNamespace(
        input_layernorm=norm, qkv_proj=reader, attn_boundary=attn_boundary
    )


def _read_input(layer, kind, previous, residual):
    from sglang.srt.layers.layer_boundary import PLAIN_ADD
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models import qwen3_5
    from sglang.test.boundary_fixtures import parallel_of, planning

    stream = ResidualStream(residual)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.zeros(previous.shape[0]),
        attn_cp_metadata=None,
        residual_stream=stream,
    )
    with planning(parallel_of(attn_dp=1, attn_tp=1)):
        x = layer.attn_boundary.prepare(stream.record(previous, PLAIN_ADD), fb)
    if kind == "gdn":
        qwen3_5.Qwen3_5GatedDeltaNet._forward_input_proj(layer.linear_attn, x)
    else:
        layer.qkv_proj(x)
    return x, fb.residual_stream


@pytest.mark.parametrize("kind", ["gdn", "attention"])
def test_the_norm_hands_off_through_sglangs_layer_boundary(
    dual_kernel, monkeypatch, caplog, kind
):
    from sglang.srt.models import qwen3_5

    calls = dual_kernel
    # a prefill outside piecewise CUDA graphs (_forward_input_proj asks)
    monkeypatch.setattr(qwen3_5, "check_cuda_graph_backend", lambda *a: False)
    norm, reader = _sglang_norm_and_reader(calls, monkeypatch)
    layer = _sglang_layer(kind, norm, reader)
    with caplog.at_level("INFO"):
        built = prefill_fp8.PrefillFp8.build(
            _model([layer]),
            ("prefill_norm_fp8",),
            {0} if kind == "gdn" else set(),
        )
    assert built.norm_in == {norm: reader} and built.rows() == ["prefill_norm_fp8"]
    assert calls == [("build", 16)] and "uses stock" not in caplog.text

    previous, residual = torch.randn(5, 16), torch.randn(5, 16)
    calls.clear()
    with caplog.at_level("INFO"), built.body():
        x, stream = _read_input(layer, kind, previous, residual)
    (dual, *rest), (apply, *applied) = calls
    assert dual == "dual" and apply == "apply"
    x_in, residual_in, weight, eps, scale, q = rest
    assert x_in is previous and residual_in is residual and x is previous
    assert weight.data_ptr() == norm.weight.data_ptr() and not weight.requires_grad
    assert eps == norm.variance_epsilon
    assert scale is reader.input_scale
    assert applied[0] is reader and applied[1] is q
    assert stream.residual is residual and built._stash is None
    assert "slot prefill_norm_fp8 runs at 5 rows" in caplog.text

    calls.clear()
    x, stream = _read_input(layer, kind, previous, residual)
    assert [c[0] for c in calls] == ["stock_norm", "apply"]
    assert calls[0][1] is previous and calls[0][2] is residual
    assert calls[1][2] is x is previous
