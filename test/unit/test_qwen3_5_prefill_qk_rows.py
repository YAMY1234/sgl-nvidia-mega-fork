# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import sys
from contextlib import contextmanager
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.qwen3_5.attention import prefill_rows

H, KVH, D, ROT, EPS = 4, 2, 16, 8, 1e-6
WIDTH = H * 2 * D + 2 * KVH * D
HIDDEN = 24
ROWS_MODULE = "sgl_nvidia_mega.qwen3_5.sm103.attention.qk_norm_rope_rows"


def _cos_sin(maxpos=64):
    inv = 1.0 / (1e4 ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT))
    f = torch.outer(torch.arange(maxpos, dtype=torch.float32), inv)
    return torch.cat([f.cos(), f.sin()], dim=-1)


def _norm_rope(src, heads, w, cos_sin, positions, axis_map):
    T = src.shape[0]
    x = src.reshape(T, heads, D).float()
    xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * (w.float() + 1.0)
    xn = xn.to(src.dtype).float()
    if axis_map is None:
        pos = positions.unsqueeze(1).expand(T, ROT // 2)
    else:
        pos = positions[axis_map.long()].T
    lanes = torch.arange(ROT // 2)
    c = cos_sin[pos, lanes].unsqueeze(1)
    s = cos_sin[pos, ROT // 2 + lanes].unsqueeze(1)
    x1, x2 = xn[..., : ROT // 2], xn[..., ROT // 2 : ROT]
    out = xn.clone()
    out[..., : ROT // 2] = x1 * c - x2 * s
    out[..., ROT // 2 : ROT] = x2 * c + x1 * s
    return out.to(src.dtype).reshape(T, heads * D)


def _qk(q_gate, k, qw, kw, cos_sin, positions, eps, nq, nkv, hd, rot, axis_map):
    assert (eps, nq, nkv, hd, rot) == (EPS, H, KVH, D, ROT)
    T = q_gate.shape[0]
    q = q_gate.reshape(T, H, 2 * D)[..., :D].reshape(T, -1)
    return (
        _norm_rope(q, H, qw, cos_sin, positions, axis_map),
        _norm_rope(k, KVH, kw, cos_sin, positions, axis_map),
    )


class _Kernels:
    def __init__(self, supported=True):
        self.calls = []
        self.gates = []
        self.supported = supported
        self.rows = ModuleType(ROWS_MODULE)
        self.rows.qk_norm_rope_rows_supported = lambda q_gate, k, hd, rot: (
            self.calls.append(("supported", hd, rot)) or self.supported
        )
        self.rows.qk_gemma_rmsnorm_rope_rows = self._rows
        self.rows.shape_unsupported = lambda hd, rot: ""
        self.rows.warm = lambda *args: self.calls.append(("warm", args[4:9]))

    def _record(self, name, args, kwargs):
        q_gate, k = args[:2]
        self.calls.append(
            (
                name,
                (q_gate.storage_offset(), q_gate.shape, q_gate.stride()),
                (k.storage_offset(), k.shape, k.stride()),
                tuple(
                    (a.dtype, a.shape, a.tolist()) if torch.is_tensor(a) else a
                    for a in args[2:]
                ),
                kwargs,
            )
        )

    def stock(self, *args, has_gate=True, mrope_axis_map=None):
        self._record("stock", args, dict(has_gate=has_gate, axis=mrope_axis_map))
        q, k = _qk(*args, mrope_axis_map)
        T = args[0].shape[0]
        gate = args[0].reshape(T, H, 2 * D)[..., D:].clone()  # SGLang's gate copy
        return q, k, gate

    def _rows(self, *args, has_gate=True, mrope_axis_map=None):
        self._record("rows", args, dict(has_gate=has_gate, axis=mrope_axis_map))
        q, k = _qk(*args, mrope_axis_map)
        T = args[0].shape[0]
        return q, k, args[0].view(T, H, 2 * D)[:, :, D:]

    def sigmoid_mul(self, attn_output, gate, inplace=False, quant=False):
        assert inplace and not quant
        self.gates.append(gate)
        g = gate.reshape(attn_output.shape[0], -1).float()
        attn_output.copy_(
            (attn_output.float() * torch.sigmoid(g)).to(attn_output.dtype)
        )
        return attn_output


@pytest.fixture
def kernels(monkeypatch):
    from sglang.kernels.ops.elementwise import elementwise
    from sglang.srt.models import qwen3_5

    stand_ins = _Kernels()
    monkeypatch.setitem(sys.modules, ROWS_MODULE, stand_ins.rows)
    package = importlib.import_module(ROWS_MODULE.rpartition(".")[0])
    monkeypatch.setattr(package, "qk_norm_rope_rows", stand_ins.rows, raising=False)
    # SGLang's CUDA arm of the layer (the CPU build of the module has no CUDA kernels)
    monkeypatch.setattr(qwen3_5, "_is_cuda", True)
    monkeypatch.setattr(
        qwen3_5, "fused_qk_gemma_rmsnorm_rope_gate", stand_ins.stock, raising=False
    )
    monkeypatch.setattr(qwen3_5, "fused_sigmoid_mul", stand_ins.sigmoid_mul)
    monkeypatch.setattr(elementwise, "fused_sigmoid_mul", stand_ins.sigmoid_mul)
    return stand_ins


class _Boundary:
    """A layer boundary stand-in: each stage changes the rows and logs its call."""

    def __init__(self, log, name, scale):
        self.log, self.name, self.scale = log, name, scale

    def prepare(self, hidden, fb, capture_gathered=None):
        self.log.append((self.name, "prepare", capture_gathered))
        return hidden * self.scale

    def finish(self, hidden, fb):
        self.log.append((self.name, "finish"))
        return hidden + self.scale

    def exit(self, fb):
        boundary = self

        class _Exit:
            defer_moe_finalize = "defer"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def finish(self, out):
                boundary.log.append((boundary.name, "exit.finish"))
                return out - 1

        return _Exit()


def _moe_block(log):
    from sglang.srt.models.qwen3_5 import Qwen2MoeSparseMoeBlock

    class _Moe(Qwen2MoeSparseMoeBlock):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, hidden, forward_batch=None, defer_finalize=False):
            log.append(("moe", defer_finalize))
            return hidden * 3

    return _Moe()


def _layer(seed=0, moe=True, **changes):
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

    g = torch.Generator().manual_seed(seed)
    w_qkv = (torch.randn(HIDDEN, WIDTH, generator=g) * 0.3).to(torch.bfloat16)
    w_o = (torch.randn(H * D, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
    log = []
    # SGLang's loader makes a BF16 model's norm weights BF16
    with set_default_torch_dtype(torch.bfloat16):
        q_norm, k_norm = GemmaRMSNorm(D, eps=EPS), GemmaRMSNorm(D, eps=EPS)
    q_norm.weight.data.copy_(torch.randn(D, generator=g) * 0.3)
    k_norm.weight.data.copy_(torch.randn(D, generator=g) * 0.3)

    def attn(q, k, v, fb):
        log.append(("attn", q.shape, k.shape, v.shape))
        T = q.shape[0]
        kv = (k.float() + v.float()).reshape(T, KVH, D).repeat_interleave(H // KVH, 1)
        return (q.float().reshape(T, H, D) * 0.5 + kv).reshape(T, -1).to(q.dtype)

    fields = dict(
        attn_output_gate=True,
        q_size=H * D,
        kv_size=KVH * D,
        num_heads=H,
        num_kv_heads=KVH,
        head_dim=D,
        q_norm=q_norm,
        k_norm=k_norm,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=_cos_sin(),
            rotary_dim=ROT,
            axis_map=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        ),
        qkv_proj=lambda x: (x @ w_qkv, None),
        o_proj=lambda x: (x @ w_o, None),
        attn=attn,
        attn_boundary=_Boundary(log, "attn_boundary", 2),
        ffn_boundary=_Boundary(log, "ffn_boundary", 3),
        mlp=_moe_block(log) if moe else (lambda x: log.append(("mlp",)) or x * 5),
        log=log,
    )
    layer = SimpleNamespace(**(fields | changes))
    for name in ("self_attention", "_prepare_qkv_gate", "forward_prepare_cuda_fused"):
        setattr(
            layer, name, MethodType(getattr(Qwen3_5AttentionDecoderLayer, name), layer)
        )
    return layer


def _batch(idle=False):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: idle))


def _inputs(T, mrope, seed=1):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(T, HIDDEN, generator=g)).to(torch.bfloat16)
    pos = torch.randint(0, 40, (T,), generator=g)
    positions = torch.stack([pos, pos + 1, pos + 2]).contiguous() if mrope else pos
    return hidden, positions


def _sglang(layer, hidden, positions, fb, capture=None):
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

    return Qwen3_5AttentionDecoderLayer.forward(
        layer,
        positions=positions,
        hidden_states=hidden,
        forward_batch=fb,
        captured_last_layer_outputs=capture,
    )


def _bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x


@pytest.mark.parametrize("mrope", [True, False], ids=["mrope", "1d-positions"])
@pytest.mark.parametrize("moe", [True, False], ids=["moe", "dense-mlp"])
def test_the_layer_is_sglangs_with_the_row_tiles_and_the_gate_read_in_place(
    kernels, mrope, moe
):
    T = 7
    hidden, positions = _inputs(T, mrope)
    layer = _layer(moe=moe)
    want = _sglang(layer, hidden, positions, _batch(), capture="aux")
    want_log = list(layer.log)
    (stock,) = [c for c in kernels.calls if c[0] == "stock"]
    stock_gate = kernels.gates.pop()
    kernels.calls.clear()
    layer.log.clear()
    got = prefill_rows.forward(layer, positions, hidden, _batch(), "aux")
    assert torch.equal(_bits(got), _bits(want))
    # the same boundary, attention and MLP calls in SGLang's order
    assert layer.log == want_log
    # the row tiles take SGLang's kernel's arguments
    (supported, rows) = kernels.calls
    assert supported == ("supported", D, ROT)
    assert rows[0] == "rows" and rows[1:4] == stock[1:4]
    assert rows[4]["has_gate"] is stock[4]["has_gate"] is True
    axis = layer.rotary_emb.axis_map if mrope else None
    assert rows[4]["axis"] is stock[4]["axis"] is axis
    # SGLang's output gate reads its copy; the slot's reads the projection in place
    (gate,) = kernels.gates
    assert stock_gate.dim() == 2 and stock_gate.is_contiguous()
    assert gate.shape == (T, H, D) and gate.stride() == (WIDTH, 2 * D, 1)
    assert gate.storage_offset() == rows[1][0] + D
    assert torch.equal(_bits(gate.reshape(T, -1)), _bits(stock_gate))


def test_where_the_tiles_do_not_take_the_step_the_layer_runs_sglangs_kernel(kernels):
    kernels.supported = False
    hidden, positions = _inputs(5, True)
    want = _sglang(_layer(), hidden, positions, _batch())
    kernels.calls.clear()
    got = prefill_rows.forward(_layer(), positions, hidden, _batch())
    assert torch.equal(_bits(got), _bits(want))
    assert [c[0] for c in kernels.calls] == ["supported", "stock"]
    assert all(g.dim() == 2 for g in kernels.gates)


@pytest.mark.parametrize("idle, T", [(True, 3), (False, 0)], ids=["idle", "no-rows"])
def test_a_step_without_rows_runs_only_the_boundaries_and_the_mlp(kernels, idle, T):
    hidden, positions = _inputs(T, True)
    want_layer, got_layer = _layer(), _layer()
    want = _sglang(want_layer, hidden, positions, _batch(idle))
    got = prefill_rows.forward(got_layer, positions, hidden, _batch(idle))
    assert torch.equal(_bits(got), _bits(want))
    assert got_layer.log == want_layer.log
    assert not any(entry[0] == "attn" for entry in got_layer.log)
    assert kernels.calls == []


def _rows_check(**changes):
    from sgl_nvidia_mega.qwen3_5.sm103.attention import qk_norm_rope_rows as rows

    return prefill_rows.layer_unsupported(_layer(**changes), rows)


@pytest.fixture
def real_rows(monkeypatch):
    launches = []

    class _Jit:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))

    triton = ModuleType("triton")
    language = ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    triton.jit = _Jit
    triton.cdiv = lambda a, b: -(-a // b)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.delitem(sys.modules, ROWS_MODULE, raising=False)
    package = importlib.import_module("sgl_nvidia_mega.qwen3_5.sm103.attention")
    module = importlib.import_module(ROWS_MODULE)
    module.launches = launches
    yield module
    sys.modules.pop(ROWS_MODULE, None)
    if getattr(package, "qk_norm_rope_rows", None) is module:
        delattr(package, "qk_norm_rope_rows")


@pytest.mark.parametrize(
    "changes, why",
    [
        pytest.param({}, "", id="qwen3_5"),
        pytest.param({"attn_output_gate": False}, "no output gate", id="no-gate"),
        pytest.param(
            {"q_norm": torch.nn.LayerNorm(D)},
            "Q / K norm is not GemmaRMSNorm",
            id="norm-type",
        ),
        pytest.param({"eps": 1e-5}, "Q and K norm epsilons differ", id="eps"),
        pytest.param(
            {"head_dim": 24}, "Q / K norm weight is not one head wide", id="norm-width"
        ),
        pytest.param(
            {"dtype": torch.float16}, "the model is torch.float16, not BF16", id="fp16"
        ),
        pytest.param(
            {"rotary_emb": SimpleNamespace(rotary_dim=ROT)},
            "no cos / sin cache",
            id="no-cache",
        ),
        pytest.param(
            {"rot": 12},
            "rotary_dim 12 is not twice a power of two <= head_dim",
            id="rotary-dim",
        ),
    ],
)
def test_the_layer_check_names_what_the_row_tiles_do_not_take(real_rows, changes, why):
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    if "eps" in changes:
        k_norm = GemmaRMSNorm(D, eps=changes.pop("eps"))
        changes["k_norm"] = k_norm
    if "rot" in changes:
        changes["rotary_emb"] = SimpleNamespace(
            cos_sin_cache=_cos_sin(), rotary_dim=changes.pop("rot")
        )
    if "dtype" in changes:
        dtype = changes.pop("dtype")
        for name in ("q_norm", "k_norm"):
            changes[name] = GemmaRMSNorm(D, eps=EPS)
            changes[name].weight.data = changes[name].weight.data.to(dtype)
    assert _rows_check(**changes) == why


@pytest.mark.parametrize(
    "head_dim, rotary_dim, why",
    [
        (256, 64, ""),
        (128, 128, ""),
        (16, 2, ""),
        (8, 4, "head_dim 8 is not a power of two >= 16"),
        (192, 64, "head_dim 192 is not a power of two >= 16"),
        (256, 0, "rotary_dim 0 is not twice a power of two <= head_dim"),
        (256, 96, "rotary_dim 96 is not twice a power of two <= head_dim"),
        (256, 512, "rotary_dim 512 is not twice a power of two <= head_dim"),
        (256, 63, "rotary_dim 63 is not twice a power of two <= head_dim"),
    ],
)
def test_the_kernel_takes_power_of_two_heads_and_rotary_halves(
    real_rows, head_dim, rotary_dim, why
):
    assert real_rows.shape_unsupported(head_dim, rotary_dim) == why


class _Cuda:
    """A tensor's fields as the supported check reads them, on a CUDA device."""

    def __init__(self, t, dtype=None):
        self.t, self.dtype, self.is_cuda = t, dtype or t.dtype, True

    def dim(self):
        return self.t.dim()

    def stride(self, i):
        return self.t.stride(i)


def test_the_kernel_takes_bf16_or_fp16_unit_stride_rows_on_cuda(real_rows):
    qkv = torch.zeros(5, WIDTH, dtype=torch.bfloat16)
    q_gate, k, _ = qkv.split([H * 2 * D, KVH * D, KVH * D], dim=-1)
    supported = real_rows.qk_norm_rope_rows_supported
    assert supported(_Cuda(q_gate), _Cuda(k), D, ROT)
    assert not supported(q_gate, k, D, ROT)  # not on CUDA
    assert not supported(_Cuda(q_gate.float()), _Cuda(k.float()), D, ROT)
    assert not supported(_Cuda(q_gate), _Cuda(k, torch.float16), D, ROT)
    assert not supported(_Cuda(q_gate[:, ::2]), _Cuda(k), D, ROT)
    assert not supported(_Cuda(q_gate), _Cuda(k), 24, ROT)


@pytest.mark.parametrize("mrope", [True, False], ids=["mrope", "1d-positions"])
@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no-pdl"])
def test_the_wrapper_launches_q_and_k_tiles_and_returns_the_gate_in_place(
    real_rows, monkeypatch, mrope, pdl
):
    monkeypatch.setattr(real_rows, "_ENABLE_PDL", pdl)
    T = 9
    qkv = torch.randn(T, WIDTH).to(torch.bfloat16)
    q_gate, k, _ = qkv.split([H * 2 * D, KVH * D, KVH * D], dim=-1)
    qw, kw = torch.zeros(D), torch.zeros(D)
    cos_sin = _cos_sin()
    pos = torch.arange(T)
    positions = torch.stack([pos] * 3).contiguous() if mrope else pos
    axis = torch.zeros(ROT // 2, dtype=torch.int32) if mrope else None
    q, kk, gate = real_rows.qk_gemma_rmsnorm_rope_rows(
        q_gate, k, qw, kw, cos_sin, positions, EPS, H, KVH, D, ROT, True, axis
    )
    assert q.shape == (T, H * D) and kk.shape == (T, KVH * D)
    assert gate.shape == (T, H, D) and gate.stride() == (WIDTH, 2 * D, 1)
    assert gate.data_ptr() == qkv.data_ptr() + D * qkv.element_size()
    assert torch.equal(gate, qkv[:, : H * 2 * D].reshape(T, H, 2 * D)[:, :, D:])
    ((grid, args, kwargs),) = real_rows.launches
    q_progs = -(-T * H // 8)
    assert grid == (q_progs + -(-T * KVH // 8),)
    assert args[0] is q_gate and args[1] is k and args[2] is q and args[3] is kk
    assert args[6] is cos_sin and args[7] is positions
    assert args[8] is (axis if mrope else positions)
    assert args[9:] == (
        T,
        q_progs,
        WIDTH,
        WIDTH,
        H * D,
        KVH * D,
        ROT,
        T if mrope else 0,
    )
    want = dict(
        NUM_Q_HEADS=H,
        NUM_KV_HEADS=KVH,
        Q_HEAD_STRIDE=2 * D,
        HEAD_DIM=D,
        ROT=ROT,
        HALF=ROT // 2,
        ROWS=8,
        EPS=EPS,
        WEIGHT_SHIFT=1.0,
        FP16=False,
        MROPE=mrope,
        ENABLE_PDL=pdl,
        num_warps=2,
    )
    if pdl:
        want["launch_pdl"] = True
    assert kwargs == want


def test_the_wrapper_launches_nothing_without_tokens(real_rows):
    qkv = torch.zeros(0, WIDTH, dtype=torch.bfloat16)
    q_gate, k, _ = qkv.split([H * 2 * D, KVH * D, KVH * D], dim=-1)
    q, kk, gate = real_rows.qk_gemma_rmsnorm_rope_rows(
        q_gate,
        k,
        torch.zeros(D),
        torch.zeros(D),
        _cos_sin(),
        torch.zeros(0),
        EPS,
        H,
        KVH,
        D,
        ROT,
    )
    assert q.shape == (0, H * D) and gate.shape == (0, H, D)
    assert real_rows.launches == []


def test_the_warm_up_launches_each_token_count_and_position_form(real_rows):
    real_rows.warm(
        torch.zeros(D),
        torch.zeros(D),
        _cos_sin(),
        EPS,
        H,
        KVH,
        D,
        ROT,
        torch.zeros(ROT // 2, dtype=torch.int32),
    )
    forms = [(args[9], kwargs["MROPE"]) for _, args, kwargs in real_rows.launches]
    assert forms == [
        (tokens, mrope) for tokens in real_rows.WARM_TOKENS for mrope in (False, True)
    ]
    assert {args[0].dtype for _, args, _ in real_rows.launches} == {torch.bfloat16}
    real_rows.launches.clear()
    real_rows.warm(torch.zeros(D), torch.zeros(D), _cos_sin(), EPS, H, KVH, D, ROT)
    assert [kwargs["MROPE"] for _, _, kwargs in real_rows.launches] == [False] * len(
        real_rows.WARM_TOKENS
    )


class _AttnLayer:
    def __init__(self, **changes):
        self.__dict__.update(_layer(**changes).__dict__)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(("sglang", sorted(kwargs)))
        return kwargs["hidden_states"]


class _GdnLayer:
    def __init__(self):
        attn = SimpleNamespace(q_dim=4, k_dim=4, v_dim=4)
        self.linear_attn = SimpleNamespace(attn=attn, attn_tp_size=1)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(("sglang", sorted(kwargs)))
        return kwargs["hidden_states"] + 1


def _prefill(
    monkeypatch, layers, kernels, offered=True, qk_rows=True, front=False, fp8_rows=()
):
    from sgl_nvidia_mega.qwen3_5 import spec
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    monkeypatch.setattr(
        "sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer", _GdnLayer
    )
    monkeypatch.setattr(
        "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer", _AttnLayer
    )
    monkeypatch.setattr(prefill.parity, "ENABLED", False)
    monkeypatch.setattr(spec, "prefill_body_offered", lambda: offered)
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=len(layers), layers=layers)
    )
    return prefill.Qwen35Prefill(
        model,
        SimpleNamespace(tokens_per_request=7, binding=None),
        False,
        label="TP1/PP2",
        split_conv=False,
        front=front,
        qk_rows=qk_rows,
        fp8_rows=fp8_rows,
    )


def test_the_build_checks_each_attention_layer_and_compiles_once_a_shape(
    monkeypatch, kernels
):
    layers = [_GdnLayer(), _AttnLayer(), _GdnLayer(), _AttnLayer()]
    built = _prefill(monkeypatch, layers, kernels)
    assert built.qk_rows and not built.front
    assert (
        built.summary() == "prefill TP1/PP2; attn_prefill_qk_rows built (prefill body)"
    )
    # one warm-up for the two layers' one shape
    assert kernels.calls == [("warm", (H, KVH, D, ROT, layers[1].rotary_emb.axis_map))]


@pytest.mark.parametrize(
    "layer, why",
    [
        pytest.param(
            {"attn_output_gate": False}, "layer 1: no output gate", id="no-gate"
        ),
        pytest.param(
            {"offered": False}, "the spec offers no prefill body here", id="no-body"
        ),
        pytest.param(
            {"other": True}, "layer 1: not a full-attention layer", id="other"
        ),
    ],
)
def test_the_build_keeps_sglangs_layers_where_the_row_tiles_cannot_run(
    monkeypatch, caplog, kernels, layer, why
):
    offered = layer.pop("offered", True)
    second = SimpleNamespace() if layer.pop("other", False) else _AttnLayer(**layer)
    with caplog.at_level("INFO"):
        built = _prefill(monkeypatch, [_GdnLayer(), second], kernels, offered)
    assert not built.qk_rows and kernels.calls == []
    assert built.summary() == "prefill TP1/PP2; SGLang's forward"
    assert f"slot attn_prefill_qk_rows uses stock ({why})" in caplog.text
    fb = SimpleNamespace(
        mm_inputs=None,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    assert built.plan_body(torch.zeros(4), fb) is None


def test_the_body_plans_and_logs_the_row_tiles_without_the_front(
    monkeypatch, caplog, kernels
):
    built = _prefill(monkeypatch, [_GdnLayer(), _AttnLayer()], kernels)
    fb = SimpleNamespace(
        input_ids=torch.zeros(5),
        mm_inputs=None,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    assert built.plan_body(torch.zeros(4), fb) == Step(mode="prefill", rows=4)
    assert built.plan(torch.zeros(4), fb) is None
    monkeypatch.setattr(built, "_body", lambda *args: "hidden", raising=False)
    step = Step(mode="prefill", rows=4, view="view")
    original = lambda *args, **kw: pytest.fail("ran the stock body")
    with caplog.at_level("INFO"):
        for _ in range(2):
            assert built.forward_body(step, original, "ids", "pos", fb) == "hidden"
    assert caplog.text.count("slot attn_prefill_qk_rows runs at 4 rows") == 1
    assert "slot gdn_prefill_front runs" not in caplog.text


def test_the_body_runs_attention_layers_through_the_row_tiles(monkeypatch, kernels):
    from sglang.srt.eplb import expert_distribution

    layers = [_GdnLayer(), _AttnLayer(), _GdnLayer()]
    built = _prefill(monkeypatch, layers, kernels)
    ran = []
    monkeypatch.setattr(
        prefill_rows,
        "forward",
        lambda layer, pos, hidden, fb, capture, gate=None: (
            ran.append((layer, pos, capture, gate)) or hidden * 2
        ),
    )
    monkeypatch.setattr(
        expert_distribution,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda i: _Nothing()),
    )
    hidden, aux = built._layers(
        torch.ones(2), "pos", "fb", None, built._attention_layer
    )
    # GDN layers keep SGLang's layer forward without the front
    assert [len(layer.calls) for layer in (layers[0], layers[2])] == [1, 1]
    assert ran == [(layers[1], "pos", None, None)] and layers[1].calls == []
    assert torch.equal(hidden, (torch.ones(2) + 1) * 2 + 1)
    # without the slot the attention layer is SGLang's
    built._layers(torch.ones(2), "pos", "fb", None)
    assert layers[1].calls == [
        (
            "sglang",
            [
                "captured_last_layer_outputs",
                "forward_batch",
                "hidden_states",
                "positions",
            ],
        )
    ]


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def _tp1():
    """SGLang's layer planning and parallel state on a TP1 rank."""
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.boundary_fixtures import parallel_of, planning

    parallel = parallel_of(attn_dp=1, attn_tp=1)
    with planning(parallel), get_parallel().override(**vars(parallel)):
        yield


def _tp1_stages(n_layers):
    from sglang.srt.layers.layer_boundary.residual.add_norm import NormQuantReadout
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.models.qwen3_5 import _qwen3_5_stage_facts
    from sglang.test.boundary_fixtures import build_stages

    g = torch.Generator().manual_seed(7)
    config = SimpleNamespace(model_type="qwen3_5_moe_text")
    stages = []
    for i in range(n_layers):
        declared = _qwen3_5_stage_facts(
            config, i, attn_read=NormQuantReadout(fp8_input=None)
        )
        for declaration in declared:
            norm = GemmaRMSNorm(HIDDEN, eps=EPS)
            norm.weight.data = torch.randn(HIDDEN, generator=g) * 0.2
            norm.refresh_gemma_weight()
            stages.append((declaration, norm, {"fusions": None}))
    with _tp1():
        bound = build_stages(*stages, terminal=True)
    return [bound[i : i + 2] for i in range(0, len(bound), 2)]


def _mix(x):
    """The GDN layers' mixer, as SGLang's layer and as the front run it."""
    return (x.float() * 0.5 + 0.25).to(x.dtype)


class _Mixer:
    def __init__(self, log):
        self.attn = SimpleNamespace(q_dim=4, k_dim=4, v_dim=4)
        self.attn_tp_size = 1
        self.log = log

    def __call__(self, hidden, fb):
        self.log.append(("sglang mixer", self))
        return _mix(hidden)


class _SglangAttnLayer(_AttnLayer):
    def __init__(self, forward, stages, seed, log, **changes):
        super().__init__(
            seed=seed, attn_boundary=stages[0], ffn_boundary=stages[1], **changes
        )
        self.mlp = _moe_block(log)
        self.sglang_forward = MethodType(forward, self)

    def __call__(self, **kwargs):
        return self.sglang_forward(**kwargs)


class _SglangGdnLayer(_GdnLayer):
    def __init__(self, forward, stages, log):
        super().__init__()
        self.attn_boundary, self.ffn_boundary = stages
        self.linear_attn = _Mixer(log)
        self.mlp = _moe_block(log)
        self.sglang_forward = MethodType(forward, self)

    def __call__(self, **kwargs):
        return self.sglang_forward(**kwargs)


@pytest.mark.parametrize(
    "front, qk_rows",
    [(True, True), (True, False), (False, True)],
    ids=["front-and-rows", "front-only", "rows-only"],
)
@pytest.mark.parametrize("mrope", [True, False], ids=["mrope", "1d-positions"])
def test_the_body_runs_each_layer_through_its_built_slot_on_sglangs_tp1_stages(
    monkeypatch, caplog, kernels, front, qk_rows, mrope
):
    from sglang.srt.eplb import expert_distribution
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.models import qwen3_5

    from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    real = qwen3_5.Qwen3_5AttentionDecoderLayer, qwen3_5.Qwen3_5LinearDecoderLayer
    recorder = SimpleNamespace(with_current_layer=lambda i: _Nothing())
    for module in (expert_distribution, qwen3_5):
        monkeypatch.setattr(
            module, "get_global_expert_distribution_recorder", lambda: recorder
        )
    log = []
    kinds = ["gdn", "attn", "gdn", "attn"]
    layers = [
        _SglangAttnLayer(real[0].forward, stages, i, log)
        if kind == "attn"
        else _SglangGdnLayer(real[1].forward, stages, log)
        for i, (kind, stages) in enumerate(zip(kinds, _tp1_stages(len(kinds))))
    ]
    core = object()

    def build_front(self, model):
        self._core = core
        return True

    monkeypatch.setattr(prefill.Qwen35Prefill, "_build_front", build_front)
    monkeypatch.setattr(
        prefill_fi,
        "step",
        lambda gdn, x, run_core, norm=None: (
            log.append(("front", gdn, run_core, norm)) or _mix(x)
        ),
    )
    built = _prefill(monkeypatch, layers, kernels, qk_rows=qk_rows, front=front)
    assert (built.front, built.qk_rows) == (front, qk_rows)
    # the row tiles' build compiled the layers' one shape at load
    assert [c[0] for c in kernels.calls] == ["warm"] * qk_rows
    kernels.calls.clear()
    model = built.lm.model
    g = torch.Generator().manual_seed(3)
    table = (torch.randn(40, HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
    final_norm = GemmaRMSNorm(HIDDEN, eps=EPS)
    final_norm.weight.data = torch.randn(HIDDEN, generator=g) * 0.2
    final_norm.refresh_gemma_weight()
    model.__dict__.update(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        embed_tokens=lambda ids: table[ids],
        norm=final_norm,
        flashinfer_mnnvl_cutedsl_fusion=None,
        hidden_size=HIDDEN,
    )
    T = 9
    ids = torch.randint(0, 40, (T,), generator=g)
    _, positions = _inputs(T, mrope)

    def batch():
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=ids,
            mm_inputs=None,
            residual_stream=None,
            attn_cp_metadata=None,
        )

    with _tp1():
        want = qwen3_5.Qwen3_5ForCausalLM.forward(model, ids, positions, batch())
    sglang_log, sglang_calls = list(log), list(kernels.calls)
    log.clear()
    kernels.calls.clear()
    kernels.gates.clear()
    fb = batch()
    step = built.plan_body(ids, fb)
    assert step == Step(mode="prefill", rows=T)
    original = lambda *args, **kwargs: pytest.fail("ran SGLang's model forward")
    with caplog.at_level("INFO"), _tp1():
        got = built.forward_body(step, original, ids, positions, fb)
    assert torch.equal(_bits(got), _bits(want))
    # SGLang's forward ran its own kernel and mixer on every layer
    assert [c[0] for c in sglang_calls] == ["stock", "stock"]
    sglang_mixers = [entry[0] for entry in sglang_log if "mixer" in entry[0]]
    assert sglang_mixers == ["sglang mixer"] * 2
    # the body: the row tiles on each full-attention layer only with the slot built
    attn_kernels = [c[0] for c in kernels.calls]
    if qk_rows:
        assert attn_kernels == ["supported", "rows"] * 2
        assert all(gate.dim() == 3 for gate in kernels.gates)
    else:
        assert attn_kernels == ["stock", "stock"]
        assert all(gate.dim() == 2 for gate in kernels.gates)
    # the front on each GDN layer only with the slot built, else SGLang's mixer
    mixers = [entry for entry in log if entry[0] in ("front", "sglang mixer")]
    gdn = [layers[0].linear_attn, layers[2].linear_attn]
    if front:
        assert mixers == [("front", mixer, core, None) for mixer in gdn]
    else:
        assert mixers == [("sglang mixer", mixer) for mixer in gdn]
    for name, on in (("gdn_prefill_front", front), ("attn_prefill_qk_rows", qk_rows)):
        assert (f"slot {name} runs at {T} rows" in caplog.text) is on
    assert "uses stock" not in caplog.text and "use stock" not in caplog.text


def _sglang_stage(monkeypatch, built):
    from sglang.srt.eplb import expert_distribution
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.models import qwen3_5

    recorder = SimpleNamespace(with_current_layer=lambda i: _Nothing())
    for module in (expert_distribution, qwen3_5):
        monkeypatch.setattr(
            module, "get_global_expert_distribution_recorder", lambda: recorder
        )
    g = torch.Generator().manual_seed(3)
    table = (torch.randn(40, HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
    final_norm = GemmaRMSNorm(HIDDEN, eps=EPS)
    final_norm.weight.data = torch.randn(HIDDEN, generator=g) * 0.2
    final_norm.refresh_gemma_weight()
    built.lm.model.__dict__.update(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        embed_tokens=lambda ids: table[ids],
        norm=final_norm,
        flashinfer_mnnvl_cutedsl_fusion=None,
        hidden_size=HIDDEN,
    )
    T = 9
    ids = torch.randint(0, 40, (T,), generator=g)
    _, positions = _inputs(T, True)

    def batch():
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=ids,
            mm_inputs=None,
            residual_stream=None,
            attn_cp_metadata=None,
        )

    return ids, positions, batch


def _stage_layers(log, **changes):
    from sglang.srt.models import qwen3_5

    real = qwen3_5.Qwen3_5AttentionDecoderLayer, qwen3_5.Qwen3_5LinearDecoderLayer
    kinds = ["gdn", "attn", "gdn", "attn"]
    return [
        _SglangAttnLayer(real[0].forward, stages, i, log, **changes)
        if kind == "attn"
        else _SglangGdnLayer(real[1].forward, stages, log)
        for i, (kind, stages) in enumerate(zip(kinds, _tp1_stages(len(kinds))))
    ]


def test_the_body_hands_each_captured_layers_output_to_the_aux_states(
    monkeypatch, kernels
):
    from sglang.srt.models import qwen3_5

    log = []
    layers = _stage_layers(log)
    layers[2]._is_layer_to_capture = True
    layers[3]._is_layer_to_capture = True
    built = _prefill(monkeypatch, layers, kernels)
    assert built.qk_rows and not built.front
    ids, positions, batch = _sglang_stage(monkeypatch, built)
    with _tp1():
        want = qwen3_5.Qwen3_5ForCausalLM.forward(
            built.lm.model, ids, positions, batch()
        )
    fb = batch()
    step = built.plan_body(ids, fb)
    with _tp1():
        got = built.forward_body(step, None, ids, positions, fb)
    assert isinstance(want, tuple) and isinstance(got, tuple)
    assert torch.equal(_bits(got[0]), _bits(want[0]))
    assert len(got[1]) == len(want[1]) == 2
    for a, b in zip(got[1], want[1]):
        assert torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize(
    "front, qk_rows",
    [(True, False), (False, True), (True, True)],
    ids=["front-only", "rows-only", "front-and-rows"],
)
def test_the_body_compares_numerically_only_with_the_row_tiles(
    monkeypatch, kernels, front, qk_rows
):
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    monkeypatch.setattr(prefill.Qwen35Prefill, "_build_front", lambda self, m: True)
    built = _prefill(
        monkeypatch, [_GdnLayer(), _AttnLayer()], kernels, qk_rows=qk_rows, front=front
    )
    seen = []
    built.parity = SimpleNamespace(
        run=lambda rows, view, stock, body, bitwise=True: seen.append(bitwise) or "x"
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    step = Step(mode="prefill", rows=4, view="view")
    assert built.forward_body(step, None, "ids", "pos", SimpleNamespace()) == "x"
    assert seen == [not qk_rows]


class _CudaScale(torch.Tensor):
    """A CPU tensor that reads as a CUDA one (the static scale's device check)."""

    @property
    def is_cuda(self):
        return True


class _OnCuda:
    """A CPU tensor as the FP8 gate's layout check reads it, on a CUDA device."""

    is_cuda = True

    def __init__(self, t):
        self.t = t

    def __getattr__(self, name):
        return getattr(self.t, name)


def _static_fp8(y, s):
    inv = 1.0 / s.float()
    return (
        (y.to(torch.bfloat16).float() * inv)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )


class _Fp8OProj:
    def __init__(self, w, log):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp8LinearMethod,
        )

        method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = False
        method.use_sm120_fp8 = False
        self.quant_method = method
        self.input_scale = torch.full((1,), 0.25).as_subclass(_CudaScale)
        self.w, self.log = w, log

    def __call__(self, x):
        s = self.input_scale.as_subclass(torch.Tensor)
        if x.dtype == torch.float8_e4m3fn:
            self.log.append(("o_proj", "e4m3"))
        else:
            self.log.append(("o_proj", "quantizes"))
            x = _static_fp8(x, s)
        return (x.float() * s).to(torch.bfloat16) @ self.w, None


def test_the_row_tiles_and_the_fp8_gate_run_in_one_layer_on_sglangs_tp1_stages(
    monkeypatch, caplog, kernels
):
    from sglang.srt.models import qwen3_5

    from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi
    from sgl_nvidia_mega.qwen3_5.runtime import prefill
    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    log, o_log, fp8_gates = [], [], []
    g = torch.Generator().manual_seed(11)
    w_o = (torch.randn(H * D, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
    layers = _stage_layers(log, o_proj=_Fp8OProj(w_o, o_log))
    supported = gate_quant.supported
    monkeypatch.setattr(
        gate_quant, "supported", lambda out, gate: supported(_OnCuda(out), gate)
    )

    def fp8_gate(attn_output, gate, scale):
        fp8_gates.append(gate)
        g = gate.reshape(attn_output.shape[0], -1).float()
        product = (attn_output.float() * torch.sigmoid(g)).to(attn_output.dtype)
        return _static_fp8(product, scale.as_subclass(torch.Tensor))

    monkeypatch.setattr(gate_quant, "fused_sigmoid_mul_fp8", fp8_gate)

    def build_front(self, model):
        self._core = object()
        return True

    monkeypatch.setattr(prefill.Qwen35Prefill, "_build_front", build_front)
    monkeypatch.setattr(prefill_fi, "step", lambda gdn, x, run_core, norm=None: _mix(x))
    built = _prefill(
        monkeypatch, layers, kernels, front=True, fp8_rows=("prefill_gate_fp8",)
    )
    assert built.qk_rows and sorted(built.fp8.gate) == [1, 3]
    kernels.calls.clear()
    ids, positions, batch = _sglang_stage(monkeypatch, built)
    with _tp1():
        want = qwen3_5.Qwen3_5ForCausalLM.forward(
            built.lm.model, ids, positions, batch()
        )
    assert o_log == [("o_proj", "quantizes")] * 2 and fp8_gates == []
    o_log.clear()
    kernels.calls.clear()
    kernels.gates.clear()
    fb = batch()
    step = built.plan_body(ids, fb)
    with caplog.at_level("INFO"), _tp1():
        got = built.forward_body(step, None, ids, positions, fb)
    assert torch.equal(_bits(got), _bits(want))
    # Q / K from the row tiles, the gate written as o_proj's E4M3 input, no BF16 gate
    assert [c[0] for c in kernels.calls] == ["supported", "rows"] * 2
    assert kernels.gates == [] and o_log == [("o_proj", "e4m3")] * 2
    T = ids.shape[0]
    rows = [c for c in kernels.calls if c[0] == "rows"]
    for gate, call in zip(fp8_gates, rows, strict=True):
        assert gate.shape == (T, H, D) and gate.stride() == (WIDTH, 2 * D, 1)
        assert gate.storage_offset() == call[1][0] + D
    for name in ("attn_prefill_qk_rows", "prefill_gate_fp8"):
        assert f"slot {name} runs at {T} rows" in caplog.text
    assert "uses stock" not in caplog.text and "use stock" not in caplog.text


@pytest.mark.parametrize("bitwise", [True, False])
def test_the_test_only_parity_enforces_bits_only_without_a_numeric_slot(
    caplog, bitwise
):
    from sgl_nvidia_mega.qwen3_5.runtime.prefill import PrefillParity

    pool = torch.zeros(1, 4)
    view = SimpleNamespace(
        state_indices=torch.tensor([1]),
        has_track=False,
        state=SimpleNamespace(conv=[pool], temporal=torch.zeros(1, 4)),
    )
    stock = lambda: torch.zeros(3)
    numeric = lambda: torch.ones(3)
    with caplog.at_level("INFO"):
        if bitwise:
            with pytest.raises(RuntimeError, match="differs from stock"):
                PrefillParity().run(3, view, stock, numeric)
        else:
            assert torch.equal(
                PrefillParity().run(3, view, stock, numeric, bitwise=False),
                torch.ones(3),
            )
    assert "output_differ=3 state_differ=0" in caplog.text
    assert f"bitwise={int(bitwise)}" in caplog.text


def test_the_slot_is_a_numeric_prefill_row_at_tp1_pp2_only():
    from sgl_nvidia_mega.qwen3_5 import runtime, spec, topology

    assert prefill_rows.SLOT in runtime.OPT_IN_SLOTS
    assert prefill_rows.SLOT in runtime.PREFILL_SLOTS
    assert prefill_rows.SLOT in runtime.MEASURED_PREFILL
    assert prefill_rows.SLOT in runtime.NUMERICS_CHANGE
    assert prefill_rows.SLOT not in runtime.MEASURED_BITWISE
    assert topology.SLOT_TOPOLOGIES[prefill_rows.SLOT] == {"p_tp1_pp2"}
    assert spec.NUMERIC_TOLERANCES[prefill_rows.SLOT] == {
        "attn_prefill_qk_rows.q_k.bf16_differ_fraction_max": 1e-4,
    }
