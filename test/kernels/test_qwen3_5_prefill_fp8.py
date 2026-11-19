# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

ROWS = [1, 7, 1023, 1024, 4097, 32768]
HEADS, HEAD_DIM = 32, 256


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer.norm  # noqa: F401
        import sglang.kernels.ops.elementwise.elementwise  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM100-family GPU + SGLang"
)


def _scales(y):
    """The calibrated scale, one that saturates, and two fixed ones."""
    amax = y.float().abs().max().item()
    return [amax / 448.0, amax / 448.0 / 16.0, 1.0, 2.0**-10]


def _static_quant(y, s):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    return static_quant_fp8(y.contiguous(), s)[0]


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _gate_case(rows, seed, layout="flat"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    attn = torch.randn(rows, HEADS * HEAD_DIM, device="cuda", generator=g)
    qg = torch.randn(rows, HEADS, 2 * HEAD_DIM, device="cuda", generator=g)
    qg[:, :, HEAD_DIM : HEAD_DIM + 8] *= 30.0  # past sigmoid's saturation
    qg = qg.to(torch.bfloat16)
    if layout == "flat":
        # the CUDA prepare's [tokens, hidden] gate view
        gate = qg[:, :, HEAD_DIM:].contiguous().view(rows, -1)
    else:
        # the second half of each head's q / gate row (a strided view)
        gate = qg[:, :, HEAD_DIM:]
    return attn.to(torch.bfloat16), gate


@pytest.mark.parametrize("layout", ["flat", "strided"])
@pytest.mark.parametrize("rows", ROWS)
def test_the_gate_writes_static_quant_fp8_of_the_stock_gate(rows, layout):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    attn, gate = _gate_case(rows, rows, layout)
    assert gate_quant.supported(attn, gate)
    stock = fused_sigmoid_mul(attn.clone(), gate, inplace=True)
    before = attn.clone()
    for value in _scales(stock):
        s = torch.tensor(value, dtype=torch.float32, device="cuda")  # 0-d, as ModelOpt
        got = gate_quant.fused_sigmoid_mul_fp8(attn, gate, s)
        assert got.dtype == torch.float8_e4m3fn and got.shape == attn.shape
        assert torch.equal(_bits(got), _bits(_static_quant(stock, s)))
    # attn_output is not written
    assert torch.equal(attn, before)


def _fp8_weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.float8_e4m3fn)
    # ModelOpt keeps the weight transposed ([K, N]) with a per-tensor scale
    return w.t(), torch.tensor(0.0123, dtype=torch.float32, device="cuda")


@pytest.mark.parametrize("route", ["cutlass", "flashinfer_bmm"])
@pytest.mark.parametrize("rows", [7, 4097])
def test_the_linear_gives_the_stock_output_from_the_prequantized_input(route, rows):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
    from sglang.srt.layers.quantization import fp8_utils

    from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

    attn, gate = _gate_case(rows, 7)
    stock = fused_sigmoid_mul(attn.clone(), gate, inplace=True)
    weight, weight_scale = _fp8_weight(4096, HEADS * HEAD_DIM, 3)
    s = torch.tensor(
        stock.float().abs().max().item() / 448.0, dtype=torch.float32, device="cuda"
    )
    q = gate_quant.fused_sigmoid_mul_fp8(attn, gate, s)
    if route == "cutlass":
        run = lambda x: fp8_utils.apply_fp8_linear(  # noqa: E731
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=s,
            pre_quant_output_dtype=torch.bfloat16,
        )
    else:
        run = lambda x: fp8_utils.apply_fp8_linear_bmm_flashinfer(  # noqa: E731
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=s,
            pre_quant_output_dtype=torch.bfloat16,
        )
    assert torch.equal(run(q), run(stock))


ACT_WIDTHS = [512, 1024, 2048, 528]


@pytest.mark.parametrize("width", ACT_WIDTHS)
@pytest.mark.parametrize("rows", ROWS)
def test_the_act_writes_static_quant_fp8_of_sglangs_silu_and_mul(rows, width):
    from sglang.srt.layers.activation import SiluAndMul

    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_act_fp8

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + width)
    gate_up = torch.randn(rows, 2 * width, device="cuda", generator=g) * 4.0
    gate_up[:, :8] *= 20.0  # far into SiLU's tails
    gate_up = gate_up.to(torch.bfloat16)
    stock = SiluAndMul()(gate_up)
    before = gate_up.clone()
    for value in _scales(stock):
        s = torch.tensor(value, dtype=torch.float32, device="cuda")
        got = prefill_act_fp8.silu_and_mul_static_fp8(gate_up, s)
        assert got.dtype == torch.float8_e4m3fn and got.shape == stock.shape
        assert torch.equal(_bits(got), _bits(_static_quant(stock, s)))
    assert torch.equal(gate_up, before)


NV, DV = 64, 128


def _norm_case(tokens, seed, activation):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    g = torch.Generator(device="cuda").manual_seed(seed)
    norm = RMSNorm(DV, eps=1e-6, norm_before_gate=True, activation=activation)
    norm = norm.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(DV, device="cuda", generator=g) * 0.5 + 1)
    x = torch.randn(tokens * NV, DV, device="cuda", generator=g).to(torch.bfloat16)
    # z: the projection's [tokens, heads, DV] view, inside a wider row
    qkvz = torch.randn(tokens, 2 * NV * DV, device="cuda", generator=g)
    qkvz[:, -16:] *= 30.0  # past the gate's saturation
    z = qkvz.to(torch.bfloat16)[:, NV * DV :].view(tokens, NV, DV)
    return norm, x, z


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("rows", ROWS)
def test_the_gated_norm_writes_static_quant_fp8_of_the_stock_norm(rows, activation):
    from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

    norm, x, z = _norm_case(rows, rows + 1, activation)
    assert norm_fp8.unsupported(norm) == ""
    stock = norm(x, z)
    before = x.clone()
    for value in _scales(stock):
        s = torch.tensor(value, dtype=torch.float32, device="cuda")
        got = norm_fp8.prefill_rms_norm_gated_fp8(norm, x, z, s)
        assert got.dtype == torch.float8_e4m3fn and got.shape == x.shape
        assert torch.equal(_bits(got), _bits(_static_quant(stock, s)))
    assert torch.equal(x, before)


NORM_HIDDEN = [2048, 4096, 8192]


@pytest.mark.parametrize("hidden", NORM_HIDDEN)
@pytest.mark.parametrize("rows", ROWS)
def test_the_dual_norm_writes_the_stock_norm_and_static_quant_fp8_of_it(rows, hidden):
    from flashinfer.norm import gemma_fused_add_rmsnorm

    from sgl_nvidia_mega.qwen3_5.sm103 import gemma_norm_fp8

    if not gemma_norm_fp8.cute_route(torch.device("cuda")):
        pytest.skip("FlashInfer runs its CUDA norm kernel here")
    g = torch.Generator(device="cuda").manual_seed(rows + hidden)
    x0 = (torch.randn(rows, hidden, device="cuda", generator=g) * 3).to(torch.bfloat16)
    r0 = torch.randn(rows, hidden, device="cuda", generator=g).to(torch.bfloat16)
    weight = (torch.randn(hidden, device="cuda", generator=g) * 0.2).to(torch.bfloat16)
    x_want, r_want = x0.clone(), r0.clone()
    gemma_fused_add_rmsnorm(x_want, r_want, weight, 1e-6)
    for value in _scales(x_want):
        s = torch.tensor(value, dtype=torch.float32, device="cuda")
        x, r = x0.clone(), r0.clone()
        q = gemma_norm_fp8.gemma_fused_add_rmsnorm_fp8(x, r, weight, 1e-6, s)
        assert torch.equal(_bits(x), _bits(x_want))
        assert torch.equal(_bits(r), _bits(r_want))
        assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
        assert torch.equal(_bits(q), _bits(_static_quant(x_want, s)))
