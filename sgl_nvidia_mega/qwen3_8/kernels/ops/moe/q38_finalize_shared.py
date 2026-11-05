# SPDX-License-Identifier: Apache-2.0

"""Ordered MK finalize and native shared gate/add, without a routed temporary.

The capture sink is used only around a matching MK producer call. The caller
retains both real stream joins; fallback producers do not enter the sink.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_finalize_shared(
    Hidden,
    Gate,
    Shared,
    Rows,
    Weights,
    Out,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    COLS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    gate_cols = tl.arange(0, 4096)
    w = tl.load(Gate + gate_cols, gate_cols < H, 0).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    h = tl.load(Hidden + row * H + gate_cols, gate_cols < H, 0).to(tl.float32)
    col = tl.program_id(0) * COLS + tl.arange(0, COLS)
    s = tl.load(Shared + row * H + col, col < H, 0).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    total = tl.full((COLS,), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        value = tl.load(Rows + (row * TOPK + slot) * H + col, col < H, 0).to(tl.float32)
        weight = tl.load(Weights + row * TOPK + slot).to(tl.float32)

        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [value, weight],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        total = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [total, product],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    routed = total.to(tl.bfloat16).to(tl.float32)
    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = gate_val * s
    result += routed
    tl.store(Out + row * H + col, result, col < H)


def finalize_shared(
    hidden, gate_weight, shared, rows, weights, output, *, cols=256, use_pdl=True
):
    m, h = hidden.shape
    assert h == 2560 and rows.shape == (m, 10, h) and weights.shape == (m, 10)
    assert shared.shape == output.shape == (m, h)
    assert all(
        t.dtype == torch.bfloat16 and t.is_contiguous()
        for t in (hidden, gate_weight, shared, rows, weights, output)
    )
    assert gate_weight.numel() == h and cols in (256, 512, 1024, 4096)
    _ordered_finalize_shared[(triton.cdiv(h, cols), m)](
        hidden,
        gate_weight,
        shared,
        rows,
        weights,
        output,
        H=h,
        TOPK=10,
        COLS=cols,
        USE_PDL=use_pdl,
        num_warps=16,
        launch_pdl=use_pdl,
    )
    return output
