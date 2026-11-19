# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import sys

import pytest
import torch
import triton
import triton.language as tl

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from sgl_nvidia_mega.qwen3_5.sm103 import sharded_gumbel as sg  # noqa: E402

VOCAB, TP = 248_320, 8
COLS = VOCAB // TP


@triton.jit
def _values_ref_kernel(
    logits_ptr,
    stride_row,
    temperature_ptr,
    seed_ptr,
    values_ptr,
    vocab_start,
    valid,
    vocab_size,
    span,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    temperature = tl.load(temperature_ptr + row)
    inverse_temperature = 1.0 / tl.maximum(temperature, 1e-20)
    seed = tl.load(seed_ptr + row)
    quads = tl.arange(0, BLOCK // 4)
    for offset in range(lo, hi, BLOCK):
        columns = offset + tl.arange(0, BLOCK)
        token_ids = vocab_start + columns
        mask = (columns < hi) & (token_ids < vocab_size)
        values = tl.load(
            logits_ptr + row * stride_row + columns, mask=mask, other=float("-inf")
        ).to(tl.float32)
        if temperature > 0.0:
            u0, u1, u2, u3 = tl.rand4x(seed, (vocab_start + offset) // 4 + quads)
            uniform = tl.reshape(tl.join(tl.join(u0, u1), tl.join(u2, u3)), [BLOCK])
            uniform = tl.minimum(tl.maximum(uniform, 1e-12), 0.99999994)
            values = values * inverse_temperature - tl.log(-tl.log(uniform))
        tl.store(values_ptr + row * stride_row + columns, values, mask=mask)


def _case(rows, dtype, rank, seed_hi, gen):
    dev = torch.device("cuda")
    logits = (torch.randn(rows, COLS, generator=gen, device=dev) * 4.0).to(dtype)
    logits[0, 5] = float("nan")
    logits[0, 6] = float("inf")
    logits[1 % rows, 7:300] = float("-inf")
    logits[min(2, rows - 1), 11] = -0.0
    if rows > 3:
        logits[3, 100] = 1e4
        logits[3, 200] = 1e4
        logits[4 % rows, 1940:3880] = float("-inf")
    temps = torch.tensor([1.0, 0.7, 0.0, -1.0, 1.3, 1e-40, 2.0], device=dev)[
        torch.arange(rows, device=dev) % 7
    ]
    temps = temps.to(torch.float32).contiguous()
    hi = (1 << 62) if seed_hi else (2**31 - 1)
    seeds = torch.randint(0, hi, (rows,), generator=gen, device=dev, dtype=torch.int64)
    return logits, temps, seeds, COLS * rank


@pytest.mark.parametrize("rows", [1, 7, 98, 112])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("seed_hi", [False, True])
def test_bitwise(rows, dtype, rank, seed_hi):
    gen = torch.Generator(device="cuda").manual_seed(
        rows * 131 + rank * 7 + int(seed_hi)
    )
    logits, temps, seeds, start = _case(rows, dtype, rank, seed_hi, gen)
    splits = sg.key_splits(COLS)
    ref_k = torch.empty(rows, splits, dtype=torch.int64, device="cuda")
    ref_g = torch.empty_like(ref_k)
    sg._gumbel_shard_key_fast_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temps,
        seeds,
        ref_k,
        ref_g,
        start,
        COLS,
        VOCAB,
        -(-COLS // splits),
        KEY_FLOOR=sg._KEY_FLOOR,
        BLOCK=sg._FAST_BLOCK,
        GREEDY=True,
        num_warps=4,
    )
    got_k = torch.full_like(ref_k, 7)
    got_g = torch.full_like(ref_k, 7)
    values = torch.full((rows, COLS), float("nan"), dtype=torch.float32, device="cuda")
    sg.gumbel_shard_keys_cuda(
        logits,
        temps,
        seeds,
        got_k,
        got_g,
        vocab_start=start,
        valid=COLS,
        vocab_size=VOCAB,
        splits=splits,
        values=values,
    )
    ref_v = torch.full_like(values, float("nan"))
    _values_ref_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temps,
        seeds,
        ref_v,
        start,
        COLS,
        VOCAB,
        -(-COLS // splits),
        BLOCK=sg._FAST_BLOCK,
        num_warps=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(got_k, ref_k), int((got_k != ref_k).sum())
    assert torch.equal(got_g, ref_g), int((got_g != ref_g).sum())
    assert torch.equal(values.view(torch.int32), ref_v.view(torch.int32)), int(
        (values.view(torch.int32) != ref_v.view(torch.int32)).sum()
    )
    # the greedy-only variant and the keys-only variant agree with the combined one
    k2 = torch.empty_like(ref_k)
    sg.gumbel_shard_keys_cuda(
        logits,
        temps,
        seeds,
        k2,
        None,
        vocab_start=start,
        valid=COLS,
        vocab_size=VOCAB,
        splits=splits,
    )
    torch.cuda.synchronize()
    assert torch.equal(k2, ref_k)


def test_dispatch(monkeypatch):
    gen = torch.Generator(device="cuda").manual_seed(5)
    logits, temps, seeds, start = _case(98, torch.bfloat16, 2, False, gen)
    splits = sg.key_splits(COLS)
    assert sg.cuda_supported(
        logits, vocab_start=start, valid=COLS, splits=splits, greedy=True
    )
    cuda_supported = sg.cuda_supported
    outs = {}
    for flag in ("1", "0"):
        monkeypatch.setattr(
            sg,
            "cuda_supported",
            cuda_supported if flag == "1" else (lambda *a, **k: False),
        )
        k = torch.empty(98, splits, dtype=torch.int64, device="cuda")
        g = torch.empty_like(k)
        sg.gumbel_shard_keys_fast(
            logits,
            temps,
            seeds,
            vocab_start=start,
            valid=COLS,
            vocab_size=VOCAB,
            splits=splits,
            out=k,
            greedy_out=g,
        )
        outs[flag] = (k, g)
    torch.cuda.synchronize()
    assert torch.equal(outs["1"][0], outs["0"][0]) and torch.equal(
        outs["1"][1], outs["0"][1]
    )


def _graph_us(fn, calls=20, reps=7):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        t.append(s.elapsed_time(e) * 1e3 / calls)
    t.sort()
    return round(t[0], 2), round(t[len(t) // 2], 2)


def bench(widths=(91, 98, 112), threads=(128, 256)):
    gen = torch.Generator(device="cuda").manual_seed(0)
    splits = sg.key_splits(COLS)
    for rows in widths:
        logits, _, seeds, start = _case(rows, torch.bfloat16, 3, False, gen)
        temps = torch.ones(rows, dtype=torch.float32, device="cuda")
        k = torch.empty(rows, splits, dtype=torch.int64, device="cuda")
        g = torch.empty_like(k)
        rec = {"rows": rows}
        rec["triton_fast"] = _graph_us(
            lambda: sg._gumbel_shard_key_fast_kernel[(rows, splits)](
                logits,
                logits.stride(0),
                temps,
                seeds,
                k,
                g,
                start,
                COLS,
                VOCAB,
                -(-COLS // splits),
                KEY_FLOOR=sg._KEY_FLOOR,
                BLOCK=sg._FAST_BLOCK,
                GREEDY=True,
                num_warps=4,
            )
        )
        for th in threads:
            rec[f"cuda_t{th}"] = _graph_us(
                lambda: sg.gumbel_shard_keys_cuda(
                    logits,
                    temps,
                    seeds,
                    k,
                    g,
                    vocab_start=start,
                    valid=COLS,
                    vocab_size=VOCAB,
                    splits=splits,
                    threads=th,
                )
            )
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"]) or bench())
