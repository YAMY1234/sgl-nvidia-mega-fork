# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from sglang.kernels.plugin_kernels.nvidia.spec import VerifyShardHeadView  # noqa: E402

from sgl_nvidia_mega.qwen3_5.sm103 import sharded_gumbel as sg  # noqa: E402
from sgl_nvidia_mega.qwen3_5.sm103.head import verify_head as vh  # noqa: E402

VOCAB, TP, RANK, H = 248_320, 8, 3, 4096
COLS = VOCAB // TP


def _weights():
    ckpt = os.environ.get("Q35_CKPT")
    dev = torch.device("cuda")
    if ckpt:
        import json as _json

        from safetensors import safe_open

        index = _json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))[
            "weight_map"
        ]
        with safe_open(
            os.path.join(ckpt, index["lm_head.weight"]), framework="pt"
        ) as f:
            w = (
                f.get_slice("lm_head.weight")[RANK * COLS : (RANK + 1) * COLS]
                .to(dev)
                .contiguous()
            )
        name = "model.language_model.norm.weight"
        with safe_open(os.path.join(ckpt, index[name]), framework="pt") as f:
            norm = f.get_tensor(name).float().to(dev)
        scale = (
            norm if float(norm.mean()) > 0.5 else 1.0 + norm
        )  # zero-centred RMSNorm weights scale by 1 + w
        return w.to(torch.bfloat16), scale, "checkpoint"
    g = torch.Generator(device=dev).manual_seed(0)
    w = (torch.randn(COLS, H, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    return w, torch.ones(H, device=dev), "random"


_W = None


class _Hook:
    def __init__(self, head):
        self.head = head
        self.keys = head.keys
        self._epoch = head._epoch

    def __call__(self, x, lm, out):
        return self.head(
            VerifyShardHeadView(hidden_states=x, weight=lm.weight, out=out)
        )


def _setup():
    global _W
    if _W is None:
        w, scale, src = _weights()
        lm = SimpleNamespace(weight=w)
        hook = _Hook(vh.VerifyHeadMma(w, vocab_start=RANK * COLS, valid=COLS))
        _W = (w, scale, src, lm, hook)
    return _W


def _x(m, scale, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, H, generator=g, device="cuda") * scale).to(torch.bfloat16)


def test_declines():
    w, scale, src, lm, hook = _setup()
    x = _x(98, scale, 1)
    out = torch.zeros(98, COLS, dtype=torch.bfloat16, device="cuda")
    assert hook(x[:7], lm, out[:7]) is None  # rows below MIN_ROWS
    assert hook(x, SimpleNamespace(weight=w.clone()), out) is None  # another weight
    alloc = hook(x, lm, None)
    assert (
        alloc is not None
        and alloc.dtype == torch.bfloat16
        and tuple(alloc.shape) == (98, COLS)
    )
    ref = hook(x, lm, torch.empty_like(alloc))
    torch.cuda.synchronize()
    assert torch.equal(alloc.view(torch.int16), ref.view(torch.int16))
    assert hook(x, lm, out.float()) is None  # fp32 logits
    assert hook(x, lm, out[:, :-8]) is None  # not [rows, shard rows]
    assert hook(x.float(), lm, out) is None  # fp32 hidden rows
    torch.cuda.synchronize()
    assert int(out.abs().sum()) == 0


def _fused_hook(w):
    hook = _Hook(vh.VerifyHeadMma(w, vocab_start=RANK * COLS, valid=COLS, keys=True))
    assert hook.keys
    return hook


@pytest.mark.parametrize("m", (70, 98, 112, 128))
def test_fused_keys(m):
    w, scale, src, lm, plain = _setup()
    hook = _fused_hook(w)
    x = _x(m, scale, 3)
    out = torch.empty(m, COLS, dtype=torch.bfloat16, device="cuda")
    ref_out = torch.empty_like(out)
    assert plain(x, lm, ref_out) is ref_out
    torch.cuda.synchronize()
    e0 = int(hook._epoch[0])
    assert hook(x, lm, out) is out
    pre = vh.lookup(out)
    assert pre is not None
    from sgl_nvidia_mega.qwen3_5.sm103.head import mma as _mma

    seeds_head = _mma.epoch_seeds(e0, m, out.device)
    torch.cuda.synchronize()
    assert hook._epoch.tolist() == [
        e0 + 1,
        0,
        0,
        0,
    ]  # the head advanced the epoch, its ticket re-armed
    assert torch.equal(
        out.view(torch.int16), ref_out.view(torch.int16)
    )  # the same logits as the logits-only build
    splits = sg.key_splits(COLS)
    temps = torch.tensor([1.0, 1.0, 0.7, 0.0, -1.0, 1.0, 1.3], device="cuda")[
        torch.arange(m, device="cuda") % 7
    ]
    temps = temps.float().contiguous()
    seeds_post = torch.randint(0, 2**31 - 1, (m,), device="cuda", dtype=torch.int64)
    k = torch.empty(m, splits, dtype=torch.int64, device="cuda")
    g = torch.empty_like(k)
    sg.gumbel_shard_keys_fast(
        out,
        temps,
        seeds_post,
        vocab_start=RANK * COLS,
        valid=COLS,
        vocab_size=VOCAB,
        splits=splits,
        out=k,
        greedy_out=g,
        pre=pre,
    )
    torch.cuda.synchronize()
    assert hook._epoch.tolist() == [e0 + 2, 0, 0, 0]  # the keys launch spent them
    use_head = (temps == 1.0) | (temps <= 0)
    ref = {}
    for name, sd in (("head", seeds_head), ("post", seeds_post)):
        rk = torch.empty_like(k)
        rg = torch.empty_like(k)
        sg._gumbel_shard_key_fast_kernel[(m, splits)](
            out,
            out.stride(0),
            temps,
            sd,
            rk,
            rg,
            RANK * COLS,
            COLS,
            VOCAB,
            -(-COLS // splits),
            KEY_FLOOR=sg._KEY_FLOOR,
            BLOCK=sg._FAST_BLOCK,
            GREEDY=True,
            num_warps=4,
        )
        ref[name] = (rk.max(1).values, rg.max(1).values)
    want_k = torch.where(use_head, ref["head"][0], ref["post"][0])
    torch.cuda.synchronize()
    assert torch.equal(k.max(1).values, want_k), int((k.max(1).values != want_k).sum())
    assert torch.equal(g.max(1).values, ref["head"][1])
    k2 = torch.empty_like(k)
    g2 = torch.empty_like(k)
    sg.gumbel_shard_keys_fast(
        out,
        temps,
        seeds_post,
        vocab_start=RANK * COLS,
        valid=COLS,
        vocab_size=VOCAB,
        splits=splits,
        out=k2,
        greedy_out=g2,
        pre=pre,
    )
    torch.cuda.synchronize()
    assert torch.equal(k2.max(1).values, ref["post"][0]) and torch.equal(
        g2.max(1).values, ref["post"][1]
    )
    assert hook(x, lm, out) is out
    pre = vh.lookup(out)
    m2 = m - 7 if m - 7 >= vh.MIN_ROWS else m + 7
    x2 = _x(m2, scale, 4)
    out2 = torch.empty(m2, COLS, dtype=torch.bfloat16, device="cuda")
    assert hook(x2, lm, out2) is out2
    k3 = torch.empty_like(k)
    g3 = torch.empty_like(k)
    sg.gumbel_shard_keys_fast(
        out,
        temps,
        seeds_post,
        vocab_start=RANK * COLS,
        valid=COLS,
        vocab_size=VOCAB,
        splits=splits,
        out=k3,
        greedy_out=g3,
        pre=pre,
    )
    torch.cuda.synchronize()
    assert torch.equal(k3.max(1).values, ref["post"][0]) and torch.equal(
        g3.max(1).values, ref["post"][1]
    )
    own = hook(x, lm, None)
    assert own is not None and vh.lookup(own) is None
    assert hook(x.float(), lm, out) is None and vh.lookup(out) is None
    assert hook(x, lm, out) is out and vh.lookup(out) is not None


def test_fused_keys_stale_after_decline():
    w, scale, src, lm, plain = _setup()
    hook = _fused_hook(w)
    m = 98
    x1, x2 = _x(m, scale, 21), _x(m, scale, 22)
    out = torch.empty(m, COLS, dtype=torch.bfloat16, device="cuda")
    temps = torch.ones(m, dtype=torch.float32, device="cuda")
    seeds = torch.randint(0, 2**31 - 1, (m,), device="cuda", dtype=torch.int64)
    assert hook(x1, lm, out) is out
    pre = vh.lookup(out)
    assert pre is not None
    assert hook(x2.float(), lm, out) is None and vh.lookup(out) is None
    torch.matmul(x2, w.T, out=out)
    splits = sg.key_splits(COLS)
    k = torch.empty(m, splits, dtype=torch.int64, device="cuda")
    g = torch.empty_like(k)
    sg.gumbel_shard_keys_fast(
        out,
        temps,
        seeds,
        vocab_start=RANK * COLS,
        valid=COLS,
        vocab_size=VOCAB,
        splits=splits,
        out=k,
        greedy_out=g,
        pre=pre,
    )
    rk = torch.empty_like(k)
    rg = torch.empty_like(k)
    sg._gumbel_shard_key_fast_kernel[(m, splits)](
        out,
        out.stride(0),
        temps,
        seeds,
        rk,
        rg,
        RANK * COLS,
        COLS,
        VOCAB,
        -(-COLS // splits),
        KEY_FLOOR=sg._KEY_FLOOR,
        BLOCK=sg._FAST_BLOCK,
        GREEDY=True,
        num_warps=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(k.max(1).values, rk.max(1).values) and torch.equal(
        g.max(1).values, rg.max(1).values
    )


@pytest.mark.parametrize("m", (70, 98))
def test_fused_keys_denormal(m):
    from sgl_nvidia_mega.qwen3_5.sm103.head import mma as _mma

    g = torch.Generator(device="cuda").manual_seed(5)
    w = torch.zeros(COLS, H, dtype=torch.float32, device="cuda")
    w[:, :2] = (1.0 + torch.rand(COLS, 2, generator=g, device="cuda")) * 2.0**-10
    w = w.to(torch.bfloat16)
    x = torch.zeros(m, H, dtype=torch.float32, device="cuda")
    x[:, 0] = 2.0**-120
    x[1::2, 1] = 2.0**-121
    x = x.to(torch.bfloat16)
    out = torch.empty(m, COLS, dtype=torch.bfloat16, device="cuda")
    ctas = -(-COLS // 128)
    pre = torch.full((2 * m * ctas + 1,), -1, dtype=torch.int64, device="cuda")
    epoch = torch.zeros(4, dtype=torch.int32, device="cuda")
    _mma.head_mma_keys(
        x,
        w,
        None,
        vocab_start=RANK * COLS,
        valid=COLS,
        nan_hi=0,
        logits=out,
        box_rows=128,
        cluster=1,
        stages=3,
        pre=pre,
        epoch=epoch,
    )
    torch.cuda.synchronize()
    f = out.float()
    assert bool(
        ((f != 0) & (f.abs() < 2.0**-126)).any()
    )  # denormal logits reached the keys
    splits = sg.key_splits(COLS)
    for t in (0.0, 1.0):
        temps = torch.full((m,), t, dtype=torch.float32, device="cuda")
        seeds = _mma.epoch_seeds(0, m, out.device)
        rk = torch.empty(m, splits, dtype=torch.int64, device="cuda")
        rg = torch.empty_like(rk)
        sg._gumbel_shard_key_fast_kernel[(m, splits)](
            out,
            out.stride(0),
            temps,
            seeds,
            rk,
            rg,
            RANK * COLS,
            COLS,
            VOCAB,
            -(-COLS // splits),
            KEY_FLOOR=sg._KEY_FLOOR,
            BLOCK=sg._FAST_BLOCK,
            GREEDY=True,
            num_warps=4,
        )
        ck = torch.empty_like(rk)
        cg = torch.empty_like(rk)
        sg.gumbel_shard_keys_cuda(
            out,
            temps,
            seeds,
            ck,
            cg,
            vocab_start=RANK * COLS,
            valid=COLS,
            vocab_size=VOCAB,
            splits=splits,
        )
        torch.cuda.synchronize()
        assert torch.equal(ck, rk) and torch.equal(cg, rg), t
        assert torch.equal(
            pre[m * ctas : 2 * m * ctas].view(m, ctas).max(1).values, rg.max(1).values
        )
        if t == 1.0:
            assert torch.equal(
                pre[: m * ctas].view(m, ctas).max(1).values, rk.max(1).values
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


def bench(widths=(56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 128)):
    w, scale, src, lm, hook = _setup()
    ws = [w, w.clone()]  # two copies: consecutive calls read different weight buffers
    hooks = [hook, _Hook(vh.VerifyHeadMma(ws[1], vocab_start=RANK * COLS, valid=COLS))]
    assert not hooks[0].keys and not hooks[1].keys
    lms = [lm, SimpleNamespace(weight=ws[1])]
    lo = vh.MIN_ROWS
    vh.MIN_ROWS = min(widths)
    try:
        for m in widths:
            x = _x(m, scale, 2)
            out = torch.empty(m, COLS, dtype=torch.bfloat16, device="cuda")
            it = [0]

            def flip():
                it[0] ^= 1
                return it[0]

            def run_hook():
                i = flip()
                assert hooks[i](x, lms[i], out) is out

            rec = {
                "rows": m,
                "cublas": _graph_us(lambda: torch.matmul(x, ws[flip()].T, out=out)),
                "mma": _graph_us(run_hook),
            }
            if m >= 70:
                splits = sg.key_splits(COLS)
                temps = torch.ones(m, dtype=torch.float32, device="cuda")
                sd = torch.randint(0, 2**31 - 1, (m,), device="cuda", dtype=torch.int64)
                k = torch.empty(m, splits, dtype=torch.int64, device="cuda")
                g = torch.empty_like(k)
                fh = [_fused_hook(ws[0]), _fused_hook(ws[1])]

                def keys(pre):
                    sg.gumbel_shard_keys_fast(
                        out,
                        temps,
                        sd,
                        vocab_start=RANK * COLS,
                        valid=COLS,
                        vocab_size=VOCAB,
                        splits=splits,
                        out=k,
                        greedy_out=g,
                        pre=pre,
                    )

                def plain_pair():
                    run_hook()
                    keys(None)

                def fused_pair():
                    i = flip()
                    assert fh[i](x, lms[i], out) is out
                    keys(vh.lookup(out))

                rec["head+keys"] = _graph_us(plain_pair)
                rec["fused head+keys"] = _graph_us(fused_pair)
            print(json.dumps(rec), flush=True)
    finally:
        vh.MIN_ROWS = lo


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x", "-k", "declines or fused"]) or bench())
