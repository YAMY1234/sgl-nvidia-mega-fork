# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os

import pytest
import torch

H, EPS, WORLD8, VOCAB = 4096, 1e-6, 8, 4096
DRAFT_ROWS = (2, 8, 13, 14, 16)
EXTEND_ROWS = (28, 35, 56, 91, 98, 112)
HOP_MAX_M = 28

gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


class _Norm:
    """The GemmaRMSNorm fields the draft reads (raw ``weight``, folded ``gemma_weight``)."""

    def __init__(self, weight):
        self.weight = weight
        self.gemma_weight = torch.add(weight.float(), 1.0).bfloat16()
        self.variance_epsilon = EPS


def _ckpt_tensors(path, device):
    """The MTP front's tensors from an HF checkpoint (by name suffix), or None."""
    from safetensors import safe_open

    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    names = index["weight_map"]
    want = {
        "fc": "mtp.fc.weight",
        "emb": "mtp.pre_fc_norm_embedding.weight",
        "hid": "mtp.pre_fc_norm_hidden.weight",
        "inn": "mtp.layers.0.input_layernorm.weight",
        "table": "embed_tokens.weight",
    }
    got = {}
    for key, suffix in want.items():
        hits = sorted(n for n in names if n.endswith(suffix))
        if key == "table":
            hits = [n for n in hits if not n.startswith("mtp.")] or hits
        if not hits:
            return None
        with safe_open(os.path.join(path, names[hits[0]]), "pt", device="cpu") as f:
            got[key] = f.get_tensor(hits[0]).to(device=device, dtype=torch.bfloat16)
    return got


def _front_tensors(device, seed=0):
    path = os.environ.get("FC_TP_CKPT")
    t = _ckpt_tensors(path, device) if path else None
    if t is not None:
        print(
            f"weights from {path}: fc {tuple(t['fc'].shape)} table {tuple(t['table'].shape)}"
        )
        return t
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (scale * torch.randn(*shape, device=device, generator=g)).bfloat16()

    return dict(
        fc=randn(H, 2 * H, scale=(2 * H) ** -0.5),
        emb=randn(H, scale=0.1),
        hid=randn(H, scale=0.1),
        inn=randn(H, scale=0.1),
        table=randn(VOCAB, H, scale=0.02),
    )


def _inputs(t, rows, device, seed):
    g = torch.Generator(device=device).manual_seed(1000 + seed)
    ids = torch.randint(0, t["table"].shape[0], (rows,), device=device, generator=g)
    hidden = torch.randn(rows, H, device=device, generator=g).bfloat16()
    return ids, hidden


def _chunk(t, ids, hidden, world, rank):
    from sgl_nvidia_mega.qwen3_5.draft.fc_input import mtp_fused_input

    return mtp_fused_input(
        ids, t["table"], hidden, t["emb"], t["hid"], EPS, EPS, world, rank
    )


def _lowm_spec(m, n, k, tactic=None, out_stride=None):
    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import PDL
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import default_tactic

    tac = tactic or tuple(default_tactic(m, n, k, fp8=False))
    s = trt_lowm.spec(
        m=m,
        k=k,
        n_bf16=n,
        fp8_act="bf16",
        tactic_bf16=tac,
        out_stride=out_stride,
        **PDL,
    )
    trt_lowm.prepare(s)
    return s


def _lowm_into(s, x, w, out):
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    trt_lowm.run(s, out=out, x16=x, w_bf16=w)
    return out


def _block(full, w, c0, s=None):
    block = torch.zeros(full.shape[0], H, dtype=torch.bfloat16, device=full.device)
    win = block[:, c0 : c0 + w.shape[0]]
    if s is None:
        torch.mm(full, w.t(), out=win)
    else:
        _lowm_into(s, full, w, win)
    return block


def _rank_order_sum(partials):
    acc = None
    for p in partials:
        p = torch.where(p.view(torch.int16) == -32768, torch.zeros_like(p), p)
        acc = p.float() if acc is None else acc + p.float()
    return acc.bfloat16()


def _hop_norm(summed, inn):
    from sgl_kernel import gemma_fused_add_rmsnorm

    x, r = summed.clone(), torch.zeros_like(summed)
    gemma_fused_add_rmsnorm(x, r, inn, EPS)
    return r, x


def _bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@gpu
@pytest.mark.parametrize("rows", DRAFT_ROWS + EXTEND_ROWS)
def test_fc_tp_blocks_sum_exactly_world8(rows):
    from sgl_nvidia_mega.qwen3_5.draft.fc_tp import LOWM_CANDIDATES, LOWM_RACE_ROWS

    device = torch.device("cuda")
    t = _front_tensors(device)
    n = H // WORLD8
    with torch.inference_mode():
        ids, hidden = _inputs(t, rows, device, seed=rows)
        full = _chunk(t, ids, hidden, 1, 0)
        windows = [t["fc"][r * n : (r + 1) * n] for r in range(WORLD8)]
        paths = {"cublas": [_block(full, w, r * n) for r, w in enumerate(windows)]}
        if rows <= LOWM_RACE_ROWS:
            for tac in [c for c in LOWM_CANDIDATES if c[3] == 0 or rows <= 8] + [None]:
                try:
                    s = _lowm_spec(rows, n, 2 * H, tac, out_stride=H)
                except Exception as exc:  # noqa: BLE001 - a refused tactic is skipped
                    print(f"m {rows} tactic {tac} refused: {exc}")
                    continue
                paths[f"low-M {tac or 'default'}"] = [
                    _block(full, w, r * n, s) for r, w in enumerate(windows)
                ]
        for name, blocks in paths.items():
            for r, b in enumerate(blocks):
                rest = b.clone()
                rest[:, r * n : (r + 1) * n] = 0
                assert not rest.view(torch.int16).any(), (name, r)
            summed = _rank_order_sum(blocks)
            cat = torch.cat(
                [b[:, r * n : (r + 1) * n] for r, b in enumerate(blocks)], 1
            )
            assert _bits(summed, cat), name  # the zero-padded sum is exact
            res, _ = _hop_norm(summed, t["inn"])
            assert _bits(res, summed), name  # the hop's residual out: the sum + 0


@gpu
def test_the_cublas_window_write_is_in_place():
    device = torch.device("cuda")
    t = _front_tensors(device)
    n = H // WORLD8
    w = t["fc"][n : 2 * n]
    with torch.inference_mode():
        for rows in (1, 13, 35, 91, 112):
            ids, hidden = _inputs(t, rows, device, seed=rows)
            full = _chunk(t, ids, hidden, 1, 0)
            block = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
            win = block[:, n : 2 * n]
            ptr = win.data_ptr()
            torch.mm(full, w.t(), out=win)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                torch.mm(full, w.t(), out=win)
                torch.cuda.synchronize()
            ops = {e.key for e in prof.key_averages()}
            assert win.data_ptr() == ptr, rows
            assert not ops & {"aten::copy_", "aten::clone", "aten::contiguous"}, (
                rows,
                sorted(ops),
            )
            rest = block.clone()
            rest[:, n : 2 * n] = 0
            assert not rest.view(torch.int16).any() and win.any(), rows
        torch.cuda.synchronize()
