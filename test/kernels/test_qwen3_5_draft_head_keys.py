# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

V = 31040
H = 4096

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(V, H, device="cuda", generator=g) * 0.02).bfloat16()
    w[V - 3].copy_(w[5])
    return w


def _keys_head(weight):
    from sgl_nvidia_mega.qwen3_5.draft.head_keys import (
        MAX_ROWS,
        MIN_ROWS,
        STAGES,
        DraftHeadKeys,
    )
    from sgl_nvidia_mega.qwen3_5.sm103.head import mma, xchg
    from sgl_nvidia_mega.qwen3_5.sm103.head.keys import KEY_FLOOR

    dev = weight.device
    h = DraftHeadKeys.__new__(DraftHeadKeys)
    h.weight, h.rows, h.vocab_start, h.valid = (
        weight,
        frozenset(range(MIN_ROWS, MAX_ROWS + 1)),
        0,
        V,
    )
    for m in sorted(h.rows):
        mma._kernel(dev.index, *mma.variant(m, stages=STAGES))
    xchg.build(dev.index)
    h.world, h.rank, h.device = 1, 0, dev
    h._buf = torch.zeros(xchg.buffer_words(1), dtype=torch.int64, device=dev)
    h.peers = [h._buf.data_ptr()]
    h._keys = torch.full(
        (xchg.MAX_ROWS, xchg.KEY_LINE), KEY_FLOOR, dtype=torch.int64, device=dev
    )
    h.keys = h._keys[:, 0]
    h.epoch = xchg._epoch(1, dev)
    return h


def _stock(dev):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import top1 as T

    t = T.Top1Exchange.__new__(T.Top1Exchange)
    t.world, t.rank, t.device = 1, 0, dev
    t.vocab_local, t.vocab_start, t.vocab_size = V, 0, V
    t._buf = torch.zeros(T.SLOTS * T.MAX_ROWS * 2, dtype=torch.int64, device=dev)
    t.uc, t.mc, t.peers, t.cfg = (
        t._buf.data_ptr(),
        0,
        [t._buf.data_ptr()],
        T.DEFAULT_CFG,
    )
    t.epoch = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    t.keys = torch.zeros(T.MAX_ROWS, dtype=torch.int64, device=dev)
    return t


def _outputs(m, dev, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = torch.randint(0, 99, (m, 6), device=dev, generator=g)
    pos = torch.randint(0, 1000, (m,), device=dev, generator=g)
    mrope = torch.randint(0, 1000, (3, m + 3), device=dev, generator=g)
    topk_p = torch.zeros(m, 1, dtype=torch.float32, device=dev)
    return tokens, pos, mrope, topk_p


@pytest.mark.parametrize("nan_rule", ["split", "torch"])
def test_keys_head_writes_what_the_cublas_path_writes(weight, nan_rule):
    from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import (
        NAN_KEY_SPLIT_ARGMAX,
        NAN_KEY_TORCH_ARGMAX,
    )

    dev = weight.device
    nan_key = NAN_KEY_SPLIT_ARGMAX if nan_rule == "split" else NAN_KEY_TORCH_ARGMAX
    w = weight.clone()
    w[11].fill_(float("nan"))  # a NaN logit in every row
    head, stock = _keys_head(w), _stock(dev)
    g = torch.Generator(device="cuda").manual_seed(3)
    for m in range(9, 17):
        x = torch.randn(m, H, device=dev, generator=g).bfloat16()
        x[1].copy_(
            (w[5].float() * 30).bfloat16()
        )  # the planted tie wins this row (split rule)
        a = _outputs(m, dev, m)
        b = tuple(t.clone() for t in a)
        head.top1_into(
            x,
            a[0],
            2,
            nan_key=nan_key,
            positions=a[1],
            mrope_positions=a[2],
            topk_p=a[3],
        )
        stock(
            torch.mm(x, w.t()),
            b[0],
            2,
            nan_key=nan_key,
            positions=b[1],
            mrope_positions=b[2],
            topk_p=b[3],
        )
        torch.cuda.synchronize()
        for got, want in zip(a, b):
            assert torch.equal(got, want), (m, nan_rule)
        if nan_rule == "torch":
            assert (a[0][:, 2] == 11).all()
        else:
            assert int(a[0][1, 2]) == 5 and (a[0][:, 2] != 11).all()


def test_keys_head_keeps_its_protocol_from_a_graph(weight):
    dev = weight.device
    head, stock = _keys_head(weight), _stock(dev)
    from sgl_nvidia_mega.qwen3_5.sm103.draft.xchg import NAN_KEY_SPLIT_ARGMAX

    g = torch.Generator(device="cuda").manual_seed(4)
    widths = [16, 9, 13, 14, 16, 12, 10, 16, 11, 15]
    xs = [torch.randn(m, H, device=dev, generator=g).bfloat16() for m in widths]
    toks = [torch.zeros(m, 2, dtype=torch.int64, device=dev) for m in widths]
    head.top1_into(xs[0], toks[0], 1, nan_key=NAN_KEY_SPLIT_ARGMAX)  # warm
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for x, t in zip(xs, toks):
            head.top1_into(x, t, 1, nan_key=NAN_KEY_SPLIT_ARGMAX)
    want = []
    for x in xs:
        t = torch.zeros(x.shape[0], 2, dtype=torch.int64, device=dev)
        stock(torch.mm(x, weight.t()), t, 1, nan_key=NAN_KEY_SPLIT_ARGMAX)
        want.append(t)
    for _ in range(25):  # tags past the slot count, every fold re-armed
        for t in toks:
            t.zero_()
        gr.replay()
        torch.cuda.synchronize()
        for t, w_ in zip(toks, want):
            assert torch.equal(t, w_)


def test_keys_head_key_ids_are_its_logits_argmax(weight):
    from sgl_nvidia_mega.qwen3_5.draft.head_keys import STAGES
    from sgl_nvidia_mega.qwen3_5.sm103.head import mma
    from sgl_nvidia_mega.qwen3_5.sm103.head.keys import KEY_FLOOR

    dev = weight.device
    g = torch.Generator(device="cuda").manual_seed(5)
    keys = torch.full((16,), KEY_FLOOR, dtype=torch.int64, device=dev)
    for m in (9, 13, 16):
        x = torch.randn(m, H, device=dev, generator=g).bfloat16()
        lg = torch.empty(m, V, dtype=torch.bfloat16, device=dev)
        keys.fill_(KEY_FLOOR)
        mma.head_mma_keys(
            x,
            weight,
            keys,
            vocab_start=0,
            valid=V,
            nan_hi=0x7FFFFFFF,
            logits=lg,
            stages=STAGES,
        )
        ids = (0xFFFFFFFF - (keys[:m] & 0xFFFFFFFF)).to(torch.int64)
        assert torch.equal(ids, lg.float().argmax(dim=1))
