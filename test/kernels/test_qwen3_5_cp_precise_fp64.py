# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

PAGE, DIM, CP, HQ = 64, 256, 4, 4
HEADS = CP * HQ  # query heads sharing one KV head
CONTEXT = 262144
K_SCALE, V_SCALE, SCALING = 0.75, 0.5, DIM**-0.5


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
        import sglang  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


class _Fb:
    """A forward batch stand-in (CP-lite keeps a weak reference to it)."""


class CpGroup:
    def __init__(self, cp: int = CP, hq: int = HQ):
        import torch.distributed._symmetric_memory as symm_mem

        from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

        self.cp, self.hq = cp, hq
        self.bufs: list[torch.Tensor] = []
        rank = [0]

        def empty(shape, dtype, device):
            buf = torch.zeros(shape, dtype=dtype, device=device)
            self.bufs.append(buf)
            return buf

        def rendezvous(buf, group):
            return SimpleNamespace(rank=rank[0], buffer_ptrs=[0] * cp)

        patches = [
            mock.patch.object(symm_mem, "empty", empty),
            mock.patch.object(symm_mem, "rendezvous", rendezvous),
            mock.patch.object(symm_mem, "get_backend", lambda device: "NCCL"),
            mock.patch.object(torch.distributed, "barrier", lambda *a, **k: None),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            self.ranks = []
            for r in range(cp):
                rank[0] = r
                self.ranks.append(CpLite(None, r, cp, cp, hq, "cuda"))
        peers = torch.tensor(
            [b.data_ptr() for b in self.bufs], dtype=torch.int64, device="cuda"
        )
        for inst in self.ranks:
            inst.peers = peers.clone()
            inst._ll_tab_t = None
            inst._ll_tab()
        self.o_pushed = [False] * cp
        for r, inst in enumerate(self.ranks):
            inst._exact_partial = self._record_o_push(r, inst._exact_partial)

    def _record_o_push(self, r: int, fn):
        def wrapped(*args, **kwargs):
            self.o_pushed[r] = bool(kwargs.get("o_push", False))
            return fn(*args, **kwargs)

        return wrapped

    @contextlib.contextmanager
    def as_rank(self, r: int):
        from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

        old = CpLite._instance
        CpLite._instance = self.ranks[r]
        try:
            yield self.ranks[r]
        finally:
            CpLite._instance = old

    @staticmethod
    def _pack(epoch: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        return (epoch.to(torch.int64) << 32) | (words.to(torch.int64) & 0xFFFFFFFF)

    def publish(self, r: int, qs: list[torch.Tensor], rows: int) -> None:
        from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import (
            _LL_PWORDS,
            _MAX_ROWS,
        )

        inst, cp, hq = self.ranks[r], self.cp, self.hq
        words = hq * DIM // 4
        buf = self.bufs[r].view(torch.int64)
        e_q = inst.cnt[0][:rows] + 1
        reg = buf[inst.llq_off // 8 :][: rows * cp * words].view(rows, cp, words)
        dw, no = DIM // 2, hq * DIM // 2
        hr = min(2 * rows, _MAX_ROWS)
        e_h = (inst.cnt_h[: hr * hq].view(hr, hq) + 1)[:, :, None]
        regh = buf[inst.llh_off // 8 :][: hr * cp * _LL_PWORDS].view(hr, cp, _LL_PWORDS)
        for c in range(cp):
            if c == r:
                continue
            q32 = qs[c][:rows].reshape(rows, -1).view(torch.int32)
            reg[:, c, :] = self._pack(e_q[:, None], q32)
            if self.o_pushed[c]:
                data = regh[:, c, : hq * dw].view(hr, hq, dw)
                lse = regh[:, c, no : no + hq]
                regh[:, c, : hq * dw] = self._pack(e_h, data).view(hr, hq * dw)
                regh[:, c, no : no + hq] = self._pack(e_h[:, :, 0], lse)
                continue
            peer = self.ranks[c]
            po = peer.po_x[r, :hr].contiguous().view(torch.int32).view(hr, hq, dw)
            regh[:, c, : hq * dw] = self._pack(e_h, po).view(hr, hq * dw)
            pl = peer.pl_x[r, :hr].contiguous().view(torch.int32).view(hr, hq)
            regh[:, c, no : no + hq] = self._pack(e_h[:, :, 0], pl)

    def run(self, call, qs: list[torch.Tensor], rows: int) -> list[torch.Tensor]:
        out = None
        for _ in range(2):
            out = []
            for r in range(self.cp):
                self.publish(r, qs, rows)
                with self.as_rank(r):
                    out.append(call(r, qs[r]))
        return out


_GROUP: list = []


def group() -> CpGroup:
    if not _GROUP:
        _GROUP.append(CpGroup())
    return _GROUP[0]


def make_inputs(lengths, q_len, dist="flat", seed=0, pad_rows=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    b = len(lengths)
    pages = [math.ceil(L / PAGE) for L in lengths]
    total = sum(pages) + 8
    perm = torch.randperm(total - 1, device="cuda", generator=g) + 1
    table = torch.zeros(b, CONTEXT // PAGE, dtype=torch.int32, device="cuda")
    at = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[at : at + n].int()
        at += n
    slots = total * PAGE
    rows = b * q_len + pad_rows

    def randn(*shape, s=1.0):
        return torch.randn(*shape, device="cuda", generator=g) * s

    if dist == "flat":
        q = randn(rows, HEADS, DIM, s=2.0)
        k = randn(slots, 1, DIM, s=2.0)
    elif dist == "peaky":  # a few keys per request aligned with every head's query
        u = (torch.randint(0, 2, (DIM,), device="cuda", generator=g) * 2 - 1) * 0.25
        q = randn(rows, HEADS, DIM) + 6 * u
        k = randn(slots, 1, DIM)
        for i, L in enumerate(lengths):
            hot = torch.randint(0, L, (8,), device="cuda", generator=g)
            loc = table[i, hot // PAGE].long() * PAGE + hot % PAGE
            k[loc, 0] = 4 * u + randn(8, DIM, s=0.5)
    else:
        raise ValueError(dist)
    v = randn(slots, 1, DIM, s=2.0)
    fp8 = torch.float8_e4m3fn
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q.to(fp8), (k.to(fp8), v.to(fp8)), table, lens


def mix(b: int, seed: int, regime: str) -> list[int]:
    g = torch.Generator().manual_seed(1000 * b + seed)
    if regime == "wide":
        return [int(x) for x in torch.randint(1024, CONTEXT, (b,), generator=g)]
    short = [int(x) for x in torch.randint(64, 4096, ((b + 1) // 2,), generator=g)]
    long = [int(x) for x in torch.randint(100000, CONTEXT, (b // 2,), generator=g)]
    return [x for pair in zip(short, long + [0]) for x in pair if x][:b]


def per_rank(q8: torch.Tensor, cp: int = CP) -> list[torch.Tensor]:
    """Rank r's query heads [r*HQ, (r+1)*HQ) as its [rows, HQ, DIM] FP8 rows."""
    return [q8[:, r * HQ : (r + 1) * HQ].contiguous() for r in range(cp)]


def layer():
    return SimpleNamespace(
        layer_id=3,
        tp_q_head_num=HQ,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=DIM,
        v_head_dim=DIM,
        scaling=SCALING,
        sliding_window_size=-1,
    )


def kv_view(kv, scales=(K_SCALE, V_SCALE)):
    return SimpleNamespace(
        layer_ids=[3], k=[kv[0]], v=[kv[1]], k_scale=[scales[0]], v_scale=[scales[1]]
    )


def precise_call(grp, q8, kv, table, lens, q_len, scales=(K_SCALE, V_SCALE), **over):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    attn, kvv = layer(), kv_view(kv, scales)
    meta = SimpleNamespace(page_table=table, cache_seqlens=lens)
    rows, b = q8.shape[0], lens.shape[0]
    multi = b > 1
    options = dict(
        min_bs=None if multi else 1,
        early_merge=False if multi else None,
        merge_h=multi,
        precise=True,
    )
    options.update(over)
    fb = _Fb()

    def call(r, q_r):
        return split_kv.cp_core(
            attn, q_r.reshape(rows, HQ * DIM), kvv, meta, fb, q_len, **options
        )

    outs = grp.run(call, per_rank(q8, grp.cp), min(rows, b * q_len))
    if any(o is None for o in outs):
        return None
    return torch.cat([o.view(rows, HQ, DIM) for o in outs], dim=1)


@pytest.mark.parametrize("q_len", [1, 7])
@pytest.mark.parametrize("b", [2, 9, 16, 33, 64])
@pytest.mark.parametrize(
    "regime,dist", [("wide", "flat"), ("split", "flat"), ("split", "peaky")]
)
def test_the_precise_multi_request_call_is_accepted_and_finite(b, q_len, regime, dist):
    lengths = [max(L, q_len) for L in mix(b, 0, regime)]
    q8, kv, table, lens = make_inputs(lengths, q_len, dist, seed=31 * b + q_len)
    got = precise_call(group(), q8, kv, table, lens, q_len)
    assert got is not None, (b, q_len, "CP-lite declined")
    assert bool(torch.isfinite(got.float()).all())


@pytest.mark.parametrize("q_len", [1, 7])
@pytest.mark.parametrize("length", [32768, 100003, CONTEXT])
def test_the_precise_one_request_call_is_accepted_and_finite(q_len, length):
    q8, kv, table, lens = make_inputs([length], q_len, "peaky", seed=length + q_len)
    got = precise_call(group(), q8, kv, table, lens, q_len)
    assert got is not None
    assert bool(torch.isfinite(got.float()).all())


@pytest.mark.parametrize("q_len", [1, 7])
def test_the_draft_pools_unit_scales_are_accepted_and_finite(q_len):
    lengths = [max(L, q_len) for L in mix(15, 3, "split")]
    q8, kv, table, lens = make_inputs(lengths, q_len, "flat", seed=7 + q_len)
    got = precise_call(group(), q8, kv, table, lens, q_len, scales=(1.0, 1.0))
    assert got is not None
    assert bool(torch.isfinite(got.float()).all())


def test_a_padded_call_serves_its_requests_rows_and_zero_pad_rows():
    grp = group()
    lengths = [50000 + 7919 * i for i in range(9)]
    q8, kv, table, lens = make_inputs(lengths, 1, "flat", seed=7, pad_rows=7)
    want = precise_call(grp, q8[:9].contiguous(), kv, table, lens, 1)
    got = precise_call(grp, q8, kv, table, lens, 1, pad_rows=True)
    assert got is not None and got.shape[0] == 16
    assert torch.equal(got[:9].view(torch.int16), want.view(torch.int16))
    assert not got[9:].view(torch.int16).any()
    # without pad_rows such a call is declined, as before
    assert precise_call(grp, q8, kv, table, lens, 1) is None


def test_past_the_one_slot_plans_the_call_keeps_one_term_p_and_bf16_partials():
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

    grp = group()
    seen = []
    for inst in grp.ranks:
        fn = inst._ungated_main

        def wrapped(*a, _fn=fn, **k):
            seen.append((k.get("hilo"), k.get("p_hilo")))
            return _fn(*a, **k)

        inst._ungated_main = wrapped
    try:
        lengths = [4096 + 97 * i for i in range(L._PRECISE_MAX_BATCH + 1)]
        q8, kv, table, lens = make_inputs(lengths, 2, "flat", seed=65)
        precise_call(grp, q8, kv, table, lens, 2)
    finally:
        for inst in grp.ranks:
            del inst._ungated_main
    # every ungated exact call it made (CP-lite's own gate may also decline it)
    assert all(s == (False, False) for s in seen), seen
