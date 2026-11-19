# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
import itertools
from types import FunctionType, SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime  # noqa: F401  (installs the CPU stubs)
from sgl_nvidia_mega.qwen3_5.gdn import replayssm

H, HV, K, V, DRAFT, RING, SLOTS = 1, 2, 32, 32, 7, 16, 5


class _Ptr:
    """A Triton pointer block: a flat view of a tensor's storage and element offsets."""

    def __init__(self, flat, off):
        self.flat, self.off = flat, off
        self.dtype = SimpleNamespace(element_ty=flat.dtype)

    def __add__(self, x):
        return _Ptr(self.flat, self.off + x)

    __radd__ = __add__


def _ptr(t):
    flat = t.as_strided((t.untyped_storage().nbytes() // t.element_size(),), (1,), 0)
    return _Ptr(flat, torch.tensor(t.storage_offset()))


def _masked(ptr, mask, *rest):
    mask = torch.ones((), dtype=torch.bool) if mask is None else mask
    return torch.broadcast_tensors(ptr.off, torch.as_tensor(mask), *rest)


def _load(ptr, mask=None, other=0.0):
    off, mask = _masked(ptr, mask)
    val = ptr.flat[torch.where(mask, off, 0).long()]
    return torch.where(mask, val, torch.as_tensor(other, dtype=val.dtype))


def _store(ptr, val, mask=None):
    off, mask, val = _masked(ptr, mask, torch.as_tensor(val))
    ptr.flat[off[mask].long()] = val[mask].to(ptr.flat.dtype)


def _language(pid):

    def where(cond, x, y):
        return torch.where(cond, torch.as_tensor(x), torch.as_tensor(y))

    return SimpleNamespace(
        program_id=lambda axis: pid[axis],
        arange=torch.arange,
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        load=_load,
        store=_store,
        where=where,
        sum=lambda x, axis: x.sum(axis),
        cumsum=lambda x, axis: x.cumsum(axis),
        exp=torch.exp,
        log=torch.log,
        sqrt=torch.sqrt,
        fma=lambda x, y, z: x * y + z,
        dot=lambda x, y, input_precision=None: x.float() @ y.float(),
        trans=lambda x: x.T,
        static_range=range,
        float32=torch.float32,
        int32=torch.int32,
        int64=torch.int64,
    )


class _Interpreted:
    """``kernel[grid](...)`` running the kernel's Python body once per program id."""

    def __init__(self, kernel):
        self.fn = getattr(kernel, "fn", kernel)

    def __getitem__(self, grid):
        def launch(*args, num_warps=None, num_stages=None, **kwargs):
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            bound.apply_defaults()
            call = {
                name: _ptr(x) if isinstance(x, torch.Tensor) else x
                for name, x in bound.arguments.items()
            }
            pid = [0, 0, 0]
            scope = {**self.fn.__globals__, "tl": _language(pid)}
            body = FunctionType(self.fn.__code__, scope, self.fn.__name__)
            for program in itertools.product(*map(range, grid)):
                pid[:] = program
                body(**call)

        return launch


@pytest.fixture(autouse=True)
def _interpreted(monkeypatch):
    kernel = replayssm.gdn_replayssm_spec_circular_kernel
    monkeypatch.setattr(
        replayssm, "gdn_replayssm_spec_circular_kernel", _Interpreted(kernel)
    )


def _operands(seed, requests):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    tokens = requests * DRAFT
    return SimpleNamespace(
        qkv=rand(tokens, 2 * H * K + HV * V),
        ba=rand(tokens, 2 * HV),
        A_log=torch.rand(HV, generator=g),
        dt_bias=torch.rand(HV, generator=g),
        state=rand(SLOTS, HV, V, K, scale=0.05),
        ring=[
            rand(SLOTS, HV, RING, V, scale=0.05),
            rand(SLOTS, H, RING, K, scale=0.09),
            -torch.rand(SLOTS, HV, RING, generator=g) * 0.5,
            rand(SLOTS, HV, RING, V, scale=1e-4),
            rand(SLOTS, H, RING, K, scale=1e-4),
        ],
        # 0 or several committed steps in each request's ring
        write_pos=torch.tensor([0, 5, 2, 8, 1], dtype=torch.int32),
        cache_base=torch.tensor([3, 14, 0, 9, 6], dtype=torch.int32),
        states=torch.tensor([4, 1, 3][:requests], dtype=torch.int32),
        replay=torch.tensor([2, 0, 4][:requests], dtype=torch.int32),
        qsl=torch.arange(0, tokens + 1, DRAFT, dtype=torch.int32),
    )


def _verify(o, qkv, ba):
    tokens = qkv.shape[0]
    ring = [t.clone() for t in o.ring]
    out = torch.zeros(tokens, HV, V, dtype=torch.bfloat16)
    replayssm.verify(
        qkv[:, : H * K].view(tokens, H, K),
        qkv[:, H * K : 2 * H * K].view(tokens, H, K),
        qkv[:, 2 * H * K :].view(tokens, HV, V),
        ba[:, HV:],
        ba[:, :HV],
        o.A_log,
        o.dt_bias,
        out,
        o.state,
        *ring,
        None,
        o.qsl,
        o.states,
        o.replay,
        o.write_pos,
        o.cache_base,
        torch.zeros(SLOTS, dtype=torch.int8),
        max_cache_len=RING,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=16,
        num_warps=2,
    )
    return out, ring


# the rows a non-finite value takes: the whole token row, or one of its inputs
POISONS = {
    "token": lambda qkv, ba: (qkv.fill_(float("nan")), ba.fill_(float("nan"))),
    "k": lambda qkv, ba: qkv[H * K : 2 * H * K].fill_(float("nan")),
    "v": lambda qkv, ba: qkv[2 * H * K :].fill_(float("inf")),
    "b": lambda qkv, ba: ba[:HV].fill_(float("nan")),
    "a": lambda qkv, ba: ba[HV:].fill_(float("nan")),
}


@pytest.mark.parametrize("requests, poisoned", [(1, 0), (3, 1)])
@pytest.mark.parametrize("first", [1, 3, 6])
@pytest.mark.parametrize("poison", list(POISONS))
def test_a_non_finite_rejected_draft_leaves_earlier_positions_as_on_finite_rows(
    poison, first, requests, poisoned
):
    o = _operands(10 * first + requests, requests)
    want, want_ring = _verify(o, o.qkv, o.ba)
    qkv, ba = o.qkv.clone(), o.ba.clone()
    row = poisoned * DRAFT + first
    POISONS[poison](qkv[row], ba[row])
    got, got_ring = _verify(o, qkv, ba)

    assert not got[row].isfinite().all()
    kept = torch.ones(requests * DRAFT, dtype=torch.bool)
    kept[row : (poisoned + 1) * DRAFT] = False
    assert got[kept].isfinite().all()
    assert torch.equal(got[kept], want[kept])
    # the ring records of the rejected positions are the only ones that may differ
    slot = int(o.replay[poisoned])
    at = int(o.cache_base[slot]) + int(o.write_pos[slot])
    rejected = [(at + s) % RING for s in range(first, DRAFT)]
    for g, w in zip(got_ring, want_ring):
        keep = torch.ones_like(g, dtype=torch.bool)
        keep[slot, :, rejected] = False
        assert g[keep].isfinite().all()
        assert torch.equal(g[keep], w[keep])
