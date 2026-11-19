# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import importlib.abc
import importlib.machinery
import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

PLAIN = (
    "w1_ptr w1sf_ptr w2_ptr w2sf_ptr b_dummy_ptr xq_addr xsf_addr p2t_addr batch_addr "
    "limit_addr nctas_addr max_ctas scale_c_addr alpha1_addr alpha2_addr h_addr hsf_addr "
    "out_addr state_addr pred_addr npred rec_addr num_experts grid_ctas"
).split()
ROUTED = [p for p in PLAIN if p not in ("pred_addr", "npred", "rec_addr")] + (
    "cand_addr ids_addr wts_addr e2p_addr tp_addr m_rows local_start p2tp_addr".split()
)
ROUTER_PLAIN = ["x_ptr", "w_ptr", "out_ptr"]
ROUTER_TOPK = ROUTER_PLAIN + ["cand_ptr", "gs_ptr", "xq_ptr", "xsf_ptr"]


class _Any:
    """Any attribute / call / subscript of the stand-in DSL; decorators keep the function."""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Any()

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and isinstance(a[0], (types.FunctionType, type)):
            return a[0]
        return (lambda f: f) if (k and not a) else _Any()

    def __getitem__(self, k):
        return _Any()

    def __mro_entries__(self, bases):
        return (object,)

    def _op(self, *o):
        return _Any()

    __add__ = __radd__ = __mul__ = __rmul__ = __sub__ = __rsub__ = __floordiv__ = _op
    __or__ = __ror__ = __and__ = __lshift__ = __rshift__ = __neg__ = __truediv__ = _op


class _Num:
    def __init__(self, v):
        assert v is not None, "None to integer conversion is not supported"
        self.v = v

    def __eq__(self, o):
        return type(self) is type(o) and self.v == o.v

    __hash__ = None


class Int32(_Num):
    pass


class Int64(_Num):
    pass


class _Mod(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return {"Int32": Int32, "Int64": Int64}.get(k) or _Any()


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("cutlass", "cuda"):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _Mod(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


@pytest.fixture
def dsl(monkeypatch):
    finder = _Stub()
    stubbed = ("cutlass", "cuda")
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in stubbed}
    try:
        for k in saved:
            del sys.modules[k]
        sys.meta_path.insert(0, finder)
        with patch.dict(sys.modules):
            for k in [k for k in sys.modules if k.startswith(_MOE) or k == _ROUTE]:
                del sys.modules[k]
            eng = importlib.import_module(_MOE + ".q35_moe_engine")
            rg = importlib.import_module(_MOE + ".q35_router_gemm")
            route = importlib.import_module(_ROUTE)
            yield _record(monkeypatch, eng, rg, route)
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.split(".")[0] in stubbed]:
            del sys.modules[k]
        sys.modules.update(saved)
        _unbind_stub_imports()


def _unbind_stub_imports():
    for name, mod in list(sys.modules.items()):
        if not name.startswith("sgl_nvidia_mega") or not hasattr(mod, "__path__"):
            continue
        for attr, val in list(vars(mod).items()):
            if isinstance(val, types.ModuleType) and val.__name__ == f"{name}.{attr}":
                real = sys.modules.get(val.__name__)
                if real is None:
                    delattr(mod, attr)
                elif real is not val:
                    setattr(mod, attr, real)


_MOE = "sgl_nvidia_mega.qwen3_5.sm103.moe"
_ROUTE = "sgl_nvidia_mega.qwen3_5.moe.route"


def _record(monkeypatch, eng, rg, route):
    compiles = []

    def compile_(kern, *ops):
        rec = SimpleNamespace(kern=kern, ops=ops, launches=[])
        compiles.append(rec)
        return lambda *a: rec.launches.append(a)

    class Kern:
        def __init__(self, *a, **k):
            self.a, self.k = a, k

    out = SimpleNamespace(
        eng=eng,
        route=route,
        compiles=compiles,
        engine=eng.Q35MoeEngine,
        router=rg.MoeRouterGemmKernel,
    )
    monkeypatch.setattr(sys.modules["cutlass.cute"], "compile", compile_, raising=False)
    monkeypatch.setattr(eng, "Q35MoeEngine", Kern)
    monkeypatch.setattr(rg, "MoeRouterGemmKernel", Kern)
    monkeypatch.setattr(eng, "_ptr", lambda dtype, t, align: ("ptr", t.data_ptr()))
    props = SimpleNamespace(multi_processor_count=148)
    stream = SimpleNamespace(cuda_stream=0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d=None: props)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d=None: stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    empty = torch.empty  # the router GEMM's compile-time operands, on the CPU here
    monkeypatch.setattr(torch, "empty", lambda *a, device=None, **k: empty(*a, **k))
    eng._COMPILED.clear()
    return out


def _params(fn):
    """[(name, annotation string)] of a recorded method without self / stream."""
    return [
        (p.name, str(p.annotation))
        for p in inspect.signature(fn).parameters.values()
        if p.name not in ("self", "stream")
    ]


def _check(rec, call, kernel, own):
    assert len(rec.launches) >= 1
    for ops in [rec.ops] + rec.launches:
        ops = ops[:-1]  # the stream
        assert [o is None for o in ops] == [o is None for o in rec.ops[:-1]]
        params = _params(call)
        assert len(ops) == len(params)
        assert [n for (n, _), o in zip(params, ops) if o is not None] == own
        kann = dict(_params(kernel))
        for (n, ann), o in zip(params, ops):
            if o is None:
                assert ann.endswith("| None"), (n, ann)
                assert kann[n].endswith("| None"), (n, kann[n])
            else:
                assert not isinstance(o, _Num) or o.v is not None


def _engine_call(eng, m, tile, routed=False, **kw):
    e, h, i = 128, 4096, 512
    tables = -(-(m * 10) // tile) + e
    rows = tables * tile
    u8, i32 = torch.uint8, torch.int32
    t = lambda *s, dt=u8: torch.zeros(s, dtype=dt)  # noqa: E731
    w = lambda *s: torch.zeros(1, dtype=u8).expand(*s)  # noqa: E731  (shape only)
    args = (
        t(m, h // 2),
        t(m, h // 16),
        t(rows, dt=i32),
        t(tables, dt=i32),
        t(tables, dt=i32),
    )
    args += (t(1, dt=i32), w(e, 2 * i, h // 2), w(e, 2 * i, h // 16), w(e, h, i // 2))
    args += (w(e, h, i // 16), t(e, dt=torch.float32), t(e, dt=torch.float32))
    args += (t(e, dt=torch.float32), t(rows, i // 2), t(rows * i // 16))
    args += (t(rows, h, dt=torch.bfloat16), t(144, dt=i32))
    n_max = 16 if m <= 16 else 32 if m <= 224 else 64
    kw.update(tile=tile, n_max=n_max, tile_k=512 if n_max <= 32 else 256)
    if routed:
        kw.update(
            cand=t(m, 4, 10, dt=i32),
            ids=t(m, 10, dt=i32),
            wts=t(m, 10, dt=torch.bfloat16),
            e2p=t(m * 10, dt=i32),
            total_padded=t(1, dt=i32),
            m=m,
            local_start=128,
        )
    eng.q35_moe_engine(*args, **kw)


PLAIN_CALLS = [
    dict(m=14, tile=8, early_unit=1),
    dict(m=126, tile=32, early_unit=0),
    dict(m=126, tile=32, early_unit=1, early_trigger=2, grid_ctas=136),
    dict(m=231, tile=128, early_unit=1),
    dict(
        m=70,
        tile=16,
        early_unit=1,
        ring=3,
        q0=1,
        rec=torch.zeros(256, dtype=torch.int32),
    ),
    dict(
        m=126,
        tile=8,
        early_unit=1,
        ring=3,
        q0=1,
        pred=torch.zeros(32, dtype=torch.int32),
    ),
]
ROUTED_CALLS = [
    dict(m=35, tile=8, early_unit=1),
    dict(m=63, tile=32, early_unit=0, early_trigger=3, grid_ctas=136),
    dict(m=63, tile=128, early_unit=1, early_trigger=2),
    dict(m=49, tile=16, early_unit=1, early_trigger=3, test_route_delay_ns=20000),
]


@pytest.mark.parametrize("kw", PLAIN_CALLS + ROUTED_CALLS)
def test_an_engine_launch_passes_its_variants_operands_and_none_only_where_optional(
    dsl, kw
):
    routed = kw in ROUTED_CALLS
    _engine_call(dsl.eng, routed=routed, **kw)
    _engine_call(dsl.eng, routed=routed, **kw)  # the second call reuses the compile
    (rec,) = dsl.compiles
    assert len(rec.launches) == 2
    assert rec.kern.k["prerouted"] == int(routed)
    _check(rec, dsl.engine.__call__, dsl.engine.kernel, ROUTED if routed else PLAIN)


def test_plain_and_routed_engine_calls_never_share_a_compile(dsl):
    for kw in PLAIN_CALLS[:2] + ROUTED_CALLS[:1]:
        _engine_call(dsl.eng, routed=kw in ROUTED_CALLS, **kw)
    keys = list(dsl.eng._COMPILED)
    assert len(keys) == len(dsl.compiles) == 3
    assert [c.kern.k["prerouted"] for c in dsl.compiles] == [0, 0, 1]


def test_the_router_gemm_launch_passes_none_for_the_topk_pointers_only_where_optional(
    dsl,
):
    fn = dsl.route._big_gemm_compile(63, 4096, 512, 4096, True)
    x, w = (
        torch.zeros(63, 4096, dtype=torch.bfloat16),
        torch.zeros(512, 4096, dtype=torch.bfloat16),
    )
    dsl.route.big_gemm_launch(fn, x, w)
    (plain,) = dsl.compiles
    assert not plain.kern.k.get("emit_topk") and not plain.kern.k.get("quant")
    _check(plain, dsl.router.__call__, dsl.router.kernel, ROUTER_PLAIN)
    dsl.route._engine_route_compile(63, 4096, 512, 4096, True)
    routed = dsl.compiles[1]
    assert routed.kern.k["emit_topk"] and routed.kern.k["quant"]
    routed.launches.append(routed.ops)
    _check(routed, dsl.router.__call__, dsl.router.kernel, ROUTER_TOPK)
