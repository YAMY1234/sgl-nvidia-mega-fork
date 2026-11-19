# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_qwen3_5_cp_precise import _GATE, _attend, _cp, _Launch, _merge, _ungated

from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L


@pytest.mark.parametrize(
    "b, kw, want",
    [
        (13, dict(p2f16=True), (True, True, False)),
        (64, dict(p2f16=True), (True, True, False)),
        (65, dict(p2f16=True), (False, False, False)),
        # precise wins where both are asked
        (13, dict(p2f16=True, precise=True), (True, False, True)),
        # neither: the one-term P / BF16 call
        (13, dict(), (False, False, False)),
    ],
)
def test_attend_asks_for_two_term_p_and_fp16_partials(b, kw, want):
    got = _attend(_cp(), b, 7, plain_wait=True, **kw)
    assert (got["p_hilo"], got["f16"], got["hilo"]) == want
    assert got["plain_wait"] is True


def test_a_p2f16_call_keeps_its_wait_choice():
    assert _attend(_cp(), 13, 7, p2f16=True)["plain_wait"] is False


def test_the_precise_call_log_names_each_calls_partial_push(monkeypatch, caplog):
    monkeypatch.setattr(L, "_cp_gather_q_ll", _Launch())
    monkeypatch.setattr(L, "_WAIT_ONLY_GATHER", _Launch())
    cp = _cp()
    partials = []
    cp._exact_partial = lambda *a, **k: partials.append(k)
    cp._merge_h = lambda *a, **k: None
    b = 4
    rows = 7 * b
    q = torch.zeros(rows, 4, 256, dtype=torch.int8)
    with caplog.at_level(logging.INFO):
        for pushed in (False, True, True):
            if pushed:
                cp._q_pushed = rows
            cp._ungated_main(
                q,
                "k",
                "v",
                "table",
                "lens",
                7,
                b,
                rows,
                1.0,
                1.0,
                _GATE,
                plain_wait=True,
                p_hilo=True,
                f16=True,
            )
    assert [k["o_push"] for k in partials] == [False, True, True]
    text = caplog.text
    assert text.count("FP16 partials True (partial pushes False)") == 1
    assert text.count("FP16 partials True (partial pushes True)") == 1


def test_a_p2f16_call_keeps_the_calls_transport(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        _, part, (_, merge) = _ungated(
            monkeypatch, pushed=True, plain_wait=True, p_hilo=True, f16=True
        )
    assert (part["o_push"], part["o_f16"], part["p_hilo"], part["o_hilo"]) == (
        True,
        True,
        True,
        False,
    )
    assert (merge["nopush"], merge["f16"], merge["hilo"]) == (True, True, False)
    assert "FP16 partials True (partial pushes True)" in caplog.text
    _, part0, (_, merge0) = _ungated(monkeypatch, pushed=True, plain_wait=True)
    assert part0["o_push"] is True and part0["o_f16"] is False
    assert merge0["nopush"] is True and merge0["f16"] is False
    drop = ("o_f16", "p_hilo")
    assert {k: v for k, v in part.items() if k not in drop} == {
        k: v for k, v in part0.items() if k not in drop
    }
    assert {k: v for k, v in merge.items() if k != "f16"} == {
        k: v for k, v in merge0.items() if k != "f16"
    }
    # precise (BF16 + residual partials) still stands the partial's push down
    _, partp, (_, mergep) = _ungated(
        monkeypatch, pushed=True, plain_wait=True, hilo=True, p_hilo=True
    )
    assert partp["o_push"] is False and partp["o_f16"] is False
    assert mergep["f16"] is False


@pytest.mark.parametrize("merge_h, kept", [(False, False), (True, True)])
def test_the_plain_merge_keeps_the_call_as_before(monkeypatch, merge_h, kept):
    _, part, (_, merge) = _ungated(
        monkeypatch,
        plain_wait=True,
        early_merge=False,
        merge_h=merge_h,
        p_hilo=True,
        f16=True,
    )
    assert part["p_hilo"] is kept and part["o_f16"] is kept and merge["f16"] is kept
    assert merge["plain"] is not merge_h


def test_f16_and_residual_partials_exclude_each_other(monkeypatch):
    with pytest.raises(AssertionError, match="no residual words"):
        _ungated(monkeypatch, plain_wait=True, hilo=True, p_hilo=True, f16=True)


def test_the_per_head_merge_takes_f16(monkeypatch):
    _, opts = _merge(monkeypatch, early=True, plain_wait=True, f16=True)
    assert opts["F16"] is True and "HILO" not in opts
    _, opts = _merge(monkeypatch, early=True, plain_wait=True, nopush=True, f16=True)
    assert opts["F16"] is True and opts["NOPUSH"] is True
    _, opts = _merge(monkeypatch, early=True, plain_wait=True)
    assert "F16" not in opts


@pytest.mark.parametrize("kw", [dict(plain=True), dict(hilo=True, early=True)])
def test_f16_is_the_per_head_merges_without_residual_words(monkeypatch, kw):
    with pytest.raises(AssertionError, match="F16"):
        _merge(monkeypatch, f16=True, **kw)


def test_the_partial_takes_o_f16(monkeypatch):
    calls = []
    monkeypatch.setattr(F, "q35_cp_fused_partial", lambda *a, **k: calls.append(k))
    cp = _cp(
        po_x=torch.zeros(4, 56, 4, 256, dtype=torch.bfloat16),
        pl_x=torch.zeros(4, 56, 4),
        q_all=torch.zeros(28, 16, 256, dtype=torch.float8_e4m3fn),
    )
    args = ("k", "v", "table", "lens", 7, 4, 28, 1.0, 1.0)
    cp._exact_partial(*args, pdl=True, len_sub=7, p_hilo=True, o_f16=True)
    cp._exact_partial(*args, pdl=True, len_sub=7, p_hilo=True, o_hilo=True)
    cp._exact_partial(*args, pdl=True, len_sub=7)
    f16, precise, plain = calls
    assert f16["o_f16"] is True and f16["p_hilo"] is True and "o_hilo" not in f16
    assert "o_f16" not in precise and "o_f16" not in plain


def test_the_kernels_take_the_options_default_off():
    assert inspect.signature(F._compiled).parameters["o_f16"].default is False
    assert (
        inspect.signature(F.q35_cp_fused_partial).parameters["o_f16"].default is False
    )
    merge = L._cp_merge_h_ll
    fn = getattr(merge, "fn", merge)
    assert inspect.signature(fn).parameters["F16"].default is False


ATTENTION_TP = {"tp4": 4, "tp8": 8, "tep8": 8}
_BUILT: dict = {}


def _built(tp: int):
    if tp in _BUILT:
        return _BUILT[tp]
    import torch.distributed._symmetric_memory as symm_mem

    from sgl_nvidia_mega.qwen3_5 import cp_lite_topology

    cp, hq = cp_lite_topology(tp)
    device = SimpleNamespace(multi_processor_count=152)
    stubs = [
        patch.object(
            symm_mem,
            "empty",
            lambda shape, dtype, device: torch.zeros(shape, dtype=dtype),
        ),
        patch.object(
            symm_mem,
            "rendezvous",
            lambda buf, group: SimpleNamespace(rank=0, buffer_ptrs=[0] * tp),
        ),
        patch.object(symm_mem, "get_backend", lambda device: "NCCL"),
        patch.object(torch.distributed, "barrier", lambda *a, **k: None, create=True),
        patch.object(torch.cuda, "get_device_properties", lambda d: device),
        patch.object(torch.cuda, "synchronize", lambda *a, **k: None),
        patch.object(torch.cuda, "Stream", lambda *a, **k: None),
        patch.object(F, "q35_cp_fused_prepare", lambda *a, **k: None),
    ]
    with contextlib.ExitStack() as stack:
        for stub in stubs:
            stack.enter_context(stub)
        _BUILT[tp] = L.CpLite(None, 0, tp, cp, hq, "cpu")
    return _BUILT[tp]


def test_tp4s_cp_lite_has_no_ll_path():
    tp4, tp8 = _built(4), _built(8)
    assert (tp4.cp, tp4.hq, tp4.ll, tp4.fused, tp4.exact) == (2, 8, False, False, False)
    assert (tp8.cp, tp8.hq, tp8.ll, tp8.fused, tp8.exact) == (4, 4, True, True, True)


def _fold_call(monkeypatch, cp, b, **kw):
    launches = {}
    for name in ("_cp_gather_q", "_cp_gather_q_ll", "_cp_a2a_merge"):
        launches[name] = _Launch()
        monkeypatch.setattr(L, name, launches[name])
    folds = []

    def scalars(args):
        return [a for a in args if not isinstance(a, torch.Tensor)]

    monkeypatch.setattr(
        cp, "_fold_partial", lambda *a, **k: folds.append((scalars(a), k))
    )
    rows = 7 * b
    q = torch.zeros(rows, cp.hq, L.HEAD_DIM, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(4, 1, 64, L.HEAD_DIM, dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(tp_k_head_num=1, sliding_window_size=None, layer_id=3)
    lens = torch.full((b,), 65536, dtype=torch.int32)
    ungated = AssertionError("the ungated exact call")
    with patch.object(L.CpLite, "_ungated_main", side_effect=ungated):
        out = cp.attend(
            q,
            (kv, kv),
            torch.zeros(b, 1, dtype=torch.int32),
            lens,
            7,
            1.0,
            1.0,
            "workspace",
            layer,
            None,
            min_bs=1 if b == 1 else None,
            **kw,
        )
    assert out is not None and tuple(out.shape) == (rows, cp.hq, L.HEAD_DIM)
    ran = {
        name: [(grid, scalars(args), opts) for grid, args, opts in launch.calls]
        for name, launch in launches.items()
    }
    return ran, folds


@pytest.mark.parametrize("b", [1, 2, 13, 64, 65, 128])
@pytest.mark.parametrize(
    "kw",
    [
        dict(p2f16=True),
        dict(p2f16=True, plain_wait=True),
        dict(precise=True),
        dict(precise=True, early_merge=False, merge_h=True),
    ],
)
def test_tp4s_calls_run_the_fold_path_as_before(monkeypatch, b, kw):
    cp = _built(4)
    ran, folds = _fold_call(monkeypatch, cp, b, **kw)
    rows = 7 * b
    assert [call[0] for call in ran["_cp_gather_q"]] == [(rows,)]
    assert [call[0] for call in ran["_cp_a2a_merge"]] == [(rows,)]
    assert ran["_cp_gather_q_ll"] == [] and len(folds) == 1
    assert (ran, folds) == _fold_call(monkeypatch, cp, b)


def test_the_skv2_cp_exceptions_name_the_calls_without_p2f16():
    import pathlib

    from sgl_nvidia_mega.qwen3_5 import spec, topology

    exc = spec.NUMERIC_EXCEPTIONS["skv2_cp"]
    assert set(exc) == {"two_slot_plans", "tp4_fold_path"}
    for e in exc.values():
        assert e["worst_request"] == "unbounded"
    slot_topologies = topology.SLOT_TOPOLOGIES["skv2_cp"]
    assert slot_topologies == set(ATTENTION_TP)
    with_ll = {t for t in slot_topologies if _built(ATTENTION_TP[t]).ll}
    assert with_ll == set(exc["two_slot_plans"]["topologies"]) == {"tp8", "tep8"}
    without_ll = slot_topologies - with_ll
    assert without_ll == set(exc["tp4_fold_path"]["topologies"]) == {"tp4"}
    lo, hi = exc["two_slot_plans"]["requests"]
    assert (lo, hi) == (L._PRECISE_MAX_BATCH + 1, F.MAX_BATCH)
    for b, took in ((2, True), (lo - 1, True), (lo, False), (hi, False)):
        got = _attend(_cp(), b, 7, plain_wait=True, p2f16=True)
        assert (got["p_hilo"], got["f16"]) == (took, took), b
    assert exc["tp4_fold_path"]["requests"] == (1, L._MAX_ROWS // 7)
    readme = " ".join(
        (pathlib.Path(__file__).parents[2] / "README.md").read_text().split()
    )
    assert "Numerics exceptions (`spec.NUMERIC_EXCEPTIONS`): at TP8 / TEP8" in readme
    # en dash, as the README's other ranges
    assert f"At {lo}\u2013{hi} requests, on CP-lite's two-slot plans" in readme
    assert "At TP4 (CP 2 x 8 query heads a rank" in readme
