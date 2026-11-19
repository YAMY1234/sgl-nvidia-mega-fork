# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import inspect
import re
import textwrap

import pytest
import torch
from test_qwen3_5_cp_precise import _cp, _Launch, _merge, _ungated
from test_qwen3_5_cp_q_gather import _cp as _gather_cp
from test_qwen3_5_cp_q_gather import _front_call

from sgl_nvidia_mega.qwen3_5 import runtime
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

NS = 30_000_000_000
ON = {"tmo": NS, "DEADLINE": True}


@pytest.fixture
def deadline(monkeypatch):
    monkeypatch.setattr(L, "_TMO_NS", 0)
    L.set_wait_deadline(NS / 1e9)
    return NS


def test_set_wait_deadline_converts_and_turns_off(monkeypatch):
    monkeypatch.setattr(L, "_TMO_NS", 0)
    assert L._dl_args() == {}
    L.set_wait_deadline(5.0)
    assert L._dl_args() == {"tmo": 5_000_000_000, "DEADLINE": True}
    for off in (None, 0, 0.0):
        L.set_wait_deadline(off)
        assert L._dl_args() == {}
    with pytest.raises(ValueError):
        L.set_wait_deadline(-1.0)


def _params(kernel):
    return inspect.signature(getattr(kernel, "fn", kernel)).parameters


@pytest.mark.parametrize(
    "name",
    [
        "_cp_gather_q",
        "_cp_a2a_merge",
        "_cp_gather_q_ll",
        "_cp_gather_q_wait",
        "_cp_a2a_merge_ll",
        "_cp_a2a_merge4_ll",
        "_cp_merge_h_ll",
        "_cp_merge_h_ll_plain",
    ],
)
def test_every_waiting_kernel_takes_the_deadline_off_by_default(name):
    params = _params(getattr(L, name))
    assert params["tmo"].default == 0 and params["DEADLINE"].default is False


def test_the_q_writers_gather_takes_the_deadline_off_by_default():
    from sgl_nvidia_mega.qwen3_5.sm103.draft import front

    params = _params(front._fused_qk_rmsnorm_rope_gate_kernel)
    assert params["gq_tmo"].default == 0 and params["GQ_DEADLINE"].default is False


@pytest.mark.parametrize("on", [True, False])
def test_the_ungated_exact_call_gathers_under_the_deadline(monkeypatch, on):
    monkeypatch.setattr(L, "_TMO_NS", NS if on else 0)
    _ungated(monkeypatch, plain_wait=True)
    ((_, _, kw),) = L._cp_gather_q_ll.calls
    assert {k: kw[k] for k in ON if k in kw} == (ON if on else {})


@pytest.mark.parametrize(
    "kw",
    [
        dict(early=True),
        dict(early=True, plain_wait=True),
        dict(early=True, plain_wait=True, nopush=True),
        dict(early=True, plain_wait=True, hilo=True),
        dict(early=True, plain_wait=True, f16=True),
        dict(plain_wait=True, nopush=True, hilo=True),
        dict(early=True, plain_wait=True, nopush=True, hilo=True),
        # option draft_hilo_trig: that early merge also releases its dependents at entry
        dict(early=True, plain_wait=True, nopush=True, hilo=True, trig=True),
        dict(plain=True),
    ],
)
def test_every_merge_waits_under_the_deadline(monkeypatch, deadline, kw):
    _, opts = _merge(monkeypatch, **kw)
    assert {k: opts[k] for k in ON} == ON


def test_a_merge_without_the_deadline_launches_as_before(monkeypatch):
    monkeypatch.setattr(L, "_TMO_NS", 0)
    _, opts = _merge(monkeypatch, early=True, plain_wait=True)
    assert "tmo" not in opts and "DEADLINE" not in opts


def test_the_wait_half_forwards_the_deadline(monkeypatch, deadline):
    wait = _Launch()
    monkeypatch.setattr(L, "_cp_gather_q_wait", wait)
    cp = _cp()
    cp.q_push_args(28, 4, 7)
    gather = L._cp_gather_q_ll_or_wait(cp, 28)
    assert gather is L._WAIT_ONLY_GATHER
    args = (None,) * 12
    gather[(28,)](
        *args,
        CP=4,
        WORDS=256,
        QL=7,
        FOLD=True,
        GATE=False,
        NB=4,
        GMAX=False,
        GWRITE=False,
        TRIG=True,
        WAIT=True,
        launch_pdl=True,
        **L._dl_args(),
    )
    ((_, _, kw),) = wait.calls
    assert {k: kw[k] for k in ON} == ON


@pytest.mark.parametrize("on", [True, False])
def test_the_q_writers_gather_polls_under_the_deadline(monkeypatch, on):
    monkeypatch.setattr(L, "_TMO_NS", NS if on else 0)
    cp = _gather_cp()
    args = cp.q_push_args(
        28, 4, 7, gather=True, seq_lens=torch.zeros(4, dtype=torch.int32)
    )
    assert args["gather"].get("tmo", 0) == (NS if on else 0)
    from sgl_nvidia_mega.qwen3_5.sm103.draft import front

    kernel = _Launch()
    monkeypatch.setattr(front, "_fused_qk_rmsnorm_rope_gate_kernel", kernel)
    _front_call({**args, "peers": "peers", "cnt": "cnt"})
    kw = kernel.calls[0][2]
    assert kw["GATHER_Q"] is True
    got = {k: kw[k] for k in ("gq_tmo", "GQ_DEADLINE") if k in kw}
    assert got == ({"gq_tmo": NS, "GQ_DEADLINE": True} if on else {})


def test_the_build_hands_sglangs_deadline_to_cp_lite(monkeypatch):
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused

    monkeypatch.setattr(L, "_TMO_NS", 0)
    monkeypatch.setattr(L.CpLite, "_instance", SimpleNamespace())
    monkeypatch.setattr(q35_cp_fused, "set_partial_build", lambda **kw: None)
    monkeypatch.setattr(q35_cp_fused, "set_cp_phase", lambda on: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp"}))
    runtime.apply_cp_lite_build()
    assert L._dl_args() == {}
    runtime.apply_cp_lite_build(30.0)
    assert L._dl_args() == ON
    # without a CP slot nothing is set
    monkeypatch.setattr(L, "_TMO_NS", 0)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2"}))
    runtime.apply_cp_lite_build(30.0)
    assert L._dl_args() == {}


def _src(name: str, mod=L) -> str:
    fn = getattr(mod, name)
    return textwrap.dedent(inspect.getsource(getattr(fn, "fn", fn)))


def _tree(name: str, mod=L) -> ast.FunctionDef:
    return next(
        n for n in ast.parse(_src(name, mod)).body if isinstance(n, ast.FunctionDef)
    )


@pytest.mark.parametrize(
    "name", ["_fail_late", "_ll_spin_dl", "_send_release_dl", "_wait_acquire_dl"]
)
def test_a_missed_deadline_stores_only_to_address_0x0_then_traps(name):
    asm = _src(name)
    assert re.findall(r"\bst\.[\w.]+", asm) == ["st.volatile.global.u32"]
    assert "mov.u64 %z, 0;" in asm
    assert re.search(r"st\.volatile\.global\.u32 \[%z\], 0;\s*(@%p\s+)?trap;", asm)
    # the flag CAS of the barrier loops is the wait itself, not a status
    atoms = re.findall(r"\batom\.[\w.]+", asm)
    assert atoms == {
        "_send_release_dl": ["atom.global.release.sys.cas.b32"],
        "_wait_acquire_dl": ["atom.global.sys.acquire.cas.b32"],
    }.get(name, [])
    assert "red." not in asm


def _scalar_names(fn: ast.FunctionDef) -> set[str]:
    """Names assigned from a full tl.min reduction (axis=0 over a vector, or nested)."""
    out = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "tl.min"
            and any(
                k.arg == "axis" and ast.unparse(k.value) == "0"
                for k in node.value.keywords
            )
        ):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return out


@pytest.mark.parametrize(
    "mod, name",
    [
        (L, "_ll_wait_plain"),
        (L, "_ll_wait_plain_ee"),
        (L, "_ll_wait_all"),
        (L, "_cp_gather_q_wait"),
        ("front", "_fused_qk_rmsnorm_rope_gate_kernel"),
    ],
)
def test_the_plain_and_one_pass_waits_read_the_timer_once_a_poll(mod, name):
    if mod == "front":
        from sgl_nvidia_mega.qwen3_5.sm103.draft import front as mod
    fn = _tree(name, mod)
    calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_fail_late"
    ]
    assert calls, name
    scalars = _scalar_names(fn)
    for c in calls:
        dep = c.args[1]
        assert isinstance(dep, ast.Name) and dep.id in scalars, ast.unparse(c)
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) in (
            "DL",
            "DEADLINE",
            "GQ_DEADLINE",
        ):
            for c in ast.walk(node):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                    assert c.func.id in {"_fail_late", "_t_end"} or c.func.id in {
                        "_ll_spin_dl",
                        "_group_barrier",
                    }, ast.unparse(c)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute):
                    # the poll itself (volatile loads and its minimum) only
                    assert ast.unparse(c.func) in {
                        "tl.load",
                        "tl.min",
                    } or ast.unparse(c.func).startswith("("), ast.unparse(c)


def _waits(node) -> bool:
    return any(
        isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_gdc_wait_clobber"
        for c in ast.walk(node)
    )


def _stores_nothing(body):
    writes = {"tl.store", "tl.atomic_add", "tl.atomic_xchg", "tl.atomic_cas"}
    for n in body:
        if isinstance(n, ast.For) and ast.unparse(n.iter) == (
            "tl.static_range(0 if NOPUSH else CP)"
        ):
            continue  # the push loop: no step under NOPUSH
        for c in ast.walk(n):
            if isinstance(c, ast.Call):
                f = ast.unparse(c.func)
                assert f not in writes and f != "_st_relaxed_sys_b64", ast.unparse(n)


def test_the_pushed_merge_stores_nothing_before_its_tail_wait():
    fn = _tree("_cp_merge_h_ll")
    tail = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "NOPUSH and EARLY"
        and _waits(n)
    )
    _stores_nothing(fn.body[:tail])


def test_the_pushed_merge_reads_its_gate_after_its_tail_wait():
    fn = _tree("_cp_merge_h_ll")
    tail = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "NOPUSH and EARLY"
        and _waits(n)
    )

    def gate_loads(nodes):
        return [
            c
            for n in nodes
            for c in ast.walk(n)
            if isinstance(c, ast.Call)
            and ast.unparse(c.func) == "tl.load"
            and c.args
            and any(
                isinstance(x, ast.Name) and x.id == "gate" for x in ast.walk(c.args[0])
            )
        ]

    assert not gate_loads(fn.body[:tail])
    assert any(
        isinstance(n, ast.If) and ast.unparse(n.test) == "GATE_BF16" and gate_loads([n])
        for n in fn.body[tail + 1 :]
    )


def test_the_late_pushed_merge_waits_at_entry():
    fn = _tree("_cp_merge_h_ll")
    entry = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "WAIT and (not EARLY)"
        and _waits(n)
    )
    _stores_nothing(fn.body[:entry])


def test_every_poll_of_the_per_head_merge_takes_the_deadline():
    fn = _tree("_cp_merge_h_ll")
    waits = {"_ll_wait", "_ll_wait_plain", "_ll_wait_plain_ee", "_ll_wait_all"}
    seen = set()
    for c in ast.walk(fn):
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) in waits:
            seen.add(c.func.id)
            assert [ast.unparse(a) for a in c.args[-2:]] == ["t_end", "DEADLINE"], (
                ast.unparse(c)
            )
    assert seen == waits
    spins = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.orelse:
            value = getattr(stmt, "value", None)
            if (
                isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "_ll_spin"
            ):
                spins += 1
                assert ast.unparse(node.test) == "DEADLINE"
                (bounded,) = node.body
                assert ast.unparse(bounded.value.func) == "_ll_spin_dl"
    assert spins == sum(
        isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_ll_spin"
        for c in ast.walk(fn)
    )
    assert spins  # the HILO merge's own spin
