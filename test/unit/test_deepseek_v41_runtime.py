# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The executor's layer loop on CPU: the order in which it calls the step
functions, which every package codes against."""

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.runtime import executor
from sgl_nvidia_mega.deepseek_v41.runtime.state import StepState


@pytest.fixture
def calls(monkeypatch):
    """Every step function replaced by a recorder."""
    seen = []

    def record(name, ret=None):
        def fn(*args, **kwargs):
            layer = next((a for a in args if isinstance(a, SimpleNamespace)), None)
            sub = next((a for a in args if a in ("attn", "ffn")), None)
            seen.append((name, getattr(layer, "i", None), sub))
            return ret

        return fn

    for pkg, names in {
        "boundary": ["start", "materialize", "pre", "post"],
        "attention": ["step"],
        "moe": ["step"],
        "engram": ["hashes", "apply"],
    }.items():
        for name in names:
            monkeypatch.setattr(
                getattr(executor, pkg), name, record(f"{pkg}.{name}", "x")
            )
    monkeypatch.setattr(executor.boundary, "final", record("boundary.final", (1, 2)))

    @contextlib.contextmanager
    def with_current_layer(i):
        yield

    recorder = SimpleNamespace(with_current_layer=with_current_layer)
    monkeypatch.setattr(
        "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
        lambda: recorder,
    )
    return seen


def _state(engram_layers=(), capture=None):
    layers = [
        SimpleNamespace(i=i, engram=object() if i in engram_layers else None)
        for i in range(3)
    ]
    model = SimpleNamespace(
        start_layer=0,
        end_layer=3,
        layers=layers,
        dspark_layers_to_capture=capture,
    )
    st = StepState(
        lm=SimpleNamespace(model=model),
        forward_batch=None,
        positions=None,
        input_ids=torch.zeros(6, dtype=torch.int64),
        input_ids_global=None,
        slots=None,
        bound={},
    )
    st.aux = [] if capture is not None else None
    return st


LAYER = [
    ("boundary.pre", "attn"),
    ("attention.step", None),
    ("boundary.post", "attn"),
    ("moe.step", None),
    ("boundary.post", "ffn"),
]


def test_each_layer_runs_pre_attention_post_moe_post(calls):
    out = executor.MegakernelDecode(None, {}, {}, ())._layers(
        _state(), torch.zeros(6, 4, 8)
    )
    assert out == (1, 2)
    assert calls[:2] == [("engram.hashes", None, None), ("boundary.start", None, None)]
    body = [(name, sub) for name, _, sub in calls[2:-1]]
    assert body == LAYER * 3
    assert [i for _, i, _ in calls[2:-1]] == [i for i in range(3) for _ in LAYER]
    assert calls[-1][0] == "boundary.final"


def test_engram_and_capture_layers_read_the_materialized_streams(calls, monkeypatch):
    st = _state(engram_layers=(1,), capture=[2])
    st_residual = torch.ones(6, 4, 8)

    def materialize(layer, st):
        calls.append(("boundary.materialize", layer.i, None))
        st.state = SimpleNamespace(residual=st_residual)

    monkeypatch.setattr(executor.boundary, "materialize", materialize)
    executor.MegakernelDecode(None, {}, {}, ())._layers(st, torch.zeros(6, 4, 8))
    names = [(name, i) for name, i, _ in calls]
    # Engram before layer 1's attention; the capture before layer 2's
    assert names.index(("boundary.materialize", 1)) < names.index(("engram.apply", 1))
    assert names.index(("engram.apply", 1)) < names.index(("boundary.pre", 1))
    assert names.index(("boundary.materialize", 2)) < names.index(("boundary.pre", 2))
    assert ("engram.apply", 2) not in names and ("boundary.materialize", 0) not in names
    assert len(st.aux) == 1 and torch.equal(st.aux[0], st_residual.mean(dim=1))


def _batch(mode="TARGET_VERIFY"):
    is_verify = mode == "TARGET_VERIFY"
    return SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode, is_target_verify=lambda: is_verify)
    )


def test_only_verify_steps_of_built_widths_are_covered():
    decode = executor.MegakernelDecode(None, {}, {}, (6, 12, 48))
    ids = torch.zeros(12, dtype=torch.int64)
    assert decode.plan(ids, _batch()) == executor.Step("TARGET_VERIFY", 12)
    assert decode.plan(torch.zeros(18, dtype=torch.int64), _batch()) is None
    assert decode.plan(ids, _batch("DECODE")) is None
    assert decode.plan(ids, _batch(), None, pp_proxy_tensors=object()) is None


def test_the_forward_is_replaced_on_the_instance_only():
    from sgl_nvidia_mega.deepseek_v41.runtime import hooks

    class _Model:
        def forward(self, input_ids, positions, forward_batch, input_embeds=None):
            return "sglang"

    class _Executor:
        def plan(self, input_ids, forward_batch, *args, **kwargs):
            return "step" if forward_batch == "covered" else None

        def forward(self, step, original, *args, **kwargs):
            return ("megakernel", original(*args, **kwargs))

    model, other = _Model(), _Model()
    hooks.install(model, _Executor())
    assert "forward" in vars(model) and "forward" not in vars(other)
    assert other.forward(1, 2, "covered") == "sglang"
    assert model.forward(1, 2, "other") == "sglang"
    assert model.forward(1, 2, "covered") == ("megakernel", "sglang")
    # SGLang reads the forward's parameters (pipeline-parallel support)
    params = inspect.signature(model.forward).parameters
    assert list(params) == ["input_ids", "positions", "forward_batch", "input_embeds"]


def test_slot_selection_is_validated(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    assert runtime._requested_slots() == frozenset(runtime.SHIPPED_SLOTS)
    monkeypatch.setenv(runtime.SLOTS_ENV, "")
    assert runtime._requested_slots() == frozenset()
    monkeypatch.setenv(runtime.SLOTS_ENV, " moe , gate ")
    assert runtime._requested_slots() == {"moe", "gate"}
    for raw, want in (
        ("moe,gatee", "unknown slot 'gatee'"),
        ("gate", "needs slot 'moe'"),
    ):
        monkeypatch.setattr(runtime, "SLOTS", frozenset(raw.split(",")))
        assert any(want in r for r in runtime.slot_reasons()), raw
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    assert runtime.slot_reasons() == []
