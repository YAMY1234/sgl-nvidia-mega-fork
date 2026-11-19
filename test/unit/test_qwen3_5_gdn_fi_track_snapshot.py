# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot as ts


def test_the_rows_store_each_tracked_entry_in_its_order_and_skip_the_rest():
    src = torch.tensor([7, 2, 4], dtype=torch.int64)
    plan = ts.checkpoint_rows(src, 9)
    assert plan.rows.dtype == torch.int32
    assert plan.rows.tolist() == [-1, -1, 1, -1, 2, -1, -1, 0, -1]
    assert plan.slots.dtype == src.dtype and plan.slots.tolist() == [0, 1, 2]


def _recording_kernel(calls):
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
    kernel._prefill_fn = lambda **kw: calls.append(kw) or ("out", "state")
    return kernel


def test_the_call_is_sglangs_with_the_destination_rows_on_the_native_kernel():
    args = dict(
        q=torch.zeros(4, 1, 2),
        k=torch.ones(4, 1, 2),
        v=torch.full((4, 2, 2), 2.0),
        alpha=torch.full((4, 2), 3.0),
        beta=torch.full((4, 2), 4.0),
        initial_state=torch.full((2, 2, 2, 2), 5.0),
        cu_seqlens=torch.tensor([0, 2, 4]),
        output=None,
        checkpoint_cu_starts=torch.tensor([0, 1, 2]),
        checkpoint_every_n_tokens=2,
    )
    stock, slot = [], []
    assert _recording_kernel(stock)._run_prefill(state_checkpoints="all", **args) == (
        "out",
        "state",
    )
    rows = torch.tensor([-1, 0], dtype=torch.int32)
    assert ts.run_prefill(
        _recording_kernel(slot),
        state_checkpoints="tracked",
        checkpoint_rows=rows,
        **args,
    ) == ("out", "state")
    ((want,), (got,)) = stock, slot
    for call in (want, got):
        # a fresh final-state buffer like the initial state's
        state = call.pop("output_state")
        assert state.shape == args["initial_state"].shape
        assert state.dtype == args["initial_state"].dtype
        assert state.data_ptr() != args["initial_state"].data_ptr()
    assert want.pop("use_cp") == "auto" and got.pop("use_cp") is False
    assert want.pop("state_checkpoints") == "all"
    assert got.pop("state_checkpoints") == "tracked"
    assert got.pop("checkpoint_state_indices") is rows
    assert got == want


@pytest.fixture
def fake_flashinfer(monkeypatch):
    state = SimpleNamespace(cp=False, capability=(10, 3), asked=[], reads=0)
    helper = ModuleType("flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper")

    def should_use_cp_host(work, sms, name, device_capability=None):
        state.asked.append((work, sms, name, device_capability))
        return state.cp

    def capability(device):
        state.reads += 1
        return state.capability

    helper.should_use_cp_host = should_use_cp_host
    state.entry = object()
    prefill = ModuleType("flashinfer.gdn_prefill")
    prefill.chunk_gated_delta_rule = state.entry
    monkeypatch.setitem(sys.modules, prefill.__name__, prefill)
    monkeypatch.setattr(ts, "_ENTRY", [])
    utils = ModuleType("flashinfer.utils")
    utils.get_compute_capability = capability
    utils.get_device_sm_count = lambda device: 148
    utils.get_device_name = lambda device: "GB300"
    for name in (
        "flashinfer",
        "flashinfer.gdn_kernels",
        "flashinfer.gdn_kernels.delta_rule_dsl",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return state


def test_the_routing_is_flashinfers_context_parallel_heuristic(fake_flashinfer):
    assert ts.routes_native(3, 64, "cpu") is True
    assert fake_flashinfer.asked == [(192, 148, "GB300", (10, 3))]
    fake_flashinfer.cp = True
    assert ts.routes_native(1, 8, "cpu") is False
    # only the SM100 family runs the slot's kernel
    fake_flashinfer.cp = False
    for capability in ((9, 0), (12, 0), (8, 0)):
        fake_flashinfer.capability = capability
        assert ts.routes_native(1, 64, "cpu") is False


def _view(src, **changes):
    return SimpleNamespace(
        **{
            **dict(
                has_track=True,
                track_ssm_h_src=src,
                num_state_checkpoints=6,
                state_checkpoint_every_n_tokens=2,
                state_checkpoint_cu_starts=torch.tensor([0, 2, 3, 5, 6]),
                query_start_loc=torch.tensor([0, 5, 8, 12, 14], dtype=torch.int32),
            ),
            **changes,
        }
    )


def test_a_step_builds_its_rows_once_and_again_for_a_new_source(fake_flashinfer):
    state = ts.TrackSnapshot()
    kernel = SimpleNamespace(_prefill_fn=fake_flashinfer.entry)
    src = torch.tensor([1, 2])
    plan = state.plan(_view(src), kernel, 2)
    assert plan.rows.tolist() == [-1, 0, 1, -1, -1, -1]
    # every layer of the step reads the same rows; the routing is asked once a shape
    assert state.plan(_view(src), kernel, 2) is plan
    assert len(fake_flashinfer.asked) == 1
    # the next step's tensor, or the same tensor written again, builds them anew
    other = torch.tensor([1, 2])
    assert state.plan(_view(other), kernel, 2) is not plan
    again = state.plan(_view(other), kernel, 2)
    other.copy_(torch.tensor([0, 4]))
    rebuilt = state.plan(_view(other), kernel, 2)
    assert rebuilt is not again and rebuilt.rows.tolist() == [0, -1, -1, -1, 1, -1]
    assert len(fake_flashinfer.asked) == 1


def test_a_step_builds_its_rows_from_inference_tensors(fake_flashinfer):
    state = ts.TrackSnapshot()
    kernel = SimpleNamespace(_prefill_fn=fake_flashinfer.entry)
    with torch.inference_mode():
        src = torch.tensor([1, 2])
        plan = state.plan(_view(src), kernel, 2)
        assert src.is_inference() and plan.rows.tolist() == [-1, 0, 1, -1, -1, -1]
        assert state.plan(_view(src), kernel, 2) is plan
        rebuilt = state.plan(_view(torch.tensor([0, 4])), kernel, 2)
    assert rebuilt is not plan and rebuilt.rows.tolist() == [0, -1, -1, -1, 1, -1]
    assert len(fake_flashinfer.asked) == 1


@pytest.mark.parametrize(
    "changes, kernel",
    [
        pytest.param(dict(has_track=False), True, id="no-track"),
        pytest.param(dict(track_ssm_h_src=None), True, id="no-sources"),
        pytest.param(
            dict(track_ssm_h_src=torch.tensor([], dtype=torch.int64)),
            True,
            id="no-tracked-checkpoint",
        ),
        pytest.param(dict(num_state_checkpoints=0), True, id="no-checkpoints"),
        pytest.param(dict(state_checkpoint_every_n_tokens=0), True, id="no-interval"),
        pytest.param(dict(state_checkpoint_cu_starts=None), True, id="no-starts"),
        pytest.param({}, False, id="kernel-without-flashinfer"),
        pytest.param({}, "wrapped", id="kernel-with-another-prefill-function"),
        pytest.param({}, "context-parallel", id="context-parallel-batch"),
    ],
)
def test_a_step_keeps_sglangs_all_row_checkpoints(fake_flashinfer, changes, kernel):
    fake_flashinfer.cp = kernel == "context-parallel"
    prefill_fn = None
    if kernel:
        prefill_fn = fake_flashinfer.entry if kernel != "wrapped" else object()
    view = _view(torch.tensor([1, 2]), **changes)
    assert (
        ts.TrackSnapshot().plan(view, SimpleNamespace(_prefill_fn=prefill_fn), 2)
        is None
    )


def test_the_pins_refuse_another_flashinfer(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "_flashinfer_root", lambda: None)
    assert ts.flashinfer_reasons() == ["needs FlashInfer"]
    for relative in ts.FLASHINFER_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    pinned = {
        r: (hashlib.sha256(r.encode()).hexdigest(),) for r in ts.FLASHINFER_SOURCES
    }
    monkeypatch.setattr(ts, "FLASHINFER_SOURCES", pinned)
    monkeypatch.setattr(ts, "_flashinfer_root", lambda: tmp_path)
    assert ts.flashinfer_reasons() == []
    changed = "gdn_kernels/blackwell/gated_delta_net_chunked.py"
    (tmp_path / changed).write_text("another kernel")
    (why,) = ts.flashinfer_reasons()
    assert changed in why and "gdn_prefill.py," not in why
    (tmp_path / changed).unlink()
    (why,) = ts.flashinfer_reasons()
    assert changed in why
