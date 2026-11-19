# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.qwen3_5 import gdn, runtime
from sgl_nvidia_mega.qwen3_5.gdn import verify
from sgl_nvidia_mega.qwen3_5.runtime import executor

CORE_TC = "sgl_nvidia_mega.qwen3_5.sm103.gdn.core_tc"


def _model():
    return SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))


def _fold(widths=(7, 14)):
    return executor.Qwen35FoldVerify(_model(), {}, {}, widths)


def _batch(mode="target_verify"):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_target_verify=lambda: mode == "target_verify",
        )
    )


def test_only_the_fold_build_commits_and_keeps_no_host_backup():
    plain = executor.Qwen35Verify(_model(), {}, {}, (7,))
    fold = _fold()
    for name in ("commit_recurrent_state", "backs_up_state", "reset_state_slots"):
        assert not hasattr(plain, name) and hasattr(fold, name)
    assert fold.backs_up_state is False
    assert fold.keeps_draft_ssm and fold.tracks_decode_state
    assert fold.state_bytes_per_slot == fold.state_bytes_per_request_row == 0
    assert fold.reset_state_slots(torch.zeros(2), torch.zeros(2)) is None
    for proto in ("SupportsRecurrentCommit", "SupportsStateSlotReset"):
        if not hasattr(registry, proto):
            continue  # an SGLang without the protocol never calls it
        assert isinstance(fold, getattr(registry, proto))
        assert not isinstance(plain, getattr(registry, proto))


def test_an_uncovered_target_verify_fails_closed():
    fold = _fold()
    assert fold.plan(torch.zeros(14), _batch()) == Step(mode="target_verify", rows=14)
    assert fold.plan(torch.zeros(21), _batch("decode")) is None
    with pytest.raises(RuntimeError, match="21 rows is not covered"):
        fold.plan(torch.zeros(21), _batch())
    # an argument the executor does not take leaves the step uncovered too
    with pytest.raises(RuntimeError, match="not covered"):
        fold.plan(torch.zeros(14), _batch(), get_embedding=True)


def _io(track=None):
    return SimpleNamespace(
        state_slots=torch.tensor([3, 5]),
        track_slots=track,
        request_rows=torch.tensor([1, 2]),
        accept_lens=torch.tensor([4, 2]),
        last_step=torch.tensor([3, 1]),
        track_steps=None,
        draft_token_num=7,
        src_indices_raw=None,
    )


@pytest.fixture
def launches(monkeypatch):
    """SGLang's commit kernels, recording their calls."""
    calls = []
    for name, fn in (
        (
            "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode",
            "commit_gdn_replayssm_spec",
        ),
        (
            "sglang.kernels.ops.mamba.mamba_state_scatter_triton",
            "fused_conv_window_scatter_with_mask",
        ),
    ):
        module = types.ModuleType(name)
        setattr(module, fn, lambda *a, _fn=fn, **kw: calls.append((_fn, a, kw)))
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _pools():
    return SimpleNamespace(
        ring_write_pos=torch.zeros(4, dtype=torch.int32),
        ring_cache_base=torch.zeros(4, dtype=torch.int32),
        ring_is_flush=torch.zeros(4, dtype=torch.int8),
        ring_d=torch.zeros(2, 4, 1, 16, 1),
        conv=(torch.zeros(1), torch.zeros(2)),
        intermediate_conv_window=(torch.zeros(3), torch.zeros(4)),
    )


def test_the_commit_advances_the_cursors_and_rolls_back_conv_without_the_fold(
    launches,
):
    fold, io = _fold(), _io()
    fold._mamba = pools = _pools()
    assert fold.commit_recurrent_state(io) is True
    (name, args, kw), *conv = launches
    assert name == "commit_gdn_replayssm_spec" and args == ()
    assert kw == dict(
        write_pos=pools.ring_write_pos,
        cache_base=pools.ring_cache_base,
        is_flush=pools.ring_is_flush,
        num_accepted=io.accept_lens,
        replay_indices=io.request_rows,
        state_batch_indices=io.state_slots,
        max_cache_len=16,
        max_spec_len=None,
        fold_every_commit=True,
        null_block_id=-1,
    )
    assert [c[0] for c in conv] == ["fused_conv_window_scatter_with_mask"] * 2
    for (_, args, _), layer, window in zip(
        conv, pools.conv, pools.intermediate_conv_window
    ):
        assert args == (layer, window, io.state_slots, io.last_step, None)


def test_a_tracked_batch_takes_sglangs_whole_commit(launches):
    fold = _fold()
    fold._mamba = _pools()
    assert fold.commit_recurrent_state(_io(track=torch.tensor([7, 8]))) is False
    assert launches == []


def test_a_commit_before_any_covered_verify_fails_closed(launches):
    with pytest.raises(RuntimeError, match="before any covered verify"):
        _fold().commit_recurrent_state(_io())
    assert launches == []


def test_the_first_covered_step_keeps_its_pools(monkeypatch):
    fold = _fold()
    view = SimpleNamespace(mamba=object())
    monkeypatch.setattr(executor.Qwen35Verify, "_check_pools", lambda self, v: None)
    fold._check_pools(view)
    assert fold._mamba is view.mamba


# ------------------------------------------------------------- slot gdn_core_tc


@pytest.fixture
def core_tc(monkeypatch):
    """A stand-in for the tcgen05 launch (CuTe DSL needs a GPU)."""
    module = types.ModuleType(CORE_TC)
    module.calls = []
    module.launchable_ = True
    module.launchable = lambda *a, **kw: module.launchable_
    module.core_tc = lambda *a, **kw: module.calls.append(kw)
    module.build = lambda *a, **kw: module.calls.append(("build", a, kw))
    monkeypatch.setitem(sys.modules, CORE_TC, module)
    monkeypatch.setattr(verify, "_ENGAGED", set())
    return module


def _bound(tc, fold=False):
    return gdn.Bound(
        gdn_verify=True, gdn_core=True, gdn_core_tc=tc, gdn_commit_fold=fold
    )


def _kw():
    return dict(rows_per_request=7, num_k_heads=16, early_dependents=False)


def test_the_core_launch_stays_without_the_slot(core_tc):
    core = object()
    assert verify._core_launch(_bound(False), 14, (), _kw(), core) is core


def test_the_tcgen05_launch_takes_the_step_and_the_fold(core_tc, caplog):
    kw = _kw()
    with caplog.at_level("INFO"):
        assert verify._core_launch(_bound(True), 14, (), kw, None) is core_tc.core_tc
    assert "fold" not in kw and "runs on tcgen05 at 14 rows" in caplog.text
    kw = _kw()
    assert verify._core_launch(_bound(True, True), 21, (), kw, None) is core_tc.core_tc
    assert kw["fold"] is True


def test_an_unbuilt_capture_falls_back_or_fails_closed_with_the_fold(core_tc):
    core_tc.launchable_ = False
    core = object()
    assert verify._core_launch(_bound(True), 14, (), _kw(), core) is core
    with pytest.raises(RuntimeError, match="gdn_commit_fold"):
        verify._core_launch(_bound(True, True), 14, (), _kw(), core)


def _gdns():
    attn = SimpleNamespace(num_k_heads=16, num_v_heads=64, A_log=1, dt_bias=2)
    norm = SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))
    layer = SimpleNamespace(attn=attn, norm=norm)
    return [(0, layer)]


@pytest.mark.parametrize(
    "slots, core, want, reason",
    [
        ({"gdn_core"}, True, (False, False), ""),
        ({"gdn_core_tc"}, False, (False, False), "requires gdn_core"),
        ({"gdn_core_tc"}, True, (True, False), ""),
        ({"gdn_core_tc", "gdn_commit_fold"}, True, (True, True), ""),
    ],
)
def test_the_build_compiles_the_forms_it_runs(
    core_tc, monkeypatch, slots, core, want, reason
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    reasons = []
    ctx = SimpleNamespace(tokens_per_request=7)
    assert gdn._build_core_tc(ctx, _gdns(), core, reasons) == want
    assert reason in " ".join(reasons) and bool(reason) == bool(reasons)
    built = [c for c in core_tc.calls if c[0] == "build"]
    if want[0]:
        ((_, args, kw),) = built
        assert args == (7, 16, 64, False)
        assert kw["fold"] == (want[1],) and kw["early_dependents"] == (False, True)
    else:
        assert built == []


def test_a_failed_compile_keeps_gdn_cores_launch(core_tc, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("no CuTe DSL")

    core_tc.build = fail
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"gdn_core_tc", "gdn_commit_fold"}))
    reasons = []
    ctx = SimpleNamespace(tokens_per_request=7)
    assert gdn._build_core_tc(ctx, _gdns(), True, reasons) == (False, False)
    assert "no CuTe DSL" in reasons[0]


_FOLD_SLOTS = frozenset({"gdn_verify", "gdn_core", "gdn_core_tc", "gdn_commit_fold"})


@pytest.fixture
def fold_config(monkeypatch):
    import sglang.srt.runtime_context as context

    from sgl_nvidia_mega.qwen3_5.runtime import parity

    memory = SimpleNamespace(disable_radix_cache=True, enable_unified_memory=False)
    monkeypatch.setattr(context, "get_memory", lambda: memory, raising=False)
    monkeypatch.setattr(runtime, "_offers_recurrent_commit", lambda: True)
    monkeypatch.setattr(parity, "ENABLED", False)
    monkeypatch.setattr(runtime, "SLOTS", _FOLD_SLOTS)
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    return SimpleNamespace(memory=memory, parity=parity, context=context)


def test_the_fold_holds_where_every_offered_width_folds(fold_config):
    widths = (7, 14, 896)
    assert runtime.apply_commit_fold(widths, widths) == frozenset()
    assert runtime.SLOTS == _FOLD_SLOTS


def _dropped(caplog, offered=(7, 14), candidates=(7, 14)):
    with caplog.at_level("INFO"):
        assert runtime.apply_commit_fold(offered, candidates) == {"gdn_commit_fold"}
    assert runtime.SLOTS == _FOLD_SLOTS - {"gdn_commit_fold"}
    return caplog.text


def test_the_radix_cache_drops_the_fold(fold_config, caplog):
    fold_config.memory.disable_radix_cache = False
    assert "the radix cache is on" in _dropped(caplog)


def test_unified_memory_drops_the_fold(fold_config, caplog):
    fold_config.memory.enable_unified_memory = True
    assert "request sub-pool unit" in _dropped(caplog)


def test_an_unreadable_memory_configuration_drops_the_fold(
    fold_config, monkeypatch, caplog
):
    def unpublished():
        raise ValueError("memory not published")

    monkeypatch.setattr(fold_config.context, "get_memory", unpublished)
    assert "memory not published" in _dropped(caplog)


def test_the_parity_check_drops_the_fold(fold_config, monkeypatch, caplog):
    monkeypatch.setattr(fold_config.parity, "ENABLED", True)
    assert "parity check" in _dropped(caplog)


def test_an_sglang_without_the_commit_seam_drops_the_fold(
    fold_config, monkeypatch, caplog
):
    monkeypatch.setattr(runtime, "_offers_recurrent_commit", lambda: False)
    assert "no SupportsRecurrentCommit" in _dropped(caplog)


@pytest.mark.parametrize(
    "offered, candidates, rows",
    [
        ((7, 903), (7, 903), "[903]"),  # past gdn_core's rows
        ((7, 14, 21), (7, 14), "[21]"),  # offered but not a candidate width
    ],
)
def test_a_width_without_gdn_core_drops_the_fold(
    fold_config, caplog, offered, candidates, rows
):
    assert f"target verifies of {rows} rows" in _dropped(caplog, offered, candidates)


def test_the_fold_is_decided_before_any_package_builds(fold_config, monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import moe

    fold_config.memory.disable_radix_cache = False
    seen = []

    class Built(Exception):
        pass

    def first_build(model, pctx):
        seen.append(runtime.SLOTS)
        raise Built

    for name in ("apply_measured_plan", "build_cp_lite", "apply_cp_lite_topology"):
        monkeypatch.setattr(runtime, name, lambda *a: frozenset())
    monkeypatch.setattr(runtime, "apply_expert_parallel", lambda *a: frozenset())
    monkeypatch.setattr(runtime, "apply_topology", lambda *a: frozenset())
    monkeypatch.setattr(moe, "build", first_build)
    dep4 = SimpleNamespace(tp_size=4, attn_tp_size=1, attn_dp_size=4, moe_ep_size=4)
    monkeypatch.setattr(fold_config.context, "get_parallel", lambda: dep4)
    ctx = SimpleNamespace(
        widths=(7, 14), tokens_per_request=7, deadline_s=60.0, binding=None
    )
    with pytest.raises(Built):
        runtime.build(_model(), ctx)
    assert seen == [_FOLD_SLOTS - {"gdn_commit_fold"}]
