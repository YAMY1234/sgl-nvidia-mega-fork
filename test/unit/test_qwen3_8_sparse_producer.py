# SPDX-License-Identifier: Apache-2.0
"""Actual framework producer -> canonical registry -> provider, with CPU launch spies."""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericRegistry
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.layers.attention.qsa import qsa_indexer
from test_qwen3_8_grouped_sparse import grouped_spies as grouped_spies
from test_qwen3_8_sparse import BOUNDARIES
from test_qwen3_8_sparse import raw_spies as raw_spies

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8.spec import SPEC


class DeviceMetadata(torch.Tensor):
    """CPU storage with CUDA eligibility metadata, never a GPU execution claim."""

    @property
    def is_cuda(self):
        return True


@pytest.fixture
def context(monkeypatch, raw_spies):
    registry = NumericRegistry()
    providers.make_offer.cache_clear()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setattr(qsa_indexer, "get_parallel", lambda: SimpleNamespace(tp_size=4))
    monkeypatch.setattr(qsa_indexer, "is_sm100_supported", lambda: True)
    monkeypatch.setattr(qsa_indexer, "is_batch_invariant_mode_enabled", lambda: False)
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_QSA_MK_SCORE.override(True))
        stack.enter_context(envs.SGLANG_QSA_SPLIT_TOPK.override(False))
        stack.enter_context(
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True)
        )
        try:
            yield registry, raw_spies
        finally:
            providers.make_offer.cache_clear()


def indexer():
    obj = object.__new__(qsa_indexer.QSAIndexer)
    torch.nn.Module.__init__(obj)
    obj.index_n_heads, obj.index_kv_heads, obj.index_head_dim = 4, 1, 128
    obj.block_topk, obj.compress_ratio, obj.token_topk = 512, 4, 2048
    return obj


def prepare(obj, context_len, monkeypatch, omitted=None):
    model = torch.nn.Module()
    model.add_module("indexer", obj)
    bound = binding.MegakernelBinding(model=model)
    # Match installation: startup admission uses the selected immutable SPEC.
    bound._numeric_spec = SPEC
    # The old post-pool producer implicitly enumerated the 1..64 preparation
    # program. Install the same explicit widths through the real collector now.
    numeric_binding.collect_numeric_signatures(
        (obj,),
        tuple(range(1, 65)),
        4,
        capability_spec=SPEC,
    )
    monkeypatch.setattr(bound, "_prepare_hybrid", lambda runner: None)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(context_len=context_len),
        token_to_kv_pool=SimpleNamespace(qsa_compressed_page_size=16),
    )
    view = bound.preparation_view(runner)
    assert isinstance(view, binding.PreparationView)
    assert all(
        s.context_bound == ((context_len + 63) // 64) * 16
        for s in view.numeric_signatures
    )
    built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=()))
    if omitted is not None:
        view = binding.PreparationView(
            numeric_signatures=tuple(
                s for s in view.numeric_signatures if s.operation != omitted
            )
        )
    SPEC.prepare(built, view)
    return built


def inputs(obj, length):
    capacity = obj._numeric_score_capacity
    return (
        torch.ones((4, 8, 128), dtype=torch.bfloat16).as_subclass(DeviceMetadata),
        torch.ones((1, 16, 1, 128), dtype=torch.bfloat16),
        torch.zeros((4, capacity // 16), dtype=torch.int64),
        torch.full((4,), length, dtype=torch.int64),
        capacity,
        torch.ones(4, dtype=torch.int64),
    )


@pytest.mark.parametrize("length", BOUNDARIES)
@pytest.mark.parametrize("short", (False, True))
def test_real_preparation_facade_registry_and_raw_calls(
    context, monkeypatch, length, short
):
    registry, (calls, _, _, _) = context
    obj = indexer()
    with envs.SGLANG_QSA_MK_SCORE_SHORT32.override(short):
        built = prepare(obj, length * 4, monkeypatch)
        assert len(built.offers) == 2
        score, select = built.offers
        assert score.signature.variant == (3 if short else 0)
        assert select.signature.variant == 0
        assert registry.get(score.signature) is score
        result = obj._numeric_score_select(*inputs(obj, length))
    assert result.shape == (4, 512) and result.dtype == torch.int32
    names = [name for name, _ in calls]
    assert names == [
        "short-prepare" if short else "mk-prepare",
        "short-launch" if short else "mk-launch",
        "select-prepare",
        "select-launch",
    ]
    score_args, select_args = calls[1][1], calls[3][1]
    assert score_args[5] is select_args[0]
    assert select_args[-1] is result
    assert score_args[2].dtype == score_args[3].dtype == torch.int32
    assert score_args[4].dtype == torch.int64


@pytest.mark.parametrize("omitted", (O.SPARSE_SCORE, O.SPARSE_SELECT))
def test_pair_is_atomic_when_either_real_offer_is_missing(
    context, monkeypatch, omitted
):
    _, (calls, _, _, _) = context
    obj = indexer()
    prepare(obj, 32768, monkeypatch, omitted)
    assert obj._numeric_score_select(*inputs(obj, 8192)) is None
    assert calls == []


@pytest.mark.parametrize("off", ("source", "plugin", "runtime", "split_precedence"))
def test_actual_facade_and_source_policy_keep_stock(context, monkeypatch, off):
    _, (calls, _, _, _) = context
    obj = indexer()
    prepare(obj, 32768, monkeypatch)
    with ExitStack() as stack:
        if off == "source":
            stack.enter_context(envs.SGLANG_QSA_MK_SCORE.override(False))
        elif off == "runtime":
            stack.enter_context(
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False)
            )
        elif off == "plugin":
            monkeypatch.setattr(binding, "megakernel_enabled", lambda: False)
        else:
            stack.enter_context(envs.SGLANG_QSA_SPLIT_TOPK.override(True))
            monkeypatch.setattr(
                qsa_indexer, "get_parallel", lambda: SimpleNamespace(tp_size=1)
            )
        assert obj._numeric_score_select(*inputs(obj, 8192)) is None
    assert calls == []


@pytest.mark.parametrize("rows", range(4, 65, 4))
@pytest.mark.parametrize("logical_heads", (4, 16))
def test_tp1_public_grouped_route_through_real_registry_and_provider(
    context, grouped_spies, monkeypatch, rows, logical_heads
):
    _, (unused_mk_calls, _, _, _) = context
    calls, _, _, _ = grouped_spies
    obj = indexer()
    obj.index_n_heads = logical_heads
    monkeypatch.setattr(qsa_indexer, "get_parallel", lambda: SimpleNamespace(tp_size=1))
    with (
        envs.SGLANG_QSA_MK_SCORE.override(False),
        envs.SGLANG_QSA_SPLIT_TOPK.override(True),
        envs.SGLANG_QSA_GROUPED_SCORE.override(True),
    ):
        prepare(obj, 32769, monkeypatch)
        heads = ((logical_heads + 7) // 8) * 8
        q = torch.ones((rows, heads, 128), dtype=torch.bfloat16)
        q[:, logical_heads:] = 0
        q = q.as_subclass(DeviceMetadata)
        capacity = obj._numeric_score_capacity
        cache = torch.ones((1, 16, 1, 128), dtype=torch.bfloat16)
        table = torch.zeros((rows, capacity // 16), dtype=torch.int64)
        lengths = torch.full((rows,), 8193, dtype=torch.int64)
        result = obj.select_decode_tokens(
            q,
            cache,
            table,
            lengths,
            capacity,
            torch.arange(rows),
            lengths,
            defer_expansion=True,
            rows_per_group=4,
        )
    assert [name for name, _ in calls] == [
        "v7-prepare",
        "v7-launch",
        "split-prepare",
        "split-launch",
    ]
    assert calls[0][1]["heads"] == heads
    assert calls[1][1][0].data_ptr() == q.data_ptr()
    assert calls[-1][1][-1] is result
    assert torch.equal(result, torch.arange(511, -1, -1).expand(rows, -1))
    assert unused_mk_calls == []


@pytest.mark.parametrize("rows", (1, 4, 8, 16, 32, 64))
def test_tp1_draft_perrow_split_pair_and_actual_capture_reuse(
    context, grouped_spies, monkeypatch, rows
):
    calls, state, _, _ = grouped_spies
    obj = indexer()
    obj.index_n_heads = 16
    monkeypatch.setattr(qsa_indexer, "get_parallel", lambda: SimpleNamespace(tp_size=1))
    with (
        envs.SGLANG_QSA_MK_SCORE.override(False),
        envs.SGLANG_QSA_SPLIT_TOPK.override(True),
    ):
        prepare(obj, 32769, monkeypatch)
        capacity = obj._numeric_score_capacity
        args = (
            torch.ones((rows, 16, 128), dtype=torch.bfloat16).as_subclass(
                DeviceMetadata
            ),
            torch.ones((1, 16, 1, 128), dtype=torch.bfloat16),
            torch.zeros((rows, capacity // 16), dtype=torch.int64),
            torch.full((rows,), 8193, dtype=torch.int64),
            capacity,
            None,
        )
        first = obj._numeric_score_select(*args, rows_per_group=1)
        state.capturing = True
        second = obj._numeric_score_select(*args, rows_per_group=1)
    assert [name for name, _ in calls] == [
        "row-prepare",
        "row-launch",
        "split-prepare",
        "split-launch",
        "row-launch",
        "split-launch",
    ]
    assert torch.equal(first, second)


@pytest.mark.parametrize("omitted", (O.SPARSE_SCORE, O.SPARSE_SELECT))
def test_tp1_grouped_pair_missing_operation_does_not_execute(
    context, grouped_spies, monkeypatch, omitted
):
    calls, _, _, _ = grouped_spies
    obj = indexer()
    obj.index_n_heads = 16
    monkeypatch.setattr(qsa_indexer, "get_parallel", lambda: SimpleNamespace(tp_size=1))
    with (
        envs.SGLANG_QSA_MK_SCORE.override(False),
        envs.SGLANG_QSA_SPLIT_TOPK.override(True),
    ):
        prepare(obj, 32768, monkeypatch, omitted)
        assert (
            obj._numeric_score_select(
                torch.ones((32, 16, 128), dtype=torch.bfloat16).as_subclass(
                    DeviceMetadata
                ),
                torch.ones((1, 16, 1, 128), dtype=torch.bfloat16),
                torch.zeros((32, 512), dtype=torch.int64),
                torch.full((32,), 8192, dtype=torch.int64),
                8192,
                None,
                4,
            )
            is None
        )
    assert calls == []
