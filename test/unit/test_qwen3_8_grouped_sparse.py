# SPDX-License-Identifier: Apache-2.0
"""Grouped/per-row/split raw-launch ABI, using source config and CPU spies."""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    SparseScoreIO,
    SparseSelectIO,
)
from test_qwen3_8_sparse import install

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import sparse

ROOT = Path(sparse.__file__).parents[1] / "qsa"


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def signature(rows=32, heads=16, variant=4, select=False, capacity=8208):
    return NumericSignature(
        O.SPARSE_SELECT if select else O.SPARSE_SCORE,
        rows,
        heads,
        1,
        128,
        512,
        4,
        D.FP32 if select else D.BF16,
        context_bound=capacity,
        variant=variant,
    )


def score_io(rows=32, heads=16, length=8193):
    return SparseScoreIO(
        torch.ones((rows, heads, 128), dtype=torch.bfloat16),
        torch.ones((1, 16, 1, 128), dtype=torch.bfloat16),
        torch.zeros((rows, 513), dtype=torch.int32),
        torch.full((rows,), length, dtype=torch.int32),
        None,
        None,
        16,
        4,
        math.sqrt(128),
        torch.full((rows, 8208), float("nan")),
        torch.empty(0, dtype=torch.uint8),
    )


def select_io(rows=32, aligned=True):
    values = torch.empty(rows * 8208 + (0 if aligned else 1))
    scores = values.view(rows, 8208) if aligned else values[1:].view(rows, 8208)
    lengths = torch.full((rows,), 8193, dtype=torch.int32)
    return SparseSelectIO(
        scores,
        lengths,
        512,
        torch.empty((rows, 512), dtype=torch.int32),
        lengths.clamp(max=512),
        torch.empty(0, dtype=torch.uint8),
    )


@pytest.fixture
def grouped_spies(monkeypatch):
    calls = []
    state = SimpleNamespace(capturing=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    for name in tuple(os.environ):
        if name.startswith("SGLANG_QSA_SCORER_V2") or name.startswith(
            "SGLANG_QSA_TOPK_"
        ):
            monkeypatch.delenv(name)
    mqa = source_functions(
        ROOT / "mqa.py",
        ("_env_int", "_v2_config", "_decode_ctas_per_row"),
        dict(
            torch=torch,
            _num_sms=lambda device: 148,
            _DECODE_CTAS_PER_ROW_LEVELS=(1, 16, 256),
            _V2_CTA_LEVELS=(16, 32, 64, 128, 256, 512, 1024),
            _V2_TARGET_CTAS=4096,
        ),
    )
    split = source_functions(
        ROOT / "split_topk.py",
        ("_env_int", "_config"),
        dict(
            Optional=__import__("typing").Optional,
            os=os,
            _THREADS=1024,
            _MAX_CLUSTER=16,
            _MAX_ITERS=8,
        ),
    )

    def factory(kind):
        def prepare(**kwargs):
            calls.append((kind + "-prepare", kwargs))

            def run(q, keys, table, lengths, output, scale):
                calls.append(
                    (kind + "-launch", (q, keys, table, lengths, output, scale))
                )
                for row, length in enumerate(lengths.tolist()):
                    if length > 512:
                        output[row, :length] = row

            return run

        return prepare

    mqa["_tilelang_qsa_mqa_decode_grouped_v7_kernel"] = factory("v7")
    mqa["_tilelang_qsa_mqa_decode_kernel"] = factory("row")

    def module(threads, cluster, iters, vec):
        calls.append(("split-prepare", (threads, cluster, iters, vec)))

        def run(scores, lengths, output):
            calls.append(("split-launch", (scores, lengths, output)))
            # Deliberately unsorted marker; no CPU sort can hide index order.
            output.copy_(
                torch.arange(511, -1, -1, dtype=torch.int32).expand(output.shape[0], -1)
            )

        return SimpleNamespace(cluster_topk=run)

    split["_module"] = module
    install(monkeypatch, "sgl_nvidia_mega.qwen3_8.qsa.mqa", **mqa)
    install(monkeypatch, "sgl_nvidia_mega.qwen3_8.qsa.split_topk", **split)
    return calls, state, mqa, split


@pytest.mark.parametrize("heads", (8, 16))
@pytest.mark.parametrize("rows", range(4, 65, 4))
def test_grouped_v7_raw_factory_and_zero_allocation_input_view(
    grouped_spies, rows, heads
):
    calls, _, mqa, _ = grouped_spies
    s = signature(rows, heads)
    assert providers.supported(s)
    io = score_io(rows, heads)
    run = sparse.build(s)
    assert run(io) is None
    cfg = mqa["_v2_config"](rows, 4, io.query.device)
    assert calls[0] == (
        "v7-prepare",
        dict(
            heads=heads,
            head_dim=128,
            page_size=16,
            rows_per_group=4,
            ctas_per_group=cfg["C"],
            min_scored_len=512,
            dtype="bfloat16",
        ),
    )
    q, keys, table, lengths, output, scale = calls[1][1]
    assert q.shape == (rows, 1, heads, 128) and q.data_ptr() == io.query.data_ptr()
    assert q.untyped_storage().data_ptr() == io.query.untyped_storage().data_ptr()
    assert keys is io.compressed_keys and table is io.compressed_slot_map
    assert (
        lengths is io.compressed_lengths
        and output is io.scores_out
        and scale == io.scale
    )


@pytest.mark.parametrize("heads", (8, 16))
@pytest.mark.parametrize("rows", (1, 4, 8, 16, 32, 64))
def test_per_row_uses_original_device_sm_policy(grouped_spies, rows, heads):
    calls, _, mqa, _ = grouped_spies
    io = score_io(rows, heads)
    assert sparse.build(signature(rows, heads, variant=5))(io) is None
    assert calls[0] == (
        "row-prepare",
        dict(
            heads=heads,
            head_dim=128,
            page_size=16,
            ctas_per_row=mqa["_decode_ctas_per_row"](rows, io.query.device),
            min_scored_len=512,
            dtype="bfloat16",
        ),
    )
    assert calls[1][1][0].data_ptr() == io.query.data_ptr()


@pytest.mark.parametrize("rows", range(1, 65))
def test_split_all_declared_rows_use_source_config_and_preserve_order(
    grouped_spies, rows
):
    calls, _, _, split = grouped_spies
    io = select_io(rows)
    counts = io.counts_out.clone()
    assert sparse.build(signature(rows, variant=1, select=True))(io) is None
    expected = (*split["_config"](rows, 8208), True)
    assert calls[0] == ("split-prepare", expected)
    assert calls[1] == (
        "split-launch",
        (io.scores, io.lengths, io.selected_indices_out),
    )
    assert torch.equal(
        io.selected_indices_out, torch.arange(511, -1, -1).expand(rows, -1)
    )
    assert torch.equal(io.counts_out, counts)


@pytest.mark.parametrize("variant", (4, 5))
def test_non_mk_requests_are_optional_but_shape_is_exact(grouped_spies, variant):
    from dataclasses import replace

    io = score_io()
    assert io.row_requests is None
    s = signature(variant=variant)
    for bad in (
        replace(s, query_heads=4),
        replace(s, rows=65),
        replace(s, scalar_type=D.FP16),
    ):
        assert providers.make_offer(bad) is None
    with pytest.raises(ValueError, match="prepared capability"):
        sparse.build(s)(replace(io, query=io.query[:, :8].contiguous()))


@pytest.mark.parametrize("kind", (4, 5, 1))
def test_grouped_row_and_split_capture_miss_and_warm_replay(grouped_spies, kind):
    calls, state, _, _ = grouped_spies
    run = sparse.build(signature(variant=kind, select=kind == 1))
    io = select_io() if kind == 1 else score_io()
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        run(io)
    assert calls == []
    state.capturing = False
    run(io)
    state.capturing = True
    run(io)
    assert sum(name.endswith("prepare") for name, _ in calls) == 1


@pytest.mark.parametrize(
    "env,value,variant",
    (
        ("SGLANG_QSA_SCORER_V2_VER", "6", 4),
        ("SGLANG_QSA_SCORER_V2_C", "64", 4),
        ("SGLANG_QSA_SCORER_V2_TARGET", "1024", 4),
        ("SGLANG_QSA_SCORER_V2_NT", "2", 4),
        ("SGLANG_QSA_SCORER_V2_GUARD", "1", 4),
        ("SGLANG_QSA_SCORER_V2_PEEL", "1", 4),
        ("SGLANG_QSA_SCORER_V2_ROWS", "1", 5),
    ),
)
def test_nondefault_scorer_does_not_launch(
    grouped_spies, monkeypatch, env, value, variant
):
    calls, _, _, _ = grouped_spies
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match="overrides require stock"):
        sparse.build(signature(variant=variant))(score_io())
    assert calls == []


@pytest.mark.parametrize(
    "env,value", (("SGLANG_QSA_TOPK_THREADS", "512"), ("SGLANG_QSA_TOPK_CLUSTER", "4"))
)
def test_nondefault_split_does_not_launch(grouped_spies, monkeypatch, env, value):
    calls, _, _, _ = grouped_spies
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match="overrides require stock"):
        sparse.build(signature(variant=1, select=True))(select_io())
    assert calls == []


def test_split_unaligned_buffer_uses_same_source_scalar_variant(grouped_spies):
    calls, _, _, _ = grouped_spies
    io = select_io(aligned=False)
    assert sparse.build(signature(variant=1, select=True))(io) is None
    assert calls[0][1][-1] is False


def test_source_cache_requires_explicit_absolute_root(monkeypatch):
    from sgl_nvidia_mega.qwen3_8._jit import source_cache

    name = "SGLANG_QSA_SPLIT_TOPK_SRC_DIR"
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="absolute node-local"):
        source_cache(name)
    monkeypatch.setenv(name, "relative/source")
    with pytest.raises(ValueError, match="absolute node-local"):
        source_cache(name)
    # Resolves a name only: it creates no directory and cannot establish live
    # mount locality. The owning job's deployment guard must establish that.
    monkeypatch.setenv(name, "/tmp/q38-sparse-source-cpu")
    assert source_cache(name) == Path("/tmp/q38-sparse-source-cpu").resolve()
