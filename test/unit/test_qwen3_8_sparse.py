# SPDX-License-Identifier: Apache-2.0
"""CPU ABI replays with original wrappers and raw-kernel spies, not GPU math."""

import ast
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
    NumericSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    SparseScoreIO,
    SparseSelectIO,
)

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import sparse

BOUNDARIES = (499, 512, 513, 4096, 4097, 8192, 8193)


def signature(operation=O.SPARSE_SCORE, capacity=8208, variant=0):
    s = NumericSignature(
        operation,
        4,
        8,
        1,
        128,
        512,
        4,
        D.BF16 if operation == O.SPARSE_SCORE else D.FP32,
        context_bound=capacity,
    )
    return replace(s, variant=variant) if variant else s


def score_io(length=8193, capacity=8208, request_dtype=torch.int32):
    return SparseScoreIO(
        query=torch.ones((4, 8, 128), dtype=torch.bfloat16),
        compressed_keys=torch.ones((1, 16, 1, 128), dtype=torch.bfloat16),
        compressed_slot_map=torch.zeros((4, (capacity + 15) // 16), dtype=torch.int32),
        compressed_lengths=torch.full((4,), length, dtype=torch.int32),
        row_requests=torch.full((4,), 7, dtype=request_dtype),
        query_weights=None,
        page_size=16,
        compress_ratio=4,
        scale=math.sqrt(128),
        scores_out=torch.full((4, capacity), float("nan")),
        scratch_out=torch.empty(0, dtype=torch.uint8),
    )


def select_io(length=8193, capacity=8208, tied=False):
    scores = (
        torch.ones((4, capacity))
        if tied
        else torch.arange(capacity).float().expand(4, -1).contiguous()
    )
    lengths = torch.full((4,), length, dtype=torch.int32)
    return SparseSelectIO(
        scores,
        lengths,
        512,
        torch.empty((4, 512), dtype=torch.int32),
        lengths.clamp(max=512),
        torch.empty(0, dtype=torch.uint8),
    )


def install(monkeypatch, name, **fields):
    module = ModuleType(name)
    module.__dict__.update(fields)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def raw_spies(monkeypatch):
    calls = []
    state = SimpleNamespace(capturing=False, fail=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )

    def score_factory(kind):
        def factory(**kwargs):
            calls.append((kind + "-prepare", kwargs))

            def launch(q, keys, table, lengths, requests, output, scale):
                calls.append(
                    (
                        kind + "-launch",
                        (q, keys, table, lengths, requests, output, scale),
                    )
                )
                if state.fail:
                    raise RuntimeError("spy eager failure")
                # Deterministic ABI marker only. This is not a scorer oracle.
                for row, length in enumerate(lengths.tolist()):
                    if length > 512:
                        output[row, :length] = torch.arange(length) + row

            return launch

        return factory

    def module_factory(items, zero_start):
        calls.append(("select-prepare", (items, zero_start)))

        def select_rows(scores, lengths, starts, output):
            calls.append(("select-launch", (scores, lengths, starts, output)))
            if state.fail:
                raise RuntimeError("spy eager failure")
            # A deterministic CPU spy shared with the original wrapper. Equality
            # proves buffer/argument/order preservation, not CUDA tie correctness.
            for row, length in enumerate(lengths.tolist()):
                output[row].fill_(-1)
                if length <= 512:
                    ids = torch.arange(length)
                else:
                    ids = torch.argsort(
                        scores[row, :length], descending=True, stable=True
                    )[:512]
                output[row, : len(ids)] = ids

        return SimpleNamespace(select_rows=select_rows)

    prefix = "sgl_nvidia_mega.qwen3_8.kernels.ops.attention."
    install(monkeypatch, prefix + "q38_mk_score", _mk_score_kernel=score_factory("mk"))
    install(
        monkeypatch,
        prefix + "q38_perf_score_short",
        _perf_score_kernel=score_factory("short"),
    )
    install(monkeypatch, prefix + "q38_mk_topk", _module=module_factory)
    return calls, state, score_factory, module_factory


def original_wrapper(name, namespace):
    root = Path(sparse.__file__).parents[1] / "kernels/ops/attention"
    path = root / ("q38_mk_score.py" if name == "score_mk" else "q38_mk_topk.py")
    tree = ast.parse(path.read_text())
    body = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(ast.Module(body=[body], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


@pytest.mark.parametrize("capacity", (499, 512, 513, 4096, 4097, 8192, 8193, 131072))
@pytest.mark.parametrize("operation", (O.SPARSE_SCORE, O.SPARSE_SELECT))
def test_finite_exact_capability(operation, capacity):
    s = signature(operation, capacity)
    assert sparse.supported(s) and providers.supported(s)
    for change in (
        {"rows": 8},
        {"query_heads": 4},
        {"kv_heads": 2},
        {"head_dim": 256},
        {"selected_count": 511},
        {"compress_ratio": 2},
        {"rotary_dim": 64},
        {"context_bound": 0},
        {"context_bound": 131073},
        {"scalar_type": D.FP16},
    ):
        assert not sparse.supported(replace(s, **change))
        assert providers.make_offer(replace(s, **change)) is None


@pytest.mark.parametrize("variant", (0, 1, 2, 3, 4, 5, -1))
@pytest.mark.parametrize("operation", (O.SPARSE_SCORE, O.SPARSE_SELECT))
def test_source_variants_are_exact(operation, variant):
    s = signature(operation, variant=variant)
    expected = variant in ((0, 3, 4, 5) if operation == O.SPARSE_SCORE else (0, 1))
    assert sparse.supported(s) == expected
    assert (providers.make_offer(s) is not None) == expected


def test_short32_calls_original_short_factory(raw_spies):
    calls, _, _, _ = raw_spies
    run = sparse.build(signature(variant=3))
    io = score_io(513)
    assert run(io) is None
    assert [name for name, _ in calls] == ["short-prepare", "short-launch"]
    assert calls[0][1] == dict(
        min_scored_len=512, request_dtype="int32", long_ctas=1024
    )


@pytest.mark.parametrize("length", BOUNDARIES)
@pytest.mark.parametrize("request_dtype", (torch.int32, torch.int64))
def test_score_wrapper_elision_keeps_raw_args_and_scored_domain(
    raw_spies, length, request_dtype
):
    calls, _, factory, _ = raw_spies
    io = score_io(length, request_dtype=request_dtype)
    ref = original_wrapper(
        "score_mk",
        dict(
            torch=torch,
            math=math,
            _SHORT_Q4=False,
            _mk_score_kernel=factory("reference"),
        ),
    )
    expected = ref(
        io.query,
        io.compressed_keys,
        io.compressed_slot_map,
        io.compressed_lengths,
        8208,
        io.row_requests,
        min_scored_len=512,
    )
    run = sparse.build(signature())
    assert run(io) is None
    args = calls[-1][1]
    assert all(
        actual is wanted
        for actual, wanted in zip(
            args[:6],
            (
                io.query,
                io.compressed_keys,
                io.compressed_slot_map,
                io.compressed_lengths,
                io.row_requests,
                io.scores_out,
            ),
        )
    )
    assert args[6] == math.sqrt(128)
    if length > 512:
        assert torch.equal(expected[:, :length], io.scores_out[:, :length])
    assert torch.isnan(
        io.scores_out[:, length:] if length > 512 else io.scores_out
    ).all()
    assert calls[-2] == (
        "mk-prepare",
        dict(
            min_scored_len=512,
            request_dtype="int32" if request_dtype == torch.int32 else "int64",
            long_ctas=1024,
        ),
    )


@pytest.mark.parametrize("length", BOUNDARIES)
@pytest.mark.parametrize("pattern", ("unique", "ties", "nan", "signed_zero"))
def test_select_wrapper_elision_preserves_ordered_raw_output_and_counts(
    raw_spies, length, pattern
):
    calls, _, _, factory = raw_spies
    io = select_io(length, tied=pattern != "unique")
    if pattern == "nan":
        io.scores[:, ::7] = float("nan")
    if pattern == "signed_zero":
        io.scores.zero_()
        io.scores[:, ::2] = -0.0
    counts = io.counts_out.clone()
    ref = original_wrapper("topk_mk", dict(torch=torch, _module=factory))
    expected = ref(io.scores, io.lengths)
    run = sparse.build(signature(O.SPARSE_SELECT))
    assert run(io) is None
    assert calls[-1] == (
        "select-launch",
        (io.scores, io.lengths, None, io.selected_indices_out),
    )
    assert torch.equal(expected, io.selected_indices_out)
    assert torch.equal(counts, io.counts_out)
    assert calls[-2] == ("select-prepare", (3, True))


@pytest.mark.parametrize("operation", (O.SPARSE_SCORE, O.SPARSE_SELECT))
def test_capture_cold_miss_then_eager_and_replay(raw_spies, operation):
    calls, state, _, _ = raw_spies
    run = sparse.build(signature(operation))
    io = score_io() if operation == O.SPARSE_SCORE else select_io()
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        run(io)
    assert calls == []
    state.capturing = False
    run(io)
    prepared = sum(name.endswith("prepare") for name, _ in calls)
    state.capturing = True
    run(io)
    assert sum(name.endswith("prepare") for name, _ in calls) == prepared == 1


@pytest.mark.parametrize("operation", (O.SPARSE_SCORE, O.SPARSE_SELECT))
def test_failed_eager_does_not_mark_layout_warm(raw_spies, operation):
    _, state, _, _ = raw_spies
    run = sparse.build(signature(operation))
    io = score_io() if operation == O.SPARSE_SCORE else select_io()
    state.fail = True
    with pytest.raises(RuntimeError, match="spy eager failure"):
        run(io)
    assert not run._resources
    state.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        run(io)


@pytest.mark.parametrize(
    "change",
    (
        {"query": torch.empty((4, 4, 128), dtype=torch.bfloat16)},
        {"query": torch.empty((4, 8, 128), dtype=torch.float32)},
        {"compressed_lengths": torch.zeros(4, dtype=torch.int64)},
        {"row_requests": torch.zeros(4, dtype=torch.int16)},
        {"query_weights": torch.ones(4)},
        {"page_size": 32},
        {"scores_out": torch.empty((4, 8192))},
    ),
)
def test_wrong_score_view_fails_before_raw_prepare(raw_spies, change):
    calls, _, _, _ = raw_spies
    with pytest.raises(ValueError, match="prepared capability"):
        sparse.build(signature())(replace(score_io(), **change))
    assert calls == []


def test_register_and_prepare_are_finite_and_idempotent(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia.binding import PreparationView

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    s = signature()
    selected = signature(O.SPARSE_SELECT)
    built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=()))
    assert built.offers == ()
    view = PreparationView(numeric_signatures=(s, selected))
    SPEC.prepare(built, view)
    assert len(built.offers) == 2
    SPEC.prepare(built, view)
    assert len(built.offers) == 2 and registry.get(s) is providers.make_offer(s)
    with pytest.raises(TypeError, match="tuple"):
        SPEC.prepare(built, SimpleNamespace(numeric_signatures=[s]))


def test_prepare_rejects_capture_before_registering(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia.binding import PreparationView

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=()))
    view = PreparationView(numeric_signatures=(signature(),))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before capture"):
        SPEC.prepare(built, view)
    assert not built.offers and registry.get(signature()) is None


def prefill_signature(**kw):
    return replace(
        NumericSignature(
            O.PREFILL_SELECT, 0, 4, 1, 128, 512, 4, D.BF16, context_bound=8208
        ),
        **kw,
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(rows=33),
        dict(query_heads=8),
        dict(context_bound=0),
        dict(context_bound=131073),
        dict(scalar_type=D.FP8_E4M3),
        dict(variant=2),
        dict(variant=32),
        dict(variant=256 << 8),
        dict(variant=96 << 20),
        dict(variant=1 << 31),
        dict(ctas_per_group=128),
    ],
)
def test_prefill_select_declaration_is_row_independent_and_exact(kw):
    from sgl_nvidia_mega.qwen3_8._providers import prefill_select

    assert providers.supported(prefill_signature())
    # Bits 0-3: the top-k (1 one-pass); bit 4: the tensor-core scorer; bits 8-19 and
    # 20-30: a wide logits budget (MiB) and the width it starts at (256-key units).
    for variant in (1, 16, 17, 256 << 8 | 96 << 20, 17 | 256 << 8 | 96 << 20):
        assert providers.supported(prefill_signature(variant=variant))
    assert providers.make_offer(prefill_signature(**kw)) is None
    assert not prefill_select.supported(prefill_signature(**kw))


def prefill_io(rows=33, width=8193, sms=148, chunk=None):
    from sglang.kernels.plugin_kernels.nvidia.numeric import PrefillSelectIO

    class Cuda(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    chunk = rows if chunk is None else chunk
    padded = (chunk + 31) // 32 * 32

    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype)

    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full((rows,), width, dtype=torch.int32)
    return PrefillSelectIO(
        empty((rows, 4, 128), torch.bfloat16).as_subclass(Cuda),
        empty((2 * width, 1, 128), torch.bfloat16),
        starts,
        ends,
        starts.clone(),
        ends.clone(),
        empty((rows, 2051)),
        empty((padded, 4, 128), torch.bfloat16),
        empty(padded),
        empty(padded),
        ends - starts,
        torch.zeros(chunk, dtype=torch.int32),
        empty((rows, 512)),
        empty((padded, width), torch.float32),
        empty(4096),
        chunk,
        sms,
    )


@pytest.fixture
def prefill_raw(monkeypatch):
    import sgl_nvidia_mega.qwen3_8.qsa as package

    raw = ModuleType(package.__name__ + ".prefill_select")
    calls = []
    raw.plan_chunk_rows = lambda rows, width, slots, budget: (
        calls.append(("plan", rows, width, slots, budget)) or rows
    )
    raw._topk_impl_for = lambda width, impl: "stock" if width <= 4096 else impl
    raw.select_into = lambda io, width, **kw: calls.append(("select", io, width, kw))
    monkeypatch.setitem(sys.modules, raw.__name__, raw)
    monkeypatch.setattr(package, "prefill_select", raw, raising=False)
    capturing = SimpleNamespace(value=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing.value
    )
    return calls, capturing


def test_prefill_select_launches_with_call_width_and_capture_warmup(prefill_raw):
    from sgl_nvidia_mega.qwen3_8._providers import prefill_select

    calls, capturing = prefill_raw
    run = prefill_select.build(prefill_signature())
    io = prefill_io()
    capturing.value = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(io)
    capturing.value = False
    assert run(io) is None
    capturing.value = True
    assert run(io) is None
    narrow = prefill_io(width=4000)
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(narrow)
    assert ("plan", 33, 8193, 296, 128 * 1024 * 1024) in calls
    assert [c for c in calls if c[0] == "select"] == [
        ("select", io, 8193, dict(scorer_tc=False, topk_impl="sampled"))
    ] * 2


@pytest.mark.parametrize(
    "variant,width,budget",
    [
        (0, 30000, 128),
        (256 << 8 | 96 << 20, 24575, 128),
        (256 << 8 | 96 << 20, 24576, 256),
        (1 | 64 << 8 | 4 << 20, 8193, 64),
    ],
)
def test_prefill_select_checks_the_caller_plan_against_the_declared_budget(
    prefill_raw, variant, width, budget
):
    from sgl_nvidia_mega.qwen3_8._providers import prefill_select

    calls, _ = prefill_raw
    signature = prefill_signature(variant=variant, context_bound=32768)
    prefill_select.build(signature)(prefill_io(width=width))
    assert ("plan", 33, width, 296, budget << 20) in calls


@pytest.mark.parametrize(
    "variant,options",
    [
        (16, dict(scorer_tc=True, topk_impl="sampled")),
        (1, dict(scorer_tc=False, topk_impl="onepass")),
        (17, dict(scorer_tc=True, topk_impl="onepass")),
    ],
)
def test_prefill_select_passes_the_declared_options_to_the_launch(
    prefill_raw, variant, options
):
    from sgl_nvidia_mega.qwen3_8._providers import prefill_select

    calls, _ = prefill_raw
    io = prefill_io()
    assert prefill_select.build(prefill_signature(variant=variant))(io) is None
    assert calls[-1] == ("select", io, 8193, options)


@pytest.mark.parametrize(
    "bad",
    [
        dict(width=8209),
        dict(chunk=32),
        dict(field="row_starts", value=torch.zeros(33, dtype=torch.int64)),
        dict(field="logits_out", value=torch.empty((32, 8193))),
        dict(field="split_tag", value=torch.empty(2048, dtype=torch.int32)),
        dict(field="selected_indices_out", value=torch.empty((33, 2048))),
    ],
)
def test_prefill_select_refuses_incomplete_workspace_before_launch(prefill_raw, bad):
    from sgl_nvidia_mega.qwen3_8._providers import prefill_select

    calls, _ = prefill_raw
    if "field" in bad:
        io = replace(prefill_io(), **{bad["field"]: bad["value"]})
    else:
        io = prefill_io(**bad)
    with pytest.raises(ValueError):
        prefill_select.build(prefill_signature())(io)
    assert not [c for c in calls if c[0] == "select"]


def raw_functions(names, scope):
    path = Path(providers.__file__).parent / "qsa" / "prefill_select.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_prefill_select_source_chunk_and_split_plans():
    scope = raw_functions(("plan_chunk_rows", "plan_key_splits"), {"_BLOCK_Q": 32})
    plan = scope["plan_chunk_rows"]
    assert plan(17, 8192, 424, 128 * 1024 * 1024) == 17
    n = plan(20000, 32768, 424, 128 * 1024 * 1024)
    assert n % 416 == 0 and n * 32768 * 4 <= 128 * 1024 * 1024
    assert scope["plan_key_splits"](2, 65, 128, 212) == 1
    assert scope["plan_key_splits"](1, 8192, 128, 212) == 64


@pytest.mark.parametrize("rows", [31, 32, 33, 65])
def test_prefill_select_raw_chunks_pad_from_caller_scratch(monkeypatch, rows):
    import sgl_nvidia_mega.qwen3_8.qsa as package

    fake = ModuleType(package.__name__ + ".kernel")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(package, "kernel", fake, raising=False)
    io = prefill_io(rows, width=16, chunk=32)
    q = (
        torch.arange(rows, dtype=torch.bfloat16)[:, None, None]
        .expand(rows, 4, 128)
        .contiguous()
    )
    starts = torch.arange(rows, dtype=torch.int32)
    io = replace(io, query=q, row_starts=starts, row_ends=starts + 16)
    calls = []

    def score(q, keys, logits, s, e, width, config, **kw):
        calls.append((q.clone(), s.clone(), e.clone()))
        logits[:, 0].copy_(q[:, 0, 0])

    def topk(logits, zeros, output, lengths):
        assert torch.all(zeros == 0)
        output.copy_(logits[:, 0, None].int().expand_as(output))

    def expand(kernel, blocks, positions, lengths, output, ratio, topk):
        output.copy_(blocks[:, 0, None].expand_as(output))

    scope = raw_functions(
        ("select_into",),
        dict(
            __package__=package.__name__,
            CONFIG=(128, 512, 2, "Square", 1),
            score_chunk=score,
            _topk_launcher=lambda impl: topk,
            _topk_impl_for=lambda *a: "sampled",
            _expand_into=expand,
        ),
    )
    before = q.clone()
    scope["select_into"](io, 16)
    assert torch.equal(q, before)
    assert torch.equal(io.selected_indices_out[:, 0], torch.arange(rows))
    assert len(calls) == (rows + 31) // 32
    tail = rows % 32
    if tail:
        last, s, e = calls[-1]
        assert torch.count_nonzero(last[tail:]) == 0
        assert torch.all(s[tail:] == starts[-1]) and torch.all(
            e[tail:] == starts[-1] + 16
        )


@pytest.mark.parametrize("covered", [False, True])
def test_prefill_select_raw_scores_with_the_tensor_core_scorer(monkeypatch, covered):
    import sgl_nvidia_mega.qwen3_8.qsa as package

    fake = ModuleType(package.__name__ + ".kernel")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(package, "kernel", fake, raising=False)
    io = prefill_io(64, width=16, chunk=32)
    stock, tc = [], []

    def score_tc(q, keys, logits, starts, ends, width, sms):
        tc.append((q.shape[0], logits.shape[0], width, sms))
        return covered

    scope = raw_functions(
        ("select_into",),
        dict(
            __package__=package.__name__,
            CONFIG=(128, 512, 2, "Square", 1),
            score_chunk=lambda q, *a, **kw: stock.append(q.shape[0]),
            _score_tc=score_tc,
            _topk_launcher=lambda impl: lambda *a: None,
            _topk_impl_for=lambda *a: "sampled",
            _expand_into=lambda *a: None,
        ),
    )
    scope["select_into"](io, 16)
    assert stock == [32, 32] and tc == []
    scope["select_into"](io, 16, scorer_tc=True)
    # A chunk the tensor-core scorer does not cover keeps the source scorer.
    assert tc == [(32, 32, 16, 148)] * 2
    assert stock == [32, 32] + ([] if covered else [32, 32])


@pytest.mark.parametrize("impl", ["sampled", "onepass"])
def test_prefill_select_raw_one_pass_flags_use_the_starts_scratch(monkeypatch, impl):
    import sgl_nvidia_mega.qwen3_8.qsa as package

    fake = ModuleType(package.__name__ + ".kernel")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(package, "kernel", fake, raising=False)
    io = prefill_io(40, width=16, chunk=32)
    launched = []

    def topk(logits, zeros, output, lengths, *flags):
        launched.append((impl, logits.shape[0], [f.data_ptr() for f in flags]))

    scope = raw_functions(
        ("select_into",),
        dict(
            __package__=package.__name__,
            CONFIG=(128, 512, 2, "Square", 1),
            score_chunk=lambda *a, **kw: None,
            _topk_launcher=lambda name: topk,
            _topk_impl_for=lambda width, name: name,
            _expand_into=lambda *a: None,
        ),
    )
    scope["select_into"](io, 16, topk_impl=impl)
    scratch = [io.starts_scratch_out.data_ptr()] if impl == "onepass" else []
    assert launched == [(impl, 32, scratch), (impl, 8, scratch)]
