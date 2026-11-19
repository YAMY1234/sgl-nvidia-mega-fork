# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import gemm_tactics, lowm, runtime, topology

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
N, K = gemm_tactics.SHAPE_NK


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(gemm_tactics, "_TACTIC", {})
    monkeypatch.setattr(gemm_tactics, "_WARNED", set())
    monkeypatch.setattr(
        gemm_tactics, "cublaslt_version", lambda: gemm_tactics.CUBLASLT_VERSION
    )
    gemm_tactics._table.cache_clear()
    yield
    gemm_tactics._table.cache_clear()


@pytest.fixture
def measured(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV, "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(runtime, "MEASURED_KEY", ("tep8", topology.GB300))


def test_the_table_covers_each_verify_width_through_448_rows_once():
    entries = gemm_tactics.ENTRIES
    assert all(a.hi < b.lo for a, b in zip(entries, entries[1:]))
    assert all(e.lo <= e.hi for e in entries)
    for rows in range(7, 449, 7):
        assert sum(e.lo <= rows <= e.hi for e in entries) == 1, rows
    assert all(e.config is None or len(e.config) == 9 for e in entries)
    assert set(gemm_tactics.TABLES) == {
        (name, topology.GB300, gemm_tactics.CUBLASLT_VERSION)
        for name in ("tp8", "tep8")
    }


@pytest.mark.parametrize(
    "switch, slots, key, on",
    [
        pytest.param("1", None, ("tep8", topology.GB300), True, id="tep8"),
        pytest.param("1", None, ("tp8", topology.GB300), True, id="tp8"),
        pytest.param(None, None, ("tep8", topology.GB300), False, id="mk-alone"),
        pytest.param("1", "hop", ("tep8", topology.GB300), False, id="slot-list"),
        pytest.param("1", None, ("tp4", topology.GB300), False, id="tp4"),
        pytest.param("1", None, ("dep8", topology.GB300), False, id="dep8"),
        pytest.param("1", None, ("tep8", topology.Sku(10, 3, 148)), False, id="b300"),
        pytest.param("1", None, None, False, id="no-key"),
    ],
)
def test_the_table_is_looked_up_only_where_the_measured_option_holds(
    monkeypatch, switch, slots, key, on
):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "MEASURED_KEY", key)
    got = gemm_tactics.entry(112, N, K)
    assert (got is not None) is on
    assert gemm_tactics.enabled() is on
    if on:
        assert got.config == gemm_tactics._64x64_2x2
        # only the pinned shape
        assert gemm_tactics.entry(112, N, 1024) is None
        assert gemm_tactics.entry(112, 16, K) is None
        assert gemm_tactics.entry(29, N, K) is None


def test_another_cublaslt_version_runs_flashinfers_bmm_with_one_warning(
    measured, monkeypatch, caplog
):
    monkeypatch.setattr(gemm_tactics, "cublaslt_version", lambda: 120000)
    with caplog.at_level("WARNING"):
        assert gemm_tactics.entry(112, N, K) is None
        assert gemm_tactics.entry(84, N, K) is None
    assert not gemm_tactics.enabled()
    assert caplog.text.count("option gemm_tactics FALLBACK") == 1
    assert "cuBLASLt 130101, this process runs 120000" in caplog.text


def _fake_runner(monkeypatch, listed, bitwise=True):
    ran = []
    monkeypatch.setattr(gemm_tactics, "algorithms", lambda *a: list(listed))
    monkeypatch.setattr(gemm_tactics, "_bitwise", lambda *a: bitwise)
    monkeypatch.setattr(
        gemm_tactics, "run_tactic", lambda x8, w, xs, ws, out, t: ran.append(t) or out
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return ran


def _call(rows):
    x8 = torch.zeros((rows, K), dtype=torch.float8_e4m3fn)
    w = torch.zeros((K, N), dtype=torch.float8_e4m3fn)
    one = torch.ones(1)
    return gemm_tactics.pinned(x8, w, one, one, torch.bfloat16)


def test_a_width_runs_its_listed_algorithm_once_resolved(measured, monkeypatch, caplog):
    other = gemm_tactics._64x8_4x1
    ran = _fake_runner(monkeypatch, [other, None, gemm_tactics._64x64_2x2])
    with caplog.at_level("INFO"):
        assert _call(112).shape == (112, N)
        _call(112)
    assert ran == [2, 2]
    assert "cuBLASLt 64x64 2x2 2-CTA (tactic 2) at 112 rows" in caplog.text
    # a heuristic entry never lists the algorithms
    monkeypatch.setattr(gemm_tactics, "algorithms", None)
    _call(56)
    assert ran[-1] == -1
    # no entry: the caller runs FlashInfer's BMM
    assert _call(1024) is None


@pytest.mark.parametrize("why", ["unlisted", "check", "capture"])
def test_a_width_falls_back_to_the_heuristic_loudly(measured, monkeypatch, caplog, why):
    listed = [] if why == "unlisted" else [gemm_tactics._64x64_2x2]
    ran = _fake_runner(monkeypatch, listed, bitwise=why != "check")
    if why == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with caplog.at_level("WARNING"):
        _call(112)
        _call(112)
    assert ran == [-1, -1]
    assert caplog.text.count("option gemm_tactics FALLBACK") == 1


@pytest.mark.parametrize("missing", ["_get_algos", "forward(tactic)"])
def test_a_flashinfer_runner_without_a_read_name_runs_flashinfers_bmm_loudly(
    measured, monkeypatch, caplog, missing
):
    from sglang.srt.layers.quantization import fp8_utils

    assert missing in gemm_tactics.FLASHINFER_READS["flashinfer/gemm/gemm_base.py"]
    ran = []

    class Runner:
        def _get_algos(self, inputs):
            if missing == "_get_algos":
                raise AttributeError("'Runner' object has no attribute '_get_algos'")
            return None, 1

        def forward(self, inputs, **kwargs):
            if missing == "forward(tactic)":
                raise TypeError("forward() got an unexpected keyword argument 'tactic'")
            ran.append(kwargs["tactic"])

    monkeypatch.setattr(gemm_tactics, "_runner", Runner)
    monkeypatch.setattr(gemm_tactics, "_inputs", lambda *a: [])
    monkeypatch.setattr(gemm_tactics, "config", lambda buf, i: gemm_tactics._64x64_2x2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        fp8_utils, "flashinfer_bmm_fp8", lambda *a: "flashinfer", raising=False
    )
    w = torch.zeros((K, N), dtype=torch.float8_e4m3fn)
    one = torch.ones(1)
    got = []
    with caplog.at_level("WARNING"):
        for rows in (112, 56, 112, 56):
            x8 = torch.zeros((rows, K), dtype=torch.float8_e4m3fn)
            out = gemm_tactics.bmm(x8, w, one, one, torch.bfloat16)
            got.append(out if isinstance(out, str) else "table")
    assert caplog.text.count("option gemm_tactics FALLBACK") == 1
    assert "does not match FLASHINFER_READS" in caplog.text
    if missing == "_get_algos":
        # the heuristic width never lists the algorithms
        assert got == ["flashinfer", "table"] * 2 and ran == [-1, -1]
    else:
        assert got == ["flashinfer"] * 4 and ran == []


def test_an_uncovered_call_runs_flashinfers_bmm(monkeypatch):
    from sglang.srt.layers.quantization import fp8_utils

    monkeypatch.delenv(RUNTIME_ENV, raising=False)
    # SGLang defines it only where FlashInfer imports
    monkeypatch.setattr(
        fp8_utils, "flashinfer_bmm_fp8", lambda *a: "flashinfer", raising=False
    )
    x8 = torch.zeros((112, K), dtype=torch.float8_e4m3fn)
    w = torch.zeros((K, N), dtype=torch.float8_e4m3fn)
    assert gemm_tactics.bmm(x8, w, torch.ones(1), torch.ones(1), torch.bfloat16) == (
        "flashinfer"
    )


def test_the_in_step_projection_takes_the_table_only_under_the_option(monkeypatch):
    # no option: nothing of the layer is read
    monkeypatch.delenv(RUNTIME_ENV, raising=False)
    assert lowm.in_proj_pinned(object(), torch.zeros(112, K)) is None


@pytest.mark.parametrize("capture", [False, True])
def test_the_in_step_projection_quantizes_then_runs_the_tables_bmm(
    measured, monkeypatch, capture
):
    from sglang.kernels.ops.quantization import fp8_kernel
    from sglang.srt.models import qwen3_5

    order = []
    x = torch.zeros((112, K), dtype=torch.bfloat16)
    x8 = torch.zeros((112, K), dtype=torch.float8_e4m3fn)
    lin = SimpleNamespace(
        weight=torch.zeros((K, N), dtype=torch.float8_e4m3fn),
        input_scale=torch.ones(1),
        weight_scale=torch.ones(1),
    )
    monkeypatch.setattr(lowm, "fp8_unsupported", lambda _: "")
    monkeypatch.setattr(lowm, "_w8", lambda _: torch.zeros(N, K))
    monkeypatch.setattr(
        fp8_kernel,
        "static_quant_fp8",
        lambda t, s, repeat_scale: order.append("quant") or (x8, s),
    )
    monkeypatch.setattr(
        gemm_tactics,
        "bmm",
        lambda a, *rest: order.append("bmm") or (a is x8 and "qkvz"),
    )
    monkeypatch.setattr(qwen3_5, "_gdn_use_alt_stream", True)
    monkeypatch.setattr(qwen3_5, "get_is_capture_mode", lambda: capture)
    alt = SimpleNamespace(wait_stream=lambda s: order.append("fork"))
    current = SimpleNamespace(wait_stream=lambda s: order.append("join"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: _Ctx(order, s is alt))
    gdn = SimpleNamespace(
        in_proj_qkvz=lin,
        in_proj_ba=lambda t: (order.append("ba") or "ba", None),
        alt_stream=alt,
    )
    assert lowm.in_proj_pinned(gdn, x) == ("qkvz", "ba")
    if capture:
        assert order == ["fork", "quant", "bmm", "alt", "ba", "join"]
    else:
        assert order == ["quant", "bmm", "ba"]


class _Ctx:
    def __init__(self, order, alt):
        self.order, self.alt = order, alt

    def __enter__(self):
        if self.alt:
            self.order.append("alt")

    def __exit__(self, *exc):
        return False


def test_slot_lowm_ba_takes_the_in_step_path_without_the_table(monkeypatch):
    from sglang.kernels.ops.quantization import fp8_kernel
    from sglang.srt.models import qwen3_5

    monkeypatch.delenv(RUNTIME_ENV, raising=False)
    x = torch.zeros((112, K), dtype=torch.bfloat16)
    x8 = torch.zeros((112, K), dtype=torch.float8_e4m3fn)
    lin = SimpleNamespace(
        weight=torch.zeros((K, N), dtype=torch.float8_e4m3fn),
        input_scale=torch.ones(1),
        weight_scale=torch.ones(1),
    )
    monkeypatch.setattr(lowm, "fp8_unsupported", lambda _: "")
    monkeypatch.setattr(lowm, "_w8", lambda _: torch.zeros(N, K))
    monkeypatch.setattr(
        fp8_kernel, "static_quant_fp8", lambda t, s, repeat_scale: (x8, s)
    )
    monkeypatch.setattr(gemm_tactics, "pinned", lambda *a: None)
    from sglang.srt.layers.quantization import fp8_utils

    monkeypatch.setattr(
        fp8_utils, "flashinfer_bmm_fp8", lambda *a: "flashinfer", raising=False
    )
    monkeypatch.setattr(qwen3_5, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace())
    monkeypatch.setattr(lowm, "in_proj_ba", lambda g, t, tactic: ("low-M", tactic))
    gdn = SimpleNamespace(in_proj_qkvz=lin, alt_stream=None)
    assert lowm.in_proj_pinned(gdn, x) is None
    assert lowm.in_proj_pinned(gdn, x, "t") == ("flashinfer", ("low-M", "t"))


@pytest.mark.parametrize("alt", [False, True])
def test_the_e4m3_projection_runs_the_tables_bmm_and_in_proj_bas_tactic(
    monkeypatch, alt
):
    order = []
    x = torch.zeros((98, K), dtype=torch.bfloat16)
    x8 = torch.zeros((98, K), dtype=torch.float8_e4m3fn)
    lin = SimpleNamespace(
        weight=None, input_scale=torch.ones(1), weight_scale=torch.ones(1)
    )
    monkeypatch.setattr(
        gemm_tactics,
        "bmm",
        lambda a, *rest: order.append("bmm") or (a is x8 and "qkvz"),
    )
    monkeypatch.setattr(
        lowm,
        "in_proj_ba",
        lambda g, t, tactic: order.append("ba") or ("low-M" if tactic else "sglang"),
    )
    stream = SimpleNamespace(wait_stream=lambda s: order.append("wait"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: _Ctx(order, True))
    gdn = SimpleNamespace(in_proj_qkvz=lin, alt_stream=stream if alt else None)
    assert lowm.in_proj_fp8(gdn, x, x8, "t") == ("qkvz", "low-M")
    assert lowm.in_proj_fp8(gdn, x, x8) == ("qkvz", "sglang")
    one = ["wait", "bmm", "alt", "ba", "wait"] if alt else ["bmm", "ba"]
    assert order == one * 2
