# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
import logging

import pytest
import torch

HIDDEN, N = 4096, 2560
WIDTHS = (*range(7, 449, 7),)
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


def _measured_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    from sgl_nvidia_mega.qwen3_5 import gemm_tactics, topology

    return (
        topology.device_sku() == topology.GB300
        and gemm_tactics.cublaslt_version() == gemm_tactics.CUBLASLT_VERSION
    )


pytestmark = pytest.mark.skipif(
    not _measured_gpu(), reason="needs GB300 with the table's cuBLASLt version"
)


class _Fp8Linear:
    def __init__(self, n, k, g):
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(
            torch.float8_e4m3fn
        )
        self.weight = w.t()  # [K, N] column-major, as ModelOpt holds it
        self.input_scale = torch.tensor([0.02], device="cuda")
        self.weight_scale = torch.tensor([0.002], device="cuda")


def _x8(rows, g):
    return (torch.randn(rows, HIDDEN, device="cuda", generator=g) * 40).to(
        torch.float8_e4m3fn
    )


def _bits(t):
    return t.view(torch.int16)


@pytest.fixture
def measured(monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import gemm_tactics, runtime, topology

    monkeypatch.setenv(RUNTIME_ENV, "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(runtime, "MEASURED_KEY", ("tep8", topology.GB300))
    monkeypatch.setattr(gemm_tactics, "_TACTIC", {})
    monkeypatch.setattr(gemm_tactics, "_WARNED", set())


@pytest.fixture
def no_autotuner(monkeypatch):
    from flashinfer.autotuner import AutoTuner

    def asked(*a, **k):
        raise AssertionError("the autotuner was asked")

    monkeypatch.setattr(AutoTuner, "choose_one", asked)


@pytest.mark.parametrize("rows", WIDTHS)
def test_each_covered_width_runs_its_entry_with_the_heuristics_bits(
    measured, no_autotuner, caplog, rows
):
    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    e = gemm_tactics.entry(rows, N, HIDDEN)
    if e is None:
        pytest.skip(f"{rows} rows: no entry")
    g = torch.Generator(device="cuda").manual_seed(rows)
    lin = _Fp8Linear(N, HIDDEN, g)
    x8 = _x8(rows, g)
    args = (x8, lin.weight, lin.input_scale, lin.weight_scale, torch.bfloat16)
    with caplog.at_level(logging.INFO, logger=gemm_tactics.__name__):
        a = gemm_tactics.bmm(*args)
        b = gemm_tactics.bmm(*args)
    out = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
    ref = gemm_tactics.run_tactic(*args[:4], out, -1)
    assert torch.equal(_bits(a), _bits(ref)) and torch.equal(_bits(a), _bits(b))
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "FALLBACK" in m], msgs
    assert len([m for m in msgs if f"at {rows} rows" in m]) == 1, msgs


@pytest.mark.parametrize("rows", [7, 84, 112, 448])
def test_a_resolved_width_replays_from_a_cuda_graph(measured, rows):
    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    lin = _Fp8Linear(N, HIDDEN, g)
    x8 = _x8(rows, g)
    args = (lin.weight, lin.input_scale, lin.weight_scale, torch.bfloat16)
    gemm_tactics.bmm(x8, *args)  # resolves outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = gemm_tactics.bmm(x8, *args)
    for _ in range(3):
        x8.copy_(_x8(rows, g))
        graph.replay()
        torch.cuda.synchronize()
        out = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
        want = gemm_tactics.run_tactic(x8, *args[:3], out, -1)
        assert torch.equal(_bits(got), _bits(want))


def test_outside_the_table_or_the_option_it_is_flashinfers_bmm(measured, monkeypatch):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    g = torch.Generator(device="cuda").manual_seed(5)
    lin = _Fp8Linear(N, HIDDEN, g)
    args = (lin.weight, lin.input_scale, lin.weight_scale, torch.bfloat16)
    for rows in (16, 32, 449):
        assert gemm_tactics.pinned(_x8(rows, g), *args) is None
    monkeypatch.delenv(RUNTIME_ENV)
    x8 = _x8(112, g)
    assert gemm_tactics.pinned(x8, *args) is None
    got = gemm_tactics.bmm(x8, *args)
    assert torch.equal(_bits(got), _bits(flashinfer_bmm_fp8(x8, *args)))


def test_flashinfer_provides_what_the_module_reads():
    import flashinfer.gemm.gemm_base as base
    import flashinfer.utils as utils

    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    runner = base.get_gemm_module().cublas_fp8_gemm_runner()
    found = {
        "flashinfer/gemm/gemm_base.py": {
            "get_gemm_module": callable(getattr(base, "get_gemm_module", None)),
            "cublas_fp8_gemm_runner": True,
            "_get_algos": callable(getattr(runner, "_get_algos", None)),
            "forward(tactic)": "tactic" in inspect.signature(runner.forward).parameters,
            "DEFAULT_WORKSPACE_SIZE": isinstance(
                getattr(base, "DEFAULT_WORKSPACE_SIZE", None), int
            ),
        },
        "flashinfer/utils.py": {
            "_get_cache_buf": callable(getattr(utils, "_get_cache_buf", None)),
        },
    }
    assert {f: tuple(n) for f, n in found.items()} == gemm_tactics.FLASHINFER_READS
    missing = [n for names in found.values() for n, ok in names.items() if not ok]
    assert not missing, missing
