# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103.gdn import core as C

H, HV, T = 1, 2, 2


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setattr(C, "_F32", {}, raising=False)
    monkeypatch.setattr(C, "_F32_IN_CAPTURE", set(), raising=False)


def _launch(monkeypatch, A_log, dt_bias):
    seen = {}
    monkeypatch.setattr(C, "_kernel", lambda *a: "kernel")
    monkeypatch.setattr(
        C.jit,
        "launch",
        lambda kernel, grid, params, stream, **kw: seen.update(p=params[0]),
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=0)
    )
    rows, K, V, R = T, C.K, C.V, C.RING
    ab = torch.zeros(rows, 2 * HV, dtype=torch.bfloat16)
    C.core(
        torch.zeros(rows, (2 * H + HV) * K, dtype=torch.bfloat16),
        torch.zeros(rows, HV, V, dtype=torch.bfloat16),
        ab[:, :HV],
        ab[:, HV:],
        A_log,
        dt_bias,
        torch.zeros(1, HV, V, K, dtype=torch.bfloat16),
        torch.zeros(1, HV, R, V, dtype=torch.bfloat16),
        torch.zeros(1, H, R, K, dtype=torch.bfloat16),
        torch.zeros(1, HV, R, dtype=torch.float32),
        None,
        None,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int8),
        torch.ones(V, dtype=torch.bfloat16),
        torch.ones(1, dtype=torch.float32),
        1e-6,
        torch.zeros(rows, HV * V, dtype=torch.float8_e4m3fn),
        rows_per_request=T,
        num_k_heads=H,
        min_blocks=1,
    )
    return seen["p"]


def _f32_at(ptr: int, n: int) -> list:
    """The ``n`` FP32 values the kernel would read at ``ptr``."""
    import ctypes

    return list((ctypes.c_float * n).from_address(ptr))


def test_a_bf16_dt_bias_reaches_the_kernel_as_its_exact_fp32_copy(monkeypatch):
    A_log = torch.tensor([-0.5, 0.25], dtype=torch.float32)
    dt_bias = torch.tensor([1.0078125, -3.25], dtype=torch.bfloat16)
    p = _launch(monkeypatch, A_log, dt_bias)
    # FP32 A_log is passed as it is; dt_bias as an FP32 copy of every head's value
    assert p.A_log == A_log.data_ptr()
    assert p.dt_bias != dt_bias.data_ptr()
    assert _f32_at(p.dt_bias, HV) == dt_bias.float().tolist()
    assert _f32_at(p.A_log, HV) == A_log.tolist()


def test_the_copies_made_at_build_are_the_ones_the_launch_reads(monkeypatch):
    A_log = torch.tensor([-0.5, 0.25], dtype=torch.bfloat16)
    dt_bias = torch.tensor([2.0, 0.5], dtype=torch.float16)
    C.prime_gate_params(A_log, dt_bias)
    primed = (C.gate_f32(A_log).data_ptr(), C.gate_f32(dt_bias).data_ptr())
    p = _launch(monkeypatch, A_log, dt_bias)
    assert (p.A_log, p.dt_bias) == primed
    # a weight update (a new version) makes a new copy
    with torch.no_grad():
        dt_bias.add_(1)
    copy = C.gate_f32(dt_bias)
    assert copy.data_ptr() != primed[1] and copy.tolist() == [3.0, 1.5]


def test_a_copy_first_made_inside_a_capture_is_not_kept(monkeypatch, caplog):
    monkeypatch.setattr(C, "_capturing", lambda t: True)
    dt_bias = torch.ones(HV, dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING):
        a, b = C.gate_f32(dt_bias), C.gate_f32(dt_bias)
    assert a is not b and C._F32 == {}
    assert (
        sum("first seen inside a capture" in r.getMessage() for r in caplog.records)
        == 1
    )


@pytest.mark.parametrize(
    "t",
    [torch.ones(HV, dtype=torch.int32), torch.ones(1, HV, dtype=torch.bfloat16)],
)
def test_a_gate_parameter_that_is_not_a_float_vector_raises(t):
    with pytest.raises(ValueError, match="a \\[HV\\] float tensor"):
        C.gate_f32(t)


def test_too_few_heads_raise(monkeypatch):
    with pytest.raises(ValueError, match="heads for 2"):
        _launch(
            monkeypatch,
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(HV, dtype=torch.bfloat16),
        )
