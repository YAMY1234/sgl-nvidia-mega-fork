"""CPU checks of the Q35FLY MKA input contract; no device initialization."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.attention import mka
from sgl_nvidia_mega.qwen3_5.sm103.attention import mka as kernels


@pytest.mark.parametrize("enabled", [False, True])
def test_hop_fp8_is_capture_time_and_only_tp8_b1(monkeypatch, enabled):
    monkeypatch.setenv("SGLANG_Q35FLY_MKA_HOP_FP8", str(int(enabled)))
    for rows in (1, 7, 8, 14, 56, 168, 448, 896):
        for heads in (4, 8):
            assert mka.hop_fp8_input(rows, heads) == (
                enabled and rows == 7 and heads == 4
            )
    monkeypatch.setenv("SGLANG_Q35FLY_MKA_HOP_FP8", str(int(not enabled)))
    assert mka.hop_fp8_input(7, 4) is not enabled


def _inputs():
    x = torch.empty((7, 4096), dtype=torch.bfloat16)
    scalar = torch.ones(1, dtype=torch.float32)
    return dict(
        x=x,
        w_packed=(torch.empty(16, dtype=torch.uint8), torch.empty(0)),
        sx=scalar,
        sw=scalar,
        q_norm_w=torch.empty(256, dtype=torch.bfloat16),
        k_norm_w=torch.empty(256, dtype=torch.bfloat16),
        cos_sin=torch.empty((128, 64), dtype=torch.float32),
        positions=torch.empty((3, 7), dtype=torch.int64),
        axis_map=torch.empty(32, dtype=torch.int64),
        cache_loc=torch.empty(7, dtype=torch.int64),
        k_cache=torch.empty((64, 256), dtype=torch.float8_e4m3fn),
        v_cache=torch.empty((64, 256), dtype=torch.float8_e4m3fn),
        k_scale=scalar,
        v_scale=scalar,
        q_out=torch.empty((7, 1024), dtype=torch.float8_e4m3fn),
        gate_out=torch.empty((7, 1024), dtype=torch.bfloat16),
        nq=4,
        eps=1e-6,
    )


@pytest.mark.parametrize("early", [False, True])
def test_host_dispatch_preserves_abi_and_falls_back_without_copy(monkeypatch, early):
    calls = []
    monkeypatch.setattr(kernels, "_launch", lambda *args, **kwargs: calls.append(args))
    args = _inputs()
    kernels.prologue(**args, early_trigger=early)
    x8 = torch.empty(args["x"].shape, dtype=torch.float8_e4m3fn)
    kernels.prologue(**args, early_trigger=early, x8=x8)
    suffix = "" if early else "_late"
    assert calls[0][0] == "q35mka_prologue" + suffix
    assert calls[1][0] == "q35mka_prologue_fp8" + suffix
    assert calls[0][1] == calls[1][1] == 80
    assert calls[0][2].x == args["x"].data_ptr()
    assert calls[1][2].x == x8.data_ptr()
    for field, _ in kernels.ProParams._fields_:
        if field != "x":
            assert getattr(calls[0][2], field) == getattr(calls[1][2], field)


@pytest.mark.parametrize("bad", ["dtype", "shape", "heads", "stride"])
def test_rejects_invalid_fp8_contract(monkeypatch, bad):
    monkeypatch.setattr(kernels, "_launch", lambda *a, **k: pytest.fail("unexpected launch"))
    args = _inputs()
    x8 = torch.empty((7, 4096), dtype=torch.float8_e4m3fn)
    if bad == "dtype":
        x8 = args["x"]
    elif bad == "shape":
        x8 = x8[:1]
    elif bad == "heads":
        args["nq"] = 8
    else:
        x8 = torch.empty((4096, 7), dtype=torch.float8_e4m3fn).t()
    with pytest.raises(ValueError, match="x8 requires"):
        kernels.prologue(**args, x8=x8)
