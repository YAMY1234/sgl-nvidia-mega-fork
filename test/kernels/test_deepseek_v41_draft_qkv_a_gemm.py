# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft's fused wqkv_a on an SM100-family GPU (GB300): one MXFP8 GEMM over
[wq_a; wkv] through SGLang's own CuTe-DSL dispatch against the two GEMMs SGLang runs.
q_lora feeds the q path and kv the draft KV store, so any difference here is the draft
KV-byte tolerance the slot declares. Prints the max |difference| per width and mode."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

HIDDEN, Q_LORA, HEAD_DIM = 5120, 1280, 512
ROWS = (1, 2, 4, 5, 6, 8, 10, 16, 20, 24, 30, 40, 48)


def _mxfp8_weight(n, k, gen):
    from sgl_nvidia_mega.deepseek_v41.sm103.mxfp8_ll import quantize

    w = torch.randn(n, k, generator=gen, device="cuda").to(torch.bfloat16) * 0.02
    q, s = quantize(w)
    return q.view(n, k), s.reshape(-1)


@pytest.mark.parametrize("autotune", [False, True])
def test_fused_wqkv_a_against_sglangs_two_gemms(autotune):
    from flashinfer.autotuner import autotune as tuning
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    gen = torch.Generator(device="cuda").manual_seed(0)
    wq, sq = _mxfp8_weight(Q_LORA, HIDDEN, gen)
    wkv, skv = _mxfp8_weight(HEAD_DIM, HIDDEN, gen)
    w, s = torch.cat([wq, wkv]), torch.cat([sq, skv])

    def gemm(x, weight, scale):
        return flashinfer_mxfp8_blockscaled_linear(x, weight, scale, backend="cute-dsl")

    worst = {}
    for rows in ROWS:
        x = torch.randn(rows, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
        with tuning(autotune):
            q_ref, kv_ref = gemm(x, wq, sq), gemm(x, wkv, skv)
            fused = gemm(x, w, s)
        q, kv = fused[:, :Q_LORA], fused[:, Q_LORA:]
        dq = (q.float() - q_ref.float()).abs().max().item()
        dkv = (kv.float() - kv_ref.float()).abs().max().item()
        rel = dkv / max(kv_ref.float().abs().max().item(), 1e-30)
        worst[rows] = (dq, dkv)
        print(
            f"DRAFT_QKV_A autotune={int(autotune)} rows={rows} "
            f"q_max_abs={dq:.3e} kv_max_abs={dkv:.3e} kv_rel={rel:.3e} "
            f"bitwise={int(torch.equal(q, q_ref) and torch.equal(kv, kv_ref))}"
        )
        assert rel < 1e-2, f"rows {rows}: kv differs by {rel:.3e} of its range"
    assert worst
