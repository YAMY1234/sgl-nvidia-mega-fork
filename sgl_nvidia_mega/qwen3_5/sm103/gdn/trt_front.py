# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM r3 (tekit 7df45abe5c), tensorrt_llm/_torch/modules/mamba/gdn_mixer.py
# (Qwen3NextGatedDeltaNet._lowm_in_proj and its TRTLLM_Q35_GDN_FUSE=4 caller).
"""The GatedDeltaNet verify front of one request of 7 rows on TensorRT-LLM r3's own in_proj kernel
(``SGLANG_PERFLL_GDN_TRTFRONT=1``, default off): ``lowm_gemm`` as TensorRT-LLM's GDN mixer calls it with
``TRTLLM_Q35_GDN_FUSE=4`` (``sm103/trt_lowm``): in one launch the static-FP8 ``in_proj_qkvz`` on the hop's E4M3 rows
(``fp8``, TensorRT-LLM's mixed8 tactic) or on the BF16 rows quantized in registers (``quant``, its mixed tactic,
where no E4M3 copy exists), the BF16 ``in_proj_ba`` on the BF16 rows, and the causal conv update of the q / k / v
channels as the epilogue. TP8 or TP16 shapes.

Outputs: ``out`` [7, QKVZ] BF16 holds the convolved q | k | v then the raw z (the front's ``conv_out`` and
``projected[:, CONV:]`` in one buffer), ``ba`` [7, 2 HV] BF16 b then a; the conv state is updated at the request's
state index (none for a pad index, -1) and the windows after every token at intermediate row 0 (SGLang's
``verify_intermediate_state_indices`` gives batch row r row r). The sums run in TensorRT-LLM's order: numeric, not
SGLang's GEMMs bit for bit; the conv is the arithmetic of Triton's ``causal_conv1d_update``.

The launch is a programmatic dependent that reads only weights, scales and conv weights before its
``griddepcontrol.wait`` (the rows, the state index and the conv state after it) and releases its dependents once
those weight loads are issued; the next launch waits for it before reading any output. ``prepare`` compiles both
activation forms where the pools first exist (before graph capture); ``front`` refuses an unprepared one.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("SGLANG_PERFLL_GDN_TRTFRONT", "0") == "1"
ROWS, HIDDEN = 7, 4096
# TensorRT-LLM's _TACTICS mode of each activation form (lowm_linear.default_tactic)
MODES = {"fp8": "mixed8", "quant": "mixed"}

_SPECS: dict = {}  # (activation form, qkvz rows) -> the prepared trt_lowm.Spec


def spec(act: str, qkvz: int, ba: int, conv_weight, conv_state, windows):
    """The ``trt_lowm`` launch of one request's front with the ``act`` activations (``fp8`` / ``quant``) over a
    [qkvz, 4096] FP8 and a [ba, 4096] BF16 weight, on these conv tensors' layouts: TensorRT-LLM's tactic of the
    shape (BF16 segment: its k_warps and ksplit, one tile a CTA), as its GDN mixer passes it."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import (
        default_tactic,
        lowm_conv_supported,
    )

    tactic = default_tactic(ROWS, qkvz, HIDDEN, mode=MODES[act])
    if not lowm_conv_supported(ROWS, tactic, conv_weight.shape[0], ROWS):
        raise ValueError(
            f"TensorRT-LLM's {MODES[act]} tactic {tactic} has no conv epilogue for {qkvz} rows"
        )
    return trt_lowm.spec(
        m=ROWS,
        k=HIDDEN,
        n_fp8=qkvz,
        n_bf16=ba,
        fp8_act=act,
        tactic=tactic,
        tactic_bf16=(tactic[0], 1, tactic[2] if len(tactic) > 2 else 1),
        split_out=qkvz,
        out_dtype=torch.bfloat16,
        conv=trt_lowm.conv_spec(
            conv_weight, conv_state, windows, requests=1, tokens=ROWS
        ),
    )


def prepare(qkvz: int, ba: int, conv_weight, conv_state, windows) -> None:
    """Compile both activation forms on these layouts (not inside a graph capture)."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    for act in MODES:
        s = spec(act, qkvz, ba, conv_weight, conv_state, windows)
        trt_lowm.prepare(s)
        _SPECS[act, qkvz] = s


def front(
    x: torch.Tensor,
    x_fp8,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    ba_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    windows: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    ba: torch.Tensor,
) -> None:
    """One request's front of the BF16 rows ``x`` [7, 4096] (and their E4M3 copy ``x_fp8``, quantized with
    ``input_scale``, or None) over ``weight`` [QKVZ, 4096] E4M3 (row-major [N, K]) and ``ba_weight`` [2 HV, 4096]
    BF16 into ``out`` [7, QKVZ] and ``ba`` [7, 2 HV]; ``conv_state`` [slots, C, 3] / ``windows`` [rows, 7, C, 3]
    updated at ``state_indices`` [1] int32 and intermediate row 0."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    act = "fp8" if x_fp8 is not None else "quant"
    s = _SPECS.get((act, weight.shape[0]))
    if s is None:
        raise RuntimeError(
            f"the TensorRT-LLM GDN front ({act}, {weight.shape[0]} rows) was not prepared"
        )
    trt_lowm.run(
        s,
        out=out,
        out2=ba,
        x16=x,
        x8=x_fp8,
        w_fp8=weight,
        weight_scale=weight_scale,
        input_scale=input_scale,
        w_bf16=ba_weight,
        conv=(conv_weight, conv_state, state_indices, windows),
    )
