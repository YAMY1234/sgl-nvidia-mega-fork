# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash on SM103 (GB300): the monokernel spec.

A routed step (a DSpark verify step of 1-48 rows) runs ``forward.Routed``, which
rebuilds SGLang's step from SGLang's own modules and swaps three slots for this
package's kernels (``plan.py``): wo_b publishing into a Lamport all-reduce + mHC at
1-8 rows, the router's top-k, and the routed experts on an 8-row-tile GEMM pair. The
step stays a chain of kernels launched back to back with programmatic dependent
launch: on GB300 a kernel boundary with PDL costs less than an exchange inside one
persistent grid.
"""

from __future__ import annotations

import importlib.util

from sglang.srt.monokernels.spec import (
    MONOKERNEL_API_VERSION,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG, plan

ISA = "sm_103"
WIDTHS = tuple(range(1, plan.MAX_ROWS + 1))
# The largest grid the kernels need co-resident: the GEMM pair's persistent schedule
# and the Lamport consumer stay within GB300's 152 SMs
RESIDENT_BLOCKS = 152
# The FlashInfer low-latency GEMM file the wo_b publish epilogue subclasses, by the
# sha256 of its source; it is byte-identical in FlashInfer 0.7.0, 0.7.1 and 0.7.2
FLASHINFER_LL_GEMM = "gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py"
FLASHINFER_LL_GEMM_SHA256 = frozenset(
    {"118744bf9553da5fcbf6504739d499ec208dcf2034d473799ec11661e1313280"}
)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "speculative_algorithm",
        "moe_a2a_backend",
        "dsv4_attn_backend",
        "enable_encoder_swa_bounded_replay",
        "enable_decoder_swa_bounded_replay",
        "startup_weight_load_mode",
    ),
}

# Per kernel output against SGLang's op on the same GPU (test/kernels); the
# tolerances are the measured ones
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # only SGLang's own writers touch the KV caches
        "attn.kv.max_abs": 0.0,
        # wo_b's published partial, as FlashInfer's low-latency GEMM writes it
        "wo_b.partial.max_abs": 0.0,
        # the router's expert ids and weights, as moe_fused_gate picks them
        "gate.topk.max_abs": 0.0,
        # the Lamport hop's streams and normed row against all-reduce + mHC in torch
        "hop_a.residual.rel": 2e-2,
        "hop_a.normed.rel": 3e-2,
        # the routed experts against FlashInfer's routed MoE: one BF16 rounding
        "moe.routed.rel": 2**-7,
        "draft_moe.out.rel": 2**-7,
    },
)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the kernel toolchains."""
    reasons = []
    for module, what in (
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("flashinfer", "FlashInfer"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ):
        if importlib.util.find_spec(module) is None:
            reasons.append(f"needs {what}")
    digest = _flashinfer_ll_gemm_digest()
    if digest is not None and digest not in FLASHINFER_LL_GEMM_SHA256:
        reasons.append(
            f"FlashInfer's {FLASHINFER_LL_GEMM} (sha256 {digest[:12]}) is not the "
            "low-latency GEMM the wo_b publish epilogue extends"
        )
    return reasons


def _flashinfer_ll_gemm_digest():
    """The sha256 of the installed FlashInfer low-latency GEMM source, or None
    without FlashInfer (refused above)."""
    import hashlib
    from pathlib import Path

    found = importlib.util.find_spec("flashinfer")
    if found is None or found.origin is None:
        return None
    path = Path(found.origin).parent / FLASHINFER_LL_GEMM
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sgl_nvidia_mega.deepseek_v41 import forward

    return forward.model_disable_reasons(model, hf_config)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.deepseek_v41 import forward

    return forward.build(model, hf_config, ctx)


SPEC = MonoSpec(
    name="deepseek_v41_nvidia",
    devices={
        ISA: DeviceLimits(widths=WIDTHS, tp_sizes=(4,), resident_blocks=RESIDENT_BLOCKS)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsv4"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            # DSpark's target steps are uniform verify steps
            "speculative_algorithm": frozenset({"DSPARK"}),
            "moe_a2a_backend": frozenset({"none"}),
            # the hop-A slot replaces wo_b after FlashMLA's sparse decode; on GB300
            # flashmla is also the faster backend at batch 1-8
            "dsv4_attn_backend": frozenset({"flashmla"}),
            # the paged SWA cache, not a request window
            "enable_encoder_swa_bounded_replay": frozenset({False}),
            "enable_decoder_swa_bounded_replay": frozenset({False, True}),
            # the kernels read weights in place, after SGLang's last conversion
            "startup_weight_load_mode": frozenset({"serial"}),
        },
        hf_config={**HF_CONFIG, "model_type": "deepseek_v41"},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=MONOKERNEL_API_VERSION,
    config_disable_reasons=_config_disable_reasons,
)
