# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.srt.megakernels``); an unsupported setup stops the server on every rank,
listing every reason. This module stays light: SGLang imports it in the launcher.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from sglang.srt.megakernels.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG
from sgl_nvidia_mega.deepseek_v41.plan import MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "TOLERANCE",
        "Requirements",
        "decode_graph_backends",
    ),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "speculative_algorithm",
        "moe_a2a_backend",
        "dsv4_attn_backend",
        "enable_encoder_swa_bounded_replay",
        "startup_weight_load_mode",
    ),
}

ARCHITECTURE = "DeepseekV4ForCausalLM"
# the sglang.srt.megakernels contract this spec is written against
API_VERSION = 1
# The FlashInfer low-latency GEMM file the wo_b publish epilogue subclasses, by the
# sha256 of its source; it is byte-identical in FlashInfer 0.7.0, 0.7.1 and 0.7.2
FLASHINFER_LL_GEMM = "gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py"
FLASHINFER_LL_GEMM_SHA256 = frozenset(
    {"118744bf9553da5fcbf6504739d499ec208dcf2034d473799ec11661e1313280"}
)

REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    tp_sizes=frozenset({4}),
    # the GEMM pair's persistent schedule and the Lamport consumer stay within
    # GB300's 152 SMs
    resident_blocks=152,
    dtypes=frozenset({"bfloat16"}),
    # the decode CUDA graphs the covered step is tested with (eager steps too)
    decode_graph_backends=frozenset({"full", "disabled"}),
    server_args={
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        # DSpark's target steps are uniform verify steps
        "speculative_algorithm": frozenset({"DSPARK"}),
        "moe_a2a_backend": frozenset({"none"}),
        # the wo_b publish follows FlashMLA's sparse decode; on GB300 it is also
        # the faster backend at batch 1-8
        "dsv4_attn_backend": frozenset({"flashmla"}),
        # the paged SWA cache, not a request window
        "enable_encoder_swa_bounded_replay": frozenset({False}),
        # the kernels read weights in place, after SGLang's last conversion
        "startup_weight_load_mode": frozenset({"serial"}),
    },
    hf_config={**HF_CONFIG, "model_type": "deepseek_v41"},
)

# Per kernel output against SGLang's op on the same GPU (test/kernels); the
# tolerances are the measured ones
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
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
    },
)


def toolchain_reasons() -> list[str]:
    """The kernel toolchains, and the FlashInfer GEMM the wo_b publish extends."""
    reasons = []
    for module, what in (
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("flashinfer", "FlashInfer"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ):
        if importlib.util.find_spec(module) is None:
            reasons.append(f"needs {what}")
    digest = flashinfer_ll_gemm_digest()
    if digest is not None and digest not in FLASHINFER_LL_GEMM_SHA256:
        reasons.append(
            f"FlashInfer's {FLASHINFER_LL_GEMM} (sha256 {digest[:12]}) is not the "
            "low-latency GEMM the wo_b publish epilogue extends"
        )
    return reasons


def flashinfer_ll_gemm_digest():
    """The sha256 of the installed FlashInfer low-latency GEMM source, or None
    without FlashInfer (refused above)."""
    found = importlib.util.find_spec("flashinfer")
    if found is None or found.origin is None:
        return None
    path = Path(found.origin).parent / FLASHINFER_LL_GEMM
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sgl_nvidia_mega.deepseek_v41 import runtime

    return toolchain_reasons() + runtime.slot_reasons()


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.deepseek_v41 import runtime

    return runtime.model_disable_reasons(model)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="DeepSeek-V4.1",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
