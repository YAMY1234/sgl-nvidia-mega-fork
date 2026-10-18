# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The setups the DeepSeek-V4.1 megakernel serves, checked when the model is loaded.

With the megakernel switched on, an unsupported setup stops the server at startup on
every rank, listing every reason; the megakernel never falls back silently.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch

from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "tp_size",
        "pp_size",
        "nnodes",
        "ep_size",
        "enable_dp_attention",
        "enable_eplb",
        "attn_cp_size",
        "dcp_size",
        "enable_prefill_cp",
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_lora",
        "lora_paths",
        "disaggregation_mode",
        "enable_hierarchical_cache",
        "enable_memory_saver",
        "enable_torch_compile",
        "kv_cache_dtype",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "moe_a2a_backend",
        "dsv4_attn_backend",
        "enable_encoder_swa_bounded_replay",
        "startup_weight_load_mode",
        "watchdog_timeout",
    ),
    "srt/model_executor/model_runner.py": ("server_args", "model_config", "model"),
    "srt/configs/model_config.py": ("hf_config", "dtype"),
}

ISA = "sm_103"
TP_SIZE = 4
# The largest grid the kernels need co-resident: the GEMM pair's persistent schedule
# and the Lamport consumer stay within GB300's 152 SMs
RESIDENT_BLOCKS = 152
# The FlashInfer low-latency GEMM file the wo_b publish epilogue subclasses, by the
# sha256 of its source; it is byte-identical in FlashInfer 0.7.0, 0.7.1 and 0.7.2
FLASHINFER_LL_GEMM = "gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py"
FLASHINFER_LL_GEMM_SHA256 = frozenset(
    {"118744bf9553da5fcbf6504739d499ec208dcf2034d473799ec11661e1313280"}
)
# Leave time for peers to abort before SGLang's watchdog fires
MAX_DEADLINE_S, MIN_DEADLINE_S = 60.0, 5.0

# Server arguments the megakernel needs, each with the values it accepts
SERVER_ARGS = {
    "tp_size": {TP_SIZE},
    "kv_cache_dtype": {"fp8_e4m3"},
    # DSpark's target steps are uniform verify steps
    "speculative_algorithm": {"DSPARK"},
    "moe_a2a_backend": {"none"},
    # the wo_b publish follows FlashMLA's sparse decode; on GB300 it is also the
    # faster backend at batch 1-8
    "dsv4_attn_backend": {"flashmla"},
    # the paged SWA cache, not a request window
    "enable_encoder_swa_bounded_replay": {False},
    # the kernels read weights in place, after SGLang's last conversion
    "startup_weight_load_mode": {"serial"},
}

# SGLang features the covered step does not reproduce, by how a server enables them
FEATURES = {
    "pipeline parallelism": lambda a: a.pp_size > 1,
    "multi-node tensor parallelism": lambda a: a.nnodes > 1,
    "DP attention": lambda a: bool(a.enable_dp_attention),
    "expert parallelism": lambda a: (a.ep_size or 1) > 1,
    "EPLB": lambda a: bool(a.enable_eplb),
    "context parallelism": lambda a: (
        (a.attn_cp_size or 1) > 1 or (a.dcp_size or 1) > 1 or bool(a.enable_prefill_cp)
    ),
    "two-batch overlap": lambda a: bool(
        a.enable_two_batch_overlap or a.enable_single_batch_overlap
    ),
    "LoRA": lambda a: bool(a.enable_lora or a.lora_paths),
    "PD disaggregation": lambda a: (a.disaggregation_mode or "null") != "null",
    "hierarchical KV cache": lambda a: bool(a.enable_hierarchical_cache),
    "memory saver": lambda a: bool(a.enable_memory_saver),
    "torch.compile": lambda a: bool(a.enable_torch_compile),
}

# Per kernel output against SGLang's op on the same GPU (test/kernels); the
# tolerances are the measured ones
NUMERICS = {
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
}


def deadline_s(watchdog_timeout) -> float:
    """A cross-GPU wait's deadline: half the watchdog timeout, at most 60 s."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(MAX_DEADLINE_S, 0.5 * float(watchdog_timeout))


def disable_reasons(server_args, model_config) -> list[str]:
    """Every reason the megakernel cannot serve this setup ([] when it can)."""
    reasons = [f"{name} is in use" for name, on in FEATURES.items() if on(server_args)]
    for arg, allowed in SERVER_ARGS.items():
        value = getattr(server_args, arg)
        if value not in allowed:
            reasons.append(f"{arg} {value!r} is not one of {sorted(map(str, allowed))}")
    dtype = str(model_config.dtype).replace("torch.", "")
    if dtype != "bfloat16":
        reasons.append(f"dtype {dtype!r} is not 'bfloat16'")
    hf_config = model_config.hf_config
    for key, want in {**HF_CONFIG, "model_type": "deepseek_v41"}.items():
        value = getattr(hf_config, key, None)
        if value != want:
            reasons.append(f"config.{key} {value!r} is not {want!r}")
    if deadline_s(server_args.watchdog_timeout) < MIN_DEADLINE_S:
        reasons.append(
            f"--watchdog-timeout {server_args.watchdog_timeout:g} s leaves the waits "
            f"a deadline under {MIN_DEADLINE_S:g} s"
        )
    return reasons + device_reasons() + toolchain_reasons()


def device_reasons() -> list[str]:
    """The GPU: SM103 (GB300), with every SM the kernels' grids need."""
    if not torch.cuda.is_available():
        return ["no GPU is visible"]
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    isa = f"sm_{props.major}{props.minor}"
    reasons = [] if isa == ISA else [f"the GPU is {isa}, not {ISA} (GB300)"]
    if props.multi_processor_count < RESIDENT_BLOCKS:
        reasons.append(
            f"{props.multi_processor_count} SMs; the kernels' grids need "
            f"{RESIDENT_BLOCKS} (a partitioned GPU)"
        )
    return reasons


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
