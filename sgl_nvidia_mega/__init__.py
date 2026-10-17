# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Monokernels for SGLang on NVIDIA GPUs: the kernels of a decode or verify step
that only make sense inside a fused step, and the forward that runs them.

SGLang holds the interface (``sglang.srt.monokernels``). This package holds the
shared CUDA runtime (``common``) and, per model, its specs and their forwards
(``<model>/spec.py``, ``<model>/draft.py``) and its kernels per ISA
(``<model>/<isa>/``). SGLang loads the package as a general plugin
(``pyproject.toml``: the ``sglang.srt.plugins`` entry point), which registers
each spec with SGLang's monokernel framework.
"""

__version__ = "0.1.0.dev0"

# The --monokernel-backend name this package registers
BACKEND = "nvidia"
# Each spec by the model's HF architecture; a spec module imports only when a
# server runs this backend
SPECS = {
    "DeepseekV4ForCausalLM": "sgl_nvidia_mega.deepseek_v41.spec:SPEC",
    "DeepseekV4ForCausalLMDSpark": "sgl_nvidia_mega.deepseek_v41.draft:SPEC",
}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"srt/monokernels/spec.py": ("register",)}


def register_specs() -> None:
    """The plugin entry point, which SGLang runs in every process."""
    from sglang.srt.monokernels.spec import register

    for architecture, path in SPECS.items():
        register(backend=BACKEND, architecture=architecture, path=path)
