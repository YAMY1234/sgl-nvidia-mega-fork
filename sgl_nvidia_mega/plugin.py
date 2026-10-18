# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang plugin entry point: the hooks it registers and the server flag that
turns the megakernel on."""

from __future__ import annotations

import os

# The switch: SGLang's server flag (a ServerArgs field); without it, or on an SGLang
# that has no such flag, the hooks return at once and SGLang runs untouched
FLAG = "enable_nvidia_megakernel"
FLAG_CLI = "--enable-nvidia-megakernel"

# SGLang targets the megakernel hooks, by dotted path, with the hook type and the
# plugin function (test/unit/test_contract.py resolves each target). The model's
# forward is replaced per instance by the build, not on SGLang's class.
HOOKS = (
    (
        "sglang.srt.model_executor.model_runner.ModelRunner.initialize",
        "AFTER",
        "sgl_nvidia_mega.deepseek_v41.runtime.hooks:after_initialize",
    ),
)

# Test only: with SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK=1, the composition check
# sorts the sparse attention indices in both forwards (their emission order varies)
PARITY_SORT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK"
TEST_HOOKS = (
    (
        "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend.forward",
        "AROUND",
        "sgl_nvidia_mega.deepseek_v41.runtime.parity:around_attention_forward",
    ),
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": (
        "HookRegistry",
        "HookType",
        "register",
        "AFTER",
        "AROUND",
    ),
}


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built.
    The hooks act only on a server started with ``FLAG_CLI``."""
    import pkgutil

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    hooks = HOOKS + (TEST_HOOKS if os.environ.get(PARITY_SORT_ENV) == "1" else ())
    for target, kind, hook in hooks:
        HookRegistry.register(target, pkgutil.resolve_name(hook), HookType[kind])
