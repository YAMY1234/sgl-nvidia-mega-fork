# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang plugin entry point: the switch and the hooks it registers."""

from __future__ import annotations

import os

# The switch: "1" enables the NVIDIA megakernel; anything else registers nothing
ENABLE_ENV = "SGLANG_ENABLE_NVIDIA_MEGAKERNEL"

# SGLang targets the megakernel hooks, by dotted path, with the hook type and the
# plugin function (test/unit/test_contract.py resolves each target)
HOOKS = (
    (
        "sglang.srt.model_executor.model_runner.ModelRunner.load_model",
        "AFTER",
        "sgl_nvidia_mega.deepseek_v41.runtime.hooks:after_load_model",
    ),
    (
        "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.forward",
        "AROUND",
        "sgl_nvidia_mega.deepseek_v41.runtime.hooks:around_forward",
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


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "") == "1"


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built."""
    if not enabled():
        return
    import pkgutil

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, kind, hook in HOOKS:
        HookRegistry.register(target, pkgutil.resolve_name(hook), HookType[kind])
