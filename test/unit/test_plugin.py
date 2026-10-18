# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The plugin entry point on CPU: without the switch it registers nothing and SGLang
runs untouched; with it, it registers each hook of ``plugin.HOOKS``."""

import os
import subprocess
import sys
import textwrap

from sgl_nvidia_mega import plugin

UNSET = textwrap.dedent(
    """
    import sys

    from sglang.srt.plugins.hook_registry import HookRegistry

    from sgl_nvidia_mega import plugin

    plugin.register()
    assert not HookRegistry._hooks, dict(HookRegistry._hooks)
    ours = sorted(m for m in sys.modules if m.startswith("sgl_nvidia_mega."))
    assert ours == ["sgl_nvidia_mega.plugin"], ours
    print("UNSET OK")
    """
)


def test_nothing_is_registered_without_the_switch():
    env = {k: v for k, v in os.environ.items() if k != plugin.ENABLE_ENV}
    out = subprocess.run(
        [sys.executable, "-c", UNSET],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert "UNSET OK" in out.stdout, out.stdout[-2000:] + out.stderr[-4000:]


def test_the_switch_registers_every_hook(monkeypatch):
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    seen = []
    monkeypatch.setattr(
        HookRegistry, "register", lambda target, hook, kind: seen.append((target, kind))
    )
    monkeypatch.setenv(plugin.ENABLE_ENV, "0")
    plugin.register()
    assert seen == []
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    plugin.register()
    assert seen == [(target, HookType[kind]) for target, kind, _ in plugin.HOOKS]
