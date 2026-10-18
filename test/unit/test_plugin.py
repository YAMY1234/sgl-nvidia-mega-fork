# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The plugin entry point on CPU: it registers each hook of ``plugin.HOOKS``, and the
hook leaves SGLang untouched on a server without ``--enable-nvidia-megakernel`` (or
on an SGLang that has no such flag)."""

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.deepseek_v41.runtime import hooks


@pytest.fixture
def registered(monkeypatch):
    from sglang.srt.plugins.hook_registry import HookRegistry

    seen = []
    monkeypatch.setattr(
        HookRegistry, "register", lambda target, hook, kind: seen.append((target, kind))
    )
    return seen


def test_register_registers_every_hook(registered, monkeypatch):
    from sglang.srt.plugins.hook_registry import HookType

    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()
    assert registered == [(target, HookType[kind]) for target, kind, _ in plugin.HOOKS]


def test_the_parity_sort_hook_registers_only_for_its_test(registered, monkeypatch):
    monkeypatch.setenv(plugin.PARITY_SORT_ENV, "1")
    plugin.register()
    targets = [target for target, _ in registered]
    assert targets == [t for t, _, _ in plugin.HOOKS + plugin.TEST_HOOKS]


class _Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"the hook read model.{name} without the flag")


@pytest.mark.parametrize(
    "server_args",
    [SimpleNamespace(), SimpleNamespace(**{plugin.FLAG: False})],
    ids=["no-such-flag", "flag-off"],
)
def test_the_hook_does_nothing_without_the_flag(server_args):
    runner = SimpleNamespace(server_args=server_args, model=_Untouchable())
    result = object()
    assert hooks.after_initialize(result, runner) is result
