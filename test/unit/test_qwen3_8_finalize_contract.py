# SPDX-License-Identifier: Apache-2.0
"""Plugin finalize offers: real provider, caller-owned views, explicit raw spy.

Framework sink lifetime, source eligibility and stream retention belong to the
binding tests. These checks never execute or extract those framework helpers.
"""

import builtins
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizeSharedIO,
    NumericOperation,
    ScalarType,
)

from sgl_nvidia_mega.qwen3_8._providers import experts

RAW_MODULE = "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_finalize_shared"


def signature():
    return ExpertSignature(
        NumericOperation.FINALIZE_SHARED, 4, 2560, 192, 512, 10, ScalarType.BF16
    )


def view(cols=256, pdl=True):
    values = [object() for _ in range(6)]
    return FinalizeSharedIO(*values[:5], cols, pdl, values[5]), tuple(values)


def raw_spy(monkeypatch, launch):
    module = ModuleType(RAW_MODULE)
    module.finalize_shared = launch
    monkeypatch.setitem(sys.modules, RAW_MODULE, module)


def test_finalize_offer_calls_plugin_without_framework_sink_helpers(monkeypatch):
    original_import = builtins.__import__

    def forbid_framework_helpers(name, *args, **kwargs):
        if name == "sglang.srt.kernel_plugins.nvidia.local_experts":
            pytest.fail("plugin finalize imported framework-only sink helpers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_framework_helpers)
    calls = []
    raw_spy(monkeypatch, lambda *args, **kwargs: calls.append((args, kwargs)))
    run = experts.build(signature())
    assert run is experts.finalize_shared
    io, values = view()
    assert run(io) is None
    assert calls == [(values, {"cols": 256, "use_pdl": True})]


def test_finalize_consumes_each_calls_own_buffers_and_launch_options(monkeypatch):
    calls = []
    raw_spy(monkeypatch, lambda *args, **kwargs: calls.append((args, kwargs)))
    run = experts.build(signature())
    first, first_values = view(cols=256, pdl=True)
    second, second_values = view(cols=512, pdl=False)
    assert all(a is not b for a, b in zip(first_values, second_values))
    assert run(first) is None
    assert run(second) is None
    assert calls == [
        (first_values, {"cols": 256, "use_pdl": True}),
        (second_values, {"cols": 512, "use_pdl": False}),
    ]
    assert calls[0][0][-1] is first.output
    assert calls[1][0][-1] is second.output


def test_selected_finalize_error_propagates_once_without_retry(monkeypatch):
    calls = []
    error = RuntimeError("selected finalize failure")

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    raw_spy(monkeypatch, fail)
    io, values = view()
    with pytest.raises(RuntimeError) as caught:
        experts.build(signature())(io)
    assert caught.value is error
    assert calls == [(values, {"cols": 256, "use_pdl": True})]


def test_unsupported_finalize_signature_refuses_before_raw_import(monkeypatch):
    original_import = builtins.__import__

    def forbid_raw(name, *args, **kwargs):
        if name == RAW_MODULE:
            pytest.fail("unsupported finalize imported raw implementation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_raw)
    for changes in (
        {"rows": 0},
        {"hidden_size": 2561},
        {"topk": 9},
        {"scalar_type": ScalarType.FP32},
        {"variant": 1},
    ):
        candidate = replace(signature(), **changes)
        assert not experts.supported(candidate)
        with pytest.raises(ValueError, match="unsupported local expert signature"):
            experts.build(candidate)
