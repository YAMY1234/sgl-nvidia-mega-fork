# SPDX-License-Identifier: Apache-2.0
"""Check the actual router caller's complete NVRTC target, without compilation."""

import pytest

from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_router_front as router


@pytest.mark.parametrize(
    "capability,target", [((10, 3), "sm_103a"), ((10, 7), "sm_107a")]
)
@pytest.mark.parametrize("capacity,ksplit", [(1, 2), (8, 2), (16, 4)])
def test_router_passes_complete_target_to_shared_compiler(
    monkeypatch, capability, target, capacity, ksplit
):
    class AtCompiler(Exception):
        pass

    def compile_source(source, name, arch, options):
        assert source == router._SOURCE.read_text()
        assert name == f"q38_router_h2560_m{capacity}_k{ksplit}"
        assert arch == target
        assert f"-DQ35_ROUTER_MAX_M={capacity}" in options
        assert f"-DQ35_ROUTER_KSPLIT={ksplit}" in options
        raise AtCompiler

    monkeypatch.setattr(
        router.torch.cuda, "get_device_capability", lambda device: capability
    )
    monkeypatch.setattr(router.jit, "compile_source", compile_source)
    router.kernel.cache_clear()
    with pytest.raises(AtCompiler):
        router.kernel(0, capacity, ksplit)
