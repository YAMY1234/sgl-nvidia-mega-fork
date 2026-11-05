# SPDX-License-Identifier: Apache-2.0
"""Compare direct caller-buffer glue with the actual original wrapper on CPU."""

import ast
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnLaunchConfig,
    GdnRingIO,
    GdnSignature,
    GdnVerifyGlueIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers
from sgl_nvidia_mega.qwen3_8._providers import gdn_verify_glue as glue


def signature(m=8, h=4, hv=12):
    return GdnSignature(
        O.GDN_VERIFY_GLUE, m, h, hv, 128, 128, 4, D.BF16, gate_scalar_type=D.BF16
    )


def view(s, mode="exact", norm=2, dt_dtype=torch.bfloat16):
    def t(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    m, h, hv, slots, cap = s.rows, s.key_heads, s.value_heads, 4, 32
    channels = (2 * h + hv) * 128
    d, k, g = (
        t((slots, hv, cap, 128)),
        t((slots, h, cap, 128)),
        t((slots, hv, cap), torch.float32),
    )
    rawv = t(d.shape) if mode != "none" else None
    rawk = t(k.shape) if mode != "none" else None
    beta = t(g.shape, torch.float32) if mode == "exact" else None
    ring = GdnRingIO(
        d,
        k,
        g,
        rawv,
        rawk,
        beta,
        None,
        None,
        t((m // 4,), torch.int32),
        t((m // 4,), torch.int64),
        t((slots,), torch.int32),
        t((slots,), torch.int32),
        t((slots,), torch.int8),
    )
    return GdnVerifyGlueIO(
        t((m, channels + hv * 128)),
        t((m, 2 * hv)),
        t((slots, channels, 3)),
        t((channels, 4)),
        None,
        t((m // 4,), torch.int64),
        t((hv,), torch.float32),
        t((hv,), dt_dtype),
        t((128,)),
        1e-6,
        t((slots, hv, 128, 128)),
        ring,
        h,
        128,
        4,
        -1,
        1,
        norm,
        GdnLaunchConfig(0, 4, 1, False, 0, True, 1, True, True),
        t((m, hv * 128)),
        t((slots, 4, channels, 3)),
    )


def raw_spy(monkeypatch):
    path = (
        Path(glue.__file__).parents[1]
        / "kernels/ops/attention/linear/gdn_fused_verify_glue.py"
    )
    tree = ast.parse(path.read_text())
    wrapper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "gdn_fused_verify_glue"
    )
    state = SimpleNamespace(capturing=False, calls=[])

    def freeze(v):
        if isinstance(v, torch.Tensor):
            base = v if v._base is None else v._base
            return (
                id(base),
                tuple(v.shape),
                tuple(v.stride()),
                v.dtype,
                v.storage_offset(),
            )
        return v

    class Kernel:
        def __getitem__(self, grid):
            def run(*a, **kw):
                state.calls.append((grid, tuple(map(freeze, a)), kw))

            return run

    module = ModuleType(
        "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_fused_verify_glue"
    )
    module.torch = torch
    module.triton = SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length())
    module._NORM_ACT = {None: 0, "": 0, "swish": 1, "silu": 1, "sigmoid": 2}
    module.gdn_fused_verify_glue_kernel = Kernel()
    exec(
        compile(ast.Module(body=[wrapper], type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    return module, state


def original(module, io):
    r = io.ring
    return module.gdn_fused_verify_glue(
        mixed_qkvz=io.mixed_qkvz,
        mixed_ba=io.mixed_ba,
        conv_state=io.conv_state,
        conv_weight=io.conv_weight,
        conv_bias=io.conv_bias,
        conv_activation="silu",
        intermediate_conv_window=io.intermediate_conv_window_out,
        intermediate_state_indices=io.intermediate_indices,
        A_log=io.a_log,
        dt_bias=io.dt_bias,
        norm_weight=io.norm_weight,
        norm_eps=io.norm_epsilon,
        norm_activation="silu" if io.norm_activation == 1 else "sigmoid",
        checkpoint_state=io.checkpoint,
        d_cache=r.d,
        k_cache=r.k,
        g_cache=r.g,
        rawv_cache=r.raw_v,
        rawk_cache=r.raw_k,
        beta_cache=r.beta,
        cache_indices=r.state_indices,
        replay_indices=r.replay_indices,
        write_pos=r.write_pos,
        cache_base=r.cache_base,
        is_flush=r.is_flush,
        num_k_heads=io.key_heads,
        head_k_dim=io.key_dim,
        draft_token_num=io.tokens_per_request,
        out=io.output,
        null_block_id=io.null_slot,
    )


@pytest.mark.parametrize("m", (8, 12, 16))
@pytest.mark.parametrize("h,hv", ((4, 12), (16, 48)))
@pytest.mark.parametrize("norm", (1, 2))
@pytest.mark.parametrize("mode", ("exact", "residual", "none"))
@pytest.mark.parametrize("dt_dtype", (torch.bfloat16, torch.float32))
def test_original_full_abi_and_store_modes(monkeypatch, m, h, hv, norm, mode, dt_dtype):
    module, state = raw_spy(monkeypatch)
    s = signature(m, h, hv)
    io = view(s, mode, norm, dt_dtype)
    assert original(module, io) is io.output
    expected = list(state.calls)
    state.calls.clear()
    run = glue.build(s)
    assert run(io) is None and state.calls == expected
    state.capturing = True
    state.calls.clear()
    io = replace(io, mixed_qkvz=torch.empty_like(io.mixed_qkvz))
    run(io)
    assert len(state.calls) == 1
    assert state.calls[0][1][0][0] == id(io.mixed_qkvz)
    assert len(run._warmed) == 1
    # Changing a real specialization while capturing must not call raw Triton.
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(replace(io, null_slot=0))
    assert len(state.calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 4),
        ("rows", 32),
        ("key_dim", 64),
        ("value_dim", 64),
        ("key_heads", 8),
        ("value_heads", 24),
        ("tokens_per_request", 1),
        ("variant", 1),
        ("scalar_type", D.FP32),
        ("gate_scalar_type", D.FP32),
    ],
)
def test_unsupported_signature_stays_stock(field, value):
    bad = replace(signature(), **{field: value})
    assert not providers.supported(bad) and providers.make_offer(bad) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_n", 32),
        ("num_warps", 8),
        ("num_stages", 2),
        ("hoist", True),
        ("dot_precision", 1),
        ("prefetch", False),
        ("historical_branch", 0),
        ("use_pdl", False),
        ("trigger_early", False),
    ],
)
def test_nondefault_policy_rejected_before_state_launch(monkeypatch, field, value):
    _, state = raw_spy(monkeypatch)
    io = view(signature())
    with pytest.raises(ValueError, match="prepared source"):
        glue.build(signature())(
            replace(io, launch=replace(io.launch, **{field: value}))
        )
    assert state.calls == []


@pytest.mark.parametrize(
    "field", ("mixed_qkvz", "mixed_ba", "checkpoint", "norm_weight", "output")
)
def test_bf16_actual_operands_no_cast(monkeypatch, field):
    _, state = raw_spy(monkeypatch)
    io = view(signature())
    with pytest.raises(ValueError, match="prepared source"):
        glue.build(signature())(replace(io, **{field: getattr(io, field).float()}))
    assert state.calls == []


def test_framework_flush_must_already_be_int8(monkeypatch):
    _, state = raw_spy(monkeypatch)
    io = view(signature())
    io = replace(io, ring=replace(io.ring, is_flush=io.ring.is_flush.int()))
    with pytest.raises(ValueError, match="prepared source"):
        glue.build(signature())(io)
    assert state.calls == []


def test_cold_capture_and_no_implicit_allocation_cast_or_sync(monkeypatch):
    _, state = raw_spy(monkeypatch)
    state.capturing = True
    run = glue.build(signature())
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(view(signature()))
    assert state.calls == [] and run._warmed == set()
    tree = ast.parse(Path(glue.__file__).read_text())
    forbidden = {
        "empty",
        "empty_like",
        "new_empty",
        "to",
        "contiguous",
        "copy_",
        "synchronize",
    }
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in forbidden
    ]


def test_registry_facade_and_no_partial_launch_on_off_or_missing_metadata(monkeypatch):
    from sglang.kernels.plugin_kernels.nvidia.numeric import NumericRegistry
    from sglang.srt.environ import envs
    from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding

    from sgl_nvidia_mega.qwen3_8.spec import SPEC

    _, state = raw_spy(monkeypatch)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    providers.make_offer.cache_clear()
    sig = signature()
    try:
        built = SPEC.build(None, None, SimpleNamespace(numeric_signatures=(sig,)))
        assert len(built.offers) == 1 and registry.get(sig) is built.offers[0]
        assert numeric_binding.numeric_operator(sig, prepared=False) is None
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False):
            assert numeric_binding.numeric_operator(sig, prepared=True) is None
        assert state.calls == []
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
            run = numeric_binding.numeric_operator(sig, prepared=True)
            assert run is not None and run(view(sig)) is None
            monkeypatch.setattr(binding, "megakernel_enabled", lambda: False)
            assert numeric_binding.numeric_operator(sig, prepared=True) is None
        assert len(state.calls) == 1
    finally:
        providers.make_offer.cache_clear()


def test_warm_cache_does_not_hold_activations_or_state_owners(monkeypatch):
    _, state = raw_spy(monkeypatch)
    run = glue.build(signature())
    io = view(signature())
    owners = [weakref.ref(t) for t in (io.mixed_qkvz, io.checkpoint, io.ring.d)]
    run(io)
    del io
    assert all(ref() is None for ref in owners)
    state.capturing = True
    run(view(signature()))
    assert len(state.calls) == 2


def test_pointer_divisibility_change_is_a_cold_triton_key(monkeypatch):
    class PointerMetadata(torch.Tensor):
        def data_ptr(self):
            return self.address

    _, state = raw_spy(monkeypatch)
    io = view(signature())
    first = io.mixed_ba.as_subclass(PointerMetadata)
    first.address = 0x1000
    run = glue.build(signature())
    run(replace(io, mixed_ba=first))
    second = io.mixed_ba.as_subclass(PointerMetadata)
    second.address = 0x2002
    state.capturing = True
    with pytest.raises(RuntimeError, match="warmed before capture"):
        run(replace(io, mixed_ba=second))
    assert len(state.calls) == 1
