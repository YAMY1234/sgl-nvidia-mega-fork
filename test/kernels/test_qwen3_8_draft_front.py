# SPDX-License-Identifier: Apache-2.0
"""Pure BF16 primitive vs native, including strided QKV and changed graph inputs.

The lane's GPU runner injects CPU-compiled kernels into both Python wrappers;
normal pytest may compile before capture. No KV cache or model slot is changed.
"""
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.draft.front import build_front

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def inputs(rows, strided, mrope, cos_dtype, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    def randn(*shape, scale=1.):
        return (torch.randn(*shape, generator=g, device=device) * scale).bfloat16()
    if strided:
        qkv = randn(rows, 3584, scale=2.)
        q, k = qkv[:, :3072], qkv[:, 3072:3328]
    else:
        q, k = randn(rows, 3072, scale=2.), randn(rows, 256, scale=2.)
    q[0, :8] = -0.
    k[-1, 16:32] *= 40
    qw, kw = randn(256, scale=.1), randn(256, scale=.1)
    qw[0] = -1.
    pos = torch.arange(4096, device=device, dtype=torch.float32)[:, None]
    freq = 1. / (10000000 ** (torch.arange(0, 64, 2, device=device) / 64))
    cache = torch.cat([(pos * freq).cos(), (pos * freq).sin()], dim=-1).to(cos_dtype)
    positions = torch.randint(0, 4096, (3, rows) if mrope else (rows,), generator=g, device=device)
    # Interleaved sections [11, 11, 10], as the native rotary module constructs it.
    axis = torch.arange(32, device=device, dtype=torch.int64) % 3 if mrope else None
    return q, k, qw, kw, cache, positions, axis


def front():
    cfg = SimpleNamespace(hidden_size=2560, num_attention_heads=24,
                          num_key_value_heads=2, head_dim=256,
                          partial_rotary_factor=.25, rms_norm_eps=1e-6)
    primitive, reasons = build_front(cfg, 4)
    assert not reasons
    return primitive


def native(args):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )
    q, k, qw, kw, cache, pos, axis = args
    return fused_qk_gemma_rmsnorm_rope_gate(
        q, k, qw, kw, cache, pos, 1e-6, 6, 1, 256, 64, mrope_axis_map=axis,
    )


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def check_outputs(expected, actual):
    assert len(expected) == len(actual) == 3
    for a, b in zip(expected, actual):
        assert a.dtype == b.dtype == torch.bfloat16
        assert same(a, b)


@cuda
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16])
def test_native_bits_and_dynamic_graph(rows, strided, mrope, cos_dtype):
    run_case(rows, strided, mrope, cos_dtype, seed=35)


def run_case(rows, strided, mrope, cos_dtype, seed):
    args = inputs(rows, strided, mrope, cos_dtype, seed)
    primitive = front()
    before = [None if x is None else x.clone() for x in args]
    check_outputs(native(args), primitive(*args))
    for old, value in zip(before, args):
        if value is not None:
            assert same(old, value), "front mutated an input"
    # Keep the native strided input views; cloning them would hide the QKV stride.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = primitive(*args)
    for offset in [1, 2, 17]:
        fresh = inputs(rows, strided, mrope, cos_dtype, seed + offset)
        for dst, src in zip(args, fresh):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        check_outputs(native(args), actual)
    return {"rows": rows, "q_stride": args[0].stride(0), "k_stride": args[1].stride(0),
            "mrope": mrope, "cos_dtype": str(cos_dtype), "eager_bitwise": True,
            "graph_replays_bitwise": 3, "inputs_read_only": True}
