# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The kernels slot gdn_verify launches its own way, on one SM100-family GPU (GB300),
against SGLang's, bit for bit, at Qwen3.5's TP8 GatedDeltaNet shapes:

- the causal conv update with the 128-channel tile and the token-major output, against
  ``causal_conv1d_update`` (output, conv states and intermediate windows), on the
  contiguous split and on the token-major projection views;
- the gated RMSNorm writing static FP8, against SGLang's gated norm followed by the FP8
  linear's static input quantization (``static_quant_fp8``), with z as rows and as
  token-major views;
- the ReplaySSM verify launch, with and without the dependent launch, against
  ``gdn_replayssm_spec_decode``'s (output and ring entries).
"""

import pytest
import torch

H, K, HV, V = 2, 128, 8, 128  # per rank at TP8: 16 / 64 heads over 8
D = 2 * H * K + HV * V  # the conv's q / k / v channels
DRAFT, WIDTH, SLOTS = 7, 4, 9


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _projection(bs, views, g):
    """A verify step's mixed q / k / v rows [bs * 7, D]: a contiguous copy, or the
    token-major view of the q / k / v / z projection."""
    qkvz = torch.randn(bs * DRAFT, D + HV * V, device="cuda", generator=g).to(
        torch.bfloat16
    )
    return qkvz[:, :D] if views else qkvz[:, :D].contiguous()


@pytest.mark.parametrize("views", [False, True])
@pytest.mark.parametrize("block_n", [256, 128])
def test_conv_update_is_sglangs(views, block_n):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update

    g = torch.Generator(device="cuda").manual_seed(block_n + views)
    bs = 2
    rows = _projection(bs, views, g)
    weight = torch.randn(D, WIDTH, device="cuda", generator=g).to(torch.bfloat16)
    bias = torch.randn(D, device="cuda", generator=g).to(torch.bfloat16)
    state = torch.randn(SLOTS, D, WIDTH - 1, device="cuda", generator=g).to(
        torch.bfloat16
    )
    window = torch.zeros(
        SLOTS, DRAFT, D, WIDTH - 1, dtype=torch.bfloat16, device="cuda"
    )
    slots = torch.tensor([4, 7], dtype=torch.int32, device="cuda")
    inter = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    x = rows.view(bs, DRAFT, D).transpose(1, 2)
    want_state, want_window = state.clone(), window.clone()
    want = causal_conv1d_update(
        x,
        want_state,
        weight,
        bias,
        "silu",
        conv_state_indices=slots,
        intermediate_conv_window=want_window,
        intermediate_state_indices=inter,
    )
    got = conv_update(
        x,
        state,
        weight,
        bias,
        "silu",
        slots,
        window,
        inter,
        block_n=block_n,
        token_major_out=views,
    )
    assert torch.equal(got.contiguous(), want.contiguous())
    assert torch.equal(state, want_state) and torch.equal(window, want_window)
    # the slot reads the output back as token rows without a copy
    assert got.transpose(1, 2).is_contiguous()


@pytest.mark.parametrize("views", [False, True])
def test_norm_fp8_is_sglangs_norm_then_static_quant(views):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.qwen3_5.gdn.norm_fp8 import rms_norm_gated_fp8

    g = torch.Generator(device="cuda").manual_seed(int(views))
    tokens = 2 * DRAFT
    core = torch.randn(tokens * HV, V, device="cuda", generator=g).to(torch.bfloat16)
    qkvz = torch.randn(tokens, D + HV * V, device="cuda", generator=g).to(
        torch.bfloat16
    )
    z = qkvz[:, D:].view(tokens, HV, V)  # token-major view, rows with gaps
    if not views:
        z = z.reshape(-1, V)
    norm = RMSNorm(
        V, eps=1e-6, norm_before_gate=True, device="cuda", dtype=torch.bfloat16
    )
    norm.weight.data = torch.randn(V, device="cuda", generator=g).to(torch.bfloat16)
    scale = torch.tensor([0.02], device="cuda")
    want, _ = static_quant_fp8(
        norm(core, z).reshape(tokens, HV * V), scale, repeat_scale=False
    )
    got = rms_norm_gated_fp8(
        core, norm.weight, norm.eps, z, scale, activation=norm.activation
    )
    assert torch.equal(
        got.view(torch.uint8).reshape(tokens, -1), want.view(torch.uint8)
    )


@pytest.mark.parametrize("pdl", [False, True])
def test_replayssm_verify_is_sglangs(pdl):
    """The slot's verify launch (32-wide value tiles on 2 warps, and with or without the
    dependent launch) against gdn_replayssm_spec_decode's verify launch: the output and
    this step's ring entries."""
    try:
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )
    except ImportError:  # in fla/ before sgl-project/sglang#45112
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

    from sgl_nvidia_mega.qwen3_5.gdn import replayssm

    g = torch.Generator(device="cuda").manual_seed(3 + int(pdl))
    bs, ring_len, slots = 2, 16, 9
    tokens = bs * DRAFT

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=g).to(dtype)

    qkv = rand(tokens, D)  # strided q / k / v, as the slot's split leaves them
    q = qkv[:, : H * K].view(tokens, H, K)
    k = qkv[:, H * K : 2 * H * K].view(tokens, H, K)
    v = qkv[:, 2 * H * K :].view(tokens, HV, V)
    ba = rand(tokens, 2 * HV)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log, dt_bias = rand(HV, dtype=torch.float32), rand(HV, dtype=torch.float32)
    state = rand(slots, HV, V, K)
    ring = [
        rand(slots, HV, ring_len, V),
        rand(slots, H, ring_len, K),
        -torch.rand(slots, HV, ring_len, device="cuda", generator=g),
        rand(slots, HV, ring_len, V),
        rand(slots, H, ring_len, K),
    ]
    write_pos = torch.randint(0, ring_len - DRAFT, (slots,), device="cuda", generator=g)
    write_pos = write_pos.to(torch.int32)
    cache_base = torch.randint(0, ring_len, (slots,), device="cuda", generator=g)
    cache_base = cache_base.to(torch.int32)
    is_flush = torch.zeros(slots, dtype=torch.int8, device="cuda")
    qsl = torch.arange(0, tokens + 1, DRAFT, dtype=torch.int32, device="cuda")
    states = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    replay = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    want_ring, got_ring = [t.clone() for t in ring], [t.clone() for t in ring]
    want, got = (q.new_empty(tokens, HV, V) for _ in range(2))
    gdn_replayssm_spec_decode(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        checkpoint_state=state,
        d_cache=want_ring[0],
        k_cache=want_ring[1],
        g_cache=want_ring[2],
        rawv_cache=want_ring[3],
        rawk_cache=want_ring[4],
        beta_cache=None,
        out=want,
        query_start_loc=qsl,
        ssm_state_indices=states,
        replay_indices=replay,
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
        launch_mode="verify",
        block_v=32,
        num_warps=2,
    )
    replayssm.verify(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        got,
        state,
        *got_ring,
        None,
        qsl,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
        pdl=pdl,
    )
    assert torch.equal(got, want)
    assert all(torch.equal(x, y) for x, y in zip(got_ring, want_ring))
