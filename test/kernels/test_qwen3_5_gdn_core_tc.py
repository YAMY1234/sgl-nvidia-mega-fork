# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

H, K, HV, V, T, RING = 2, 128, 8, 128, 7, 16
D = 2 * H * K + HV * V


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM100-family GPU and CuTe DSL"
)


def _operands(bs, history, pad, flush, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots, tokens = bs + 3, bs * T

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(
            torch.bfloat16
        )

    qkvz = rand(tokens, D + HV * V)
    o = dict(qkv=qkvz[:, :D], z=qkvz[:, D:].view(tokens, HV, V))
    ba = rand(tokens, 2 * HV)
    o["b"], o["a"] = ba[:, :HV], ba[:, HV:]
    o["A_log"] = torch.rand(HV, device="cuda", generator=g)
    o["dt_bias"] = torch.rand(HV, device="cuda", generator=g)
    o["state"] = rand(slots, HV, V, K, scale=0.05)
    o["ring"] = [
        rand(slots, HV, RING, V, scale=0.05),
        rand(slots, H, RING, K, scale=0.09),
        -torch.rand(slots, HV, RING, device="cuda", generator=g) * 0.5,
        rand(slots, HV, RING, V, scale=1e-4),
        rand(slots, H, RING, K, scale=1e-4),
    ]
    states = torch.randperm(slots, device="cuda", generator=g)[:bs].to(torch.int32)
    replay = torch.randperm(slots, device="cuda", generator=g)[:bs].to(torch.int32)
    if pad:
        states[-1] = -1
    o["states"], o["replay"] = states, replay
    o["write_pos"] = torch.randint(
        0, RING - T + 1 if history else 1, (slots,), device="cuda", generator=g
    ).to(torch.int32)
    o["cache_base"] = torch.randint(0, RING, (slots,), device="cuda", generator=g).to(
        torch.int32
    )
    o["is_flush"] = torch.zeros(slots, dtype=torch.int8, device="cuda")
    if flush:
        o["is_flush"][replay[0]] = 1
    o["weight"] = rand(V)
    o["scale"] = torch.tensor([0.02], device="cuda")
    return o


def _stock(o, ring):
    from sgl_nvidia_mega.qwen3_5.gdn import replayssm
    from sgl_nvidia_mega.qwen3_5.gdn.norm_fp8 import rms_norm_gated_fp8

    qkv = o["qkv"]
    tokens = qkv.shape[0]
    qsl = torch.arange(0, tokens + 1, T, dtype=torch.int32, device="cuda")
    out = torch.zeros(tokens, HV, V, dtype=torch.bfloat16, device="cuda")
    replayssm.verify(
        qkv[:, : H * K].view(tokens, H, K),
        qkv[:, H * K : 2 * H * K].view(tokens, H, K),
        qkv[:, 2 * H * K :].view(tokens, HV, V),
        o["a"],
        o["b"],
        o["A_log"],
        o["dt_bias"],
        out,
        o["state"],
        *ring,
        None,
        qsl,
        o["states"],
        o["replay"].long(),
        o["write_pos"],
        o["cache_base"],
        o["is_flush"],
        max_cache_len=RING,
        max_spec_len=T,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
    )
    y = rms_norm_gated_fp8(out.view(-1, V), o["weight"], 1e-6, o["z"], o["scale"])
    return out, y.view(tokens, -1)


def _tc(o, ring, replay=None, early=False):
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core_tc import core_tc

    tokens = o["qkv"].shape[0]
    out = torch.zeros(tokens, HV, V, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(tokens, HV * V, dtype=torch.float8_e4m3fn, device="cuda")
    core_tc(
        o["qkv"],
        o["z"],
        o["a"],
        o["b"],
        o["A_log"],
        o["dt_bias"],
        o["state"],
        *ring,
        o["states"],
        o["replay"] if replay is None else replay,
        o["write_pos"],
        o["cache_base"],
        o["is_flush"],
        o["weight"],
        o["scale"],
        1e-6,
        y,
        rows_per_request=T,
        num_k_heads=H,
        early_dependents=early,
        o_out=out,
    )
    return out, y


def _live_rows(o):
    tokens = o["qkv"].shape[0]
    live = torch.ones(tokens, dtype=torch.bool, device="cuda")
    for n in range(tokens // T):
        rid = int(o["replay"][n])
        if int(o["states"][n]) < 0 or int(o["is_flush"][rid]):
            live[n * T : (n + 1) * T] = False
    return live


@pytest.mark.parametrize("bs", [13, 16, 64, 96, 128])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_core_tc_is_gdn_verify_numerically(pad, history, bs):
    o = _operands(bs, history, pad, flush=False, seed=100 * bs + 10 * history + pad)
    want_ring = [t.clone() for t in o["ring"]]
    got_ring = [t.clone() for t in o["ring"]]
    o_want, y_want = _stock(o, want_ring)
    o_got, y_got = _tc(o, got_ring)
    torch.cuda.synchronize()
    live = _live_rows(o)
    assert not y_got[~live].float().any() and not o_got[~live].float().any()
    ow, og = o_want[live].float(), o_got[live].float()
    assert (og - ow).norm() <= 4e-3 * ow.norm()
    same = y_got.view(torch.uint8)[live] == y_want.view(torch.uint8)[live]
    assert same.float().mean() >= 0.90
    yw, yg = y_want[live].float(), y_got[live].float()
    assert (yg - yw).norm() <= 2e-2 * yw.norm()
    for hi, lo, tol in ((0, 3, 5e-3), (1, 4, 1e-5)):
        w = want_ring[hi].float() + want_ring[lo].float()
        g = got_ring[hi].float() + got_ring[lo].float()
        changed = w - (o["ring"][hi].float() + o["ring"][lo].float())
        assert (g - w).norm() <= tol * max(changed.norm().item(), 1e-30), (
            hi,
            (g - w).norm(),
            changed.norm(),
        )
    assert torch.allclose(got_ring[2], want_ring[2], rtol=0, atol=1e-5)
    for n in range(bs):
        rid = int(o["replay"][n])
        if int(o["states"][n]) < 0 or int(o["is_flush"][rid]):
            assert all(
                torch.equal(gr[rid], r0[rid]) for gr, r0 in zip(got_ring, o["ring"])
            )


def test_core_tc_graph_replays_and_int64_indices_equal_eager():
    o = _operands(64, True, True, True, seed=5)
    ring_e = [t.clone() for t in o["ring"]]
    o_e, y_e = _tc(o, ring_e)
    ring_l = [t.clone() for t in o["ring"]]
    o_l, y_l = _tc(o, ring_l, replay=o["replay"].long(), early=True)
    torch.cuda.synchronize()
    assert torch.equal(y_e.view(torch.uint8), y_l.view(torch.uint8)) and torch.equal(
        o_e, o_l
    )
    assert all(torch.equal(a, b) for a, b in zip(ring_e, ring_l))
    live = _live_rows(o)
    assert int((~live).sum()) == 2 * T
    assert not y_e[~live].float().any() and not o_e[~live].float().any()
    for n in (0, 63):
        rid = int(o["replay"][n])
        assert all(torch.equal(a[rid], r0[rid]) for a, r0 in zip(ring_e, o["ring"]))
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core_tc import core_tc

    ring_g = [t.clone() for t in o["ring"]]
    y_g = torch.empty_like(y_e)
    o_g = torch.zeros_like(o_e)

    def run():
        core_tc(
            o["qkv"],
            o["z"],
            o["a"],
            o["b"],
            o["A_log"],
            o["dt_bias"],
            o["state"],
            *ring_g,
            o["states"],
            o["replay"],
            o["write_pos"],
            o["cache_base"],
            o["is_flush"],
            o["weight"],
            o["scale"],
            1e-6,
            y_g,
            rows_per_request=T,
            num_k_heads=H,
            o_out=o_g,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for t, r0 in zip(ring_g, o["ring"]):
        t.copy_(r0)
    y_g.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_g.view(torch.uint8), y_e.view(torch.uint8)) and torch.equal(
        o_g, o_e
    )
    assert all(torch.equal(a, b) for a, b in zip(ring_g, ring_e))
