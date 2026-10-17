# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The two-shot all-reduce + mHC hops (TwoShotMHC) on 4 SM100-family GPUs (GB300), at
49-384 rows: the attention hop, the MoE hop with and without the MXFP8 quant, and the
closed-seam MoE hop.

Bit for bit against SGLang's two-shot kernel when the SGLang tree on PYTHONPATH has it,
and against SGLang's one-shot fused kernel (streams and quant bytes; the normalized row
is reported, and its mismatches must stay within one BF16 ulp); within tolerance of a
torch reference; unchanged over 50 back-to-back calls of mixed row counts and over 20
replays of one CUDA graph with a late rank.

Run with ``python test/kernels/test_two_shot_mhc.py --num-gpu 4 -s``.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

HIDDEN, HC, TOP_K, EPS = 5120, 4, 6, 1e-6
ROWS = (49, 64, 96, 128, 152, 192, 256, 384)
ONE_SHOT_QUANT_ROWS = (
    128  # the one-shot quant epilogue addresses one 128-row scale tile
)


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield rank
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _inputs(rows, device, gen):
    p = rows * TOP_K
    idx = torch.randperm(p, device=device, generator=gen).int()
    idx[::7] = -1  # dropped slots
    weights = torch.rand(rows, TOP_K, device=device, generator=gen)
    shared = torch.randn(rows, HIDDEN, device=device, generator=gen).bfloat16() / 8
    # an all-zero row: its +0 words must be sent as -0, not taken for empty slots
    weights[0] = 0
    shared[0] = 0
    return SimpleNamespace(
        gemm2=torch.randn(p, HIDDEN, device=device, generator=gen).bfloat16() / 8,
        idx=idx,
        weights=weights,
        shared=shared,
        x=torch.randn(rows, HIDDEN, device=device, generator=gen).bfloat16() / 4,
        residual=torch.randn(rows, HC, HIDDEN, device=device, generator=gen).bfloat16(),
        post=torch.rand(rows, HC, device=device, generator=gen),
        comb=torch.randn(rows, HC, HC, device=device, generator=gen) * 0.1,
        pre=torch.rand(rows, HC, device=device, generator=gen),
        norm_weight=torch.randn(HIDDEN, device=device, generator=gen).bfloat16(),
    )


def _run(hop, a, variant):
    """Our hop: the outputs as a tuple of tensors."""
    if variant == "attn":
        return hop.attn_post(a.x, a.residual, a.post, a.comb, a.pre, a.norm_weight, EPS)
    moe = (a.gemm2, a.idx, a.weights, TOP_K, a.shared, a.residual, a.post, a.comb)
    if variant == "post":
        return (hop.moe_post_only(*moe),)
    if variant == "moe":
        return hop.moe_post(*moe, a.pre, a.norm_weight, EPS)
    rows = a.weights.shape[0]
    q = torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=a.x.device)
    sf = torch.empty(hop.scale_bytes(rows), dtype=torch.uint8, device=a.x.device)
    updated, normalized = hop.moe_post(
        *moe, a.pre, a.norm_weight, EPS, mxfp8_out=(q, sf)
    )
    return updated, normalized, q, sf


def _sglang(a, variant, world, two_shot):
    """SGLang's kernel, outputs in _run's order; None where it does not apply."""
    if two_shot:
        from sglang.kernels.ops.communication import all_reduce_mhc_highbatch as m

        post_only = m.moe_finalize_all_reduce_mhc_post_2shot
        norm = m.moe_finalize_all_reduce_mhc_post_combine_norm_2shot
        attn = m.all_reduce_mhc_post_combine_norm_2shot
        quant_rows = m.MAX_ROWS
    else:
        from sglang.kernels.ops.communication import all_reduce_mhc as m

        post_only = m.moe_finalize_all_reduce_mhc_post
        norm = m.moe_finalize_all_reduce_mhc_post_combine_norm
        attn = m.all_reduce_mhc_post_combine_norm
        quant_rows = ONE_SHOT_QUANT_ROWS
    mix = (a.residual, a.post, a.comb)
    moe = (a.gemm2, a.idx, a.weights, TOP_K, a.shared, *mix)
    if variant == "attn":
        return attn(a.x, *mix, a.pre, a.norm_weight, EPS, world_size=world)[1:]
    if variant == "post":
        return post_only(*moe, world_size=world)[1:]
    if variant == "moe":
        return norm(*moe, a.pre, a.norm_weight, EPS, world_size=world)[1:]
    if a.weights.shape[0] > quant_rows:
        return None
    if two_shot:
        return norm(*moe, a.pre, a.norm_weight, EPS, world_size=world, quant=True)[1:]
    return m.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
        *moe, a.pre, a.norm_weight, EPS, world_size=world
    )[1:]


def _torch(a, variant, group, world):
    """FP32 finalize, rank-order sum of the BF16 partials, then post / collapse / norm."""
    rows = a.weights.shape[0]
    if variant == "attn":
        part = a.x
    else:
        g = a.gemm2.float()
        acc = torch.zeros(rows, HIDDEN, device=g.device)
        idx = a.idx.view(rows, TOP_K).long()
        for k in range(TOP_K):
            valid = (idx[:, k] >= 0).unsqueeze(1)
            contrib = g[idx[:, k].clamp(min=0)] * a.weights[:, k : k + 1]
            acc += torch.where(valid, contrib, 0.0)
        part = (acc.bfloat16().float() + a.shared.float()).bfloat16()
    parts = [torch.empty_like(part) for _ in range(world)]
    dist.all_gather(parts, part.contiguous(), group=group)
    y = parts[0].float()
    for p in parts[1:]:
        y = y + p.float()
    y = y.bfloat16().float()
    streams = a.post.unsqueeze(-1) * y.unsqueeze(1) + (
        a.comb.unsqueeze(-1) * a.residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    if variant == "post":
        return (streams,)
    collapsed = (a.pre.unsqueeze(-1) * streams.float()).sum(dim=1).bfloat16().float()
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + EPS)
    return streams, (normed * a.norm_weight.float()).bfloat16()


def _scale_offsets(rows, device):
    r = torch.arange(rows, device=device)[:, None]
    g = torch.arange(HIDDEN // 32, device=device)[None, :]
    return (
        (r // 128) * (HIDDEN // 128) * 512
        + (g // 4) * 512
        + ((r % 32) * 4 + (r // 32) % 4) * 4
        + g % 4
    )


def _check_quant(normalized, q, sf):
    """MXFP8 of the normalized row: dequantizes within FP8 E4M3 rounding; pad rows zero."""
    rows = normalized.shape[0]
    offsets = _scale_offsets(rows, normalized.device)
    exponent = sf[offsets].int()
    scale = torch.exp2((exponent - 127).float()).repeat_interleave(32, dim=1)
    dequant = q.float() * scale
    ref = normalized.float()
    assert ((dequant - ref).abs() <= ref.abs() / 16 + scale / 512).all()
    pad = torch.ones(sf.numel(), dtype=torch.bool, device=sf.device)
    pad[offsets.flatten()] = False
    assert not sf[pad].any()


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _equal(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _bf16_ulps(a, b):
    """Max distance in BF16 ulps (same-sign values; ordered bit patterns)."""
    ia = a.contiguous().view(torch.int16).int()
    ib = b.contiguous().view(torch.int16).int()
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return int((ia - ib).abs().max())


def _have_two_shot_ref() -> bool:
    try:
        from sglang.kernels.ops.communication import (  # noqa: F401
            all_reduce_mhc_highbatch,
        )
        from sglang.srt.layers.communication import hop_highbatch  # noqa: F401
    except ImportError:
        return False
    return True


def _register_references(device):
    """SGLang's one-shot plane (sized for 384 rows) and, if present, its two-shot plane."""
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    cpu_group = ps.get_tp_group().cpu_group
    plane = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 << 20,
        max_push_blocks=512,
    )
    if plane.disabled:
        return None, False
    all_reduce_fusion.register_comm(plane.obj)
    two_shot = _have_two_shot_ref()
    if two_shot:
        from sglang.srt.layers.communication import hop_highbatch

        assert hop_highbatch.register_highbatch_push_plane(
            SimpleNamespace(group=cpu_group, device=device)
        )
    return plane, two_shot


def _max_over_ranks(value: float) -> float:
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ps.get_tp_group().device_group)
    return float(t.item())


VARIANTS = ("attn", "moe", "quant", "post")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)
def test_two_shot_mhc(rank):
    from sgl_nvidia_mega.deepseek_v41.sm103 import all_reduce_mhc
    from sgl_nvidia_mega.deepseek_v41.sm103.two_shot_mhc import MAX_ROWS, TwoShotMHC

    tp = ps.get_tp_group()
    world = tp.world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # a live deadline: the polls run the timer path without faulting
    all_reduce_mhc.set_wait_deadline(30.0)
    log = print if rank == 0 else (lambda *_, **__: None)
    with torch.inference_mode():
        one_shot_plane, two_shot_ref = _register_references(device)
        hop = TwoShotMHC(
            hidden_size=HIDDEN,
            hc_mult=HC,
            max_num_tokens=MAX_ROWS,
            top_k=TOP_K,
            device=device,
            group=tp.device_group,
        )
        assert hop.fits(49, HIDDEN) and hop.fits(384, HIDDEN)
        assert not hop.fits(48, HIDDEN) and not hop.fits(385, HIDDEN)
        assert not hop.fits(64, 7168)
        log(
            f"\nSGLang references: one-shot {'yes' if one_shot_plane else 'absent'}, "
            f"two-shot {'yes' if two_shot_ref else 'absent'}"
        )

        gen = torch.Generator(device=device).manual_seed(1000 + rank)
        expected = {}
        for rows in ROWS:
            a = _inputs(rows, device, gen)
            for variant in VARIANTS:
                got = _run(hop, a, variant)
                expected[(rows, variant)] = (a, got)
                line = f"rows={rows:3d} {variant:5s}"
                if variant == "quant":
                    _check_quant(got[1], got[2], got[3])
                # torch reference (streams, normalized)
                ref = _torch(a, variant, tp.device_group, world)
                errs = [
                    (g.float() - r.float()).abs().max().item() for g, r in zip(got, ref)
                ]
                torch.testing.assert_close(
                    got[0].float(), ref[0].float(), atol=3e-2, rtol=2e-2
                )
                if len(ref) > 1:
                    torch.testing.assert_close(
                        got[1].float(), ref[1].float(), atol=6e-2, rtol=3e-2
                    )
                line += " torch max_abs " + "/".join(
                    f"{_max_over_ranks(e):.3g}" for e in errs
                )
                if two_shot_ref:
                    sg = _sglang(a, variant, world, two_shot=True)
                    same = all(_equal(g, s) for g, s in zip(got, sg))
                    assert same, (
                        f"differs from SGLang's two-shot: rows={rows} {variant}"
                    )
                    line += " | two-shot bitwise"
                one = _sglang(a, variant, world, False) if one_shot_plane else None
                if one is None:
                    line += " | one-shot n/a"
                else:
                    assert _equal(got[0], one[0]), (
                        f"streams differ from one-shot: {rows} {variant}"
                    )
                    note = " | one-shot streams bitwise"
                    if len(got) > 1:
                        diff = _bits(got[1]).view(torch.int16) != _bits(one[1]).view(
                            torch.int16
                        )
                        n = int(diff.sum())
                        ulps = _bf16_ulps(got[1], one[1]) if n else 0
                        assert ulps <= 1, (rows, variant, ulps)
                        n_all = int(_max_over_ranks(n))
                        note += f", norm {'bitwise' if n_all == 0 else f'{n_all} elems differ (<=1 ulp)'}"
                    if len(got) > 2:
                        q_same = _equal(got[2], one[2]) and _equal(got[3], one[3])
                        # quant bytes may follow a differing normalized element only
                        if not q_same:
                            rows_diff = diff.any(dim=1)
                            qd = (
                                (_bits(got[2]) != _bits(one[2]))
                                .view(rows, HIDDEN)
                                .any(dim=1)
                            )
                            assert not (qd & ~rows_diff).any(), (
                                rows,
                                "quant differs off the norm diffs",
                            )
                        q_all = _max_over_ranks(0.0 if q_same else 1.0) == 0.0
                        note += f", quant {'bitwise' if q_all else 'differs where norm differs'}"
                    line += note
                log(line, flush=True)

        # 50 back-to-back calls, mixed row counts and variants: epochs and markers cycle
        order = [(rows, v) for v in VARIANTS for rows in ROWS]
        seq = [order[(i * 7) % len(order)] for i in range(50)]
        outs = []
        for i, key in enumerate(seq):
            if rank == i % world:
                torch.cuda._sleep(5_000)
            outs.append(_run(hop, expected[key][0], key[1]))
        torch.cuda.synchronize()
        for key, got in zip(seq, outs):
            for g, e in zip(got, expected[key][1]):
                assert _equal(g, e), f"back-to-back differs: {key}"
        log("50 back-to-back calls: bitwise equal to the first calls", flush=True)

        # one CUDA graph of hops at several row counts, replayed with a late rank
        keys = [(rows, v) for rows in (49, 192, 384, 128) for v in VARIANTS]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            captured = [_run(hop, expected[k][0], k[1]) for k in keys]
        torch.cuda.current_stream().wait_stream(stream)
        for replay in range(20):
            for got in captured:
                for g in got:
                    g.view(torch.uint8).zero_()
            if rank == replay % world:
                torch.cuda._sleep(50_000)
            graph.replay()
            torch.cuda.synchronize()
            for key, got in zip(keys, captured):
                for g, e in zip(got, expected[key][1]):
                    assert _equal(g, e), f"graph replay {replay} differs: {key}"
            # eager calls between replays keep the epochs moving
            _run(hop, expected[(96, "moe")][0], "moe")
        torch.cuda.synchronize()
        log("CUDA graph (16 hops) x 20 replays: bitwise equal", flush=True)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=1800)
