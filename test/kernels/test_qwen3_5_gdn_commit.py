# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

LAYERS, HV, H, D, T, RING = 6, 8, 2, 128, 7, 16


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _ops():
    from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as g

    return g


def _setup(B, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    slots_n, ring_n = B + 12, B + 4
    state = (
        torch.randn(LAYERS, slots_n, HV, D, D, device="cuda", generator=gen) * 0.05
    ).to(torch.bfloat16)
    rings = (
        torch.randn(LAYERS, ring_n, HV, RING, D, device="cuda", generator=gen).to(
            torch.bfloat16
        ),
        torch.zeros(LAYERS, ring_n, HV, RING, D, device="cuda", dtype=torch.bfloat16),
        torch.randn(LAYERS, ring_n, H, RING, D, device="cuda", generator=gen).to(
            torch.bfloat16
        ),
        torch.zeros(LAYERS, ring_n, H, RING, D, device="cuda", dtype=torch.bfloat16),
        -torch.rand(LAYERS, ring_n, HV, RING, device="cuda", generator=gen),
    )
    slots = torch.randperm(slots_n - 1, generator=cpu)[:B].tolist()
    slots[-1] = -1  # a padded row
    track_pool = [s for s in range(slots_n - 1) if s not in slots][:4]
    rows = torch.randperm(ring_n - 1, generator=cpu)[:B].tolist()
    return gen, cpu, state, rings, slots, track_pool, rows, ring_n


def _cursors(ring_n):
    return (
        torch.zeros(ring_n, dtype=torch.int32, device="cuda"),
        torch.zeros(ring_n, dtype=torch.int32, device="cuda"),
        torch.zeros(ring_n, dtype=torch.int8, device="cuda"),
    )


def _ring_write(rings, cursors, rows, gen):
    d, dr, k, kr, gg = rings
    write_pos, cache_base, _ = cursors
    B = rows.shape[0]
    t = torch.arange(T, device="cuda")
    phys = (cache_base[rows].long()[:, None] + write_pos[rows].long()[:, None] + t) & (
        RING - 1
    )
    R = rows[:, None].expand(B, T)
    upd = torch.randn(B, T, LAYERS, HV, D, device="cuda", generator=gen) * 0.2
    hi = upd.to(torch.bfloat16)
    d[:, R, :, phys, :] = hi
    dr[:, R, :, phys, :] = (upd - hi.float()).to(torch.bfloat16)
    kk = torch.randn(B, T, LAYERS, H, D, device="cuda", generator=gen)
    kk = kk / kk.norm(dim=-1, keepdim=True)
    khi = kk.to(torch.bfloat16)
    k[:, R, :, phys, :] = khi
    kr[:, R, :, phys, :] = (kk - khi.float()).to(torch.bfloat16)
    gg[:, R, :, phys] = -torch.exp(
        torch.randn(B, T, LAYERS, HV, device="cuda", generator=gen) - 1.5
    ) * torch.nn.functional.softplus(
        torch.randn(B, T, LAYERS, HV, device="cuda", generator=gen) * 2.0
    )


def _written(cursors, rows, slots, track):
    """The checkpoint slots a commit writes: each flushed request's and each tracked one."""
    is_flush = cursors[2]
    out = set()
    for i, (s, r) in enumerate(zip(slots, rows)):
        if s < 0:
            continue
        if int(is_flush[r]) != 0:
            out.add(s)
        if track is not None and track[0][i] >= 0 and track[1][i] >= 0:
            out.add(int(track[0][i]))
    return out


def _advance(cursors, acc, rows, sidx):
    _ops().commit_gdn_replayssm_spec(
        write_pos=cursors[0],
        cache_base=cursors[1],
        is_flush=cursors[2],
        num_accepted=acc,
        replay_indices=rows,
        state_batch_indices=sidx,
        max_cache_len=RING,
        max_spec_len=T,
        fold_every_commit=True,
        null_block_id=-1,
    )


def _track(cpu, acc_l, track_pool, B):
    ti, ts = [-1] * B, [-1] * B
    for j in range(min(3, B - 1)):
        ti[j] = track_pool[j]
        ts[j] = int(torch.randint(0, int(acc_l[j]), (1,), generator=cpu))
    return ti, ts


def _commit_rounds(B, rounds, with_track, index_dtype):
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import commit as kernel

    g = _ops()
    gen, cpu, state, rings, slots, track_pool, rows_l, ring_n = _setup(B, B)
    cur_a, cur_b = _cursors(ring_n), _cursors(ring_n)
    st_a, st_b = state.clone(), state.clone()
    sidx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    rows = torch.tensor(rows_l, dtype=index_dtype, device="cuda")
    written = set()
    for rnd in range(rounds):
        _ring_write(rings, cur_a, rows, gen)
        acc_l = torch.randint(1, T + 1, (B,), generator=cpu)
        acc = acc_l.to(torch.int32).cuda()
        track = (
            _track(cpu, acc_l, track_pool, B) if with_track and rnd % 2 == 0 else None
        )
        for cur in (cur_a, cur_b):
            _advance(cur, acc, rows, sidx)
        written |= _written(cur_a, rows_l, slots, track)
        tt = (
            None
            if track is None
            else tuple(torch.tensor(x, dtype=index_dtype, device="cuda") for x in track)
        )
        g.commit_gdn_replayssm_circular(
            checkpoint_state=st_a,
            d_cache=rings[0],
            k_cache=rings[2],
            g_cache=rings[4],
            d_residual_cache=rings[1],
            k_residual_cache=rings[3],
            state_batch_indices=sidx,
            replay_indices=rows,
            write_pos=cur_a[0],
            cache_base=cur_a[1],
            is_flush=cur_a[2],
            accept_lens=acc,
            mamba_track_indices=None if tt is None else tt[0],
            mamba_steps_to_track=None if tt is None else tt[1],
            null_block_id=-1,
        )
        kernel.commit(
            checkpoint=st_b,
            ring_d=rings[0],
            ring_dlo=rings[1],
            ring_k=rings[2],
            ring_klo=rings[3],
            ring_g=rings[4],
            state_indices=sidx,
            ring_rows=rows,
            write_pos=cur_b[0],
            cache_base=cur_b[1],
            is_flush=cur_b[2],
            accept_lens=acc,
            track_indices=None if tt is None else tt[0],
            track_steps=None if tt is None else tt[1],
        )
        torch.cuda.synchronize()
        for c_a, c_b in zip(cur_a, cur_b):
            assert torch.equal(c_a, c_b)
    for s in set(range(state.shape[1])) - written:
        assert torch.equal(st_b[:, s].view(torch.int16), state[:, s].view(torch.int16))


@pytest.mark.parametrize(
    "B, rounds, with_track, index_dtype",
    [
        (6, 24, True, torch.int64),
        (5, 16, False, torch.int32),
        (2, 8, True, torch.int64),
    ],
)
def test_commit_keeps_sglangs_cursors_and_untouched_slots(
    B, rounds, with_track, index_dtype
):
    _commit_rounds(B, rounds, with_track, index_dtype)


def test_the_provider_commits_as_sglangs_after_verify():
    from sgl_nvidia_mega.qwen3_5.gdn.commit import RecurrentCommit, pool_reasons

    g = _ops()
    B, rounds = 6, 12
    gen, cpu, state, rings, slots, track_pool, rows_l, ring_n = _setup(B, 7)
    cur_a, cur_b = _cursors(ring_n), _cursors(ring_n)
    st_a, st_b = state.clone(), state.clone()
    sidx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    rows = torch.tensor(rows_l, dtype=torch.int64, device="cuda")
    spec_state = SimpleNamespace(
        temporal=st_a,
        replayssm_d=rings[0],
        replayssm_rawv=rings[1],
        replayssm_k=rings[2],
        replayssm_rawk=rings[3],
        replayssm_g=rings[4],
        conv=[],
        intermediate_conv_window=[],
    )
    pools = SimpleNamespace(
        conv=(),
        intermediate_conv_window=(),
        temporal=st_b,
        ring_d=rings[0],
        ring_k=rings[2],
        ring_g=rings[4],
        ring_rawv=rings[1],
        ring_rawk=rings[3],
        ring_beta=None,
        ring_write_pos=cur_b[0],
        ring_cache_base=cur_b[1],
        ring_is_flush=cur_b[2],
    )
    assert pool_reasons(pools) == []
    provider = RecurrentCommit(torch.cuda.current_device())
    provider.bind(pools)
    for rnd in range(rounds):
        _ring_write(rings, cur_a, rows, gen)
        acc_l = torch.randint(1, T + 1, (B,), generator=cpu)
        acc = acc_l.to(torch.int32).cuda()
        track = _track(cpu, acc_l, track_pool, B) if rnd % 2 else None
        tt = (
            None
            if track is None
            else tuple(torch.tensor(x, dtype=torch.int64, device="cuda") for x in track)
        )
        last = (acc - 1).to(torch.int32)
        g.commit_gdn_replayssm_after_verify(
            spec_state=spec_state,
            write_pos=cur_a[0],
            cache_base=cur_a[1],
            is_flush=cur_a[2],
            state_batch_indices=sidx,
            replay_indices=rows,
            accept_lens=acc,
            last_correct_step_indices=last,
            fold_every_commit=True,
            max_spec_len=T,
            mamba_track_indices=None if tt is None else tt[0],
            mamba_steps_to_track=None if tt is None else tt[1],
            null_block_id=-1,
        )
        io = SimpleNamespace(
            state_slots=sidx,
            track_slots=None if tt is None else tt[0],
            request_rows=rows,
            accept_lens=acc,
            last_step=last,
            track_steps=None if tt is None else tt[1],
            draft_token_num=T,
            src_indices_raw=None,
        )
        assert provider(io) is True
        torch.cuda.synchronize()
        for c_a, c_b in zip(cur_a, cur_b):
            assert torch.equal(c_a, c_b)


def test_graph_replays_equal_eager_launches():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import commit as kernel

    B, rounds, warmups = 5, 12, 2
    gen, cpu, state, rings, slots, track_pool, rows_l, ring_n = _setup(B, 2)
    cur_e, cur_g = _cursors(ring_n), _cursors(ring_n)
    st_e, st_g = state.clone(), state.clone()
    sidx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    rows = torch.tensor(rows_l, dtype=torch.int64, device="cuda")
    acc_s = torch.ones(B, dtype=torch.int32, device="cuda")
    ti_s = torch.full((B,), -1, dtype=torch.int64, device="cuda")
    ts_s = torch.full((B,), -1, dtype=torch.int64, device="cuda")

    def step(st, cur, acc, ti, ts):
        _advance(cur, acc, rows, sidx)
        kernel.commit(
            checkpoint=st,
            ring_d=rings[0],
            ring_dlo=rings[1],
            ring_k=rings[2],
            ring_klo=rings[3],
            ring_g=rings[4],
            state_indices=sidx,
            ring_rows=rows,
            write_pos=cur[0],
            cache_base=cur[1],
            is_flush=cur[2],
            accept_lens=acc,
            track_indices=ti,
            track_steps=ts,
        )

    side = torch.cuda.Stream()
    graph = None
    for rnd in range(warmups + rounds):
        _ring_write(rings, cur_e, rows, gen)
        acc = torch.randint(1, T + 1, (B,), generator=cpu).to(torch.int32)
        ti = torch.full((B,), -1, dtype=torch.int64)
        ts = torch.full((B,), -1, dtype=torch.int64)
        if rnd % 2 == 0:
            ti[0] = track_pool[0]
            ts[0] = int(torch.randint(0, int(acc[0]), (1,), generator=cpu))
        acc, ti, ts = acc.cuda(), ti.cuda(), ts.cuda()
        step(st_e, cur_e, acc, ti, ts)
        acc_s.copy_(acc)
        ti_s.copy_(ti)
        ts_s.copy_(ts)
        if rnd < warmups:
            step(st_g, cur_g, acc_s, ti_s, ts_s)
        else:
            if graph is None:
                snap = st_g.clone()
                graph = torch.cuda.CUDAGraph()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(
                    graph,
                    pool=torch.cuda.graph_pool_handle(),
                    stream=side,
                    capture_error_mode="thread_local",
                ):
                    step(st_g, cur_g, acc_s, ti_s, ts_s)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                # capture launches nothing
                assert torch.equal(snap.view(torch.int16), st_g.view(torch.int16))
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                graph.replay()
            torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        assert torch.equal(st_e.view(torch.int16), st_g.view(torch.int16))
        for c_e, c_g in zip(cur_e, cur_g):
            assert torch.equal(c_e, c_g)
