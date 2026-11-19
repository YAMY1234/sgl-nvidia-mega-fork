# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

D, RING, T = 128, 16, 7
DEP4 = (16, 64)
TEP8 = (2, 8)


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        from sglang.kernels.ops.attention.linear import (  # noqa: F401
            gdn_replayssm_spec_decode,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM100-family GPU, CuTe DSL and SGLang"
)


class _Ops:
    def __init__(self, shape, bs, history, seed, flush=False):
        H, HV = shape
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.H, self.HV, self.bs = H, HV, bs
        slots, tok, Dq = bs + 3, bs * T, (2 * H + HV) * D

        def rand(*s, scale=1.0):
            return (torch.randn(*s, device="cuda", generator=g) * scale).to(
                torch.bfloat16
            )

        qkvz = rand(tok, Dq + HV * D)
        self.qkv, self.z = qkvz[:, :Dq], qkvz[:, Dq:].view(tok, HV, D)
        ba = rand(tok, 2 * HV)
        self.b, self.a = ba[:, :HV], ba[:, HV:]
        self.A_log = torch.rand(HV, device="cuda", generator=g)
        self.dt_bias = torch.rand(HV, device="cuda", generator=g)
        self.state = rand(1, slots, HV, D, D, scale=0.05)
        self.ring = [
            rand(1, slots, HV, RING, D, scale=0.05),
            rand(1, slots, H, RING, D, scale=0.09),
            -torch.rand(1, slots, HV, RING, device="cuda", generator=g) * 0.5,
            rand(1, slots, HV, RING, D, scale=1e-4),
            rand(1, slots, H, RING, D, scale=1e-4),
        ]
        self.states = torch.randperm(slots, device="cuda", generator=g)[:bs].to(
            torch.int32
        )
        self.replay = torch.randperm(slots, device="cuda", generator=g)[:bs].to(
            torch.int32
        )
        if history == "fold":
            self.write_pos = torch.randint(
                1, T + 1, (slots,), device="cuda", generator=g
            ).to(torch.int32)
        else:
            hi = RING - T + 1 if history == "hist" else 1
            self.write_pos = torch.randint(
                0, hi, (slots,), device="cuda", generator=g
            ).to(torch.int32)
        self.cache_base = torch.randint(
            0, RING, (slots,), device="cuda", generator=g
        ).to(torch.int32)
        self.is_flush = torch.full(
            (slots,), int(flush), dtype=torch.int8, device="cuda"
        )
        self.weight = rand(D)
        self.scale = torch.tensor([0.02], device="cuda")

    def clone(self):
        o = _Ops.__new__(_Ops)
        o.__dict__.update(self.__dict__)
        o.state = self.state.clone()
        o.ring = [t.clone() for t in self.ring]
        o.write_pos, o.cache_base, o.is_flush = (
            self.write_pos.clone(),
            self.cache_base.clone(),
            self.is_flush.clone(),
        )
        return o


def _sglang_commit(o):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_circular,
    )

    commit_gdn_replayssm_circular(
        checkpoint_state=o.state,
        d_cache=o.ring[0],
        k_cache=o.ring[1],
        g_cache=o.ring[2],
        d_residual_cache=o.ring[3],
        k_residual_cache=o.ring[4],
        state_batch_indices=o.states,
        replay_indices=o.replay,
        write_pos=o.write_pos,
        cache_base=o.cache_base,
        is_flush=o.is_flush,
        accept_lens=torch.ones(o.bs, dtype=torch.int32, device="cuda"),
        null_block_id=-1,
    )


def _sglang_verify(o):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )

    H, HV, tok = o.H, o.HV, o.bs * T
    out = torch.zeros(tok, HV, D, dtype=torch.bfloat16, device="cuda")
    gdn_replayssm_spec_decode(
        q=o.qkv[:, : H * D].view(tok, H, D),
        k=o.qkv[:, H * D : 2 * H * D].view(tok, H, D),
        v=o.qkv[:, 2 * H * D :].view(tok, HV, D),
        a=o.a,
        b=o.b,
        A_log=o.A_log,
        dt_bias=o.dt_bias,
        checkpoint_state=o.state[0],
        d_cache=o.ring[0][0],
        k_cache=o.ring[1][0],
        g_cache=o.ring[2][0],
        rawv_cache=o.ring[3][0],
        rawk_cache=o.ring[4][0],
        beta_cache=None,
        out=out,
        query_start_loc=torch.arange(0, tok + 1, T, dtype=torch.int32, device="cuda"),
        ssm_state_indices=o.states,
        replay_indices=o.replay,
        write_pos=o.write_pos,
        cache_base=o.cache_base,
        is_flush=o.is_flush,
        max_cache_len=RING,
        max_spec_len=T,
        scale=D**-0.5,
        null_block_id=-1,
        launch_mode="verify",
    )
    return out


def _tc(o, fold=False, replay=None):
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core_tc import core_tc

    tok = o.bs * T
    out = torch.zeros(tok, o.HV, D, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(tok, o.HV * D, dtype=torch.float8_e4m3fn, device="cuda")
    core_tc(
        o.qkv,
        o.z,
        o.a,
        o.b,
        o.A_log,
        o.dt_bias,
        o.state[0],
        *[t[0] for t in o.ring],
        o.states,
        o.replay if replay is None else replay,
        o.write_pos,
        o.cache_base,
        o.is_flush,
        o.weight,
        o.scale,
        1e-6,
        y,
        rows_per_request=T,
        num_k_heads=o.H,
        o_out=out,
        fold=fold,
    )
    return out, y


@pytest.mark.parametrize("bs", [8, 32, 40, 47])
@pytest.mark.parametrize("history", ["empty", "hist"])
def test_dep4_launch_is_sglangs_verify_numerically(bs, history):
    o = _Ops(DEP4, bs, history, seed=97 * bs + len(history))
    s, t = o.clone(), o.clone()
    so = _sglang_verify(s)
    to, _ = _tc(t)
    torch.cuda.synchronize()
    assert (to.float() - so.float()).norm() <= 4e-3 * so.float().norm()


@pytest.mark.parametrize("shape, bs", [(DEP4, 8), (DEP4, 36), (DEP4, 47), (TEP8, 64)])
def test_fold_is_the_commit_then_the_verify(shape, bs):
    o = _Ops(shape, bs, "fold", seed=31 * bs + shape[1], flush=True)
    s, t = o.clone(), o.clone()
    _sglang_commit(s)  # folds, then base += write_pos, write_pos = 0, is_flush = 0
    so = _sglang_verify(s)
    to, _ = _tc(t, fold=True)
    torch.cuda.synchronize()
    idx = o.states.long()
    # the folded checkpoint: SGLang's commit's, up to FP32 summation order
    eq = (t.state[0, idx] == s.state[0, idx]).float().mean().item()
    assert eq >= 0.99, eq
    assert (to.float() - so.float()).norm() <= 4e-3 * so.float().norm()
    assert torch.equal(t.write_pos, o.write_pos) and torch.equal(t.is_flush, o.is_flush)
    other = torch.ones(o.state.shape[1], dtype=torch.bool, device="cuda")
    other[idx] = False
    assert torch.equal(t.state[0, other], o.state[0, other])


@pytest.mark.parametrize("shape, bs", [(DEP4, 36), (TEP8, 64)])
def test_fold_build_without_flush_rows_is_the_plain_launch(shape, bs):
    o = _Ops(shape, bs, "hist", seed=5 + bs)
    a, b = o.clone(), o.clone()
    ao, ay = _tc(a)
    bo, by = _tc(b, fold=True)
    torch.cuda.synchronize()
    assert torch.equal(ao, bo) and torch.equal(
        ay.view(torch.uint8), by.view(torch.uint8)
    )
    assert all(torch.equal(x, y) for x, y in zip(a.ring, b.ring)) and torch.equal(
        a.state, b.state
    )


def _fold_build(P):
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5.runtime.executor import Qwen35FoldVerify

    lm = SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))
    built = Qwen35FoldVerify(lm, {}, {}, (T,))
    built._mamba = SimpleNamespace(
        ring_write_pos=P.write_pos,
        ring_cache_base=P.cache_base,
        ring_is_flush=P.is_flush,
        ring_d=P.ring[0],
        conv=(),
        intermediate_conv_window=(),
    )
    return built


def test_fold_protocol_over_steps():
    from inspect import signature
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_spec,
    )

    bs, steps = 8, 8
    o = _Ops(DEP4, bs, "empty", seed=1234)
    for t in o.ring:
        t.zero_()
    o.write_pos.zero_()
    o.cache_base.zero_()
    A, B = o.clone(), o.clone()
    fold_build = _fold_build(B)
    params = signature(commit_gdn_replayssm_spec).parameters
    slots = {"state_batch_indices": A.states} if "state_batch_indices" in params else {}
    g = torch.Generator(device="cuda").manual_seed(77)
    for step in range(steps):
        x = _Ops(DEP4, bs, "empty", seed=9000 + step)  # this step's rows
        for P in (A, B, o):
            P.qkv, P.z, P.a, P.b = x.qkv, x.z, x.a, x.b
        acc = torch.randint(1, T + 1, (bs,), device="cuda", generator=g).to(torch.int32)
        _sglang_verify(A)
        _tc(B, fold=True)
        commit_gdn_replayssm_spec(
            write_pos=A.write_pos,
            cache_base=A.cache_base,
            is_flush=A.is_flush,
            num_accepted=acc,
            replay_indices=A.replay,
            max_cache_len=RING,
            max_spec_len=T,
            fold_every_commit=True,
            null_block_id=-1,
            **slots,
        )
        _sglang_commit(A)
        io = SimpleNamespace(
            state_slots=B.states,
            track_slots=None,
            request_rows=B.replay,
            accept_lens=acc,
            last_step=acc - 1,
            src_indices_raw=None,
        )
        assert fold_build.commit_recurrent_state(io)
        torch.cuda.synchronize()
    _sglang_commit(B)
    torch.cuda.synchronize()


def test_fold_graph_replay_equals_eager():
    o = _Ops(DEP4, 16, "fold", seed=3, flush=True)
    e = o.clone()
    eo, ey = _tc(e, fold=True)
    g_ = o.clone()
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core_tc import core_tc

    out = torch.zeros_like(eo)
    y = torch.empty_like(ey)

    def run():
        core_tc(
            g_.qkv,
            g_.z,
            g_.a,
            g_.b,
            g_.A_log,
            g_.dt_bias,
            g_.state[0],
            *[t[0] for t in g_.ring],
            g_.states,
            g_.replay,
            g_.write_pos,
            g_.cache_base,
            g_.is_flush,
            g_.weight,
            g_.scale,
            1e-6,
            y,
            rows_per_request=T,
            num_k_heads=g_.H,
            o_out=out,
            fold=True,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    g_.state.copy_(o.state)
    for t, t0 in zip(g_.ring, o.ring):
        t.copy_(t0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eo) and torch.equal(
        y.view(torch.uint8), ey.view(torch.uint8)
    )
    assert torch.equal(g_.state, e.state) and all(
        torch.equal(a, b) for a, b in zip(g_.ring, e.ring)
    )
