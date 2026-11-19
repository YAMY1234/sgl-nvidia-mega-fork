# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

K, V, WIDTH, RING = 128, 128, 4, 16


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _heads(tp):
    h, hv = 16 // tp, 64 // tp
    return h, hv, 2 * h * K + hv * V


class _Case:
    def __init__(
        self, tp, bs, T, seed, *, history=True, pad=False, flush=False, i64=False
    ):
        H, HV, D = _heads(tp)
        self.H, self.HV, self.D, self.T, self.bs = H, HV, D, T, bs
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.g = g
        slots, tokens = bs + 5, bs * T
        self.slots, self.tokens = slots, tokens

        def rand(*shape, scale=1.0):
            return (torch.randn(*shape, device="cuda", generator=g) * scale).to(
                torch.bfloat16
            )

        self.rand = rand
        # the q / k / v / z projection; the conv reads its first D columns as views
        self.qkvz = rand(tokens, D + HV * V)
        self.ba = rand(tokens, 2 * HV)
        self.A_log = torch.rand(HV, device="cuda", generator=g)
        self.dt_bias = torch.rand(HV, device="cuda", generator=g)
        self.weight = rand(D, WIDTH, scale=0.5)
        self.bias = None
        self.state = rand(slots, D, WIDTH - 1)
        # deduplicated windows: one row of T + 2 a channel (window[t][w] = row[t + w])
        self.shared = rand(slots, D, T + 2)
        self.checkpoint = rand(slots, HV, V, K, scale=0.05)
        self.ring = [
            rand(slots, HV, RING, V, scale=0.05),
            rand(slots, H, RING, K, scale=0.09),
            -torch.rand(slots, HV, RING, device="cuda", generator=g) * 0.5,
            rand(slots, HV, RING, V, scale=1e-4),
            rand(slots, H, RING, K, scale=1e-4),
        ]
        dt = torch.int64 if i64 else torch.int32
        self.states = torch.randperm(slots, device="cuda", generator=g)[:bs].to(dt)
        self.windex = torch.randperm(slots, device="cuda", generator=g)[:bs].to(dt)
        self.replay = torch.randperm(slots, device="cuda", generator=g)[:bs].to(dt)
        if pad:
            self.states[bs // 2] = -1
        self.write_pos = torch.randint(
            1 if history else 0,
            RING - T + 1 if history else 1,
            (slots,),
            device="cuda",
            generator=g,
        ).to(torch.int32)
        self.cache_base = torch.randint(
            0, RING, (slots,), device="cuda", generator=g
        ).to(torch.int32)
        self.is_flush = torch.zeros(slots, dtype=torch.int8, device="cuda")
        if flush:
            self.is_flush[self.replay[0]] = 1
        self.norm = rand(V)
        self.scale = torch.tensor([0.02], device="cuda")

    def windows(self, shared):
        st = shared.stride()
        return shared.as_strided(
            (self.slots, self.T, self.D, WIDTH - 1), (st[0], 1, st[1], 1)
        )

    def views(self, qkvz):
        D, HV = self.D, self.HV
        return qkvz[:, :D], qkvz[:, D:].view(self.tokens, HV, V)

    def outputs(self):
        """Fresh copies of everything a launch writes."""
        return dict(
            state=self.state.clone(),
            shared=self.shared.clone(),
            ring=[t.clone() for t in self.ring],
            y=torch.zeros(
                self.tokens, self.HV * V, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            o=torch.zeros(self.tokens, self.HV, V, dtype=torch.bfloat16, device="cuda"),
        )

    def _core_args(self, qkv, z, out):
        b, a = self.ba[:, : self.HV], self.ba[:, self.HV :]
        return (
            qkv,
            z,
            a,
            b,
            self.A_log,
            self.dt_bias,
            self.checkpoint,
            *out["ring"],
            self.states,
            self.replay,
            self.write_pos,
            self.cache_base,
            self.is_flush,
            self.norm,
            self.scale,
            1e-6,
            out["y"],
        )

    def stock(self, out, qkvz=None, **kw):
        """Slot gdn_conv's launch, then slot gdn_core's on its output."""
        from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update
        from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import core

        qkv, z = self.views(self.qkvz if qkvz is None else qkvz)
        y = conv_update(
            qkv.view(self.bs, self.T, -1).transpose(1, 2),
            out["state"],
            self.weight,
            self.bias,
            "silu",
            self.states,
            self.windows(out["shared"]),
            self.windex,
            block_n=128,
            token_major_out=True,
            hoist=True,
            window_mode=1,
        )
        core(
            *self._core_args(y.transpose(1, 2).view(self.tokens, -1), z, out),
            rows_per_request=self.T,
            num_k_heads=self.H,
            o_out=out["o"],
            **kw,
        )

    def fused(self, out, arrive, qkvz=None, **kw):
        """Slot gdn_conv_core's one launch."""
        from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import Conv, core

        qkv, z = self.views(self.qkvz if qkvz is None else qkvz)
        core(
            *self._core_args(qkv, z, out),
            rows_per_request=self.T,
            num_k_heads=self.H,
            o_out=out.get("o"),
            conv=Conv(
                self.weight,
                self.bias,
                out["state"],
                self.windows(out["shared"]),
                self.windex,
                arrive,
            ),
            **kw,
        )

    def arrive(self):
        return torch.zeros(self.bs * self.H, dtype=torch.int32, device="cuda")


def _same(got, want, pad_rows=None):
    assert torch.equal(got["y"].view(torch.uint8), want["y"].view(torch.uint8))
    assert torch.equal(got["o"], want["o"])
    assert all(torch.equal(x, y) for x, y in zip(got["ring"], want["ring"]))
    assert torch.equal(got["state"], want["state"])
    assert torch.equal(got["shared"], want["shared"])


CASES = [(8, bs) for bs in (1, 2, 4, 5, 6, 8, 12, 16, 24, 32)] + [
    (4, 8),
    (4, 32),
    (2, 10),
    (2, 32),
]


@pytest.mark.parametrize("tp, bs", CASES)
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_conv_core_is_conv_then_core(pad, history, tp, bs):
    if pad and bs == 1:
        pytest.skip("a padded single request leaves nothing to compare")
    c = _Case(
        tp, bs, 7, 1000 * tp + 10 * bs + 2 * history + pad, history=history, pad=pad
    )
    want, got = c.outputs(), c.outputs()
    c.stock(want)
    arrive = c.arrive()
    c.fused(got, arrive)
    _same(got, want)
    assert not (arrive % (c.HV // c.H)).any()
    # the conv rolled every live request's state and wrote its windows
    live = c.states[c.states >= 0].long()
    assert not torch.equal(got["state"][live], c.state[live])
    if pad:
        dead = c.states[c.bs // 2]
        assert dead == -1


@pytest.mark.parametrize("tp, bs", [(8, 1), (8, 8), (8, 32), (4, 12)])
def test_conv_core_with_a_conv_bias(tp, bs):
    c = _Case(tp, bs, 7, 4040 + 10 * tp + bs)
    c.bias = c.rand(c.D)
    want, got = c.outputs(), c.outputs()
    c.stock(want)
    c.fused(got, c.arrive())
    _same(got, want)


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("i64", [False, True])
def test_conv_core_rows_a_request_and_index_widths(i64, T):
    """4 and 8 rows a request (the conv's bounds), int32 and int64 indices."""
    c = _Case(8, 6, T, 77 + T + i64, i64=i64)
    want, got = c.outputs(), c.outputs()
    c.stock(want)
    c.fused(got, c.arrive())
    _same(got, want)


@pytest.mark.parametrize("bs", [6, 32])
def test_conv_core_stores_padded_windows_and_strided_states(bs):
    c = _Case(8, bs, 7, 606 + bs, pad=True)
    padded = c.rand(c.slots, c.D, c.T + 6)
    c.shared = padded[:, :, : c.T + 2]
    wide = c.rand(c.slots, c.D, 4)
    c.state = wide[:, :, :3]
    want, got = c.outputs(), c.outputs()
    # outputs() clones contiguously: rebuild the padded layouts for both
    for out in (want, got):
        w, s = padded.clone(), wide.clone()
        out["shared"], out["state"] = w[:, :, : c.T + 2], s[:, :, :3]
        out["_keep"] = (w, s)
    c.stock(want)
    c.fused(got, c.arrive())
    _same(got, want)
    assert torch.equal(got["_keep"][0], want["_keep"][0])
    assert torch.equal(got["_keep"][1], want["_keep"][1])


def test_conv_core_flush_request_still_rolls_the_conv():
    c = _Case(8, 8, 7, 4242, flush=True)
    want, got = c.outputs(), c.outputs()
    c.stock(want)
    c.fused(got, c.arrive())
    _same(got, want)
    assert not got["y"][:7].float().any()  # request 0 is the flush request


@pytest.mark.parametrize(
    "variant",
    [
        {"early_dependents": True},
        {"min_blocks": 3},
        {"flat": False},
        {"early_dependents": True, "min_blocks": 3},
    ],
)
@pytest.mark.parametrize("tp, bs", [(8, 8), (8, 32), (4, 32)])
def test_conv_core_variants_leave_every_byte_as_it_was(tp, bs, variant):
    c = _Case(tp, bs, 7, 31 * bs + len(variant) + tp, pad=bs > 8)
    want, got = c.outputs(), c.outputs()
    c.stock(want, **variant)
    c.fused(got, c.arrive(), **variant)
    _same(got, want)


def test_conv_core_reads_state_its_key_head_has_not_rolled():
    for tp, bs in ((8, 32), (4, 32), (2, 32)):
        c = _Case(tp, bs, 7, 9000 + tp)
        want, got = c.outputs(), c.outputs()
        arrive = c.arrive()
        for i in range(6):
            qkvz = c.rand(c.tokens, c.D + c.HV * V)
            c.stock(want, qkvz=qkvz)
            c.fused(got, arrive, qkvz=qkvz, early_dependents=bool(i % 2))
            _same(got, want)
        assert not (arrive % (c.HV // c.H)).any()
        assert (arrive == 6 * (c.HV // c.H)).all()


@pytest.mark.parametrize("bs", [6, 8, 12, 16, 32])
def test_conv_core_replays_from_a_cuda_graph(bs):
    c = _Case(8, bs, 7, 555 + bs, pad=bs > 8)
    qkvz = c.qkvz.clone()
    got = c.outputs()
    arrive = c.arrive()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        c.fused(c.outputs(), arrive, qkvz=qkvz)  # warm up (compiles outside capture)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.fused(got, arrive, qkvz=qkvz)
    for i in range(4):
        qkvz.copy_(c.rand(*qkvz.shape))
        got["state"].copy_(c.rand(*c.state.shape))
        got["shared"].copy_(c.rand(*c.shared.shape))
        want = dict(
            state=got["state"].clone(),
            shared=got["shared"].clone(),
            ring=[t.clone() for t in got["ring"]],
            y=torch.zeros_like(got["y"]),
            o=torch.zeros_like(got["o"]),
        )
        c.stock(want, qkvz=qkvz)
        graph.replay()
        torch.cuda.synchronize()
        _same(got, want)
    assert not (arrive % (c.HV // c.H)).any()


def test_conv_core_refuses_what_it_cannot_serve():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import Conv

    c = _Case(8, 4, 7, 3)
    got = c.outputs()
    with pytest.raises(ValueError, match="arrival counters"):
        c.fused(got, torch.zeros(c.bs * c.H, dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="arrival counters"):
        c.fused(got, torch.zeros(c.bs * c.H - 1, dtype=torch.int32, device="cuda"))
    per_token = torch.zeros(
        c.slots, c.T, c.D, WIDTH - 1, dtype=torch.bfloat16, device="cuda"
    )
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import core

    qkv, z = c.views(c.qkvz)
    with pytest.raises(ValueError, match="deduplicated"):
        core(
            *c._core_args(qkv, z, got),
            rows_per_request=c.T,
            num_k_heads=c.H,
            conv=Conv(c.weight, None, got["state"], per_token, c.windex, c.arrive()),
        )
    c3 = _Case(8, 4, 3, 5)
    with pytest.raises(ValueError, match="rows a request"):
        c3.fused(c3.outputs(), c3.arrive())
    with pytest.raises(ValueError, match="replay indices"):
        c.replay = c.replay[:-1]
        c.fused(got, c.arrive())
