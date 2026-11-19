# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The low-M projection launches (slots lowm_in / lowm_out / lowm_se, sm103/lowm) on one SM100-family GPU
(GB300), numeric against SGLang's static quantize + FlashInfer FP8 BMM, eager and replayed from a CUDA graph:
- the in-register quantization is SGLang's static quantize bit for bit: a BF16-input launch equals, byte for
  byte, the same launch on SGLang's E4M3 quantization of the rows;
- each TP8 projection (in_proj_qkvz with in_proj_ba's BF16 rows, qkv_proj, out_proj, o_proj) at the tactics
  the slots use and at the default one, against fp64 on the same E4M3 operands: at most 1.05 x the max
  relative error of FlashInfer's BMM plus one BF16 ulp of the output scale;
- the cluster K split equals the same tactic without it up to the FP32 sum order;
- the shared expert's two launches against SGLang's chain (quantize, gate_up BMM, SiluAndMul, quantize,
  down BMM, fused_gate_sigmoid_mul)."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103 import lowm

FP8 = torch.float8_e4m3fn


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
        from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: F401
            static_quant_fp8,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _weight(g, n, k, std=0.05):
    w = torch.randn(n, k, device="cuda", generator=g) * std
    s = (w.abs().amax() / 448.0).float().reshape(1)
    return (w / s).to(FP8).contiguous(), s


def _rows(g, m, k):
    x = (torch.randn(m, k, device="cuda", generator=g) * 2).to(torch.bfloat16)
    s = (x.float().abs().amax() / 448.0 * 1.1).reshape(1).float()
    return x, s


def _bmm(x8, w8, s, sw):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    return flashinfer_bmm_fp8(x8, w8.t(), s, sw, torch.bfloat16)


def _rel(a, ref):
    return float((a.float() - ref).abs().max() / ref.abs().max())


def _tactics():
    from sgl_nvidia_mega.qwen3_5.lowm import TACTICS

    return TACTICS


@pytest.mark.parametrize(
    "n, n16, k, rows",
    [
        (2560, 16, 4096, 14),
        (2560, 0, 4096, 14),
        (4096, 0, 1024, 14),
        (4096, 0, 1024, 28),
        (2560, 0, 4096, 28),
        (4096, 0, 1024, 56),
    ],
)
def test_projection_is_bmm_numerically(n, n16, k, rows):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + n16)
    w8, sw = _weight(g, n, k)
    w16 = (
        (torch.randn(n16, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        if n16
        else None
    )
    x, s = _rows(g, rows, k)
    x8, _ = static_quant_fp8(x, s)
    exact = (x8.double() @ w8.double().t()) * float(s) * float(sw)
    ref = _bmm(x8, w8, s, sw)
    bound = 1.05 * _rel(ref, exact) + 2.0**-8
    used = _tactics().get((n, n16, k, rows, False))
    for tactic in ([used] if used is not None else []) + [
        lowm.default_tactic(rows, n, n16, k),
        None,
    ]:
        out_q = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        out_8 = torch.empty_like(out_q)
        out2 = (
            torch.empty(rows, n16, dtype=torch.bfloat16, device="cuda") if n16 else None
        )
        lowm.gemm(
            x16=x,
            w8=w8,
            w_scale=sw,
            in_scale=s,
            w16=w16,
            out=out_q,
            out2=out2,
            tactic=tactic,
        )
        torch.cuda.synchronize()
        assert _rel(out_q, exact) <= bound, (tactic, _rel(out_q, exact), bound)
        if tactic is not None:
            # the same FP8 CTAs on SGLang's E4M3 rows: the quantization is SGLang's, byte for byte
            lowm.gemm(x8=x8, w8=w8, w_scale=sw, in_scale=s, out=out_8, tactic=tactic)
            torch.cuda.synchronize()
            assert torch.equal(out_q.view(torch.int16), out_8.view(torch.int16)), tactic
        if n16:
            e2 = x.double() @ w16.double().t()
            r2 = torch.nn.functional.linear(x, w16)
            assert _rel(out2, e2) <= 1.05 * _rel(r2, e2) + 2.0**-8


@pytest.mark.parametrize("rows", [14, 28])
def test_cluster_split_is_the_same_sums(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    w8, sw = _weight(g, 4096, 1024)
    x, s = _rows(g, rows, 1024)
    a = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    lowm.gemm(
        x16=x, w8=w8, w_scale=sw, in_scale=s, out=a, tactic=lowm.Tactic(4, 4, 1, 1, 1)
    )
    lowm.gemm(
        x16=x, w8=w8, w_scale=sw, in_scale=s, out=b, tactic=lowm.Tactic(4, 4, 1, 1, 2)
    )
    torch.cuda.synchronize()
    d = (a.float() - b.float()).abs()
    assert float(d.max()) <= float(a.float().abs().max()) * 2.0**-7


@pytest.mark.parametrize("rows", [14, 28, 56])
def test_shared_expert_is_sglang_numerically(rows):
    from sgl_kernel import silu_and_mul
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    gu, sgu = _weight(g, 256, 4096)
    dn, sdn = _weight(g, 4096, 128)
    gate = (torch.randn(1, 4096, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x, s = _rows(g, rows, 4096)
    s_h = torch.tensor([0.02], device="cuda")
    # SGLang: quantize, gate_up BMM, SiluAndMul, quantize, down BMM, gate
    q, _ = static_quant_fp8(x, s)
    h = torch.empty(rows, 128, dtype=torch.bfloat16, device="cuda")
    silu_and_mul(_bmm(q, gu, s, sgu), h)
    q2, _ = static_quant_fp8(h, s_h)
    ref = fused_gate_sigmoid_mul(x, gate.squeeze(0), _bmm(q2, dn, s_h, sdn))
    gate8 = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    gate8[:1].copy_(gate)
    t = _tactics()
    h8 = torch.empty(rows, 128, dtype=FP8, device="cuda")
    logits = torch.empty(rows, 8, dtype=torch.float32, device="cuda")
    out = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")

    def run():
        lowm.gemm(
            x16=x,
            w8=lowm.swiglu_pairs(gu),
            w_scale=sgu,
            in_scale=s,
            w16=gate8,
            out=h8,
            out2=logits,
            swiglu_scale=s_h,
            tactic=t.get((256, 8, 4096, rows, False)),
        )
        lowm.gemm(
            x8=h8,
            w8=dn,
            w_scale=sdn,
            in_scale=s_h,
            out=out,
            gate=logits[:, 0],
            tactic=t.get((4096, 0, 128, rows, True)),
        )

    run()
    torch.cuda.synchronize()
    # the hidden FP8 rows differ from SGLang's only where the FP32 sum order moves a BF16 rounding
    assert float((h8.float() != q2.float()).float().mean()) < 0.02
    rel = float((out.float() - ref.float()).norm() / ref.float().norm())
    assert rel < 2e-2, rel
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first)


TEP8_ROWS = [33, 56, 63, 64, 70, 84, 98, 112, 126, 128]
TEP8_TACTICS = {
    "default": (lowm.Tactic(8, 2, 1, 1, 4), lowm.Tactic(2, 1, 1, 1, 1)),
    "down256": (lowm.Tactic(8, 2, 1, 1, 4), lowm.Tactic(2, 2, 1, 1, 1)),
    "down128": (lowm.Tactic(16, 2, 1, 1, 4), lowm.Tactic(2, 4, 1, 1, 1)),
}


def _shared_case(seed, rows):
    g = torch.Generator(device="cuda").manual_seed(500 + 31 * rows + seed)
    gu, sgu = _weight(g, 256, 4096)
    dn, sdn = _weight(g, 4096, 128)
    gate = (torch.randn(1, 4096, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x, s = _rows(g, rows, 4096)
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    q, _ = static_quant_fp8(x, s)
    gu64 = (q.double() @ gu.double().t()) * float(s) * float(sgu)
    h64 = torch.nn.functional.silu(gu64[:, :128]) * gu64[:, 128:]
    s_h = (h64.abs().amax() / 448.0 * 1.05).float().reshape(1)
    exact = (
        (h64 @ dn.double().t())
        * float(sdn)
        * torch.sigmoid(x.double() @ gate.double().t())
    )
    return x, s, gu, sgu, dn, sdn, gate, s_h, exact


def _sglang_shared(x, s, gu, sgu, dn, sdn, gate, s_h):
    from sgl_kernel import silu_and_mul
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    rows = x.shape[0]
    q, _ = static_quant_fp8(x, s)
    h = torch.empty(rows, 128, dtype=torch.bfloat16, device="cuda")
    silu_and_mul(_bmm(q, gu, s, sgu), h)
    q2, _ = static_quant_fp8(h, s_h)
    return fused_gate_sigmoid_mul(x, gate.squeeze(0), _bmm(q2, dn, s_h, sdn)), q2


def _lowm_shared(x, s, gu, sgu, dn, sdn, gate, s_h, t1, t2):
    rows = x.shape[0]
    gate8 = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    gate8[:1].copy_(gate)
    h8 = torch.empty(rows, 128, dtype=FP8, device="cuda")
    logits = torch.empty(rows, 8, dtype=torch.float32, device="cuda")
    out = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
    pairs = lowm.swiglu_pairs(gu)

    def run():
        lowm.gemm(
            x16=x,
            w8=pairs,
            w_scale=sgu,
            in_scale=s,
            w16=gate8,
            out=h8,
            out2=logits,
            swiglu_scale=s_h,
            tactic=t1,
        )
        lowm.gemm(
            x8=h8,
            w8=dn,
            w_scale=sdn,
            in_scale=s_h,
            out=out,
            gate=logits[:, 0],
            tactic=t2,
        )

    return run, out, h8


@pytest.mark.parametrize("tactics", list(TEP8_TACTICS))
@pytest.mark.parametrize("rows", TEP8_ROWS)
def test_tep8_shared_expert_fp8_hidden_and_graph_replay(rows, tactics):
    t1, t2 = TEP8_TACTICS[tactics]
    for seed in range(3):
        x, s, gu, sgu, dn, sdn, gate, s_h, _ = _shared_case(seed, rows)
        _, q2 = _sglang_shared(x, s, gu, sgu, dn, sdn, gate, s_h)
        run, out, h8 = _lowm_shared(x, s, gu, sgu, dn, sdn, gate, s_h, t1, t2)
        run()
        torch.cuda.synchronize()
        assert float((h8.float() != q2.float()).float().mean()) < 0.02
        if seed == 0:
            first = out.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, first)


def test_tep8_shared_expert_slot_takes_its_tactics(monkeypatch):
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5 import lowm as slot

    rows = 98
    x, s, gu, sgu, dn, sdn, gate, s_h, _ = _shared_case(0, rows)
    gate8 = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    gate8[:1].copy_(gate)
    mlp = SimpleNamespace(
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=gu.t(), weight_scale=sgu, input_scale=s
            ),
            down_proj=SimpleNamespace(weight=dn.t(), weight_scale=sdn, input_scale=s_h),
        )
    )
    t1, t2 = TEP8_TACTICS["default"]
    slot._configure_tep8_se(slot.TEP8_SE_ROWS, False)
    cfg = slot.tep8_se_config()
    assert cfg["rows"] == (33, 128) and cfg["t1"] == t1
    assert slot._tep8_se_tactics(cfg["rows"], rows, 0, 2) == t2
    bound = slot.Bound(
        se=True, se_tep8=cfg["rows"], shared={id(mlp): (lowm.swiglu_pairs(gu), gate8)}
    )
    got = slot.shared_expert(mlp, x, bound)
    run, out, _ = _lowm_shared(x, s, gu, sgu, dn, sdn, gate, s_h, t1, t2)
    run()
    torch.cuda.synchronize()
    assert torch.equal(got, out)
    assert slot._tep8_se_tactics((33, 128), 28, 8, 1) is None
    slot._configure_tep8_se((), False)


@pytest.mark.parametrize("rows", [84, 91, 98, 112, 128])
def test_tep8_shared_expert_coresident_is_the_same_bytes(monkeypatch, rows):
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5 import lowm as slot

    x, s, gu, sgu, dn, sdn, gate, s_h, _ = _shared_case(2, rows)
    gate8 = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    gate8[:1].copy_(gate)
    mlp = SimpleNamespace(
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=gu.t(), weight_scale=sgu, input_scale=s
            ),
            down_proj=SimpleNamespace(weight=dn.t(), weight_scale=sdn, input_scale=s_h),
        )
    )
    t1, t2 = TEP8_TACTICS["default"]
    run, want, _ = _lowm_shared(x, s, gu, sgu, dn, sdn, gate, s_h, t1, t2)
    run()
    torch.cuda.synchronize()
    want = want.clone()
    for coresident, trigger in ((True, False), (False, True)):
        slot._configure_tep8_se(slot.TEP8_SE_ROWS, coresident)
        cfg = slot.tep8_se_config()
        assert slot._tep8_se_trigger(cfg["rows"], rows) is trigger
        bound = slot.Bound(
            se=True,
            se_tep8=cfg["rows"],
            shared={id(mlp): (lowm.swiglu_pairs(gu), gate8)},
        )
        lowm.prepare(
            torch.cuda.current_device(),
            rows,
            4096,
            256,
            8,
            1,
            1,
            True,
            tactic=t1,
            trigger=trigger,
        )
        got = slot.shared_expert(mlp, x, bound)
        torch.cuda.synchronize()
        assert torch.equal(got, want)
        px = torch.randn(rows, 1024, device="cuda").to(torch.bfloat16)
        pw = (torch.randn(4096, 1024, device="cuda") * 0.05).to(FP8)
        pout = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
        one = torch.ones(1, device="cuda")
        outs = []

        def chain():
            lowm.gemm(x16=px, w8=pw, w_scale=one, in_scale=one, out=pout, trigger=True)
            outs.append(slot.shared_expert(mlp, x, bound))

        chain()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain()
        for _ in range(20):
            outs[-1].zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(outs[-1], want)
    slot._configure_tep8_se((), False)


TEP8_WIDE_ROWS = [133, 147, 154, 160]


def _wide_slot_case(monkeypatch, rows, seed=0):
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5 import lowm as slot

    slot._configure_tep8_se((33, slot.TEP8_SE_WIDE_HI), True)
    cfg = slot.tep8_se_config()
    x, s, gu, sgu, dn, sdn, gate, s_h, exact = _shared_case(seed, rows)
    gate8 = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    gate8[:1].copy_(gate)
    mlp = SimpleNamespace(
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=gu.t(), weight_scale=sgu, input_scale=s
            ),
            down_proj=SimpleNamespace(weight=dn.t(), weight_scale=sdn, input_scale=s_h),
        )
    )
    bound = slot.Bound(
        se=True, se_tep8=cfg["rows"], shared={id(mlp): (lowm.swiglu_pairs(gu), gate8)}
    )
    return slot, mlp, bound, (x, s, gu, sgu, dn, sdn, gate, s_h, exact)


@pytest.mark.parametrize("rows", TEP8_WIDE_ROWS)
def test_tep8_wide_shared_expert_is_the_128_row_launch_row_by_row(monkeypatch, rows):
    slot, mlp, bound, case = _wide_slot_case(monkeypatch, rows)
    x, s, gu, sgu, dn, sdn, gate, s_h, _ = case
    pairs = slot.se_launches(bound.se_tep8, rows)
    assert [b - a for a, b, *_ in pairs] == [128, rows - 128]
    assert pairs[-1][3] is True and all(p[3] is False for p in pairs[:-1])
    got = slot.shared_expert(mlp, x, bound)
    torch.cuda.synchronize()
    t1, t2 = TEP8_TACTICS["default"]
    for a, n in ((rows - 128, 128), (7, 112), (100, 64)):
        n = min(n, rows - a)
        run, out, _ = _lowm_shared(
            x[a : a + n].contiguous(), s, gu, sgu, dn, sdn, gate, s_h, t1, t2
        )
        run()
        torch.cuda.synchronize()
        assert torch.equal(got[a : a + n], out), (rows, a, n)
    slot._configure_tep8_se((), False)


@pytest.mark.parametrize("rows", [133, 154, 160])
def test_tep8_wide_shared_expert_replays_behind_a_programmatic_producer(
    monkeypatch, rows
):
    slot, mlp, bound, case = _wide_slot_case(monkeypatch, rows, 1)
    x = case[0]
    want = slot.shared_expert(mlp, x, bound).clone()
    torch.cuda.synchronize()
    px = torch.randn(rows, 1024, device="cuda").to(torch.bfloat16)
    pw = (torch.randn(4096, 1024, device="cuda") * 0.05).to(FP8)
    pout = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
    one = torch.ones(1, device="cuda")
    outs = []

    def chain():
        lowm.gemm(
            x16=px[:128], w8=pw, w_scale=one, in_scale=one, out=pout[:128], trigger=True
        )
        outs.append(slot.shared_expert(mlp, x, bound))

    chain()
    torch.cuda.synchronize()
    assert torch.equal(outs[-1], want)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    for _ in range(50):
        outs[-1].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outs[-1], want)
    slot._configure_tep8_se(slot.TEP8_SE_ROWS, True)
    assert slot.se_launches(slot.tep8_se_config()["rows"], rows) == (
        (0, rows, True, True),
    )
    slot._configure_tep8_se((), False)


def _ba_case(g, rows, regime):
    w = torch.randn(16, 4096, device="cuda", generator=g) * 0.02
    w = (w * (1 + 3 * torch.rand(16, 1, device="cuda", generator=g))).to(torch.bfloat16)
    x = torch.randn(rows, 4096, device="cuda", generator=g, dtype=torch.float64)
    if regime == "gamma":
        gamma = 1 + 0.5 * torch.randn(
            4096, device="cuda", generator=g, dtype=torch.float64
        )
        gamma[torch.randperm(4096, device="cuda", generator=g)[:16]] *= 20
        x = x * gamma
    return w, x.to(torch.bfloat16)


class _BaLinear:
    def __init__(self, w):
        self.weight = w

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


@pytest.mark.parametrize("rows", [84, 98, 112])
def test_lowm_ba_replays_behind_a_programmatic_producer(rows):
    from sgl_nvidia_mega.qwen3_5 import lowm as slot

    torch.zeros(1, device="cuda")  # a current CUDA context before the module loads
    dev = torch.cuda.current_device()
    tactic, tactic84 = slot.prepare_ba(dev, rows), slot.prepare_ba(dev, 84)
    g = torch.Generator(device="cuda").manual_seed(rows)
    w, _ = _ba_case(g, rows, "gamma")
    gdn = type("G", (), {"in_proj_ba": _BaLinear(w)})()
    px = torch.randn(rows, 1024, device="cuda", generator=g).to(torch.bfloat16)
    pw = (torch.randn(4096, 1024, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    x = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
    side = torch.cuda.Stream()
    outs = []

    def chain():
        lowm.gemm(
            x16=px, w16=pw, out2=x, tactic=lowm.Tactic(4, 1, 1, 1, 1), trigger=True
        )
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            ba = slot.in_proj_ba(gdn, x, tactic)
        cur.wait_stream(side)
        outs.append(ba * 1)

    chain()
    torch.cuda.synchronize()
    want = outs[-1].clone()
    assert torch.equal(want, slot.in_proj_ba(gdn, x, tactic) * 1)
    assert torch.equal(slot.in_proj_ba(gdn, x[:84].contiguous(), tactic84), want[:84])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    for _ in range(50):
        outs[-1].zero_()
        x.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outs[-1], want)
