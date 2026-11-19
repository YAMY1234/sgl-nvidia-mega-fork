# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

HIDDEN, EPS, CAP = 4096, 1e-6, 256
SENTINEL = -32768  # bf16 -0.0: an empty mailbox slot


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt import device_kernels  # noqa: F401
        from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: F401
            static_quant_fp8,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


@pytest.fixture(scope="module", params=(1024, 512), ids=("rms1024", "rms512"))
def flashinfer_rms(request):
    import cutlass.cute as cute
    from cutlass import BFloat16, Int32
    from cutlass.cute.runtime import make_fake_compact_tensor
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
        LAMPORT_GENERATIONS,
        STAGE_STATE_WORDS,
        _MaterializeRMSNormDeviceKernel,
    )
    from flashinfer.comm.mnnvl_cutedsl.runtime import (
        current_cu_stream,
        make_fake_dynamic_compact_tensor,
        to_cute,
        to_cute_dynamic,
    )

    kernel = _MaterializeRMSNormDeviceKernel(
        hidden_size=HIDDEN,
        capacity_m=CAP,
        threads=request.param,
        rms_epsilon=EPS,
        weight_bias=0.0,
        write_residual_output=True,
        enable_pdl=True,
        apply_rms_norm=True,
    )
    compiled = cute.compile(
        kernel,
        make_fake_compact_tensor(
            BFloat16, (LAMPORT_GENERATIONS * CAP * HIDDEN,), assumed_align=16
        ),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=HIDDEN),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=HIDDEN),
        None,  # the BF16 norm companion of an FP8 workspace
        None,  # the FP8 scale
        make_fake_compact_tensor(BFloat16, (HIDDEN,), assumed_align=16),
        make_fake_compact_tensor(Int32, (STAGE_STATE_WORDS,), assumed_align=4),
        Int32(CAP),
        current_cu_stream(),
    )

    def run(prenorm, stage_state, gamma, m):
        res = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        norm = torch.empty_like(res)
        compiled(
            to_cute(prenorm.flatten(), 16),
            to_cute_dynamic(res.flatten(), 16, divisibility=HIDDEN),
            to_cute_dynamic(norm.flatten(), 16, divisibility=HIDDEN),
            None,
            None,
            to_cute(gamma, 16),
            to_cute(stage_state, 4),
            Int32(m),
            current_cu_stream(),
        )
        return norm, res

    return run


@pytest.fixture(scope="module")
def tail():
    from sgl_nvidia_mega.qwen3_5.sm103.hop import bt_tail

    torch.zeros(1, device="cuda")
    compiled = bt_tail.kernels(torch.cuda.current_device())

    def run(prenorm, stage_state, gamma, m, scale=None, early=False):
        res = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        norm = torch.empty_like(res)
        fp8 = (
            None if scale is None else torch.empty_like(res, dtype=torch.float8_e4m3fn)
        )
        bt_tail.tail(
            compiled,
            prenorm=prenorm,
            stage_state=stage_state,
            gamma=gamma,
            eps=EPS,
            m=m,
            residual_out=res,
            norm_out=norm,
            fp8_scale=scale,
            fp8_out=fp8,
            early=early,
        )
        return norm, res, fp8

    return run


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.uint8)


def _bf16(bits: list) -> torch.Tensor:
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


SPECIAL = [0x7FC0, -0x0040, 0x7F80, -0x0080, 0x7F7F, -0x0081, 0x0001, -0x7FFF]
SPECIAL += [0x0040, -0x7FC0, 0x0080, -0x7F80]


def _rows(m, gen, kind):
    """``m`` prenorm rows as the owner reduce stores them (no -0.0, its sentinel)."""
    x = torch.randn(m, HIDDEN, device="cuda", generator=gen)
    if kind == "scaled":
        x = x * torch.logspace(-3, 3, m, device="cuda")[:, None]
    elif kind == "tiny":
        x = x * 1e-38  # subnormal-heavy: their squares flush to 0
    elif kind == "huge":
        x = x * 1e19  # squares overflow: inf sum, zero inverse RMS
    elif kind == "special":
        x[:, ::97] = (
            _bf16(SPECIAL * (HIDDEN // 97 // len(SPECIAL) + 1))[: x[:, ::97].shape[1]]
            .float()
            .cuda()
        )
        x[::3] = 0.0  # all-zero rows: rsqrt(eps)
    bits = x.to(torch.bfloat16).view(torch.int16)
    return bits.masked_fill(bits == SENTINEL, 0).view(torch.bfloat16)


def _gamma(gen, kind):
    g = 1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)
    if kind == "special":
        g[::61] = (
            _bf16(SPECIAL * (HIDDEN // 61 // len(SPECIAL) + 1))[: g[::61].shape[0]]
            .float()
            .cuda()
        )
        g[1::61] = 1e30
        g[2::61] = 1e-39  # subnormal norms
    return g.to(torch.bfloat16)


def _mailbox(rows, stage):
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import STAGE_STATE_WORDS

    box = torch.full((3, CAP, HIDDEN), SENTINEL, dtype=torch.int16, device="cuda")
    box[stage, : rows.shape[0]] = rows.view(torch.int16)
    state = torch.zeros(STAGE_STATE_WORDS, dtype=torch.int32, device="cuda")
    state[:2] = torch.tensor([(stage + 1) % 3, stage], dtype=torch.int32)
    return box.view(torch.bfloat16), state


def _armed(box):
    return bool((box.view(torch.int16) == SENTINEL).all())


@pytest.mark.parametrize("kind", ("normal", "scaled", "tiny", "huge", "special"))
def test_tail_is_flashinfers_rmsnorm_bit_for_bit(flashinfer_rms, tail, kind):
    gen = torch.Generator(device="cuda").manual_seed(7)
    for i, m in enumerate((1, 7, 29, 56, 84, 112, 224, CAP)):
        rows = _rows(m, gen, kind)
        gamma = _gamma(gen, kind)
        stage = i % 3
        box, state = _mailbox(rows, stage)
        want = flashinfer_rms(box, state, gamma, m)
        assert _armed(box)
        for early in (False, True):
            box, state = _mailbox(rows, stage)
            norm, res, _ = tail(box, state, gamma, m, early=early)
            assert torch.equal(_bits(res), _bits(want[1])), (kind, m, early)
            assert torch.equal(_bits(norm), _bits(want[0])), (kind, m, early)
            assert torch.equal(_bits(res), _bits(rows))
            assert _armed(box)
    torch.cuda.synchronize()


@pytest.mark.parametrize("scale", (0.05, 1 / 3, 3.0, 1e-3, 2.0**-120))
def test_fp8_is_sglangs_static_quantization_of_the_norm(tail, scale):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    gen = torch.Generator(device="cuda").manual_seed(11)
    s = torch.tensor([scale], dtype=torch.float32, device="cuda")
    seen = dict(values=0, nan=0, inf=0, clamped=0, subnormal=0)
    for i, kind in enumerate(("special", "scaled", "tiny", "huge", "normal")):
        rows = _rows(CAP, gen, kind)
        gamma = _gamma(gen, "special")
        box, state = _mailbox(rows, i % 3)
        norm, _, fp8 = tail(box, state, gamma, CAP, scale=s, early=i % 2 == 1)
        want, _ = static_quant_fp8(norm, s, repeat_scale=False)
        assert torch.equal(fp8.view(torch.uint8), want.view(torch.uint8)), (scale, kind)
        f = norm.float()
        seen["values"] += f.numel()
        seen["nan"] += int(f.isnan().sum())
        seen["inf"] += int(f.isinf().sum())
        seen["clamped"] += int(((f / scale).abs() > 448).sum())
        seen["subnormal"] += int(((f != 0) & (f.abs() < 2.0**-126)).sum())
    torch.cuda.synchronize()
    print(f"T0 bt_tail fp8 scale {scale:g}: {seen}", flush=True)
    assert seen["values"] >= 10**6
    assert min(seen["nan"], seen["inf"], seen["clamped"], seen["subnormal"]) > 0, seen


def test_tail_replays_from_a_cuda_graph(flashinfer_rms, tail):
    gen = torch.Generator(device="cuda").manual_seed(3)
    m, stage = 56, 2
    box, state = _mailbox(_rows(m, gen, "normal"), stage)
    gamma = _gamma(gen, "normal")
    s = torch.tensor([0.05], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tail(box, state, gamma, m, scale=s)  # eager warmup on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm, res, fp8 = tail(box, state, gamma, m, scale=s)
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    for replay in range(3):
        rows = _rows(m, gen, ("normal", "scaled", "special")[replay])
        box.view(torch.int16)[stage, :m] = rows.view(torch.int16)
        graph.replay()
        ref_box, ref_state = _mailbox(rows, stage)
        want = flashinfer_rms(ref_box, ref_state, gamma, m)
        assert torch.equal(_bits(norm), _bits(want[0]))
        assert torch.equal(_bits(res), _bits(want[1]))
        q, _ = static_quant_fp8(norm, s, repeat_scale=False)
        assert torch.equal(fp8.view(torch.uint8), q.view(torch.uint8))
        assert _armed(box)
    torch.cuda.synchronize()
