"""FP8 pool union numerical gate, NOT bitwise. Run on sm107 before GSM8K.

Declared tolerance rtol=0.02/atol=0.0002 and relative L2 <= 0.01 against dense FP64 softmax. This is a
qualification ceiling, not a measured result; record actual max/relative errors.
Passing this synthetic gate does not waive full GSM8K and natural AL checks.
"""

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    DirectPoolSelection,
    KvTensors,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericSignature,
    PoolUnionAttentionIO,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8._providers.pool_union import build


def check_tolerance(out, reference):
    """Proposed fixed qualification gate; GSM8K remains mandatory afterwards."""
    assert torch.isfinite(out).all() and torch.isfinite(reference).all()
    difference = out.double() - reference.double()
    rel_l2 = (difference.norm() / reference.double().norm().clamp_min(1e-30)).item()
    assert rel_l2 <= 0.01, ("relative L2", rel_l2)
    torch.testing.assert_close(out.double(), reference.double(), rtol=0.02, atol=0.0002)
    return {"max_abs": difference.abs().max().item(), "rel_l2": rel_l2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")
@pytest.mark.parametrize("rows", [64, 128, 416, 576])
# Variant bits 0-5: kernel U warps (0 = 8); bit 6: the warp-specialized kernel 2;
# bit 7: PDL into it.
@pytest.mark.parametrize("variant", [0, 16, 64 | 16, 128 | 64 | 16])
def test_pool_union_tolerance(rows, variant, record_property):
    torch.manual_seed(942)
    device = "cuda"
    width = 4
    requests = rows // width
    length = 2051
    q = torch.randn(rows, 24, 256, device=device, dtype=torch.bfloat16) * 0.25
    k = (torch.randn(requests * length + 1, 2, 256, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn_like(k, dtype=torch.bfloat16) * 0.25).to(torch.float8_e4m3fn)
    table = torch.arange(
        1, requests * length + 1, device=device, dtype=torch.int32
    ).reshape(requests, length)
    req = torch.arange(requests, device=device, dtype=torch.int32).repeat_interleave(
        width
    )
    positions = 2047 + torch.arange(rows, device=device, dtype=torch.int32) % width
    lengths = positions + 1
    loc = table[req.long(), positions.long()].long()
    idx = (
        torch.arange(512, device=device, dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    # Include counted invalid entries and an unselected stale-NaN slot.
    idx[:, 0] = -1
    k[0] = float("nan")
    v[0] = float("nan")
    nk = k[loc].contiguous()
    nv = v[loc].contiguous()
    out = torch.empty_like(q)
    from sglang.srt.kernel_plugins.nvidia.qsa_union import workspace_plan

    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits, ni, nf = workspace_plan(rows, sms)
    if variant & 64:
        from sgl_nvidia_mega.qwen3_8.qsa.pool_union_ws import choose_ws_splits

        # The caller resolves the warp-specialized kernel's own split count.
        splits = choose_ws_splits(requests, 2, sms)
        nf = requests * 2 * splits * 64 * 258
    io = PoolUnionAttentionIO(
        q,
        None,
        nk,
        nv,
        loc,
        KvTensors(k, v, 1.0, 1.0, 1),
        DirectPoolSelection(table, req, idx, lengths, positions),
        0.0625,
        4,
        True,
        out,
        torch.empty(nf, device=device, dtype=torch.float32),
        torch.empty(ni, device=device, dtype=torch.int32),
        splits,
    )
    run = build(
        NumericSignature(
            O.POOL_UNION_ATTENTION, rows, 24, 2, 256, 512, 4, D.BF16, variant=variant
        )
    )
    run(io)
    torch.cuda.synchronize()
    references = []
    for row in range(rows):
        n = int(lengths[row] // 4) * 4
        pos = (
            idx[row].repeat_interleave(4) * 4
            + torch.arange(4, device=device).repeat(512)
        )[:n]
        valid = idx[row].repeat_interleave(4)[:n] >= 0
        tail = torch.arange(
            (int(positions[row]) + 1) // 4 * 4, int(positions[row]) + 1, device=device
        )
        pos = torch.cat((pos, tail))
        valid = torch.cat((valid, torch.ones_like(tail, dtype=torch.bool)))
        slots = table[int(req[row]), pos.clamp_min(0).long()]
        kk = k[slots].double()
        vv = v[slots].double()
        kk[~valid] = 0
        vv[~valid] = 0
        kk = kk.repeat_interleave(12, 1)
        vv = vv.repeat_interleave(12, 1)
        probs = torch.softmax(
            torch.einsum("hd,nhd->hn", q[row].double(), kk) * 0.0625, dim=-1
        )
        references.append(torch.einsum("hn,nhd->hd", probs, vv))
    ref = torch.stack(references)
    assert torch.isfinite(out).all()
    record_property("max_abs", (out.double() - ref).abs().max().item())
    record_property("rel_l2", ((out.double() - ref).norm() / ref.norm()).item())
    check_tolerance(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")
@pytest.mark.parametrize("rows,variant", [(4, 1), (64, 1), (1, 2), (48, 2)])
def test_pool_merge_gate_is_the_separate_sigmoid_mul(rows, variant):
    """Variant bit 2 (the source's SGLANG_ENABLE_QSA_POOL_MERGE_GATE): the merge
    stores bf16(bf16(attn) * sigmoid(gate)), fused_sigmoid_mul's arithmetic."""
    import triton
    import triton.language as tl
    from sglang.kernels.plugin_kernels.nvidia.numeric import DirectPoolAttentionIO
    from sglang.srt.kernel_plugins.nvidia.pool_attention import pool_scratch_elements

    from sgl_nvidia_mega.qwen3_8._providers import pool

    @triton.jit
    def sigmoid_mul(out, gate, row_stride, head_stride, D: tl.constexpr):
        row, head = tl.program_id(0).to(tl.int64), tl.program_id(1)
        d = tl.arange(0, D)
        o = out + (row * tl.num_programs(1) + head) * D + d
        g = tl.load(gate + row * row_stride + head * head_stride + d).to(tl.float32)
        tl.store(o, (tl.load(o).to(tl.float32) * tl.sigmoid(g)).to(tl.bfloat16))

    torch.manual_seed(7)
    device, width = "cuda", 4 if variant == 1 else 1
    requests, length = max(1, rows // width), 4096
    q = torch.randn(rows, 24, 256, device=device, dtype=torch.bfloat16) * 0.25
    k = torch.randn(requests * length + 1, 2, 256, device=device).bfloat16() * 0.25
    v = torch.randn_like(k) * 0.25
    table = torch.arange(1, requests * length + 1, device=device).reshape(
        requests, length
    )
    req = torch.arange(requests, device=device).repeat_interleave(width)[:rows]
    positions = 4000 + torch.arange(rows, device=device) % width
    loc = table[req, positions]
    idx = torch.randperm(990, device=device)[:512].int().expand(rows, -1).contiguous()
    nk, nv = k[loc].contiguous(), v[loc].contiguous()
    q_gate = torch.randn(rows, 24, 512, device=device, dtype=torch.bfloat16)
    gate = q_gate[:, :, 256:]
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    scratch = pool_scratch_elements(
        rows,
        sms,
        kv_heads=2,
        head_dim=256,
        query_heads=24,
        selected_count=512,
        compress_ratio=4,
    )
    outputs = []
    for bit, g in ((0, None), (pool.MERGE_GATE, gate)):
        out = torch.empty_like(q)
        io = DirectPoolAttentionIO(
            q,
            g,
            nk,
            nv,
            loc,
            KvTensors(k, v, None, None, 1),
            DirectPoolSelection(table, req, idx, positions + 1, positions),
            0.0625,
            width,
            True,
            out,
            torch.empty(scratch, device=device, dtype=torch.float32),
        )
        signature = NumericSignature(
            O.DIRECT_POOL_ATTENTION,
            rows,
            24,
            2,
            256,
            512,
            4,
            D.BF16,
            variant=variant | bit,
        )
        pool.build(signature)(io)
        outputs.append(out)
    sigmoid_mul[(rows, 24)](outputs[0], gate, gate.stride(0), gate.stride(1), 256)
    torch.cuda.synchronize()
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")
@pytest.mark.parametrize("variant", [0, 64 | 16])
def test_pool_union_cast_write_is_the_torch_cast(variant):
    """Variant bit 8 (the source's SGLANG_QSA_FUSED_FP8_KV_WRITE): BF16 new K/V cast to
    E4M3 in the pool write give the same pool bytes and output as torch's cast first."""
    from sglang.srt.kernel_plugins.nvidia.qsa_union import workspace_plan

    from sgl_nvidia_mega.qwen3_8.qsa import pool_decode

    torch.manual_seed(5)
    device, rows, width, length = "cuda", 64, 4, 2051
    requests = rows // width
    q = torch.randn(rows, 24, 256, device=device, dtype=torch.bfloat16) * 0.25
    k = (torch.randn(requests * length + 1, 2, 256, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn_like(k, dtype=torch.bfloat16) * 0.25).to(torch.float8_e4m3fn)
    table = torch.arange(
        1, requests * length + 1, device=device, dtype=torch.int32
    ).reshape(requests, length)
    req = torch.arange(requests, device=device, dtype=torch.int32).repeat_interleave(
        width
    )
    positions = 2047 + torch.arange(rows, device=device, dtype=torch.int32) % width
    loc = table[req.long(), positions.long()].long()
    idx = torch.arange(512, device=device, dtype=torch.int32).expand(rows, -1)
    idx = idx.contiguous()
    packed = torch.randn(rows, 3, 2, 256, device=device).bfloat16() * 8
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits, ni, nf = workspace_plan(rows, sms)
    if variant & 64:
        from sgl_nvidia_mega.qwen3_8.qsa.pool_union_ws import choose_ws_splits

        splits = choose_ws_splits(requests, 2, sms)
        nf = requests * 2 * splits * 64 * 258
    assert pool_decode.write_new_kv_e4m3_ok(k, v)
    fp8 = torch.float8_e4m3fn
    results = []
    for bit, nk, nv in (
        (0, packed[:, 0].to(fp8), packed[:, 1].to(fp8)),
        (256, packed[:, 0], packed[:, 1]),
    ):
        kc, vc, out = k.clone(), v.clone(), torch.empty_like(q)
        io = PoolUnionAttentionIO(
            q,
            None,
            nk,
            nv,
            loc,
            KvTensors(kc, vc, 1.0, 1.0, 1),
            DirectPoolSelection(table, req, idx, positions + 1, positions),
            0.0625,
            4,
            True,
            out,
            torch.empty(nf, device=device, dtype=torch.float32),
            torch.empty(ni, device=device, dtype=torch.int32),
            splits,
        )
        signature = NumericSignature(
            O.POOL_UNION_ATTENTION,
            rows,
            24,
            2,
            256,
            512,
            4,
            D.BF16,
            variant=variant | bit,
        )
        build(signature)(io)
        results.append((kc, vc, out))
    torch.cuda.synchronize()
    for a, b in zip(*results):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
