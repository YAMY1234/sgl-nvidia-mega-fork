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
# Variant bits 0-5: kernel U warps (0 = 8); bit 6: the warp-specialized kernel 2.
@pytest.mark.parametrize("variant", [0, 16, 64 | 16])
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
