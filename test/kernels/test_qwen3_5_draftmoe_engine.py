# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

E_ALL, E, H, I, K = 512, 128, 4096, 512, 10
RENORM = 1


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer.fused_moe  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM100-family GPU and FlashInfer"
)


@pytest.fixture(scope="module")
def moe():
    from flashinfer.fused_moe import trtllm_bf16_moe
    from flashinfer.fused_moe.core import (
        ActivationType,
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(11)

    def block(w_canon, gated):
        cache, out = {}, []
        for e in range(w_canon.shape[0]):
            t = w_canon[e].contiguous().view(torch.uint8)
            idx = (
                _maybe_get_cached_w3_w1_permute_indices(
                    cache, t, 128, is_gated_act_gemm=True
                )
                if gated
                else get_w2_permute_indices_with_cache(cache, t, 128)
            )
            out.append(
                convert_to_block_layout(t[idx.to(dev)].contiguous(), 128).view(
                    torch.bfloat16
                )
            )
        return torch.stack(out).contiguous()

    w13 = block(
        (torch.randn(E, 2 * I, H, device=dev, generator=g) * 0.02).bfloat16(), True
    )
    w2 = block((torch.randn(E, H, I, device=dev, generator=g) * 0.02).bfloat16(), False)

    def fi(x, lg, local_start):
        return trtllm_bf16_moe(
            routing_logits=lg,
            routing_bias=None,
            hidden_states=x,
            gemm1_weights=w13,
            gemm2_weights=w2,
            num_experts=E_ALL,
            top_k=K,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=local_start,
            local_num_experts=E,
            routing_method_type=RENORM,
            routed_scaling_factor=None,
            tune_max_num_tokens=max(8, 1 << (x.shape[0] - 1).bit_length()),
            activation_type=ActivationType.Swiglu.value,
            do_finalize=False,
        )

    return w13, w2, fi


def _engine(w13, w2, local_start):
    from types import SimpleNamespace

    from sgl_nvidia_mega.qwen3_5.draft.draftmoe import DraftMoeEngine

    experts = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        moe_ep_rank=local_start // E,
        num_local_experts=E,
    )
    return DraftMoeEngine(experts, wide=True)


def _inputs(m, kind, seed):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, H, device=dev, generator=g).bfloat16()
    if kind == "rand":
        lg = torch.randn(m, E_ALL, device=dev, generator=g) * 2
        lg[2] = lg[1]  # duplicate rows
        lg[3, :20] = 3.0
    elif kind == "extend":  # 7 correlated rows a request
        base = torch.randn(
            (m + 6) // 7, E_ALL, device=dev, generator=g
        ).repeat_interleave(7, 0)[:m]
        lg = (0.9 * base + 0.45 * torch.randn(m, E_ALL, device=dev, generator=g)) * 2
    else:
        lg = torch.randn(m, E_ALL, device=dev, generator=g)
        lg[:, 3:13] += 20.0
    return x, lg.bfloat16().contiguous()


def _same(ref, got, m):
    ia, ib = ref[2].reshape(m, K).long(), got[2].reshape(m, K).long()
    assert torch.equal(
        ref[1].reshape(m, K).view(torch.int16), got[1].reshape(m, K).view(torch.int16)
    ), "weights"
    assert torch.equal(ia < 0, ib < 0), "local (token, k) mask"
    nv = int((ib >= 0).sum())
    assert torch.equal(
        torch.sort(ib[ib >= 0]).values, torch.arange(nv, device=ib.device)
    ), "dense permutation"
    valid = (ia >= 0) & (ib >= 0)
    if valid.any():
        assert torch.equal(
            ref[0][ia[valid]].view(torch.int16), got[0][ib[valid]].view(torch.int16)
        ), "gemm2 rows"


@pytest.mark.parametrize("m", [4, 16, 17, 32, 33, 64, 91, 112, 128])
@pytest.mark.parametrize("kind", ["rand", "extend", "hot"])
@pytest.mark.parametrize("local_start", [0, 256])
def test_the_rows_are_bitwise_flashinfers(moe, m, kind, local_start):
    w13, w2, fi = moe
    engine = _engine(w13, w2, local_start)
    x, lg = _inputs(m, kind, 100 * m + len(kind) + local_start)
    assert engine.takes(x, lg)
    _same(fi(x, lg, local_start), engine.run(x, lg), m)


def test_builds_interleaved_on_one_state(moe):
    w13, w2, fi = moe
    engine = _engine(w13, w2, 128)
    for it, m in enumerate([8, 112, 33, 16, 128, 128, 32, 64, 1, 91, 98, 112, 24, 40]):
        x, lg = (
            _inputs(m, "extend" if it % 2 else "rand", 7000 + it)
            if m >= 4
            else _inputs(4, "extend", it)
        )
        x, lg = x[:m].contiguous(), lg[:m].contiguous()
        _same(fi(x, lg, 128), engine.run(x, lg), m)


def test_wide_graph_replay(moe):
    w13, w2, fi = moe
    engine = _engine(w13, w2, 0)
    m = 112
    x, lg = _inputs(m, "extend", 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = engine.run(x, lg)
    assert out is not None
    for seed in (2, 3, 4):
        x2, lg2 = _inputs(m, "rand" if seed % 2 else "extend", seed)
        x.copy_(x2)
        lg.copy_(lg2)
        graph.replay()
        torch.cuda.synchronize()
        _same(fi(x, lg, 0), out, m)
