# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random

import pytest
import torch

NK, NV, D = 16, 64, 128


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer.gdn_prefill  # noqa: F401
        from sglang.srt.layers.attention.linear.kernels import (  # noqa: F401
            gdn_flashinfer,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM100-family GPU, FlashInfer and SGLang"
)

CASES = {
    "chunk_32k_7req": [4681] * 6 + [4682],
    "cold_32k": [32768],
    "ragged": [64, 65, 127, 128, 192, 4000, 1, 256, 100, 6464, 30, 2048],
}


def _kernel():
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    return FlashInferGDNKernel()


def _inputs(lens, state_dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)
    norm = torch.nn.functional.normalize
    q = norm(torch.randn(tokens, NK, D, generator=g, device="cuda"), dim=-1)
    k = norm(torch.randn(tokens, NK, D, generator=g, device="cuda"), dim=-1)
    v = 0.5 * torch.randn(tokens, NV, D, generator=g, device="cuda")
    alpha = torch.exp(-torch.rand(tokens, NV, generator=g, device="cuda") * 0.2)
    beta = torch.sigmoid(torch.randn(tokens, NV, generator=g, device="cuda"))
    initial = 0.05 * torch.randn(len(lens), NV, D, D, generator=g, device="cuda")
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return dict(
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        alpha=alpha,
        beta=beta,
        initial_state=initial.to(state_dtype),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device="cuda"),
        output=None,
    )


def _tracked(lens, every, pick, rnd):
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n // every)
    src = []
    for i, n in enumerate(lens):
        count = n // every
        if not count or (pick == "mixed" and i % 2):
            continue
        j = count - 1 if pick == "last" else rnd.randrange(count)
        src.append(starts[i] + j)
    return starts, src


def _u8(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("every", [64, 128])
@pytest.mark.parametrize("pick", ["last", "random", "mixed"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_tracked_rows_are_sglangs_checkpoints(case, state_dtype, pick, every):
    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot as ts

    lens = CASES[case]
    rnd = random.Random(f"{case}{pick}{every}")
    starts, src = _tracked(lens, every, pick, rnd)
    assert src, "every pick tracks at least one request"
    kernel = _kernel()
    x = _inputs(lens, state_dtype, seed=len(lens) + every)
    cu_starts = torch.tensor(starts, dtype=torch.int64, device="cuda")
    track_src = torch.tensor(src, dtype=torch.int64, device="cuda")
    assert ts.routes_native(len(lens), NV, torch.device("cuda"))

    every_row = x["initial_state"].new_empty((starts[-1], NV, D, D))
    o_s, s_s = kernel._run_prefill(
        **x,
        state_checkpoints=every_row,
        checkpoint_cu_starts=cu_starts,
        checkpoint_every_n_tokens=every,
    )
    plan = ts.checkpoint_rows(track_src, starts[-1])
    rows = x["initial_state"].new_empty((len(src), NV, D, D))
    o_t, s_t = ts.run_prefill(
        kernel,
        **x,
        state_checkpoints=rows,
        checkpoint_cu_starts=cu_starts,
        checkpoint_every_n_tokens=every,
        checkpoint_rows=plan.rows,
    )
    assert torch.equal(_u8(o_s), _u8(o_t)), "output differs"
    assert torch.equal(_u8(s_s), _u8(s_t)), "final states differ"
    # SGLang's track copy reads every_row[track_src]; the slot's reads rows[slots]
    assert torch.equal(_u8(every_row[track_src]), _u8(rows[plan.slots]))


def test_the_installed_flashinfer_is_the_pinned_one():
    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot as ts

    assert ts.flashinfer_reasons() == []


def test_the_routing_keeps_qwen3_5s_tp1_heads_on_the_native_kernel():
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
    from flashinfer.utils import (
        get_compute_capability,
        get_device_name,
        get_device_sm_count,
    )

    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot as ts

    device = torch.device("cuda")
    assert all(ts.routes_native(n, NV, device) for n in range(1, 65))
    capability = get_compute_capability(device)
    sms, name = get_device_sm_count(device), get_device_name(device)
    for heads in (8, 16):
        for n in (1, 2, 4):
            cp = should_use_cp_host(n * heads, sms, name, device_capability=capability)
            assert ts.routes_native(n, heads, device) is not cp
