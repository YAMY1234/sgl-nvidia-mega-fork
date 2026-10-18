# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention chain on CPU: its swap-AB gates, the query RoPE applied exactly
once, and its slot reported as built."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

backend = pytest.importorskip("sglang.srt.layers.attention.deepseek_v4_backend")

from sgl_nvidia_mega.deepseek_v41 import runtime  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.attention import chain  # noqa: E402


def _mode(*, decode=False, verify=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_target_verify=lambda: verify
        )
    )


def _pool(*, window=None, width=584):
    return SimpleNamespace(
        request_window=window, get_swa_key_bytes_per_token=lambda: width
    )


def test_swapab_takes_only_covered_steps():
    """Swap-AB only at 1-24 rows of decode / verify on the V4-layout window cache."""
    meta = backend.DSV4AttnMetadata.__new__(backend.DSV4AttnMetadata)

    def takes(rows, mode=None, pool=None):
        attn_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(core_attn_metadata=meta),
            token_to_kv_pool=pool or _pool(),
        )
        return chain.swapab_takes(rows, mode or _mode(verify=True), attn_backend)

    assert takes(1) and takes(24) and takes(6, _mode(decode=True))
    assert not takes(25)
    assert not takes(6, _mode())
    assert not takes(6, pool=_pool(window=object()))
    assert not takes(6, pool=_pool(width=528))


def test_chain_ropes_the_query_once():
    """With the RoPE in attention, wq_b's BF16 output must reach attention unroped."""
    q = torch.randn(6, 1280, dtype=torch.bfloat16)
    wq_b_out = torch.randn(6, 16 * 512, dtype=torch.bfloat16)
    attn = SimpleNamespace(
        wq_b=mock.Mock(return_value=(wq_b_out, None)), n_local_heads=16, head_dim=512
    )
    ops = chain.ChainOps.__new__(chain.ChainOps)
    ops.rope_in_attention, ops.wq_b_ll = True, False
    with mock.patch("sglang.kernels.ops.attention.dsv4.fused_q_norm_rope") as rope:
        out = ops.compute_q_b(attn, q, torch.arange(6), None)
    rope.assert_not_called()
    assert torch.equal(out, wq_b_out.view(6, 16, 512))


def test_every_shipped_slot_reports_its_build():
    """The build checks each requested slot against this table; a slot missing
    from it stops the server at build."""
    slots = {6: SimpleNamespace(hop_a=True, fast_gate=True)}
    moe = SimpleNamespace(slots={0: SimpleNamespace(gate=True)})
    layers = {0: SimpleNamespace(chain=None), 1: SimpleNamespace(chain=object())}
    facts = SimpleNamespace(routed=True)
    built = runtime.built_slots(slots, {"moe": moe, "attention": layers}, facts)
    assert set(built) == set(runtime.SHIPPED_SLOTS) and all(built.values())
    stock = {0: SimpleNamespace(chain=None)}
    assert not runtime.built_slots(slots, {"moe": moe, "attention": stock}, facts)[
        "attn"
    ]
