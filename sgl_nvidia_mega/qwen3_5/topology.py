# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot capabilities, resolved once before any package allocates or compiles.

Widths are local target-verify rows (requests times tokens_per_request). DEP8's
attention is DP8/TP1 and its MoE is EP8; it must not inherit TP8/TEP8 kernels just
because its global TP group also contains eight ranks.
"""

from __future__ import annotations

import msgspec

SGLANG_READS = {
    "srt/runtime_context.py": ('"tp_size"', '"attn_dp_size"', '"moe_ep_size"'),
    "srt/arg_groups/fields/*.py": (
        "tp_size",
        "attn_dp_size",
        "ep_size",
        "dp_size",
        "enable_dp_attention",
    ),
}

TOPOLOGIES = frozenset({"tp2", "tp4", "tp8", "tep8", "dep8"})
LEGACY = frozenset({"tp8", "tep8"})

# Extend a slot's entry only alongside its topology-specific build/shape checks
# and correctness evidence. A missing entry supports no topology.
SLOT_TOPOLOGIES = {
    name: LEGACY
    for name in (
        "route",
        "route_topk",
        "route_gemm",
        "mkm",
        "mkm_wide",
        "mkm_m32",
        "mkm_m64",
        "mkm_m128",
        "mkm_m256",
        "mkm_m512",
        "mkm_wide_bal",
        "gdn_verify",
        "gdn_pdl",
        "gdn_conv",
        "gdn_core",
        "gdn_push",
        "gdn_front",
        "gdn_trt_front",
        "hop",
        "hop_mid",
        "hop_fp8",
        "route_shared",
        "route_lin",
        "skv2",
        "skv2_cp",
        "mka",
        "mka_push",
        "lowm_in",
        "lowm_out",
        "lowm_se",
        "lowm_push",
        "draft_loop",
        "draft_extend",
        "draft_moe",
        "draft_q1",
        "draft_xchg",
        "draft_hop",
        "draft_fin",
        "draft_front",
        "draft_skv2",
        "draft_multi",
        "draft_embed",
        "draft_q1_cp",
        "draft_skv2_cp",
    )
}
# These names select coverage or SGLang's own runtime policy, not a plugin kernel.
SLOT_TOPOLOGIES.update(
    multi_request=TOPOLOGIES,
    gdn_commit=TOPOLOGIES,
    sharded_gumbel=TOPOLOGIES,
)


class Topology(msgspec.Struct, frozen=True):
    tp_size: int
    attn_dp_size: int = 1
    moe_ep_size: int = 1

    @property
    def name(self) -> str:
        if self.attn_dp_size > 1:
            return (
                "dep8"
                if (self.tp_size, self.attn_dp_size, self.moe_ep_size) == (8, 8, 8)
                else "unsupported"
            )
        if self.tp_size == 8 and self.moe_ep_size > 1:
            return "tep8"
        return f"tp{self.tp_size}"

    @property
    def stock_widths(self) -> bool:
        """New topologies cover all offered widths even with every slot off."""
        return self.name in {"tp2", "tp4", "dep8"}


def from_parallel(parallel) -> Topology:
    return Topology(
        parallel.tp_size,
        getattr(parallel, "attn_dp_size", 1),
        parallel.moe_ep_size,
    )


def from_config(cfg) -> Topology:
    # Startup guards may run before legacy --dp-size/--enable-dp-attention is
    # normalized to attn_dp_size; the published ParallelContext is already resolved.
    attention_dp = cfg.attn_dp_size or 1
    if cfg.enable_dp_attention:
        attention_dp = max(attention_dp, cfg.dp_size or 1)
    return Topology(cfg.tp_size, attention_dp, cfg.ep_size or 1)


def select(requested: frozenset, topology: Topology) -> frozenset:
    """Only explicitly declared support enables a kernel on this topology."""
    return frozenset(
        name
        for name in requested
        if topology.name in SLOT_TOPOLOGIES.get(name, frozenset())
    )
