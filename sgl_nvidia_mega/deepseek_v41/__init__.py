# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash monokernels: the specs (``spec``, ``draft``), the slot plan
(``plan``), the routed forward and its SGLang glue (``forward``), and the SM103
(GB300) kernels (``sm103``)."""

# DeepSeek-V4.1-Flash's shapes, routing and activation, which the kernels build in
HF_CONFIG = dict(
    hidden_size=5120,
    moe_intermediate_size=2304,
    n_routed_experts=384,
    num_experts_per_tok=6,
    n_shared_experts=1,
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    topk_method="noaux_tc",
    hidden_act="silu",
    swiglu_limit=10.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-20,
    head_dim=512,
    q_lora_rank=1280,
    qk_rope_head_dim=64,
    sliding_window=128,
)
