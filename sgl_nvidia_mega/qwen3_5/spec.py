# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the server on
every rank, listing every reason. This module stays light: SGLang imports it in the
launcher.
"""

from __future__ import annotations

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_nvidia_mega.qwen3_5 import TEXT_CONFIG, runtime
from sgl_nvidia_mega.qwen3_5.plan import MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MULTI_NODE_TP",
        "KV_OFFLOAD",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "BITWISE",
        "TOLERANCE",
        "Requirements",
        "decode_graph_backends",
    ),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_use_rejection_sampling",
        "moe_a2a_backend",
        "moe_runner_backend",
        "flashinfer_allreduce_fusion_backend",
        "attention_backend",
        "decode_attention_backend",
        "linear_attn_backend",
        "linear_attn_decode_backend",
        "linear_attn_verify_backend",
        "enable_linear_replayssm_spec",
        "mamba_ssm_dtype",
        "startup_weight_load_mode",
    ),
    "srt/configs/qwen3_5.py": ("text_config",),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "EagleForwardView",
    ),
}

ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
# The SGLang megakernel contract this spec is written against.
API_VERSION = 2

REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    # TP8 over the two 4-GPU trays of one NVL72 rack, TP4 on one tray
    tp_sizes=frozenset({4, 8}),
    dtypes=frozenset({"bfloat16"}),
    # the decode CUDA graphs the covered step is tested with (eager steps too)
    decode_graph_backends=frozenset({"full", "disabled"}),
    server_args={
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        # MTP (NEXTN resolves to EAGLE) as a chain: uniform verify steps of 7 rows
        "speculative_algorithm": frozenset({"EAGLE"}),
        "speculative_eagle_topk": frozenset({1}),
        # SGLang's EAGLE entries serve a chain without rejection sampling
        "speculative_use_rejection_sampling": frozenset({False}),
        "moe_a2a_backend": frozenset({"none"}),
        # the deferred MoE finalize the boundary takes (the measured TP8 setup)
        "moe_runner_backend": frozenset({"flashinfer_trtllm"}),
        "flashinfer_allreduce_fusion_backend": frozenset({"cutedsl"}),
        "attention_backend": frozenset({"trtllm_mha"}),
        "decode_attention_backend": frozenset({None, "trtllm_mha"}),
        # the Triton GDN verify on the ReplaySSM ring with BF16 checkpoints
        "linear_attn_backend": frozenset({"triton"}),
        "linear_attn_decode_backend": frozenset({None, "triton"}),
        "linear_attn_verify_backend": frozenset({None, "triton"}),
        "enable_linear_replayssm_spec": frozenset({True}),
        "mamba_ssm_dtype": frozenset({"bfloat16"}),
        # the kernels read the weights SGLang loaded, after its last conversion
        "startup_weight_load_mode": frozenset({"serial"}),
    },
    # the text shapes live in text_config, which config_disable_reasons checks
    hf_config={"model_type": "qwen3_5_moe"},
)

# Against SGLang's forward on the same GPU (test/e2e/test_qwen3_5_stock_parity.py): the
# stock step and the bitwise slots (mkm, gdn_verify, gdn_pdl, hop) reproduce it bit for
# bit. A numeric slot's outputs are held to stock or fp64 on the same quantized operands
# (test/kernels), at most these multiples of the error of SGLang's own op.
NUMERIC_TOLERANCES = {
    # Four router GEMV chains sum in another order than SGLang's cuBLAS GEMM.
    "route": {"route.router_logits.bf16_ulp_vs_stock": 1.0},
    # The tiled router front stays within one BF16 ULP of SGLang's whole-M GEMM.
    "route_gemm": {"route_gemm.router_logits.bf16_ulp_vs_stock": 1.0},
    "gdn_push": {
        "gdn_push.out_proj.rel_l2_vs_stock": 1.05,
        "gdn_push.out_proj.max_abs_vs_stock": 2.0,
    },
    "gdn_front": {
        "gdn_front.in_proj_qkvz.rel_l2_vs_stock": 1.05,
        "gdn_front.in_proj_qkvz.max_abs_vs_stock": 2.0,
        "gdn_front.in_proj_ba.rel_l2_vs_stock": 1.05,
        "gdn_front.in_proj_ba.max_abs_vs_stock": 2.0,
    },
    "route_shared": {
        "route_shared.gate_up.rel_l2_vs_stock": 1.05,
        "route_shared.gate_up.max_abs_vs_stock": 2.0,
    },
    # SKV2 against SGLang's TRT-LLM MHA decode (test_skv2_is_within_twice_sglangs_error)
    "skv2": {"skv2.attention.rel_l2_vs_stock": 2.0},
    # CP-lite's exact path (SGLang's q35_cp_lite) at one request, as skv2's bound
    "skv2_cp": {"skv2_cp.attention.rel_l2_vs_stock": 2.0},
    "draft_q1_cp": {"draft_q1_cp.attention.rel_l2_vs_stock": 2.0},
    "draft_skv2_cp": {"draft_skv2_cp.attention.rel_l2_vs_stock": 2.0},
    # MKA's own qkv / o_proj GEMMs; the GEMM slots' bounds, no fp64 kernel test yet
    "mka": {
        "mka.qkv_proj.rel_l2_vs_stock": 1.05,
        "mka.qkv_proj.max_abs_vs_stock": 2.0,
        "mka.o_proj.rel_l2_vs_stock": 1.05,
        "mka.o_proj.max_abs_vs_stock": 2.0,
    },
    # the low-M launches against fp64 on the same E4M3 operands (test/kernels/test_qwen3_5_lowm.py)
    "lowm_in": {
        "lowm_in.in_proj_qkvz.rel_max_vs_stock": 1.05,
        "lowm_in.in_proj_ba.rel_max_vs_stock": 1.05,
        "lowm_in.qkv_proj.rel_max_vs_stock": 1.05,
    },
    "lowm_out": {
        "lowm_out.out_proj.rel_max_vs_stock": 1.05,
        "lowm_out.o_proj.rel_max_vs_stock": 1.05,
    },
    "lowm_se": {"lowm_se.shared_expert.rel_l2_vs_stock": 2e-2},
    # the pushed partial is lowm_out's; the all-reduce that consumes it is the fusion's (test_qwen3_5_hop.py)
    "lowm_push": {
        "lowm_push.out_proj.rel_max_vs_stock": 1.05,
        "lowm_push.o_proj.rel_max_vs_stock": 1.05,
    },
}


def numerics(slots) -> Numerics:
    """The claim for the requested ``slots``: bitwise unless a numeric slot is on."""
    tolerances = {
        k: v
        for slot, held in NUMERIC_TOLERANCES.items()
        if slot in slots
        for k, v in held.items()
    }
    if not tolerances:
        return Numerics(kind=NumericsKind.BITWISE)
    return Numerics(kind=NumericsKind.TOLERANCE, tolerances=tolerances)


NUMERICS = numerics(runtime.SLOTS)


def text_config_reasons(hf_config) -> list[str]:
    """The text_config fields the kernels are built for that differ."""
    text = hf_config.text_config
    return [
        f"config.text_config.{key} {getattr(text, key, None)!r} is not {want!r}"
        for key, want in TEXT_CONFIG.items()
        if getattr(text, key, None) != want
    ]


def binding_reasons() -> list[str]:
    """[] when this SGLang has the EAGLE binding whose forward view the covered steps
    read (``Step.view``, an ``EagleForwardView``)."""
    try:
        from sglang.srt.kernel_plugins.nvidia.eagle_binding import (  # noqa: F401
            EagleForwardView,
        )
    except ImportError as e:
        return [
            f"this SGLang has no EAGLE forward view "
            f"(kernel_plugins/nvidia/eagle_binding.py: {e})"
        ]
    return []


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    runtime.apply_expert_parallel(cfg.ep_size or 1)
    return text_config_reasons(hf_config) + binding_reasons() + runtime.slot_reasons()


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.qwen3_5 import runtime

    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="Qwen3.5",
    requires=REQUIREMENTS,
    features=frozenset(
        {
            Feature.SPECULATIVE_DECODING,
            # TP8 spans the two 4-GPU trays of one NVL72 rack
            Feature.MULTI_NODE_TP,
            # HiCache: covered steps read only running requests' pages
            Feature.KV_OFFLOAD,
            # TEP (MoE TP x EP): the slots in runtime.EP_STOCK_SLOTS stay off
            Feature.EXPERT_PARALLEL,
        }
    ),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    config_disable_reasons=_config_disable_reasons,
)
