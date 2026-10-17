# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash's DSpark draft on SM103 (GB300): its spec.

SGLang attaches the draft model as it does the target: a routed draft step (a block
of rows a request, run as a TARGET_VERIFY batch) runs the draft forward, which takes
and returns what ``DeepseekV4ForCausalLMDSpark.forward`` does, with the routed
experts of each stage on the GEMM pair. The draft never publishes into the hop's
mailbox.
"""

from __future__ import annotations

from sglang.srt.monokernels.spec import DeviceLimits, MonoSpec, Requirements

from sgl_nvidia_mega.deepseek_v41 import spec as target

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/monokernels/spec.py": ("DeviceLimits", "MonoSpec", "Requirements"),
    "srt/runtime_context.py": ("get_spec",),
    "srt/arg_groups/fields/spec.py": ("speculative_num_draft_tokens",),
    "srt/speculative/dspark_components/dspark_config.py": (
        "dspark_gamma_from_num_draft_tokens",
    ),
    "srt/models/deepseek_v4_dspark.py": ("sample_from_anchor",),
}


def _tokens_per_request(model) -> int:
    """A draft block's rows a request: gamma, or gamma + 1 without the anchor."""
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.dspark_components.dspark_config import (
        dspark_gamma_from_num_draft_tokens,
    )

    gamma = dspark_gamma_from_num_draft_tokens(
        int(get_spec().speculative_num_draft_tokens)
    )
    return gamma if model.sample_from_anchor else gamma + 1


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sgl_nvidia_mega.deepseek_v41 import forward

    return forward.draft_disable_reasons(model, hf_config)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.deepseek_v41 import forward

    return forward.build_draft(model, hf_config, ctx)


SPEC = MonoSpec(
    name="deepseek_v41_dspark_draft_nvidia",
    devices={
        target.ISA: DeviceLimits(
            widths=target.ROUTED_WIDTHS,
            tp_sizes=(4,),
            resident_blocks=target.RESIDENT_BLOCKS,
        )
    },
    features=target.SPEC.features,
    # the framework checks nothing else for a draft spec; its forward re-checks
    requires=Requirements(attention_backends=target.SPEC.requires.attention_backends),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=target.SPEC.api_version,
    tokens_per_request=_tokens_per_request,
)
