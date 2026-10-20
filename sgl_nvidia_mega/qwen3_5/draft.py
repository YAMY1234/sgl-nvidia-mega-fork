# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5's MTP draft on SM103 (GB300): the megakernel draft spec.

The draft's forward stays SGLang's (the built object covers no step width), and the
built object offers no EAGLE entry yet (``draft_loop``, ``draft_extend``,
``begin_draft_round``): every draft phase runs SGLang's own path.
"""

from __future__ import annotations

from sglang.srt.megakernels.spec import Feature, MegakernelSpec, Requirements

from sgl_nvidia_mega.qwen3_5 import spec as target

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Requirements",
    ),
}

ARCHITECTURE = "Qwen3_5ForCausalLMMTP"


class MtpDraft:
    """The draft spec's built object: SGLang's forward on every step."""

    widths: tuple[int, ...] = ()
    draft_sampler = None  # SGLang's draft proposal

    def __init__(self, lm):
        self.lm = lm

    def summary(self) -> str:
        return "Qwen3.5 MTP draft: SGLang's forward (no draft phase ported yet)"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        return None

    def forward(self, step, original, *args, **kwargs):
        raise AssertionError("the MTP draft covers no step width")


def _build(model, hf_config, ctx) -> MtpDraft:
    return MtpDraft(model)


SPEC = MegakernelSpec(
    name="Qwen3.5 MTP draft",
    # the target's devices, TP size, dtype and decode graphs: the draft shares its GPUs
    requires=Requirements(
        isas=target.REQUIREMENTS.isas,
        tp_sizes=target.REQUIREMENTS.tp_sizes,
        dtypes=target.REQUIREMENTS.dtypes,
        decode_graph_backends=target.REQUIREMENTS.decode_graph_backends,
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=target.NUMERICS,
    # the draft's forward stays SGLang's: no width is built
    max_rows=0,
    build=_build,
    api_version=target.API_VERSION,
)
