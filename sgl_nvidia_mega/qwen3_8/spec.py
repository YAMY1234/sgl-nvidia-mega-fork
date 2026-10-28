# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.8 Flash Next (FN) megakernel's spec: the setups it serves and its build.

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

from sgl_nvidia_mega.qwen3_8 import runtime
from sgl_nvidia_mega.qwen3_8.plan import MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
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
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_use_rejection_sampling",
        "moe_a2a_backend",
    ),
}

ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
# The SGLang megakernel contract this spec is written against.
API_VERSION = 2

REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    # TP4 on one 4-GPU tray
    tp_sizes=frozenset({4}),
    dtypes=frozenset({"bfloat16"}),
    # the decode CUDA graphs the covered step is tested with (eager steps too)
    decode_graph_backends=frozenset({"full", "disabled"}),
    server_args={
        # MTP (NEXTN resolves to EAGLE) as a chain: uniform verify steps
        "speculative_algorithm": frozenset({"EAGLE"}),
        "speculative_eagle_topk": frozenset({1}),
        "speculative_use_rejection_sampling": frozenset({False}),
        "moe_a2a_backend": frozenset({"none"}),
    },
    hf_config={"model_type": "qwen4_exp"},
)

# A numeric slot's outputs are held to fp64 on the same operands (test/kernels), at
# most these multiples of the error of SGLang's own op; every other slot is bitwise.
NUMERIC_TOLERANCES: dict[str, dict[str, float]] = {}


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


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    return runtime.slot_reasons()


def _build(model, hf_config, ctx):
    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="Qwen3.8 Flash Next",
    requires=REQUIREMENTS,
    features=frozenset(
        {
            Feature.SPECULATIVE_DECODING,
            # HiCache: covered steps read only running requests' pages
            Feature.KV_OFFLOAD,
        }
    ),
    numerics=NUMERICS,
    max_rows=MAX_ROWS,
    build=_build,
    api_version=API_VERSION,
    config_disable_reasons=_config_disable_reasons,
)
