# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 specs: the executor owns the target's and the MTP draft's forward."""

from sglang.kernels.plugin_kernels.nvidia.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.qwen3_8 import runtime

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
}

_REQUIRES = Requirements(
    isas=frozenset({"sm_100", "sm_100a", "sm_103", "sm_103a", "sm_107", "sm_107a"}),
    tp_sizes=frozenset({1, 4}),
    dtypes=frozenset({"bfloat16"}),
)
# The served recipes' features; the executor's stock step functions are SGLang's
# own forward, so these change no step it runs.
_FEATURES = frozenset(
    {
        Feature.SPECULATIVE_DECODING,
        Feature.KV_OFFLOAD,
        Feature.PD_DISAGGREGATION,
        Feature.MEMORY_RELEASE,
        Feature.DETERMINISTIC_INFERENCE,
    }
)
# S0 runs SGLang's own ops; kernel slots declare their tolerances as they land.
_NUMERICS = Numerics(kind=NumericsKind.BITWISE)
# The target's step view: the typed EAGLE hybrid view (QSA step/pool views, GDN pools
# incl. the u-cache rings, layer readiness) where SGLang offers binding kinds. The
# MTP draft keeps the default view: the EAGLE prefill view needs a linear-attention
# backend, which the draft has not.
_TARGET_BINDING = (
    {"binding": "eagle_hybrid"}
    if "binding" in getattr(MegakernelSpec, "__struct_fields__", ())
    else {}
)


SPEC = MegakernelSpec(
    name="Qwen3.8 forward executor",
    requires=_REQUIRES,
    features=_FEATURES,
    numerics=_NUMERICS,
    max_rows=runtime.MAX_ROWS,
    build=runtime.build,
    api_version=2,
    model_disable_reasons=runtime.model_reasons,
    config_disable_reasons=runtime.config_reasons,
    **_TARGET_BINDING,
)

DRAFT_SPEC = MegakernelSpec(
    name="Qwen3.8 MTP draft executor",
    requires=_REQUIRES,
    features=_FEATURES,
    numerics=_NUMERICS,
    max_rows=runtime.MAX_ROWS,
    build=runtime.build,
    api_version=2,
    model_disable_reasons=runtime.model_reasons,
)
