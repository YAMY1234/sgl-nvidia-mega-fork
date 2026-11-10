# SPDX-License-Identifier: Apache-2.0
"""Numeric providers only: the framework always retains the model forward."""

from dataclasses import dataclass

from sglang.kernels.plugin_kernels.nvidia.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from .capabilities import numeric_capability
from .providers import register_signatures

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "numeric_max_rows",
        "numeric_signatures",
        "numeric_capability",
    ),
    "srt/kernel_plugins/nvidia/binding.py": (
        "PreparationView",
        "numeric_signatures",
    ),
}

# This bounds framework enumeration; every registration is still an exact tuple.
# Rows 0 is one row-independent declaration (prefill select), not an enumeration.
NUMERIC_MAX_ROWS = 1024


@dataclass
class BuiltNumeric:
    offers: tuple
    widths: tuple = ()
    draft_sampler: object = None
    draft_loop: object = None
    draft_extend: object = None
    begin_draft_round: object = None
    graph_variants: object = None
    speculative_runtime_hooks: object = None

    def summary(self):
        return (
            f"{len(self.offers)} rank-local numeric capabilities; stock model forward"
        )

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        return None

    def forward(self, step, original, *args, **kwargs):
        return original(*args, **kwargs)


def build(model, hf_config, ctx):
    # Model and runner objects never reach an operator adapter.
    signatures = tuple(
        s for s in ctx.numeric_signatures if 0 <= s.rows <= NUMERIC_MAX_ROWS
    )
    return BuiltNumeric(register_signatures(signatures))


def prepare(built, view):
    """Register framework-resolved pool capacities before eager graph warmup."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("numeric capabilities must be prepared before capture")
    signatures = getattr(view, "numeric_signatures", ())
    if not isinstance(signatures, tuple):
        raise TypeError("prepared numeric signatures must be a frozen tuple")
    offers = register_signatures(
        tuple(s for s in signatures if 0 <= s.rows <= NUMERIC_MAX_ROWS)
    )
    existing = {offer.signature for offer in built.offers}
    built.offers += tuple(offer for offer in offers if offer.signature not in existing)


SPEC = MegakernelSpec(
    name="Qwen3.8 rank-local numeric providers",
    requires=Requirements(
        isas=frozenset({"sm_107", "sm_107a"}),
        tp_sizes=frozenset({1, 4}),
        dtypes=frozenset({"bfloat16"}),
    ),
    # Every unsupported feature/layout stays stock via the producer's prepared gate.
    features=frozenset(Feature),
    # Recurrence tensor-core arithmetic requires a tolerance gate; exact operators
    # retain their original arithmetic and ordered top-k semantics.
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        tolerances={
            "gdn.output.rel_l2_vs_stock": 1.05,
            "gdn.output.max_abs_vs_stock": 2.0,
        },
    ),
    max_rows=0,
    numeric_max_rows=NUMERIC_MAX_ROWS,
    build=build,
    prepare=prepare,
    api_version=2,
    numeric_capability=numeric_capability,
)
