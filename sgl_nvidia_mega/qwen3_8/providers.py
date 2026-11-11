# SPDX-License-Identifier: Apache-2.0
"""Finite, exact declarations built before autotune and graph capture.

The framework calls register_signatures from model build/preparation, after it
knows the shapes and has prepared the matching views. Importing this module does
not register shapes, load GPU libraries, or change any framework entry point.
"""

from functools import cache

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizePermutedSignature,
    GdnSignature,
    HcSignature,
    MtpCaptureSeedSignature,
    MtpDeferredKvSignature,
    NativeGdnPdlSignature,
    NumericOffer,
    NumericOperation,
    NumericSignature,
    RouterQuantSignature,
    ScalarType,
    numeric_registry,
)

from .slots import enabled, log_selection

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "ExpertSignature",
        "FinalizePermutedSignature",
        "GdnSignature",
        "HcSignature",
        "NativeGdnPdlSignature",
        "MtpCaptureSeedSignature",
        "MtpDeferredKvSignature",
        "NumericOffer",
        "NumericOperation",
        "NumericSignature",
        "RouterQuantSignature",
        "ScalarType",
        "numeric_registry",
    )
}


def supported(s):
    """False means no offer; stock stays selected before any output is written."""
    op = s.operation
    if op == NumericOperation.POOL_UNION_ATTENTION:
        from ._providers.pool_union import supported as union_supported

        return enabled(op) and union_supported(s)
    if op == NumericOperation.PREFILL_SELECT:
        from ._providers.prefill_select import supported as prefill_supported

        return enabled(op) and prefill_supported(s)
    if op == NumericOperation.PACKED_PREFILL_ATTENTION:
        from ._providers.packed_prefill import supported as packed_supported

        return enabled(op) and packed_supported(s)
    if op == NumericOperation.FINALIZE_PERMUTED_SHARED:
        from ._providers.finalize_permuted import supported as finalize_supported

        return enabled(op) and finalize_supported(s)
    if not enabled(op):
        return False
    if s.rows <= 0:
        return False
    if isinstance(s, MtpDeferredKvSignature):
        from ._providers.mtp_deferred_kv import supported as deferred_supported

        return deferred_supported(s)
    if isinstance(s, MtpCaptureSeedSignature):
        from ._providers.mtp_capture_seed import supported as capture_supported

        return capture_supported(s)
    if isinstance(s, NativeGdnPdlSignature):
        from ._providers.gdn_native_pdl import supported as native_supported

        return native_supported(s)
    if isinstance(s, RouterQuantSignature):
        from ._providers.router_quant import supported as router_supported

        return router_supported(s)
    if isinstance(s, HcSignature):
        if s.scalar_type != ScalarType.BF16:
            return False
        if op == NumericOperation.HC_MIX:
            return (
                s.stream_count == 4
                and s.hidden_size == 2560
                and s.lowrank > 0
                and s.variant in (0, 1)
            )
        return (
            s.stream_count == 4
            and s.hidden_size == 2560
            and s.rows <= 32
            and s.parts in (1, 2, 4)
            and op
            in (
                NumericOperation.HC_GATE,
                NumericOperation.HC_APPLY_NORM,
                NumericOperation.HC_APPLY_NORM_GATE,
            )
        )
    if isinstance(s, ExpertSignature):
        from ._providers.experts import supported as expert_supported

        return expert_supported(s)
    if isinstance(s, GdnSignature):
        if op == NumericOperation.GDN_VERIFY_GLUE:
            from ._providers.gdn_verify_glue import supported as glue_supported

            return glue_supported(s)
        if op == NumericOperation.GDN_FRONT:
            from ._providers.gdn_front import supported as front_supported

            return front_supported(s)
        return (
            op == NumericOperation.GDN_CORE
            and s.variant in (0, 1)
            and (
                s.rows in (4, 8, 12)
                or (
                    (s.key_heads, s.value_heads) == (4, 12)
                    and (s.rows == 16 or (s.rows == 32 and s.variant == 1))
                )
            )
            and (s.key_heads, s.value_heads) in ((4, 12), (16, 48))
            and s.key_dim == s.value_dim == 128
            and s.tokens_per_request == 4
            and s.scalar_type == ScalarType.BF16
            and s.gate_scalar_type == ScalarType.BF16
        )
    if isinstance(s, NumericSignature):
        if op in (NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT):
            from ._providers.sparse import supported as sparse_supported

            return sparse_supported(s)
        if op == NumericOperation.DIRECT_POOL_ATTENTION and s.variant & ~4 in (1, 2):
            from ._providers.pool import supported as pool_supported

            return pool_supported(s)
        return (
            op == NumericOperation.DIRECT_POOL_ATTENTION
            and s.variant == 0
            and (
                s.rows,
                s.query_heads,
                s.kv_heads,
                s.head_dim,
                s.selected_count,
                s.compress_ratio,
                s.scalar_type,
            )
            == (4, 6, 1, 256, 512, 4, ScalarType.BF16)
        )
    return False


@cache
def _cached_offer(signature):
    log_selection()
    if not supported(signature):
        return None
    if signature.operation == NumericOperation.POOL_UNION_ATTENTION:
        from ._providers.pool_union import build

        run = build(signature)
    elif signature.operation == NumericOperation.PREFILL_SELECT:
        from ._providers.prefill_select import build

        run = build(signature)
    elif signature.operation == NumericOperation.PACKED_PREFILL_ATTENTION:
        from ._providers.packed_prefill import build

        run = build(signature)
    elif isinstance(signature, MtpDeferredKvSignature):
        from ._providers.mtp_deferred_kv import build

        run = build(signature)
    elif isinstance(signature, MtpCaptureSeedSignature):
        from ._providers.mtp_capture_seed import build

        run = build(signature)
    elif isinstance(signature, NativeGdnPdlSignature):
        from ._providers.gdn_native_pdl import build

        run = build(signature)
    elif isinstance(signature, RouterQuantSignature):
        from ._providers.router_quant import build

        run = build(signature)
    elif isinstance(signature, FinalizePermutedSignature):
        from ._providers.finalize_permuted import build

        run = build(signature)
    elif isinstance(signature, HcSignature):
        if signature.operation == NumericOperation.HC_MIX:
            from ._providers.hc import build_mix

            run = build_mix(signature)
        else:
            from ._providers.hc import build

            run = build(signature)
    elif isinstance(signature, ExpertSignature):
        from ._providers.experts import build

        run = build(signature)
    elif isinstance(signature, GdnSignature):
        if signature.operation == NumericOperation.GDN_VERIFY_GLUE:
            from ._providers.gdn_verify_glue import build

            run = build(signature)
        elif signature.operation == NumericOperation.GDN_FRONT:
            from ._providers.gdn_front import build

            run = build(signature)
        else:
            from ._providers.gdn import build_core

            run = build_core(signature)
    else:
        if signature.operation in (
            NumericOperation.SPARSE_SCORE,
            NumericOperation.SPARSE_SELECT,
        ):
            from ._providers.sparse import build

            run = build(signature)
        elif signature.variant & ~4 in (1, 2):
            from ._providers.pool import build

            run = build(signature)
        else:
            from ._providers.qsa import direct_pool as run
    return NumericOffer(signature, run)


def make_offer(signature):
    if signature.operation == NumericOperation.POOL_UNION_ATTENTION:
        from ._providers.pool_union import supported as union_supported

        if not union_supported(signature):
            return None
    if signature.operation == NumericOperation.PREFILL_SELECT:
        from ._providers.prefill_select import supported as prefill_supported

        if not prefill_supported(signature):
            return None
    if signature.operation == NumericOperation.PACKED_PREFILL_ATTENTION:
        from ._providers.packed_prefill import supported as packed_supported

        if not packed_supported(signature):
            return None
    if isinstance(signature, MtpDeferredKvSignature):
        from ._providers.mtp_deferred_kv import supported as deferred_supported

        if not deferred_supported(signature):
            return None
    if isinstance(signature, MtpCaptureSeedSignature):
        from ._providers.mtp_capture_seed import supported as capture_supported

        if not capture_supported(signature):
            return None
    # A bool/float can compare equal to an integer dataclass cache key.
    if isinstance(signature, NativeGdnPdlSignature):
        from ._providers.gdn_native_pdl import supported as native_supported

        if not native_supported(signature):
            return None
    # Reject noninteger router declarations before bool/int cache equality.
    if isinstance(signature, RouterQuantSignature):
        from ._providers.router_quant import supported as router_supported

        if not router_supported(signature):
            return None
    # bool/int fields compare equal inside frozen dataclasses. Reject invalid F
    # declarations before a cached integer signature can alias a bool request.
    if isinstance(signature, FinalizePermutedSignature):
        from ._providers.finalize_permuted import supported as finalize_supported

        if not finalize_supported(signature):
            return None
    return _cached_offer(signature)


make_offer.cache_clear = _cached_offer.cache_clear
make_offer.cache_info = _cached_offer.cache_info
make_offer.cache_parameters = _cached_offer.cache_parameters
make_offer.__wrapped__ = _cached_offer.__wrapped__


def register_signatures(signatures):
    """Register only the explicit finite capabilities from build/preparation.

    All required descriptor/JIT preparation and eager warmup must finish before
    capture. Nothing calls this function from numeric_operator or graph replay.
    """
    log_selection()
    offers = tuple(offer for s in signatures if (offer := make_offer(s)) is not None)
    for offer in offers:
        numeric_registry.register(offer)
    return offers
