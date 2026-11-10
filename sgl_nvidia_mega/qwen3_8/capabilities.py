# SPDX-License-Identifier: Apache-2.0
"""Pure startup admission; preparing an exact offer is a separate step."""

from dataclasses import fields

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizePermutedSignature,
    GdnFrontVariant,
    GdnSignature,
    HcSignature,
    MtpCaptureSeedSignature,
    MtpDeferredKvSignature,
    NativeGdnPdlSignature,
    NumericCapabilityQuery,
    NumericOperation,
    NumericSignature,
    RouterQuantSignature,
)

from .providers import supported

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "ExpertSignature",
        "FinalizePermutedSignature",
        "GdnFrontVariant",
        "GdnSignature",
        "HcSignature",
        "MtpCaptureSeedSignature",
        "MtpDeferredKvSignature",
        "NativeGdnPdlSignature",
        "NumericCapabilityQuery",
        "NumericOperation",
        "NumericSignature",
        "RouterQuantSignature",
    ),
}

_FIXED_CONTEXTS = {
    NumericOperation.MTP_CAPTURE_SEED: {
        "tp_size": 4,
        "sm100_supported": 1,
        "is_draft_worker": 1,
        "indexer_topk": 2048,
        "compress_ratio": 4,
        "tail_width": 4,
        "tokens_per_request": 4,
    },
    NumericOperation.MTP_DEFERRED_KV_PREPARE: {
        "tp_size": 4,
        "sm100_supported": 1,
        "is_draft_worker": 1,
        "hidden_size": 2560,
        "hc_count": 4,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "indexer_topk": 2048,
        "compress_ratio": 4,
        "query_heads": 6,
        "tokens_per_request": 4,
    },
    NumericOperation.FINALIZE_PERMUTED_SHARED: {
        "tp_size": 1,
        "ep_size": 1,
        "expert_count": 512,
    },
}

_GDN_CONTEXT = (
    "tp_size",
    "hidden_size",
    "wide_enabled",
    "m32_norm_enabled",
    "projection_norm_available",
    "m32_norm_available",
    "requested_front_stages",
    "front_stages_specified",
)
_SPARSE_CONTEXT = ("tp_size", "sm100_supported", "page_size", "unpadded_query_heads")
_SCHEMAS = {
    NumericOperation.POOL_UNION_ATTENTION: (NumericSignature, ("tp_size",)),
    **{
        op: (HcSignature, ())
        for op in (
            NumericOperation.HC_GATE,
            NumericOperation.HC_APPLY_NORM,
            NumericOperation.HC_APPLY_NORM_GATE,
            NumericOperation.HC_MIX,
        )
    },
    **{
        op: (GdnSignature, _GDN_CONTEXT)
        for op in (
            NumericOperation.GDN_FRONT,
            NumericOperation.GDN_VERIFY_GLUE,
        )
    },
    **{
        op: (NumericSignature, _SPARSE_CONTEXT)
        for op in (
            NumericOperation.SPARSE_SCORE,
            NumericOperation.SPARSE_SELECT,
            NumericOperation.PREFILL_SELECT,
        )
    },
    NumericOperation.GDN_CORE: (
        GdnSignature,
        _GDN_CONTEXT + ("normalization_program",),
    ),
    NumericOperation.GDN_NATIVE_PDL: (NativeGdnPdlSignature, ()),
    NumericOperation.DIRECT_POOL_ATTENTION: (NumericSignature, ("tp_size",)),
    NumericOperation.LOCAL_EXPERTS: (
        ExpertSignature,
        (
            "tokens_per_request",
            "sf_pack_enabled",
            "mkm_wide_enabled",
            "mkm_wide_m8_enabled",
            "wide_batch34_enabled",
            "is_target_verify",
            "batch_size",
        ),
    ),
    NumericOperation.FINALIZE_SHARED: (ExpertSignature, ()),
    NumericOperation.FINALIZE_PERMUTED_SHARED: (
        FinalizePermutedSignature,
        tuple(_FIXED_CONTEXTS[NumericOperation.FINALIZE_PERMUTED_SHARED]),
    ),
    NumericOperation.ROUTER_QUANT: (
        RouterQuantSignature,
        ("tp_size", "ep_size", "wide_enabled", "k4_enabled"),
    ),
    NumericOperation.MTP_CAPTURE_SEED: (
        MtpCaptureSeedSignature,
        tuple(_FIXED_CONTEXTS[NumericOperation.MTP_CAPTURE_SEED]),
    ),
    NumericOperation.MTP_DEFERRED_KV_PREPARE: (
        MtpDeferredKvSignature,
        tuple(_FIXED_CONTEXTS[NumericOperation.MTP_DEFERRED_KV_PREPARE]),
    ),
}


def _source_supported(s, actual):
    op = s.operation
    if op == NumericOperation.HC_MIX:
        return s.rows <= 32
    if op == NumericOperation.POOL_UNION_ATTENTION:
        return actual["tp_size"] == 1
    if op == NumericOperation.PREFILL_SELECT:
        return (
            actual["tp_size"] > 0
            and actual["sm100_supported"] == 1
            and actual["unpadded_query_heads"] == s.query_heads
        )
    fixed = _FIXED_CONTEXTS.get(op, {})
    if any(actual[name] != value for name, value in fixed.items()):
        return False
    if op == NumericOperation.MTP_CAPTURE_SEED:
        return s.trash_row > 0
    if op == NumericOperation.MTP_DEFERRED_KV_PREPARE:
        # The graph producer is narrower than synthetic raw-layout coverage.
        return (
            s.new_k_row_stride == s.new_v_row_stride == 256
            and s.metadata_dtype_mask in (17, 25)
        )
    if isinstance(s, GdnSignature):
        tp = actual["tp_size"]
        flags = (
            "wide_enabled",
            "m32_norm_enabled",
            "projection_norm_available",
            "m32_norm_available",
            "front_stages_specified",
        )
        if (
            tp <= 0
            or actual["hidden_size"] <= 0
            or any(actual[name] not in (0, 1) for name in flags)
            or (
                not actual["front_stages_specified"]
                and actual["requested_front_stages"] != 0
            )
        ):
            return False
        if op == NumericOperation.GDN_VERIFY_GLUE:
            return tp == 1
        if op == NumericOperation.GDN_CORE:
            program = actual["normalization_program"]
            if s.variant == 0:
                if program != 0:
                    return False
            elif s.variant != 1 or program not in (1, 2):
                return False
            if program == 2:
                return (
                    s.rows == 32
                    and tp == 4
                    and actual["m32_norm_enabled"] == 1
                    and actual["m32_norm_available"] == 1
                )
        if s.rows == 32:
            return False
        if s.rows == 16 and not (tp == 4 and actual["wide_enabled"] == 1):
            return False
        if op == NumericOperation.GDN_FRONT or s.variant == 1:
            if tp not in (1, 4) or actual["projection_norm_available"] != 1:
                return False
            channels = 2 * s.key_heads * s.key_dim + s.value_heads * s.value_dim
            stages = (
                actual["requested_front_stages"]
                if s.rows == 4 and channels == 2560 and actual["front_stages_specified"]
                else 6
            )
            if stages not in (6, 12):
                return False
            if op == NumericOperation.GDN_FRONT:
                variant = (
                    GdnFrontVariant.MK_STAGES_6
                    if stages == 6
                    else GdnFrontVariant.MK_STAGES_12
                )
                return actual["hidden_size"] == 2560 and s.variant == variant
        return True
    if op in (NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT):
        if (
            actual["tp_size"] <= 0
            or actual["sm100_supported"] != 1
            or actual["page_size"] != 16
        ):
            return False
        heads = actual["unpadded_query_heads"]
        if not 0 < heads <= 16 or s.query_heads != ((heads + 7) // 8) * 8:
            return False
        if (
            (op == NumericOperation.SPARSE_SCORE and s.variant in (0, 3, 6, 7))
            or (op == NumericOperation.SPARSE_SELECT and s.variant == 0)
        ) and heads != 4:
            return False
        if (op == NumericOperation.SPARSE_SCORE and s.variant in (4, 5, 6, 7)) or (
            op == NumericOperation.SPARSE_SELECT and s.variant == 1
        ):
            return actual["tp_size"] == 1
        return True
    if op == NumericOperation.DIRECT_POOL_ATTENTION:
        return actual["tp_size"] == (4 if s.variant == 0 else 1)
    if op == NumericOperation.LOCAL_EXPERTS:
        flags = (
            "sf_pack_enabled",
            "mkm_wide_enabled",
            "mkm_wide_m8_enabled",
            "wide_batch34_enabled",
            "is_target_verify",
        )
        if (
            any(actual[key] not in (0, 1) for key in flags)
            or actual["tokens_per_request"] <= 0
            or actual["batch_size"] < 0
        ):
            return False
        if s.rows <= 7:
            return s.variant == int(s.rows in (1, 4) and actual["sf_pack_enabled"])
        if not actual["mkm_wide_enabled"]:
            return False
        if actual["wide_batch34_enabled"]:
            return (
                s.variant == 2
                and s.rows in (12, 16)
                and actual["tokens_per_request"] == 4
                and actual["is_target_verify"] == 1
                and actual["batch_size"] in (3, 4)
                and s.rows == 4 * actual["batch_size"]
            )
        return s.variant == 0 and (s.rows != 8 or actual["mkm_wide_m8_enabled"] == 1)
    if op == NumericOperation.ROUTER_QUANT:
        if (
            actual["tp_size"] != 4
            or actual["ep_size"] != 1
            or actual["wide_enabled"] not in (0, 1)
            or actual["k4_enabled"] not in (0, 1)
        ):
            return False
        if s.rows in (1, 4):
            return s.variant == actual["k4_enabled"]
        return actual["wide_enabled"] == 1 and s.variant == 2
    return True


def numeric_capability(query):
    """Answer from host facts only, before registry entries or pools are built."""
    if not isinstance(query, NumericCapabilityQuery):
        return False
    schema = _SCHEMAS.get(query.operation)
    if schema is None:
        return False
    signature_type, context = schema
    names = {field.name for field in fields(signature_type)} - {"operation"}
    actual = dict(query.attributes)
    if actual.keys() != names | set(context):
        return False
    signature = signature_type(
        operation=query.operation,
        **{name: actual[name] for name in names},
    )
    return supported(signature) and _source_supported(signature, actual)
