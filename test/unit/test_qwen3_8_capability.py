# SPDX-License-Identifier: Apache-2.0
"""Startup queries cannot compile, register, write outputs, or widen source domains."""

from dataclasses import fields, replace

import pytest
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizePermutedSignature,
    GdnSignature,
    HcSignature,
    MtpCaptureSeedSignature,
    MtpDeferredKvSignature,
    NativeGdnPdlSignature,
    NumericCapabilityQuery,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericSignature,
    RouterQuantSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import capabilities, providers, slots
from sgl_nvidia_mega.qwen3_8.spec import SPEC

SEED_CONTEXT = dict(
    tp_size=4,
    sm100_supported=1,
    is_draft_worker=1,
    indexer_topk=2048,
    compress_ratio=4,
    tail_width=4,
    tokens_per_request=4,
)
KV_CONTEXT = dict(
    tp_size=4,
    sm100_supported=1,
    is_draft_worker=1,
    hidden_size=2560,
    hc_count=4,
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    indexer_topk=2048,
    compress_ratio=4,
    query_heads=6,
    tokens_per_request=4,
)
SEED = MtpCaptureSeedSignature(
    O.MTP_CAPTURE_SEED,
    4,
    2051,
    2051,
    8,
    2055,
    2055,
    0,
    7,
    D.INT32,
    D.INT32,
    D.INT64,
    D.INT32,
    1,
)
KV = MtpDeferredKvSignature(
    O.MTP_DEFERRED_KV_PREPARE,
    1,
    1,
    256,
    D.BF16,
    2055,
    2112,
    4,
    8,
    32,
    7,
    4096,
    4096,
    256,
    256,
    17,
)
GDN_CONTEXT = dict(
    tp_size=4,
    hidden_size=2560,
    wide_enabled=0,
    m32_norm_enabled=0,
    projection_norm_available=1,
    m32_norm_available=1,
    requested_front_stages=0,
    front_stages_specified=0,
)
LOCAL_CONTEXT = dict(
    tokens_per_request=4,
    sf_pack_enabled=0,
    mkm_wide_enabled=0,
    mkm_wide_m8_enabled=1,
    wide_batch34_enabled=0,
    is_target_verify=1,
    batch_size=1,
)
ROUTER_CONTEXT = dict(tp_size=4, ep_size=1, wide_enabled=0, k4_enabled=0)


def representatives():
    """Typed CPU ABI examples, not measured tensors or per-role declarations."""
    return (
        (
            NumericSignature(O.POOL_UNION_ATTENTION, 64, 24, 2, 256, 512, 4, D.BF16),
            dict(tp_size=1),
        ),
        *(
            (HcSignature(op, 4, 4, 2560, D.BF16, parts=2), {})
            for op in (O.HC_GATE, O.HC_APPLY_NORM, O.HC_APPLY_NORM_GATE)
        ),
        (HcSignature(O.HC_MIX, 4, 4, 2560, D.BF16, lowrank=128, variant=1), {}),
        (
            GdnSignature(
                O.GDN_CORE, 4, 4, 12, 128, 128, 4, D.BF16, gate_scalar_type=D.BF16
            ),
            GDN_CONTEXT | {"normalization_program": 0},
        ),
        (GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16), GDN_CONTEXT),
        (
            GdnSignature(
                O.GDN_VERIFY_GLUE,
                8,
                16,
                48,
                128,
                128,
                4,
                D.BF16,
                gate_scalar_type=D.BF16,
            ),
            GDN_CONTEXT | {"tp_size": 1},
        ),
        (NativeGdnPdlSignature(O.GDN_NATIVE_PDL, 32, 16, 48, 128, 128, 4, D.BF16), {}),
        (
            NumericSignature(
                O.SPARSE_SCORE, 4, 8, 1, 128, 512, 4, D.BF16, context_bound=65536
            ),
            dict(tp_size=4, sm100_supported=1, page_size=16, unpadded_query_heads=4),
        ),
        (
            NumericSignature(
                O.SPARSE_SELECT,
                4,
                8,
                1,
                128,
                512,
                4,
                D.FP32,
                context_bound=65536,
                variant=1,
            ),
            dict(tp_size=1, sm100_supported=1, page_size=16, unpadded_query_heads=4),
        ),
        (
            NumericSignature(O.DIRECT_POOL_ATTENTION, 4, 6, 1, 256, 512, 4, D.BF16),
            dict(tp_size=4),
        ),
        (
            ExpertSignature(O.LOCAL_EXPERTS, 4, 2560, 192, 512, 10, D.PACKED_FP4),
            LOCAL_CONTEXT,
        ),
        (ExpertSignature(O.FINALIZE_SHARED, 4, 2560, 192, 512, 10, D.BF16), {}),
        (
            FinalizePermutedSignature(
                O.FINALIZE_PERMUTED_SHARED, 4, 2560, 10, D.BF16, D.FP32
            ),
            dict(tp_size=1, ep_size=1, expert_count=512),
        ),
        (RouterQuantSignature(O.ROUTER_QUANT, 1, 2560, 512, D.BF16), ROUTER_CONTEXT),
        (SEED, SEED_CONTEXT),
        (KV, KV_CONTEXT),
    )


def role_representatives(tp_size):
    """Source-domain CPU examples; callers must still apply actual role policy."""
    if tp_size not in (1, 4):
        return ()
    source_tp4 = {
        O.LOCAL_EXPERTS,
        O.FINALIZE_SHARED,
        O.ROUTER_QUANT,
        O.MTP_CAPTURE_SEED,
        O.MTP_DEFERRED_KV_PREPARE,
    }
    source_tp1 = {O.GDN_VERIFY_GLUE, O.FINALIZE_PERMUTED_SHARED}
    result = []
    for s, context in representatives():
        if (
            tp_size == 1
            and s.operation in source_tp4
            or tp_size == 4
            and s.operation in source_tp1
        ):
            continue
        if s.operation == O.GDN_NATIVE_PDL and tp_size == 4:
            # The TP4 representative follows MK precedence, not native verify.
            continue
        if isinstance(s, GdnSignature):
            s = replace(
                s,
                key_heads=16 if tp_size == 1 else 4,
                value_heads=48 if tp_size == 1 else 12,
            )
            context = context | {"tp_size": tp_size}
        elif s.operation in (O.SPARSE_SCORE, O.SPARSE_SELECT):
            s = replace(
                s,
                variant=(4 if s.operation == O.SPARSE_SCORE else 1)
                if tp_size == 1
                else 0,
            )
            context = context | {"tp_size": tp_size}
        elif s.operation == O.DIRECT_POOL_ATTENTION:
            if tp_size == 1:
                s = replace(s, query_heads=24, kv_heads=2, variant=1)
            context = context | {"tp_size": tp_size}
        result.append((s, context))
    return tuple(result)


def query(signature, context):
    actual = {
        f.name: getattr(signature, f.name)
        for f in fields(signature)
        if f.name != "operation"
    }
    assert not actual.keys() & context.keys()
    return NumericCapabilityQuery(
        signature.operation, tuple((actual | context).items())
    )


@pytest.fixture(autouse=True)
def enabled_slots(monkeypatch):
    monkeypatch.setattr(slots, "SLOTS", slots.KNOWN_SLOTS)


@pytest.mark.parametrize("signature,context", representatives())
def test_all_selected_operations_complete_query_and_exact_keys(signature, context):
    q = query(signature, context)
    assert capabilities.numeric_capability(q) is True
    assert (
        capabilities.numeric_capability(
            replace(q, attributes=q.attributes + (("unknown", 1),))
        )
        is False
    )
    for key, _ in q.attributes:
        incomplete = tuple(pair for pair in q.attributes if pair[0] != key)
        assert (
            capabilities.numeric_capability(replace(q, attributes=incomplete)) is False
        ), key
    assert (
        capabilities.numeric_capability(query(replace(signature, rows=0), context))
        is False
    )


def test_query_catalogue_does_not_add_or_remove_slots():
    operations = {s.operation for s, _ in representatives()}
    assert operations == set(slots.OPERATION_SLOTS) == set(capabilities._SCHEMAS)
    assert len(operations) == 18 and O.PACKED_ATTENTION not in operations


@pytest.mark.parametrize("tp_size", [1, 4])
def test_role_examples_do_not_force_source_inapplicable_positives(tp_size):
    examples = role_representatives(tp_size)
    assert examples and all(
        capabilities.numeric_capability(query(s, ctx)) for s, ctx in examples
    )
    operations = {s.operation for s, _ in examples}
    if tp_size == 1:
        assert not operations & {
            O.LOCAL_EXPERTS,
            O.FINALIZE_SHARED,
            O.ROUTER_QUANT,
            O.MTP_CAPTURE_SEED,
            O.MTP_DEFERRED_KV_PREPARE,
        }
    else:
        assert not operations & {
            O.GDN_VERIFY_GLUE,
            O.GDN_NATIVE_PDL,
            O.FINALIZE_PERMUTED_SHARED,
        }


@pytest.mark.parametrize("signature,context", [(SEED, SEED_CONTEXT), (KV, KV_CONTEXT)])
def test_query_before_registration_only_reads_host_facts(
    monkeypatch, signature, context
):
    def forbidden(*args, **kwargs):
        pytest.fail("support query attempted preparation or registration")

    class NoRegistry:
        def __getattr__(self, name):
            return forbidden

    monkeypatch.setattr(providers, "numeric_registry", NoRegistry())
    monkeypatch.setattr(providers, "register_signatures", forbidden)
    monkeypatch.setattr(providers, "make_offer", forbidden)
    before = providers._cached_offer.cache_info()
    assert SPEC.numeric_capability is capabilities.numeric_capability
    assert SPEC.numeric_capability(query(signature, context)) is True
    assert providers._cached_offer.cache_info() == before


@pytest.mark.parametrize(
    "signature,context,key",
    [
        (signature, context, key)
        for signature, context in ((SEED, SEED_CONTEXT), (KV, KV_CONTEXT))
        for key in context
    ],
)
def test_every_source_boundary_rejects_a_changed_fact(signature, context, key):
    assert (
        capabilities.numeric_capability(query(signature, context | {key: 0})) is False
    )


@pytest.mark.parametrize("signature,context", [(SEED, SEED_CONTEXT), (KV, KV_CONTEXT)])
def test_missing_extra_and_reordered_keys(signature, context):
    q = query(signature, context)
    assert (
        capabilities.numeric_capability(replace(q, attributes=q.attributes[:-1]))
        is False
    )
    assert (
        capabilities.numeric_capability(
            replace(q, attributes=q.attributes + (("unknown", 1),))
        )
        is False
    )
    assert (
        capabilities.numeric_capability(
            replace(q, attributes=tuple(reversed(q.attributes)))
        )
        is True
    )


@pytest.mark.parametrize("mask", [17, 25])
def test_actual_graph_metadata_masks_remain_supported(mask):
    assert capabilities.numeric_capability(
        query(replace(KV, metadata_dtype_mask=mask), KV_CONTEXT)
    )


@pytest.mark.parametrize(
    "signature,context",
    [
        (replace(SEED, rows=8), SEED_CONTEXT),
        (replace(SEED, front_tokens=1), SEED_CONTEXT),
        (replace(SEED, columns=2050), SEED_CONTEXT),
        (replace(SEED, seed_columns=2054), SEED_CONTEXT),
        (replace(SEED, trash_row=0, seed_rows=1), SEED_CONTEXT),
        (replace(KV, rows=2), KV_CONTEXT),
        (replace(KV, kv_heads=2), KV_CONTEXT),
        (replace(KV, scalar_type=D.FP8_E4M3), KV_CONTEXT),
        (replace(KV, new_k_row_stride=512), KV_CONTEXT),
        (replace(KV, new_v_row_stride=512), KV_CONTEXT),
        (replace(KV, metadata_dtype_mask=0), KV_CONTEXT),
        (replace(KV, index_columns=2054), KV_CONTEXT),
        (replace(KV, packed_columns=2048), KV_CONTEXT),
    ],
)
def test_changed_geometry_remains_stock(signature, context):
    assert capabilities.numeric_capability(query(signature, context)) is False


def test_unknown_operation_wrong_type_and_slot_off(monkeypatch):
    assert capabilities.numeric_capability(None) is False
    assert (
        capabilities.numeric_capability(NumericCapabilityQuery(O.PACKED_ATTENTION, ()))
        is False
    )
    monkeypatch.setattr(slots, "SLOTS", frozenset())
    assert capabilities.numeric_capability(query(SEED, SEED_CONTEXT)) is False
    assert capabilities.numeric_capability(query(KV, KV_CONTEXT)) is False


@pytest.mark.parametrize(
    "op,rows,variant,tp,hidden,wide,norm,accepted",
    [
        (O.GDN_CORE, 4, 0, 2, 1024, 0, 0, True),
        (O.GDN_CORE, 4, 1, 1, 1024, 0, 0, True),
        (O.GDN_CORE, 16, 0, 4, 1024, 1, 0, True),
        (O.GDN_CORE, 16, 1, 4, 1024, 1, 0, True),
        (O.GDN_CORE, 16, 0, 1, 2560, 1, 0, False),
        (O.GDN_CORE, 16, 0, 4, 2560, 0, 0, False),
        (O.GDN_CORE, 32, 1, 4, 1024, 0, 1, True),
        (O.GDN_CORE, 32, 1, 1, 2560, 1, 1, False),
        (O.GDN_CORE, 32, 1, 4, 2560, 1, 0, False),
        (O.GDN_FRONT, 4, 0, 4, 1024, 0, 0, False),
        (O.GDN_FRONT, 4, 0, 4, 2560, 0, 0, True),
        (O.GDN_FRONT, 32, 0, 4, 2560, 1, 1, False),
        (O.GDN_VERIFY_GLUE, 16, 0, 1, 1024, 0, 0, True),
        (O.GDN_VERIFY_GLUE, 16, 0, 4, 2560, 1, 1, False),
        (O.GDN_VERIFY_GLUE, 32, 0, 1, 2560, 1, 1, False),
    ],
)
def test_gdn_projection_and_recurrence_source_envelopes(
    op, rows, variant, tp, hidden, wide, norm, accepted
):
    s = GdnSignature(
        op,
        rows,
        4,
        12,
        128,
        128,
        4,
        D.BF16,
        variant,
        3 if op == O.GDN_FRONT else D.BF16,
    )
    context = GDN_CONTEXT | dict(
        tp_size=tp, hidden_size=hidden, wide_enabled=wide, m32_norm_enabled=norm
    )
    if op == O.GDN_CORE:
        context |= {
            "normalization_program": 0 if variant == 0 else 2 if rows == 32 else 1
        }
    assert capabilities.numeric_capability(query(s, context)) is accepted


@pytest.mark.parametrize(
    "rows,variant,sf,wide,m8,batch34,target,batch,tokens,accepted",
    [
        (1, 0, 0, 0, 1, 0, 0, 0, 1, True),
        (1, 1, 1, 0, 1, 0, 0, 0, 1, True),
        (1, 0, 1, 0, 1, 0, 0, 0, 1, False),
        (2, 0, 1, 0, 1, 0, 0, 0, 1, True),
        (7, 0, 1, 0, 1, 0, 0, 0, 1, True),
        (8, 0, 0, 1, 1, 0, 1, 2, 4, True),
        (8, 0, 0, 0, 1, 0, 1, 2, 4, False),
        (8, 0, 0, 1, 0, 0, 1, 2, 4, False),
        (12, 0, 0, 1, 1, 0, 1, 3, 4, True),
        (12, 2, 0, 1, 1, 1, 1, 3, 4, True),
        (12, 2, 0, 1, 1, 1, 0, 3, 4, False),
        (12, 2, 0, 1, 1, 1, 1, 3, 1, False),
        (16, 2, 0, 1, 1, 1, 1, 4, 4, True),
        (16, 2, 0, 1, 1, 1, 1, 3, 4, False),
        (32, 2, 0, 1, 1, 1, 1, 8, 4, False),
        (64, 2, 0, 1, 1, 1, 1, 16, 4, False),
    ],
)
def test_local_expert_original_variant_selection(
    rows, variant, sf, wide, m8, batch34, target, batch, tokens, accepted
):
    s = ExpertSignature(
        O.LOCAL_EXPERTS, rows, 2560, 192, 512, 10, D.PACKED_FP4, variant
    )
    context = dict(
        tokens_per_request=tokens,
        sf_pack_enabled=sf,
        mkm_wide_enabled=wide,
        mkm_wide_m8_enabled=m8,
        wide_batch34_enabled=batch34,
        is_target_verify=target,
        batch_size=batch,
    )
    assert capabilities.numeric_capability(query(s, context)) is accepted


def test_shared_finalize_is_independent_of_local_expert_widths():
    s = ExpertSignature(O.FINALIZE_SHARED, 32, 2560, 192, 512, 10, D.BF16)
    assert capabilities.numeric_capability(query(s, {})) is True


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("rows,accepted", [(32, True), (33, False), (64, False)])
def test_hc_mix_declaration_bound_is_narrower_than_raw_support(variant, rows, accepted):
    s = HcSignature(O.HC_MIX, rows, 4, 2560, D.BF16, lowrank=128, variant=variant)
    assert providers.supported(s)
    assert capabilities.numeric_capability(query(s, {})) is accepted


@pytest.mark.parametrize(
    "op,variant,tp,accepted",
    [
        (O.SPARSE_SCORE, 0, 1, True),
        (O.SPARSE_SCORE, 0, 4, True),
        (O.SPARSE_SCORE, 3, 1, True),
        (O.SPARSE_SCORE, 4, 1, True),
        (O.SPARSE_SCORE, 4, 4, False),
        (O.SPARSE_SCORE, 5, 1, True),
        (O.SPARSE_SELECT, 0, 4, True),
        (O.SPARSE_SELECT, 1, 1, True),
        (O.SPARSE_SELECT, 1, 4, False),
    ],
)
def test_native_score_and_select_keep_separate_admission(op, variant, tp, accepted):
    s = NumericSignature(
        op,
        4,
        8,
        1,
        128,
        512,
        4,
        D.FP32 if op == O.SPARSE_SELECT else D.BF16,
        context_bound=65536,
        variant=variant,
    )
    context = dict(tp_size=tp, sm100_supported=1, page_size=16, unpadded_query_heads=4)
    assert capabilities.numeric_capability(query(s, context)) is accepted


@pytest.mark.parametrize(
    "rows,variant,ctas,tp,actual,accepted",
    [
        (68, 7, 128, 1, 4, True),
        (576, 6, 512, 1, 4, True),
        (88, 7, 128, 4, 4, False),
        (88, 7, 128, 1, 8, False),
        (88, 7, 0, 1, 4, False),
        (64, 7, 128, 1, 4, False),
        (580, 6, 512, 1, 4, False),
        (32, 4, 128, 1, 4, False),
    ],
)
def test_grouped4_score_admission_carries_its_cta_count(
    rows, variant, ctas, tp, actual, accepted
):
    s = NumericSignature(
        O.SPARSE_SCORE,
        rows,
        8,
        1,
        128,
        512,
        4,
        D.BF16,
        context_bound=65536,
        variant=variant,
        ctas_per_group=ctas,
    )
    context = dict(
        tp_size=tp, sm100_supported=1, page_size=16, unpadded_query_heads=actual
    )
    assert capabilities.numeric_capability(query(s, context)) is accepted


def test_sparse_unpadded_fact_is_mandatory():
    s = NumericSignature(
        O.SPARSE_SCORE, 4, 8, 1, 128, 512, 4, D.BF16, context_bound=65536
    )
    old_context = dict(tp_size=4, sm100_supported=1, page_size=16)
    assert capabilities.numeric_capability(query(s, old_context)) is False


@pytest.mark.parametrize(
    "op,variant,padded,actual,accepted",
    [
        (O.SPARSE_SCORE, 0, 8, 4, True),
        (O.SPARSE_SCORE, 0, 8, 8, False),
        (O.SPARSE_SCORE, 3, 8, 4, True),
        (O.SPARSE_SCORE, 3, 8, 8, False),
        (O.SPARSE_SELECT, 0, 8, 4, True),
        (O.SPARSE_SELECT, 0, 8, 8, False),
        (O.SPARSE_SCORE, 4, 8, 8, True),
        (O.SPARSE_SCORE, 5, 16, 9, True),
        (O.SPARSE_SELECT, 1, 8, 1, True),
        (O.SPARSE_SELECT, 1, 8, 8, True),
        (O.SPARSE_SELECT, 1, 8, 9, False),
        (O.SPARSE_SELECT, 1, 16, 9, True),
        (O.SPARSE_SELECT, 1, 16, 16, True),
        (O.SPARSE_SELECT, 1, 16, 17, False),
        (O.SPARSE_SELECT, 1, 8, 0, False),
    ],
)
def test_sparse_padding_cannot_alias_the_original_head_domain(
    op, variant, padded, actual, accepted
):
    s = NumericSignature(
        op,
        4,
        padded,
        1,
        128,
        512,
        4,
        D.FP32 if op == O.SPARSE_SELECT else D.BF16,
        context_bound=65536,
        variant=variant,
    )
    context = dict(
        tp_size=1, sm100_supported=1, page_size=16, unpadded_query_heads=actual
    )
    assert capabilities.numeric_capability(query(s, context)) is accepted


@pytest.mark.parametrize(
    "rows,variant,tp,ep,wide,k4,accepted",
    [
        (1, 0, 4, 1, 0, 0, True),
        (1, 1, 4, 1, 0, 1, True),
        (4, 0, 4, 1, 0, 1, False),
        (8, 2, 4, 1, 1, 0, True),
        (16, 2, 4, 1, 1, 1, True),
        (8, 2, 4, 1, 0, 0, False),
        (1, 0, 1, 1, 0, 0, False),
        (1, 0, 4, 2, 0, 0, False),
    ],
)
def test_router_source_variants(rows, variant, tp, ep, wide, k4, accepted):
    s = RouterQuantSignature(O.ROUTER_QUANT, rows, 2560, 512, D.BF16, variant)
    assert (
        capabilities.numeric_capability(
            query(s, dict(tp_size=tp, ep_size=ep, wide_enabled=wide, k4_enabled=k4))
        )
        is accepted
    )


@pytest.mark.parametrize(
    "key,value", [("tp_size", 4), ("ep_size", 2), ("expert_count", 256)]
)
def test_permuted_finalize_retains_original_transitive_source_domain(key, value):
    s = FinalizePermutedSignature(
        O.FINALIZE_PERMUTED_SHARED, 4, 2560, 10, D.BF16, D.FP32
    )
    context = dict(tp_size=1, ep_size=1, expert_count=512)
    assert capabilities.numeric_capability(query(s, context | {key: value})) is False


@pytest.mark.parametrize(
    "op,rows,variant,projection,m32,accepted",
    [
        (O.GDN_CORE, 4, 0, 0, 0, True),
        (O.GDN_VERIFY_GLUE, 8, 0, 0, 0, True),
        (O.GDN_CORE, 4, 1, 0, 1, False),
        (O.GDN_CORE, 8, 1, 0, 1, False),
        (O.GDN_CORE, 12, 1, 0, 1, False),
        (O.GDN_CORE, 16, 1, 0, 1, False),
        (O.GDN_FRONT, 4, 0, 0, 1, False),
        (O.GDN_CORE, 4, 1, 1, 0, True),
        (O.GDN_CORE, 32, 1, 0, 1, True),
        (O.GDN_CORE, 32, 1, 1, 0, False),
    ],
)
def test_gdn_projection_and_m32_protocols_are_independent(
    op, rows, variant, projection, m32, accepted
):
    s = GdnSignature(
        op,
        rows,
        4,
        12,
        128,
        128,
        4,
        D.BF16,
        variant,
        3 if op == O.GDN_FRONT else D.BF16,
    )
    context = GDN_CONTEXT | dict(
        tp_size=1 if op == O.GDN_VERIFY_GLUE else 4,
        wide_enabled=1,
        m32_norm_enabled=1,
        projection_norm_available=projection,
        m32_norm_available=m32,
    )
    if op == O.GDN_CORE:
        context |= {
            "normalization_program": 0 if variant == 0 else 2 if rows == 32 else 1
        }
    assert capabilities.numeric_capability(query(s, context)) is accepted


@pytest.mark.parametrize(
    "rows,key_heads,value_heads", [(4, 4, 12), (8, 4, 12), (4, 16, 48)]
)
@pytest.mark.parametrize(
    "specified,requested", [(0, 0), (1, 0), (1, -1), (1, 6), (1, 12)]
)
def test_gdn_front_stages_presence_and_original_override_geometry(
    rows, key_heads, value_heads, specified, requested
):
    sensitive = rows == 4 and (key_heads, value_heads) == (4, 12)
    stages = requested if sensitive and specified else 6
    context = GDN_CONTEXT | dict(
        front_stages_specified=specified, requested_front_stages=requested
    )
    for variant in (0, 1, 2, 3):
        s = GdnSignature(
            O.GDN_FRONT, rows, key_heads, value_heads, 128, 128, 4, D.BF16, variant, 3
        )
        expected = stages in (6, 12) and variant == (0 if stages == 6 else 1)
        assert capabilities.numeric_capability(query(s, context)) is expected
    core = GdnSignature(
        O.GDN_CORE, rows, key_heads, value_heads, 128, 128, 4, D.BF16, 1, D.BF16
    )
    assert capabilities.numeric_capability(
        query(core, context | {"normalization_program": 1})
    ) is (stages in (6, 12))


@pytest.mark.parametrize(
    "op,rows,variant",
    [(O.GDN_CORE, 4, 0), (O.GDN_CORE, 32, 1), (O.GDN_VERIFY_GLUE, 8, 0)],
)
def test_gdn_unrelated_paths_do_not_inherit_projection_launch_override(
    op, rows, variant
):
    s = GdnSignature(
        op, rows, 4, 12, 128, 128, 4, D.BF16, variant, gate_scalar_type=D.BF16
    )
    context = GDN_CONTEXT | dict(
        tp_size=1 if op == O.GDN_VERIFY_GLUE else 4,
        m32_norm_enabled=1,
        projection_norm_available=0,
        front_stages_specified=1,
        requested_front_stages=-1,
    )
    if op == O.GDN_CORE:
        context |= {"normalization_program": 2 if rows == 32 else 0}
    assert capabilities.numeric_capability(query(s, context)) is True


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16, 32, 64])
@pytest.mark.parametrize("program", [0, 1, 2, -1, 3])
@pytest.mark.parametrize("variant", [0, 1])
def test_core_preparation_program_preserves_independent_normalized_return_route(
    rows, program, variant
):
    signature = GdnSignature(
        O.GDN_CORE, rows, 4, 12, 128, 128, 4, D.BF16, variant, D.BF16
    )
    context = GDN_CONTEXT | dict(
        wide_enabled=1, m32_norm_enabled=1, normalization_program=program
    )
    expected = (
        (variant == 0 and program == 0 and rows in (4, 8, 12, 16))
        or (variant == 1 and program == 1 and rows in (4, 8, 12, 16))
        or (variant == 1 and program == 2 and rows == 32)
    )
    assert capabilities.numeric_capability(query(signature, context)) is expected


@pytest.mark.parametrize(
    "projection,m32,wide,norm,tp,heads,accepted",
    [
        (0, 1, 0, 1, 4, (4, 12), True),
        (1, 1, 0, 1, 4, (4, 12), True),
        (1, 0, 1, 1, 4, (4, 12), False),
        (0, 1, 1, 0, 4, (4, 12), False),
        (0, 1, 1, 1, 1, (4, 12), False),
        (0, 1, 1, 1, 4, (16, 48), False),
    ],
)
def test_standalone_normalized_return_requires_original_source_and_no_projection(
    projection, m32, wide, norm, tp, heads, accepted
):
    signature = GdnSignature(O.GDN_CORE, 32, *heads, 128, 128, 4, D.BF16, 1, D.BF16)
    context = GDN_CONTEXT | dict(
        projection_norm_available=projection,
        m32_norm_available=m32,
        wide_enabled=wide,
        m32_norm_enabled=norm,
        tp_size=tp,
        normalization_program=2,
        front_stages_specified=1,
        requested_front_stages=-1,
    )
    assert capabilities.numeric_capability(query(signature, context)) is accepted
