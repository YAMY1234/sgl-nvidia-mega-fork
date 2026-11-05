# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks: real DTOs, recorded raw launches, no GPU qualification."""

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    DirectPoolAttentionIO,
    DirectPoolSelection,
    ExpertSignature,
    FinalizeSharedIO,
    GdnSignature,
    HcSignature,
    KvTensors,
    NumericOffer,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
    NumericSignature,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D

from sgl_nvidia_mega.qwen3_8 import providers


def hc(op, **kw):
    fields = dict(
        operation=op,
        rows=4,
        stream_count=4,
        hidden_size=2560,
        scalar_type=D.BF16,
        parts=2,
    )
    fields.update(kw)
    return HcSignature(**fields)


@pytest.mark.parametrize(
    "s",
    [
        hc(O.HC_GATE),
        hc(O.HC_APPLY_NORM, rows=32),
        hc(O.HC_APPLY_NORM_GATE),
        ExpertSignature(O.FINALIZE_SHARED, 4, 2560, 192, 512, 10, D.BF16),
        GdnSignature(
            O.GDN_CORE, 4, 4, 12, 128, 128, 4, D.BF16, gate_scalar_type=D.BF16
        ),
        NumericSignature(O.DIRECT_POOL_ATTENTION, 4, 6, 1, 256, 512, 4, D.BF16),
    ],
)
def test_exact_supported_shapes(s):
    assert providers.supported(s)
    assert not providers.supported(replace(s, rows=0))
    assert not providers.supported(replace(s, scalar_type=999))


@pytest.mark.parametrize(
    "s",
    [
        hc(O.HC_GATE, parts=8),
        GdnSignature(
            O.GDN_CORE, 4, 4, 12, 128, 128, 4, D.BF16, gate_scalar_type=D.FP32
        ),
        hc(O.HC_APPLY_NORM, rows=33),
        hc(13, variant=4),
        hc(O.HC_MIX),
        GdnSignature(O.GDN_CORE, 1, 4, 12, 128, 128, 4, D.BF16),
        NumericSignature(O.DIRECT_POOL_ATTENTION, 4, 6, 1, 128, 512, 4, D.BF16),
    ],
)
def test_unsupported_declares_no_offer(s):
    assert providers.make_offer(s) is None


def test_offer_is_canonical_and_registration_idempotent(monkeypatch):
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    s = ExpertSignature(O.FINALIZE_SHARED, 4, 2560, 192, 512, 10, D.BF16)
    first = providers.register_signatures((s,))
    assert first == providers.register_signatures((s,))
    assert isinstance(first[0], NumericOffer)
    assert registry.get(s) is first[0]
    with pytest.raises(ValueError, match="conflicting"):
        registry.register(NumericOffer(s, lambda io: None))
    assert registry.get(replace(s, rows=8)) is None


def install(monkeypatch, module_name, **values):
    module = ModuleType(module_name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_hc_uses_caller_buffers_and_combine_parts(monkeypatch):
    calls = []
    module = SimpleNamespace(apply_norm=lambda *a: calls.append(a))
    name = "sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode"
    install(
        monkeypatch, name, _module_for_rows=lambda p, r: calls.append((p, r)) or module
    )
    from sgl_nvidia_mega.qwen3_8._providers.hc import build

    run = build(hc(O.HC_APPLY_NORM, parts=4))
    io = SimpleNamespace(
        block=object(),
        residual=object(),
        gate_partials=object(),
        norm_weight=object(),
        combined_out=object(),
        normalized_out=object(),
        epsilon=1e-6,
    )
    assert run(io) is None
    assert calls == [
        (4, 4),
        (
            io.block,
            io.residual,
            io.gate_partials,
            io.norm_weight,
            io.combined_out,
            io.normalized_out,
            1e-6,
        ),
    ]


def test_finalize_preserves_raw_argument_identity(monkeypatch):
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return args[-1]

    install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_finalize_shared",
        finalize_shared=launch,
    )
    from sgl_nvidia_mega.qwen3_8._providers.local import finalize_shared

    tensors = [object() for _ in range(6)]
    io = FinalizeSharedIO(*tensors[:5], 256, True, tensors[5])
    assert finalize_shared(io) is None
    assert calls == [(tuple(tensors), {"cols": 256, "use_pdl": True})]


def test_no_implicit_registration_or_framework_objects():
    import ast
    from pathlib import Path

    root = Path(providers.__file__).parent
    forbidden = {
        "get_parallel",
        "get_forward_context",
        "get_token_to_kv_pool",
        "HookRegistry",
        "ForwardBatch",
        "ModelRunner",
    }
    for path in [root / "providers.py", *sorted((root / "_providers").glob("*.py"))]:
        tree = ast.parse(path.read_text())
        assert not {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & forbidden
        # Allocation belongs to the caller, not a numeric adapter.
        assert not {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} & {
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "new_empty",
            "clone",
        }


class Tensor:
    def __init__(self, shape, dtype="torch.bfloat16"):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def stride(self, dim):
        stride = 1
        for x in self.shape[dim + 1 :]:
            stride *= x
        return stride

    def numel(self):
        n = 1
        for x in self.shape:
            n *= x
        return n

    def is_contiguous(self):
        return True

    def view(self, *shape):
        return Tensor(shape, self.dtype)

    def __getitem__(self, item):
        return Tensor((item.stop - (item.start or 0),), self.dtype)


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kw):
            self.calls.append((grid, args, kw))

        return run


def test_direct_attention_sequence_length_and_scratch_abi(monkeypatch):
    from sgl_nvidia_mega.qwen3_8._providers.qsa import direct_pool

    store, partial, merge = Launch(), Launch(), Launch()
    install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_attn_core_tp4",
        _PDL_SMALL=True,
        _store_tp4=store,
        _partial_tp4=partial,
        _merge_gate_tp4=merge,
    )
    tensor = lambda *shape: Tensor(shape)
    selection = DirectPoolSelection(
        full_slot_map=tensor(32, 65536),
        row_requests=tensor(4),
        selected_indices=tensor(4, 512),
        sequence_lengths=tensor(4),
        query_positions=tensor(4),
    )
    io = DirectPoolAttentionIO(
        q=tensor(4, 6, 256),
        gate=tensor(4, 6, 256),
        kv=KvTensors(
            k=tensor(65536, 1, 256),
            v=tensor(65536, 1, 256),
            k_scale=1.0,
            v_scale=1.0,
            page_size=1,
        ),
        new_k=tensor(4, 1, 256),
        new_v=tensor(4, 1, 256),
        out_cache_loc=tensor(4),
        selection=selection,
        scale=0.0625,
        query_width=4,
        store_new_kv=True,
        output=tensor(4, 6, 256),
        scratch_out=Tensor((4 * 33 * 16 * (256 + 2),), "torch.float32"),
    )
    assert direct_pool(io) is None
    assert store.calls[0][0] == (4, 1)
    assert partial.calls[0][1][9] is selection.sequence_lengths
    assert not hasattr(selection, "compressed_lengths")
    assert not hasattr(io, "compressed_kv")
    assert partial.calls[0][2]["WIDTH"] == 4
    assert merge.calls[0][1][3] is io.output
    assert merge.calls[0][1][7:9] == (33, 64)


def test_capture_cannot_prepare_expert_descriptors(monkeypatch):
    from sgl_nvidia_mega.qwen3_8._providers.local import LocalExperts

    install(
        monkeypatch,
        "torch",
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: True),
    )
    # prepare checks capture before touching a device or descriptors.
    install(monkeypatch, "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mk_q35")
    install(monkeypatch, "sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_mkm_wide")
    s = ExpertSignature(O.LOCAL_EXPERTS, 4, 2560, 192, 512, 10, D.PACKED_FP4)
    with pytest.raises(RuntimeError, match="before capture"):
        LocalExperts(s).prepare(SimpleNamespace())


def test_numeric_spec_build_only_consumes_finite_signatures(monkeypatch):
    from sgl_nvidia_mega.qwen3_8.spec import NUMERIC_MAX_ROWS, SPEC

    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)

    class NoModelAccess:
        def __getattribute__(self, name):
            raise AssertionError("numeric build read model/config: " + name)

    accepted = ExpertSignature(O.FINALIZE_SHARED, 4, 2560, 192, 512, 10, D.BF16)
    too_large = replace(accepted, rows=NUMERIC_MAX_ROWS + 1)
    ctx = SimpleNamespace(numeric_signatures=(accepted, too_large))
    built = SPEC.build(NoModelAccess(), NoModelAccess(), ctx)
    assert SPEC.max_rows == 0 and SPEC.numeric_max_rows == NUMERIC_MAX_ROWS
    assert built.widths == () and built.plan(None, None) is None
    assert built.speculative_runtime_hooks is None
    assert registry.get(accepted) is not None and registry.get(too_large) is None
    assert len(built.offers) == 1
    assert SPEC.build(NoModelAccess(), NoModelAccess(), ctx).offers == built.offers


def test_selected_base_operation_offers_use_real_registry_with_mock_gpu_build(
    monkeypatch,
):
    # This tests startup wiring only; GPU compiler/launch qualification is separate.
    from sgl_nvidia_mega.qwen3_8._providers import hc as hc_module

    monkeypatch.setattr(hc_module, "build", lambda s: lambda io: None)
    monkeypatch.setattr(hc_module, "build_mix", lambda s: lambda io: None)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    signatures = (
        hc(O.HC_GATE),
        hc(O.HC_APPLY_NORM),
        hc(O.HC_APPLY_NORM_GATE),
        hc(O.HC_MIX, lowrank=128, variant=1),
        ExpertSignature(O.FINALIZE_SHARED, 4, 2560, 192, 512, 10, D.BF16),
        ExpertSignature(O.LOCAL_EXPERTS, 4, 2560, 192, 512, 10, D.PACKED_FP4),
        GdnSignature(
            O.GDN_CORE, 4, 4, 12, 128, 128, 4, D.BF16, gate_scalar_type=D.BF16
        ),
        NumericSignature(O.DIRECT_POOL_ATTENTION, 4, 6, 1, 256, 512, 4, D.BF16),
    )
    providers.make_offer.cache_clear()
    try:
        first = providers.register_signatures(signatures)
        assert len(first) == 8 and first == providers.register_signatures(signatures)
        assert all(registry.get(s) is offer for s, offer in zip(signatures, first))
    finally:
        providers.make_offer.cache_clear()


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("false", False),
    ],
)
def test_hc_cluster_env_does_not_require_model_specific_framework_field(
    monkeypatch, value, expected
):
    import ast
    import os
    from pathlib import Path

    from sglang.srt.environ import EnvBool

    if value is None:
        monkeypatch.delenv("SGLANG_Q38_HC_CLUSTER_M4ONLY", raising=False)
    else:
        monkeypatch.setenv("SGLANG_Q38_HC_CLUSTER_M4ONLY", value)
    source = (
        Path(providers.__file__).parent / "kernels/ops/elementwise/hc_combine_decode.py"
    )
    tree = ast.parse(source.read_text())
    names = {"_cluster_m4only", "_HC_CLUSTER_M4ONLY"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            (isinstance(t, ast.Name) and t.id in names)
            or (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id in names
            )
            for t in node.targets
        ):
            selected.append(node)
    scope = {"EnvBool": EnvBool, "os": os}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope
    )
    assert scope["_HC_CLUSTER_M4ONLY"] is expected


@pytest.mark.parametrize("variant", (0, 1, 2, 3, 6, 7))
def test_unpublished_hc_id_has_no_slot_offer_or_compile(monkeypatch, variant):
    from sgl_nvidia_mega.qwen3_8 import slots
    from sgl_nvidia_mega.qwen3_8._providers import hc as hc_module

    # ID13 remains unassigned here; do not depend on an optional framework enum.
    signature = hc(13, rows=1024, variant=variant)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode",
        _jit_module=lambda: pytest.fail("unpublished HC operation compiled"),
        _module_for_rows=lambda *_: pytest.fail("unpublished HC operation compiled"),
        _jit_norm_gate_module=lambda: pytest.fail("unpublished HC operation compiled"),
    )
    providers.make_offer.cache_clear()
    try:
        assert 13 not in slots.OPERATION_SLOTS and not slots.enabled(13)
        assert not providers.supported(signature)
        assert providers.make_offer(signature) is None
        assert providers.register_signatures((signature,)) == ()
        assert registry.get(signature) is None
        with pytest.raises(ValueError, match="unsupported HC operation"):
            hc_module.build(signature)
    finally:
        providers.make_offer.cache_clear()


@pytest.mark.parametrize("operation,parts", ((15, 1), (16, 4)))
@pytest.mark.parametrize("rows", (2, 4, 16))
def test_removed_shared_ids_have_no_slot_offer_or_compile(
    monkeypatch, operation, parts, rows
):
    from sgl_nvidia_mega.qwen3_8 import slots
    from sgl_nvidia_mega.qwen3_8._providers import hc as hc_module

    signature = hc(operation, rows=rows, parts=parts)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    install(
        monkeypatch,
        "sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode",
        _jit_module=lambda: pytest.fail("removed shared operation compiled"),
        _module_for_rows=lambda *_: pytest.fail("removed shared operation compiled"),
        _jit_norm_gate_module=lambda: pytest.fail("removed shared operation compiled"),
    )
    providers.make_offer.cache_clear()
    try:
        assert operation not in slots.OPERATION_SLOTS and not slots.enabled(operation)
        assert not providers.supported(signature)
        assert providers.make_offer(signature) is None
        assert providers.register_signatures((signature,)) == ()
        assert registry.get(signature) is None
        with pytest.raises(ValueError, match="unsupported HC operation"):
            hc_module.build(signature)
    finally:
        providers.make_offer.cache_clear()


@pytest.mark.parametrize("operation", (1, 20, 21))
def test_removed_qk_ple_ids_never_offer_or_register(monkeypatch, operation):
    from sgl_nvidia_mega.qwen3_8 import slots

    # Numeric IDs remain reserved; do not require deleted framework DTOs/enums.
    signature = NumericSignature(operation, 4, 8, 1, 128, 512, 4, D.FP32)
    registry = NumericRegistry()
    monkeypatch.setattr(providers, "numeric_registry", registry)
    providers.make_offer.cache_clear()
    try:
        assert operation not in slots.OPERATION_SLOTS and not slots.enabled(operation)
        assert not providers.supported(signature)
        assert providers.make_offer(signature) is None
        assert providers.register_signatures((signature,)) == ()
        assert registry.get(signature) is None
    finally:
        providers.make_offer.cache_clear()
