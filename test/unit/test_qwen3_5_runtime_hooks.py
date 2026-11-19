# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for Qwen3.5's post-verify local-key provider."""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace

import msgspec
import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec

from sgl_nvidia_mega.qwen3_5.runtime import hooks


class _Tensor:
    """CUDA tensor metadata for CPU-only provider policy tests."""

    def __init__(
        self,
        shape,
        dtype,
        *,
        device="cuda:0",
        is_cuda=True,
        stride=None,
        contiguous=True,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = is_cuda
        self._stride = stride or (self.shape[-1], 1)
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def stride(self, dimension):
        return self._stride[dimension]

    def is_contiguous(self):
        return self._contiguous


class _Weight:
    def __init__(self, columns=4, *, is_cuda=True, dimensions=2):
        self.shape = (columns, 8) if dimensions == 2 else (columns,)
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0")
        self._dimensions = dimensions

    def dim(self):
        return self._dimensions


def _descriptor(**changes):
    values = {
        "columns": 4,
        "start": 4,
        "valid": 4,
        "vocab_size": 8,
        "splits": 2,
        "device_index": 0,
    }
    values.update(changes)
    return hooks._Descriptor(**values)


def _model(
    *,
    columns=4,
    start=4,
    valid=4,
    vocab_size=8,
    added=0,
    is_cuda=True,
    dimensions=2,
):
    return SimpleNamespace(
        lm_head=SimpleNamespace(
            weight=_Weight(columns, is_cuda=is_cuda, dimensions=dimensions),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=start,
                num_org_elements=valid,
                num_added_elements_padded=added,
            ),
        ),
        logits_processor=SimpleNamespace(vocab_size=vocab_size),
    )


def _defaults(default_local=None):
    if default_local is None:
        default_local = lambda view: None

    def run(io, local_keys):
        return io, local_keys

    return spec.SpeculativeRuntimeHooks(
        post_verify=spec.PostVerifyStrategy(run=run, local_keys=default_local)
    )


def _view(
    values,
    temperatures,
    *,
    seeds=None,
    start=4,
    valid=4,
    vocab_size=8,
    splits=2,
):
    return spec.ShardedLogitsView(
        values=values,
        shard=spec.VocabShardView(
            start=start,
            valid=valid,
            vocab_size=vocab_size,
        ),
        temperatures=temperatures,
        seeds=seeds,
        splits=splits,
    )


def _local(default_local=None):
    defaults = _defaults(default_local)
    factory = lambda stock: spec.SpeculativeRuntimeHooks(
        post_verify=spec.PostVerifyStrategy(
            run=stock.post_verify.run,
            local_keys=hooks._LocalKeys(_descriptor(), stock.post_verify.local_keys),
        )
    )
    return factory(defaults).post_verify.local_keys, defaults


def test_build_collects_descriptors_then_prebuilds_exact_local_shard(monkeypatch):
    model = _model()
    group = object()
    binding = SimpleNamespace(tp_size=2, tp_rank=1, tp_cpu_group=group)
    prebuilt = []

    def gather(output, local, *, group):
        assert group is binding.tp_cpu_group
        output[:] = [(_descriptor(start=0), ""), local]

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: prebuilt.append((args, kwargs)),
    )

    factory = hooks.build(model, binding)

    assert callable(factory)
    assert prebuilt == [
        (
            (0,),
            {
                "columns": 4,
                "vocab_start": 4,
                "valid": 4,
                "vocab_size": 8,
                "splits": 2,
                "fast": False,
                "pre": False,
            },
        )
    ]
    defaults = _defaults()
    selected = factory(defaults)
    assert selected.post_verify.run is defaults.post_verify.run
    assert selected.post_verify.local_keys is not defaults.post_verify.local_keys


def test_padded_columns_keep_splits_uniform_across_an_uneven_valid_tail(monkeypatch):
    columns, tail, vocab_size = 4104, 4041, 8145
    model = _model(
        columns=columns,
        start=columns,
        valid=tail,
        vocab_size=vocab_size,
    )
    binding = SimpleNamespace(tp_size=2, tp_rank=1, tp_cpu_group=object())
    prebuilt = []

    def gather(output, local, *, group):
        output[:] = [
            (
                hooks._Descriptor(
                    columns=columns,
                    start=0,
                    valid=columns,
                    vocab_size=vocab_size,
                    splits=4,
                    device_index=0,
                ),
                "",
            ),
            local,
        ]

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: prebuilt.append((args, kwargs)),
    )
    factory = hooks.build(model, binding)

    assert callable(factory)
    assert prebuilt == [
        (
            (0,),
            {
                "columns": columns,
                "vocab_start": columns,
                "valid": tail,
                "vocab_size": vocab_size,
                "splits": 4,
                "fast": False,
                "pre": False,
            },
        )
    ]

    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    generated = _Tensor((2,), torch.int64)
    launches = []
    monkeypatch.setattr(hooks.torch, "randint", lambda *args, **kwargs: generated)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "gumbel_shard_keys",
        lambda *args, **kwargs: launches.append((args, kwargs)) or object(),
    )
    local_keys = factory(_defaults()).post_verify.local_keys
    values = _Tensor((2, columns), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    local_keys(
        _view(
            values,
            temperatures,
            start=columns,
            valid=tail,
            vocab_size=vocab_size,
            splits=4,
        )
    )
    assert launches == [
        (
            (values, temperatures, generated),
            {
                "vocab_start": columns,
                "valid": tail,
                "vocab_size": vocab_size,
                "splits": 4,
            },
        )
    ]


@pytest.mark.parametrize(
    "gathered, match",
    [
        (
            [(_descriptor(start=0), ""), (None, "missing LM head")],
            "rank 1: missing LM head",
        ),
        (
            [(_descriptor(start=0), ""), (_descriptor(start=5), "")],
            "not contiguous",
        ),
        (
            [(_descriptor(start=0), ""), (_descriptor(start=4, splits=4), "")],
            "disagree",
        ),
    ],
)
def test_unsupported_topology_returns_none_uniformly_without_prebuild(
    monkeypatch, caplog, gathered, match
):
    binding = SimpleNamespace(tp_size=2, tp_rank=0, tp_cpu_group=object())

    def gather(output, local, *, group):
        output[:] = gathered

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: pytest.fail("unsupported provider was prebuilt"),
    )

    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert hooks.build(_model(start=0), binding) is None
    assert match in caplog.text


@pytest.mark.parametrize(
    "model, match",
    [
        (SimpleNamespace(), "metadata is unavailable"),
        (_model(is_cuda=False), "not a CUDA matrix"),
        (_model(dimensions=1), "not a CUDA matrix"),
        (_model(added=1), "added-vocabulary"),
        (_model(valid=0), "valid columns"),
        (_model(valid=5), "valid columns"),
        (_model(start=7, valid=4), "outside"),
    ],
)
def test_local_metadata_refusal_is_a_build_result_not_an_exception(
    monkeypatch, caplog, model, match
):
    binding = SimpleNamespace(tp_size=1, tp_rank=0)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: pytest.fail("unsupported provider was prebuilt"),
    )
    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert hooks.build(model, binding) is None
    assert match in caplog.text


def test_greedy_always_delegates_by_identity_even_with_seed_material():
    sentinel = object()
    calls = []

    def default(view):
        calls.append(view)
        return sentinel

    local, defaults = _local(default)
    view = _view(object(), None, seeds=object())

    assert local(view) is sentinel
    assert calls == [view]
    assert local.default is defaults.post_verify.local_keys


def test_sampled_request_seeds_decline_before_checks_or_rng(monkeypatch):
    local, _ = _local()
    monkeypatch.setattr(
        hooks.torch,
        "randint",
        lambda *args, **kwargs: pytest.fail("seeded fallback consumed RNG"),
    )
    # Neither object implements tensor metadata; an early capability read fails.
    view = _view(object(), object(), seeds=object())
    assert local(view) is None


@pytest.mark.parametrize(
    "change",
    [
        {"values": object()},
        {"values.is_cuda": False},
        {"values.device": "cuda:1"},
        {"values.shape": (2, 4, 1)},
        {"values.dtype": torch.float16},
        {"values.stride": (8, 2)},
        {"values.shape": (2, 5)},
        {"temperatures.device": "cuda:1"},
        {"temperatures.dtype": torch.bfloat16},
        {"temperatures.shape": (2, 1)},
        {"temperatures.contiguous": False},
        {"valid": 3},
        {"vocab_size": 9},
        {"splits": 4},
    ],
)
def test_every_sampled_capability_refusal_precedes_rng(monkeypatch, change):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((2, 4), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    view_kwargs = {}
    if "values" in change:
        values = change["values"]
    for name, value in change.items():
        if name.startswith("values."):
            attribute = name.split(".", 1)[1]
            if attribute == "shape":
                values.shape = value
            elif attribute == "stride":
                values._stride = value
            elif attribute == "device":
                values.device = torch.device(value)
            else:
                setattr(values, attribute, value)
        elif name.startswith("temperatures."):
            attribute = name.split(".", 1)[1]
            if attribute == "contiguous":
                temperatures._contiguous = value
            elif attribute == "device":
                temperatures.device = torch.device(value)
            else:
                setattr(temperatures, attribute, value)
        elif name != "values":
            view_kwargs[name] = value
    local, _ = _local()
    monkeypatch.setattr(
        hooks.torch,
        "randint",
        lambda *args, **kwargs: pytest.fail("unsupported view consumed RNG"),
    )

    assert local(_view(values, temperatures, **view_kwargs)) is None


def test_supported_sample_draws_exact_seed_then_launches_without_collective(
    monkeypatch, caplog
):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((3, 4), torch.float32)
    temperatures = _Tensor((3,), torch.float32)
    generated = _Tensor((3,), torch.int64)
    keys = object()
    events = []

    def randint(*args, **kwargs):
        events.append(("rng", args, kwargs))
        return generated

    def launch(*args, **kwargs):
        events.append(("launch", args, kwargs))
        return keys

    monkeypatch.setattr(hooks.torch, "randint", randint)
    monkeypatch.setattr(hooks.dist, "all_gather_object", pytest.fail)
    monkeypatch.setattr(hooks.sharded_gumbel, "gumbel_shard_keys", launch)
    local, _ = _local()
    # The runtime descriptor's rank-varying start is not a call-time applicability
    # branch.  The build-captured start remains the kernel's authoritative value.
    view = _view(values, temperatures, start=777)

    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert local(view) is keys
        assert local(view) is keys

    assert events[0] == (
        "rng",
        (0, 2**31 - 1, (3,)),
        {"dtype": torch.int64, "device": torch.device("cuda:0")},
    )
    assert events[1] == (
        "launch",
        (values, temperatures, generated),
        {
            "vocab_start": 4,
            "valid": 4,
            "vocab_size": 8,
            "splits": 2,
        },
    )
    marker = "NVIDIA megakernel: Qwen3.5 slot sharded_gumbel runs at 3 rows"
    assert caplog.text.count(marker) == 1


def test_failure_after_rng_is_hard_and_does_not_set_engagement(monkeypatch, caplog):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((2, 4), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    generated = _Tensor((2,), torch.int64)
    monkeypatch.setattr(hooks.torch, "randint", lambda *args, **kwargs: generated)
    attempts = []

    def launch(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("launch failed after RNG")
        return object()

    monkeypatch.setattr(hooks.sharded_gumbel, "gumbel_shard_keys", launch)
    local, _ = _local()
    view = _view(values, temperatures)

    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        with pytest.raises(RuntimeError, match="launch failed after RNG"):
            local(view)
        assert "slot sharded_gumbel runs" not in caplog.text
        local(view)
    assert caplog.text.count("slot sharded_gumbel runs at 2 rows") == 1


def _build(monkeypatch, slots, *, head_refusals=("", ""), shard=None):
    from sgl_nvidia_mega.qwen3_5 import runtime

    shard = shard or {"columns": 8, "valid": 8, "vocab_size": 16}
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    model = _model(start=shard["columns"], **shard)
    binding = SimpleNamespace(tp_size=2, tp_rank=1, tp_cpu_group=object())

    def gather(output, local, *, group):
        assert group is binding.tp_cpu_group
        if isinstance(local, tuple):
            output[:] = [(dataclasses.replace(local[0], start=0), ""), local]
        else:
            output[:] = list(head_refusals)

    prebuilt, heads = [], []
    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(hooks.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hooks.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(hooks.verify_head, "head_refusal", lambda processor, head: "")
    monkeypatch.setattr(
        hooks.sharded_gumbel, "build", lambda *args, **kwargs: prebuilt.append(kwargs)
    )

    def build_head(model, *, keys):
        heads.append(keys)
        return SimpleNamespace(keys=keys)

    monkeypatch.setattr(hooks.verify_head, "build", build_head)
    return hooks.build(model, binding), prebuilt, heads


@pytest.mark.parametrize(
    "slots, fast, head, keys",
    [
        ({"sharded_gumbel"}, False, False, False),
        ({"sharded_gumbel", "gumbel_fast"}, True, False, False),
        ({"sharded_gumbel", "verify_head"}, False, True, False),
        ({"sharded_gumbel", "gumbel_fast", "verify_head"}, True, True, False),
        (
            {"sharded_gumbel", "gumbel_fast", "verify_head", "verify_head_keys"},
            True,
            True,
            True,
        ),
    ],
)
def test_each_option_engages_only_where_the_slots_hold_it(
    monkeypatch, slots, fast, head, keys
):
    factory, prebuilt, heads = _build(monkeypatch, slots)
    assert [(p["fast"], p["pre"]) for p in prebuilt] == [(fast, keys)]
    assert heads == ([keys] if head else [])
    selected = factory(_defaults())
    assert selected.post_verify.local_keys.fast is fast
    assert (selected.post_verify.local_keys.head is not None) is keys
    assert (selected.verify_shard_head is not None) is head
    if head:
        assert selected.verify_shard_head.keys is keys


def test_a_head_refused_on_any_rank_stays_off_on_every_rank(monkeypatch, caplog):
    slots = {"sharded_gumbel", "gumbel_fast", "verify_head", "verify_head_keys"}
    with caplog.at_level(logging.INFO):
        factory, prebuilt, heads = _build(
            monkeypatch, slots, head_refusals=("", "a quantized LM head (Fp8)")
        )
    assert heads == []
    assert [(p["fast"], p["pre"]) for p in prebuilt] == [(True, False)]
    selected = factory(_defaults())
    assert selected.verify_shard_head is None
    assert selected.post_verify.local_keys.head is None
    assert "['verify_head'] use stock (a quantized LM head (Fp8))" in caplog.text


def test_an_sglang_without_the_head_seam_keeps_its_gemm(monkeypatch, caplog):
    class Hooks(msgspec.Struct, frozen=True):
        post_verify: spec.PostVerifyStrategy

    monkeypatch.setattr(hooks, "SpeculativeRuntimeHooks", Hooks)
    slots = {"sharded_gumbel", "gumbel_fast", "verify_head", "verify_head_keys"}
    with caplog.at_level(logging.INFO):
        factory, prebuilt, heads = _build(monkeypatch, slots)
    assert heads == []
    assert [(p["fast"], p["pre"]) for p in prebuilt] == [(True, False)]
    selected = factory(_defaults())
    assert isinstance(selected, Hooks) and selected.post_verify.local_keys.fast
    assert "['verify_head'] use stock (this SGLang's hooks" in caplog.text
    assert "['verify_head_keys'] use stock" in caplog.text


def test_fast_noise_declines_uniformly_when_a_shard_is_off_quads(monkeypatch, caplog):
    slots = {"sharded_gumbel", "gumbel_fast", "verify_head", "verify_head_keys"}
    # two 6-column shards in 2 key splits: tiles of 3 ids, not on multiples of 4
    shard = {"columns": 6, "valid": 6, "vocab_size": 12}
    with caplog.at_level(logging.INFO):
        factory, prebuilt, heads = _build(monkeypatch, slots, shard=shard)
    assert [(p["fast"], p["pre"]) for p in prebuilt] == [(False, False)]
    # the head is built without the keys, which need the fast launch
    assert heads == [False]
    selected = factory(_defaults())
    assert not selected.post_verify.local_keys.fast
    assert "['gumbel_fast'] use stock (rank 0: its key tiles" in caplog.text


def test_fast_keys_take_the_verify_heads_keys_of_their_logits(monkeypatch):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((3, 4), torch.bfloat16)
    temperatures = _Tensor((3,), torch.float32)
    generated = _Tensor((3,), torch.int64)
    pre = object()
    launches = []
    monkeypatch.setattr(hooks.torch, "randint", lambda *args, **kwargs: generated)
    monkeypatch.setattr(
        hooks.verify_head, "lookup", lambda logits: pre if logits is values else None
    )
    monkeypatch.setattr(
        hooks.sharded_gumbel, "gumbel_shard_keys", pytest.fail, raising=True
    )
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "gumbel_shard_keys_fast",
        lambda *args, **kwargs: launches.append((args, kwargs)) or "keys",
    )
    for head, want in (
        (SimpleNamespace(keys=True), pre),
        (SimpleNamespace(keys=False), None),
        (None, None),
    ):
        local = hooks._LocalKeys(_descriptor(), lambda view: None, fast=True, head=head)
        assert local(_view(values, temperatures)) == "keys"
        assert launches[-1] == (
            (values, temperatures, generated),
            {
                "vocab_start": 4,
                "valid": 4,
                "vocab_size": 8,
                "splits": 2,
                "pre": want,
            },
        )


def test_a_slot_list_without_the_options_keeps_the_stock_stream(monkeypatch):
    from sgl_nvidia_mega.qwen3_5 import runtime

    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setenv(runtime.SLOTS_ENV, "hop,sharded_gumbel")
    requested = runtime._requested_slots()
    assert not requested & set(runtime.MEASURED_HOOKS)
    factory, prebuilt, heads = _build(monkeypatch, requested)
    selected = factory(_defaults())
    assert not selected.post_verify.local_keys.fast and heads == []
    assert selected.verify_shard_head is None


# ------------------------------------------------------------- slot gathered_cdf


def _gathered_view(values, temperatures):
    return SimpleNamespace(values=values, temperatures=temperatures)


@pytest.mark.parametrize(
    "change",
    [
        {"values": object()},
        {"values.is_cuda": False},
        {"values.device": "cuda:1"},
        {"values.shape": (2, 4, 1)},
        {"values.dtype": torch.int32},
        {"values.stride": (8, 2)},
        {"values.shape": (0, 4)},
        {"values.shape": (4097, 4)},
        {"values.shape": (2, 5)},
        {"temperatures": None},
        {"temperatures.device": "cuda:1"},
        {"temperatures.dtype": torch.bfloat16},
        {"temperatures.shape": (2, 1)},
        {"temperatures.contiguous": False},
    ],
)
def test_every_gathered_refusal_precedes_rng(monkeypatch, change):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((2, 4), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    for name, value in change.items():
        if name == "values":
            values = value
        elif name == "temperatures":
            temperatures = value
        else:
            target, attribute = name.split(".", 1)
            tensor = values if target == "values" else temperatures
            if attribute == "stride":
                tensor._stride = value
            elif attribute == "contiguous":
                tensor._contiguous = value
            elif attribute == "device":
                tensor.device = torch.device(value)
            else:
                setattr(tensor, attribute, value)
    seeds = SimpleNamespace(
        next=lambda rows: pytest.fail("an unsupported view consumed RNG")
    )
    if change.get("values.shape") == (4097, 4):
        temperatures.shape = (4097,)
    sampler = hooks._GatheredCdf(0, 4, seeds)
    assert sampler(_gathered_view(values, temperatures)) is None


def test_a_supported_gathered_view_draws_seeds_then_tokens(monkeypatch, caplog):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((3, 8), torch.float32)
    temperatures = _Tensor((3,), torch.float32)
    events = []
    seeds, tokens = object(), object()

    class Seeds:
        def next(self, rows):
            events.append(("seeds", rows))
            return seeds

    monkeypatch.setattr(
        hooks.gathered_cdf,
        "cdf_tokens",
        lambda *a: events.append(("tokens", a)) or tokens,
    )
    monkeypatch.setattr(hooks.dist, "all_gather_object", pytest.fail)
    sampler = hooks._GatheredCdf(0, 8, Seeds())
    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert sampler(_gathered_view(values, temperatures)) is tokens
        assert sampler(_gathered_view(values, temperatures)) is tokens
    assert events[:2] == [("seeds", 3), ("tokens", (values, temperatures, seeds))]
    marker = "NVIDIA megakernel: Qwen3.5 slot gathered_cdf runs at 3 rows"
    assert caplog.text.count(marker) == 1


def _gathered_build(monkeypatch):
    """gathered_cdf.build, recording its calls and returning one counter."""
    calls, counter = [], object()

    def build(device, columns, **kw):
        calls.append((device, columns, kw))
        return counter

    monkeypatch.setattr(hooks.gathered_cdf, "build", build)
    monkeypatch.setattr(hooks.torch.cuda, "current_device", lambda: 0)
    return calls, counter


def test_the_gathered_sampler_alone_needs_no_descriptor_or_collective(monkeypatch):
    if not hooks._offers_gathered_tokens():
        pytest.skip("this SGLang's post-verify strategy takes no gathered sampler")
    monkeypatch.setattr(hooks.dist, "all_gather_object", pytest.fail)
    monkeypatch.setattr(hooks.sharded_gumbel, "build", pytest.fail)
    _gathered_build(monkeypatch)
    model = SimpleNamespace(logits_processor=SimpleNamespace(vocab_size=8))
    factory = hooks.build(model, object(), sharded=False, gathered=True)
    defaults = _defaults()
    selected = factory(defaults).post_verify
    assert selected.run is defaults.post_verify.run
    assert selected.local_keys is defaults.post_verify.local_keys
    assert isinstance(selected.gathered_tokens, hooks._GatheredCdf)


def test_the_gathered_kernels_and_seed_counter_are_built_at_load(monkeypatch):
    calls, counter = _gathered_build(monkeypatch)
    model = SimpleNamespace(logits_processor=SimpleNamespace(vocab_size=248))
    sampler = hooks._gathered_sampler(model)
    assert calls == [(torch.device("cuda", 0), 248, {"max_rows": 4096})]
    assert sampler.seeds is counter and sampler.columns == 248
    assert sampler.device_index == 0


def _cpu_build(monkeypatch, capturing=False):
    """gathered_cdf.build on CPU tensors, recording each compile launch."""
    from sgl_nvidia_mega.qwen3_5.sm103 import gathered_cdf

    launches = []

    def launch(logits, temperatures, seeds):
        launches.append((logits.dtype, logits.shape, logits.stride(0), seeds.tolist()))

    monkeypatch.setattr(gathered_cdf, "cdf_tokens", launch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    return gathered_cdf, launches


def test_the_gathered_build_compiles_every_dtype_with_a_fresh_counter(monkeypatch):
    gathered_cdf, launches = _cpu_build(monkeypatch)
    seeds = gathered_cdf.build(torch.device("cpu"), 20, max_rows=8)
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    assert launches == [
        (dtype, (2, 20), pitch, [0, 1]) for dtype in dtypes for pitch in (20, 32)
    ]
    assert int(seeds.round) == 0 and seeds.rows.numel() == 8
    assert gathered_cdf.build(torch.device("cpu"), 32).round is not seeds.round
    assert [launch[2] for launch in launches[6:]] == [32] * 3


def test_the_gathered_build_refuses_inside_a_capture(monkeypatch):
    gathered_cdf, launches = _cpu_build(monkeypatch, capturing=True)
    monkeypatch.setattr(gathered_cdf, "RoundSeeds", pytest.fail)
    with pytest.raises(RuntimeError, match="inside CUDA-graph capture"):
        gathered_cdf.build(torch.device("cpu"), 20)
    assert launches == []


def test_without_the_gathered_seam_the_slot_stays_stock(monkeypatch, caplog):
    monkeypatch.setattr(hooks, "_offers_gathered_tokens", lambda: False)
    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert hooks.build(object(), object(), sharded=False, gathered=True) is None
    assert "slot gathered_cdf uses stock" in caplog.text


def test_both_samplers_share_one_strategy(monkeypatch):
    if not hooks._offers_gathered_tokens():
        pytest.skip("this SGLang's post-verify strategy takes no gathered sampler")
    model = _model()
    binding = SimpleNamespace(tp_size=2, tp_rank=1, tp_cpu_group=object())

    def gather(output, local, *, group):
        output[:] = [(_descriptor(start=0), ""), local]

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(hooks.sharded_gumbel, "build", lambda *a, **kw: None)
    _gathered_build(monkeypatch)
    selected = hooks.build(model, binding, gathered=True)(_defaults()).post_verify
    assert isinstance(selected.local_keys, hooks._LocalKeys)
    assert isinstance(selected.gathered_tokens, hooks._GatheredCdf)
