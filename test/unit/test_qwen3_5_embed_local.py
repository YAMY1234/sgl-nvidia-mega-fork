# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.sm103.draft import embed_local

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
REPLICATION_ENV = "SGLANG_ENABLE_EMBED_REPLICATION"
GB300_KEYS = (("tp8", topology.GB300), ("tep8", topology.GB300))


def test_the_two_halves_are_bitwise_rows_of_the_tp8_and_tep8_plans():
    assert "draft_embed_local" in runtime.DRAFT_SLOTS
    assert topology.SLOT_TOPOLOGIES["draft_embed_local"] == {"tp4", "tp8", "tep8"}
    assert "draft_embed_local" in runtime.MEASURED_DRAFT
    assert "embed_local" in runtime.MEASURED_OPTIONS
    for name in ("embed_local", "draft_embed_local"):
        assert name in runtime.MEASURED_BITWISE
        assert name not in runtime.NUMERICS_CHANGE
        for key in GB300_KEYS:
            assert name in runtime.MEASURED_PLAN[key]
    # an option is no slot: the launcher never requests it
    assert "embed_local" not in topology.SLOT_TOPOLOGIES
    assert "embed_local" not in runtime._measured_rows()


def test_draft_embed_local_adds_no_numerics_claim():
    from sgl_nvidia_mega.qwen3_5 import draft

    base = frozenset({"draft_loop", "draft_extend"})
    assert draft.numerics(base | {"draft_embed_local"}) == draft.numerics(base)


def test_draft_embed_local_runs_inside_the_draft_loop():
    reasons = runtime.slot_reasons(frozenset({"draft_extend", "draft_embed_local"}))
    assert any(
        "'draft_embed_local' runs inside entry 'draft_loop'" in r for r in reasons
    )


@pytest.mark.parametrize(
    "switch, slots, key, want",
    [
        ("1", None, ("tep8", topology.GB300), True),
        ("1", None, ("tp8", topology.GB300), True),
        (None, None, ("tep8", topology.GB300), False),
        ("1", "draft_loop,draft_embed_local", ("tep8", topology.GB300), False),
        ("1", None, ("tp4", topology.GB300), False),
        ("1", None, ("tep8", topology.Sku(10, 3, 148)), False),
        ("1", None, None, False),
    ],
)
def test_the_verify_half_follows_the_measured_rows(
    monkeypatch, switch, slots, key, want
):
    from sgl_nvidia_mega.qwen3_5.runtime import embedding

    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "MEASURED_KEY", key)
    built = []

    def build(**kwargs):
        built.append(kwargs)
        return "table"

    monkeypatch.setattr(embed_local, "build", build)
    model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens="embed"),
        logits_processor=SimpleNamespace(vocab_size=64),
    )
    got = embedding.build_local(model, "binding")
    assert (got == "table") is want
    if want:
        (kwargs,) = built
        assert kwargs["name"] == "embed_local" and kwargs["vocab"] == 64
        assert kwargs["embed"] == "embed" and kwargs["binding"] == "binding"
    else:
        assert built == []


# ------------------------------------------------------------------- the build


class _Shard(SimpleNamespace):
    pass


def _embed(rank, world, vocab=64, hidden=8, dtype=torch.bfloat16, added=0, shard=True):
    local = vocab // world
    weight = torch.arange(local * hidden, dtype=torch.float32).reshape(local, hidden)
    embed = SimpleNamespace(weight=SimpleNamespace(data=weight.to(dtype)))
    if shard:
        embed.shard_indices = _Shard(
            org_vocab_start_index=rank * local,
            org_vocab_end_index=(rank + 1) * local,
            added_vocab_start_index=vocab,
            added_vocab_end_index=vocab + added,
        )
    return embed


def _binding(rank=1, world=4):
    return SimpleNamespace(
        tp_rank=rank,
        tp_size=world,
        tp_cpu_group=None,
        tp_device_group="device group",
    )


@pytest.mark.parametrize(
    "case, reason",
    [
        ("tp1", "TP1: no vocabulary-parallel all-reduce to replace"),
        ("no_shard", "is not vocabulary-parallel"),
        ("added", "the embedding has added vocabulary"),
        ("fp32", "needs BF16 rows"),
        ("not_rank_major", "is not a rank-major slice of 60 ids"),
    ],
)
def test_the_build_refuses_a_table_it_cannot_replicate(monkeypatch, case, reason):
    monkeypatch.delenv(REPLICATION_ENV, raising=False)
    world = 1 if case == "tp1" else 4
    embed = _embed(
        1 if world > 1 else 0,
        world,
        dtype=torch.float32 if case == "fp32" else torch.bfloat16,
        added=8 if case == "added" else 0,
        shard=case != "no_shard",
    )
    vocab = 60 if case == "not_rank_major" else 64
    seen = []

    def supported(name, why, *, cpu_group=None):
        seen.append((name, why))
        return not why

    monkeypatch.setattr(runtime, "slot_supported", supported)

    def allocate(**_):
        raise AssertionError("a refused table is never allocated")

    monkeypatch.setattr(embed_local.EmbedLocal, "allocate", staticmethod(allocate))
    got = embed_local.build(
        name="draft_embed_local",
        embed=embed,
        vocab=vocab,
        binding=_binding(1 if world > 1 else 0, world),
        what="the MTP draft",
    )
    assert got is None
    ((name, why),) = seen
    assert name == "draft_embed_local" and reason in why


def test_it_steps_aside_when_sglang_replicates_the_embedding(monkeypatch, caplog):
    monkeypatch.setenv(REPLICATION_ENV, "1")

    def supported(*_, **__):
        raise AssertionError("no agreement and no table under SGLang's replication")

    monkeypatch.setattr(runtime, "slot_supported", supported)
    with caplog.at_level("INFO"):
        got = embed_local.build(
            name="embed_local",
            embed=_embed(1, 4),
            vocab=64,
            binding=_binding(),
            what="the verify",
        )
    assert got is None
    assert "embed_local steps aside" in caplog.text
    assert REPLICATION_ENV in caplog.text


@pytest.fixture
def fake_collectives(monkeypatch):
    import torch.distributed as dist

    monkeypatch.delenv(REPLICATION_ENV, raising=False)
    monkeypatch.setattr(embed_local, "_SHARED", {})
    world, vocab, hidden = 4, 64, 8
    full = (
        torch.arange(vocab * hidden, dtype=torch.float32)
        .reshape(vocab, hidden)
        .bfloat16()
    )
    full[3, 2] = -0.0
    local = vocab // world
    made = []

    class FakeLocal:
        def __init__(self, *, table, shard, vocab_start, vocab_local, **_):
            made.append(vocab_start)
            self.table = full.clone()
            self.gib = 0.0

        @staticmethod
        def allocate(**_):
            return "buffer"

        def __call__(self, ids):
            rows = self.table[ids].clone()
            rows[rows.view(torch.int16) == -32768] = 0.0
            return rows

    monkeypatch.setattr(embed_local, "EmbedLocal", FakeLocal)

    def all_reduce(want, group=None):
        ids = torch.linspace(0, vocab - 1, embed_local.CHECK_IDS).long()
        ids[:world] = torch.arange(world) * local
        want.zero_()
        want += full[ids].float()

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(runtime, "slot_supported", lambda name, why, **_: not why)

    def embed(rank):
        e = _embed(rank, world, vocab, hidden)
        e.weight.data = full[rank * local : (rank + 1) * local].clone()
        return e

    return SimpleNamespace(embed=embed, made=made, full=full, world=world)


def test_the_draft_reuses_the_verifys_table(fake_collectives, caplog):
    embed = fake_collectives.embed(1)
    binding = _binding(1, fake_collectives.world)
    with caplog.at_level("INFO"):
        verify = embed_local.build(
            name="embed_local",
            embed=embed,
            vocab=64,
            binding=binding,
            what="the verify",
        )
        draft = embed_local.build(
            name="draft_embed_local",
            embed=embed,
            vocab=64,
            binding=binding,
            what="the MTP draft",
        )
    assert verify is not None and draft is verify
    # one table a rank
    assert fake_collectives.made == [16]
    assert "the verify gathers its embedding rows" in caplog.text
    assert "the MTP draft gathers its embedding rows" in caplog.text
    assert "a rank, shared)" in caplog.text


def test_a_lookup_that_differs_from_sglangs_is_refused(fake_collectives, monkeypatch):
    seen = []

    def supported(name, why, **_):
        seen.append(why)
        return not why

    monkeypatch.setattr(runtime, "slot_supported", supported)
    real_call = embed_local.EmbedLocal.__call__

    def wrong(self, ids):
        rows = real_call(self, ids)
        rows[0, 0] += 1
        return rows

    monkeypatch.setattr(embed_local.EmbedLocal, "__call__", wrong)
    got = embed_local.build(
        name="draft_embed_local",
        embed=fake_collectives.embed(2),
        vocab=64,
        binding=_binding(2, fake_collectives.world),
        what="the MTP draft",
    )
    assert got is None
    assert seen == ["", "1 lookup values differ from SGLang's embedding"]
    assert embed_local._SHARED == {}


# ------------------------------------------------------------------- the steps


@pytest.mark.parametrize("language_only", [True, False])
@pytest.mark.parametrize("buffer", [True, False])
def test_the_verify_takes_the_replicated_rows(monkeypatch, language_only, buffer):
    from sgl_nvidia_mega.qwen3_5.runtime import embedding

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ids = torch.tensor([3, 1, 2])
    rows = torch.full((3, 4), 5.0)
    calls = []

    def local(token_ids):
        calls.append(token_ids)
        return rows

    def stock(*_, **__):
        raise AssertionError("the replicated table serves the verify")

    buf = torch.zeros(3, 4) if buffer else None
    st = SimpleNamespace(
        lm=SimpleNamespace(language_model_only=language_only, use_deepstack=False),
        model=SimpleNamespace(embed_tokens=stock),
        input_ids=ids,
        fb=SimpleNamespace(input_embeds=buf),
        bound={"embed_local": local},
    )
    monkeypatch.setattr("sglang.srt.managers.mm_utils.prepare_mm_inputs", stock)
    out = embedding.embed(st)
    assert calls == [ids]
    if buffer and not language_only:
        assert out is buf and torch.equal(buf, rows)
    else:
        assert out is rows


def test_without_the_table_the_verify_keeps_sglangs_lookup(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.runtime import embedding

    ids = torch.tensor([3])
    st = SimpleNamespace(
        lm=SimpleNamespace(language_model_only=True),
        model=SimpleNamespace(embed_tokens=lambda token_ids: ("stock", token_ids)),
        input_ids=ids,
        bound={"embed_local": None},
    )
    assert embedding.embed(st) == ("stock", ids)


def test_the_draft_prefers_the_replicated_table():
    from sgl_nvidia_mega.qwen3_5.draft.parts import Parts

    parts = Parts.__new__(Parts)
    parts.embed_local = lambda ids: ("local", ids)
    parts.embedx = lambda ids: ("exchange", ids)
    parts.model = SimpleNamespace(embed_tokens=lambda ids: ("stock", ids))
    ids = torch.tensor([1, 2])
    assert parts.embed(ids)[0] == "local"
    parts.embed_local = None
    assert parts.embed(ids)[0] == "exchange"


@pytest.mark.parametrize("built", [True, False])
def test_a_built_draft_embed_local_leaves_draft_embed_unbuilt(monkeypatch, built):
    from sgl_nvidia_mega.qwen3_5.draft import attention
    from sgl_nvidia_mega.qwen3_5.draft.parts import Parts

    model = SimpleNamespace(layers=["decoder"], start_layer=0, end_layer=1)
    draft = SimpleNamespace(
        lm=SimpleNamespace(model=model),
        binding="binding",
        slots=frozenset({"draft_loop", "draft_embed", "draft_embed_local"}),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(attention, "Attention", lambda _: None)
    table = object()
    monkeypatch.setattr(
        Parts, "_build_embed_local", lambda self: table if built else None
    )
    monkeypatch.setattr(Parts, "_build_embedx", lambda self: "exchange")
    parts = Parts.build(draft)
    if built:
        assert parts.embed_local is table and parts.embedx is None
        assert parts.slots == {"draft_loop", "draft_embed_local"}
    else:
        assert parts.embed_local is None and parts.embedx == "exchange"
        assert parts.slots == {"draft_loop", "draft_embed"}
