# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec as sgl_spec

from sgl_nvidia_mega.qwen3_5 import runtime, spec, topology
from sgl_nvidia_mega.qwen3_5.attention import draft_cp, split_kv
from sgl_nvidia_mega.qwen3_5.draft.attention import DraftCP
from sgl_nvidia_mega.qwen3_5.runtime import hooks
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

SLOT = "draft_cp"
RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"


@pytest.fixture(autouse=True)
def fresh_logs(monkeypatch):
    monkeypatch.setattr(draft_cp, "_LOGGED", set())


def test_declared_as_a_numeric_provider_row_of_the_measured_plans():
    assert SLOT in runtime.HOOK_SLOTS and SLOT not in runtime.SHIPPED_SLOTS
    assert SLOT in runtime.MEASURED_DRAFT and SLOT in runtime.NUMERICS_CHANGE
    assert SLOT not in runtime.MEASURED_BITWISE and SLOT in spec.NUMERIC_TOLERANCES
    assert topology.SLOT_TOPOLOGIES[SLOT] == {"tp8", "tep8"}
    for key in (("tp8", topology.GB300), ("tep8", topology.GB300)):
        assert SLOT in runtime.MEASURED_PLAN[key]
    assert SLOT not in runtime.MEASURED_PLAN[("dep4", topology.GB300)]


def test_it_needs_cp_lite(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT, "skv2"}))
    assert any(f"slot '{SLOT}' runs on CP-lite" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({SLOT, "skv2", "skv2_cp"}))
    assert not [r for r in runtime.slot_reasons() if SLOT in r]


def test_unsupported_cp_lite_topology_drops_it(monkeypatch):
    import sgl_nvidia_mega.qwen3_5 as q35

    monkeypatch.setattr(q35, "cp_lite_topology", lambda attention_tp_size: None)
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"skv2", "skv2_cp", SLOT}))
    assert runtime.apply_cp_lite_topology(2) == {"skv2_cp", SLOT}


@pytest.mark.parametrize(
    "switch, slots, on",
    [("1", None, True), (None, None, False), ("1", "skv2,skv2_cp", False)],
)
def test_the_measured_plan_runs_it_at_tp8_and_tep8(monkeypatch, switch, slots, on):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for topo in ("tp8", "tep8"):
        assert runtime._measured_rows_on((topo, topology.GB300), SLOT) is on
    assert (SLOT in runtime._requested_slots()) is on


def _layer(**over):
    fields = dict(
        layer_id=3,
        tp_q_head_num=4,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=L.HEAD_DIM,
        scaling=1.0,
        sliding_window_size=-1,
    )
    fields.update(over)
    return SimpleNamespace(**fields)


def _pool(dtype=torch.float8_e4m3fn, heads=1):
    return torch.zeros(4 * split_kv.PAGE_SIZE, heads, L.HEAD_DIM, dtype=dtype)


@pytest.fixture
def cp(monkeypatch):
    fake = SimpleNamespace(min_bs=2, hq=4)
    monkeypatch.setattr(split_kv, "cp_lite", lambda: fake)
    return fake


@pytest.mark.parametrize(
    "case, why",
    [
        (dict(requests=1, rows=1), "1 request(s) (CP-lite serves 2 or more)"),
        (dict(requests=65, rows=65), "65 requests (the draft's CP-lite serves 64"),
        (dict(rows=7), "7 rows for 4 requests of 7"),
        (dict(requests=64, q_len=15, rows=960), "960 rows (CP-lite serves 896)"),
        (dict(page_size=32), "pages of 32 slots"),
        (dict(layer=_layer(tp_q_head_num=8)), "8 query / 1 KV heads"),
        (dict(layer=_layer(head_dim=128)), "of 128 a rank"),
        (dict(layer=_layer(sliding_window_size=128)), "a sliding window of 128"),
        (dict(pool=_pool(torch.bfloat16)), "a KV pool of torch.bfloat16"),
        (dict(pool=_pool(heads=2)), "a KV pool of"),
    ],
)
def test_reasons_decline_from_shapes_only(cp, case, why):
    args = dict(
        layer=_layer(), pool=_pool(), rows=28, requests=4, q_len=7, page_size=64
    )
    args.update(case)
    got = draft_cp.reasons(
        args["layer"],
        args["pool"],
        args["rows"],
        args["requests"],
        args["q_len"],
        args["page_size"],
    )
    assert why in got
    assert draft_cp.reasons(_layer(), _pool(), 28, 4, 7, 64) == ""


def test_reasons_without_cp_lite(monkeypatch):
    monkeypatch.setattr(split_kv, "cp_lite", lambda: None)
    assert "did not build CP-lite" in draft_cp.reasons(_layer(), _pool(), 28, 4, 7, 64)


def test_the_cutover_warns_once(caplog):
    with caplog.at_level(logging.WARNING):
        assert not draft_cp.stock_cutover(64, 1)
        assert draft_cp.stock_cutover(65, 1)
        assert draft_cp.stock_cutover(128, 7)
    assert caplog.text.count("runs the stock kernel") == 1


def _view(requests=4, q_len=7, rows=None, **over):
    rows = requests * q_len if rows is None else rows
    fields = dict(
        layer=_layer(),
        q=torch.zeros(rows, 4 * L.HEAD_DIM, dtype=torch.float8_e4m3fn),
        k_pool=_pool(),
        v_pool=_pool(),
        k_scale=0.5,
        v_scale=2.0,
        page_size=split_kv.PAGE_SIZE,
        page_table=torch.zeros(requests, 2, dtype=torch.int32),
        cache_seqlens=torch.full((requests,), 100, dtype=torch.int32),
        q_len=q_len,
        out_dtype=torch.bfloat16,
        step=object(),
    )
    fields.update(over)
    return sgl_spec.DraftAttentionView(**fields)


def test_attend_runs_cp_lites_precise_call_on_the_view(cp, monkeypatch, caplog):
    calls = []
    out = torch.ones(32, 4 * L.HEAD_DIM, dtype=torch.bfloat16)

    def core(attn, q, kv, meta, forward_batch, q_len, **kw):
        calls.append((attn, q, kv, meta, forward_batch, q_len, kw))
        return out

    monkeypatch.setattr(split_kv, "cp_core", core)
    view = _view(rows=32)
    with caplog.at_level(logging.INFO):
        assert draft_cp.attend(view) is out
        assert draft_cp.attend(view) is out
    ((attn, q, kv, meta, step, q_len, kw), _) = calls
    assert attn is view.layer and q.shape == (32, 4 * L.HEAD_DIM) and q_len == 7
    assert step is view.step
    assert kv.layer_ids == [3] and kv.k[0] is view.k_pool and kv.v[0] is view.v_pool
    assert (kv.k_scale, kv.v_scale) == ([0.5], [2.0])
    assert meta.page_table is view.page_table
    assert meta.cache_seqlens is view.cache_seqlens
    assert kw == dict(min_bs=None, pad_rows=True, precise=True)
    assert caplog.text.count("runs CP-lite's precise call") == 1


@pytest.mark.parametrize(
    "view, why",
    [
        (lambda: _view(out_dtype=torch.float16), "torch.float16 output"),
        (lambda: _view(requests=1, q_len=1), "1 request(s)"),
    ],
)
def test_attend_declines_without_calling_cp_lite(cp, monkeypatch, caplog, view, why):
    monkeypatch.setattr(split_kv, "cp_core", lambda *a, **k: pytest.fail("called"))
    with caplog.at_level(logging.INFO):
        assert draft_cp.attend(view()) is None
    assert f"runs stock ({why}" in caplog.text


def test_attend_hands_a_declined_call_back(cp, monkeypatch, caplog):
    monkeypatch.setattr(split_kv, "cp_core", lambda *a, **k: None)
    with caplog.at_level(logging.INFO):
        assert draft_cp.attend(_view()) is None
    assert "runs stock (CP-lite declined the call)" in caplog.text


def _bind(monkeypatch, *, offered=True, cp_why=""):
    refused = []
    monkeypatch.setattr(hooks, "_offers_draft_attention", lambda: offered)
    monkeypatch.setattr(split_kv, "cp_reason", lambda: cp_why)
    monkeypatch.setattr(
        runtime, "unavailable_slots", lambda slots, why, **k: refused.append(why)
    )
    factory = hooks.build(
        SimpleNamespace(), SimpleNamespace(), sharded=False, draft=True
    )
    return factory, refused


def test_the_factory_offers_the_draft_attention(monkeypatch):
    factory, refused = _bind(monkeypatch)
    assert refused == []
    stock = SimpleNamespace(
        post_verify=sgl_spec.PostVerifyStrategy(
            run=lambda io, keys: None, local_keys=lambda view: None
        )
    )
    bound = factory(stock)
    assert bound.draft_attention is draft_cp.attend
    assert bound.post_verify.run is stock.post_verify.run
    assert bound.verify_shard_head is None


@pytest.mark.parametrize(
    "offered, cp_why, why",
    [
        (False, "", "this SGLang's hooks have no draft attention"),
        (True, "the plugin did not build CP-lite", "the plugin did not build CP-lite"),
    ],
)
def test_no_factory_where_the_draft_attention_cannot_run(
    monkeypatch, offered, cp_why, why
):
    factory, refused = _bind(monkeypatch, offered=offered, cp_why=cp_why)
    assert factory is None and refused == [why]


def test_this_sglang_takes_a_draft_attention():
    assert hooks._offers_draft_attention()


class _SplitKV:
    def __init__(self):
        self.calls = []

    def cp_core(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "o"


def _draft_cp(multi=True, precise=True):
    return DraftCP(_SplitKV(), SimpleNamespace(), True, True, multi, precise)


@pytest.mark.parametrize("rows_per_request, rows", [(1, 4), (7, 28), (1, 1), (7, 7)])
def test_the_megakernel_drafts_cp_calls_are_precise(
    monkeypatch, rows_per_request, rows
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    d = _draft_cp()
    attention = SimpleNamespace(layer=SimpleNamespace(attn=_layer()))
    assert d.run(attention, torch.zeros(rows, 4), "meta", "kv", rows_per_request) == "o"
    (kw,) = d.split_kv.calls
    multi = rows > rows_per_request
    assert kw["precise"] is True and kw["merge_h"] is multi
    assert kw["early_merge"] is (False if multi else None)


def test_the_megakernel_draft_keeps_the_stock_decode_past_the_cutover():
    d = _draft_cp()
    assert d.takes(1, 64) and d.takes(7, 64 * 7)
    assert not d.takes(1, 65) and not d.takes(7, 65 * 7)


PUBLIC_SLOTS = frozenset(
    "route,mkm,gdn_verify,gdn_pdl,gdn_conv,hop,hop_fp8,hop_mid,draft_loop,draft_extend,"
    "draft_moe,draft_q1,draft_xchg,draft_hop,draft_fin,draft_front,draft_skv2,draft_multi,"
    "draft_embed,gdn_push,gdn_front,route_shared,skv2,skv2_cp,mka,mka_push,mkm_wide,mkm_m32,"
    "mkm_m64,mkm_m128,mkm_m256,mkm_m512,multi_request,lowm_in,lowm_out,lowm_se,lowm_push,"
    "draft_q1_cp,draft_skv2_cp,sharded_gumbel".split(",")
)


def _built(monkeypatch, slots):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    monkeypatch.setattr(split_kv, "cp_reason", lambda: "")
    parts = SimpleNamespace(slots=frozenset(slots) & set(runtime.DRAFT_SLOTS))
    d = DraftCP.build(parts)
    d.split_kv = _SplitKV()
    return d


@pytest.mark.parametrize("rows_per_request, rows", [(1, 4), (7, 28), (1, 1), (7, 7)])
def test_a_slot_list_without_it_keeps_the_drafts_earlier_cp_calls(
    monkeypatch, rows_per_request, rows
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv(runtime.SLOTS_ENV, ",".join(sorted(PUBLIC_SLOTS)))
    d = _built(monkeypatch, PUBLIC_SLOTS)
    assert d.precise is False
    attention = SimpleNamespace(layer=SimpleNamespace(attn=_layer()))
    assert d.run(attention, torch.zeros(rows, 4), "meta", "kv", rows_per_request) == "o"
    multi = rows > rows_per_request
    # exactly the call before the slot: no precise, no merge_h
    assert d.split_kv.calls == [
        {"min_bs": None if multi else 1, "early_merge": False if multi else None}
    ]
    assert d.takes(1, 65) and d.takes(7, 455) and d.takes(1, 128)


def test_the_slot_makes_the_drafts_cp_calls_precise(monkeypatch):
    d = _built(monkeypatch, PUBLIC_SLOTS | {SLOT})
    assert d.precise is True
    assert d.takes(1, 64) and not d.takes(1, 65) and not d.takes(7, 455)
