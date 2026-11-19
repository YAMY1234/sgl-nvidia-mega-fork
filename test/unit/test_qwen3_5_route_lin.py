# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.models import qwen3_5

from sgl_nvidia_mega.qwen3_5 import attention, gdn, runtime
from sgl_nvidia_mega.qwen3_5.attention import mka
from sgl_nvidia_mega.qwen3_5.sm103 import hop as sm103_hop

SM103 = Path(sm103_hop.__file__).parent.parent
RL_PACKED = 16 * 8 * 12 * 2 * 32 * 8 + sm103_hop.RL_TAIL
A_FRAGMENT = [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)]


def test_the_router_mailbox_holds_the_partials_then_the_cells():
    assert sm103_hop.RL_CELLS == sm103_hop.HIDDEN // 32 == 128
    assert sm103_hop.rl_box_words(8, 15) == 3 * 8 * 15 * 512 + 3 * 8 * 128 * 16
    assert sm103_hop.rl_box_words(8, 8) == 3 * 8 * 8 * 512 + 3 * 8 * 128 * 8


def test_the_cell_constants_agree_across_the_sources():
    rlin = (SM103 / "hop" / "src" / "rlin.cu").read_text()
    mka_cu = (SM103 / "attention" / "src" / "mka.cu").read_text()
    assert re.search(r"constexpr int kCells = kHidden / 32;", rlin)
    assert re.search(r"constexpr int kRlCells = 4096 / 32;", mka_cu)
    assert "static_cast<size_t>(kStages) * s.world * s.m_cap * kExperts +" in rlin
    assert "* kCells + g) * static_cast<size_t>((s.m_cap + 3) & ~3) + token;" in rlin
    assert "(size_t)q35hop::kStages * s.world * s.m_cap * kRlExperts +" in mka_cu
    assert "* kRlCells + g) * (size_t)((s.m_cap + 3) & ~3) + token;" in mka_cu


def _struct_fields(source: str) -> list:
    body = re.search(r"\nstruct RlArgs \{[^\n]*\n(.*?)\n\};", source, re.S).group(1)
    code = "\n".join(line.split("//")[0] for line in body.splitlines())
    return re.findall(r"(\w+)\s*;", code)


def test_the_ctypes_rl_args_carry_every_field_of_the_device_structs():
    names = [name for name, _ in sm103_hop.RlArgs._fields_]
    mka_cu = (SM103 / "attention" / "src" / "mka.cu").read_text()
    rlin = (SM103 / "hop" / "src" / "rlin.cu").read_text()
    assert _struct_fields(mka_cu) == names
    assert _struct_fields(rlin) == names[:-1]
    assert ctypes.sizeof(sm103_hop.RlArgs) == 80
    assert sm103_hop.RlArgs().trace is None


def _pack_inputs(*, router_dtype=torch.bfloat16, gamma=None, weight_scale=1.0):
    gen = torch.Generator().manual_seed(0)
    router = (torch.randn(512, 4096, generator=gen) * 0.02).to(router_dtype)
    out_w = (torch.randn(4096, 1024, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    if gamma is None:
        gamma = (1.0 + torch.rand(4096, generator=gen)).to(torch.bfloat16)
    return router, gamma, out_w, torch.tensor(0.25), torch.tensor(weight_scale)


def test_the_pack_folds_the_core_under_one_scale_and_appends_the_tail():
    router, gamma, out_w, sx, sw = _pack_inputs()
    packed = sm103_hop.pack_router_lin(router, gamma, out_w, sx, sw, 1)
    assert packed.dtype == torch.int16 and packed.numel() == RL_PACKED
    tail = packed[-sm103_hop.RL_TAIL :]
    unscale = float(tail[512:514].clone().view(torch.float32))
    scale = 1.0 / unscale
    assert scale == 2.0 ** round(torch.log2(torch.tensor(scale)).item())
    assert torch.equal(
        tail[:512].view(torch.bfloat16).double(), gamma[512:1024].double() * scale
    )
    assert not tail[514:].any()


def _unpack_router_lin(packed):
    frags = packed[: -sm103_hop.RL_TAIL].view(16, 8, 12, 2, 32, 8)
    a = torch.empty(16, 2, 16, 8, 12, 16, dtype=torch.int16)
    for lane in range(32):
        g, t = divmod(lane, 4)
        for half, (row, col) in enumerate(A_FRAGMENT):
            a[:, :, g + row, :, :, 2 * t + col] = frags[..., lane, half].permute(
                0, 3, 1, 2
            )
    a = a.reshape(512, 8, 12, 16)
    core = a[:, :, :8].reshape(512, 1024).view(torch.float16)
    return core, a[:, :, 8:].reshape(512, 512).view(torch.bfloat16)


def test_the_packed_fragments_give_the_router_partial_within_fp16_rounding_of_fp64():
    router, gamma, out_w, sx, sw = _pack_inputs()
    rank, k_r = 3, slice(1536, 2048)
    packed = sm103_hop.pack_router_lin(router, gamma, out_w, sx, sw, rank)
    core, router_slice = _unpack_router_lin(packed)
    tail = packed[-sm103_hop.RL_TAIL :]
    gamma_2s = tail[:512].view(torch.bfloat16).double()
    unscale = float(tail[512:514].clone().view(torch.float32))
    a = router.double() * gamma.double()
    c = a @ out_w.float().double() * float(sx * sw)
    # the slice as stored; the core is C 2^s to the nearest fp16
    assert torch.equal(router_slice.view(torch.int16), router[:, k_r].view(torch.int16))
    c_2s = c / unscale
    assert ((core.double() - c_2s).abs() <= c_2s.abs() * 2.0**-11 + 2.0**-25).all()
    # this rank's router partial for 7 rows from the decoded operands, against FP64
    gen = torch.Generator().manual_seed(1)
    x = (torch.randn(1024, 7, generator=gen) * 2).to(torch.float8_e4m3fn).double()
    h0 = torch.randn(512, 7, generator=gen).to(torch.bfloat16).double()
    ref = c @ x + a[:, k_r] @ h0
    got = (
        core.double() @ x + router_slice.double() @ (h0 * gamma_2s[:, None])
    ) * unscale
    assert ((got - ref).norm(dim=0) / ref.norm(dim=0)).max() <= 2.0**-11


def test_the_pack_declines_a_router_that_is_not_bf16():
    router, gamma, out_w, sx, sw = _pack_inputs(router_dtype=torch.float16)
    with pytest.raises(ValueError, match="needs a bf16 router, not torch.float16"):
        sm103_hop.pack_router_lin(router, gamma, out_w, sx, sw, 0)


def test_the_pack_declines_a_core_with_no_fp16_scale():
    router, gamma, out_w, sx, sw = _pack_inputs(weight_scale=1e30)
    with pytest.raises(ValueError, match="has no fp16 scale"):
        sm103_hop.pack_router_lin(router, gamma, out_w, sx, sw, 0)


def test_the_pack_declines_a_gamma_slice_not_exact_in_bf16_on_its_rank_only():
    gamma = torch.ones(4096, dtype=torch.float32)
    gamma[7] = 1.0 + 2.0**-12
    router, gamma, out_w, sx, sw = _pack_inputs(gamma=gamma)
    with pytest.raises(ValueError, match=r"gamma \* 2\^s is not exact in bf16"):
        sm103_hop.pack_router_lin(router, gamma, out_w, sx, sw, 0)
    # the same weights pack on rank 1: the check reads each rank's own slice
    assert (
        sm103_hop.pack_router_lin(router, gamma, out_w, sx, sw, 1).numel() == RL_PACKED
    )


def _tp8(monkeypatch, others: dict):
    """An 8-rank TP group of which this process is rank 0; rank r reports others[r]."""
    tp_group = SimpleNamespace(world_size=8, device_group="device", cpu_group="cpu")
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_group=tp_group),
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    def gather(out, mine, group=None):
        assert group == "cpu"
        out[:] = [mine] + [others.get(r, "") for r in range(1, 8)]

    monkeypatch.setattr(dist, "all_gather_object", gather)


def test_the_ranks_agree_on_every_ranks_decline_and_warn(monkeypatch, caplog):
    _tp8(monkeypatch, {})
    with caplog.at_level(logging.INFO, logger=runtime.logger.name):
        assert runtime.agreed_decline("route_lin", "", cpu_group="cpu") == ""
        assert not caplog.records
        assert runtime.agreed_decline("route_lin", "x", cpu_group=None) == "x"
        _tp8(monkeypatch, {3: "a", 5: "b", 6: "a"})
        both = "a (ranks [3, 6]); b (ranks [5])"
        assert runtime.agreed_decline("route_lin", "", cpu_group="cpu") == both
        # slot_supported (the draft slots' shard checks) keeps its de-duplicated reasons
        assert not runtime.slot_supported("draft_multi", "", cpu_group="cpu")
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("WARNING", "NVIDIA megakernel: slot route_lin uses stock on every rank: x"),
        (
            "WARNING",
            f"NVIDIA megakernel: slot route_lin uses stock on every rank: {both}",
        ),
        ("INFO", "NVIDIA megakernel: requested slots ['draft_multi'] use stock (a; b)"),
    ]


def _declines(caplog) -> list:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _gdn_layer():
    layer = SimpleNamespace(
        layer_id=0,
        mlp=SimpleNamespace(gate=SimpleNamespace(weight="router"), layer_id=0),
        ffn_boundary=SimpleNamespace(plan=SimpleNamespace(norm="norm")),
    )
    mixer = SimpleNamespace(
        attn=SimpleNamespace(layer_id=0),
        out_proj=SimpleNamespace(
            weight=torch.zeros(2, 2), input_scale=1, weight_scale=1
        ),
    )
    return SimpleNamespace(layers=[layer]), [(0, mixer)]


@pytest.mark.parametrize(
    "decline", ["", "layer 0: router-lin gamma * 2^s is not exact in bf16"]
)
def test_a_gdn_pack_decline_on_one_rank_leaves_every_rank_on_stock(
    monkeypatch, caplog, decline
):
    _tp8(monkeypatch, {3: decline})
    monkeypatch.setattr(
        "sglang.srt.layers.layer_boundary.fusions.norm.fused_norm_gamma",
        lambda norm: "gamma",
    )
    monkeypatch.setattr(sm103_hop, "pack_router_lin", lambda *a: "packed")
    model, gdns = _gdn_layer()
    rlin, reason = gdn._route_lin(model, gdns)
    if decline:
        assert rlin == {} and reason == f"{decline} (ranks [3])"
        assert _declines(caplog) == [
            f"NVIDIA megakernel: slot route_lin uses stock on every rank: {reason}"
        ]
    else:
        assert rlin == {0: "packed"} and reason == "" and not _declines(caplog)


def _attention_model():
    layers = []
    for _ in range(2):
        made = qwen3_5.Qwen3_5AttentionDecoderLayer.__new__(
            qwen3_5.Qwen3_5AttentionDecoderLayer
        )
        torch.nn.Module.__init__(made)
        layers.append(made)
    return SimpleNamespace(
        is_mrope_enabled=True,
        model=SimpleNamespace(start_layer=0, end_layer=2, layers=layers),
    )


@pytest.mark.parametrize(
    "decline", ["", "router-lin core max 1.0e+30 has no fp16 scale"]
)
def test_an_attention_pack_decline_on_one_rank_leaves_every_rank_on_stock(
    monkeypatch, caplog, decline
):
    _tp8(monkeypatch, {5: f"layer 1: {decline}" if decline else ""})
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"mka", "mka_push", "route_lin"}))
    monkeypatch.setattr(mka, "unsupported", lambda layer: "")
    monkeypatch.setattr(mka, "RL_WEIGHTS", {})

    def pack(layer, route_lin=False):
        mka.RL_WEIGHTS[id(layer)] = "packed"
        return ""

    monkeypatch.setattr(mka, "pack", pack)
    bound = attention.build(_attention_model(), SimpleNamespace(tokens_per_request=7))
    assert bound.mka and bound.mka_push
    if decline:
        assert not bound.route_lin and mka.RL_WEIGHTS == {}
        assert bound.reason == f"slot route_lin: layer 1: {decline} (ranks [5])"
        assert _declines(caplog) == [
            "NVIDIA megakernel: slot route_lin uses stock on every rank: "
            f"layer 1: {decline} (ranks [5])"
        ]
    else:
        assert bound.route_lin and len(mka.RL_WEIGHTS) == 2 and bound.reason == ""
        assert not _declines(caplog)


def _rl_hop():
    hop = object.__new__(sm103_hop.Hop)
    hop.max_m, hop._pushed, hop._pushed_rl, hop._pushed_cells = 8, None, False, False
    return hop


def test_a_router_lin_push_without_cells_is_refused_before_any_launch():
    hop = _rl_hop()
    out = torch.zeros(7, 16)
    hop.mark_pushed_rl(out)
    with pytest.raises(RuntimeError, match="no rounding cells"):
        hop.all_reduce_residual_rms_norm_rl(
            local_contribution=out, residual=None, gamma=None, global_scale=None
        )


def test_mark_pushed_rl_records_the_cells():
    hop = _rl_hop()
    out = torch.zeros(7, 16)
    hop.mark_pushed_rl(out, cells=True)
    assert hop._pushed_rl and hop._pushed_cells and hop.has_pushed(out)
