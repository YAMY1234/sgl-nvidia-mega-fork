# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5 import runtime, topology
from sgl_nvidia_mega.qwen3_5.gdn import commit as gdn_commit
from sgl_nvidia_mega.qwen3_5.runtime import executor
from sgl_nvidia_mega.qwen3_5.sm103.gdn import commit as kernel

RUNTIME_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
LAYERS, SLOTS, ROWS, HV, H, RING, CONV = 3, 10, 6, 8, 2, 16, 4


@pytest.fixture(autouse=True)
def agreed(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    monkeypatch.setattr(
        plugin_binding, "megakernel_agree", lambda *args, **kwargs: None
    )


@pytest.fixture
def measured(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV, "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(runtime, "MEASURED_KEY", None)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())


def _parallel(tp, attn_tp, dp, ep, pp):
    return SimpleNamespace(
        tp_size=tp,
        attn_tp_size=attn_tp,
        attn_dp_size=dp,
        moe_ep_size=ep,
        pp_size=pp,
        tp_group=SimpleNamespace(world_size=tp),
    )


@pytest.mark.parametrize(
    "switch, slots", [(None, None), ("0", None), ("1", "hop,route_topk")]
)
def test_option_gdn_commit_is_off_without_the_measured_stack(
    monkeypatch, switch, slots
):
    for name, value in ((RUNTIME_ENV, switch), (runtime.SLOTS_ENV, slots)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "MEASURED_KEY", None)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    runtime.apply_measured_plan(_parallel(8, 8, 1, 4, 1), topology.GB300)
    assert not runtime.measured_option("gdn_commit")
    monkeypatch.setattr(runtime, "MEASURED_KEY", ("tep8", topology.GB300))
    assert not runtime.measured_option("gdn_commit")
    assert runtime.build_recurrent_commit() is None


def test_the_measured_numerics_claim_covers_the_commit(measured):
    from sgl_nvidia_mega.qwen3_5 import spec

    assert "gdn_commit" in runtime.measured_options()
    assert "gdn_commit" in runtime.NUMERICS_CHANGE
    claim = spec.numerics(
        topology.plannable(runtime.SLOTS) | runtime.measured_options()
    )
    assert "gdn_commit" in spec.NUMERIC_TOLERANCES
    assert claim.kind.value == "tolerance"


def test_the_build_compiles_the_commit_only_under_the_option(monkeypatch):
    built = []
    monkeypatch.setattr(kernel, "build", built.append)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(runtime, "measured_option", lambda name: False)
    assert runtime.build_recurrent_commit() is None and built == []
    monkeypatch.setattr(runtime, "measured_option", lambda name: name == "gdn_commit")
    provider = runtime.build_recurrent_commit()
    assert isinstance(provider, gdn_commit.RecurrentCommit)
    assert built == [3] and provider.pools is None


# ------------------------------------------------------------------- the executor


def _verify(provider=None):
    lm = SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))
    return executor.Qwen35Verify(
        lm, {}, {}, (7,), binding="binding", recurrent_commit=provider
    )


def test_the_executor_offers_sglangs_recurrent_commit_only_with_a_provider():
    spec = pytest.importorskip("sglang.kernels.plugin_kernels.nvidia.spec")
    if not hasattr(spec, "SupportsRecurrentCommit"):
        pytest.skip("this SGLang has no SupportsRecurrentCommit")
    assert not isinstance(_verify(), spec.SupportsRecurrentCommit)
    calls = []
    built = _verify(lambda io: calls.append(io) or True)
    assert isinstance(built, spec.SupportsRecurrentCommit)
    assert built.commit_recurrent_state("io") is True and calls == ["io"]


def test_the_first_covered_step_binds_the_pools_once():
    bound = []
    provider = SimpleNamespace(bind=bound.append)
    built = _verify(provider)
    view = SimpleNamespace(mamba="pools")
    built._check_pools(view)
    assert bound == ["pools"] and built._pools_checked
    # without a provider the check binds nothing
    _verify()._check_pools(view)
    assert bound == ["pools"]


def _pools(**overrides):
    meta = dict(device="meta")
    bf16 = dict(dtype=torch.bfloat16, **meta)
    pools = dict(
        conv=tuple(torch.empty(LAYERS, SLOTS, 3, CONV, **bf16) for _ in range(2)),
        intermediate_conv_window=tuple(
            torch.empty(LAYERS, SLOTS, 7, 3, CONV, **bf16) for _ in range(2)
        ),
        temporal=torch.empty(LAYERS, SLOTS, HV, 128, 128, **bf16),
        ring_d=torch.empty(LAYERS, SLOTS, HV, RING, 128, **bf16),
        ring_k=torch.empty(LAYERS, SLOTS, H, RING, 128, **bf16),
        ring_g=torch.empty(LAYERS, SLOTS, HV, RING, dtype=torch.float32, **meta),
        ring_rawv=torch.empty(LAYERS, SLOTS, HV, RING, 128, **bf16),
        ring_rawk=torch.empty(LAYERS, SLOTS, H, RING, 128, **bf16),
        ring_beta=None,
        ring_write_pos=torch.empty(SLOTS, dtype=torch.int32, **meta),
        ring_cache_base=torch.empty(SLOTS, dtype=torch.int32, **meta),
        ring_is_flush=torch.empty(SLOTS, dtype=torch.int8, **meta),
    )
    pools.update(overrides)
    return SimpleNamespace(**pools)


def test_the_compact_bf16_ring_is_served():
    assert gdn_commit.pool_reasons(_pools()) == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(ring_cache_base=None), "no circular ReplaySSM verify ring"),
        (dict(ring_d=None), "no circular ReplaySSM verify ring"),
        (dict(ring_g=None), "no circular ReplaySSM verify ring"),
        (
            dict(
                ring_d=torch.empty(LAYERS, SLOTS, HV, 32, 128, device="meta"),
            ),
            "ring length 32 (serves 16)",
        ),
        (dict(ring_beta=torch.empty(1, device="meta")), "not the compact"),
        (dict(ring_rawk=None), "not the compact"),
        (
            dict(
                temporal=torch.empty(
                    LAYERS, SLOTS, HV, 128, 128, dtype=torch.float32, device="meta"
                )
            ),
            "torch.float32 checkpoints",
        ),
        (
            dict(
                temporal=torch.empty(
                    LAYERS, SLOTS, HV, 128, 256, dtype=torch.bfloat16, device="meta"
                )[..., :128]
            ),
            "checkpoints",
        ),
        (
            dict(
                ring_k=torch.empty(
                    LAYERS, SLOTS, 3, RING, 128, dtype=torch.bfloat16, device="meta"
                )
            ),
            "8 value heads over 3 key heads",
        ),
        (
            dict(
                ring_rawv=torch.empty(
                    LAYERS, SLOTS, HV, RING, 128, dtype=torch.float16, device="meta"
                )
            ),
            "ring torch.float16",
        ),
        (
            dict(
                ring_g=torch.empty(
                    LAYERS, SLOTS, HV, RING, dtype=torch.bfloat16, device="meta"
                )
            ),
            "ring gates torch.bfloat16",
        ),
        (
            dict(ring_write_pos=torch.empty(SLOTS, dtype=torch.int64, device="meta")),
            "ring cursors are not unit-stride torch.int32",
        ),
        (
            dict(
                ring_is_flush=torch.empty(SLOTS, 2, dtype=torch.int8, device="meta")[
                    :, 0
                ]
            ),
            "ring cursors are not unit-stride torch.int8",
        ),
        (
            dict(ring_g=torch.empty(LAYERS, SLOTS, HV, RING, dtype=torch.float32)),
            "different devices",
        ),
    ],
)
def test_the_commit_declines_pools_it_does_not_serve(overrides, reason):
    assert any(reason in r for r in gdn_commit.pool_reasons(_pools(**overrides)))


def test_without_pools_the_commit_declines():
    assert gdn_commit.pool_reasons(None) == ["no GDN pools"]


# ------------------------------------------------------------------ one commit


@pytest.fixture
def launches(monkeypatch):
    """SGLang's two ops and the CUDA launcher, recorded in order."""
    calls = []
    advance = ModuleType("advance")
    advance.commit_gdn_replayssm_spec = lambda **kw: calls.append(("advance", kw))
    scatter = ModuleType("scatter")
    scatter.fused_conv_window_scatter_with_mask = lambda *a: calls.append(("conv", a))
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode",
        advance,
    )
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.ops.mamba.mamba_state_scatter_triton", scatter
    )
    monkeypatch.setattr(kernel, "build", lambda device_index: None)
    monkeypatch.setattr(kernel, "commit", lambda **kw: calls.append(("cuda", kw)))
    return calls


def _io(track):
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return SimpleNamespace(
        state_slots=i32(4, 7, -1),
        track_slots=torch.tensor([8, -1, -1]) if track else None,
        request_rows=torch.tensor([2, 5, 0]),
        accept_lens=i32(3, 1, 1),
        last_step=i32(2, 0, 0),
        track_steps=torch.tensor([1, -1, -1]),
        draft_token_num=7,
        src_indices_raw=None,
    )


def test_before_the_pools_are_bound_the_commit_declines(launches):
    provider = gdn_commit.RecurrentCommit(0)
    assert provider(_io(False)) is False and launches == []


def test_refused_pools_leave_sglangs_commit_with_one_line(launches, caplog):
    provider = gdn_commit.RecurrentCommit(0)
    with caplog.at_level("INFO"):
        provider.bind(_pools(ring_rawk=None))
        provider.bind(_pools())  # bound once: the first covered step decides
    assert caplog.text.count("SGLang's recurrent commit runs") == 1
    assert "CUDA ReplaySSM commit runs" not in caplog.text
    assert provider(_io(False)) is False and launches == []


@pytest.mark.parametrize("track", [False, True])
def test_one_commit_launches_sglangs_order(launches, caplog, track):
    pools = _pools()
    provider = gdn_commit.RecurrentCommit(0)
    with caplog.at_level("INFO"):
        provider.bind(pools)
    assert "the CUDA ReplaySSM commit runs (ring 16, 3 GDN layers)" in caplog.text
    io = _io(track)
    assert provider(io) is True
    kinds = [kind for kind, _ in launches]
    conv = ["conv", "conv"] if track else ["conv"]
    assert kinds == ["advance", "cuda"] + conv * 2  # two conv layers
    advance, cuda = launches[0][1], launches[1][1]
    assert advance == dict(
        write_pos=pools.ring_write_pos,
        cache_base=pools.ring_cache_base,
        is_flush=pools.ring_is_flush,
        num_accepted=io.accept_lens,
        replay_indices=io.request_rows,
        state_batch_indices=io.state_slots,
        max_cache_len=RING,
        max_spec_len=7,
        fold_every_commit=True,
        null_block_id=-1,
    )
    assert cuda == dict(
        checkpoint=pools.temporal,
        ring_d=pools.ring_d,
        ring_dlo=pools.ring_rawv,
        ring_k=pools.ring_k,
        ring_klo=pools.ring_rawk,
        ring_g=pools.ring_g,
        state_indices=io.state_slots,
        ring_rows=io.request_rows,
        write_pos=pools.ring_write_pos,
        cache_base=pools.ring_cache_base,
        is_flush=pools.ring_is_flush,
        accept_lens=io.accept_lens,
        track_indices=io.track_slots,
        track_steps=io.track_steps,
        null_block=-1,
    )
    rollback = launches[2][1]
    assert rollback == (
        pools.conv[0],
        pools.intermediate_conv_window[0],
        io.state_slots,
        io.last_step,
        None,
    )
    if track:
        assert launches[3][1] == (
            pools.conv[0],
            pools.intermediate_conv_window[0],
            io.track_slots,
            io.track_steps,
            None,
        )


# ---------------------------------------------------------- the kernels' launches


def test_the_params_struct_is_commit_cus():
    source = Path(kernel.SOURCE).read_text()
    body = re.search(r"struct Params \{(.*?)\n\};", source, re.S).group(1)
    sizes = {"i32": 4, "i64": 8, "i8": 1}
    fields = []
    for line in body.splitlines():
        line = line.split("//")[0].strip().rstrip(";")
        if not line:
            continue
        kind, names = line.split(None, 1) if "*" not in line else line.split("*", 1)
        size = 8 if "*" in line else sizes[kind.split()[-1]]
        fields += [(n.strip().lstrip("*"), size) for n in names.split(",")]
    assert [(n, ctypes.sizeof(t)) for n, t in kernel.Params._fields_] == fields
    assert ctypes.sizeof(kernel.Params) == 248


def test_the_shared_memory_is_two_stages_of_the_ring():
    assert kernel.shared_bytes(16) == 2 * (16384 + 8192 + 4096 + 64)
    assert kernel.shared_bytes(32) == 2 * (16384 + 16384 + 8192 + 128)


@pytest.mark.parametrize("track", [False, True])
def test_the_commit_launches_one_cta_an_item_then_the_cursor_reset(monkeypatch, track):
    launched = []
    monkeypatch.setattr(kernel, "_kernels", lambda device, ring: ("main", "finish"))
    monkeypatch.setattr(
        jit,
        "launch",
        lambda k, grid, args, stream, pdl=True: launched.append(
            (k, grid, args, stream, pdl)
        ),
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=9)
    )
    layers, hv, hk = 7, HV, H
    temporal = torch.zeros(layers, SLOTS, hv, 128, 128, dtype=torch.bfloat16)
    ring = lambda heads: torch.zeros(  # noqa: E731
        layers, SLOTS, heads, RING, 128, dtype=torch.bfloat16
    )
    io = _io(track)
    kernel.commit(
        checkpoint=temporal,
        ring_d=ring(hv),
        ring_dlo=ring(hv),
        ring_k=ring(hk),
        ring_klo=ring(hk),
        ring_g=torch.zeros(layers, SLOTS, hv, RING),
        state_indices=io.state_slots,
        ring_rows=io.request_rows,
        write_pos=torch.zeros(SLOTS, dtype=torch.int32),
        cache_base=torch.zeros(SLOTS, dtype=torch.int32),
        is_flush=torch.zeros(SLOTS, dtype=torch.int8),
        accept_lens=io.accept_lens,
        track_indices=io.track_slots,
        track_steps=io.track_steps,
    )
    (main, grid, (p,), stream, pdl), finish = launched
    assert (main, grid, stream, pdl) == ("main", (3 * 8 * 2 * 2, 1, 1), 9, False)
    assert (p.items, p.chunks, p.layers_per_cta, p.num_layers) == (96, 2, 5, 7)
    assert (p.hv, p.hk, p.L, p.null_block, p.has_track) == (8, 2, RING, -1, track)
    assert p.idx64 == (0b11010 if track else 0b00010)
    assert p.s_state_slot == hv * 128 * 128 and p.s_g_slot == hv * RING
    assert finish[0] == "finish" and finish[1] == (1, 1, 1) and not finish[4]
    assert finish[2][1].value == 3


def test_an_empty_batch_launches_nothing(monkeypatch):
    monkeypatch.setattr(jit, "launch", lambda *a, **kw: pytest.fail("launched"))
    empty = torch.zeros(0, dtype=torch.int32)
    kernel.commit(
        checkpoint=None,
        ring_d=None,
        ring_dlo=None,
        ring_k=None,
        ring_klo=None,
        ring_g=None,
        state_indices=empty,
        ring_rows=empty,
        write_pos=None,
        cache_base=None,
        is_flush=None,
        accept_lens=empty,
        track_indices=None,
        track_steps=None,
    )


@pytest.mark.parametrize("pdl, attributes", [(True, 1), (False, 0)])
def test_a_launch_without_pdl_sets_no_launch_attribute(monkeypatch, pdl, attributes):
    seen = []

    class Config:
        pass

    class Attribute:
        def __init__(self):
            self.value = SimpleNamespace()

    fake = SimpleNamespace(
        CUtensorMap=type("CUtensorMap", (), {}),
        CUlaunchConfig=Config,
        CUlaunchAttribute=Attribute,
        CUlaunchAttributeID=SimpleNamespace(
            CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION=1,
            CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION=2,
        ),
        CUstream=lambda s: s,
        cuLaunchKernelEx=lambda config, *a: seen.append(config) or (0,),
    )
    monkeypatch.setattr(jit, "driver", lambda: fake)
    k = jit.Kernel("function", (128, 1, 1), 0, b"")
    jit.launch(k, (1, 1, 1), [ctypes.c_int(1)], 0, pdl=pdl)
    assert seen[0].numAttrs == attributes == len(seen[0].attrs)
