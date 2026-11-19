# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.qwen3_5.gdn import prefill_fi

NK, NV, D, W = 1, 2, 4, 4
QKV = (2 * NK + NV) * D
SLOTS = 9
LENS = [5, 3, 4, 2]
INDICES = [3, 7, -1, 5]
PREFIX = [2, 0, 0, 1]
EVERY = 2


def _conv(
    x,
    weight,
    bias,
    *,
    conv_states,
    cache_indices,
    query_start_loc,
    has_initial_state,
    **kw,
):
    """A causal conv stand-in over the token-major rows, [conv_dim, T] in and out."""
    out = x.clone()
    cu = query_start_loc.tolist()
    for i, slot in enumerate(cache_indices.tolist()):
        if slot < 0 or cu[i + 1] == cu[i]:
            continue
        rows = slice(cu[i], cu[i + 1])
        state = conv_states[slot] * has_initial_state[i]
        out[:, rows] = x[:, rows] * 2 + state[:, -1:] * weight[:, :1]
        tail = torch.cat([state, x[:, rows]], dim=1)[:, -state.shape[1] :]
        conv_states[slot] = tail
    return out


def _split(mixed):
    q, k, v = torch.split(mixed, [NK * D, NK * D, NV * D], dim=-1)
    return (t.reshape(t.shape[0], -1, D) for t in (q, k, v))


def _prepare(q, k, v):
    return (q * 0.5).contiguous(), (k * 0.25).contiguous(), v.contiguous()


def _gates(A_log, a, b, dt_bias, *args):
    g = (a.float() + dt_bias.float()) * -A_log.float().exp()
    return g.unsqueeze(0), b.float().sigmoid().unsqueeze(0)


def _run_prefill(
    *,
    q,
    k,
    v,
    alpha,
    beta,
    initial_state,
    cu_seqlens,
    output,
    state_checkpoints,
    checkpoint_cu_starts,
    checkpoint_every_n_tokens,
    use_cp="auto",
):
    out = v * alpha[..., None] * beta[..., None] + q.sum(-1, keepdim=True)
    final = initial_state.clone()
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        for t in range(cu[i], cu[i + 1]):
            final[i] = final[i] * 0.5 + k[t, 0].sum() + v[t, 0].sum()
            n = t - cu[i] + 1
            if state_checkpoints is not None and n % checkpoint_every_n_tokens == 0:
                c = checkpoint_cu_starts[i] + n // checkpoint_every_n_tokens - 1
                state_checkpoints[c] = final[i]
            out[t] += final[i].sum()
    return out, final


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = sum(LENS)
    cu = [0]
    for n in LENS:
        cu.append(cu[-1] + n)
    conv = torch.randn(SLOTS, QKV, W - 1, generator=g)
    ssm = torch.randn(SLOTS, NV, D, D, generator=g)
    mixed = torch.randn(tokens, QKV, generator=g)
    a = torch.randn(tokens, NV, generator=g)
    b = torch.randn(tokens, NV, generator=g)
    counts = [n // EVERY for n in LENS]
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + c)
    meta = SimpleNamespace(
        query_start_loc=torch.tensor(cu, dtype=torch.int32),
        mamba_cache_indices=torch.tensor(INDICES, dtype=torch.int32),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        conv_block_table=None,
        has_mamba_track_mask=True,
        track_conv_indices=torch.tensor([[1, 2, 3], [10, 11, 12]]),
        conv_states_mask_indices=torch.tensor([8, 6]),
        # request 0's checkpoint 1 (token 4) into slot 2, request 1's (token 2) into 4
        track_ssm_h_src=torch.tensor([starts[0] + 1, starts[1]]),
        track_ssm_h_dst=torch.tensor([2, 4]),
        track_ssm_final_src=torch.tensor([5]),
        track_ssm_final_dst=torch.tensor([1]),
        track_ssm_recompute_dst=None,
        track_chunk_idx=None,
        state_checkpoint_cu_starts=torch.tensor(starts, dtype=torch.int64),
        num_state_checkpoints=starts[-1],
        state_checkpoint_every_n_tokens=EVERY,
    )
    layer = SimpleNamespace(
        layer_id=0,
        num_q_heads=NK,
        num_k_heads=NK,
        num_v_heads=NV,
        head_q_dim=D,
        head_k_dim=D,
        head_v_dim=D,
        q_dim=NK * D,
        k_dim=NK * D,
        v_dim=NV * D,
        conv_weights=torch.randn(QKV, W, generator=g),
        bias=None,
        activation="silu",
        A_log=torch.rand(NV, generator=g),
        dt_bias=torch.randn(NV, generator=g),
    )
    fb = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        extend_prefix_lens=torch.tensor(PREFIX, dtype=torch.int32),
        extend_seq_lens_cpu=LENS,
        global_num_token_non_padded_cpu=tokens,
    )
    return SimpleNamespace(
        conv=conv, ssm=ssm, mixed=mixed, a=a, b=b, meta=meta, layer=layer, fb=fb
    )


def _kernel():
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = True
    kernel._prefill_needs_fp32_state = False
    kernel._run_prefill = _run_prefill
    return kernel


def _backend(case, conv, ssm, kernel):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from sglang.srt.layers.attention.linear import gdn_backend

    backend = SimpleNamespace(
        prefill_graph_metadata=None,
        forward_metadata=case.meta,
        mis_metadata=None,
        requires_contiguous_prefill_state=True,
        state_layer_cache=lambda _: SimpleNamespace(conv=(conv,), temporal=ssm),
        kernel_dispatcher=SimpleNamespace(
            extend_kernel=kernel,
            extend=lambda q, k, v, g, beta, **kw: kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                **{
                    name: kw[name]
                    for name in (
                        "ssm_states",
                        "cache_indices",
                        "query_start_loc",
                        "state_checkpoint_cu_starts",
                        "num_state_checkpoints",
                        "state_checkpoint_every_n_tokens",
                    )
                },
                output=kw["output"],
            ),
        ),
        _prefill_gates=gdn_backend.GDNAttnBackend._prefill_gates,
    )
    backend._convolve_prefill = lambda *args: (
        gdn_backend.GDNAttnBackend._convolve_prefill(backend, *args)
    )
    backend._track_mamba_state_extend = lambda *args, **kw: (
        MambaAttnBackendBase._track_mamba_state_extend(backend, *args, **kw)
    )
    return backend


def _view(case, conv, ssm):
    meta = case.meta
    return SimpleNamespace(
        state=SimpleNamespace(layer_ids=(0,), conv=(conv[None],), temporal=ssm[None]),
        query_start_loc=meta.query_start_loc,
        state_indices=meta.mamba_cache_indices,
        conv_block_table=None,
        kernel_id=0,
        requires_contiguous_state=True,
        is_graph=False,
        has_mis=False,
        has_track=True,
        track_conv_indices=meta.track_conv_indices,
        track_conv_dst=meta.conv_states_mask_indices,
        track_ssm_h_src=meta.track_ssm_h_src,
        track_ssm_h_dst=meta.track_ssm_h_dst,
        track_ssm_final_src=meta.track_ssm_final_src,
        track_ssm_final_dst=meta.track_ssm_final_dst,
        track_ssm_recompute_dst=None,
        state_checkpoint_cu_starts=meta.state_checkpoint_cu_starts,
        num_state_checkpoints=meta.num_state_checkpoints,
        state_checkpoint_every_n_tokens=meta.state_checkpoint_every_n_tokens,
    )


def _front_module(calls=None):
    front = ModuleType("sgl_nvidia_mega.qwen3_5.sm103.gdn.prefill_front")

    def gdn_prefill_front(
        mixed, w, bias, conv_states, qsl, cidx, hinit, *, num_k_heads, num_v_heads, **kw
    ):
        if calls is not None:
            calls.append("front")
        out = _conv(
            mixed.transpose(0, 1),
            w,
            bias,
            conv_states=conv_states,
            cache_indices=cidx,
            query_start_loc=qsl,
            has_initial_state=hinit,
        ).transpose(0, 1)
        return _prepare(*_split(out))

    front.gdn_prefill_front = gdn_prefill_front
    front.gdn_prefill_gates = _gates
    front.gates_eligible = lambda *args: True
    front.eligible = lambda *args: ""
    front.supported = lambda **kw: True
    front.precompile = lambda *args: calls.append(("precompile", args))
    return front


@pytest.fixture
def stand_ins(monkeypatch):
    from sglang.kernels.ops.attention.fla import l2norm
    from sglang.srt.layers.attention.linear import gdn_backend

    monkeypatch.setattr(gdn_backend, "causal_conv1d_fn", _conv)
    monkeypatch.setattr(gdn_backend, "fused_gdn_gating", _gates)
    for name in ("is_cpu", "is_cuda", "is_hip", "is_xpu", "is_npu"):
        monkeypatch.setattr(gdn_backend, name, lambda: False)
    monkeypatch.setattr(
        l2norm, "gdn_prefill_qkv_prepare_fwd", lambda q, k, v: _prepare(q, k, v)
    )
    monkeypatch.setitem(
        sys.modules, "sgl_nvidia_mega.qwen3_5.sm103.gdn.prefill_front", _front_module()
    )


def _bits(t):
    return t.contiguous().view(torch.int32)


def test_the_extend_writes_sglangs_flashinfer_route_states_and_output(stand_ins):
    from sglang.srt.layers.attention.linear import gdn_backend

    case = _case()
    conv_s, ssm_s = case.conv.clone(), case.ssm.clone()
    conv_m, ssm_m = case.conv.clone(), case.ssm.clone()
    stock = gdn_backend.GDNAttnBackend.forward_extend(
        _backend(case, conv_s, ssm_s, _kernel()),
        case.layer,
        case.fb,
        case.mixed.clone(),
        case.a,
        case.b,
    )
    linear = _backend(case, conv_m, ssm_m, _kernel())
    got = prefill_fi.extend(
        case.layer,
        case.mixed.clone(),
        case.a,
        case.b,
        _view(case, conv_m, ssm_m),
        case.fb,
        linear,
    )
    live = torch.ones(sum(LENS), dtype=torch.bool)
    cu = case.meta.query_start_loc.tolist()
    live[cu[2] : cu[3]] = False
    assert stock.shape == got.shape == (1, sum(LENS), NV, D)
    assert torch.equal(_bits(stock[0, live]), _bits(got[0, live]))
    assert torch.equal(_bits(conv_s), _bits(conv_m))
    assert torch.equal(_bits(ssm_s), _bits(ssm_m))
    # every kind of write happened: windows, checkpoints, finals and the request rows
    assert not torch.equal(conv_m[8], case.conv[8])
    for slot in (1, 2, 3, 4, 5, 7):
        assert not torch.equal(ssm_m[slot], case.ssm[slot])
    assert torch.equal(ssm_m[6], case.ssm[6]) and torch.equal(ssm_m[0], 0 * ssm_m[0])


class _Launch:
    def __init__(self, name, fn, log):
        self.name, self.fn, self.log = name, fn, log

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.log.append(self.name)
            self.fn(grid, *args, **kwargs)

        return launch


def _rows(t):
    return t.reshape(t.shape[0], -1)


def _gather_rows(grid, states, indices, out, *args, **kw):
    for r, slot in enumerate(indices.tolist()):
        _rows(out)[r] = _rows(states)[slot] if slot > 0 else 0


def _scatter_rows(grid, states, indices, updates, *args, **kw):
    for r, slot in enumerate(indices.tolist()):
        _rows(states)[max(slot, 0)] = _rows(updates)[r] if slot > 0 else 0


def _track_states(grid, states, h, h_src, h_dst, final_src, final_dst, n_h, *a, **kw):
    for r in range(grid[0]):
        if r < n_h:
            _rows(states)[h_dst[r]] = _rows(h)[h_src[r]]
        else:
            j = r - n_h
            _rows(states)[final_dst[j]] = _rows(states)[final_src[j]]


def _track_conv(grid, conv, x, tokens, dst, *args, **kw):
    for r in range(grid[0]):
        for j, t in enumerate(tokens[r].tolist()):
            conv[dst[r], :, j] = x[t]


@pytest.fixture
def one_launch_copies(monkeypatch):
    from sglang.kernels.ops.mamba import gdn_extend_state_copies as copies

    log = []
    monkeypatch.setattr(
        copies,
        "current_arch",
        lambda: SimpleNamespace(is_sm_at_least=lambda major: True),
    )
    for name, fn in (
        ("_gather_pool_rows_kernel", _gather_rows),
        ("_scatter_pool_rows_kernel", _scatter_rows),
        ("_track_state_rows_kernel", _track_states),
        ("_track_conv_rows_kernel", _track_conv),
    ):
        monkeypatch.setattr(copies, name, _Launch(name, fn, log))
    return log


def test_the_extend_takes_sglangs_one_launch_state_copies_where_sglang_does(
    stand_ins, one_launch_copies
):
    from sglang.srt.layers.attention.linear import gdn_backend

    case = _case(seed=3)
    conv_s, ssm_s = case.conv.clone(), case.ssm.clone()
    conv_m, ssm_m = case.conv.clone(), case.ssm.clone()
    stock = gdn_backend.GDNAttnBackend.forward_extend(
        _backend(case, conv_s, ssm_s, _kernel()),
        case.layer,
        case.fb,
        case.mixed.clone(),
        case.a,
        case.b,
    )
    stock_launches = list(one_launch_copies)
    one_launch_copies.clear()
    got = prefill_fi.extend(
        case.layer,
        case.mixed.clone(),
        case.a,
        case.b,
        _view(case, conv_m, ssm_m),
        case.fb,
        _backend(case, conv_m, ssm_m, _kernel()),
    )
    assert (
        one_launch_copies
        == stock_launches
        == [
            "_track_conv_rows_kernel",
            "_gather_pool_rows_kernel",
            "_scatter_pool_rows_kernel",
            "_track_state_rows_kernel",
        ]
    )
    live = torch.ones(sum(LENS), dtype=torch.bool)
    cu = case.meta.query_start_loc.tolist()
    live[cu[2] : cu[3]] = False
    assert torch.equal(_bits(stock[0, live]), _bits(got[0, live]))
    assert torch.equal(_bits(conv_s), _bits(conv_m))
    assert torch.equal(_bits(ssm_s), _bits(ssm_m))
    assert not torch.equal(conv_m[8], case.conv[8])


def _flashinfer(calls):

    def prefill(
        *,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output,
        output_state,
        state_checkpoints,
        checkpoint_cu_starts,
        checkpoint_every_n_tokens,
        checkpoint_state_indices=None,
        **kw,
    ):
        calls.append(
            dict(
                kw,
                state_checkpoints=state_checkpoints,
                checkpoint_state_indices=checkpoint_state_indices,
            )
        )
        rows = None
        if checkpoint_state_indices is not None:
            rows = checkpoint_state_indices.tolist()
            buffer = state_checkpoints
            state_checkpoints = buffer.new_empty((len(rows), *buffer.shape[1:]))
        out, final = _run_prefill(
            q=q,
            k=k,
            v=v,
            alpha=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            output=None,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )
        if rows is not None:
            for entry, row in enumerate(rows):
                if row >= 0:
                    buffer[row] = state_checkpoints[entry]
        if output is not None:
            output.copy_(out)
            out = output
        output_state.copy_(final)
        return out, output_state

    return prefill


def _fi_kernel(calls):
    """FlashInferGDNKernel with SGLang's own _run_prefill on the stand-in FlashInfer."""
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = True
    kernel._prefill_needs_fp32_state = False
    kernel._prefill_fn = _flashinfer(calls)
    return kernel


def _is_flashinfers_entry(monkeypatch, kernel):
    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot

    monkeypatch.setattr(
        fi_track_snapshot, "flashinfer_entry", lambda: kernel._prefill_fn
    )


@pytest.fixture
def snapshot(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot

    routed = []
    monkeypatch.setattr(
        fi_track_snapshot,
        "routes_native",
        lambda *args: routed.append(args) or True,
    )
    state = fi_track_snapshot.TrackSnapshot()
    state.routed = routed
    return state


@pytest.mark.parametrize("copies", ["index-ops", "one-launch"])
def test_the_snapshot_extend_writes_sglangs_states_from_the_tracked_checkpoints_only(
    monkeypatch, request, stand_ins, snapshot, copies
):
    from sglang.srt.layers.attention.linear import gdn_backend

    launches = (
        request.getfixturevalue("one_launch_copies") if copies != "index-ops" else []
    )
    case = _case(seed=5)
    conv_s, ssm_s = case.conv.clone(), case.ssm.clone()
    conv_m, ssm_m = case.conv.clone(), case.ssm.clone()
    stock_calls, calls = [], []
    stock = gdn_backend.GDNAttnBackend.forward_extend(
        _backend(case, conv_s, ssm_s, _fi_kernel(stock_calls)),
        case.layer,
        case.fb,
        case.mixed.clone(),
        case.a,
        case.b,
    )
    stock_launches = list(launches)
    launches.clear()
    kernel = _fi_kernel(calls)
    _is_flashinfers_entry(monkeypatch, kernel)
    got = prefill_fi.extend(
        case.layer,
        case.mixed.clone(),
        case.a,
        case.b,
        _view(case, conv_m, ssm_m),
        case.fb,
        _backend(case, conv_m, ssm_m, kernel),
        snapshot=snapshot,
    )
    live = torch.ones(sum(LENS), dtype=torch.bool)
    cu = case.meta.query_start_loc.tolist()
    live[cu[2] : cu[3]] = False
    assert torch.equal(_bits(stock[0, live]), _bits(got[0, live]))
    assert torch.equal(_bits(conv_s), _bits(conv_m))
    assert torch.equal(_bits(ssm_s), _bits(ssm_m))
    # the tracked checkpoints and the finals were written
    for slot in (1, 2, 4):
        assert not torch.equal(ssm_m[slot], case.ssm[slot])
    assert launches == stock_launches
    # one FlashInfer call each: SGLang's arguments, with the destination rows
    ((want,), (call,)) = stock_calls, calls
    assert want.pop("use_cp") == "auto" and call.pop("use_cp") is False
    assert want.pop("checkpoint_state_indices") is None
    rows = call.pop("checkpoint_state_indices")
    assert want.pop("state_checkpoints").shape[0] == case.meta.num_state_checkpoints
    assert call.pop("state_checkpoints").shape[0] == 2
    assert call == want
    # request 0's entry 1 into row 0, request 1's entry (2) into row 1
    assert rows.dtype == torch.int32
    assert rows.tolist() == [-1, 0, 1, -1, -1, -1]
    assert snapshot.routed == [(len(LENS), NV, torch.device("cpu"))]


def test_the_snapshot_keeps_sglangs_checkpoints_where_flashinfer_runs_context_parallel(
    monkeypatch, stand_ins
):
    from sglang.srt.layers.attention.linear import gdn_backend

    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot

    monkeypatch.setattr(fi_track_snapshot, "routes_native", lambda *args: False)
    for untracked in (False, True):
        case = _case(seed=6)
        if untracked:
            case.meta.track_ssm_h_src = case.meta.track_ssm_h_src[:0]
            case.meta.track_ssm_h_dst = case.meta.track_ssm_h_dst[:0]
        conv_s, ssm_s = case.conv.clone(), case.ssm.clone()
        conv_m, ssm_m = case.conv.clone(), case.ssm.clone()
        stock_calls, calls = [], []
        gdn_backend.GDNAttnBackend.forward_extend(
            _backend(case, conv_s, ssm_s, _fi_kernel(stock_calls)),
            case.layer,
            case.fb,
            case.mixed.clone(),
            case.a,
            case.b,
        )
        prefill_fi.extend(
            case.layer,
            case.mixed.clone(),
            case.a,
            case.b,
            _view(case, conv_m, ssm_m),
            case.fb,
            _backend(case, conv_m, ssm_m, _fi_kernel(calls)),
            snapshot=fi_track_snapshot.TrackSnapshot(),
        )
        assert torch.equal(_bits(ssm_s), _bits(ssm_m))
        assert calls[0]["use_cp"] == "auto"
        assert calls[0]["checkpoint_state_indices"] is None
        assert calls[0]["state_checkpoints"].shape[0] == case.meta.num_state_checkpoints


@pytest.mark.parametrize("slot", [False, True], ids=["front", "front-and-snapshot"])
@pytest.mark.parametrize("checkpoints", [False, True])
def test_the_flashinfer_call_of_an_untracked_step_reads_no_track_row(
    stand_ins, snapshot, checkpoints, slot
):
    case = _case(seed=7)
    meta = case.meta
    plan = dict(
        state_checkpoint_cu_starts=(
            meta.state_checkpoint_cu_starts if checkpoints else None
        ),
        num_state_checkpoints=meta.num_state_checkpoints if checkpoints else 0,
        state_checkpoint_every_n_tokens=EVERY if checkpoints else 0,
    )
    q, k, v = _split(case.mixed)
    g, beta = _gates(case.layer.A_log, case.a, case.b, case.layer.dt_bias)
    ssm_s, ssm_m = case.ssm.clone(), case.ssm.clone()
    stock_calls, calls = [], []
    o_s, _, h_s = _fi_kernel(stock_calls).extend(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g,
        beta,
        ssm_states=ssm_s,
        cache_indices=meta.mamba_cache_indices,
        query_start_loc=meta.query_start_loc,
        **plan,
    )
    view = SimpleNamespace(
        state_indices=meta.mamba_cache_indices,
        query_start_loc=meta.query_start_loc,
        **plan,
    )
    if slot:
        view.has_track, view.track_ssm_h_src = False, None
    o_m, h_m, h_src = prefill_fi._flashinfer_extend(
        _fi_kernel(calls),
        *_prepare(q, k, v),
        g,
        beta,
        ssm_m,
        view,
        None,
        None,
        snapshot if slot else None,
    )
    assert h_src is None
    assert torch.equal(_bits(o_s), _bits(o_m))
    assert torch.equal(_bits(ssm_s), _bits(ssm_m))
    assert (h_s is None) == (h_m is None) == (not checkpoints)
    if checkpoints:
        assert torch.equal(_bits(h_s), _bits(h_m))
    ((want,), (call,)) = stock_calls, calls
    assert want["use_cp"] == call["use_cp"] == "auto"
    assert want["checkpoint_state_indices"] is call["checkpoint_state_indices"] is None
    assert snapshot.routed == []


TRACK_SEQLENS = [7, 0, 3, 3]
TRACK_MASK = [True, False, True, True]
STATE_SLOTS = [3, 7, 6, 5]
TRACK_SLOTS = [2, 8, 4, 1]
LAYERS = 2


@pytest.fixture
def chunk_size():
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(_mamba_cache_chunk_size=EVERY)
    override.install()
    yield EVERY
    override.restore()


def _sglang_batch():
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode

    cu = [0]
    for n in LENS:
        cu.append(cu[-1] + n)
    fb = SimpleNamespace(
        batch_size=len(LENS),
        forward_mode=ForwardMode.EXTEND,
        logical_forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.arange(1, len(LENS) + 1),
        extend_start_loc=torch.tensor(cu[:-1], dtype=torch.int32),
        extend_seq_lens=torch.tensor(LENS, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(PREFIX, dtype=torch.int32),
        extend_seq_lens_cpu=LENS,
        extend_prefix_lens_cpu=PREFIX,
        mamba_track_mask=torch.tensor(TRACK_MASK),
        mamba_track_seqlens=torch.tensor(TRACK_SEQLENS),
        mamba_track_indices=torch.tensor(TRACK_SLOTS),
        mamba_prefill_track_mask_cpu=TRACK_MASK,
        mamba_track_seqlens_cpu=TRACK_SEQLENS,
        multi_item_delimiter_indices=None,
        tbo_parent_token_range=None,
        spec_info=None,
        out_cache_loc=torch.zeros(sum(LENS), dtype=torch.int64),
        global_num_token_non_padded_cpu=sum(LENS),
    )
    fb.mamba_track_aligned_lens = lambda: ForwardBatch.mamba_track_aligned_lens(fb)
    return fb


def _sglang_backend(fb, conv, ssm, kernel):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear import gdn_backend

    dispatcher = object.__new__(gdn_backend.GDNKernelDispatcher)
    dispatcher.extend_kernel = kernel
    linear = object.__new__(gdn_backend.GDNAttnBackend)
    linear.__dict__.update(
        device="cpu",
        _mamba_chunk_size=EVERY,
        conv_states_shape=conv.shape[2:],
        state_pool=None,
        enable_mis=False,
        _state_ids=SimpleNamespace(ids=lambda ids: ids),
        # request-table row 0 pads; rows 1.. are the batch's requests
        req_to_token_pool=SimpleNamespace(
            get_mamba_indices=lambda rows: torch.tensor(
                [0, *STATE_SLOTS], dtype=torch.int32
            )[rows]
        ),
        kernel_dispatcher=dispatcher,
        state_layer_cache=lambda i: SimpleNamespace(conv=(conv[i],), temporal=ssm[i]),
    )
    linear.init_forward_metadata(fb)
    hybrid = object.__new__(HybridLinearAttnBackend)
    hybrid.__dict__.update(
        full_attn_layers=[],
        full_attn_backend=None,
        linear_attn_backend=linear,
        req_to_token_pool=SimpleNamespace(
            mamba_map={i: i for i in range(LAYERS)},
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(conv=(conv,), temporal=ssm)
            ),
        ),
    )
    return hybrid


def _routing(monkeypatch, asked):
    helper = ModuleType("flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper")
    helper.should_use_cp_host = lambda work, *args, **kw: asked.append(work) or False
    utils = ModuleType("flashinfer.utils")
    utils.get_compute_capability = lambda device: (10, 3)
    utils.get_device_sm_count = lambda device: 148
    utils.get_device_name = lambda device: "GB300"
    for name in (
        "flashinfer",
        "flashinfer.gdn_kernels",
        "flashinfer.gdn_kernels.delta_rule_dsl",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)


def _measured_tp1_pp2(monkeypatch, binding):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    from sgl_nvidia_mega.qwen3_5 import runtime, spec, topology
    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    # one rank: nothing to agree on across TP
    monkeypatch.setattr(plugin_binding, "megakernel_agree", lambda *a, **kw: None)
    front = _front_module([])
    front.warm = lambda attn, heads: None
    monkeypatch.setitem(
        sys.modules, "sgl_nvidia_mega.qwen3_5.sm103.gdn.prefill_front", front
    )
    monkeypatch.setattr(
        "sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer", _LinearLayer
    )
    monkeypatch.setattr(prefill.parity, "ENABLED", False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fi_track_snapshot, "flashinfer_reasons", lambda: [])
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(spec, "prefill_body_offered", spec._offers_prefill_body)
    monkeypatch.setattr(runtime, "MEASURED_KEY", None)
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    parallel = SimpleNamespace(
        tp_size=1,
        attn_tp_size=1,
        attn_dp_size=1,
        moe_ep_size=1,
        pp_size=2,
        tp_group=SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr("sglang.srt.runtime_context.get_parallel", lambda: parallel)
    monkeypatch.setattr(topology, "device_sku", lambda: topology.GB300)
    model = SimpleNamespace(
        model=SimpleNamespace(
            start_layer=0,
            end_layer=LAYERS,
            layers=[_LinearLayer() for _ in range(LAYERS)],
            flashinfer_mnnvl_cutedsl_fusion=None,
        )
    )
    built = runtime.build(model, SimpleNamespace(tokens_per_request=1, binding=binding))
    assert runtime.MEASURED_KEY == ("p_tp1_pp2", topology.GB300)
    return built


@pytest.mark.parametrize("inference", [False, True], ids=["no-grad", "inference-mode"])
def test_the_measured_tp1_pp2_prefill_runs_the_snapshot_on_sglangs_own_step(
    monkeypatch, caplog, stand_ins, chunk_size, inference
):
    from sglang.srt.kernel_plugins.nvidia.eagle_binding import EagleBinding
    from sglang.srt.layers.radix_linear_attention import (
        _linear_attention_with_output_impl,
    )
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.model_executor.runner_utils.forward_batch import (
        set_forward_batch,
    )

    from sgl_nvidia_mega.qwen3_5.gdn import fi_track_snapshot

    asked = []
    _routing(monkeypatch, asked)
    binding = EagleBinding(model=torch.nn.Module())
    with caplog.at_level("INFO"):
        built = _measured_tp1_pp2(monkeypatch, binding)
    assert built.summary() == (
        "prefill TP1/PP2; gdn_prefill_front, attn_prefill_qk_rows built (prefill body); "
        "gdn_prefill_track_snapshot built"
    )
    assert isinstance(built.snapshot, fi_track_snapshot.TrackSnapshot)
    g = torch.Generator().manual_seed(11)
    conv = torch.randn(LAYERS, SLOTS, QKV, W - 1, generator=g).bfloat16()
    ssm = torch.randn(LAYERS, SLOTS, NV, D, D, generator=g)
    steps = []
    for i in range(LAYERS):
        case = _case(seed=20 + i)
        layer = SimpleNamespace(**(case.layer.__dict__ | {"layer_id": i}))
        steps.append((layer, *(t.bfloat16() for t in (case.mixed, case.a, case.b))))
    mode = torch.inference_mode if inference else torch.no_grad
    runs = {}
    with caplog.at_level("INFO"):
        for arm in ("stock", "mega"):
            calls, pools, outs = [], (conv.clone(), ssm.clone()), []
            fb = _sglang_batch()
            with mode():
                kernel = _fi_kernel(calls)
                _is_flashinfers_entry(monkeypatch, kernel)
                backend = _sglang_backend(fb, *pools, kernel)
                with (
                    forward_context(ForwardContext(attn_backend=backend)),
                    set_forward_batch(fb),
                ):
                    for layer, mixed, a, b in steps:
                        out = torch.full((1, sum(LENS), NV, D), 7.0).bfloat16()
                        if arm == "stock":
                            _linear_attention_with_output_impl(
                                mixed.clone(), a, b, out, layer, fb
                            )
                        else:
                            built._core(layer, mixed.clone(), a, b, out)
                        outs.append(out)
            meta = backend.linear_attn_backend.forward_metadata
            runs[arm] = SimpleNamespace(calls=calls, pools=pools, outs=outs, meta=meta)
    stock, mega = runs["stock"], runs["mega"]
    assert mega.meta.track_ssm_h_src.tolist() == [1, 3]
    assert mega.meta.track_ssm_h_src.is_inference() is inference
    assert mega.meta.state_checkpoint_cu_starts.tolist() == [0, 2, 3, 5, 6]
    assert mega.meta.num_state_checkpoints == 6
    for slot in ("gdn_prefill_front", "gdn_prefill_track_snapshot"):
        assert f"slot {slot} uses stock" not in caplog.text
    assert "SGLang's GDN core runs this step" not in caplog.text
    assert asked == [len(LENS) * NV]
    # each layer's FlashInfer call: SGLang's all-row checkpoints vs the tracked rows'
    assert len(stock.calls) == len(mega.calls) == LAYERS
    for want, got in zip(stock.calls, mega.calls):
        assert want["use_cp"] == "auto" and want["checkpoint_state_indices"] is None
        assert want["state_checkpoints"].shape[0] == 6
        assert got["use_cp"] is False and got["state_checkpoints"].shape[0] == 2
        assert got["checkpoint_state_indices"].tolist() == [-1, 0, -1, 1, -1, -1]
    # the step's rows are built once and read by every layer
    rows = {id(call["checkpoint_state_indices"]) for call in mega.calls}
    assert len(rows) == 1
    for want, got in zip(stock.outs, mega.outs):
        assert torch.equal(_bits(want), _bits(got)) and not (got == 7.0).all()
    assert torch.equal(
        stock.pools[0].view(torch.int16), mega.pools[0].view(torch.int16)
    )
    assert torch.equal(_bits(stock.pools[1]), _bits(mega.pools[1]))
    # the tracked states (slots 2, 4) and the final-state copy (slot 1) were written
    for slot in TRACK_SLOTS[:1] + TRACK_SLOTS[2:]:
        assert not torch.equal(mega.pools[1][:, slot], ssm[:, slot])


def _gdn(case, seen, qkvz, ba, z):
    return SimpleNamespace(
        attn=case.layer,
        num_k_heads=NK,
        num_v_heads=NV,
        attn_tp_size=1,
        head_k_dim=D,
        head_v_dim=D,
        _forward_input_proj=lambda x: (qkvz, ba),
        norm=lambda core, z_: seen.setdefault("norm", (core, z_))[0] + 1,
        out_proj=lambda x: (seen.setdefault("out", x) * 3, None),
    )


def test_the_step_projects_and_norms_as_the_gdn_layer(monkeypatch):
    case = _case(seed=1)
    tokens = sum(LENS)
    qkvz = torch.randn(tokens, QKV + NV * D)
    ba = torch.randn(tokens, 2 * NV)
    z = torch.randn(tokens, NV, D)
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.triton_gdn_fused_proj."
        "qwen3_5_gdn_prefill_projection_views",
        lambda qkvz_, ba_, nk, nv, dk, dv: (
            (case.mixed, z, case.b, case.a)
            if (qkvz_ is qkvz and (nk, nv, dk, dv) == (NK, NV, D, D))
            else pytest.fail("projection views")
        ),
    )
    seen = {}

    def run_core(attn, mixed, a, b, output):
        assert (attn, mixed, a, b) == (case.layer, case.mixed, case.a, case.b)
        assert output.shape == (1, tokens, NV, D)
        output.copy_(torch.arange(output.numel()).view(output.shape))

    out = prefill_fi.step(
        _gdn(case, seen, qkvz, ba, z), torch.zeros(tokens, 8), run_core
    )
    core, z_seen = seen["norm"]
    assert core.shape == (tokens * NV, D) and z_seen is z
    assert torch.equal(core.flatten(), torch.arange(core.numel()).to(core.dtype))
    assert seen["out"].shape == (tokens, NV * D)
    assert torch.equal(out, seen["out"] * 3)


PAD = 3


def _bf16_view(case, conv, ssm):
    view = _view(case, conv, ssm)
    view.state = SimpleNamespace(
        conv=(torch.zeros(1, SLOTS, QKV, 3, dtype=torch.bfloat16),),
        temporal=ssm[None],
    )
    return view


def _padded(case):
    """The case's batch with PAD physical rows past its real tokens."""
    pad = lambda t: torch.cat([t, torch.randn(PAD, *t.shape[1:])])
    fb = SimpleNamespace(**case.fb.__dict__, out_cache_loc=torch.zeros(sum(LENS) + PAD))
    return pad(case.mixed), pad(case.a), pad(case.b), fb


@pytest.fixture
def live_batch(monkeypatch):
    from sglang.srt.layers import radix_linear_attention
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.model_executor.runner_utils import forward_batch

    def bind(fb, linear):
        monkeypatch.setattr(forward_batch, "get_forward_batch", lambda: fb)
        monkeypatch.setattr(prefill_fi, "linear_backend", lambda: linear)
        monkeypatch.setattr(
            radix_linear_attention,
            "get_attn_backend",
            lambda: SimpleNamespace(
                forward=lambda layer, forward_batch, mixed_qkv, a, b, **kw: (
                    gdn_backend.GDNAttnBackend.forward_extend(
                        linear, layer, forward_batch, mixed_qkv, a, b, **kw
                    )
                )
            ),
        )

    return bind


def test_the_core_runs_the_front_on_the_real_rows_and_zeros_the_padding(
    stand_ins, live_batch
):
    from sglang.srt.layers.radix_linear_attention import (
        _linear_attention_with_output_impl,
    )

    case = _case(seed=4)
    mixed, a, b, fb = _padded(case)
    rows = sum(LENS) + PAD
    conv_s, ssm_s = case.conv.clone(), case.ssm.clone()
    stock = torch.full((1, rows, NV, D), 7.0)
    live_batch(fb, _backend(case, conv_s, ssm_s, _kernel()))
    _linear_attention_with_output_impl(mixed, a, b, stock, case.layer, fb)
    conv_m, ssm_m = case.conv.clone(), case.ssm.clone()
    got = torch.full((1, rows, NV, D), 7.0)
    linear = _backend(case, conv_m, ssm_m, _kernel())
    live_batch(fb, linear)
    binding = SimpleNamespace(prefill_view=lambda: _bf16_view(case, conv_m, ssm_m))
    calls = []
    sys.modules["sgl_nvidia_mega.qwen3_5.sm103.gdn.prefill_front"] = _front_module(
        calls
    )
    prefill_fi.core(case.layer, mixed, a, b, got, binding, set())
    assert calls == ["front"]
    live = torch.ones(rows, dtype=torch.bool)
    cu = case.meta.query_start_loc.tolist()
    live[cu[2] : cu[3]] = False
    assert torch.equal(_bits(stock[0, live]), _bits(got[0, live]))
    assert torch.equal(got[0, sum(LENS) :], torch.zeros(PAD, NV, D))
    assert torch.equal(_bits(conv_s), _bits(conv_m))
    assert torch.equal(_bits(ssm_s), _bits(ssm_m))


@pytest.mark.parametrize(
    "change, why",
    [
        pytest.param(
            {"view": {"is_graph": True}},
            "linear extend inside the graph or multi-item layout",
            id="step",
        ),
        pytest.param({"eligible": "mixed_qkv layout"}, "layer 0's tensors", id="layer"),
    ],
)
def test_the_core_runs_sglangs_core_where_the_front_does_not_reproduce_the_step(
    monkeypatch, stand_ins, live_batch, caplog, change, why
):
    from sglang.srt.layers import radix_linear_attention

    case = _case(seed=5)
    mixed, a, b, fb = _padded(case)
    conv, ssm = case.conv.clone(), case.ssm.clone()
    view = _bf16_view(case, conv, ssm)
    view.__dict__.update(change.get("view", {}))
    live_batch(fb, _backend(case, conv, ssm, _kernel()))
    calls = []
    front = _front_module(calls)
    front.eligible = lambda *args: change.get("eligible", "")
    monkeypatch.setitem(
        sys.modules, "sgl_nvidia_mega.qwen3_5.sm103.gdn.prefill_front", front
    )
    stock = []
    monkeypatch.setattr(
        radix_linear_attention,
        "_linear_attention_with_output_impl",
        lambda *args: stock.append(args),
    )
    declined = set()
    out = torch.empty(1, sum(LENS) + PAD, NV, D)
    binding = SimpleNamespace(prefill_view=lambda: view)
    with caplog.at_level("INFO"):
        for _ in range(2):
            prefill_fi.core(case.layer, mixed, a, b, out, binding, declined)
    # SGLang's core takes the whole physical batch, before the front wrote anything
    assert calls == [] and torch.equal(conv, case.conv) and torch.equal(ssm, case.ssm)
    assert [args[:5] for args in stock] == [(mixed, a, b, out, case.layer)] * 2
    assert declined == {why}
    assert caplog.text.count(f"SGLang's GDN core runs this step ({why})") == 1


def test_the_core_of_a_batch_without_real_rows_is_sglangs(
    monkeypatch, stand_ins, live_batch
):
    from sglang.srt.layers import radix_linear_attention

    case = _case()
    mixed, a, b, fb = _padded(case)
    fb.global_num_token_non_padded_cpu = 0
    live_batch(fb, None)
    stock = []
    monkeypatch.setattr(
        radix_linear_attention,
        "_linear_attention_with_output_impl",
        lambda *args: stock.append(args),
    )
    binding = SimpleNamespace(prefill_view=lambda: pytest.fail("read the view"))
    out = torch.empty(1, sum(LENS) + PAD, NV, D)
    prefill_fi.core(case.layer, mixed, a, b, out, binding, set())
    assert len(stock) == 1


def test_the_core_without_a_real_row_count_runs_the_backend_on_every_row(
    monkeypatch, stand_ins, live_batch
):
    from sglang.srt.layers import radix_linear_attention

    case = _case()
    mixed, a, b, fb = _padded(case)
    fb.global_num_token_non_padded_cpu = None
    live_batch(fb, None)
    monkeypatch.setattr(prefill_fi, "unsupported", lambda view, linear: "a reason")
    got = torch.randn(1, mixed.shape[0], NV, D)
    seen = []
    monkeypatch.setattr(
        radix_linear_attention,
        "get_attn_backend",
        lambda: SimpleNamespace(
            forward=lambda **kw: seen.append(kw["mixed_qkv"]) or got.clone()
        ),
    )
    monkeypatch.setattr(
        radix_linear_attention,
        "_linear_attention_with_output_impl",
        lambda *args: pytest.fail("needs a real-row count"),
    )
    binding = SimpleNamespace(prefill_view=lambda: "view")
    out = torch.empty(1, mixed.shape[0], NV, D)
    prefill_fi.core(case.layer, mixed, a, b, out, binding, set())
    assert len(seen) == 1 and seen[0] is mixed
    assert torch.equal(out, got)


def _unsupported_case(**changes):
    case = _case()
    view = _view(case, case.conv, case.ssm)
    view.__dict__.update(changes.pop("view", {}))
    linear = SimpleNamespace(
        kernel_dispatcher=SimpleNamespace(
            extend_kernel=changes.pop("kernel", None) or _kernel()
        )
    )
    return prefill_fi.unsupported(view, linear)


@pytest.mark.parametrize(
    "changes, why",
    [
        pytest.param({}, "", id="supported"),
        pytest.param(
            {"kernel": SimpleNamespace()},
            "not the FlashInfer prefill kernel",
            id="triton-kernel",
        ),
        pytest.param(
            {"view": {"is_graph": True}},
            "linear extend inside the graph or multi-item layout",
            id="graph",
        ),
        pytest.param(
            {"view": {"has_mis": True}},
            "linear extend inside the graph or multi-item layout",
            id="multi-item",
        ),
        pytest.param(
            {
                "view": {
                    "state": SimpleNamespace(
                        conv=(torch.zeros(1, SLOTS, QKV, 6)[..., ::2],),
                        temporal=torch.zeros(1, SLOTS, NV, D, D),
                    )
                }
            },
            "strided committed state pool",
            id="strided-conv",
        ),
        pytest.param(
            {"view": {"track_ssm_recompute_dst": torch.tensor([3])}},
            "recomputed checkpoint layout",
            id="recompute",
        ),
        pytest.param(
            {"view": {"state_indices": torch.tensor(INDICES, dtype=torch.int64)}},
            "index dtype / layout",
            id="index-dtype",
        ),
        pytest.param(
            {
                "view": {
                    "state": SimpleNamespace(
                        conv=(torch.zeros(1, SLOTS, QKV, 3, dtype=torch.float16),),
                        temporal=torch.zeros(1, SLOTS, NV, D, D),
                    )
                }
            },
            "conv state dtype",
            id="conv-dtype",
        ),
    ],
)
def test_the_step_refuses_what_it_does_not_reproduce(changes, why):
    if why != "conv state dtype" and "state" not in changes.get("view", {}):
        changes.setdefault("view", {})["state"] = SimpleNamespace(
            conv=(torch.zeros(1, SLOTS, QKV, 3, dtype=torch.bfloat16),),
            temporal=torch.zeros(1, SLOTS, NV, D, D),
        )
    assert _unsupported_case(**changes) == why


class _LinearLayer:
    def __init__(self, attn_tp=1, **attn):
        base = _case().layer.__dict__ | {
            "conv_weights": torch.zeros(QKV, 1, W, dtype=torch.bfloat16)
        }
        self.linear_attn = SimpleNamespace(
            attn=SimpleNamespace(**(base | attn)),
            attn_tp_size=attn_tp,
            num_v_heads=NV,
            out_proj=None,
            norm=None,
        )


def _prefill(monkeypatch, layers, calls, offered=True, binding=None, reads=False):
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    from sgl_nvidia_mega.qwen3_5 import spec
    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    if not reads:
        for name in ("_run_prefill", "_prefill_state_dtype"):
            if not hasattr(FlashInferGDNKernel, name):
                monkeypatch.setattr(FlashInferGDNKernel, name, object(), raising=False)

    front = _front_module(calls)
    front.warm = lambda attn, heads: calls.append(("warm", attn.layer_id, heads))
    monkeypatch.setitem(
        sys.modules, "sgl_nvidia_mega.qwen3_5.sm103.gdn.prefill_front", front
    )
    monkeypatch.setattr(
        "sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer", _LinearLayer
    )
    monkeypatch.setattr(prefill.parity, "ENABLED", False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(spec, "prefill_body_offered", lambda: offered)
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=len(layers), layers=layers)
    )
    return prefill.Qwen35Prefill(
        model,
        SimpleNamespace(
            tokens_per_request=7,
            binding=binding or SimpleNamespace(prefill_view=lambda: "view"),
        ),
        False,
        label="TP1/PP2",
        split_conv=False,
        front=True,
    )


def test_the_build_compiles_and_runs_the_front_once_a_layer_shape(monkeypatch):
    calls = []
    built = _prefill(monkeypatch, [_LinearLayer() for _ in range(3)], calls)
    assert built.front
    assert built.summary() == (
        "prefill TP1/PP2; gdn_prefill_front built (prefill body)"
    )
    # the CUDA kernel, then the Triton launches on one of the shape's layers
    assert calls == [("precompile", (0, NK, NV, False, "silu")), ("warm", 0, NV)]


@pytest.mark.parametrize(
    "layer, why",
    [
        pytest.param({"attn_tp": 2}, "layer 1: attention TP", id="attention-tp"),
        pytest.param(
            {"conv_weights": torch.zeros(QKV, 1, W)},
            "layer 1: conv weight layout",
            id="fp32-conv-weight",
        ),
        pytest.param({"num_q_heads": 2}, "layer 1: layer shape", id="shape"),
        pytest.param(
            {"offered": False}, "the spec offers no prefill body here", id="no-body"
        ),
        pytest.param(
            {"binding": SimpleNamespace()}, "no linear prefill view", id="no-view"
        ),
    ],
)
def test_the_build_keeps_sglangs_forward_where_the_front_cannot_run(
    monkeypatch, caplog, layer, why
):
    calls = []
    attn_tp = layer.pop("attn_tp", 1)
    offered = layer.pop("offered", True)
    binding = layer.pop("binding", None)
    layers = [_LinearLayer(), _LinearLayer(attn_tp, **layer)]
    with caplog.at_level("INFO"):
        built = _prefill(monkeypatch, layers, calls, offered, binding)
    assert not built.front and calls == []
    assert built.summary() == "prefill TP1/PP2; SGLang's forward"
    assert f"gdn_prefill_front uses stock ({why})" in caplog.text
    fb = SimpleNamespace(
        mm_inputs=None,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    assert built.plan(torch.zeros(4), fb) is None
    assert built.plan_body(torch.zeros(4), fb) is None


def test_the_build_needs_sglangs_flashinfer_prefill_reads(monkeypatch, caplog):
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    monkeypatch.delattr(FlashInferGDNKernel, "_run_prefill", raising=False)
    calls = []
    with caplog.at_level("INFO"):
        built = _prefill(monkeypatch, [_LinearLayer()], calls, reads=True)
    assert not built.front and calls == []
    assert "uses stock (no FlashInferGDNKernel._run_prefill)" in caplog.text


def test_the_build_on_the_installed_sglang_runs_the_front_or_names_the_read(
    monkeypatch, caplog
):
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    missing = [
        name
        for name in ("_run_prefill", "_prefill_state_dtype")
        if not hasattr(FlashInferGDNKernel, name)
    ]
    calls = []
    with caplog.at_level("INFO"):
        built = _prefill(monkeypatch, [_LinearLayer()], calls, reads=True)
    if missing:
        assert not built.front and calls == []
        assert f"uses stock (no FlashInferGDNKernel.{missing[0]})" in caplog.text
    else:
        assert built.front and calls


def _extend_batch(**changes):
    fields = dict(
        input_ids=torch.zeros(5),
        mm_inputs=None,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    return SimpleNamespace(**(fields | changes))


@pytest.mark.parametrize(
    "args, kwargs, fb, planned",
    [
        pytest.param((torch.zeros(4),), {}, {}, 4, id="text-extend"),
        pytest.param((None,), {"pp_proxy_tensors": "proxy"}, {}, 5, id="later-stage"),
        pytest.param(
            (torch.zeros(4),), {"input_embeds": torch.zeros(4, 8)}, {}, 4, id="embeds"
        ),
        pytest.param(
            (torch.zeros(4),),
            {"input_deepstack_embeds": torch.zeros(4, 8)},
            {},
            None,
            id="deepstack",
        ),
        pytest.param((torch.zeros(4),), {"other": 1}, {}, None, id="other-input"),
        pytest.param(
            (torch.zeros(4),), {}, {"mm_inputs": ["image"]}, None, id="multimodal"
        ),
        pytest.param(
            (torch.zeros(4),),
            {},
            {
                "forward_mode": SimpleNamespace(
                    is_extend_without_speculative=lambda: False
                )
            },
            None,
            id="not-a-plain-extend",
        ),
    ],
)
def test_the_prefill_body_plans_a_text_extend(monkeypatch, args, kwargs, fb, planned):
    built = _prefill(monkeypatch, [_LinearLayer()], [])
    batch = _extend_batch(**fb)
    want = None if planned is None else Step(mode="prefill", rows=planned)
    assert built.plan_body(*args, batch, **kwargs) == want
    # the whole-model forward leaves the step to the body
    assert built.plan(torch.zeros(4), batch) is None


def test_the_prefill_body_runs_the_front_body_and_logs_once(monkeypatch, caplog):
    built = _prefill(monkeypatch, [_LinearLayer()], [])
    ran = []
    monkeypatch.setattr(
        built, "_body", lambda *args: ran.append(args) or "hidden", raising=False
    )
    step = Step(mode="prefill", rows=4, view="view")
    original = lambda *args, **kw: pytest.fail("ran the stock body")
    with caplog.at_level("INFO"):
        for _ in range(2):
            got = built.forward_body(
                step, original, "ids", "pos", "fb", None, "proxy", None
            )
            assert got == "hidden"
    assert ran == [("ids", "pos", "fb", None, "proxy")] * 2
    assert caplog.text.count("slot gdn_prefill_front runs at 4 rows") == 1
    # the GDN layers' cores stay graph breaks
    assert built.linear_prefill_graph_seqs(object()) is None
    assert built.refill_prefill_graph(object(), object()) is None


def _stage_forward(input_embeds, pp_proxy_tensors):
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    if pp_proxy_tensors is None:
        hidden, residual = input_embeds * 0.5, input_embeds
    else:
        hidden, residual = (
            pp_proxy_tensors["hidden_states"],
            pp_proxy_tensors["residual"],
        )
    for scale in (2.0, 3.0):
        residual.add_(hidden * scale + 1)
        hidden = residual * 0.25
    return PPProxyTensors({"hidden_states": hidden, "residual": residual})


@pytest.mark.parametrize("stage", ["first", "later"])
def test_the_parity_step_runs_sglangs_forward_on_copies_of_the_stage_inputs(
    monkeypatch, stage
):
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    from sgl_nvidia_mega.qwen3_5.runtime import prefill

    built = _prefill(monkeypatch, [_LinearLayer()], [])
    built.parity = prefill.PrefillParity()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    embeds = torch.linspace(-1, 1, 8).reshape(2, 4)
    if stage == "first":
        inputs = dict(input_embeds=embeds, pp_proxy_tensors=None)
    else:
        proxy = {"hidden_states": embeds * 3, "residual": embeds + 1}
        inputs = dict(input_embeds=None, pp_proxy_tensors=PPProxyTensors(proxy))
    pristine = {name: prefill.stage_input_copy(v) for name, v in inputs.items()}
    seen = []

    def original(ids, pos, fb, *, input_embeds, pp_proxy_tensors, **kw):
        seen.append(("stock", input_embeds, pp_proxy_tensors))
        return _stage_forward(input_embeds, pp_proxy_tensors)

    def body(ids, pos, fb, input_embeds, pp_proxy_tensors):
        seen.append(("body", input_embeds, pp_proxy_tensors))
        return _stage_forward(input_embeds, pp_proxy_tensors)

    monkeypatch.setattr(built, "_body", body, raising=False)
    view = SimpleNamespace(
        state=SimpleNamespace(
            conv=(torch.zeros(1, 3, 2, 2),), temporal=torch.zeros(1, 3, 2, 2, 2)
        ),
        state_indices=torch.tensor([1], dtype=torch.int32),
        has_track=False,
    )
    got = built.forward_body(
        Step(mode="prefill", rows=2, view=view),
        original,
        None,
        "pos",
        "fb",
        inputs["input_embeds"],
        inputs["pp_proxy_tensors"],
        None,
    )
    want = _stage_forward(**pristine)
    for name in ("hidden_states", "residual"):
        assert torch.equal(_bits(got[name]), _bits(want[name]))
    # the body read the step's own inputs, SGLang's forward copies of them
    ((who_s, embeds_s, proxy_s), (who_b, embeds_b, proxy_b)) = seen
    assert (who_s, who_b) == ("stock", "body")
    assert embeds_b is inputs["input_embeds"] and proxy_b is inputs["pp_proxy_tensors"]
    if stage == "first":
        assert embeds_s is not embeds and proxy_s is None
    else:
        assert embeds_s is None and isinstance(proxy_s, PPProxyTensors)
        assert all(proxy_s[n] is not inputs["pp_proxy_tensors"][n] for n in proxy)


def test_the_core_is_a_graph_break_that_runs_the_slots_core(monkeypatch):
    binding = SimpleNamespace(prefill_view=lambda: "view")
    built = _prefill(monkeypatch, [_LinearLayer()], [], binding=binding)
    cores = []
    monkeypatch.setattr(prefill_fi, "core", lambda *args: cores.append(args))
    # outside a capture the break runs at once (SGLang's eager_on_graph)
    assert built._core("attn", "mixed", "a", "b", "out") is None
    assert cores == [
        ("attn", "mixed", "a", "b", "out", binding, built._declined, built.snapshot)
    ]
    assert built._core.__wrapped__ == built._run_core


def test_the_body_publishes_its_batch_for_the_cores(monkeypatch):
    from sglang.srt.model_executor.runner_utils.forward_batch import (
        get_forward_batch,
        set_forward_batch,
    )

    built = _prefill(monkeypatch, [_LinearLayer()], [])
    seen = []
    monkeypatch.setattr(
        prefill_fi, "core", lambda *args: seen.append(get_forward_batch())
    )
    monkeypatch.setattr(
        built,
        "_body",
        lambda *args: built._core("attn", "mixed", "a", "b", "out"),
        raising=False,
    )
    step = Step(mode="prefill", rows=4, view="view")
    original = lambda *args, **kw: pytest.fail("ran the stock body")
    fb = SimpleNamespace()
    built.forward_body(step, original, "ids", "pos", fb, None, None, None)
    # under the graph runner's scope, the step's batch is the published one
    with set_forward_batch(fb):
        built.forward_body(step, original, "ids", "pos", fb, None, None, None)
        assert get_forward_batch() is fb
    assert seen == [fb, fb]
    with pytest.raises(RuntimeError, match="No forward batch"):
        get_forward_batch()


def test_the_builds_offer_the_prefill_body_seam(monkeypatch):
    from sglang.kernels.plugin_kernels.nvidia.spec import SupportsPrefillBody

    from sgl_nvidia_mega.qwen3_5.runtime.executor import Qwen35Verify

    built = _prefill(monkeypatch, [_LinearLayer()], [])
    verify = Qwen35Verify.__new__(Qwen35Verify)
    assert isinstance(built, SupportsPrefillBody)
    assert isinstance(verify, SupportsPrefillBody)
    assert verify.plan_body(torch.zeros(4), _extend_batch()) is None
    assert verify.forward_body(None, lambda *a, **k: (a, k), 1, x=2) == ((1,), {"x": 2})


@pytest.mark.parametrize(
    "switch, slots, offered",
    [("1", None, True), (None, None, False), ("0", None, False), ("1", "", False)],
    ids=["measured-stack", "mk-alone", "switch-off", "slot-list"],
)
def test_the_spec_offers_the_prefill_body_under_the_measured_stack_only(
    monkeypatch, switch, slots, offered
):
    from sgl_nvidia_mega.qwen3_5 import runtime, spec

    for name, value in (
        ("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", switch),
        (runtime.SLOTS_ENV, slots),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    seam = "prefill_graph" in spec.MegakernelSpec.__struct_fields__
    assert spec._offers_prefill_body() is (offered and seam)
