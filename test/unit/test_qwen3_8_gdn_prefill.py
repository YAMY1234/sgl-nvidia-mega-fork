# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's GDN prefill step on CPU: the kernels each step runs and in
which order, the stock fallbacks, the options and the build refusals. SGLang's
backend and model modules are replaced by fakes and the kernel launches by spies, so
no GPU, no kernel build and no model weights are needed."""

import enum
import logging
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_flashinfer as fi_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_gating as gating_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_norm as norm_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import gdn_prefill_conv as conv_mod
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (
    gdn_prefill_state_glue as glue_mod,
)
from sgl_nvidia_mega.qwen3_8.runtime import gdn_prefill as gp

# A small GDN layer of Qwen3.8's shape: 2 key heads, 6 value heads (ratio 3), 128 wide
H, HV, D = 2, 6, 128
Q_DIM, V_DIM = H * D, HV * D
QKV = 2 * Q_DIM + V_DIM
HIDDEN = 64
SLOTS = 9


class _Mode(enum.Enum):
    EXTEND = 1
    MIXED = 2
    DECODE = 3
    TARGET_VERIFY = 4
    DRAFT_EXTEND_V2 = 5

    def is_extend_or_draft_extend_or_mixed(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)

    def is_extend_without_speculative(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)


class _FlashInferKernel:  # the fake FlashInferGDNKernel
    use_state_pool = True
    _prefill_needs_fp32_state = False


class _TritonKernel:
    pass


class _Linear:  # the fake Qwen4ExpLinearDecoderLayer
    def __init__(self, layer_id, gdn):
        self.layer_id, self.linear_attn = layer_id, gdn


_STATE = {}


def _install(test):
    """SGLang's modules the step imports, as fakes, restored after the test."""
    _STATE.update(breakable=False, full_graph=False, cp=False, backend=None, fi=True)
    fakes = {
        "sglang.srt.model_executor.forward_context": dict(
            get_attn_backend=lambda: _STATE["backend"]
        ),
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": dict(
            is_in_breakable_cuda_graph=lambda: _STATE["breakable"]
        ),
        "sglang.srt.model_executor.runner_utils.prefill_graph": dict(
            is_in_full_prefill_graph=lambda: _STATE["full_graph"]
        ),
        "sglang.srt.models.qwen4_exp": dict(
            _shards_rows_over_cp=lambda fb: _STATE["cp"],
            Qwen4ExpLinearDecoderLayer=_Linear,
        ),
        "sglang.srt.models.qwen3_5": dict(_GDN_FUSED_QKVZBA_RATIOS=(1, 2, 3, 4)),
        "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer": dict(
            FlashInferGDNKernel=_FlashInferKernel,
            is_flashinfer_gdn_prefill_available=lambda: _STATE["fi"],
        ),
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        old = sys.modules.get(name)
        sys.modules[name] = module
        test.addCleanup(_restore, name, old)
    tmem = gp._tmem_fixed
    test.addCleanup(setattr, gp, "_tmem_fixed", tmem)
    gp._tmem_fixed = False


def _restore(name, old):
    if old is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old


class _Spies:
    """The kernel launches the step makes, in order, with what they were given."""

    def __init__(self, test, *, snapshot=True, state_rows=True):
        self.calls = []
        self.snapshot, self.state_rows = snapshot, state_rows
        for module, name, fn in (
            (conv_mod, "causal_conv1d_fn_split_qkv", self.split_conv),
            (conv_mod, "causal_conv1d_fn", self.conv),
            (gating_mod, "fused_gdn_gating", self.gating),
            (fi_mod, "extend", self.flashinfer),
            (norm_mod, "_layer_norm_fwd", self.norm),
            (glue_mod, "gdn_track_conv_snapshot", self.track_conv),
            (glue_mod, "gdn_track_state_rows", self.track_states),
        ):
            patcher = mock.patch.object(module, name, fn)
            patcher.start()
            test.addCleanup(patcher.stop)

    def names(self):
        return [c[0] for c in self.calls]

    def kwargs(self, name):
        return next(c[1] for c in self.calls if c[0] == name)

    def split_conv(self, x, weight, bias, **kw):
        self.calls.append(("split_conv", dict(kw, x=x, weight=weight, bias=bias)))
        dim, total = x.shape
        q_dim, k_dim = kw["q_dim"], kw["k_dim"]
        out = torch.arange(dim * total, dtype=torch.float32)
        return (
            out[: total * q_dim].view(total, q_dim),
            out[total * q_dim : total * (q_dim + k_dim)].view(total, k_dim),
            out[total * (q_dim + k_dim) :].view(total, dim - q_dim - k_dim),
        )

    def conv(self, x, weight, bias, **kw):
        self.calls.append(("conv", dict(kw, x=x)))
        return torch.zeros(x.shape[0], x.shape[1])

    def gating(self, A_log, a, b, dt_bias, **kw):
        self.calls.append(("gating", dict(kw, a=a, b=b)))
        return torch.zeros(1, a.shape[0], HV), torch.zeros(1, a.shape[0], HV)

    def flashinfer(self, kernel, q, k, v, g, beta, **kw):
        self.calls.append(("flashinfer", dict(kw, kernel=kernel, q=q, k=k, v=v)))
        h = (
            torch.zeros(1, kw["num_state_checkpoints"], HV, D, D)
            if kw["num_state_checkpoints"]
            else None
        )
        return torch.ones(1, q.shape[1], HV, D), None, h

    def norm(self, x, weight, bias, eps, **kw):
        self.calls.append(("norm", dict(kw, x=x, weight=weight, bias=bias, eps=eps)))
        return x * 2, None, None

    def track_conv(self, **kw):
        self.calls.append(("track_conv", kw))
        return self.snapshot

    def track_states(self, **kw):
        self.calls.append(("track_states", kw))
        return self.state_rows


def _gdn(tp=1, head_dim=D, num_v_heads=HV, norm=None, out_proj_calls=None):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    attn = SimpleNamespace(
        layer_id=3,
        q_dim=Q_DIM,
        k_dim=Q_DIM,
        v_dim=V_DIM,
        num_q_heads=H,
        num_k_heads=H,
        num_v_heads=HV,
        head_q_dim=D,
        head_k_dim=D,
        head_v_dim=D,
        conv_weights=torch.zeros(QKV, 4),
        bias=None,
        activation="silu",
        A_log=torch.zeros(HV),
        dt_bias=torch.zeros(HV),
    )
    calls = out_proj_calls if out_proj_calls is not None else []

    def input_proj(x):
        rows = x.shape[0]
        qkvz = torch.arange(rows * (QKV + V_DIM), dtype=torch.float32)
        return qkvz.view(rows, QKV + V_DIM), torch.zeros(rows, 2 * HV)

    def out_proj(x):
        calls.append(x)
        return x.sum(dim=-1, keepdim=True), None

    return SimpleNamespace(
        attn=attn,
        attn_tp_size=tp,
        num_k_heads=H,
        num_v_heads=num_v_heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        _forward_input_proj=input_proj,
        norm=norm if norm is not None else RMSNorm(D, eps=1e-6, activation="sigmoid"),
        out_proj=out_proj,
    )


def _metadata(lens, *, track=False):
    starts = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32
    )
    md = SimpleNamespace(
        query_start_loc=starts,
        mamba_cache_indices=torch.arange(1, len(lens) + 1, dtype=torch.int32),
        conv_block_table=None,
        has_mamba_track_mask=track,
        track_conv_indices=None,
        conv_states_mask_indices=None,
        track_ssm_h_src=None,
        track_ssm_h_dst=None,
        track_ssm_final_src=None,
        track_ssm_final_dst=None,
        track_ssm_recompute_dst=None,
        state_checkpoint_cu_starts=None,
        num_state_checkpoints=0,
        state_checkpoint_every_n_tokens=0,
    )
    if track:
        # one tracked row: the conv window of tokens 1..3 into track slot 7, the
        # checkpoint row 0 into slot 8
        md.track_conv_indices = torch.tensor([[1, 2, 3]])
        md.conv_states_mask_indices = torch.tensor([7])
        md.track_ssm_h_src = torch.tensor([0])
        md.track_ssm_h_dst = torch.tensor([8])
        md.track_ssm_final_src = torch.zeros(0, dtype=torch.int64)
        md.track_ssm_final_dst = torch.zeros(0, dtype=torch.int64)
        md.state_checkpoint_cu_starts = torch.tensor([0, 1])
        md.num_state_checkpoints = 1
        md.state_checkpoint_every_n_tokens = 64
    return md


class _Backend:
    def __init__(self, md, kernel=None):
        self.forward_metadata = md
        self.kernel_dispatcher = SimpleNamespace(
            extend_kernel=kernel if kernel is not None else _FlashInferKernel()
        )
        self.mis_metadata = None
        self.prefill_graph_metadata = None
        self.conv = torch.zeros(SLOTS, QKV, 3)
        self.temporal = torch.zeros(SLOTS, HV, D, D)
        self.cache_calls = []
        self.split_calls = []
        self.track_calls = []

    def state_layer_cache(self, layer_id):
        self.cache_calls.append(layer_id)
        return SimpleNamespace(conv=[self.conv], temporal=self.temporal)

    def _split_prefill_qkv(self, layer, mixed_qkv):
        self.split_calls.append(tuple(mixed_qkv.shape))
        rows = mixed_qkv.shape[0]
        q, k, v = torch.split(mixed_qkv, [Q_DIM, Q_DIM, V_DIM], dim=-1)
        return q.view(1, rows, H, D), k.view(1, rows, H, D), v.view(1, rows, HV, D)

    def _track_mamba_state_extend(self, fb, h, ssm_states, md, h_track_buf=None):
        self.track_calls.append((h, ssm_states, md, h_track_buf))


class _St:
    def __init__(self, fb):
        self.fb = fb
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


def _batch(lens, mode=_Mode.EXTEND, logical=None, padded=None):
    return SimpleNamespace(
        forward_mode=mode,
        logical_forward_mode=logical if logical is not None else mode,
        extend_seq_lens_cpu=list(lens),
        extend_prefix_lens=torch.tensor([0, 5][: len(lens)]),
        global_num_token_non_padded_cpu=padded,
    )


class _Case(unittest.TestCase):
    def setUp(self):
        _install(self)

    def run_step(
        self,
        lens=(5, 3),
        options=gp.DEFAULT,
        *,
        track=False,
        spies=None,
        backend=None,
        gdn=None,
        fb=None,
        x=None,
    ):
        self.spies = spies or _Spies(self)
        self.backend = backend or _Backend(_metadata(lens, track=track))
        _STATE["backend"] = self.backend
        self.out_proj = []
        self.gdn = gdn or _gdn(out_proj_calls=self.out_proj)
        rows = sum(lens)
        self.x = x if x is not None else torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        self.st = _St(fb or _batch(lens))
        ex = SimpleNamespace(gdn_prefill=options)
        return gp.extend(ex, SimpleNamespace(linear_attn=self.gdn), self.x, self.st)


class TestDispatch(_Case):
    def test_slot_off_runs_stock(self):
        spies = _Spies(self)
        st = _St(_batch((4,)))
        ex = SimpleNamespace(gdn_prefill=None)
        out = gp.extend(ex, SimpleNamespace(linear_attn=_gdn()), torch.zeros(4, 8), st)
        self.assertIsNone(out)
        self.assertEqual(spies.calls, [])
        self.assertEqual(st.ran_ops, [])

    def test_prefill_runs_the_kernels_in_the_stock_order(self):
        out = self.run_step()
        s = self.spies
        self.assertEqual(s.names(), ["split_conv", "gating", "flashinfer", "norm"])
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi+rows_fused")])
        self.assertEqual(self.backend.cache_calls, [3])  # the layer's state, once
        # the conv: the prefill view of q|k|v, channel-last, in the stock call
        conv = s.kwargs("split_conv")
        self.assertEqual(tuple(conv["x"].shape), (QKV, 8))
        self.assertEqual(conv["x"].stride(), (1, QKV + V_DIM))
        self.assertIs(conv["conv_states"], self.backend.conv)
        self.assertEqual((conv["q_dim"], conv["k_dim"]), (Q_DIM, Q_DIM))
        self.assertEqual(conv["strip_cfg"], (32, 128, 4, 2048))
        self.assertEqual(conv["activation"], "silu")
        self.assertEqual(conv["seq_lens_cpu"], [5, 3])
        self.assertEqual(conv["has_initial_state"].tolist(), [False, True])
        self.assertIs(
            conv["cache_indices"], self.backend.forward_metadata.mamba_cache_indices
        )
        self.assertNotIn("block_table", conv)
        # the gates read b / a in place from the ba projection
        self.assertEqual(s.kwargs("gating")["gating_2d"], True)
        self.assertEqual(s.kwargs("gating")["a"].stride(), (2 * HV, 1))
        # the core: q / k / v views of the split planes, the stock kernel, the plan
        core = s.kwargs("flashinfer")
        self.assertIsInstance(core["kernel"], _FlashInferKernel)
        self.assertEqual(tuple(core["q"].shape), (1, 8, H, D))
        self.assertEqual(tuple(core["v"].shape), (1, 8, HV, D))
        self.assertTrue(core["q"].is_contiguous() and core["k"].is_contiguous())
        self.assertIs(core["ssm_states"], self.backend.temporal)
        self.assertEqual(
            (core["fused_rows"], core["l2norm_retile"], core["qk_one_launch"]),
            (True, True, True),
        )
        self.assertEqual(core["num_state_checkpoints"], 0)
        # the norm: the stock gated RMSNorm call with the strided 3D z
        norm = s.kwargs("norm")
        self.assertEqual(tuple(norm["x"].shape), (8 * HV, D))
        self.assertEqual(tuple(norm["z"].shape), (8, HV, D))
        self.assertFalse(norm["z"].is_contiguous())
        self.assertIsNone(norm["bias"])
        self.assertEqual(
            (norm["is_rms_norm"], norm["activation"], norm["norm_before_gate"]),
            (True, "sigmoid", True),
        )
        self.assertEqual((norm["group_size"], norm["retile"]), (None, True))
        # the output projection takes [T, HV * V]
        self.assertEqual(tuple(self.out_proj[0].shape), (8, V_DIM))
        self.assertEqual(tuple(out.shape), (8, 1))
        self.assertTrue(torch.equal(out, torch.full((8, 1), 2.0 * V_DIM)))

    def test_track_copies_fused(self):
        self.run_step(track=True)
        s = self.spies
        self.assertEqual(
            s.names(),
            [
                "track_conv",
                "split_conv",
                "gating",
                "flashinfer",
                "track_states",
                "norm",
            ],
        )
        md = self.backend.forward_metadata
        self.assertIs(s.kwargs("track_conv")["token_rows"], md.track_conv_indices)
        self.assertIs(s.kwargs("track_conv")["dst_slots"], md.conv_states_mask_indices)
        self.assertEqual(tuple(s.kwargs("track_states")["h"].shape), (1, HV, D, D))
        self.assertEqual(s.kwargs("flashinfer")["num_state_checkpoints"], 1)
        self.assertEqual(self.backend.track_calls, [])
        self.assertTrue(
            torch.equal(self.backend.conv, torch.zeros_like(self.backend.conv))
        )
        self.assertEqual(
            self.st.ran_ops, [("gdn", "conv_split+fi+rows_fused+track_fused")]
        )

    def test_track_copies_stock_where_the_fused_ones_decline(self):
        spies = _Spies(self, snapshot=False, state_rows=False)
        self.run_step(track=True, spies=spies)
        # the conv window by torch indexing, before the conv
        qkvz, _ = self.gdn._forward_input_proj(self.x)
        expected = qkvz[:, :QKV][torch.tensor([[1, 2, 3]])].transpose(1, 2)
        self.assertTrue(torch.equal(self.backend.conv[7], expected[0]))
        ((h, ssm, md, buf),) = self.backend.track_calls
        self.assertEqual(tuple(h.shape), (1, 1, HV, D, D))
        self.assertIs(ssm, self.backend.temporal)
        self.assertIsNone(buf)
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi+rows_fused+track")])

    def test_recompute_rows_take_the_stock_track_copy(self):
        backend = _Backend(_metadata((5, 3), track=True))
        backend.forward_metadata.track_ssm_recompute_dst = torch.tensor([6])
        self.run_step(track=True, backend=backend)
        self.assertNotIn("track_states", self.spies.names())
        self.assertEqual(len(backend.track_calls), 1)

    def test_plain_conv_where_the_split_conv_does_not_apply(self):
        # one token: the split conv takes two or more
        self.run_step(lens=(1,), fb=_batch((1,)))
        conv = self.spies.kwargs("conv")
        self.assertEqual(
            conv["fwd_launch"],
            {"BLOCK_M": 8, "BLOCK_N": 256, "num_warps": 2, "num_stages": 2},
        )
        self.assertEqual(self.backend.split_calls, [(1, QKV)])
        self.assertEqual(self.st.ran_ops, [("gdn", "conv+fi+rows_fused")])

    def test_block_table_reaches_the_conv(self):
        backend = _Backend(_metadata((5, 3)))
        backend.forward_metadata.conv_block_table = torch.zeros(2, 2, dtype=torch.int32)
        self.run_step(backend=backend)
        table = self.spies.kwargs("split_conv")["block_table"]
        self.assertIs(table, backend.forward_metadata.conv_block_table)

    def test_options_off_run_the_stock_launches(self):
        off = gp.Options(
            split_conv=False,
            conv_strip=None,
            conv_tile=None,
            gating_2d=False,
            l2norm_retile=False,
            qk_l2norm_one_launch=False,
            norm_retile=False,
            fused_state_copy=False,
            fused_track=False,
            tmem_dealloc=False,
        )
        self.run_step(options=off, track=True)
        s = self.spies
        self.assertEqual(s.names(), ["conv", "gating", "flashinfer", "norm"])
        self.assertIsNone(s.kwargs("conv")["fwd_launch"])
        self.assertFalse(s.kwargs("gating")["gating_2d"])
        core = s.kwargs("flashinfer")
        self.assertEqual(
            (core["fused_rows"], core["l2norm_retile"], core["qk_one_launch"]),
            (False, False, False),
        )
        self.assertFalse(s.kwargs("norm")["retile"])
        self.assertEqual(len(self.backend.track_calls), 1)
        self.assertEqual(self.st.ran_ops, [("gdn", "conv+fi+rows+track")])

    def test_strided_slots_take_the_stock_row_copies(self):
        backend = _Backend(_metadata((5, 3)))
        backend.forward_metadata.mamba_cache_indices = torch.tensor(
            [1, 0, 2, 0], dtype=torch.int32
        )[::2]
        self.run_step(backend=backend)
        self.assertFalse(self.spies.kwargs("flashinfer")["fused_rows"])
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi+rows")])

    def test_installed_tmem_fix_is_recorded(self):
        gp._tmem_fixed = True
        self.run_step()
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi_tmem+rows_fused")])


class TestFallbacks(_Case):
    """Every step the kernels do not serve returns None before anything is written."""

    def assert_stock(self, reason_part, **kw):
        out = self.run_step(**kw)
        self.assertIsNone(out)
        self.assertEqual(self.spies.calls, [])
        self.assertEqual(self.st.ran_ops, [])
        self.assertEqual(self.out_proj, [])
        self.assertTrue(
            torch.equal(self.backend.conv, torch.zeros_like(self.backend.conv))
        )
        reason, step = gp.prepare(self.gdn, self.x, self.st.fb)
        self.assertIsNone(step)
        self.assertIn(reason_part, reason)

    def test_modes(self):
        self.assert_stock("extend", fb=_batch((5, 3), mode=_Mode.TARGET_VERIFY))
        self.assert_stock("extend", fb=_batch((5, 3), mode=_Mode.DRAFT_EXTEND_V2))
        self.assert_stock(
            "extend", fb=_batch((5, 3), mode=_Mode.MIXED, logical=_Mode.DECODE)
        )

    def test_tp(self):
        self.assert_stock("TP2", gdn=_gdn(tp=2))

    def test_dtype(self):
        self.assert_stock("BF16", x=torch.zeros(8, HIDDEN, dtype=torch.float16))

    def test_prefill_graphs(self):
        _STATE["breakable"] = True
        self.assert_stock("graph")
        _STATE.update(breakable=False, full_graph=True)
        self.assert_stock("graph")

    def test_cp(self):
        _STATE["cp"] = True
        self.assert_stock("CP")

    def test_dp_padding(self):
        self.assert_stock("DP padding", fb=_batch((5, 3), padded=6))

    def test_other_prefill_kernel(self):
        self.assert_stock(
            "_TritonKernel", backend=_Backend(_metadata((5, 3)), kernel=_TritonKernel())
        )

    def test_mis(self):
        backend = _Backend(_metadata((5, 3)))
        backend.mis_metadata = object()
        self.assert_stock("MIS", backend=backend)

    def test_captured_bucket(self):
        backend = _Backend(_metadata((5, 3)))
        backend.prefill_graph_metadata = object()
        self.assert_stock("bucket", backend=backend)

    def test_page_major_pool(self):
        backend = _Backend(_metadata((5, 3)))
        backend.conv = torch.zeros(QKV, SLOTS, 3).transpose(0, 1)
        self.assert_stock("strided", backend=backend)
        backend = _Backend(_metadata((5, 3)))
        backend.temporal = torch.zeros(SLOTS, HV, D, 2 * D)[..., :D]
        self.assert_stock("strided", backend=backend)


class TestOptions(unittest.TestCase):
    def test_defaults_are_the_recipe_values(self):
        o = gp.DEFAULT
        self.assertEqual(o.strip_cfg(), (32, 128, 4, 2048))
        self.assertEqual(
            o.fwd_launch(),
            {"BLOCK_M": 8, "BLOCK_N": 256, "num_warps": 2, "num_stages": 2},
        )
        for name in (
            "split_conv",
            "gating_2d",
            "l2norm_retile",
            "qk_l2norm_one_launch",
            "norm_retile",
            "fused_state_copy",
            "fused_track",
            "tmem_dealloc",
        ):
            self.assertTrue(getattr(o, name), name)

    def test_tiles_are_validated(self):
        self.assertEqual(
            gp.Options(conv_strip=(64, 256, 8)).strip_cfg().min_tokens, 2048
        )
        for bad in (
            dict(conv_strip=(24, 128, 4, 2048)),
            dict(conv_strip=(32, 128, 3, 2048)),
            dict(conv_strip=(0,)),
            dict(conv_tile=(4, 256, 2, 2)),
            dict(conv_tile=(8, 32, 2, 2)),
            dict(conv_tile=(0,)),
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                gp.Options(**bad)


class TestBuild(unittest.TestCase):
    def setUp(self):
        _install(self)

    def test_layer_reasons(self):
        self.assertEqual(gp.layer_reasons(_gdn()), "")
        self.assertIn("TP2", gp.layer_reasons(_gdn(tp=2)))
        self.assertIn("head dims", gp.layer_reasons(_gdn(head_dim=64)))
        self.assertIn("value heads", gp.layer_reasons(_gdn(num_v_heads=5 * H)))
        self.assertIn(
            "RMSNorm", gp.layer_reasons(_gdn(norm=SimpleNamespace(bias=None)))
        )
        norm = _gdn().norm
        norm.bias = torch.nn.Parameter(torch.zeros(D))
        self.assertIn("RMSNorm", gp.layer_reasons(_gdn(norm=norm)))

    def test_needs_an_sm10x_gpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIn("SM10x", gp.build(self.model()))
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            self.assertIn("SM10x", gp.build(self.model()))

    def model(self, gdns=None):
        gdns = gdns if gdns is not None else [_gdn(), _gdn()]
        layers = [_Linear(i, g) for i, g in enumerate(gdns)] + [object()]
        m = SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
        return SimpleNamespace(model=m)

    def build(self, model, options=gp.DEFAULT, *, strip_error=None, tmem=None):
        """build on a faked SM100 with the strip warmup and the #8854 install spied."""
        self.warmups, self.installs = [], []

        def warmup(cfg):
            self.warmups.append(cfg)
            if strip_error is not None:
                raise strip_error

        def install():
            self.installs.append(1)
            if isinstance(tmem, Exception):
                raise tmem
            return True

        fake = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_tmem_dealloc"
        )
        fake.install = install
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ),
            mock.patch.object(conv_mod, "warmup_prefill_conv_strip", warmup),
            mock.patch.dict(sys.modules, {fake.__name__: fake}),
        ):
            return gp.build(model, options)

    def test_builds_the_strip_and_installs_the_tmem_fix(self):
        self.assertEqual(self.build(self.model()), "")
        self.assertEqual(self.warmups, [(32, 128, 4, 2048)])
        self.assertEqual(self.installs, [1])
        self.assertTrue(gp._tmem_fixed)
        self.assertEqual(self.build(self.model()), "")  # installed once per process
        self.assertEqual(self.installs, [])

    def test_refusals(self):
        self.assertEqual(self.build(self.model(gdns=[])), "no GDN layer")
        reason = self.build(self.model(gdns=[_gdn(), _gdn(tp=2)]))
        self.assertTrue(reason.startswith("layer 1: ") and "TP2" in reason, reason)
        _STATE["fi"] = False
        self.assertIn("FlashInfer", self.build(self.model()))
        _STATE["fi"] = True
        reason = self.build(self.model(), strip_error=RuntimeError("nvcc failed"))
        self.assertIn("strip does not build: RuntimeError: nvcc failed", reason)

    def test_strip_off_builds_nothing(self):
        options = gp.Options(conv_strip=None, tmem_dealloc=False)
        self.assertEqual(self.build(self.model(), options), "")
        self.assertEqual((self.warmups, self.installs), ([], []))
        self.assertFalse(gp._tmem_fixed)

    def test_tmem_fix_on_another_flashinfer_warns_and_builds(self):
        with self.assertLogs(gp.logger, logging.WARNING) as logs:
            reason = self.build(
                self.model(), tmem=ImportError("not FlashInfer v0.7.3's")
            )
        self.assertEqual(reason, "")
        self.assertFalse(gp._tmem_fixed)
        self.assertIn("not FlashInfer v0.7.3's", logs.output[0])


if __name__ == "__main__":
    unittest.main()
