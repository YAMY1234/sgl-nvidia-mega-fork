# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's host-side decisions on CPU: which forwards plan() covers,
the startup refusals and the engagement log. SGLang's model module and graph
helpers are replaced by fakes, so no GPU and no model weights are needed."""

import enum
import logging
import sys
import types
import unittest
from types import SimpleNamespace

import torch


class _Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    MIXED = 3
    IDLE = 4
    TARGET_VERIFY = 5
    DRAFT_EXTEND_V2 = 6
    PREBUILT = 7

    def is_idle(self):
        return self is _Mode.IDLE

    def is_target_verify(self):
        return self is _Mode.TARGET_VERIFY

    def is_decode(self):
        return self is _Mode.DECODE

    def is_draft_extend_v2(self):
        return self is _Mode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)


class _Linear:  # the fake GDN layer type
    ple = None


class _Attention:
    ple = None


_STATE = {"breakable": False, "cp": False}


def _install_fakes():
    qwen4 = types.ModuleType("sglang.srt.models.qwen4_exp")
    qwen4.Qwen4ExpLinearDecoderLayer = _Linear
    qwen4.Qwen4ExpAttentionDecoderLayer = _Attention
    qwen4._shards_rows_over_cp = lambda fb: _STATE["cp"]
    sys.modules["sglang.srt.models.qwen4_exp"] = qwen4
    bcg = types.ModuleType(
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    )
    bcg.is_in_breakable_cuda_graph = lambda: _STATE["breakable"]
    sys.modules[bcg.__name__] = bcg


_install_fakes()

from sgl_nvidia_mega.qwen3_8 import runtime  # noqa: E402
from sgl_nvidia_mega.qwen3_8.runtime.engagement import Engagement  # noqa: E402
from sgl_nvidia_mega.qwen3_8.runtime.executor import (  # noqa: E402
    Qwen3_8Executor,
    Qwen3_8UcacheExecutor,
)


def _model(layers=(_Linear, _Linear, _Attention)):
    m = SimpleNamespace(
        layers=[cls() for cls in layers], start_layer=0, end_layer=len(layers)
    )
    return SimpleNamespace(model=m)


def _batch(mode, rows):
    return SimpleNamespace(forward_mode=mode, input_ids=torch.zeros(rows))


def _executor(draft=False):
    widths = tuple(range(1, 1025)) if draft else tuple(range(4, 1025, 4))
    return Qwen3_8Executor(_model(), widths=widths, draft=draft, slots=frozenset())


class TestPlan(unittest.TestCase):
    def setUp(self):
        _STATE.update(breakable=False, cp=False)

    def plan(self, ex, mode, rows, **kw):
        step = ex.plan(torch.zeros(rows), _batch(mode, rows), **kw)
        return None if step is None else (step.mode, step.rows)

    def test_target(self):
        ex = _executor()
        self.assertEqual(self.plan(ex, _Mode.TARGET_VERIFY, 8), ("target_verify", 8))
        self.assertEqual(self.plan(ex, _Mode.TARGET_VERIFY, 1024), ("target_verify", 1024))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 6))  # not a built width
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 1028))
        self.assertEqual(self.plan(ex, _Mode.DECODE, 4), ("decode", 4))
        self.assertIsNone(self.plan(ex, _Mode.DRAFT_EXTEND_V2, 8))
        self.assertEqual(self.plan(ex, _Mode.EXTEND, 16384), ("prefill", 16384))
        self.assertEqual(self.plan(ex, _Mode.MIXED, 37), ("prefill", 37))

    def test_draft(self):
        ex = _executor(draft=True)
        self.assertEqual(self.plan(ex, _Mode.DECODE, 3), ("decode", 3))
        self.assertEqual(self.plan(ex, _Mode.DRAFT_EXTEND_V2, 12), ("draft_extend", 12))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8))
        self.assertEqual(self.plan(ex, _Mode.EXTEND, 5), ("prefill", 5))

    def test_declined(self):
        ex = _executor()
        self.assertIsNone(self.plan(ex, _Mode.IDLE, 0))
        self.assertIsNone(self.plan(ex, _Mode.PREBUILT, 4))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8, get_embedding=True))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8, pp_proxy_tensors=1))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8, unknown=1))
        _STATE["breakable"] = True
        self.assertIsNone(self.plan(ex, _Mode.EXTEND, 64))
        _STATE.update(breakable=False, cp=True)
        self.assertIsNone(self.plan(ex, _Mode.EXTEND, 64))

    def test_mixers(self):
        ex = _executor()
        names = [ex.mixers[i].__name__ for i in range(3)]
        self.assertEqual(names, ["_gdn", "_gdn", "_qsa"])


class _St:
    def __init__(self, mode, rows=8):
        self.fb = SimpleNamespace(
            forward_mode=mode, spec_info=SimpleNamespace(draft_token_num=4)
        )
        self.positions = torch.zeros(rows)
        self.step = SimpleNamespace(
            mode="prefill" if mode in (_Mode.EXTEND, _Mode.MIXED) else str(mode),
            rows=rows,
        )
        self.ran_ops = []
        self.scratch = {}

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


class TestStepFallbacks(unittest.TestCase):
    """Each step function runs SGLang's module when its slot is off or the step is
    outside what the kernel serves, and records that it ran stock."""

    def setUp(self):
        qwen2 = types.ModuleType("sglang.srt.models.qwen2_moe")

        class Block:
            def __call__(self, x, fb=None, defer_finalize=None):
                return ("moe", defer_finalize)

        qwen2.Qwen2MoeSparseMoeBlock = Block
        sys.modules["sglang.srt.models.qwen2_moe"] = qwen2
        self.Block = Block

    def test_moe_stock(self):
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        ex = SimpleNamespace(moe_finalize_ready=False)
        layer = SimpleNamespace(mlp=self.Block())
        st = _St(_Mode.TARGET_VERIFY)
        self.assertEqual(moe.step(ex, layer, torch.zeros(8, 2560), st), ("moe", False))
        self.assertEqual(st.ran_ops, [("moe", "stock")])

    def test_moe_rows_bound(self):
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        ex = SimpleNamespace(moe_finalize_ready=True)
        layer = SimpleNamespace(mlp=self.Block())
        st = _St(_Mode.EXTEND)
        moe.step(ex, layer, torch.zeros(moe.MAX_ROWS + 1, 2560), st)
        self.assertEqual(st.ran_ops, [("moe", "stock")])

    def test_moe_routed_before_shared(self):
        """Eager (not capturing): SGLang's order, the routed experts then the shared
        expert, then the finalize kernel."""
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        calls = []
        runner = types.ModuleType("sglang.srt.model_executor.runner")
        runner.get_is_capture_mode = lambda: False
        fin = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.moe_finalize"
        )

        class Kernel:
            def __getitem__(self, grid):
                return lambda *a, **k: calls.append("finalize")

        fin._fused_finalize_gate_sigmoid_mul_add_kernel = Kernel()
        saved = {n: sys.modules.get(n) for n in (runner.__name__, fin.__name__)}
        sys.modules.update({runner.__name__: runner, fin.__name__: fin})
        self.addCleanup(
            lambda: [
                sys.modules.pop(n) if v is None else sys.modules.__setitem__(n, v)
                for n, v in saved.items()
            ]
        )
        rows = 8

        class Block(self.Block):
            alt_stream = None
            shared_expert = None
            shared_expert_gate = SimpleNamespace(weight=torch.zeros(1, 2560))

            def _forward_router_experts(self, x, defer_finalize):
                calls.append("routed")
                return SimpleNamespace(
                    top_k=10,
                    gemm2_out=torch.zeros(rows * 10, 2560),
                    expanded_idx_to_permuted_idx=torch.zeros(rows, 10, dtype=torch.int32),
                    expert_weights=torch.zeros(rows, 10),
                )

            def _forward_shared_experts(self, x, apply_gate):
                calls.append("shared")
                return torch.zeros_like(x)

        ex = SimpleNamespace(moe_finalize_ready=True, moe_routing=None, smallm_gemm=None)
        layer = SimpleNamespace(mlp=Block())
        st = _St(_Mode.TARGET_VERIFY)
        moe.step(ex, layer, torch.zeros(rows, 2560), st)
        self.assertEqual(calls, ["routed", "shared", "finalize"])
        self.assertEqual(st.ran_ops, [("moe", "finalize_permuted")])

    def test_gdn_stock_outside_verify(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = SimpleNamespace(gdn_pdl_ready=True, gdn_ucache=None, gdn_prefill=None)
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        st = _St(_Mode.EXTEND)
        self.assertEqual(gdn.step(ex, layer, torch.zeros(8, 4), st), "gdn-stock")
        self.assertEqual(st.ran_ops, [("gdn", "stock")])

    def test_gdn_glue_declines_one_request(self):
        """The glue serves 2-8 requests; a 1-request verify falls through."""
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = SimpleNamespace(
            gdn_glue_ready=True, gdn_pdl_ready=False, gdn_ucache=None, gdn_prefill=None,
            gdn_front_ready=False,
        )
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        st = _St(_Mode.TARGET_VERIFY, rows=4)
        self.assertEqual(gdn.step(ex, layer, torch.zeros(4, 8), st), "gdn-stock")
        self.assertEqual(st.ran_ops, [("gdn", "stock")])

    def test_gdn_glue_runs_the_projections_around_one_launch(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        rows, seen = 8, {}
        kmod = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_fused_verify_glue"
        )

        def glue(**kw):
            seen.update(kw)
            return torch.ones(rows, 6)

        kmod.gdn_fused_verify_glue = glue
        umod = types.ModuleType("sglang.srt.layers.attention.linear.utils")
        selected = []
        umod.select_verify_intermediate_state_indices = (
            lambda idx, req, valid, size: selected.append(valid) or idx
        )
        umod.pp_spec_stable_rows_enabled = lambda: False
        cache = SimpleNamespace(
            conv=[torch.zeros(1)], intermediate_conv_window=[torch.zeros(1)],
            temporal="ckpt", replayssm_d="d", replayssm_k="k", replayssm_g="g",
            replayssm_rawv=None, replayssm_rawk=None, replayssm_beta=None,
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                retrieve_next_token=None, retrieve_next_sibling=None,
                retrieve_parent_token=None, mamba_cache_indices=torch.arange(2),
            ),
            state_layer_cache=lambda layer_id: cache,
            state_pool=SimpleNamespace(
                replayssm_spec_write_pos="wp", replayssm_cache_base="cb",
                replayssm_is_flush="fl",
            ),
            verify_intermediate_state_indices=torch.arange(2),
            req_to_token_pool=SimpleNamespace(size=16),
        )
        attn = SimpleNamespace(
            layer_id=0, conv_weights="w", activation="silu", A_log="al", dt_bias="dt",
            num_k_heads=4, head_k_dim=128,
        )
        linear = SimpleNamespace(
            attn=attn,
            norm=SimpleNamespace(weight="nw", eps=1e-6, activation="swish"),
            _forward_input_proj=lambda x: ("qkvz", "ba"),
            out_proj=lambda core: (core * 2, None),
        )
        # slot gdn_front declines 8 rows before it imports its kernels
        ex = SimpleNamespace(
            gdn_glue_ready=True, gdn_pdl_ready=False, gdn_ucache=None,
            gdn_prefill=None, smallm_gemm=None, gdn_front_ready=True,
        )
        st = _St(_Mode.TARGET_VERIFY, rows=rows)
        st.fb.req_pool_indices = torch.arange(2)
        with mock.patch.dict(sys.modules, {kmod.__name__: kmod, umod.__name__: umod}), \
                mock.patch.object(gdn, "linear_backend", lambda: backend):
            out = gdn.step(ex, SimpleNamespace(linear_attn=linear), torch.zeros(rows, 8), st)
        self.assertTrue(torch.equal(out, torch.full((rows, 6), 2.0)))
        self.assertEqual(st.ran_ops, [("gdn", "verify_glue")])
        self.assertEqual((seen["mixed_qkvz"], seen["mixed_ba"]), ("qkvz", "ba"))
        self.assertEqual((seen["draft_token_num"], seen["norm_activation"]), (4, "swish"))
        self.assertIsNone(seen["conv_bias"])
        # without PP stable rows, the source's call: no mask, the backend's own indices
        self.assertEqual(selected, [])
        self.assertIs(seen["intermediate_state_indices"], backend.verify_intermediate_state_indices)

    def test_gdn_front_runs_one_request(self):
        """4 rows: the front's projection + conv, then the core with the fused norm on
        the packed [q|k|v|z|b|a] columns, then the out_proj; the window rows are
        selected once a step."""
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        rows, hv, vd, qk = 4, 3, 2, 4  # packed: q 4 | k 4 | v 6 | z 6 | b 3 | a 3
        width = 2 * qk + 2 * hv * vd + 2 * hv
        seen, selects = {}, []
        pkg = "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front."
        front, core = types.ModuleType(pkg + "mk_front"), types.ModuleType(pkg + "mk_core")

        def run(x, weight, conv_w, conv_b, conv_state, state_idx, window, w_rows, pad_slot):
            seen["run"] = (weight, conv_b, state_idx, w_rows, pad_slot)
            return torch.arange(width, dtype=torch.float32).repeat(rows, 1), "qkv"

        def core_fn(qkv, a, b, *pools, z, norm_weight, normalized, eps):
            seen["core"] = (qkv, a[0].tolist(), b[0].tolist(), z[0].tolist(), pools)
            return normalized.fill_(1.0)

        front.run, core.core = run, core_fn
        conv = types.ModuleType("sglang.kernels.ops.mamba.causal_conv1d_triton")
        conv.PAD_SLOT_ID = -1
        umod = types.ModuleType("sglang.srt.layers.attention.linear.utils")
        umod.select_verify_intermediate_state_indices = (
            lambda idx, req, valid, size: selects.append(size) or idx
        )
        cache = SimpleNamespace(
            conv=["conv"], intermediate_conv_window=["window"], temporal="ckpt",
            replayssm_d="d", replayssm_k="k", replayssm_g="g", replayssm_rawv="dlo",
            replayssm_rawk="klo",
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                retrieve_next_token=None, retrieve_next_sibling=None,
                retrieve_parent_token=None, mamba_cache_indices=torch.tensor([5, -1]),
            ),
            state_layer_cache=lambda layer_id: cache,
            state_pool=SimpleNamespace(
                replayssm_spec_write_pos="wp", replayssm_cache_base="cb",
                replayssm_is_flush="fl",
            ),
            verify_intermediate_state_indices=torch.tensor([7, 8]),
            req_to_token_pool=SimpleNamespace(size=16),
        )
        attn = SimpleNamespace(
            layer_id=0, conv_weights="w", A_log="al", dt_bias="dt", q_dim=qk, k_dim=qk,
            v_dim=hv * vd, num_v_heads=hv, head_v_dim=vd,
        )
        linear = SimpleNamespace(
            attn=attn,
            norm=SimpleNamespace(weight="nw", eps=1e-6),
            _fused_in_proj_weight="packed_w",
            out_proj=lambda core: (core * 2, None),
        )
        ex = SimpleNamespace(
            gdn_front_ready=True, gdn_glue_ready=True, gdn_pdl_ready=False,
            gdn_ucache=None, gdn_prefill=None, smallm_gemm=None,
        )
        st = _St(_Mode.TARGET_VERIFY, rows=rows)
        st.fb.req_pool_indices = torch.tensor([3])
        mods = {m.__name__: m for m in (front, core, conv, umod)}
        with mock.patch.dict(sys.modules, mods), \
                mock.patch.object(gdn, "linear_backend", lambda: backend):
            for _ in range(2):  # two GDN layers of one step
                out = gdn.step(
                    ex, SimpleNamespace(linear_attn=linear), torch.zeros(rows, 8), st
                )
        self.assertTrue(torch.equal(out, torch.full((rows, hv * vd), 2.0)))
        self.assertEqual(st.ran_ops, [("gdn", "front")] * 2)
        self.assertEqual(selects, [16])
        weight, conv_b, state_idx, w_rows, pad_slot = seen["run"]
        self.assertEqual((weight, conv_b, pad_slot), ("packed_w", None, -1))
        self.assertEqual((state_idx.tolist(), w_rows.tolist()), ([5], [7]))
        qkv, a, b, z, pools = seen["core"]
        self.assertEqual(qkv, "qkv")
        # mk_core.core's order: A_log ... rings, state / replay indices, cursors, out
        self.assertEqual(pools[:8], ("al", "dt", "ckpt", "d", "k", "g", "dlo", "klo"))
        self.assertEqual(pools[8].tolist() + pools[9].tolist(), [5, -1, 3])
        self.assertEqual(pools[10:13], ("wp", "cb", "fl"))
        self.assertEqual(z, [float(c) for c in range(14, 20)])
        self.assertEqual((b, a), ([20.0, 21.0, 22.0], [23.0, 24.0, 25.0]))
        self.assertEqual(tuple(pools[13].shape), (rows, hv, vd))

    def test_qsa_prefill_slot_or_stock(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention, qsa_prefill

        layer = SimpleNamespace(
            is_qsa=True, self_attention=lambda **kw: "qsa-stock"
        )
        ex = SimpleNamespace(qsa_pool_ready=False, qsa_indexer=None, qsa_prefill="o")
        for served, want in ((None, "qsa-stock"), ("qsa-prefill", "qsa-prefill")):
            with mock.patch.object(qsa_prefill, "extend", return_value=served) as ext:
                out = attention.step(ex, layer, torch.zeros(8, 4), _St(_Mode.EXTEND))
                self.assertEqual(out, want)
                out = attention.step(ex, layer, torch.zeros(8, 4), _St(_Mode.DECODE))
                self.assertEqual(out, "qsa-stock")
                ext.assert_called_once()

    def test_gdn_prefill_slot_or_stock(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn, gdn_prefill

        ex = SimpleNamespace(gdn_pdl_ready=False, gdn_ucache=None, gdn_prefill="opts")
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        for served, want in ((None, "gdn-stock"), ("gdn-prefill", "gdn-prefill")):
            with mock.patch.object(gdn_prefill, "extend", return_value=served) as ext:
                st = _St(_Mode.EXTEND)
                self.assertEqual(gdn.step(ex, layer, torch.zeros(8, 4), st), want)
                ext.assert_called_once()
                # a verify never asks the prefill kernels
                st = _St(_Mode.DECODE)
                self.assertEqual(gdn.step(ex, layer, torch.zeros(8, 4), st), "gdn-stock")
                ext.assert_called_once()

    def test_qsa_stock_when_off_or_prefill(self):
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        calls = []
        layer = SimpleNamespace(
            is_qsa=True, self_attention=lambda **kw: calls.append(kw) or "qsa-stock"
        )
        for ready, mode in ((False, _Mode.TARGET_VERIFY), (True, _Mode.EXTEND)):
            st = _St(mode, 8)
            ex = SimpleNamespace(qsa_pool_ready=ready, qsa_indexer=None, qsa_prefill=None)
            out = attention.step(ex, layer, torch.zeros(8, 4), st)
            self.assertEqual(out, "qsa-stock")
            self.assertEqual(st.ran_ops, [("qsa", "stock")])
        self.assertEqual(len(calls), 2)


class TestReasons(unittest.TestCase):
    def test_prefill_graphs_refused(self):
        cfg = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="breakable"))
        )
        self.assertTrue(runtime.config_reasons(cfg, None))
        cfg.cuda_graph_config.prefill.backend = "disabled"
        self.assertEqual(runtime.config_reasons(cfg, None), [])

    def test_model_reasons(self):
        pp = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        self.assertTrue(runtime.model_reasons(SimpleNamespace(pp_group=pp), None))
        model = SimpleNamespace(capture_aux_hidden_states=True)
        self.assertTrue(runtime.model_reasons(model, None))
        self.assertEqual(runtime.model_reasons(SimpleNamespace(), None), [])


def _gdn_front_model():
    """One GDN layer of the TP1 shape slot gdn_front takes."""
    attn = SimpleNamespace(
        q_dim=2048, k_dim=2048, v_dim=6144, num_v_heads=48, num_q_heads=16,
        head_q_dim=128, num_k_heads=16, head_k_dim=128, bias=None, activation="silu",
        A_log=torch.zeros(48), dt_bias=torch.zeros(48, dtype=torch.bfloat16),
    )
    norm = SimpleNamespace(
        group_size=None, norm_before_gate=True, bias=None, activation="sigmoid",
        weight=torch.zeros(128, dtype=torch.bfloat16),
    )
    layer = _Linear()
    layer.linear_attn = SimpleNamespace(
        attn=attn, norm=norm, attn_tp_size=1, num_k_heads=16, num_v_heads=48,
        head_k_dim=128, head_v_dim=128,
        _fused_in_proj_weight=SimpleNamespace(shape=(16480, 2560), dtype=torch.bfloat16),
    )
    m = SimpleNamespace(layers=[layer, _Attention()], start_layer=0, end_layer=2)
    return SimpleNamespace(model=m), layer.linear_attn


def _gdn_front_backend():
    bf16 = torch.bfloat16
    cache = SimpleNamespace(
        replayssm_d=torch.zeros(2, 1, 16, 1, dtype=bf16), replayssm_rawv="dlo",
        replayssm_rawk="klo", replayssm_beta=None, temporal=torch.zeros(1, dtype=bf16),
        conv=[torch.zeros(2, 4, 3, dtype=bf16)],
        intermediate_conv_window=[torch.zeros(2, 4, 4, 3, dtype=bf16)],
    )
    pool = SimpleNamespace(
        replayssm_cache_base="cb", replayssm_spec_fold=False, replayssm_is_kda=False
    )
    return SimpleNamespace(state_pool=pool, state_layer_cache=lambda i: cache), cache


class TestGdnFront(unittest.TestCase):
    def test_model_reasons(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        model, linear = _gdn_front_model()
        self.assertEqual(gdn.front_reasons(model), "")
        for change in (
            lambda: setattr(linear, "_fused_in_proj_weight", None),
            lambda: setattr(linear._fused_in_proj_weight, "shape", (16480, 4096)),
            lambda: setattr(linear.norm, "activation", "swish"),  # the core's is sigmoid
            lambda: setattr(linear, "attn_tp_size", 4),
            lambda: setattr(linear.attn, "bias", torch.zeros(1)),
            lambda: setattr(linear.attn, "A_log", torch.zeros(48, dtype=torch.bfloat16)),
        ):
            model, linear = _gdn_front_model()
            change()
            self.assertTrue(gdn.front_reasons(model))

    def test_pool_reasons(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        backend, _ = _gdn_front_backend()
        self.assertEqual(gdn.front_pool_reasons(backend, 0, 4), "")
        self.assertTrue(gdn.front_pool_reasons(backend, 0, 8))
        for name, value in (
            ("replayssm_beta", "beta"),  # the exact-raw rings
            ("replayssm_rawv", None),
            ("replayssm_d", torch.zeros(2, 1, 16, 1)),
            ("temporal", torch.zeros(1)),
        ):
            backend, cache = _gdn_front_backend()
            setattr(cache, name, value)
            self.assertTrue(gdn.front_pool_reasons(backend, 0, 4), name)

    def test_pool_check_turns_the_slot_off(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = _executor()
        attn = SimpleNamespace(layer_id=3)
        ex.lm.model.layers[0].linear_attn = SimpleNamespace(attn=attn)
        ex.gdn_front_ready = True
        fb = SimpleNamespace(spec_info=SimpleNamespace(draft_token_num=4))
        with mock.patch.object(gdn, "linear_backend", lambda: "backend"), \
                mock.patch.object(gdn, "front_pool_reasons", return_value="no rings") as r:
            ex._check_pools(fb)
        r.assert_called_once_with("backend", 3, 4)
        self.assertFalse(ex.gdn_front_ready)

    def test_front_compiles_eagerly_never_while_capturing(self):
        """A layout compiles at its first eager call and a capture reuses it; a cold
        layout while capturing raises before any compile or launch."""
        import importlib
        from unittest import mock

        import sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front as package

        calls, capturing = [], [False]
        cuda, bindings, driver = (
            types.ModuleType(n) for n in ("cuda", "cuda.bindings", "cuda.bindings.driver")
        )
        cuda.bindings, bindings.driver, driver.CUstream = bindings, driver, lambda s: s
        cutlass, cute, runtime = (
            types.ModuleType(n) for n in ("cutlass", "cutlass.cute", "cutlass.cute.runtime")
        )
        cutlass.cute, cute.runtime, cutlass.BFloat16 = cute, runtime, "bf16"
        cute.experimental = SimpleNamespace(
            compile=lambda op, *args: calls.append("compile")
            or (lambda *a: calls.append("launch"))
        )
        runtime.from_dlpack = lambda t, assumed_align: t
        kernel = types.ModuleType(package.__name__ + ".mk_front_kernel")
        kernel.MkGdnFront = lambda **kw: SimpleNamespace(**kw)
        mods = (cuda, bindings, driver, cutlass, cute, runtime, kernel)
        mods = {m.__name__: m for m in mods}
        bf = lambda *shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        idx = lambda: torch.empty(1, dtype=torch.int32, device="meta")
        args = lambda: (
            bf(4, 2560), bf(16480, 2560), bf(10240, 4), None, bf(3, 10240, 3), idx(),
            bf(2, 4, 10240, 3), idx(),
        )
        cuda_fns = dict(
            is_current_stream_capturing=lambda: capturing[0],
            current_stream=lambda d: SimpleNamespace(cuda_stream=7),
            get_device_capability=lambda d: (10, 7),
        )
        self.addCleanup(lambda: vars(package).pop("mk_front", None))
        with mock.patch.dict(sys.modules, mods), mock.patch.multiple(torch.cuda, **cuda_fns):
            sys.modules.pop(package.__name__ + ".mk_front", None)
            front = importlib.import_module(package.__name__ + ".mk_front")
            packed, conv = front.run(*args(), pad_slot=-1)
            self.assertEqual(tuple(packed.shape) + tuple(conv.shape), (4, 16480, 4, 10240))
            capturing[0] = True
            front.run(*args(), pad_slot=-1)
            self.assertEqual(calls, ["compile", "launch", "launch"])
            with self.assertRaises(RuntimeError):
                front.run(*args(), pad_slot=0)  # a layout no eager step compiled
            self.assertEqual(calls, ["compile", "launch", "launch"])


class _State:  # a UcacheState double that records the calls
    state_bytes_per_slot = 0
    state_bytes_per_request_row = 123
    keeps_draft_ssm = tracks_decode_state = backs_up_state = False
    tokens_per_request = 4
    buffers = "rings"

    def __init__(self):
        self.calls = []

    def reset_state_slots(self, slots, request_rows):
        self.calls.append(("reset", slots, request_rows))

    def commit(self, io):
        self.calls.append(("commit", io))
        return True

    def prepare(self, view):
        self.calls.append(("prepare", view))


class TestOwnUcache(unittest.TestCase):
    def _executor(self):
        ex = Qwen3_8UcacheExecutor(
            _model(), widths=tuple(range(4, 1025, 4)), draft=False, slots=frozenset()
        )
        ex.ucache_state = _State()
        return ex

    def test_binding_protocols(self):
        try:
            from sglang.kernels.plugin_kernels.nvidia.spec import (
                SupportsRecurrentCommit,
                SupportsStateSlotReset,
            )
        except ImportError:
            self.skipTest("binding v2 (SupportsStateSlotReset) not in this SGLang")
        ex = self._executor()
        self.assertIsInstance(ex, SupportsStateSlotReset)
        self.assertIsInstance(ex, SupportsRecurrentCommit)
        self.assertNotIsInstance(_executor(), SupportsStateSlotReset)
        self.assertNotIsInstance(_executor(), SupportsRecurrentCommit)

    def test_delegates_to_state(self):
        ex = self._executor()
        self.assertEqual((ex.state_bytes_per_slot, ex.state_bytes_per_request_row), (0, 123))
        self.assertFalse(ex.keeps_draft_ssm or ex.tracks_decode_state)
        self.assertFalse(ex.backs_up_state)
        ex.reset_state_slots("slots", "rows")
        self.assertTrue(ex.commit_recurrent_state("io"))
        self.assertEqual(
            ex.ucache_state.calls, [("reset", "slots", "rows"), ("commit", "io")]
        )

    def test_prepare_builds_the_verify_once(self):
        ex = self._executor()
        built = []
        ex._build_ucache_verify = lambda mamba, buffers, tokens: (
            built.append((mamba, buffers, tokens)),
            setattr(ex, "gdn_ucache", "verify"),
        )
        view = SimpleNamespace(mamba="pools")
        runtime.prepare(ex, view)
        runtime.prepare(ex, view)  # a recapture repeats prepare
        self.assertEqual(built, [("pools", "rings", 4)])
        self.assertEqual(
            ex.ucache_state.calls, [("prepare", view), ("prepare", view)]
        )
        runtime.prepare(_executor(), view)  # no own u-cache: nothing to prepare

    def test_config_reasons(self):
        cfg = SimpleNamespace(
            mamba_ssm_dtype="float16",
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer_lazy",
            speculative_algorithm="NEXTN",
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            max_running_requests=8,
        )
        env = runtime.UCACHE_ENV
        old = runtime.os.environ.get(env)
        runtime.os.environ[env] = "1"
        try:
            self.assertEqual(runtime.config_reasons(cfg, None), [])
            cfg.mamba_ssm_dtype = "bfloat16"
            cfg.max_running_requests = 512  # 512 x 4 rows > MAX_ROWS
            self.assertEqual(len(runtime.config_reasons(cfg, None)), 2)
        finally:
            if old is None:
                del runtime.os.environ[env]
            else:
                runtime.os.environ[env] = old


class TestSpecs(unittest.TestCase):
    def test_target_and_draft(self):
        from sgl_nvidia_mega.qwen3_8 import spec

        self.assertIn("forward", spec.SPEC.name)
        self.assertIn("draft", spec.DRAFT_SPEC.name)
        self.assertIsNone(getattr(spec.DRAFT_SPEC, "prepare", None))
        if "prepare" in spec.MegakernelSpec.__struct_fields__:
            self.assertIs(spec.SPEC.prepare, runtime.prepare)


class TestEngagement(unittest.TestCase):
    def test_logs_each_new_set_once(self):
        e = Engagement("target")
        with self.assertLogs(
            "sgl_nvidia_mega.qwen3_8.runtime.engagement", logging.INFO
        ) as logs:
            for _ in range(3):
                e.begin()
                e.note("target_verify", 8, "gdn", "stock")
                e.note("target_verify", 8, "qsa", "stock")
                e.end("target_verify", 8)
            e.begin()
            e.note("target_verify", 8, "gdn", "gdn_core")
            e.end("target_verify", 8)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("gdn=stock; qsa=stock", logs.output[0])
        self.assertIn("gdn=gdn_core", logs.output[1])


if __name__ == "__main__":
    unittest.main()
