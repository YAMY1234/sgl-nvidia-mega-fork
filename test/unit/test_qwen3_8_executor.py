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

    def test_gdn_stock_outside_verify(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = SimpleNamespace(gdn_pdl_ready=True, gdn_ucache=None, gdn_prefill=None)
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        st = _St(_Mode.EXTEND)
        self.assertEqual(gdn.step(ex, layer, torch.zeros(8, 4), st), "gdn-stock")
        self.assertEqual(st.ran_ops, [("gdn", "stock")])

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
