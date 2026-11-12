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
from sgl_nvidia_mega.qwen3_8.runtime.executor import Qwen38Executor  # noqa: E402


def _model(layers=(_Linear, _Linear, _Attention)):
    m = SimpleNamespace(
        layers=[cls() for cls in layers], start_layer=0, end_layer=len(layers)
    )
    return SimpleNamespace(model=m)


def _batch(mode, rows):
    return SimpleNamespace(forward_mode=mode, input_ids=torch.zeros(rows))


def _executor(draft=False):
    widths = tuple(range(1, 1025)) if draft else tuple(range(4, 1025, 4))
    return Qwen38Executor(_model(), widths=widths, draft=draft, slots=frozenset())


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
        self.assertEqual(names, ["_gdn", "_gdn", "_qsa_stock"])


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
