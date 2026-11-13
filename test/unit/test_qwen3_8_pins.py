# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's tuning-pin tables on CPU (runtime/flashinfer_tactic_pins.py,
runtime/bf16_gemm_pins.py): the file formats, the SHA-256 and metadata gates, which
autotuner entries the MoE decode table rewrites and which it refuses, when a BF16 pin
serves an executor projection call and when SGLang's dispatch keeps it. FlashInfer's
and SGLang's modules are fakes and the pinned kernels CPU stand-ins, so no GPU is
needed."""

import hashlib
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import torch
import torch.nn.functional as F
import triton  # noqa: F401 - imported after the custom-op test, it aborts the CPU process

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins as bp
from sgl_nvidia_mega.qwen3_8.runtime import flashinfer_tactic_pins as tp

BF16 = torch.bfloat16
SHAPE = (2560, 640, 512, 10)
META = {
    "flashinfer_version": "9.9.9",
    "cuda_version": "13.4",
    "cublas_version": "13.8.1",
    "gpu": "NVIDIA VR NVL72",
    "sm": "10.7",
}
SPLIT = [8, 54]


def _moe_document(**section):
    hidden, inter, experts, top_k = SHAPE
    body = dict(
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=experts,
        top_k=top_k,
        decode={"4": [8, 322], "8": [8, 352], "16": SPLIT, "32": [8, 231], "64": -1},
        prefill={"768": [32, 125]},
        split_k_tactics=[SPLIT],
    )
    body.update(section)
    return {"_metadata": dict(META), "trtllm_fp4_block_scale_moe": body}


def _extras(shape=SHAPE, shared=0):
    hidden, inter, experts, top_k = shape
    # get_cache_key_extras: op, top_k, local, global, shared, dtypes x3, hidden,
    # hidden_out, intermediate, activation, shuffled, layout, packed, per_token
    return ("Module", top_k, experts, experts, shared, 1, 1, 0, hidden, hidden,
            inter, 3, True, 0, False, False)  # fmt: skip


class _Key(NamedTuple):  # flashinfer's ProfilingCacheKey, the fields read
    custom_op: str
    nearest_profile: tuple
    extras: tuple


def _key(bucket, shape=SHAPE, op=tp.MOE_OP, shared=0):
    profile = ((bucket, 0), (bucket, 512), (bucket, 10), (bucket, 10), (bucket, 1280))
    return _Key(op, profile, _extras(shape, shared))


def _query(bucket, shape=SHAPE, layout=15):
    hidden, inter, experts, top_k = shape
    head = (1, 1, 0, top_k, hidden, hidden, inter, experts)
    tail = (3, True, 0, False, bucket, False)
    return head + ((experts,) if layout == 15 else ()) + tail


class _Files(unittest.TestCase):
    """Pin files in a temporary directory and a fresh process-wide state per test."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        for cached in (tp.read_tuning_file, tp._moe_pins_of, tp.file_digest,
                       tp._warn_inert_once):  # fmt: skip
            cached.cache_clear()
        self.addCleanup(tp._warn_inert_once.cache_clear)
        patcher = mock.patch.object(tp, "current_metadata", lambda: dict(self.here))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.here = dict(META)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in (tp.FILE_ENV, tp.SHA256_ENV, bp.FILE_ENV, bp.SHA256_ENV):
            os.environ.pop(var, None)

    def write(self, name, document):
        path = f"{self.dir.name}/{name}"
        text = json.dumps(document)
        with open(path, "w") as f:
            f.write(text)
        return path, hashlib.sha256(text.encode()).hexdigest()

    def set_env(self, var, value):
        os.environ[var] = value


class TestMoeFile(_Files):
    def test_tables_parse(self):
        pins = tp.parse_moe_tactic_pins(_moe_document())
        self.assertEqual(pins.shape, SHAPE)
        self.assertEqual(pins.decode[4], (8, 322))
        self.assertEqual(pins.decode[64], tp.DEFAULT_TACTIC)
        self.assertEqual(pins.prefill, {768: (32, 125)})
        self.assertEqual(pins.split_k, frozenset({tuple(SPLIT)}))

    def test_malformed_files_raise(self):
        for document in (
            _moe_document(decode={"24": [8, 1]}),  # not a power of two
            _moe_document(decode={"1024": [8, 1]}),  # past the decode buckets
            _moe_document(decode={"4": [0, 1]}),
            _moe_document(decode={"4": "8,1"}),
            _moe_document(split_k_tactics=[-1]),
            _moe_document(top_k="ten"),
            {"trtllm_fp4_block_scale_moe": {}},  # no metadata
        ):
            with self.assertRaises(ValueError):
                tp.parse_moe_tactic_pins(document)

    def test_digest_and_metadata_gate(self):
        path, digest = self.write("moe.json.py", _moe_document())
        self.assertIsNone(tp.active_moe_tactic_pins())  # unset: off
        self.set_env(tp.FILE_ENV, path)
        self.assertIsNotNone(tp.active_moe_tactic_pins())  # no digest given
        self.set_env(tp.SHA256_ENV, digest.upper())
        self.assertIsNotNone(tp.active_moe_tactic_pins())
        self.set_env(tp.SHA256_ENV, "0" * 64)
        with self.assertLogs(tp.logger, "WARNING") as logs:
            self.assertIsNone(tp.active_moe_tactic_pins())
        self.assertIn(digest, logs.output[0])
        self.set_env(tp.SHA256_ENV, digest)
        for field, value in (("flashinfer_version", "9.9.8"), ("sm", "10.0"),
                             ("cublas_version", "unknown")):  # fmt: skip
            self.here = dict(META, **{field: value})
            self.assertIsNone(tp.active_moe_tactic_pins(), field)
        self.here = dict(META)
        self.assertIsNotNone(tp.active_moe_tactic_pins())


class TestMoeSettle(unittest.TestCase):
    def setUp(self):
        self.pins = tp.parse_moe_tactic_pins(_moe_document())

    def test_valid_tactics_by_query_layout(self):
        valid = {
            (1, _query(4)): [[8, 322], [8, 1]],
            (1, _query(8, layout=14)): [[8, 352]],
            (1, _query(16, shape=(4096, 640, 512, 10))): [[8, 9]],  # other shape
            (1, _query(32)[:-3]): [[8, 231]],  # unknown layout
        }
        got = tp.valid_tactics_by_bucket(valid, SHAPE)
        self.assertEqual(got, {4: {(8, 322), (8, 1)}, 8: {(8, 352)}})

    def test_listed_buckets_take_their_pins(self):
        keys = {b: _key(b) for b in (4, 8, 16, 32, 64, 128)}
        winners = {k: ([8, 1], f"p{b}") for b, k in keys.items()}
        other = _key(4, shape=(4096, 640, 512, 10))
        shared = _key(8, shared=1)
        not_moe = _key(32, op="flashinfer::mm_bf16")
        winners.update({other: ([8, 1], "o"), shared: ([8, 1], "s"),
                        not_moe: ([8, 1], "m")})  # fmt: skip
        valid = {4: {(8, 322)}, 16: {tuple(SPLIT)}, 32: {(8, 1)}}
        rows = tp.settle_decode_tactics(winners, pins=self.pins, valid=valid)
        sources = {(b, s) for b, _, _, s in rows}
        self.assertEqual(
            sources,
            {
                (4, "pin"),
                (4, "pin_shape_skip"),
                (8, "pin_invalid"),
                (8, "pin_shape_skip"),
                (16, "pin_split_k_refused"),
                (32, "pin_invalid"),
                (64, "pin_default"),
            },  # fmt: skip
        )
        self.assertEqual(winners[keys[4]], ([8, 322], "p4"))
        self.assertEqual(winners[keys[64]], (tp.DEFAULT_TACTIC, "p64"))
        for b in (8, 16, 32, 128):  # refused or unlisted: the tuner's pick stands
            self.assertEqual(winners[keys[b]], ([8, 1], f"p{b}"))
        for k in (other, shared, not_moe):
            self.assertEqual(winners[k][0], [8, 1])


class _FakeTuner:
    def __init__(self, winners):
        self._lock = threading.RLock()
        self.is_tuning_mode = False
        self.winners = winners

    def _winner_cache(self):
        return self.winners


class TestMoeAfterForward(_Files):
    def setUp(self):
        super().setUp()
        path, _ = self.write("moe.json.py", _moe_document())
        self.set_env(tp.FILE_ENV, path)
        self.key = _key(4)
        self.tuner = _FakeTuner({self.key: ([8, 1], "p")})
        runner = SimpleNamespace(valid_tactics_dict={(1, _query(4)): [[8, 322]]})
        fakes = {
            "flashinfer.autotuner": dict(
                AutoTuner=SimpleNamespace(get=lambda: self.tuner)
            ),
            "flashinfer.fused_moe.backends.trtllm.sm100_runner": dict(MoERunner=runner),
        }
        for name, attrs in fakes.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            patcher = mock.patch.dict(sys.modules, {name: module})
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pins_only_inside_a_tuning_pass(self):
        tp.after_forward()
        self.assertEqual(self.tuner.winners[self.key][0], [8, 1])
        self.tuner.is_tuning_mode = True
        tp.after_forward()
        self.assertEqual(self.tuner.winners[self.key], ([8, 322], "p"))

    def test_a_file_for_another_stack_pins_nothing(self):
        self.here = dict(META, gpu="NVIDIA B200")
        self.tuner.is_tuning_mode = True
        self.assertEqual(tp.pin_decode_moe_tactics(), 0)
        self.assertEqual(self.tuner.winners[self.key][0], [8, 1])

    def test_build_checks_the_model_shape(self):
        model = object()
        with mock.patch.object(tp, "model_shape", lambda m: SHAPE):
            self.assertEqual(tp.build(model), "")
        with mock.patch.object(tp, "model_shape", lambda m: (4096, 640, 512, 10)):
            self.assertIn("this model's is (4096", tp.build(model))
        self.here = dict(META, sm="10.0")
        with mock.patch.object(tp, "model_shape", lambda m: SHAPE):
            self.assertIn("does not apply here", tp.build(model))


# -- BF16 GEMM pins ------------------------------------------------------------------

N, K = 2560, 6144  # o_proj


def _gemm(x, w):
    return (x.float() @ w.float().T).to(BF16)


class _Unquant:
    pass


class _Linear:
    def __init__(self, n=N, k=K, bias=None, quant=_Unquant):
        g = torch.Generator().manual_seed(n + k)
        self.weight = (torch.randn(n, k, generator=g) * 0.02).to(BF16)
        self.bias = bias
        self.quant_method = quant()
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return _gemm(x, self.weight), None


def _bf16_document(entries):
    return {"_metadata": dict(META), "bf16_gemm_pins": entries}


ALGO = "ab" * 64


class _Gemm(_Files):
    def setUp(self):
        super().setUp()
        self.state = dict(capturing=False, backend="cutedsl", tp=1)
        for name, value in (("_state", None), ("_runners", {}), ("_routed", False)):
            patcher = mock.patch.object(bp, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        backend = SimpleNamespace(
            is_cutedsl=lambda: self.state["backend"] == "cutedsl",
            is_gemv=lambda: self.state["backend"] == "gemv",
        )
        self.stock_calls = []

        def stock(x, w, bias):
            self.stock_calls.append(x.shape[0])
            return _gemm(x, w)

        fakes = {
            "sglang.srt.layers.quantization.unquant": dict(
                UnquantizedLinearMethod=_Unquant,
                get_bf16_gemm_backend=lambda: backend,
                _bf16_gemm_dispatch_impl=stock,
            ),
            "sglang.srt.batch_invariant_ops": dict(
                is_batch_invariant_mode_enabled=lambda: False
            ),
            "sglang.srt.runtime_context": dict(
                get_parallel=lambda: SimpleNamespace(tp_size=self.state["tp"])
            ),
        }
        for name, attrs in fakes.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            patcher = mock.patch.dict(sys.modules, {name: module})
            patcher.start()
            self.addCleanup(patcher.stop)
        # CPU tensors stand in for CUDA ones
        for target, attr, value in (
            (torch.Tensor, "is_cuda", property(lambda t: True)),
            (
                torch.cuda,
                "is_current_stream_capturing",
                lambda: self.state["capturing"],
            ),
        ):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.flip = set()  # kinds whose stand-in differs from stock in one bit
        self.runs = []

        def runner(kind):
            def run(x, w, out=None):
                self.runs.append((kind, x.shape[0]))
                got = _gemm(x, w)
                if kind in self.flip:
                    got.view(-1).view(torch.int16)[0] ^= 1
                return got if out is None else out.copy_(got)

            return run

        for name, kind in (("_cublaslt_runner", "cublaslt"), ("_tgv_runner", "tgv")):
            patcher = mock.patch.object(bp, name, lambda *a, kind=kind: runner(kind))
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_module(self, name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        patcher = mock.patch.dict(sys.modules, {name: module})
        patcher.start()
        self.addCleanup(patcher.stop)
        return module

    def install(self, entries, digest_env=None):
        path, digest = self.write("gemm.json.py", _bf16_document(entries))
        self.set_env(bp.FILE_ENV, path)
        if digest_env is not None:
            self.set_env(bp.SHA256_ENV, digest_env)
        return path, digest


def _x(m, k=K, seed=3):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(BF16)


class TestGemmFile(_Gemm):
    def test_entries_parse(self):
        entries = bp._parse(_bf16_document([
            {"m": 36, "n": N, "k": K, "kind": "cublaslt", "algo": ALGO, "workspace": 0},
            {"m": 40, "n": N, "k": K, "kind": "tgv", "tactic": 21},
        ]), "f")  # fmt: skip
        self.assertEqual(entries[(36, N, K)], ("cublaslt", bytes.fromhex(ALGO), 0))
        self.assertEqual(entries[(40, N, K)], ("tgv", 21))

    def test_malformed_entries_raise(self):
        good = {"m": 36, "n": N, "k": K, "kind": "tgv", "tactic": 1}
        for entries in (
            [dict(good, kind="cublas")],
            [dict(good, m=0)],
            [{"m": 36, "n": N, "k": K, "kind": "cublaslt", "algo": "ab"}],
            [good, dict(good, tactic=2)],  # duplicate key
        ):
            with self.assertRaises(ValueError):
                bp._parse(_bf16_document(entries), "f")

    def test_digest_and_metadata_gate(self):
        entries = [{"m": 36, "n": N, "k": K, "kind": "tgv", "tactic": 1}]
        _, digest = self.install(entries, digest_env="f" * 64)
        self.assertFalse(bp.enabled())
        bp._state = None
        self.set_env(bp.SHA256_ENV, digest)
        self.here = dict(META, cuda_version="13.0")
        self.assertFalse(bp.enabled())
        bp._state = None
        self.here = dict(META)
        self.assertTrue(bp.enabled())
        self.assertTrue(bp.has_shape(N, K))


class TestGemm(_Gemm):
    def setUp(self):
        super().setUp()
        self.install([
            {"m": 36, "n": N, "k": K, "kind": "cublaslt", "algo": ALGO},
            {"m": 40, "n": N, "k": K, "kind": "tgv", "tactic": 21},
        ])  # fmt: skip
        self.proj = _Linear()

    def stock(self, x, w):
        return _gemm(x, w)

    def test_a_listed_key_is_checked_once_then_served(self):
        for _ in range(2):
            out = bp.gemm(_x(36), self.proj.weight, self.stock)
            self.assertTrue(torch.equal(out, _gemm(_x(36), self.proj.weight)))
        # the check (seeded input) and two serves; the check ran once
        self.assertEqual(self.runs, [("cublaslt", 36)] * 3)

    def test_a_pin_that_differs_keeps_stock_for_good(self):
        self.flip.add("tgv")
        with self.assertLogs(bp.logger, "WARNING"):
            self.assertIsNone(bp.gemm(_x(40), self.proj.weight, self.stock))
        self.flip.clear()
        self.assertIsNone(bp.gemm(_x(40), self.proj.weight, self.stock))
        self.assertEqual(self.runs, [("tgv", 40)])

    def test_first_seen_under_capture_keeps_stock_in_that_graph(self):
        self.state["capturing"] = True
        with self.assertLogs(bp.logger, "WARNING"):
            self.assertIsNone(bp.gemm(_x(36), self.proj.weight, self.stock))
        self.assertEqual(self.runs, [])
        self.state["capturing"] = False
        self.assertIsNotNone(bp.gemm(_x(36), self.proj.weight, self.stock))

    def test_other_calls_take_stock(self):
        w = self.proj.weight
        for x, weight in (
            (_x(44), w),  # M not listed
            (_x(36, k=N), _Linear(N, N).weight),  # shape not listed
            (_x(36).half(), w),
            (_x(K, k=36).t(), w),  # not contiguous
        ):
            self.assertIsNone(bp.gemm(x, weight, self.stock))
        self.assertEqual(self.runs, [])


class TestExecutorCalls(_Gemm):
    def setUp(self):
        super().setUp()
        self.install([
            {"m": 36, "n": N, "k": K, "kind": "tgv", "tactic": 1},
            {"m": 36, "n": 16480, "k": 2560, "kind": "tgv", "tactic": 2},
        ])  # fmt: skip

    def test_linear_serves_only_once_routed(self):
        proj = _Linear()
        out = bp.linear(proj, _x(36))
        self.assertEqual((proj.calls, self.runs), (1, []))
        bp.enable_routes()
        out = bp.linear(proj, _x(36))
        self.assertTrue(torch.equal(out, _gemm(_x(36), proj.weight)))
        self.assertEqual(proj.calls, 1)
        self.assertEqual(self.runs, [("tgv", 36)] * 2)
        # the stock dispatch ran once, for the check
        self.assertEqual(self.stock_calls, [36])

    def test_linear_keeps_the_module_where_sglang_skips_its_dispatch(self):
        bp.enable_routes()
        cases = [
            (_Linear(bias=torch.zeros(N, dtype=BF16)), "cutedsl"),
            (_Linear(quant=type("Fp8", (), {})), "cutedsl"),
            (_Linear(), "torch"),  # cuBLAS through F.linear: no dispatch
            (_Linear(), "gemv"),
        ]
        for proj, backend in cases:
            self.state["backend"] = backend
            bp.linear(proj, _x(36))
        self.assertEqual([p.calls for p, _ in cases], [1, 1, 1, 0])

    def test_in_proj_splits_the_packed_output(self):
        g = torch.Generator().manual_seed(5)
        weight = (torch.randn(16480, 2560, generator=g) * 0.02).to(BF16)
        gdn = SimpleNamespace(
            _fused_in_proj_weight=weight, _fused_in_proj_qkvz_width=16384, calls=0
        )

        def method(x):
            gdn.calls += 1
            out = _gemm(x, weight)
            return out[:, :16384], out[:, 16384:]

        gdn._forward_input_proj = method
        bp.enable_routes()
        x = _x(36, k=2560)
        qkvz, ba = bp.in_proj(gdn, x)
        want = _gemm(x, weight)
        self.assertTrue(torch.equal(qkvz, want[:, :16384]))
        self.assertTrue(torch.equal(ba, want[:, 16384:]))
        self.assertEqual(gdn.calls, 0)
        bp.in_proj(gdn, _x(40, k=2560))  # no pin at 40 rows
        gdn._fused_in_proj_weight = None
        bp.in_proj(gdn, x)
        self.assertEqual(gdn.calls, 2)

    def test_build(self):
        sites = {(N, K): "gdn.out_proj/qsa.o_proj"}
        with self.assertLogs(bp.logger, "WARNING") as logs:
            self.assertEqual(bp.build(object(), sites), "")
        self.assertIn("(16480, 2560)", logs.output[0])  # no executor call takes it
        self.assertIn("no executor projection", bp.build(object(), {}))
        self.state["tp"] = 2
        self.assertIn("TP1 only", bp.build(object(), sites))


class TestQkvGate(_Gemm):
    """attention._qkv_gate: SGLang's _prepare_qkv_gate without pins; with them its CUDA
    gated branch, the projection through ``linear``."""

    HEADS, KV_HEADS, DIM = 2, 1, 4

    def setUp(self):
        super().setUp()
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        self.attention = attention
        self.install([{"m": 6, "n": N, "k": K, "kind": "tgv", "tactic": 1}])
        self.fused = []
        rows_out = lambda t, heads: torch.zeros(t.shape[0], heads, self.DIM)

        def fused(q_gate, k, *args, has_gate, mrope_axis_map):
            self.fused.append((q_gate, k, args, has_gate, mrope_axis_map))
            heads = self.HEADS
            return (
                rows_out(q_gate, heads),
                rows_out(k, self.KV_HEADS),
                rows_out(q_gate, heads),
            )

        name = "sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate"
        module = types.ModuleType(name)
        module.fused_qk_gemma_rmsnorm_rope_gate = fused
        patcher = mock.patch.dict(sys.modules, {name: module})
        patcher.start()
        self.addCleanup(patcher.stop)
        q, kv = self.HEADS * self.DIM, self.KV_HEADS * self.DIM
        norm = lambda: SimpleNamespace(
            weight=torch.ones(self.DIM), variance_epsilon=1e-6
        )
        self.layer = SimpleNamespace(
            attn_output_gate=True,
            qkv_proj=_Linear(n=2 * q + 2 * kv, k=16),
            q_size=q,
            kv_size=kv,
            q_norm=norm(),
            k_norm=norm(),
            rotary_emb=SimpleNamespace(
                cos_sin_cache="cache", rotary_dim=4, axis_map="axes"
            ),
            num_heads=self.HEADS,
            num_kv_heads=self.KV_HEADS,
            head_dim=self.DIM,
            _prepare_qkv_gate=mock.Mock(return_value="sglang"),
        )

    def test_pins_off_runs_sglangs_method(self):
        got = self.attention._qkv_gate(self.layer, _x(6, k=16), torch.arange(6), "fb")
        self.assertEqual(got, "sglang")
        self.assertEqual(self.fused, [])

    def test_pins_on_runs_the_cuda_branch(self):
        bp.enable_routes()
        x = _x(6, k=16)
        for positions, axes in ((torch.arange(6), None), (torch.zeros(3, 6), "axes")):
            q, k, v, gate = self.attention._qkv_gate(self.layer, x, positions, "fb")
            q_gate, k_in, args, has_gate, mrope = self.fused[-1]
            qkv = _gemm(x, self.layer.qkv_proj.weight)
            self.assertTrue(torch.equal(q_gate, qkv[:, :16]))
            self.assertTrue(torch.equal(k_in, qkv[:, 16:20]))
            self.assertTrue(torch.equal(v, qkv[:, 20:]))
            self.assertEqual(args[2], "cache")
            self.assertIs(args[3], positions)
            self.assertEqual(args[4:], (1e-6, 2, 1, 4, 4))
            self.assertEqual((has_gate, mrope), (True, axes))
            self.assertEqual(
                [tuple(t.shape) for t in (q, k, gate)], [(6, 8), (6, 4), (6, 8)]
            )
        self.layer._prepare_qkv_gate.assert_not_called()


class _LogitsProcessor:
    """The fake LogitsProcessor: forward computes the lm_head GEMM, then the softcap."""

    def __init__(self):
        self.use_fp32_lm_head = False
        self.use_fp32_lm_head_output = False
        self.rl_on_policy_target = None
        self.softcap = 2.0
        self.stock_calls = 0

    def __call__(self, hidden, lm_head, out=None):
        logits = self._compute_lm_head(hidden, lm_head, None, out=out)
        return logits * self.softcap

    def _compute_lm_head(self, hidden, lm_head, embedding_bias=None, *, out=None):
        self.stock_calls += 1
        logits = _gemm(hidden, lm_head.weight)  # CPU stand-in for the BF16 matmul
        if out is not None and out.dtype == lm_head.weight.dtype:
            return out.copy_(logits)
        return logits


class TestLmHead(_Gemm):
    V = 512

    def setUp(self):
        super().setUp()
        self.fake_module(
            "sglang.srt.layers.logits_processor",
            LogitsProcessor=_LogitsProcessor,
            should_apply_lm_head_quant_method=lambda lm_head, qm: False,
            use_intel_amx_backend=lambda lm_head: False,
            _aiter_lm_head_ok=lambda h, w: False,
        )
        self.install([{"m": 6, "n": self.V, "k": 16, "kind": "tgv", "tactic": 1}])
        patcher = mock.patch.object(bp, "_lm_head_matmul", _gemm)
        patcher.start()
        self.addCleanup(patcher.stop)
        g = torch.Generator().manual_seed(9)
        self.lm_head = SimpleNamespace(
            weight=torch.randn(self.V, 16, generator=g).to(BF16)
        )
        self.processor = _LogitsProcessor()

    def test_unrouted_keeps_the_processor(self):
        self.assertIs(bp.logits_processor(self.processor), self.processor)

    def test_the_view_pins_the_gemm_and_keeps_the_rest(self):
        bp.enable_routes()
        view = bp.logits_processor(self.processor)
        self.assertIsNot(view, self.processor)
        self.assertIs(bp.logits_processor(self.processor), view)
        self.processor.softcap = 3.0  # state is shared
        x = _x(6, k=16)
        logits = _gemm(x, self.lm_head.weight)
        buffer = torch.empty(6, self.V, dtype=BF16)
        for out in (None, buffer):
            self.assertTrue(torch.equal(view(x, self.lm_head, out=out), logits * 3.0))
        self.assertTrue(torch.equal(buffer, logits))  # the pin wrote the buffer
        self.assertEqual(self.runs, [("tgv", 6)] * 3)  # check, then two serves
        self.assertEqual(self.processor.stock_calls, 0)
        view(_x(8, k=16), self.lm_head)  # no pin at 8 rows
        self.processor.use_fp32_lm_head_output = True
        view(x, self.lm_head)  # another branch of _compute_lm_head
        self.assertEqual(self.processor.stock_calls, 2)


class _GatedResidual:
    def __init__(self, hc=2, hidden=8, lowrank=4, jit=True):
        g = torch.Generator().manual_seed(4)
        self.hc_count, self.hidden_size, self.params_dtype = hc, hidden, BF16
        self.config = SimpleNamespace(hc_lowrank=lowrank, hc_per_branch_norm=True)
        self.input_mix_weight_down = SimpleNamespace(
            weight=torch.randn(lowrank, hc * hidden, generator=g).to(BF16)
        )
        self.input_mix_weight_up = SimpleNamespace(
            weight=torch.randn(hc * hidden, lowrank, generator=g).to(BF16)
        )
        self.hc_norm = lambda t: t * 2
        self._jit_mix_ok = jit
        self._mix_up_weight_padded = None
        self._gate_stream = None
        self._flydsl_mix_ok = lambda: False
        self.mix = mock.Mock(return_value="sglang")


class TestGatedMix(_Gemm):
    def setUp(self):
        super().setUp()
        self.routes = []
        self.fused = {"rows": 16}
        self.fake_module(
            "sglang.srt.layers.hyperconnection",
            flydsl_hc_mix_supported=lambda t, ok: False,
            fused_hc_mix_supported=lambda t, d, u: t.shape[0] <= self.fused["rows"],
            fused_hc_mix=lambda t, d, u, hc, hs: self.route("fused", t, hs),
        )
        self.fake_module(
            "sglang.kernels.ops.gemm.hc_mix_splitk",
            hc_mix=lambda t, d, u, hc, hs: self.route("splitk", t, hs),
            permute_pad_up_weight=lambda w, hc: "padded",
        )
        self.gr = _GatedResidual()
        self.install([
            {"m": 28, "n": 4, "k": 16, "kind": "tgv", "tactic": 1},
            {"m": 28, "n": 16, "k": 4, "kind": "tgv", "tactic": 2},
        ])  # fmt: skip
        from sgl_nvidia_mega.qwen3_8.runtime import hc_mix_tuned

        self.tuned = {"on": False}
        for name, value in (
            ("enabled", lambda: self.tuned["on"]),
            ("hc_mix", lambda t, d, u, hc, hs: self.route("tuned", t, hs)),
        ):
            patcher = mock.patch.object(hc_mix_tuned, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def route(self, name, t, hs):
        self.routes.append((name, t.shape[0]))
        return torch.zeros(t.shape[0], hs)

    def test_unrouted_or_unpinned_is_sglangs_mix(self):
        self.assertEqual(bp.gated_mix(self.gr, _x(28, k=16)), "sglang")
        bp.enable_routes()
        other = _GatedResidual(lowrank=8)
        self.assertEqual(bp.gated_mix(other, _x(28, k=16)), "sglang")

    def test_routes_in_sglangs_order(self):
        bp.enable_routes()
        compute = mock.Mock(
            side_effect=lambda t, d, u, hc, hs: self.route("pinned", t, hs)
        )
        with mock.patch.object(bp, "_pinned_mix_compute", lambda: compute):
            for rows in (8, 20, 28):
                x = _x(rows, k=16)
                mixed, (res, normed, partials) = bp.gated_mix(self.gr, x)
                self.assertIs(res, x)
                self.assertTrue(torch.equal(normed, x * 2))
                self.assertIsNone(partials)
            self.gr._jit_mix_ok = False
            bp.gated_mix(self.gr, _x(8, k=16), normalized_input=_x(8, k=16))
        self.assertEqual(
            self.routes, [("splitk", 8), ("splitk", 20), ("pinned", 28), ("fused", 8)]
        )
        self.gr.mix.assert_not_called()

    def test_the_sources_tuned_tactics_serve_the_jit_rows(self):
        # SGLANG_HC_MACRO (on by default) at TP1: the source's split-K mix rows (<= 32,
        # SGLang's <= 24) run its tactics, pinned or not; wider rows keep SGLang's mix
        # when unpinned and the pinned order when pinned
        self.tuned["on"] = True
        mixed, (res, normed, partials) = bp.gated_mix(self.gr, _x(12, k=16))
        self.assertTrue(torch.equal(normed, _x(12, k=16) * 2))
        bp.gated_mix(self.gr, _x(28, k=16))
        bp.gated_mix(self.gr, _x(32, k=16))
        self.assertEqual(bp.gated_mix(self.gr, _x(36, k=16)), "sglang")
        self.assertEqual(self.routes, [("tuned", 12), ("tuned", 28), ("tuned", 32)])
        bp.enable_routes()
        self.fused["rows"] = 40
        bp.gated_mix(self.gr, _x(36, k=16))
        self.assertEqual(self.routes[-1], ("fused", 36))

    def test_the_pinned_compute_is_mix_compute_on_the_pins(self):
        bp.enable_routes()
        with mock.patch.object(torch, "compile", lambda f: f):
            compute = bp._pinned_mix_compute()
        x = _x(28, k=16)
        down, up = (
            self.gr.input_mix_weight_down.weight,
            self.gr.input_mix_weight_up.weight,
        )
        got = compute(x, down, up, 2, 8)
        mix = torch.sigmoid(_gemm((F.silu(_gemm(x, down) / 2)), up)).unflatten(
            -1, (2, 8)
        )
        want = (mix * x.unflatten(-1, (2, 8))).mean(dim=-2)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(self.runs, [("tgv", 28)] * 4)  # each GEMM: check, then serve

    def test_layer_reads_and_the_final_mix(self):
        layer = SimpleNamespace(
            ple=None,
            _hc_norms=SimpleNamespace(attn_input="a", ffn_input="f"),
            attn_hyper_connection="attn_hc",
            mlp_hyper_connection="mlp_hc",
            _attn_mix=mock.Mock(return_value="stock attn"),
            _ffn_mix=mock.Mock(return_value="stock ffn"),
        )
        self.assertEqual(bp.attn_mix(layer, "s"), "stock attn")
        self.assertEqual(bp.ffn_mix(layer, "s"), "stock ffn")
        bp.enable_routes()
        calls = []
        with mock.patch.object(
            bp,
            "gated_mix",
            lambda gr, x, n: calls.append((gr, x, n)) or ("mixed", None),
        ):
            bp.attn_mix(layer, "s")
            bp.ffn_mix(layer, "s")
            self.assertEqual(bp.MixStreams("final", "n")("s"), "mixed")
        self.assertEqual(
            calls, [("attn_hc", "s", "a"), ("mlp_hc", "s", "f"), ("final", "s", "n")]
        )


class TestMoeMirrors(_Gemm):
    def setUp(self):
        super().setUp()
        self.fake_module(
            "sglang.srt.models.qwen2_moe",
            TopKOutputChecker=SimpleNamespace(format_is_bypassed=lambda t: t == "topk"),
            can_run_silu_fp4_quant_fusion=lambda module, gate_up: False,
            silu_and_mul_fp4_quant=None,
        )
        self.install([
            {"m": 6, "n": 512, "k": 16, "kind": "tgv", "tactic": 1},
            {"m": 6, "n": 8, "k": 16, "kind": "tgv", "tactic": 2},
        ])  # fmt: skip
        shared = SimpleNamespace(
            gate_up_proj=_Linear(n=8, k=16),
            down_proj=_Linear(n=16, k=4),
            act_fn=lambda t: t[:, :4] * t[:, 4:],
        )
        self.mlp = SimpleNamespace(
            gate=_Linear(n=512, k=16),
            topk=mock.Mock(return_value="topk"),
            supports_deferred_finalize=True,
            experts=SimpleNamespace(
                forward_deferred_finalize=mock.Mock(return_value="d")
            ),
            shared_expert=shared,
            _forward_router_experts=mock.Mock(return_value="stock router"),
            _forward_shared_experts=mock.Mock(return_value="stock shared"),
        )

    def test_unrouted_calls_sglang(self):
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        x = _x(6, k=16)
        self.assertEqual(moe._router_experts(self.mlp, x), "stock router")
        self.assertEqual(moe._shared_experts(self.mlp, x), "stock shared")
        self.mlp._forward_router_experts.assert_called_once_with(x, defer_finalize=True)
        self.mlp._forward_shared_experts.assert_called_once_with(x, apply_gate=False)

    def test_routed_runs_the_mirrors_on_the_pins(self):
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        bp.enable_routes()
        x = _x(6, k=16)
        self.assertEqual(moe._router_experts(self.mlp, x), "d")
        logits = self.mlp.topk.call_args.args[1]
        self.assertTrue(torch.equal(logits, _gemm(x, self.mlp.gate.weight)))
        shared = self.mlp.shared_expert
        gate_up = _gemm(x, shared.gate_up_proj.weight)
        want = _gemm(gate_up[:, :4] * gate_up[:, 4:], shared.down_proj.weight)
        self.assertTrue(torch.equal(moe._shared_experts(self.mlp, x), want))
        self.assertEqual(self.runs, [("tgv", 6)] * 4)  # gate and gate_up pinned
        self.assertEqual((self.mlp.gate.calls, shared.gate_up_proj.calls), (0, 0))
        self.assertEqual(shared.down_proj.calls, 1)  # no pin at (16, 4)
        self.mlp.topk.return_value = "standard"
        with self.assertRaises(RuntimeError):
            moe._router_experts(self.mlp, x)


class TestProjectQk(_Gemm):
    HEADS, KV, DIM = 2, 1, 4

    def setUp(self):
        super().setUp()
        n = (self.HEADS + self.KV) * self.DIM
        self.install([{"m": 6, "n": n, "k": 16, "kind": "tgv", "tactic": 1}])
        self.indexer = SimpleNamespace(
            index_qk_proj=_Linear(n=n, k=16),
            index_n_heads=self.HEADS,
            index_kv_heads=self.KV,
            index_head_dim=self.DIM,
            _use_fused_prep=lambda qk: False,
            q_layernorm=lambda t: t * 2,
            apply_rope=lambda positions, q: q + 1,
            project_qk=mock.Mock(return_value="stock"),
        )

    def test_mirror(self):
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer

        x, positions = _x(6, k=16), torch.arange(6)
        self.assertEqual(qsa_indexer._project_qk(self.indexer, x, positions), "stock")
        bp.enable_routes()
        pool = SimpleNamespace(qsa_compressed_dtype=torch.float32)
        q, token_k, stored = qsa_indexer._project_qk(
            self.indexer, x, positions, pool=pool
        )
        qk = _gemm(x, self.indexer.index_qk_proj.weight)
        want_q = (qk[:, :8].reshape(-1, 4) * 2).reshape(-1, 2, 4) + 1
        self.assertTrue(torch.equal(q, want_q.float()))
        self.assertTrue(torch.equal(token_k, qk[:, 8:].reshape(-1, 1, 4)))
        self.assertFalse(stored)
        self.assertEqual(self.indexer.index_qk_proj.calls, 0)


class TestStockStepMirrors(_Gemm):
    """gdn.forward and attention._self_attention: SGLang's stock GDN and QSA steps with
    their projections on the pins."""

    def test_gdn_forward(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn as gdn_step

        def split(qkvz, ba, nk, nv, dk, dv):
            rows = qkvz.shape[0]
            return qkvz[:, :6], qkvz[:, 6:10].reshape(rows, 2, 2), ba[:, :2], ba[:, 2:]

        self.fake_module(
            "sglang.srt.models.qwen3_5",
            _GDN_FUSED_QKVZBA_RATIOS=(2,),
            _fp8_tuple_input=lambda linear, rows: False,
            _gdn_decode_fused_proj_conv=False,
            _is_cpu=False,
            _is_cuda=True,
            _is_xpu=False,
            _layer_norm_fwd=None,
            fused_qkvzba_split_reshape_cat_contiguous=split,
            get_exec=None,
            qwen3_5_gdn_prefill_projection_views=None,
        )
        self.install([
            {"m": 6, "n": 14, "k": 16, "kind": "tgv", "tactic": 1},
            {"m": 6, "n": 16, "k": 4, "kind": "tgv", "tactic": 2},
        ])  # fmt: skip
        bp.enable_routes()
        g = torch.Generator().manual_seed(11)
        layer = SimpleNamespace(
            num_k_heads=1,
            num_v_heads=2,
            attn_tp_size=1,
            head_k_dim=2,
            head_v_dim=2,
            _fused_in_proj_weight=torch.randn(14, 16, generator=g).to(BF16),
            _fused_in_proj_qkvz_width=10,
            _forward_input_proj=mock.Mock(),
            attn=lambda fb, mixed_qkv, a, b: mixed_qkv[:, :4].reshape(-1, 2, 2),
            norm=lambda core, z: core * z,
            out_proj=_Linear(n=16, k=4),
        )
        mode = SimpleNamespace(is_target_verify=lambda: False, is_decode=lambda: True)
        fb = SimpleNamespace(
            forward_mode=mode,
            logical_forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: False
            ),
        )
        x = _x(6, k=16)
        out = gdn_step.forward(layer, x, fb)
        packed = _gemm(x, layer._fused_in_proj_weight)
        core = packed[:, :4] * packed[:, 6:10]
        self.assertTrue(torch.equal(out, _gemm(core, layer.out_proj.weight)))
        layer._forward_input_proj.assert_not_called()
        self.assertEqual(layer.out_proj.calls, 0)
        self.assertEqual(self.runs, [("tgv", 6)] * 4)

    def test_self_attention(self):
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        self.fake_module(
            "sglang.srt.models.qwen4_exp",
            _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD=1024,
            _shards_rows_over_cp=lambda fb: False,
            fused_sigmoid_mul=None,
            get_attn_backend=None,
            get_is_capture_mode=lambda: False,
            is_in_breakable_cuda_graph=lambda: False,
        )
        self.install([{"m": 6, "n": N, "k": K, "kind": "tgv", "tactic": 1}])
        bp.enable_routes()
        attn_out = _x(6)
        layer = SimpleNamespace(
            is_qsa=True,
            alt_stream=None,
            attn_output_gate=False,
            _prepare_qkv_gate=mock.Mock(return_value=("q", "k", "v", None)),
            _compute_qsa_topk_indices=mock.Mock(return_value="idx"),
            attn=mock.Mock(return_value=attn_out),
            o_proj=_Linear(),
        )
        x = _x(6, k=16)
        out = attention._self_attention(layer, x, "pos", "fb")
        self.assertTrue(torch.equal(out, _gemm(attn_out, layer.o_proj.weight)))
        layer.attn.assert_called_once_with("q", "k", "v", "fb", topk_indices="idx")
        layer._compute_qsa_topk_indices.assert_called_once_with(x, "pos", "fb")
        self.assertEqual(layer.o_proj.calls, 0)


if __name__ == "__main__":
    unittest.main()
