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
            def run(x, w):
                self.runs.append((kind, x.shape[0]))
                out = _gemm(x, w)
                if kind in self.flip:
                    out.view(-1).view(torch.int16)[0] ^= 1
                return out

            return run

        for name, kind in (("_cublaslt_runner", "cublaslt"), ("_tgv_runner", "tgv")):
            patcher = mock.patch.object(bp, name, lambda *a, kind=kind: runner(kind))
            patcher.start()
            self.addCleanup(patcher.stop)

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


if __name__ == "__main__":
    unittest.main()
