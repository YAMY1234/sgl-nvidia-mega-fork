# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Kimi K3.1 split DSA selection on CPU: the merge storage comes from SGLang's
``DcpTopKMerge.create`` (which may decline) and the replicated split launch gets
that storage."""

import ast
import contextlib
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k31/dsa"
# split.py imports its kernel module as ..sm103.dsa.split_merge
KERNEL = "split_test.sm103.dsa.split_merge"


class SplitMergeCompatibility(unittest.TestCase):
    def load_split(self):
        node = next(
            n
            for n in ast.parse((ROOT / "split.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "SplitMerge"
        )
        scope = {"TOPK": 2048, "PAGE": 64, "__package__": "split_test.dsa"}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "split.py", "exec"), scope
        )
        return scope["SplitMerge"]

    def test_factory_storage_and_replicated_kernel(self):
        for cand_k in (256, 384):
            factory_calls, launch_calls = [], []
            storage = types.SimpleNamespace(
                buffer=torch.zeros(2 * 8 * (16 * cand_k + 2048), dtype=torch.int64),
                multicast_ptr=123,
                peer_ptrs=torch.arange(16),
                counters=torch.zeros(8),
                union_scratch=torch.empty(8, 16 * 2048),
                fallback=torch.zeros(8),
                dcp_rank=5,
                usable=lambda rows: rows <= 8,
            )

            def create(*args):
                factory_calls.append(args)
                return storage

            host = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_dcp_sharded")
            host.DcpTopKMerge = types.SimpleNamespace(create=create)
            kernel = types.ModuleType(KERNEL)
            kernel.dcp_topk_merge = lambda *a, **k: launch_calls.append((a, k))
            with patch.dict(
                sys.modules, {host.__name__: host, kernel.__name__: kernel}
            ):
                group = types.SimpleNamespace(rank_in_group=5, world_size=16)
                merge = self.load_split()(group, 8)
                self.assertEqual(factory_calls, [(group, 8, 2048, 5)])
                self.assertTrue(merge.usable(8))
                self.assertFalse(merge.usable(9))
                out = torch.empty(4, 2048, dtype=torch.int32)
                self.assertIs(merge.run("candidates", "lengths", "pages", out), out)
                args, kwargs = launch_calls[0]
                self.assertEqual(args, ("candidates", "lengths", "pages"))
                self.assertTrue(kwargs["split"])
                self.assertEqual(
                    (kwargs["rank"], kwargs["span"], kwargs["cand_k"]), (5, 64, cand_k)
                )
                self.assertIs(kwargs["buffer"], storage.buffer)
                self.assertEqual(kwargs["out_fallback"].shape, (4,))

    def test_declined_factory_remains_unusable(self):
        host = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_dcp_sharded")
        host.DcpTopKMerge = types.SimpleNamespace(create=lambda *a: None)
        kernel = types.ModuleType(KERNEL)
        kernel.dcp_topk_merge = lambda *a, **k: self.fail("declined merge launched")
        with patch.dict(sys.modules, {host.__name__: host, kernel.__name__: kernel}):
            merge = self.load_split()(types.SimpleNamespace(rank_in_group=0), 8)
            self.assertFalse(merge.usable(1))
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                merge.run(None, None, None, None)


if __name__ == "__main__":
    unittest.main()


class K31DFlashRuntimeRound(unittest.TestCase):
    """The K3.1 DFlash runtime on CPU, over fakes of SGLang's worker bindings and the
    megakernel's target and draft chain: what it launches, in which order, and what it
    hands back to SGLang's stock round."""

    BLOCK = 8

    def setUp(self):
        try:
            from sglang.srt.kernel_plugins.nvidia import (  # noqa: F401
                kimi_k31_dflash_binding,
            )
        except ImportError:
            self.skipTest("this SGLang has no DFlash runtime binding")
        from sgl_nvidia_mega.kimi_k31.dflash import protocol
        from sgl_nvidia_mega.kimi_k31.dflash import runtime as rt

        self.rt, self.calls = rt, []
        # No MLP sync unless a test says so (SGLang reads it from the parallel config).
        mlp_sync = patch.object(rt, "require_mlp_sync", return_value=False)
        mlp_sync.start()
        self.addCleanup(mlp_sync.stop)
        log = self.calls.append
        self.chain = types.SimpleNamespace(
            draft_aux_rows=(self.BLOCK,),
            draft_head_requests=(1,),
            draft_head_sampling=True,
            draft_accept_requests=(),
            draft_embed_requests=(),
            attach_draft_head=lambda **kw: (
                log("attach_head")
                or (torch.arange(7), torch.zeros(1, 7, 16), torch.zeros(1, 7, 16))
            ),
            stage_draft_head_sampling=lambda **kw: log("stage_head"),
        )
        protocol.DRAFT_CHAINS.append(self.chain)
        self.addCleanup(protocol.DRAFT_CHAINS.remove, self.chain)
        self.target = types.SimpleNamespace(
            kda_commit_enabled=True,
            prepare_sampled_accept=lambda **kw: log("prepare_sampled") or True,
            sampled_accept_fits=lambda bs, logits=None: True,
            sampled_accept=lambda **kw: (
                log("sampled_accept")
                or (torch.tensor([2], dtype=torch.int32), torch.tensor([9]), None)
            ),
        )
        lp = types.SimpleNamespace(
            vocab_size=32,
            final_logit_softcapping=None,
            defer_plain_gather=False,
            is_deferred_shard=lambda x: log("is_deferred_shard") or True,
            deferred_argmax=lambda x: log("deferred_argmax") or torch.zeros(1, 8),
        )
        runner = types.SimpleNamespace(
            model=types.SimpleNamespace(
                lm_head=types.SimpleNamespace(num_embeddings_per_partition=128),
                get_input_embeddings=lambda: (
                    lambda ids: log("embed") or torch.zeros(*ids.shape, 4)
                ),
            ),
            req_to_token_pool=types.SimpleNamespace(req_to_token=None),
            runner_ids=types.SimpleNamespace(plan=self._plan),
            attn_backend="target attention",
        )
        self.verify_out = types.SimpleNamespace(
            logits_output=types.SimpleNamespace(
                next_token_logits="shard", hidden_states=torch.zeros(8, 4)
            ),
            can_run_cuda_graph=True,
            routed_experts_output=None,
            indexer_topk_output=None,
            kv_weight_version_record=None,
        )
        self.bindings = {
            "block_size": self.BLOCK,
            "device": "cpu",  # the worker holds the server arg
            "mask_token_id": 3,
            "noise_embed_scale": 1.0,
            "batched_small_copy": False,
            "draft_model": types.SimpleNamespace(
                candidate_selector=types.SimpleNamespace(top_k=16),
                draft_config=None,
            ),
            "draft_model_runner": types.SimpleNamespace(
                runner_ids=types.SimpleNamespace(space=lambda kind: None),
                token_to_kv_pool="draft pool",
                forward=lambda fb: (
                    log("draft_forward") or types.SimpleNamespace(can_run_graph=True)
                ),
            ),
            "draft_parallel_layout": None,
            "draft_block_spec_info": None,
            "target_worker": types.SimpleNamespace(
                model_runner=runner,
                forward_batch_generation=lambda **kw: log("verify") or self.verify_out,
            ),
            "target_logits_processor": lp,
            "sim_acc_rng": None,
            "need_mamba_verify_commit": True,
            "compute_compact_draft_seq_lens": lambda lens: log("compact_lens") or lens,
            "fill_compact_seq_lens_cpu_bound": lambda **kw: log("compact_bound"),
            "rebuild_compact_draft_cache": lambda **kw: log("compact_rebuild"),
            "accept_block": lambda **kw: (
                log(("accept_block", kw["target_predict"]))
                or (
                    torch.tensor([1]),
                    torch.tensor([2]),
                    torch.tensor([7]),
                    torch.zeros(1, 8),
                    None,
                    None,
                )
            ),
        }
        graphs = types.SimpleNamespace(
            graph=types.SimpleNamespace(
                cuda_graph_config=types.SimpleNamespace(
                    decode=types.SimpleNamespace(bs=[1])
                )
            )
        )
        verify_input = types.SimpleNamespace(
            prepare_for_verify=lambda *a, **kw: log("prepare_verify") or ("fb", True)
        )
        for name, value in {
            "megakernel_agree": lambda value, what: log(("agree", what)),
            "built_megakernel": lambda model: self.chain,
            "get_parallel": lambda: types.SimpleNamespace(tp_rank=0),
            "get_exec": lambda: graphs,
            "_prepare_dflash_draft_block_unchecked": lambda **kw: log("prepare_block"),
            "ForwardBatch": lambda **kw: log("forward_batch") or kw,
            "DFlashVerifyInput": lambda **kw: verify_input,
            "draft_tp_context": lambda layout: contextlib.nullcontext(),
            "apply_dflash_simulated_acceptance": lambda **kw: (
                log("simulate") or torch.tensor([11])
            ),
            "SIMULATE_ACC_LEN": 3.69,
            "enable_num_token_non_padded": lambda: False,
            "GenerationBatchResult": lambda **kw: kw,
        }.items():
            mocked = patch.object(rt, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def _plan(self, **kw):
        log = self.calls.append
        return types.SimpleNamespace(
            reads_from=lambda *a, **kw: types.SimpleNamespace(
                bind=lambda fb, translator: log("bind")
            ),
            write_ids=lambda reader: torch.zeros(8, dtype=torch.int64),
        )

    def _binding(self, bindings=None):
        bindings = bindings or self.bindings
        return types.SimpleNamespace(pull=lambda name: bindings[name])

    def _runtime(self):
        runtime = self.rt.K31DFlashRuntime(self._binding(), self.target)
        log = self.calls.append
        runtime.draft_sampler(lm_head="lm_head")
        runtime._kda._megakernel_kda_commit = lambda **kw: log("kda_commit") or True
        runtime._aux._append_target_hidden_megakernel = lambda **kw: log("draft_aux")
        return runtime

    def _batch(self, **sampling):
        info = dict(
            is_all_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones(1, 1),
            top_ks=torch.tensor([20]),
            penalizer_orchestrator=None,
            has_custom_logit_processor=False,
            grammar_mask=None,
            logit_bias=None,
            sampling_mask_batch_indices=None,
        )
        info.update(sampling)
        return types.SimpleNamespace(
            seq_lens=torch.tensor([5]),
            seq_lens_cpu=torch.tensor([5]),
            seq_lens_sum=5,
            req_pool_indices=torch.tensor([0]),
            spec_info=types.SimpleNamespace(
                bonus_tokens=torch.tensor([4]), nxt_kv_lens_cpu=None
            ),
            sampling_info=types.SimpleNamespace(**info),
            has_grammar=False,
            return_logprob=False,
            can_run_decode_cuda_graph=True,
        )

    def _round(self, runtime, batch):
        published = []

        def publish(new_seq_lens):
            self.calls.append("publish")
            published.append(new_seq_lens)

        with patch.object(torch.Tensor, "record_stream"):
            out = runtime.decode_round(batch, on_publish=publish, grammar_barrier=None)
        return out, published

    def test_init_builds_the_sampled_accept_and_arms_the_deferral_before_capture(self):
        runtime = self._runtime()
        self.assertEqual(
            self.calls,
            [("agree", "DFlash draft chain offers"), "prepare_sampled", "attach_head"],
        )
        self.assertTrue(self.bindings["target_logits_processor"].defer_plain_gather)
        self.assertEqual(runtime._round_bs, {1})
        self.assertEqual(runtime._aux._draft_kv_pool, "draft pool")
        self.assertEqual(runtime._kda._target_attn_backend, "target attention")

    def test_a_sampled_round_runs_the_measured_order(self):
        runtime = self._runtime()
        del self.calls[:]
        out, published = self._round(runtime, self._batch())
        self.assertEqual(
            self.calls,
            [
                "prepare_block",
                "embed",
                "compact_lens",
                "compact_bound",
                "compact_rebuild",
                "forward_batch",
                "bind",
                "stage_head",
                "draft_forward",
                "prepare_verify",
                "verify",
                "is_deferred_shard",
                "sampled_accept",
                "simulate",
                "kda_commit",
                "publish",
                "draft_aux",
            ],
        )
        # published once, between the KDA commit and the aux, with the simulated lengths
        self.assertEqual([t.tolist() for t in published], [[11]])
        self.assertEqual(out["new_seq_lens"].tolist(), [11])
        self.assertEqual(out["accept_lens"].tolist(), [3])
        self.assertIsNone(out["side_stream_done"])
        self.assertIsNone(self.verify_out.logits_output.next_token_logits)

    def test_a_greedy_round_takes_sglangs_accept_on_the_shard_argmax(self):
        runtime = self._runtime()
        del self.calls[:]
        self._round(runtime, self._batch(is_all_greedy=True))
        accepts = [c for c in self.calls if c[0] in ("accept_block", "sampled_accept")]
        self.assertEqual([c[0] for c in accepts], ["accept_block"])
        self.assertIn("deferred_argmax", self.calls)
        self.assertNotIn("simulate", self.calls)  # SGLang's accept simulates itself

    def test_rounds_it_does_not_serve_go_to_sglang_before_any_launch(self):
        runtime = self._runtime()
        cases = {
            "grammar": dict(has_grammar=True),
            "logprob": dict(return_logprob=True),
            "bs 2": dict(seq_lens=torch.tensor([5, 6])),
        }
        sampling = {
            "top_p": dict(need_top_p_sampling=True),
            "top_k": dict(need_top_k_sampling=True),
            "logit bias": dict(logit_bias=torch.zeros(1)),
            "sampling mask": dict(sampling_mask_batch_indices=[0]),
        }
        for name, fields in [*cases.items(), *sampling.items()]:
            with self.subTest(name):
                batch = self._batch(**(fields if name in sampling else {}))
                if name in cases:
                    batch.__dict__.update(fields)
                del self.calls[:]
                out, published = self._round(runtime, batch)
                self.assertIsNone(out)
                self.assertEqual((self.calls, published), ([], []))

    def test_the_decode_graph_flag_counts_only_under_mlp_sync(self):
        """SGLang sets can_run_decode_cuda_graph only under MLP sync; without it the
        flag stays False and the round still runs here."""
        for mlp_sync, served in ((False, True), (True, False)):
            with (
                self.subTest(mlp_sync=mlp_sync),
                patch.object(self.rt, "require_mlp_sync", return_value=mlp_sync),
            ):
                runtime = self._runtime()
                batch = self._batch()
                batch.can_run_decode_cuda_graph = False
                del self.calls[:]
                out, published = self._round(runtime, batch)
                self.assertEqual(out is not None, served)
                self.assertEqual(bool(self.calls), served)

    def test_init_refuses_what_the_round_does_not_serve(self):
        for name, change in {
            "KDA commit": lambda: setattr(self.target, "kda_commit_enabled", False),
            "opt-in accept": lambda: setattr(self.chain, "draft_accept_requests", (1,)),
            "sampled accept": lambda: setattr(
                self.target, "prepare_sampled_accept", lambda **kw: False
            ),
        }.items():
            with (
                self.subTest(name),
                patch.object(self, "target", copy.copy(self.target)),
            ):
                saved = dict(self.chain.__dict__)
                change()
                with self.assertRaisesRegex(RuntimeError, "K3.1 DFlash runtime"):
                    self.rt.K31DFlashRuntime(self._binding(), self.target)
                self.chain.__dict__.update(saved)
        # without the draft graph (and its head) the round refuses to run
        runtime = self.rt.K31DFlashRuntime(self._binding(), self.target)
        with self.assertRaisesRegex(RuntimeError, "never attached"):
            self._round(runtime, self._batch())


class K31DFlashRuntimeGuard(unittest.TestCase):
    """SGLang's guard (sglang.srt.kernel_plugins.nvidia.kimi_k31_dflash_binding) against
    the runtime's declared bindings and guards, on the measured server arguments
    (G0/G2: TP16, page 64, block 8, FP8 draft KV, draft window 1024, decode graph bs
    [1]) and on setups it does not serve."""

    PATH = "sgl_nvidia_mega.kimi_k31.dflash.runtime:K31DFlashRuntime"

    def setUp(self):
        try:
            from sglang.srt.kernel_plugins.nvidia import kimi_k31_dflash_binding
        except ImportError:
            self.skipTest("this SGLang has no DFlash runtime binding")
        from sglang.srt.kernel_plugins.nvidia import binding
        from sglang.srt.runtime_context import get_parallel, reset_context

        from sgl_nvidia_mega.kimi_k31.dflash.runtime import K31DFlashRuntime

        self.sgl, self.runtime = kimi_k31_dflash_binding, K31DFlashRuntime
        self.addCleanup(reset_context)
        self._publish()
        for patched in (
            get_parallel().override(tp_rank=0),
            patch.object(self.sgl, "is_cuda", return_value=True),
            patch.object(self.sgl, "mambaish_config", return_value=object()),
            patch.object(binding, "_gather_tp", side_effect=lambda v, group=None: [v]),
            patch.object(self.sgl, "built_megakernel"),
        ):
            patched.__enter__()
            self.addCleanup(patched.__exit__, None, None, None)
        self.sgl.built_megakernel.return_value = types.SimpleNamespace(
            dflash_runtime=self.PATH
        )
        config = types.SimpleNamespace(dtype=torch.bfloat16)
        self.worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(model_config=config),
            draft_model_runner=types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(architectures=["DFlash2DraftModel"])
                )
            ),
            block_size=8,
            use_compact_draft_cache=True,
            draft_kv_ring=None,
            selector=object(),
            _selector_sampling_enabled=True,
            lilicorr=None,
            _is_domino=False,
            _block_verification=False,
            _full_embed_gpu=None,
            _use_aux_graph=True,
        )

    def _publish(self, tp_size=16, graph_bs=(1,)):
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
            with_phase,
        )
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        publish(
            ServerArgs(
                model_path="dummy",
                tp_size=tp_size,
                page_size=64,
                kv_cache_dtype="fp8_e4m3",
                speculative_algorithm="DFLASH",
                speculative_num_draft_tokens=8,
                speculative_draft_kv_cache_dtype="fp8_e4m3",
                speculative_draft_window_size=1024,
                enable_nvidia_megakernel=True,
                cuda_graph_config=with_phase(
                    default_cuda_graph_config(), "decode", bs=list(graph_bs)
                ),
            ),
            role="test",
        )

    def test_the_measured_setup_is_served_and_any_other_is_named(self):
        guard = self.sgl.DFlashGuard(self.runtime.guards)
        self.assertEqual(guard.unmet(self.worker), [])
        self.worker.block_size = 5
        self.worker._use_aux_graph = False
        unserved = guard.unmet(self.worker)
        self.assertEqual([m.split()[0] for m in unserved], ["aux_graph", "block_size"])
        self.assertEqual(unserved[1], "block_size is 5, expected 8")

    def test_tp8_or_wider_decode_graphs_decline_to_sglangs_round(self):
        """Setups that ran SGLang's stock DFlash round beside the target megakernel
        before the runtime existed must still start, with one warning."""
        for tp_size, graph_bs, key in ((8, (1,), "tp_size"), (16, (1, 2, 4), "decode")):
            with self.subTest(tp_size=tp_size, graph_bs=graph_bs):
                self._publish(tp_size=tp_size, graph_bs=graph_bs)
                with self.assertLogs(self.sgl.logger, "WARNING") as logs:
                    bound = self.sgl.bind_dflash_runtime(self.worker, None)
                self.assertIsNone(bound)
                (warning,) = logs.output
                self.assertIn("does not serve this DFLASH setup", warning)
                self.assertIn(f"{key}", warning)
                self.assertIn("DFlash phases run stock", warning)
