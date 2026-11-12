# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen4_exp.py (Qwen4ExpForConditionalGeneration.forward,
# Qwen4ExpVLModel.forward, Qwen4ExpModel.forward and its decoder layers' forward),
# python/sglang/srt/models/qwen3_vl.py (Qwen3VLForConditionalGeneration.forward) and
# python/sglang/srt/models/qwen4_exp_mtp.py (Qwen4ExpForCausalLMMTP.forward).
"""The executor: every covered forward of Qwen3.8, target and MTP draft.

``Qwen3_8Executor.plan`` decides from host state alone, identical on every rank, which
forwards it runs: target verify and decode steps of a built width (in CUDA graphs),
draft-extend steps (in the draft's graphs) and prefill / extend steps (eager). It
declines idle batches, embedding calls and pipeline-parallel proxies, which run the
model's own forward.

``Qwen3_8Executor.forward`` takes and returns what the replaced forward does. It runs
the layer loop itself from the executor's step functions, never through a model-level
or decoder-layer ``forward``:

1. the positions the layers take (M-RoPE when the model uses it) and the embedding
   (the draft: the MTP input fusion of the embedding and the target's streams);
2. the PLE batch, then per layer, under the expert-distribution recorder's layer
   scope and the layer's prepared HC norms: the attention stage's read (HC mix, PLE
   add), the mixer (QSA attention or GDN), the attention stage's write (HC combine),
   and the FFN stage (HC read, MoE, HC write);
3. the PLE commit, the final mix of the streams and the logits, with the HC streams
   handed back as the hidden states.

Each op's implementation is chosen per width at build (``slots``); an op no kernel
serves calls SGLang's own module, and ``engagement`` records which one ran.
"""

from __future__ import annotations

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.qwen3_8.runtime import attention, gdn, moe
from sgl_nvidia_mega.qwen3_8.runtime.engagement import Engagement
from sgl_nvidia_mega.qwen3_8.runtime.state import StepState

STOCK = "stock"

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Step",
        "mode",
        "rows",
        "view",
        "widths",
        "summary",
        "plan",
        "forward",
    ),
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpAttentionDecoderLayer",
        "Qwen4ExpLinearDecoderLayer",
        "_prepare_ple_batch",
        "_commit_ple_batch",
        "start_prefetch",
        "_shards_rows_over_cp",
        "_MixStreams",
        "_prepare_attn_stage",
        "_run_ffn_stage",
        "self_attention",
        "attn_boundary",
        "hc_norms",
        "hyper_connection_mixer",
        "attn_hyper_connection",
        "has_ple",
        "ple_ngram_size",
        "ple_ngram_eos_token_id",
        "last_hc_hidden_states",
        "residual_batch",
    ),
    "srt/models/qwen3_5.py": ("linear_attn",),
    "srt/models/qwen4_exp_mtp.py": (
        "_prepare_input_embeds",
        "_mtp_input_fusion",
        "_set_hc_logits_hidden_states",
    ),
    "srt/models/qwen3_vl.py": (
        "is_mrope_enabled",
        "language_model_only",
        "use_deepstack",
        "logits_processor",
        "lm_head",
    ),
    "srt/managers/mm_utils.py": ("prepare_mm_inputs",),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "final_norm"),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
        "disable_this_region",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
}

# The SGLang functions whose body this executor mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang). Filled in from
# the pinned SGLang revision by test/unit/test_contract.py --update.
SGLANG_MIRRORS = {}


class Qwen3_8Executor:
    """One rank's covered forwards of the target or the MTP draft."""

    # EAGLE entries: the draft is covered through plan(), SGLang runs the loop.
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(
        self, lm, *, widths, draft, slots, binding=None, parity=None, decompose=False
    ):
        from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

        self.lm = lm
        self.widths = tuple(sorted(widths))
        self._covered = frozenset(self.widths)
        self.draft = draft
        self.slots = slots
        self.binding = binding
        self.parity = parity
        self.engagement = Engagement("draft" if draft else "target")
        # S1: the layer split into its HC read / mixer / HC write / MoE pieces,
        # called from here instead of through the stage boundaries. Only where the
        # boundaries do no collective (TP1, no DP / CP / EP).
        self.decompose = decompose
        m = lm.model
        self.mixers = {
            i: (
                self._gdn
                if isinstance(m.layers[i], Qwen4ExpLinearDecoderLayer)
                else self._qsa
            )
            for i in range(m.start_layer, m.end_layer)
        }

    def summary(self) -> str:
        what = "draft" if self.draft else "target"
        return f"Qwen3.8 {what} executor; slots {sorted(self.slots) or 'none (stock)'}"

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        """The step if this forward is covered, else None (the model's forward runs).
        Host state only, so a captured graph replays the decision made at capture."""
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

        if self.parity is not None:
            self.parity.flush()
        if kwargs or get_embedding or pp_proxy_tensors is not None:
            return None
        mode = forward_batch.forward_mode
        if mode.is_idle() or _shards_rows_over_cp(forward_batch):
            return None
        rows = int(
            (input_ids if input_ids is not None else forward_batch.input_ids).shape[0]
        )
        if mode.is_target_verify():
            name = None if self.draft else "target_verify"
        elif mode.is_decode():
            name = "decode"
        elif mode.is_draft_extend_v2():
            name = "draft_extend" if self.draft else None
        elif mode.is_extend_or_draft_extend_or_mixed():
            if is_in_breakable_cuda_graph():
                return None
            return Step(mode="prefill", rows=rows)
        else:
            name = None
        if name is None or rows not in self._covered:
            return None
        return Step(mode=name, rows=rows)

    def forward(
        self, step, original, input_ids, positions, forward_batch, *args, **kwargs
    ):
        """The covered step (``original``: the model's own forward, bound, which the
        parity check runs)."""
        run = lambda: self._forward(step, input_ids, positions, forward_batch)
        if self.parity is not None:
            stock = lambda: original(
                input_ids, positions, forward_batch, *args, **kwargs
            )
            return self.parity.run(step, forward_batch, stock, run)
        return run()

    def _gdn(self, layer, x, st):
        return gdn.step(self, layer, x, st)

    def _qsa(self, layer, x, st):
        return attention.step(self, layer, x, st)

    @torch.no_grad()
    def _forward(self, step, input_ids, positions, forward_batch):
        lm = self.lm
        st = StepState(
            lm=lm,
            step=step,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=self.slots,
            binding=self.binding,
            engagement=self.engagement,
        )
        self.engagement.begin()
        out = self._draft(st) if self.draft else self._target(st)
        self.engagement.end(step.mode, step.rows)
        return out

    def _target(self, st):
        """``Qwen4ExpForConditionalGeneration.forward`` (through
        ``Qwen3VLForConditionalGeneration.forward`` and ``Qwen4ExpVLModel.forward``)."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        lm, fb = st.lm, st.fb
        if lm.is_mrope_enabled:
            st.positions = fb.mrope_positions
        lm.model.last_hc_hidden_states = None
        if lm.language_model_only:
            ids, embeds = st.input_ids, None
        else:
            from sglang.srt.managers.mm_utils import prepare_mm_inputs

            # general_mm_embed_routine: the language model then gets input_ids=None
            # and reads the batch's own ids (the PLE needs them).
            embeds = prepare_mm_inputs(
                st.input_ids,
                fb,
                st.model,
                {"positions": st.positions, "pp_proxy_tensors": None},
                multimodal_model=lm,
                use_deepstack=lm.use_deepstack,
            )
            ids = fb.input_ids
        if ids is None:
            ids = fb.input_ids
        st.ran("embed", STOCK)
        hidden, hc_hidden = self._language_model(st, ids, embeds)
        lm.model.last_hc_hidden_states = hc_hidden
        out = lm.logits_processor(st.input_ids, hidden, lm.lm_head, fb, None)
        st.ran("logits", STOCK)
        if hc_hidden is not None and isinstance(out, LogitsProcessorOutput):
            out.hidden_states = hc_hidden
        return out

    def _draft(self, st):
        """``Qwen4ExpForCausalLMMTP.forward``."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )

        lm, fb = st.lm, st.fb
        embeds = lm._prepare_input_embeds(st.input_ids, fb, None)
        st.ran("embed", STOCK)
        streams = lm._mtp_input_fusion(embeds, fb.spec_info.hidden_states)
        st.ran("mtp_fusion", STOCK)
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden, hc_hidden = self._language_model(st, st.input_ids, streams)
        out = lm.logits_processor(st.input_ids, hidden, lm.lm_head, fb)
        st.ran("logits", STOCK)
        lm._set_hc_logits_hidden_states(out, hc_hidden, fb)
        return out

    def _language_model(self, st, input_ids, inputs_embeds):
        """``Qwen4ExpModel.forward`` on one stage holding every layer, without CP row
        sharding (plan declines both): the layers, the PLE commit and the final mix;
        returns the mixed hidden rows and the HC streams."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models.qwen4_exp import (
            _commit_ple_batch,
            _MixStreams,
            _prepare_ple_batch,
        )

        m, fb = st.model, st.fb
        hidden = inputs_embeds if inputs_embeds is not None else m.embed_tokens(input_ids)
        residual_batch.start(fb)
        if m.has_ple:
            st.ple_batch = _prepare_ple_batch(
                input_ids=input_ids,
                forward_batch=fb,
                ngram_size=m.ple_ngram_size,
                ngram_eos_token_id=m.ple_ngram_eos_token_id,
            )
            st.ran("ple", STOCK)
        recorder = get_global_expert_distribution_recorder()
        normalized_input = None
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            next_norm = None
            if i + 1 < m.end_layer:
                next_layer = m.layers[i + 1]
                if next_layer.ple is not None:
                    next_layer.ple.start_prefetch(batch=st.ple_batch, forward_batch=fb)
                else:
                    next_norm = next_layer.attn_hyper_connection.hc_norm
            elif m.pp_group.is_last_rank:
                next_norm = m.hyper_connection_mixer.hc_norm
            with (
                recorder.with_current_layer(i),
                layer.hc_norms(normalized_input, next_norm) as norms,
            ):
                if self.decompose:
                    hidden = self._layer_streams(i, layer, hidden, st)
                else:
                    hidden = self._layer(i, layer, hidden, st)
            normalized_input = norms.output
            if self.parity is not None:
                self.parity.record_layer(i, hidden)
        _commit_ple_batch(batch=st.ple_batch, forward_batch=fb)
        hc_hidden = hidden
        mix = _MixStreams(m.hyper_connection_mixer, normalized_input)
        if self.decompose:
            residual_batch.clear(fb)
            hidden = mix(hidden)
        else:
            hidden = residual_batch.final_norm(hidden, fb, mix)
        st.ran("hc.final_mix", STOCK)
        return hidden, hc_hidden

    def _layer(self, i, layer, hidden, st):
        """A decoder layer: the attention stage's read, the mixer and the stage's
        write, then the FFN stage (read, MoE, write)."""
        fb = st.fb
        hidden = layer._prepare_attn_stage(hidden, fb, st.ple_batch)
        st.ran("hc.attn_read", STOCK)
        if not fb.forward_mode.is_idle():
            hidden = self.mixers[i](layer, hidden, st)
        hidden = layer.attn_boundary.finish(hidden, fb)
        st.ran("hc.attn_write", STOCK)
        hidden = layer._run_ffn_stage(hidden, fb)
        st.ran("moe+hc.ffn", STOCK)
        return hidden

    def _layer_streams(self, i, layer, streams, st):
        """S1: the decoder layer on the HC streams, at TP1. The stage boundaries
        there apply the attention write-back at the FFN read and the FFN
        write-back at the stage's finish, both inside the layer's prepared-norm
        scope, so this is the same sequence of the layer's own HC calls."""
        streams = layer._widen_streams(streams)
        fb = st.fb
        if layer.ple is not None:
            layer._ple_forward_batch, layer._ple_batch = fb, st.ple_batch
        try:
            x, (res, normed, partials) = layer._attn_mix(streams)
        finally:
            layer._ple_forward_batch = layer._ple_batch = None
        st.ran("hc.attn_read", STOCK)
        if not fb.forward_mode.is_idle():
            x = self.mixers[i](layer, x, st)
        streams = layer._attn_combine(x, (res, normed, partials))
        st.ran("hc.attn_write", STOCK)
        x, (res, normed, partials) = layer._ffn_mix(streams)
        st.ran("hc.ffn_read", STOCK)
        x = moe.step(self, layer, x, st)
        streams = layer._ffn_combine(x, (res, normed, partials))
        st.ran("hc.ffn_write", STOCK)
        return streams
