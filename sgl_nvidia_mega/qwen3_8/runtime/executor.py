# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_vl.py (Qwen3VLForConditionalGeneration.forward) and
# python/sglang/srt/models/qwen4_exp.py (the FN model's and its layers' forward).
"""The executor: one covered target-verify step of Qwen3.8 Flash Next (FN).

``Qwen38Verify.plan`` decides from host state alone, identical on every rank, whether a
step is covered: a target-verify step of a built width. ``Qwen38Verify.forward`` takes
and returns what ``Qwen4ExpForConditionalGeneration.forward`` does, from the plugin's
step functions over SGLang's loaded modules, never through a model-level or
decoder-layer ``forward``:

1. the positions the layers take (M-RoPE when the model uses it) and the embedding;
2. the residual stream's start and this step's PLE batch (its n-gram history);
3. per layer, under the expert-distribution recorder's layer scope and the layer's HC
   norm scope (which carries the next layer's HC norm across the FFN exit): the next
   PLE layer's prefetch, ``x = layer._prepare_attn_stage(h)`` (the HC read, with the
   PLE embedding on a PLE layer), the mixer (GDN or QSA attention),
   ``layer.attn_boundary.finish``, then the FFN stage (``layer._run_ffn_stage``: the
   HC read, the MoE and the HC write);
4. the PLE batch's commit, the HC mix down to one stream (the stack's terminal read)
   and the logits, whose hidden states are the HC streams the MTP draft takes.

Pipeline and context parallelism and breakable decode graphs are refused at startup,
so their branches of the mirrored code do not appear here.
"""

from __future__ import annotations

from functools import partial

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.qwen3_5.runtime import embedding
from sgl_nvidia_mega.qwen3_5.runtime.state import StepState

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # Step, and the BuiltMegakernel members SGLang reads
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Step",
        "mode",
        "rows",
        "view",
        "widths",
        "draft_sampler",
        "summary",
        "plan",
        "forward",
    ),
    "srt/models/qwen3_vl.py": (
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "is_mrope_enabled",
        "language_model_only",
        "logits_processor",
        "lm_head",
        "model",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "mrope_positions",
        "forward_mode",
        "input_ids",
    ),
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpLinearDecoderLayer",
        "_prepare_ple_batch",
        "_commit_ple_batch",
        "_MixStreams",
        "has_ple",
        "ple_ngram_size",
        "ple_ngram_eos_token_id",
        "ple",
        "start_prefetch",
        "attn_hyper_connection",
        "hyper_connection_mixer",
        "hc_norms",
        "_prepare_attn_stage",
        "_run_ffn_stage",
        "attn_boundary",
        "self_attention",
        "last_hc_hidden_states",
    ),
    "srt/models/qwen3_5.py": ("layers", "start_layer", "end_layer", "linear_attn"),
    "srt/layers/hyperconnection.py": ("hc_norm",),
    "srt/layers/layer_boundary/stage.py": ("finish",),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "final_norm"),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_vl.py:Qwen3VLForConditionalGeneration.forward": "d394d79ba7fa7b31",
    "srt/models/qwen4_exp.py:Qwen4ExpForConditionalGeneration.forward": (
        "1d988276834c06bb"
    ),
    "srt/models/qwen4_exp.py:Qwen4ExpVLModel.forward": "ff08ad8dfece1d16",
    "srt/models/qwen4_exp.py:Qwen4ExpModel.forward": "69e0064f63e30a25",
    "srt/models/qwen4_exp.py:Qwen4ExpLinearDecoderLayer.forward": "7680125d5fa1a372",
    "srt/models/qwen4_exp.py:Qwen4ExpAttentionDecoderLayer.forward": (
        "572467872c848570"
    ),
}


def gdn_stock(layer, x, st):
    """``layer``'s GatedDeltaNet on its HC read ``x``: the TP-partial output."""
    return layer.linear_attn(x, st.fb)


def attention_stock(layer, x, st):
    """``layer``'s QSA attention on its HC read ``x``: the TP-partial output."""
    return layer.self_attention(
        positions=st.positions, hidden_states=x, forward_batch=st.fb
    )


class Qwen38Verify:
    """One rank's covered steps: the model, each width's slots and each package's
    build result."""

    # the target offers no draft entries (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None

    def __init__(
        self,
        lm,
        slots: dict,
        bound: dict,
        widths: tuple[int, ...],
        binding=None,
        summary: str = "",
        parity=None,
    ):
        from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer
        from sgl_nvidia_mega.qwen3_8.gdn import step as gdn_step
        from sgl_nvidia_mega.qwen3_8.attention import step as attention_step

        self.lm, self.slots, self.bound = lm, slots, bound
        self.widths = tuple(sorted(widths))
        self.binding = binding  # SGLang's generic MegakernelBinding
        self._summary = summary
        self.parity = parity  # qwen3_5.runtime.parity.Parity under the test-only switch
        m = lm.model
        self.mixers = {
            i: (gdn_step if any(s.gdn_verify or s.gdn_mk or s.gdn_front
                               for s in slots.values()) else gdn_stock)
            if isinstance(m.layers[i], Qwen4ExpLinearDecoderLayer)
            else (attention_step if any(s.qsa_core for s in slots.values()) else attention_stock)
            for i in range(m.start_layer, m.end_layer)
        }

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(
        self,
        input_ids,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ) -> Step | None:
        """The step if this forward is covered, else None (SGLang's forward runs).
        An argument this executor does not know also leaves the step to SGLang."""
        if self.parity is not None:
            self.parity.flush()
        if kwargs or get_embedding or pp_proxy_tensors is not None:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        rows = int(input_ids.shape[0])
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def forward(
        self, step: Step, original, input_ids, positions, forward_batch, *args, **kwargs
    ):
        """The covered step (``original``: the model's own forward, bound, which the
        parity check runs)."""
        run = partial(self._forward, step, input_ids, positions, forward_batch)
        if self.parity is not None:
            stock = partial(
                original, input_ids, positions, forward_batch, *args, **kwargs
            )
            return self.parity.run(step, stock, run)
        return run()

    @torch.no_grad()
    def _forward(self, step, input_ids, positions, forward_batch):
        """``Qwen4ExpForConditionalGeneration.forward`` for a covered step."""
        lm = self.lm
        if lm.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        st = StepState(
            lm=lm,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            slots=self.slots[step.rows],
            bound=self.bound,
            binding=self.binding,
            view=step.view,
        )
        hidden = self._language_model(st, embedding.embed(st))
        out = lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)
        out.hidden_states = st.model.last_hc_hidden_states
        return out

    def _language_model(self, st, hidden):
        """``Qwen4ExpVLModel.forward`` and ``Qwen4ExpModel.forward``: the layers, the
        PLE commit and the HC mix to one stream; keeps the HC streams for the draft."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.layer_boundary.residual.batch import final_norm, start
        from sglang.srt.models.qwen4_exp import (
            _commit_ple_batch,
            _MixStreams,
            _prepare_ple_batch,
        )

        m = st.model
        m.last_hc_hidden_states = None
        start(st.fb)
        ple_batch = None
        if m.has_ple:
            # the multimodal routine hands the language model no ids; it reads the batch's
            ids = st.input_ids if st.lm.language_model_only else st.fb.input_ids
            ple_batch = _prepare_ple_batch(
                input_ids=ids,
                forward_batch=st.fb,
                ngram_size=m.ple_ngram_size,
                ngram_eos_token_id=m.ple_ngram_eos_token_id,
            )
        normalized_input = None
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            if i + 1 < m.end_layer:
                next_layer = m.layers[i + 1]
                if next_layer.ple is not None:
                    next_layer.ple.start_prefetch(batch=ple_batch, forward_batch=st.fb)
                    next_norm = None
                else:
                    next_norm = next_layer.attn_hyper_connection.hc_norm
            else:
                next_norm = m.hyper_connection_mixer.hc_norm
            with (
                get_global_expert_distribution_recorder().with_current_layer(i),
                layer.hc_norms(normalized_input, next_norm) as norms,
            ):
                hidden = self._layer(i, layer, hidden, st, ple_batch)
            normalized_input = norms.output
        _commit_ple_batch(batch=ple_batch, forward_batch=st.fb)
        m.last_hc_hidden_states = hidden
        return final_norm(
            hidden, st.fb, _MixStreams(m.hyper_connection_mixer, normalized_input)
        )

    def _layer(self, i, layer, hidden, st, ple_batch):
        """A decoder layer's forward: the mixer stage, then the FFN stage."""
        x = layer._prepare_attn_stage(hidden, st.fb, ple_batch)
        part = self.mixers[i](layer, x, st)
        hidden = layer.attn_boundary.finish(part, st.fb)
        if st.slots.route:
            from sgl_nvidia_mega.qwen3_8.moe.step import ffn

            return ffn(layer, hidden, st)
        return layer._run_ffn_stage(hidden, st.fb)
