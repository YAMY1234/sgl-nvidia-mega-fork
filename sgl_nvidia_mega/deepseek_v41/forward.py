# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The routed DeepSeek-V4.1 step: SGLang's own modules, with this package's slots.

``Routed.forward`` takes and returns what ``DeepseekV4ForCausalLM.forward`` does for
a DSpark verify step on CUDA: SGLang's input preparation, the embedding and its mHC
broadcast, every layer through SGLang's stage boundaries (which own the residual
streams, the mHC stats stream and the fused posts), the head's combine and norm, and
SGLang's logits processor. It follows SGLang's step for the one configuration the
spec accepts (TP4, no pipeline, no attention DP, no context parallelism, no
two-batch overlap, no late-layer tail in verify steps); the contract test pins every
SGLang name it reads, and the composition-parity check (``SGLANG_NVIDIA_MEGA_PARITY``)
compares it with SGLang's forward step by step.
"""

from __future__ import annotations

import logging
import os

import msgspec
import torch

from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "maybe_input_scattered",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "set_written",
        "read_output",
        "clear",
    ),
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast", "hc_combine"),
    "srt/models/deepseek_v4_mhc.py": ("HcState", "with_residual", "materialized"),
    "srt/models/deepseek_v4_boundary.py": (
        "LayerCall",
        "HcRead",
        "bind",
        "next_boundary",
        "materialize",
        "materialize(next)",
        "FinalRead",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_size"',
        '"attn_dp_size"',
        '"tp_group"',
    ),
    "srt/distributed/parallel_state.py": ("rank_in_group", "cpu_group", "device_group"),
    # DeepseekV4ForCausalLM.forward, which the routed forward stands in for
    "srt/models/deepseek_v4.py": (
        "prepare_language_model_inputs",
        "logits_processor",
        "lm_head",
        "capture_aux_hidden_states",
        "model",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "forward(input_ids_global)",
        "embed_tokens",
        "hc_mult",
        "hc_pre_from_prev_sublayer",
        "norm",
        "start_layer",
        "end_layer",
        "engram_hasher",
        "engram",
        "dspark_layers_to_capture",
        "layers",
        "config",
        "hc_cfg",
        '"freqs_cis_c4"',
        '"freqs_cis_c128"',
        "late_layer_start",
        "DeepseekV4CudaDecoderLayer",
        "MQALayer",
        "attn_boundary",
        "ffn_boundary",
        "self_attn",
        "mlp",
        "wo_b",
        "maybe_use_decode_attn_tp",
        "_run_boundary_attention",
        "layer_id",
        "_project_wo_b(defer_all_reduce)",
        "forward(x_quant)",
        # the module aliases SGLang's DSV4 model imports too
        "hc_boundary",
        "mhc",
        "residual_batch",
    ),
    "srt/layers/engram.py": ("forward(cp_all_tokens)", "layer_hash_index"),
    "srt/configs/deepseek_v41.py": ("vision_n_layers",),
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": (
        "Mxfp4FlashinferTrtllmMoEMethod",
        "routed_hidden_size",
        "quantize_routed_input",
        "_gemm1_clamp_limit_tensor",
        "w13_weight_scale_inv",
        '"w2_weight_scale_inv"',
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "num_local_experts",
        "num_experts",
        "top_k",
        "quant_method",
    ),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    # DeepseekV4CudaDecoderLayer.forward, which the hop-A layer runs piece by piece
    "srt/layers/layer_boundary/stage.py": (
        "prepare",
        "finish",
        "transport_context",
        "exit",
    ),
    "srt/layers/layer_boundary/residual/batch.py": ("written_residual",),
    "srt/models/deepseek_v4_boundary.py": ("quantized", "return_moe_output"),
    "srt/models/deepseek_common/moe_transport.py": (
        "input_ids",
        "input_ids_global",
        "do_shared_local",
    ),
    "srt/models/deepseek_v2.py": (
        "forward(skip_shared_experts)",
        "forward(return_moe_output)",
        "forward(leave_output_sum)",
        "DeepseekV2MoE",
        "forward_normal_dual_stream",
        "alt_stream",
        "num_fused_shared_experts",
        "topk",
        "gate",
        "e_score_correction_bias_vl",
        "is_hash",
        "experts",
    ),
    "srt/layers/moe/topk.py": (
        "_RENORMALIZE_SUM_EPSILON",
        "topk_config",
        "routed_scaling_factor",
        "apply_routed_scaling_factor_on_output",
        "correction_bias_for_dtype",
        "scoring_func",
        "sqrtsoftplus_log1p",
        "renormalize",
        "use_grouped_topk",
        "custom_routing_function",
        "correction_bias",
        "topk_ids",
        "topk_weights",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "FlashInferTrtllmDeferredFinalizeOutput",
        "gemm2_out",
        "expert_weights",
        "expanded_idx_to_permuted_idx",
        "top_k",
    ),
    "srt/model_executor/forward_batch_info.py": ("moe_num_token_non_padded",),
    # the test-only sorted sparse indices (SGLANG_NVIDIA_MEGA_PARITY_SORT_TOPK)
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DeepseekV4AttnBackend",
        "DSV4AttnMetadata",
        "forward_metadata",
        "core_attn_metadata",
        "sparse_page_indices",
        "sparse_topk_lengths",
    ),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
        "disable_this_region",
    ),
    # DeepseekV4ForCausalLMDSpark.forward, which the draft forward stands in for
    "srt/models/deepseek_v4_dspark.py": (
        "forward_embed",
        "stages",
        "hc_mult",
        "hc_pre_from_prev_sublayer",
        "forward(positions)",
        "forward(hidden_states)",
        "forward(forward_batch)",
    ),
}

# The SGLang seams the hop-A and MoE slots need, probed at build (``hop_a_seams``,
# ``moe_seam``): without them a slot stays off and the reason is logged;
# test/unit/test_contract.py pins them against SGLang trees that have them
SGLANG_SEAMS = {
    "srt/models/deepseek_v4.py": ("forward_before_wo_b",),
    "srt/models/deepseek_v4_mhc.py": ("fused_post",),
    "srt/models/deepseek_v2.py": (
        "forward_normal_dual_stream(select_experts)",
        "forward_normal_dual_stream(run_routed_experts)",
    ),
    "srt/models/deepseek_v4_boundary.py": ("mhc_fusions",),
}

# Composition parity (test only): run SGLang's own forward on the same step first and
# compare its logits and auxiliary hidden states with the routed forward's
# ("control": SGLang's forward twice, the run-to-run floor the check is read against)
PARITY_MODE = os.environ.get("SGLANG_NVIDIA_MEGA_PARITY", "0")
PARITY = PARITY_MODE in ("1", "control")
# Test only (with PARITY): sort each row's sparse attention indices before FlashMLA in
# both arms, so the indexer top-k's emission order (atomic arrival order) does not
# change the attention's accumulation order between the two forwards
PARITY_SORT_TOPK = os.environ.get("SGLANG_NVIDIA_MEGA_PARITY_SORT_TOPK", "0") == "1"
# Test only: the slots build may fill (hop_a, moe, gate; draft_post, draft_seam: the
# draft stages on the fused posts, and each stage's MoE post writing the next stage's
# input; draft_tail: the draft head combines the BF16 streams; draft_qkv_a: the draft's
# wq_a + wkv as one GEMM, off until its KV bytes are declared; "dual" runs SGLang's own
# MoE on the dual-stream path); default all but draft_qkv_a
SLOTS = frozenset(
    os.environ.get(
        "SGLANG_NVIDIA_MEGA_SLOTS", "hop_a,moe,gate,draft_post,draft_seam,draft_tail"
    ).split(",")
) - {""}


def model_disable_reasons(model, hf_config) -> list[str]:
    from sglang.srt.models.deepseek_v4 import DeepseekV4CudaDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    m = model.model
    reasons = []
    if not m.hc_pre_from_prev_sublayer:
        reasons.append("needs the mHC pre-mix carried from the previous sublayer")
    if get_parallel().attn_dp_size > 1:
        reasons.append("attention DP: the routed step reads replicated token ids")
    other = [
        i
        for i in range(m.start_layer, m.end_layer)
        if not isinstance(m.layers[i], DeepseekV4CudaDecoderLayer)
    ]
    if other:
        reasons.append(f"layers {other}: not SGLang's CUDA decoder layer")
    return reasons


def draft_disable_reasons(model, hf_config) -> list[str]:
    """What the framework does not check for a draft spec."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4CudaDecoderLayer
    from sglang.srt.runtime_context import get_parallel

    reasons = []
    if get_parallel().tp_size != 4:
        reasons.append(f"the draft needs TP 4, not {get_parallel().tp_size}")
    if not model.hc_pre_from_prev_sublayer:
        reasons.append("the draft needs the mHC pre-mix carried between stages")
    other = [
        i
        for i, stage in enumerate(model.stages)
        if not isinstance(stage, DeepseekV4CudaDecoderLayer)
    ]
    if other:
        reasons.append(f"draft stages {other}: not SGLang's CUDA decoder layer")
    return reasons


class Routed:
    """One rank's routed step: the model, the slot plan per width, the kernels, and
    the forward."""

    def __init__(self, lm, slots: dict[int, slot_plan.Slots], runner=None, moe=None):
        self.lm, self.model, self.slots, self.runner = lm, lm.model, slots, runner
        self.moe = moe or {}  # layer id -> MoeSlot

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__
    ):
        if PARITY and not torch.cuda.is_current_stream_capturing():
            return self._with_parity(
                plan, input_ids, positions, forward_batch, input_embeds
            )
        return self._forward(plan, input_ids, positions, forward_batch, input_embeds)

    def _forward(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4ForCausalLM.forward`` for a routed step."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        lm = self.lm
        input_ids, input_embeds = lm.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            out = self._model_forward(
                plan, input_ids, positions, forward_batch, input_embeds
            )
        aux = None
        if lm.capture_aux_hidden_states:
            out, aux = out
        hidden, pre_hc_head = out
        return lm.logits_processor(
            input_ids,
            hidden,
            lm.lm_head,
            forward_batch,
            aux,
            hidden_states_before_norm=None if aux is not None else pre_hc_head,
        )

    def _model_forward(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4Model.forward``: the embedding, the layers, the head."""
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast, hc_combine
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        m = self.model
        try:
            x = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
            x = hc_broadcast(x, m.hc_mult)
            input_ids_global = getattr(forward_batch, "input_ids_global", input_ids)
            aux = [] if m.dspark_layers_to_capture is not None else None
            # the compressor caches each step's rotary tables on the batch
            for attr in ("freqs_cis_c4", "freqs_cis_c128"):
                if hasattr(forward_batch, attr):
                    delattr(forward_batch, attr)
            residual, pre = self._layers(
                plan, x, positions, forward_batch, input_ids, input_ids_global, aux
            )
        finally:
            residual_batch.clear(forward_batch)
        pre_hc_head = residual.flatten(1)
        hidden = m.norm(hc_combine(pre_hc_head.float(), pre, m.hc_mult, residual.dtype))
        if aux is not None:
            return (hidden, pre_hc_head), aux
        return hidden, pre_hc_head

    def _layers(self, plan, x, positions, fb, input_ids, input_ids_global, aux):
        """The layer loop of ``DeepseekV4Model._forward_layers_hc_pre_from_prev`` on
        CUDA for a verify step: Engram and capture seams, then each layer."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.models import deepseek_v4_boundary as hc_boundary
        from sglang.srt.models import deepseek_v4_mhc as mhc

        m = self.model
        slots = self.slots[plan.key.rows]
        capture = m.dspark_layers_to_capture if aux is not None else ()
        hash_ids = (
            m.engram_hasher(input_ids, fb) if m.engram_hasher is not None else None
        )
        state = mhc.HcState(x)
        residual_batch.start(fb)
        hidden = residual_batch.set_written(state, fb)
        recorder = get_global_expert_distribution_recorder()
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            engram = layer.engram
            if engram is not None or i in capture:
                call = hc_boundary.LayerCall(input_ids, input_ids_global)
                next_boundary = (
                    hc_boundary.HcRead(layer, True).bind(fb, call).next_boundary
                )
                hidden = state = hc_boundary.materialize(hidden, fb, next=next_boundary)
            if engram is not None:
                before = state.residual
                hidden = engram(
                    state.residual,
                    hash_ids[:, engram.layer_hash_index],
                    fb,
                    cp_all_tokens=False,
                )
                if (
                    m.config.model_type == "deepseek_v41"
                    and m.config.vision_n_layers > 0
                ):
                    image = input_ids == m.config.image_token_id
                    hidden = torch.where(image[:, None, None], before, hidden)
                state = state.with_residual(hidden)
                hidden = residual_batch.set_written(state, fb)
            if i in capture:
                # the draft head reads the attention input of its target layers
                aux.append(state.residual.mean(dim=1))
            with recorder.with_current_layer(i):
                hidden = self._layer(
                    layer, slots, positions, hidden, input_ids, fb, input_ids_global
                )
        state = residual_batch.read_output(hidden, fb, hc_boundary.FinalRead(), None)
        residual_batch.clear(fb)
        state = state.materialized(m.layers[m.end_layer - 1].hc_cfg)
        return state.residual, state.pre

    def _layer(self, layer, slots, positions, hidden, input_ids, fb, input_ids_global):
        if not (slots.hop_a or slots.routed or "dual" in SLOTS):
            return layer(
                positions=positions,
                hidden_states=hidden,
                input_ids=input_ids,
                forward_batch=fb,
                input_ids_global=input_ids_global,
            )
        return run_layer(
            layer,
            positions,
            hidden,
            input_ids,
            fb,
            input_ids_global,
            cross_layer=True,
            runner=self.runner,
            hop_a=slots.hop_a,
            moe=self.moe.get(layer.layer_id) if slots.routed else None,
            fast_gate=slots.fast_gate,
        )

    def _with_parity(self, plan, input_ids, positions, forward_batch, input_embeds):
        """SGLang's forward, then the routed one, on the same step; logs the largest
        difference of the logits and the auxiliary hidden states."""
        lm = self.lm
        want = type(lm).forward(lm, input_ids, positions, forward_batch, input_embeds)
        want = (_clone(want.next_token_logits), _clone(want.hidden_states))
        if PARITY_MODE == "control":
            got = type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds
            )
        else:
            got = self._forward(plan, input_ids, positions, forward_batch, input_embeds)
        diffs = [
            _max_abs(a, b)
            for a, b in zip(want, (got.next_token_logits, got.hidden_states))
        ]
        logger.info(
            "sgl-nvidia-mega parity%s %s rows=%d logits_max_abs=%s hidden_max_abs=%s",
            " control" if PARITY_MODE == "control" else "",
            plan.key.forward_mode,
            plan.key.rows,
            *diffs,
        )
        return got


_ENGAGED: set = set()


def _engaged(slot: str, rows: int) -> None:
    """Log once per slot and width that the model's dispatch took the slot."""
    if (slot, rows) not in _ENGAGED:
        _ENGAGED.add((slot, rows))
        logger.info("sgl-nvidia-mega engaged %s rows=%d", slot, rows)


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b):
    """max |a - b|, that over max |a|, and the fraction of elements that differ."""
    if a is None or b is None:
        return "n/a" if a is None and b is None else "missing"
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.float() - b.float()).abs()
    peak = a.float().abs().max().clamp_min(1e-30)
    return (
        f"{d.max().item():.3g} (rel {(d.max() / peak).item():.2e}, "
        f"{(d > 0).float().mean().item():.2%} differ)"
    )


def run_layer(
    layer,
    positions,
    hidden,
    input_ids,
    fb,
    input_ids_global,
    *,
    cross_layer: bool,
    runner,
    hop_a: bool,
    moe,
    fast_gate: bool,
    mhc_fusions=None,
    accepts_mxfp8=None,
):
    """``DeepseekV4CudaDecoderLayer.forward`` with this package's slots: SGLang's
    stage boundaries own the residual streams, the mHC stats and the posts. At hop-A
    widths wo_b publishes into the Lamport hop, which the attention post runs as its
    fused reduction; with a MoE slot the routed experts (and at fast-gate widths the
    router's top-k) run on the GEMM pair inside SGLang's dual-stream MoE."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.models import deepseek_v4_boundary as hc_boundary
    from sglang.srt.models import deepseek_v4_mhc as mhc

    # mhc_fusions: the call's choice of SGLang's fused posts (None: the layer's own)
    extra = {} if mhc_fusions is None else {"mhc_fusions": mhc_fusions}
    if accepts_mxfp8 is not None:
        extra["accepts_mxfp8"] = accepts_mxfp8
    call = hc_boundary.LayerCall(input_ids, input_ids_global, cross_layer, **extra)
    hidden = layer.attn_boundary.prepare(hidden, fb, read_context=call)
    quantized = residual_batch.written_residual(fb).quantized
    attn = layer.self_attn
    with attn.maybe_use_decode_attn_tp(fb):
        out = None
        if hop_a:
            o = attn.forward_before_wo_b(
                x=hidden, positions=positions, forward_batch=fb, x_quant=quantized
            )
            # wo_b's input as _project_wo_b forms it: [rows, heads * dim]
            partial = runner.publish_wo_b(
                attn.wo_b, o.flatten(1) if isinstance(o, torch.Tensor) else o
            )
            out = (
                attn._project_wo_b(o, defer_all_reduce=True)
                if partial is None
                else mhc.AttnOutput(partial, fused_post=runner.hop_a)
            )
        else:
            out = layer._run_boundary_attention(hidden, positions, fb, quantized)
        hidden = layer.attn_boundary.finish(out, fb)
    hidden = layer.ffn_boundary.prepare(hidden, fb, read_context=call)
    if hop_a:
        # the attention post ran by now; it must have been the Lamport consumer (on
        # the host, so a width that never defers its sum fails before capture)
        runner.check_consumed()
    transport = layer.ffn_boundary.transport_context(fb)
    with layer.ffn_boundary.exit(fb) as output:
        if moe is None and "dual" in SLOTS and not transport.do_shared_local:
            # test only: SGLang's own MoE on the dual-stream path the MoE slot uses,
            # which an eager SGLang step does not take
            mlp = layer.mlp
            hidden = mlp.forward_normal_dual_stream(
                hidden,
                None,
                transport.input_ids,
                input_ids_global=transport.input_ids_global,
                num_token_non_padded=fb.moe_num_token_non_padded(),
                use_vision_topk=mlp.gate.e_score_correction_bias_vl is not None,
                return_moe_output=transport.return_moe_output,
                leave_output_sum=True,
            )
        elif moe is None or transport.do_shared_local:
            hidden = layer.mlp(
                hidden,
                fb,
                input_ids=transport.input_ids,
                input_ids_global=transport.input_ids_global,
                skip_shared_experts=transport.do_shared_local,
                return_moe_output=transport.return_moe_output,
                leave_output_sum=True,
            )
        else:
            mlp = layer.mlp
            hidden = mlp.forward_normal_dual_stream(
                hidden,
                None,
                transport.input_ids,
                input_ids_global=transport.input_ids_global,
                num_token_non_padded=fb.moe_num_token_non_padded(),
                use_vision_topk=mlp.gate.e_score_correction_bias_vl is not None,
                return_moe_output=transport.return_moe_output,
                leave_output_sum=True,
                select_experts=moe.select if fast_gate and moe.gate else None,
                run_routed_experts=moe.run,
            )
        return output.finish(hidden)


class _GateTopK(msgspec.Struct):
    """The router's logits and routing inputs, for gate_topk.cu inside the routed call."""

    router_logits: torch.Tensor
    bias: torch.Tensor
    scale: float
    num_valid: torch.Tensor
    alt: object
    eps: float


class MoeSlot:
    """One MoE layer's routed-expert pair as the two callables of SGLang's dual-stream
    MoE: the router's top-k (fast gate) and the routed experts."""

    def __init__(self, mlp, gemm, *, gate: bool):
        from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

        self.mlp, self.gemm, self.gate = mlp, gemm, gate
        config = mlp.topk.topk_config
        self.scale = (
            float(config.routed_scaling_factor)
            if config.apply_routed_scaling_factor_on_output
            and config.routed_scaling_factor is not None
            else 1.0
        )
        self.bias = config.correction_bias_for_dtype(torch.float32) if gate else None
        self.eps = _RENORMALIZE_SUM_EPSILON

    def select(
        self, *, hidden_states, router_logits, input_ids_global, num_token_non_padded,
        use_vision_topk,
    ):  # fmt: skip
        if not self.gemm.supports_gate_topk(router_logits, self.bias):
            raise RuntimeError(
                f"sgl-nvidia-mega: the fast gate cannot take router logits "
                f"{router_logits.dtype} {tuple(router_logits.shape)}"
            )
        rows = router_logits.shape[0]
        num_valid = (
            num_token_non_padded
            if num_token_non_padded is not None
            else torch.full((1,), rows, dtype=torch.int32, device=router_logits.device)
        )
        alt, eps = None, 0.0
        if use_vision_topk:
            # vision routing: image tokens rank with the alternate bias
            alt = (
                self.mlp.gate.e_score_correction_bias_vl,
                input_ids_global,
                int(self.mlp.config.image_token_id),
            )
            eps = self.eps
        return _GateTopK(router_logits, self.bias, self.scale, num_valid, alt, eps)

    def run(self, *, hidden_states, topk_output, pre_quant_input):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            routed_hidden_size,
        )

        experts = self.mlp.experts
        ready = None
        if pre_quant_input is not None:
            x_q, x_s, ready = pre_quant_input
        else:
            x_q, x_s = experts.quant_method.quantize_routed_input(
                hidden_states, routed_hidden_size(experts)
            )
        if isinstance(topk_output, _GateTopK):
            rows, weights, permuted, _ = self.gemm.gate_topk_and_gemms(
                x_q,
                x_s,
                topk_output.router_logits,
                topk_output.bias,
                topk_output.scale,
                topk_output.num_valid,
                input_ready=ready,
                alt=topk_output.alt,
                renormalize_epsilon=topk_output.eps,
            )
        else:
            if ready is not None:
                torch.cuda.current_stream().wait_event(ready)
            rows, weights, permuted = self.gemm(
                x_q, x_s, topk_output.topk_ids, topk_output.topk_weights
            )
        return FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=rows,
            expert_weights=weights,
            # flat [rows * top_k] slot -> row map, as FlashInfer's routed MoE returns it
            expanded_idx_to_permuted_idx=permuted.view(-1),
            top_k=self.gemm.top_k,
        )


def moe_seam() -> str:
    """Why SGLang cannot run the MoE slot, or "" when it can: the dual-stream MoE's
    top-k and routed-expert slots."""
    import inspect

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    params = inspect.signature(DeepseekV2MoE.forward_normal_dual_stream).parameters
    if not {"select_experts", "run_routed_experts"} <= set(params):
        return "this SGLang lacks DeepseekV2MoE.forward_normal_dual_stream's slots"
    return ""


def _gate_applies(mlp) -> bool:
    """The DSV4.1 target router gate_topk.cu reproduces bit for bit."""
    config = mlp.topk.topk_config
    bias_vl = getattr(mlp.gate, "e_score_correction_bias_vl", None)
    return (
        not getattr(mlp, "is_hash", False)
        and config.scoring_func == "sqrtsoftplus"
        and getattr(config, "sqrtsoftplus_log1p", False)
        and config.renormalize
        and config.num_fused_shared_experts == 0
        and not config.use_grouped_topk
        and config.custom_routing_function is None
        and config.correction_bias is not None
        and (bias_vl is None or bias_vl.dtype == torch.float32)
    )


def _moe_slots(runner, layers, key=lambda i, layer: layer.layer_id) -> dict:
    """The MoE slot of every layer whose pair built and whose MoE runs SGLang's
    dual-stream path (alternate stream, no fused shared experts)."""
    slots = {}
    for i, layer in layers:
        mlp, gemm = layer.mlp, runner.moe.get(i)
        if gemm is None or mlp.alt_stream is None or mlp.num_fused_shared_experts:
            continue
        slots[key(i, layer)] = MoeSlot(
            mlp,
            gemm,
            gate="gate" in SLOTS
            and gemm.gate_topk_kernel is not None
            and _gate_applies(mlp),
        )
    return slots


def draft_post_seam() -> str:
    """Why SGLang cannot run the draft stages on its fused posts, or "": a layer call
    that turns on the mHC fusions of a stage built without them."""
    from sglang.srt.models import deepseek_v4_boundary as hc_boundary

    if "mhc_fusions" not in hc_boundary.LayerCall.__struct_fields__:
        return "this SGLang lacks LayerCall.mhc_fusions"
    return ""


def hop_a_seams() -> str:
    """Why SGLang cannot run the hop-A slot, or "" when it can: the attention output
    before wo_b (``MQALayer.forward_before_wo_b``), and an attention output that names
    the fused post reducing it (``AttnOutput.fused_post``)."""
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.models.deepseek_v4 import MQALayer

    missing = []
    if not hasattr(MQALayer, "forward_before_wo_b"):
        missing.append("MQALayer.forward_before_wo_b")
    if "fused_post" not in getattr(mhc.AttnOutput, "_fields", ()):
        missing.append("AttnOutput.fused_post")
    return f"this SGLang lacks {', '.join(missing)}" if missing else ""


def _gather_errors(group, error: str, value, what: str):
    """All-gather each rank's (error, value) on the CPU group; raise on every rank if
    any rank failed or the ranks disagree, so none waits in a collective alone."""
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, (error, value), group=group.cpu_group)
    failed = [f"rank {r}: {e}" for r, (e, _) in enumerate(gathered) if e]
    if failed:
        raise RuntimeError(f"sgl-nvidia-mega: {what} failed: " + "; ".join(failed))
    if any(v != value for _, v in gathered):
        raise RuntimeError(f"sgl-nvidia-mega: the ranks disagree on {what}: {gathered}")
    return value


def _build_hop_a(runner, model, hf_config, ctx, group) -> list[int]:
    """The Lamport mailbox, consumer and publishing GEMM; the rows it takes."""
    m = model.model
    wo_b = m.layers[m.start_layer].self_attn.wo_b
    rows = [w for w in ctx.widths if w <= slot_plan.HOP_A_MAX_ROWS]
    error, taken = "", []
    try:
        runner.build_hop(
            group=group.device_group,
            hidden=hf_config.hidden_size,
            hc_mult=hf_config.hc_mult,
            top_k=hf_config.num_experts_per_tok,
            hop_a_rows=slot_plan.HOP_A_MAX_ROWS,
            deadline_s=ctx.deadline_s,
        )
        taken = runner.precompile_publish(wo_b, rows)
    except Exception as e:  # gathered, then raised on every rank
        error = f"{type(e).__name__}: {e}"
    return _gather_errors(group, error, taken, "the hop-A build")


def _build_moe(runner, layers, *, finalize: bool, gate_topk: bool, group) -> str:
    """A routed-expert GEMM pair per MoE layer, reading SGLang's loaded MXFP4
    experts in place; the reason none was built, or ""."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        SERVED_MAX_ROWS,
    )

    reason = ""
    try:
        for i, layer in layers:
            experts = layer.mlp.experts
            method = experts.quant_method
            if not isinstance(method, Mxfp4FlashinferTrtllmMoEMethod):
                raise ValueError(f"layer {i}: experts on {type(method).__name__}")
            if experts.num_local_experts != experts.num_experts:
                raise ValueError(f"layer {i}: expert parallelism")
            runner.add_moe(
                i,
                w13=experts.w13_weight,
                w13_scale=experts.w13_weight_scale_inv,
                w2=experts.w2_weight,
                w2_scale=experts.w2_weight_scale_inv,
                clamp=method._gemm1_clamp_limit_tensor,
                top_k=experts.top_k,
                max_rows=SERVED_MAX_ROWS,
                finalize=finalize,
                gate_topk=gate_topk,
            )
    except Exception as e:  # gathered, then the slot is off on every rank
        reason = f"{type(e).__name__}: {e}"
        runner.moe.clear()
    built = _gather_errors(group, "", (reason == "", len(runner.moe)), "the MoE build")
    if not built[0]:
        reasons = [None] * group.world_size
        torch.distributed.all_gather_object(reasons, reason, group=group.cpu_group)
        return "; ".join(f"rank {r}: {e}" for r, e in enumerate(reasons) if e)
    return ""


def _install_parity_sorted_topk() -> None:
    """Test only: sort each decode or verify row's valid sparse indices in place
    before SGLang's DSV4 attention reads them (``SGLANG_NVIDIA_MEGA_PARITY_SORT_TOPK``)."""
    import functools

    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
        DSV4AttnMetadata,
    )

    original = DeepseekV4AttnBackend.forward
    if getattr(original, "_sgl_mega_sorted", False):
        return

    @functools.wraps(original)
    def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        core = getattr(self.forward_metadata, "core_attn_metadata", None)
        ratio = getattr(layer, "compress_ratio", 0)
        if (
            ratio
            and isinstance(core, DSV4AttnMetadata)
            and (mode.is_decode() or mode.is_target_verify())
        ):
            table = core.sparse_page_indices(ratio)
            lengths = core.sparse_topk_lengths(ratio)
            if table is not None and lengths is not None:
                rows = table.view(-1, table.shape[-1])
                n = lengths.view(-1)[: rows.shape[0]].to(torch.int64)
                col = torch.arange(rows.shape[1], device=rows.device)
                valid = col[None, :] < n[:, None]
                big = torch.iinfo(rows.dtype).max
                key = torch.where(valid, rows, torch.full_like(rows, big))
                key = key.sort(dim=-1).values
                rows.copy_(torch.where(valid, key, rows))
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)

    forward._sgl_mega_sorted = True
    DeepseekV4AttnBackend.forward = forward
    logger.warning("sgl-nvidia-mega: test-only sorted sparse indices are on")


def build(model, hf_config, ctx) -> Routed:
    from sglang.srt.runtime_context import get_parallel

    if PARITY_SORT_TOPK:
        _install_parity_sorted_topk()

    from sgl_nvidia_mega.deepseek_v41.sm103.runner import Runner

    group = get_parallel().tp_group
    m = model.model
    device = m.layers[m.start_layer].self_attn.wo_b.weight.device
    runner = Runner(device=device, rank=group.rank_in_group)
    reason, taken = hop_a_seams(), []
    if "hop_a" not in SLOTS:
        reason = "off (SGLANG_NVIDIA_MEGA_SLOTS)"
    if not reason:
        taken = _build_hop_a(runner, model, hf_config, ctx, group)
        if not taken:
            reason = "wo_b is not an MXFP8 linear the low-latency GEMM takes"
    layers = [(i, m.layers[i]) for i in range(m.start_layer, m.end_layer)]
    moe_reason = moe_seam() or (
        "" if "moe" in SLOTS else "off (SGLANG_NVIDIA_MEGA_SLOTS)"
    )
    moe = {}
    if not moe_reason:
        moe_reason = _build_moe(
            runner, layers, finalize=False, gate_topk=True, group=group
        )
        moe = _moe_slots(runner, layers)
    facts = slot_plan.Facts(hop_a=not reason, hop_a_reason=reason)
    slots = {
        w: msgspec.structs.replace(s, hop_a=s.hop_a and w in taken)
        for w, s in slot_plan.plan(ctx.widths, facts).items()
    }
    logger.info(
        "sgl-nvidia-mega deepseek_v41: widths %s; hop A at %s rows (%s; mailbox %.1f "
        "MiB); routed experts on the GEMM pair in %d layers (%s), fast gate in %d of "
        "them at %s rows",
        list(ctx.widths),
        [w for w, s in slots.items() if s.hop_a],
        reason or "built",
        runner.mailbox_bytes() / 2**20,
        len(moe),
        moe_reason or "built",
        sum(m.gate for m in moe.values()),
        [w for w, s in slots.items() if s.fast_gate],
    )
    return Routed(model, slots, runner, moe)


class Draft:
    """One rank's routed DSpark draft step."""

    def __init__(
        self,
        lm,
        slots: dict[int, slot_plan.Slots],
        runner=None,
        moe=None,
        *,
        post=False,
        seam=False,
        tail_bf16=False,
        mxfp8_input=False,
    ):
        self.lm, self.slots, self.runner = lm, slots, runner
        self.moe = moe or {}  # stage index -> MoeSlot
        # the stages run on SGLang's fused all-reduce + mHC posts; with the seam each
        # stage's MoE post also writes the next stage's attention input
        self.post, self.seam = post, seam
        self.tail_bf16 = tail_bf16
        # each stage's attention reads the post's MXFP8 rows (the fused wqkv_a takes them)
        self.mxfp8_input = mxfp8_input

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__
    ):
        if PARITY and not torch.cuda.is_current_stream_capturing():
            lm = self.lm
            want = type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds
            )
            want = _clone(want.hidden_states)
            if PARITY_MODE == "control":
                got = type(lm).forward(
                    lm, input_ids, positions, forward_batch, input_embeds
                )
            else:
                got = self._forward(
                    plan, input_ids, positions, forward_batch, input_embeds
                )
            logger.info(
                "sgl-nvidia-mega parity%s draft %s rows=%d hidden_max_abs=%s",
                " control" if PARITY_MODE == "control" else "",
                plan.key.forward_mode,
                plan.key.rows,
                _max_abs(want, got.hidden_states),
            )
            return got
        return self._forward(plan, input_ids, positions, forward_batch, input_embeds)

    def _forward(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4ForCausalLMDSpark.forward`` for a routed step: each stage
        through SGLang's stage boundaries, then the head's combine."""
        from sglang.kernels.ops.layernorm.mhc import hc_combine
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.models import deepseek_v4_boundary as hc_boundary

        lm = self.lm
        rows = plan.key.rows
        x = lm.forward_embed(input_ids) if input_embeds is None else input_embeds
        if self.post:
            _engaged("draft_seam" if self.seam else "draft_post", rows)
        try:
            residual_batch.start(forward_batch)
            # draft stages have no recorder layer indices
            slots = self.slots[rows]
            with get_global_expert_distribution_recorder().disable_this_region():
                for i, stage in enumerate(lm.stages):
                    moe = self.moe.get(i) if slots.routed else None
                    if moe is None and not self.post:
                        x = stage(positions, x, forward_batch)
                        continue
                    # DSparkV4CudaStage.forward: no token ids; cross-stage fusion only
                    # with the seam
                    x = run_layer(
                        stage,
                        positions,
                        x,
                        None,
                        forward_batch,
                        None,
                        cross_layer=self.seam,
                        runner=self.runner,
                        hop_a=False,
                        moe=moe,
                        fast_gate=False,
                        mhc_fusions=True if self.post else None,
                        accepts_mxfp8=True if self.mxfp8_input else None,
                    )
            state = residual_batch.read_output(
                x, forward_batch, hc_boundary.FinalRead(), None
            )
        finally:
            residual_batch.clear(forward_batch)
        streams = state.residual.flatten(1)
        if self.tail_bf16:
            # hc_combine widens each load to FP32: the same result without the copy
            _engaged("draft_tail", rows)
        else:
            streams = streams.float()
        x = hc_combine(streams, state.pre, lm.hc_mult, state.residual.dtype)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)


def build_draft(model, hf_config, ctx) -> Draft:
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41.sm103.runner import Runner

    group = get_parallel().tp_group
    stages = list(enumerate(model.stages))
    device = next(model.parameters()).device
    runner = Runner(device=device, rank=group.rank_in_group)
    # the draft's stages route with SGLang's top-k; their posts finalize the
    # deferred output (fused or not), so the pair needs no finalize kernel
    moe_reason = moe_seam() or (
        "" if "moe" in SLOTS else "off (SGLANG_NVIDIA_MEGA_SLOTS)"
    )
    moe = {}
    if not moe_reason:
        moe_reason = _build_moe(
            runner, stages, finalize=False, gate_topk=False, group=group
        )
        moe = _moe_slots(runner, stages, key=lambda i, _: i)
    slots = slot_plan.plan(
        ctx.widths, slot_plan.Facts(hop_a=False, hop_a_reason="draft")
    )
    post_reason = draft_post_seam() or (
        "" if "draft_post" in SLOTS else "off (SGLANG_NVIDIA_MEGA_SLOTS)"
    )
    seam = not post_reason and "draft_seam" in SLOTS
    from sgl_nvidia_mega.deepseek_v41.draft_layers import install_qkv_a

    qkv_reason = (
        install_qkv_a(model.stages)
        if "draft_qkv_a" in SLOTS
        else "off (SGLANG_NVIDIA_MEGA_SLOTS)"
    )
    logger.info(
        "sgl-nvidia-mega deepseek_v41 draft: widths %s; routed experts on the GEMM "
        "pair in %d stages (%s); fused posts %s; seam %s; fused wqkv_a %s",
        list(ctx.widths),
        len(moe),
        moe_reason or "built",
        post_reason or "on",
        "on" if seam else "off",
        qkv_reason or "on",
    )
    return Draft(
        model,
        slots,
        runner,
        moe,
        post=not post_reason,
        seam=seam,
        tail_bf16="draft_tail" in SLOTS,
        mxfp8_input=not post_reason and not qkv_reason,
    )
