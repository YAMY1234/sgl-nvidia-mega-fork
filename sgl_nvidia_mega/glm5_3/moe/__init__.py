# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (the NVFP4 TRT-LLM weight processing the layout self-test replays).
"""CuTe DSL small-M NVFP4 MoE in the GLM-5.3 verify step (slot ``moe``); the stock
step is in ``step``.

At the covered widths (1..16 verify rows, ``moe_split``) each MoE layer runs the stock
router GEMM (``mlp.gate``), ``moe_route_top8`` (the top-8 selection of the stock TRT-LLM
NVFP4 MoE routing: sigmoid scores, correction bias, renormalized weights x the routed
scaling factor), one engine launch for the routed experts and the BF16 shared expert, and
``moe_finalize`` (the layer's un-reduced output, which the next hop step all-reduces as
it does the stock MoE output). When the layer's ``o_proj`` hop has already routed the
rows (``step.routing``), the router GEMM and ``moe_route_top8`` are skipped, the engine
runs without the finalize, and the step returns ``MoeRecords`` (the per-expert outputs)
for the MLP-output hop, which finalizes them inside its all-reduce.

The engine reads the server's processed routed FC1 weights in place. FC2 is read from
compact per-quarter records and the shared expert from a packed copy; both are derived
at load and registered (226.5 MB + 9.4 MB per MoE layer and rank). ``build`` declines
(``FamilyDeclined``; with slot ``moe`` requested the server stops) unless the engine API
accepts the configuration, every MoE layer has the expected tensors and the server's own
NVFP4 weight processing passes the engine's layout self-test.

With ``SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_LOCALIZE=1`` on a GPU with two locality domains
(``sm103/moe/locality.py``) the routed FC1 weights and scales are re-bound to copies homed
per domain (experts 0..127 on domain 0; the originals are freed, net memory unchanged),
the FC2 records and shared packs are made on the domains' banks, and the engines run
their die-local forms (same output bits). Otherwise the build is the stock one;
a device with one domain, or a failed topology probe, logs why and stays stock.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from sgl_nvidia_mega.glm5_3 import HIDDEN
from sgl_nvidia_mega.glm5_3.moe.step import MlpSteps
from sgl_nvidia_mega.glm5_3.plan import MAX_ROWS, FamilyDeclined
from sgl_nvidia_mega.glm5_3.runtime.compile import compile_cached, cu_stream

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "align_fp4_moe_weights_for_flashinfer_trtllm",
    ),
    # the replaced model's modules, weights and quant method
    "srt/models/deepseek_v2.py": (
        "DeepseekV2MoE",
        "model",
        "layers",
        "mlp",
        "gate",
        "experts",
        "shared_experts",
        "routed_scaling_factor",
        "e_score_correction_bias",
        "layer_id",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "g1_alphas",
        "g1_alphas_up",
        '"g2_alphas"',
        '"w13_input_scale_quant"',
        '"w2_input_scale_quant"',
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("moe_runner_config",),
    "srt/layers/moe/moe_runner/base.py": (
        "activation",
        "is_gated",
        "gemm1_clamp_limit",
        "swiglu_limit",
        "gemm1_alpha",
    ),
}

EXPERTS = 256
TOP_K = 8
INTER = 256  # routed and shared intermediate size per rank (TP 8)
# The engine's row groups: at most 4 groups of at most 4 rows (one group: g1, two: g2,
# three or four: the four-group engine)
GROUP_ROWS = 4
MAX_GROUPS = 4


def moe_split(rows: int) -> Tuple[int, ...]:
    """The engine's row groups for ``rows`` verify rows: groups of 4 rows, the
    remainder last (4 -> (4,), 6 -> (4, 2), 12 -> (4, 4, 4)). The engine's groups are
    token sets over one expert list, so they need not follow the request boundaries."""
    split = (GROUP_ROWS,) * (rows // GROUP_ROWS) + (
        (rows % GROUP_ROWS,) if rows % GROUP_ROWS else ()
    )
    assert sum(split) == rows and len(split) <= MAX_GROUPS, (rows, split)
    return split


@dataclass
class MoeRecords:
    """The engine's per-expert outputs of one MoE layer, for a hop that finalizes them
    inside its all-reduce (row stride ``rows``): ``pq`` [4, rows, 8, 6144] fp32 (the four
    quarters' FC2 outputs per top-k slot), ``shpartials`` [4, rows, 6144] fp32 (shared expert),
    ``ids_fin`` / ``w_fin`` / ``g2s`` [rows, 8] (slot expert id or -1, routing weight, FC2
    scale)."""

    pq: torch.Tensor
    shpartials: torch.Tensor
    ids_fin: torch.Tensor
    w_fin: torch.Tensor
    g2s: torch.Tensor


def _process_like_server(gate, up, gate_sf, up_sf, w2, w2_sf):
    """The server's NVFP4 TRT-LLM weight processing on a synthetic layer: the loader's
    w13 order for this backend (up rows first) and the same processing call."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        align_fp4_moe_weights_for_flashinfer_trtllm,
    )

    e = gate.shape[0]
    f8 = torch.float8_e4m3fn
    p = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
    layer = torch.nn.Module()
    layer.w13_weight = p(torch.cat([up, gate], dim=1).contiguous())
    layer.w2_weight = p(w2.contiguous())
    layer.w13_weight_scale = p(torch.cat([up_sf, gate_sf], dim=1).view(f8).contiguous())
    layer.w2_weight_scale = p(w2_sf.view(f8).contiguous())
    ones = torch.ones(e, dtype=torch.float32, device=gate.device)
    layer.w2_input_scale_quant = p(ones.clone())
    layer.g1_alphas = p(ones.clone())
    layer.g1_alphas_up = p(ones.clone())
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    align_fp4_moe_weights_for_flashinfer_trtllm(layer)
    return (
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
    )


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _bad(t, dtype, shape=None, numel=None, contiguous=True) -> bool:
    return (
        t is None
        or t.dtype != dtype
        or (shape is not None and tuple(t.shape) != shape)
        or (numel is not None and t.numel() != numel)
        or (contiguous and not t.is_contiguous())
    )


def _layer_reason(mlp) -> Optional[str]:
    """Why this MoE layer cannot run on the engine (None if it can)."""
    ex = mlp.experts
    for name, shape in (
        ("w13_weight", (EXPERTS, 2 * INTER, HIDDEN // 2)),
        ("w2_weight", (EXPERTS, HIDDEN, INTER // 2)),
    ):
        if _bad(getattr(ex, name, None), torch.uint8, shape):
            return f"{name} is not contiguous uint8 {shape}"
    for name, n in (
        ("w13_weight_scale", EXPERTS * 2 * INTER * HIDDEN // 16),
        ("w2_weight_scale", EXPERTS * HIDDEN * INTER // 16),
    ):
        if _bad(getattr(ex, name, None), torch.float8_e4m3fn, numel=n):
            return f"{name} is not contiguous e4m3 with {n} elements"
    for name in ("g1_alphas", "g1_alphas_up", "g2_alphas"):
        if _bad(getattr(ex, name, None), torch.float32, (EXPERTS,), contiguous=False):
            return f"{name} is not fp32 [{EXPERTS}]"
    if not torch.equal(ex.g1_alphas, ex.g1_alphas_up):
        return "gate and up GEMM1 scales differ (the engine applies one per expert)"
    for name in ("w13_input_scale_quant", "w2_input_scale_quant"):
        t = getattr(ex, name, None)
        if t is None or t.numel() != 1:
            return f"{name} is not a single scale"
    cfg = ex.moe_runner_config
    if (
        getattr(cfg, "activation", "silu") != "silu"
        or getattr(cfg, "gemm1_clamp_limit", None)
        or getattr(cfg, "swiglu_limit", None)
        or getattr(cfg, "gemm1_alpha", None) is not None
    ):
        return "only an unclamped SiLU-gated MLP is supported"
    gu, dn = mlp.shared_experts.gate_up_proj.weight, mlp.shared_experts.down_proj.weight
    if _bad(gu, torch.bfloat16, (2 * INTER, HIDDEN), contiguous=False):
        return "shared expert gate_up_proj is not bf16 [512, 6144]"
    if _bad(dn, torch.bfloat16, (HIDDEN, INTER), contiguous=False):
        return "shared expert down_proj is not bf16 [6144, 256]"
    bias = mlp.gate.e_score_correction_bias
    if _bad(bias, torch.float32, (EXPERTS,), contiguous=False):
        return "router correction bias is not fp32 [256]"
    return None


class MoeSteps(MlpSteps):
    """The MoE layers of one verify row count on the engine; dense layers stay stock."""

    def __init__(self, glue: MoeGlue, rows: int) -> None:
        self.glue, self.rows = glue, rows
        self.split = moe_split(rows)

    def forward(self, layer, x, buffer: Optional[torch.Tensor], step):
        glue = self.glue
        lt = glue.layers.get(layer.layer_id)
        if lt is None:
            return super().forward(layer, x, buffer, step)
        m = x.shape[0]
        if m != self.rows or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise RuntimeError(
                f"fused verify MoE: unexpected input {tuple(x.shape)} {x.dtype} at {self.rows} rows"
            )
        routing, step.routing = step.routing, None
        if routing is not None:
            # routed by this layer's o_proj hop; its MLP-output hop finalizes the records
            ids, weights = routing
            glue.engine(lt, x, ids, weights, self.split, what=1)
            return glue.records(m)
        ids, weights = glue.ids[:m], glue.weights[:m]
        gate = layer.mlp.gate
        scale = layer.mlp.routed_scaling_factor
        glue.route(gate(x), gate.e_score_correction_bias, scale, ids, weights)
        if (
            isinstance(buffer, torch.Tensor)
            and buffer.dtype == torch.bfloat16
            and buffer.dim() == 2
            and buffer.shape[0] >= m
            and buffer.shape[1] == HIDDEN
            and buffer.is_contiguous()
        ):
            out = buffer[:m]
        else:
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=x.device)
        glue.engine(lt, x, ids, weights, self.split, out=out, what=3)
        return out


class MoeGlue:
    """Derived MoE weights, compiled engines and per-row-count steps."""

    def __init__(
        self, api, layers: Dict[int, object], specs, localize=None, nclusters=None
    ) -> None:
        self.api, self.specs, self.layers = api, specs, layers
        # (topology, banks, locality blocks) of the die-local form, else None
        self.localize = localize
        self.nclusters = nclusters
        self.device = dev = next(iter(layers.values())).w13.device
        self.ids = torch.empty(16, TOP_K, dtype=torch.int32, device=dev)
        self.weights = torch.empty(16, TOP_K, dtype=torch.float32, device=dev)
        self.fn: Dict[str, object] = {}
        self.engine = None

    def compile(self, rows: Tuple[int, ...]) -> None:
        import cutlass.cute as cute

        modes = {self.api.mode_for(moe_split(m)) for m in rows if 1 <= m <= MAX_ROWS}
        for name in sorted({"route", "finalize"} | modes):
            kernel, args = self.specs[name]
            constexprs = {"unit": name, "shared": True}
            if hasattr(kernel, "knobs"):  # the g4 engine's opt-in forms
                constexprs.update(kernel.knobs())
            self.fn[name] = compile_cached(
                kernel,
                constexprs,
                lambda k=kernel, a=args: cute.compile(k, *a),
            )
        nclusters = self.nclusters or self.api.default_clusters(self.device)
        self.engine = self.api.SmallMMoe(
            self.fn,
            nclusters,
            self.device,
            shared=True,
            localize=self.localize is not None,
        )

    def records(self, m: int) -> MoeRecords:
        sc = self.engine.scratch
        return MoeRecords(
            pq=sc.partials[: 4 * m * TOP_K * HIDDEN].view(4, m, TOP_K, HIDDEN),
            shpartials=sc.shared_partials[: 4 * m * HIDDEN].view(4, m, HIDDEN),
            ids_fin=sc.slot_ids[: m * TOP_K].view(m, TOP_K),
            w_fin=sc.slot_weights[: m * TOP_K].view(m, TOP_K),
            g2s=sc.slot_scales[: m * TOP_K].view(m, TOP_K),
        )

    def route(self, logits, bias, scale, ids, weights) -> None:
        from cutlass import Float32, Int32, Int64

        m = logits.shape[0]
        if (
            logits.dtype != torch.float32
            or tuple(logits.shape) != (m, EXPERTS)
            or not logits.is_contiguous()
        ):
            raise RuntimeError(
                f"fused verify MoE: router logits {tuple(logits.shape)} {logits.dtype}"
            )
        self.fn["route"](
            Int64(logits.data_ptr()),
            Int64(bias.data_ptr()),
            Float32(float(scale)),
            Int64(ids.data_ptr()),
            Int64(weights.data_ptr()),
            Int32(m),
            cu_stream(logits.device),
        )

    def for_rows(self, m: int) -> Optional[MoeSteps]:
        if (
            self.engine is None
            or not 1 <= m <= MAX_ROWS
            or self.api.mode_for(moe_split(m)) not in self.fn
        ):
            return None
        return MoeSteps(self, m)


def build(model, derived) -> Optional[MoeGlue]:
    """The MoE family of ``model``: None without MoE layers; raises ``FamilyDeclined``
    naming why the engine cannot serve them."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    from sgl_nvidia_mega.glm5_3.sm103.moe import api
    from sgl_nvidia_mega.glm5_3.sm103.moe import weights as W

    def stock(why) -> None:
        raise FamilyDeclined(f"MoE: {why}")

    moe_layers = [l for l in model.model.layers if isinstance(l.mlp, DeepseekV2MoE)]
    if not moe_layers:
        return None
    for layer in moe_layers:
        why = _layer_reason(layer.mlp)
        if why is not None:
            return stock(f"layer {layer.layer_id}: {why}")
    dev = moe_layers[0].mlp.experts.w13_weight.device
    topo, nclusters = _localize_topology(api, dev)
    try:
        specs = api.compile_specs(shared=True, localize=topo is not None)
    except ValueError as e:
        return stock(e)
    try:
        W.layout_self_test(_process_like_server, dev)
    except W.LayoutMismatch as e:
        return stock(f"layout self-test: {e}")
    layers = {}
    loc = None
    if topo is not None:
        from sgl_nvidia_mega.glm5_3.sm103.moe import locality as LOC

        banks = LOC.Banks(dev.index if dev.index is not None else 0)
        blocks = LOC.LocBlocks(topo, len(moe_layers), dev)
        derived.add("moe.loc_blocks", blocks.blocks)
        loc = (topo, banks, blocks)
        relocated = 0
    for li, layer in enumerate(moe_layers):
        ex, sh = layer.mlp.experts, layer.mlp.shared_experts
        records = W.fc2_records(
            _u8(ex.w2_weight.data).reshape(EXPERTS, HIDDEN, INTER // 2),
            _u8(ex.w2_weight_scale.data).reshape(EXPERTS, -1),
        )
        pack = W.shared_pack(sh.gate_up_proj.weight.data, sh.down_proj.weight.data)
        if loc is not None:
            # this layer's routed FC1 weights / scales re-bound to per-domain copies
            # (the server's own MoE path reads the same tensors), FC2 records and
            # shared pack made on the banks
            relocated += LOC.relocate_param(ex, "w13_weight", banks)
            relocated += LOC.relocate_param(ex, "w13_weight_scale", banks)
            records, pack = LOC.to_banks(records, banks), LOC.to_banks(pack, banks)
            torch.cuda.empty_cache()
        records = derived.add("moe.fc2_records", records)
        pack = derived.add("moe.shared_pack", pack)
        lt = api.LayerTensors(
            w13=ex.w13_weight.data,
            w13_scales=_u8(ex.w13_weight_scale.data).reshape(EXPERTS, -1),
            fc2_records=records.view(EXPERTS, 4, 4, -1),
            shared_pack=pack,
            g1_alpha=ex.g1_alphas.data,
            g2_alpha=ex.g2_alphas.data,
            gs1=float(ex.w13_input_scale_quant.item()),
            gs2=float(ex.w2_input_scale_quant.item()),
            loc=None if loc is None else blocks[li],
        )
        lt.check()
        layers[layer.layer_id] = lt
    logger.info(
        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: MoE engine on %d layers", len(layers)
    )
    from sgl_nvidia_mega.glm5_3.sm103.moe import api as _moe_api

    logger.info(
        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: four-request MoE engine form %s, schedule %s",
        _moe_api.g4_knobs(loc is not None),
        _moe_api.g4_schedule_knobs(),
    )
    if loc is not None:
        torch.cuda.synchronize(dev)
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: die-local MoE: %s; FC1 weights and "
            "scales re-bound to per-domain copies (%.1f GiB, originals freed); banks "
            "%.1f GiB on domain 0 / %.1f GiB on domain 1 (FC2 records and shared packs "
            "counted in the derived copies)",
            LOC.describe(topo),
            relocated / 2**30,
            banks.bytes_by_die[0] / 2**30,
            banks.bytes_by_die[1] / 2**30,
        )
        if rank_zero():
            logger.warning(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: die-local MoE: the re-bound FC1 "
                "weights hold experts 0..127 on one die; a MoE step this plugin does not "
                "cover (more rows than its verify widths, prefill) runs SGLang's MoE on "
                "them, which can read one die's memory at a time (the stock engine "
                "schedule on this placement measured 17-43%% slower on GB300). Use it on "
                "servers whose decode steps are all covered (--max-running-requests <= "
                "the covered requests)."
            )
    return MoeGlue(api, layers, specs, localize=loc, nclusters=nclusters)


def rank_zero() -> bool:
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _localize_topology(api, dev):
    """(topology, engine clusters) when the die-local form is requested and this GPU
    can run it, else (None, None) with the reason logged."""
    from sgl_nvidia_mega.glm5_3.sm103.moe import locality as LOC

    if not LOC.requested():
        return None, None
    try:
        nclusters = api.default_clusters(dev)
        topo = LOC.probe(dev, (nclusters, min(nclusters, 32)))
        # the slot rule covers every unit once within each engine's unit cap
        for n, n0, sw, cap, max_d in api.schedule_checks(nclusters, topo.n0):
            LOC.check_schedule(n, n0, sw, cap, max_d)
    except Exception as e:  # noqa: BLE001 (any probe failure keeps the stock form)
        logger.warning(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: %s=1 but the die-local MoE is off "
            "on this GPU (%s: %s); the stock engine serves",
            LOC.LOCALIZE_ENV,
            type(e).__name__,
            str(e)[:300],
        )
        return None, None
    return topo, nclusters
