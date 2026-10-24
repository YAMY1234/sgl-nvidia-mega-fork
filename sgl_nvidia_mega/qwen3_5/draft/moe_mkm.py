# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PERF-DRAFT: the MTP draft's MoE on the target's TP8 MoE path, as TensorRT-LLM's ``MTP_MOE_QUANT=nvfp4`` runs the
quantized draft experts on its target path.

With the draft's routed experts in NVFP4 (``SGLANG_PERFLP_DRAFT_MOE_NVFP4``, the target experts' format and layout),
``SGLANG_PERFDRAFT_MOE_MKM=1`` (default off) runs the draft MoE through the target's slots: the router front and slot
mkm's 7 / 16-token engine up to 16 rows, slot route_topk with the 32 / 64-token engine past them (``moe.step``), on the
draft layer's own weight maps. The engines are the process's (built once per intermediate size); the shared expert
stays the draft's BF16 one on the alternate stream, and the finalize goes to the draft hop (slot draft_fin) or to
SGLang's deferred finalize. Draft steps of fewer than ``SGLANG_PERFDRAFT_MOE_MKM_MIN_ROWS`` rows keep the single launch.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": ("experts", "layer_id"),
    "srt/layers/quantization/modelopt_quant.py": ('"w13_weight"', '"w2_weight"'),
}

ENABLED = os.environ.get("SGLANG_PERFDRAFT_MOE_MKM", "0") == "1"
# one-token steps keep the single launch (BF16 twin): 16.8 us against 20.5 on this path (single-GPU micro, pdraft m1)
MIN_ROWS = int(os.environ.get("SGLANG_PERFDRAFT_MOE_MKM_MIN_ROWS", "2"))
# the widest pass it serves: the draft extend of 8 requests (7 rows each)
MAX_ROWS = 64

# the MoE slots moe.step reads, all off but the ones a width takes
_SLOT_NAMES = (
    "route",
    "mkm",
    "route_topk",
    "route_gemm",
    "route_shared",
    "mkm_m32",
    "mkm_m64",
    "mkm_m128",
    "mkm_m256",
    "mkm_m512",
    "lowm_se",
    "hop_fp8",
)


class DraftMkm:
    """The draft layer's bound for moe.step and the per-width slots it takes."""

    def __init__(self, bound, wide: bool):
        self.bound = bound
        self.wide = wide
        self._noted: set = set()

    @classmethod
    def build(cls, mlp) -> tuple[Optional["DraftMkm"], str]:
        from sgl_nvidia_mega.qwen3_5.moe import mkm, route
        from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as mkm_kernel
        from sgl_nvidia_mega.qwen3_5.sm103.moe import router

        reason = route.unsupported(mlp) or mkm.unsupported(mlp)
        if reason:
            return None, reason
        device = torch.cuda.current_device()
        intermediate = mlp.experts.w2_weight.shape[-1] * 2
        reason = mkm_kernel.m32_unsupported(device, intermediate)
        if reason:
            return None, f"32 / 64-token engines: {reason}"
        # compiled kernels are cached per shape: what the target built is reused, the rest added
        router.build(device, router.MAX_TOKENS)
        router.build_topk(device)
        mkm_kernel.build(device, intermediate, MAX_ROWS)
        wide = intermediate in mkm_kernel.WIDE_ENGINES
        maps = mkm_kernel.weight_maps(mlp.experts.w13_weight, mlp.experts.w2_weight)
        bound = route.Bound(
            route=True,
            mkm={mlp.layer_id: maps},
            mkm_wide=wide,
            # the hop's finalize takes registered identity rows without a map load
            identity=route._identity_rows(mkm_kernel.M128_MAX_TOKENS * mkm_kernel.TOP_K),
            topk=True,
            m32={mlp.layer_id: maps},
            m64=True,
        )
        return cls(bound, wide), ""

    def takes(self, rows: int, rows_per_request: int) -> bool:
        """Whether a draft pass of ``rows`` rows (a draft step, or the extend's rows_per_request rows a request) runs
        here."""
        if rows_per_request == 1 and rows < MIN_ROWS:
            return False
        return 1 <= rows <= MAX_ROWS

    def state(self, rows: int):
        """moe.step's step state for ``rows`` rows: the MoE slots of that width, this bound."""
        from sgl_nvidia_mega.qwen3_5 import plan

        route = rows <= plan.ROUTE_MAX_ROWS
        on = dict.fromkeys(_SLOT_NAMES, False)
        on.update(
            route=route,
            mkm=route and rows <= (plan.MKM_WIDE_MAX_ROWS if self.wide else plan.MKM_MAX_ROWS),
            route_topk=not route,
            mkm_m32=not route and rows <= plan.MKM_M32_MAX_ROWS,
            mkm_m64=plan.MKM_M32_MAX_ROWS < rows <= plan.MKM_M64_MAX_ROWS,
        )
        if rows not in self._noted and torch.cuda.is_current_stream_capturing():
            self._noted.add(rows)
            logger.info(
                "NVIDIA megakernel: Qwen3.5 MTP draft MoE of %d rows on the target's path (PERF-DRAFT): %s",
                rows,
                [k for k, v in on.items() if v],
            )
        return SimpleNamespace(slots=SimpleNamespace(**on), bound={"moe": self.bound}, scratch={})

    def __call__(self, layer, x: torch.Tensor):
        """The draft layer's MoE on ``x`` [rows, hidden]: SGLang's deferred-finalize handoff (MoeDeferredFinalize)."""
        from sgl_nvidia_mega.qwen3_5 import moe

        return moe.step(layer, x, self.state(int(x.shape[0])), defer=True)
