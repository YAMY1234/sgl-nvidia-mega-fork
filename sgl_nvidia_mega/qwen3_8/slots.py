# SPDX-License-Identifier: Apache-2.0
"""Startup selection of rank-local numeric slots; omitted slots keep stock."""

import logging
import os
from functools import cache

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": ("NumericOperation",),
}

SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
OPERATION_SLOTS = {
    NumericOperation.MTP_DEFERRED_KV_PREPARE: "mtp_deferred_kv",
    NumericOperation.MTP_CAPTURE_SEED: "mtp_capture_seed",
    NumericOperation.HC_GATE: "hc_gate",
    NumericOperation.HC_APPLY_NORM: "hc_apply_norm",
    NumericOperation.HC_APPLY_NORM_GATE: "hc_apply_norm_gate",
    NumericOperation.HC_MIX: "hc_mix",
    NumericOperation.FINALIZE_SHARED: "finalize_shared",
    NumericOperation.LOCAL_EXPERTS: "mkm",
    NumericOperation.GDN_CORE: "gdn_mk",
    NumericOperation.GDN_FRONT: "gdn_front",
    NumericOperation.GDN_VERIFY_GLUE: "gdn_verify_glue",
    NumericOperation.GDN_NATIVE_PDL: "gdn_native_pdl",
    NumericOperation.DIRECT_POOL_ATTENTION: "qsa_core",
    NumericOperation.SPARSE_SCORE: "qsa_score",
    NumericOperation.SPARSE_SELECT: "qsa_select",
    NumericOperation.FINALIZE_PERMUTED_SHARED: "finalize_permuted_shared",
    NumericOperation.ROUTER_QUANT: "router_quant",
}
KNOWN_SLOTS = frozenset(OPERATION_SLOTS.values())
# Like the other architecture families, slots are fixed when this module loads.
_raw = os.environ.get(SLOTS_ENV)
REQUESTED_SLOTS = (
    None if _raw is None else frozenset(s.strip() for s in _raw.split(",") if s.strip())
)
SLOTS = KNOWN_SLOTS if REQUESTED_SLOTS is None else KNOWN_SLOTS & REQUESTED_SLOTS


def enabled(operation):
    return OPERATION_SLOTS.get(operation) in SLOTS


@cache
def log_selection():
    if REQUESTED_SLOTS is not None:
        logging.getLogger(__name__).info(
            "Qwen3.8 numeric slots: enabled=%s; disabled=%s; disabled offers use stock",
            ",".join(sorted(SLOTS)) or "none",
            ",".join(sorted(KNOWN_SLOTS - SLOTS)) or "none",
        )
