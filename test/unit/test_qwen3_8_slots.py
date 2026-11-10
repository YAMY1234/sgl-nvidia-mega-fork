# SPDX-License-Identifier: Apache-2.0
"""Startup slot filtering with real DTOs, SPEC and registry in fresh processes.

HC and F GPU compiler entry points are CPU spies. No GPU qualification.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SLOT_NAMES = (
    "qsa_pool_union",
    "qsa_prefill_select",
    "qsa_prefill_attention",
    "hc_gate",
    "hc_apply_norm",
    "hc_apply_norm_gate",
    "hc_mix",
    "finalize_shared",
    "mkm",
    "gdn_mk",
    "qsa_core",
    "qsa_score",
    "qsa_select",
    "gdn_front",
    "gdn_verify_glue",
    "finalize_permuted_shared",
    "router_quant",
    "gdn_native_pdl",
    "mtp_capture_seed",
    "mtp_deferred_kv",
)

CHILD = r"""
import json
import logging
from types import SimpleNamespace
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    HcSignature, ExpertSignature, GdnSignature, NumericSignature,
    FinalizePermutedSignature, RouterQuantSignature, NativeGdnPdlSignature, MtpCaptureSeedSignature, MtpDeferredKvSignature,
    NumericRegistry, NumericOperation as O, ScalarType as D,
)
from sgl_nvidia_mega.qwen3_8 import providers, slots
from sgl_nvidia_mega.qwen3_8.spec import SPEC
from sgl_nvidia_mega.qwen3_8._providers import hc, finalize_permuted, router_quant

compile_calls = []
def cpu_build(signature):
    compile_calls.append(signature.operation)
    return lambda io: None
hc.build = cpu_build
hc.build_mix = cpu_build
finalize_permuted.build = lambda signature: (lambda io: None)
router_quant.build = cpu_build
providers.numeric_registry = registry = NumericRegistry()
signatures = (
    NumericSignature(O.POOL_UNION_ATTENTION, 64, 24, 2, 256, 512, 4, D.BF16),
    NumericSignature(O.PREFILL_SELECT, 0, 4, 1, 128, 512, 4, D.BF16, context_bound=8192),
    NumericSignature(O.PACKED_PREFILL_ATTENTION, 0, 24, 2, 256, 2051, 4, D.BF16, variant=3),
    HcSignature(O.HC_GATE, 4, 4, 2560, D.BF16, parts=2),
    HcSignature(O.HC_APPLY_NORM, 4, 4, 2560, D.BF16, parts=2),
    HcSignature(O.HC_APPLY_NORM_GATE, 4, 4, 2560, D.BF16, parts=2),
    HcSignature(O.HC_MIX, 4, 4, 2560, D.BF16, parts=2, lowrank=128, variant=1),
    ExpertSignature(O.FINALIZE_SHARED, 4, 2560, 192, 512, 10, D.BF16),
    ExpertSignature(O.LOCAL_EXPERTS, 4, 2560, 192, 512, 10, D.PACKED_FP4),
    GdnSignature(O.GDN_CORE, 4, 4, 12, 128, 128, 4, D.BF16, gate_scalar_type=D.BF16),
    NumericSignature(O.DIRECT_POOL_ATTENTION, 4, 6, 1, 256, 512, 4, D.BF16),
    NumericSignature(O.SPARSE_SCORE, 4, 8, 1, 128, 512, 4, D.BF16, context_bound=8192),
    NumericSignature(O.SPARSE_SELECT, 4, 8, 1, 128, 512, 4, D.FP32, context_bound=8192),
    GdnSignature(O.GDN_FRONT, 4, 4, 12, 128, 128, 4, D.BF16),
    GdnSignature(O.GDN_VERIFY_GLUE, 8, 16, 48, 128, 128, 4, D.BF16, gate_scalar_type=D.BF16),
    FinalizePermutedSignature(O.FINALIZE_PERMUTED_SHARED, 4, 2560, 10, D.BF16, D.FP32),
    RouterQuantSignature(O.ROUTER_QUANT, 4, 2560, 512, D.BF16),
    NativeGdnPdlSignature(O.GDN_NATIVE_PDL, 32, 16, 48, 128, 128, 4, D.BF16),
    MtpCaptureSeedSignature(O.MTP_CAPTURE_SEED,4,2051,2051,8,2055,2055,0,7,D.INT32,D.INT32,D.INT64,D.INT32,1),
    MtpDeferredKvSignature(O.MTP_DEFERRED_KV_PREPARE,1,1,256,D.BF16,2055,2112,4,8,32,8,4096,4096,512,512,0),
)
expected_names = (
    'qsa_pool_union', 'qsa_prefill_select', 'qsa_prefill_attention',
    'hc_gate', 'hc_apply_norm', 'hc_apply_norm_gate', 'hc_mix',
    'finalize_shared', 'mkm', 'gdn_mk', 'qsa_core',
    'qsa_score', 'qsa_select', 'gdn_front', 'gdn_verify_glue',
    'finalize_permuted_shared', 'router_quant',
    'gdn_native_pdl', 'mtp_capture_seed', 'mtp_deferred_kv',
)
assert tuple(slots.OPERATION_SLOTS[s.operation] for s in signatures) == expected_names
assert slots.KNOWN_SLOTS == frozenset(expected_names)
messages = []
class Handler(logging.Handler):
    def emit(self, record):
        messages.append(record.getMessage())
logger = logging.getLogger(slots.__name__)
logger.setLevel(logging.INFO)
logger.addHandler(Handler())
class NoFrameworkAccess:
    def __getattribute__(self, name):
        raise AssertionError('numeric build must not inspect framework objects: ' + name)
ctx = SimpleNamespace(numeric_signatures=signatures)
built = SPEC.build(NoFrameworkAccess(), NoFrameworkAccess(), ctx)
assert built.offers == SPEC.build(NoFrameworkAccess(), NoFrameworkAccess(), ctx).offers
assert built.plan(None, None) is None
actual = []
for name, signature in zip(expected_names, signatures):
    offer = providers.make_offer(signature)
    if name in slots.SLOTS:
        assert providers.supported(signature)
        assert offer is not None and registry.get(signature) is offer
        assert offer in built.offers
        actual.append(name)
    else:
        assert not providers.supported(signature)
        assert offer is None and registry.get(signature) is None
assert len(built.offers) == len(actual)
assert compile_calls == [s.operation for s in signatures if s.operation in (O.HC_GATE, O.HC_APPLY_NORM, O.HC_APPLY_NORM_GATE, O.HC_MIX, O.ROUTER_QUANT) and slots.enabled(s.operation)]
assert len(messages) == (0 if slots.REQUESTED_SLOTS is None else 1), messages
if messages:
    assert 'disabled offers use stock' in messages[0]
print(json.dumps({'enabled': actual, 'offers': len(built.offers), 'log': messages}))
"""


def run_case(value):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if value is None:
        env.pop("SGLANG_NVIDIA_MEGAKERNEL_SLOTS", None)
    else:
        env["SGLANG_NVIDIA_MEGAKERNEL_SLOTS"] = value
    # Propagate pytest's source roots into the new interpreter without depending
    # on a package installation or the parent's import/module cache.
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    result = subprocess.run(
        [sys.executable, "-c", CHILD],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (value, result.stdout, result.stderr)
    actual = json.loads(result.stdout.splitlines()[-1])
    requested = (
        set(SLOT_NAMES)
        if value is None
        else {name.strip() for name in value.split(",") if name.strip()}
    )
    assert actual["enabled"] == [name for name in SLOT_NAMES if name in requested]
    return actual


def test_fresh_process_slot_matrix():
    full = ",".join(SLOT_NAMES)
    cases = [
        None,
        "",
        full,
        "gdn_pdl",
        "gdn_mk,gdn_pdl,qsa_core",
        " , mkm, mkm, qsa_core, ",
    ]
    cases += [
        ",".join(name for name in SLOT_NAMES if name != omitted)
        for omitted in SLOT_NAMES
    ]
    cases += list(SLOT_NAMES)
    cases.append(
        "hc_shared_apply,hc_shared_apply_norm"
    )  # Removed S slot names enable no capabilities.
    cases.append("qk_front,ple_hash,ple_gather")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_case, cases))
    assert len(results) == 48
    assert results[0]["offers"] == 20 and results[1]["offers"] == 0
    assert [r["offers"] for r in results[6:26]] == [19] * 20
    assert [r["offers"] for r in results[26:-2]] == [1] * 20
    assert results[-1]["offers"] == results[-2]["offers"] == 0
