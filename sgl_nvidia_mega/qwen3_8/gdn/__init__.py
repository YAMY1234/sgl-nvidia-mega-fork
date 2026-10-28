# SPDX-License-Identifier: Apache-2.0
"""FN GDN slots, on generic ForwardView and native backend-owned state."""
import msgspec

from .config import shapes

SGLANG_READS = {"srt/models/qwen3_5.py": ("_fused_in_proj_weight",)}

GDN_SLOTS = frozenset({"gdn_verify", "gdn_pdl", "gdn_mk"})


class Bound(msgspec.Struct, frozen=True):
    gdn_verify: bool = False
    gdn_pdl: bool = False
    gdn_mk: bool = False
    shape: object = None
    reason: str = ""


def build(model, ctx):
    from sgl_nvidia_mega.qwen3_8.runtime import SLOTS
    requested = GDN_SLOTS & SLOTS
    if not requested:
        return Bound()
    from . import verify
    gdns = [layer.linear_attn for layer in model.model.layers
            if hasattr(layer, "linear_attn")]
    if not gdns:
        return Bound(reason="model has no GDN layers")
    shape = shapes(model.config, ctx.tokens_per_request, gdns[0].attn_tp_size)
    for gdn in gdns:
        reason = verify.unsupported(gdn)
        if reason:
            return Bound(reason=f"GDN layer {gdn.attn.layer_id}: {reason}")
    mk = "gdn_mk" in requested
    if mk and ctx.tokens_per_request != 4:
        return Bound(reason="gdn_mk is qualified for four verify rows/request")
    if mk:
        import torch
        for gdn in gdns:
            weight = gdn._fused_in_proj_weight
            if (weight is None or weight.dtype != torch.bfloat16
                    or tuple(weight.shape) != (shape.packed, shape.hidden)):
                return Bound(reason="gdn_mk needs the finalized native BF16 input projection")
    # The unchanged e5 numeric core includes its PDL wait/entry trigger.
    return Bound(gdn_verify=not mk, gdn_pdl=mk or "gdn_pdl" in requested,
                 gdn_mk=mk, shape=shape)


def step(layer, x, st):
    if not (st.slots.gdn_verify or st.slots.gdn_mk):
        return layer.linear_attn(x, st.fb)
    from . import verify
    return verify.step(layer, x, st)
