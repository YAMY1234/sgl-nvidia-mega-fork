# SPDX-License-Identifier: Apache-2.0
"""FN GDN slots, on generic ForwardView and native backend-owned state."""
import msgspec

from .config import Shapes, shapes

GDN_SLOTS = frozenset({"gdn_verify", "gdn_pdl", "gdn_mk", "gdn_front"})


class Bound(msgspec.Struct, frozen=True):
    gdn_verify: bool = False
    gdn_pdl: bool = False
    gdn_mk: bool = False
    gdn_front: bool = False
    shape: object = None
    reason: str = ""


def build(model, ctx):
    from sgl_nvidia_mega.qwen3_8.runtime import SLOTS
    requested = GDN_SLOTS & SLOTS
    if not requested:
        return Bound()
    if requested & {"gdn_mk", "gdn_front"}:
        return Bound(reason="gdn_mk/gdn_front numeric qualification pending")
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
    return Bound(gdn_verify=True, gdn_pdl="gdn_pdl" in requested, shape=shape)


def step(layer, x, st):
    if not st.slots.gdn_verify:
        return layer.linear_attn(x, st.fb)
    from . import verify
    return verify.step(layer, x, st)
