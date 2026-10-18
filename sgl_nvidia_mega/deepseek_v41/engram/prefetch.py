# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram prefetch (slot ``engram``): every Engram layer's lookup and WKV GEMM on a
side stream.

An Engram layer's table lookup, MXFP8 quantize and WKV GEMM read only the token
hashes, yet SGLang runs them on that layer's critical path. For a routed step of at
most ``MAX_ROWS`` rows the routed forward issues them for every Engram layer, in layer
order, on one prefetch stream right after the hashes; each layer then only waits for
its rows before ``engram_gate``.

With the per-rank host table each rank holds a slice of the table rows, so the lookup
needs an all-reduce. It runs on a push-only all-reduce plane of its own, whose epochs
never meet the TP group's collectives on the main stream. Each rank also multiplies
only its quarter of the WKV's output columns (views of the weight and of its 128-row
scale tiles) and the plane gathers the quarters as an all-reduce of zero-padded rows.
Both all-reduces sum one owner's value with zeros, and the WKV runs the same CuTe DSL
GEMM, so the rows equal SGLang's inline ones bit for bit; build checks that.
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.deepseek_v41 import plan

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/engram.py": (
        "engram_gate",
        "embed",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "layer_hash_index",
        "_owned_rows",
        "host_table",
        "layout",
        "rows",
        "dim",
    ),
    "srt/models/deepseek_v4.py": (
        "model",
        "layers",
        "start_layer",
        "end_layer",
        "engram",
    ),
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "use_mxfp8",
        "block_fp8_as_mxfp8",
        "block_fp8_mxfp8_ready",
        "mxfp8_dense_backend",
        "weight_scale_inv_swizzled",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "flashinfer_mxfp8_blockscaled_linear",
        "flashinfer_mxfp8_blockscaled_linear(backend)",
        "flashinfer_mxfp8_blockscaled_linear(pin_tactic)",
        '"cute-dsl"',
        "is_flashinfer_cutlass",
        "is_flashinfer_cutedsl",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
        "__init__(max_pull_size)",
        "__init__(max_push_size)",
        "__init__(max_pull_blocks)",
        "override_algo",
        "custom_all_reduce",
        "disabled",
        "group",
        "device",
    ),
    "kernels/ops/communication/all_reduce.py": ("AllReduceAlgo", "ONE_SHOT_PUSH"),
    "srt/distributed/parallel_state.py": (
        "ca_comm",
        "world_size",
        "rank_in_group",
        "cpu_group",
    ),
}

MAX_ROWS = plan.ENGRAM_PREFETCH_MAX_ROWS
# The plane's push slot: the lookup (16 rows x 6144 x 2 B) and the WKV gather
# (16 x 25600 x 2 B = 800 KiB)
PLANE_BYTES = 1 << 20
# The WKV slice of each rank is whole 128-row scale tiles
SCALE_TILE_ROWS = 128


def reason(engram, tp_size: int) -> str:
    """Why this Engram layer cannot prefetch, or "": the per-rank host table (a
    lookup all-reduce the plane can take) and an MXFP8 WKV with swizzled scales."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    table = engram.embed.host_table
    if table is None or table.layout != "per_rank":
        return (
            "needs the per-rank host Engram table (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1"
            ", SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank)"
        )
    wkv = engram.wkv
    method = wkv.quant_method
    if not isinstance(method, Fp8LinearMethod):
        return f"the WKV is on {type(method).__name__}"
    backend = method.mxfp8_dense_backend
    mxfp8 = method.use_mxfp8 or (
        method.block_fp8_as_mxfp8 and getattr(wkv, "block_fp8_mxfp8_ready", False)
    )
    if not (
        mxfp8
        and backend is not None
        and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
    ):
        return "the WKV is not a FlashInfer MXFP8 GEMM"
    if wkv.weight.size(0) % (tp_size * SCALE_TILE_ROWS):
        return f"{wkv.weight.size(0)} WKV columns do not split into scale tiles"
    return ""


def build_plane(group):
    """A push-only one-shot all-reduce plane over the TP group (collective), or None
    without SGLang's custom all-reduce on that group."""
    from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    tp_comm = group.ca_comm
    if not isinstance(tp_comm, CustomAllReduceV2) or tp_comm.disabled:
        return None
    plane = CustomAllReduceV2(
        tp_comm.group,
        tp_comm.device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=PLANE_BYTES,
    )
    if plane.disabled:
        return None
    # one-shot push registers no graph inputs, so it runs on any stream
    plane.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
    return plane


class _Layer:
    """One Engram layer's prefetched rows, their event and its WKV quarter."""

    def __init__(self, engram, rank: int, tp_size: int):
        wkv = engram.wkv
        # detached: CuTe DSL's DLPack import refuses tensors that require grad
        weight = wkv.weight.detach()
        n = weight.size(0)
        cols = n // tp_size
        start = rank * cols
        tiles = wkv.weight_scale_inv_swizzled.reshape(n // SCALE_TILE_ROWS, -1)
        self.engram = engram
        self.weight = weight[start : start + cols]
        self.scale = tiles[
            start // SCALE_TILE_ROWS : (start + cols) // SCALE_TILE_ROWS
        ].reshape(-1)
        self.cols = slice(start, start + cols)
        # persistent, so every captured graph reads one address
        self.out = torch.empty(
            (MAX_ROWS, n), dtype=torch.bfloat16, device=weight.device
        )
        self.ready = torch.cuda.Event()

    def kv(self, hash_ids, plane):
        """SGLang's ``wkv(embed(hash_ids))`` rows, with both all-reduces on ``plane``."""
        from sglang.srt.layers.quantization.fp8_utils import (
            flashinfer_mxfp8_blockscaled_linear,
        )

        emb = plane.custom_all_reduce(self.engram.embed._owned_rows(hash_ids))
        # CUTLASS's MXFP8 GEMM keeps one process-wide workspace; CuTe DSL keeps none
        part = flashinfer_mxfp8_blockscaled_linear(
            input=emb.flatten(-2),
            weight=self.weight,
            weight_scale=self.scale,
            backend="cute-dsl",
            pin_tactic=True,
        )
        rows = part.new_zeros((part.size(0), self.out.size(1)))
        rows[:, self.cols] = part
        return plane.custom_all_reduce(rows)


class EngramPrefetch:
    """The prefetch stream, its all-reduce plane and every Engram layer's rows."""

    def __init__(self, engrams: dict, plane, rank: int, tp_size: int):
        self.plane = plane
        self.stream = torch.cuda.Stream()
        self.layers = {i: _Layer(e, rank, tp_size) for i, e in engrams.items()}

    def issue(self, hash_ids: torch.Tensor) -> None:
        """Every Engram layer's rows for ``hash_ids`` [rows, hash columns], on the
        prefetch stream; ``gate`` of each layer joins them."""
        if not 0 < hash_ids.size(0) <= MAX_ROWS:
            raise ValueError(f"the Engram prefetch takes 1-{MAX_ROWS} rows")
        self.stream.wait_stream(torch.cuda.current_stream())
        hash_ids.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            # temporaries stay on this stream; only the persistent rows cross
            for layer in self.layers.values():
                ids = hash_ids[:, layer.engram.layer_hash_index]
                kv = layer.kv(ids, self.plane)
                layer.out[: kv.size(0)].copy_(kv)
                layer.ready.record(self.stream)

    def gate(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        """``Engram.forward(x, ...)`` of layer ``layer_id`` from its prefetched rows."""
        from sglang.srt.layers.engram import engram_gate

        layer = self.layers[layer_id]
        torch.cuda.current_stream().wait_event(layer.ready)
        e = layer.engram
        kv = layer.out[: x.size(0)]
        return engram_gate(x, kv, e.q_weight, e.k_weight, e.eps, e.clamp_value)


def _any_error(group, error: str) -> str:
    """Every rank's error, joined ("" when none failed): the same on every rank."""
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    return "; ".join(f"rank {r}: {e}" for r, e in enumerate(errors) if e)


def build(model, ctx, group):
    """The Engram prefetch, or None, and the reason it is off. Build issues it at every
    width it takes and compares each layer's rows with SGLang's inline lookup + WKV
    bit for bit (which also compiles its GEMMs before capture)."""
    m = model.model
    engrams = {
        i: m.layers[i].engram
        for i in range(m.start_layer, m.end_layer)
        if m.layers[i].engram is not None
    }
    widths = [w for w in ctx.widths if w <= MAX_ROWS]
    if not engrams or not widths:
        return None, "no Engram layer" if not engrams else "no width it takes"
    reasons = (reason(e, group.world_size) for e in engrams.values())
    error = _any_error(group, next((r for r in reasons if r), ""))
    if error:
        return None, error
    plane = build_plane(group)  # collective
    error = _any_error(group, "" if plane else "no custom all-reduce on the TP group")
    if error:
        return None, error
    # the same hash ids on every rank, spread over every rank's rows of each table
    rows = [None] * group.world_size
    torch.distributed.all_gather_object(
        rows, [e.embed.rows for e in engrams.values()], group=group.cpu_group
    )
    # rank-local setup only: a failure here is gathered before any collective runs
    try:
        prefetch = EngramPrefetch(engrams, plane, group.rank_in_group, group.world_size)
        bound = torch.tensor([min(r) * group.world_size for r in zip(*rows)])
        first = next(iter(engrams.values()))
        cols = first.wkv.weight.size(1) // first.embed.dim
        device = first.wkv.weight.device
        ids = {
            w: (
                torch.arange(w * len(engrams) * cols).view(w, len(engrams), cols)
                * 7919
                % bound[None, :, None]
            ).to(device)
            for w in widths
        }
        error = ""
    except Exception as e:  # gathered: the slot is then off on every rank
        error = f"{type(e).__name__}: {e}"
    error = _any_error(group, error)
    if error:
        return None, error
    # every rank runs the same collectives; a mismatch is recorded, not raised, so
    # no rank leaves them early
    mismatch = ""
    for w in widths:
        prefetch.issue(ids[w])
        for i, layer in prefetch.layers.items():
            e = layer.engram
            torch.cuda.current_stream().wait_event(layer.ready)
            want, _ = e.wkv(e.embed(ids[w][:, e.layer_hash_index]).flatten(-2))
            if not mismatch and not torch.equal(layer.out[:w], want):
                mismatch = f"layer {i} at {w} rows: not SGLang's rows"
    error = _any_error(group, mismatch)
    return (None, error) if error else (prefetch, "")
