# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): python/sglang/srt/models/qwen3_5.py
# (Qwen3_5GatedDeltaNet.forward, its prefill branch) and
# python/sglang/srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_extend,
# the eager non-verify branch) as the source runs them
# (_split_conv_ok, _convolve_prefill_split, _fused_track_conv and _fused_track_states).
"""The GDN layer step of a covered prefill / extend: ``layer.linear_attn(x, fb)`` on the
source's prefill kernels, eagerly, with FlashInfer's chunked delta rule as
the core (``--linear-attn-prefill-backend flashinfer``).

In the stock order:

1. the input projection and SGLang's prefill views of it
   (``qwen3_5_gdn_prefill_projection_views``: q/k/v, z, b and a in place);
2. the conv windows of the prefix-cache tracked rows, taken before the conv
   (``gdn_track_conv_snapshot``, one launch);
3. the causal conv writing contiguous q / k / v planes (``causal_conv1d_fn_split_qkv``:
   the token-strip CUDA kernel from ``conv_strip``'s token count, else the Triton 2D
   tile); where the split conv does not apply (one token, other strides or widths) the
   plain conv on ``conv_tile`` and SGLang's q / k / v split;
4. the gates (``fused_gdn_gating``, the 2D launch from 512 rows);
5. FlashInfer's chunked kernel (``gdn_prefill_flashinfer.extend``): q and k
   l2-normalized (one launch when the split conv stored them back to back, 64 rows a
   program from 16384 rows), the state rows gathered and written back in one launch
   each, FlashInfer #8854's TMEM dealloc order once ``build`` installed it;
6. the track copies of the checkpoint and final states (``gdn_track_state_rows``, one
   launch);
7. the gated RMSNorm (32 rows a program from 32768 rows) and the output projection.

Each fused copy that declines a layout (2, 5's row copies, 6) runs SGLang's torch
indexing or its ``_track_mamba_state_extend`` instead. Reads and writes are the stock
step's: the backend's forward metadata (query_start_loc, mamba_cache_indices,
conv_block_table, the track rows and FlashInfer's checkpoint plan), the layer's conv and
SSM states by state slot (updated in place) and the track slots.

Numerics: the stock step's bits (SGLang ``main`` with FlashInfer's prefill). Every
change is a launch geometry, an output layout or a copy, each verified bitwise against
the stock launch by the source's tests (``test_gdn_prefill_layout.py``,
``test_causal_conv1d_split_output.py``, ``test_gdn_prefill_tmem_dealloc.py``): the split
conv, the 2D tile and the strip give the stock conv's values and conv states; the 2D
gating keeps the per-element expressions; the l2norm and norm tiles keep each row's
16-lane x 8-element reduction; the q | k one launch normalizes rows independently;
#8854 adds a fence and a barrier only.

``extend`` returns None (the caller runs SGLang's forward) for anything the kernels do
not serve: not an EXTEND / MIXED step (a target verify, a draft extend, DP-relabeled
decode rows), linear-attention TP > 1, activations other than BF16, a prefill CUDA graph
(breakable or full) or a captured prefill bucket, CP row sharding, DP padding rows,
MIS, a page-major (strided) state pool, and an SGLang prefill kernel other than
FlashInfer's (its checkpoint plan and track rows are FlashInfer's only then).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)

OP = "gdn"
HEAD_DIM = 128  # FlashInfer's GDN prefill and the l2norm / norm tiles

SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "_GDN_FUSED_QKVZBA_RATIOS",
        "_forward_input_proj",
        "linear_attn",
        "layer_id",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "attn_tp_size",
        "attn",
        "norm",
        "out_proj",
    ),
    "srt/models/qwen4_exp.py": ("Qwen4ExpLinearDecoderLayer", "_shards_rows_over_cp"),
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "qwen3_5_gdn_prefill_projection_views",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "RMSNorm",
        "weight",
        "bias",
        "eps",
        "group_size",
        "norm_before_gate",
        "activation",
    ),
    "srt/layers/radix_linear_attention.py": (
        "layer_id",
        "q_dim",
        "k_dim",
        "v_dim",
        "num_q_heads",
        "num_k_heads",
        "num_v_heads",
        "head_q_dim",
        "head_k_dim",
        "head_v_dim",
        "conv_weights",
        "bias",
        "activation",
        "A_log",
        "dt_bias",
    ),
    "srt/layers/attention/linear/gdn_backend.py": (
        "kernel_dispatcher",
        "extend_kernel",
        "mis_metadata",
        "prefill_graph_metadata",
        "_split_prefill_qkv",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "forward_metadata",
        "state_layer_cache",
        "_track_mamba_state_extend(h_track_buf)",
    ),
    "srt/mem_cache/memory_pool.py": ("conv", "temporal"),
    "srt/layers/attention/mamba/mamba2_metadata.py": (
        "query_start_loc",
        "mamba_cache_indices",
        "conv_block_table",
        "has_mamba_track_mask",
        "track_conv_indices",
        "conv_states_mask_indices",
        "track_ssm_h_src",
        "track_ssm_h_dst",
        "track_ssm_final_src",
        "track_ssm_final_dst",
        "track_ssm_recompute_dst",
        "state_checkpoint_cu_starts",
        "num_state_checkpoints",
        "state_checkpoint_every_n_tokens",
    ),
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py": (
        "FlashInferGDNKernel",
        "is_flashinfer_gdn_prefill_available",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "logical_forward_mode",
        "extend_seq_lens_cpu",
        "extend_prefix_lens",
        "global_num_token_non_padded_cpu",
    ),
    "srt/model_executor/forward_mode.py": (
        "is_extend_or_draft_extend_or_mixed",
        "is_extend_without_speculative",
    ),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/model_executor/runner_utils/prefill_graph.py": ("is_in_full_prefill_graph",),
}


@dataclass(frozen=True)
class Options:
    """The source's GDN prefill knobs as values, with its tuned defaults."""

    # SGLANG_OPT_GDN_PREFILL_SPLIT_CONV=1 : the conv
    # writes contiguous q / k / v planes
    split_conv: bool = True
    # SGLANG_OPT_GDN_PREFILL_CONV_STRIP=1 with SGLANG_GDN_PREFILL_CONV_RETILE=1 (A's
    # name): the split conv's token-strip CUDA kernel, (tokens, threads, unroll,
    # min_tokens); None keeps the Triton 2D tile at every length
    conv_strip: Optional[tuple] = (32, 128, 4, 2048)
    # SGLANG_CAUSAL_CONV1D_FWD_CONFIG="8,256,2,2": the plain conv's tile (BLOCK_M,
    # BLOCK_N, num_warps, num_stages); None keeps the stock tile
    conv_tile: Optional[tuple] = (8, 256, 2, 2)
    gating_2d: bool = True  # SGLANG_OPT_GDN_PREFILL_GATING_2D=1
    l2norm_retile: bool = True  # SGLANG_OPT_GDN_PREFILL_L2NORM_RETILE=1
    qk_l2norm_one_launch: bool = True  # SGLANG_OPT_GDN_PREFILL_QK_L2NORM_ONE_LAUNCH=1
    norm_retile: bool = True  # SGLANG_OPT_GDN_PREFILL_NORM_RETILE=1
    fused_state_copy: bool = True  # SGLANG_OPT_GDN_PREFILL_FUSED_STATE_COPY=1
    fused_track: bool = True  # SGLANG_OPT_GDN_PREFILL_FUSED_TRACK=1
    # SGLANG_OPT_GDN_PREFILL_TMEM_DEALLOC=1 (a master pin): FlashInfer #8854 on
    # FlashInfer 0.7.3's prefill kernels, installed by build
    tmem_dealloc: bool = True

    def __post_init__(self):
        # the source's parsers validate the tiles (ValueError)
        self.strip_cfg()
        self.fwd_launch()

    def strip_cfg(self):
        """The strip tile (``_PrefillConvStrip``), or None."""
        return None if self.conv_strip is None else _strip_cfg(self.conv_strip)

    def fwd_launch(self) -> Optional[dict]:
        """The plain conv's launch tile, or None."""
        return None if self.conv_tile is None else _fwd_launch(self.conv_tile)


@lru_cache(maxsize=None)
def _strip_cfg(tile: tuple):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_conv import (
        _parse_prefill_conv_strip,
    )

    cfg = _parse_prefill_conv_strip(",".join(str(v) for v in tile))
    if cfg is None:
        raise ValueError(f"conv_strip {tile!r}: use None to turn the strip off")
    return cfg


@lru_cache(maxsize=None)
def _fwd_launch(tile: tuple) -> dict:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_conv import (
        _parse_fwd_launch_config,
    )

    launch = _parse_fwd_launch_config(",".join(str(v) for v in tile))
    if launch is None:
        raise ValueError(f"conv_tile {tile!r}: use None for the stock tile")
    return launch


DEFAULT = Options()

# Set by build once FlashInfer #8854 is installed (process-wide, FlashInfer's own
# launcher modules rebound)
_tmem_fixed = False


def layer_reasons(gdn) -> str:
    """'' when the prefill kernels take this ``Qwen3_5GatedDeltaNet``, else why not."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
    from sglang.srt.models.qwen3_5 import _GDN_FUSED_QKVZBA_RATIOS

    if gdn.attn_tp_size != 1:
        return (
            f"linear-attention TP{gdn.attn_tp_size}: the kernels are qualified at TP1"
        )
    if (gdn.head_k_dim, gdn.head_v_dim) != (HEAD_DIM, HEAD_DIM):
        return (
            f"GDN head dims {(gdn.head_k_dim, gdn.head_v_dim)}, the kernels take "
            f"{HEAD_DIM}"
        )
    if gdn.num_v_heads // gdn.num_k_heads not in _GDN_FUSED_QKVZBA_RATIOS:
        return (
            f"{gdn.num_v_heads} value heads to {gdn.num_k_heads} key heads: SGLang's "
            "prefill projection views need the fused q/k/v/z/b/a layout"
        )
    norm = gdn.norm
    if not isinstance(norm, RMSNorm) or norm.bias is not None:
        return "the GDN output norm is not SGLang's gated RMSNorm without bias"
    return ""


def build(model, options: Options = DEFAULT) -> str:
    """'' when the prefill kernels serve every GDN layer of ``model`` (target or MTP
    draft) and are built, or why the GDN prefill stays stock. Compiles the strip conv
    and installs FlashInfer #8854 (process-wide; FlashInfer 0.7.3's files only: on any
    other FlashInfer its own kernels run, with a warning, as in the source)."""
    global _tmem_fixed
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the GDN prefill kernels need an SM10x GPU"
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    m = model.model
    layers = [
        m.layers[i]
        for i in range(m.start_layer, m.end_layer)
        if isinstance(m.layers[i], Qwen4ExpLinearDecoderLayer)
    ]
    if not layers:
        return "no GDN layer"
    for layer in layers:
        reason = layer_reasons(layer.linear_attn)
        if reason:
            return f"layer {layer.layer_id}: {reason}"
    if not is_flashinfer_gdn_prefill_available():
        return "FlashInfer's GDN prefill kernels are unavailable"
    strip = options.strip_cfg()
    if options.split_conv and strip is not None:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_conv import (
            warmup_prefill_conv_strip,
        )

        try:
            warmup_prefill_conv_strip(strip)
        except Exception as e:  # a failed build keeps the stock GDN prefill
            return f"the prefill conv strip does not build: {type(e).__name__}: {e}"
    if options.tmem_dealloc and not _tmem_fixed:
        try:
            from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_tmem_dealloc import (
                install,
            )

            _tmem_fixed = install()
        except Exception as e:  # the source warns and keeps FlashInfer's kernels
            logger.warning(
                "GDN prefill TMEM dealloc fix (FlashInfer #8854) not applied: %s", e
            )
    return ""


@dataclass
class _Step:
    """One layer's host decisions and state handles, made before any launch."""

    backend: object
    kernel: object  # SGLang's FlashInferGDNKernel (its prefill function and flags)
    conv_states: torch.Tensor
    ssm_states: torch.Tensor


def prepare(gdn, x, fb) -> tuple:
    """(reason, None) when the step stays stock, else ('', the _Step). Host state
    only; nothing is written."""
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.prefill_graph import (
        is_in_full_prefill_graph,
    )
    from sglang.srt.models.qwen4_exp import _shards_rows_over_cp

    from sgl_nvidia_mega.qwen3_8.runtime.gdn import linear_backend

    if not (
        fb.forward_mode.is_extend_or_draft_extend_or_mixed()
        and fb.logical_forward_mode.is_extend_without_speculative()
    ):
        return "not an extend / mixed step", None
    if gdn.attn_tp_size != 1:
        return f"linear-attention TP{gdn.attn_tp_size}", None
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return "activations are not a BF16 tensor", None
    if is_in_breakable_cuda_graph() or is_in_full_prefill_graph():
        return "a prefill CUDA graph", None
    if _shards_rows_over_cp(fb):
        return "CP row sharding", None
    real = fb.global_num_token_non_padded_cpu
    if real is not None and real < x.shape[0]:
        return "DP padding rows", None
    backend = linear_backend()
    kernel = backend.kernel_dispatcher.extend_kernel
    if not isinstance(kernel, FlashInferGDNKernel):
        return f"SGLang's GDN prefill kernel is {type(kernel).__name__}", None
    if backend.mis_metadata is not None:
        return "MIS", None
    if backend.prefill_graph_metadata is not None:
        return "a captured prefill bucket", None
    cache = backend.state_layer_cache(gdn.attn.layer_id)
    conv_states, ssm_states = cache.conv[0], cache.temporal
    if not (conv_states.is_contiguous() and ssm_states.is_contiguous()):
        return "a page-major (strided) state pool", None
    return "", _Step(backend, kernel, conv_states, ssm_states)


def extend(executor, layer, x, st):
    """What ``layer.linear_attn(x, st.fb)`` returns for a prefill / extend step, or
    None when the kernels do not serve it (the caller then runs stock).
    ``executor.gdn_prefill``: the ``Options`` its build succeeded with, or None."""
    options = executor.gdn_prefill
    if options is None:
        return None
    gdn = layer.linear_attn
    reason, step = prepare(gdn, x, st.fb)
    if reason:
        return None
    out, impl = _forward(options, step, gdn, x, st.fb)
    st.ran(OP, impl)
    return out


def _forward(options, step, gdn, x, fb):
    """``Qwen3_5GatedDeltaNet.forward``'s prefill branch; returns (output, impl)."""
    import triton
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        qwen3_5_gdn_prefill_projection_views,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_prefill_norm import (
        _layer_norm_fwd,
    )

    qkvz, ba = gdn._forward_input_proj(x)
    mixed_qkv, z, b, a = qwen3_5_gdn_prefill_projection_views(
        qkvz,
        ba,
        triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size),
        triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size),
        gdn.head_k_dim,
        gdn.head_v_dim,
    )
    core, impl = _extend_core(options, step, gdn.attn, fb, mixed_qkv, a, b)
    # the prefill views' z stays [T, HV, V] (strided); DP padding was declined, so the
    # core has z's rows
    z_shape = z.shape
    core = core.reshape(-1, core.shape[-1])
    norm = gdn.norm
    # norm(core, z): RMSNorm.forward_triton -> rms_norm_gated -> _layer_norm_fwd
    core, _, _ = _layer_norm_fwd(
        core,
        norm.weight.contiguous(),
        norm.bias,
        norm.eps,
        z=z,
        group_size=norm.group_size,
        norm_before_gate=norm.norm_before_gate,
        is_rms_norm=True,
        activation=norm.activation,
        retile=options.norm_retile,
    )
    core = core.reshape(z_shape)
    core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
    out, _ = gdn.out_proj(core)
    return out, impl


def _extend_core(options, step, layer, fb, mixed_qkv, a, b):
    """``GDNAttnBackend.forward_extend``'s eager non-verify branch on a contiguous pool
    with FlashInfer's prefill kernel; returns ([1, T, HV, V], impl)."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_prefill_flashinfer import (
        extend as flashinfer_extend,
    )
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_prefill_gating import (
        fused_gdn_gating,
    )
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_conv import (
        causal_conv1d_fn,
        causal_conv1d_fn_split_qkv,
    )
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_state_glue import (
        gdn_state_rows_supported,
    )

    backend = step.backend
    md = backend.forward_metadata
    conv_states, ssm_states = step.conv_states, step.ssm_states
    cache_indices = md.mamba_cache_indices
    track = md.has_mamba_track_mask
    track_fused = True
    if track and not (
        options.fused_track and _fused_track_conv(md, mixed_qkv, conv_states)
    ):
        track_fused = False
        mixed_qkv_to_track = mixed_qkv[md.track_conv_indices].transpose(1, 2)
        conv_states[md.conv_states_mask_indices] = mixed_qkv_to_track
    block_table = md.conv_block_table
    conv_kwargs = {"block_table": block_table} if block_table is not None else {}
    if options.split_conv and _split_conv_ok(layer, mixed_qkv):
        conv = "conv_split"
        q, k, v = causal_conv1d_fn_split_qkv(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            conv_states=conv_states,
            query_start_loc=md.query_start_loc,
            seq_lens_cpu=fb.extend_seq_lens_cpu,
            q_dim=layer.q_dim,
            k_dim=layer.k_dim,
            strip_cfg=options.strip_cfg(),
            activation=layer.activation,
            has_initial_state=fb.extend_prefix_lens > 0,
            cache_indices=cache_indices,
            **conv_kwargs,
        )
        query, key, value = _split_conv_views(layer, q, k, v)
    else:
        conv = "conv"
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=fb.extend_prefix_lens > 0,
            cache_indices=cache_indices,
            query_start_loc=md.query_start_loc,
            seq_lens_cpu=fb.extend_seq_lens_cpu,
            fwd_launch=options.fwd_launch(),
            **conv_kwargs,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]
        query, key, value = backend._split_prefill_qkv(layer, mixed_qkv)
    g, beta = fused_gdn_gating(
        layer.A_log, a, b, layer.dt_bias, gating_2d=options.gating_2d
    )
    fused_rows = options.fused_state_copy and gdn_state_rows_supported(
        ssm_states, cache_indices
    )
    core_attn_out, _, h = flashinfer_extend(
        step.kernel,
        query,
        key,
        value,
        g,
        beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=md.query_start_loc,
        state_checkpoint_cu_starts=md.state_checkpoint_cu_starts,
        num_state_checkpoints=md.num_state_checkpoints,
        state_checkpoint_every_n_tokens=md.state_checkpoint_every_n_tokens,
        fused_rows=fused_rows,
        l2norm_retile=options.l2norm_retile,
        qk_one_launch=options.qk_l2norm_one_launch,
    )
    if track and not (options.fused_track and _fused_track_states(md, h, ssm_states)):
        track_fused = False
        # FlashInfer has no fp32 track snapshot (supports_track_state_snapshot)
        backend._track_mamba_state_extend(fb, h, ssm_states, md, h_track_buf=None)
    impl = [conv, "fi_tmem" if _tmem_fixed else "fi"]
    impl.append("rows_fused" if fused_rows else "rows")
    if track:
        impl.append("track_fused" if track_fused else "track")
    return core_attn_out, "+".join(impl)


def _split_conv_ok(layer, mixed_qkv: torch.Tensor) -> bool:
    """Whether the extend conv can emit split contiguous q / k / v: the Triton
    launch on a channel-last view (what the stock CUDA dispatch also takes), width
    4, q and q + k on 256-channel boundaries."""
    return (
        mixed_qkv.dim() == 2
        and mixed_qkv.shape[0] > 1
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.stride(0) != 1
        and mixed_qkv.shape[1] == layer.q_dim + layer.k_dim + layer.v_dim
        and layer.conv_weights.shape[-1] == 4
        and layer.q_dim % 256 == 0
        and (layer.q_dim + layer.k_dim) % 256 == 0
        and layer.activation in (None, "silu", "swish")
    )


def _split_conv_views(layer, q, k, v):
    seq_len = q.shape[0]
    return (
        q.view(1, seq_len, layer.num_q_heads, layer.head_q_dim),
        k.view(1, seq_len, layer.num_k_heads, layer.head_k_dim),
        v.view(1, seq_len, layer.num_v_heads, layer.head_v_dim),
    )


def _fused_track_conv(meta, mixed_qkv: torch.Tensor, conv_states: torch.Tensor) -> bool:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_state_glue import (
        gdn_track_conv_snapshot,
    )

    return gdn_track_conv_snapshot(
        conv_pool=conv_states,
        x=mixed_qkv,
        token_rows=meta.track_conv_indices,
        dst_slots=meta.conv_states_mask_indices,
    )


def _fused_track_states(
    meta, h: Optional[torch.Tensor], ssm_states: torch.Tensor
) -> bool:
    """The track copies of _track_mamba_state_extend in one launch; False (nothing
    written) when a row set it does not cover is present."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_state_glue import (
        gdn_track_state_rows,
    )

    if (
        meta.track_ssm_h_src is None
        or meta.track_ssm_h_dst is None
        or meta.track_ssm_final_src is None
        or meta.track_ssm_final_dst is None
        or (
            meta.track_ssm_recompute_dst is not None
            and meta.track_ssm_recompute_dst.numel() > 0
        )
    ):
        return False
    return gdn_track_state_rows(
        pool=ssm_states,
        h=h.squeeze(0) if h is not None else None,
        h_src=meta.track_ssm_h_src,
        h_dst=meta.track_ssm_h_dst,
        f_src=meta.track_ssm_final_src,
        f_dst=meta.track_ssm_final_dst,
    )
