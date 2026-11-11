# SPDX-License-Identifier: Apache-2.0
"""Direct sparse attention cores on framework-owned pools and scratch."""

# DIRECT_POOL_ATTENTION variant bit 4: the TP1 TCGEN05 core with the output gate in
# its merge (the source's Q38_ATTN_CORE_V1); clear of the pool variants' bits 0-3.
TCGEN05_CORE = 16


def supported_tcgen05_core(s):
    return (
        s.variant == TCGEN05_CORE
        and (
            s.rows,
            s.query_heads,
            s.kv_heads,
            s.head_dim,
            s.selected_count,
            s.compress_ratio,
            s.rotary_dim,
            s.context_bound,
        )
        == (4, 24, 2, 256, 512, 4, 0, 0)
    )


def direct_pool(io):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import q38_attn_core_tp4 as core

    q, gate = io.q, io.gate
    m, qh, d = q.shape
    kh = io.kv.k.shape[1]
    if (m, qh, kh, d, io.query_width) != (4, 6, 1, 256, 4) or gate is None:
        raise ValueError("unsupported direct sparse attention layout")
    selection = io.selection
    if selection.selected_indices.shape != (m, 512):
        raise ValueError("selected indices must be prepared as [4,512]")
    # Same partition and reduction order as run_gated; only allocation is removed.
    nt = 33
    partial_size = m * kh * nt * 16 * d
    stats_size = m * kh * nt * 16
    scratch = io.scratch_out
    if (
        str(scratch.dtype) != "torch.float32"
        or not scratch.is_contiguous()
        or scratch.numel() < partial_size + 2 * stats_size
    ):
        raise ValueError("direct attention requires contiguous FP32 scratch")
    partial = scratch[:partial_size].view(m, kh, nt, 16, d)
    maximum = scratch[partial_size : partial_size + stats_size].view(m, kh, nt, 16)
    sums = scratch[partial_size + stats_size : partial_size + 2 * stats_size].view(
        m, kh, nt, 16
    )
    k, v, nk, nv = io.kv.k, io.kv.v, io.new_k, io.new_v
    pdl = core._PDL_SMALL
    if io.store_new_kv:
        core._store_tp4[(m, kh)](
            k,
            v,
            nk,
            nv,
            io.out_cache_loc,
            k.stride(0),
            v.stride(0),
            nk.stride(0),
            nv.stride(0),
            kh,
            d,
            num_warps=4,
            launch_pdl=pdl,
            PDL_SMALL=pdl,
        )
    core._partial_tp4[(m, nt, kh)](
        q,
        k,
        v,
        nk,
        nv,
        io.out_cache_loc,
        selection.full_slot_map,
        selection.row_requests,
        selection.selected_indices,
        selection.sequence_lengths,
        selection.query_positions,
        partial,
        maximum,
        sums,
        QS=q.stride(0),
        QH=q.stride(1),
        KS=k.stride(0),
        VS=v.stride(0),
        NKS=nk.stride(0),
        NVS=nv.stride(0),
        MS=selection.full_slot_map.stride(0),
        IS=selection.selected_indices.stride(0),
        NT=nt,
        GROUP=qh // kh,
        WIDTH=io.query_width,
        SCALE=float(io.scale),
        KVH=kh,
        num_warps=4,
        launch_pdl=pdl,
        PDL_SMALL=pdl,
    )
    core._merge_gate_tp4[(m, qh, (d + 63) // 64)](
        partial,
        maximum,
        sums,
        io.output,
        gate,
        gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else 256,
        nt,
        64,
        qh,
        kh,
        qh // kh,
        num_warps=4,
        launch_pdl=True,
    )


def direct_core_tp1(io):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import qsa_gate_epilogue_v2 as gated
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import qsa_tcgen05_v1 as core

    q, gate = io.q, io.gate
    m, qh, d = q.shape
    kh = io.kv.k.shape[1]
    if (m, qh, kh, d, io.query_width) != (4, 24, 2, 256, 4) or gate is None:
        raise ValueError("unsupported direct sparse attention layout")
    if q.stride(2) != 1 or q.stride(1) != d:
        raise ValueError("direct attention requires [rows, heads, 256] dense heads")
    selection = io.selection
    if selection.selected_indices.shape != (m, 512):
        raise ValueError("selected indices must be prepared as [4,512]")
    # Same partition and reduction order as run_gated; only allocation is removed.
    nt = 33
    partial_size = m * kh * nt * 16 * d
    stats_size = m * kh * nt * 16
    scratch = io.scratch_out
    if (
        str(scratch.dtype) != "torch.float32"
        or not scratch.is_contiguous()
        or scratch.numel() < partial_size + 2 * stats_size
    ):
        raise ValueError("direct attention requires contiguous FP32 scratch")
    partial = scratch[:partial_size].view(m, kh, nt, 16, d)
    maximum = scratch[partial_size : partial_size + stats_size].view(m, kh, nt, 16)
    sums = scratch[partial_size + stats_size : partial_size + 2 * stats_size].view(
        m, kh, nt, 16
    )
    k, v, nk, nv = io.kv.k, io.kv.v, io.new_k, io.new_v
    if io.store_new_kv:
        core._store[(m, kh)](
            k,
            v,
            nk,
            nv,
            io.out_cache_loc,
            k.stride(0),
            v.stride(0),
            nk.stride(0),
            nv.stride(0),
            kh,
            d,
            num_warps=4,
        )
    core.partial_tcgen05[(m, nt, kh)](
        q,
        k,
        v,
        nk,
        nv,
        io.out_cache_loc,
        selection.full_slot_map,
        selection.row_requests,
        selection.selected_indices,
        selection.sequence_lengths,
        selection.query_positions,
        partial,
        maximum,
        sums,
        QS=q.stride(0),
        QH=q.stride(1),
        KS=k.stride(0),
        VS=v.stride(0),
        NKS=nk.stride(0),
        NVS=nv.stride(0),
        MS=selection.full_slot_map.stride(0),
        IS=selection.selected_indices.stride(0),
        NT=nt,
        GROUP=12,
        WIDTH=io.query_width,
        SCALE=float(io.scale),
        num_warps=4,
    )
    gated._merge_gate_pdl_v2[(m, qh, (d + 63) // 64)](
        partial,
        maximum,
        sums,
        io.output,
        gate,
        gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else 256,
        nt,
        64,
        WAIT_PRIMARY=gated._Q38_QSA_PDL_FIX,
        num_warps=4,
        launch_pdl=True,
    )
