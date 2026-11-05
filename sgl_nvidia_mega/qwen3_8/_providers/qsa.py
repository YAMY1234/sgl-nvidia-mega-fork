# SPDX-License-Identifier: Apache-2.0
"""Direct TP4 sparse attention on framework-owned pools and scratch."""


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
