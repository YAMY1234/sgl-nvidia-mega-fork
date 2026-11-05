# SPDX-License-Identifier: Apache-2.0
"""Rank-local state and expert boundary launches, with explicit outputs."""


def finalize_shared(io):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe.q38_finalize_shared import (
        finalize_shared as launch,
    )

    launch(
        io.hidden,
        io.gate_weight,
        io.shared,
        io.expert_rows,
        io.route_weights,
        io.output,
        cols=io.cols,
        use_pdl=io.use_pdl,
    )


def gdn_core(io):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.q38_mkgdn import core

    if str(io.a.dtype) != "torch.bfloat16" or str(io.b.dtype) != "torch.bfloat16":
        raise ValueError("the GDN core a/b ABI is BF16, independent of dt_bias dtype")
    if io.a.stride(0) != io.a_row_stride or io.b.stride(0) != io.b_row_stride:
        raise ValueError("GDN a/b row strides differ from the prepared view")
    ring = io.ring
    core(
        io.qkv,
        io.a,
        io.b,
        io.a_log,
        io.dt_bias,
        io.checkpoint,
        ring.d,
        ring.k,
        ring.g,
        ring.d_low,
        ring.k_low,
        ring.state_indices,
        ring.replay_indices,
        ring.write_pos,
        ring.cache_base,
        ring.is_flush,
        io.raw_out,
        z=io.z,
        norm_weight=io.norm_weight,
        normalized=io.normalized_out,
        eps=io.epsilon,
        vrows=io.value_rows,
    )


class LocalExperts:
    """Descriptors and AOT handles are prepared by eager warmup, never capture."""

    def __init__(self, signature):
        self.signature = signature
        self._resources = {}

    def prepare(self, io):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "local expert descriptors must be prepared before capture"
            )
        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_mk_q35 as narrow
        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_mkm_wide as wide

        module = narrow if self.signature.rows <= narrow.MAX_M else wide
        bundle = (
            narrow._bundle(io.packed_x.device.index)
            if module is narrow
            else wide.bundle(io.packed_x.device.index, self.signature.rows == 8)
        )
        maps = module.maps(io.packed_w13, io.packed_w2, io.packed_x)
        key = tuple(t.data_ptr() for t in (io.packed_w13, io.packed_w2, io.packed_x))
        # Retain the tensor owners along with descriptors; addresses cannot be recycled.
        self._resources[key] = (
            module,
            bundle,
            maps,
            (io.packed_w13, io.packed_w2, io.packed_x),
        )

    def __call__(self, io):
        from flashinfer.tllm_enums import ActivationType

        from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import q38_mk_q35 as narrow

        s = self.signature
        if (
            (io.hidden_size, io.intermediate_size, io.experts, io.topk)
            != (s.hidden_size, s.intermediate_size, s.experts, s.topk)
            or io.local_expert_start != 0
            or io.local_expert_count != s.experts
            or io.routed_scale != 1.0
            or io.activation != int(ActivationType.Swiglu)
            or io.packed_x.shape != (s.rows, s.hidden_size // 2)
        ):
            raise ValueError("local expert view does not match its offered layout")
        key = tuple(t.data_ptr() for t in (io.packed_w13, io.packed_w2, io.packed_x))
        if key not in self._resources:
            # Ordinary eager warmup may prepare resources. Capture never does.
            self.prepare(io)
        module, (kernel, finalize, metadata), maps, owners = self._resources[key]
        layer = dict(
            gemm1_weights_scale=io.scale13,
            gemm2_weights_scale=io.scale2,
            output1_scale_gate_scalar=io.global_scale1,
            output1_scale_scalar=io.global_scale1_c,
            output2_scale_scalar=io.global_scale2,
        )
        if module is narrow:
            narrow.run(
                kernel,
                maps,
                io.routing_logits,
                io.route_weights_out,
                io.route_ids_out,
                io.x_scale,
                layer,
                io.expert_rows_out,
            )
        else:
            kwargs = dict(
                layer,
                hidden_states=io.packed_x,
                hidden_states_scale=io.x_scale,
                routing_logits=io.routing_logits,
            )
            module.run(
                kernel,
                maps,
                kwargs,
                io.expert_rows_out,
                io.route_weights_out,
                io.route_ids_out,
            )
        if io.final_out is not None:
            narrow._finalize(
                finalize,
                metadata,
                io.expert_rows_out,
                io.route_weights_out,
                io.final_out,
            )
