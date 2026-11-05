# SPDX-License-Identifier: Apache-2.0
"""Build/launch-only fixture for the owning lane's GPU smoke, not a numerical gate.

Use with canonical binding and the plugin commit in the delivery receipt.
The node/container entry, AOT assets, GPU selection and mount guard are supplied
by the job owner. This file does not submit, clean up, or start a model service.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path


def router_logits(rows, experts, device):
    """BF16 raw ABI, with a strict ordering among the selected first ten."""
    import torch

    return (-torch.arange(experts, dtype=torch.bfloat16, device=device) / 16).repeat(
        rows, 1
    )


def run(out):
    import torch
    from flashinfer.tllm_enums import ActivationType
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        ExpertSignature,
        FinalizeSharedIO,
        LocalExpertsIO,
    )
    from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
    from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        numeric_registry,
    )

    from sgl_nvidia_mega.qwen3_8.providers import register_signatures

    assert torch.cuda.get_device_capability() == (10, 7)
    device = torch.device("cuda", torch.cuda.current_device())
    h, i, e, k = 2560, 192, 512, 10
    # Uniform E4M3 1.0 bytes are valid in every 128x4 SF swizzle position.
    common = dict(
        packed_w13=torch.full(
            (e, 2 * i, h // 2), 0x11, dtype=torch.uint8, device=device
        ),
        packed_w2=torch.full((e, h, i // 2), 0x11, dtype=torch.uint8, device=device),
        scale13=torch.full(
            (e, 3 * 40 * 512), 1.0, dtype=torch.float8_e4m3fn, device=device
        ),
        scale2=torch.full(
            (e, 20 * 3 * 512), 1.0, dtype=torch.float8_e4m3fn, device=device
        ),
        global_scale1=torch.full((e,), 1 / h, dtype=torch.float32, device=device),
        global_scale1_c=torch.full((e,), 1 / h, dtype=torch.float32, device=device),
        global_scale2=torch.full((e,), 1 / i, dtype=torch.float32, device=device),
    )
    cases = [(1, 0), (4, 0), (1, 1), (4, 1), (12, 2), (16, 2)]
    signatures = tuple(
        ExpertSignature(O.LOCAL_EXPERTS, m, h, i, e, k, D.PACKED_FP4, v)
        for m, v in cases
    )
    final_signature = ExpertSignature(O.FINALIZE_SHARED, 4, h, i, e, k, D.BF16)
    register_signatures((*signatures, final_signature))
    records = []

    def launch(signature, io):
        offer = numeric_registry.get(signature)
        assert offer is not None and offer.signature == signature
        offer.run(io)  # eager AOT/maps/JIT preparation
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert offer.run(io) is None
        graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        records.append(
            dict(
                operation=int(signature.operation),
                rows=signature.rows,
                variant=signature.variant,
                registry=True,
                eager=True,
                capture_replay=True,
                numerical_qualification=False,
                **(
                    dict(
                        routing_logits_dtype=str(io.routing_logits.dtype),
                        routing_logits_shape=list(io.routing_logits.shape),
                        routing_logits_data_ptr=io.routing_logits.data_ptr(),
                        raw_router_abi="uint16_t BF16",
                    )
                    if isinstance(io, LocalExpertsIO)
                    else {}
                ),
            )
        )

    for signature in signatures:
        m = signature.rows
        logits = router_logits(m, e, device)
        assert logits.dtype == torch.bfloat16 and logits.shape == (m, e)
        io = LocalExpertsIO(
            **common,
            packed_x=torch.full((m, h // 2), 0x11, dtype=torch.uint8, device=device),
            x_scale=torch.full(
                (m, h // 16), 1.0, dtype=torch.float8_e4m3fn, device=device
            ),
            routing_logits=logits,
            experts=e,
            intermediate_size=i,
            hidden_size=h,
            topk=k,
            activation=int(ActivationType.Swiglu),
            local_expert_start=0,
            local_expert_count=e,
            routed_scale=1.0,
            route_weights_out=torch.empty((m, k), dtype=torch.bfloat16, device=device),
            route_ids_out=torch.empty((m, k), dtype=torch.int32, device=device),
            expert_rows_out=torch.empty((m, k, h), dtype=torch.bfloat16, device=device),
            final_out=torch.empty((m, h), dtype=torch.bfloat16, device=device),
            scratch_out=torch.empty((0,), dtype=torch.uint8, device=device),
        )
        launch(signature, io)
        # Exercise the unfinalized output path, too; ownership remains with caller.
        launch(signature, replace(io, final_out=None))
    final_io = FinalizeSharedIO(
        hidden=torch.ones((4, h), dtype=torch.bfloat16, device=device),
        gate_weight=torch.ones((h,), dtype=torch.bfloat16, device=device),
        shared=torch.ones((4, h), dtype=torch.bfloat16, device=device),
        expert_rows=torch.ones((4, k, h), dtype=torch.bfloat16, device=device),
        route_weights=torch.full((4, k), 1 / k, dtype=torch.bfloat16, device=device),
        cols=256,
        use_pdl=True,
        output=torch.empty((4, h), dtype=torch.bfloat16, device=device),
    )
    launch(final_signature, final_io)
    out.write_text(
        json.dumps(
            dict(
                status="BUILD_LAUNCH_ONLY_PASS",
                records=records,
                numerical_qualification=False,
                model_qualification=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
