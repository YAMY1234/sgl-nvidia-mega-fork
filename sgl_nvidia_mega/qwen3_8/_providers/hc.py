# SPDX-License-Identifier: Apache-2.0
"""Caller-buffer HC launches; stream joins stay in the framework."""

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": ("NumericOperation",),
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": (
        "SplitKTactic",
        "default_tactic",
        "run_splitk_dense_gate",
        "run_splitk_dense_silu",
        "validate_tactic",
    ),
}


def build(signature):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import (
        hc_combine_decode as decode,
    )

    op = signature.operation
    if op == NumericOperation.HC_GATE:
        module = decode._jit_module()

        def run(io):
            module.gate(io.normed, io.inject_weight, io.partials_out)

    elif op == NumericOperation.HC_APPLY_NORM:
        module = decode._module_for_rows(signature.parts, signature.rows)

        def run(io):
            module.apply_norm(
                io.block,
                io.residual,
                io.gate_partials,
                io.norm_weight,
                io.combined_out,
                io.normalized_out,
                io.epsilon,
            )

    elif op == NumericOperation.HC_APPLY_NORM_GATE:
        module = decode._jit_norm_gate_module()

        def run(io):
            module.gate(
                io.block,
                io.residual,
                io.gate_partials,
                io.norm_weight,
                io.next_inject_weight,
                io.combined_out,
                io.normalized_out,
                io.next_partials_out,
                io.epsilon,
            )

    else:
        raise ValueError("unsupported HC operation")
    return run


def build_mix(signature):
    """The framework resolves TP policy into variant 0 (base) or 1 (tuned)."""
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        SplitKTactic,
        default_tactic,
        run_splitk_dense_gate,
        run_splitk_dense_silu,
        validate_tactic,
    )

    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_mix_splitk import _tuned_tactic

    def tactic(m, n, k):
        if signature.variant == 1:
            result = _tuned_tactic(m, n, k, SplitKTactic, validate_tactic)
            if result is not None:
                return result
        if n <= 512 and k % (128 * 16) == 0:
            result = SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=6)
            try:
                validate_tactic(result, m, n, k)
            except ValueError:
                return default_tactic(m, n, k)
            return result
        return default_tactic(m, n, k)

    k = signature.stream_count * signature.hidden_size
    lowrank = signature.lowrank
    kpad = (lowrank + 127) // 128 * 128
    choices = {
        m: (tactic(m, lowrank, k), tactic(m, k, kpad))
        for m in {min(32, signature.rows), signature.rows % 32}
        if m
    }

    def run(io):
        rows = signature.rows
        scratch = io.scratch_out.view(min(rows, 32), kpad)
        inv_hc = 1.0 / signature.stream_count
        for start in range(0, rows, 32):
            end = min(start + 32, rows)
            m = end - start
            chunk = io.normed[start:end]
            down, up = choices[m]
            run_splitk_dense_silu(
                chunk, io.down_weight.T, scratch[:m, :lowrank], True, down, inv_hc
            )
            run_splitk_dense_gate(
                scratch[:m],
                io.permuted_up_weight.T,
                chunk,
                io.output[start:end],
                True,
                up,
                inv_hc,
                signature.stream_count,
            )

    return run
