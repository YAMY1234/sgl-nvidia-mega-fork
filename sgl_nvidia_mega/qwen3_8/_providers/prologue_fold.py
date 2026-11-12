# SPDX-License-Identifier: Apache-2.0
"""QSA indexer prologue (q_prep and the graph group compression) in one launch."""

from dataclasses import fields
from enum import IntEnum

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericOperation,
    QsaPrologueFoldIO,
    QsaPrologueFoldSignature,
    ScalarType,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/numeric.py": (
        "NumericOperation",
        "QsaPrologueFoldIO",
        "QsaPrologueFoldSignature",
        "ScalarType",
    ),
}

_INPUT_SCALARS = (ScalarType.BF16, ScalarType.FP16)


def supported(s):
    if not isinstance(s, QsaPrologueFoldSignature):
        return False
    if any(
        type(v := getattr(s, f.name)) is not int and not isinstance(v, IntEnum)
        for f in fields(s)
    ):
        return False
    return (
        s.operation == NumericOperation.QSA_PROLOGUE_FOLD
        and s.rows == 0
        and 0 < s.q_heads <= s.q_heads_padded
        and s.head_dim in (64, 128, 256)
        and 0 < s.rotary_dim <= s.head_dim
        and s.rotary_dim % 2 == 0
        and s.rope_axes in (1, 3)
        and 0 < s.compress_ratio <= 16
        and s.scalar_type in _INPUT_SCALARS
        # q_prep's rule: the output is the input dtype or FP8 E4M3.
        and s.out_scalar_type in (s.scalar_type, ScalarType.FP8_E4M3)
        and s.neox in (0, 1)
    )


def build(signature):
    if not supported(signature):
        raise ValueError("unsupported QSA prologue fold signature")
    return QsaPrologueFold(signature)


def _torch_dtype(scalar_type):
    import torch

    return {
        ScalarType.BF16: torch.bfloat16,
        ScalarType.FP16: torch.float16,
        ScalarType.FP8_E4M3: torch.float8_e4m3fn,
    }[scalar_type]


class QsaPrologueFold:
    def __init__(self, signature):
        from sgl_nvidia_mega.qwen3_8.qsa.prologue_fold import precompile

        self.signature = signature
        self._dtypes = (
            _torch_dtype(signature.scalar_type),
            _torch_dtype(signature.out_scalar_type),
        )
        # Build at registration: a first build at serving time stalls the scheduler.
        precompile(*self._dtypes, signature.head_dim, bool(signature.neox))

    def __call__(self, io):
        from sgl_nvidia_mega.qwen3_8.qsa.prologue_fold import _module

        if not isinstance(io, QsaPrologueFoldIO):
            raise TypeError("QSA prologue fold requires its borrowed tensor view")
        s = self.signature
        if (io.qk.dtype, io.q_out.dtype) != self._dtypes or io.positions.shape[
            0
        ] != s.rope_axes:
            raise ValueError("QSA prologue fold view differs from its signature")
        # The launcher checks every shape, dtype and stride against the signature's
        # geometry before it writes.
        _module(*self._dtypes, s.head_dim, bool(s.neox)).fold(
            io.qk,
            io.q_out,
            io.q_weight,
            io.cos_sin_cache,
            io.axis_map,
            io.positions,
            s.rope_axes,
            io.cache_loc,
            io.key_state_buffer,
            io.rope_position_buffer,
            s.q_heads,
            s.rotary_dim,
            io.q_eps,
            io.group_locs,
            io.k_weight,
            io.write_locs,
            io.compressed_k_buffer,
            s.compress_ratio,
            io.k_eps,
        )
