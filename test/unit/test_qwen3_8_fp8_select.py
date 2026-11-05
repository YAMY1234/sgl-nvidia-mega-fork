# SPDX-License-Identifier: Apache-2.0
"""SELECT-only boundary after native FP8 scoring; CPU storage and raw spies.

Scoring and CUDA execution are not exercised here. Native scorer logits enter
the existing selector as FP32, without transferring its query/cache operands.
"""

from dataclasses import fields, replace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType as D
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    SparseSelectIO,
)
from sglang.srt.kernel_plugins.nvidia.numeric_binding import bind_numeric_operator
from test_qwen3_8_grouped_sparse import grouped_spies as grouped_spies
from test_qwen3_8_grouped_sparse import signature

from sgl_nvidia_mega.qwen3_8 import providers


@pytest.mark.parametrize("rows", (4, 8, 12, 16))
@pytest.mark.parametrize("heads", (8, 16))
def test_native_score_buffer_select_only_ABC(grouped_spies, rows, heads):
    calls, state, _, config = grouped_spies
    s = signature(rows, heads, variant=1, select=True, capacity=65536)
    providers._cached_offer.cache_clear()
    try:
        offer = providers.make_offer(s)
        assert offer is not None and offer.signature.scalar_type == D.FP32
        run = bind_numeric_operator(
            enabled=True,
            signature=s,
            prepared=(s,),
            offers=(offer,),
            stock=lambda _: pytest.fail("selected provider retried stock"),
        )
        # Three live allocations establish that warmup does not pin tensor
        # addresses. The unsorted raw marker must pass through without sorting.
        views = [
            SparseSelectIO(
                torch.empty((rows, 65536), dtype=torch.float32),
                torch.full((rows,), n, dtype=torch.int32),
                512,
                torch.empty((rows, 512), dtype=torch.int32),
                None,
                torch.empty(0, dtype=torch.uint8),
            )
            for n in (499, 513, 8193)
        ]
        for i, io in enumerate(views):
            state.capturing = i == 2
            assert run(io) is None
            assert io.counts_out is None
            assert torch.equal(
                io.selected_indices_out, torch.arange(511, -1, -1).expand(rows, -1)
            )
        assert calls[0] == ("split-prepare", (*config["_config"](rows, 65536), True))
        assert sum(kind == "split-prepare" for kind, _ in calls) == 1
        launches = [args for kind, args in calls if kind == "split-launch"]
        assert len(launches) == 3
        for args, io in zip(launches, views):
            assert all(
                got is want
                for got, want in zip(
                    args, (io.scores, io.lengths, io.selected_indices_out)
                )
            )
        assert len({io.scores.data_ptr() for io in views}) == 3
        assert len({io.selected_indices_out.data_ptr() for io in views}) == 3
    finally:
        providers._cached_offer.cache_clear()


def test_fp8_is_not_the_select_operand_or_a_score_offer(grouped_spies):
    s = signature(4, 8, variant=1, select=True, capacity=65536)
    assert not providers.supported(replace(s, scalar_type=D.FP8_E4M3))
    assert not providers.supported(
        replace(s, operation=O.SPARSE_SCORE, scalar_type=D.FP8_E4M3)
    )
    names = {f.name for f in fields(SparseSelectIO)}
    assert not names.intersection(("query", "compressed_keys", "query_weights"))


def test_select_registration_alone_does_not_warm_capture(grouped_spies):
    calls, state, _, _ = grouped_spies
    s = signature(4, 8, variant=1, select=True, capacity=65536)
    providers._cached_offer.cache_clear()
    try:
        offer = providers.make_offer(s)
        io = SparseSelectIO(
            torch.empty((4, 65536)),
            torch.full((4,), 513, dtype=torch.int32),
            512,
            torch.empty((4, 512), dtype=torch.int32),
            None,
            torch.empty(0, dtype=torch.uint8),
        )
        state.capturing = True
        with pytest.raises(RuntimeError, match="cold layout"):
            offer.run(io)
        assert calls == []
    finally:
        providers._cached_offer.cache_clear()
