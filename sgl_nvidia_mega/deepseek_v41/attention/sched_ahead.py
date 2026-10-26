# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashMLA's split-KV schedules for a covered step (slot "sched_ahead"): built on
side streams at the step's head and joined before the first attention layer.

SGLang builds each ratio's schedule lazily in the first FlashMLA layer of that ratio,
on the critical path, and skips the build when the step's schedule object already has
one. This fills the same objects with the same kernel and the same inputs, so the
schedule is SGLang's bit for bit; the objects are fresh in every metadata build."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DSV4AttnMetadata",
        "_FLASHMLA_SCHED_BLOCK_SIZE_N",
        "_FLASHMLA_SCHED_FIXED_OVERHEAD",
        "_num_sms",
        "_pad_tensor_to_size",
        "core_attn_metadata",
        "get_flashmla_metadata",
        "sparse_page_indices",
        "sparse_topk_lengths",
        "swa_page_indices",
        "swa_topk_lengths",
        "present_ratios",
        "low_ratios",
        '"tile_scheduler_metadata"',
        "num_splits",
    ),
    "kernels/ops/attention/dsv4/flashmla_sched_meta.py": (
        "META_INTS",
        "flashmla_sched_meta",
    ),
    "kernels/ops/attention/dsv4/decode_attention_sm100.py": ("MAX_BATCH",),
    "srt/runtime_context.py": ("get_platform", '"is_blackwell"'),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "metadata",
        "kv_pool",
        "trtllm_attn",
        "request_window",
    ),
}

# One stream per job in flight; jobs beyond these share them round-robin
NUM_STREAMS = 4


class _Job(NamedTuple):
    fm: Any  # the step's FlashMLA schedule object of one ratio
    meta: torch.Tensor
    num_splits: torch.Tensor
    event: torch.cuda.Event
    # the side stream reads these: a padded length tensor is allocated on the step's
    # stream, so it must outlive the join, not start()
    inputs: tuple


class _Streams:
    streams: Optional[list] = None


def first_flashmla_rows(swapab_rows: int) -> int:
    """The fewest rows whose attention runs FlashMLA: past the chain's swap-AB
    (``swapab_rows``, 0 without it) and past SGLang's own swap-AB."""
    from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import MAX_BATCH

    return max(swapab_rows, MAX_BATCH) + 1


def _inputs(core, rows: int) -> list:
    """(schedule object, SWA lengths, SWA width, extra lengths, extra width) per
    ratio, as SGLang's FlashMLA layer passes them (lengths padded to the rows)."""
    from sglang.srt.layers.attention.deepseek_v4_backend import _pad_tensor_to_size

    def match(x):
        if x is None or x.shape[0] == rows:
            return x
        return x[:rows] if x.shape[0] > rows else _pad_tensor_to_size(x, rows, value=1)

    out = []
    for ratio in (0, *core.low_ratios, 4, 128):
        if ratio != 0 and ratio not in core.present_ratios:
            continue
        fm = core.get_flashmla_metadata(ratio)
        if fm is None or getattr(fm, "tile_scheduler_metadata", None) is not None:
            continue
        extra = None if ratio == 0 else core.sparse_page_indices(ratio)
        out.append(
            (
                fm,
                match(core.swa_topk_lengths),
                core.swa_page_indices.shape[-1],
                None if ratio == 0 else match(core.sparse_topk_lengths(ratio)),
                0 if extra is None else extra.shape[-1],
            )
        )
    return out


def start(view, fb, rows: int, first_rows: int) -> list[_Job]:
    """Fork this step's schedules onto the side streams ([] when the step's attention
    does not run FlashMLA's decode kernel)."""
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        META_INTS,
        flashmla_sched_meta,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _FLASHMLA_SCHED_BLOCK_SIZE_N,
        _FLASHMLA_SCHED_FIXED_OVERHEAD,
        DSV4AttnMetadata,
        _num_sms,
    )
    from sglang.srt.runtime_context import get_platform

    core = getattr(view.metadata, "core_attn_metadata", None)
    if (
        rows < first_rows
        or not get_platform().is_blackwell
        or view.trtllm_attn
        or not fb.forward_mode.is_target_verify()
        or not isinstance(core, DSV4AttnMetadata)
        or view.kv_pool.request_window is not None
    ):
        return []
    device = fb.positions.device
    # one query per row (s_q 1), as SGLang's lazy build sizes it
    num_sm_parts = max(_num_sms(device.index), 1)
    if 4 * (5 * rows + 1 + num_sm_parts * META_INTS) > 48 * 1024:
        return []
    if _Streams.streams is None:
        _Streams.streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]
    current = torch.cuda.current_stream()
    jobs = []
    for i, (fm, lengths, width, extra_lengths, extra_width) in enumerate(
        _inputs(core, rows)
    ):
        meta = torch.empty((num_sm_parts, META_INTS), dtype=torch.int32, device=device)
        num_splits = torch.empty((rows + 1,), dtype=torch.int32, device=device)
        stream = _Streams.streams[i % NUM_STREAMS]
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            flashmla_sched_meta(
                meta,
                num_splits,
                topk_length=lengths,
                extra_topk_length=extra_lengths,
                block_size_n=_FLASHMLA_SCHED_BLOCK_SIZE_N,
                fixed_overhead_num_blocks=_FLASHMLA_SCHED_FIXED_OVERHEAD,
                topk=width,
                extra_topk=extra_width,
            )
            event = stream.record_event()
        jobs.append(_Job(fm, meta, num_splits, event, (lengths, extra_lengths)))
    return jobs


def join(jobs: list[_Job]) -> None:
    """Make the step's stream wait for every schedule (none may stay unjoined in a
    capture) and hand them to their schedule objects."""
    current = torch.cuda.current_stream()
    for job in jobs:
        current.wait_event(job.event)
        job.fm.tile_scheduler_metadata = job.meta
        job.fm.num_splits = job.num_splits
