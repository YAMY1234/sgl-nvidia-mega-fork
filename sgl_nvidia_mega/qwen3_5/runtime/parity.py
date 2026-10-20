# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: the covered step against SGLang's own forward, bit for bit, in eager
steps and inside the decode CUDA graphs (``SGLANG_NVIDIA_MEGAKERNEL_PARITY``).

``1`` runs SGLang's forward on the step, then the megakernel's, and compares the logits
and the hidden rows they return bit for bit. An eager step logs the comparison at once.
A captured step records both forwards and the comparison, which adds to device counters
on every replay; the next eager forward on the model (a new request's prefill) logs
what the replays added. ``control`` runs SGLang's forward twice instead: the run-to-run
floor of the same comparison. ``trace`` also hooks every module both forwards call and
logs the first module call whose input or output differs (eager steps only).

A verify forward rewrites its requests' conv windows in place (SGLang's commit later
rolls them back to the accepted token), so the second forward runs on the windows as
they were before the first: their rows are saved and put back between the two. The
verify only reads the SSM checkpoints and only writes this step's ReplaySSM window.
An eager line's valid_rows counts the step's requests with a mamba slot. The forwards
SGLang runs before capturing a graph log as ``capture-warmup``: the graph runner's
placeholder inputs give every row the same KV slot, and there two runs of SGLang's own
forward already differ (``control``), first at the first full-attention layer's
attention, so those lines say nothing about the step.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "req_to_token_pool",
        "forward_metadata",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/mem_cache/memory_pool.py": ("mamba_pool", "mamba_cache", "conv"),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}

MODE = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "0")
ENABLED = MODE in ("1", "control", "trace")

# element size -> the integer type its bits are compared as
_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class Parity:
    """One rank's comparison counters per width: covered steps, the elements of the
    logits and of the hidden rows whose bits differ, and the elements compared (zero
    when a forward returned no such output, e.g. SGLang's autotune dummy run)."""

    def __init__(self, widths: tuple[int, ...], device, model=None):
        self.counters = {
            w: torch.zeros(5, dtype=torch.int64, device=device) for w in widths
        }
        self.logged = {w: [0] * 5 for w in widths}
        self.label = " control" if MODE == "control" else ""
        self.model = model  # whose modules trace mode hooks

    def run(self, step, original, megakernel):
        """``original()`` then ``megakernel()`` (``original()`` again for control) on the
        same step; returns the second's output."""
        traced = MODE == "trace" and not torch.cuda.is_current_stream_capturing()
        slots, convs = conv_rows()
        # padded rows hold -1; their slot 0 stand-in is saved and put back unchanged
        rows = None if slots is None else slots.clamp(min=0)
        saved = [conv[:, rows].clone() for conv in convs]
        want, calls = self._traced(original) if traced else (original(), None)
        want = (_clone(want.next_token_logits), _clone(want.hidden_states))
        for conv, rows_before in zip(convs, saved):
            conv[:, rows] = rows_before
        if traced:
            got, got_calls = self._traced(megakernel)
            logger.info(
                "NVIDIA megakernel parity trace %s rows=%d: %s",
                step.mode,
                step.rows,
                first_divergence(calls, got_calls),
            )
        else:
            got = original() if MODE == "control" else megakernel()
        pairs = list(zip(want, (got.next_token_logits, got.hidden_states)))
        if torch.cuda.is_current_stream_capturing():
            counts = self.counters[step.rows]
            counts[0:1].add_(1)
            for i, (a, b) in enumerate(pairs, start=1):
                counts[i : i + 1].add_(differing(a, b))
                counts[i + 2 : i + 3].add_(0 if a is None else a.numel())
            return got
        from sglang.srt.model_executor.runner import get_is_capture_mode

        logger.info(
            "NVIDIA megakernel parity%s %s %s rows=%d logits_differ=%d "
            "hidden_differ=%d logits_compared=%d hidden_compared=%d "
            "logits_max_abs=%s hidden_max_abs=%s valid_rows=%s",
            self.label,
            "capture-warmup" if get_is_capture_mode() else "eager",
            step.mode,
            step.rows,
            *(int(differing(a, b)) for a, b in pairs),
            *(0 if a is None else a.numel() for a, _ in pairs),
            *(_max_abs(a, b) for a, b in pairs),
            "n/a" if slots is None else int((slots >= 0).sum()),
        )
        return got

    def flush(self) -> None:
        """Log what the replays of captured steps added since the last flush; a host
        sync, so only outside capture."""
        if torch.cuda.is_current_stream_capturing():
            return
        for rows, counts in self.counters.items():
            now = counts.tolist()
            if now == self.logged[rows]:
                continue
            added = [n - o for n, o in zip(now, self.logged[rows])]
            self.logged[rows] = now
            logger.info(
                "NVIDIA megakernel parity%s graph target_verify rows=%d steps=%d "
                "logits_differ=%d hidden_differ=%d logits_compared=%d "
                "hidden_compared=%d",
                self.label,
                rows,
                *added,
            )

    def _traced(self, fn):
        """fn()'s output and every module call it made: (name, input, output)."""
        names = {m: n or "model" for n, m in self.model.named_modules()}
        calls = []

        def hook(module, args, out):
            calls.append((names[module], _first_tensor(args), _first_tensor(out)))

        handles = [m.register_forward_hook(hook) for m in names]
        try:
            return fn(), calls
        finally:
            for h in handles:
                h.remove()


def conv_rows():
    """This step's mamba slots and the pool's conv-window tensors ([layers, slots,
    ...]); (None, []) without a linear-attention backend."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    linear = getattr(backend, "linear_attn_backend", None)
    if linear is None:
        return None, []
    cache = linear.req_to_token_pool.mamba_pool.mamba_cache
    return linear.forward_metadata.mamba_cache_indices, list(cache.conv)


def first_divergence(want, got) -> str:
    """The first call in got whose input or output bits differ from the same module's
    call of the same index in want."""
    index, seen = {}, {}
    for name, a_in, a_out in want:
        seen[name] = seen.get(name, -1) + 1
        index[(name, seen[name])] = (a_in, a_out)
    seen, common = {}, 0
    for name, b_in, b_out in got:
        seen[name] = seen.get(name, -1) + 1
        pair = index.get((name, seen[name]))
        if pair is None:
            continue
        common += 1
        for what, a, b in (("input", pair[0], b_in), ("output", pair[1], b_out)):
            if a is None or b is None:
                continue
            if a.shape != b.shape or a.dtype != b.dtype:
                return f"{name}#{seen[name]} {what}: {a.dtype}{tuple(a.shape)} against {b.dtype}{tuple(b.shape)}"
            n = int(differing(a, b))
            if n:
                return f"{name}#{seen[name]} {what}: {n} of {a.numel()} elements differ"
    return f"no diverging call among {common} common module calls"


def _first_tensor(value):
    """A clone of the first tensor in value (a tensor, a tuple / list of them, or an
    output with next_token_logits), or None."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (tuple, list)):
        return next((v.clone() for v in value if isinstance(v, torch.Tensor)), None)
    logits = getattr(value, "next_token_logits", None)
    return logits.clone() if isinstance(logits, torch.Tensor) else None


def differing(a, b):
    """The number of elements of a and b whose bits differ (a device scalar)."""
    if a is None or b is None:
        if a is None and b is None:
            return 0
        raise RuntimeError("parity: only one forward returned this output")
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError(
            f"parity: {a.dtype}{tuple(a.shape)} against {b.dtype}{tuple(b.shape)}"
        )
    bits = _BITS[a.element_size()]
    return (a.contiguous().view(bits) != b.contiguous().view(bits)).sum()


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b) -> str:
    if a is None:
        return "n/a"
    return f"{(a.float() - b.float()).abs().max().item():.3g}"
