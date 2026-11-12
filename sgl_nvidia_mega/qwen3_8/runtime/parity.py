# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: covered steps against SGLang's own forward (``SGLANG_NVIDIA_MEGAKERNEL_PARITY``).

``1`` runs SGLang's forward on the step, puts back the recurrent state it changed,
runs the executor, and compares the logits and the HC streams (``hidden_states``) bit
for bit. ``layer`` also compares every decoder layer's output streams (SGLang's
through forward hooks on ``model.layers[i]``, the executor's from its layer loop) and
reports the first differing layer. ``control`` runs SGLang's forward twice: the
run-to-run floor of the same comparison.

An eager step logs its line at once. A captured step records both forwards, the state
restore and the comparison in the graph; the comparison adds to device counters on
every replay, and the next eager forward on the model logs what the replays added
(``graph`` lines).

State a forward changes in place and later reads, put back between the two forwards
(rows of this step's state slots only): the GDN conv states (a verify's conv update
overwrites them; SGLang's commit restores them later), on a prefill also the GDN
temporal state, and the PLE n-gram context and short-conv state. Everything else a
covered step writes is written with the same values by both forwards (K/V and QSA
rings at this step's slots, verify intermediates, ring entries past the cursors)."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

MODE = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "0")
ENABLED = MODE in ("1", "layer", "control")

_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# steps, logits differ, hidden differ, logits compared, hidden compared
_FIELDS = 5


def _differing(a, b) -> torch.Tensor:
    if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
        return torch.tensor(-1, dtype=torch.int64, device=_device(a, b))
    bits = _BITS[a.element_size()]
    return (a.contiguous().view(bits) != b.contiguous().view(bits)).sum()


def _device(*ts):
    for t in ts:
        if t is not None:
            return t.device
    return torch.device("cuda", torch.cuda.current_device())


def _numel(t) -> int:
    return 0 if t is None else t.numel()


def _outputs(out):
    logits = getattr(out, "next_token_logits", None)
    hidden = getattr(out, "hidden_states", None)
    return (
        None if logits is None else logits.clone(),
        None if hidden is None else hidden.clone(),
    )


def _state_rows(step, forward_batch):
    """(tensor, slot dim, slots) of the recurrent state this step's forward changes."""
    from sglang.srt.mem_cache.kv.decl import STATE_PART
    from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, SHORT_CONV_PART
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    router = get_token_to_kv_pool().layer_router
    mamba = router.part_view(STATE_PART)
    if mamba is None:
        return []
    slots = (
        get_attn_backend()
        .forward_state_slots(int(forward_batch.batch_size), None)
        .working.long()
        .clamp(min=0)
    )
    tensors = [(conv, 1) for conv in mamba.mamba_cache.conv]
    if step.mode == "prefill":
        tensors.append((mamba.mamba_cache.temporal, 1))
    ngram = router.part_view(NGRAM_PART)
    if ngram is not None and getattr(ngram, "context", None) is not None:
        tensors.append((ngram.context, 0))
    short_conv = router.part_view(SHORT_CONV_PART)
    if short_conv is not None and getattr(short_conv, "conv_state", None) is not None:
        tensors.append((short_conv.conv_state, 1))
    return [(t, dim, slots) for t, dim in tensors]


class Parity:
    def __init__(self, model, *, draft: bool):
        self.model = model
        self.label = ("draft" if draft else "target") + (
            " control" if MODE == "control" else ""
        )
        self.counters: dict[tuple, torch.Tensor] = {}
        self.layer_counters: dict[tuple, torch.Tensor] = {}
        self.logged: dict[tuple, list] = {}
        # set while a forward runs: the executor's per-layer outputs
        self.layers_out: list | None = None

    # --- per-layer capture ---------------------------------------------------
    def record_layer(self, i: int, hidden: torch.Tensor) -> None:
        if self.layers_out is not None:
            self.layers_out.append((i, hidden.clone()))

    def _stock_layers(self, stock):
        outs, handles = [], []
        m = self.model.model
        for i in range(m.start_layer, m.end_layer):

            def hook(module, args, out, i=i):
                outs.append((i, (out[0] if isinstance(out, tuple) else out).clone()))

            handles.append(m.layers[i].register_forward_hook(hook))
        try:
            result = stock()
        finally:
            for h in handles:
                h.remove()
        return result, outs

    # --- the comparison --------------------------------------------------------
    def run(self, step, forward_batch, stock, executor):
        saved = [
            (t, dim, slots, t.index_select(dim, slots))
            for t, dim, slots in _state_rows(step, forward_batch)
        ]
        if MODE == "layer":
            want, want_layers = self._stock_layers(stock)
        else:
            want, want_layers = stock(), []
        want_logits, want_hidden = _outputs(want)
        for t, dim, slots, rows in saved:
            t.index_copy_(dim, slots, rows)
        if MODE == "control":
            got = stock()
            got_layers = []
        else:
            self.layers_out = [] if MODE == "layer" else None
            try:
                got = executor()
            finally:
                got_layers, self.layers_out = self.layers_out or [], None
        got_logits, got_hidden = _outputs(got)
        values = torch.stack(
            [
                torch.ones((), dtype=torch.int64, device=_device(got_logits)),
                _differing(want_logits, got_logits),
                _differing(want_hidden, got_hidden),
                torch.tensor(_numel(got_logits), dtype=torch.int64, device=_device(got_logits)),
                torch.tensor(_numel(got_hidden), dtype=torch.int64, device=_device(got_logits)),
            ]
        )
        per_layer = None
        if want_layers and got_layers:
            got_by = dict(got_layers)
            per_layer = torch.stack(
                [_differing(w, got_by.get(i)) for i, w in want_layers]
            )
        key = (step.mode, step.rows)
        if torch.cuda.is_current_stream_capturing():
            counters = self.counters.setdefault(
                key, torch.zeros(_FIELDS, dtype=torch.int64, device=values.device)
            )
            counters.add_(values)
            if per_layer is not None:
                layer_counters = self.layer_counters.setdefault(
                    key, torch.zeros_like(per_layer)
                )
                layer_counters.add_(per_layer)
        else:
            self._log("eager", key, values.tolist(), per_layer)
        return got

    def flush(self) -> None:
        """Log what the captured steps' replays added since the last flush."""
        if torch.cuda.is_current_stream_capturing():
            return
        for key, counters in self.counters.items():
            now = counters.tolist()
            before = self.logged.setdefault(key, [0] * _FIELDS)
            delta = [a - b for a, b in zip(now, before)]
            if delta[0] <= 0:
                continue
            self.logged[key] = now
            layers = self.layer_counters.get(key)
            self._log("graph", key, delta, layers)

    def _log(self, kind, key, values, per_layer) -> None:
        steps, logits_differ, hidden_differ, logits_n, hidden_n = values
        first = ""
        if per_layer is not None:
            differ = [(i, int(n)) for i, n in enumerate(per_layer.tolist()) if n]
            first = (
                f" first_layer_differ={differ[0][0]} layers_differ={len(differ)}"
                if differ
                else " layers_differ=0"
            )
        logger.info(
            "NVIDIA megakernel parity %s %s %s rows=%d steps=%d logits_differ=%d "
            "hidden_differ=%d logits_compared=%d hidden_compared=%d%s",
            self.label,
            kind,
            key[0],
            key[1],
            steps,
            logits_differ,
            hidden_differ,
            logits_n,
            hidden_n,
            first,
        )
