# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered decode step carries between its steps.

The runtime creates one per forward and passes it to every step function. Fields
fixed for the step are set at creation; the boundary owns the residual-stream
fields; the per-layer fields are rewritten for each layer. A package keeps values
of its own in ``scratch`` under ``"<package>.<name>"``, and its build result in
``bound["<package>"]``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch


class StepState:
    def __init__(
        self,
        *,
        lm,
        forward_batch,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        plan,
        slots,
        bound: dict,
        streams: dict,
    ):
        # fixed for the step
        self.lm = lm  # DeepseekV4ForCausalLM
        self.model = lm.model  # DeepseekV4Model: layers, embedding, head norm
        self.fb = forward_batch
        self.positions = positions
        self.input_ids = input_ids
        self.input_ids_global = input_ids_global
        self.rows = int(input_ids.shape[0])
        self.plan = plan  # the covered step: its forward mode and rows
        self.slots = slots  # plan.Slots: which kernels fill this width's steps
        self.bound: dict[str, Any] = bound  # package -> its build() result
        self.streams: dict[str, torch.cuda.Stream] = streams  # side streams, by name
        self.events: dict[str, torch.cuda.Event] = {}
        # the residual streams between sublayers, in the boundary's representation:
        # ``hidden`` is what the next boundary call reads; ``state`` the materialized
        # streams (residual + pre-mix) where a reader needs them (Engram, capture)
        self.hidden: Any = None
        self.state: Any = None
        # per layer
        self.layer_id: int = -1
        self.layer = None
        self.call: Any = None  # the layer's read context (input ids, fusion permission)
        # the sublayer input's MXFP8 pair, when the boundary wrote one
        self.x_quant: Any = None
        self.transport: Any = None  # the MoE operand transport of this layer
        # scopes a boundary opens across a sublayer; the runtime closes it per layer
        self.exit: contextlib.ExitStack = contextlib.ExitStack()
        # step outputs and per-step inputs
        self.hash_ids: Optional[torch.Tensor] = None  # Engram token hashes
        self.aux: Optional[list] = None  # the draft's captured hidden states
        self.scratch: dict[str, Any] = {}
