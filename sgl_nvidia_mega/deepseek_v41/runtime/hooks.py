# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel's hook into SGLang (registered by ``sgl_nvidia_mega.plugin``).

``after_initialize`` runs once a model runner has loaded and prepared its model
(after the post-load transforms, before SGLang sizes the KV cache). For the
DeepSeek-V4.1 target it checks the setup on every rank, builds the executor, and
installs the megakernel's forward on that model instance only: a covered step runs
the executor, every other step the model's own forward.
"""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": ("DeepseekV4ForCausalLM",),
    "srt/model_executor/model_runner.py": ("model", "server_args", "model_config"),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
    "srt/model_executor/model_runner_components/weight_updater.py": (
        '"_derived_weight_cache_error"',
    ),
}

# What SGLang's weight updater reports instead of refreshing the derived state
WEIGHT_UPDATE_ERROR = (
    "Online weight updates are not supported with SGLANG_ENABLE_NVIDIA_MEGAKERNEL=1: "
    "the megakernel's buffers and captured graphs keep state derived from the weights"
)


def after_initialize(result, runner, *args, **kwargs):
    """Build and install the megakernel for the DeepSeek-V4.1 target this runner
    prepared; raise on every rank, with every rank's reasons, if any rank cannot
    serve the setup."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41 import requirements, runtime

    model = runner.model
    if type(model) is not DeepseekV4ForCausalLM:
        return result
    reasons = requirements.disable_reasons(runner.server_args, runner.model_config)
    reasons += runtime.model_disable_reasons(model)
    if "forward" in model.__dict__:
        reasons.append("the model's forward is already replaced on this instance")
    group = get_parallel().tp_group
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, reasons, group=group.cpu_group)
    failed = [f"rank {r}: {why}" for r, rs in enumerate(gathered) for why in rs]
    if failed:
        raise ValueError(
            "SGLANG_ENABLE_NVIDIA_MEGAKERNEL=1 cannot run this setup:\n  - "
            + "\n  - ".join(failed)
        )
    executor = runtime.build(model, runner.server_args)
    install(model, executor)
    model._derived_weight_cache_error = WEIGHT_UPDATE_ERROR
    logger.info(
        "NVIDIA megakernel: active (DeepSeek-V4.1 verify steps of %s rows)",
        sorted(executor.widths),
    )
    return result


def install(model, executor) -> None:
    """Replace ``model.forward`` on this instance: a step ``executor.plan`` covers
    runs the executor, every other step the model's own forward."""
    original = model.forward  # bound to this instance

    @functools.wraps(original)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        step = executor.plan(input_ids, forward_batch, *args, **kwargs)
        if step is None:
            return original(input_ids, positions, forward_batch, *args, **kwargs)
        return executor.forward(
            step, original, input_ids, positions, forward_batch, *args, **kwargs
        )

    model.forward = forward
