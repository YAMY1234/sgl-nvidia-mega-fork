# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel's two hooks into SGLang (registered by ``sgl_nvidia_mega.plugin``).

``after_load_model`` runs when a model runner has loaded its model: for the
DeepSeek-V4.1 target it checks the setup on every rank, then builds the executor
before SGLang sizes the KV cache. ``around_forward`` wraps the target's ``forward``:
a covered step runs the executor, every other step SGLang's own forward.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": ("DeepseekV4ForCausalLM",),
    "srt/model_executor/model_runner.py": ("model", "server_args", "model_config"),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
}

# Where the built executor rides on the model (not setattr: an nn.Module attribute
# would join the state dict)
EXECUTOR = "_nvidia_megakernel"


def after_load_model(result, runner, *args, **kwargs):
    """Build the megakernel for the DeepSeek-V4.1 target this runner loaded; raise on
    every rank, with every rank's reasons, if any rank cannot serve the setup."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41 import requirements, runtime

    model = runner.model
    if type(model) is not DeepseekV4ForCausalLM:
        return result
    reasons = requirements.disable_reasons(runner.server_args, runner.model_config)
    reasons += runtime.model_disable_reasons(model)
    group = get_parallel().tp_group
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, reasons, group=group.cpu_group)
    failed = [f"rank {r}: {why}" for r, rs in enumerate(gathered) for why in rs]
    if failed:
        raise ValueError(
            "SGLANG_ENABLE_NVIDIA_MEGAKERNEL=1 cannot run this setup:\n  - "
            + "\n  - ".join(failed)
        )
    model.__dict__[EXECUTOR] = runtime.build(model, runner.server_args)
    return result


def around_forward(original, lm, input_ids, positions, forward_batch, *args, **kwargs):
    """The executor for a covered step, else SGLang's ``forward``."""
    executor = lm.__dict__.get(EXECUTOR)
    step = (
        None
        if executor is None
        else executor.plan(input_ids, forward_batch, *args, **kwargs)
    )
    if step is None:
        return original(lm, input_ids, positions, forward_batch, *args, **kwargs)
    return executor.forward(
        step, original, input_ids, positions, forward_batch, *args, **kwargs
    )
