# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVIDIA megakernels for SGLang: per model, the decode step's own code and the
kernels that only make sense inside a fused step.

SGLang loads the package as a general plugin (``pyproject.toml``: the
``sglang.srt.plugins`` entry point, ``plugin.register``). With
``SGLANG_ENABLE_NVIDIA_MEGAKERNEL=1`` it hooks the megakernel into SGLang's model
loading and the model's ``forward``; without it the package registers nothing. The
shared CUDA runtime is in ``common``; each model's step code and kernels per ISA are
in ``<model>/``.
"""

__version__ = "0.1.0.dev0"
