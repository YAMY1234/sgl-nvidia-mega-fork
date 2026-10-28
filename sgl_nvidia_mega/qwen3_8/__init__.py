# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.8 Flash Next (FN, ``Qwen4ExpForConditionalGeneration``) megakernel: its spec
and the setups it serves (``spec``), the slot plan (``plan``) and the executor of a
covered target-verify step and its build (``runtime``).

FN's decoder layers are Qwen3.5's with a four-stream hyper-connection (HC) residual
inside the layer boundaries, QSA sparse attention on the full-attention layers and a
PLE n-gram embedding on its PLE layers. The step state, the embedding and the parity
check are Qwen3.5's (``qwen3_5.runtime``). The kernels read FN's sizes from
text_config at build. The MTP draft runs SGLang's own forward."""
