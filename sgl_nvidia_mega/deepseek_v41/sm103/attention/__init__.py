# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The decode attention chain at 1-48 rows: the q-LoRA norm + MXFP8 quant, the swap-AB
sparse attention and its combine, and the fused inverse RoPE + wo_a."""
