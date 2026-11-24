# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixed geometry of the GLM-5.3 routed-expert layer at tensor parallel 8."""

HIDDEN = 6144  # model hidden size (K of FC1, N of FC2)
NUM_EXPERTS = 256  # routed experts
TOP_K = 8  # experts per token
INTER = 256  # per-rank intermediate size of one expert (gate or up rows)
QUARTERS = 4  # an expert's FC2 output is reduced from four quarter partials
SHARED_PARTS = 4  # the shared expert's output is reduced from four partials
MAX_ROWS = 16  # verify rows of up to four requests
ROUTED_SCALE = 2.5  # routed scaling factor of the checkpoint

# Die-local form (SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_LOCALIZE): the per-layer locality block
# the engine reads in place of its schedule table (``locality.py``). Experts
# 0..NUM_EXPERTS / 2 - 1 live in locality domain 0, the rest in domain 1.
LOC_CTR = 0  # u64 claim word: [0:16) die-0 claims, [16:32) die-1 claims, [32:48) total
LOC_N0 = 128  # u8 [128]: die-0 virtual slots for a grid of N clusters (index N)
LOC_DIE = 1024  # u8 [1024]: locality domain of each %smid
LOC_BYTES = 2048
LOC_EXPERTS_DIE0 = NUM_EXPERTS // 2
