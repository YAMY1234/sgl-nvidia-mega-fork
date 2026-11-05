# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch

_MAX_ROWS = 32
_KPAD_ALIGN = 128


def pad_lowrank(lowrank: int) -> int:
    return (lowrank + _KPAD_ALIGN - 1) // _KPAD_ALIGN * _KPAD_ALIGN


def permute_pad_up_weight(w_up: torch.Tensor, hc_count: int) -> torch.Tensor:
    k, lowrank = w_up.shape
    hidden_size = k // hc_count
    padded = torch.zeros(
        (k, pad_lowrank(lowrank)), dtype=w_up.dtype, device=w_up.device
    )
    padded[:, :lowrank] = w_up.detach()
    src = torch.arange(k, device=w_up.device)
    perm = torch.empty(k, dtype=torch.long, device=w_up.device)
    perm[(src % hidden_size) * hc_count + (src // hidden_size)] = src
    return padded[perm].contiguous()


_TUNED_TACTICS = os.environ.get("SGLANG_HC_MACRO", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _tuned_tactic(m: int, n: int, k: int, SplitKTactic, validate_tactic):
    if n <= 512 and k % (128 * 16) == 0:
        tactic = (
            SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=6)
            if m <= 8
            else SplitKTactic(mma_m=64, mma_n=8, split_k=8, ab_stages=8)
        )
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        tactic = SplitKTactic(mma_m=64, mma_n=mma_n, split_k=1, ab_stages=4)
    try:
        validate_tactic(tactic, m, n, k)
    except ValueError:
        return None
    return tactic
