# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect

import pytest
import torch  # noqa: F401  (SGLang's platform stubs need torch first)

from sgl_nvidia_mega.qwen3_5 import runtime  # noqa: F401  (installs the CPU stubs)
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L

LL_PUSHERS = (
    "_cp_gather_q_ll",
    "_cp_a2a_merge_ll",
    "_cp_a2a_merge4_ll",
    "_cp_merge_h_ll",
    "_cp_merge_h_ll_plain",
)


def _source(name: str) -> str:
    fn = getattr(L, name)
    return inspect.getsource(getattr(fn, "fn", fn))


def test_the_store_is_a_relaxed_system_scope_64_bit_store():
    src = _source("_st_relaxed_sys_b64")
    assert "st.relaxed.sys.global.b64 [$1], $2;" in src
    assert "is_pure=False" in src


@pytest.mark.parametrize("name", LL_PUSHERS)
def test_every_ll_push_is_relaxed_sys(name):
    src = _source(name)
    assert src.count("_st_relaxed_sys_b64(") >= 1, name
    assert "tl.store(dst" not in src, name


def test_the_precise_merges_residual_push_is_relaxed_sys():
    src = _source("_cp_merge_h_ll")
    assert "_st_relaxed_sys_b64(dst_lo + " in src
    assert "tl.store(dst_lo" not in src


def test_the_flag_protected_push_keeps_its_stores():
    assert "tl.store(dst + hd, x)" in _source("_cp_a2a_merge")


def test_the_per_head_merges_pushes_are_its_data_lse_and_residual_words():
    src = _source("_cp_merge_h_ll")
    assert src.count("_st_relaxed_sys_b64(") == 3
    assert "_st_relaxed_sys_b64(dst_lo + " in src
