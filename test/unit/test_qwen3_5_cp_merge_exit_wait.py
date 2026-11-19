# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import inspect
import textwrap

import torch  # noqa: F401  (SGLang's platform stubs need torch first)

from sgl_nvidia_mega.qwen3_5 import runtime  # noqa: F401  (installs the CPU stubs)
from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_lite as L


def _tree(name: str) -> ast.FunctionDef:
    fn = getattr(L, name)
    src = textwrap.dedent(inspect.getsource(getattr(fn, "fn", fn)))
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))


def _calls(node, name: str) -> bool:
    return any(
        isinstance(c, ast.Call) and getattr(c.func, "id", None) == name
        for c in ast.walk(node)
    )


def _gate_exit(fn: ast.FunctionDef) -> ast.If:
    """The top-level ``if _gate_on(...) == 0:`` block of the kernel."""
    return next(
        n for n in fn.body if isinstance(n, ast.If) and _calls(n.test, "_gate_on")
    )


def test_the_early_merge_waits_before_its_gated_exit():
    block = _gate_exit(_tree("_cp_merge_h_ll"))
    assert isinstance(block.body[-1], ast.Return)
    guard = block.body[0]
    assert isinstance(guard, ast.If) and _calls(guard, "_gdc_wait_clobber")
    names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    # WAIT and EARLY (NOPUSH implies both), not NOPUSH alone
    assert names == {"WAIT", "EARLY"}, ast.dump(guard.test)


def test_a_merge_that_waits_first_needs_no_second_wait():
    fn = _tree("_cp_merge_h_ll_plain")
    first_wait = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.If) and _calls(n, "_gdc_wait_clobber")
    )
    gate = fn.body.index(_gate_exit(fn))
    assert first_wait < gate
