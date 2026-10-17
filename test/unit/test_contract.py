# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang names this package reads, pinned on CPU against the
installed SGLang.

The specs and kernels live outside SGLang, so they read SGLang by name: the
monokernel framework (``sglang.srt.monokernels``) and SGLang internals.
Every module that imports from ``sglang`` lists what it reads in
``SGLANG_READS``. A refactor that renames one fails here, naming the file to
repair, instead of failing at a GPU server's startup. The checks are static
(AST): no module of this package is imported. Run it against each SGLang this
package supports, and nightly against SGLang main.
"""

import ast
import os
from functools import lru_cache
from pathlib import Path

import pytest
import sglang

import sgl_nvidia_mega

ROOT = Path(sglang.__file__).resolve().parent  # the installed sglang package
PACKAGE = Path(sgl_nvidia_mega.__file__).resolve().parent


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


@lru_cache(maxsize=None)
def _provided(path: Path) -> frozenset:
    """What a source file provides: definitions, assignment targets (``x.name = ...``
    too), imported names, parameters as ``f(p)`` and string literals as ``"x"``."""
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                names.update(f"{node.name}({p.arg})" for p in params)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute):
                        names.add(t.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(f'"{node.value}"')
    return frozenset(names)


def _declared(path: Path) -> dict:
    """The module's ``SGLANG_READS``, or {} without one."""
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_READS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def _sglang_imports(path: Path) -> set:
    return {
        a.asname or a.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("sglang.")
        for a in node.names
    }


READERS = [
    p for p in sorted(PACKAGE.rglob("*.py")) if _declared(p) or _sglang_imports(p)
]


def _id(path: Path) -> str:
    return str(path.relative_to(PACKAGE.parent))


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_every_sglang_import_is_declared(path):
    declared = {n for names in _declared(path).values() for n in names}
    undeclared = _sglang_imports(path) - declared
    assert not undeclared, f"{_id(path)}: declare {sorted(undeclared)} in SGLANG_READS"


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_sglang_provides_what_the_plugin_reads(path):
    missing = []
    for pattern, names in _declared(path).items():
        files = sorted(ROOT.glob(pattern))
        assert files, f"{_id(path)}: no file matches {pattern}"
        provided = frozenset().union(*map(_provided, files))
        missing += [f"{pattern}: {n}" for n in names if n not in provided]
    assert not missing, (
        f"SGLang no longer provides what {_id(path)} reads; repair it and its "
        "SGLANG_READS:\n  " + "\n  ".join(missing)
    )


def _seams(path: Path) -> dict:
    """The module's ``SGLANG_SEAMS``: SGLang names a slot probes for at build."""
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_SEAMS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


SEAM_READERS = [p for p in sorted(PACKAGE.rglob("*.py")) if _seams(p)]


@pytest.mark.skipif(
    os.environ.get("SGL_MEGA_REQUIRE_SEAMS") != "1",
    reason="the slots' SGLang seams are optional; set SGL_MEGA_REQUIRE_SEAMS=1 "
    "against an SGLang that has them",
)
@pytest.mark.parametrize("path", SEAM_READERS, ids=_id)
def test_sglang_provides_the_seams_the_slots_use(path):
    missing = []
    for pattern, names in _seams(path).items():
        files = sorted(ROOT.glob(pattern))
        provided = frozenset().union(*map(_provided, files))
        missing += [f"{pattern}: {n}" for n in names if n not in provided]
    assert not missing, f"{_id(path)}: this SGLang lacks " + ", ".join(missing)
