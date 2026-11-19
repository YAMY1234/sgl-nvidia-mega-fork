# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import functools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "sgl_nvidia_mega.qwen3_5"


def _module_file(module: str):
    path = ROOT.joinpath(*module.split("."))
    for f in (path.with_suffix(".py"), path / "__init__.py"):
        if f.is_file():
            return f
    return None


@functools.lru_cache(maxsize=None)
def _defined(path: Path):
    names = set()
    for node in ast.parse(path.read_text()).body:
        scope = ast.walk(node) if isinstance(node, (ast.If, ast.Try)) else [node]
        for sub in scope:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in sub.names)
            elif isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for t in targets:
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
    return None if "__getattr__" in names or "*" in names else names


def _sources():
    yield from sorted((ROOT / "sgl_nvidia_mega" / "qwen3_5").rglob("*.py"))
    for kind in ("unit", "kernels"):
        yield from sorted((ROOT / "test" / kind).glob("test_qwen3_5_*.py"))
    yield from sorted((ROOT / "benchmark").glob("bench_qwen3_5_*.py"))


def _imports(path: Path):
    """(line, module, name) of every package import in ``path``, at any depth."""
    package = path.relative_to(ROOT).parent.parts
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if node.level:
            base = package[: len(package) - node.level + 1]
            module = ".".join([*base, module] if module else base)
        if module == PACKAGE or module.startswith(PACKAGE + "."):
            for alias in node.names:
                yield node.lineno, module, alias.name


def test_every_qwen3_5_import_names_a_definition():
    missing, checked = [], 0
    for path in _sources():
        for line, module, name in _imports(path):
            checked += 1
            where = f"{path.relative_to(ROOT)}:{line}: from {module} import {name}"
            target = _module_file(module)
            if target is None:
                missing.append(f"{where} (no module)")
                continue
            names = _defined(target)
            if names is None or name == "*" or name in names:
                continue
            if _module_file(f"{module}.{name}") is None:
                missing.append(where)
    assert checked > 1000, checked
    assert not missing, "\n".join(missing)
