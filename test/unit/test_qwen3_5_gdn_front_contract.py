# SPDX-License-Identifier: Apache-2.0
"""Reviewed optional numeric front: exact digest and unchanged stock continuation."""

import ast

import pytest
import test_contract as contract

KEY = "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward"
REVIEWED = (
    "a5e9e3bd054ff18f",
    "098d62c5d4dcbaf4",
    "103ba5ab4134e962",
    "dcb17527f75b4ce7",
    "a4680caed736c525",
    "fef05061e3986ead",
    "336d8b0724589d88",
)


def mirror_path():
    return contract.PACKAGE / "qwen3_5/gdn/__init__.py"


def test_gdn_front_mirror_accepts_only_reviewed_exact_bodies():
    assert contract._mirrors(mirror_path())[KEY] == REVIEWED
    contract.test_the_mirrored_sglang_code_is_unchanged(mirror_path(), KEY, REVIEWED)


def test_unknown_forward_digest_is_still_rejected(monkeypatch):
    monkeypatch.setattr(contract, "mirror_digest", lambda *args: "ffffffffffffffff")
    with pytest.raises(AssertionError, match="SGLang changed"):
        contract.test_the_mirrored_sglang_code_is_unchanged(
            mirror_path(), KEY, REVIEWED
        )


def test_installed_stock_body_is_exact_after_optional_early_return():
    path = contract.ROOT / "srt/models/qwen3_5.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5GatedDeltaNet"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    text = path.read_text()
    got = contract.mirror_digest(path, "Qwen3_5GatedDeltaNet.forward")
    ranges = []
    if got in REVIEWED[1:]:
        assert ast.unparse(fn.body[1]) == (
            "numeric_output = self._try_numeric_gdn_front(hidden_states, forward_batch)"
        )
        assert ast.unparse(fn.body[2]) == (
            "if numeric_output is not None:\n    return numeric_output"
        )
        ranges.append((fn.body[1].lineno - 1, fn.body[2].end_lineno, []))
    if got in REVIEWED[2:]:
        assert ast.unparse(fn.body[4]) == (
            "glue_output = self._try_numeric_gdn_verify_glue(hidden_states, forward_batch, "
            "projected_states_qkvz, projected_states_ba)"
        )
        assert ast.unparse(fn.body[5]) == (
            "if glue_output is not None:\n    return glue_output"
        )
        ranges.append((fn.body[4].lineno - 1, fn.body[5].end_lineno, []))
    if got in REVIEWED[3:]:
        node = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.BoolOp)
            and any(
                isinstance(t, ast.Name) and t.id == "use_strided_prefill_z"
                for t in n.targets
            )
        )
        expected = ast.parse(
            "_is_cuda and ("
            "forward_batch.logical_forward_mode.is_extend_without_speculative() or "
            "(os.environ.get('Q38_GDN_FRONT') == '1' and "
            "forward_batch.forward_mode.is_target_verify()))",
            mode="eval",
        ).body
        assert ast.dump(node.value) == ast.dump(expected)
        # Restore only this independently reviewed source-route seam. Its view
        # helper and all arithmetic/stock continuation remain byte/token checked.
        replacement = " " * node.col_offset + (
            "use_strided_prefill_z = (_is_cuda and "
            "forward_batch.logical_forward_mode.is_extend_without_speculative())\n"
        )
        ranges.append((node.lineno - 1, node.end_lineno, [replacement]))
    if got in (REVIEWED[5], REVIEWED[6]):
        imported = [
            n
            for n in fn.body
            if isinstance(n, ast.ImportFrom)
            and n.module == "sglang.kernels.plugin_kernels.nvidia.numeric"
        ]
        assert len(imported) == 1
        node = imported[0]
        assert ast.unparse(node) == (
            "from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation"
        )
        ranges.append((node.lineno - 1, node.end_lineno, []))
    if got in (REVIEWED[4], REVIEWED[5], REVIEWED[6]):
        route = next(
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and "_numeric_gdn_m32_norm_supported" in ast.unparse(n.test)
        )
        row_program = (
            "hidden_states.shape[0] == 32"
            if got == REVIEWED[4]
            else "any(s.operation == NumericOperation.GDN_CORE and s.variant == 1 "
            "and s.rows == hidden_states.shape[0] for s in "
            "getattr(self, '_numeric_gdn_standalone_norm_signatures', ()))"
        )
        expected = ast.parse(
            "isinstance(hidden_states, torch.Tensor) and " + row_program + " "
            "and self._numeric_gdn_m32_norm_supported() "
            "and forward_batch.forward_mode.is_target_verify() "
            "and not use_fused_proj_conv and z is not None "
            "and hidden_states.dtype == z.dtype == torch.bfloat16",
            mode="eval",
        ).body
        assert ast.dump(route.test) == ast.dump(expected)
        assert len(route.orelse) == 1
        old_call = route.orelse[0]
        assert ast.unparse(old_call) == (
            "attn_result = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=a, b=b)"
        )
        # The independently reviewed source M32 path returns only after successful
        # fused normalization; its else is exactly the existing attention call.
        replacement = [
            line[4:]
            for line in text.splitlines(keepends=True)[
                old_call.lineno - 1 : old_call.end_lineno
            ]
        ]
        ranges.append((route.lineno - 1, route.end_lineno, replacement))
    lines = text.splitlines(keepends=True)
    for start, end, replacement in sorted(ranges, reverse=True):
        lines[start:end] = replacement
    text = "".join(lines)

    # Use the repository's token digest, independent of Python's evolving AST
    # node fields. Remove exactly the reviewed early-route/target-view seams; retain all stock text.
    class SourceText:
        def read_text(self):
            return text

    assert (
        contract.mirror_digest(SourceText(), "Qwen3_5GatedDeltaNet.forward")
        == REVIEWED[0]
    )
