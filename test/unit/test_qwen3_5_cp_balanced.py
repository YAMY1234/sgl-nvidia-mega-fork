"""CPU checks for CP admission and the shipped KV-length allocation mirror.

No CUDA import is needed to validate the integer plan. Numerical equality between
different split boundaries is a separate gate, not implied by tile coverage.
"""

import ast
import pathlib
import unittest


def _host_plan():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "sgl_nvidia_mega/qwen3_5/sm103/attention/q35_cp_fused.py"
    )
    names = {
        "PAGE",
        "HEAD_DIM",
        "MAX_ROWS",
        "_QUAD_PLAN_MAX_BATCH",
        "_WIDE_PLAN_MAX_BATCH",
        "_LANE_PLAN_MAX_BATCH",
        "_DEFAULT_BALANCED",
        "MAX_BATCH",
        "MAX_BATCH_Q1",
        "_MAX_COMB_ROWS",
        "_SM_RESERVE",
        "cp_partial_geometry",
        "balanced_split_plan",
        "_num_splits",
        "_num_ctas",
        "q35_cp_fused_supported",
    }
    selected = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            selected.append(node)
    namespace = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def _split_mode(batch, override=None):
    """Evaluate the actual dispatch expression without importing CUDA modules."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "sgl_nvidia_mega/qwen3_5/sm103/attention/q35_cp_fused.py"
    )
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "q35_cp_fused_partial"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "bal" for t in node.targets)
    )
    namespace = dict(_host_plan(), b=batch, balanced=override)
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["bal"]


class TestBalancedCP(unittest.TestCase):
    def setUp(self):
        self.host = _host_plan()

    def test_default_policy_and_64_request_admission(self):
        self.assertTrue(self.host["_DEFAULT_BALANCED"])
        self.assertEqual((self.host["MAX_BATCH"], self.host["MAX_BATCH_Q1"]), (64, 64))
        supported = self.host["q35_cp_fused_supported"]
        for q_len in (1, 7):
            for batch in (1, 8, 32, 33, 64):
                self.assertTrue(supported(q_len, 4, 1, batch, 152), (q_len, batch))
            self.assertFalse(supported(q_len, 4, 1, 65, 152))
        self.assertFalse(supported(7, 4, 1, 64, 64))
        self.assertFalse(supported(7, 4, 1, 64, 152, max_batch=32))

    def test_small_batches_keep_original_split_mode(self):
        for batch in range(1, 33):
            self.assertFalse(_split_mode(batch), batch)
            self.assertEqual(self.host["_num_splits"](batch, 152), 152 // batch)

    def test_wide_batches_use_balanced_split_mode(self):
        for batch in range(33, 65):
            self.assertTrue(_split_mode(batch), batch)

    def test_explicit_split_mode_override_is_preserved(self):
        for batch in (1, 2, 4, 8, 16, 32, 33, 64):
            self.assertTrue(_split_mode(batch, True), batch)
            self.assertFalse(_split_mode(batch, False), batch)

    def test_all_owned_tiles_appear_once_in_request_order(self):
        plan = self.host["balanced_split_plan"]
        lengths_pool = (0, 1, 63, 64, 65, 512, 8192, 65536, 131072)
        for batch in (1, 2, 8, 16, 32, 33, 64):
            lengths = [lengths_pool[i % len(lengths_pool)] for i in range(batch)]
            for rank in range(4):
                _, counts, assignments = plan(lengths, rank, 4, 152)
                self.assertEqual(sum(counts), 152)
                self.assertEqual(len(assignments), 152)
                for request, length in enumerate(lengths):
                    tiles = ((((length + 63) // 64) - rank + 3) // 4 + 1) // 2
                    actual = [
                        tile
                        for r, _, start, n in assignments
                        if r == request
                        for tile in range(start, start + n)
                    ]
                    self.assertEqual(actual, list(range(tiles)), (batch, rank, request))

    def test_ragged_batch_reduces_longest_cta_tile_range(self):
        lengths = [65536, 8192, 4096, 2048, 1024, 512, 256, 256]
        _, _, assignments = self.host["balanced_split_plan"](lengths, 0, 4, 152)
        equal_splits = self.host["_num_splits"](len(lengths), 152)
        longest_tiles = (((lengths[0] + 63) // 64 + 3) // 4 + 1) // 2
        equal_longest = (longest_tiles + equal_splits - 1) // equal_splits
        self.assertLess(max(n for _, _, _, n in assignments), equal_longest)


if __name__ == "__main__":
    unittest.main()
