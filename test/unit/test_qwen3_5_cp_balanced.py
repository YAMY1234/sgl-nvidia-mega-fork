"""CPU checks for CP admission and the shipped KV-length allocation mirror.

No CUDA import is needed to validate the integer plan. Numerical equality between
different split boundaries is a separate gate, not implied by tile coverage.
"""

import ast
import pathlib
import unittest
from types import SimpleNamespace


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
        "_TWO_SLOT_MAX_BATCH",
        "_two_slot_pack_starts",
        "_two_slot_pack_T",
        "balanced_split_plan_two",
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


class TestBalancedCP(unittest.TestCase):
    def setUp(self):
        self.host = _host_plan()

    def test_verify_128_and_q1_64_admission(self):
        self.assertTrue(self.host["_DEFAULT_BALANCED"])
        self.assertEqual((self.host["MAX_BATCH"], self.host["MAX_BATCH_Q1"]), (128, 64))
        supported = self.host["q35_cp_fused_supported"]
        for batch in (1, 8, 32, 33, 64, 65, 96, 128):
            self.assertTrue(supported(7, 4, 1, batch, 152), batch)
            # Production exact-prefix calls fold verify rows into query heads.
            self.assertTrue(supported(1, 4, 7, batch, 152, max_batch=128), batch)
            self.assertEqual(supported(1, 4, 1, batch, 152), batch <= 64)
        self.assertFalse(supported(7, 4, 1, 129, 152))
        self.assertFalse(supported(7, 4, 1, 128, 128))
        self.assertFalse(supported(7, 4, 1, 64, 152, max_batch=32))

    def test_kv_gate_uses_the_greatest_batch_key_and_mean_length(self):
        path = pathlib.Path(__file__).resolve().parents[2] / (
            "sgl_nvidia_mega/qwen3_5/sm103/attention/q35_cp_lite.py"
        )
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "CpLite"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        policy = next(
            n.value
            for n in ast.walk(init)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "min_kv" for t in n.targets
            )
        )
        policy = ast.literal_eval(policy)
        self.assertEqual(policy, {2: 12288, 4: 10240, 16: 6144})
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_gate_thr"
        )
        scope = {}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            scope,
        )
        owner = SimpleNamespace(min_kv=policy, gate_max=False)
        for batch, mean in [
            (1, None),
            (2, 12288),
            (3, 12288),
            (4, 10240),
            (8, 10240),
            (15, 10240),
            (16, 6144),
            (64, 6144),
            (65, 6144),
            (128, 6144),
        ]:
            self.assertEqual(
                scope["_gate_thr"](owner, batch), None if mean is None else batch * mean
            )

    def test_two_slot_owned_tile_coverage(self):
        pool = (0, 1, 63, 64, 65, 512, 8192, 65536, 131072)
        for batch in (65, 96, 128):
            lengths = [pool[i % len(pool)] for i in range(batch)]
            for rank in range(4):
                for packed in (False, True):
                    _, ctas = self.host["balanced_split_plan_two"](
                        lengths, rank, 4, 152, pack=packed
                    )
                    self.assertEqual(len(ctas), 152)
                    self.assertTrue(all(len(slots) == 2 for slots in ctas))
                    for req, length in enumerate(lengths):
                        tiles = ((((length + 63) // 64) - rank + 3) // 4 + 1) // 2
                        pieces = sorted(
                            (
                                slot
                                for slots in ctas
                                for slot in slots
                                if slot is not None and slot[0] == req
                            ),
                            key=lambda slot: slot[1],
                        )
                        actual = [
                            t
                            for slot in pieces
                            for t in range(slot[4], slot[4] + slot[5])
                        ]
                        self.assertEqual(
                            actual, list(range(tiles)), (batch, rank, packed, req)
                        )
                        self.assertEqual(
                            [slot[1] for slot in pieces], list(range(len(pieces)))
                        )
                        self.assertTrue(all(slot[2] == len(pieces) for slot in pieces))

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
