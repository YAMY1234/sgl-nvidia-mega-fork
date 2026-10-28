# SPDX-License-Identifier: Apache-2.0
"""Fresh native FI comparison on externally prepared real FN rank fixtures.

The CPU phase supplies cubins, native.json and a fixture containing layer/router/
cases; the GPU phase must not compile. See qwen3_8/moe/README.md.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM103")
def test_route_mkm_matches_native_and_replays(tmp_path):
    required = ("FN_MOE_FIXTURE", "FN_PREPARED", "FN_SGLANG", "SGLANG_CACHE_DIR")
    if any(not os.environ.get(name) for name in required):
        pytest.skip(
            "requires CPU-prepared real FN fixture and cubins: " + ", ".join(required)
        )
    rank = os.environ.get("FN_MOE_RANK", "0")
    env = {**os.environ, "Q38CC_ARM_OUTPUT": str(tmp_path)}
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_qwen3_8_moe_probe.py")),
            "--rank",
            rank,
        ],
        env=env,
        check=True,
    )
    result = json.loads((tmp_path / "parity-micro.json").read_text())
    assert result["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM103")
def test_shared_gate_up_holds_to_fp64(tmp_path):
    required = ("FN_MOE_FIXTURE", "FN_SHARED_FIXTURE", "FN_SGLANG", "SGLANG_CACHE_DIR")
    if any(not os.environ.get(name) for name in required):
        pytest.skip("requires CPU-prepared real BF16 shared fixture and cubins")
    env = {**os.environ, "Q38CC_ARM_OUTPUT": str(tmp_path)}
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_qwen3_8_shared_probe.py"))],
        env=env,
        check=True,
    )
    assert json.loads((tmp_path / "shared-numeric.json").read_text())["passed"]
