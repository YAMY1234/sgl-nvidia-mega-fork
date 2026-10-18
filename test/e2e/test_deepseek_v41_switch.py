# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""--enable-nvidia-megakernel fails at argument resolution, before the weights load,
when it cannot run: without the plugin it names it, and with the plugin it lists every
reason the setup is not supported."""

import os
import subprocess
import sys
import unittest

from deepseek_v41_server import ARGS, ENV, MODEL, URL
from sglang.test.test_utils import CustomTestCase

from sgl_nvidia_mega.plugin import FLAG_CLI

# Argument resolution reads only the model's config; the weights take minutes
RESOLVE_TIMEOUT_S = 300
REFUSAL = f"{FLAG_CLI} cannot run this setup:"


def _launch(extra_args=(), extra_env=None) -> tuple[int, str]:
    """A server launch that must stop at argument resolution: its exit code and stderr."""
    proc = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL]
        + ARGS
        + [FLAG_CLI, *extra_args],
        env={**os.environ, **ENV, **(extra_env or {})},
        capture_output=True,
        text=True,
        timeout=RESOLVE_TIMEOUT_S,
    )
    return proc.returncode, proc.stderr


@unittest.skipIf(URL, "launches its own servers")
class TestFlagFailsAtStartup(CustomTestCase):
    def test_without_the_plugin(self):
        # an SGLANG_PLUGINS list without it keeps the installed plugin from loading
        code, err = _launch(extra_env={"SGLANG_PLUGINS": "another_plugin"})
        self.assertNotEqual(code, 0)
        self.assertIn(REFUSAL, err)
        self.assertIn("does not list the 'nvidia_megakernel' plugin", err)

    def test_an_unsupported_setup_lists_every_reason(self):
        # the last --tp wins; a 6 s watchdog leaves the cross-GPU waits under 5 s
        code, err = _launch(["--tp", "2", "--watchdog-timeout", "6"])
        self.assertNotEqual(code, 0)
        self.assertIn(REFUSAL, err)
        self.assertIn("tp_size 2 is not one of ['4']", err)
        self.assertIn("--watchdog-timeout 6 s", err)


if __name__ == "__main__":
    unittest.main()
