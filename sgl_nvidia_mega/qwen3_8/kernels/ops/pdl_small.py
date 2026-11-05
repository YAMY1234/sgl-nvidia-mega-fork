# SPDX-License-Identifier: Apache-2.0
"""Opt-in Rubin PDL launch coverage; arithmetic and stream ownership are unchanged."""

import os

ENABLED = os.environ.get("SGLANG_Q38_PDL_SMALL", "0") == "1"
