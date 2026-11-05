# SPDX-License-Identifier: Apache-2.0
"""FN MTP segment kernels; opt-in, with native paths for unsupported shapes."""

import os


def enabled() -> bool:
    return os.environ.get("SGLANG_Q38_MK_MTP", "0") == "1"
