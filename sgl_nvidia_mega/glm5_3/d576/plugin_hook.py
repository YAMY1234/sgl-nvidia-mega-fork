"""Opt-in override of D576's PrimTS module (SGL_MEGA_D576_PRIMTS=1).

SGLang's D576 in-place NVFP4 decode loads PrimTS through
``sglang.srt.mem_cache.dsa_trtrow_d._primts_mla_decode``; this AROUND hook returns this
package's PrimTS (SM107 table converter, early conversion, two-slot raw ring) instead.
"""
from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)
ENV = "SGL_MEGA_D576_PRIMTS"
TARGET = "sglang.srt.mem_cache.dsa_trtrow_d._primts_mla_decode"
MODULE = "sgl_nvidia_mega.glm5_3.d576.prims_ts.mla_decode"
_logged = False


def register() -> bool:
    if os.environ.get(ENV) != "1":
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        global _logged
        module = importlib.import_module(MODULE)
        if not _logged:
            _logged = True
            logger.info("SGL_MEGA_D576_PRIMTS engaged module=%s file=%s", MODULE, module.__file__)
        return module

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    return True
