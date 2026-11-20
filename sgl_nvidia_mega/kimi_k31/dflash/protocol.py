# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Which draft ``K31DFlashRuntime`` serves.

The runtime serves SGLang's built draft only if it is one of the process's draft
chains (``DraftLayersOnChain`` registers itself at build); any other draft offers
nothing, so no phase is served and no collective runs for it.
"""

from __future__ import annotations

from typing import Any


class _NoDraftChain:
    """Stands in for the draft when it is not on the megakernel chain: offers nothing."""

    draft_aux_rows: tuple[int, ...] = ()
    draft_head_requests: tuple[int, ...] = ()
    draft_accept_requests: tuple[int, ...] = ()
    draft_embed_requests: tuple[int, ...] = ()


NO_DRAFT_CHAIN = _NoDraftChain()
# The process's draft chains (draft/serving.py DraftLayersOnChain registers itself at
# build): prepare serves a draft only if SGLang's built draft is one of them.
DRAFT_CHAINS: list = []


def offered(built: Any, sizes: tuple[int, ...]) -> Any:
    """``built`` when it offers a phase for some sizes, else None."""
    return built if sizes else None
