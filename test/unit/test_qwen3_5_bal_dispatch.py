# SPDX-License-Identifier: Apache-2.0
"""BAL must affect only wide-engine work assignment, including its row cap."""
import pytest

from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm


@pytest.mark.parametrize("m", [56, 112, 126, 128, 129, 133, 140, 168, 224, 256, 280, 512])
def test_bal_preserves_control_dispatch(monkeypatch, m):
    rows = mkm.engine_rows(m)
    monkeypatch.setattr(mkm, "ENGINE_WIDE_BAL_MAX_ROWS", 0)
    monkeypatch.setattr(mkm, "ENGINE_WIDE_BAL", False)
    off = mkm.engine_options(rows)
    plain_off = mkm.onepass_plain(rows, m)
    monkeypatch.setattr(mkm, "ENGINE_WIDE_BAL", True)
    on = mkm.engine_options(rows)
    assert on == off + (("-DBAL_SPLIT=1",) if m > 128 else ())
    assert mkm.onepass_plain(rows, m) == plain_off
    assert mkm.engine_options(rows, plain=True) == tuple(x for x in off if x != "-DONEPASS=1")


@pytest.mark.parametrize("m,plain", [(129, False), (210, False), (211, True), (280, True)])
def test_bal_row_cap_selects_stock_build(monkeypatch, m, plain):
    monkeypatch.setattr(mkm, "ENGINE_WIDE_BAL", True)
    monkeypatch.setattr(mkm, "ENGINE_WIDE_BAL_MAX_ROWS", 210)
    rows = mkm.engine_rows(m)
    assert mkm.onepass_plain(rows, m) is plain
    assert ("-DBAL_SPLIT=1" in mkm.engine_options(rows, plain)) is not plain
