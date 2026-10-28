"""Startup refusals and parameter extraction do not require a CUDA context."""
from types import SimpleNamespace

from sgl_nvidia_mega.qwen3_8.boundary.hop import Config, config_disable_reasons


def test_hc_hop_uses_the_models_text_configuration():
    cfg = Config.from_text_config(SimpleNamespace(
        hidden_size=2560, hc_count=4, hc_lowrank=320, rms_norm_eps=1e-6,
    ))
    assert cfg == Config(2560, 4, 320, 1e-6)
    assert config_disable_reasons(cfg, world=4, max_m=4) == []


def test_unsupported_hop_setups_report_reasons_without_asserting():
    reasons = config_disable_reasons(Config(2576, 5, 0, 0), world=8, max_m=0)
    assert len(reasons) == 5
    assert any("TP8" in reason for reason in reasons)
    assert any("five" in reason or "5" in reason for reason in reasons)
