# SPDX-License-Identifier: Apache-2.0
"""FN GDN dimensions from HF text_config and the build's verify width."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Shapes:
    hidden: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    tp: int
    tokens_per_request: int

    @property
    def hk(self):
        return self.key_heads // self.tp

    @property
    def hv(self):
        return self.value_heads // self.tp

    @property
    def channels(self):
        return 2 * self.hk * self.key_dim + self.hv * self.value_dim

    @property
    def packed(self):
        return self.channels + self.hv * self.value_dim + 2 * self.hv


def config_disable_reasons(hf_config, tokens_per_request, tp):
    text = getattr(hf_config, "text_config", hf_config)
    reasons = []
    for name in ("hidden_size", "linear_num_key_heads", "linear_num_value_heads",
                 "linear_key_head_dim", "linear_value_head_dim"):
        if not isinstance(getattr(text, name, None), int) or getattr(text, name) <= 0:
            reasons.append(f"GDN: missing/invalid text_config.{name}")
    if reasons:
        return reasons
    if tp <= 0 or text.linear_num_key_heads % tp or text.linear_num_value_heads % tp:
        reasons.append(f"GDN: head counts are not divisible by TP{tp}")
    if text.linear_num_value_heads % text.linear_num_key_heads:
        reasons.append("GDN: value heads must be a multiple of key heads")
    if tokens_per_request is not None and not 1 <= tokens_per_request <= 8:
        reasons.append(f"GDN: unsupported {tokens_per_request} verify rows/request (1–8)")
    if text.linear_key_head_dim != 128 or text.linear_value_head_dim != 128:
        reasons.append("GDN: only 128-dimensional K/V heads are qualified")
    return reasons


def shapes(hf_config, tokens_per_request, tp):
    reasons = config_disable_reasons(hf_config, tokens_per_request, tp)
    if reasons:
        raise ValueError("; ".join(reasons))
    text = getattr(hf_config, "text_config", hf_config)
    return Shapes(text.hidden_size, text.linear_num_key_heads,
                  text.linear_num_value_heads, text.linear_key_head_dim,
                  text.linear_value_head_dim, tp, tokens_per_request)
