# SPDX-License-Identifier: Apache-2.0
"""CPU configuration tests: FN draft front is a primitive, not an enabled entry."""
from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.qwen3_8.draft.front import build_front, config_disable_reasons


def config(**changes):
    values = dict(hidden_size=2560, num_attention_heads=24, num_key_value_heads=2,
                  head_dim=256, partial_rotary_factor=0.25, rms_norm_eps=1e-6)
    values.update(changes)
    return SimpleNamespace(**values)


def test_fn_tp4_reads_text_config():
    front, why = build_front(SimpleNamespace(text_config=config()), 4)
    assert not why
    s = front.shape
    assert (s.hidden_size, s.q_heads, s.kv_heads, s.head_dim, s.rotary_dim) == (2560, 6, 1, 256, 64)


def test_sizes_are_not_fixed_to_fn():
    c = config()
    c.hidden_size, c.num_attention_heads, c.head_dim, c.partial_rotary_factor = 3072, 12, 128, 0.5
    front, why = build_front(c, 4)
    assert not why
    assert (front.shape.q_heads, front.shape.head_dim, front.shape.rotary_dim) == (3, 128, 64)


@pytest.mark.parametrize('field,value,word', [
    ('num_attention_heads', 25, 'attention_heads'),
    ('num_key_value_heads', 3, 'KV heads'),
    ('head_dim', 2048, 'head_dim'),
    ('partial_rotary_factor', 0, 'partial_rotary_factor'),
    ('partial_rotary_factor', 65/256, 'rotary dimension'),
    ('rms_norm_eps', 0, 'rms_norm_eps'),
    ('hidden_size', -1, 'hidden_size'),
])
def test_refusal_names_the_field(field, value, word):
    c = config();setattr(c, field, value)
    front, reasons = build_front(c, 4)
    assert front is None and any(word in r for r in reasons)


def test_bad_dtype_is_a_reason_not_a_kernel_assert():
    assert 'bfloat16' in ';'.join(config_disable_reasons(config(), 4, 'float16'))
