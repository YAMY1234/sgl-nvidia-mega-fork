from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sgl_nvidia_mega.qwen3_8.gdn.config import config_disable_reasons, shapes
from sgl_nvidia_mega.qwen3_8.gdn import verify


def hf():
    return SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560,
        linear_num_key_heads=16,linear_num_value_heads=48,
        linear_key_head_dim=128,linear_value_head_dim=128))


@pytest.mark.parametrize("tp,hk,hv",[(1,16,48),(4,4,12)])
def test_dimensions_are_read_from_text_config(tp,hk,hv):
    shape=shapes(hf(),4,tp)
    assert (shape.hk,shape.hv,shape.tokens_per_request)==(hk,hv,4)
    assert shape.packed==2*hk*128+2*hv*128+2*hv


def test_unsupported_configuration_returns_reasons():
    cfg=hf();cfg.text_config.linear_value_head_dim=64
    assert any("128-dimensional" in x for x in config_disable_reasons(cfg,4,4))
    assert any("verify rows" in x for x in config_disable_reasons(hf(),9,4))


def test_generic_view_is_not_an_eagle_view():
    cache=object();pool=object();metadata=object()
    impl=SimpleNamespace(forward_metadata=metadata,state_pool=pool,
                         state_layer_cache=lambda layer: cache)
    st=SimpleNamespace(view=SimpleNamespace(metadata=None),binding=object())
    with patch.object(verify,"backend",return_value=impl):
        assert verify.context(SimpleNamespace(layer_id=3),st)==(impl,metadata,pool,cache)
