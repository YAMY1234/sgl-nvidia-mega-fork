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


def fake_model():
    norm=SimpleNamespace(activation="sigmoid",group_size=None,norm_before_gate=True)
    out=SimpleNamespace(reduce_results=False,input_is_parallel=True,bias=None)
    import torch
    gdn=SimpleNamespace(norm=norm,out_proj=out,attn_tp_size=4,
                        _fused_in_proj_weight=torch.empty(4120,2560,dtype=torch.bfloat16),
                        attn=SimpleNamespace(layer_id=3))
    return SimpleNamespace(config=hf(),model=SimpleNamespace(
        layers=[SimpleNamespace(linear_attn=gdn)]))


def test_mk_has_separate_numeric_claim_and_owns_pdl():
    from sgl_nvidia_mega.qwen3_8 import gdn, runtime, plan
    with patch.object(runtime,"SLOTS",frozenset({"gdn_mk"})):
        bound=gdn.build(fake_model(),SimpleNamespace(tokens_per_request=4))
    assert bound.gdn_mk and bound.gdn_pdl and not bound.gdn_verify
    held=plan.slots(4,plan.Facts(gdn_mk=True,gdn_pdl=True))
    assert held.gdn_mk and held.gdn_pdl and not held.gdn_verify


def test_unqualified_mk_row_count_has_a_reason():
    from sgl_nvidia_mega.qwen3_8 import gdn, runtime
    with patch.object(runtime,"SLOTS",frozenset({"gdn_mk"})):
        bound=gdn.build(fake_model(),SimpleNamespace(tokens_per_request=7))
    assert "four verify rows" in bound.reason


@pytest.mark.parametrize('numeric',[False,True])
def test_gdn_and_moe_build_and_plan_together(monkeypatch,numeric):
    from sgl_nvidia_mega.qwen3_5.runtime import parity
    from sgl_nvidia_mega.qwen3_8 import runtime
    from sgl_nvidia_mega.qwen3_8.runtime import executor
    from sgl_nvidia_mega.qwen3_8.moe import route
    from sglang.srt import runtime_context
    names={'gdn_mk'} if numeric else {'gdn_verify','gdn_pdl'}
    monkeypatch.setattr(runtime,'SLOTS',frozenset(names|{'route','mkm','route_shared'}))
    monkeypatch.setattr(parity,'ENABLED',False)
    monkeypatch.setattr(runtime_context,'get_parallel',lambda:SimpleNamespace(tp_size=4))
    seen={}
    def moe_build(model,max_rows,**kwargs):
        seen.update(max_rows=max_rows,**kwargs)
        return route.Bound(route=True,mkm={3:object()},shared=True)
    monkeypatch.setattr(route,'build',moe_build)
    # This is a wiring/plan test: GPU resource construction/execution has its
    # independent primitive and service gates, not simulated CPU success.
    monkeypatch.setattr(executor,'Qwen38Verify',lambda model,slots,**kwargs:
                        SimpleNamespace(slots=slots,**kwargs))
    model=fake_model()
    ctx=SimpleNamespace(widths=(4,8,12),tokens_per_request=4,deadline_s=2.,binding=object())
    built=runtime.build(model,ctx,hf_config=model.config)
    assert built.widths==(4,) and set(built.bound)=={'gdn','moe','attention'}
    held=built.slots[4]
    assert held.gdn_mk==numeric and held.gdn_verify!=numeric and held.gdn_pdl
    assert held.route and held.mkm and held.route_shared
    assert seen==dict(max_rows=4,fused=True,shared=True,hf_config=model.config,tp_size=4)


def test_stopped_front_slot_is_rejected():
    from sgl_nvidia_mega.qwen3_8 import runtime
    with patch.object(runtime,'SLOTS',frozenset({'gdn_front'})):
        assert any("unknown FN slot 'gdn_front'" in reason
                   for reason in runtime.slot_reasons())
