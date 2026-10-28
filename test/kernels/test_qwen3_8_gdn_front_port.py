# SPDX-License-Identifier: Apache-2.0
"""The unchanged e5 BF16 front port against its frozen native source.

The frozen parent is supplied as an explicit test asset, never imported from
an in-tree serving checkout. This checks the port, not the new N32 algorithm.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sgl_nvidia_mega.qwen3_8.gdn.config import shapes


@pytest.mark.parametrize('requests',[1,2,3])
@pytest.mark.parametrize('has_bias',[False,True])
def test_e5_front_port_is_bitwise(requests,has_bias):
    import cutlass
    from cutlass.cute import experimental as ext
    from sgl_nvidia_mega.qwen3_8.sm103.gdn.front import run
    from sgl_nvidia_mega.qwen3_8.sm103.gdn import front
    from cuda.bindings import driver as cu
    spec=importlib.util.spec_from_file_location('frozen_e5_front',os.environ['GDN_NATIVE_FRONT'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    shape=shapes(SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560,
        linear_num_key_heads=16,linear_num_value_heads=48,
        linear_key_head_dim=128,linear_value_head_dim=128)),4,4)
    M=requests*shape.tokens_per_request
    torch.manual_seed(123+int(os.environ.get('GDN_TEST_RANK','0'))*100+requests+has_bias)
    def rand(*dims):return (torch.randn(*dims,device='cuda')*.1).to(torch.bfloat16)
    x=rand(M,shape.hidden);weight=rand(shape.packed,shape.hidden)
    cw=rand(shape.channels,4);bias=rand(shape.channels) if has_bias else None
    original=rand(8,shape.channels,3)
    state,cstate=original.clone(),original.clone()
    def window():
        store=torch.full((8,shape.channels,6),-9.,device='cuda',dtype=torch.bfloat16)
        return store.as_strided((8,4,shape.channels,3),(shape.channels*6,1,6,1))
    win,cwin=window(),window()
    index_dtype=torch.int64 if has_bias else torch.int32
    slots=torch.tensor([2,5,1][:requests],device='cuda',dtype=index_dtype)
    inter=torch.tensor([6,3,0][:requests],device='cuda',dtype=index_dtype)
    packed=torch.empty(M,shape.packed,device='cuda',dtype=torch.bfloat16)
    conv=torch.empty(M,shape.channels,device='cuda',dtype=torch.bfloat16)
    op=module.MkGdnFront(cta_m=64,cta_n=8 if M==4 else 16,cta_k=128,
        num_ab_stage=6,use_2cta=False,use_pdl=True,has_bias=False,out_dtype=cutlass.BFloat16)
    op.qkv_channels=shape.channels;op.verify_rows=M;op.conv_has_bias=has_bias;op.pad_slot=-1
    tensors=(weight.unsqueeze(-1),x.unsqueeze(-1),packed.t().unsqueeze(-1),
             packed.t().unsqueeze(-1),cw,cw[:,0] if bias is None else bias,
             state,slots,win,inter,conv)
    args=tuple(front._view(t,16 if i<4 else 2) for i,t in enumerate(tensors))
    stream=cu.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled=ext.compile(op,*args,stream,options='--gpu-arch sm_103a')
    compiled(*args,stream)
    got,cgot=run(x,weight,cw,bias,cstate,slots,cwin,inter,shape=shape)
    torch.cuda.synchronize()
    assert torch.equal(got,packed)
    assert torch.equal(cgot,conv)
    assert torch.equal(state,cstate)
    assert torch.equal(win,cwin)
    # Change both dynamic inputs and state, then replay an actual captured graph.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream=cu.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(*args,capture_stream)
        result=run(x,weight,cw,bias,cstate,slots,cwin,inter,shape=shape)
    for _ in range(3):
        x.copy_(rand(*x.shape));newstate=rand(*original.shape)
        state.copy_(newstate);cstate.copy_(newstate)
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(result[0],packed) and torch.equal(result[1],conv)
        assert torch.equal(state,cstate) and torch.equal(win,cwin)
