# SPDX-License-Identifier: Apache-2.0
"""FN GDN numeric qualification against FP64, with native commit on both lanes.

This is a synthetic same-input reference, not a model-accuracy or speed gate.
The FP64 state never rounds at commit; native and MK both use the native compact
commit/track implementation and BF16 checkpoints. Compare the single verify and
continuous 2048-verify tracks separately, retaining standard-bound failures.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode, commit_gdn_replayssm_spec,
    commit_gdn_replayssm_circular,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sgl_nvidia_mega.qwen3_8.gdn.config import shapes
from sgl_nvidia_mega.qwen3_8.sm103.gdn.core import core, _kernel, _check


def errors(actual, reference):
    d = actual.double() - reference
    return {"rel_l2": (d.norm()/reference.norm().clamp_min(1e-300)).item(),
            "max_abs": d.abs().max().item()}


def compared(native, candidate, reference):
    n, c = errors(native, reference), errors(candidate, reference)
    ratios = {k: c[k]/n[k] if n[k] else (0.0 if c[k] == 0 else None)
              for k in n}
    return {"native_vs_fp64": n, "mk_vs_fp64": c, "mk_over_native": ratios,
            "standard_bound": {"rel_l2": c["rel_l2"] <= 1.05*n["rel_l2"],
                               "max_abs": c["max_abs"] <= 2.0*n["max_abs"]}}


def fp64_step(checkpoint, x, a, b, A_log, dt_bias, weight, z, accepted, shape):
    """Real delta rule on the exact supplied BF16/FP32 operands, in FP64.

    State layout [B,HV,V,K]; normalized Q/K are not BF16-rounded in the ideal
    reference. The accepted prefix alone becomes the next committed state.
    """
    B, T = x.shape[:2]
    hk, hv, K, V = shape.hk, shape.hv, shape.key_dim, shape.value_dim
    q, k, v = x.double().split((hk*K, hk*K, hv*V), -1)
    q = q.reshape(B,T,hk,K); k = k.reshape(B,T,hk,K)
    q = q * (q.square().sum(-1,keepdim=True)+1e-6).rsqrt() * K**-0.5
    k = k * (k.square().sum(-1,keepdim=True)+1e-6).rsqrt()
    q = q.repeat_interleave(hv//hk,dim=2)
    k = k.repeat_interleave(hv//hk,dim=2)
    v = v.reshape(B,T,hv,V)
    decay = (-A_log.double().exp()*torch.nn.functional.softplus(
        a.double()+dt_bias.double(),threshold=20)).exp()
    beta = b.double().sigmoid()
    h = checkpoint
    outputs = []
    selected = None
    for t in range(T):
        h = h*decay[:,t,:,None,None]
        delta = (v[:,t] - (h*k[:,t,:,None,:]).sum(-1))*beta[:,t,:,None]
        h = h + delta[:,:,:,None]*k[:,t,:,None,:]
        outputs.append((h*q[:,t,:,None,:]).sum(-1))
        if t+1 == accepted:
            selected = h
    o = torch.stack(outputs,dim=1)
    y = o*(o.square().mean(-1,keepdim=True)+1e-6).rsqrt()
    y = y*weight.double()*z.double().reshape(B,T,hv,V).sigmoid()
    return selected, o, y


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--requests',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,default=2048)
    args=parser.parse_args()
    assert args.steps >= 2048
    hf=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560,
        linear_num_key_heads=16,linear_num_value_heads=48,
        linear_key_head_dim=128,linear_value_head_dim=128))
    shape=shapes(hf,tokens_per_request=4,tp=4)
    B,T,HK,HV,K,V=args.requests,shape.tokens_per_request,shape.hk,shape.hv,shape.key_dim,shape.value_dim
    M,S=B*T,2*B
    torch.manual_seed(args.seed)
    device='cuda'
    def rand(*dims,scale=1,dtype=torch.bfloat16):
        return (torch.randn(*dims,device=device)*scale).to(dtype)
    x=rand(args.steps,B,T,shape.channels,scale=.3)
    ab=rand(args.steps,B,T,2*HV)
    z=rand(args.steps,M,HV,V)
    A=torch.linspace(-6,0,HV,device=device)
    dt=rand(HV,scale=.5)
    weight=rand(V,scale=.1)+1
    initial=rand(S,HV,V,K,scale=.02)
    idx=torch.arange(B,device=device,dtype=torch.int64)
    track=idx+B
    qloc=torch.arange(0,M+1,T,device=device,dtype=torch.int32)
    accepts=[torch.full((B,),a,device=device,dtype=torch.int32) for a in range(1,T+1)]
    track_steps=[a-1 for a in accepts]
    def lane():
        return dict(state=initial.clone(),
            d=torch.zeros(S,HV,16,V,device=device,dtype=torch.bfloat16),
            k=torch.zeros(S,HK,16,K,device=device,dtype=torch.bfloat16),
            g=torch.zeros(S,HV,16,device=device),
            dl=torch.zeros(S,HV,16,V,device=device,dtype=torch.bfloat16),
            kl=torch.zeros(S,HK,16,K,device=device,dtype=torch.bfloat16),
            wp=torch.zeros(S,device=device,dtype=torch.int32),
            base=torch.zeros(S,device=device,dtype=torch.int32),
            flush=torch.zeros(S,device=device,dtype=torch.int8),
            out=torch.empty(M,HV,V,device=device,dtype=torch.bfloat16),
            norm=torch.empty(M,HV*V,device=device,dtype=torch.bfloat16))
    native,mk=lane(),lane()
    ref=initial[:B].double()
    report={'tp':4,'requests':B,'rows':M,'tokens_per_request':T,
        'heads':[HK,HV],'seed':args.seed,'verify_steps':args.steps,
        'accepted_pattern':list(range(1,T+1)),
        'reference':'FP64 exact delta rule, no intermediate/checkpoint BF16 round; same BF16/FP32 inputs',
        'native':'main4517 Triton verify BV64/NW1 + native compact commit/track',
        'candidate':'e5 core source, NW4 full head + sigmoid norm; same native compact commit/track',
        'qualification':'MEASURED_ONLY; empirical exceptions need lead decision',
        'samples':[]}
    for t in range(args.steps):
        accept=t%T+1
        mark=t==0 or (t+1)%256==0
        xx=x[t].reshape(M,-1)
        a=ab[t,:,:,:HV].reshape(M,HV)
        b=ab[t,:,:,HV:].reshape(M,HV)
        q,k,v=xx.split((HK*K,HK*K,HV*V),-1)
        for candidate,l in ((False,native),(True,mk)):
            if candidate:
                core(xx,a,b,A,dt,l['state'],l['d'],l['k'],l['g'],l['dl'],l['kl'],
                     idx,idx,l['wp'],l['base'],l['flush'],l['out'],z=z[t],
                     norm_weight=weight,normalized=l['norm'],tokens_per_request=T)
            else:
                gdn_replayssm_spec_decode(q=q.view(M,HK,K),k=k.view(M,HK,K),v=v.view(M,HV,V),
                    a=a,b=b,A_log=A,dt_bias=dt,checkpoint_state=l['state'],
                    d_cache=l['d'],k_cache=l['k'],g_cache=l['g'],rawv_cache=l['dl'],
                    rawk_cache=l['kl'],beta_cache=None,out=l['out'],query_start_loc=qloc,
                    ssm_state_indices=idx,replay_indices=idx,write_pos=l['wp'],cache_base=l['base'],
                    is_flush=l['flush'],max_cache_len=16,max_spec_len=T,scale=K**-.5,
                    use_qk_l2norm_in_kernel=True,null_block_id=-1,launch_mode='verify',
                    block_v=64,num_warps=1)
                if mark:
                    l['norm'].copy_(rms_norm_gated(x=l['out'],weight=weight,bias=None,z=z[t],
                        eps=1e-6,norm_before_gate=True,is_rms_norm=True,activation='sigmoid').reshape(M,-1))
            commit_gdn_replayssm_spec(l['wp'],l['base'],l['flush'],accepts[accept-1],idx,
                                     16,T,null_block_id=-1)
            commit_gdn_replayssm_circular(checkpoint_state=l['state'].unsqueeze(0),
                d_cache=l['d'].unsqueeze(0),k_cache=l['k'].unsqueeze(0),g_cache=l['g'].unsqueeze(0),
                d_residual_cache=l['dl'].unsqueeze(0),k_residual_cache=l['kl'].unsqueeze(0),
                state_batch_indices=idx,replay_indices=idx,write_pos=l['wp'],cache_base=l['base'],
                is_flush=l['flush'],accept_lens=accepts[accept-1],mamba_track_indices=track if mark else None,
                mamba_steps_to_track=track_steps[accept-1] if mark else None,null_block_id=-1)
        ref,ro,ry=fp64_step(ref,x[t],a.reshape(B,T,HV),b.reshape(B,T,HV),A,dt,
                           weight,z[t],accept,shape)
        if mark:
            torch.cuda.synchronize()
            row={'verify_step':t+1,'accepted_tokens':(t//T)*(T*(T+1)//2)+(t%T+1)*(t%T+2)//2,
                 'raw_O':compared(native['out'],mk['out'],ro.reshape(M,HV,V)),
                 'gated_norm':compared(native['norm'],mk['norm'],ry.reshape(M,-1)),
                 'state_track':compared(native['state'][B:],mk['state'][B:],ref)}
            report['samples'].append(row)
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    from cuda.bindings import driver as cu
    _,fn=_kernel(0,HK,HV,T,True,True,False,128)
    report['resources']={}
    for name in ('NUM_REGS','SHARED_SIZE_BYTES','LOCAL_SIZE_BYTES','MAX_THREADS_PER_BLOCK'):
        report['resources'][name]=int(_check(cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute,
            'CU_FUNC_ATTRIBUTE_'+name),fn),name))
    report['completed']=True
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
