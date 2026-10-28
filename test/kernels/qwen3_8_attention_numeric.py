"""Load-only main4517 TRTLLM sparse QSA / plugin core / FP64 comparison."""
import argparse,ctypes,json,os,statistics,tarfile,tempfile,shutil
from pathlib import Path
from types import SimpleNamespace as NS
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
out=a.output or Path(os.environ['Q38CC_ARM_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
work=Path(tempfile.mkdtemp(prefix='qsa-fn-native-',dir='/tmp'))
with tarfile.open(a.archive) as t:
 for m in t.getmembers():
  if m.isfile():
   name=Path(m.name).name
   if name.endswith(('.cubin','.json','.pt')):(work/name).write_bytes(t.extractfile(m).read())
import torch
from cuda.bindings import driver as cu
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
import qwen3_8_attention_native_backend as native_source
torch.set_num_threads(1);torch.empty(1,device='cuda')
def checked(result):
 if int(result[0]):raise RuntimeError(str(result))
 return result[1:]

class Cubin:
 def __init__(self,name):
  self.meta=json.loads((work/(name+'.json')).read_text())
  self.module,=checked(cu.cuModuleLoadData((work/(name+'.cubin')).read_bytes()))
  self.fn,=checked(cu.cuModuleGetFunction(self.module,self.meta['symbol'].encode()))
  self.regs,=checked(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,self.fn))
  if self.meta['shared']>48*1024:
   checked(cu.cuFuncSetAttribute(self.fn,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,self.meta['shared']))
 def __call__(self,grid,tensors):
  values=[ctypes.c_void_p(t.data_ptr()) for t in tensors]
  values += [ctypes.c_void_p(0) for _ in range(self.meta['params']-len(values))]
  assert len(values)==self.meta['params']
  ptrs=(ctypes.c_void_p*len(values))(*(ctypes.addressof(v) for v in values))
  cfg=cu.CUlaunchConfig();cfg.gridDimX,cfg.gridDimY,cfg.gridDimZ=grid
  cfg.blockDimX,cfg.blockDimY,cfg.blockDimZ=self.meta['warps']*32,1,1
  cfg.sharedMemBytes=self.meta['shared'];cfg.hStream=cu.CUstream(torch.cuda.current_stream().cuda_stream)
  attrs=[]
  if self.meta['options'].get('launch_pdl'):
   attr=cu.CUlaunchAttribute()
   attr.id=cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
   attr.value.programmaticStreamSerializationAllowed=1
   attrs=[attr]
  cfg.attrs,cfg.numAttrs=attrs,len(attrs)
  checked(cu.cuLaunchKernelEx(cfg,self.fn,ctypes.addressof(ptrs),0))


def bench(fn):
 for _ in range(3):fn()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  for _ in range(32):fn()
 for _ in range(2):graph.replay()
 samples=[]
 for _ in range(5):
  b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  b.record()
  for _ in range(4):graph.replay()
  e.record();e.synchronize();samples.append(b.elapsed_time(e)*1000/128)
 return dict(median_us=statistics.median(samples),samples_us=samples)

kernels={n:Cubin(n) for n in ['partial','merge','store','native-pack-4517','native-gate-4517']}
reports=[]
for cpu in torch.load(work/f'rank{a.rank}-cases.pt',map_location='cpu',weights_only=False):
 x={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cpu.items() if not k.startswith('producer_')}
 vr=torch.empty((4,3584),dtype=torch.bfloat16,device='cuda');vr[:,:256].copy_(x['v_new'].reshape(4,256));vn=vr[:,:256].view(4,1,256)
 pk,pv=x['k_pool'].clone(),x['v_pool'].clone();ck,cv=pk.clone(),pv.clone()
 parts=torch.empty((4,1,33,16,256),dtype=torch.float32,device='cuda');mx=torch.empty((4,1,33,16),dtype=torch.float32,device='cuda');den=torch.empty_like(mx)
 result=torch.empty((4,6,256),dtype=torch.bfloat16,device='cuda');native_gated=torch.empty_like(result)
 meta=NS(sequence_lengths=x['lengths'],row_req_pool_indices=x['row_requests'],is_cuda_graph=True,fa2_valid_counts=torch.empty(4,dtype=torch.int32,device='cuda'),indexer_metadata=NS(decode_logical_positions=x['query_positions']))
 backend=native_source.NativeQSA();backend.compress_ratio=4;backend.qsa_profile=NS(block_topk=512);backend._cuda_graph_max_tokens=4
 backend._fa2_scratch={};backend._trtllm_sparse_tables={};backend._trtllm_workspace=None;backend._trtllm_counters={};backend.req_to_token_pool=NS(req_to_token=x['req_to_token'])
 def pack(kc,vc,kn,nv,loc,mapping,req,idx,lengths,counts,ok,ov,width,**kwargs):
  assert width==4 and kwargs['compress_ratio']==4 and kwargs['visible_page_size']==64
  kernels['native-pack-4517']((1064,1,1),[kc,vc,kn,nv,loc,mapping,req,idx,lengths,kwargs['query_positions'],counts,ok,ov])
 native_source.fused_kv_prepare=pack
 def native():
  pre=backend._forward_trtllm_sparse(x['q'],pk,pv,NS(scaling=float(cpu['scale'])),NS(req_pool_indices=x['row_requests']),meta,x['indices'],trtllm_batch_decode_with_kv_cache,new_kv=(x['k_new'],vn,x['loc'],4))
  kernels['native-gate-4517']((4,2,1),[native_gated,pre,x['gate']])
  return pre
 def candidate():
  kernels['store']((4,1,1),[ck,cv,x['k_new'],vn,x['loc']])
  kernels['partial']((4,33,1),[x['q'],ck,cv,x['k_new'],vn,x['loc'],x['req_to_token'],x['row_requests'],x['indices'],x['lengths'],x['query_positions'],parts,mx,den])
  kernels['merge']((4,6,4),[parts,mx,den,result,x['gate']])
 native_runs=[];candidate_runs=[]
 for _ in range(5):native();native_runs.append(native_gated.clone())
 for _ in range(5):candidate();candidate_runs.append(result.clone())
 torch.cuda.synchronize()
 refs=[]
 for row in range(4):
  length=int(cpu['lengths'][row]);qpos=int(cpu['query_positions'][row]);req=int(cpu['row_requests'][row])
  ids=cpu['indices'][row].long();pos=(ids[:,None]*4+torch.arange(4)).flatten()[:min(length//4,512)*4]
  pos=torch.cat([pos,torch.arange((qpos+1)//4*4,min(qpos+1,length))]);pos=pos[(pos>=0)&(pos<length)]
  slots=cpu['req_to_token'][req,pos].long().cuda();keys=pk[slots,0].double();vals=pv[slots,0].double()
  first=length-1-row%4;rel=pos-first;fresh=rel>=0
  if fresh.any():
   fr=(row//4*4+rel[fresh]).long().cuda();keys[fresh.cuda()]=x['k_new'][fr,0].double();vals[fresh.cuda()]=vn[fr,0].double()
  scores=x['q'][row].double()@keys.T*float(cpu['scale'])
  refs.append((torch.softmax(scores,-1)@vals)*torch.sigmoid(x['gate'][row].double()))
 ref=torch.stack(refs)
 def errors(t):
  diff=t.double()-ref
  return dict(max_abs=float(diff.abs().max()),rel_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref)))
 ne=[errors(t) for t in native_runs];ce=[errors(t) for t in candidate_runs]
 ratio=dict(rel_l2=max(z['rel_l2'] for z in ce)/max(z['rel_l2'] for z in ne),max_abs=max(z['max_abs'] for z in ce)/max(z['max_abs'] for z in ne))
 aa=[float((native_runs[i].float()-native_runs[j].float()).abs().max()) for i in range(5) for j in range(i)]
 report=dict(rank=a.rank,name=cpu['name'],scope=cpu['scope'],reference='FP64 attention+sigmoid on identical BF16 operands; native exact4517 backend body + nativeCPU pack/gate cubins',native_errors=ne,candidate_errors=ce,error_ratios=ratio,standard_pass=ratio['rel_l2']<=1.05 and ratio['max_abs']<=2.,native_aa_max_abs=max(aa),cache_k=bool(torch.equal(pk,ck)),cache_v=bool(torch.equal(pv,cv)),timing=dict(native=bench(native),core=bench(candidate)))
 reports.append(report);print('NUMERIC_CASE',json.dumps(report),flush=True)
(out/'report.json').write_text(json.dumps(dict(reports=reports,all_standard=all(r['standard_pass'] for r in reports),scope='kernel qualification; not service admission',compile_allowed=False),indent=2)+'\n')
shutil.rmtree(work)
# Exceeding the nominal bound is a measured qualification result, not an
# infrastructure failure. The lead explicitly requested the actual ratios.
