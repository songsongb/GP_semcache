"""Conditional native eager OPTAttention replay. Never executes model.forward()."""
import inspect
import math
import struct
import os
import time
from pathlib import Path
from unittest.mock import patch
from .communication import require,digest_bytes,digest
from .r3a_artifacts import write,tensor_bytes,tensor_from_bytes,full_base_fingerprint,validate_prepared,versions
from .r5_i5_evidence import seal
from .r5_i5r2_tensors import Evidence,slice_axis,DTYPES
from .r5_i5r2b_capture import backend,operation_mode
from .r5_i5r2bs1_plan import CORE
from .r5_i5r2bs1_runtime import Store,layout,parameters,apply_recorded_backend,reconstruct
from .r5_i5r2c3 import load,history,fresh,BOUNDARIES,compare_aligned,verdict


def schedule():
 return [dict(warmup=w,repetition=i,label=k) for w,n in ((True,2),(False,6)) for i in range(n) for k in (('S','F') if i%2==0 else ('F','S'))]


def validate_repetitions(rows):
 expected=[s for s in schedule() if not s['warmup']]
 require([{k:r[k] for k in ('warmup','repetition','label')} for r in rows]==expected,'incomplete/duplicate/reordered repetitions')
 require(all(set(r['boundaries'])==set(BOUNDARIES)|{'output_projection_input'} for r in rows),'missing attention boundary')


def canonical_boundary_shape(shape,role,heads,n,d):
 """Only a unit batch dimension may be removed; no head/token flattening."""
 shape=list(shape)
 if role in ('scores','masked_scores','probabilities','pv'):
  wanted=[heads,n,d if role=='pv' else n]
  require(shape==wanted or shape==[1]+wanted,'native attention rank/batch/head/query/key shape differs')
  return wanted
 require(role in ('output_projection_input','output_projection','module_output') and shape==[1,n,heads*d],'native output projection shape differs')
 return shape


def canonical_boundary(t,role,heads,n,d):
 shape=canonical_boundary_shape(t.shape,role,heads,n,d)
 return t.squeeze(0) if t.ndim==4 else t


def operand_prefix(data,shape,dtype,axis,*,also_keys=False):
 prefix,ps=slice_axis(data,shape,axis,0,64,DTYPES[dtype][1])
 if also_keys:
  require(axis==len(shape)-2,'probability query axis differs')
  prefix,ps=slice_axis(prefix,ps,len(shape)-1,0,64,DTYPES[dtype][1])
 return prefix,ps


def head_prefix_bytes(raw,n,heads,d,*,transpose=False,width=2):
 from .r4b2_analysis import raw_head_bytes
 head=raw_head_bytes(raw,[1,n,heads*d],heads,width)
 prefix,_=slice_axis(head,[1,heads,n,d],2,0,64,width)
 if transpose:
  return b''.join(prefix[((h*64+t)*d+j)*width:((h*64+t)*d+j+1)*width] for h in range(heads) for j in range(d) for t in range(64))
 return prefix


class AttentionOperations:
 """Observe eager QK -> mask -> softmax[/cast/dropout] -> PV in native order."""
 def __init__(self,attention,operands,observe):
  self.a=attention;self.x=operands;self.observe=observe;self.heads=attention.num_heads;self.d=attention.head_dim
  self.n=operands['q'].shape[1];self.phase='START';self.prob=[];self.q_lineage=[operands['q']];self.score_values=[];self.mask_values=[]
  self.trace=dict(operations=[],softmax=None,dropout=dict(observed=False),post_qk_scaling=[],probability_transitions=[],mask_postprocessing=[])
  require(not attention.training and all(list(operands[r].shape)==[1,self.n,self.heads*self.d] for r in 'qkv') and list(operands['mask'].shape)==[1,1,self.n,self.n],'native replay input/mask/batch shape differs')
 def head(self,t):return t.view(1,self.n,self.heads,self.d).transpose(1,2)
 def equal_head(self,t,raw,transpose=False):
  import torch
  ref=self.head(raw)
  if transpose:ref=ref.transpose(-2,-1)
  if t.ndim==3:ref=ref.reshape(list(ref.shape)[1:])
  return torch.equal(t,ref)
 def operation(self,name,args,kwargs,out):
  if name in ('scaled_dot_product_attention','_scaled_dot_product_efficient_attention','_native_multi_head_attention'):
   require(False,'unsupported fused attention path; no manual fallback')
  if name in ('mul','__mul__') and args:
   if any(args[0] is x for x in self.q_lineage):self.q_lineage.append(out)
   if any(args[0] is x for x in self.score_values):
    self.score_values.append(out);self.trace['post_qk_scaling'].append(dict(operation=name,shape=list(out.shape)))
  if name in ('view','reshape','contiguous','to','type_as','half','float','squeeze','unsqueeze','transpose','permute') and args:
   if any(args[0] is x for x in self.q_lineage):self.q_lineage.append(out)
   if any(args[0] is x for x in self.prob):
    self.prob.append(out);self.trace['probability_transitions'].append(dict(operation=name,input=layout(args[0]),output=layout(out),input_sha256=digest_bytes(tensor_bytes(args[0])),output_sha256=digest_bytes(tensor_bytes(out))))
   if any(args[0] is x for x in self.score_values):self.score_values.append(out)
   if any(args[0] is x for x in self.mask_values):self.mask_values.append(out)
  if name in ('max','maximum','clamp','clamp_min') and args and any(args[0] is x for x in self.mask_values):
   require(hasattr(out,'shape'),'unsupported mask reduction output')
   self.mask_values.append(out);self.trace['mask_postprocessing'].append(dict(operation=name,input=layout(args[0]),output=layout(out)))
  if name in ('bmm','matmul','__matmul__'):
   require(len(args)>=2,'missing native matmul operands');left,right=args[:2]
   rank=left.ndim
   require(left.dtype==right.dtype==out.dtype and left.device==right.device==out.device,'native product dtype/device differs')
   require(rank in (3,4) and right.ndim==out.ndim==rank and (name!='bmm' or rank==3),'unsupported matmul rank/broadcast')
   lead=[self.heads] if rank==3 else [1,self.heads]
   if self.phase=='START':
    require(list(left.shape)==lead+[self.n,self.d] and list(right.shape)==lead+[self.d,self.n] and list(out.shape)==lead+[self.n,self.n],'invalid QK tensor roles/axes')
    require(right.dtype==self.x['k'].dtype and self.equal_head(right,self.x['k'],True) and any(left is x for x in self.q_lineage),'QK operands differ from recorded K/native scaled Q')
    self.qk_inputs=[(left,rank-2),(right,rank-1)];self.score_values=[out];self.phase='QK'
    self.trace['operations'].append(dict(role='QK',operation=name,left=layout(left),right=layout(right),output=layout(out)))
   else:
    require(self.phase=='SOFTMAX' and list(left.shape)==lead+[self.n,self.n] and list(right.shape)==list(out.shape)==lead+[self.n,self.d],'invalid PV operation order/roles/axes')
    require(right.dtype==self.x['v'].dtype and any(left is x for x in self.prob) and self.equal_head(right,self.x['v']),'PV probabilities are not native softmax/cast/dropout output or V differs')
    self.observe('probabilities',left);self.observe('pv',out,[(left,rank-2),(right,rank-2)])
    self.trace['operations'].append(dict(role='PV',operation=name,left=layout(left),right=layout(right),output=layout(out)))
    self.phase='PV'
  if name in ('__iadd__','add_') and self.phase=='QK':require(False,'in-place mask add needs pre-operation evidence; unsupported passive boundary')
  if name in ('add','__add__') and self.phase=='QK' and len(args)>=2:
   import torch
   left,right=args[:2]
   require(kwargs.get('out') is None and out is not left,'in-place mask add cannot supply a passive pre-add snapshot')
   require(list(right.shape)==list(self.x['mask'].shape) and torch.equal(right,self.x['mask']),'native mask add operand differs')
   canonical_boundary_shape(left.shape,'scores',self.heads,self.n,self.d)
   require(any(left is x for x in self.score_values),'mask input is not native QK/scaling/view output')
   self.observe('scores',left,self.qk_inputs);self.mask_values=[out];self.phase='MASK'
   self.trace['mask']=dict(operation=name,mask=layout(right),sha256=digest_bytes(tensor_bytes(right)))
  if name=='softmax':
   require(self.phase=='MASK','softmax before verified native mask add or duplicate softmax')
   t=args[0];require(any(t is x for x in self.mask_values),'softmax input is not native masked/clamped scores');canonical_boundary_shape(t.shape,'masked_scores',self.heads,self.n,self.d)
   dim=kwargs.get('dim',args[1] if len(args)>1 else None);require(dim in (-1,t.ndim-1),'softmax on wrong key axis')
   self.observe('masked_scores',t);self.prob=[out];self.phase='SOFTMAX'
   self.trace['softmax']=dict(input=layout(t),output=layout(out),dim=dim,input_sha256=digest_bytes(tensor_bytes(t)),output_sha256=digest_bytes(tensor_bytes(out)),boundary='ACTUAL_SOFTMAX_INPUT_AFTER_MASK_AND_OPTIONAL_NATIVE_CLAMP')
  if name=='dropout':
   require(self.phase=='SOFTMAX' and args and any(args[0] is x for x in self.prob),'dropout not linked to native softmax')
   training=kwargs.get('training',args[2] if len(args)>2 else True);require(training is False,'stochastic dropout unsupported')
   self.prob.append(out);self.trace['probability_transitions'].append(dict(operation='dropout',input=layout(args[0]),output=layout(out),input_sha256=digest_bytes(tensor_bytes(args[0])),output_sha256=digest_bytes(tensor_bytes(out))))
   self.trace['dropout']=dict(observed=True,input=layout(args[0]),output=layout(out),training=training,p=kwargs.get('p',args[1] if len(args)>1 else 0.5))
 def finish(self):
  require(self.phase=='PV' and [x['role'] for x in self.trace['operations']]==['QK','PV'] and self.trace['softmax'] is not None,'unsupported/incomplete native attention operations; no manual fallback')


def native_replay(attention,operands,observe,*,trace=None):
 """Run installed eager attention unchanged, substituting recorded QKV only."""
 handles=[];state=AttentionOperations(attention,operands,observe)
 def pre(module,args):
  require(state.phase=='PV','output projection before verified PV');observe('output_projection_input',args[0])
 def post(module,args,out):observe('output_projection',out)
 try:
  handles=[attention.out_proj.register_forward_pre_hook(pre),attention.out_proj.register_forward_hook(post)]
  with patch.object(attention.q_proj,'forward',lambda x:operands['q']),patch.object(attention.k_proj,'forward',lambda x:operands['k']),patch.object(attention.v_proj,'forward',lambda x:operands['v']),operation_mode(state.operation):
   out=attention(hidden_states=operands['q'],attention_mask=operands['mask'],output_attentions=False)
  state.finish();observe('module_output',out[0])
  if trace is not None:trace.update(state.trace)
  return out[0]
 finally:
  for h in handles:h.remove()


def execute(experiment,output,commit):
 p,pe=load(experiment,current=True)
 require(p['source']['commit']==commit and os.environ.get('C3_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'manual opted-in pinned allocation required')
 require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline assets required')
 require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'four CPUs/64 GiB allocation required')
 import shutil,torch
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one CUDA-visible GPU required')
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 require(available>=32*2**30 and shutil.disk_usage(Path(experiment).parent).free>2**30,'CPU/disk reserve unavailable')
 from .r5_i5_gpu import diagnostics
 from .r3a_runtime import load_bare,memory
 pm,_,_=validate_prepared(p['base_prepared']);require(pm['identity']==p['model_identity'] and versions()==pm['identity']['versions'],'model/software identity differs')
 torch.cuda.set_device(0);gpu=diagnostics(pm);require(gpu['eligible'] and gpu['gpu']['free_bytes']>5400000000+5*2**30,'compatible RTX3090/memory required')
 initial=backend();effective=apply_recorded_backend(p['backend']);require(all(effective[k]==p['backend'][k] for k in ('torch','transformers','cuda_compiled_version')),'historical runtime versions differ')
 out=fresh(output,(experiment,p['history']['run'],p['history']['audit'],p['base_prepared']));store=Store(out,p['budget'],p['max_files']);start=time.monotonic();status=dict(status='FAILED',GPU_NOT_RUN=True);rows=[]
 try:
  model,tok,meta=load_bare(pm['identity'],'cuda:0')
  from semcache.models.model_adapter import OPTModelAdapter
  adapter=OPTModelAdapter(model);attn=adapter.layers[0].self_attn
  from semcache.system_cost.es_base_profile import assert_no_lora
  assert_no_lora(model,adapter)
  fp=full_base_fingerprint(model);require(fp['sha256']==p['model_identity']['base_weight_sha256'],'loaded base fingerprint differs')
  require(type(attn).__name__=='OPTAttention' and digest_bytes(inspect.getsource(attn.forward).encode())==p['contract']['methods']['attention']['source_sha256'],'installed native OPTAttention source differs')
  require(attn.num_heads==32 and attn.head_dim==80 and not attn.training,'actual Layer0 OPT-2.7B eager attention required')
  original=parameters(attn.out_proj);name='model.decoder.layers.0.self_attn.out_proj'
  require(all(v is None or fp['tensors'][name+'.'+k]['sha256']==v['sha256'] for k,v in original.items()),'native out_proj fingerprint differs')
  operands={};input_layouts={}
  for label,items in p['blobs'].items():
   operands[label]={};input_layouts[label]={}
   for role in ('q','k','v','mask'):
    item=items[role];t=tensor_from_bytes(pe.tensor('',item),item['dtype'],item['shape']).to('cuda:0')
    recorded=item['historical'];shape=recorded.get('observed_shape',recorded['shape'])
    # Only reproduce layout with proven, complete metadata. Otherwise declare a
    # contiguous control; historical module output equality remains mandatory.
    repro=recorded.get('observed_stride')
    if isinstance(repro,list) and shape==list(t.shape) and type(recorded.get('storage_offset')) is int:t=reconstruct(t,recorded)
    require(tensor_bytes(t)==pe.tensor('',item),'operand layout reconstruction changed bytes')
    operands[label][role]=t;input_layouts[label][role]=dict(actual=layout(t),historical=recorded,layout_reproduction='RECORDED_STRIDED' if isinstance(repro,list) and shape==list(t.shape) and type(recorded.get('storage_offset')) is int else 'CONTIGUOUS_CONTROL_ORIGINAL_LAYOUT_UNAVAILABLE')
  fixed={k:{r:digest_bytes(tensor_bytes(t)) for r,t in v.items()} for k,v in operands.items()}
  with torch.inference_mode():
   for step in schedule():
    require(time.monotonic()-start<p['deadline_seconds'],'bounded replay deadline exceeded')
    require(parameters(attn.out_proj)==original and all(backend()[k]==effective[k] for k in CORE) and memory('cuda:0')['uuid']==gpu['execution_gpu_uuid'],'weights/backend/GPU changed')
    label=step['label'];xs=operands[label];require({r:digest_bytes(tensor_bytes(t)) for r,t in xs.items()}==fixed[label],'replay input changed')
    captured={};native_trace={}
    def observe(role,t,operands=None):
     require(role not in captured,'duplicate native attention boundary')
     observed=layout(t);canonical=canonical_boundary(t,role,attn.num_heads,xs['q'].shape[1],attn.head_dim)
     data=tensor_bytes(canonical);shape=list(canonical.shape);dtype=str(t.dtype).removeprefix('torch.');width=DTYPES[dtype][1]
     data,selected=slice_axis(data,shape,1,0,64,width)
     input_receipts=[]
     for index,(operand,axis) in enumerate(operands or []):
      blob=tensor_bytes(operand);dt=str(operand.dtype).removeprefix('torch.')
      prefix,ps=operand_prefix(blob,list(operand.shape),dt,axis,also_keys=role=='pv' and index==0)
      input_receipts.append(dict(layout=layout(operand),full_sha256=digest_bytes(blob),prefix_sha256=digest_bytes(prefix),prefix_shape=ps,canonical_prefix_shape=ps[1:] if operand.ndim==4 else ps,token_axis=axis,
       selected_axes=[axis,operand.ndim-1] if role=='pv' and index==0 else [axis]))
     captured[role]=dict(data=data,shape=selected,dtype=dtype,observed=observed,input_receipts=input_receipts,canonicalization='DROP_UNIT_BATCH_AXIS' if t.ndim==4 else 'IDENTITY')
    torch.cuda.synchronize();native_replay(attn,xs,observe,trace=native_trace);torch.cuda.synchronize()
    require(set(captured)==set(BOUNDARIES)|{'output_projection_input'},'native capture incomplete')
    require(parameters(attn.out_proj)==original and {r:digest_bytes(tensor_bytes(t)) for r,t in xs.items()}==fixed[label],'native replay mutated operands/weights')
    if not step['warmup']:
     items={}
     for role,item in captured.items():items[role]=dict(store.tensor('tensors/'+label+'_'+str(step['repetition'])+'_'+role+'.bin',item['data'],item['shape'],item['dtype']),observed=item['observed'],input_receipts=item['input_receipts'],canonicalization=item['canonicalization'])
     rows.append(dict(step,inputs=fixed[label],boundaries=items,native_operations=native_trace))
  final=full_base_fingerprint(model);require(final['sha256']==fp['sha256'],'base weights changed')
  store.put('i5r2c3_weights.json',dict(before=original,after=parameters(attn.out_proj),base_fingerprint=fp))
  store.put('i5r2c3_backend.json',dict(initial=initial,effective=effective,after=backend(),gpu=gpu,inputs=input_layouts))
  store.put('i5r2c3_plan.json',p);store.put('i5r2c3_records.json',rows)
  status.update(status='COMPLETE',GPU_NOT_RUN=False,source_commit=commit,experiment_id=p['experiment_id'],model_load_count=1,model_forward_calls=0,generation_tokens=0)
 finally:
  store.put('i5r2c3_status.json',status);store.writer.reconcile();seal(out)
 return status


def analyze(experiment,run,output):
 p,pe=load(experiment);ev=Evidence(run,max_input_bytes=p['budget']);receipt=ev.verify();st=ev.json('i5r2c3_status.json')
 require(st['status']=='COMPLETE' and not st['GPU_NOT_RUN'] and st['experiment_id']==p['experiment_id'] and st['source_commit']==p['source']['commit'] and ev.json('i5r2c3_plan.json')==p and st['model_load_count']==1 and st['model_forward_calls']==st['generation_tokens']==0,'replay completion/identity differs')
 require(len(ev.files)<=p['max_files']+2,'replay file bound exceeded')
 env=ev.json('i5r2c3_backend.json');w=ev.json('i5r2c3_weights.json');fp=w['base_fingerprint']
 require(digest(fp['tensors'])==fp['sha256']==p['model_identity']['base_weight_sha256'] and w['before']==w['after'],'base fingerprint/weight immutability differs')
 require(all(v is None or fp['tensors']['model.decoder.layers.0.self_attn.out_proj.'+k]['sha256']==v['sha256'] for k,v in w['before'].items()),'out_proj/base fingerprint differs')
 from .gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS
 pm,_,_=validate_prepared(p['base_prepared']);gpu=env['gpu'];checked=validate_gpu(pm['gpu'],gpu['gpu'],'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
 require(all(gpu[k]==v for k,v in checked.items()),'GPU receipt differs')
 require(all(env['effective'][k]==env['after'][k]==p['backend'][k] for k in CORE),'replay backend differs')
 rows=ev.json('i5r2c3_records.json');validate_repetitions(rows)
 comparisons=[];repeats=[];historical=[]
 for i in range(6):
  pair={r['label']:r for r in rows if r['repetition']==i}
  for label,row in pair.items():
   require(row['inputs']=={r:p['blobs'][label][r]['sha256'] for r in ('q','k','v','mask')},'recorded operand hashes differ')
   trace=row['native_operations'];heads=p['contract']['heads'];h=p['model_identity']['hidden_size'];n=64 if label=='S' else 256
   require([x['role'] for x in trace['operations']]==['QK','PV'],'native operation order differs')
   for op in trace['operations']:
    rank=len(op['left']['shape']);lead=[heads] if rank==3 else [1,heads]
    require(rank in (3,4) and op['operation'] in ('matmul','__matmul__','bmm') and (op['operation']!='bmm' or rank==3),'unsupported native product receipt')
    shapes=(lead+[n,h//heads],lead+[h//heads,n],lead+[n,n]) if op['role']=='QK' else (lead+[n,n],lead+[n,h//heads],lead+[n,h//heads])
    require(all(op[k]['dtype']=='float16' for k in ('left','right','output')) and tuple(op[k]['shape'] for k in ('left','right','output'))==shapes,'native QK/PV receipt role/axis differs')
   canonical_boundary_shape(trace['softmax']['input']['shape'],'masked_scores',heads,n,h//heads)
   probability_hash=trace['softmax']['output_sha256']
   for event in trace['probability_transitions']:
    require(event['input_sha256']==probability_hash,'native probability transition broken')
    probability_hash=event['output_sha256']
   require(row['boundaries']['pv']['input_receipts'][0]['full_sha256']==probability_hash,'PV did not consume native softmax/cast/dropout output')
   if trace['dropout']['observed']:require(trace['dropout']['training'] is False,'stochastic dropout receipt')
   for role,item in row['boundaries'].items():
    ev.tensor('',item)
    heads=p['contract']['heads'];h=p['model_identity']['hidden_size'];n=64 if label=='S' else 256
    shape=[heads,64,n] if role in ('scores','masked_scores','probabilities') else [heads,64,h//heads] if role=='pv' else [1,64,h]
    observed_shape=canonical_boundary_shape(item['observed']['shape'],role,heads,n,h//heads)
    allowed_dtype=('float16','float32') if role in ('scores','masked_scores') else ('float16',)
    require(item['shape']==shape and item['dtype'] in allowed_dtype and observed_shape[1]==n,'native boundary shape/dtype/layout differs')
    require(item['canonicalization']==('DROP_UNIT_BATCH_AXIS' if len(item['observed']['shape'])==4 else 'IDENTITY'),'unverified canonical batch representation')
    if role=='probabilities':
     data=ev.tensor('',item);values=struct.iter_unpack('<e',data)
     require(all(v>=0 and math.isfinite(v) and (v==0 if k%n> (k//n)%64 else True) for k,(v,) in enumerate(values)),'future probability not zero or probability invalid')
    if i:
     ref=next(r for r in rows if r['repetition']==0 and r['label']==label)['boundaries'][role]
     require(ref['shape']==item['shape'] and ref['dtype']==item['dtype'],'repeat shape/dtype differs')
     repeats.append(dict(label=label,repetition=i,boundary=role,bitwise_equal=ev.tensor('',ref)==ev.tensor('',item)))
   for role in ('scores','pv'):
    operands=row['boundaries'][role]['input_receipts'];require(len(operands)==2,'missing native operand receipts')
    for j,r in enumerate(operands):
     rank=len(r['layout']['shape']);axis=rank-1 if role=='scores' and j==1 else rank-2
     expected_axes=[axis,rank-1] if role=='pv' and j==0 else [axis]
     require(r['token_axis']==axis and r['selected_axes']==expected_axes,'query/key token receipt axis differs')
     expected=list(r['layout']['shape'])
     for selected_axis in expected_axes:expected[selected_axis]=64
     require(r['prefix_shape']==expected and r['canonical_prefix_shape']==(expected[1:] if rank==4 else expected),'operand prefix shape differs')
   for role,operand_role,transpose in (('scores','k',True),('pv','v',False)):
    expected_prefix=head_prefix_bytes(pe.tensor('',p['blobs'][label][operand_role]),n,heads,h//heads,transpose=transpose)
    require(row['boundaries'][role]['input_receipts'][1]['prefix_sha256']==digest_bytes(expected_prefix),'K/V native operand prefix hash differs from recorded operands')
   probs=row['boundaries']['probabilities'];common,_=slice_axis(ev.tensor('',probs),probs['shape'],2,0,64,2)
   require(row['boundaries']['pv']['input_receipts'][0]['prefix_sha256']==digest_bytes(common),'PV probability prefix hash differs from actual consumed probability tensor')
   module=row['boundaries']['module_output'];ref=p['blobs'][label]['reference']
   require(module['shape']==ref['shape'] and module['dtype']==ref['dtype'],'historical module output alignment differs')
   historical.append(dict(label=label,repetition=i,metrics=compare_aligned((pe.tensor('',ref),ref['shape']),(ev.tensor('',module),module['shape']),ref['dtype'],'module_output')))
  for role in BOUNDARIES:
   a=pair['S']['boundaries'][role];b=pair['F']['boundaries'][role];require(a['dtype']==b['dtype'],'boundary dtype differs')
   comparisons.append(dict(boundary=role,repetition=i,metrics=compare_aligned((ev.tensor('',a),a['shape']),(ev.tensor('',b),b['shape']),a['dtype'],role),reference=a,actual=b,matching_aligned_operand_prefix_hashes=[x['prefix_sha256']==y['prefix_sha256'] for x,y in zip(a.get('input_receipts',[]),b.get('input_receipts',[]))]))
 result=verdict(comparisons,all(x['bitwise_equal'] for x in repeats),all(x['metrics']['bitwise_equal'] for x in historical))
 result['score_operand_prefixes_equal']=all(all(x['matching_aligned_operand_prefix_hashes']) and len(x['matching_aligned_operand_prefix_hashes'])==2 for x in comparisons if x['boundary']=='scores')
 if not result['score_operand_prefixes_equal']:
  result['FIRST_DIVERGENCE_LOCALIZED']='UPSTREAM_ATTENTION_SCORE_OPERANDS_DIFFER' if result['HISTORICAL_REPLAY_REPRODUCED'] else 'NOT_ESTABLISHED'
 out=fresh(output,(experiment,run));write(out/'i5r2c3_tensor_comparisons.json',dict(cross_shape=comparisons,repeatability=repeats,historical=historical));write(out/'i5r2c3_verdicts.json',result)
 write(out/'i5r2c3_independent_audit.json',dict(status='PASS' if result['HISTORICAL_REPLAY_REPRODUCED'] else 'REPLAY_NOT_ESTABLISHED',inventory=receipt,source=p['source'],scope='Independent CPU checks of sealed native operation captures; no fresh physical cache test'))
 (out/'i5r2c3_report.md').write_text('# C3 native attention replay review\n\nHistorical replay reproduced: '+str(result['HISTORICAL_REPLAY_REPRODUCED'])+'\n\nFirst localized boundary: '+str(result['FIRST_DIVERGENCE_LOCALIZED'])+'\n\nROOT_CAUSE remains UNRESOLVED. No performance or full-model numerical restoration claim. Non-reproduced isolated observations cannot localize historical arithmetic.\n')
 seal(out);require(result['HISTORICAL_REPLAY_REPRODUCED'],'HISTORICAL_REPLAY_NOT_ESTABLISHED; isolated observations sealed without historical localization claim');return result
