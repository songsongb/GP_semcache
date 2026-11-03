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


def native_replay(attention,operands,observe):
 """Run installed native attention with recorded total projection outputs.

 The Q carrier supplies ONLY [batch,length,hidden] to the native forward; three
 temporary projection wrappers return verified recorded QKV, without a Linear
 call or invented hidden-state values. No model/cache/LoRA forward is performed.
 """
 handles=[];bmm=0;softmax=0
 def operation(name,args,kwargs,out):
  nonlocal bmm,softmax
  if name=='bmm':
   bmm+=1
   if bmm==1:observe('scores',out,[(args[0],1),(args[1],2)])
   elif bmm==2:observe('probabilities',args[0]);observe('pv',out,[(args[0],1),(args[1],1)])
  if name=='softmax':
   softmax+=1;observe('masked_scores',args[0])
 def pre(module,args):observe('output_projection_input',args[0])
 def post(module,args,out):observe('output_projection',out)
 try:
  handles=[attention.out_proj.register_forward_pre_hook(pre),attention.out_proj.register_forward_hook(post)]
  with patch.object(attention.q_proj,'forward',lambda x:operands['q']),patch.object(attention.k_proj,'forward',lambda x:operands['k']),patch.object(attention.v_proj,'forward',lambda x:operands['v']),operation_mode(operation):
   out=attention(hidden_states=operands['q'],attention_mask=operands['mask'],output_attentions=False)
  require(bmm==2 and softmax==1,'unsupported native attention operations; no manual fallback')
  observe('module_output',out[0]);return out[0]
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
    captured={}
    def observe(role,t,operands=None):
     require(role not in captured,'duplicate native attention boundary')
     data=tensor_bytes(t);shape=list(t.shape);dtype=str(t.dtype).removeprefix('torch.');width=DTYPES[dtype][1]
     require(t.ndim==3 and shape[1]>=64,'unsupported native attention boundary axes')
     data,selected=slice_axis(data,shape,1,0,64,width)
     input_receipts=[]
     for operand,axis in operands or []:
      blob=tensor_bytes(operand);dt=str(operand.dtype).removeprefix('torch.');prefix,ps=slice_axis(blob,list(operand.shape),axis,0,64,DTYPES[dt][1])
      input_receipts.append(dict(layout=layout(operand),full_sha256=digest_bytes(blob),prefix_sha256=digest_bytes(prefix),prefix_shape=ps,token_axis=axis))
     captured[role]=dict(data=data,shape=selected,dtype=dtype,observed=layout(t),input_receipts=input_receipts)
    torch.cuda.synchronize();native_replay(attn,xs,observe);torch.cuda.synchronize()
    require(set(captured)==set(BOUNDARIES)|{'output_projection_input'},'native capture incomplete')
    require(parameters(attn.out_proj)==original and {r:digest_bytes(tensor_bytes(t)) for r,t in xs.items()}==fixed[label],'native replay mutated operands/weights')
    if not step['warmup']:
     items={}
     for role,item in captured.items():items[role]=dict(store.tensor('tensors/'+label+'_'+str(step['repetition'])+'_'+role+'.bin',item['data'],item['shape'],item['dtype']),observed=item['observed'],input_receipts=item['input_receipts'])
     rows.append(dict(step,inputs=fixed[label],boundaries=items))
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
   for role,item in row['boundaries'].items():
    ev.tensor('',item)
    heads=p['contract']['heads'];h=p['model_identity']['hidden_size'];n=64 if label=='S' else 256
    shape=[heads,64,n] if role in ('scores','masked_scores','probabilities') else [heads,64,h//heads] if role=='pv' else [1,64,h]
    require(item['shape']==shape and item['dtype']=='float16' and item['observed']['shape'][1]==n,'native boundary shape/dtype/layout differs')
    if role=='probabilities':
     data=ev.tensor('',item);values=struct.iter_unpack('<e',data)
     require(all(v>=0 and math.isfinite(v) and (v==0 if k%n> (k//n)%64 else True) for k,(v,) in enumerate(values)),'future probability not zero or probability invalid')
    if i:
     ref=next(r for r in rows if r['repetition']==0 and r['label']==label)['boundaries'][role]
     require(ref['shape']==item['shape'] and ref['dtype']==item['dtype'],'repeat shape/dtype differs')
     repeats.append(dict(label=label,repetition=i,boundary=role,bitwise_equal=ev.tensor('',ref)==ev.tensor('',item)))
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
