"""DIAGNOSTIC_ONLY native QK recovery and bounded operator controls."""
import inspect,os,time
from pathlib import Path
from .communication import require,digest_bytes
from .r3a_artifacts import tensor_bytes,tensor_from_bytes,full_base_fingerprint,validate_prepared,versions
from .r5_i5_evidence import seal
from .r5_i5r2b_capture import backend
from .r5_i5r2bs1_plan import CORE
from .r5_i5r2bs1_runtime import Store,layout,reconstruct,apply_recorded_backend
from .r5_i5r2c3_runtime import native_replay,canonical_boundary,operand_prefix
from .r5_i5r2_tensors import slice_axis
from .r5_i5r2c4 import load,fresh,CASES,schedule


def common(t):
 require(list(t.shape)[0:2]==[1,32] and t.ndim==4 and min(t.shape[-2:])>=64,'invalid QK output common-region shape')
 return t[0,:,:64,:64]


def make_pair(recovered,case,layout_name,precision):
 import torch
 require(case in CASES and layout_name in ('NATIVE','CONTIGUOUS') and precision in ('FP16','FP32'),'unsupported C4 condition')
 nq,nk=CASES[case];q=recovered['S' if nq==64 else 'F']['q'];kt=recovered['S' if nk==64 else 'F']['kt']
 if layout_name=='CONTIGUOUS':q=q.clone(memory_format=torch.contiguous_format);kt=kt.clone(memory_format=torch.contiguous_format)
 if precision=='FP32':
  # Cast numeric values of the already-scaled native operands; do not rescale.
  q32=q.float();k32=kt.float()
  if layout_name=='NATIVE':
   q32=reconstruct(q32,dict(observed_stride=list(q.stride()),storage_offset=q.storage_offset()))
   k32=reconstruct(k32,dict(observed_stride=list(kt.stride()),storage_offset=kt.storage_offset()))
  q,kt=q32,k32
 return q,kt


def execute(experiment,output,commit):
 p,h=load(experiment,current=True);require(commit==p['source']['commit'] and os.environ.get('C4_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'manual C4 GPU opt-in/source required')
 require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline checkpoints required')
 require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'four CPUs/64 GiB allocation required')
 import torch,shutil
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one CUDA-visible GPU required')
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 require(available>=32*2**30 and shutil.disk_usage(Path(experiment).parent).free>2**30,'CPU/disk reserve unavailable')
 from .r5_i5_gpu import diagnostics
 from .r3a_runtime import load_bare,memory
 pm,_,_=validate_prepared(p['base_prepared']);require(pm['identity']==p['model_identity'] and versions()==pm['identity']['versions'],'model/software differs')
 torch.cuda.set_device(0);gpu=diagnostics(pm);require(gpu['eligible'] and gpu['gpu']['free_bytes']>5400000000+5*2**30,'compatible RTX3090/memory required')
 initial=backend();effective=apply_recorded_backend(p['backend']);require(all(effective[k]==p['backend'][k] for k in ('torch','transformers','cuda_compiled_version')),'historical backend versions differ')
 out=fresh(output,(experiment,*p['inputs'].values(),p['base_prepared']));store=Store(out,p['bounds']['bytes'],p['bounds']['files']);started=time.monotonic()
 st=dict(status='FAILED',GPU_NOT_RUN=True,source_commit=commit,experiment_id=p['experiment_id'],model_load_count=0,model_forward_calls=0,generation_tokens=0)
 primary=None;reproduction={'passed':False};recovered={};inputs={};rows=[]
 store.put('i5r2c4_plan.json',p)
 try:
  model,tok,meta=load_bare(pm['identity'],'cuda:0');st['model_load_count']=1
  from semcache.models.model_adapter import OPTModelAdapter
  from semcache.system_cost.es_base_profile import assert_no_lora
  adapter=OPTModelAdapter(model);assert_no_lora(model,adapter);attn=adapter.layers[0].self_attn
  fp=full_base_fingerprint(model);require(fp['sha256']==p['model_identity']['base_weight_sha256'],'base fingerprint differs');store.put('i5r2c4_base_fingerprint.json',fp)
  require(attn.num_heads==32 and attn.head_dim==80 and type(attn).__name__=='OPTAttention' and digest_bytes(inspect.getsource(attn.forward).encode())==h['plan']['contract']['methods']['attention']['source_sha256'],'native attention implementation differs')
  from types import SimpleNamespace
  from .r4b2_kv import installed_contract
  from .r5_i5r2c3b0_runtime import verify_contract
  native_contract=installed_contract(SimpleNamespace(model=model,fixture=False,heads=32,identity=p['model_identity']))
  expected_contract={key:h['plan']['contract'][key] for key in native_contract}
  verify_contract(expected_contract,native_contract,store.writer,out/'native_contract')
  c3p=h['evidence']['c3_experiment'];c3run=h['evidence']['c3_run'];oldp=h['plan']
  with torch.inference_mode():
   for label,n in (('S',64),('F',256)):
    xs={}
    for role in ('q','k','v','mask'):
     item=oldp['blobs'][label][role];t=tensor_from_bytes(c3p.tensor('',item),item['dtype'],item['shape']).to('cuda:0');original=item['historical']
     if isinstance(original.get('observed_stride'),list) and original.get('observed_shape',original['shape'])==list(t.shape) and type(original.get('storage_offset')) is int:t=reconstruct(t,original)
     xs[role]=t
    qk={};captured={}
    def observe(role,t,operands=None):
     if role=='scores':
      require(len(operands)==2,'native QK operands unavailable')
      for key,(operand,axis) in zip(('q','kt'),operands):
       require(operand.ndim==4 and operand.dtype==torch.float16,'C4 requires actual rank4 FP16 native operands')
       info=layout(operand);data=tensor_bytes(operand)
       saved=store.tensor('operands/'+label+'_'+key+'.bin',data,list(operand.shape),'float16',info)
       qk[key]=reconstruct(tensor_from_bytes(data,'float16',list(operand.shape)).to('cuda:0'),dict(observed_stride=info['stride'],storage_offset=info['storage_offset']))
       inputs.setdefault(label,{})[key]=saved
       prefix,_=operand_prefix(data,list(operand.shape),'float16',axis);inputs[label][key+'_prefix_sha256' if key=='q' else 'k_prefix_sha256']=digest_bytes(prefix)
     if role in ('scores','module_output'):
      canonical=canonical_boundary(t,role,32,n,80)
      data=tensor_bytes(canonical[:,:64,:64] if role=='scores' else canonical[:,:64])
      captured[role]=data
    st['GPU_NOT_RUN']=False;trace={};native_replay(attn,xs,observe,trace=trace)
    original_rows=[r for r in h['rows'] if r['label']==label]
    store.put('i5r2c4_operands.json',inputs)
    recovery_checks=[]
    for j,key in enumerate(('q','kt')):
     ref=inputs[label][key]
     for r in original_rows:
      old=r['boundaries']['scores']['input_receipts'][j]
      recovery_checks.append(dict(role=key,repetition=r['repetition'],actual_sha256=ref['sha256'],expected_sha256=old['full_sha256'],layout_equal=ref['layout']==old['layout'],value_equal=ref['sha256']==old['full_sha256']))
    reproduction[label]=dict(operand_checks=recovery_checks);store.put('i5r2c4_reproduction.json',reproduction)
    for j,key in enumerate(('q','kt')):
     ref=inputs[label][key]
     require(all(ref['sha256']==r['boundaries']['scores']['input_receipts'][j]['full_sha256'] and ref['layout']==r['boundaries']['scores']['input_receipts'][j]['layout'] for r in original_rows),'reconstructed native QK value/layout receipt differs')
    native=torch.matmul(qk['q'],qk['kt']);torch.cuda.synchronize();data=tensor_bytes(common(native))
    score=store.tensor('baseline/'+label+'_score.bin',data,[32,64,64],'float16')
    checks=[]
    for r in original_rows:
     item=r['boundaries']['scores'];ref,_=slice_axis(c3run.tensor('',item),item['shape'],2,0,64,2)
     checks.append(dict(repetition=r['repetition'],native_attention_score_sha256=digest_bytes(captured['scores']),matmul_sha256=digest_bytes(data),historical_sha256=digest_bytes(ref),score_equal=data==ref==captured['scores'],module_equal=captured['module_output']==c3run.tensor('',r['boundaries']['module_output'])))
    reproduction[label].update(score=score,checks=checks,native_score=store.tensor('baseline/'+label+'_native_score.bin',captured['scores'],[32,64,64],'float16'),module_output=store.tensor('baseline/'+label+'_module.bin',captured['module_output'],[1,64,2560],'float16'))
    store.put('i5r2c4_reproduction.json',reproduction)
    require(all(x['score_equal'] and x['module_equal'] for x in checks),'native scaled-score baseline not reproduced; scaling order/post-QK scaling or runtime context unresolved')
    recovered[label]=qk
   require(inputs['S']['q_prefix_sha256']==inputs['F']['q_prefix_sha256'] and inputs['S']['k_prefix_sha256']==inputs['F']['k_prefix_sha256'],'native S/F operand prefixes differ')
   reproduction['passed']=True;store.put('i5r2c4_reproduction.json',reproduction);store.put('i5r2c4_operands.json',inputs)
   pairs={}
   for l in ('NATIVE','CONTIGUOUS'):
    for c in CASES:
     for precision in ('FP16','FP32'):pairs[l,c,precision]=make_pair(recovered,c,l,precision)
   for s in schedule():
    require(time.monotonic()-started<p['bounds']['seconds'] and memory('cuda:0')['uuid']==gpu['execution_gpu_uuid'],'C4 deadline/GPU changed')
    require(all(backend()[k]==effective[k] for k in CORE),'C4 backend changed')
    q,kt=pairs[s['layout'],s['case'],s['precision']];nq,nk=CASES[s['case']];sourceq=recovered['S' if nq==64 else 'F']['q'];sourcek=recovered['S' if nk==64 else 'F']['kt']
    require(torch.equal(q,sourceq.to(q.dtype)) and torch.equal(kt,sourcek.to(kt.dtype)),'control input values changed')
    before=(digest_bytes(tensor_bytes(q)),digest_bytes(tensor_bytes(kt)));torch.cuda.synchronize();result=torch.matmul(q,kt);torch.cuda.synchronize()
    require(before==(digest_bytes(tensor_bytes(q)),digest_bytes(tensor_bytes(kt))),'operator mutated inputs')
    if not s['warmup']:
     for precision,value in ((('FP32',result),('FP32_CAST_FP16',result.half())) if s['precision']=='FP32' else (('FP16',result),)):
      data=tensor_bytes(common(value));name='_'.join((s['layout'],s['case'],precision,str(s['repetition'])))
      row=dict(s,precision=precision,q=layout(q),kt=layout(kt),q_alignment_mod_256=q.data_ptr()%256,kt_alignment_mod_256=kt.data_ptr()%256,prefix_hash_dtype='float16_source',prefix_equal=True,input_values_verified=True,
       operand_hashes=[digest_bytes(tensor_bytes(sourceq)),digest_bytes(tensor_bytes(sourcek))],execution_input_hashes=list(before),q_prefix_sha256=inputs['S']['q_prefix_sha256'],k_prefix_sha256=inputs['S']['k_prefix_sha256'],
       layout_matches_native=all(list(t.stride())==list(ref.stride()) and t.storage_offset()==ref.storage_offset() for t,ref in ((q,sourceq),(kt,sourcek))),
       output=store.tensor('scores/'+name+'.bin',data,[32,64,64],'float32' if precision=='FP32' else 'float16'))
      rows.append(row)
   final=full_base_fingerprint(model);require(final['sha256']==fp['sha256'],'model weights changed');st['base_fingerprint_after']=final['sha256']
  store.put('i5r2c4_records.json',rows);store.put('i5r2c4_backend.json',dict(initial=initial,effective=effective,after=backend(),gpu=gpu))
  st['status']='COMPLETE'
 except BaseException as exc:primary=exc;st['reason']=type(exc).__name__+': '+str(exc)[:512];raise
 finally:
  try:store.put('i5r2c4_status.json',st);store.writer.reconcile();seal(out)
  except BaseException as secondary:
   if primary is None:raise
   from .r5_i5r2c3b0_runtime import failure_note
   failure_note(primary,'C4 failure evidence finalization: '+type(secondary).__name__)
 return st
