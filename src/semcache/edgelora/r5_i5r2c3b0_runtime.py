"""Two real distributed prefills; passive, complete Layer-0 operand capture only."""
from contextlib import contextmanager
import inspect,os,re,sys,time,uuid
from pathlib import Path
from .communication import Channel,connect,header,digest,require,digest_bytes
from .r3a_artifacts import ROOT,read,full_base_fingerprint,write as write_status
from .r5_i5_evidence import seal
from .r3b_cache import raw
from .r5_i5r2c3b0 import load,prepared,fresh
from .r5_i5r2b_capture import backend
from .r5_i5r2bs1_plan import CORE
from .r5_i5r2bs1_runtime import Store,layout,apply_recorded_backend


def contract_difference(expected,actual,*,limit=64,node_limit=20000):
 """Bounded JSON-pointer diff. Embedded code/config objects are hash-only."""
 require(1<=limit<=64 and 1<=node_limit<=20000,'invalid contract diff bounds')
 missing=object();rows=[];visited=0;truncated=False
 def summary(value,path):
  if value is missing:return dict(type='MISSING',sha256=None)
  result=dict(type=type(value).__name__,sha256=digest(value))
  if value is None or type(value) in (bool,int,float):result['value']=value
  elif isinstance(value,str) and not path.endswith('/source') and len(value)<=80 and re.fullmatch(r'[A-Za-z0-9_.:+/\-]*',value):result['value']=value
  return result
 def walk(a,b,path):
  nonlocal visited,truncated
  if visited>=node_limit or len(rows)>=limit:truncated=True;return
  visited+=1
  if a==b and digest(a)==digest(b):return
  if isinstance(a,dict) and isinstance(b,dict):
   for key in sorted(set(a)|set(b)):
    walk(a.get(key,missing),b.get(key,missing),path+'/'+key.replace('~','~0').replace('/','~1'))
    if truncated:return
  elif isinstance(a,list) and isinstance(b,list):
   for i in range(max(len(a),len(b))):
    walk(a[i] if i<len(a) else missing,b[i] if i<len(b) else missing,path+'/'+str(i))
    if truncated:return
  else:
   row=dict(path=path if len(path)<=512 else None,kind='missing_in_actual' if b is missing else 'extra_in_actual' if a is missing else 'type' if type(a) is not type(b) else 'value',expected=summary(a,path),actual=summary(b,path))
   if row['path'] is None:row['path_sha256']=digest(path);truncated=True
   rows.append(row)
 walk(expected,actual,'')
 return dict(exact_equal=expected==actual and digest(expected)==digest(actual),expected_sha256=digest(expected),actual_sha256=digest(actual),
  differences=rows,truncated=truncated,visited_nodes=visited,limits=dict(differences=limit,nodes=node_limit),normalization='NONE')


def failure_note(error,message):
 # Python 3.10 is supported; BaseException.add_note starts at 3.11.
 try:
  error.b0_secondary_failures=getattr(error,'b0_secondary_failures',[])+[message]
  if callable(getattr(error,'add_note',None)):error.add_note(message)
 except BaseException:pass


def verify_contract(expected,actual,writer,out):
 report=contract_difference(expected,actual)
 if not report['exact_equal']:
  try:require(False,'native OPT/adapter contract differs')
  except BaseException as primary:
   try:writer.write(Path(out)/'i5r2c3b0_contract_difference.json',report)
   except BaseException as secondary:failure_note(primary,'Contract difference receipt failed: '+type(secondary).__name__)
   raise
 writer.write(Path(out)/'i5r2c3b0_contract_difference.json',report)


def finalize_attempt(out,writer,status,*,primary=None,channel=None,child=None,finish_child=None):
 """Best-effort failure evidence; cleanup never replaces a primary exception.

 A failed accounting ledger remains failed. A last-resort FAILED status is
 written directly only inside this fresh attempt; it cannot authorize an audit.
 """
 errors=[];first=primary;out=Path(out)
 def attempt(stage,action):
  nonlocal first
  try:action();return True
  except BaseException as exc:
   errors.append(dict(stage=stage,type=type(exc).__name__,message=str(exc)[:256]))
   if first is None:first=exc
   return False
 if channel is not None:attempt('channel_close',channel.close)
 if child is not None:
  try:alive=child[0].poll() is None
  except BaseException as exc:
   alive=True;errors.append(dict(stage='child_poll',type=type(exc).__name__,message=str(exc)[:256]))
   if first is None:first=exc
  if alive:
   attempt('child_terminate',child[0].terminate)
   attempt('child_finish',lambda:finish_child(child,timeout=10))
 if not attempt('writer_reconcile',writer.reconcile):status['evidence_accounting']='FAILED'
 def save_status():
  if first is not None:status.update(status='FAILED',reason=type(first).__name__+': '+str(first)[:512])
  status['finalization_errors']=list(errors)
  if not attempt('status_write',lambda:writer.write(out/'i5r2c3b0_status.json',status)):
   status.update(status='FAILED',reason=type(first).__name__+': '+str(first)[:512],evidence_accounting='FAILED',finalization_errors=list(errors))
   attempt('failed_status_fallback',lambda:write_status(out/'i5r2c3b0_status.json',status))
 save_status()
 if not attempt('seal',lambda:seal(out)):
  status['failure_seal']='FAILED';save_status()
 if primary is not None:
  if errors:failure_note(primary,'B0 secondary finalization failures: '+', '.join(e['stage']+':'+e['type'] for e in errors))
 elif first is not None:raise first


@contextmanager
def layer0_hooks(driver,save):
 """Observe real TOTAL projection returns and native mask/module output."""
 a=driver.adapter.layers[0].self_attn;handles=[]
 try:
  for role,module in driver.adapter.projection_modules(0).items():
   handles.append(module.register_forward_hook(lambda m,args,out,role=role:save(role,out)))
  def mask(module,args,kwargs):
   values=inspect.signature(module.forward).bind_partial(*args,**kwargs).arguments
   require(values.get('attention_mask') is not None,'native additive mask unavailable')
   save('mask',values['attention_mask'])
  handles.append(a.register_forward_pre_hook(mask,with_kwargs=True))
  handles.append(a.register_forward_hook(lambda m,args,out:save('reference',out[0],prefix=True)))
  yield
 finally:
  for h in handles:h.remove()


def capture_prefill(driver,task,mode,cohort,save):
 # QualityDriver enters physical_path BEFORE its observer, preserving the bare
 # projection guard. Hooks are removed before physical_path's final checks.
 ids=task['token_ids']
 return driver.c_forward(task['request_id'],0,ids,ids,None,task['policy'],mode,cohort,use_cache=True,observer=lambda adapter:layer0_hooks(driver,save))


def execute(experiment,output,commit):
 p,pe,h=load(experiment,current=True);require(not p['TEST_FIXTURE_ONLY'] and commit==p['source']['commit'] and os.environ.get('B0_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'manual pinned GPU opt-in required')
 require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline checkpoints required')
 require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'four CPUs/64 GiB allocation required')
 import shutil,torch
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 require(available>=32*2**30 and shutil.disk_usage(Path(experiment).parent).free>2**30,'CPU/disk reserve unavailable')
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one CUDA-visible GPU required')
 m,pm,specs,registry=prepared(h,p['multi_prepared']);spec=specs['Adapter_A']
 from .r5_i5_gpu import diagnostics,receipt
 gpu=diagnostics(pm);require(gpu['eligible'] and gpu['gpu']['free_bytes']>5400000000+5*2**30,'compatible RTX3090 with headroom required')
 initial_backend=backend();apply_recorded_backend(p['backend'])
 from .r4a_runtime import setup
 from .r5_i5_driver import QualityDriver
 from .r5_i5r2c2 import padded,gate
 from .r5_multi_artifacts import runtime_view,physical,tokenizer_identity
 from .r5_i5_contracts import request
 from .r5_i5_session import RoutedChannel,VERSION as TCP_VERSION
 from .r5_i5_protocol import VERSION
 from .r5_i5_limits import wire_limits
 from .r4c_workload import termination
 from .r5_perf_evidence import PerformanceEvidence,spawn,finish,accounted_modules
 from .r3a_runtime import memory
 started=time.monotonic();setup_plan=dict(base_prepared=p['base_prepared'],fixture=False,device='cuda:0',i5_gpu_policy_scope='R5_I5')
 out=fresh(output,(experiment,p['history']['run'],p['history']['audit'],p['multi_prepared']));store=Store(out,p['budget'],p['max_files']);writer=store.writer
 status=dict(status='FAILED',GPU_NOT_RUN=True,source_commit=commit,experiment_id=p['experiment_id'],model_load_count=0,prefill_forwards=0,decode_forwards=0,cache_reads=0,cache_admissions=0,protocol_selected_tokens=0)
 child=channel=None;primary=None;records={};padcfg=dict(mode='SOURCE_ONLY',source=p['tasks'][0]['request_id'],target=p['tasks'][1]['request_id'])
 class Driver(QualityDriver):
  label=None
  @property
  def experimental_native_projection(self):return self.project if self.label=='S' else None
  def project(self,c,layer,role,native,x):
   direction=gate(padcfg,'SEMCACHE_INTEGRATED',c);require(direction=='APPEND','B0 source padding phase differs')
   value,r=padded(native,x,direction,bias=native.__self__.bias is not None)
   writer.append(out/'padding.jsonl',dict(r,layer=layer,role=role,context=c));return value
 try:
  with accounted_modules(writer):
   _,_,_,old,tok,current=setup(setup_plan,native=False,gpu_identity_policy='SAME_COMPATIBLE_GPU_CLASS');status['model_load_count']=1
   apply_recorded_backend(p['backend']);require(all(backend()[k]==p['backend'][k] for k in (*CORE,'torch','transformers','cuda_compiled_version')),'effective historical backend differs')
   require(tokenizer_identity(pm,tok)==p['tokenizer'],'runtime tokenizer identity differs')
   view=runtime_view(pm,spec);es=out/'es';ud=out/'ud';es.mkdir();ud.mkdir()
   driver=Driver(view['identity'],'cuda:0',model=old.model,adapter=old.adapter,native=False);driver.model_tokenizer=tok
   contract=driver.configure(es,budget_bytes=p['budget']);ident,contract=physical(view,p['tokenizer'],'cuda:0',current['uuid'],contract,spec,registry)
   writer.write(out/'i5r2c3b0_backend.json',dict(initial=initial_backend,effective=backend()))
   writer.write(out/'i5r2c3b0_allocation.json',dict(gpu=current,preparation_gpu_uuid=pm['gpu']['uuid'],gpu_identity_policy='SAME_COMPATIBLE_GPU_CLASS',prefill_forwards=0))
   verify_contract(p['contract'],contract,writer,out);driver.contract=contract
   driver.evidence=PerformanceEvidence(es,writer,profile='SCALABLE_EVALUATION')
   identity=dict(model_identity=view['identity'],adapter=spec,physical_identity=ident,contract=contract)
   cohort=digest(dict(experiment_id=p['experiment_id'],physical_identity=ident));session=uuid.uuid4().hex
   child_plan=dict(multi_prepared=p['multi_prepared'],base_prepared=p['base_prepared'],fixture=False,adapter_id='Adapter_A',physical_identity=ident,
    cohort_sha256=cohort,context_limit=driver.context_limit,session_id=session,runtime_budget_seconds=p['deadline_seconds'],source_commit=commit,sources=p['source']['files'],
    account_root=str(out),condition_budget_bytes=p['budget'],output=str(ud))
   writer.write(ud/'plan.json',child_plan)
   env=dict(os.environ,R5_I5_ENABLE_GPU='1',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
   child=spawn([sys.executable,'-u',str(ROOT/'scripts/119_run_r5_i5.py'),'--worker','ud','--plan',str(ud/'plan.json')],writer,ud/'stdout.log',ud/'stderr.log',env=env)
   deadline=time.monotonic()+120
   while not (ud/'ready.json').exists():
    require(child[0].poll() is None and time.monotonic()<deadline,'B0 UD startup failed/timed out');time.sleep(.01)
   info=read(ud/'ready.json');require(info['pid']==child[0].pid!=os.getpid(),'independent CPU UD required')
   channel=Channel(connect(info['port']),session,spec['matrix_sha256'],timeout=p['deadline_seconds'],trace=lambda f:writer.append(es/'wire.jsonl',f))
   hello=dict(application=TCP_VERSION,adapter_id='Adapter_A',matrix_sha256=spec['matrix_sha256'],registry_sha256=registry['sha256'],physical_identity=ident,cohort_sha256=cohort)
   channel.send(header('HELLO','handshake',-1,**hello));reply,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
   require(reply['metadata']==dict(hello,pid=info['pid'],user_device_cuda_used=False),'UD handshake differs')
   driver.channel=RoutedChannel(channel,registry,ident)
   for label,task in zip(('S','F'),p['tasks']):
    require(time.monotonic()-started<p['deadline_seconds'] and memory('cuda:0')['uuid']==current['uuid'],'B0 deadline/GPU identity changed')
    driver.label=label;driver.sample_id=task['sample_id'];ids=task['token_ids'];req=request(task,ids,task['policy'],ident,fixture=False)
    driver.channel.request=req;writer.append(out/'wire_limits.jsonl',wire_limits(req,registry,cohort,driver.context_limit))
    tensors={}
    def save(role,t,prefix=False):
     require(role not in tensors,'duplicate B0 boundary');metadata=layout(t)
     require(t.dtype==torch.float16 and t.device.type=='cuda','actual FP16 GPU capture required')
     selected=t[:,:64] if prefix else t
     tensors[role]=dict(store.tensor('tensors/'+label+'_'+role+'.bin',raw(selected),list(selected.shape),'float16',metadata),
      observed_shape=metadata['shape'],observed_stride=metadata['stride'],storage_offset=metadata['storage_offset'],original_contiguous=metadata['contiguous'],semantic_positions=[0,64 if prefix else len(ids)])
    before=backend()
    status['GPU_NOT_RUN']=False
    result,record,blocks,values,blob=capture_prefill(driver,task,h['contexts'][label]['mode'],cohort,save)
    require(set(tensors)==set('qkv')|{'mask','reference'},'B0 operand capture incomplete')
    selected=max(range(len(values)),key=values.__getitem__);reason=termination(task['policy'],[selected]);require(reason is not None,'prefill protocol completion unavailable')
    complete=dict(version=VERSION,cohort_sha256=cohort,action='COMPLETE_REQUEST',parent_request_id=task['request_id'],generated_token_ids=[selected],termination=reason)
    channel.send(header('HELLO','complete_'+label,-1,application=TCP_VERSION,control=complete));reply,_,_=channel.recv(kind='READY',request_id='complete_'+label,layer_id=-1)
    require(reply['metadata']['control_sha256']==digest(complete),'request completion receipt differs')
    records[label]=dict(tensors=tensors,identity=identity,context=record['context'],position_observation=record['position_observation'],backend_before=before,backend_after=backend(),
     request=req,projection_calls=record['executed_projection_calls'],real_projection_rows=record['executed_projection_token_rows'],protocol_selected_token=selected,prefill_logits_sha256=digest_bytes(blob))
    status['prefill_forwards']+=1;status['protocol_selected_tokens']+=1
    del result,blocks,values,blob;driver.synchronize()  # No in-request or cross-request past survives.
   channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();channel=None
   life=finish(child,timeout=30);writer.write(ud/'lifecycle.json',life);require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'UD completion failed')
   require(full_base_fingerprint(old.model)['sha256']==pm['identity']['base_weight_sha256'],'model weights changed')
   writer.write(out/'gpu_identity.json',dict(receipt=receipt(pm,current,'SAME_COMPATIBLE_GPU_CLASS'),gpu=current))
   writer.write(out/'i5r2c3b0_backend.json',dict(initial=initial_backend,effective=backend()))
   writer.write(out/'i5r2c3b0_records.json',records);writer.write(out/'i5r2c3b0_plan.json',p)
   require(writer.status()['data_files']<=p['max_files'],'B0 file bound exceeded')
   status.update(status='COMPLETE',GPU_NOT_RUN=False,es_pid=os.getpid(),ud_pid=child[0].pid)
 except BaseException as exc:
  primary=exc;status.update(status='FAILED',reason=type(exc).__name__+': '+str(exc)[:512]);raise
 finally:
  finalize_attempt(out,writer,status,primary=primary,channel=channel,child=child,finish_child=finish)
 return status


def audit_receipts(p,ev,records,h):
 """Independent all-layer TCP/CPU proof; tensor bytes remain Layer-0 only."""
 from collections import Counter
 from .r5_i5_gpu import verify_receipt
 from .r5_i5_protocol import validate_context
 from .r5_i5_contracts import validate_request
 m,pm,specs,registry=prepared(h,p['multi_prepared']);gpu=ev.json('gpu_identity.json')
 verify_receipt(gpu['receipt'],pm,gpu['gpu'],'SAME_COMPATIBLE_GPU_CLASS');uuid_now=gpu['receipt']['execution_gpu_uuid']
 st=ev.json('i5r2c3b0_status.json');life=ev.json('ud/lifecycle.json');ud=ev.json('ud/result.json');initial=ev.json('ud/initial.json')
 require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'] and ud['status']=='COMPLETE' and not ud['user_device_cuda_used'] and not ud['full_model_loaded'] and ud['layers']==64 and ud['forwards']==2 and ud['parents']==2 and not ud['grants'] and len(ud['finished'])==2 and st['ud_pid']==ud['pid']==life['pid']!=st['es_pid'] and initial['cpu']['effective_threads']==4,'CPU UD lifecycle/isolation differs')
 esframes=ev.jsonl('es/wire.jsonl');udframes=ev.jsonl('ud/wire.jsonl')
 completions=[f['header']['metadata']['control'] for f in esframes if f['operation']=='send' and f['header']['kind']=='HELLO' and 'control' in f['header']['metadata']]
 require(len(completions)==2 and [x['action'] for x in completions]==['COMPLETE_REQUEST']*2 and [x['parent_request_id'] for x in completions]==[t['request_id'] for t in p['tasks']] and all(len(x['generated_token_ids'])==1 for x in completions),'B0 unexpected control/cache/generation lifecycle')
 for a,b in (('send','recv'),('recv','send')):
  require(Counter(f['frame_sha256'] for f in esframes if f['operation']==a)==Counter(f['frame_sha256'] for f in udframes if f['operation']==b),'ES/UD TCP frame integrity differs')
 layers=ev.jsonl('ud/layers.jsonl');require(len(layers)==64,'CPU correction coverage incomplete')
 oldev=h['historical_evidence'];padding=ev.jsonl('padding.jsonl')
 require([(r['layer'],r['role']) for r in padding]==[(l,r) for l in range(32) for r in 'qkv'],'source padding coverage differs')
 for label,task,n in zip(('S','F'),p['tasks'],(64,256)):
  r=records[label];c=validate_context(r['context']);req=validate_request(r['request']);ident=r['identity']
  require(ident['model_identity']==runtime_identity(pm,specs['Adapter_A']) and ident['physical_identity']['gpu_uuid']==uuid_now and req['physical_identity']==ident['physical_identity'] and req['token_ids']==task['token_ids'] and req['generation_policy']==task['policy'] and req['user_id']==task['user_id'] and req['adapter_id']==task['adapter_id'] and c['policy']==task['policy'] and c['cohort_sha256']==ev.json('ud/plan.json')['cohort_sha256'],'B0 request/runtime identity differs')
  require(r['projection_calls']==96 and r['real_projection_rows']==n*96,'actual real-row projection accounting differs')
  require(completions[0 if label=='S' else 1]['generated_token_ids']==[r['protocol_selected_token']],'protocol token receipt differs')
  selected=[x for x in layers if x['forward_id']==task['request_id']+'.s00'];require([x['layer'] for x in selected]==list(range(32)),'UD layer order differs')
  oldcond='SEMCACHE_INTEGRATED' if label=='S' else 'FULL_RECOMPUTE'
  original=[x for x in oldev.jsonl(oldcond+'/Adapter_A/ud/layers.jsonl') if x['forward_id']==task['request_id']+'.s00']
  require(len(original)==32,'historical CPU precision receipts missing')
  frames=[f['header'] for f in esframes if f['operation']=='send' and f['header']['kind']=='HIDDEN' and f['header']['request_id']==task['request_id']+'.s00']
  require([f['layer_id'] for f in frames]==list(range(32)) and all(f['shape']==[1,n,2560] and f['dtype']=='float16' and f['metadata']['r4c_forward']==c and f['metadata']['r5_multi']['request']==req for f in frames),'actual CPU LoRA received padded/mismatched rows')
  for i,(new,old) in enumerate(zip(selected,original)):
   require(new['context']==c and new['metrics']['actual_precision']==old['metrics']['actual_precision'] and new['metrics']['user_id']==task['user_id'] and new['metrics']['matrix_sha256']==p['adapter']['matrix_sha256'] and new['metrics']['hidden_payload_sha256']==frames[i]['payload_sha256'],'CPU precision/adapter/payload receipt differs')
  if label=='S':
   for row in padding:
    require(row['context']==c and row['direction']=='APPEND' and row['real_rows']==64 and row['extra_rows']==192 and row['native_rows']==256 and row['retained']==[0,64] and row['input_sha256']==row['real_region_sha256']==frames[row['layer']]['payload_sha256'] and row['zero_region_sha256']==digest_bytes(b'\0'*(192*2560*2)),'native-only source padding differs')


def runtime_identity(pm,spec):
 from .r5_multi_artifacts import runtime_view
 return runtime_view(pm,spec)['identity']
