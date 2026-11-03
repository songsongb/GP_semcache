"""Opt-in integrated native-only padding. Cache, LoRA and real context stay unchanged."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
from .communication import require,digest,digest_bytes
from .r3a_artifacts import read,write,pack
from .r3b_cache import raw
from .r3b_fixture import ArrayTensor
from .r5_i5_evidence import fresh,seal,jsonl
from .r5_i5r2_tensors import Evidence,metrics
from .r5_i5r2_analysis import source_provenance
from .r5_i5r2b_plan import prepare as base_prepare,validate_plan
MODES=('OFF','SOURCE_ONLY','SOURCE_AND_SUFFIX')

def gate(cfg,condition,c):
 require(cfg['mode'] in MODES,'unknown padding mode')
 if condition!='SEMCACHE_INTEGRATED' or cfg['mode']=='OFF' or c['pass_index']!=0:return None
 if c['parent_request_id']==cfg['source']:
  require(c['mode']=='COLD_MISS' and c['past_length']==0 and c['input_length']==len(c['target_token_ids'])==64,'invalid source phase')
  return 'APPEND'
 if c['parent_request_id']==cfg['target'] and cfg['mode']=='SOURCE_AND_SUFFIX':
  require(c['mode']=='PARTIAL_PREFIX_HIT' and c['past_length']==64 and c['input_length']==192 and len(c['target_token_ids'])==256 and c['source_request_id']==cfg['source'] and c['source_entry_id'],'invalid suffix/cache grant')
  return 'PREPEND'
 return None

def padded(native,x,direction,*,bias=True):
 if direction is None:return native(x),None
 n=x.shape[1];h=x.shape[2];require((direction,n) in (('APPEND',64),('PREPEND',192)),'padding shape/direction mismatch')
 offset=0 if direction=='APPEND' else 64
 if hasattr(x,'fixture_bytes'):
  zeros=[0.]*(256-n)*h;values=x.values+zeros if direction=='APPEND' else zeros+x.values
  enlarged=ArrayTensor(values,[1,256,h]);allout=native(enlarged)
  out=ArrayTensor(allout.values[offset*h:(offset+n)*h],[1,n,h])
 else:
  import torch
  require(x.dtype==torch.float16 and x.shape[0]==1,'native FP16 batch-one required')
  enlarged=x.new_zeros((1,256,h));enlarged[:,offset:offset+n].copy_(x)
  allout=native(enlarged);out=allout[:,offset:offset+n].contiguous()
 require(list(out.shape)==list(x.shape),'padded rows escaped projection')
 return out,dict(direction=direction,real_rows=n,native_rows=256,extra_rows=256-n,retained=[offset,offset+n],hidden=h,
  input_sha256=digest_bytes(raw(x)),padded_input_sha256=digest_bytes(raw(enlarged)),padded_output_sha256=digest_bytes(raw(allout)),output_sha256=digest_bytes(raw(out)),
  real_region_sha256=digest_bytes(raw(ArrayTensor(enlarged.values[offset*h:(offset+n)*h],[1,n,h]) if hasattr(x,'fixture_bytes') else enlarged[:,offset:offset+n])),
  zero_region_sha256=digest_bytes(raw(ArrayTensor((enlarged.values[n*h:] if direction=='APPEND' else enlarged.values[:offset*h]),[1,256-n,h]) if hasattr(x,'fixture_bytes') else (enlarged[:,n:] if direction=='APPEND' else enlarged[:,:offset]))),
  element_bytes=4 if hasattr(x,'fixture_bytes') else 2,extra_flops=(256-n)*(2*h*h+(h if bias else 0)),temporary_bytes=(256*h+256*h+n*h)*(4 if hasattr(x,'fixture_bytes') else 2))


def prepare(historical_plan,multi,review,c1_plan,c1_run,c1_audit,output,fixture=False):
 proof={'TEST_FIXTURE_ONLY':True}
 if not fixture:
  pe,re,ae=(Evidence(p,max_input_bytes=512*2**20) for p in (c1_plan,c1_run,c1_audit))
  proof=dict(plan=pe.verify(),run=re.verify(),audit=ae.verify());cp=pe.json('i5r2c1_plan.json');status=re.json('i5r2c1_status.json')
  require(status['status']=='COMPLETE' and re.json('i5r2c1_plan.json')==cp and status['experiment_id']==cp['experiment_id'] and not status['GPU_NOT_RUN'],'C1 run incomplete')
  ar=ae.json('i5r2c1_provenance.json');require(ar['independent_audit']=='PASS' and ar['current']==cp['source'],'C1 audit identity differs')
  proof['source']=source_provenance(dict(source_commit=cp['source']['commit'],sources=cp['source']['files']),expected_commit=cp['source']['commit'],fixture=False)
 with tempfile.TemporaryDirectory() as d:
  prototype=base_prepare(historical_plan,multi,Path(d)/'prototype',r2a_review=review,fixture=fixture)
  dataset=read(Path(d)/'prototype/i5_dataset_manifest.json')
 if not fixture:
  sp=Evidence(cp['s1_experiment']);sp.verify();oldpair=sp.json('i5r2bs1_plan.json')['historical_pair']['requests']
  require(cp['model_identity']==prototype['i5r1']['model_identity'] and [x['token_ids'] for x in oldpair]==[x['token_ids'] for x in prototype['blocks'][0]['tasks']],'C1/R1 model or controlled pair differs')
 out=fresh(output,(historical_plan,multi,*(() if fixture else (review,c1_plan,c1_run,c1_audit))))
 for mode in MODES:
  p=deepcopy(prototype);p.pop('experiment_id');s,t=p['blocks'][0]['tasks']
  p['i5r2c2']=dict(mode=mode,source=s['request_id'],target=t['request_id'],C1_provenance=proof,DIAGNOSTIC_ONLY=True)
  p['experiment_id']=digest(p);dest=out/mode;dest.mkdir()
  write(dest/'i5_workload_manifest.json',p);write(dest/'i5r2b_protocol.json',p['i5r2b']);write(dest/'i5_dataset_manifest.json',dataset);seal(dest)
 seal(out);return out

def validate(path):
 p=validate_plan(path);cfg=p['i5r2c2'];require(cfg['mode'] in MODES and cfg['DIAGNOSTIC_ONLY'],'padding plan invalid')
 s,t=p['blocks'][0]['tasks'];require(cfg['source']==s['request_id'] and cfg['target']==t['request_id'] and t['policy']['max_new_tokens']==2,'padding pair/generation differs');return p

def execute(plan,root,writer):
 from . import r5_i5r2b_capture as capture
 original=capture.DiagnosticDriver;oldhooks=capture.model_hooks;cfg=plan['i5r2c2']
 class Driver(original):
  @property
  def experimental_native_projection(self):
   return self.project_native if plan['condition']=='SEMCACHE_INTEGRATED' and cfg['mode']!='OFF' else None
  def project_native(self,c,layer,role,native,x):
   direction=gate(cfg,plan['condition'],c) if self.identity['adapter']=='Adapter_A' else None
   out,receipt=padded(native,x,direction,bias=not hasattr(x,'fixture_bytes') and native.__self__.bias is not None)
   if receipt:
    writer.append(self.output/'i5r2c2_projection.jsonl',dict(receipt,layer=layer,role=role,context=c,mode=cfg['mode']))
   return out
 @contextmanager
 def hooks(driver,store):
  handles=[]
  try:
   for name,module in [('attention',driver.adapter.layers[0].self_attn),('layer',driver.adapter.layers[0])]:
    def hook(m,args,kwargs,out,name=name):
     t=out[0] if isinstance(out,tuple) else out;t=t[:,-1:].detach()
     rid=store.current['request_id'];fn='tensors/c2_last_'+rid+'_'+name+'.bin';data=raw(t)
     writer.bytes(driver.output/fn,data,exclusive=True);writer.append(driver.output/'i5r2c2_last.jsonl',dict(request_id=rid,component=name,file=fn,shape=list(t.shape),dtype=str(t.dtype).removeprefix('torch.'),sha256=digest_bytes(data)))
    handles.append(module.register_forward_hook(hook,with_kwargs=True))
   with oldhooks(driver,store):yield
  finally:
   for h in handles:h.remove()
 try:capture.DiagnosticDriver=Driver;capture.model_hooks=hooks;capture.execute(plan,root,writer)
 finally:capture.DiagnosticDriver=original;capture.model_hooks=oldhooks

def serve(plan,output,writer):
 from .r5_i5r2b_capture import serve as original
 original(plan,output,writer)


def analyze(experiment,run,output):
 from .r5_i5_analysis import analyze as baseline
 p=validate(experiment);cfg=p['i5r2c2'];report=baseline(run);ev=Evidence(run,max_input_bytes=4*2**30);inventory=ev.verify()
 require(report['config'].get('i5r2c2')==cfg and report['config']['experiment_id']==p['experiment_id'],'run/plan mismatch')
 rows=report['request_metrics'];hit=next(x for x in rows if x['condition']=='SEMCACHE_INTEGRATED' and x['request_id']==cfg['target'])
 require(hit['reused_prefix_tokens']==64 and hit['source']['source_request_id']==cfg['source'] and not hit['cached_prefix_q_consumed'],'genuine 64-token physical HIT missing')
 expected_requests=[] if cfg['mode']=='OFF' else [cfg['source']]+([cfg['target']] if cfg['mode']=='SOURCE_AND_SUFFIX' else [])
 receipts=ev.jsonl('SEMCACHE_INTEGRATED/Adapter_A/es/i5r2c2_projection.jsonl',optional=not expected_requests)
 require([(r['context']['parent_request_id'],r['layer'],r['role']) for r in receipts]==[(rid,l,r) for rid in expected_requests for l in range(32) for r in 'qkv'],'padding receipt coverage/order differs')
 for condition in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE'):
  require(not ev.jsonl(condition+'/Adapter_A/es/i5r2c2_projection.jsonl',optional=True),'baseline was padded')
 for r in receipts:
  d=gate(cfg,'SEMCACHE_INTEGRATED',r['context']);n=r['context']['input_length'];offset=64 if d=='PREPEND' else 0
  require(r['direction']==d and r['real_rows']==n and r['native_rows']==256 and r['extra_rows']==256-n and r['retained']==[offset,offset+n],'padding/slicing receipt differs')
 frames=ev.jsonl('SEMCACHE_INTEGRATED/Adapter_A/es/wire.jsonl')
 for r in receipts:
  hidden=[f['header'] for f in frames if f['header']['kind']=='HIDDEN' and f['header']['layer_id']==r['layer'] and f['header']['request_id']==r['context']['parent_request_id']+'.s00']
  require(len(hidden)==1 and hidden[0]['payload_sha256']==r['input_sha256']==r['real_region_sha256'] and hidden[0]['shape']==[1,r['real_rows'],r['hidden']],'padding input/real-row TCP receipt mismatch')
  require(r['zero_region_sha256']==digest_bytes(b'\0'*(r['extra_rows']*r['hidden']*r['element_bytes'])),'dummy rows not zero')
 # Layer-0 saved/consumed K/V byte identity; all-layer metadata/cache proofs are checked by baseline().
 for role in 'kv':
  base='SEMCACHE_INTEGRATED/Adapter_A/es';cap=ev.json(base+'/i5r2b_capture.json')['entries']
  a=cap[cfg['source']+'/returned_prefix_'+role];b=cap[cfg['target']+'/consumed_prefix_'+role]
  require(ev.tensor(base,a)==ev.tensor(base,b),'consumed cached prefix differs')
 prefix=[]
 abase='FULL_RECOMPUTE/Adapter_A/es';cbase='SEMCACHE_INTEGRATED/Adapter_A/es'
 acap=ev.json(abase+'/i5r2b_capture.json')['entries'];ccap=ev.json(cbase+'/i5r2b_capture.json')['entries']
 for name in ['projection_input_q','norm_output']+[n+'_'+r for n in ('base','delta_on_ES','raw_qkv') for r in 'qkv']+['attention_output']:
  a=acap.get(cfg['target']+'/'+name);b=ccap.get(cfg['source']+'/'+name)
  if not a or not b or a['status']!='CAPTURED' or b['status']!='CAPTURED':
   prefix.append(dict(boundary=name,status='NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE'));continue
  require(a['shape']==b['shape'] and a['dtype']==b['dtype'],'Layer0 prefix alignment differs')
  prefix.append(dict(boundary=name,status='CAPTURED',positions=[0,64],metrics=metrics(ev.tensor(abase,a),ev.tensor(cbase,b),a['dtype'])))
 last=[]
 maps={c:{(x['request_id'],x['component']):x for x in ev.jsonl(c+'/Adapter_A/es/i5r2c2_last.jsonl',optional=p['TEST_FIXTURE_ONLY'])} for c in ('FULL_RECOMPUTE','SEMCACHE_INTEGRATED')}
 for key,a in maps['FULL_RECOMPUTE'].items():
  b=maps['SEMCACHE_INTEGRATED'].get(key);require(b is not None,'missing Layer0 diagnostic')
  last.append(dict(request_id=key[0],component=key[1],metrics=metrics(ev.tensor('FULL_RECOMPUTE/Adapter_A/es',a),ev.tensor('SEMCACHE_INTEGRATED/Adapter_A/es',b),a['dtype'])))
 from .r5_multi_analysis import fidelity
 by={(x['condition'],x['request_id']):x for x in rows}
 source_numerical=[]
 for a,b in (('FULL_RECOMPUTE','SEMANTIC_NO_REUSE'),('FULL_RECOMPUTE','SEMCACHE_INTEGRATED')):
  ra,rb=by[a,cfg['source']],by[b,cfg['source']]
  source_numerical.append(dict(comparison=a+'_vs_'+b,**fidelity(ra,rb,Path(run)/a/'Adapter_A/es',Path(run)/b/'Adapter_A/es')))
 out=fresh(output,(experiment,run));write(out/'i5r2c2_numerical.json',dict(target=report['numerical_fidelity'],priming_source=source_numerical))
 write(out/'i5r2c2_layer0.json',dict(source_vs_full_target_prefix=prefix,last_position=last,FIRST_OBSERVED_DIFFERENCE=next((x for x in prefix if x.get('metrics',{}).get('bitwise_equal') is False),next((x for x in last if not x['metrics']['bitwise_equal']),'NOT_LOCALIZED_IN_CAPTURED_LAYER0_BOUNDARIES')),scope='Layer0 same-request final real prefill position; earliest captured difference only, not root cause'))
 write(out/'i5r2c2_accounting.json',dict(padding=receipts,extra_projection_rows=sum(x['extra_rows'] for x in receipts),extra_flops=sum(x['extra_flops'] for x in receipts),
  peak_temporary_bytes=max((x['temporary_bytes'] for x in receipts),default=0),real_reused_tokens=64,
  semantics='Real prefix projection rows omitted remain in legacy metrics; extra padded Linear rows above must be subtracted from any compute savings claim',
  temporary_memory_scope='Conservative live input/output/cropped tensor byte estimate; allocator peak not isolated',
  projection_rows=[dict(condition=x['condition'],request_id=x['request_id'],real_prefill_rows=x['executed_prefill_projection_token_rows'],extra_padded_rows=sum(r['extra_rows'] for r in receipts if x['condition']=='SEMCACHE_INTEGRATED' and r['context']['parent_request_id']==x['request_id']),actual_native_prefill_rows=x['executed_prefill_projection_token_rows']+sum(r['extra_rows'] for r in receipts if x['condition']=='SEMCACHE_INTEGRATED' and r['context']['parent_request_id']==x['request_id'])) for x in rows],
  measured=[{k:x[k] for k in ('condition','request_id','application_e2e_ms','application_ttft_ms','model_execution_e2e_ms')} for x in rows],
  source_inclusive_e2e_ms={c:sum(x['application_e2e_ms'] for x in rows if x['condition']==c) for c in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED')}))
 source_pass=all(x['frozen_tolerance_pass'] and x['generated_tokens_equal'] and x['generated_text_equal'] for x in source_numerical)
 decision=dict(SOURCE_STRICT_FIDELITY='PASS' if source_pass else 'FAIL',TARGET_STRICT_FIDELITY=report['decision']['numerical_tolerance'],mode=cfg['mode'],TEST_FIXTURE_ONLY=p['TEST_FIXTURE_ONLY'],PHYSICAL_REUSE_PRESERVED='CPU_FIXTURE_ONLY' if p['TEST_FIXTURE_ONLY'] else 'PASS',
  STRICT_FIDELITY_RESTORED='CPU_FIXTURE_ONLY' if p['TEST_FIXTURE_ONLY'] else report['decision']['numerical_tolerance'] if source_pass and report['decision']['token_text_fidelity']=='PASS' and report['decision']['task_score_completeness']=='PASS' and all(q['absolute_delta']==0 for q in report['quality_deltas']) else 'FAIL',
  PERFORMANCE_BENEFIT='NOT_ESTABLISHED',ROOT_CAUSE_CONFIRMED=False,independent_audit=report['independent_audit'],inventory=inventory,
  NATIVE_PEFT_STRICT_PARITY=report['decision']['native_peft_strict_parity'],audit_scope='All-layer cache/TCP/shape/hash proofs plus Layer0 consumed KV bytes; not an all-layer full-tensor dump')
 write(out/'i5r2c2_decision.json',decision);seal(out);return decision
