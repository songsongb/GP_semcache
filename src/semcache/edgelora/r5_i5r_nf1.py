"""Immutable BE numerical forensics; no CUDA/Torch/model/encoder execution."""
import csv,json,math,re,struct,time
from collections import defaultdict
from pathlib import Path
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,THRESHOLDS,check_parity,write
from .r5_i5_evidence import fresh,seal
from .r5_i5r1_plan import source_identity
from .r5_i5r2_analysis import source_provenance
from .r5_i5_contracts import validate_request
from .r5_i5_protocol import context
from .r5_i5_data import validate_sample,SOURCES
from .r5_multi_artifacts import validate_registry,sharing
from .r5_i5r3bbe import economics
from .r5_i5r_nf1_tensors import StreamEvidence,aligned_ranges,compare,UNKNOWN,file_sha,safe_path

COMMIT='f7157c9b8b97fdce56f87dcbc3566cb677404bb8'
BASE=ROOT/'results/r5/i5r/controlled_hit/i5r3b/be'
PLAN=BASE/'plan_N3_20261106T064627Z'
RUN=BASE/'attempt_N3_20261106T064627Z/block_0_153343'
A='FULL_RECOMPUTE';B='SEMANTIC_NO_REUSE';C='SEMCACHE_INTEGRATED'
S='P000000S';T1='P000000T';T2='P000001T';R='P000000T_BE03';N='P000002T'
IDS=(S,T1,T2,R,N)
NAMESPACE=ROOT/'results/r5/i5r/controlled_hit/i5r3b/nf1'

def source_check(source,expected):
 return source_provenance(dict(source_commit=source['commit'],sources=source['files']),expected_commit=expected,fixture=False)

def retain(ev,name,key='request_id'):
 rows={}
 for r in ev.records(name):
  require(r[key] not in rows,'duplicate record identity');rows[r[key]]=r
  require(len(rows)<=16,'request row bound exceeded')
 return rows

def read_inputs(experiment,run,audit,verification,expected_commit=COMMIT):
 es={k:StreamEvidence(v) for k,v in dict(experiment=experiment,run=run,audit=audit,verification=verification).items()};receipts={k:e.verify() for k,e in es.items()}
 p=es['experiment'].json('i5r3bbe_protocol.json');provenance=source_check(p['source'],expected_commit)
 require(p['schema']=='I5R3B_BE_V1' and p['positive_requests']==3 and p['repetitions']==1 and p['label']=='CONTROLLED_REPEAT_AMORTIZATION','exact N=3 baseline required')
 helpers=('communication.py','r3a_artifacts.py','r4b2_analysis.py','r5_i5_protocol.py','r5_i5_contracts.py','r5_i5r3bbe.py')
 provenance['reused_helper_hashes']={name:file_sha(ROOT/'src/semcache/edgelora'/name) for name in helpers}
 require(all(value==p['source']['files']['src/semcache/edgelora/'+name] for name,value in provenance['reused_helper_hashes'].items()),'reused historical helper source differs; no source-check bypass')
 rv=es['run'];status=rv.json('i5r3bbe_attempt_status.json');require(status['status']=='COMPLETE' and status['block_index']==0 and status['source_commit']==expected_commit and status['execution_order']==p['execution_order'][0],'BE source/completion differs')
 exported=es['audit'].json('i5r3bbe_results.json');ad=es['audit'].json('i5r3bbe_independent_audit.json');vr=es['verification'].json('i5r3bbe_verification.json')
 require(ad['status']==vr['status']=='PASS' and ad['source']==exported['source']==vr['source']==p['source'] and not exported['GPU_NOT_RUN'] and exported['positive_requests']==3,'BE independent audit/source differs')
 require(vr['audit_inventory']==receipts['audit'] and ad['run_inventories']==exported['run_inventories'] and len(ad['run_inventories'])==1 and ad['run_inventories'][0]['receipt']==receipts['run'] and Path(ad['run_inventories'][0]['run']).resolve()==Path(run).resolve(),'BE inventory binding differs')
 require(len(exported['blocks'])==1 and exported['blocks'][0]['block_index']==0,'ambiguous exported block')
 old=exported['blocks'][0];plans={};cfgs={}
 for control in ('abc','a0'):
  child=StreamEvidence(Path(experiment)/control);child.verify();ex=child.json('i5_workload_manifest.json');q=dict(ex);eid=q.pop('experiment_id')
  require(digest(q)==eid and ex['i5r3bbe']['source']==p['source'] and not ex['TEST_FIXTURE_ONLY'],'BE child identity differs');plans[control]=ex
  cfg=rv.json(control+'_run/i5_execution_config.json');cfgs[control]=cfg
  require(cfg['sources']==p['source']['files'] and cfg['source_commit']==expected_commit and cfg['experiment_id']==eid and cfg['block']==ex['blocks'][0] and not cfg['fixture'] and Path(cfg['experiment_prepared']).resolve()==(Path(experiment)/control).resolve(),'raw execution/plan/source differs')
  raw_status=rv.json(control+'_run/i5_block_status.json');require(raw_status['status']=='COMPLETE' and raw_status['GPU_NOT_RUN'] is False,'GPU run incomplete')
  raw_audit=StreamEvidence(Path(run)/(control+'_run_audit'));raw_audit.verify();require(raw_audit.json('i5_independent_audit.json')==old['legacy_strict_audits'][control] and old['legacy_strict_audits'][control]['status']=='PASS','archived I5 audit differs')
 tasks={t['request_id']:t for t in cfgs['abc']['block']['tasks']};require(tuple(tasks)==IDS,'unexpected source/target/repeat identities')
 require(tasks[S]['policy']['max_new_tokens']==1 and all(tasks[r]['policy']['max_new_tokens']==8 for r in IDS[1:]),'historical generation policies differ')
 require(tasks[S]['token_ids']==tasks[T1]['token_ids'][:512]==tasks[T2]['token_ids'][:512] and len(tasks[S]['token_ids'])==512 and all(len(tasks[x]['token_ids'])==1280 for x in IDS[1:]) and tasks[N]['token_ids'][:512]!=tasks[S]['token_ids'],'token source/target lengths/prefix differ')
 for k in tasks[T1]:
  if k not in ('request_id','request_sequence_number'):require(tasks[T1][k]==tasks[R][k],'repeated request/sample/content differs')
 require(len({tasks[x]['sample_id'] for x in (T1,T2,N)})==3,'not three unique original samples')
 targets=cfgs['a0']['block']['tasks'];require([t['request_id'] for t in targets]==list(IDS[1:]),'A0 ordering differs')
 for t in targets:
  require(all(t[k]==tasks[t['request_id']][k] for k in t if k!='request_sequence_number'),'A0 target content differs')
 # Verify original normalized records without accumulating the entire dataset.
 manifest=rv.json('abc_run/i5_dataset_manifest.json')['qasper'];ds=StreamEvidence(manifest['prepared_path']);ds_receipt=ds.verify();require(ds_receipt['manifest_sha256']==manifest['prepared_manifest_sha256'],'dataset inventory differs')
 dm=ds.json('i5_dataset_manifest.json');require(dm['dataset']=='qasper' and dm['version']=='LongBench-v1' and dm['TEST_FIXTURE_ONLY'] is False and dm['source']==SOURCES['qasper'] and dm['split'] and re.fullmatch('[0-9a-f]{40,64}',dm['revision']) and dm['source_files'] and all(re.fullmatch('[0-9a-f]{64}',v) for v in dm['source_files'].values()),'real Qasper provenance required')
 require(all(manifest[k]==v for k,v in dm.items()),'copied dataset provenance differs')
 selected={};seen=set()
 for sample in ds.records('samples.jsonl'):
  validate_sample(sample);require(sample['id'] not in seen and sample['TEST_FIXTURE_ONLY'] is False,'duplicate/mixed original samples');seen.add(sample['id'])
  if sample['id'] in {t['sample_id'] for t in tasks.values()}:selected[sample['id']]=sample
 require(len(seen)==dm['samples'] and file_sha(ds.root/'samples.jsonl')==dm['samples_sha256'],'dataset count/SHA differs')
 for t in tasks.values():require(t['prepared']['sample']==selected.get(t['sample_id']),'immutable original sample/reference differs')
 rows={};requests={};prefills={};logit_steps=defaultdict(list);identities={};context_bytes=0
 reg=rv.json('abc_run/i5_adapter_registry.json');validate_registry(reg['registry'],reg['adapters'])
 for cond in (A,B,C):
  base='abc_run/'+cond;cp=rv.json(base+'/plan.json');result=rv.json(base+'/result.json')
  require(cp['sources']==p['source']['files'] and cp['source_commit']==expected_commit and cp['block']==cfgs['abc']['block'] and result['status']=='COMPLETE' and result['gpu_inference_performed'] is True,'condition snapshot/completion differs')
  gpu=rv.json(base+'/gpu_identity.json');require(gpu==old['legacy_strict_audits']['abc']['conditions'][cond]['gpu_identity'] and gpu['gpu_class_compatible'] and '3090' in gpu['execution_gpu_name'] and not gpu['historical_live_kv_restored'],'GPU receipt differs')
  identities[cond]=rv.json(base+'/Adapter_A/identity.json');ident=identities[cond]
  require(ident['model_identity']['model']=='facebook/opt-2.7b' and ident['model_identity']['dtype']=='float16' and ident['model_identity']['layers']==32 and ident['model_identity']['hidden_size']==2560 and ident['contract']['heads']==32 and ident['adapter']==reg['adapters']['Adapter_A'],'model/adapter/dimensions differ')
  require(ident['physical_identity']['base_fingerprint']==ident['model_identity']['base_weight_sha256'] and ident['physical_identity']['adapter_matrix_fingerprint']==ident['adapter']['matrix_sha256'] and ident['physical_identity']['gpu_uuid']==gpu['execution_gpu_uuid'],'physical/model/adapter identity differs')
  for method in ident['contract']['methods'].values():require(digest_bytes(method['source'].encode())==method['source_sha256'],'installed method text/source hash differs')
  rs=retain(rv,base+'/generation.jsonl');require(tuple(rs)==IDS,'generation request order differs')
  arrivals={};complete=[]
  for item in rv.records(base+'/requests.jsonl'):
   if item['kind']=='ARRIVAL':
    req=validate_request(item['request']);rid=req['request_id'];require(rid not in arrivals,'duplicate arrival');arrivals[rid]=req
   elif item['kind']=='COMPLETION':complete.append(item['request_id'])
  require(tuple(arrivals)==tuple(complete)==IDS,'arrival/completion ordering differs')
  for rid,t in tasks.items():
   req=arrivals[rid];r=rs[rid];require(req['token_ids']==t['token_ids'] and req['input_text']==t['input_text'] and req['generation_policy']==t['policy'] and req['sample_id']==t['sample_id'] and req['user_id']==t['user_id'] and req['adapter_id']=='Adapter_A' and req['physical_identity']==r['physical_identity']==ident['physical_identity'] and r['request_sha256']==req['request_sha256'],'actual request identity differs')
   require(r['status']=='COMPLETE' and 0<r['generated_tokens']==len(r['generated_token_ids'])<=8 and r['user_id']==req['user_id'] and r['adapter_id']==req['adapter_id'] and r['sample_id']==t['sample_id'] and r['scored']==t['scored'],'missing generation or row identity differs')
   require(r['matched_identity_sha256']==digest(dict(task=t,adapter_sha256=reg['adapters']['Adapter_A']['matrix_sha256'],tokenizer=r['physical_identity']['tokenizer'])),'matched request hash differs')
   if cond in (A,B):require(r['source'] is None and r['reused_prefix_tokens']==0,'baseline unexpectedly reused cache')
   rows[cond,rid]=r;requests[cond,rid]=req
  for step in rv.records(base+'/Adapter_A/es/r5_tokens.jsonl'):
   rid=step['request_id']
   if rid not in tasks:continue # Known warmup captures are not diagnostic targets.
   key=(cond,rid);r=rs[rid];i=step['step'];require(i==len(logit_steps[key]) and i<len(r['generated_token_ids']),'duplicate/missing generation steps')
   tokens=tasks[rid]['token_ids'][r['reused_prefix_tokens']:] if not i else [r['generated_token_ids'][i-1]];past=r['reused_prefix_tokens'] if not i else 1280+i-1
   if rid==S and i:past=512+i-1
   expected=context(rid,i,tokens,tasks[rid]['token_ids'],past,tasks[rid]['policy'],r['mode'],step['context']['cohort_sha256'],r['source'],context_limit=2048)
   require(step['context']==expected and step['position_observation']['positions']==expected['positions'] and step['position_observation']['past_length']==past and step['position_observation']['source']=='ACTUAL_OPT_POSITION_MODULE' and step['position_observation']['lookup_output_equal'] is True,'actual token/position/mask alignment differs')
   logit_steps[key].append(dict(step=i,sha256=step['logits_sha256'],selected_token_id=step['selected_token_id']))
   if i==0:
    context_bytes+=len(json.dumps(step));require(context_bytes<=32*2**20,'retained prefill metadata budget exceeded')
    require(len(step['qkv'])==96 and len(step['kv'])==32,'incomplete layer metadata')
    prefills[key]=step
  for rid in tasks:require(len(logit_steps[cond,rid])==rs[rid]['generated_tokens'],'missing logit step records')
 for cond in (B,C):require(identities[cond]['physical_identity']==identities[A]['physical_identity'] and identities[cond]['contract']==identities[A]['contract'],'matched A/B/C model/GPU/contract differs')
 source=rows[C,S]
 for rid in (T1,T2,R):
  r=rows[C,rid];grant=r['source'];require(grant and grant['source_request_id']==S and source['admission'] and grant['entry_id']==source['admission']['entry_id'] and grant['entry_id'] in r['cache_before']['lru_to_mru'] and grant['entry_id'] in r['cache_after']['lru_to_mru'] and r['admission'] is None and r['reused_prefix_tokens']==512 and r['skipped_prefix_projection_token_rows']==512*96 and not r['cached_prefix_q_consumed'] and source['completion_ns']<r['arrival_ns'],'three genuine completed/resident HIT receipts required')
  require(sharing(reg['registry'],source['user_id'],r['user_id'],'Adapter_A','Adapter_A') is None,'unauthorized reuse')
 require(rows[C,N]['source'] is None and rows[C,N]['decision']=='CACHE_MISS','negative control not a MISS')
 require(old['numerical']['thresholds']==THRESHOLDS,'frozen numerical thresholds differ')
 # Raw economic result is recomputed, never copied from the prompt/handoff.
 a0rows=retain(rv,'a0_run/'+A+'/generation.jsonl');economic=economics([rows[C,x] for x in IDS],[a0rows[x] for x in IDS[1:]])
 require(abs(economic['D_ms']-old['economics']['D_ms'])<1e-5,'archived D(3) disagrees with raw timing')
 return dict(evidence=rv,receipts=receipts,protocol=p,tasks=tasks,rows=rows,requests=requests,prefills=prefills,logit_steps=logit_steps,identities=identities,old=old,provenance=provenance,dataset=dict(path=str(ds.root),inventory=ds_receipt,manifest=dm),economics=economic,layers=32,heads=32,hidden=2560)

def tensor_ref(data,cond,rid,kind,layer,role):
 step=data['prefills'][cond,rid];n=len(step['context']['input_token_ids']);base='abc_run/'+cond+'/Adapter_A/es'
 if kind=='raw':item=step['qkv'].get(f'{layer}.{role}');shape=[1,n,data['hidden']]
 elif kind=='kv':item=step['kv'][layer].get(role);shape=[1,data['heads'],len(step['context']['target_token_ids']),data['hidden']//data['heads']]
 elif kind=='cached':item=data['rows'][cond,rid]['strict_cached_qkv'].get(f'{layer}.{role}');shape=[1,512,data['hidden']]
 else:item=step['components'].get(f'{layer}.{role}');shape=[1,n,data['hidden']]
 require(item is not None and item['shape']==shape and item['dtype']=='float16','missing or incompatible tensor metadata: '+str((cond,rid,kind,layer,role)))
 return base,item

def compare_all(data,deadline_seconds=7200):
 ev=data['evidence'];rows=[];start=time.monotonic()
 def add(family,left,right,layer,role,ra,rb,*,head_a=False,required=True,integrity=False):
  require(time.monotonic()-start<deadline_seconds,'CPU diagnostic deadline exceeded; no partial PASS')
  a,ai=tensor_ref(data,*left,layer,role);b,bi=tensor_ref(data,*right,layer,role)
  result=compare(ev,a,ai,b,bi,ra,rb,head_a=head_a,heads=data['heads'],required=required)
  rows.append(dict(family=family,layer=layer,role=role,reference_request=left[1],candidate_request=right[1],reference_kind=left[2],candidate_kind=right[2],integrity_check=integrity,**result))
  require(len(rows)<=4096,'comparison record bound exceeded')
 for rid in (T1,T2,R):
  src=data['prefills'][C,S]['context'];full=data['prefills'][A,rid]['context'];hit=data['prefills'][C,rid]['context']
  prefix,_=aligned_ranges(src,full,'PREFIX');fr,hr=aligned_ranges(full,hit,'SUFFIX')
  for layer in range(data['layers']):
   for role in 'qkv':
    add('PREFIX_TOTAL_QKV',(C,S,'raw'),(A,rid,'raw'),layer,role,prefix,prefix)
    add('SUFFIX_TOTAL_QKV',(A,rid,'raw'),(C,rid,'raw'),layer,role,fr,hr)
   for role in 'kv':
    add('PREFIX_SOURCE_ATTENTION_KV_VS_FULL',(C,S,'kv'),(A,rid,'kv'),layer,role,prefix,prefix)
    add('PREFIX_CONSUMED_ATTENTION_KV_VS_FULL',(C,rid,'kv'),(A,rid,'kv'),layer,role,prefix,prefix)
   for role in ('attention','ffn','layer'):
    add('PREFIX_MODULE_BOUNDARIES',(C,S,'component'),(A,rid,'component'),layer,role,prefix,prefix)
    add('SUFFIX_MODULE_BOUNDARIES',(A,rid,'component'),(C,rid,'component'),layer,role,fr,hr)
 for rid in (T1,T2,R):
  for layer in range(data['layers']):
   for role in 'qkv':add('SOURCE_RAW_VS_STORED_CACHE',(C,S,'raw'),(C,rid,'cached'),layer,role,(0,512),(0,512),integrity=True)
   for role in 'kv':
    add('SOURCE_ATTENTION_KV_VS_CONSUMED',(C,S,'kv'),(C,rid,'kv'),layer,role,(0,512),(0,512),integrity=True)
    add('CACHED_RAW_HEAD_TRANSFORM_VS_CONSUMED',(C,rid,'cached'),(C,rid,'kv'),layer,role,(0,512),(0,512),head_a=True,integrity=True)
    inp=data['prefills'][C,rid]['input_kv'][layer]['kv'.index(role)];src=data['prefills'][C,S]['kv'][layer][role]
    require(inp=={k:v for k,v in src.items() if k!='file'},'actual input-past KV digest/shape differs from source')
 for cond in (A,C):aligned_ranges(data['prefills'][cond,T1]['context'],data['prefills'][cond,R]['context'],'MATCHED')
 for layer in range(data['layers']):
  for role in 'qkv':
   add('SOURCE_FULL_A_VS_C_CONTROL',(A,S,'raw'),(C,S,'raw'),layer,role,(0,512),(0,512))
   add('REPEAT_FULL_PREFIX',(A,T1,'raw'),(A,R,'raw'),layer,role,(0,512),(0,512))
   add('REPEAT_FULL_SUFFIX',(A,T1,'raw'),(A,R,'raw'),layer,role,(512,1280),(512,1280))
   add('REPEAT_HIT_SUFFIX',(C,T1,'raw'),(C,R,'raw'),layer,role,(0,768),(0,768))
  for role in ('q','k','v','attention','ffn','layer'):
   kind='raw' if role in 'qkv' and len(role)==1 else 'component'
   add('NEGATIVE_CONTROL',(A,N,kind),(C,N,kind),layer,role,(0,1280),(0,1280),required=False)
 return rows

def logits(data):
 ev=data['evidence'];blobs={};result=[]
 for cond in (A,B,C):
  for rid in IDS:
   row=data['rows'][cond,rid];path=safe_path(ev.root,'abc_run/'+cond+'/Adapter_A/es/'+row['logits_file']);size=path.stat().st_size
   require(size==math.prod(row['logits_shape'])*4 and row['logits_shape']==[1,row['generated_tokens'],50272] and size<=16*2**20 and file_sha(path)==row['logits_sha256'],'full-vocabulary logit shape/SHA differs')
   require(str(path.relative_to(ev.root)) in ev.inventory and ev.inventory[str(path.relative_to(ev.root))]==row['logits_sha256'],'logit inventory binding differs')
   blobs[cond,rid]=path
   with path.open('rb') as stream:
    for step in data['logit_steps'][cond,rid]:
     chunk=stream.read(50272*4);require(digest_bytes(chunk)==step['sha256'],'per-step logit hash differs')
     values=[x[0] for x in struct.iter_unpack('<f',chunk)];require(all(math.isfinite(x) for x in values) and max(range(50272),key=values.__getitem__)==step['selected_token_id']==row['generated_token_ids'][step['step']],'nonfinite/incorrect greedy logit receipt')
 for ca,cb,label in ((A,B,'FULL_RECOMPUTE_vs_SEMANTIC_NO_REUSE'),(B,C,'SEMANTIC_NO_REUSE_vs_SEMCACHE_INTEGRATED'),(A,C,'FULL_RECOMPUTE_vs_SEMCACHE_INTEGRATED'),(A,A,'REPEAT_FULL'),(C,C,'REPEAT_HIT')):
  for rid in ((T1,) if label.startswith('REPEAT') else IDS):
   other=R if label.startswith('REPEAT') else rid;ra=data['rows'][ca,rid];rb=data['rows'][cb,other];steps=[];skipped=[]
   with blobs[ca,rid].open('rb') as fa,blobs[cb,other].open('rb') as fb:
    for i in range(max(ra['generated_tokens'],rb['generated_tokens'])):
     x=fa.read(50272*4);y=fb.read(50272*4)
     if not x or not y or ra['generated_token_ids'][:i]!=rb['generated_token_ids'][:i]:skipped.append(dict(step=i,reason='UNMATCHED_GENERATION_HISTORY'));continue
     xv=[v[0] for v in struct.iter_unpack('<f',x)];yv=[v[0] for v in struct.iter_unpack('<f',y)]
     steps.append(dict(step=i,**check_parity(yv,xv,[1,1,50272])))
   record=dict(comparison=label,request_id=other,reference_request_id=rid,steps=steps,skipped=skipped,generated_tokens_equal=ra['generated_token_ids']==rb['generated_token_ids'],generated_text_equal=ra['generated_text']==rb['generated_text'],bitwise_logits_equal=file_sha(blobs[ca,rid])==file_sha(blobs[cb,other]),frozen_tolerance_pass=bool(steps) and not skipped and all(s['passed'] for s in steps))
   result.append(record)
   if ca!=cb and data['tasks'][rid]['scored']:
    exported=[n for n in data['old']['numerical']['comparisons'] if n['comparison']==label and n['request_id']==rid]
    require(len(exported)==1,'missing/duplicate archived numerical comparison')
    require(all(exported[0][k]==record[k] for k in ('steps','skipped','generated_tokens_equal','generated_text_equal','bitwise_logits_equal','frozen_tolerance_pass')),'frozen full-vocabulary fidelity does not reproduce archived result')
 scored=[r for r in result if not r['comparison'].startswith('REPEAT') and data['tasks'][r['request_id']]['scored']]
 numerical_gate='PASS' if all(r['frozen_tolerance_pass'] for r in scored) else 'FAIL'
 token_gate='PASS' if all(r['generated_tokens_equal'] and r['generated_text_equal'] for r in scored) else 'FAIL'
 require(numerical_gate==data['old']['verdicts']['NUMERICAL_FIDELITY'] and token_gate==data['old']['verdicts']['TOKEN_TEXT_FIDELITY'],'aggregate historical fidelity gate does not reproduce')
 return dict(thresholds=THRESHOLDS,comparisons=result,scope='Recorded full-vocabulary float32 exports of actual FP16 model logits; no inference or missing-value regeneration',historical_numerical_gate=data['old']['verdicts']['NUMERICAL_FIDELITY'],recomputed_numerical_gate=numerical_gate,recomputed_token_text_gate=token_gate)

def summarize(records):
 groups=defaultdict(list)
 for r in records:groups[r['family']].append(r)
 groups['PREFIX_PATH_INSPECTION']=[r for r in records if r['family'] in ('PREFIX_TOTAL_QKV','PREFIX_SOURCE_ATTENTION_KV_VS_FULL','PREFIX_MODULE_BOUNDARIES')]
 groups['SUFFIX_PATH_INSPECTION']=[r for r in records if r['family'] in ('SUFFIX_TOTAL_QKV','SUFFIX_MODULE_BOUNDARIES')]
 out={};rank=lambda r:(r['layer'],0 if r['reference_kind']=='raw' else 1 if r['reference_kind']=='kv' else {'attention':2,'ffn':3,'layer':4}.get(r['role'],5))
 for family,rs in groups.items():
  values=[r for r in rs if r['status']=='OBSERVED_VALUES'];unequal=[r for r in values if not r['metrics']['bitwise_equal']];fail=[r for r in values if not r['metrics']['activation_tolerance_pass']]
  def first(items):
   if not items:return None
   key=min(map(rank,items));return [dict(layer=r['layer'],role=r['role'],reference_request=r['reference_request'],candidate_request=r['candidate_request'],max_abs=r['metrics']['max_abs'],relative_l2=r['metrics']['relative_l2']) for r in items if rank(r)==key]
  out[family]=dict(comparisons=len(rs),value_comparisons=len(values),hash_only=len(rs)-len(values),hash_only_unequal=sum(r.get('bitwise_equal_by_full_sha256') is False for r in rs),hash_only_equal=sum(r.get('bitwise_equal_by_full_sha256') is True for r in rs),bitwise_differences=len(unequal),frozen_activation_failures=len(fail),first_observed_bitwise_difference=first(unequal),first_observed_tolerance_failure=first(fail))
 return out

def hypotheses(summary):
 def differs(name):return summary.get(name,{}).get('bitwise_differences',0)>0
 return dict(H1=dict(status='SUPPORTED_HYPOTHESIS' if differs('PREFIX_SOURCE_ATTENTION_KV_VS_FULL') else 'NOT_SUPPORTED_BY_RECORDED_VALUES',observation='Standalone-source vs full-prefix attention-layout KV comparison; source/storage integrity is a separate comparison',causal_logit_contribution='UNRESOLVED'),H2=dict(status='UNRESOLVED',observed_suffix_total_qkv_difference=differs('SUFFIX_TOTAL_QKV'),observation='1280-full vs 768-suffix total QKV; differing outputs alone do not isolate projection shape. Base Linear/LoRA/input-state causes not separated'),H3=dict(status='UNRESOLVED',observation='Module attention/FFN/layer outputs are available; internal QK/mask/softmax/PV, live strides, projection inputs, Base/Delta boundaries are not captured by normal I5'),source_no_cache_control=dict(A_source_matches_C_source=not differs('SOURCE_FULL_A_VS_C_CONTROL'),interpretation='If source QKV matches across conditions, its full-target prefix difference also exists within no-cache A; otherwise this equivalence is not established'),repeat_control=dict(status='OBSERVED',source_discrepancy_reproduction='SUPPORTED_BY_EXACT_REPEAT_FULL_PREFIX' if not differs('REPEAT_FULL_PREFIX') else 'REPEAT_PREFIX_VALUES_DIFFER',suffix_discrepancy_reproduction='SUPPORTED_BY_EXACT_REPEAT_FULL_AND_HIT_SUFFIX' if not differs('REPEAT_FULL_SUFFIX') and not differs('REPEAT_HIT_SUFFIX') else 'REPEAT_SUFFIX_VALUES_DIFFER',independent_samples=1),ROOT_CAUSE='UNRESOLVED',ROOT_CAUSE_CONFIRMED=False,NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',inspection_order='Layer and observed boundary class only; first recorded unequal QKV is not the first causal arithmetic operation')

def phase2():
 return dict(status='PLANNING_ONLY_NOT_EXECUTED',requires_explicit_GPU_approval=True,prerequisite='Inspect NF1 value coverage before any new forward; reuse existing full Raw QKV/KV first',cases=dict(F='Unchanged full 1280-token distributed reference, same IDs, policy/adapter/masks/positions',H='Genuine independently completed/admitted Source512 and actual suffix768 HIT, unchanged production controller/TCP',O='DIAGNOSTIC_ONLY attention replay with Full-target prefix KV replacing source prefix KV and fixed observed H suffix operands; direct native attention, no controller HIT flag/grant/admission and no performance validity',P='DIAGNOSTIC_ONLY actual native Layer0 Linear on verified identical row values in full1280 and suffix768 shapes; same weights/bias/dtype, balanced same-shape repeats'),intervention_points=['If native-mask/layout gap blocks replay, bounded opt-in hooks at Layer0 self_attn pre/post; actual additive mask and Q/K/V head layouts before native QK, native masked scores, softmax/PV and out_proj boundaries. Existing C3 observer must be explicitly extended/validated for q768 x k1280 and actual HF past, not reused assuming square attention.','If projection cause is not isolated, passive physical_path miss() capture at projection x, native(x), actual CPU UD input/A/B/scaled delta and returned device delta, real base+delta total. Preserve CPU LoRA/TCP arithmetic.','O uses r4b2_kv.construct_past on request-owned Full prefix copies only in an isolated attention diagnostic adapter. No oracle tensors enter global cache; normal QualityDriver/ForwardOrder are not bypassed with fake grants.'],gates=['All source/model/adapter/backend identities pinned; first reproduce recorded F/H module output bitwise before historical localization.','Matching absolute positions, real native causal masks, heads/strides/dtypes and Q scaling required. Missing masks are not silently synthesized as historical evidence.','Two warmups/six balanced replay repetitions; compare same-shape stability before shape attribution.','If masks/inputs/layout from new captures cannot reproduce history, label new-run-only evidence; historical localization remains UNRESOLVED.'],falsifiers=dict(H1='All measured source-prefix KV equal full-prefix KV falsifies this recorded-value difference; O failing to repair output with equivalent other operands shows prefix difference alone insufficient.',H2='Matching Layer0 input and Base/Delta/Total suffix values bound projection shape as the first cause; unequal input does not implicate Linear kernels.',H3='Matching prefix/suffix QKV, masks/layout/scaling with differing native attention intermediates localizes attention execution, but does not identify a cuBLAS algorithm.'),bounds=dict(gpus=1,gpu_class='RTX3090 24GB',model_loads=1,CPU_cores=4,Slurm_RAM_GiB=64,layers=[0],generation_policy='Original source one token; target up to eight, unchanged greedy policy',GPU_runtime_seconds=900,Slurm_minutes=30,capture_bytes=512*2**20,attention_matrix_limit='Layer0 only; selected first differing query/key regions after full hash validation'),audit='Independent CPU inventory/source verification, exact aligned operand receipts, native replay acceptance, Base/Delta separation and frozen full-vocabulary comparisons; oracle results cannot authorize legitimate reuse or speedup')

def prior_research():
 return dict(scope='Source/report review and operator handoffs; prior GPU exports are not newly audited by NF1',studies=[
  dict(stage='R2B/BS1',tested='64 vs 256 actual native Linear; matching input and CPU delta control, repeats, FP16/FP32',handoff='Native base projection was the first differing recorded arithmetic boundary; exact GPU mechanism not established',limit='Does not establish 512/1280/768 behavior'),
  dict(stage='C1/C2',tested='Fixed-shape native Linear, then integrated source/suffix projection padding',handoff='C2 Layer0 prefix QKV matched full reference while attention output/final logits still differed',limit='Linear alignment is not whole-model fidelity recovery'),
  dict(stage='C3/B0',tested='Bounded full operand recapture and native eager attention replay',handoff='Historical module output reproduced; scaled QK scores were the first observed internal difference at earlier lengths',limit='No specific CUDA/cuBLAS mechanism was proved'),
  dict(stage='C4',tested='Framework for query/key shape, stride and FP32 controls',handoff='Implementation available; no C4 GPU export supplied to NF1',limit='Do not assume completion or conclusions from implementation'),
  dict(stage='3B/BE',tested='Real unoptimized Source512/Target1280/Suffix768 KV reuse',handoff='Three HITs, one negative MISS, token/text match, numerical FAIL; NF1 must reproduce from sealed values',limit='Existing module boundaries do not separate native base from LoRA or expose live layouts')])

def analyze(experiment,run,audit,verification,output_dir,expected_commit=COMMIT,deadline_seconds=7200):
 require(60<=deadline_seconds<=14400,'CPU deadline outside bounded 60..14400 seconds')
 data=read_inputs(experiment,run,audit,verification,expected_commit);records=compare_all(data,deadline_seconds);numerical=logits(data);summary=summarize(records);hyp=hypotheses(summary)
 integrity=all(r['metrics']['bitwise_equal'] for r in records if r['integrity_check'])
 available=dict(status='HISTORICAL_EXPORTS_AND_NEW_CPU_VALUE_FORENSICS',input_inventories=data['receipts'],original_strides=UNKNOWN,base_projection_vs_CPU_delta='NOT_CAPTURED',actual_internal_additive_mask='NOT_CAPTURED: only actual input mask/positions metadata retained',internal_attention_scores_softmax_PV='NOT_CAPTURED',full_token_total_QKV='STRICT_AUDIT source/positive/repeat, metadata-validated before comparison',negative='Scalable hash-only activations allowed; full-vocabulary logit files required',capture_scope='All aligned tokens, no sampling; normal BE full-token captures distinct from old last-position traces')
 available['backend_coverage']=dict(versions=data['identities'][C]['contract'].get('versions',UNKNOWN),actual_TF32_determinism_and_cuBLAS_algorithm_receipts=UNKNOWN)
 proposal=phase2()
 proposal['whole_model_oracle_followup']=dict(status='PROPOSAL_ONLY_CONDITIONAL_ON_LAYER0_GAP',intervention='Request-owned per-layer Full-target KV slices 0:512 passed as actual past_key_values to a diagnostic-only native model forward on real suffix IDs 512:1280; same actual positions/mask and greedy policy. No cache insertion or HIT claim.',CPU_UD='A separate explicitly versioned ORACLE_DIAGNOSTIC_ONLY endpoint/session must validate real user/matrix identity and an oracle provenance receipt, then reuse unchanged cpu_deltas. Existing production ForwardOrder/REGISTER_PREFIX guards must not be bypassed with a fake source grant.',audit='Compare F/H/O full-vocabulary logits and all input-past fingerprints; direct oracle results cannot qualify physical reuse or serving performance.',cost='Reuse already recorded Full prefix values when verified; all-layer prefix KV is about160 MiB, keep disk captures bounded by references/hashes; no second model copy. Separate review before extending beyond Layer0.')
 outputs=dict(nf1_prior_research=prior_research(),nf1_layer_summary=summary,nf1_numerical_fidelity=numerical,nf1_evidence_availability=available,nf1_hypotheses=hyp,nf1_phase2_proposal=proposal,nf1_source_provenance=dict(historical=data['provenance'],analysis_source=source_identity(fixture=True),dataset=data['dataset']),nf1_decision=dict(status='FORENSICS_COMPLETE' if integrity else 'CACHE_INTEGRITY_REPRODUCTION_FAILED',CPU_ONLY=True,GPU_EXECUTED=False,historical_export_verdicts=data['old']['verdicts'],recomputed_economics=data['economics'],source_cache_integrity_reproduced=integrity,root_cause='UNRESOLVED'))
 require(sum(len(json.dumps(v).encode()) for v in outputs.values())+len(json.dumps(records).encode())<=24*2**20,'compact diagnostic output budget exceeded')
 out=Path(output_dir).resolve();require(not Path(output_dir).is_symlink() and (not out.is_relative_to(ROOT) or out.is_relative_to(NAMESPACE)),'fresh NF1 namespace required')
 require(not any((parent/'i5_output_manifest.json').exists() for parent in out.parents),'output would modify an existing sealed ancestor')
 out=fresh(out,(experiment,run,audit,verification,data['dataset']['path'],data['dataset']['manifest']['source_path']))
 for name,value in outputs.items():write(out/(name+'.json'),value)
 write(out/'nf1_tensor_comparisons.json',records)
 with (out/'nf1_tensor_comparisons.csv').open('x',newline='') as stream:
  fields=['family','layer','role','reference_request','candidate_request','status','bitwise_equal','max_abs','mean_abs','relative_l2','differing_values','bitwise_differing_elements','elementwise_tolerance_violations'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
  for r in records:writer.writerow({**{k:r[k] for k in fields if k in r},**{k:r['metrics'][k] for k in fields if isinstance(r['metrics'],dict) and k in r['metrics']}})
 lines=['# NF1 immutable evidence forensics','','CPU-only new tensor/logit comparisons over archived GPU evidence. No new inference or historical audit rerun.','',f'Historical source: {expected_commit}',f'Cache integrity reproduced: {integrity}',f'Recomputed D(3): {data["economics"]["D_ms"]} ms','','## Observed comparisons','']
 for family,value in summary.items():lines.append(f'- {family}: {value["value_comparisons"]} value comparisons; {value["bitwise_differences"]} unequal; {value["frozen_activation_failures"]} frozen activation failures; hash-only={value["hash_only"]} (unequal hashes={value["hash_only_unequal"]}); first={value["first_observed_bitwise_difference"]}')
 lines+=['','## Supported hypotheses and limitations','',json.dumps(hyp,indent=2),'','Raw QKV means Base + CPU LoRA total before attention scaling. K/V head-layout comparisons are separately labeled. Strides and Base/Delta/hidden input values are unobserved; no CUDA kernel mechanism is proved.','Repeat equality is an execution control, not another independent sample. Negative hash-only activation evidence cannot supply error magnitudes. Full-vocabulary frozen logit comparisons remain separate from activation tolerance and generated-token equality.','','## Phase 2 (not executed)','','Read nf1_phase2_proposal.json: F/H native replay gate, isolated oracle prefix O and projection-shape P, conditional on actual evidence gaps. No oracle cache is a legitimate HIT or speedup result.']
 (out/'nf1_report.md').write_text('\n'.join(lines)+'\n');seal(out);require(integrity,'source/storage/consumption mismatch; diagnostic sealed, no PASS');return outputs['nf1_decision']
