"""Unoptimized I5 runtime workloads; no semantic/cache/inference policy changes."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,read,write,sha
from .r5_i5_evidence import fresh,seal
from .r5_i5r2_tensors import Evidence
from .r5_i5r2_analysis import source_provenance
from .r5_i5r1_plan import source_identity
from .r5_i5r3a import compute as compute_3a,summary_record,inputs as inputs_3a
from .r5_i5_workload import prepare_prompt,requests,resources,CONDITIONS,ORDERS,SCALE_SEMANTICS,routing
from .r5_multi_artifacts import sharing,validate
from .r5_i5r3b_transport import preflight as transport_preflight,LABEL

SCHEMA='I5R3B_UNOPTIMIZED_V1'
NAMESPACE=ROOT/'results/r5/i5r/controlled_hit/i5r3b'
CAPACITY=512*2**20

def new_output(path,protected):
 p=Path(path).resolve();require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'3B outputs require fresh dedicated namespace')
 return fresh(path,protected)

def archived_source(source):
 return source_provenance(dict(source_commit=source['commit'],sources=source['files']),expected_commit=source['commit'],fixture=False)

def history(plan,audit,verification,expected_commit):
 p,result=compute_3a(plan);require(p['source']['commit']==expected_commit,'3A pinned source mismatch')
 ae=Evidence(audit);receipt=ae.verify();ve=Evidence(verification);vr=ve.verify();a=ve.json('i5r3a_independent_audit.json');s=ae.json('i5r3a_summary.json')
 require(a['status']=='PASS' and a['scope']=='CPU static recomputation only' and a['real_runtime_hit_status']=='NOT_EVALUATED' and a['plan_id']==p['plan_id'] and a['inventory']==receipt,'3A independent verification differs')
 require(ae.json('i5r3a_results.json')==result and ae.json('i5r3a_dataset_provenance.json')==p['datasets'] and s==summary_record(p,result,s['analysis_source']),'3A audit recomputation differs')
 archived_source(s['analysis_source']);archived_source(a['verification_source'])
 return p,result,dict(plan=Evidence(plan).verify(),audit=receipt,verification=vr,source=p['source'],scope='STATIC_ONLY; no historical physical HIT inferred')

def lcp(a,b):
 return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))

def shared_pairs(prompts,users):
 require(len({p['sample']['id'] for p in prompts})==len(prompts),'duplicate sample identity')
 found=[]
 for a,b in combinations(prompts,2):
  sa,sb=a['sample'],b['sample'];ra,rb=a['route'],b['route']
  if sa['context']!=sb['context'] or sa['question']==sb['question'] or a['token_ids']==b['token_ids']:continue
  common=lcp(a['token_ids'],b['token_ids'])
  denied=sharing(users,ra['user_id'],rb['user_id'],ra['adapter_id'],rb['adapter_id'])
  if denied is None and common>=512 and min(len(a['token_ids']),len(b['token_ids']))>512:
   found.append(dict(a=a,b=b,common_tokens=common,context_sha256=digest_bytes(sa['context'].encode()),adapter=ra['adapter_id']))
 return sorted(found,key=lambda x:digest([x['context_sha256'],x['a']['sample']['id'],x['b']['sample']['id']]))

def shared_statistics(pairs):
 return dict(authorized_shared_pairs_at_least_512=len(pairs),authorized_shared_pairs_at_least_1000=sum(p['common_tokens']>=1000 for p in pairs),independent_contexts_at_least_1000=len({p['context_sha256'] for p in pairs if p['common_tokens']>=1000}),by_adapter={a:sum(p['adapter']==a for p in pairs) for a in ('Adapter_A','Adapter_B','Adapter_C')})

def selection(prompts,users,profile,adapter='Adapter_A'):
 require(profile in ('B1','B2_SMOKE','B2_THREE_ADAPTER'),'unsupported profile')
 pairs=shared_pairs(prompts,users)
 report=shared_statistics(pairs)
 if profile=='B1':
  options=[p for p in prompts if p['route']['adapter_id']==adapter and len(p['token_ids'])==256]
  require(options,'INSUFFICIENT_B1_ELIGIBLE: exact 256-token Qasper prompt required')
  return [dict(targets=[sorted(options,key=lambda p:digest(p['sample']['id']))[0]],context_sha256=digest_bytes(sorted(options,key=lambda p:digest(p['sample']['id']))[0]['sample']['context'].encode()),adapter=adapter)],report
 chosen=[];seen=set()
 for aid in (('Adapter_A','Adapter_B','Adapter_C') if profile=='B2_THREE_ADAPTER' else (adapter,)):
  eligible=[p for p in pairs if p['adapter']==aid and p['context_sha256'] not in seen]
  require(eligible,'INSUFFICIENT_AUTHORIZED_SHARED_CONTEXT: '+aid+'; '+str(report))
  item=eligible[0];chosen.append(item);seen.add(item['context_sha256'])
 negative_ids=set();negative_contexts=set()
 for item in chosen:
  a,b=item['a'],item['b'];route=a['route'];prefix=a['token_ids'][:512]
  options=[p for p in prompts if digest_bytes(p['sample']['context'].encode()) not in seen|negative_contexts and p['sample']['id'] not in negative_ids and p['token_ids'][:512]!=prefix and sharing(users,route['user_id'],p['route']['user_id'],route['adapter_id'],p['route']['adapter_id']) is None]
  require(options,'INSUFFICIENT_NEGATIVE_CONTROL: different context, same authorized original adapter required')
  negative=sorted(options,key=lambda p:digest(p['sample']['id']))[0];negative_ids.add(negative['sample']['id']);negative_contexts.add(digest_bytes(negative['sample']['context'].encode()))
  item['targets']=[a,b,negative]
 return chosen,report

def checks(block,users,seed):
 tasks=block['tasks'];require(2<=len(tasks)<=20 and tasks[0]['role']=='CONTROLLED_STANDALONE_SOURCE' and not tasks[0]['scored'],'missing first completed standalone source lifecycle')
 require([t['request_sequence_number'] for t in tasks]==list(range(len(tasks))) and len({t['request_id'] for t in tasks})==len(tasks),'duplicate/reordered requests')
 require(block['track']=='CONTROLLED' and block['cache_resets_per_condition']==1 and block['cache_lifetime_requests']==block['physical_requests_per_condition']==len(tasks),'cache reset/lifetime boundary differs')
 source=tasks[0];p=block['prefix_length'];require(p in (64,512) and len(source['token_ids'])==p and source['policy']['max_new_tokens']==1,'invalid source size/generation')
 require(source['token_ids']==source['prepared']['token_ids'][:p] and source['strict'],'source is not actual standalone sliced prefix')
 scored=tasks[1:];require(len({t['sample_id'] for t in scored})==len(scored),'duplicate/repeated scored sample')
 proof=[]
 for i,t in enumerate(tasks):
  require(t['adapter_id']==source['adapter_id'] and sharing(users,source['user_id'],t['user_id'],source['adapter_id'],t['adapter_id']) is None,'unauthorized sharing')
  route=routing(t['prepared']['sample'],seed);require(all(t[k]==v for k,v in route.items()),'forced routing prohibited')
  require(len(t['token_ids'])+t['policy']['max_new_tokens']<=2048 and t['policy']['max_new_tokens']==(8 if t['scored'] else 1),'context/generation bound')
  require(t['prepared']['sample']['id']==t['sample_id'] and t['group_id']==t['prepared']['sample']['group_id'],'original identity changed')
  require(not any(k in t for k in ('forced_hit','selected_entry_id','cache_grant')),'no controller grant may enter workload')
  if not i:continue
  require(t['scored'] and t['token_ids']==t['prepared']['token_ids'] and t['request_sequence_number']>source['request_sequence_number'],'target identity/order mismatch')
  negative=t['role']=='NEGATIVE_TARGET';exact=source['token_ids']==t['token_ids'][:p] and len(t['token_ids'])>p
  require(exact is not negative,'positive/negative exact-prefix constraint differs')
  require(not any(k in t for k in ('forced_hit','selected_entry_id','cache_grant')),'no controller grant may enter workload')
  if not negative:require(t['strict'],'positive physical target must have strict tensor audit')
  proof.append(dict(source_request_id=source['request_id'],target_request_id=t['request_id'],source_original_id=source['sample_id'],target_original_id=t['sample_id'],source_user=source['user_id'],target_user=t['user_id'],adapter=t['adapter_id'],target_tokens=len(t['token_ids']),source_tokens=p,source_sha256=digest(source['token_ids']),target_sha256=digest(t['token_ids']),exact_prefix=exact,negative_control=negative,source_raw_qkv_bytes=p*491520,source_capacity_eligible=p*491520<=CAPACITY,positions='zero-based actual request; all-ones mask',completion_before_target='REQUIRED_RUNTIME_RECEIPT_NOT_STATIC_PROOF',cache_lifetime='SAME_INDEPENDENT_BLOCK',runtime_HIT_authorized_by_static_check=False))
 if p==512:
  require(len(scored)==3 and scored[0]['prepared']['sample']['context']==scored[1]['prepared']['sample']['context'] and scored[0]['prepared']['sample']['question']!=scored[1]['prepared']['sample']['question'],'B2 distinct genuine shared-context questions required')
  require(scored[2]['role']=='NEGATIVE_TARGET' and scored[2]['prepared']['sample']['context']!=scored[0]['prepared']['sample']['context'],'negative context differs')
 require(block['resources']['per_adapter_capacity_bytes']==CAPACITY,'capacity policy changed')
 return proof

def build_blocks(chosen,profile,special,tok,users,seed):
 blocks=[];prefix=64 if profile=='B1' else 512
 for ordinal,item in enumerate(chosen):
  ps=deepcopy(item['targets'])
  for p in ps:p.update(strict=True,max_new_tokens=8)
  if len(ps)==3:ps[-1]['strict']=False
  ps[0]['source_semantic_text']=tok.decode(ps[0]['token_ids'][:prefix],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  tasks=requests(ps[:1],track='CONTROLLED',prefix=prefix,special=special,routing_seed=seed)
  tasks.extend(requests(ps[1:],track='NATURAL',prefix=None,offset=1,special=special,routing_seed=seed))
  tasks[0]['role']='CONTROLLED_STANDALONE_SOURCE'
  if len(ps)==3:tasks[-1]['role']='NEGATIVE_TARGET'
  for i,t in enumerate(tasks):t['request_sequence_number']=i
  b=dict(dataset='qasper',track='CONTROLLED',prefix_length=prefix,repetition=0,scale='SMOKE',scale_semantics=SCALE_SEMANTICS,routing_seed=seed,i5r3b_profile=profile,independent_context_sha256=item['context_sha256'],tasks=tasks,condition_order=[CONDITIONS[i] for i in ORDERS[ordinal%len(ORDERS)]],cache_resets_per_condition=1,cache_lifetime_requests=len(tasks),physical_requests_per_condition=len(tasks),priming_requests=1,scored_unique_samples=len(ps),repeated_evaluation_samples=0,reset='new ES/model/CPU UD sessions and empty cache/index per condition',warmup='unchanged I5 one uncached one-token request per adapter')
  b['resources']=resources(tasks,capacity=CAPACITY,evidence_budget=(8 if profile=='B1' else 16)*2**30)
  b['pair_checks']=checks(b,users,seed);b['block_id']=digest(b);blocks.append(b)
 return blocks

def historical_r1(path):
 if path is None:return dict(status='NOT_PROVIDED',scope='No historical metrics reused or pooled')
 ev=Evidence(path,max_input_bytes=256*2**20);receipt=ev.verify();d=ev.json('i5r1_decision.json');a=ev.json('i5r1_independent_audit.json')
 require(a['status']=='PASS' and not a['TEST_FIXTURE_ONLY'] and not d['GPU_NOT_RUN'],'real historical R1 audit required');archived_source(d['source'])
 require(any(g['dataset']=='qasper' and g['case']=='64:256' and g['target_repeats']==1 for g in d['groups']),'historical R1 scope not comparable')
 return dict(status='SEALED_EXPORTED_HISTORICAL_AUDIT',path=str(Path(path).resolve()),receipt=receipt,source=d['source'],decisions=d['groups'],scope='64:256 arithmetic/configuration context only; new targets/routing not asserted identical; never pooled as new measurements')

def selection_table(blocks):
 return [dict(block_index=i,independent_context_sha256=b['independent_context_sha256'],targets=b['pair_checks'],source_original_id=b['tasks'][0]['sample_id'],source_records=[dict(sample_id=t['sample_id'],original_3a=t['prepared'].get('original_3a'),derived_prompt_sha256=digest(t['token_ids']),derived_prompt_tokens=len(t['token_ids'])) for t in b['tasks'] if t['scored']]) for i,b in enumerate(blocks)]

def original_link(sample,record):
 require(sample['id']==record['id'] and sample['source_record_sha256']==record['source_record_sha256'] and digest_bytes(sample['context'].encode())==record['shared_context_key'],'3A original source linkage differs')
 require(digest(record['token_ids'])==record['token_ids_sha256'],'3A original prompt hash differs')
 return dict(sample_id=sample['id'],group_id=sample['group_id'],source_record_sha256=sample['source_record_sha256'],context_sha256=record['shared_context_key'],original_3a_prompt_sha256=record['token_ids_sha256'],original_3a_prompt_tokens=len(record['token_ids']),original_routing=record['route'],question_sha256=digest(sample['question']),references_sha256=digest(sample['references']))


def prepare(plan_3a,audit_3a,verification_3a,expected_3a_commit,output,profile='B2_SMOKE',adapter='Adapter_A',historical_r1_audit=None,max_prompt_tokens=None):
 require(profile!='B1' or max_prompt_tokens is None,'--max-prompt-tokens is B2-only')
 limit=None if profile=='B1' else (1280 if max_prompt_tokens is None else max_prompt_tokens)
 require(limit is None or type(limit) is int and 512<limit<=2040,'B2 prompt limit must allow a nonempty 512-token suffix and generation reservation')
 old,result,proof=history(plan_3a,audit_3a,verification_3a,expected_3a_commit)
 tok,users,prov,data,pm,m=inputs_3a(old['registry_path'],old['multi_prepared'])
 require(users==old['users'] and pm['identity']==old['model_identity'] and tok.identity==old['tokenizer_identity']==m['tokenizer'],'3A/model/tokenizer preparation mismatch')
 historical_prompts=[dict(sample=s,route=routing(s,old['seed']),token_ids=r['token_ids']) for s in data['qasper'] for r in result['qasper']['samples'] if r['id']==s['id'] and 'token_ids' in r]
 historical_shared_statistics=shared_statistics(shared_pairs(historical_prompts,users))
 prompts=[];derived_exclusions=[]
 for s in data['qasper']:
  try:p=prepare_prompt(s,tok.tokenizer,2048,8 if profile=='B1' else 32,allow_truncation=True,input_limit=256 if profile=='B1' else limit)
  except ValueError as exc:derived_exclusions.append(dict(sample_id=s['id'],reason=str(exc)));continue
  p.update(route=routing(s,old['seed']))
  if profile!='B1':
   recorded=next(r for r in result['qasper']['samples'] if r['id']==s['id'])
   p['original_3a']=original_link(s,recorded)
   p['workload_label']=LABEL
   require(p['route']==recorded['route'] and len(p['token_ids'])<=limit,'derived routing/length differs')
  prompts.append(p)
 chosen,eligibility=selection(prompts,users,profile,adapter)
 special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
 blocks=build_blocks(chosen,profile,special,tok.tokenizer,users,old['seed'])
 _,_,specs,_=validate(old['multi_prepared'])
 protocol=dict(schema=SCHEMA,source=source_identity(),profile=profile,adapter=adapter,inputs=dict(plan_3a=str(Path(plan_3a).resolve()),audit_3a=str(Path(audit_3a).resolve()),verification_3a=str(Path(verification_3a).resolve()),expected_3a_commit=expected_3a_commit),proof=proof,multi_prepared=old['multi_prepared'],multi_inventory=Evidence(old['multi_prepared']).verify('r5_output_manifest.json'),adapter_fingerprints={k:v['matrix_sha256'] for k,v in specs.items()},model_identity=pm['identity'],eligibility=eligibility,historical_r1=historical_r1(historical_r1_audit),revalidated_3a_summaries={k:v['summary'] for k,v in result.items()},max_prompt_tokens=limit,historical_3a_shared_statistics=historical_shared_statistics,derived_exclusions=derived_exclusions,label=LABEL if limit else 'CONTROLLED_SHARED_CONTEXT_SETUP_NOT_NATURAL_PHYSICAL_REUSE',observed_runtime_HIT='NOT_EVALUATED')
 protocol['derived_workload_statistics']=dict(label=LABEL if limit else 'B1',prompt_limit=limit,eligible_prompts=len(prompts),shared_context_selection=eligibility,original_3a_full_prompt_equivalence='NOT_CLAIMED' if limit else 'NOT_APPLICABLE')
 protocol['handoff_comparison']=dict(supplied_authorized_1000_pairs=19,supplied_independent_1000_contexts=17,recalculated=historical_shared_statistics,agrees=historical_shared_statistics['authorized_shared_pairs_at_least_1000']==19 and historical_shared_statistics['independent_contexts_at_least_1000']==17) if profile!='B1' else dict(status='B1_INPUT_LIMIT_DIFFERS_NOT_COMPARABLE')
 base=dict(schema='I5_EXPERIMENT_V1',scale='SMOKE',scale_semantics=SCALE_SEMANTICS,seed=old['seed'],tokenizer_identity=tok.identity,context_limit=2048,special_tokens=special,template_version='I5_PROMPTS_V1',TEST_FIXTURE_ONLY=False,conditions=CONDITIONS,clusters=1,policy='LRU_FIT_COMPLETE_SOURCE_V1',max_entries_per_adapter=64,routing_algorithm='ORIGINAL_I5_ROUTING_UNCHANGED',i5r3b=protocol)
 plans={}
 dataset_manifest={'qasper':dict(prov['qasper']['manifest'],prepared_path=prov['qasper']['path'],prepared_manifest_sha256=sha(Path(prov['qasper']['path'])/'i5_output_manifest.json'))}
 protected=(plan_3a,audit_3a,verification_3a,old['multi_prepared'],*[x['path'] for x in prov.values()],*([historical_r1_audit] if historical_r1_audit else []))
 for name in ('abc','a0'):
  bs=deepcopy(blocks)
  if name=='a0':
   for b in bs:
    b.pop('block_id');b['tasks']=b['tasks'][1:]
    for i,t in enumerate(b['tasks']):t['request_sequence_number']=i
    b.update(track='NATURAL',priming_requests=0,cache_lifetime_requests=len(b['tasks']),physical_requests_per_condition=len(b['tasks']),pair_checks=[],a0_scope='Only FULL_RECOMPUTE is the economic reference; auxiliary B/C retained for unchanged I5 audit')
    b['resources']=resources(b['tasks'],capacity=CAPACITY,evidence_budget=(8 if profile=='B1' else 16)*2**30);b['block_id']=digest(b)
  ex=dict(deepcopy(base),blocks=bs,control='ABC' if name=='abc' else 'A0_TARGET_ONLY');ex['experiment_id']=digest(ex);plans[name]=ex

 transport=transport_preflight(plans,pm,specs,tok.identity,users)
 out=new_output(output,protected)
 for name,ex in plans.items():
  dest=out/name;dest.mkdir();write(dest/'i5_workload_manifest.json',ex);write(dest/'i5_dataset_manifest.json',dataset_manifest);seal(dest)
 write(out/'i5r3b_transport.json',transport)
 write(out/'i5r3b_protocol.json',protocol);write(out/'i5r3b_selection.json',selection_table(blocks));seal(out)
 return inspect(output)

def validate_plan(path,current=True):
 root=Path(path);ev=Evidence(root,max_input_bytes=128*2**20);ev.verify();p=ev.json('i5r3b_protocol.json');require(p['schema']==SCHEMA,'3B schema differs');archived_source(p['source'])
 if current:require(p['source']==source_identity(),'3B reviewed source differs')
 old,result,proof=history(p['inputs']['plan_3a'],p['inputs']['audit_3a'],p['inputs']['verification_3a'],p['inputs']['expected_3a_commit'])
 require(proof==p['proof'] and Evidence(p['multi_prepared']).verify('r5_output_manifest.json')==p['multi_inventory'],'3A/adapter evidence changed')
 require(historical_r1(p['historical_r1'].get('path'))==p['historical_r1'],'optional historical R1 audit changed')
 m,pm,specs,users=validate(p['multi_prepared']);require(pm['identity']==p['model_identity'] and {k:v['matrix_sha256'] for k,v in specs.items()}==p['adapter_fingerprints'],'model/adapter changed')
 plans={}
 for name in ('abc','a0'):
  ee=Evidence(root/name);ee.verify();ex=ee.json('i5_workload_manifest.json');q=dict(ex);eid=q.pop('experiment_id');require(digest(q)==eid and ex['i5r3b']==p and ex['conditions']==list(CONDITIONS) and ex['clusters']==1 and ex['max_entries_per_adapter']==64,'3B child plan differs')
  require(ex['context_limit']==2048 and ex['seed']==old['seed'] and ex['tokenizer_identity']==m['tokenizer'] and ex['TEST_FIXTURE_ONLY'] is False and ex['policy']=='LRU_FIT_COMPLETE_SOURCE_V1' and not ex.get('i5r2b') and not ex.get('i5r2c2'),'unoptimized runtime contract differs')
  require(1<=len(ex['blocks'])<=3,'bounded groups required')
  for b in ex['blocks']:
   bb=dict(b);bid=bb.pop('block_id');require(digest(bb)==bid and b['resources']['per_adapter_capacity_bytes']==CAPACITY,'block seal/capacity differs')
   require(b['prefix_length']==(64 if p['profile']=='B1' else 512) and b['i5r3b_profile']==p['profile'] and b['condition_order']==[CONDITIONS[i] for i in ORDERS[ex['blocks'].index(b)%len(ORDERS)]],'profile prefix/condition order differs')
   if name=='abc':require(checks(b,users,ex['seed'])==b['pair_checks'],'3B pair proof differs')
   if p['profile']!='B1':
    require(type(p['max_prompt_tokens']) is int and 512<p['max_prompt_tokens']<=2040 and p['label']==LABEL,'derived workload limit/label differs')
    for task in b['tasks']:
     prepared=task['prepared'];original=next(r for r in result['qasper']['samples'] if r['id']==task['sample_id'])
     require(prepared['original_3a']==original_link(prepared['sample'],original) and prepared['workload_label']==LABEL and len(prepared['token_ids'])<=p['max_prompt_tokens'],'derived-to-original provenance changed')
  plans[name]=ex
 require(len(plans['abc']['blocks'])==len(plans['a0']['blocks'])==(3 if p['profile']=='B2_THREE_ADAPTER' else 1),'A0/profile block pairing differs')
 require(len({b['independent_context_sha256'] for b in plans['abc']['blocks']})==len(plans['abc']['blocks']),'independent context groups repeated')
 if p['profile']=='B2_THREE_ADAPTER':require({b['tasks'][0]['adapter_id'] for b in plans['abc']['blocks']}=={'Adapter_A','Adapter_B','Adapter_C'},'three-adapter coverage differs')
 for abc,a0 in zip(plans['abc']['blocks'],plans['a0']['blocks']):
  targets=deepcopy(abc['tasks'][1:])
  for i,t in enumerate(targets):t['request_sequence_number']=i
  require(a0['tasks']==targets and a0['priming_requests']==0 and a0['independent_context_sha256']==abc['independent_context_sha256'],'A0 altered targets or inserted source')
 require(ev.json('i5r3b_selection.json')==selection_table(plans['abc']['blocks']),'selection table differs from sealed workloads')
 require(ev.json('i5r3b_transport.json')==transport_preflight(plans,pm,specs,m['tokenizer'],users),'CPU transport envelope evidence differs')
 return p,plans

def inspect(path):
 p,plans=validate_plan(path);bs=plans['abc']['blocks']
 return dict(status='PREPARED_GPU_NOT_RUN',profile=p['profile'],independent_groups=len({b['independent_context_sha256'] for b in bs}),blocks=len(bs),valid_array=f'0-{len(bs)-1}%1',source_commit=p['source']['commit'],workload_label=p['label'],max_prompt_tokens=p['max_prompt_tokens'],transport=read(Path(path)/'i5r3b_transport.json'),historical_3a_statistics=p['revalidated_3a_summaries'],historical_3a_shared_context_statistics=p['historical_3a_shared_statistics'],historical_handoff_comparison=p['handoff_comparison'],derived_statistics=p['derived_workload_statistics'],eligibility=p['eligibility'],selection=read(Path(path)/'i5r3b_selection.json'),requests_per_condition=[len(b['tasks']) for b in bs],a0_requests_per_condition=[len(b['tasks']) for b in plans['a0']['blocks']],conditions=list(CONDITIONS),A0_reference='Separate target-only block; only FULL_RECOMPUTE enters economics; two auxiliary B/C runs cost extra',maximum_gpu_estimate_bytes=max(b['resources']['gpu_estimate_bytes'] for b in bs),worst_case_all_condition_evidence_bytes=sum(3*b['resources']['evidence_budget_bytes'] for ex in plans.values() for b in ex['blocks']),capacity_bytes_per_adapter=CAPACITY,max_entries=64)
