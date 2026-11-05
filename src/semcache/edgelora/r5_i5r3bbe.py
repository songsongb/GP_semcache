"""Measured break-even analysis and a separate immutable repeat-request contract."""
from copy import deepcopy
import math
from pathlib import Path
from .communication import require,digest
from .r3a_artifacts import ROOT,read,write,THRESHOLDS
from .r5_i5_evidence import seal,jsonl
from .r5_i5r2_tensors import Evidence
from .r5_i5r1_plan import source_identity
from .r5_i5r3b_plan import archived_source,new_output,CONDITIONS,CAPACITY
from .r5_i5r3b_transport import preflight as transport_preflight
from .r5_multi_artifacts import validate,sharing
from .r5_i5_workload import resources,ORDERS
from .r5_i5_contracts import validate_request

SCHEMA='I5R3B_BE_V1'
NAMESPACE=ROOT/'results/r5/i5r/controlled_hit/i5r3b/be'
BASE_COMMIT='a28e770779568554ad0bac9a49c888a968b6ffed'
C=CONDITIONS[2];A=CONDITIONS[0]

def output(path,protected):
 p=Path(path).resolve();require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'fresh BE namespace required')
 return new_output(path,protected)

def sequence(rows):
 require(rows and len({r['request_id'] for r in rows})==len(rows),'duplicate/missing execution requests')
 require(all(r['status']=='COMPLETE' and r['completion_ns']>r['arrival_ns'] for r in rows),'incomplete timing receipts')
 require(all(r['completion_ns']<=s['arrival_ns'] for r,s in zip(rows,rows[1:])),'reordered sequence')
 span=(rows[-1]['completion_ns']-rows[0]['arrival_ns'])/1e6
 total=sum(r['application_e2e_ms'] for r in rows)
 require(all(abs(r['application_e2e_ms']-(r['completion_ns']-r['arrival_ns'])/1e6)<1e-6 for r in rows) and span>=total-1e-6,'invalid request/sequence timing')
 return dict(span_ms=span,request_intervals_ms=total,gaps_ms=span-total)

def economics(crows,a0rows):
 require(len(crows)==len(a0rows)+1 and not crows[0]['scored'] and all(r['scored'] for r in crows[1:]+a0rows),'source/A0 denominator differs')
 require([r['request_id'] for r in crows[1:]]==[r['request_id'] for r in a0rows],'unpaired target requests')
 cs,aa=sequence(crows),sequence(a0rows);positives=[];negatives=[]
 for c,a in zip(crows[1:],a0rows):
  require(c['sample_id']==a['sample_id'] and c['user_id']==a['user_id'] and c['adapter_id']==a['adapter_id'],'matched sample/route differs')
  item=dict(request_id=c['request_id'],sample_id=c['sample_id'],a0_e2e_ms=a['application_e2e_ms'],c_e2e_ms=c['application_e2e_ms'],savings_ms=a['application_e2e_ms']-c['application_e2e_ms'])
  (negatives if c['role']=='NEGATIVE_TARGET' else positives).append(item)
 require(positives and len(negatives)==1,'positive/negative denominator differs')
 source=crows[0]['application_e2e_ms'];savings=sum(r['savings_ms'] for r in positives);negative=-negatives[0]['savings_ms'];gap=cs['gaps_ms']-aa['gaps_ms'];d=cs['span_ms']-aa['span_ms']
 require(abs(d-(source-savings+negative+gap))<1e-5,'source-inclusive cost decomposition differs')
 return dict(measured_positive_requests=len(positives),actual_physical_hits=sum(r.get('source') is not None for r in crows[1:] if r['role']!='NEGATIVE_TARGET'),source_cost_ms=source,source_forward_ms=crows[0]['model_execution_e2e_ms'],source_admission_ms=crows[0]['cache_admission_eviction_ms'],source_cost_scope='Application E2E already includes admission; forward/admission are nested descriptions, never added again',positive_target_savings_ms=positives,total_positive_target_savings_ms=savings,negative_target_overhead_ms=negative,sequence_gap_overhead_ms=gap,C_sequence=cs,A0_sequence=aa,D_ms=d,measured_break_even=d<0,observed_amortized_e2e_ms=cs['span_ms']/len(a0rows),A0_amortized_e2e_ms=aa['span_ms']/len(a0rows),scored_request_denominator=len(a0rows),positive_only_C_span_ms=(crows[-2]['completion_ns']-crows[0]['arrival_ns'])/1e6,positive_only_A0_span_ms=(a0rows[-2]['completion_ns']-a0rows[0]['arrival_ns'])/1e6,positive_target_interval_ms=sum(r['c_e2e_ms'] for r in positives),negative_control_C_ms=negatives[0]['c_e2e_ms'],measured_minimum_reuses='NOT_ESTABLISHED: only this tested N; no interpolation is a measurement')

def estimate(measured):
 require(measured['measured_positive_requests']==2 and measured['actual_physical_hits']==2,'N=2 verified physical baseline required')
 s=[r['savings_ms'] for r in measured['positive_target_savings_ms']];d=measured['D_ms'];scenarios={}
 for label,value in (('repeat_target1',s[0]),('repeat_target2',s[1]),('mean',sum(s)/2),('minimum_observed_saving',min(s)),('maximum_observed_saving',max(s))):
  minimum=2 if d<0 else (2+math.floor(d/value)+1 if value>0 else None)
  scenarios[label]=dict(assumed_additional_saving_ms=value,estimated_minimum_reuses=minimum,estimated_N3_D_ms=d-value,estimated_N3_break_even=d-value<0,status='EXTRAPOLATED_NOT_MEASURED')
 return dict(scenarios=scenarios,estimated_minimum_reuses=scenarios['repeat_target1']['estimated_minimum_reuses'],measured_N2=measured,assumptions='Existing two savings retained; source/negative/gap overhead fixed; each extra request saves scenario amount. Additional gaps, eviction, miss, thermal/order/run variation not predicted.',uncertainty='Two dependent target observations from one context; sensitivity is not a confidence interval. Run-to-run variance NOT_MEASURED.',statistical_independent_groups=1)

def baseline(experiment,run,audit,verification,expected_commit=BASE_COMMIT):
 es={k:Evidence(v,max_input_bytes=128*2**30 if k=='run' else 256*2**20) for k,v in dict(experiment=experiment,run=run,audit=audit,verification=verification).items()};receipts={k:e.verify() for k,e in es.items()}
 p=es['experiment'].json('i5r3b_protocol.json');archived_source(p['source']);require(p['source']['commit']==expected_commit,'baseline source differs')
 r=es['audit'].json('i5r3b_results.json');a=es['audit'].json('i5r3b_independent_audit.json');v=es['verification'].json('i5r3b_verification.json')
 require(a['status']==v['status']=='PASS' and a['source']==r['source']==p['source']==v['source'] and a['run_inventory']==r['receipt']==receipts['run'] and v['audit_inventory']==receipts['audit'] and v['verdicts']==r['verdicts']==es['audit'].json('i5r3b_decision.json'),'baseline audit/provenance binding differs')
 require(r['verdicts']['FUNCTIONAL_PHYSICAL_REUSE']==r['verdicts']['CACHE_INTEGRITY']=='PASS' and r['numerical']['thresholds']==THRESHOLDS,'baseline physical integrity or frozen thresholds differ')
 require(all(r['legacy_independent_audits'][k]['status']=='PASS' for k in ('abc','a0')),'baseline legacy strict audits missing')
 st=es['run'].json('i5r3b_attempt_status.json');require(st['status']=='COMPLETE' and st['with_a0'] and st['source_commit']==expected_commit,'baseline incomplete')
 plans={};measured_rows={}
 for name in ('abc','a0'):
  child=Evidence(Path(experiment)/name);child.verify();ex=child.json('i5_workload_manifest.json');q=dict(ex);eid=q.pop('experiment_id');require(digest(q)==eid and ex['i5r3b']==p and not ex['TEST_FIXTURE_ONLY'],'baseline plan identity differs');plans[name]=ex
  cfg=es['run'].json(name+'_run/i5_execution_config.json');require(cfg['experiment_id']==eid and cfg['sources']==p['source']['files'] and cfg['source_commit']==expected_commit and cfg['block']==ex['blocks'][st['block_index']] and not cfg['fixture'],'baseline raw identity differs')
  raw_root=Path(run)/(name+'_run');Evidence(raw_root,max_input_bytes=128*2**30).verify();raw_status=read(raw_root/'i5_block_status.json')
  require(raw_status['status']=='COMPLETE' and raw_status['GPU_NOT_RUN'] is False,'baseline raw GPU execution incomplete')
  exported_audit=Evidence(Path(run)/(name+'_run_audit'));exported_audit.verify()
  require(exported_audit.json('i5_independent_audit.json')==r['legacy_independent_audits'][name],'baseline original I5 audit differs')
  for condition in CONDITIONS:
   gpu=r['legacy_independent_audits'][name]['conditions'][condition]['gpu_identity']
   require(gpu['TEST_FIXTURE_ONLY'] is False and gpu['gpu_class_compatible'] is True and '3090' in gpu['execution_gpu_name'] and gpu['execution_gpu_uuid']==cfg['expected_execution_gpu_uuid'] and gpu['historical_live_kv_restored'] is False,'baseline GPU receipt differs')
   raw=es['run'].jsonl(name+'_run/'+condition+'/generation.jsonl')
   exported=[x for x in r['metrics'] if x['control']==name and x['condition']==condition]
   require([x['request_id'] for x in raw]==[t['request_id'] for t in cfg['block']['tasks']],'baseline request order differs')
   # Export rows have I5 analysis-added fields; every original raw field must agree.
   require(len(raw)==len(exported) and all(all(x[k]==y[k] for k in x) for x,y in zip(raw,exported)),'raw/export per-request values differ')
   arrivals=[x for x in es['run'].jsonl(name+'_run/'+condition+'/requests.jsonl') if x['kind']=='ARRIVAL']
   require(len(arrivals)==len(raw),'missing baseline original requests')
   for task,item,row in zip(cfg['block']['tasks'],arrivals,raw):
    req=validate_request(item['request'])
    require(req['request_id']==row['request_id']==task['request_id'] and req['token_ids']==task['token_ids'] and req['input_text']==task['input_text'] and req['generation_policy']==task['policy'] and req['user_id']==task['user_id'] and req['adapter_id']==task['adapter_id'] and req['sample_id']==task['sample_id'],'baseline request/task mismatch')
    require(req['physical_identity']['base_fingerprint']==p['model_identity']['base_weight_sha256'] and req['physical_identity']['adapter_matrix_fingerprint']==p['adapter_fingerprints'][task['adapter_id']],'baseline physical fingerprints differ')
   measured_rows[name,condition]=raw
 b=plans['abc']['blocks'][st['block_index']];require(p['profile']=='B2_SMOKE' and p['max_prompt_tokens']==1280 and len(b['tasks'])==4,'exact transport-limited B2 smoke baseline required')
 require([t['user_id'] for t in b['tasks'][1:]]==['User_08','User_06','User_10'] and all(t['adapter_id']=='Adapter_A' for t in b['tasks']) and len(b['tasks'][0]['token_ids'])==512 and all(len(t['token_ids'])==1280 for t in b['tasks'][1:]),'recorded baseline selection/user/shape differs')
 crows=measured_rows['abc',C];arows=measured_rows['a0',A]
 require(crows[-1]['source'] is None and all(x['source'] is not None and x['reused_prefix_tokens']==512 for x in crows[1:-1]),'baseline N=2 HIT/negative evidence differs')
 m=economics(crows,arows)
 require(abs(m['D_ms']-r['A0_economic_comparison']['absolute_delta'])<1e-5,'baseline exported economics differs')
 projection=estimate(m);projection['baseline_verdicts']=r['verdicts']
 return dict(protocol=p,plans=plans,block=b,report=r,measured=m,estimate=projection,receipts=receipts,paths={k:str(e.root) for k,e in es.items()},expected_commit=expected_commit,scope='Archived independently verified GPU evidence plus new CPU byte/provenance/timing checks; not a fresh original-source I5 GPU audit')

def repeat_blocks(original,n=3,repetitions=1):
 require(type(n) is int and 3<=n<=8 and type(repetitions) is int and 1<=repetitions<=3,'bounded BE repeat profile required')
 s,t1,t2,neg=original['tasks'];blocks=[]
 for rep in range(repetitions):
  b=deepcopy(original);b.pop('block_id');positives=[deepcopy(t1),deepcopy(t2)]
  for i in range(2,n):
   item=deepcopy((t1,t2)[(i-2)%2]);item['request_id']+=f'_BE{i+1:02d}';positives.append(item)
  b['tasks']=[deepcopy(s)]+positives+[deepcopy(neg)]
  for i,t in enumerate(b['tasks']):t['request_sequence_number']=i
  b.update(track='CONTROLLED',i5r3bbe_label='CONTROLLED_REPEAT_AMORTIZATION',repetition=rep,priming_requests=1,scored_unique_samples=3,repeated_evaluation_samples=n-2,cache_lifetime_requests=n+2,physical_requests_per_condition=n+2,condition_order=[CONDITIONS[i] for i in ORDERS[rep%len(ORDERS)]])
  b.pop('pair_checks',None);b['resources']=resources(b['tasks'],capacity=CAPACITY,evidence_budget=16*2**30);b['block_id']=digest(b);blocks.append(b)
 return blocks

def check_repeat(block,original,users,n):
 expected=repeat_blocks(original,n,block['repetition']+1)[-1]
 require(block==expected,'BE immutable repeat/task contract differs')
 tasks=block['tasks'];require(len({t['request_id'] for t in tasks})==len(tasks),'duplicate request IDs')
 source=tasks[0]
 for t in tasks[1:]:
  require(sharing(users,source['user_id'],t['user_id'],source['adapter_id'],t['adapter_id']) is None,'unauthorized repeat user/adapter')
 require(len({t['sample_id'] for t in tasks[1:]})==3 and sum(t['sample_id']==tasks[1]['sample_id'] for t in tasks[1:-1])>=2,'repeat inflated unique samples')
 return dict(total_scored_requests=n+1,unique_scored_samples=3,repeated_request_observations=n-2,distinct_original_users=len({t['user_id'] for t in tasks}),independent_context_groups=1,source_tokens=512,source_raw_qkv_bytes=512*491520,label='CONTROLLED_REPEAT_AMORTIZATION',request_table=[dict(request_id=t['request_id'],sequence=t['request_sequence_number'],sample_id=t['sample_id'],group_id=t['group_id'],user=t['user_id'],adapter=t['adapter_id'],tokens=len(t['token_ids']),token_hash=digest(t['token_ids']),role=t['role']) for t in tasks])

def child_plans(base,source,n,repetitions):
 blocks=repeat_blocks(base['block'],n,repetitions);plans={}
 for name in ('abc','a0'):
  ex=deepcopy(base['plans'][name]);ex.pop('experiment_id');ex.pop('i5r3b');ex['i5r3bbe']=dict(schema=SCHEMA,source=source,label='CONTROLLED_REPEAT_AMORTIZATION',baseline_source=base['protocol']['source']);ex['blocks']=deepcopy(blocks)
  if name=='a0':
   for b in ex['blocks']:
    b.pop('block_id');b['tasks']=b['tasks'][1:]
    for i,t in enumerate(b['tasks']):t['request_sequence_number']=i
    b.update(track='NATURAL',priming_requests=0,cache_lifetime_requests=len(b['tasks']),physical_requests_per_condition=len(b['tasks']))
    b['resources']=resources(b['tasks'],capacity=CAPACITY,evidence_budget=16*2**30);b['block_id']=digest(b)
  ex['experiment_id']=digest(ex);plans[name]=ex
 return plans

def approval(path,n):
 if n==3 and path is None:return None
 require(path,'additional hits require explicit previously validated real 3-hit BE audit')
 ev=Evidence(path);receipt=ev.verify();approved=ev.json('i5r3bbe_results.json');proof=ev.json('i5r3bbe_independent_audit.json');archived_source(proof['source'])
 require(proof['status']=='PASS' and proof['source']==approved['source'] and approved['positive_requests']==3 and not approved['GPU_NOT_RUN'] and approved['blocks'] and all(b['physical_reuse_pass'] and b['verdicts']['CACHE_INTEGRITY']=='PASS' for b in approved['blocks']),'real three-HIT validation prerequisite failed')
 return receipt

def compatible_preparation(base):
 m,pm,specs,users=validate(base['protocol']['multi_prepared'])
 require(Evidence(base['protocol']['multi_prepared']).verify('r5_output_manifest.json')==base['protocol']['multi_inventory'],'baseline registry/preparation inventory changed')
 require(pm['identity']==base['protocol']['model_identity'] and {k:v['matrix_sha256'] for k,v in specs.items()}==base['protocol']['adapter_fingerprints'] and m['tokenizer']==base['plans']['abc']['tokenizer_identity'],'baseline model/tokenizer/adapter fingerprints changed')
 return m,pm,specs,users

def prepare(experiment,run,audit,verification,output_dir,expected_commit=BASE_COMMIT,positive_requests=3,repetitions=1,validated_be_audit=None):
 base=baseline(experiment,run,audit,verification,expected_commit);approved=approval(validated_be_audit,positive_requests)
 source=source_identity();m,pm,specs,users=compatible_preparation(base);plans=child_plans(base,source,positive_requests,repetitions)
 tables=[check_repeat(b,base['block'],users,positive_requests) for b in plans['abc']['blocks']]
 transport=transport_preflight(plans,pm,specs,m['tokenizer'],users)
 protocol=dict(schema=SCHEMA,source=source,baseline_inputs=base['paths'],baseline_expected_commit=expected_commit,baseline_inventories=base['receipts'],positive_requests=positive_requests,repetitions=repetitions,multi_prepared=base['protocol']['multi_prepared'],adapter_inventory=Evidence(base['protocol']['multi_prepared']).verify('r5_output_manifest.json'),execution_order=[['abc','a0'] if i%2==0 else ['a0','abc'] for i in range(repetitions)],label='CONTROLLED_REPEAT_AMORTIZATION',validated_be_audit=str(Path(validated_be_audit).resolve()) if validated_be_audit else None,approval_inventory=approved)
 out=output(output_dir,(*base['paths'].values(),protocol['multi_prepared'],*([validated_be_audit] if validated_be_audit else [])))
 write(out/'i5r3bbe_protocol.json',protocol);write(out/'i5r3bbe_selection.json',tables);write(out/'i5r3bbe_transport.json',transport);write(out/'i5r3bbe_N2_economics.json',base['estimate'])
 for name,ex in plans.items():
  dest=out/name;dest.mkdir();write(dest/'i5_workload_manifest.json',ex);write(dest/'i5_dataset_manifest.json',read(Path(experiment)/name/'i5_dataset_manifest.json'));seal(dest)
 seal(out);return inspect(output_dir)

def load(path):
 ev=Evidence(path,max_input_bytes=128*2**20);ev.verify();p=ev.json('i5r3bbe_protocol.json');require(p['schema']==SCHEMA and p['source']==source_identity(),'BE reviewed source differs')
 base=baseline(**p['baseline_inputs'],expected_commit=p['baseline_expected_commit']);require(base['receipts']==p['baseline_inventories'],'baseline inputs changed')
 require(approval(p['validated_be_audit'],p['positive_requests'])==p['approval_inventory'],'optional repeat approval changed')
 m,pm,specs,users=compatible_preparation(base);require(Evidence(p['multi_prepared']).verify('r5_output_manifest.json')==p['adapter_inventory'],'adapter evidence changed')
 expected=child_plans(base,p['source'],p['positive_requests'],p['repetitions']);tables=[]
 for name,ex in expected.items():
  child=Evidence(Path(path)/name);child.verify();require(child.json('i5_workload_manifest.json')==ex and child.json('i5_dataset_manifest.json')==read(Path(base['paths']['experiment'])/name/'i5_dataset_manifest.json'),'BE child manifest differs')
 tables=[check_repeat(b,base['block'],users,p['positive_requests']) for b in expected['abc']['blocks']]
 require(ev.json('i5r3bbe_selection.json')==tables and ev.json('i5r3bbe_transport.json')==transport_preflight(expected,pm,specs,m['tokenizer'],users) and ev.json('i5r3bbe_N2_economics.json')==base['estimate'],'BE selection/envelope/economics changed')
 require(p['execution_order']==[['abc','a0'] if i%2==0 else ['a0','abc'] for i in range(p['repetitions'])],'paired execution order changed')
 return p,expected,base

def inspect(path):
 p,plans,base=load(path)
 return dict(status='PREPARED_GPU_NOT_RUN',positive_requests=p['positive_requests'],repetitions=p['repetitions'],selection=read(Path(path)/'i5r3bbe_selection.json'),transport=read(Path(path)/'i5r3bbe_transport.json'),N2_economics=base['estimate'],source_commit=p['source']['commit'],execution_order=p['execution_order'],blocks=len(plans['abc']['blocks']),resources=[b['resources'] for b in plans['abc']['blocks']],A0_scope='Separate target-only ABC block; only FULL_RECOMPUTE enters economics; auxiliary B/C excluded')

def offline(experiment,run,audit,verification,output_dir,expected_commit=BASE_COMMIT):
 base=baseline(experiment,run,audit,verification,expected_commit);out=output(output_dir,base['paths'].values());write(out/'i5r3bbe_offline.json',base['estimate']);write(out/'i5r3bbe_baseline_provenance.json',dict(paths=base['paths'],receipts=base['receipts'],source=base['protocol']['source'],analysis_source=source_identity(),scope=base['scope']));seal(out);return base['estimate']
