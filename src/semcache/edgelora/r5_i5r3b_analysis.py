"""Independent strict I5 audit, measured cache classes and source-inclusive costs."""
from pathlib import Path
from .communication import require,digest
from .r3a_artifacts import read,write,THRESHOLDS
from .r5_i5_evidence import fresh,seal,jsonl
from .r5_i5r2_tensors import Evidence
from .r5_i5_analysis import analyze as audit_i5,METRICS,paired
from .r5_i5r1_analysis import costs,delta,layer_diagnostics
from .r5_multi_analysis import fidelity
from .r5_i5r3b_plan import validate_plan,CONDITIONS,new_output


def classify(row,events):
 relevant=[e['event'] for e in events if e['condition']==row['condition'] and e['adapter_id']==row['adapter_id'] and e['event'].get('request_id')==row['request_id']]
 if row['condition']!='SEMCACHE_INTEGRATED':kind='NO_PHYSICAL_CACHE_REUSE'
 elif row['source'] is not None:
  require(row['decision']=='PHYSICAL_PREFIX_HIT' and row['reused_prefix_tokens']>0 and not row['cached_prefix_q_consumed'],'inconsistent physical grant')
  require(row['source']['entry_id'] in row['cache_before']['lru_to_mru'] and any(e['kind']=='PHYSICAL_REUSE' for e in relevant),'no resident physical reuse receipt')
  kind='PHYSICAL_PREFIX_HIT'
 else:
  require(row['decision']=='CACHE_MISS' and row['reused_prefix_tokens']==0,'miss incorrectly claims reuse')
  kind='COLD_SOURCE' if row['role']=='CONTROLLED_STANDALONE_SOURCE' and row['cache_before']['entry_count']==0 else 'WARM_MISS' if row['cache_before']['entry_count']>0 else 'COLD_MISS'
 denied=any(e['kind']=='ADMISSION_DENIED' for e in relevant)
 return dict(condition=row['condition'],request_id=row['request_id'],classification=kind,secondary_outcomes=(['ADMISSION_DENIED'] if denied else [])+(['EVICTION'] if row['eviction_count'] else []),admission_denied=denied,evictions=row['eviction_count'],eviction_basis='audited request-scoped real owner events',candidate_reasons=row['candidate_reasons'],actual_source=row['source'],admission=row['admission'],cache_before=row['cache_before'],cache_after=row['cache_after'])

def scope_cost(rows,initialization,warmup,life):
 require(rows and all(r['completion_ns']<=s['arrival_ns'] for r,s in zip(rows,rows[1:])),'incomplete or reordered execution')
 if any(not r['scored'] for r in rows):out=costs(rows,initialization,warmup,life)
 else:
  require(all(r['scored'] for r in rows),'A0 contains constructed source')
  span=(rows[-1]['completion_ns']-rows[0]['arrival_ns'])/1e6
  require(life['execution_start_ns']<=rows[0]['arrival_ns']<=rows[-1]['completion_ns']<=life['execution_end_ns'] and span>=sum(r['application_e2e_ms'] for r in rows)-1e-6,'A0 lifecycle/span differs')
  out=dict(targets=len(rows),priming_sources=0,target_interval_sum_ms=sum(r['application_e2e_ms'] for r in rows),source_to_last_target_span_ms=span,amortized_sequence_ms_per_target=span/len(rows),process_lifecycle_ms=life['process_wall_ms'],model_session_initialization_ms=initialization,warmup_ms=warmup)
 positives=[r for r in rows if r['scored'] and r['role']!='NEGATIVE_TARGET']
 require(positives,'no positive target measurements')
 out.update(positive_target_count=len(positives),negative_target_count=sum(r['role']=='NEGATIVE_TARGET' for r in rows),warm_positive_application_e2e_ms=sum(r['application_e2e_ms'] for r in positives)/len(positives),warm_positive_application_ttft_ms=sum(r['application_ttft_ms'] for r in positives)/len(positives),source_through_positive_span_ms=(positives[-1]['completion_ns']-rows[0]['arrival_ns'])/1e6,scope='Instrumented source-inclusive sequence and target intervals; lifecycle separate; nested timers are not additive; localhost TCP')
 return out

def evaluate_trace(tasks,rows):
 require([r['request_id'] for r in rows]==[t['request_id'] for t in tasks],'runtime request order/identity differs')
 source=rows[0];require(not source['scored'] and source['status']=='COMPLETE','source incomplete')
 checks=[]
 for task,target in zip(tasks[1:],rows[1:]):
  require(source['completion_ns']<target['arrival_ns'],'source not completed before target')
  if task['role']=='NEGATIVE_TARGET':
   checks.append(dict(request_id=target['request_id'],negative=True,pass_=target['source'] is None));continue
  grant=target['source'];ok=grant is not None
  if grant:
   require(source['admission'] is not None and grant['source_request_id']==source['request_id'] and grant['entry_id']==source['admission']['entry_id'],'grant references wrong/absent source')
   p=len(tasks[0]['token_ids']);require(target['reused_prefix_tokens']==p and target['skipped_prefix_projection_token_rows']==p*3*target['physical_identity']['layers'] and not target['cached_prefix_q_consumed'],'hidden prefix recomputation/Q consumption')
  checks.append(dict(request_id=target['request_id'],negative=False,pass_=ok,reused_tokens=target['reused_prefix_tokens']))
 return checks

def compute(experiment,run):
 protocol,plans=validate_plan(experiment);root=Path(run);receipt=Evidence(root,max_input_bytes=128*2**30).verify();status=read(root/'i5r3b_attempt_status.json')
 require(status['status']=='COMPLETE' and status['source_commit']==protocol['source']['commit'] and status['with_a0'] is True,'missing/incomplete ABC or separate A0 block')
 reports={};cost={};classes=[];traces=[];metrics=[]
 for name in ('abc','a0'):
  raw=root/(name+'_run');report=audit_i5(raw);cfg=report['config'];index=status['block_index'];expected=plans[name]['blocks'][index]
  require(cfg['block']==expected and cfg['source_commit']==protocol['source']['commit'] and cfg['sources']==protocol['source']['files'] and cfg['experiment_id']==plans[name]['experiment_id'] and Path(cfg['experiment_prepared']).resolve()==(Path(experiment)/name).resolve(),'run/plan/source mismatch')
  require(Path(cfg['multi_prepared']).resolve()==Path(protocol['multi_prepared']).resolve() and not cfg['fixture'],'real prepared multi-adapter evidence required')
  reports[name]=report;by={(r['condition'],r['request_id']):r for r in report['request_metrics']};cost[name]={}
  for condition in CONDITIONS:
   rows=[by[condition,t['request_id']] for t in expected['tasks']];info=report['independent_audit']['conditions'][condition]
   cost[name][condition]=scope_cost(rows,info['initialization']['model_and_session_initialization_ms'],info['warmup_total_ms'],read(raw/condition/'lifecycle.json'))
   if name=='abc':
    classes.extend(classify(r,report['cache_events']) for r in rows)
    if condition=='SEMCACHE_INTEGRATED':traces=evaluate_trace(expected['tasks'],rows)
   metrics.extend(dict(control=name,**r) for r in rows)
 require(reports['abc']['config']['expected_execution_gpu_uuid']==reports['a0']['config']['expected_execution_gpu_uuid'],'A0/ABC execution GPU mismatch; unmatched economics')
 abc=reports['abc'];a0=reports['a0'];targets=[t for t in plans['abc']['blocks'][status['block_index']]['tasks'] if t['scored']]
 correction_work=[]
 for task in targets:
  counts={}
  for condition in CONDITIONS:
   es=root/'abc_run'/condition/task['adapter_id']/'es'
   first=[s for s in jsonl(es/'r5_tokens.jsonl') if s['request_id']==task['request_id'] and s['step']==0]
   require(len(first)==1,'missing first-forward CPU correction context')
   frames=[f for f in jsonl(es/'wire.jsonl') if f['header']['kind']=='HIDDEN' and f['header']['request_id']==first[0]['forward_id']]
   require(len(frames)==protocol['model_identity']['layers'],'missing real CPU LoRA layer payloads')
   counts[condition]=3*sum(f['header']['shape'][-2] for f in frames)
  correction_work.append(dict(request_id=task['request_id'],actual_prefill_cpu_lora_token_role_layer_rows=counts,rows_omitted_A_vs_C=counts[CONDITIONS[0]]-counts[CONDITIONS[2]],basis='Audited actual first-forward HIDDEN TCP shapes; correction calls remain per layer'))
 for c in CONDITIONS:
  ar={r['request_id']:r for r in abc['request_metrics'] if r['condition']==c};br={r['request_id']:r for r in a0['request_metrics'] if r['condition']==c}
  for t in targets:
   x,y=ar[t['request_id']],br[t['request_id']]
   # Task sequence numbers differ with omitted source; compare actual immutable inputs instead.
   require(x['physical_identity']==y['physical_identity'] and x['sample_id']==y['sample_id'] and x['user_id']==y['user_id'] and x['input_tokens']==y['input_tokens'],'A0 target identity differs')
 a0_numerical=[]
 for task in targets:
  aa=next(r for r in abc['request_metrics'] if r['condition']==CONDITIONS[0] and r['request_id']==task['request_id'])
  bb=next(r for r in a0['request_metrics'] if r['condition']==CONDITIONS[0] and r['request_id']==task['request_id'])
  a0_numerical.append(fidelity(aa,bb,root/'abc_run'/CONDITIONS[0]/task['adapter_id']/'es',root/'a0_run'/CONDITIONS[0]/task['adapter_id']/'es'))
 layers=[]
 for task in targets:
  if task['role']=='NEGATIVE_TARGET':continue
  ar=next(r for r in abc['request_metrics'] if r['condition']==CONDITIONS[0] and r['request_id']==task['request_id'])
  cr=next(r for r in abc['request_metrics'] if r['condition']==CONDITIONS[2] and r['request_id']==task['request_id'])
  sa=next(s for s in jsonl(root/'abc_run'/CONDITIONS[0]/task['adapter_id']/'es/r5_tokens.jsonl') if s['request_id']==task['request_id'] and s['step']==0)
  sc=next(s for s in jsonl(root/'abc_run'/CONDITIONS[2]/task['adapter_id']/'es/r5_tokens.jsonl') if s['request_id']==task['request_id'] and s['step']==0)
  layers.append(dict(request_id=task['request_id'],**layer_diagnostics(root/'abc_run',ar,cr,sa,sc)))
 comparisons={}
 for a,b in ((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2])):
  comparisons[a+'_vs_'+b]={k:delta(cost['abc'][a][k],cost['abc'][b][k]) for k in ('warm_positive_application_e2e_ms','warm_positive_application_ttft_ms','source_to_last_target_span_ms','amortized_sequence_ms_per_target','process_lifecycle_ms')}
 economy=delta(cost['a0'][CONDITIONS[0]]['source_to_last_target_span_ms'],cost['abc'][CONDITIONS[2]]['source_to_last_target_span_ms'])
 numerical=abc['numerical_fidelity'];require(numerical['thresholds']==THRESHOLDS,'frozen thresholds differ')
 verdict=dict(FUNCTIONAL_PHYSICAL_REUSE='PASS' if traces and all(t['pass_'] for t in traces) else 'FAIL',CACHE_INTEGRITY=abc['decision']['cache_correctness'],NUMERICAL_FIDELITY=abc['decision']['numerical_tolerance'],TOKEN_TEXT_FIDELITY=abc['decision']['token_text_fidelity'],TASK_QUALITY='DESCRIPTIVE_QASPER_F1_8_TOKEN_BUDGET',WARM_TARGET_BENEFIT='PASS' if comparisons[CONDITIONS[0]+'_vs_'+CONDITIONS[2]]['warm_positive_application_e2e_ms']['absolute_delta']<0 else 'FAIL',AMORTIZED_E2E_BENEFIT='PASS' if economy['absolute_delta']<0 else 'FAIL',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',NATIVE_PEFT_STRICT_PARITY='UNESTABLISHED',STATISTICAL_EVIDENCE='ONE_INDEPENDENT_GROUP_DESCRIPTIVE_ONLY',ROOT_CAUSE='UNRESOLVED; recorded FP16 attention-score shape sensitivity does not establish a resolved kernel mechanism')
 verdict['NONEMPTY_GENERATED_OUTPUTS']='PASS' if all(r['generated_token_ids'] and r['generated_text'].strip() for r in abc['request_metrics'] if r['scored']) else 'FAIL'
 # All reconstruction/consumption details were independently replayed above;
 # last-position layer diagnostics are supplementary, not numerical causality.
 abc['numerical_fidelity']['prefill_last_position_layer_diagnostics']=layers
 return dict(source=protocol['source'],block_index=status['block_index'],independent_context_sha256=expected['independent_context_sha256'],receipt=receipt,verdicts=verdict,classifications=classes,physical_trace_checks=traces,cpu_correction_work=correction_work,costs=cost,comparisons=comparisons,A0_economic_comparison=economy,A0_scope='FULL_RECOMPUTE target-only reference; auxiliary B/C excluded from economic denominator',A0_vs_A_numerical=a0_numerical,metrics=metrics,numerical=numerical,quality=abc['quality_per_sample'],A0_quality=a0['quality_per_sample'],quality_deltas=abc['quality_deltas'],cache_summary=abc['cache_summary'],A0_cache_summary=a0['cache_summary'],matched_metrics=abc['paired_latency'],cache_events=abc['cache_events'],legacy_independent_audits={k:v['independent_audit'] for k,v in reports.items()},localhost_tcp=True,instrumented_performance_only=True)

def analyze(experiment,run,output):
 result=compute(experiment,run);out=new_output(output,(experiment,run));write(out/'i5r3b_results.json',result);write(out/'i5r3b_decision.json',result['verdicts']);write(out/'i5r3b_independent_audit.json',dict(status='PASS',source=result['source'],run_inventory=result['receipt'],scope='Unchanged strict I5 raw/consumed KV, TCP, isolation, full-vocabulary frozen logits and 3B immutable workloads; functional/numerical verdicts separate'));seal(out);return result['verdicts']

def verify(experiment,run,audit,output):
 result=compute(experiment,run);ev=Evidence(audit,max_input_bytes=256*2**20);receipt=ev.verify();a=ev.json('i5r3b_independent_audit.json')
 require(ev.json('i5r3b_results.json')==result and ev.json('i5r3b_decision.json')==result['verdicts'] and a['status']=='PASS' and a['source']==result['source'] and a['run_inventory']==result['receipt'],'independent 3B audit changed')
 out=new_output(output,(experiment,run,audit));write(out/'i5r3b_verification.json',dict(status='PASS',audit_inventory=receipt,source=result['source'],verdicts=result['verdicts']));seal(out);return dict(status='PASS',verdicts=result['verdicts'])
