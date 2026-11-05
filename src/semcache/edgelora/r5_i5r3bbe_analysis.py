"""Strict existing I5 audits plus measured, repeat-aware economic reporting."""
import statistics
from pathlib import Path
from .communication import require
from .r3a_artifacts import read,write,THRESHOLDS
from .r5_i5_evidence import seal,jsonl
from .r5_i5r2_tensors import Evidence
from .r5_i5_analysis import analyze as audit_i5
from .r5_i5r3b_analysis import classify,evaluate_trace,scope_cost
from .r5_i5r1_analysis import layer_diagnostics
from .r5_i5r3bbe import load,output,economics,CONDITIONS,A,C


def physical_checks(tasks,rows):
 checks=evaluate_trace(tasks,rows);source=rows[0];entry=None if source['admission'] is None else source['admission']['entry_id']
 for check,row in zip(checks,rows[1:]):
  if check['negative']:continue
  resident=entry is not None and entry in row['cache_before']['lru_to_mru'] and entry in row['cache_after']['lru_to_mru']
  check.update(source_resident_before_and_after=resident,no_positive_admission=row['admission'] is None,evictions=row['eviction_count'])
  check['pass_']=check['pass_'] and resident and row['admission'] is None
 return checks

def compute(experiment,runs):
 p,plans,base=load(experiment);expected=set(range(p['repetitions']));seen=set();blocks=[];receipts=[]
 for run in runs:
  root=Path(run);receipt=Evidence(root,max_input_bytes=128*2**30).verify();status=read(root/'i5r3bbe_attempt_status.json');index=status['block_index']
  require(index in expected and index not in seen and status['status']=='COMPLETE' and status['source_commit']==p['source']['commit'] and status['execution_order']==p['execution_order'][index],'missing/duplicate/failed BE paired execution')
  seen.add(index);reports={};costs={};rows_by={};bounds={};sessions=[]
  for control in ('abc','a0'):
   raw=root/(control+'_run');r=audit_i5(raw);cfg=r['config'];b=plans[control]['blocks'][index]
   require(cfg['block']==b and cfg['experiment_id']==plans[control]['experiment_id'] and cfg['sources']==p['source']['files'] and cfg['source_commit']==p['source']['commit'] and Path(cfg['experiment_prepared']).resolve()==(Path(experiment)/control).resolve() and not cfg['fixture'],'BE pinned plan/execution differs')
   reports[control]=r;costs[control]={};bounds[control]=[]
   for cond in CONDITIONS:
    rows=[next(x for x in r['request_metrics'] if x['condition']==cond and x['request_id']==t['request_id']) for t in b['tasks']];rows_by[control,cond]=rows;info=r['independent_audit']['conditions'][cond];life=read(raw/cond/'lifecycle.json');bounds[control].append(life)
    costs[control][cond]=scope_cost(rows,info['initialization']['model_and_session_initialization_ms'],info['warmup_total_ms'],life)
    sessions.extend(a['session_id'] for a in info['adapters'].values())
  require(len(sessions)==len(set(sessions)),'ABC/A0 reused a CPU UD session')
  first,last=p['execution_order'][index]
  require(max(x['execution_end_ns'] for x in bounds[first])<=min(x['execution_start_ns'] for x in bounds[last]),'actual paired execution order differs')
  require(reports['abc']['config']['expected_execution_gpu_uuid']==reports['a0']['config']['expected_execution_gpu_uuid'],'unmatched execution GPU')
  cr,ar=rows_by['abc',C],rows_by['a0',A];economy=economics(cr,ar);checks=physical_checks(plans['abc']['blocks'][index]['tasks'],cr)
  numerics=reports['abc']['numerical_fidelity'];require(numerics['thresholds']==THRESHOLDS,'frozen thresholds changed')
  classes=[classify(row,reports['abc']['cache_events']) for cond in CONDITIONS for row in rows_by['abc',cond]]
  layers=[]
  for row in cr[1:-1]:
   aa=next(x for x in rows_by['abc',A] if x['request_id']==row['request_id']);sa=next(s for s in jsonl(root/'abc_run'/A/row['adapter_id']/'es/r5_tokens.jsonl') if s['request_id']==row['request_id'] and s['step']==0);sc=next(s for s in jsonl(root/'abc_run'/C/row['adapter_id']/'es/r5_tokens.jsonl') if s['request_id']==row['request_id'] and s['step']==0)
   layers.append(dict(request_id=row['request_id'],**layer_diagnostics(root/'abc_run',aa,row,sa,sc)))
  functional=all(c['pass_'] for c in checks)
  verdicts=dict(FUNCTIONAL_PHYSICAL_REUSE='PASS' if functional else 'FAIL',CACHE_INTEGRITY=reports['abc']['decision']['cache_correctness'],NUMERICAL_FIDELITY=reports['abc']['decision']['numerical_tolerance'],TOKEN_TEXT_FIDELITY=reports['abc']['decision']['token_text_fidelity'],MEASURED_ECONOMIC_BREAK_EVEN='PASS' if economy['D_ms']<0 else 'FAIL',MEASURED_PHYSICAL_HIT_BREAK_EVEN='PASS' if functional and economy['D_ms']<0 else 'FAIL',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',NATIVE_PEFT_STRICT_PARITY='UNESTABLISHED')
  blocks.append(dict(block_index=index,execution_order=p['execution_order'][index],physical_reuse_pass=functional,verdicts=verdicts,economics=economy,costs=costs,actual_positive_requests=p['positive_requests'],actual_physical_hits=economy['actual_physical_hits'],physical_checks=checks,classifications=classes,hit_metrics=cr[1:-1],all_request_metrics=[dict(control=control,**x) for control,r in reports.items() for x in r['request_metrics']],numerical=numerics,layer_diagnostics=layers,quality_per_request=reports['abc']['quality_per_sample'],unique_scored_samples=3,repeated_request_observations=p['positive_requests']-2,independent_context_groups=1,legacy_strict_audits={k:v['independent_audit'] for k,v in reports.items()},runtime_inventories=receipt))
  receipts.append(dict(run=str(root.resolve()),receipt=receipt))
 require(seen==expected,'missing paired repetition blocks; no pooling partial evidence')
 blocks.sort(key=lambda b:b['block_index']);ds=[b['economics']['D_ms'] for b in blocks]
 return dict(schema='I5R3B_BE_RESULTS_V1',source=p['source'],GPU_NOT_RUN=False,positive_requests=p['positive_requests'],baseline_N2=base['estimate'],baseline_verdicts=base['report']['verdicts'],baseline_separate_not_pooled=True,blocks=blocks,run_inventories=receipts,statistical_scope=dict(fresh_paired_execution_blocks=len(blocks),independent_context_groups=1,mean_D_ms=statistics.mean(ds),sample_variance_D_ms=statistics.variance(ds) if len(ds)>1 else None,confidence_interval='NOT_ESTABLISHED: one shared context, bounded order-sensitive repeats'),label='CONTROLLED_REPEAT_AMORTIZATION',ROOT_CAUSE='UNRESOLVED',NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',correctness_qualified_speedup='NOT_ESTABLISHED')

def analyze(experiment,runs,output_dir):
 result=compute(experiment,runs);out=output(output_dir,(experiment,*runs));write(out/'i5r3bbe_results.json',result);write(out/'i5r3bbe_independent_audit.json',dict(status='PASS',source=result['source'],run_inventories=result['run_inventories'],scope='Unchanged I5 strict KV/TCP/logit audit; BE canonical repeat contract; measured timing decomposition; numerical/functional FAIL remain separate'));seal(out);return dict(status='EVIDENCE_AUDIT_PASS',verdicts=[b['verdicts'] for b in result['blocks']])

def verify(experiment,runs,audit,output_dir):
 result=compute(experiment,runs);ev=Evidence(audit,max_input_bytes=256*2**20);receipt=ev.verify();a=ev.json('i5r3bbe_independent_audit.json')
 require(ev.json('i5r3bbe_results.json')==result and a['status']=='PASS' and a['source']==result['source'] and a['run_inventories']==result['run_inventories'],'BE independent evidence tampering')
 out=output(output_dir,(experiment,*runs,audit));write(out/'i5r3bbe_verification.json',dict(status='PASS',audit_inventory=receipt,source=result['source'],numerical_failures_preserved=True));seal(out);return dict(status='PASS')
