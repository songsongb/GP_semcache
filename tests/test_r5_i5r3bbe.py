"""Synthetic CPU economics/contracts only; not real GPU performance evidence."""
import copy,contextlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,create_autospec
from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import read,write,THRESHOLDS
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5r3b_plan import build_blocks,CONDITIONS
from semcache.edgelora.r5_i5_workload import routing
from semcache.edgelora import r5_i5r3bbe as be
from semcache.edgelora import r5_i5r3bbe_analysis as analysis
from test_r5_i5r3b import prompt,USERS,Tokenizer,SPECIAL


def original():
 ps=[]
 for user,ctx in (('User_08','shared'),('User_06','shared'),('User_10','negative')):
  for i in range(2000):
   p=prompt(user+str(i),ctx,[1 if ctx=='shared' else 3]*1279+[len(ps)+1])
   if routing(p['sample'])['user_id']==user:ps.append(p);break
 return build_blocks([dict(targets=ps,context_sha256='f'*64,adapter='Adapter_A')],'B2_SMOKE',SPECIAL,Tokenizer(),USERS,42)[0]

def row(rid,sample,ms,start,role='SCORED_TARGET',scored=True):
 return dict(request_id=rid,sample_id=sample,user_id='User_08',adapter_id='Adapter_A',scored=scored,role=role,status='COMPLETE',arrival_ns=start,completion_ns=start+int(ms*1e6),application_e2e_ms=ms,model_execution_e2e_ms=ms*.7,cache_admission_eviction_ms=ms*.1,source=None)

def measured():
 # C=20+5+7+6+10gap=48; A0=12+10+5+4gap=31; D=17 ms.
 cr=[row('S','a',20,0,'CONTROLLED_STANDALONE_SOURCE',False),row('T1','a',5,22000000),row('T2','b',7,30000000),row('N','n',6,42000000,'NEGATIVE_TARGET')]
 ar=[row('T1','a',12,0),row('T2','b',10,14000000),row('N','n',5,26000000,'NEGATIVE_TARGET')]
 for r in cr[1:-1]:r['source']={'entry_id':'e'}
 return cr,ar


def base_fixture():
 b=original();specs={a:dict(adapter_id=a,matrix_sha256=digest(a),rank=8,matrix_dtype='float32') for a in ('Adapter_A','Adapter_B','Adapter_C')};pm={'identity':dict(model='facebook/opt-2.7b',base_revision='a'*40,base_weight_sha256='b'*64,dtype='float16',hidden_dtype='float16',layers=32,hidden_size=2560)};tok={'tokenizer_revision':'c'*40}
 p=dict(source={'commit':be.BASE_COMMIT,'files':{}},profile='B2_SMOKE',max_prompt_tokens=1280,model_identity=pm['identity'],adapter_fingerprints={a:s['matrix_sha256'] for a,s in specs.items()},multi_prepared='fixture_adapters')
 plans={k:dict(blocks=[copy.deepcopy(b)],tokenizer_identity=tok,seed=42,context_limit=2048,TEST_FIXTURE_ONLY=False,conditions=list(CONDITIONS),i5r3b=p) for k in ('abc','a0')}
 plans['a0']['blocks'][0]['tasks']=plans['a0']['blocks'][0]['tasks'][1:]
 for ex in plans.values():ex['experiment_id']=digest(ex)
 base=dict(block=b,plans=plans,protocol=p,report={'verdicts':{'NUMERICAL_FIDELITY':'FAIL'}},estimate=be.estimate(be.economics(*measured())))
 return base,pm,specs,{'tokenizer':tok}

class Economics(unittest.TestCase):
 def test_source_gap_negative_and_positive_decomposition(self):
  m=be.economics(*measured());self.assertEqual(m['D_ms'],17);self.assertEqual(m['source_cost_ms'],20);self.assertEqual(m['negative_target_overhead_ms'],1);self.assertEqual(m['sequence_gap_overhead_ms'],6);self.assertEqual(m['total_positive_target_savings_ms'],10)
  self.assertEqual(m['observed_amortized_e2e_ms'],16);self.assertEqual(m['scored_request_denominator'],3);self.assertFalse(m['measured_break_even'])
 def test_sensitivity_not_a_measured_n3(self):
  e=be.estimate(be.economics(*measured()));self.assertEqual(e['estimated_minimum_reuses'],5);self.assertEqual(e['scenarios']['repeat_target1']['estimated_N3_D_ms'],10);self.assertEqual(e['scenarios']['minimum_observed_saving']['estimated_minimum_reuses'],8);self.assertEqual(e['scenarios']['mean']['estimated_minimum_reuses'],6);self.assertEqual(e['scenarios']['repeat_target1']['status'],'EXTRAPOLATED_NOT_MEASURED')
 def test_strict_negative_break_even_not_zero(self):
  m=be.economics(*measured());m['D_ms']=14;e=be.estimate(m);self.assertEqual(e['estimated_minimum_reuses'],5)
 def test_no_positive_savings_does_not_invent_crossing(self):
  m=be.economics(*measured())
  for p in m['positive_target_savings_ms']:p['savings_ms']=-1
  self.assertIsNone(be.estimate(m)['estimated_minimum_reuses'])
 def test_negative_observed_difference_is_separate(self):
  cr,ar=measured();cr[0]['application_e2e_ms']=1;cr[0]['completion_ns']=1000000
  # Move all later C requests to keep gaps unchanged, not just a nested timer.
  for r in cr[1:]:r['arrival_ns']-=19000000;r['completion_ns']-=19000000
  m=be.economics(cr,ar);self.assertLess(m['D_ms'],0);self.assertTrue(m['measured_break_even'])
 def test_nested_source_timers_do_not_change_economics(self):
  cr,ar=measured();first=be.economics(cr,ar);cr[0]['model_execution_e2e_ms']=999;cr[0]['cache_admission_eviction_ms']=888
  self.assertEqual(be.economics(cr,ar)['D_ms'],first['D_ms'])
 def test_no_fake_source_or_unmatched_a0(self):
  cr,ar=measured();cr[0]['scored']=True
  with self.assertRaises(ValueError):be.economics(cr,ar)
  cr,ar=measured();ar[0]['request_id']='wrong'
  with self.assertRaises(ValueError):be.economics(cr,ar)
 def test_reordered_or_corrupt_timing_rejected(self):
  cr,ar=measured();cr[1]['arrival_ns']=0
  with self.assertRaises(ValueError):be.economics(cr,ar)
  cr,ar=measured();ar[0]['application_e2e_ms']=900
  with self.assertRaises(ValueError):be.economics(cr,ar)

class RepeatContract(unittest.TestCase):
 def test_exact_repeat_order_unique_requests_and_original_samples(self):
  b=original();r=be.repeat_blocks(b)[0];stats=be.check_repeat(r,b,USERS,3);s,t1,t2,repeat,n=r['tasks']
  for key in t1:
   if key not in ('request_id','request_sequence_number'):self.assertEqual(repeat[key],t1[key])
  self.assertNotEqual(repeat['request_id'],t1['request_id']);self.assertEqual([t['user_id'] for t in r['tasks'][1:]],['User_08','User_06','User_08','User_10']);self.assertEqual(stats['total_scored_requests'],4);self.assertEqual(stats['unique_scored_samples'],3);self.assertEqual(stats['repeated_request_observations'],1);self.assertEqual(stats['distinct_original_users'],3);self.assertEqual(stats['independent_context_groups'],1)
 def test_generic_3b_repetition_guard_not_weakened(self):
  from semcache.edgelora.r5_i5r3b_plan import checks
  with self.assertRaises(ValueError):checks(be.repeat_blocks(original())[0],USERS,42)
 def test_changed_repeat_prompt_policy_user_or_id_rejected(self):
  b=original()
  for field,value in (('user_id','User_11'),('sample_id','new_sample'),('token_ids',[1]),('request_id',b['tasks'][1]['request_id'])):
   r=be.repeat_blocks(b)[0];r['tasks'][3][field]=value
   with self.assertRaises(ValueError):be.check_repeat(r,b,USERS,3)
 def test_alternating_extra_repeats_and_resource_bound(self):
  b=original();r=be.repeat_blocks(b,4)[0]
  self.assertEqual([t['sample_id'] for t in r['tasks'][1:-1]],[b['tasks'][1]['sample_id'],b['tasks'][2]['sample_id']]*2)
  with self.assertRaises(ValueError):be.repeat_blocks(b,100)
  with self.assertRaises(ValueError):be.approval(None,4)
 def test_repetitions_rotate_conditions_but_one_context(self):
  rs=be.repeat_blocks(original(),3,3);self.assertEqual(len(rs),3);self.assertNotEqual(rs[0]['condition_order'],rs[1]['condition_order']);self.assertEqual(len({r['independent_context_sha256'] for r in rs}),1)
 def test_a0_has_only_identical_scored_requests(self):
  base,_,_,_=base_fixture();plans=be.child_plans(base,{'fixture':True},3,1);a=plans['abc']['blocks'][0];b=plans['a0']['blocks'][0]
  self.assertEqual(len(a['tasks']),5);self.assertEqual(len(b['tasks']),4);self.assertTrue(all(t['scored'] for t in b['tasks']));self.assertEqual([t['request_id'] for t in b['tasks']],[t['request_id'] for t in a['tasks'][1:]]);self.assertNotIn('i5r3b',plans['abc'])
 def test_every_request_passes_unchanged_transport_headroom(self):
  from semcache.edgelora.r5_i5r3b_transport import preflight
  base,pm,specs,m=base_fixture();r=preflight(be.child_plans(base,{},3,1),pm,specs,m['tokenizer'],USERS)
  self.assertEqual(len(r['rows']),9);self.assertTrue(all(x['remaining_headroom_bytes']>=8192 for x in r['rows']))
 def trace(self):
  tasks=be.repeat_blocks(original())[0]['tasks'];sid=tasks[0]['request_id'];source=dict(request_id=sid,scored=False,status='COMPLETE',completion_ns=1,admission={'entry_id':'e'})
  rows=[source]
  for t in tasks[1:]:
   negative=t['role']=='NEGATIVE_TARGET';rows.append(dict(request_id=t['request_id'],arrival_ns=2,source=None if negative else dict(entry_id='e',source_request_id=sid),reused_prefix_tokens=0 if negative else 512,skipped_prefix_projection_token_rows=49152,cached_prefix_q_consumed=False,physical_identity={'layers':32},cache_before={'lru_to_mru':['e']},cache_after={'lru_to_mru':['e']},admission=None,eviction_count=0))
  return tasks,rows
 def test_resident_three_hits_zero_evictions(self):
  t,r=self.trace();self.assertTrue(all(x['pass_'] for x in analysis.physical_checks(t,r)))
 def test_missing_residency_or_unexpected_admission_is_not_success(self):
  for key,value in (('cache_after',{'lru_to_mru':[]}),('admission',{'entry_id':'new'}),('source',None)):
   t,r=self.trace();r[3][key]=value;self.assertFalse(analysis.physical_checks(t,r)[2]['pass_'])
 def test_false_grant_and_negative_hit_rejected(self):
  t,r=self.trace();r[3]['source']['source_request_id']='other'
  with self.assertRaises(ValueError):analysis.physical_checks(t,r)
  t,r=self.trace();r[-1]['source']={'entry_id':'e'};self.assertFalse(analysis.physical_checks(t,r)[-1]['pass_'])
 def test_frozen_thresholds_and_no_padding_mode(self):
  self.assertEqual(THRESHOLDS['max_abs'],.01);self.assertEqual(THRESHOLDS['relative_l2'],.001);base,_,_,_=base_fixture();plans=be.child_plans(base,{},3,1);self.assertNotIn('i5r2c2',plans['abc']);self.assertNotIn('i5r2b',plans['abc'])

class Integrity(unittest.TestCase):
 def test_archived_baseline_reader_real_seals_and_raw_crosschecks(self):
  # Sparse synthetic metadata fixture for the historical reader, NOT GPU evidence.
  from semcache.edgelora.r5_i5r3b_transport import predicted_request
  base,pm,specs,m=base_fixture();cr,ar=measured()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);paths={k:root/k for k in ('experiment','run','audit','verification')}
   for p in paths.values():p.mkdir()
   proto=base['protocol'];write(paths['experiment']/'i5r3b_protocol.json',proto)
   exported=[];legacy={}
   for name in ('abc','a0'):
    ex=base['plans'][name];child=paths['experiment']/name;child.mkdir();write(child/'i5_workload_manifest.json',ex);seal(child)
    raw=paths['run']/(name+'_run');raw.mkdir();cfg=dict(experiment_id=ex['experiment_id'],sources={},source_commit=be.BASE_COMMIT,block=ex['blocks'][0],fixture=False,expected_execution_gpu_uuid='TEST_FIXTURE_UUID')
    write(raw/'i5_execution_config.json',cfg);write(raw/'i5_block_status.json',{'status':'COMPLETE','GPU_NOT_RUN':False})
    legacy[name]={'status':'PASS','conditions':{}}
    for cond in CONDITIONS:
     dest=raw/cond;dest.mkdir();gpu=dict(TEST_FIXTURE_ONLY=False,gpu_class_compatible=True,execution_gpu_name='TEST_FIXTURE RTX 3090',execution_gpu_uuid='TEST_FIXTURE_UUID',historical_live_kv_restored=False);legacy[name]['conditions'][cond]={'gpu_identity':gpu}
     rows=copy.deepcopy(cr if name=='abc' else ar);arrivals=[]
     for r,t in zip(rows,ex['blocks'][0]['tasks']):
      r.update(request_id=t['request_id'],sample_id=t['sample_id'],user_id=t['user_id'],adapter_id=t['adapter_id'],condition=cond,reused_prefix_tokens=512 if name=='abc' and r['source'] else 0)
      req=predicted_request(t,pm,specs[t['adapter_id']],m['tokenizer']);arrivals.append({'kind':'ARRIVAL','request':req})
      exported.append(dict(control=name,**r))
     (dest/'generation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(dest/'requests.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in arrivals))
    seal(raw);audit=paths['run']/(name+'_run_audit');audit.mkdir();write(audit/'i5_independent_audit.json',legacy[name]);seal(audit)
   write(paths['run']/'i5r3b_attempt_status.json',dict(status='COMPLETE',with_a0=True,source_commit=be.BASE_COMMIT,block_index=0));seal(paths['run']);receipt=Evidence(paths['run']).verify()
   verdicts=dict(FUNCTIONAL_PHYSICAL_REUSE='PASS',CACHE_INTEGRITY='PASS',NUMERICAL_FIDELITY='FAIL',TOKEN_TEXT_FIDELITY='PASS');r=dict(source=proto['source'],receipt=receipt,verdicts=verdicts,numerical={'thresholds':THRESHOLDS},legacy_independent_audits=legacy,metrics=exported,A0_economic_comparison={'absolute_delta':17})
   write(paths['audit']/'i5r3b_results.json',r);write(paths['audit']/'i5r3b_decision.json',verdicts);write(paths['audit']/'i5r3b_independent_audit.json',dict(status='PASS',source=proto['source'],run_inventory=receipt));seal(paths['audit']);audit_receipt=Evidence(paths['audit']).verify()
   write(paths['verification']/'i5r3b_verification.json',dict(status='PASS',source=proto['source'],audit_inventory=audit_receipt,verdicts=verdicts));seal(paths['verification']);seal(paths['experiment'])
   with patch.object(be,'archived_source'):
    parsed=be.baseline(**paths);self.assertEqual(parsed['measured']['D_ms'],17);self.assertEqual(parsed['estimate']['baseline_verdicts']['NUMERICAL_FIDELITY'],'FAIL')
    with self.assertRaisesRegex(ValueError,'source differs'):be.baseline(**paths,expected_commit='b'*40)
    # Resealed wrong summary still cannot replace actual request timing data.
    r['metrics'][1]['application_e2e_ms']=999;write(paths['audit']/'i5r3b_results.json',r);seal(paths['audit'])
    with self.assertRaises(ValueError):be.baseline(**paths)
 def test_plan_prepare_load_real_seals_synthetic_assets(self):
  base,pm,specs,m=base_fixture()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base['paths']={k:str(root/k) for k in ('experiment','run','audit','verification')};base['receipts']={'TEST_FIXTURE_ONLY':True};base['protocol']['multi_prepared']=str(root/'multi');multi=root/'multi';multi.mkdir();write(multi/'fixture.json',{});write(multi/'r5_output_manifest.json',{'files':{'fixture.json':__import__('hashlib').sha256((multi/'fixture.json').read_bytes()).hexdigest()}})
   for name in ('abc','a0'):
    p=root/'experiment'/name;p.mkdir(parents=True);write(p/'i5_dataset_manifest.json',{'TEST_FIXTURE_ONLY':True})
   source={'commit':'TEST_FIXTURE_ONLY','files':{}}
   with patch.object(be,'baseline',return_value=base),patch.object(be,'source_identity',return_value=source),patch.object(be,'compatible_preparation',return_value=(m,pm,specs,USERS)):
    report=be.prepare(*(root/k for k in ('experiment','run','audit','verification')),output_dir=root/'plan')
    self.assertEqual(report['blocks'],1);p,plans,_=be.load(root/'plan');self.assertEqual(p['execution_order'],[['abc','a0']]);self.assertEqual(len(plans['abc']['blocks'][0]['tasks']),5)
    receipt=root/'plan/i5r3bbe_selection.json';write(receipt,[]);seal(root/'plan')
    with self.assertRaises(ValueError):be.load(root/'plan')
 def test_independent_audit_tamper_is_rejected(self):
  result={'source':{'TEST_FIXTURE_ONLY':True},'run_inventories':[],'GPU_NOT_RUN':True}
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'audit';a.mkdir();write(a/'i5r3bbe_results.json',result);write(a/'i5r3bbe_independent_audit.json',{'status':'PASS','source':result['source'],'run_inventories':[]});seal(a)
   with patch.object(analysis,'compute',return_value=result):
    self.assertEqual(analysis.verify('p',['r'],a,root/'verified')['status'],'PASS');write(a/'i5r3bbe_results.json',{'altered':True});seal(a)
    with self.assertRaises(ValueError):analysis.verify('p',['r'],a,root/'bad')
 def test_missing_or_corrupted_baseline_evidence_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with self.assertRaises(ValueError):be.baseline(root/'missing',root/'run',root/'audit',root/'verification')
   for k in ('experiment','run','audit','verification'):
    p=root/k;p.mkdir();write(p/'fixture.json',{});seal(p)
   write(root/'experiment/fixture.json',{'changed':True})
   with self.assertRaises(ValueError):be.baseline(*(root/k for k in ('experiment','run','audit','verification')))
 def test_fresh_outputs_reject_history_overlap(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=root/'old';old.mkdir()
   with self.assertRaises(ValueError):be.output(old/'new',[old])
 def test_cli_dispatch_signature(self):
  import sys
  spec=importlib.util.spec_from_file_location('be_cli',Path(__file__).resolve().parents[1]/'scripts/146_r5_i5r3bbe.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
  fn=create_autospec(cli.api.prepare,return_value={'TEST_FIXTURE_ONLY':True});argv=['146','prepare','--experiment','p','--run','r','--audit','a','--verification','v','--output-dir','fresh']
  with patch.object(sys,'argv',argv),patch.object(cli.api,'prepare',fn),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(cli.main(),0)
  fn.assert_called_once_with(experiment='p',run='r',audit='a',verification='v',output_dir='fresh',expected_commit=be.BASE_COMMIT,positive_requests=3,repetitions=1,validated_be_audit=None)

if __name__=='__main__':unittest.main()
