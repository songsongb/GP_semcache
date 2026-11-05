"""CPU synthetic workloads only; no claims about real GPU latency or HITs."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import write,read,THRESHOLDS
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5_data import normalize
from semcache.edgelora.r5_i5_workload import routing
from semcache.edgelora.r5_multi_artifacts import registry_for
from semcache.edgelora.r5_i5r3b_plan import shared_pairs,selection,build_blocks,checks,lcp,CAPACITY,history
from semcache.edgelora.r5_i5r3b_analysis import classify,scope_cost,evaluate_trace,verify

SPECIAL=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0)
USERS=registry_for(['Adapter_A','Adapter_B','Adapter_C'],'expanded')
class Tokenizer:
 bos_token_id=0;eos_token_id=6;pad_token_id=0
 def decode(self,ids,**kw):return 'TEST_FIXTURE_ONLY_SEMANTIC_TEXT'

def prompt(rid,ctx,ids):
 s=normalize('qasper',[dict(_id=rid,context=ctx,input='Question '+rid,answers=['gold'],length=100,TEST_FIXTURE_ONLY=True)],fixture=True)[0]
 return dict(sample=s,route=routing(s),token_ids=list(ids),token_ids_sha256=digest(list(ids)),input_text='TEST_FIXTURE_ONLY',template_version='I5_PROMPTS_V1',max_new_tokens=8,context_limit=2048,original_context_tokens=1200,retained_context_tokens=1200,removed_context_tokens=0,original_prompt_tokens=len(ids),retained_prompt_tokens=len(ids),truncated=False)

def corpus():
 result=[]
 for aid in ('Adapter_A','Adapter_B','Adapter_C'):
  found=[]
  for i in range(500):
   p=prompt(aid+str(i),'context_'+aid,[1]*1000+[2,i%5+1])
   if p['route']['adapter_id']==aid and (not found or p['route']['user_id']!=found[0]['route']['user_id']):found.append(p)
   if len(found)==2:break
  found[1]['token_ids'][-1]=9;found[1]['token_ids_sha256']=digest(found[1]['token_ids']);result+=found
  for i in range(500):
   neg=prompt('negative_'+aid+str(i),'negative_context_'+aid,[3]*700)
   if neg['route']['adapter_id']==aid:result.append(neg);break
 return result

class Workload(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.prompts=corpus()
 def block(self,profile='B2_SMOKE'):
  chosen,_=selection(copy.deepcopy(self.prompts),USERS,profile)
  return build_blocks(chosen,profile,SPECIAL,Tokenizer(),USERS,42)[0]
 def test_distinct_ids_shared_context_and_lcp(self):
  ps=shared_pairs(self.prompts,USERS);self.assertEqual(len(ps),3);self.assertTrue(all(p['a']['sample']['id']!=p['b']['sample']['id'] and p['common_tokens']>=1000 for p in ps));self.assertEqual(lcp([1,2],[1,3]),1)
 def test_no_shared_context_or_prefix_rejected(self):
  for change in ('context','prefix'):
   ps=copy.deepcopy(self.prompts[:2])
   if change=='context':ps[1]['sample']['context']='different'
   else:ps[1]['token_ids'][0]=99
   with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):selection(ps,USERS,'B2_SMOKE')
 def test_512_source_exact_target_order_negative_and_label(self):
  b=self.block();s,a,c,n=b['tasks'];self.assertEqual(s['token_ids'],a['token_ids'][:512]);self.assertEqual(s['token_ids'],c['token_ids'][:512]);self.assertNotEqual(s['token_ids'],n['token_ids'][:512]);self.assertEqual(s['role'],'CONTROLLED_STANDALONE_SOURCE');self.assertEqual(n['role'],'NEGATIVE_TARGET');self.assertEqual([t['request_sequence_number'] for t in b['tasks']],list(range(4)));self.assertEqual(b['track'],'CONTROLLED')
 def test_authorized_cross_user_original_routing_preserved(self):
  b=self.block();self.assertNotEqual(b['tasks'][1]['user_id'],b['tasks'][2]['user_id']);self.assertEqual(len(checks(b,USERS,42)),3)
 def test_unauthorized_adapter_and_user_rejected(self):
  b=self.block();b['tasks'][2]['adapter_id']='Adapter_B'
  with self.assertRaises(ValueError):checks(b,USERS,42)
  b=self.block();u=copy.deepcopy(USERS);u['users'][b['tasks'][2]['user_id']]['allow_cross_user_reuse']=False
  with self.assertRaises(ValueError):checks(b,u,42)
 def test_real_raw_qkv_capacity_and_entry_policy(self):
  b=self.block();self.assertEqual(b['resources']['per_adapter_capacity_bytes'],CAPACITY);self.assertEqual(b['pair_checks'][0]['source_raw_qkv_bytes'],251658240);self.assertEqual(b['resources']['raw_qkv_bytes_per_token'],491520)
 def test_no_cache_capacity_changes(self):
  b=self.block();b['resources']['per_adapter_capacity_bytes']=1
  with self.assertRaises(ValueError):checks(b,USERS,42)
 def test_long_position_and_policy_bounds(self):
  b=self.block();b['tasks'][1]['token_ids']=[1]*2048
  with self.assertRaises(ValueError):checks(b,USERS,42)
 def test_source_after_target_and_block_boundary_rejected(self):
  for tasks in (self.block()['tasks'][1:],list(reversed(self.block()['tasks']))):
   b=self.block();b['tasks']=tasks
   with self.assertRaises(ValueError):checks(b,USERS,42)
  b=self.block();b['cache_lifetime_requests']=2
  with self.assertRaisesRegex(ValueError,'boundary'):checks(b,USERS,42)
 def test_source_not_completed_before_target(self):
  tasks=self.block()['tasks'];rows=[dict(request_id=t['request_id'],scored=t['scored'],status='COMPLETE',completion_ns=100,arrival_ns=50) for t in tasks]
  with self.assertRaisesRegex(ValueError,'not completed'):evaluate_trace(tasks,rows)
  rows[0]['status']='FAILED'
  with self.assertRaisesRegex(ValueError,'incomplete'):evaluate_trace(tasks,rows)
 def test_claimed_hit_must_omit_actual_prefix_projection_rows(self):
  tasks=self.block()['tasks'];sid=tasks[0]['request_id']
  rows=[dict(request_id=sid,scored=False,status='COMPLETE',completion_ns=1,admission={'entry_id':'e'})]
  for task in tasks[1:]:
   negative=task['role']=='NEGATIVE_TARGET'
   rows.append(dict(request_id=task['request_id'],arrival_ns=2,source=None if negative else {'entry_id':'e','source_request_id':sid},reused_prefix_tokens=0 if negative else 512,skipped_prefix_projection_token_rows=512*3*32,cached_prefix_q_consumed=False,physical_identity={'layers':32}))
  self.assertTrue(all(r['pass_'] for r in evaluate_trace(tasks,rows)))
  rows[1]['skipped_prefix_projection_token_rows']=0
  with self.assertRaisesRegex(ValueError,'recomputation'):evaluate_trace(tasks,rows)
 def test_duplicate_identity_and_repeated_targets_rejected(self):
  with self.assertRaises(ValueError):shared_pairs([self.prompts[0],self.prompts[0]],USERS)
  b=self.block();b['tasks'][2]['sample_id']=b['tasks'][1]['sample_id']
  with self.assertRaises(ValueError):checks(b,USERS,42)
 def test_no_hidden_grants_or_runtime_fallback(self):
  b=self.block();self.assertTrue(all(not p['runtime_HIT_authorized_by_static_check'] for p in b['pair_checks']))
  b['tasks'][1]['forced_hit']=True
  with self.assertRaises(ValueError):checks(b,USERS,42)
 def test_three_independent_contexts_all_adapters(self):
  chosen,_=selection(copy.deepcopy(self.prompts),USERS,'B2_THREE_ADAPTER');bs=build_blocks(chosen,'B2_THREE_ADAPTER',SPECIAL,Tokenizer(),USERS,42)
  self.assertEqual(len(bs),3);self.assertEqual(len({b['independent_context_sha256'] for b in bs}),3);self.assertEqual({b['tasks'][0]['adapter_id'] for b in bs},{'Adapter_A','Adapter_B','Adapter_C'});self.assertNotEqual(bs[0]['condition_order'],bs[1]['condition_order'])
 def test_b1_64_256_same_user_and_eight_tokens(self):
  p=copy.deepcopy(self.prompts[0]);p['token_ids']=p['token_ids'][:256];p['token_ids_sha256']=digest(p['token_ids']);chosen,_=selection([p],USERS,'B1');b=build_blocks(chosen,'B1',SPECIAL,Tokenizer(),USERS,42)[0]
  self.assertEqual([len(t['token_ids']) for t in b['tasks']],[64,256]);self.assertEqual([t['policy']['max_new_tokens'] for t in b['tasks']],[1,8]);self.assertEqual(b['tasks'][0]['user_id'],b['tasks'][1]['user_id'])
 def test_deterministic_selection(self):self.assertEqual(selection(copy.deepcopy(self.prompts),USERS,'B2_SMOKE'),selection(copy.deepcopy(self.prompts),USERS,'B2_SMOKE'))
 def test_sealed_plan_roundtrip_a0_target_only_and_tamper(self):
  from types import SimpleNamespace
  from semcache.edgelora import r5_i5r3b_plan as api
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);multi=root/'multi';multi.mkdir();write(multi/'fixture.json',{'TEST_FIXTURE_ONLY':True});write(multi/'r5_output_manifest.json',{'files':{'fixture.json':__import__('hashlib').sha256((multi/'fixture.json').read_bytes()).hexdigest()}})
   ds=root/'data';ds.mkdir();write(ds/'samples.json',{'TEST_FIXTURE_ONLY':True});seal(ds)
   raw={p['sample']['id']:p for p in self.prompts};data={'qasper':[p['sample'] for p in self.prompts]};prov={'qasper':dict(path=str(ds),manifest={'TEST_FIXTURE_ONLY':True})}
   pm={'identity':{'TEST_FIXTURE_ONLY':True,'layers':32}};m={'tokenizer':{}};tok=SimpleNamespace(identity={},tokenizer=Tokenizer())
   old=dict(registry_path=str(root/'registry'),multi_prepared=str(multi),users=USERS,model_identity=pm['identity'],tokenizer_identity={},seed=42,input_limit=None)
   result={'qasper':dict(samples=[dict(id=k,token_ids=p['token_ids']) for k,p in raw.items()],summary={'TEST_FIXTURE_ONLY':True})};proof={'scope':'TEST_FIXTURE_ONLY'};source={'commit':'TEST_FIXTURE_ONLY','files':{}}
   specs={a:{'matrix_sha256':a} for a in ('Adapter_A','Adapter_B','Adapter_C')}
   with patch.object(api,'history',return_value=(old,result,proof)),patch.object(api,'inputs_3a',return_value=(tok,USERS,prov,data,pm,m)),patch.object(api,'validate',return_value=(m,pm,specs,USERS)),patch.object(api,'source_identity',return_value=source),patch.object(api,'archived_source'),patch.object(api,'prepare_prompt',side_effect=lambda s,*a,**kw:copy.deepcopy(raw[s['id']])):
    info=api.prepare(root/'3a_plan',root/'3a_audit',root/'3a_verified','fixture',root/'plan')
    self.assertEqual(info['independent_groups'],1);self.assertEqual(info['requests_per_condition'],[4]);self.assertEqual(info['a0_requests_per_condition'],[3])
    protocol,plans=api.validate_plan(root/'plan');abc=plans['abc']['blocks'][0];a0=plans['a0']['blocks'][0]
    self.assertEqual([t['sample_id'] for t in a0['tasks']],[t['sample_id'] for t in abc['tasks'][1:]])
    self.assertTrue(all(t['scored'] for t in a0['tasks']));self.assertEqual(a0['priming_requests'],0)
    # Tampering is rejected without modifying either the old inputs or their seals.
    child=root/'plan/a0/i5_workload_manifest.json';x=read(child);x['blocks'][0]['tasks'][0]['token_ids'][0]=99;write(child,x)
    with self.assertRaises(ValueError):api.validate_plan(root/'plan')
 def test_frozen_thresholds_and_quality_interface(self):
  from semcache.edgelora.r5_i5_quality import score
  self.assertEqual(THRESHOLDS['max_abs'],.01);self.assertEqual(THRESHOLDS['relative_l2'],.001);self.assertTrue(score(self.prompts[0]['sample'],'gold','I5_PROMPTS_V1'))

class Reporting(unittest.TestCase):
 def row(self,rid,scored,start,role='SCORED_TARGET'):
  return dict(request_id=rid,scored=scored,role=role,arrival_ns=start,completion_ns=start+1000000,application_e2e_ms=1.,application_ttft_ms=.5,cache_admission_eviction_ms=.2)
 def test_cost_denominators_a0_separate_and_source_included(self):
  rows=[self.row('s',False,1000000,'CONTROLLED_STANDALONE_SOURCE'),self.row('t',True,3000000)]
  life=dict(execution_start_ns=0,execution_end_ns=6000000,process_wall_ms=6)
  abc=scope_cost(rows,1,1,life);a0=scope_cost(rows[1:],1,1,life)
  self.assertEqual(abc['priming_sources'],1);self.assertEqual(a0['priming_sources'],0);self.assertEqual(abc['amortized_sequence_ms_per_target'],3.);self.assertEqual(a0['amortized_sequence_ms_per_target'],1.)
 def test_cache_classification_uses_grant_and_residency(self):
  row=dict(condition='SEMCACHE_INTEGRATED',adapter_id='Adapter_A',request_id='s',role='CONTROLLED_STANDALONE_SOURCE',source=None,decision='CACHE_MISS',reused_prefix_tokens=0,cache_before={'entry_count':0,'lru_to_mru':[]},cache_after={},eviction_count=0,candidate_reasons=[],admission=None)
  self.assertEqual(classify(row,[])['classification'],'COLD_SOURCE');row['role']='NEGATIVE_TARGET';row['cache_before']['entry_count']=1;self.assertEqual(classify(row,[])['classification'],'WARM_MISS');self.assertEqual(classify(row,[])['evictions'],0)
 def test_false_hit_no_physical_receipt_rejected(self):
  row=dict(condition='SEMCACHE_INTEGRATED',adapter_id='Adapter_A',request_id='t',source={'entry_id':'e'},decision='PHYSICAL_PREFIX_HIT',reused_prefix_tokens=512,cached_prefix_q_consumed=False,cache_before={'lru_to_mru':['e']})
  with self.assertRaisesRegex(ValueError,'receipt'):classify(row,[])
 def test_3a_provenance_tampering_rejected(self):
  with patch('semcache.edgelora.r5_i5r3b_plan.compute_3a',return_value=({'source':{'commit':'a'*40}},{})):
   with self.assertRaisesRegex(ValueError,'pinned source'):history('plan','audit','verify','b'*40)
 def test_corrupted_3a_inventory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'audit';a.mkdir();v=root/'verified';v.mkdir();write(a/'i5r3a_results.json',{'TEST_FIXTURE_ONLY':True});seal(a);write(v/'fixture.json',{});seal(v)
   write(a/'i5r3a_results.json',{'altered':True})
   with patch('semcache.edgelora.r5_i5r3b_plan.compute_3a',return_value=({'source':{'commit':'a'*40}},{})):
    with self.assertRaisesRegex(ValueError,'corrupt sealed evidence'):history('plan',a,v,'a'*40)
 def test_independent_audit_tamper_rejection(self):
  result=dict(source={'fixture':True},receipt={'sha':'x'},verdicts={'NUMERICAL_FIDELITY':'FAIL'})
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'audit';a.mkdir();write(a/'i5r3b_results.json',result);write(a/'i5r3b_decision.json',result['verdicts']);write(a/'i5r3b_independent_audit.json',dict(status='PASS',source=result['source'],run_inventory=result['receipt']));seal(a)
   with patch('semcache.edgelora.r5_i5r3b_analysis.compute',return_value=result):
    self.assertEqual(verify(root/'ex',root/'run',a,root/'review')['status'],'PASS')
    write(a/'i5r3b_decision.json',{'NUMERICAL_FIDELITY':'PASS'});seal(a)
    with self.assertRaises(ValueError):verify(root/'ex',root/'run',a,root/'bad')
 def test_runner_delegates_only_to_existing_runtime_without_fallback(self):
  spec=importlib.util.spec_from_file_location('i3b_run',Path(__file__).resolve().parents[1]/'scripts/144_run_r5_i5r3b.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cmd=module.command('ex','multi',0,'commit',4800,'fresh')
  self.assertTrue(cmd[1].endswith('119_run_r5_i5.py'));self.assertIn('--enable-gpu-experiment',cmd);self.assertNotIn('--test-fixture-only',cmd);self.assertNotIn('--enable-arithmetic-diagnostic',cmd)
 def test_prepare_cli_argument_dispatch_uses_actual_signature(self):
  import contextlib,io,sys
  from unittest.mock import create_autospec
  spec=importlib.util.spec_from_file_location('i3b_prepare',Path(__file__).resolve().parents[1]/'scripts/143_prepare_r5_i5r3b.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  argv=['143_prepare_r5_i5r3b.py','prepare','--plan-3a','p','--audit-3a','a','--verification-3a','v','--expected-3a-commit','commit','--output','fresh']
  fn=create_autospec(module.api.prepare,return_value={'status':'TEST_FIXTURE_ONLY'})
  with patch.object(sys,'argv',argv),patch.object(module.api,'prepare',fn),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(module.main(),0)
  fn.assert_called_once_with(plan_3a='p',audit_3a='a',verification_3a='v',expected_3a_commit='commit',output='fresh',profile='B2_SMOKE',adapter='Adapter_A',historical_r1_audit=None)

if __name__=='__main__':unittest.main()
