"""CPU synthetic workloads only; no claims about real GPU latency or HITs."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest,digest_bytes
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
   pm={'identity':{'TEST_FIXTURE_ONLY':True,'model':'facebook/opt-2.7b','base_revision':'a'*40,'base_weight_sha256':'b'*64,'dtype':'float16','hidden_dtype':'float16','layers':32,'hidden_size':2560}};m={'tokenizer':{'tokenizer_revision':'c'*40}};tok=SimpleNamespace(identity=m['tokenizer'],tokenizer=Tokenizer())
   old=dict(registry_path=str(root/'registry'),multi_prepared=str(multi),users=USERS,model_identity=pm['identity'],tokenizer_identity=m['tokenizer'],seed=42,input_limit=None)
   result={'qasper':dict(samples=[dict(id=k,token_ids=p['token_ids'],token_ids_sha256=digest(p['token_ids']),source_record_sha256=p['sample']['source_record_sha256'],shared_context_key=digest_bytes(p['sample']['context'].encode()),route=p['route']) for k,p in raw.items()],summary={'TEST_FIXTURE_ONLY':True})};proof={'scope':'TEST_FIXTURE_ONLY'};source={'commit':'TEST_FIXTURE_ONLY','files':{}}
   frozen_3a=copy.deepcopy(result)
   specs={a:{'matrix_sha256':digest(a),'adapter_id':a,'rank':8,'matrix_dtype':'float32'} for a in ('Adapter_A','Adapter_B','Adapter_C')}
   with patch.object(api,'history',return_value=(old,result,proof)),patch.object(api,'inputs_3a',return_value=(tok,USERS,prov,data,pm,m)),patch.object(api,'validate',return_value=(m,pm,specs,USERS)),patch.object(api,'source_identity',return_value=source),patch.object(api,'archived_source'),patch.object(api,'prepare_prompt',side_effect=lambda s,*a,**kw:copy.deepcopy(raw[s['id']])):
    info=api.prepare(root/'3a_plan',root/'3a_audit',root/'3a_verified','fixture',root/'plan')
    self.assertEqual(info['independent_groups'],1);self.assertEqual(info['requests_per_condition'],[4]);self.assertEqual(info['a0_requests_per_condition'],[3])
    self.assertEqual(result,frozen_3a)
    with patch.object(api,'transport_preflight',side_effect=ValueError('TCP_HEADER_LIMIT fixture')) as gate:
     with self.assertRaisesRegex(ValueError,'TCP_HEADER_LIMIT'):api.prepare(root/'3a_plan',root/'3a_audit',root/'3a_verified','fixture',root/'unsafe_plan')
     gate.assert_called_once()
     self.assertFalse((root/'unsafe_plan').exists())
    protocol,plans=api.validate_plan(root/'plan');abc=plans['abc']['blocks'][0];a0=plans['a0']['blocks'][0]
    self.assertEqual([t['sample_id'] for t in a0['tasks']],[t['sample_id'] for t in abc['tasks'][1:]])
    self.assertTrue(all(t['scored'] for t in a0['tasks']));self.assertEqual(a0['priming_requests'],0)
    self.assertEqual(protocol['label'],'TRANSPORT_LIMITED_DERIVED_WORKLOAD');self.assertEqual(protocol['max_prompt_tokens'],1280)
    self.assertTrue(all(t['prepared']['original_3a']['original_routing']==t['prepared']['route'] for t in abc['tasks']))
    transport=root/'plan/i5r3b_transport.json';original_transport=read(transport);bad=copy.deepcopy(original_transport);bad['maximum_header_bytes']=0;write(transport,bad);seal(root/'plan')
    with self.assertRaisesRegex(ValueError,'transport envelope'):api.validate_plan(root/'plan')
    write(transport,original_transport);seal(root/'plan')
    # Tampering is rejected without modifying either the old inputs or their seals.
    child=root/'plan/a0/i5_workload_manifest.json';x=read(child);x['blocks'][0]['tasks'][0]['token_ids'][0]=99;write(child,x)
    with self.assertRaises(ValueError):api.validate_plan(root/'plan')
 def test_frozen_thresholds_and_quality_interface(self):
  from semcache.edgelora.r5_i5_quality import score
  self.assertEqual(THRESHOLDS['max_abs'],.01);self.assertEqual(THRESHOLDS['relative_l2'],.001);self.assertTrue(score(self.prompts[0]['sample'],'gold','I5_PROMPTS_V1'))


class Transport(unittest.TestCase):
 def setUp(self):
  self.pm={'identity':{'model':'facebook/opt-2.7b','base_revision':'a'*40,'base_weight_sha256':'b'*64,'dtype':'float16','hidden_dtype':'float16','layers':32,'hidden_size':2560}}
  self.tokenizer={'tokenizer_revision':'c'*40}
  self.specs={a:dict(adapter_id=a,matrix_sha256=digest(a),rank=8,matrix_dtype='float32') for a in ('Adapter_A','Adapter_B','Adapter_C')}
 def plans(self,n):
  ps=copy.deepcopy(corpus())
  for i,p in enumerate(ps):
   p['token_ids']=([49999]*n if not p['sample']['context'].startswith('negative') else [40000]*n)
   p['token_ids'][-1]=i;p['token_ids_sha256']=digest(p['token_ids']);p['input_text']='word '*n
  chosen,_=selection(ps,USERS,'B2_SMOKE');b=build_blocks(chosen,'B2_SMOKE',SPECIAL,Tokenizer(),USERS,42)[0]
  a0=copy.deepcopy(b);a0['tasks']=a0['tasks'][1:]
  for i,t in enumerate(a0['tasks']):t['request_sequence_number']=i
  return {'abc':{'conditions':['FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED'],'blocks':[b]},'a0':{'conditions':['FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED'],'blocks':[a0]}}
 def test_old_2016_header_exceeds_frozen_limit_and_fails(self):
  from semcache.edgelora.r5_i5r3b_transport import predicted_request,preflight
  from semcache.edgelora.r5_i5_limits import wire_limits
  plans=self.plans(2016);task=plans['abc']['blocks'][0]['tasks'][1];req=predicted_request(task,self.pm,self.specs[task['adapter_id']],self.tokenizer)
  with self.assertRaisesRegex(ValueError,'TCP_HEADER_LIMIT: required=[0-9]+, limit=65536'):wire_limits(req,USERS,'0'*64,2048)
  with self.assertRaises(ValueError):preflight(plans,self.pm,self.specs,self.tokenizer,USERS)
 def test_every_1280_abc_a0_request_has_reserved_headroom(self):
  from semcache.edgelora.r5_i5r3b_transport import preflight,HEADROOM
  result=preflight(self.plans(1280),self.pm,self.specs,self.tokenizer,USERS)
  self.assertEqual(len(result['rows']),7)
  for row in result['rows']:
   self.assertLessEqual(row['maximum_header_bytes']+HEADROOM,65536)
   self.assertEqual(row['conditions'],['FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED'])
   self.assertIn('cold_step_0',row['scenario_header_bytes'])
   if row['input_tokens']>512:self.assertIn('prefix_512_step_0',row['scenario_header_bytes'])
  self.assertLess(result['maximum_header_utilization'],1.)
 def test_request_below_hard_limit_still_requires_explicit_headroom(self):
  from semcache.edgelora.r5_i5r3b_transport import preflight,predicted_request
  from semcache.edgelora.r5_i5_limits import wire_limits
  plans=self.plans(1400);task=plans['abc']['blocks'][0]['tasks'][1]
  raw=wire_limits(predicted_request(task,self.pm,self.specs[task['adapter_id']],self.tokenizer),USERS,'0'*64,2048)
  self.assertLess(raw['maximum_header_bytes'],65536)
  with self.assertRaisesRegex(ValueError,'TCP_HEADER_HEADROOM'):preflight(plans,self.pm,self.specs,self.tokenizer,USERS)
 def test_real_context_only_truncation_preserves_question_refs_ids_route(self):
  from semcache.edgelora.r5_i5_workload import prepare_prompt
  from semcache.edgelora.r5_i5r3b_plan import original_link
  class CharacterTokenizer:
   def encode(self,text,**kwargs):return list(text.encode())
   def decode(self,ids,**kwargs):return bytes(ids).decode()
  ps=copy.deepcopy(corpus()[:2]);derived=[]
  for p in ps:
   sample=p['sample'];sample['context']='x'*5000;sample['original']['context']=sample['context'];sample['source_record_sha256']=digest(sample['original'])
   original=prepare_prompt(sample,CharacterTokenizer(),2048,32,allow_truncation=True)
   new=prepare_prompt(sample,CharacterTokenizer(),2048,32,allow_truncation=True,input_limit=1280)
   record=dict(id=sample['id'],source_record_sha256=sample['source_record_sha256'],shared_context_key=digest_bytes(sample['context'].encode()),token_ids=original['token_ids'],token_ids_sha256=digest(original['token_ids']),route=routing(sample))
   frozen=copy.deepcopy(record);link=original_link(sample,record)
   self.assertEqual(record,frozen);self.assertEqual(new['sample'],sample);self.assertIn(sample['question'],new['input_text']);self.assertEqual(new['sample']['references'],['gold']);self.assertEqual(routing(new['sample']),record['route'])
   self.assertLessEqual(len(new['token_ids']),1280);self.assertEqual(len(original['token_ids']),2016);self.assertNotEqual(new['token_ids_sha256'],link['original_3a_prompt_sha256']);derived.append(new)
  self.assertEqual(derived[0]['token_ids'][:512],derived[1]['token_ids'][:512]);self.assertGreater(len(derived[0]['token_ids']),512)
 def test_b1_rejects_b2_only_option(self):
  from semcache.edgelora.r5_i5r3b_plan import prepare
  with self.assertRaisesRegex(ValueError,'B2-only'):prepare('p','a','v','c','out',profile='B1',max_prompt_tokens=1280)

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
  argv=['143_prepare_r5_i5r3b.py','prepare','--plan-3a','p','--audit-3a','a','--verification-3a','v','--expected-3a-commit','commit','--output','fresh','--max-prompt-tokens','1280']
  fn=create_autospec(module.api.prepare,return_value={'status':'TEST_FIXTURE_ONLY'})
  with patch.object(sys,'argv',argv),patch.object(module.api,'prepare',fn),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(module.main(),0)
  fn.assert_called_once_with(plan_3a='p',audit_3a='a',verification_3a='v',expected_3a_commit='commit',output='fresh',profile='B2_SMOKE',adapter='Adapter_A',historical_r1_audit=None,max_prompt_tokens=1280)

if __name__=='__main__':unittest.main()
