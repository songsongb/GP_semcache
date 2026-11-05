"""B0 synthetic CPU evidence tests; never real GPU recapture results."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from semcache.edgelora.communication import digest_bytes,digest
from semcache.edgelora.r3a_artifacts import pack,write
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5r2c3 import history,replay_operands
from semcache.edgelora.r5_i5r2c3b0 import compare_capture,verified_companion,ORIGIN,fresh
from test_r5_i5r2c3 import fixture


def capture(root):
 p,r,a,c=fixture(root/'historical');h=history(p,r,a,c,True);h['historical_evidence']=Evidence(r);new=root/'new';new.mkdir()
 records={}
 def put(name,data,shape):
  path=new/(name+'.bin');path.write_bytes(data)
  return dict(file=path.name,sha256=digest_bytes(data),shape=shape,dtype='float16',semantic_positions=[0,64 if name.endswith('reference') else shape[2] if name.endswith('mask') else shape[1]])
 for label,n in (('S',64),('F',256)):
  ctx=h['contexts'][label];ctx.update(target_token_ids=list(range(n)),cache_positions=list(range(n)),input_length=n,mode='COLD_MISS' if label=='S' else 'NO_REUSE',phase='FULL_PREFILL')
  tensors={}
  for role in 'qkv':
   old=h['coverage'][label]['raw_'+role];base=('SEMCACHE_INTEGRATED' if label=='S' else 'FULL_RECOMPUTE')+'/Adapter_A/es'
   prefix=h['historical_evidence'].tensor(base,old)
   tensors[role]=put(label+'_'+role,prefix+pack([.5]*((n-64)*4),'float16'),[1,n,4])
  tensors['mask']=put(label+'_mask',pack([0. if k<=q else -65504 for q in range(n) for k in range(n)],'float16'),[1,1,n,n])
  tensors['reference']=put(label+'_reference',h['inputs'][label]['reference'][0],[1,64,4])
  records[label]=dict(context=copy.deepcopy(ctx),identity=copy.deepcopy(h['identities'][label]),position_observation=dict(positions=list(range(n)),past_length=0),backend_before=h['backend'],backend_after=h['backend'],tensors=tensors)
 seal(new);return h,new,records


class OperandCapture(unittest.TestCase):
 def test_full_operands_compare_to_both_original_references(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d));report,inputs=compare_capture(h,Evidence(new),records,fixture=True)
   self.assertTrue(report['NEW_CAPTURE_COMPLETE']);self.assertTrue(report['HISTORICAL_C2_PREFIX_REPRODUCED']);self.assertTrue(report['HISTORICAL_C2_ATTENTION_REPRODUCED'])
   self.assertEqual(inputs['F']['q'][1]['shape'],[1,256,4]);self.assertFalse(report['COMPANION_REPLAY_ELIGIBLE']);self.assertEqual(report['FIRST_ATTENTION_DIVERGENCE'],'NOT_EVALUATED')
 def test_missing_full256_qkv_or_mask_is_rejected(self):
  for role in ('q','k','v','mask'):
   with self.subTest(role=role),tempfile.TemporaryDirectory() as d:
    h,new,records=capture(Path(d));del records['F']['tensors'][role]
    with self.assertRaisesRegex(ValueError,'missing/unexpected'):compare_capture(h,Evidence(new),records,fixture=True)
 def test_bounded_full_mask_or_qkv_cannot_impersonate_complete_capture(self):
  for role in ('q','mask'):
   with self.subTest(role=role),tempfile.TemporaryDirectory() as d:
    h,new,records=capture(Path(d));records['F']['tensors'][role]['shape'][1 if role=='q' else 2]=64
    with self.assertRaisesRegex(ValueError,'shape/dtype/position'):compare_capture(h,Evidence(new),records,fixture=True)
 def test_new_source_full_equality_is_insufficient_for_historical_prefix_reproduction(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d))
   for label in ('S','F'):
    item=records[label]['tensors']['q'];path=new/item['file'];data=pack([.75],'float16')+path.read_bytes()[2:];path.write_bytes(data);item['sha256']=digest_bytes(data)
   report,_=compare_capture(h,Evidence(new),records,fixture=True)
   self.assertTrue(all(x['metrics']['bitwise_equal'] for x in report['new_source_vs_full']));self.assertFalse(report['HISTORICAL_C2_PREFIX_REPRODUCED']);self.assertFalse(report['COMPANION_REPLAY_ELIGIBLE'])
 def test_equal_new_module_outputs_still_require_each_historical_reference(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d))
   for label in ('S','F'):
    item=records[label]['tensors']['reference'];data=pack([.75]*256,'float16');(new/item['file']).write_bytes(data);item['sha256']=digest_bytes(data)
   report,_=compare_capture(h,Evidence(new),records,fixture=True);self.assertFalse(report['HISTORICAL_C2_ATTENTION_REPRODUCED']);self.assertFalse(report['COMPANION_REPLAY_ELIGIBLE'])
 def test_source_full_prefix_difference_rejected_by_gate(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d));item=records['S']['tensors']['v'];path=new/item['file'];data=pack([.25],'float16')+path.read_bytes()[2:];path.write_bytes(data);item['sha256']=digest_bytes(data)
   report,_=compare_capture(h,Evidence(new),records,fixture=True);self.assertFalse(report['HISTORICAL_C2_PREFIX_REPRODUCED'])
 def test_bad_actual_causal_mask_fails(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d));item=records['F']['tensors']['mask'];path=new/item['file'];data=path.read_bytes()[:2]+pack([0.],'float16')+path.read_bytes()[4:];path.write_bytes(data);item['sha256']=digest_bytes(data)
   with self.assertRaisesRegex(ValueError,'causal mask'):compare_capture(h,Evidence(new),records,fixture=True)
 def test_corrupted_or_missing_tensor_cannot_pass_manifest_or_reference(self):
  for missing in (True,False):
   with self.subTest(missing=missing),tempfile.TemporaryDirectory() as d:
    h,new,records=capture(Path(d));path=new/records['F']['tensors']['q']['file']
    if missing:path.unlink()
    else:path.write_bytes(b'bad')
    with self.assertRaises(ValueError):Evidence(new).verify()
    with self.assertRaises(ValueError):compare_capture(h,Evidence(new),records,fixture=True)
 def test_model_adapter_backend_position_incompatibilities_fail(self):
  for change in ('base','adapter','backend','position'):
   with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
    h,new,records=capture(Path(d));r=records['S']
    if change=='base':r['identity']['model_identity']['base_weight_sha256']='changed'
    elif change=='adapter':r['identity']['adapter']['matrix_sha256']='changed'
    elif change=='backend':r['backend_before']={'different':True};r['backend_after']=r['backend_before']
    else:r['position_observation']['positions'][0]=1
    with self.assertRaises(ValueError):compare_capture(h,Evidence(new),records,fixture=True)
 def test_original_c3_stays_incomplete_even_when_new_capture_exists(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d));before=copy.deepcopy(h['missing'])
   with self.assertRaisesRegex(ValueError,'INCOMPLETE_EVIDENCE'):replay_operands(h)
   compare_capture(h,Evidence(new),records,fixture=True);self.assertEqual(h['missing'],before)
 def test_companion_requires_all_explicit_paths(self):
  with self.assertRaisesRegex(ValueError,'all three'):replay_operands({},dict(experiment='p',run='r'))
 def test_companion_route_is_explicit_and_original_references_remain_separate(self):
  with tempfile.TemporaryDirectory() as d:
   h,new,records=capture(Path(d));_,inputs=compare_capture(h,Evidence(new),records,fixture=True)
   expected={label:{**inputs[label],'reference':h['inputs'][label]['reference']} for label in ('S','F')}
   with patch('semcache.edgelora.r5_i5r2c3b0.verified_companion',return_value=(expected,{'provenance':ORIGIN})) as verified:
    actual,proof=replay_operands(h,dict(experiment='new-plan',run='new-run',audit='new-audit'))
   self.assertEqual(verified.call_count,1);self.assertEqual(proof['provenance'],ORIGIN);self.assertIs(actual['F']['reference'],h['inputs']['F']['reference']);self.assertNotEqual(actual['F']['q'][1]['file'],h['coverage']['F']['raw_q']['file'])
 def test_verified_companion_uses_new_operands_and_original_c2_outputs(self):
  from semcache.edgelora.r5_i5r2c3b0 import anchor
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);h,new,records=capture(root);report,inputs=compare_capture(h,Evidence(new),records,fixture=True)
   # Mock a previously GPU-verified inspect_run to exercise export binding only.
   # This test does not claim that the synthetic tensors are GPU evidence.
   report['COMPANION_REPLAY_ELIGIBLE']=True;source=dict(commit=h['config']['source_commit'],files=h['config']['sources'])
   p=dict(history={'anchor':anchor(h)},source=source,experiment_id='mock-GPU-verified-id');plan=root/'companion_plan';plan.mkdir();write(plan/'fixture.json',{'TEST_FIXTURE_ONLY':True});seal(plan)
   audit=root/'companion_audit';audit.mkdir();receipt=Evidence(new).verify()
   write(audit/'i5r2c3b0_independent_audit.json',dict(status='PASS',inventory=receipt,experiment_id=p['experiment_id'],source=source,historical_anchor=anchor(h)))
   write(audit/'i5r2c3b0_decision.json',report);seal(audit)
   with patch('semcache.edgelora.r5_i5r2c3b0.inspect_run',return_value=(p,h,report,inputs,receipt)):
    actual,proof=verified_companion(plan,new,audit,h)
    self.assertEqual(actual['F']['q'][1]['capture_origin'],ORIGIN);self.assertIs(actual['F']['reference'],h['inputs']['F']['reference']);self.assertEqual(proof['historical_reference'],'ORIGINAL_C2_MODULE_OUTPUT')
    exported=json.loads((audit/'i5r2c3b0_decision.json').read_text());exported['HISTORICAL_C2_PREFIX_REPRODUCED']=False;write(audit/'i5r2c3b0_decision.json',exported);seal(audit)
    with self.assertRaisesRegex(ValueError,'exported independent audit differs'):verified_companion(plan,new,audit,h)
 def test_mismatch_cannot_be_authorized_by_companion_export(self):
  with patch('semcache.edgelora.r5_i5r2c3b0.inspect_run',return_value=({},None,{'COMPANION_REPLAY_ELIGIBLE':False},None,None)):
   with self.assertRaisesRegex(ValueError,'does not reproduce'):verified_companion('p','r','a',{})
 def test_sealed_plan_preserves_pair_and_bounds_with_no_decode_budget(self):
  from semcache.edgelora.r5_i5r2c3b0 import prepare,load
  from semcache.edgelora.r4c_workload import policy
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);h,new,records=capture(root)
   for task,n in zip(h['config']['block']['tasks'],(1,2)):
    task['policy']=policy(dict(bos_token_id=0,eos_token_id=2,pad_token_id=1),'quality',n)
   h['config']['i5r2c2']['mode']='SOURCE_ONLY';model=h['plan']['i5r1']['model_identity'];model['versions']={'TEST_FIXTURE_ONLY':True};source=dict(commit=h['config']['source_commit'],files=h['config']['sources'])
   pm={'identity':model};spec=h['identities']['S']['adapter'];m=dict(base_prepared='offline-base',tokenizer={'fixture':'tokenizer'})
   with patch('semcache.edgelora.r5_i5r2c3b0.history',return_value=h),patch('semcache.edgelora.r5_i5r2c3b0.prepared',return_value=(m,pm,{'Adapter_A':spec},{})),patch('semcache.edgelora.r5_i5r2c3b0.source_identity',return_value=source),patch('semcache.edgelora.r5_i5r2c3b0.versions',return_value=model['versions']):
    p=prepare(root/'historical/plan',root/'historical/run',root/'historical/audit','offline-adapters',root/'plan',h['config']['source_commit'])
    loaded,_,_=load(root/'plan',current=True)
   self.assertEqual(p,loaded);self.assertEqual([len(t['token_ids']) for t in p['tasks']],[64,256]);self.assertEqual([t['policy']['max_new_tokens'] for t in p['tasks']],[1,1]);self.assertEqual(p['deadline_seconds'],900)
 def test_prefill_observer_installs_hooks_only_after_bare_projection_guard(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import capture_prefill
  from contextlib import contextmanager
  state=[]
  @contextmanager
  def hooks(driver,save):
   state.append('hooks');yield;state.pop()
  def forward(*args,**kwargs):
   self.assertEqual(state,[])  # Same ordering required by physical_path's guard.
   self.assertIsNone(args[4]);self.assertEqual(args[2],args[3]);self.assertEqual(args[1],0)
   with kwargs['observer'](object()):self.assertEqual(state,['hooks'])
   return 'actual-forward-result'
  driver=SimpleNamespace(c_forward=forward)
  with patch('semcache.edgelora.r5_i5r2c3b0_runtime.layer0_hooks',side_effect=hooks):
   out=capture_prefill(driver,dict(request_id='S',token_ids=list(range(64)),policy={'fixture':True}),'COLD_MISS','cohort',lambda *a:None)
  self.assertEqual(out,'actual-forward-result');self.assertEqual(state,[])
 def test_optional_hooks_do_not_replace_or_expose_padding_rows(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import layer0_hooks
  class Module:
   def __init__(self):self.hooks=[]
   def register_forward_hook(self,fn):self.hooks.append(fn);return SimpleNamespace(remove=lambda:self.hooks.remove(fn))
   def register_forward_pre_hook(self,fn,**kw):self.hooks.append(fn);return SimpleNamespace(remove=lambda:self.hooks.remove(fn))
  attn=Module();mods={r:Module() for r in 'qkv'};driver=SimpleNamespace(adapter=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)],projection_modules=lambda l:mods))
  seen=[]
  with layer0_hooks(driver,lambda role,t,**kw:seen.append((role,t,kw))):
   for role,m in mods.items():self.assertIsNone(m.hooks[0](m,(),role+'-real'))
  self.assertEqual([x[0] for x in seen],list('qkv'));self.assertFalse(attn.hooks);self.assertTrue(all(not m.hooks for m in mods.values()))
 def test_independent_audit_rejects_a_different_current_source(self):
  from semcache.edgelora.r5_i5r2c3b0 import analyze
  with patch('semcache.edgelora.r5_i5r2c3b0.inspect_run',return_value=({'source':{'commit':'reviewed'}},None,{},None,None)),patch('semcache.edgelora.r5_i5r2c3b0.source_identity',return_value={'commit':'different'}):
   with self.assertRaisesRegex(ValueError,'reviewed capture source'):analyze('p','r','unused-output')
 def test_output_cannot_overlap_history_or_reuse_existing_directory(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'history').mkdir()
   with self.assertRaises(ValueError):fresh(root/'history/changed',(root/'history',))
   with self.assertRaises(ValueError):fresh(root/'history')
 @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch','numpy','transformers')),'native OPT CPU operand hooks NOT_RUN; dependencies unavailable')
 def test_actual_cpu_opt_hooks_capture_full_qkv_mask_without_replacement(self):
  import torch
  from transformers import OPTConfig
  from transformers.models.opt.modeling_opt import OPTAttention
  from semcache.edgelora.r5_i5r2c3b0_runtime import layer0_hooks
  config=OPTConfig(hidden_size=8,num_attention_heads=2,attention_dropout=0.);config._attn_implementation='eager'
  a=OPTAttention(config=config,layer_idx=0).eval();layer=SimpleNamespace(self_attn=a)
  adapter=SimpleNamespace(layers=[layer],projection_modules=lambda n:{r:getattr(a,r+'_proj') for r in 'qkv'})
  for n in (64,256):
   x=torch.zeros(1,n,8);mask=torch.zeros(1,1,n,n);seen={}
   def save(role,t,**kw):seen[role]=t[:,:64].clone() if kw.get('prefix') else t.clone()
   with torch.inference_mode():
    ref=a(hidden_states=x,attention_mask=mask,output_attentions=False)[0]
    with layer0_hooks(SimpleNamespace(adapter=adapter),save):out=a(hidden_states=x,attention_mask=mask,output_attentions=False)[0]
   self.assertTrue(torch.equal(ref,out));self.assertEqual(list(seen['q'].shape),[1,n,8]);self.assertEqual(list(seen['mask'].shape),[1,1,n,n]);self.assertEqual(list(seen['reference'].shape),[1,64,8])

class ContractFailures(unittest.TestCase):
 def writer(self,root):
  writer=Mock();writer.write.side_effect=lambda path,value:write(path,value)
  return writer
 def contract(self):
  return dict(config=dict(model_type='opt',hidden_size=2560),methods={'attention':dict(source='def forward():\n    native_attention()\n',source_sha256='a'*64)},r5_multi=dict(adapter_manifest_sha256='b'*64,rank=8,matrix_dtype='float32'))
 def test_matching_contract_is_accepted_with_receipt(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);writer=self.writer(root);expected=self.contract();actual=copy.deepcopy(expected)
   verify_contract(expected,actual,writer,root)
   report=json.loads((root/'i5r2c3b0_contract_difference.json').read_text())
   self.assertTrue(report['exact_equal']);self.assertEqual(report['CONTRACT_VERIFIED'],'EXACT');self.assertEqual(report['differences'],[]);self.assertEqual(actual,expected)
 def label_contracts(self):
  expected=self.contract();expected['config']['id2label']={'0':'LABEL_0','1':'LABEL_1'}
  actual=copy.deepcopy(expected);actual['config']['id2label']={0:'LABEL_0',1:'LABEL_1'}
  return expected,actual
 def test_confirmed_id2label_representation_verified_without_mutation(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract,contract_difference
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);expected,actual=self.label_contracts();old_expected=copy.deepcopy(expected);old_actual=copy.deepcopy(actual);raw=contract_difference(expected,actual)
   verify_contract(expected,actual,self.writer(root),root)
   report=json.loads((root/'i5r2c3b0_contract_difference.json').read_text())
   self.assertEqual(report['CONTRACT_VERIFIED'],'VERIFIED_JSON_KEY_REPRESENTATION');self.assertFalse(report['exact_equal']);self.assertTrue(report['compatibility_established']);self.assertTrue(report['physical_opt_contract_consistent'])
   self.assertEqual(report['differences'],json.loads(json.dumps(raw['differences'])));self.assertEqual(expected,old_expected);self.assertEqual(actual,old_actual)
 def test_allowlist_rejects_label_model_attention_adapter_or_collision_changes(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract
  for case in ('label','model','attention','adapter','collision','other_key_conversion','wrong_direction','boolean_key'):
   with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
    expected,actual=self.label_contracts()
    if case=='label':actual['config']['id2label'][0]='DIFFERENT'
    elif case=='model':actual['config']['hidden_size']=1024
    elif case=='attention':actual['methods']['attention']['source_sha256']='c'*64
    elif case=='adapter':actual['r5_multi']['adapter_manifest_sha256']='c'*64
    elif case=='collision':actual['config']['id2label']['0']='LABEL_0'
    elif case=='other_key_conversion':expected['config']['other']={'0':'x'};actual['config']['other']={0:'x'}
    elif case=='wrong_direction':expected,actual=actual,expected
    else:actual['config']['id2label']={False:'LABEL_0',1:'LABEL_1'}
    with self.assertRaisesRegex(ValueError,'native OPT/adapter contract differs'):verify_contract(expected,actual,self.writer(Path(d)),Path(d))
    self.assertEqual(json.loads((Path(d)/'i5r2c3b0_contract_difference.json').read_text())['CONTRACT_VERIFIED'],'REJECTED')
 def test_truncated_or_incomplete_comparison_cannot_authorize_compatibility(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract,contract_difference
  for case in ('truncated','hash_errors','rows','json_unavailable'):
   with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
    expected,actual=self.label_contracts();report=contract_difference(expected,actual)
    if case=='truncated':report['truncated']=True
    elif case=='hash_errors':report['hash_errors']=['incomplete']
    elif case=='rows':report['differences']=report['differences'][:2]
    else:report['json_round_trip']['actual']['available']=False
    with patch('semcache.edgelora.r5_i5r2c3b0_runtime.contract_difference',return_value=report):
     with self.assertRaisesRegex(ValueError,'native OPT/adapter contract differs'):verify_contract(expected,actual,self.writer(Path(d)),Path(d))
 def test_physical_opt_contract_fingerprint_matches_historical_namespace(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract
  from semcache.edgelora.r5_contracts import namespace
  expected,actual=self.label_contracts()
  identity=dict(model='facebook/opt-2.7b',base_weight_sha256='base',adapter='Adapter_A',adapter_matrix_sha256='adapter',hidden_dtype='float16',layers=32,hidden_size=2560,rank=8)
  old=namespace({'identity':identity},{},'cuda:0','historical-gpu',expected);new=namespace({'identity':identity},{},'cuda:0','current-gpu',actual)
  self.assertEqual(old['opt_contract'],new['opt_contract']);self.assertNotEqual(old['gpu_uuid'],new['gpu_uuid'])
  with tempfile.TemporaryDirectory() as d:
   verify_contract(expected,actual,self.writer(Path(d)),Path(d),expected_opt_contract=old['opt_contract'],actual_opt_contract=new['opt_contract'])
   with self.assertRaisesRegex(ValueError,'native OPT/adapter contract differs'):verify_contract(expected,actual,self.writer(Path(d)),Path(d),expected_opt_contract='changed',actual_opt_contract=new['opt_contract'])
 def test_meaningful_attention_source_hash_or_adapter_mismatch_rejected(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract
  for group,key in (('methods','attention'),('r5_multi','adapter_manifest_sha256'),('config','model_type')):
   with self.subTest(group=group),tempfile.TemporaryDirectory() as d:
    root=Path(d);expected=self.contract();actual=copy.deepcopy(expected)
    if group=='methods':actual[group][key]['source_sha256']='c'*64
    else:actual[group][key]='different'
    with self.assertRaisesRegex(ValueError,'native OPT/adapter contract differs'):verify_contract(expected,actual,self.writer(root),root)
    report=json.loads((root/'i5r2c3b0_contract_difference.json').read_text());self.assertFalse(report['exact_equal']);self.assertEqual(len(report['differences']),1)
 def test_diff_exact_paths_missing_extra_types_hashes_and_no_source_dump(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference,diagnostic_hash
  source='PRIVATE IMPLEMENTATION\n'*10000
  a={'methods':{'attention':{'source':source}},'remove':None,'rank':8,'a/b~':1}
  b={'methods':{'attention':{'source':source+'changed'}},'extra':[1,2],'rank':8.0,'a/b~':2}
  report=contract_difference(a,b);rows={x['path']:x for x in report['differences']}
  self.assertEqual(set(rows),{'/methods/attention/source','/remove','/extra','/rank','/a~1b~0'})
  self.assertEqual(rows['/remove']['kind'],'missing_in_actual');self.assertEqual(rows['/extra']['kind'],'extra_in_actual');self.assertEqual(rows['/rank']['kind'],'type')
  item=rows['/methods/attention/source'];self.assertEqual(item['expected']['sha256'],diagnostic_hash(source));self.assertEqual(item['actual']['type'],'str');self.assertNotIn('value',item['actual'])
  self.assertNotIn('PRIVATE IMPLEMENTATION',json.dumps(report));self.assertLess(len(json.dumps(report)),65536)
 def test_diff_bound_and_numeric_boolean_type_difference(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference
  report=contract_difference({str(i):0 for i in range(1000)},{str(i):1 for i in range(1000)})
  self.assertTrue(report['truncated']);self.assertEqual(len(report['differences']),64)
  report=contract_difference({'nested':{'enabled':True}},{'nested':{'enabled':1}})
  self.assertFalse(report['exact_equal']);self.assertEqual(report['differences'][0]['path'],'/nested/enabled')
  class RuntimeVersion(str):pass
  report=contract_difference({'version':'2.0'},{'version':RuntimeVersion('2.0')});self.assertTrue(report['exact_equal']);self.assertEqual(report['differences'],[])
 def test_integer_string_label_keys_reproduce_crash_and_json_conversion(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference
  direct=contract_difference({'0':'LABEL_0'},{0:'LABEL_0'});self.assertTrue(direct['json_key_conversion_only']);self.assertFalse(direct['exact_equal'])
  a={'config':{'id2label':{'0':'LABEL_0','1':'LABEL_1'}}};b={'config':{'id2label':{0:'LABEL_0',1:'LABEL_1'}}}
  report=contract_difference(a,b)
  self.assertFalse(report['exact_equal']);self.assertFalse(report['raw_python_structural_equal']);self.assertTrue(report['serialized_json_representation_equal']);self.assertTrue(report['json_key_conversion_only'])
  self.assertEqual({r['path'] for r in report['differences']},{'/config/id2label/0','/config/id2label/1'})
  self.assertEqual({r['expected_key']['type'] for r in report['differences']},{'MISSING','str'});self.assertEqual({r['actual_key']['type'] for r in report['differences']},{'MISSING','int'})
 def test_nested_mixed_keys_lists_missing_extra_and_escaping(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference
  a={'a/b~':[{0:'zero','x':{2:'two','name':'before'}}]};b={'a/b~':[{0:'zero','x':{2:'two','name':'after','extra':1}}]}
  report=contract_difference(a,b)
  self.assertFalse(report['exact_equal']);self.assertFalse(report['json_key_conversion_only']);self.assertEqual({r['path'] for r in report['differences']},{'/a~1b~0/0/x/name','/a~1b~0/0/x/extra'})
 def test_typed_hashes_order_independent_preserve_key_types(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import diagnostic_hash,contract_difference
  a={0:'zero','x':{3:'three','name':'label'}};b={'x':{'name':'label',3:'three'},0:'zero'}
  self.assertEqual(diagnostic_hash(a),diagnostic_hash(b));self.assertNotEqual(diagnostic_hash({0:'LABEL_0'}),diagnostic_hash({'0':'LABEL_0'}));self.assertTrue(contract_difference(a,b)['exact_equal'])
  with self.assertRaises(TypeError):digest(a)  # Production mixed-key behavior is untouched.
 def test_json_collision_is_not_lossless_or_compatible(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference
  report=contract_difference({'0':'last'},{0:'first','0':'last'})
  self.assertFalse(report['exact_equal']);self.assertFalse(report['json_key_conversion_only']);self.assertIsNone(report['serialized_json_representation_equal']);self.assertFalse(report['json_round_trip']['actual']['lossless_keys'])
 def test_json_equality_does_not_hide_other_type_changes(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference
  report=contract_difference({'x':[1]},{'x':(1,)})
  self.assertTrue(report['serialized_json_representation_equal']);self.assertFalse(report['json_key_conversion_only']);self.assertFalse(report['exact_equal'])
 def test_deep_traversal_is_bounded_and_hash_failure_is_explicit(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import contract_difference
  a,b=0,1
  for i in range(140):a={i:a};b={i:b}
  report=contract_difference(a,b);self.assertTrue(report['truncated']);self.assertFalse(report['exact_equal']);self.assertIsNone(report['expected_sha256']);self.assertTrue(report['hash_errors'])
 def test_key_conversion_receipt_survives_rejection_and_failed_attempt_seal(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract,finalize_attempt
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);writer=self.writer(root);primary=None
   with self.assertRaisesRegex(ValueError,'native OPT/adapter contract differs') as caught:
    try:verify_contract({'config':{'id2label':{'0':'LABEL_0'}}},{'config':{'id2label':{0:'LABEL_0'}}},writer,root)
    except BaseException as exc:primary=exc;raise
    finally:finalize_attempt(root,writer,{},primary=primary)
   self.assertIs(caught.exception,primary);report=json.loads((root/'i5r2c3b0_contract_difference.json').read_text());self.assertTrue(report['json_key_conversion_only']);self.assertFalse(report['compatibility_established']);self.assertEqual(json.loads((root/'i5r2c3b0_status.json').read_text())['status'],'FAILED');Evidence(root).verify()
 def test_primary_failure_survives_cleanup_reconcile_and_is_sealed(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import finalize_attempt
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'available.json',{'partial':True});writer=self.writer(root);writer.reconcile.side_effect=OSError('ledger failed')
   channel=Mock();channel.close.side_effect=OSError('close failed');process=Mock();process.poll.return_value=None;process.terminate.side_effect=OSError('terminate failed');finish=Mock(side_effect=OSError('finish failed'))
   primary=ValueError('native OPT/adapter contract differs');status={'status':'COMPLETE'}
   with self.assertRaises(ValueError) as caught:
    try:raise primary
    finally:finalize_attempt(root,writer,status,primary=primary,channel=channel,child=(process,),finish_child=finish)
   self.assertIs(caught.exception,primary);saved=json.loads((root/'i5r2c3b0_status.json').read_text());self.assertEqual(saved['status'],'FAILED');self.assertIn('native OPT/adapter contract differs',saved['reason']);self.assertEqual(saved['evidence_accounting'],'FAILED');self.assertEqual(len(saved['finalization_errors']),4);finish.assert_called_once();Evidence(root).verify()
 def test_real_writer_partial_attempt_is_sealed_with_primary_exception(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import finalize_attempt
  from semcache.edgelora.r5_i5r2bs1_runtime import Store
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);store=Store(root,128*2**20);store.put('partial.json',{'partial':True});primary=ValueError('primary')
   finalize_attempt(root,store.writer,{'status':'FAILED'},primary=primary)
   self.assertEqual(json.loads((root/'i5r2c3b0_status.json').read_text())['reason'],'ValueError: primary');Evidence(root).verify()
 def test_failed_writer_status_falls_back_without_clearing_accounting_failure(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import finalize_attempt
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);writer=self.writer(root);writer.write.side_effect=OSError('write unavailable');primary=ValueError('primary')
   finalize_attempt(root,writer,{},primary=primary)
   saved=json.loads((root/'i5r2c3b0_status.json').read_text());self.assertEqual(saved['status'],'FAILED');self.assertEqual(saved['reason'],'ValueError: primary');self.assertEqual(saved['evidence_accounting'],'FAILED');Evidence(root).verify()
 def test_seal_failure_never_masks_primary_and_is_not_claimed_sealed(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import finalize_attempt
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);primary=ValueError('primary')
   with patch('semcache.edgelora.r5_i5r2c3b0_runtime.seal',side_effect=OSError('cannot seal')):
    with self.assertRaises(ValueError) as caught:
     try:raise primary
     finally:finalize_attempt(root,self.writer(root),{},primary=primary)
   self.assertIs(caught.exception,primary);self.assertFalse((root/'i5_output_manifest.json').exists());self.assertEqual(json.loads((root/'i5r2c3b0_status.json').read_text())['failure_seal'],'FAILED')
 def test_finalization_error_on_success_changes_status_to_failed_and_raises(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import finalize_attempt
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);writer=self.writer(root);writer.reconcile.side_effect=OSError('ledger failed')
   with self.assertRaisesRegex(OSError,'ledger failed'):finalize_attempt(root,writer,{'status':'COMPLETE'})
   self.assertEqual(json.loads((root/'i5r2c3b0_status.json').read_text())['status'],'FAILED');Evidence(root).verify()
 def test_contract_receipt_write_failure_preserves_contract_rejection(self):
  from semcache.edgelora.r5_i5r2c3b0_runtime import verify_contract
  writer=Mock();writer.write.side_effect=OSError('disk write failed')
  with self.assertRaisesRegex(ValueError,'native OPT/adapter contract differs') as caught:verify_contract(self.contract(),{},writer,'unused')
  self.assertTrue(caught.exception.b0_secondary_failures)

if __name__=='__main__':unittest.main()
