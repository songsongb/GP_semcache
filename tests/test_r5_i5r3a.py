"""Synthetic CPU fixtures only; never evidence of GPU reuse."""
import copy,json,tempfile,unittest
import contextlib,importlib.util,io,sys
from unittest.mock import patch,create_autospec
from pathlib import Path
from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import write,sha
from semcache.edgelora.r5_i5_data import normalize,SOURCES,VERSIONS
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_multi_artifacts import registry_for
from semcache.edgelora.r5_i5r3a import audit_dataset,pair_record,provenance,fresh

class Tokenizer:
 def encode(self,text,**kwargs):return list(text.encode())
 def decode(self,ids,**kwargs):return bytes(ids).decode()

def samples(contexts=('abc','abc'),questions=('why','how')):
 return normalize('qasper',[dict(_id=str(i),context=c,input=questions[i],answers=['gold'],length=len(c),TEST_FIXTURE_ONLY=True) for i,c in enumerate(contexts)],fixture=True)

def prepared(ids,rid,user='User_01',adapter='Adapter_A'):
 return dict(sample=dict(id=rid,context='abc'),token_ids=ids,shared_context_key='key',route=dict(user_id=user,adapter_id=adapter))

class Audit(unittest.TestCase):
 def setUp(self):self.registry=registry_for(['Adapter_A','Adapter_B','Adapter_C'],'expanded')
 def run_audit(self,rows,**kw):return audit_dataset(rows,Tokenizer(),self.registry,**kw)
 def test_same_context_different_questions_not_prefix(self):
  r=self.run_audit(samples());s=r['summary'];self.assertEqual(s['shared_context_request_pairs'],1);self.assertEqual(s['strict_prefix_pairs'],0);self.assertEqual(s['result'],'NO_NATURAL_ELIGIBLE_PAIR');self.assertEqual(len(r['pairs']),2)
 def test_identical_complete_prompts_are_not_strict(self):
  r=self.run_audit(samples(questions=('why','why')));self.assertEqual(r['summary']['strict_prefix_pairs'],0);self.assertEqual(r['samples'][0]['token_ids'],r['samples'][1]['token_ids'])
 def test_genuine_complete_prompt_prefix_through_preparation(self):
  class ExactTokenizer(Tokenizer):
   def encode(self,text,**kwargs):
    if text.endswith('Question: why\nAnswer:'):return [1,2]
    if text.endswith('Question: how\nAnswer:'):return [1,2,3]
    return super().encode(text,**kwargs)
  r=audit_dataset(samples(),ExactTokenizer(),self.registry)
  self.assertEqual(r['summary']['strict_prefix_pairs'],1)
  self.assertEqual(r['summary']['same_context_strict_prefix_pairs'],1)
  self.assertEqual(r['summary']['cross_context_strict_prefix_pairs'],0)
  self.assertEqual(r['samples'][0]['token_ids'],[1,2])
 def test_order_comparison_uses_production_selection_and_common_tokens(self):
  from semcache.edgelora.r5_i5_workload import select,pack_groups
  rows=normalize('qasper',[dict(_id=str(i),context='context'+str(i),input='q'+str(i),answers=['gold'],length=8,TEST_FIXTURE_ONLY=True) for i in range(8)],fixture=True)
  class PrefixTokenizer(Tokenizer):
   def encode(self,text,**kwargs):
    if text.endswith('\nAnswer:'):return list(range(int(text.rsplit('q',1)[1].split('\n')[0])+2))
    return super().encode(text,**kwargs)
  tok=PrefixTokenizer();r=audit_dataset(rows,tok,self.registry,block_size=3)
  expected,_=select(rows,tok,count=8,seed=42,context_limit=2048,budget=32,allow_truncation=True)
  ids=[p['sample']['id'] for group in pack_groups(expected,3) for p in group]
  self.assertEqual([x['id'] for x in r['original_i5_order']],ids)
  self.assertEqual(r['summary']['same_context_strict_prefix_pairs'],0)
  self.assertEqual(r['summary']['cross_context_strict_prefix_pairs'],28)
  original={p['id']:p for p in r['original_i5_order']}
  for pair in r['pairs']:
   old=pair['original_i5_order'];a=original[pair['source_id']];b=original[pair['target_id']]
   self.assertEqual(old['SOURCE_PRECEDES_TARGET'],a['sequence']<b['sequence'])
   self.assertEqual(old['WITHIN_SAME_CACHE_LIFETIME'],a['block']==b['block'])
   self.assertEqual(pair['scope'],'STATIC_ONLY')
  self.assertTrue(any(p['SOURCE_PRECEDES_TARGET']!=p['original_i5_order']['SOURCE_PRECEDES_TARGET'] for p in r['pairs']))
  comparison=r['order_comparison'];self.assertEqual(comparison['original_i5']['strict_prefix_pairs'],28)
  self.assertEqual(comparison['original_i5']['statically_eligible_pairs'],sum(p['original_i5_order']['STATICALLY_ELIGIBLE'] for p in r['pairs']))
  self.assertEqual(comparison['gained_statically_eligible_pairs'],sum(p['STATICALLY_ELIGIBLE'] and not p['original_i5_order']['STATICALLY_ELIGIBLE'] for p in r['pairs']))
  self.assertEqual(comparison['lost_statically_eligible_pairs'],sum(not p['STATICALLY_ELIGIBLE'] and p['original_i5_order']['STATICALLY_ELIGIBLE'] for p in r['pairs']))
 def test_no_eligible_records_have_empty_order_comparison(self):
  r=self.run_audit(samples(questions=('q'*400,'q'*400)),input_limit=150)
  self.assertEqual(r['original_i5_order'],[]);self.assertEqual(r['order_comparison']['original_i5']['strict_prefix_pairs'],0)
  self.assertEqual(r['summary']['natural_actual_hit_status'],'NOT_EVALUATED')
 def test_complete_prefix_and_same_adapter_authorization(self):
  a,b=prepared([1,2],'s'),prepared([1,2,3],'t','User_02');r=pair_record(a,b,{'s':[0,0],'t':[0,1]},self.registry,491520)
  self.assertTrue(r['STATICALLY_ELIGIBLE']);self.assertEqual(r['suffix_tokens'],1);self.assertEqual(r['source_raw_qkv_bytes'],983040)
 def test_unauthorized_cross_adapter(self):
  r=pair_record(prepared([1],'s'),prepared([1,2],'t','User_11','Adapter_B'),{'s':[0,0],'t':[0,1]},self.registry,491520)
  self.assertFalse(r['STATICALLY_ELIGIBLE']);self.assertIn('CROSS_ADAPTER_REUSE_FORBIDDEN',r['rejection_reasons'])
 def test_unauthorized_same_adapter(self):
  reg=copy.deepcopy(self.registry);reg['users']['User_02']['allow_cross_user_reuse']=False
  r=pair_record(prepared([1],'s'),prepared([1,2],'t','User_02'),{'s':[0,0],'t':[0,1]},reg,491520);self.assertFalse(r['SAME_OR_AUTHORIZED_ADAPTER'])
 def test_capacity(self):
  r=pair_record(prepared([1,2],'s'),prepared([1,2,3],'t'),{'s':[0,0],'t':[0,1]},self.registry,491520,capacity=1);self.assertFalse(r['SOURCE_CAPACITY_ELIGIBLE'])
 def test_reset_boundary(self):
  r=pair_record(prepared([1],'s'),prepared([1,2],'t'),{'s':[0,0],'t':[1,1]},self.registry,491520);self.assertFalse(r['WITHIN_SAME_CACHE_LIFETIME'])
 def test_reversed_temporal_order(self):
  r=pair_record(prepared([1],'s'),prepared([1,2],'t'),{'s':[0,1],'t':[0,0]},self.registry,491520);self.assertFalse(r['SOURCE_PRECEDES_TARGET'])
 def test_tokenizer_boundary_match_is_not_text_prefix(self):
  r=pair_record(prepared([1,2],'s'),prepared([1,7,2],'t'),{'s':[0,0],'t':[0,1]},self.registry,491520);self.assertFalse(r['STRICT_TOKEN_PREFIX'])
 def test_truncation_per_sample_not_group_stable(self):
  rows=samples(('x'*600,'x'*600),('why','long question'*8));r=self.run_audit(rows,input_limit=250)
  self.assertEqual(r['summary']['truncated_count'],2);self.assertNotEqual(r['samples'][0]['retained_context_tokens'],r['samples'][1]['retained_context_tokens']);self.assertEqual(rows[0]['context'],rows[1]['context'])
 def test_fixed_fields_excluded(self):
  r=self.run_audit(samples(questions=('q'*400,'why')),input_limit=150);self.assertEqual(r['summary']['excluded_count'],1);self.assertIn('INELIGIBLE_FIXED_FIELDS',r['exclusions'][0]['reason'])
 def test_no_shared_contexts(self):
  r=self.run_audit(samples(('abc','def')));self.assertEqual(r['summary']['shared_context_groups'],0);self.assertEqual(r['summary']['strict_prefix_pairs'],0)
 def test_empty_or_duplicate_ids(self):
  with self.assertRaises(ValueError):self.run_audit([])
  rows=samples();rows[1]['id']=rows[0]['id']
  with self.assertRaises(ValueError):self.run_audit(rows)
 def test_deterministic_and_no_mutation(self):
  rows=samples();old=copy.deepcopy(rows);self.assertEqual(self.run_audit(rows),self.run_audit(rows));self.assertEqual(rows,old)
 def test_controlled_separation_and_block_bound(self):
  r=self.run_audit(samples(),block_size=1);self.assertEqual(r['summary']['maximum_lifetime_requests'],1);self.assertEqual(r['summary']['context_groups_split_by_reset'],1);self.assertEqual(r['summary']['natural_actual_hit_status'],'NOT_EVALUATED');self.assertEqual(r['pilot']['controlled'][0]['category'],'CONTROLLED_DIAGNOSTIC')
 def test_sealed_dataset_and_altered_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);rows=samples();(p/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows));m=dict(dataset='qasper',version=VERSIONS['qasper'],source=SOURCES['qasper'],split='test',revision='a'*40,source_files={'raw.jsonl':'b'*64},TEST_FIXTURE_ONLY=True,samples=2,samples_sha256=sha(p/'samples.jsonl'));write(p/'i5_dataset_manifest.json',m);seal(p)
   self.assertEqual(len(provenance(p,'qasper',True)[0]),2)
   with self.assertRaises(ValueError):provenance(p,'qasper',False)
   m['source']='wrong';write(p/'i5_dataset_manifest.json',m);seal(p)
   with self.assertRaises(ValueError):provenance(p,'qasper',True)
   (p/'samples.jsonl').write_text('{}')
   with self.assertRaises(ValueError):provenance(p,'qasper',True)
 def test_historical_protection(self):
  with tempfile.TemporaryDirectory() as d:
   old=Path(d)/'old';old.mkdir()
   with self.assertRaises(ValueError):fresh(old/'new',[old])
   with self.assertRaises(ValueError):fresh(old,[])
 def test_independent_verifier_rejects_resealed_positive_runtime_claim(self):
  import subprocess
  from unittest.mock import patch
  from semcache.edgelora.r3a_artifacts import ROOT
  from semcache.edgelora.communication import digest_bytes
  from semcache.edgelora.r5_i5r3a import verify,summary_record
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);review=root/'audit';review.mkdir()
   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r5_i5_data.py'
   source=dict(commit=commit,files={name:digest_bytes(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))})
   p=dict(source=source,plan_id='fixture',datasets={},registry_path=str(root/'registry'),multi_prepared=str(root/'multi'))
   result={'qasper':self.run_audit(samples())};summary=summary_record(p,result,source)
   write(review/'i5r3a_results.json',result);write(review/'i5r3a_dataset_provenance.json',{});write(review/'i5r3a_summary.json',summary);seal(review)
   with patch('semcache.edgelora.r5_i5r3a.compute',return_value=(p,result)),patch('semcache.edgelora.r5_i5r3a.source_identity',return_value=source):
    self.assertEqual(verify(root/'plan',review,root/'verified')['status'],'PASS')
    summary['runtime_semantics_admission_eviction_reuse_numerical_fidelity']='PASS';write(review/'i5r3a_summary.json',summary);seal(review)
    with self.assertRaisesRegex(ValueError,'summary differs'):verify(root/'plan',review,root/'bad')

class ManifestCompatibility(unittest.TestCase):
 @contextlib.contextmanager
 def adapter_fixture(self):
  # Real inventory I/O, isolated from model/tokenizer/preparation dependencies.
  # This fixture does not stand in for validation of real adapter matrices.
  from types import SimpleNamespace
  from semcache.edgelora import r5_i5r3a as api
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);multi=root/'multi';multi.mkdir();registry=root/'registry.json';write(registry,{})
   write(multi/'adapter.json',{'TEST_FIXTURE_ONLY':True})
   write(multi/'r5_output_manifest.json',{'files':{'adapter.json':sha(multi/'adapter.json')}})
   tok=SimpleNamespace(identity={'TEST_FIXTURE_ONLY':True},tokenizer=Tokenizer())
   loaded=(tok,{}, {}, {}, {'identity':{'TEST_FIXTURE_ONLY':True}}, {})
   with patch.object(api,'inputs',return_value=loaded),patch.object(api,'source_identity',return_value={'commit':'fixture','files':{}}),patch.object(api,'source_provenance'):
    yield api,root,multi,registry
 def test_valid_r5_multi_manifest_passes_prepare_and_compute(self):
  from semcache.edgelora.r5_i5r2_tensors import Evidence
  with self.adapter_fixture() as (api,root,multi,registry):
   api.prepare(registry,multi,root/'plan')
   p,result=api.compute(root/'plan')
   self.assertEqual(p['multi_inventory']['manifest'],'r5_output_manifest.json')
   self.assertEqual(result,{})
   self.assertTrue((root/'plan/i5_output_manifest.json').is_file())
   Evidence(root/'plan').verify()
 def test_missing_r5_manifest_rejected_even_with_valid_i5_manifest(self):
  with self.adapter_fixture() as (api,root,multi,registry):
   api.prepare(registry,multi,root/'plan')
   (multi/'r5_output_manifest.json').unlink();seal(multi)
   for action in (lambda:api.prepare(registry,multi,root/'other_plan'),lambda:api.compute(root/'plan')):
    with self.assertRaisesRegex(ValueError,'missing required evidence: r5_output_manifest.json'):action()
 def test_corrupted_r5_inventory_rejected_in_prepare_and_compute(self):
  with self.adapter_fixture() as (api,root,multi,registry):
   api.prepare(registry,multi,root/'plan');write(multi/'adapter.json',{'TEST_FIXTURE_ONLY':True,'altered':True})
   for action in (lambda:api.prepare(registry,multi,root/'other_plan'),lambda:api.compute(root/'plan')):
    with self.assertRaisesRegex(ValueError,'corrupt sealed evidence'):action()
 def test_i5_dataset_keeps_its_normal_i5_manifest(self):
  from semcache.edgelora.r5_i5r2_tensors import Evidence
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);rows=samples()
   (root/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows))
   write(root/'i5_dataset_manifest.json',dict(dataset='qasper',version=VERSIONS['qasper'],source=SOURCES['qasper'],split='test',revision='a'*40,source_files={'fixture.jsonl':'b'*64},TEST_FIXTURE_ONLY=True,samples=2,samples_sha256=sha(root/'samples.jsonl')))
   seal(root)
   self.assertEqual(provenance(root,'qasper',True)[0],rows)
   self.assertEqual(Evidence(root).verify()['manifest'],'i5_output_manifest.json')
   self.assertFalse((root/'r5_output_manifest.json').exists())

class CLI(unittest.TestCase):
 def setUp(self):
  spec=importlib.util.spec_from_file_location('i5r3a_cli',Path(__file__).resolve().parents[1]/'scripts/142_r5_i5r3a.py')
  self.cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.cli)
 def test_prepare_real_parser_dispatch_matches_function_signature(self):
  argv=['142_r5_i5r3a.py','prepare','--registry','registry.json','--multi-prepared','sealed_adapters','--output-dir','fresh_plan','--block-size','7','--input-limit','512','--seed','17']
  dispatch=create_autospec(self.cli.api.prepare,return_value={'status':'PREPARED'})
  with patch.object(sys,'argv',argv),patch.object(self.cli.api,'prepare',dispatch),contextlib.redirect_stdout(io.StringIO()) as output:
   self.assertEqual(self.cli.main(),0)
  dispatch.assert_called_once_with(registry='registry.json',multi='sealed_adapters',output='fresh_plan',block_size=7,input_limit=512,seed=17)
  self.assertEqual(json.loads(output.getvalue())['status'],'PREPARED')
 def test_prepare_dispatch_enters_actual_prepare_and_reports_missing_inputs(self):
  argv=['142_r5_i5r3a.py','prepare','--registry','missing_registry','--multi-prepared','missing_adapters','--output-dir','never_created']
  with patch.object(sys,'argv',argv),patch.object(self.cli.api,'inputs',side_effect=FileNotFoundError('missing real data')) as load,contextlib.redirect_stderr(io.StringIO()) as error:
   self.assertEqual(self.cli.main(),2)
  load.assert_called_once_with('missing_registry','missing_adapters')
  self.assertEqual(json.loads(error.getvalue())['status'],'REJECTED')

if __name__=='__main__':unittest.main()
