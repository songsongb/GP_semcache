"""CPU fixture logic only. No GPU conclusions from these virtual score records."""
import copy,importlib.util,json,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import Mock,patch
from semcache.edgelora.communication import digest,digest_bytes
from semcache.edgelora.r3a_artifacts import ROOT,pack,write
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5r2c4 import CASES,LAYOUTS,PRECISIONS,schedule,expanded_schedule,validate_records,summarize,assessment_data,history,fresh


def virtual_records(change=None):
 rows=[]
 for s in expanded_schedule():
  value=1. if change is None else change(s)
  rows.append(dict(s,prefix_equal=True,input_values_verified=True,output=dict(shape=[32,64,64],dtype='float32' if s['precision']=='FP32' else 'float16',value=value)))
 return rows


def read_virtual(t):return pack([t['value'],1.],'float32' if t['dtype']=='float32' else 'float16')


class C4(unittest.TestCase):
 def test_exact_matrix_and_balanced_bounded_schedule(self):
  self.assertEqual(CASES,{'A':(64,64),'B':(64,256),'C':(256,64),'D':(256,256)})
  self.assertEqual(len(schedule()),128);self.assertEqual(len(expanded_schedule()),144)
  self.assertEqual(sum(s['warmup'] for s in schedule()),32)
  self.assertEqual({s['repetition'] for s in expanded_schedule()},set(range(6)))
  self.assertEqual(schedule()[16]['case'],'D')
 def test_incomplete_duplicate_reordered_or_unverified_records_rejected(self):
  rows=virtual_records();validate_records(rows)
  for broken in (rows[:-1],rows+[rows[0]],list(reversed(rows))):
   with self.assertRaises(ValueError):validate_records(broken)
  rows[0]['prefix_equal']=False
  with self.assertRaises(ValueError):validate_records(rows)
 def test_no_shape_difference_and_no_mechanism_claim(self):
  result=summarize(virtual_records(),read_virtual,baseline=True,fixture=True)
  self.assertTrue(result['verdict']['SAME_SHAPE_REPEATABILITY']);self.assertFalse(result['verdict']['CONTROLLED_EVIDENCE_ELIGIBLE']);self.assertFalse(result['verdict']['ROOT_CAUSE_CONFIRMED'])
  self.assertTrue(all(x['metrics']['bitwise_equal'] for x in result['shape']))
 def test_query_only_and_key_only_dependence_are_separated(self):
  for cases,pair in ((('C','D'),'AC'),(('B','D'),'AB')):
   rows=virtual_records(lambda s:1.00390625 if s['case'] in cases and s['precision']=='FP16' else 1.)
   result=summarize(rows,read_virtual,baseline=True,fixture=True)
   selected=[r for r in result['shape'] if r['pair']==pair and r['precision']=='FP16'];self.assertTrue(any(not r['metrics']['bitwise_equal'] for r in selected))
   self.assertTrue(all(r['metrics']['bitwise_equal'] for r in result['shape'] if r['precision']=='FP32'))
 def test_layout_precision_and_repeat_instability_detected(self):
  rows=virtual_records(lambda s:1.00390625 if s['layout']=='NATIVE' and s['precision']=='FP16' else 1.)
  result=summarize(rows,read_virtual,baseline=True,fixture=True)
  self.assertTrue(any(not r['metrics']['bitwise_equal'] for r in result['layout']));self.assertTrue(any(not r['metrics']['bitwise_equal'] for r in result['precision']))
  rows[0]['output']['value']=2.;result=summarize(rows,read_virtual,baseline=True,fixture=True);self.assertFalse(result['verdict']['SAME_SHAPE_REPEATABILITY'])
 def test_failed_native_baseline_blocks_attribution_even_with_control_data(self):
  result=summarize(virtual_records(),read_virtual,baseline=False)
  self.assertFalse(result['verdict']['CONTROLLED_EVIDENCE_ELIGIBLE']);self.assertEqual(result['verdict']['shape_dependence'],'NOT_EVALUATED')
 def test_repeat_metrics_and_precision_shape_effect_are_explicit(self):
  rows=virtual_records(lambda s:1.00390625 if s['case']=='D' and s['precision']=='FP16' else 1.)
  result=summarize(rows,read_virtual,baseline=True)
  self.assertEqual(len(result['same_shape']),120)
  self.assertTrue(all(r['metrics']['bitwise_equal'] for r in result['same_shape']))
  effects=[r for r in result['precision_shape_effects'] if r['pair']=='AD']
  self.assertTrue(all(r['effect']=='CAST_SHAPE_DISCREPANCY_ELIMINATED' for r in effects))
  result=summarize(rows,read_virtual,baseline=False)
  self.assertTrue(all(r['effect']=='NOT_EVALUATED' for r in result['precision_shape_effects']))
 def test_deterministic_statistics_and_frozen_error_checks(self):
  rows=virtual_records(lambda s:1.03125 if s['case']=='D' else 1.)
  a=summarize(rows,read_virtual,baseline=True,fixture=True);b=summarize(rows,read_virtual,baseline=True,fixture=True)
  self.assertEqual(a,b);self.assertTrue(any(not x['metrics']['activation_tolerance_pass'] for x in a['shape']));self.assertTrue(any(x['metrics'].get('FP16_ULP',{}).get('max',0)>0 for x in a['shape']))
 def test_all_paths_explicit_and_corrupt_seal_rejected(self):
  with self.assertRaisesRegex(ValueError,'all nine'):history({}, {})
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);write(p/'fixture.json',{'TEST_FIXTURE_ONLY':True});seal(p);Evidence(p).verify();(p/'fixture.json').write_text('{}')
   with self.assertRaises(ValueError):Evidence(p).verify()
 def test_historical_c3_source_mismatch_fails_before_any_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   inputs={}
   for stage in ('c2','b0','c3'):
    for role in ('experiment','run','audit'):
     p=Path(d)/(stage+'_'+role);p.mkdir();write(p/'fixture.json',{'TEST_FIXTURE_ONLY':True});seal(p);inputs[stage+'_'+role]=p
   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r3b_runtime.py';sha=digest_bytes(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))
   p=dict(source={'commit':commit,'files':{name:sha}});p['experiment_id']=digest(p);write(inputs['c3_experiment']/'i5r2c3_plan.json',p);seal(inputs['c3_experiment'])
   with self.assertRaisesRegex(ValueError,'source commit mismatch'):history(inputs,{'c3':'0'*40})
 def test_fresh_outputs_never_modify_history(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'history').mkdir()
   with self.assertRaises(ValueError):fresh(p/'history/new',(p/'history',))
   with self.assertRaises(ValueError):fresh(p/'history')
 def test_bounded_final_assessment_preserves_negative_fidelity(self):
  num={'comparisons':[dict(frozen_tolerance_pass=False,generated_tokens_equal=True,generated_text_equal=True)]}
  audit=Mock();audit.json.side_effect=lambda name:num if 'numerical' in name else {'PHYSICAL_REUSE_PRESERVED':'PASS'}
  raw=Mock();raw.jsonl.return_value=[]
  result=summarize(virtual_records(),read_virtual,baseline=False)
  a=assessment_data({'evidence':{'c2_audit':audit,'c2_run':raw}},result)
  self.assertFalse(a['observed_facts']['frozen_numerical_tolerance']['all_recorded_checks_pass']);self.assertTrue(a['observed_facts']['generation']['all_recorded_tokens_equal']);self.assertEqual(a['controlled_findings']['shape_dependence']['status'],'NOT_EVALUATED');self.assertEqual(a['hypotheses']['kernel_level_mechanism']['status'],'UNRESOLVED');self.assertFalse(a['ROOT_CAUSE_CONFIRMED'])
 def test_sealed_tensor_reference_must_have_complete_declared_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'score.bin').write_bytes(pack([1.],'float16'));seal(p);ev=Evidence(p);ev.verify()
   with self.assertRaisesRegex(ValueError,'byte count'):
    ev.tensor('',dict(file='score.bin',dtype='float16',shape=[32,64,64],sha256=digest_bytes((p/'score.bin').read_bytes())))
 def test_reused_writer_enforces_capture_byte_and_file_bounds(self):
  from semcache.edgelora.r5_i5r2bs1_runtime import Store
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);byte_store=Store(root/'bytes',64*2**20+8,2)
   with self.assertRaises(ValueError):byte_store.tensor('too_big.bin',pack([1.]*5,'float16'),[5],'float16')
   file_store=Store(root/'files',65*2**20,1);file_store.tensor('first.bin',pack([1.],'float16'),[1],'float16')
   with self.assertRaisesRegex(ValueError,'file bound'):file_store.tensor('second.bin',pack([1.],'float16'),[1],'float16')
 def test_final_assessment_of_failed_controls_is_sealed_read_only(self):
  from semcache.edgelora.r5_i5r2c4 import final_assessment
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=root/'history';old.mkdir();(old/'immutable.txt').write_text('preserve')
   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r3b_runtime.py';source=dict(commit=commit,files={name:digest_bytes(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))})
   p=dict(source=source,input_seals={'fixture':'sealed'},inputs={'fixture':str(old)},expected_commits={'c2':commit,'b0':commit,'c3':commit})
   num={'comparisons':[dict(frozen_tolerance_pass=False,generated_tokens_equal=True,generated_text_equal=True)]};audit=Mock();audit.json.side_effect=lambda name:num if 'numerical' in name else {'PHYSICAL_REUSE_PRESERVED':'PASS'};raw=Mock();raw.jsonl.return_value=[];h={'evidence':{'c2_audit':audit,'c2_run':raw}}
   result=summarize(virtual_records(),read_virtual,baseline=False);review=root/'audit';review.mkdir();receipt={'fixture':'inventory'}
   write(review/'i5r2c4_independent_audit.json',dict(status='FAIL',source=source,analysis_source=source,input_seals=p['input_seals'],inventory=receipt));write(review/'i5r2c4_comparisons.json',result);seal(review)
   before=(old/'immutable.txt').read_bytes()
   with patch('semcache.edgelora.r5_i5r2c4.inspect_run',return_value=(p,h,result,receipt,None)),patch('semcache.edgelora.r5_i5r2c4.source_identity',return_value=source):
    a=final_assessment(root/'plan',root/'run',review,root/'assessment')
    self.assertEqual(a['controlled_findings']['shape_dependence']['status'],'NOT_EVALUATED');self.assertEqual((old/'immutable.txt').read_bytes(),before);Evidence(root/'assessment').verify()
    (review/'i5r2c4_comparisons.json').write_text('{}')
    with self.assertRaises(ValueError):final_assessment(root/'plan',root/'run',review,root/'bad-assessment')
 @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch','numpy')),'actual Torch CPU shape/layout/precision controls NOT_RUN: dependencies absent')
 def test_actual_cpu_matmul_controls_preserve_scaled_values(self):
  import torch
  from semcache.edgelora.r5_i5r2c4_runtime import make_pair,common
  torch.manual_seed(42);rawq=torch.randn(1,32,256,80).half()/8;rawk=torch.randn(1,32,80,256).half()
  recovered={'S':{'q':rawq[:,:,:64].clone(),'kt':rawk[:,:,:,:64].clone()},'F':{'q':rawq,'kt':rawk}}
  snapshots={label:{role:t.clone() for role,t in values.items()} for label,values in recovered.items()}
  for case,(nq,nk) in CASES.items():
   for l in LAYOUTS:
    for precision in ('FP16','FP32'):
     q,k=make_pair(recovered,case,l,precision);self.assertEqual(list(q.shape),[1,32,nq,80]);self.assertEqual(list(k.shape),[1,32,80,nk]);self.assertTrue(torch.equal(q.float(),recovered['S' if nq==64 else 'F']['q'].float()));self.assertEqual(list(common(torch.matmul(q,k)).shape),[32,64,64])
  for label in recovered:
   for role,t in recovered[label].items():self.assertTrue(torch.equal(t,snapshots[label][role]))

if __name__=='__main__':unittest.main()
