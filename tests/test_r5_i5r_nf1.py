"""Synthetic bounded CPU evidence tests; no native GPU arithmetic claims."""
import copy,hashlib,json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest_bytes,digest
from semcache.edgelora.r3a_artifacts import pack,check_parity,write,THRESHOLDS
from semcache.edgelora.r4b2_analysis import raw_head_bytes
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5_protocol import context
from semcache.edgelora.r4c_workload import policy
from semcache.edgelora.r5_i5r_nf1_tensors import StreamEvidence,region,compare,numerical,aligned_ranges,UNKNOWN
from semcache.edgelora import r5_i5r_nf1 as nf


def tensor(root,base,name,values,shape,dtype='float16',save=True):
 data=pack(values,dtype);item=dict(shape=shape,dtype=dtype,device='TEST_FIXTURE_CPU',sha256=digest_bytes(data))
 if save:
  p=root/base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);item['file']=name
 return item

def fixture(root):
 targets={nf.S:[1]*512,nf.T1:[1]*512+[2]*768,nf.T2:[1]*512+[3]*768,nf.R:[1]*512+[2]*768,nf.N:[9]*1280}
 data=dict(prefills={},rows={},layers=1,heads=2,hidden=4)
 specials=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0)
 originals={};rawvals={}
 for cond in (nf.A,nf.C):
  base='abc_run/'+cond+'/Adapter_A/es'
  for rid in nf.IDS:
   hit=cond==nf.C and rid in (nf.T1,nf.T2,nf.R);n=len(targets[rid]);prefix=512 if hit else 0;tokens=targets[rid][prefix:]
   grant=dict(entry_id='a'*64,source_request_id=nf.S,prefix_length=512,proof_sha256='b'*64) if hit else None
   ctx=context(rid,0,tokens,targets[rid],prefix,policy(specials,'fixed' if rid==nf.S else 'quality',1 if rid==nf.S else 8),'PARTIAL_PREFIX_HIT' if hit else 'COLD_MISS','c'*64,grant)
   step=dict(context=ctx,qkv={},kv=[],components={},input_kv=[]);byrole={}
   for role in 'qkv':
    full=[float(i%13)/8 for i in range(n*4)]
    if rid==nf.S and cond==nf.C:full[0]+=.03125
    values=full[prefix*4:]
    if hit:values[0]+=.0625
    rawvals[cond,rid,role]=values;byrole[role]=values
    step['qkv']['0.'+role]=tensor(root,base,rid+'_'+role+'.bin',values,[1,len(tokens),4],save=rid!=nf.N)
   kv={}
   for role in 'kv':
    full=rawvals[nf.C,nf.S,role]+byrole[role] if hit else byrole[role]
    head=raw_head_bytes(pack(full,'float16'),[1,n,4],2,2);values=[x[0] for x in struct.iter_unpack('<e',head)]
    kv[role]=tensor(root,base,rid+'_past_'+role+'.bin',values,[1,2,n,2],save=rid!=nf.N)
   step['kv']=[kv]
   if hit:step['input_kv']=[[{k:v for k,v in data['prefills'][nf.C,nf.S]['kv'][0][r].items() if k!='file'} for r in 'kv']]
   for role in ('attention','ffn','layer'):step['components']['0.'+role]=tensor(root,base,rid+'_'+role+'.bin',byrole['q'],[1,len(tokens),4],save=rid!=nf.N)
   data['prefills'][cond,rid]=step;data['rows'][cond,rid]={'strict_cached_qkv':{}}
   if hit:data['rows'][cond,rid]['strict_cached_qkv']=copy.deepcopy(data['prefills'][nf.C,nf.S]['qkv'])
 seal(root);data['evidence']=StreamEvidence(root);data['evidence'].verify();return data

class TensorTests(unittest.TestCase):
 def test_raw_position_slice_and_head_layout(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);item=tensor(root,'','raw.bin',range(20),[1,5,4]);seal(root);ev=StreamEvidence(root);ev.verify()
   raw,shape=region(ev,'',item,1,4);self.assertEqual(shape,[1,3,4]);self.assertEqual([v[0] for v in struct.iter_unpack('<e',raw)],list(range(4,16)))
   head,hshape=region(ev,'',item,1,4,head_transform=True,heads=2);self.assertEqual(hshape,[1,2,3,2]);self.assertEqual(head,raw_head_bytes(raw,shape,2,2))
 def test_attention_head_token_axis_is_not_head_axis(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);item=tensor(root,'','kv.bin',range(24),[1,2,6,2]);seal(root);ev=StreamEvidence(root);ev.verify();b,s=region(ev,'',item,2,4)
   self.assertEqual(s,[1,2,2,2]);self.assertEqual([v[0] for v in struct.iter_unpack('<e',b)],[4,5,6,7,16,17,18,19])
 def test_wrong_dtype_shape_byte_count_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=tensor(root,'','a.bin',[1,2,3,4],[1,1,4]);seal(root);ev=StreamEvidence(root);ev.verify()
   b=dict(a,dtype='float32')
   with self.assertRaises(ValueError):region(ev,'',b,0,1)
   with self.assertRaises(ValueError):compare(ev,'',a,'',b,(0,1),(0,1))
 def test_sha_mutation_missing_named_file_and_unsafe_path(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=tensor(root,'','a.bin',[1,2],[1,1,2]);seal(root);ev=StreamEvidence(root);ev.verify();(root/'a.bin').write_bytes(pack([3,4],'float16'))
   with self.assertRaisesRegex(ValueError,'SHA'):region(ev,'',a,0,1)
   with self.assertRaises(ValueError):region(ev,'',dict(a,file='missing.bin'),0,1)
   with self.assertRaises(ValueError):region(ev,'',dict(a,file='../escape'),0,1)
 def test_hash_only_is_coverage_not_fabricated_value(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=tensor(root,'','a.bin',[1,2],[1,1,2],save=False);write(root/'meta.json',{});seal(root);ev=StreamEvidence(root);ev.verify()
   r=compare(ev,'',a,'',a,(0,1),(0,1),required=False);self.assertEqual(r['metrics'],UNKNOWN);self.assertTrue(r['bitwise_equal_by_full_sha256'])
   with self.assertRaises(ValueError):compare(ev,'',a,'',a,(0,1),(0,1))
 def test_metrics_bitwise_ulp_and_frozen_perturbation(self):
  a=pack([1,2,3],'float16');b=pack([1.03125,2,3],'float16');m=numerical(a,b,'float16')
  self.assertFalse(m['bitwise_equal']);self.assertEqual(m['bitwise_differing_elements'],1);self.assertEqual(m['max_abs'],.03125);self.assertGreater(m['FP16_ULP']['max'],0);self.assertFalse(m['activation_tolerance_pass']);self.assertEqual(THRESHOLDS['max_abs'],.01)
  self.assertTrue(numerical(a,a,'float16')['activation_tolerance_pass'])
 def test_signed_zero_is_byte_difference_not_nonzero_ulp(self):
  m=numerical(pack([0.],'float16'),pack([-0.],'float16'),'float16');self.assertFalse(m['bitwise_equal']);self.assertEqual(m['bitwise_differing_elements'],1);self.assertEqual(m['FP16_ULP']['max'],0)
 def test_equal_fast_path_preserves_full_metrics_and_nonfinite_rejection(self):
  from semcache.edgelora.r5_i5r2_tensors import metrics
  a=pack([1.,2.,0.,-3.],'float16');expected=metrics(a,a,'float16');actual=numerical(a,a,'float16')
  self.assertTrue(all(actual[k]==v for k,v in expected.items()))
  with self.assertRaises(ValueError):numerical(pack([float('inf')],'float16'),pack([float('inf')],'float16'),'float16')
 def test_mmap_does_not_modify_historical_tensor(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=tensor(root,'','a.bin',[1,2],[1,1,2]);before=(root/'a.bin').read_bytes();seal(root);ev=StreamEvidence(root);ev.verify();region(ev,'',a,0,1);self.assertEqual((root/'a.bin').read_bytes(),before)
 def test_jsonl_duplicate_key_and_bounded_record_limit(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'rows.jsonl').write_text('{"x":1,"x":2}\n');seal(root);ev=StreamEvidence(root);ev.verify()
   with self.assertRaises(ValueError):list(ev.records('rows.jsonl'))
 def test_source_mismatch_fails_closed(self):
  with self.assertRaisesRegex(ValueError,'source commit mismatch'):nf.source_check(dict(commit='a'*40,files={}), 'b'*40)

class ForensicsTests(unittest.TestCase):
 def test_alignment_uses_exact_absolute_positions(self):
  with tempfile.TemporaryDirectory() as d:
   data=fixture(Path(d));s=data['prefills'][nf.C,nf.S]['context'];f=data['prefills'][nf.A,nf.T1]['context'];h=data['prefills'][nf.C,nf.T1]['context']
   self.assertEqual(aligned_ranges(s,f,'PREFIX'),((0,512),(0,512)));self.assertEqual(aligned_ranges(f,h,'SUFFIX'),((512,1280),(0,768)))
   wrong=copy.deepcopy(h);wrong['positions'][0]=0
   with self.assertRaises(ValueError):aligned_ranges(f,wrong,'SUFFIX')
   wrong=copy.deepcopy(f);wrong['target_token_ids'][0]=77
   with self.assertRaises(ValueError):aligned_ranges(s,wrong,'PREFIX')
 def test_all_families_integrity_repeat_negative_and_no_causal_claim(self):
  with tempfile.TemporaryDirectory() as d:
   data=fixture(Path(d));before=data['evidence'].verify();records=nf.compare_all(data);summary=nf.summarize(records);hyp=nf.hypotheses(summary)
   self.assertGreater(summary['PREFIX_TOTAL_QKV']['bitwise_differences'],0);self.assertGreater(summary['SUFFIX_TOTAL_QKV']['bitwise_differences'],0)
   self.assertEqual({r['candidate_request'] for r in records if r['family']=='PREFIX_TOTAL_QKV'},{nf.T1,nf.T2,nf.R})
   self.assertTrue(all(r['metrics']['bitwise_equal'] for r in records if r['integrity_check']));self.assertEqual(summary['REPEAT_FULL_PREFIX']['bitwise_differences'],0);self.assertEqual(summary['REPEAT_HIT_SUFFIX']['bitwise_differences'],0);self.assertEqual(summary['NEGATIVE_CONTROL']['hash_only'],6)
   self.assertFalse(hyp['ROOT_CAUSE_CONFIRMED']);self.assertEqual(hyp['H3']['status'],'UNRESOLVED');self.assertEqual(data['evidence'].verify(),before)
 def test_missing_strict_tensor_is_not_partial_success(self):
  with tempfile.TemporaryDirectory() as d:
   data=fixture(Path(d));data['prefills'][nf.A,nf.T1]['qkv']['0.q'].pop('file')
   with self.assertRaisesRegex(ValueError,'strict tensor'):nf.compare_all(data)
 def test_repeat_context_must_have_same_tokens(self):
  with tempfile.TemporaryDirectory() as d:
   data=fixture(Path(d));data['prefills'][nf.C,nf.R]['context']['target_token_ids'][800]=77
   with self.assertRaises(ValueError):nf.compare_all(data)
 def test_deterministic_metric_summary(self):
  with tempfile.TemporaryDirectory() as d:
   data=fixture(Path(d));a=nf.compare_all(data);b=nf.compare_all(data);self.assertEqual(a,b);self.assertEqual(nf.summarize(a),nf.summarize(b))
 def test_phase2_oracle_is_isolated_no_gpu_execution(self):
  p=nf.phase2();self.assertEqual(p['status'],'PLANNING_ONLY_NOT_EXECUTED');self.assertTrue(p['requires_explicit_GPU_approval']);self.assertIn('no controller HIT',p['cases']['O'])
 def test_sealed_small_report_does_not_change_inputs_or_hide_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=root/'evidence';old.mkdir();data=fixture(old);before=data['evidence'].verify()
   data.update(receipts={'TEST_FIXTURE_ONLY':True},old={'verdicts':{'NUMERICAL_FIDELITY':'FAIL','TOKEN_TEXT_FIDELITY':'PASS'}},provenance={'TEST_FIXTURE_ONLY':True},identities={nf.C:{'contract':{}}},dataset={'path':str(old),'manifest':{'source_path':str(root/'raw_source'),'TEST_FIXTURE_ONLY':True}},economics={'D_ms':-4.0})
   with patch.object(nf,'read_inputs',return_value=data),patch.object(nf,'logits',return_value={'TEST_FIXTURE_ONLY':True,'historical_numerical_gate':'FAIL'}),patch.object(nf,'source_identity',return_value={'TEST_FIXTURE_ONLY':True}):
    report=nf.analyze(root/'plan',old,root/'audit',root/'verify',root/'report')
    self.assertEqual(report['status'],'FORENSICS_COMPLETE');self.assertFalse(report['GPU_EXECUTED']);self.assertEqual(report['historical_export_verdicts']['NUMERICAL_FIDELITY'],'FAIL');StreamEvidence(root/'report').verify();self.assertLess(sum(p.stat().st_size for p in (root/'report').rglob('*') if p.is_file()),2*2**20)
    self.assertEqual(data['evidence'].verify(),before)
    with self.assertRaises(ValueError):nf.analyze(root/'plan',old,root/'audit',root/'verify',root/'report/new')
 def test_import_and_metrics_without_torch_or_transformers(self):
  code="""import sys,importlib.abc
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','transformers'):raise RuntimeError('GPU/model dependency forbidden')
sys.meta_path.insert(0,Reject())
from semcache.edgelora.r5_i5r_nf1 import phase2
from semcache.edgelora.r5_i5r_nf1_tensors import numerical
assert phase2()['status']=='PLANNING_ONLY_NOT_EXECUTED'
assert numerical(b'\\x00\\x3c',b'\\x00\\x3c','float16')['bitwise_equal']
"""
  r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_full_vocabulary_logit_reproduction_and_tampered_verdict(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);data=dict(rows={},logit_steps={},tasks={r:{'scored':r!=nf.S} for r in nf.IDS},old={'verdicts':{'NUMERICAL_FIDELITY':'FAIL','TOKEN_TEXT_FIDELITY':'PASS'},'numerical':{'comparisons':[]}})
   blobs={}
   for cond in (nf.A,nf.B,nf.C):
    for rid in nf.IDS:
     values=[0.]*50272;values[5]=1.
     if cond==nf.C and rid in (nf.T1,nf.T2,nf.R):values[7]=.03125
     blob=pack(values,'float32');base='abc_run/'+cond+'/Adapter_A/es';p=root/base/'logits'/ (rid+'.bin');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob);blobs[cond,rid]=values
     data['rows'][cond,rid]=dict(logits_file='logits/'+rid+'.bin',logits_sha256=digest_bytes(blob),logits_shape=[1,1,50272],generated_tokens=1,generated_token_ids=[5],generated_text='token')
     data['logit_steps'][cond,rid]=[dict(step=0,sha256=digest_bytes(blob),selected_token_id=5)]
   for ca,cb in ((nf.A,nf.B),(nf.B,nf.C),(nf.A,nf.C)):
    for rid in nf.IDS[1:]:
     step=dict(step=0,**check_parity(blobs[cb,rid],blobs[ca,rid],[1,1,50272]));data['old']['numerical']['comparisons'].append(dict(comparison=ca+'_vs_'+cb,request_id=rid,steps=[step],skipped=[],generated_tokens_equal=True,generated_text_equal=True,bitwise_logits_equal=blobs[ca,rid]==blobs[cb,rid],frozen_tolerance_pass=step['passed']))
   seal(root);data['evidence']=StreamEvidence(root);data['evidence'].verify();r=nf.logits(data)
   self.assertEqual(r['historical_numerical_gate'],'FAIL');self.assertTrue(any(not x['frozen_tolerance_pass'] for x in r['comparisons']));self.assertTrue(all(x['generated_tokens_equal'] for x in r['comparisons']))
   data['old']['numerical']['comparisons'][0]['frozen_tolerance_pass']=False
   with self.assertRaisesRegex(ValueError,'does not reproduce'):nf.logits(data)
 def test_missing_input_is_structured_rejection_cli(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cmd=[sys.executable,'scripts/147_r5_i5r_nf1.py','--experiment',str(root/'missing'),'--run',str(root/'run'),'--audit',str(root/'audit'),'--verification',str(root/'verify'),'--output-dir',str(root/'new')]
   r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(json.loads(r.stderr)['status'],'REJECTED');self.assertFalse((root/'new').exists())

if __name__=='__main__':unittest.main()
