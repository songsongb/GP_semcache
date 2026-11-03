"""CPU-only fixed-shape construction and historical reference tests."""
import tempfile,unittest
from pathlib import Path
from semcache.edgelora.r3a_artifacts import pack,read,write
from semcache.edgelora.r5_i5r2c1 import operands,flops,schedule,prepare,load,analyze
import test_r5_i5r2c0 as fixtures
class Fixed(unittest.TestCase):
 def test_padding_prefix_and_nonzero_distinct(self):
  short=pack([.5]*256,'float16');long=short+pack([.125]*(192*4),'float16');v,seed=operands(short,long,4)
  self.assertTrue(all(x[:len(short)]==short for x in v.values()));self.assertEqual(v['ZERO'][len(short):],b'\0'*(192*4*2));self.assertNotEqual(v['NONZERO'],long);self.assertEqual(v['SAME256'],long)
 def test_bad_prefix_rejected(self):
  with self.assertRaises(ValueError):operands(pack([1.]*256,'float16'),pack([0.]*1024,'float16'),4)
 def test_flops_bias_and_ratio(self):
  a=flops(64,2560,2560,True);b=flops(256,2560,2560,True)
  self.assertEqual(b['total'],4*a['total']);self.assertEqual(flops(192,2560,2560,True)['total'],b['total']-a['total']);self.assertEqual(a['bias_add'],64*2560)
 def test_balanced_six_repeats_and_two_warmups(self):
  s=schedule()
  for k in ('A','B','ZERO','NONZERO','SAME256'):
   self.assertEqual(sum(x['warmup'] and x['kind']==k for x in s),2);self.assertEqual(sum(not x['warmup'] and x['kind']==k for x in s),6)
  self.assertEqual([x['kind'] for x in s[:5]],list(reversed([x['kind'] for x in s[5:10]])))
 def test_prepare_verified_references_and_source(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p,r,a=fixtures.EvidenceTests().fixture(root);before=(r/'i5_output_manifest.json').read_bytes()
   result=prepare(p,r,a,root/'c1',fixture=True);self.assertEqual(load(root/'c1')[0],result);self.assertEqual((r/'i5_output_manifest.json').read_bytes(),before)
   self.assertEqual(len(result['references']['q']),6)
 def test_missing_audit_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p,r,a=fixtures.EvidenceTests().fixture(root)
   with self.assertRaises(ValueError):prepare(p,r,root/'missing',root/'bad',fixture=True)
 def test_independent_analysis_preserves_historical_reference(self):
  from semcache.edgelora.communication import digest
  from semcache.edgelora.r5_i5_evidence import seal
  from semcache.edgelora.r5_i5r2bs1_runtime import Store
  from semcache.edgelora.r5_i5r2c1 import fresh,CORE
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p,r,a=fixtures.EvidenceTests().fixture(root);plan=prepare(p,r,a,root/'c1',fixture=True)
   fp=dict(tensors={})
   for role in 'qkv':
    for name,item in plan['weights'][role].items():
     if item is not None:fp['tensors']['model.decoder.layers.0.self_attn.'+role+'_proj.'+name]=item
   fp['sha256']=digest(fp['tensors']);plan['model_identity']['base_weight_sha256']=fp['sha256']
   plan.pop('experiment_id');plan['experiment_id']=digest(plan);write(root/'c1/i5r2c1_plan.json',plan);seal(root/'c1')
   run=fresh(root/'newrun');store=Store(run,plan['budget']);rows=[];ev=__import__('semcache.edgelora.r5_i5r2_tensors',fromlist=['Evidence']).Evidence(root/'c1');ev.verify()
   for role in 'qkv':
    for step in plan['schedule']:
     if step['warmup']:continue
     k=step['kind'];data=ev.tensor('',plan['references'][role][step['repetition']]['A' if k=='A' else 'B'])
     t=store.tensor('outputs/'+role+k+str(step['repetition'])+'.bin',data,[1,64,4],'float16')
     rows.append(dict(step,role=role,output=t,operator_cuda_ms=.1,input_sha256=plan['inputs'][role][k]['sha256'],prefix_sha256=plan['inputs'][role]['A']['sha256'],weight=plan['weights'][role],gpu_uuid='FIXTURE',backend={c:plan['backend'][c] for c in CORE}))
   store.put('i5r2c1_plan.json',plan);store.put('i5r2c1_records.json',rows)
   store.put('i5r2c1_status.json',dict(status='COMPLETE',GPU_NOT_RUN=True,experiment_id=plan['experiment_id']))
   store.put('i5r2c1_environment.json',dict(effective=plan['backend'],after=plan['backend'],gpu={'execution_gpu_uuid':'FIXTURE'},model_load_count=1,full_fingerprint=fp))
   store.writer.reconcile();seal(run)
   result=analyze(root/'c1',run,root/'audit');self.assertFalse(result['GPU_DIAGNOSTIC_COMPLETE']);self.assertEqual(result['NUMERICAL_FIDELITY_RESTORED'],'NOT_EVALUATED')
   comparisons=read(root/'audit/i5r2c1_comparisons.json');self.assertTrue(all(x['FIXED_SHAPE_PREFIX_EXACT'] for x in comparisons))
   rows.pop();write(run/'i5r2c1_records.json',rows);seal(run)
   with self.assertRaises(ValueError):analyze(root/'c1',run,root/'bad')

if __name__=='__main__':unittest.main()
