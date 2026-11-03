"""Synthetic CPU evidence only; no native GPU numerical claims."""
import copy,importlib.util,json,struct,subprocess,tempfile,unittest
from pathlib import Path
from semcache.edgelora.communication import digest,digest_bytes
from semcache.edgelora.r3a_artifacts import ROOT,THRESHOLDS,pack,write
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5r2c3 import causal_mask,aligned_bytes,compare_aligned,verdict,stage_a,history,fresh,BOUNDARIES
from semcache.edgelora.r5_multi_artifacts import runtime_view


def fixture(root,complete=False,mode='SOURCE_ONLY'):
 plan=root/'plan';run=root/'run';audit=root/'audit'
 for p in (plan,run,audit):p.mkdir(parents=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r3b_runtime.py'
 sha=digest_bytes(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))
 source=dict(commit=commit,files={name:sha});model=dict(hidden_size=4,base_weight_sha256='base-fingerprint',adapter='BASE_PREPARATION',adapter_matrix_sha256='base-adapter',rank=8,delta_dtype='float32')
 adapter=dict(adapter_id='Adapter_A',matrix_sha256='adapter-matrix',base_fingerprint=model['base_weight_sha256'],rank=8,matrix_dtype='float32')
 tasks=[dict(request_id=r,token_ids=list(range(n)),user_id='User_01',adapter_id='Adapter_A') for r,n in (('S',64),('F',256))]
 block=dict(tasks=tasks);cfg2=dict(mode=mode,source='S',target='F')
 p=dict(blocks=[block],i5r2c2=cfg2,i5r2b=dict(source=source),i5r1=dict(model_identity=model,adapter_fingerprints={'Adapter_A':'adapter-matrix'}))
 p['experiment_id']=digest(p);write(plan/'i5_workload_manifest.json',p);seal(plan)
 ps=Evidence(plan).verify();cfg=dict(fixture=True,source_commit=commit,sources=source['files'],block=block,experiment_id=p['experiment_id'],experiment_manifest_sha256=ps['manifest_sha256'],i5r2c2=cfg2)
 write(run/'i5_execution_config.json',cfg);write(run/'i5_block_status.json',dict(status='COMPLETE'));write(run/'i5_adapter_registry.json',dict(adapters={'Adapter_A':adapter}))
 def tensor(base,name,values,shape):
  data=pack(values,'float16');path=base/(name+'.bin');path.write_bytes(data)
  return dict(file=name+'.bin',shape=shape,dtype='float16',sha256=digest_bytes(data),status='CAPTURED',semantic_positions=[0,64],observed_shape=shape)
 for label,cond,n in (('S','SEMCACHE_INTEGRATED',64),('F','FULL_RECOMPUTE',256)):
  base=run/cond/'Adapter_A/es';base.mkdir(parents=True)
  identity=dict(model_identity=runtime_view({'identity':model},adapter)['identity'],adapter=adapter,gpu=None,physical_identity={'gpu_uuid':'CPU_FIXTURE'},contract={})
  write(base.parent/'identity.json',identity)
  ctx=dict(pass_index=0,past_length=0,input_token_ids=list(range(n)),positions=list(range(n)),attention_mask=[1]*n)
  entries={};qkv={}
  for r in 'qkv':
   data=[(i%9)/16 for i in range(64*4)];entries[label+'/raw_qkv_'+r]=tensor(base,'bounded_'+r,data,[1,64,4]);entries[label+'/raw_qkv_'+r]['observed_shape']=[1,n,4]
   qkv['0.'+r]=tensor(base,'full_'+r,data+[0.]*(n*4-len(data)),[1,n,4]) if complete else dict(shape=[1,n,4],dtype='float16',sha256='hash-only')
  queries=n if complete else 64
  entries[label+'/actual_attention_mask']=tensor(base,'mask',[0. if k<=q else -65504. for q in range(queries) for k in range(n)],[1,1,queries,n])
  entries[label+'/attention_output']=tensor(base,'attention',[0. if label=='S' else .001]*256,[1,64,4])
  write(base/'i5r2b_capture.json',dict(status='COMPLETE',entries=entries,contexts={label:dict(actual_context=ctx,position_observation=dict(positions=list(range(n)),past_length=0),backend_before={'TEST_FIXTURE_ONLY':True},backend_after={'TEST_FIXTURE_ONLY':True})}))
  (base/'r5_tokens.jsonl').write_text(json.dumps(dict(request_id=label,step=0,context=ctx,qkv=qkv))+'\n')
 seal(run);rs=Evidence(run).verify()
 write(audit/'i5r2c2_decision.json',dict(TEST_FIXTURE_ONLY=True,mode=mode,inventory=rs,independent_audit={'status':'PASS'},PHYSICAL_REUSE_PRESERVED='CPU_FIXTURE_ONLY'))
 write(audit/'i5r2c2_numerical.json',dict(target={'thresholds':THRESHOLDS}));seal(audit)
 return plan,run,audit,commit


class Attention(unittest.TestCase):
 def test_valid_adapter_runtime_identity_is_not_base_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d));h=history(p,r,a,c,True)
   self.assertNotEqual(h['identities']['S']['model_identity'],h['plan']['i5r1']['model_identity'])
   self.assertEqual(h['identities']['S']['model_identity'],runtime_view({'identity':h['plan']['i5r1']['model_identity']},h['identities']['S']['adapter'])['identity'])
 def changed_runtime_identity(self,field,value):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d));path=r/'SEMCACHE_INTEGRATED/Adapter_A/identity.json';ident=json.loads(path.read_text());ident['model_identity'][field]=value;write(path,ident);seal(r)
   ad=json.loads((a/'i5r2c2_decision.json').read_text());ad['inventory']=Evidence(r).verify();write(a/'i5r2c2_decision.json',ad);seal(a)
   with self.assertRaisesRegex(ValueError,'historical model/adapter differs'):history(p,r,a,c,True)
 def test_changed_base_fingerprint_rejected(self):
  self.changed_runtime_identity('base_weight_sha256','changed-base')
 def test_changed_adapter_fingerprint_rejected(self):
  self.changed_runtime_identity('adapter_matrix_sha256','changed-adapter')
 def test_wrong_runtime_rank_dtype_or_adapter_rejected(self):
  for field,value in (('rank',4),('delta_dtype','float16'),('adapter','Adapter_B')):
   with self.subTest(field=field):self.changed_runtime_identity(field,value)
 def test_registry_fingerprint_rank_and_dtype_remain_pinned(self):
  for field,value in (('matrix_sha256','changed-adapter'),('rank',4),('matrix_dtype','float16'),('base_fingerprint','changed-base'),('adapter_id','Adapter_B')):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    p,r,a,c=fixture(Path(d));registry=json.loads((r/'i5_adapter_registry.json').read_text());registry['adapters']['Adapter_A'][field]=value;write(r/'i5_adapter_registry.json',registry);seal(r)
    ad=json.loads((a/'i5r2c2_decision.json').read_text());ad['inventory']=Evidence(r).verify();write(a/'i5r2c2_decision.json',ad);seal(a)
    with self.assertRaisesRegex(ValueError,'historical model/adapter differs'):history(p,r,a,c,True)
 def test_actual_causal_mask_all_future_keys_checked(self):
  good=pack([0 if k<=q else -65504 for q in range(64) for k in range(256)],'float16')
  self.assertTrue(causal_mask(good,[1,1,64,256],'float16',64,256)['future_keys_masked'])
  bad=bytearray(good);bad[128:130]=pack([0],'float16')
  with self.assertRaises(ValueError):causal_mask(bytes(bad),[1,1,64,256],'float16',64,256)
 def test_mask_negative_infinity_supported_but_nan_rejected(self):
  self.assertTrue(causal_mask(struct.pack('<4e',0,-float('inf'),0,0),[1,1,2,2],'float16',2,2)['causal'])
  with self.assertRaises(ValueError):causal_mask(struct.pack('<4e',0,float('nan'),0,0),[1,1,2,2],'float16',2,2)
 def test_score_alignment_removes_future_keys_without_unmasking(self):
  short=pack([1.]*64*64,'float16');long=pack([v for q in range(256) for v in ([1.]*64+[2.]*192)],'float16')
  m=compare_aligned((short,[1,64,64]),(long,[1,256,256]),'float16','scores');self.assertTrue(m['bitwise_equal']);self.assertEqual(m['elements'],4096)
 def test_output_alignment_does_not_include_suffix_queries(self):
  a=pack([1.]*128,'float16');b=a+pack([2.]*384,'float16')
  self.assertEqual(aligned_bytes(b,[1,256,2],'float16','pv'),a)
 def test_injected_perturbation_and_frozen_thresholds(self):
  a=pack([1.]*128,'float16');b=pack([1.04]+[1.]*127,'float16')
  m=compare_aligned((a,[1,64,2]),(b,[1,64,2]),'float16','pv');self.assertFalse(m['activation_tolerance_pass']);self.assertGreater(m['FP16_ULP']['max'],0);self.assertGreater(m['differing_percentage'],0)
 def test_replay_and_repeatability_mandatory_before_localization(self):
  rows=[dict(boundary=b,metrics={'bitwise_equal':b!='probabilities'}) for b in BOUNDARIES]
  for stable,replay in ((True,False),(False,True),(False,False)):
   v=verdict(rows,stable,replay);self.assertEqual(v['FIRST_DIVERGENCE_LOCALIZED'],'NOT_ESTABLISHED');self.assertTrue(v['REPLAY_NOT_ESTABLISHED'])
  v=verdict(rows,True,True);self.assertEqual(v['FIRST_DIVERGENCE_LOCALIZED'],'probabilities');self.assertFalse(v['ROOT_CAUSE_CONFIRMED']);self.assertEqual(v['STRICT_NUMERICAL_FIDELITY_RESTORED'],'NOT_EVALUATED')
 def test_cpu_fixture_never_reports_gpu_localization(self):
  v=verdict([dict(boundary='scores',metrics={'bitwise_equal':False})],True,True,fixture=True);self.assertFalse(v['HISTORICAL_REPLAY_REPRODUCED'])
 def test_partial_historical_coverage_sealed_and_deterministic(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d));before={str(f):digest_bytes(f.read_bytes()) for root in (p,r,a) for f in root.rglob('*') if f.is_file()}
   x=stage_a(p,r,a,Path(d)/'out',c,allow_incomplete=True,fixture=True);y=stage_a(p,r,a,Path(d)/'out2',c,allow_incomplete=True,fixture=True)
   self.assertEqual(x,y);self.assertTrue(x['INCOMPLETE_EVIDENCE']);self.assertFalse(x['GPU_REPLAY_READY']);Evidence(Path(d)/'out').verify()
   self.assertTrue(any('256-row raw q' in m for m in x['missing']));self.assertTrue(any('256-query' in m for m in x['missing']))
   self.assertEqual(before,{str(f):digest_bytes(f.read_bytes()) for root in (p,r,a) for f in root.rglob('*') if f.is_file()})
 def test_missing_requires_explicit_partial_mode(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d))
   with self.assertRaisesRegex(ValueError,'INCOMPLETE_EVIDENCE'):stage_a(p,r,a,Path(d)/'out',c,fixture=True)
   Evidence(Path(d)/'out').verify()
 def test_complete_operands_are_detected_without_cpu_arithmetic_claims(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d),complete=True);h=history(p,r,a,c,True);self.assertFalse(h['missing']);self.assertTrue(all(x['metrics']['bitwise_equal'] for x in h['observations'][:3]))
 def test_corruption_and_wrong_historical_commit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d))
   with self.assertRaises(ValueError):history(p,r,a,'0'*40,True)
   (r/'SEMCACHE_INTEGRATED/Adapter_A/es/bounded_q.bin').write_bytes(b'bad')
   with self.assertRaisesRegex(ValueError,'corrupt'):history(p,r,a,c,True)
 def test_plan_and_audit_identity_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d));ad=json.loads((a/'i5r2c2_decision.json').read_text());ad['mode']='OFF';write(a/'i5r2c2_decision.json',ad);seal(a)
   with self.assertRaisesRegex(ValueError,'audit identity'):history(p,r,a,c,True)
 def test_hash_only_full_tensor_is_never_filled(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d));h=history(p,r,a,c,True);self.assertNotIn('q',h['inputs']['F']);self.assertIn('q',h['inputs']['S'])
 def test_actual_position_mismatch_rejected_even_with_resealed_fixture(self):
  with tempfile.TemporaryDirectory() as d:
   p,r,a,c=fixture(Path(d));path=r/'SEMCACHE_INTEGRATED/Adapter_A/es/i5r2b_capture.json'
   cap=json.loads(path.read_text());cap['contexts']['S']['position_observation']['positions'][0]=1;write(path,cap);seal(r)
   ad=json.loads((a/'i5r2c2_decision.json').read_text());ad['inventory']=Evidence(r).verify();write(a/'i5r2c2_decision.json',ad);seal(a)
   with self.assertRaisesRegex(ValueError,'actual position-module'):history(p,r,a,c,True)
 def test_protected_historical_paths_and_fresh_output(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'evidence').mkdir()
   with self.assertRaises(ValueError):fresh(root/'evidence/output',(root/'evidence',))
   with self.assertRaises(ValueError):fresh(root/'evidence')
 def test_repetitions_complete_balanced_and_missing_duplicate_fail(self):
  from semcache.edgelora.r5_i5r2c3_runtime import schedule,validate_repetitions
  rows=[dict(s,boundaries={b:{} for b in (*BOUNDARIES,'output_projection_input')}) for s in schedule() if not s['warmup']]
  self.assertEqual(len(rows),12);validate_repetitions(rows)
  for broken in (rows[:-1],rows+[rows[0]],list(reversed(rows))):
   with self.assertRaises(ValueError):validate_repetitions(broken)
 def test_missing_boundary_rejected(self):
  from semcache.edgelora.r5_i5r2c3_runtime import schedule,validate_repetitions
  rows=[dict(s,boundaries={b:{} for b in BOUNDARIES}) for s in schedule() if not s['warmup']]
  with self.assertRaisesRegex(ValueError,'missing attention boundary'):validate_repetitions(rows)
 def test_historical_git_blob_hash_mismatch_fails(self):
  from semcache.edgelora.r5_i5r2_analysis import source_provenance
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
  with self.assertRaisesRegex(ValueError,'source SHA-256 mismatch'):
   source_provenance(dict(source_commit=commit,sources={'src/semcache/edgelora/r3b_runtime.py':'0'*64}),expected_commit=commit,fixture=False)
 def test_masked_finite_values_have_ulp_and_infinite_pattern_supported(self):
  finite=pack([0.]*4096,'float16');m=compare_aligned((finite,[1,64,64]),(finite,[1,64,64]),'float16','masked_scores');self.assertIn('FP16_ULP',m)
  data=struct.pack('<4096e',*([-float('inf')]*4096));m=compare_aligned((data,[1,64,64]),(data,[1,64,64]),'float16','masked_scores');self.assertTrue(m['bitwise_equal'])
 @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch','numpy','transformers')),'installed native OPTAttention CPU API NOT_RUN: dependencies unavailable')
 def test_installed_attention_native_operation_observer_cpu(self):
  import torch
  from transformers import OPTConfig
  from transformers.models.opt.modeling_opt import OPTAttention
  from semcache.edgelora.r5_i5r2c3_runtime import native_replay
  config=OPTConfig(hidden_size=8,num_attention_heads=2,attention_dropout=0.)
  config._attn_implementation='eager';module=OPTAttention(config=config,layer_idx=0).eval()
  x=torch.zeros(1,64,8);qkv={r:x.clone() for r in 'qkv'};qkv['mask']=torch.zeros(1,1,64,64);seen={};weights=module.q_proj.weight.clone();original=module.q_proj.forward
  with torch.inference_mode():result=native_replay(module,qkv,lambda n,t,*args:seen.setdefault(n,t.detach().clone()))
  self.assertEqual(set(seen),set(BOUNDARIES)|{'output_projection_input'});self.assertEqual(result.shape,x.shape);self.assertTrue(torch.equal(module.q_proj.weight,weights));self.assertEqual(module.q_proj.forward,original)

if __name__=='__main__':unittest.main()
