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
  rows=[dict(s,boundaries={b:{} for b in BOUNDARIES if b!='output_projection_input'}) for s in schedule() if not s['warmup']]
  with self.assertRaisesRegex(ValueError,'missing attention boundary'):validate_repetitions(rows)
 def test_historical_git_blob_hash_mismatch_fails(self):
  from semcache.edgelora.r5_i5r2_analysis import source_provenance
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
  with self.assertRaisesRegex(ValueError,'source SHA-256 mismatch'):
   source_provenance(dict(source_commit=commit,sources={'src/semcache/edgelora/r3b_runtime.py':'0'*64}),expected_commit=commit,fixture=False)
 def test_masked_finite_values_have_ulp_and_infinite_pattern_supported(self):
  finite=pack([0.]*4096,'float16');m=compare_aligned((finite,[1,64,64]),(finite,[1,64,64]),'float16','masked_scores');self.assertIn('FP16_ULP',m)
  data=struct.pack('<4096e',*([-float('inf')]*4096));m=compare_aligned((data,[1,64,64]),(data,[1,64,64]),'float16','masked_scores');self.assertTrue(m['bitwise_equal'])
 def test_native_rank4_canonical_shapes_and_invalid_batches(self):
  from semcache.edgelora.r5_i5r2c3_runtime import canonical_boundary_shape
  for n in (64,256):
   for role in ('scores','masked_scores','probabilities','pv'):
    last=4 if role=='pv' else n
    self.assertEqual(canonical_boundary_shape([1,2,n,last],role,2,n,4),[2,n,last])
    self.assertEqual(canonical_boundary_shape([2,n,last],role,2,n,4),[2,n,last])
    with self.assertRaises(ValueError):canonical_boundary_shape([2,2,n,last],role,2,n,4)
   with self.assertRaises(ValueError):canonical_boundary_shape([1,n,2,4],'scores',2,n,4)
 def test_rank4_query_key_prefix_axes_and_hash_alignment(self):
  from semcache.edgelora.r5_i5r2c3_runtime import operand_prefix
  for role,axis in (('q',2),('k',3),('v',2),('probabilities',2)):
   results=[]
   for n in (64,256):
    shape=[1,2,4,n] if role=='k' else [1,2,n,n if role=='probabilities' else 4]
    if role=='k':values=[float(h*1000+d*100+t) for h in range(2) for d in range(4) for t in range(n)]
    else:values=[float(h*1000+t*10+d) for h in range(2) for t in range(n) for d in range(shape[-1])]
    data=pack(values,'float32');prefix,selected=operand_prefix(data,shape,'float32',axis,also_keys=role=='probabilities')
    self.assertEqual(selected[axis],64)
    if role=='probabilities':self.assertEqual(selected[-1],64)
    results.append(digest_bytes(prefix))
   self.assertEqual(results[0],results[1],role)
 def test_independent_k_v_prefix_hash_reconstruction(self):
  from semcache.edgelora.r5_i5r2c3_runtime import head_prefix_bytes,operand_prefix
  for n in (64,256):
   raw=pack([float(t*8+h*4+j) for t in range(n) for h in range(2) for j in range(4)],'float32')
   keys=pack([float(t*8+h*4+j) for h in range(2) for j in range(4) for t in range(n)],'float32')
   values=pack([float(t*8+h*4+j) for h in range(2) for t in range(n) for j in range(4)],'float32')
   self.assertEqual(head_prefix_bytes(raw,n,2,4,width=4,transpose=True),operand_prefix(keys,[1,2,4,n],'float32',3)[0])
   self.assertEqual(head_prefix_bytes(raw,n,2,4,width=4),operand_prefix(values,[1,2,n,4],'float32',2)[0])
 def fake_observer(self):
  from types import SimpleNamespace
  from semcache.edgelora.r5_i5r2c3_runtime import AttentionOperations
  class T:
   def __init__(self,shape,dtype='float32'):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device='cpu'
   def stride(self):
    stride=[];n=1
    for size in self.shape[::-1]:stride.insert(0,n);n*=size
    return tuple(stride)
   def storage_offset(self):return 0
   def is_contiguous(self):return True
  operands={r:T([1,64,8]) for r in 'qkv'};operands['mask']=T([1,1,64,64]);seen={}
  state=AttentionOperations(SimpleNamespace(num_heads=2,head_dim=4,training=False),operands,lambda role,t,*args:seen.setdefault(role,(t,args)))
  state.equal_head=lambda *a:True
  return state,T,operands,seen
 def test_native_matmul_state_order_and_actual_pv_probability_identity(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  state,T,x,seen=self.fake_observer();q=T([1,2,64,4]);k=T([1,2,4,64]);v=T([1,2,64,4]);scores=T([1,2,64,64]);masked=T([1,2,64,64]);soft=T([1,2,64,64]);cast=T([1,2,64,64]);drop=T([1,2,64,64])
  with patch.dict('sys.modules',{'torch':SimpleNamespace(equal=lambda a,b:True)}),patch('semcache.edgelora.r5_i5r2c3_runtime.tensor_bytes',side_effect=lambda t:str(t.shape).encode()):
   state.operation('view',(x['q'],),{},q);state.operation('matmul',(q,k),{},scores)
   state.operation('add',(scores,x['mask']),{},masked);state.operation('softmax',(masked,),{'dim':-1},soft)
   state.operation('to',(soft,),{},cast);state.operation('dropout',(cast,),{'training':False,'p':0.},drop)
   state.operation('matmul',(drop,v),{},T([1,2,64,4]));state.finish()
  self.assertIs(seen['probabilities'][0],drop);self.assertEqual([o['operation'] for o in state.trace['operations']],['matmul','matmul'])
  self.assertEqual([axis for t,axis in seen['scores'][1][0]],[2,3]);self.assertEqual([axis for t,axis in seen['pv'][1][0]],[2,2]);self.assertTrue(state.trace['dropout']['observed'])
 def test_unsupported_fused_or_wrong_product_shapes_fail_closed(self):
  for name,shape in (('matmul',[2,2,64,4]),('matmul',[1,2,4,64]),('scaled_dot_product_attention',[1,2,64,4])):
   state,T,x,seen=self.fake_observer()
   with self.subTest(name=name,shape=shape),self.assertRaises(ValueError):state.operation(name,(T(shape),T([1,2,4,64])),{},T([1,2,64,64]))
  state,T,x,seen=self.fake_observer()
  with self.assertRaises(ValueError):state.finish()
 def test_softmax_and_pv_cannot_be_identified_by_count_only(self):
  state,T,x,seen=self.fake_observer()
  with self.assertRaises(ValueError):state.operation('softmax',(T([1,2,64,64]),),{'dim':-1},T([1,2,64,64]))
  state.phase='SOFTMAX';state.prob=[T([1,2,64,64])]
  with self.assertRaises(ValueError):state.operation('matmul',(T([1,2,64,64]),T([1,2,64,4])),{},T([1,2,64,4]))
 @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torch','numpy','transformers')),'installed native OPTAttention CPU API NOT_RUN: dependencies unavailable')
 def test_installed_attention_native_operation_observer_cpu(self):
  import torch
  from unittest.mock import patch
  from transformers import OPTConfig
  from transformers.models.opt.modeling_opt import OPTAttention
  from semcache.edgelora.r5_i5r2c3_runtime import native_replay,canonical_boundary,operand_prefix
  from semcache.edgelora.r3a_artifacts import tensor_bytes
  config=OPTConfig(hidden_size=8,num_attention_heads=2,attention_dropout=0.);config._attn_implementation='eager'
  module=OPTAttention(config=config,layer_idx=0).eval();weights={k:v.clone() for k,v in module.state_dict().items()};original=module.q_proj.forward
  torch.manual_seed(42);raw={r:torch.randn(1,256,8) for r in 'qkv'};receipts=[]
  for n in (64,256):
   qkv={r:t[:,:n].clone() for r,t in raw.items()};qkv['mask']=torch.triu(torch.full((1,1,n,n),torch.finfo(torch.float32).min),diagonal=1);seen={};trace={};prefixes={}
   def save(role,t,operands=None):
    canonical=canonical_boundary(t,role,2,n,4);seen[role]=(t.clone(),canonical.clone())
    if operands:
     prefixes[role]=[digest_bytes(operand_prefix(tensor_bytes(v),list(v.shape),'float32',axis,also_keys=role=='pv' and j==0)[0]) for j,(v,axis) in enumerate(operands)]
   with torch.inference_mode():
    with patch.object(module.q_proj,'forward',lambda x:qkv['q']),patch.object(module.k_proj,'forward',lambda x:qkv['k']),patch.object(module.v_proj,'forward',lambda x:qkv['v']):
     baseline=module(hidden_states=qkv['q'],attention_mask=qkv['mask'],output_attentions=False)[0]
    result=native_replay(module,qkv,save,trace=trace)
   self.assertEqual(set(seen),set(BOUNDARIES)|{'output_projection_input'});self.assertTrue(torch.equal(result,baseline));self.assertEqual(module.q_proj.forward,original)
   for key,value in weights.items():self.assertTrue(torch.equal(value,module.state_dict()[key]))
   self.assertEqual(len(trace['operations']),2)
   for role in ('scores','masked_scores','probabilities','pv'):
    observed,canonical=seen[role];self.assertEqual(list(canonical.shape),[2,n,4 if role=='pv' else n])
    if trace['operations'][0]['operation']!='bmm':self.assertEqual(observed.ndim,4)
   if trace['operations'][0]['operation']!='bmm':self.assertTrue(all(o['operation'] in ('matmul','__matmul__') for o in trace['operations']))
   receipts.append(prefixes['scores'])
   print('NATIVE_CPU_OBSERVATION',n,[(o['operation'],o['left']['shape'],o['right']['shape']) for o in trace['operations']], 'softmax',trace['softmax']['input']['shape'])
  self.assertEqual(receipts[0],receipts[1])

if __name__=='__main__':unittest.main()
