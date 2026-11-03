"""Native-only padding fixtures, not OPT/GPU fidelity evidence."""
import unittest,tempfile,importlib.util
from pathlib import Path
from semcache.edgelora.r5_i5r2c2 import gate,padded,MODES,prepare,validate
from semcache.edgelora.r3b_fixture import ArrayTensor
import test_r5_i5r2b as fixtures

def context(source=True):
 return dict(parent_request_id='S' if source else 'T',pass_index=0,mode='COLD_MISS' if source else 'PARTIAL_PREFIX_HIT',past_length=0 if source else 64,input_length=64 if source else 192,target_token_ids=[1]*(64 if source else 256),source_request_id='S' if not source else None,source_entry_id='real-entry' if not source else None)
class Padding(unittest.TestCase):
 def test_source_appended_and_output_real_only(self):
  x=ArrayTensor([1.]*128,[1,64,2]);calls=[]
  def native(t):calls.append(t);return t
  y,r=padded(native,x,'APPEND');self.assertEqual(y.values,x.values);self.assertEqual(y.shape,x.shape);self.assertEqual(calls[0].values[128:],[0.]*384);self.assertEqual(r['extra_rows'],192)
 def test_suffix_prepended_and_output_real_only(self):
  x=ArrayTensor([2.]*384,[1,192,2]);calls=[]
  def native(t):calls.append(t);return t
  y,r=padded(native,x,'PREPEND');self.assertEqual(y.values,x.values);self.assertEqual(calls[0].values[:128],[0.]*128);self.assertEqual(r['retained'],[64,256])
 def test_off_identity_no_padding(self):
  x=ArrayTensor([1.]*128,[1,64,2]);self.assertIs(padded(lambda t:t,x,None)[0],x)
 def test_gate_baselines_decode_warmup_and_modes(self):
  for mode in MODES:
   cfg=dict(mode=mode,source='S',target='T')
   for c in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE'):self.assertIsNone(gate(cfg,c,context()))
   self.assertEqual(gate(cfg,'SEMCACHE_INTEGRATED',context()),None if mode=='OFF' else 'APPEND')
   self.assertEqual(gate(cfg,'SEMCACHE_INTEGRATED',context(False)),'PREPEND' if mode=='SOURCE_AND_SUFFIX' else None)
   self.assertIsNone(gate(cfg,'SEMCACHE_INTEGRATED',dict(context(),pass_index=1)))
   self.assertIsNone(gate(cfg,'SEMCACHE_INTEGRATED',dict(context(),parent_request_id='W_Adapter_A')))
 def test_invalid_suffix_grant_and_shape_fail(self):
  with self.assertRaises(ValueError):gate(dict(mode='SOURCE_AND_SUFFIX',source='S',target='T'),'SEMCACHE_INTEGRATED',dict(context(False),source_entry_id=None))
  with self.assertRaises(ValueError):padded(lambda x:x,ArrayTensor([1.]*64,[1,32,2]),'APPEND')
 @unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch','numpy')),'actual CPU Torch padding API NOT_RUN; dependencies absent')
 def test_actual_torch_padding_and_slicing_preserve_real_rows(self):
  import torch
  module=torch.nn.Linear(4,4).half().eval()
  for n,direction,offset in ((64,'APPEND',0),(192,'PREPEND',64)):
   x=torch.arange(n*4,dtype=torch.float16).reshape(1,n,4)/1000;before=x.clone();weight=module.weight.detach().clone()
   reference=x.new_zeros((1,256,4));reference[:,offset:offset+n].copy_(x)
   with torch.inference_mode():
    out,receipt=padded(module.forward,x,direction);expected=module(reference)[:,offset:offset+n]
   self.assertTrue(torch.equal(out,expected));self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(module.weight,weight))
   self.assertEqual(list(out.shape),[1,n,4]);self.assertEqual(receipt['retained'],[offset,offset+n])
 def test_fresh_three_mode_plans(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);multi,old=fixtures.fixture_inputs(root/'inputs');prepare(old,multi,None,None,None,None,root/'plans',fixture=True)
   for mode in MODES:self.assertEqual(validate(root/'plans'/mode)['i5r2c2']['mode'],mode)
if __name__=='__main__':unittest.main()
