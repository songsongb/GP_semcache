"""Scalar/coordinate/hook lifecycle tests; real model paths are explicitly gated."""
import importlib.util
from types import SimpleNamespace as NS
import unittest
from semcache.diagnostics.r2b15_analysis import alignment,first_findings,resolve_layers
from semcache.diagnostics.r2b15_numeric import aligned_reference,Recorder
from semcache.diagnostics.r2b1_prefix_cache import PrefixError


class View:
    def __init__(self,shape): self.shape=shape
    def __getitem__(self,slices):
        slices=slices+(slice(None),)*(len(self.shape)-len(slices))
        return View(tuple(len(range(n)[s]) if isinstance(s,slice) else 1 for n,s in zip(self.shape,slices)))


class NumericContracts(unittest.TestCase):
    def test_prefix_suffix_and_attention_coordinates(self):
        for label,start,stop,actual in [('FULL_PREFIX_vs_SOURCE',0,3,(1,3,4)),('FULL_SUFFIX_vs_WARM',3,8,(1,5,4))]:
            ref,positions=aligned_reference(View((1,8,4)),View(actual),'tokens',label,3,8)
            self.assertEqual(positions,list(range(start,stop))); self.assertEqual(ref.shape,actual)
        ref,pos=aligned_reference(View((1,2,8,8)),View((1,2,3,3)),'attention','FULL_PREFIX_vs_SOURCE',3,8)
        self.assertEqual(ref.shape,(1,2,3,3)); self.assertEqual(pos,[0,1,2])
        ref,pos=aligned_reference(View((1,2,8,8)),View((1,2,5,8)),'attention','FULL_SUFFIX_vs_WARM',3,8)
        self.assertEqual(ref.shape,(1,2,5,8))
        with self.assertRaises(PrefixError): aligned_reference(View((1,8,4)),View((1,5,1)),'tokens','FULL_SUFFIX_vs_WARM',3,8)
    def test_chronological_first_bitwise_and_tolerance_separate(self):
        rows=[dict(comparison='prefix',execution_order=i,layer=layer,module=stage,exact_equal=eq,bitwise_equal=bits,allclose=close)
            for i,layer,stage,eq,bits,close in [(0,-1,'embedding',True,True,True),(1,0,'q_base',True,False,True),
                (2,1,'ln',False,False,True),(3,1,'attention',False,False,False),(4,-1,'lm_head',False,False,False)]]
        result=first_findings(list(reversed(rows)))['prefix']
        self.assertEqual(result['first_bitwise_difference']['module'],'q_base')
        self.assertEqual(result['first_nonzero']['module'],'ln')
        self.assertEqual(result['first_tolerance_failure']['module'],'attention')
        self.assertEqual(first_findings(rows+[dict(rows[0],comparison='suffix')])['suffix']['status'],'NO_DIFFERENCE_AT_CAPTURED_BOUNDARIES')
    def test_layer_scope(self):
        self.assertEqual(resolve_layers('0,2,last',4),[0,2,3])
        self.assertEqual(resolve_layers('all',2),[0,1])
        with self.assertRaises(PrefixError): resolve_layers('0,last',1)
        with self.assertRaises(PrefixError): alignment((8,4),'tokens',0,3)
    def test_cleanup_with_fake_modules_on_exception(self):
        all_modules=[]
        class Module:
            def __init__(self): self.handles=[]; all_modules.append(self)
            def register_forward_hook(self,*args,**kwargs):
                h=NS(removed=False)
                def remove(): h.removed=True
                h.remove=remove; self.handles.append(h); return h
            register_forward_pre_hook=register_forward_hook
        qkv={r:Module() for r in 'qkv'}
        for m in qkv.values():
            m.base=Module(); m.get_base_layer=lambda m=m:m.base; m.lora_B={'user_a':Module()}; m.scaling={'user_a':1.}
        layer=Module(); layer.self_attn=Module(); layer.self_attn.out_proj=Module()
        layer.self_attn_layer_norm=Module(); layer.final_layer_norm=Module(); layer.fc1=Module(); layer.fc2=Module(); layer.activation_fn=lambda x:x
        original=layer.activation_fn
        r=NS(a=NS(adapter='user_a'),adapter=NS(layers=[layer],projection_modules=lambda _:qkv),
            decoder=NS(embed_tokens=Module(),embed_positions=Module(),final_layer_norm=Module()),base=NS(lm_head=Module()))
        record=Recorder(r,[0])
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with record.hooks():
                self.assertIsNot(layer.activation_fn,original); raise RuntimeError('stop')
        self.assertIs(layer.activation_fn,original)
        self.assertTrue(all(h.removed for m in all_modules for h in m.handles))


DEPS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: actual numeric/model audit NOT EXECUTED')
class TensorTests(unittest.TestCase):
    def test_bitwise_signed_zero_and_shape_rejection(self):
        import torch
        from semcache.diagnostics.r2b15_numeric import observed_metrics
        a=torch.tensor([[[0.,1.]]]); b=torch.tensor([[[-0.,1.]]])
        r=observed_metrics(a,b,[4],'tokens')
        self.assertTrue(r['exact_equal']); self.assertFalse(r['bitwise_equal']); self.assertEqual(r['first_bitwise_element']['token_index'],4)
        c=b.clone(); c[0,0,1]+=.1; r=observed_metrics(a,c,[4],'tokens')
        self.assertFalse(r['allclose'])
    def exercise(self,device):
        from test_r2b1_prefix_runtime import OfflineOPTTests
        from semcache.diagnostics.r2b15_numeric import numeric_case
        r=OfflineOPTTests().runtime(device); r.prepare(32,16)
        rows,kv,probes,summary,audit=numeric_case(r,[0,1],32,probe=False)
        self.assertTrue(summary['quality']['full_repeat']['exact_equal']); self.assertTrue(summary['quality']['suffix']['allclose'])
        self.assertTrue(all(x['exact_equal'] for x in kv if x['comparison']=='SOURCE_vs_WARM_PAST'))
        self.assertIn('FULL_PREFIX_vs_SOURCE',summary['first_findings'])
        self.assertIn('FULL_SUFFIX_vs_WARM',summary['first_findings'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in r.model.modules()))
        record=Recorder(r,[0,1],limit_mib=.0001)
        with self.assertRaises(MemoryError):
            with record.hooks(): r.forward('FULL_USE_CACHE')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in r.model.modules()))
    def test_cpu_tiny_opt(self): self.exercise('cpu')
    def test_cuda_tiny_opt(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.exercise('cuda:0')

if __name__=='__main__': unittest.main()
