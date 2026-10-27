"""B3 numeric specifications run without Torch; tensor/hook tests explicitly skip."""
import importlib.util
import math
from types import SimpleNamespace as NS
import unittest
from semcache.diagnostics.r1b3_fp16_divergence import scalar_metrics,first_divergences,tensor_metrics,logit_metrics,StreamingAudit

class CorrectnessTests(unittest.TestCase):
    def test_strict_element_tolerance_and_zero_reference_norm(self):
        x=scalar_metrics([0.,1.,2.],[.002,1.,2.])
        self.assertFalse(x['allclose']); self.assertEqual(x['tolerance_violations'],1)
        self.assertEqual(x['first_tolerance_failure'],0); self.assertAlmostEqual(x['mean_abs_error'],.002/3)
        self.assertIsNone(scalar_metrics([0.],[1.])['relative_l2_error'])
        self.assertTrue(scalar_metrics([0.,1.],[0.,1.])['exact_equal'])
        with self.assertRaises(ValueError): scalar_metrics([math.inf],[1.])

    def test_first_divergence_uses_execution_order_and_length(self):
        rows=[dict(comparison='FULL_vs_REUSE',length=n,layer=0,stage=s,role=r,exact_equal=eq,allclose=close)
            for n in (128,256) for s,r,eq,close in [('layer_input','',True,True),('base_projection','k',False,True),('total_qkv','q',False,False)]]
        first=first_divergences(rows)
        self.assertEqual(len(first),2); self.assertEqual(first[0]['first_nonzero']['role'],'k')
        self.assertEqual(first[0]['first_tolerance_failure']['stage'],'total_qkv')

TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'NOT EXECUTED: Torch unavailable; no numerical/hook parity claim')
class TensorTests(unittest.TestCase):
    def test_tensor_metrics_strict_fail_with_argmax_pass(self):
        import torch
        a=torch.tensor([[[1.,2.,3.]]]); b=a+0.02
        q=logit_metrics(a,b)
        self.assertFalse(q['allclose']); self.assertTrue(q['last_argmax_agreement'])
        self.assertEqual(q['tolerance_violations'],3)
        self.assertTrue(q['margin_exceeds_twice_max_error'])
        with self.assertRaises(ValueError): tensor_metrics(a,torch.tensor([[[float('nan'),2.,3.]]]))

    def test_streaming_real_hooks_base_delta_and_downstream(self):
        import torch
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.base=torch.nn.Linear(4,4); self.lora_A=torch.nn.ModuleDict({'user_a':torch.nn.Linear(4,2,bias=False)})
                self.lora_B=torch.nn.ModuleDict({'user_a':torch.nn.Linear(2,4,bias=False)}); self.scaling={'user_a':1.}
            def get_base_layer(self): return self.base
            def forward(self,x): return self.base(x)+self.lora_B['user_a'](self.lora_A['user_a'](x))
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.q_proj=Projection(); self.k_proj=Projection(); self.v_proj=Projection()
            def forward(self,x): return self.q_proj(x)+self.k_proj(x)+self.v_proj(x),None
        class Layer(torch.nn.Module):
            def __init__(self): super().__init__(); self.self_attn=Attention()
            def forward(self,hidden_states): return (hidden_states+self.self_attn(hidden_states)[0],)
        decoder=torch.nn.Module(); decoder.embed_tokens=torch.nn.Embedding(8,4); decoder.embed_positions=torch.nn.Embedding(8,4); decoder.layers=torch.nn.ModuleList([Layer()])
        model=NS(model=NS(decoder=decoder)); adapter=NS(model=model,layers=decoder.layers,projection_modules=lambda _:dict(q=decoder.layers[0].self_attn.q_proj,k=decoder.layers[0].self_attn.k_proj,v=decoder.layers[0].self_attn.v_proj))
        def forward():
            x=decoder.embed_tokens(torch.tensor([[1,2,3]]))+decoder.embed_positions(torch.tensor([[0,1,2]]))
            return decoder.layers[0](x)
        reference=StreamingAudit(adapter,'user_a',[0],'REF',retain=True)
        with reference.hooks(): forward()
        candidate=StreamingAudit(adapter,'user_a',[0],'SAME',reference.snapshots)
        with candidate.hooks(): forward()
        self.assertTrue(all(r['exact_equal'] for r in candidate.rows))
        with torch.no_grad(): decoder.layers[0].self_attn.q_proj.base.weight.add_(0.1)
        candidate=StreamingAudit(adapter,'user_a',[0],'CHANGED',reference.snapshots)
        with candidate.hooks(): forward()
        first=next(r for r in candidate.rows if not r['exact_equal'])
        self.assertEqual((first['stage'],first['role']),('base_projection','q'))
        self.assertFalse(any(m._forward_hooks for m in decoder.modules()))

if __name__=='__main__': unittest.main()
