from types import SimpleNamespace as NS
import importlib.util
import unittest
from semcache.diagnostics.r2b35_executor import native_contract,decoder_then_head,Runtime
from semcache.diagnostics.r2b1_prefix_cache import PrefixError

SOURCE='''def forward(self, input_ids=None):
    outputs = self.model.decoder(input_ids=input_ids)
    logits = self.lm_head(outputs[0]).contiguous()
    return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
'''


class Tensor:
    def __init__(self,tokens,dim): self.shape=(1,tokens,dim); self.ndim=3
    def __getitem__(self,key):
        start,stop,step=key[1].indices(self.shape[1]); return Tensor(len(range(start,stop,step)),self.shape[2])
    def contiguous(self): return self


class ExecutorTests(unittest.TestCase):
    def test_source_contract_no_posthoc_or_hidden_postprocess(self):
        self.assertEqual(native_contract(SOURCE)['dtype_policy'],'native head dtype; no added cast')
        for changed in [SOURCE.replace('.contiguous()','.float()'),SOURCE.replace('outputs[0]','outputs[0][:,-1:,:]'),
            SOURCE.replace('return CausalLMOutputWithPast','logits = logits.float()\n    return CausalLMOutputWithPast'),
            SOURCE.replace('return CausalLMOutputWithPast','extra = self.lm_head(outputs[0])\n    return CausalLMOutputWithPast'),
            SOURCE.replace('logits=logits','logits=logits.float()')]:
            with self.assertRaises(PrefixError): native_contract(changed)

    def test_real_hidden_slicing_before_only_head_and_same_kv(self):
        for tokens in (1,16,32):
            calls=[]; cache=object()
            class Output:
                past_key_values=cache; hidden_states=None; attentions=None
                def __getitem__(self,key): return Tensor(tokens,8)
            def decoder(**kwargs): calls.append(('decoder',kwargs)); return Output()
            def head(hidden): calls.append(('head',hidden.shape)); return Tensor(hidden.shape[1],64)
            out=decoder_then_head(decoder,head,NS,{'input_ids','past_key_values','use_cache'},
                dict(input_ids='same',past_key_values=cache,use_cache=True,labels=None,logits_to_keep=0))
            self.assertEqual(out.logits.shape,(1,1,64)); self.assertIs(out.past_key_values,cache)
            self.assertEqual([c[0] for c in calls],['decoder','head']); self.assertEqual(calls[-1][1],(1,1,8))
            self.assertIs(calls[0][1]['past_key_values'],cache)
        with self.assertRaises(PrefixError): decoder_then_head(None,None,None,set(),{'labels':'nonempty'})

    def test_patch_restore_and_frozen_timer_prepare_inheritance(self):
        from semcache.diagnostics.r2b2_benchmark import B2Runtime
        from semcache.diagnostics.r2b1_prefix_runtime import PrefixRuntime
        self.assertIs(Runtime.forward,B2Runtime.forward); self.assertIs(Runtime.prepare,PrefixRuntime.prepare)
        r=Runtime.__new__(Runtime); decoder=object(); head=object(); original=lambda **kw:None
        r.base=NS(forward=original,get_decoder=lambda:decoder,get_output_embeddings=lambda:head)
        r.model=NS(get_base_model=lambda:r.base); r.binding=(id(r.base),id(decoder),id(head))
        r.endpoint={'effective':'b1_all_logits'}; r.matched_decoder=decoder; r.matched_head=head
        try:
            with r.matched_scope():
                self.assertIsNot(r.base.forward,original); self.assertEqual(r.endpoint['effective'],'native_last')
                raise RuntimeError('injected')
        except RuntimeError: pass
        self.assertIs(r.base.forward,original); self.assertEqual(r.endpoint['effective'],'b1_all_logits')
        r.base.get_decoder=lambda:object()
        with self.assertRaises(PrefixError):
            with r.matched_scope(): pass


DEPS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: actual OPT/LoRA/head/Cache tests NOT_RUN')
class ModelTests(unittest.TestCase):
    def exercise(self,device):
        from test_r2b1_prefix_runtime import OfflineOPTTests
        from semcache.diagnostics.r2b35_endpoint_audit import gate
        original=OfflineOPTTests().runtime(device)
        rt=Runtime.__new__(Runtime); rt.__dict__.update(original.__dict__); rt.setup_matched(); rt.configure('off','b1')
        for p in (16,31):
            rt.prepare(32,p); g=gate(rt)
            self.assertEqual(g['output_matching'],'OUTPUT_MATCHED_VALID'); self.assertTrue(g['native_endpoint_equivalent'])
            self.assertTrue(g['axis_b']['matched']['allclose'])
            for mode,row in g['audits'].items():
                if mode.startswith('MATCHED'): self.assertEqual(row['endpoint']['head']['input'][1],1)
                self.assertEqual(row['physical']['query_tokens'],32 if mode.endswith('FULL') else 32-p)
        rt.verify_weight_hashes()
    def test_cpu_opt(self): self.exercise('cpu')
    def test_cuda_opt(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable: GPU NOT_RUN')
        self.exercise('cuda:0')


if __name__=='__main__': unittest.main()
