"""No model packages: actual tiny KV math and dynamic protocol invariants."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from semcache.edgelora.communication import digest,header,ProtocolError
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r3a_artifacts import load_matrices,read,THRESHOLDS,check_parity
from semcache.edgelora.r4a_fixture import TinyBackend
from semcache.edgelora.r4a_generation import ForwardDriver,generate,generation_policy,past_summary,validate_generated
from semcache.edgelora.r4a_session import context,positions
from semcache.edgelora.r4a_dynamic_ud import ForwardOrder


class GenerationTests(unittest.TestCase):
    def test_actual_kv_decode_matches_complete_causal_prefix_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=prepare_tiny(Path(tmp)/'base');am=read(path/'r3a_adapter_manifest.json');base=read(path/'r3a_tiny_base.json')
            model=TinyBackend(base,load_matrices(path,am,torch_backend=False))
            prompt=([1,2,3,4]*8);mask=[0,0]+[1]*30
            prefill=model.forward(prompt,mask);past_summary(prefill.past_key_values,32,32,1,4,'float32','cpu')
            cached=model.forward([5],mask+[1],prefill.past_key_values)
            complete=model.forward(prompt+[5],mask+[1])
            self.assertEqual(cached.logits[-7:],complete.logits[-7:]);self.assertEqual(cached.shape,[1,1,7])
            past_summary(cached.past_key_values,33,32,1,4,'float32','cpu')
            with self.assertRaises(ProtocolError):past_summary(cached.past_key_values,34,32,1,4,'float32','cpu')

    def test_positions_and_length_transition_32_to_1_and_wrong_mask(self):
        self.assertEqual(positions([0,0,1,1,1]),[-1,-1,0,1,2])
        first=context('R1',0,[1]*32,[1]*32,0,8)
        second=context('R1',1,[2],[1]*33,32,8)
        self.assertEqual(first['input_length'],32);self.assertEqual(second['input_length'],1)
        for args in (([2],[1]*32,32),([2,3],[1]*34,32)):
            with self.assertRaises(ProtocolError):context('R1',1,*args,8)
        with self.assertRaises(ProtocolError):positions([1,0,1])

    def test_dynamic_ud_layer_context_request_order_and_hit_first_decode(self):
        identity=dict(hidden_size=4,hidden_dtype='float32');order=ForwardOrder()
        def frame(parent,step,layer,n,past,hit=False):
            c=context(parent,step,[1]*n,[1]*(n+past),past,2,hit=hit)
            h=header('HIDDEN',f'{parent}.s{step:02d}',layer,'float32',[1,n,4],r4a_forward=c);return h
        for layer in range(32):order.accept(frame('R0',0,layer,32,0),identity);order.advance(32)
        with self.assertRaises(ProtocolError):order.accept(frame('R0',1,1,1,32),identity)
        for layer in range(32):order.accept(frame('R0',1,layer,1,32),identity);order.advance(32)
        for layer in range(32):order.accept(frame('R1',1,layer,1,32,True),identity);order.advance(32)
        order.finish();self.assertEqual(order.completed,3)
        with self.assertRaises(ProtocolError):order.accept(frame('R1',1,0,1,32,True),identity)

    def test_fixed_and_natural_eos_are_explicit_and_thresholds_unchanged(self):
        special=dict(bos_token_id=0,eos_token_id=2,pad_token_id=1)
        self.assertTrue(generation_policy(special,'fixed',8)['ignore_eos'])
        self.assertFalse(generation_policy(special,'natural-eos',8)['ignore_eos'])
        for mode,n in (('implicit',8),('fixed',0),('fixed',17)):
            with self.assertRaises(ProtocolError):generation_policy(special,mode,n)
        self.assertEqual(THRESHOLDS['max_abs'],.01);self.assertEqual(THRESHOLDS['relative_l2'],.001)
        parity=check_parity([.0546875,1.,2.],[0.,1.,2.],[1,1,3])
        self.assertFalse(parity['passed']);self.assertTrue(parity['last_token_argmax_agreement'])

    def test_fixed_ignores_eos_and_natural_stops_without_extra_decode(self):
        # Deliberate tiny UNIT head control, not an OPT reference or experiment.
        with tempfile.TemporaryDirectory() as tmp:
            path=prepare_tiny(Path(tmp)/'base');pm=read(path/'r3a_prepare_manifest.json');am=read(path/'r3a_adapter_manifest.json')
            base=read(path/'r3a_tiny_base.json');base['head']=[0.]*len(base['head'])
            tiny=TinyBackend(base,load_matrices(path,am,torch_backend=False));driver=ForwardDriver(pm['identity'],'cpu',tiny=tiny,native=True)
            prompt=dict(token_ids=[1]*32,attention_mask=[1]*32,prompt_sha256='TEST_FIXTURE_ONLY')
            special=dict(bos_token_id=0,eos_token_id=0,pad_token_id=0)
            fixed=generate(driver,prompt,generation_policy(special,'fixed',3),'UNIT_FIXED')
            natural=generate(driver,prompt,generation_policy(special,'natural-eos',3),'UNIT_NATURAL')
            self.assertEqual(fixed['generated_token_ids'],[0,0,0]);self.assertEqual(fixed['decode_forward_count'],2)
            self.assertEqual(natural['generated_token_ids'],[0]);self.assertEqual(natural['decode_forward_count'],0)
            with self.assertRaises(ProtocolError):validate_generated(generation_policy(special,'natural-eos',3),[1])
            with self.assertRaises(ProtocolError):validate_generated(generation_policy(special,'natural-eos',3),[0,1])


if __name__=='__main__':unittest.main()
