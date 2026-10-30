from copy import deepcopy
import unittest

from semcache.edgelora.communication import digest
from semcache.edgelora.r4c_session import VERSION,ForwardOrder,context,validate_context
from semcache.edgelora.r4c_workload import policy,termination

SPECIAL=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0)


class SessionTests(unittest.TestCase):
    def setUp(self):self.cohort='a'*64;self.identity=dict(hidden_size=4,hidden_dtype='float32');self.p=policy(SPECIAL,'fixed',32)
    def frame(self,c,l=0):return dict(kind='HIDDEN',request_id=f'{c["parent_request_id"]}.s{c["pass_index"]:02d}',layer_id=l,shape=[1,c['input_length'],4],dtype='float32',metadata=dict(r4c_forward=c))
    def accept(self,order,c):
        for l in range(2):order.accept(self.frame(c,l),self.identity);order.advance()
    def complete(self,order,rid,ids,reason):order.control(dict(version=VERSION,cohort_sha256=self.cohort,action='COMPLETE_REQUEST',parent_request_id=rid,generated_token_ids=ids,termination=reason))

    def test_long_input_and_generation_have_new_bounds(self):
        c=context('P',0,[1]*256,[1]*256,0,self.p,'NO_REUSE',self.cohort);self.assertEqual(c['target_length'],256)
        c=context('P',31,[2],[1]*256,286,self.p,'NO_REUSE',self.cohort);self.assertEqual(c['positions'],[286])
        with self.assertRaises(ValueError):context('P',0,[1]*512,[1]*512,0,self.p,'NO_REUSE',self.cohort,context_limit=530)

    def test_eos_verified_completion_and_no_decode_after_eos(self):
        p=policy(SPECIAL,'quality',64);order=ForwardOrder(self.cohort,2,2048)
        self.accept(order,context('P',0,[1]*256,[1]*256,0,p,'NO_REUSE',self.cohort))
        with self.assertRaises(ValueError):self.accept(order,context('Q',0,[1]*256,[1]*256,0,p,'NO_REUSE',self.cohort))
        self.complete(order,'P',[6],'EOS');order.finish()
        with self.assertRaises(ValueError):self.accept(order,context('P',1,[6],[1]*256,256,p,'NO_REUSE',self.cohort))

    def test_cold_completion_registration_partial_prefill_and_decode(self):
        order=ForwardOrder(self.cohort,2,2048);p=policy(SPECIAL,'fixed',1)
        self.accept(order,context('SOURCE',0,[1]*128,[1]*128,0,p,'COLD_MISS',self.cohort))
        register=dict(version=VERSION,cohort_sha256=self.cohort,action='REGISTER_PREFIX',entry_id='b'*64,source_request_id='SOURCE',prefix_length=128,prefix_token_ids=[1]*128,proof_sha256='c'*64)
        with self.assertRaises(ValueError):order.control(register)
        self.complete(order,'SOURCE',[3],'LENGTH_CAP');order.control(register)
        grant={k:register[k] for k in ('entry_id','source_request_id','prefix_length','proof_sha256')};target=[1]*128+[2]*128;p=policy(SPECIAL,'fixed',3)
        self.accept(order,context('TARGET',0,target[128:],target,128,p,'PARTIAL_PREFIX_HIT',self.cohort,grant))
        self.accept(order,context('TARGET',1,[3],target,256,p,'PARTIAL_PREFIX_HIT',self.cohort,grant))
        self.accept(order,context('TARGET',2,[4],target,257,p,'PARTIAL_PREFIX_HIT',self.cohort,grant))
        with self.assertRaises(ValueError):self.complete(order,'TARGET',[2,4,5],'LENGTH_CAP')
        self.complete(order,'TARGET',[3,4,5],'LENGTH_CAP');order.finish();self.assertEqual(order.completed,4)

    def test_invalid_suffix_mask_position_grant_reject(self):
        c=context('P',0,[1]*256,[1]*256,0,self.p,'NO_REUSE',self.cohort)
        for key,value in (('positions',[1]*256),('past_length',128),('attention_mask',[0]*256),('input_token_ids',[2]*256)):
            bad=deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_context(bad)
        self.assertEqual(termination(policy(SPECIAL,'fixed',1),[6]),'LENGTH_CAP')
        self.assertEqual(termination(policy(SPECIAL,'quality',32),[6]),'EOS')
