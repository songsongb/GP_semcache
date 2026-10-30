"""New B2 validator; frozen R4-A must still reject initial unverified past."""
from copy import deepcopy
import unittest

from semcache.edgelora.communication import ProtocolError,digest,header
from semcache.edgelora.r4a_session import context as frozen_context
from semcache.edgelora.r4b2_session import ForwardOrder,context,validate_context

COHORT='a'*64;KEY='b'*64;PROOF='c'*64


class ProtocolTests(unittest.TestCase):
    def drive(self,order,parent,step,target,mode,grant=None):
        p=0 if grant is None else grant['prefix_length'];inputs=target[p:] if not step else [5];old=p if not step else len(target)+step-1
        c=context(parent,step,inputs,target,old,3,mode,COHORT,grant)
        for layer in range(32):order.accept(header('HIDDEN',f'{parent}.s{step:02d}',layer,'float32',[1,len(inputs),4],r4b2_forward=c),{'hidden_size':4,'hidden_dtype':'float32'});order.advance()
        return c
    def registered(self):
        order=ForwardOrder(COHORT,32);source=[1]*64
        for step in range(3):self.drive(order,'SOURCE',step,source,'COLD_MISS')
        message=dict(action='REGISTER_PREFIX',b2_version=2,cohort_sha256=COHORT,entry_id=KEY,source_request_id='SOURCE',
            prefix_length=64,prefix_token_ids=source,proof_sha256=PROOF)
        order.control(message);return order,message
    def test_realistic_suffix_prefill_and_decode_preserve_source_and_frozen_gate(self):
        order,g=self.registered();grant={k:g[k] for k in ('entry_id','source_request_id','prefix_length','proof_sha256')}
        for target in ([1]*64+[2]*64,[1]*64+[3]*64):
            parent='A' if target[-1]==2 else 'B'
            first=self.drive(order,parent,0,target,'PARTIAL_PREFIX_HIT',grant)
            self.assertEqual(first['phase'],'SUFFIX_PREFILL');self.assertEqual(first['past_length'],64);self.assertEqual(first['input_length'],64)
            for step in (1,2):self.drive(order,parent,step,target,'PARTIAL_PREFIX_HIT',grant)
        order.finish()
        with self.assertRaises(ProtocolError):frozen_context('UNCHANGED',0,[2]*64,[1]*128,64,3)
    def test_uncompleted_unknown_evicted_source_and_malformed_context_reject(self):
        order=ForwardOrder(COHORT,32);self.drive(order,'SOURCE',0,[1]*64,'COLD_MISS')
        m=dict(action='REGISTER_PREFIX',b2_version=2,cohort_sha256=COHORT,entry_id=KEY,source_request_id='SOURCE',prefix_length=64,prefix_token_ids=[1]*64,proof_sha256=PROOF)
        with self.assertRaises(ProtocolError):order.control(m)
        order,m=self.registered();order.control(dict(action='EVICT_PREFIX',b2_version=2,cohort_sha256=COHORT,entry_id=KEY))
        grant={k:m[k] for k in ('entry_id','source_request_id','prefix_length','proof_sha256')}
        c=context('A',0,[2]*64,[1]*64+[2]*64,64,3,'PARTIAL_PREFIX_HIT',COHORT,grant)
        with self.assertRaises(ProtocolError):order.accept(header('HIDDEN','A.s00',0,'float32',[1,64,4],r4b2_forward=c),{'hidden_size':4,'hidden_dtype':'float32'})
        for field,value in (('past_length',0),('prefix_length',32),('positions',[0]*64),('attention_mask',[0]*128),('phase','DECODE'),('source_request_id',None)):
            wrong=deepcopy(c);wrong[field]=value
            with self.subTest(field=field),self.assertRaises(ProtocolError):validate_context(wrong)
    def test_wrong_layer_cohort_decode_parent_and_incomplete_shutdown_reject(self):
        order,g=self.registered();grant={k:g[k] for k in ('entry_id','source_request_id','prefix_length','proof_sha256')}
        c=context('A',0,[2]*64,[1]*64+[2]*64,64,3,'PARTIAL_PREFIX_HIT',COHORT,grant)
        for layer,dtype in ((1,'float32'),(0,'float16')):
            with self.assertRaises(ProtocolError):order.accept(header('HIDDEN','A.s00',layer,dtype,[1,64,4],r4b2_forward=c),{'hidden_size':4,'hidden_dtype':'float32'})
        self.drive(order,'A',0,[1]*64+[2]*64,'PARTIAL_PREFIX_HIT',grant)
        with self.assertRaises(ProtocolError):order.finish()
        with self.assertRaises(ProtocolError):self.drive(order,'B',1,[1]*64+[2]*64,'PARTIAL_PREFIX_HIT',grant)


if __name__=='__main__':unittest.main()
