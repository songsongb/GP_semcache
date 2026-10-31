"""I5 controller admission, isolation, no-semantic A, shadow B and long-prefix ownership."""
from copy import deepcopy
import unittest
from unittest.mock import Mock
from semcache.edgelora.communication import digest
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4b2_kv import attention_kv,verify_raw_kv
from semcache.edgelora.r5_i5_cache import I5Owner
from semcache.edgelora.r5_i5_controller import I5Controller
from semcache.edgelora.r5_i5_contracts import request
from semcache.edgelora.r5_multi_artifacts import runtime_view,physical
from semcache.edgelora.r5_semantic_index import SemanticIndex
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_workload import generation_policy
from semcache.semantic.encoder import ControlledEncoder
import test_r5_multi as legacy


class Controller(unittest.TestCase):
    setUpClass=classmethod(legacy.MultiTests.setUpClass.__func__)
    tearDownClass=classmethod(legacy.MultiTests.tearDownClass.__func__)
    computed=legacy.MultiTests.computed
    def manager(self,condition='SEMCACHE_INTEGRATED',capacity=6144):
        spec=self.adapters['Adapter_A'];view=runtime_view(self.pm,spec)
        ident,contract=physical(view,self.manifest['tokenizer'],'cpu','TEST_FIXTURE_ONLY',{'scope':'TEST_FIXTURE_ONLY','output_attentions':False},spec,self.registry)
        index=SemanticIndex(ControlledEncoder({t:[1.,0.] for t in ('source','target','other')}))
        owner=I5Owner(view,'prepare','TEST_FIXTURE_ONLY','cpu',capacity,index=index,tokenizer=self.manifest['tokenizer'],sources={'fixture':'unit'},
            fixture=True,contract=contract,observed_cache_type=tuple,spec=spec,registry=self.registry)
        return owner,I5Controller(owner,index,ident,condition=condition,registry=self.registry),ident
    def req(self,rid,ids,ident,text='source',user='User_01'):
        return request(dict(request_id=rid,user_id=user,adapter_id=ident['adapter_id'],input_text=text,dataset='coqa',sample_id=rid,request_sequence_number=0),
            ids,generation_policy(TinyTokenizer(),budget=2),ident,fixture=True)
    def source(self,o,c,i,rid='S',ids=None):
        ids=ids or [1,2,3,4];req=self.req(rid,ids,i);e,_=c.decide(req);blocks,proof=self.computed(ids)
        return c.admit(req,e['assignment'],blocks,proof,completed=True)
    def test_a_avoids_semantic_computation_even_with_encoder_spy(self):
        o,c,i=self.manager('FULL_RECOMPUTE');c.index.assign=Mock(side_effect=AssertionError('semantic called'))
        e,found=c.decide(self.req('A',[1,2,3],i));self.assertIsNone(found);c.index.assign.assert_not_called()
        self.assertIsNone(c.admit(self.req('A',[1,2,3],i),e['assignment'],None,None,completed=True))
    def test_b_shadow_populates_discovers_evicts_without_physical_reads(self):
        o,c,i=self.manager('SEMANTIC_NO_REUSE');self.source(o,c,i)
        o.selected=Mock(side_effect=AssertionError('physical tensors accessed'))
        e,found=c.decide(self.req('T',[1,2,3,4,5],i,'target',user='User_02'))
        self.assertIsNone(found);self.assertTrue(e['virtual_selected_entry_id']);o.selected.assert_not_called()
        self.source(o,c,i,'X',[5,4,3,2]);self.assertEqual(len(c.shadow),1);self.assertEqual(o.bytes,0)
        e,_=c.decide(self.req('T2',[1,2,3,4,5],i,'target'));self.assertIsNone(e['virtual_selected_entry_id'])
    def test_c_eviction_and_cross_adapter_rejection(self):
        o,c,i=self.manager();g=self.source(o,c,i);self.source(o,c,i,'X',[5,4,3,2])
        self.assertNotIn(g['entry_id'],o.entries);self.assertNotIn(g['entry_id'],c.index.entries)
        before=c.index.clusterer.queries
        with self.assertRaises(ValueError):c.decide(self.req('BAD',[1,2,3],i,user='User_03'))
        self.assertEqual(c.index.clusterer.queries,before)
    def test_long_1024_token_physical_owner_and_nonempty_suffix(self):
        o,c,i=self.manager(capacity=1024*3*32*4*4)
        # Explicit fixture tensor proof; this test makes no model/GPU execution claim.
        ids=[1]*1024;blocks={l:tuple(ArrayTensor([.01*(l+1)]*(1024*4),[1,1024,4]) for _ in 'qkv') for l in range(32)}
        pairs=attention_kv(blocks,1,4);proof=verify_raw_kv(blocks,tuple(pairs),1,runtime_view(self.pm,self.adapters['Adapter_A'])['identity'],'cpu')
        req=self.req('LONG',ids,i);e,_=c.decide(req);g=c.admit(req,e['assignment'],blocks,proof,completed=True)
        self.assertEqual(o.bytes,1024*3*32*4*4)
        e,found=c.decide(self.req('TARGET',ids+[2],i,'target',user='User_02'))
        self.assertEqual(found[1]['prefix_length'],1024);self.assertEqual(found[1]['entry_id'],g['entry_id'])
    def test_corrupt_proof_fails_closed(self):
        o,c,i=self.manager();g=self.source(o,c,i);o.proofs[g['entry_id']]['prefix_length']=3
        with self.assertRaises(ValueError):c.decide(self.req('T',[1,2,3,4,5],i,'target'))
        self.assertFalse(o.entries);self.assertFalse(c.index.entries)
    def test_incomplete_source_denied(self):
        o,c,i=self.manager();req=self.req('S',[1,2,3],i);e,_=c.decide(req)
        with self.assertRaises(ValueError):c.admit(req,e['assignment'],None,None,completed=False)


if __name__=='__main__':unittest.main()
