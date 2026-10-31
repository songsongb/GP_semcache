"""Unit sources use actual TinyBackend QKV/proofs; TCP path is tested separately."""
from copy import deepcopy
import unittest

from semcache.edgelora.communication import digest
from semcache.edgelora.r5_contracts import namespace,request
from semcache.edgelora.r5_controller import Controller,RawOwner,CacheIntegrityError
from semcache.edgelora.r5_semantic_index import SemanticIndex
from semcache.edgelora.r5_workload import generation_policy
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.semantic.encoder import ControlledEncoder
from tests import test_r4b2_cache as legacy


class ControllerTests(unittest.TestCase):
    setUp=legacy.PrefixTests.setUp
    computed=legacy.PrefixTests.computed
    def manager(self,capacity):
        self.index=SemanticIndex(ControlledEncoder({'source':[0.,0.],'target':[0.,0.],'other':[0.,0.]}))
        c=RawOwner(self.pm,self.prepared_sha,'TEST_FIXTURE_ONLY','cpu',capacity,index=self.index,fixture=True,
            tokenizer={'fixture':'integer'},sources={'fixture':'unit'},contract={'scope':'TEST_FIXTURE_ONLY','output_attentions':False},observed_cache_type=tuple)
        self.ident=namespace(self.pm,c.tokenizer,'cpu','TEST_FIXTURE_ONLY',c.contract)
        self.controller=Controller(c,self.index,self.ident)
        return c

    def req(self,rid,ids,text='source',user='A'):
        raw=dict(request_id=rid,user_id=user,adapter_id='user_a',input_text=text,dataset='TEST_FIXTURE_ONLY',sample_id=rid,request_sequence_number=0)
        return request(raw,ids,generation_policy(TinyTokenizer()),self.ident,fixture=True)

    def source(self,rid='S',ids=None):
        ids=ids or [1,2,3,4];r=self.req(rid,ids);event,_=self.controller.decide(r)
        out,blocks,proof=self.computed(ids)
        return self.controller.admit(r,event['assignment'],blocks,proof,completed=True)

    def test_automatic_miss_admit_hit_same_adapter_cross_user_and_duplicate_id(self):
        g=self.source();event,found=self.controller.decide(self.req('T',[1,2,3,4,5],text='target',user='B'))
        self.assertEqual(event['outcome'],'PHYSICAL_PREFIX_HIT');self.assertEqual(found[1]['entry_id'],g['entry_id'])
        with self.assertRaisesRegex(ValueError,'duplicate'):self.controller.decide(self.req('T',[1,2,3,4,5]))

    def test_unsupported_adapter_rejected_before_semantics_or_consumption(self):
        self.source();r=self.req('BAD',[1,2,3,4,5]);r['adapter_id']='other';r['physical_identity']['adapter_id']='other'
        r.pop('request_sha256');r['request_sha256']=digest(r);before=self.index.clusterer.queries
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_ADAPTER'):self.controller.decide(r)
        self.assertEqual(self.index.clusterer.queries,before)

    def test_integrity_tensor_corruption_invalidates_and_fails_closed(self):
        grant=self.source();key=grant['entry_id'];entry=next(iter(self.cache.entries[key].pool.entries.values()))
        entry.tensors[0][1].values[0]+=1.
        with self.assertRaises(CacheIntegrityError):self.controller.decide(self.req('T',[1,2,3,4,5]))
        self.assertEqual(self.cache.bytes,0);self.assertNotIn(key,self.index.entries);self.assertIsNone(entry.tensors)

    def test_proof_corruption_invalidates_and_cannot_become_miss(self):
        grant=self.source();self.cache.proofs[grant['entry_id']]['prefix_length']+=1
        with self.assertRaises(CacheIntegrityError):self.controller.decide(self.req('T',[1,2,3,4,5]))
        self.assertTrue(any(e['kind']=='INVALID_CACHE_INTEGRITY' for e in self.cache.events))

    def test_capacity_eviction_index_synchronization_and_readmission(self):
        self.cache=self.manager(4*3*32*4*4);a=self.source()
        b=self.source('Q',[2,3,4,5]);self.assertNotIn(a['entry_id'],self.cache.entries)
        self.assertNotIn(a['entry_id'],self.index.entries)
        event,found=self.controller.decide(self.req('OLD',[1,2,3,4,5]))
        self.assertIsNone(found)
        a2=self.source('AGAIN');self.assertIn(a2['entry_id'],self.cache.entries)

    def test_source_admission_denied_and_incomplete_never_inserted(self):
        self.cache=self.manager(1);r=self.req('S',[1,2,3,4]);event,_=self.controller.decide(r);_,blocks,proof=self.computed(r['token_ids'])
        self.assertIsNone(self.controller.admit(r,event['assignment'],blocks,proof,completed=True));self.assertEqual(self.cache.bytes,0)
        with self.assertRaisesRegex(ValueError,'incomplete'):self.controller.admit(r,event['assignment'],blocks,proof,completed=False)

    def test_longest_selected_only_inside_semantic_candidates(self):
        self.source('SHORT',[1,2,3]);long=self.source('LONG',[1,2,3,4]);event,found=self.controller.decide(self.req('TARGET',[1,2,3,4,5]))
        self.assertEqual(event['selected_entry_id'],long['entry_id']);self.assertEqual(found[1]['prefix_length'],4)
