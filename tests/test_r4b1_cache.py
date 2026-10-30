"""Real compact tensor owners and strict multi-entry policy controls, CPU only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.edgelora.communication import ProtocolError,digest
from semcache.edgelora.r3a_artifacts import read,sha
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r3b_cache import ExactRawCache
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4a_generation import generation_policy
from semcache.edgelora.r4b1_cache import MultiPromptCache
from semcache.edgelora.r4b1_evidence import entry_bytes
from semcache.edgelora.r4b1_generation import generate


class MultiEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        base=prepare_tiny(Path(self.temp.name)/'base');self.pm=read(base/'r3a_prepare_manifest.json');self.base_sha=sha(base/'r3a_prepare_manifest.json')
        self.size=entry_bytes(self.pm['identity']);self.cache=self.manager(2*self.size)
        self.a=dict(token_ids=[1,2,3,4]*8,attention_mask=[1]*32)
        self.b=dict(token_ids=[2,3,4,1]*8,attention_mask=[1]*32)
        self.c=dict(token_ids=[3,4,1,2]*8,attention_mask=[1]*32)
    def manager(self,capacity):
        return MultiPromptCache(self.pm,self.base_sha,'TEST_FIXTURE_ONLY','cpu',capacity,fixture=True,
            tokenizer=dict(vocab_sha256='TEST_FIXTURE_ONLY',special_tokens=dict(pad_token_id=0)),sources={'unit':'TEST_FIXTURE_ONLY'})
    def blocks(self,n=32):return {l:tuple(ArrayTensor([float(l+r)]*(n*4),[1,n,4]) for r in range(3)) for l in range(32)}
    def cold(self,prompt,rid):
        self.cache.request_id=rid;view=self.cache.bind(prompt);view.reset('targeted cold invalidation')
        self.assertIsNone(view.lookup(view.context));self.assertTrue(view.insert(view.context,self.blocks(len(prompt['token_ids'])),rid,completed=True))
        return view

    def test_a_cold_b_cold_a_warm_retains_two_actual_owned_entries(self):
        a=self.cold(self.a,'A_COLD');self.cold(self.b,'B_COLD');entry=a.lookup(a.context)
        self.assertIsNotNone(entry);self.assertIsInstance(entry,CacheEntry)
        self.assertEqual(entry.qkv_metadata['source_request_id'],'A_COLD');self.assertTrue(entry.qkv_metadata['complete_request'])
        self.assertEqual(len(self.cache.entries),2);self.assertEqual(self.cache.bytes,2*self.size)
        for cache in self.cache.entries.values():
            self.assertIsInstance(cache,ExactRawCache);self.assertIsInstance(cache.pool,GlobalCache)
        a.reused(entry);self.assertEqual(entry.frequency,1)
        self.assertFalse(any(e['kind']=='EVICT' for e in self.cache.events))
        # Original caller tensors are cloned: mutations cannot alter residents.
        self.cache.request_id='A_REPLACE';a.reset('replace only A');blocks=self.blocks()
        self.assertTrue(a.insert(a.context,blocks,'A_REPLACE',completed=True));blocks[0][0].values[0]+=99
        self.assertIsNotNone(a.lookup(a.context));self.assertIsNotNone(self.cache.bind(self.b).lookup(self.cache.context_for(self.b)))

    def test_byte_pressure_evicts_actual_lru_and_never_authorizes_evicted_hit(self):
        a=self.cold(self.a,'A');b=self.cold(self.b,'B');old_b=b.lookup(b.context);self.assertIsNotNone(a.lookup(a.context))
        self.cold(self.c,'C')  # Successful access of A makes B the victim.
        self.assertIsNone(b.lookup(b.context));self.assertIsNotNone(a.lookup(a.context))
        evicted=[e for e in self.cache.events if e['kind']=='EVICT']
        self.assertEqual(len(evicted),1);self.assertEqual(evicted[0]['context_sha256'],b.context_digest)
        self.assertEqual(evicted[0]['released_bytes'],self.size)
        self.assertIsNone(old_b.tensors)  # Even a retained entry handle cannot pin evicted storage.
        self.assertTrue(all(e['occupancy_bytes']<=2*self.size for e in self.cache.events))
        with self.assertRaises(ProtocolError):b.reused(CacheEntry(0,tuple(self.b['token_ids']),(0,32),self.size))

    def test_all_identity_mismatches_partial_prefix_and_incomplete_provenance_reject(self):
        view=self.cold(self.a,'A')
        for field in ('token_ids','positions','attention_mask','adapter','adapter_matrix_sha256','identity','tokenizer','storage_device','execution_sources','prepared_manifest_sha256','gpu_uuid'):
            changed=deepcopy(view.context)
            if field in ('token_ids','positions','attention_mask'):changed[field][0]+=1
            elif field=='identity':changed[field]['base_weight_sha256']='CORRUPTED'
            elif field=='tokenizer':changed[field]['vocab_sha256']='CORRUPTED'
            elif field=='execution_sources':changed[field]['unit']='CORRUPTED'
            else:changed[field]='CORRUPTED'
            with self.subTest(field=field):
                self.assertIsNone(view.lookup(changed))
                with self.assertRaises(ProtocolError):view.insert(changed,self.blocks(),'A',completed=True)
        short=dict(token_ids=self.a['token_ids'][:16],attention_mask=[1]*16)
        self.assertIsNone(self.cache.bind(short).lookup(self.cache.context_for(short)))
        padded=dict(self.a,attention_mask=[0]+[1]*31)
        self.assertIsNone(self.cache.bind(padded).lookup(self.cache.context_for(padded)))
        with self.assertRaises(ProtocolError):self.cache.bind(dict(self.a,positions=[99]*32))
        with self.assertRaises(ProtocolError):self.cache.bind(dict(self.a,prompt_sha256='CORRUPTED'))
        with self.assertRaises(ProtocolError):view.insert(view.context,self.blocks(),'A',completed=False)
        entry=view.lookup(view.context);entry.qkv_metadata['complete_request']=False
        self.assertIsNone(view.lookup(view.context));self.assertEqual(self.cache.bytes,0)

    def test_wrong_source_tensor_shape_dtype_device_and_content_fail_closed(self):
        for corruption in ('source','shape','dtype','device','content'):
            view=self.cold(self.a,'A');entry=view.lookup(view.context)
            if corruption=='source':entry.qkv_metadata['source_request_id']='FOREIGN'
            elif corruption=='shape':entry.tensors[0][0].shape=(32,1,4)
            elif corruption=='dtype':entry.tensors[0][0].dtype='float16'
            elif corruption=='device':entry.tensors[0][0].device='foreign-device'
            else:entry.tensors[0][0].values[0]+=1
            with self.subTest(corruption=corruption):self.assertIsNone(view.lookup(view.context))
            self.assertEqual(self.cache.bytes,0)

    def test_oversize_denied_without_evicting_and_foreign_source_not_admitted(self):
        a=self.cold(self.a,'A');self.cold(self.b,'B');before=self.cache.snapshot()
        p=dict(token_ids=[1]*128,attention_mask=[1]*128);view=self.cache.bind(p);self.cache.request_id='BIG'
        self.assertFalse(view.insert(view.context,self.blocks(128),'BIG',completed=True));self.assertEqual(before,self.cache.snapshot())
        with self.assertRaises(ProtocolError):view.insert(view.context,self.blocks(128),'FOREIGN',completed=True)
        self.assertIsNotNone(a.lookup(a.context))
        with self.assertRaises(ProtocolError):self.manager(0)
        with self.assertRaises(ProtocolError):MultiPromptCache(self.pm,self.base_sha,'TEST','cpu',self.size,tokenizer={},sources={})

    def test_requested_warm_miss_fails_before_any_forward(self):
        driver=SimpleNamespace(native=False,synchronize=lambda:None)
        policy=generation_policy(dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),'fixed',3)
        p=dict(self.a,sample_id='A',prompt_sha256=digest(dict(tokens=self.a['token_ids'],mask=self.a['attention_mask'])))
        with self.assertRaisesRegex(ProtocolError,'invalid HIT/MISS; no fallback counted as HIT'):
            generate(driver,p,policy,'REQUESTED_WARM',self.cache,'WARM_HIT')
        self.assertEqual([e['kind'] for e in self.cache.events],['MISS'])
        self.assertEqual(self.cache.bytes,0)


if __name__=='__main__':unittest.main()
