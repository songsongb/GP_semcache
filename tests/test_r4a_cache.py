"""Actual SemCache cache ownership/provenance with explicit CPU fixture tensors."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from semcache.cache.global_cache import GlobalCache
from semcache.cache.cache_entry import CacheEntry
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.edgelora.communication import digest,ProtocolError
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r3a_artifacts import read,sha
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4a_cache import PromptCache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=prepare_tiny(Path(self.tmp.name)/'base')
        pm=read(self.base/'r3a_prepare_manifest.json')
        self.router=PromptCache(pm,sha(self.base/'r3a_prepare_manifest.json'),'TEST_FIXTURE_ONLY','cpu',fixture=True)
        self.prompt=dict(token_ids=[1,2,3,4]*8,attention_mask=[1]*32)
    def blocks(self,n):return {l:tuple(ArrayTensor([.1+r]*(n*4),[1,n,4]) for r in range(3)) for l in range(32)}
    def test_real_semcache_owned_total_qkv_and_decode_not_stored(self):
        cache=self.router.bind(self.prompt);blocks=self.blocks(32)
        self.assertIsInstance(cache.pool,GlobalCache);self.assertIsInstance(cache.matcher,ExactTokenMatcher)
        self.assertTrue(cache.insert(cache.context,blocks,'COLD_FULL_GENERATION',completed=True))
        entry=cache.lookup(cache.context);self.assertIsInstance(entry,CacheEntry)
        blocks[0][0].values[0]=9;self.assertIsNotNone(cache.lookup(cache.context))
        self.assertEqual(entry.qkv_metadata['component_scope'],'total_qkv');self.assertEqual(len(entry.tensors),32)
        entry.tensors[0][0].values[0]=8;self.assertIsNone(cache.lookup(cache.context))
    def test_varying_prompt_context_mask_positions_and_adapter_cannot_authorize_hit(self):
        cache=self.router.bind(self.prompt);cache.insert(cache.context,self.blocks(32),'SOURCE',completed=True)
        for field in ('adapter','positions','attention_mask','context'):
            changed=deepcopy(cache.context)
            if field=='adapter':changed[field]='NOT_A_REAL_EXECUTED_ADAPTER'
            elif field=='positions':changed[field][0]=99
            elif field=='attention_mask':changed[field][0]=0
            else:changed[field]['prefix_token_ids']=[123]
            self.assertIsNone(cache.lookup(changed))
        other=dict(token_ids=[2]*128,attention_mask=[1]*128)
        active=self.router.bind(other);self.assertEqual(active.bytes,0);self.assertIsNone(active.lookup(active.context))
        self.assertTrue(any(e['kind']=='NAMESPACE_SWITCH_EVICTION' for e in self.router.events))
    def test_partial_or_uncompleted_entry_and_wrong_shape_rejected(self):
        cache=self.router.bind(self.prompt)
        with self.assertRaises(ProtocolError):cache.insert(cache.context,self.blocks(32),'FAILED',completed=False)
        with self.assertRaises(ProtocolError):cache.insert(cache.context,{0:self.blocks(32)[0]},'PARTIAL',completed=True)
        with self.assertRaises(ProtocolError):cache.insert(cache.context,self.blocks(1),'DECODE_NOT_PROMPT',completed=True)


if __name__=='__main__':unittest.main()
