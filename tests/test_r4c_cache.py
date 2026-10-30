from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from semcache.edgelora.r4c_cache import PrefixCache
from semcache.edgelora.r4c_evidence import Evidence,fresh
from tests import test_r4b2_cache as legacy


class CacheTests(unittest.TestCase):
    setUp=legacy.PrefixTests.setUp
    prompt=legacy.PrefixTests.prompt
    computed=legacy.PrefixTests.computed
    admit=legacy.PrefixTests.admit

    def manager(self,capacity):
        return PrefixCache(self.pm,self.prepared_sha,'TEST_FIXTURE_ONLY','cpu',capacity,fixture=True,
            tokenizer={'fixture':'integer'},sources={'fixture':'unit'},contract={'scope':'TEST_FIXTURE_ONLY','output_attentions':False},observed_cache_type=tuple)

    def test_variable_prefix128_long_target_and_two_distinct_suffixes(self):
        p=self.prefix*4;grant=self.admit(p,'P128')
        for suffix in ([2,3]*64,[4,5]*64):
            target=p+suffix;entry,g,past=self.cache.find(self.prompt(target),'HIT',expected_length=128)
            self.assertEqual(grant['entry_id'],g['entry_id']);self.assertEqual(g['source_request_id'],'P128')
            out=self.model.forward(suffix,[1]*256,past);full=self.model.forward(target,[1]*256)
            self.assertEqual(out.logits,full.logits[-len(suffix)*self.model.vocab:])
            for t in (2,4,3):
                target.append(t);out=self.model.forward([t],[1]*len(target),out.past_key_values);full=self.model.forward(target,[1]*len(target))
                self.assertEqual(out.logits,full.logits[-self.model.vocab:])
            self.model.forward([5]*180,[1]*180)
        self.assertEqual(self.cache.bytes,3*32*128*4*4)

    def test_lru_capacity_and_no_hit_after_eviction(self):
        self.cache=self.manager(3*32*32*4*4)
        controls=[];self.cache.control=controls.append
        a=self.admit(self.prefix,'A');entry=next(iter(self.cache.entries[a['entry_id']].pool.entries.values()))
        self.admit([2,3,4,5]*8,'B');self.assertIsNone(entry.tensors)
        with self.assertRaisesRegex(ValueError,'no validated exact prefix'):self.cache.find(self.prompt(self.prefix+[5]*128),'EVICTED')
        self.assertTrue(any(e['kind']=='EVICT' for e in self.cache.events))
        _,blocks,proof=self.computed(self.prefix*4)
        self.assertIsNone(self.cache.admit(self.prompt(self.prefix*4),blocks,proof,'OVERSIZE',completed=True))
        from semcache.edgelora.r4c_session import VERSION
        self.assertEqual([c['action'] for c in controls],['REGISTER_PREFIX','EVICT_PREFIX','REGISTER_PREFIX'])
        self.assertTrue(all(c['version']==VERSION and 'b2_version' not in c for c in controls))

    def test_negative_token_position_mask_identity_and_corruption(self):
        self.admit(self.prefix,'P');target=self.prompt(self.prefix+[5]*224)
        for changed in (dict(target,token_ids=[9]+target['token_ids'][1:]),dict(target,positions=[99]*256),dict(target,attention_mask=[0]+[1]*255),dict(target,prompt_sha256='0'*64)):
            with self.assertRaises(ValueError):self.cache.find(changed,'BAD')
        for key in ('adapter','model','base_weight_sha256','adapter_matrix_sha256'):
            saved=self.cache._pm['identity'][key];self.cache._pm['identity'][key]='OTHER'
            with self.assertRaisesRegex(ValueError,'namespace changed'):self.cache.find(target,'BAD_NAMESPACE')
            self.cache._pm['identity'][key]=saved
        key=next(iter(self.cache.entries));self.cache.proofs[key]['prefix_length']+=1
        with self.assertRaisesRegex(ValueError,'corrupted prefix proof'):self.cache.find(target,'BAD_PROOF')

    def test_real_byte_evidence_budget_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=fresh(root/'out');(out/'existing').write_bytes(b'x'*30);e=Evidence(out,budget_bytes=32)
            with self.assertRaisesRegex(ValueError,'EVIDENCE_BUDGET_EXCEEDED'):e.blob(b'123','next')
            self.assertFalse((out/'next').exists())
            with self.assertRaisesRegex(ValueError,'nonempty'):fresh(out)

    def test_digest_only_profile_and_strict_tensor_serialization(self):
        with tempfile.TemporaryDirectory() as d:
            _,blocks,_=self.computed(self.prefix);e=Evidence(Path(d),budget_bytes=2**20)
            e.profile='SCALABLE_EVALUATION';s=e.save(blocks[0][0],'compact');self.assertNotIn('file',s);self.assertEqual(e.bytes,0)
            e.profile='STRICT_AUDIT';s=e.save(blocks[0][0],'strict');self.assertIn('file',s);self.assertEqual((Path(d)/s['file']).stat().st_size,32*4*4)

    def test_corrupt_owned_tensor_and_incomplete_source_fail_closed(self):
        _,blocks,proof=self.computed(self.prefix)
        with self.assertRaises(ValueError):self.cache.admit(self.prompt(self.prefix),blocks,proof,'INCOMPLETE',completed=False)
        grant=self.cache.admit(self.prompt(self.prefix),blocks,proof,'SOURCE',completed=True)
        entry=next(iter(self.cache.entries[grant['entry_id']].pool.entries.values()));entry.tensors[0][1].values[0]+=1.
        with self.assertRaises(ValueError):self.cache.find(self.prompt(self.prefix+[5]*224),'CORRUPT')
        self.assertEqual(self.cache.bytes,0)

    def test_nonfinite_actual_logits_preserve_diagnostics_even_when_final_token_finite(self):
        import json
        from types import SimpleNamespace
        from semcache.edgelora.r4c_generation import QualityDriver
        with tempfile.TemporaryDirectory() as d:
            driver=QualityDriver(self.pm['identity'],'cpu',tiny=self.model);driver.configure(Path(d))
            record=dict(context=dict(parent_request_id='FAIL',pass_index=0,input_length=2))
            out=SimpleNamespace(shape=[1,2,self.model.vocab],logits=[float('nan')]+[0.]*(2*self.model.vocab-1))
            with self.assertRaisesRegex(ValueError,'nonfinite actual model logits'):driver.last_logits(out,sample_id='SAMPLE',record=record,mask=[1,1])
            diagnostic=json.loads(driver.diagnostics_path.read_text());self.assertEqual(diagnostic['all_logits']['nonfinite'],1)
            self.assertEqual(diagnostic['last_position']['nonfinite'],0);self.assertEqual(diagnostic['valid_positions']['nonfinite'],1)

    def test_evidence_accounts_owned_writes_and_rechecks_real_trace_bytes(self):
        from unittest.mock import patch
        from semcache.edgelora.r4c_evidence import directory_bytes
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);e=Evidence(out,budget_bytes=1024)
            with patch('semcache.edgelora.r4c_evidence.directory_bytes',wraps=directory_bytes) as scans:
                e.check()
                for i in range(3):e.blob(b'x'*100,f'{i}.bin')
                self.assertEqual(scans.call_count,1)
                (out/'trace.jsonl').write_bytes(b'x'*800)
                with self.assertRaisesRegex(ValueError,'EVIDENCE_BUDGET_EXCEEDED'):e.check()
