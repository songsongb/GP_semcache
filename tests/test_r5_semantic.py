from copy import deepcopy
import unittest

from semcache.semantic.encoder import ControlledEncoder
from semcache.edgelora.r5_semantic_index import SemanticIndex,eligibility


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.index=SemanticIndex(ControlledEncoder({'a':[0.,0.],'b':[10.,10.]}),clusters=2,warmup_texts=['a','b'])
        self.meta=dict(entry_id='entry',source_request_id='source',source_user='A',token_ids=[1,2,3,4],
            attention_mask=[1]*4,positions=list(range(4)),cluster_id=0,completed=True,physical_identity=dict(adapter_id='user_a',matrix='a'))
        self.index.add('entry',self.meta)

    def test_cluster_exact_window_discovery_and_distance_updates(self):
        self.assertEqual(len(self.index.discover(0,[1,2,3,4,5])),1)
        assignment=self.index.assign('a');self.assertEqual(assignment['cluster_id'],0)
        self.assertTrue(assignment['centroid_update_applied']);self.assertIn('nearest_centroid_distance_pre_update',assignment)

    def test_different_cluster_empty_and_similar_token_mismatch(self):
        self.assertFalse(self.index.discover(1,[1,2,3,4,5]))
        self.assertFalse(self.index.discover(0,[1,9,3,8,5]))
        self.assertFalse(self.index.discover(0,[]))

    def test_middle_window_is_candidate_not_prefix_authorization(self):
        ids=[9,1,2,3,4,5];self.assertTrue(self.index.discover(0,ids))
        req=dict(token_ids=ids,attention_mask=[1]*6,positions=list(range(6)),physical_identity=self.meta['physical_identity'])
        self.assertEqual(eligibility(self.meta,req,self.index.entries),'EXACT_PREFIX_TOKEN_MISMATCH')

    def test_adapter_fingerprints_and_position_mask_rejections(self):
        req=dict(token_ids=[1,2,3,4,5],attention_mask=[1]*5,positions=list(range(5)),physical_identity=dict(adapter_id='user_a',matrix='DIFFERENT'))
        self.assertEqual(eligibility(self.meta,req,self.index.entries),'INCOMPATIBLE_PHYSICAL_IDENTITY')
        req['physical_identity']=self.meta['physical_identity'];req['positions'][0]=1
        self.assertEqual(eligibility(self.meta,req,self.index.entries),'POSITION_OR_MASK_MISMATCH')

    def test_duplicate_metadata_rejected_and_repeated_windows_invalidated_once(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):self.index.add('entry',self.meta)
        m=dict(self.meta,entry_id='repeat',token_ids=[1,2,3]*4);self.index.add('repeat',m)
        self.index.invalidate('repeat');self.index.invalidate('repeat')
        self.assertEqual(len(self.index.discover(0,[1,2,3,4,5])),1)

    def test_stale_candidate_no_hit_and_synchronous_index_removal(self):
        req=dict(token_ids=[1,2,3,4,5],attention_mask=[1]*5,positions=list(range(5)),physical_identity=self.meta['physical_identity'])
        self.assertEqual(eligibility(self.meta,req,{}),'STALE_EVICTED_ENTRY')
        self.index.invalidate('entry');self.assertFalse(self.index.discover(0,req['token_ids']))

    def test_longest_then_stable_key_and_bounded_dedup(self):
        m=dict(self.meta,entry_id='long',token_ids=[1,2,3,4,5]);self.index.add('long',m)
        self.assertEqual([c['metadata']['entry_id'] for c in self.index.discover(0,[1,2,3,4,5,6])],['long','entry'])

    def test_buffered_cluster_policy_is_reused(self):
        i=SemanticIndex(ControlledEncoder({'a':[1.,2.]}),update_mode='buffered',update_interval=2)
        self.assertFalse(i.assign('a')['centroid_update_applied']);self.assertTrue(i.assign('a')['centroid_update_applied'])
