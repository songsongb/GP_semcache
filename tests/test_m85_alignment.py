"""M8.5 regression: no model construction, execution, or downloads.

Run with unittest (stdlib). Optional tests use only literal torch tensors.
"""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.util
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.cache.attention_impact import make_impact_reducer
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.metric_manager import CacheMetricManager
from semcache.metrics.alignment import add_alignment_arguments, validate_alignment_arguments
from semcache.metrics.inference import record_from_engine
from semcache.metrics.m8 import aggregate_raw, raw_record, reuse_delta_summary
from semcache.semantic.addressing import PoolBlockAddress, SourceOccurrence, BlockDescriptor
from semcache.semantic.encoder import ControlledEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.subsequence import Subsequence
from semcache.semcache_engine import SemCacheEngine


class StubTensor:
    """Only plumbing for the engine control-flow test; never computes inference."""
    device, dtype, is_cuda = 'cpu', 'float32', False

    def detach(self):
        return self

    def cpu(self):
        return self

    def element_size(self):
        return 4


class PrefillPolicyTest(unittest.TestCase):
    def setUp(self):
        weight = StubTensor()
        module = SimpleNamespace(in_features=4, r={'u': 2},
            get_base_layer=lambda: SimpleNamespace(weight=weight),
            lora_B={'u': SimpleNamespace(weight=weight)})
        model = SimpleNamespace(parameters=lambda: iter([weight]), set_adapter=lambda _: None,
                                eval=lambda: None)
        # Callable stub, not a torch or Transformers model.
        class ForwardStub:
            parameters, set_adapter, eval = (staticmethod(model.parameters),
                staticmethod(model.set_adapter), staticmethod(model.eval))
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=StubTensor(), attentions=('literal_stub',))
        clusterer = IntentClusterer(1, update_mode='buffered', update_interval=100)
        clusterer.initialize([[0.]])
        self.cache = GlobalCache(100)
        self.entry = CacheEntry(0, (1, 2, 3), (0, 3), 10, impact=5.)
        self.other = CacheEntry(1, (4, 5, 6), (0, 3), 10, impact=9.)
        self.assertTrue(self.cache.insert(self.entry))
        self.assertTrue(self.cache.insert(self.other))
        self.engine = SemCacheEngine(ForwardStub(), lambda _: {'input_ids': [1, 2, 3]},
            SimpleNamespace(layers=[0], projection_module=lambda *_: module),
            ControlledEncoder({'q': [0.]}), clusterer, self.cache)
        self.forward_hits = []

    def query(self, mode):
        @contextmanager
        def projection_stub(adapter, user, hits, n, **kwargs):
            self.forward_hits.append(list(hits))
            yield SimpleNamespace(records={(0, 'q'): {'reused_projection_rows': 3 if hits else 0}},
                                  projections={}, cuda_event_pairs=[])
        torch_stub = SimpleNamespace(tensor=lambda *a, **k: StubTensor(),
            ones=lambda *a, **k: None, bool=bool, inference_mode=nullcontext)
        with patch.dict(sys.modules, {'torch': torch_stub}), \
             patch('semcache.semcache_engine.mixed_projection_path', projection_stub), \
             patch('semcache.semcache_engine.actual_attention_impact', return_value=2.):
            return self.engine.query('q', 'u', 'test', execution_mode=mode)

    def test_lookup_only_preserves_all_reuse_state(self):
        before = (self.entry.frequency, self.entry.last_access, self.entry.updated_at, self.entry.impact)
        for _ in range(3):
            result = self.query('SEMCACHE_LOOKUP_NO_REUSE')
            self.assertEqual(before, (self.entry.frequency, self.entry.last_access,
                                     self.entry.updated_at, self.entry.impact))
            row = result['summary']
            self.assertEqual(row['executed_nonoverlap_hits'], 0)
            self.assertEqual(row['accepted_nonoverlap_hits'], 1)
            self.assertEqual(row['logical_reusable_token_count'], 3)
            self.assertEqual(row['reuse_block_provenance'], [])
            for field in ('reused_unique_token_count', 'reused_projection_rows',
                          'paper_estimated_base_flops_saved', 'paper_estimated_lora_flops_saved',
                          'paper_estimated_comm_elements_saved', 'paper_estimated_comm_bytes_saved'):
                self.assertEqual(row[field], 0)
            self.assertFalse(row['projection_skip_used'])
            self.assertFalse({'CHU', 'FETCH', 'REUSE_PROVENANCE'} &
                             {event['event_type'] for event in result['events']})
        self.assertEqual(self.forward_hits, [[], [], []])
        self.assertEqual(self.engine.metrics.frequencies[self.entry.key], 3)
        self.assertEqual(self.cache.hits, 3)  # index observations, not physical reuse
        self.assertEqual(self.entry.age(self.cache.now), 3)  # clock advances; access does not reset
        self.assertEqual(len(self.engine.metrics.history.history[0]), 3)

    def test_physical_hit_updates_once_and_exports_reducer(self):
        result = self.query('SEMCACHE_PHYSICAL_REUSE')
        self.assertEqual(self.entry.frequency, 1)
        self.assertEqual(self.entry.last_access, 1)
        self.assertAlmostEqual(self.entry.impact, .8 * 5 + .2 * 2)
        self.assertEqual(len([e for e in result['events'] if e['event_type'] == 'CHU']), 1)
        self.assertEqual(len(result['summary']['reuse_block_provenance']), 1)
        row = record_from_engine(result, experiment_id='fixture', repeat_index=0,
            warmup_runs=0, measured_runs=1, model_metadata={'model': 'stub', 'device': 'cpu'},
            gpu_name=None, condition='repeat')
        self.assertEqual(row['impact_reducer_type'], 'paper_row_l2_sum')
        self.assertTrue(row['impact_reducer_applied'])
        self.assertEqual(aggregate_raw([row])[0]['impact_reducer_type'], 'paper_row_l2_sum')

    def test_scheduled_pbr_is_query_based_not_a_reuse_update(self):
        self.engine.pbr_interval_queries = 2
        self.engine.metrics.history.append(1, 'earlier', 0, {self.other.key: 7.})
        first = self.query('SEMCACHE_LOOKUP_NO_REUSE')
        self.assertEqual(first['summary']['pbr_updates_this_query'], 0)
        second = self.query('SEMCACHE_LOOKUP_NO_REUSE')
        self.assertEqual(self.entry.impact, 2.)  # PBR uses observed query attention even without reuse
        self.assertEqual(self.other.impact, 7.)  # all resident clusters are visited
        self.assertEqual((self.entry.frequency, self.entry.last_access), (0, 0))
        self.assertEqual((self.other.frequency, self.other.last_access), (0, 0))
        events = [e for e in second['events'] if e['event_type'] == 'PBR']
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e['trigger_policy'].endswith('REPRODUCTION_CHOICE') for e in events))
        self.assertFalse(any(e['event_type'] == 'CHU' for e in second['events']))


class ScheduleAndAddressTest(unittest.TestCase):
    def test_admission_occurrences_and_bounded_pbr_are_not_reuse(self):
        cache = GlobalCache(100)
        manager = CacheMetricManager(cache, history_lambda=2, frequency_window=2)
        entry = CacheEntry(0, (1,), (0, 1), 10, impact=9.)
        absent = CacheEntry(0, (2,), (0, 1), 10, impact=7.)
        self.assertTrue(cache.insert(entry))
        self.assertTrue(cache.insert(absent))
        manager.arrive([entry.key, entry.key])  # repeated occurrences count twice
        manager.arrive([entry.key])
        self.assertEqual(manager.frequencies[entry.key], 3)
        manager.arrive([absent.key])
        self.assertEqual(manager.frequencies[entry.key], 1)
        for order, value in enumerate([100., 2., 4.]):
            manager.history.append(0, str(order), order, {entry.key: value})
        updates = manager.pbr(0)
        self.assertEqual(entry.impact, 3.)
        self.assertEqual(absent.impact, 7.)
        self.assertEqual(entry.frequency, 0)
        self.assertEqual(entry.last_access, 0)
        self.assertTrue(next(u for u in updates if u['cache_key'] == absent.key)['denominator_zero'])

    def test_profile_defaults_and_validation(self):
        p = argparse.ArgumentParser()
        add_alignment_arguments(p)
        args = p.parse_args([])
        validate_alignment_arguments(args)
        self.assertEqual((args.impact_reducer, args.cluster_update_mode,
            args.cluster_update_interval, args.rho, args.history_lambda, args.pbr_mode,
            args.pbr_interval), ('paper_row_l2_sum', 'buffered', 100, .8, 100, 'interval', 100))
        args.rho = 1.
        with self.assertRaises(ValueError):
            validate_alignment_arguments(args)

    def test_buffered_eq9_boundary_and_diagnostics(self):
        c = IntentClusterer(2, update_interval=100, update_mode='buffered')
        c.initialize([[0.], [100.]])
        for _ in range(99):
            r = c.observe_with_diagnostics([2.])
            self.assertFalse(r['centroid_update_applied'])
        self.assertEqual(c.centroids, [[0.], [100.]])
        r = c.observe_with_diagnostics([2.])
        self.assertTrue(r['centroid_update_applied'])
        self.assertEqual(c.counts, [101, 1])
        self.assertAlmostEqual(c.centroids[0][0], 200 / 101)
        self.assertEqual(c.pending, [])
        self.assertFalse(c.observe_with_diagnostics([2.])['centroid_update_applied'])

    def test_pool_address_is_distinct_from_match_key_and_query_position(self):
        a = BlockDescriptor(PoolBlockAddress(0, 0, 3), (1, 2, 3), SourceOccurrence('q', 'u', 'a', 4, 7))
        b = BlockDescriptor(PoolBlockAddress(0, 3, 6), (1, 2, 3), SourceOccurrence('q', 'u', 'a', 10, 13))
        self.assertEqual(a.match_key, b.match_key)
        self.assertNotEqual(a.address, b.address)
        self.assertNotEqual(a.source, b.source)
        matcher = ExactTokenMatcher()
        self.assertEqual(matcher.key(0, Subsequence((1, 2, 3), 4, 7)),
                         matcher.key(0, Subsequence((1, 2, 3), 10, 13)))
        with self.assertRaises(ValueError):
            PoolBlockAddress(0, 3, 3)
        with self.assertRaises(ValueError):
            BlockDescriptor(PoolBlockAddress(0, 0, 2), a.token_ids, a.source)

    def test_artifact_grouping_does_not_mix_reducers_or_schedules(self):
        rows = [raw_record(experiment_id='x', model_id='stub', mode='SEMCACHE_PHYSICAL_REUSE',
                 query_id='q', request_wall_ms=1., impact_reducer_type=name,
                 cluster_update_interval_queries=interval)
                for name, interval in [('paper_row_l2_sum', 100),
                    ('reproduction_frobenius_mean', 100), ('paper_row_l2_sum', 2)]]
        summary = [r for r in aggregate_raw(rows) if r['metric'] == 'request_wall_ms']
        self.assertEqual(len(summary), 3)
        self.assertTrue(all(r['count'] == 1 for r in summary))
        legacy = aggregate_raw([raw_record(experiment_id='old', model_id='stub', mode='m', query_id='q')])
        self.assertIsNone(legacy[0]['impact_reducer_type'])

    def test_delta_artifact_preserves_profile_and_rejects_mixed_profiles(self):
        common = dict(experiment_id='x', model_id='stub', query_id='same_user_exact',
            candidate_blocks=2, reused_tokens=3, recomputed_tokens=3, token_reuse_ratio=.5,
            block_hits=1, attention_impact_ms=2., attention_impact_ms_per_block=1.,
            mixed_qkv_execution_ms=3., prefill_wall_ms=5., request_wall_ms=10.,
            impact_reducer_type='paper_row_l2_sum', cluster_update_interval_queries=100)
        lookup = dict(common, mode='SEMCACHE_LOOKUP_NO_REUSE')
        physical = dict(common, mode='SEMCACHE_PHYSICAL_REUSE')
        result = reuse_delta_summary([lookup, physical])
        self.assertEqual(result[0]['impact_reducer_type'], 'paper_row_l2_sum')
        self.assertEqual(result[0]['cluster_update_interval_queries'], 100)
        self.assertEqual(result[0]['qkv_reuse_delta_ms'], 0.)
        physical['impact_reducer_type'] = 'reproduction_frobenius_mean'
        self.assertEqual(reuse_delta_summary([lookup, physical]), [])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch absent; tensor-only tests not run')
class ImpactTensorTest(unittest.TestCase):
    def test_exact_layer_token_sum_and_head_mean(self):
        import torch
        # Selected rows t=1,2. Norms by head: [5,2], [10,4].
        # Layer 1: (5+10)/2 + (2+4)/2 = 10.5. Layer 2 doubled => 21.
        a = torch.tensor([[[[1., 0., 0.], [3., 4., 0.], [0., 0., 2.]],
                           [[1., 0., 0.], [6., 8., 0.], [0., 0., 4.]]]])
        paper = make_impact_reducer()
        self.assertAlmostEqual(paper.reduce([a, 2*a], 1, 3), 31.5)
        old = make_impact_reducer('reproduction_frobenius_mean')
        self.assertNotAlmostEqual(old.reduce([a, 2*a], 1, 3), 31.5)
        self.assertEqual(paper.metadata['head_aggregation'], 'mean_REPRODUCTION_CHOICE')

    def test_query_rows_not_block_key_columns_and_padding(self):
        import torch
        a = torch.ones(1, 1, 3, 3)
        paper = make_impact_reducer()
        self.assertAlmostEqual(paper.reduce([a], 0, 2), 1 + math.sqrt(2))
        self.assertAlmostEqual(make_impact_reducer('reproduction_frobenius_mean').reduce([a], 0, 2), math.sqrt(5))
        self.assertAlmostEqual(paper.reduce([a], 0, 3, torch.tensor([1, 0, 1])), 1 + math.sqrt(2))
        # Future and padded values cannot contribute, including invalid sentinels.
        a[0, 0, 0, 2] = float('nan')
        self.assertAlmostEqual(paper.reduce([a], 0, 2), 1 + math.sqrt(2))
        a[0, 0, 1, 0] = float('nan')
        with self.assertRaises(ValueError):
            paper.reduce([a], 0, 2)
        with self.assertRaises(ValueError):
            paper.reduce([torch.ones(1, 1, 2, 3)], 0, 2)


if __name__ == '__main__':
    unittest.main()
