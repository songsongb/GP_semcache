"""L2C contracts + REQUIRED real torch CPU parity. Missing torch FAILS, not SKIP."""
from collections import defaultdict
from copy import deepcopy
import importlib.util
import math
import os
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.edgelora import r5_d5_impact as candidate
from semcache.edgelora.r5_subsequence_generation import StreamingImpact
from semcache.semantic.subsequence import SubsequenceExtractor, Subsequence
from semcache.edgelora.communication import digest


def bits(values):
    return b''.join(struct.pack('<d', v) for v in values)


class Contracts(unittest.TestCase):
    """No tensor arithmetic is simulated here; these tests cover control flow."""
    def setUp(self):
        self.req = dict(request_id='CPU_CONTRACT_ONLY', token_ids=[1, 2, 3, 1, 2, 3])
        self.windows = SubsequenceExtractor(3).extract(self.req['token_ids'])

    def prob(self):
        class Prob:
            shape = (1, 2, 6, 6)
            def numel(self): return 72
            def element_size(self): return 4
            def detach(self): return self
            def double(self): return self
        return Prob()

    def test_reference_remains_default_and_candidate_is_explicit(self):
        self.assertIs(type(candidate.create_impact(self.req, 2, self.windows)), StreamingImpact)
        observed = candidate.create_impact(self.req, 2, self.windows, candidate.HOST_BATCH)
        self.assertIsInstance(observed, StreamingImpact)
        with self.assertRaises(ValueError): candidate.implementation({'impact_implementation': 'UNKNOWN'})
        with self.assertRaises(ValueError): candidate.create_impact(self.req, 2, self.windows, None)

    def test_approval_requires_new_native_module_only_for_candidate(self):
        from semcache.edgelora.r5_d5_lightperf import approval_modules, TEST_MODULES
        self.assertEqual(approval_modules({}), list(TEST_MODULES))
        self.assertEqual(approval_modules({'impact_implementation': candidate.HOST_BATCH}),
                         list(TEST_MODULES) + ['test_r5_d5_impact'])

    def test_empty_and_oversized_staging_are_bounded_without_torch(self):
        self.assertEqual(candidate.deferred_window_values(None, []), [])
        with self.assertRaises(ValueError):
            candidate.deferred_window_values(None, self.windows * 200)

    def test_streaming_keeps_original_layer_accumulation_and_record_order(self):
        observed = candidate.DeferredHostImpact(self.req, 3, self.windows)
        values = (1e16, 1., 1.)
        with patch.object(candidate, 'deferred_window_values', side_effect=[[v] * 4 for v in values]) as reduce:
            for layer in range(3): observed.observe(layer, self.prob())
        expected = 0.
        for v in values: expected += v
        self.assertEqual(bits(list(observed.values.values())), bits([expected] * 4))
        self.assertEqual(reduce.call_count, 3)
        self.assertEqual(observed.host_materialization_calls, 3)
        self.assertEqual(observed.observed, [0, 1, 2])
        self.assertEqual([r['layer'] for r in observed.records], [0, 1, 2])
        self.assertEqual(observed.values_for(dict(req=self.req, windows=self.windows)), observed.values)

    def test_invalid_reduction_never_commits_layer_or_impacts(self):
        observed = candidate.DeferredHostImpact(self.req, 2, self.windows)
        with patch.object(candidate, 'deferred_window_values', side_effect=ValueError('nonfinite')):
            with self.assertRaises(ValueError): observed.observe(0, self.prob())
        self.assertEqual(observed.observed, [])
        self.assertEqual(observed.records, [])
        self.assertEqual(observed.host_materialization_calls, 0)
        self.assertTrue(all(v == 0 for v in observed.values.values()))
        with self.assertRaises(ValueError): observed.values_for(dict(req=self.req, windows=self.windows))

    def test_existing_probability_and_working_set_guards_remain(self):
        observed = candidate.DeferredHostImpact(self.req, 1, self.windows, max_layer_bytes=128)
        with self.assertRaises(ValueError): observed.observe(0, self.prob())
        observed = candidate.DeferredHostImpact(self.req, 1, self.windows)
        with self.assertRaises(ValueError): observed.observe(1, self.prob())
        with patch.object(candidate, 'deferred_window_values', return_value=[1.] * 4):
            observed.observe(0, self.prob())
        self.assertEqual(observed.peak_working_estimate, 288 + 72*8 + 2*3*6*8 + candidate.STAGING_RESERVE_BYTES)
        with self.assertRaises(ValueError): observed.observe(0, self.prob())

    def test_resource_reserve_tightens_bounds_without_mutating_reference(self):
        resources = dict(reasons=[], status='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING', impact_layer_working_bytes=80*2**20,
                         gpu_estimate_bytes=22*2**30 - 1, gpu_allocation_ceiling_bytes=22*2**30)
        self.assertIs(candidate.resource_bounds(resources, candidate.REFERENCE), resources)
        bounded = candidate.resource_bounds(resources, candidate.HOST_BATCH)
        self.assertEqual(bounded['status'], 'BLOCKED_INFEASIBLE')
        self.assertEqual(resources['reasons'], [])
        self.assertEqual(bounded['gpu_allocation_ceiling_bytes'], 22*2**30)

    def test_policy_trace_fixture_runs_real_chu_pbr_and_eviction(self):
        trace = policy_trace(self.windows, [1., 2., 3., 4.])
        self.assertTrue(any(e[0] == 'PBR' for e in trace['events']))
        self.assertTrue(any(e[0] == 'EVICT' for e in trace['events']))
        self.assertEqual(digest(trace), digest(policy_trace(self.windows, [1., 2., 3., 4.])))

    def test_scalar_batch_does_not_call_row_precompute_or_prefix_sum(self):
        import ast, inspect
        tree = ast.parse(inspect.getsource(candidate.deferred_window_values))
        attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        self.assertNotIn('item', attrs)
        self.assertNotIn('cumsum', attrs)
        self.assertEqual(attrs.count('norm'), 1)
        self.assertEqual(attrs.count('cpu'), 1)
        self.assertEqual(attrs.count('tolist'), 1)


def policy_trace(windows, impacts):
    """Actual logical policies, synthetic cache only; includes default query100 PBR."""
    from semcache.cache.global_cache import GlobalCache
    from semcache.cache.cache_entry import CacheEntry
    from semcache.cache.metric_manager import CacheMetricManager
    from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
    cache, events = GlobalCache(20), []
    metrics = CacheMetricManager(cache)
    for q in range(100):
        keys = [(0, w.token_ids) for w in windows]
        metrics.arrive(keys)
        grouped = defaultdict(list)
        for key, value in zip(keys, impacts): grouped[key].append(value)
        history = {key: sum(v) / len(v) for key, v in grouped.items()}
        metrics.history.append(0, str(q), cache.now, history)
        hits = [CacheHit(w, cache.entries[k], cache.entries[k].impact) for w, k in zip(windows, keys) if k in cache.entries]
        selected, _ = select_nonoverlapping(hits, max(w.end for w in windows))
        for hit in selected:
            metrics.reused(hit.entry, impacts[next(i for i, w in enumerate(windows) if w.start == hit.window.start)])
            events.append(('CHU', hit.entry.key, hit.entry.frequency, hit.entry.impact))
        seen = set()
        for w, key, value in zip(windows, keys, impacts):
            if key in seen or key in cache.entries: continue
            seen.add(key)
            entry = CacheEntry(0, w.token_ids, (w.start, w.end), 10, impact=value)
            accepted = cache.insert(entry, metrics.frequencies[key], metrics.frequencies,
                on_event=lambda kind, e, score: events.append((kind, e.key, score)))
            events.append(('ADMISSION', key, accepted))
        if cache.now % 100 == 0: events.append(('PBR', metrics.pbr(0)))
    history = [dict(q, impacts=sorted(q['impacts'].items())) for q in metrics.history.history[0]]
    return dict(events=events, history=history, frequencies=sorted(metrics.frequencies.items()),
                resident=[(k, e.frequency, e.impact, e.last_access) for k, e in cache.entries.items()])


class ReviewContracts(unittest.TestCase):
    def fixtures(self):
        row = dict(request_id='R0', sample_id='S0', user_id='User_01', adapter_id='Adapter_A', input_tokens=6,
            output_tokens=2, generated_token_ids=[1, 2], generated_text_sha256='a'*64,
            quality={'metric': 'SYNTHETIC_ONLY', 'score': 1.}, scorer_version='TEST', physical_hit=True,
            physical_reuse_sources=['SOURCE'], fresh_qkv_rows=9, skipped_qkv_rows=9, cache_occupancy_bytes=64,
            cache_entry_count=1, policy_counts={'INSERT': 1, 'EVICT': 0, 'CHU': 1, 'PBR': 0}, matching_candidate_count=1,
            categories={'A': 9, 'B': 0, 'C': 0, 'D': 0}, attention_impact={'values_sha256': 'b'*64},
            integrity_validation=dict(full_entry_proof_count=1, consumed_tensor_proof_count=3, admission_entry_proof_count=1,
                full_entry_serialized_bytes=64, consumed_tensor_serialized_bytes=32, admission_entry_serialized_bytes=64,
                integrity_check_failures=0, records_expired=True), application_e2e_ms=100., application_ttft_ms=50.)
        run = dict(condition='SEMCACHE_D5_EXACT', dataset='qasper', scale=1, tasks_sha256='TEST', source={'commit': 'SAME'},
            performance_knobs={'bandwidth_mbps': None, 'ud_throughput_factor': 1.},
            gpu=dict(uuid='GPU_TEST_ONLY', name='TEST', device='cuda:0', total_bytes=1000, free_bytes=900), backend='TEST',
            versions={}, precision_policy='TEST', cpu_concurrency='TEST', warmup_requests=0, measurement_scope='NATIVE_LIGHT',
            impact_implementation=candidate.REFERENCE, requests=[row])
        optimized = deepcopy(run)
        optimized['impact_implementation'] = candidate.HOST_BATCH
        optimized['gpu']['free_bytes'] = 899  # Memory availability is metadata, not physical identity.
        optimized['requests'][0].update(application_e2e_ms=90., application_ttft_ms=40.)
        return run, optimized

    def test_pair_reports_bounded_parity_not_full_fidelity(self):
        from semcache.edgelora.r5_d5_impact_review import _compare
        result = _compare(*self.fixtures())
        self.assertEqual(result['status'], 'IMPACT_TOKEN_AND_LIGHT_SUMMARY_PARITY_PASS')
        self.assertEqual(result['requests'][0]['delta_e2e_ms'], -10.)
        self.assertEqual(result['independent_full_vocabulary_audit'], 'NOT_EVALUATED')
        self.assertEqual(result['correctness_qualified_speedup'], 'NOT_ESTABLISHED')

    def test_changed_impact_policy_or_tokens_fails_parity(self):
        from semcache.edgelora.r5_d5_impact_review import _compare
        for field in ('impact', 'tokens', 'policy'):
            a, b = self.fixtures()
            if field == 'impact': b['requests'][0]['attention_impact']['values_sha256'] = 'c'*64
            if field == 'tokens': b['requests'][0]['generated_token_ids'] = [9]
            if field == 'policy': b['requests'][0]['policy_counts']['EVICT'] = 1
            self.assertEqual(_compare(a, b)['status'], 'PARITY_FAILED')

    def test_identity_mismatch_and_failed_integrity_never_pass(self):
        from semcache.edgelora.r5_d5_impact_review import _compare
        for field in ('gpu', 'source', 'bandwidth', 'integrity'):
            a, b = self.fixtures()
            if field == 'gpu': b['gpu']['uuid'] = 'OTHER'
            if field == 'source': b['source']['commit'] = 'OTHER'
            if field == 'bandwidth': b['performance_knobs']['bandwidth_mbps'] = 50
            if field == 'integrity':
                a['requests'][0]['integrity_validation']['integrity_check_failures'] = 1
                b['requests'][0]['integrity_validation']['integrity_check_failures'] = 1
            with self.subTest(field=field), self.assertRaises(ValueError): _compare(a, b)

    def test_official_verifier_failure_prevents_output(self):
        from semcache.edgelora import r5_d5_impact_review as review
        paths = dict(run='reference', plan='plan', cpu_audit='audit')
        with patch.object(review, 'verify_run', side_effect=ValueError('bad inventory')) as check, \
             patch.object(review, 'fresh') as output:
            with self.assertRaises(ValueError): review.review(paths, dict(paths, run='candidate'), 'unused')
        self.assertEqual(check.call_count, 1)
        output.assert_not_called()

    def test_full_and_exact_different_implementations_are_not_pooled(self):
        from semcache.edgelora.r5_d5_lightperf_analysis import compare
        full, optimized = self.fixtures()
        full['condition'] = 'FULL_RECOMPUTE'
        result = compare(full, optimized)
        self.assertEqual(result['status'], 'NOT_COMPARABLE')
        self.assertIn('impact_implementation', result['mismatches'])


class TorchParity(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '', 'CPU parity must hide CUDA')
        self.assertIsNotNone(importlib.util.find_spec('torch'), 'BLOCKED_NATIVE_DEPENDENCY: torch; no bitwise approval')
        import torch
        self.torch = torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def compare(self, attention, windows, mask=None):
        from semcache.cache.attention_impact import PaperRowL2SumReducer
        reducer = PaperRowL2SumReducer()
        before = attention.clone()
        expected = [reducer.reduce([attention], w.start, w.end, mask) for w in windows]
        actual = candidate.deferred_window_values(attention, windows, mask)
        for window, a, b in zip(windows, expected, actual):
            self.assertEqual(bits([a]), bits([b]), f'FP64 mismatch window[{window.start}:{window.end}]: reference={a.hex()} candidate={b.hex()}; norm/head-mean/window-sum order must remain frozen')
        self.assertTrue(self.torch.equal(before.contiguous().view(self.torch.uint8), attention.contiguous().view(self.torch.uint8)))
        self.assertFalse(self.torch.cuda.is_initialized())
        return expected, actual

    def test_fp16_fp32_fp64_all_windows_heads_and_boundaries(self):
        t = self.torch
        for dtype in (t.float16, t.float32, t.float64):
            for n in (3, 6, 17):
                for heads in (1, 3):
                    with self.subTest(dtype=dtype, n=n, heads=heads):
                        attention = t.rand((1, heads, n, n), dtype=dtype)
                        self.compare(attention, SubsequenceExtractor(3).extract(list(range(n))))
        self.compare(t.rand((1, 3, 6, 6), dtype=t.float64).transpose(-1, -2), SubsequenceExtractor(3).extract(list(range(6))))
        self.compare(-t.zeros((1, 3, 3, 3), dtype=t.float64), SubsequenceExtractor(3).extract([1, 2, 3]))

    def test_repeated_token_windows_and_ragged_boundaries(self):
        windows = SubsequenceExtractor(3).extract([1, 2, 3, 1, 2, 3])
        windows += [Subsequence((1,), 0, 1), Subsequence((3,), 5, 6)]
        self.compare(self.torch.rand((1, 3, 6, 6), dtype=self.torch.float64), windows)

    def test_masks_future_nan_inf_and_valid_nonfinite_fail_closed(self):
        t = self.torch
        windows = [Subsequence((1, 2, 3), 0, 3)]
        attention = t.rand((1, 3, 6, 6), dtype=t.float64)
        attention[:, :, 0, 5] = float('nan')
        attention[:, :, 5, :] = float('inf')  # Outside requested query rows.
        self.compare(attention, windows)
        for mask in (t.tensor([1, 0, 1, 1, 1, 1]), t.tensor([[1, 0, 1, 1, 1, 1]])):
            self.compare(attention, windows, mask)
            padded = attention.clone(); padded[:, :, 1, :] = float('nan'); padded[:, :, :, 1] = float('nan')
            self.compare(padded, windows, mask)
        for value in (float('nan'), float('inf')):
            bad = attention.clone(); bad[0, 0, 0, 0] = value
            from semcache.cache.attention_impact import PaperRowL2SumReducer
            with self.assertRaises(ValueError): PaperRowL2SumReducer().reduce([bad], 0, 3)
            with self.assertRaises(ValueError): candidate.deferred_window_values(bad, windows)

    def test_invalid_shapes_windows_masks_and_overflow(self):
        t = self.torch
        windows = SubsequenceExtractor(3).extract(list(range(6)))
        for attention in (None, t.ones(2, 3, 6, 6), t.ones(1, 0, 6, 6), t.ones(1, 3, 6, 5)):
            with self.assertRaises(ValueError): candidate.deferred_window_values(attention, windows)
        attention = t.ones(1, 3, 6, 6, dtype=t.float64)
        for mask in (t.ones(2), t.ones(2, 6)):
            with self.assertRaises(ValueError): candidate.deferred_window_values(attention, windows, mask)
        with self.assertRaises(ValueError): candidate.deferred_window_values(attention, [SimpleNamespace(start=5, end=8)])
        huge = t.full((1, 3, 6, 6), 1e308, dtype=t.float64)
        with self.assertRaises(ValueError): candidate.deferred_window_values(huge, windows)

    def test_one_host_batch_no_tensor_bool_or_item_and_layer_order(self):
        t = self.torch
        req = dict(request_id='TINY_CPU_ONLY', token_ids=[1, 2, 3, 1, 2, 3])
        windows = SubsequenceExtractor(3).extract(req['token_ids'])
        tensors = [t.rand((1, 3, 6, 6), dtype=t.float64) for _ in range(3)]
        reference, optimized = StreamingImpact(req, 3, windows), candidate.DeferredHostImpact(req, 3, windows)
        for layer, prob in enumerate(tensors): reference.observe(layer, prob)
        calls = []
        cpu = t.Tensor.cpu
        def transfer(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return cpu(tensor, *args, **kwargs)
        with patch.object(t.Tensor, 'cpu', transfer), patch.object(t.Tensor, 'item', side_effect=AssertionError('scalar item')), \
             patch.object(t.Tensor, '__bool__', side_effect=AssertionError('scalar bool')):
            for layer, prob in enumerate(tensors): optimized.observe(layer, prob)
        self.assertEqual(calls, [(2, 4)] * 3)
        self.assertEqual(digest(reference.values), digest(optimized.values))
        self.assertEqual(reference.records, optimized.records)
        self.assertEqual(optimized.host_materialization_calls, 3)

    def test_actual_history_chu_pbr_admission_eviction_bitwise_trace(self):
        t = self.torch
        windows = SubsequenceExtractor(3).extract([1, 2, 3, 1, 2, 3])
        reference, optimized = self.compare(t.rand((1, 3, 6, 6), dtype=t.float64), windows)
        a, b = policy_trace(windows, reference), policy_trace(windows, optimized)
        self.assertEqual(a, b)
        self.assertEqual(digest(a), digest(b))
        self.assertTrue(any(e[0] == 'PBR' for e in a['events']))
        self.assertTrue(any(e[0] == 'EVICT' for e in a['events']))

    def test_tiny_opt_native_logits_policy_and_proofs(self):
        self.assertIsNotNone(importlib.util.find_spec('transformers'), 'BLOCKED_NATIVE_DEPENDENCY: transformers')
        from test_r5_d5_lightperf_native import NativeParity
        helper = NativeParity('test_strict_light_full_exact_approx_logits_tokens_impact_policy_routing')
        helper.setUp()
        reference = helper.run_case('SEMCACHE_D5_EXACT')
        optimized = helper.run_case('SEMCACHE_D5_EXACT', impact_implementation=candidate.HOST_BATCH)
        self.assertEqual(reference['logits'], optimized['logits'])
        self.assertEqual(reference['snapshots'], optimized['snapshots'])
        self.assertEqual(reference['policy'], optimized['policy'])
        for a, b in zip(reference['rows'], optimized['rows']):
            for key in ('generated_token_ids', 'fresh_qkv_rows', 'skipped_qkv_rows', 'physical_hit', 'policy_counts', 'cache_occupancy_bytes'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a['attention_impact']['values_sha256'], b['attention_impact']['values_sha256'])
            self.assertEqual(b['attention_impact']['host_materialization_calls'], 2)
            for key in ('full_entry_proof_count', 'consumed_tensor_proof_count', 'admission_entry_proof_count', 'integrity_check_failures'):
                self.assertEqual(a['integrity_validation'][key], b['integrity_validation'][key])


if __name__ == '__main__': unittest.main()
