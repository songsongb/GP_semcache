"""Required native CPU L2E parity. Missing dependencies fail, never skip."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora import r5_d5_impact as l2c
from semcache.edgelora import r5_d5_impact_compute as l2e
from semcache.edgelora.communication import digest
from semcache.edgelora.r5_subsequence_generation import StreamingImpact
from semcache.semantic.subsequence import Subsequence, SubsequenceExtractor


def bits(values):
    return b''.join(struct.pack('<d', v) for v in values)


def semantic(value):
    if isinstance(value, dict):
        return {(digest(k) if isinstance(k, tuple) else k): semantic(v) for k, v in value.items()
                if k != 'costs' and not (isinstance(k, str) and k.endswith(('_ms', '_ns')))}
    if isinstance(value, (tuple, list)):
        return [semantic(v) for v in value]
    return value


class NativeCPU(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '', 'CPU parity requires hidden CUDA')
        for name in ('torch', 'numpy', 'transformers'):
            self.assertIsNotNone(importlib.util.find_spec(name), 'BLOCKED_NATIVE_DEPENDENCY: ' + name)
        import torch
        self.t = torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(1)
        torch.manual_seed(42)


class ReducerParity(NativeCPU):
    def compare(self, attention, windows, mask=None, layer=0):
        from semcache.cache.attention_impact import PaperRowL2SumReducer
        original = attention.contiguous().view(self.t.uint8).clone()
        original_mask = None if mask is None else mask.contiguous().view(self.t.uint8).clone()
        expected = [PaperRowL2SumReducer().reduce([attention], w.start, w.end, mask) for w in windows]
        for name, fn in (('L2C', l2c.deferred_window_values), ('L2E', l2e.reused_window_values)):
            actual = fn(attention, windows, mask)
            self.assertEqual(len(actual), len(expected))
            for w, a, b in zip(windows, expected, actual):
                self.assertEqual(bits([a]), bits([b]),
                    f'{name} FP64 mismatch layer={layer} dtype={attention.dtype} '
                    f'window=[{w.start}:{w.end}] reference={a.hex()} candidate={b.hex()}')
        self.assertTrue(self.t.equal(original, attention.contiguous().view(self.t.uint8)))
        if mask is not None:
            self.assertTrue(self.t.equal(original_mask, mask.contiguous().view(self.t.uint8)))
        return expected

    def test_all_dtypes_heads_lengths_boundaries_noncontiguous_and_signed_zero(self):
        t = self.t
        for dtype in (t.float16, t.float32, t.float64):
            for n in (1, 2, 3, 6, 17, 64, 512):
                for heads in (1, 3, 8):
                    with self.subTest(dtype=dtype, n=n, heads=heads):
                        windows = SubsequenceExtractor(3).extract(list(range(n)))
                        windows += [Subsequence((1,), 0, 1), Subsequence((2,), n-1, n)]
                        self.compare(t.rand(1, heads, n, n, dtype=dtype), windows)
            attention = t.rand(1, 3, 17, 17, dtype=dtype).transpose(-1, -2)
            self.assertFalse(attention.is_contiguous())
            self.compare(attention, SubsequenceExtractor(3).extract(list(range(17))))
            self.compare(-t.zeros(1, 3, 6, 6, dtype=dtype), SubsequenceExtractor(3).extract(list(range(6))))

    def test_padding_mask_shapes_dtypes_and_masked_nonfinite(self):
        t = self.t
        windows = SubsequenceExtractor(3).extract([1, 2, 3, 1, 2, 3])
        for dtype in (t.float16, t.float32, t.float64):
            for mdtype in (t.bool, t.int64, t.float32):
                for matrix in (False, True):
                    mask = t.tensor([1, 0, 1, 1, 0, 1], dtype=mdtype)
                    mask = mask[None, :] if matrix else mask
                    attention = t.rand(1, 3, 6, 6, dtype=dtype)
                    attention[:, :, 1, :] = float('nan')
                    attention[:, :, :, 1] = float('inf')
                    attention[:, :, 4, :] = -float('inf')
                    attention[:, :, :, 4] = float('nan')
                    attention[:, :, 0, 5] = float('nan')
                    self.compare(attention, windows, mask)
        mask = t.ones(12)[::2]
        self.assertFalse(mask.is_contiguous())
        self.compare(t.rand(1, 2, 6, 6), windows, mask)
        self.compare(t.full((1, 2, 6, 6), float('nan')), windows, t.zeros(6))

    def test_repeated_and_out_of_order_windows_keep_exact_order(self):
        windows = SubsequenceExtractor(3).extract([1, 2, 3, 1, 2, 3])
        windows = windows[::-1] + windows[:2] + [Subsequence((1,), 0, 1)]
        self.compare(self.t.rand(1, 3, 6, 6, dtype=self.t.float64), windows)

    def test_invalid_shapes_masks_windows_nonfinite_overflow_and_empty(self):
        t = self.t
        windows = SubsequenceExtractor(3).extract(list(range(6)))
        errors = [(None, windows, None), (t.ones(2, 2, 6, 6), windows, None),
                  (t.ones(1, 0, 6, 6), windows, None), (t.ones(1, 2, 6, 5), windows, None),
                  (t.ones(6, 6), windows, None)]
        for mask in (t.ones(2), t.ones(2, 6), t.ones(1, 1, 6), t.tensor(1)):
            errors.append((t.ones(1, 2, 6, 6), windows, mask))
        for start, end in ((-1, 2), (1, 1), (5, 8), (7, 9)):
            errors.append((t.ones(1, 2, 6, 6), [Subsequence((1,), start, end)], None))
        errors.append((t.full((1, 2, 6, 6), 1e308, dtype=t.float64), windows, None))
        for value in (float('nan'), float('inf'), -float('inf')):
            bad = t.ones(1, 2, 6, 6, dtype=t.float64)
            bad[0, 0, 5, 0] = value  # Last window: earlier valid scalars must not escape.
            errors.append((bad, windows, None))
        errors.append((t.ones(1, 2, 6, 6), windows * 200, None))
        for attention, spans, mask in errors:
            caught = []
            for fn in (l2c.deferred_window_values, l2e.reused_window_values):
                with self.assertRaises(ValueError) as exc:
                    fn(attention, spans, mask)
                caught.append(str(exc.exception))
            self.assertEqual(caught[0], caught[1])
        self.assertEqual(l2e.reused_window_values(None, []), [])

    def test_one_index_padding_conversion_host_batch_and_no_scalar_reads(self):
        t = self.t
        attention = t.rand(1, 3, 17, 17, dtype=t.float64)
        windows = SubsequenceExtractor(3).extract(list(range(17)))
        mask = t.ones(1, 17)
        cpu, to = t.Tensor.cpu, t.Tensor.to
        transfers, conversions = [], []
        def host(tensor, *args, **kwargs):
            transfers.append(tuple(tensor.shape))
            return cpu(tensor, *args, **kwargs)
        def convert(tensor, *args, **kwargs):
            if tensor is mask:
                conversions.append(1)
            return to(tensor, *args, **kwargs)
        with patch.object(t, 'arange', wraps=t.arange) as arange, patch.object(t.Tensor, 'cpu', host), \
             patch.object(t.Tensor, 'to', convert), patch.object(t.Tensor, 'item', side_effect=AssertionError('item')), \
             patch.object(t.Tensor, '__bool__', side_effect=AssertionError('bool')):
            l2e.reused_window_values(attention, windows, mask)
        self.assertEqual(arange.call_count, 1)
        self.assertEqual(conversions, [1])
        self.assertEqual(transfers, [(2, len(windows))])

    def test_layer_accumulation_records_and_partial_failure(self):
        t = self.t
        req = dict(request_id='CPU_L2E_LAYERS', token_ids=[1, 2, 3, 1, 2, 3])
        windows = SubsequenceExtractor(3).extract(req['token_ids'])
        streams = [StreamingImpact(req, 3, windows), l2c.DeferredHostImpact(req, 3, windows),
                   l2e.IndexReuseImpact(req, 3, windows)]
        for layer in range(3):
            tensor = t.rand(1, 3, 6, 6)
            for stream in streams:
                stream.observe(layer, tensor)
        for stream in streams[1:]:
            self.assertEqual(bits(list(stream.values.values())), bits(list(streams[0].values.values())))
            self.assertEqual(stream.records, streams[0].records)
            self.assertEqual(stream.observed, [0, 1, 2])
            self.assertEqual(stream.host_materialization_calls, 3)
        self.assertEqual(streams[2].peak_working_estimate - streams[1].peak_working_estimate,
                         l2e.index_reserve_bytes(6))
        stream = l2e.IndexReuseImpact(req, 3, windows)
        stream.observe(0, t.ones(1, 3, 6, 6))
        before = (deepcopy(stream.values), deepcopy(stream.records), list(stream.observed), stream.host_materialization_calls)
        bad = t.ones(1, 3, 6, 6)
        bad[:, :, 5, 0] = float('nan')
        with self.assertRaises(ValueError):
            stream.observe(1, bad)
        self.assertEqual(before, (stream.values, stream.records, stream.observed, stream.host_materialization_calls))
        with self.assertRaises(ValueError):
            stream.values_for(dict(req=req, windows=windows))

    def test_bounds_remain_fail_closed_and_layer_sum_order_unchanged(self):
        t = self.t
        req = dict(request_id='CPU_L2E_BOUNDS', token_ids=list(range(6)))
        windows = SubsequenceExtractor(3).extract(req['token_ids'])
        with self.assertRaises(ValueError):
            l2e.IndexReuseImpact(req, 1, windows, max_layer_bytes=128).observe(0, t.ones(1, 3, 6, 6))
        for spans in (windows * 200, [Subsequence((1, 2, 3, 4), 0, 4)]):
            with self.assertRaises(ValueError):
                l2e.IndexReuseImpact(req, 1, spans).observe(0, t.ones(1, 3, 6, 6))
        with self.assertRaises(ValueError):
            l2e.IndexReuseImpact(req, 1, windows).observe(1, t.ones(1, 3, 6, 6))
        stream = l2e.IndexReuseImpact(req, 3, windows)
        with patch.object(l2e, 'reused_window_values', side_effect=[[x] * 4 for x in (1e16, 1., 1.)]):
            for layer in range(3):
                stream.observe(layer, t.ones(1, 3, 6, 6))
        expected = 0.
        for value in (1e16, 1., 1.):
            expected += value
        self.assertEqual(bits(list(stream.values.values())), bits([expected] * 4))


class TinyOPTParity(NativeCPU):
    def helper(self):
        from test_r5_d5_lightperf_native import NativeParity
        helper = NativeParity('test_strict_light_full_exact_approx_logits_tokens_impact_policy_routing')
        helper.setUp()
        return helper

    def test_three_way_real_cpu_logits_tokens_policy_integrity_and_metadata(self):
        from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
        original_complete, original_begin = PhysicalOwner.complete, PhysicalOwner.begin
        inventories, selections = [], []
        def complete(owner, **kwargs):
            result = original_complete(owner, **kwargs)
            inventories.append(digest(semantic(dict(entries=[vars(e) | {'tensors': None} for e in owner.cache.entries.values()],
                proofs=owner.proofs, hits=owner.cache.hits, misses=owner.cache.misses,
                history=list(owner.bridge.metrics.history.history[0])))))
            return result
        def begin(owner, *args, **kwargs):
            result = original_begin(owner, *args, **kwargs)
            selections.append(digest(semantic(result)))
            return result
        values, captures = [], []
        for mode in (l2c.REFERENCE, l2c.HOST_BATCH, l2e.MASK_INDEX_REUSE):
            inventories.clear()
            selections.clear()
            with patch.object(PhysicalOwner, 'complete', complete), patch.object(PhysicalOwner, 'begin', begin):
                values.append(self.helper().run_case('SEMCACHE_D5_EXACT', impact_implementation=mode))
            captures.append((list(inventories), list(selections)))
        for other, capture in zip(values[1:], captures[1:]):
            self.assertEqual(captures[0], capture)
            self.assertEqual(values[0]['logits'], other['logits'])
            self.assertEqual(values[0]['snapshots'], other['snapshots'])
            self.assertEqual(values[0]['policy'], other['policy'])
            self.assertEqual(semantic(values[0]['physicals']), semantic(other['physicals']))
            for a, b in zip(values[0]['rows'], other['rows']):
                for key in ('generated_token_ids', 'generated_text', 'fresh_qkv_rows', 'skipped_qkv_rows',
                            'physical_hit', 'policy_counts', 'cache_occupancy_bytes', 'cache_entry_count'):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(a['attention_impact']['values_sha256'], b['attention_impact']['values_sha256'])
                self.assertEqual(semantic(a['integrity_validation']), semantic(b['integrity_validation']))
        for rid, expected in values[0]['logits'].items():
            expected_bytes = b''.join(struct.pack('<f', x) for row in expected for x in row)
            for other in values[1:]:
                self.assertEqual(expected_bytes, b''.join(struct.pack('<f', x) for row in other['logits'][rid] for x in row))

    def test_native_integrity_failure_deadline_and_impact_abort_cleanup(self):
        from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
        self.helper().run_case('SEMCACHE_D5_EXACT', impact_implementation=l2e.MASK_INDEX_REUSE, corrupt=True)
        self.helper().run_case('SEMCACHE_D5_EXACT', impact_implementation=l2e.MASK_INDEX_REUSE, expire=True)
        original_abort = PhysicalOwner.abort
        aborted = []
        def abort(owner):
            original_abort(owner)
            aborted.append((owner.active, owner.bridge.pending, owner.validation.open))
        with patch.object(l2e, 'reused_window_values', side_effect=ValueError('L2E_IMPACT_FAIL')), \
             patch.object(PhysicalOwner, 'abort', abort), \
             patch.object(PhysicalOwner, 'complete', side_effect=AssertionError('admission before valid impact')):
            with self.assertRaisesRegex(ValueError, 'L2E_IMPACT_FAIL'):
                self.helper().run_case('SEMCACHE_D5_EXACT', impact_implementation=l2e.MASK_INDEX_REUSE)
        self.assertTrue(aborted)
        self.assertTrue(all(x == (None, None, False) for x in aborted))


class IntegrationContracts(unittest.TestCase):
    def test_plan_binds_selector_and_additive_resource_reserve(self):
        from semcache.edgelora import r5_d5_lightperf as api
        from semcache.edgelora.r5_d5_lightperf_io import SummaryWriter
        source = dict(approval_scope='DEVELOPMENT_NOT_GPU_APPROVED', commit='TEST_FIXTURE_ONLY', content_fingerprint='TEST')
        fields = dict(dataset_paths={}, stream=dict(strict_five_parity=True),
            resources=dict(reasons=[], status='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING',
                impact_layer_working_bytes=80*2**20, gpu_estimate_bytes=10*2**30,
                gpu_allocation_ceiling_bytes=22*2**30))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture_plan'
            def fresh(path, protected):
                Path(path).mkdir()
                return Path(path)
            with patch.object(api, 'source_identity', return_value=source), \
                 patch.object(api, 'material', side_effect=lambda *a, **k: deepcopy(fields)), \
                 patch.object(api, 'fresh', fresh):
                api.prepare('TEST_ORIGINAL', 'TEST_MULTI', 'qasper', 5, root, development=True,
                            impact_implementation=l2e.MASK_INDEX_REUSE)
                plan, inventory = api.load_plan(root)
                self.assertEqual(plan['impact_implementation'], l2e.MASK_INDEX_REUSE)
                self.assertEqual(plan['resources']['impact_index_reuse_reserve_bytes'], 8192)
                # A self-consistent reseal cannot hide a changed selector's resource contract.
                plan['impact_implementation'] = l2c.HOST_BATCH
                plan.pop('plan_id')
                plan['plan_id'] = digest(plan)
                other = Path(tmp) / 'resealed'
                other.mkdir()
                writer = SummaryWriter(other)
                writer.write('plan.json', plan)
                writer.seal()
                with self.assertRaisesRegex(ValueError, 'changed prepared input'):
                    api.load_plan(other)

    def test_official_verifier_binds_variant_host_count_and_new_approval_modules(self):
        from test_r5_d5_lightperf import AnalysisTests
        from semcache.edgelora.r5_d5_lightperf_io import SummaryWriter
        from semcache.edgelora.r5_d5_lightperf import approval_modules
        from semcache.edgelora.r5_d5_lightperf_analysis import verify_run
        original_write = SummaryWriter.write
        for invalid in (None, 'label', 'host_count', 'approval'):
            def write(writer, name, value):
                if name == 'plan.json':
                    value['impact_implementation'] = l2e.MASK_INDEX_REUSE
                    value.pop('plan_id')
                    value['plan_id'] = digest(value)
                elif name == 'cpu_validation.json':
                    value['modules'] = approval_modules({'impact_implementation':
                        l2c.HOST_BATCH if invalid == 'approval' else l2e.MASK_INDEX_REUSE})
                return original_write(writer, name, value)
            def mutate(run):
                run.update(impact_implementation=l2e.MASK_INDEX_REUSE, condition='SEMCACHE_D5_EXACT')
                for row in run['requests']:
                    row.update(condition='SEMCACHE_D5_EXACT', attention_impact=dict(layers=32,
                        implementation=l2c.HOST_BATCH if invalid == 'label' else l2e.MASK_INDEX_REUSE,
                        host_materialization_calls=31 if invalid == 'host_count' else 32))
            with tempfile.TemporaryDirectory() as tmp, patch.object(SummaryWriter, 'write', write):
                paths = AnalysisTests().receipts(Path(tmp), mutate_run=mutate)
                if invalid is None:
                    self.assertEqual(verify_run(*paths)[0]['impact_implementation'], l2e.MASK_INDEX_REUSE)
                else:
                    with self.assertRaises(ValueError):
                        verify_run(*paths)

    def test_explicit_selector_default_and_approval_modules_preserve_l2c(self):
        from semcache.edgelora.r5_d5_lightperf import approval_modules, TEST_MODULES
        req = dict(request_id='CPU_SELECTOR_ONLY', token_ids=list(range(6)))
        windows = SubsequenceExtractor(3).extract(req['token_ids'])
        self.assertIs(type(l2c.create_impact(req, 2, windows)), StreamingImpact)
        self.assertIs(type(l2c.create_impact(req, 2, windows, l2c.HOST_BATCH)), l2c.DeferredHostImpact)
        self.assertIs(type(l2c.create_impact(req, 2, windows, l2e.MASK_INDEX_REUSE)), l2e.IndexReuseImpact)
        self.assertEqual(l2c.implementation({}), l2c.REFERENCE)
        self.assertEqual(approval_modules({}), list(TEST_MODULES))
        self.assertEqual(approval_modules({'impact_implementation': l2c.HOST_BATCH}), list(TEST_MODULES)+['test_r5_d5_impact'])
        self.assertEqual(approval_modules({'impact_implementation': l2e.MASK_INDEX_REUSE}),
                         list(TEST_MODULES)+['test_r5_d5_impact','test_r5_d5_impact_compute','test_r5_d5_owner_diagnostics'])
        with self.assertRaises(ValueError):
            l2c.implementation({'impact_implementation':'UNKNOWN'})

    def test_resource_reserve_is_additive_and_can_only_tighten_bounds(self):
        resources = dict(reasons=[], status='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING',
            impact_layer_working_bytes=80*2**20, gpu_estimate_bytes=22*2**30-l2c.STAGING_RESERVE_BYTES-4096,
            gpu_allocation_ceiling_bytes=22*2**30)
        original = deepcopy(resources)
        self.assertIs(l2c.resource_bounds(resources, l2c.REFERENCE), resources)
        old = l2c.resource_bounds(resources, l2c.HOST_BATCH)
        new = l2c.resource_bounds(resources, l2e.MASK_INDEX_REUSE)
        self.assertEqual(old['status'], 'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING')
        self.assertEqual(new['status'], 'BLOCKED_INFEASIBLE')
        self.assertEqual(new['impact_index_reuse_reserve_bytes'], 8192)
        self.assertEqual(new['impact_layer_working_bytes']-old['impact_layer_working_bytes'],8192)
        self.assertEqual(resources, original)

    def test_paired_review_supports_l2c_l2e_but_rejects_pooling_variants(self):
        from test_r5_d5_impact import ReviewContracts
        from semcache.edgelora.r5_d5_impact_review import _compare
        from semcache.edgelora.r5_d5_lightperf_analysis import compare
        a, b = ReviewContracts().fixtures()
        for left in (l2c.REFERENCE, l2c.HOST_BATCH):
            a['impact_implementation'], b['impact_implementation'] = left, l2e.MASK_INDEX_REUSE
            result = _compare(a, b)
            self.assertEqual(result['status'], 'IMPACT_TOKEN_AND_LIGHT_SUMMARY_PARITY_PASS')
            self.assertEqual(result['candidate_impact_implementation'], l2e.MASK_INDEX_REUSE)
            full = deepcopy(a)
            full['condition'] = 'FULL_RECOMPUTE'
            self.assertEqual(compare(full, b)['status'], 'NOT_COMPARABLE')
        a['impact_implementation'] = b['impact_implementation']
        with self.assertRaises(ValueError):
            _compare(a, b)


if __name__ == '__main__':
    unittest.main()
