"""Model-free C0/C1 tests. No models, downloads, codec imports, or builds."""
import ast
import importlib.util
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen.common import (digest, raw_bytes, storage,
    sample_partitions, to_heads, from_heads, write_csv, read_json, summarize)
from semcache.experiments.cachegen.codecs import Baseline, Unavailable, audit, stage_status
from semcache.experiments.cachegen.harness import (main, BLOCK_FIELDS, C0_FIELDS,
    SUMMARY_FIELDS, QUALITY_FIELDS, verify_capture)


class LiteralTensor:
    """Small literal array for exercising layout adapters without installing torch."""
    def __init__(self, values, shape):
        self.values, self.shape, self.ndim = list(values), tuple(shape), len(shape)
        assert len(self.values) == math.prod(shape)

    def reshape(self, *shape):
        return LiteralTensor(self.values, shape)

    def permute(self, *axes):
        result_shape = tuple(self.shape[a] for a in axes)
        values = []
        for target in itertools.product(*(range(n) for n in result_shape)):
            source = [0]*self.ndim
            for axis, value in zip(axes, target):
                source[axis] = value
            offset = 0
            for size, index in zip(self.shape, source):
                offset = offset*size+index
            values.append(self.values[offset])
        return LiteralTensor(values, result_shape)

    def contiguous(self):
        return self


def records(n=30):
    return [dict(source_id=f'q{i:03}', conversation_id=f'd{i//3:03}', query_text=f'actual query {i}')
            for i in range(n)]


class StorageTests(unittest.TestCase):
    def test_canonical_exact_sizes(self):
        self.assertEqual(raw_bytes(3), dict(raw_q_bytes=491520, raw_kv_bytes=983040, raw_qkv_bytes=1474560))
        self.assertEqual(raw_bytes(10), dict(raw_q_bytes=1638400, raw_kv_bytes=3276800, raw_qkv_bytes=4915200))
        self.assertEqual(raw_bytes(3, 2, 8, 4)['raw_q_bytes'], 192)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                raw_bytes(invalid)

    def test_q_and_metadata_included(self):
        row = storage(3, 491520, 768)
        self.assertEqual(row['encoded_kv_total_bytes'], 492288)
        self.assertEqual(row['semcache_total_stored_bytes'], 983808)
        self.assertEqual(row['kv_compression_ratio'], 983040/492288)
        self.assertEqual(row['semcache_total_compression_ratio'], 1474560/983808)
        self.assertNotEqual(row['kv_compression_ratio'], row['semcache_total_compression_ratio'])
        self.assertLess(row['semcache_total_compression_ratio'], 1.5)
        self.assertEqual(storage(3, 983040, 0)['semcache_total_compression_ratio'], 1.)

    def test_layout_preserves_all_values_and_token_head_order(self):
        x = LiteralTensor(range(2*3*8), (2, 3, 8))
        h = to_heads(x, 2)
        self.assertEqual(h.shape, (2, 2, 3, 4))
        self.assertEqual(h.values[:12], [0,1,2,3,8,9,10,11,16,17,18,19])
        back = from_heads(h)
        self.assertEqual(back.shape, x.shape)
        self.assertEqual(back.values, x.values)
        with self.assertRaises(ValueError):
            to_heads(x, 3)
        with self.assertRaises(ValueError):
            from_heads(x)

    def test_split_determinism_source_disjoint_and_input_order_independent(self):
        a = sample_partitions(records(), 8, 42)
        b = sample_partitions(list(reversed(records())), 8, 42)
        self.assertEqual(a, b)
        self.assertNotEqual(a, sample_partitions(records(), 8, 43))
        for key in ('source_id', 'conversation_id'):
            self.assertFalse({r[key] for r in a['calibration']} & {r[key] for r in a['evaluation']})
        self.assertEqual(len(a['calibration']), 8)
        self.assertEqual(len(a['evaluation']), 8)
        self.assertEqual(a['hashes']['calibration'], digest(a['calibration']))
        with self.assertRaises(ValueError):
            sample_partitions(records(3), 3)
        with self.assertRaises(ValueError):
            sample_partitions(records()+records(), 3)

    def test_default_128_queries_per_partition(self):
        split = sample_partitions(records(900))
        self.assertEqual(len(split['calibration']), 128)
        self.assertEqual(len(split['evaluation']), 128)
        self.assertFalse({r['conversation_id'] for r in split['calibration']} &
                         {r['conversation_id'] for r in split['evaluation']})

    def test_plan_only_reuses_dataset_loader_without_model_loading(self):
        from semcache.experiments.cachegen.common import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'snips.json', [dict(id=str(i), text=f'query {i}', label='intent') for i in range(8)])
            write_json(root/'multiwoz.json', [dict(dialogue_id=str(i), turns=[
                dict(speaker='USER', utterance=f'query {i}'),
                dict(speaker='SYSTEM', utterance='response')]) for i in range(8)])
            with patch('semcache.models.loader.load_model', side_effect=AssertionError('No model allowed')):
                main(['capture', '--snips', str(root/'snips.json'), '--multiwoz', str(root/'multiwoz.json'),
                      '--revision', 'a'*40, '--queries-per-split', '2', '--plan-only', '--output-dir', str(root/'out')])
            plan = read_json(root/'out/capture_manifest.json')
            self.assertEqual(plan['status'], 'PLANNED')
            self.assertEqual(plan['blocks'], [])
            self.assertEqual(len(plan['sampling']['snips']['evaluation']), 2)
            self.assertEqual(len(plan['sampling']['multiwoz']['evaluation']), 2)

    def test_hash_stable_but_content_sensitive(self):
        self.assertEqual(digest({'a': 1, 'b': 2}), digest({'b': 2, 'a': 1}))
        self.assertNotEqual(digest(records()), digest(records()[::-1]))
        self.assertNotEqual(digest(records()), digest(records(29)))

    def test_unavailable_is_never_raw_fallback(self):
        for mode in ('CACHEGEN_FULL', 'CACHEGEN_QUANT'):
            available, reason = stage_status(mode)
            self.assertFalse(available)
            self.assertTrue(reason)
            with self.assertRaises(Unavailable):
                Baseline(mode)
        with self.assertRaises(ValueError):
            stage_status('pretend')

    def test_unavailable_excluded_from_summary(self):
        row = dict(dataset='snips', token_group_size=3, compression_mode='UNIFORM_INT8',
                   status='MEASURED', **storage(3, 491520, 768), encode_ms=1., decode_ms=2.,
                   k_rel_l2=.1, v_rel_l2=.2)
        summary = summarize([row, dict(row, status='UNAVAILABLE', compression_mode='CACHEGEN_FULL')])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]['block_count'], 1)
        self.assertEqual(summary[0]['encoded_kv_total_bytes_p95'], 492288)
        self.assertEqual(set(summary[0]), set(SUMMARY_FIELDS))

    def test_header_only_outputs_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fields in (C0_FIELDS, BLOCK_FIELDS, SUMMARY_FIELDS, QUALITY_FIELDS):
                path = Path(tmp)/'empty.csv'
                write_csv(path, [], fields)
                self.assertEqual(path.read_text().strip().split(','), fields)

    def test_audit_only_no_models_or_official_import(self):
        with tempfile.TemporaryDirectory() as tmp, patch('semcache.experiments.cachegen.harness.Official',
                                                        side_effect=AssertionError('must not import codec')):
            with patch('builtins.print'):
                main(['c0', '--audit-only', '--output-dir', tmp])
            self.assertEqual(read_json(Path(tmp)/'manifest.json')['status'], 'AUDIT_ONLY')
            self.assertEqual(read_json(Path(tmp)/'compatibility.json')['opt_support'], 'UNKNOWN')
            self.assertFalse(read_json(Path(tmp)/'environment.json')['torchac_cuda_build_attempted'])

    def test_static_audit_detects_opt_and_self_fit_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'LMCache/lmcache/storage_backend/serde'
            root.mkdir(parents=True)
            (root/'cachegen_encoder.py').write_text('from_model_name(metadata.model_name)\ncalculate_cdf(new_key)\ncalculate_cdf(new_value)\n')
            (root/'cachegen_decoder.py').write_text('')
            (root/'cachegen_basics.py').write_text('raise ValueError("Model is not supported")')
            report = audit(tmp)
            self.assertEqual(report['opt_support'], 'UNSUPPORTED')
            self.assertEqual(report['cdf_policy'], 'INPUT_SELF_FITTED')
            self.assertFalse(report['c1_full_available'])
            self.assertEqual(len(report['source_hashes']), 3)

    def test_production_has_no_harness_import_or_dependency(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root/'src/semcache').rglob('*.py'):
            if 'experiments/cachegen' in path.as_posix():
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] + [getattr(node, 'module', '') or '']
                    self.assertFalse(any('cachegen' in name.lower() or 'lmcache' in name.lower() for name in names), path)
        self.assertNotIn('lmcache', (root/'pyproject.toml').read_text().lower())
        self.assertNotIn('cachegen', (root/'pyproject.toml').read_text().lower())

    def test_capture_manifest_partition_tampering_rejected(self):
        plan = {'snips': sample_partitions(records(), 8)}
        manifest = dict(status='CAPTURED', sampling=plan, sampling_sha256=digest(plan), blocks=[])
        verify_capture(manifest)
        manifest['sampling']['snips']['evaluation'][0]['query_text'] = 'tampered'
        with self.assertRaises(ValueError):
            verify_capture(manifest)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; no installation permitted')
class TensorTests(unittest.TestCase):
    def test_real_opt_layout_sizes_and_values(self):
        import torch
        for tokens in (3, 10):
            x = (torch.arange(32*tokens*2560) % 997).reshape(32, tokens, 2560).half()
            self.assertTrue(torch.equal(x, from_heads(to_heads(x))))
            self.assertEqual(to_heads(x).shape, (32,32,tokens,80))
            self.assertEqual(x.numel()*x.element_size(), raw_bytes(tokens)['raw_q_bytes'])

    def test_int8_scale_bytes_zero_vectors_and_error_bound(self):
        import torch
        codec = Baseline('UNIFORM_INT8')
        x = torch.tensor([[[0., 1., -1., .5], [0., 0., 0., 0.]]], dtype=torch.float16)
        encoded = codec.encode(x, x)
        self.assertEqual(codec.sizes(encoded), (16, 16))
        k, v = codec.decode(encoded)
        self.assertTrue(torch.isfinite(k).all())
        self.assertTrue(torch.equal(k[:,1], x[:,1]))
        self.assertLessEqual((x-k).abs().max().item(), 1/127)
        self.assertTrue(torch.equal(k, v))
        raw = Baseline('FP16_RAW')
        kr, vr = raw.decode(raw.encode(x, x))
        self.assertTrue(torch.equal(x, kr) and torch.equal(x, vr))
        self.assertEqual(raw.sizes(raw.encode(x,x)), (32, 0))

    def test_timing_repeat_contract(self):
        import torch
        from semcache.experiments.cachegen.harness import measured_baseline
        x = torch.ones(1,3,4, dtype=torch.float16)
        encoded, decoded, times, wall, cuda = measured_baseline(Baseline('UNIFORM_INT8'), x, x, 'cpu')
        self.assertEqual(len(wall), 20)
        self.assertEqual(cuda, [])
        self.assertIsNone(times['encode_cuda_ms'])
        self.assertTrue(torch.equal(x, decoded[0]))


if __name__ == '__main__':
    unittest.main()
