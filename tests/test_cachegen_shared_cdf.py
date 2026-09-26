"""C1.5 tests: no pretrained models, downloads, CUDA, or official-code imports."""
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen.shared.core import (
    CONFIG, MODES, Profile, SUPPORT, TOTAL, arithmetic_encode, arithmetic_decode,
    cdf_from_counts, decode, encode, fit_profiles, inspect_block, map_symbols,
    partition_ids, pool_accounting, stream_counts, unmap_symbols, validate_cdf)
from semcache.experiments.cachegen.shared.harness import (
    benchmark, load_profiles, measure, profile, protect_output, quality_selection)
from semcache.experiments.cachegen.shared.source_audit import audit, REVISION
from semcache.experiments.cachegen.common import read_json


def blocks():
    return [dict(block_id='c1', partition='calibration'), dict(block_id='e1', partition='evaluation'),
            dict(block_id='c2', partition='calibration')]


def histograms(block):
    h = [[[0]*SUPPORT for _ in range(32)] for _ in 'kv']
    for c in range(2):
        for layer in range(32):
            h[c][layer][127] = 1000+layer
            h[c][layer][layer+c] = 10 if block['block_id'] == 'c1' else 50
    return h


def profiles():
    return fit_profiles(blocks(), histograms)[0]


def prepared(model, tokens, hidden=2):
    rng = random.Random(17)
    streams = tuple(bytes(rng.randrange(255) if i % 5 == 0 else 127 for i in range(n))
                    for n in stream_counts(model, (32, tokens, hidden)))
    scales = struct.pack('<'+'f'*(64*tokens), *[.25]*(64*tokens))
    return streams, scales, (32, tokens, hidden)


class ProfileTests(unittest.TestCase):
    def test_disjoint_nonempty_unique_partitions(self):
        self.assertEqual(partition_ids(blocks()), {'calibration': ['c1', 'c2'], 'evaluation': ['e1']})
        for data in (blocks()+[dict(block_id='c1', partition='evaluation')], blocks()[:1],
                     [dict(block_id='x', partition='test')]):
            with self.assertRaises(ValueError):
                partition_ids(data)

    def test_fit_only_calibration_deterministic_and_no_eval_reads(self):
        loaded = []
        def loader(block):
            self.assertEqual(block['partition'], 'calibration')
            loaded.append(block['block_id'])
            return histograms(block)
        models, fitted, counts = fit_profiles(blocks(), loader)
        self.assertEqual(loaded, ['c1', 'c2'])
        self.assertEqual(fitted, loaded)
        changed = list(reversed(blocks()))
        changed[1]['dataset'] = 'evaluation-label-must-not-matter'
        again = fit_profiles(changed, loader)[0]
        self.assertEqual(models, again)
        self.assertEqual(counts[MODES[0]][0][127], sum((1000+i)*2 for i in range(32)))
        for mode, model in models.items():
            self.assertEqual(model.to_bytes(), again[mode].to_bytes())
            self.assertEqual(Profile.from_bytes(model.to_bytes(), model.sha256), model)
            with self.assertRaises(FrozenInstanceError):
                model.mode = 'changed'
            with self.assertRaises(TypeError):
                model.cdfs[0][0] = 1

    def test_fixed_layer_boundaries_and_separate_k_v(self):
        models, _, counts = fit_profiles(blocks(), histograms)
        self.assertEqual(len(models[MODES[0]].cdfs), 2)
        self.assertEqual(len(models[MODES[1]].cdfs), 6)
        for group, (start, end) in enumerate(((0, 11), (11, 22), (22, 32))):
            for component in range(2):
                self.assertEqual(counts[MODES[1]][group*2+component][127],
                                 sum((1000+i)*2 for i in range(start, end)))

    def test_mapping_full_alphabet_and_invalid_negative_128(self):
        values = list(range(-127, 128))
        self.assertEqual(map_symbols(values), bytes(range(255)))
        self.assertEqual(unmap_symbols(map_symbols(values)), values)
        for invalid in ([-128], [128], [1.5]):
            with self.assertRaises(ValueError):
                map_symbols(invalid)
        with self.assertRaises(ValueError):
            unmap_symbols(b'\xff')

    def test_cdf_support_monotonicity_smoothing_and_large_counts(self):
        for counts in ([0]*255, [10**15]+[0]*254, list(range(255))):
            cdf = cdf_from_counts(counts)
            validate_cdf(cdf)
            self.assertEqual(cdf[-1], TOTAL)
            self.assertTrue(all(b-a >= 1 for a, b in zip(cdf, cdf[1:])))
            symbols = bytes(range(255))  # includes symbols unseen in calibration
            self.assertEqual(arithmetic_decode(arithmetic_encode(symbols, cdf), 255, cdf), symbols)
        for bad in ((0,)*256, tuple(range(256))):
            with self.assertRaises(ValueError):
                validate_cdf(bad)

    def test_profile_sha_and_format_fail_closed(self):
        model = profiles()[MODES[0]]
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            Profile.from_bytes(model.to_bytes()+b'x', model.sha256)
        malformed = b'bad'
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            Profile.from_bytes(malformed, hashlib.sha256(malformed).hexdigest())


class ArithmeticTests(unittest.TestCase):
    def test_t3_t10_global_layergroup_exact_roundtrip_no_profile_update(self):
        for model in profiles().values():
            frozen = model.to_bytes()
            for t in (3, 10):
                with self.subTest(mode=model.mode, t=t):
                    data = prepared(model, t)
                    blob = encode(model, *data)
                    self.assertEqual(decode(blob, model), data)
                    self.assertEqual(blob, encode(model, *data))
                    self.assertEqual(model.to_bytes(), frozen)
                    shape, scales, payloads, sizes = inspect_block(blob, model)
                    self.assertEqual(sizes['scale_metadata_bytes'], 2*32*t*4)
                    self.assertEqual(len(blob), sizes['encoded_payload_bytes']+sizes['local_metadata_bytes'])
                    self.assertEqual(scales, data[1])

    def test_random_skewed_streams_and_underflow_renormalization(self):
        rng = random.Random(31)
        for trial in range(12):
            counts = [rng.randrange(10000) if rng.random() < .2 else 0 for _ in range(255)]
            cdf = cdf_from_counts(counts)
            for data in (bytes([trial]*100), bytes(rng.randrange(255) for _ in range(500)), b'\x00', b'\xfe'):
                self.assertEqual(arithmetic_decode(arithmetic_encode(data, cdf), len(data), cdf), data)

    def test_corruption_truncation_wrong_profile_and_trailing_data(self):
        model = profiles()[MODES[0]]
        blob = encode(model, *prepared(model, 3))
        for i in (0, len(blob)//2, len(blob)-1):
            broken = bytearray(blob)
            broken[i] ^= 1
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                decode(bytes(broken), model)
        for broken in (blob[:-1], blob[:30], blob+b'\0'):
            with self.assertRaises(ValueError):
                decode(broken, model)
        other = Profile(MODES[0], (cdf_from_counts([0]*255),)*2)
        with self.assertRaisesRegex(ValueError, 'profile SHA256'):
            decode(blob, other)

    def test_shared_profile_charged_once_local_metadata_per_block_q_uncoded(self):
        model = profiles()[MODES[1]]
        rows = []
        for t in (3, 10):
            data = prepared(model, t)
            sizes = inspect_block(encode(model, *data), model)[3]
            rows.append(dict(raw_q_bytes=32*t*2*2, raw_kv_bytes=32*t*2*4, **sizes))
        total = pool_accounting(rows, len(model.to_bytes()))
        self.assertEqual(total['encoded_kv_pool_bytes'], len(model.to_bytes())+
                         sum(r['encoded_payload_bytes']+r['local_metadata_bytes'] for r in rows))
        self.assertEqual(total['encoded_semcache_pool_bytes'], total['encoded_kv_pool_bytes']+
                         sum(r['raw_q_bytes'] for r in rows))
        self.assertEqual(total['shared_profile_bytes'], model.logical_bytes+12)
        # Core API has only K/V streams and scale bytes, never Q or dataset.
        import inspect
        self.assertEqual(list(inspect.signature(encode).parameters), ['profile', 'streams', 'scales', 'shape'])


class HarnessTests(unittest.TestCase):
    def test_output_guard_c1_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c1 = root/'c1'
            c1.mkdir()
            for path in (c1, c1/'sub', root):
                with self.assertRaises(ValueError):
                    protect_output(path, c1)
            link = root/'c15'
            link.symlink_to(c1, target_is_directory=True)
            with self.assertRaises(ValueError):
                protect_output(link, c1)
            self.assertEqual(protect_output(root/'new', c1), root/'new')

    def test_timing_protocol_separate_calls_and_summary(self):
        calls = []
        value, stats, times = measure(lambda: calls.append(1) or b'data')
        self.assertEqual(len(calls), 25)
        self.assertEqual(len(times), 20)
        self.assertEqual(set(stats), {'mean', 'median', 'p95'})
        self.assertEqual(value, b'data')

    def test_quality_uses_recorded_c1_ids_not_new_selection(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chosen = [dict(block_id=f'e{i}', partition='evaluation', dataset=d, token_group_size=t)
                      for i, (d, t) in enumerate((('snips', 3), ('snips', 10), ('multiwoz', 3), ('multiwoz', 10)))]
            with (root/'c1_quality.csv').open('w') as f:
                w = csv.DictWriter(f, fieldnames=['fixture_id', 'compression_mode', 'status'])
                w.writeheader()
                for b in chosen:
                    for mode in ('NATIVE', 'FP16_RAW', 'UNIFORM_INT8'):
                        w.writerow(dict(fixture_id=b['block_id'], compression_mode=mode, status='MEASURED'))
            self.assertEqual(quality_selection(root, [dict(chosen[0], block_id='unselected')]+chosen), chosen)
            with self.assertRaises(ValueError):
                quality_selection(root, chosen[:-1])

    def test_pinned_source_audit_uses_no_official_import_and_records_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit(Path(tmp)/'not-mounted')
        self.assertEqual(report['official_revision'], REVISION)
        self.assertFalse(report['official_code_reused'])
        self.assertIsNone(report['imported_arithmetic_coder_path'])
        self.assertIn('MAX_LP=48', report['reason'])

    def test_profile_stage_fits_only_calibration_and_freezes_provenance(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp)/'c1', Path(tmp)/'c15'
            root.mkdir()
            out.mkdir()
            capture = dict(blocks=[dict(b, dataset='snips', token_group_size=3) for b in blocks()],
                           model_metadata={'resolved_model_revision': 'pinned'})
            ids = partition_ids(blocks())
            hashes = {'capture_manifest.json': 'literal_file_sha'}
            @contextmanager
            def input_stub(args):
                yield root, out, capture, {}, ids, [], hashes
            loaded = []
            def fixture_loader(root, block):
                loaded.append(block['block_id'])
                self.assertEqual(block['partition'], 'calibration')
                return dict(k=histograms(block), v=None)
            with patch('semcache.experiments.cachegen.shared.harness.inputs', input_stub), \
                 patch('semcache.experiments.cachegen.shared.harness.load_fixture', fixture_loader), \
                 patch('semcache.experiments.cachegen.shared.harness.quantize', lambda k, v: k), \
                 patch('semcache.experiments.cachegen.shared.harness.histograms', lambda x: x):
                args = SimpleNamespace(cachegen_repo=Path(tmp)/'missing')
                profile(args)
                self.assertEqual(loaded, ['c1', 'c2'])
                models, pm = load_profiles(out, hashes, ids)
                self.assertEqual(pm['fitted_block_ids'], ids['calibration'])
                self.assertEqual(pm['evaluation_block_ids'], ids['evaluation'])
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    profile(args)
            # Execute the real benchmark orchestration with literal byte streams.
            # Tensor-only boundaries are mocked; entropy/timing/persistence run.
            evaluated = []
            def evaluation_loader(root, block):
                self.assertEqual(block['partition'], 'evaluation')
                evaluated.append(block['block_id'])
                return dict(q='untouched', k='k', v='v')
            def check(model, encoded, blob):
                self.assertEqual(decode(blob, model), prepared(model, 3))
            with patch('semcache.experiments.cachegen.shared.harness.inputs', input_stub), \
                 patch('semcache.experiments.cachegen.shared.harness.load_fixture', evaluation_loader), \
                 patch('semcache.experiments.cachegen.shared.harness.quantize', return_value='quantized'), \
                 patch('semcache.experiments.cachegen.shared.harness.Baseline.decode', return_value=('k', 'v')), \
                 patch('semcache.experiments.cachegen.shared.harness.Baseline.sizes', return_value=(491520, 768)), \
                 patch('semcache.experiments.cachegen.shared.harness.existing_uniform') as existing, \
                 patch('semcache.experiments.cachegen.shared.harness.prepare', side_effect=lambda m, q: prepared(m, 3)), \
                 patch('semcache.experiments.cachegen.shared.harness.check_roundtrip', side_effect=check), \
                 patch('builtins.print'):
                benchmark(args)
                existing.assert_called_once()
            self.assertEqual(evaluated, ['e1'])
            bm = read_json(out/'benchmark_manifest.json')
            self.assertEqual(len(bm['bitstreams']), 2)
            self.assertEqual({b['block_id'] for b in bm['bitstreams']}, {'e1'})
            self.assertTrue(bm['profile_unchanged_after_evaluation'])
            self.assertEqual(load_profiles(out, hashes, ids)[0], models)
            timings = read_json(out/'timing_repeats.json')
            self.assertEqual(len(timings['repeats']), 2)
            for repeat in timings['repeats']:
                self.assertEqual(len(repeat['encode_wall_ms']), 20)
                self.assertEqual(len(repeat['decode_wall_ms']), 20)
            with patch('semcache.experiments.cachegen.shared.harness.inputs', input_stub):
                with self.assertRaisesRegex(ValueError, 'Benchmark output exists'):
                    benchmark(args)
            with self.subTest('profile corruption after benchmark'):
                target = out/pm['profiles'][MODES[0]]['file']
                target.write_bytes(target.read_bytes()+b'x')
                with self.assertRaisesRegex(ValueError, 'Profile SHA256'):
                    load_profiles(out, hashes, ids)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; no installation/download')
class TensorTests(unittest.TestCase):
    def test_exact_existing_uniform_reconstruction_q_untouched(self):
        import torch
        from semcache.experiments.cachegen.codecs import Baseline
        from semcache.experiments.cachegen.shared.tensors import quantize, prepare, check_roundtrip
        torch.manual_seed(9)
        for t in (3, 10):
            q, k, v = (torch.randn(32, t, 8).half() for _ in 'qkv')
            k[:, 0] = 0  # zero-vector scale fallback stays exactly one
            saved_q = q.clone()
            encoded = quantize(k, v)
            baseline = Baseline('UNIFORM_INT8').decode(encoded)
            for model in profiles().values():
                blob = encode(model, *prepare(model, encoded))
                actual = check_roundtrip(model, encoded, blob)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, baseline)))
                self.assertTrue(torch.equal(q, saved_q))
                self.assertEqual(q.dtype, torch.float16)
                self.assertTrue(torch.equal(encoded[0][1][:, 0], torch.ones(32, 1)))


if __name__ == '__main__':
    unittest.main()
