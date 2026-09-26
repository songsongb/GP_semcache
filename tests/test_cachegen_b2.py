"""Real CPU arithmetic/framing with tiny fixtures; no models, downloads or GPU."""
from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import random
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen import b1
from semcache.experiments.cachegen.b2 import format as fmt, harness as h
from semcache.experiments.cachegen.shared import core
from semcache.experiments.cachegen.shared.device_contract import make_contract

CONTRACT = make_contract('cuda', cuda_available=True, cuda_device_count=1)
SHAPE = (32, 10, 1)
SCALES = struct.pack('<640f', *([.25]*640))


def packed(seed=0):
    rng = random.Random(seed)
    k, v = [], []
    for layer in range(32):
        a = rng.randrange(255)
        k.extend([a]+[(a+i) % 255 for i in range(1, 10)])
        v.extend(rng.randrange(255) for _ in range(10))
    return bytes(k), bytes(v)


def blocks():
    return [dict(block_id=f'{s}_{d}', partition=s, dataset=d, token_group_size=10,
                 hidden_dim=1, query_id=f'{s}_{d}', file=f'{s}_{d}.pt')
            for s in ('calibration', 'evaluation') for d in ('snips', 'multiwoz')]


def profiles():
    return fmt.fit(blocks(), lambda b: (packed(), SCALES, SHAPE))[0]


class FormatTests(unittest.TestCase):
    def test_calibration_only_fit_no_evaluation_loader_equal_four_models(self):
        loaded = []
        def loader(b):
            self.assertEqual(b['partition'], 'calibration')
            loaded.append(b['block_id'])
            return packed(), SCALES, SHAPE
        models, counts, fitted = fmt.fit(blocks(), loader)
        self.assertEqual(loaded, [b['block_id'] for b in blocks()[:2]])
        self.assertEqual(fitted, loaded)
        self.assertEqual(len(models), 3)
        for mode, model in models.items():
            self.assertEqual(len(model.cdfs), 4)
            self.assertEqual(len(model.to_bytes()), 4108)
            self.assertEqual(fmt.Profile.from_bytes(model.to_bytes(), model.sha256), model)
            self.assertEqual([sum(c) for c in counts[mode]], [64, 576, 64, 576])
            for cdf, hist in zip(model.cdfs, counts[mode]):
                self.assertEqual(cdf, core.cdf_from_counts(hist))
            with self.assertRaises(FrozenInstanceError): model.mode = fmt.MODES[0]
        with self.assertRaises(ValueError): fmt.Profile(fmt.MODES[0], (models[fmt.MODES[0]].cdfs[0],)*2)
        changed = [dict(b) for b in blocks()]
        changed[-1]['block_id'] = 'different-evaluation-does-not-affect-calibration'
        self.assertEqual(fmt.fit(changed, loader)[0], models)

    def test_raw_unchanged_primary_exact_B1_hybrid_K_only(self):
        domains = packed()
        raw = fmt.streams_from_domains(fmt.MODES[0], domains, SHAPE)
        self.assertEqual(raw, b1.role_domains(domains[0], SHAPE)+b1.role_domains(domains[1], SHAPE))
        primary = fmt.streams_from_domains(fmt.MODES[1], domains, SHAPE)
        self.assertEqual(primary, b1.role_domains(b1.transform(domains[0], SHAPE), SHAPE)+
                         b1.role_domains(b1.transform(domains[1], SHAPE), SHAPE))
        hybrid = fmt.streams_from_domains(fmt.MODES[2], domains, SHAPE)
        self.assertEqual(hybrid[:2], primary[:2])
        self.assertEqual(hybrid[2:], raw[2:])
        for mode in fmt.MODES:
            self.assertEqual(fmt.domains_from_streams(mode, fmt.streams_from_domains(mode, domains, SHAPE), SHAPE), domains)

    def test_arithmetic_exact_scales_byte_sizes_core_functions_only(self):
        for mode, model in profiles().items():
            streams = fmt.streams_from_domains(mode, packed(), SHAPE)
            with patch.object(core, 'arithmetic_encode', wraps=core.arithmetic_encode) as enc, \
                 patch.object(core, 'arithmetic_decode', wraps=core.arithmetic_decode) as dec:
                blob = fmt.encode(model, streams, SCALES, SHAPE)
                self.assertEqual(fmt.decode(blob, model), (streams, SCALES, SHAPE))
                self.assertEqual((enc.call_count, dec.call_count), (4, 4))
            shape, scales, payloads, sizes = fmt.inspect(blob, model)
            self.assertEqual(shape, SHAPE)
            self.assertEqual(scales, SCALES)
            self.assertEqual(sizes['scale_metadata_bytes'], 2560)
            self.assertEqual(sizes['local_transform_metadata_bytes'], 99)
            self.assertEqual(sizes['local_metadata_bytes'], 2659)
            self.assertEqual(sizes['total_payload_bytes'], sum(map(len, payloads)))
            self.assertEqual(len(blob), sizes['total_payload_bytes']+2659)
            for bad in (blob[:-1], blob+b'x', blob[:50]):
                with self.assertRaises(ValueError): fmt.decode(bad, model)
            with self.assertRaisesRegex(ValueError, 'profile'):
                fmt.decode(blob, profiles()[fmt.MODES[(fmt.MODES.index(mode)+1) % 3]])

    def test_T3_and_invalid_domain_profile_rejected(self):
        with self.assertRaisesRegex(ValueError, 'T=3'):
            fmt.stream_counts((32, 3, 1))
        with self.assertRaises(ValueError): fmt.streams_from_domains(fmt.MODES[0], (b'\xff'*320,)*2, SHAPE)
        with self.assertRaises(ValueError): fmt.Profile.from_bytes(profiles()[fmt.MODES[0]].to_bytes(), 'bad')

    def test_posthoc_label_never_primary(self):
        self.assertEqual(fmt.labels(fmt.MODES[1])['experiment_role'], 'PRIMARY_B2')
        self.assertEqual(fmt.labels(fmt.MODES[2])['result_classification'], 'POST_HOC_EXPLORATORY')
        self.assertEqual(fmt.labels(fmt.MODES[2])['predeclaration'], 'NOT_PREDECLARED_PRIMARY')


class BoundaryTests(unittest.TestCase):
    def test_cuda_quantization_contract_forwarded_and_T3_rejected_before_load(self):
        block = blocks()[2]
        with patch.object(h, 'load_fixture', return_value=dict(k='K', v='V', q='Q')) as load, \
             patch.object(h.tensors, 'quantize', return_value='encoded') as quantize, \
             patch.object(h, 'pack_encoded', return_value=(packed(), SCALES, SHAPE)):
            self.assertEqual(h.quantized_fixture(None, block, CONTRACT), (packed(), SCALES, SHAPE, 'encoded'))
            quantize.assert_called_once_with('K', 'V', device=CONTRACT['quantization_device_resolved'])
            with self.assertRaises(ValueError): h.quantized_fixture(None, dict(block, token_group_size=3), CONTRACT)
            self.assertEqual(load.call_count, 1)

    def test_real_pair_gates_roundtrip_inverse_scales_saved_C1_and_device(self):
        # Tensor boundary stubs allow testing the real saved-C1 gate without a GPU.
        class Tensor:
            def __init__(self, value, dtype): self.value, self.dtype = value, dtype
        encoded = tuple((Tensor(i, 'int8'), Tensor(.25, 'float32')) for i in range(2))
        actual = (Tensor(1., 'float16'), Tensor(2., 'float16'))
        saved = dict(zip('kv', actual))
        stub = SimpleNamespace(Tensor=Tensor, float16='float16',
            equal=lambda a, b: a.value == b.value and a.dtype == b.dtype,
            load=lambda *a, **k: saved)
        model = profiles()[fmt.MODES[1]]
        for failure in (None, 'arithmetic', 'inverse', 'scale_bytes', 'scale_tensor', 'saved'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                (root/'saved.pt').write_bytes(b'C1_RECONSTRUCTION')
                block = blocks()[2]
                ctx = dict(root=root, contract=CONTRACT, rec=dict(reconstructed=[dict(
                    block_id=block['block_id'], compression_mode='UNIFORM_INT8', file='saved.pt',
                    sha256=h.file_hash(root/'saved.pt'))]))
                stack.enter_context(patch.dict(sys.modules, {'torch': stub}))
                stack.enter_context(patch.object(h, 'quantized_fixture', return_value=(packed(), SCALES, SHAPE, encoded)))
                restored = encoded if failure != 'scale_tensor' else ((encoded[0][0], Tensor(.250001, 'float32')), encoded[1])
                stack.enter_context(patch.object(h, 'restored_encoded', return_value=restored))
                reconstruct = stack.enter_context(patch.object(h.tensors, 'reconstruct', return_value=actual))
                stack.enter_context(patch.object(h.shared_harness, 'failure_report',
                    return_value=dict(components={}, block_id=block['block_id'])))
                check = stack.enter_context(patch.object(h.shared_harness, 'existing_uniform', wraps=h.shared_harness.existing_uniform))
                saved['k'] = actual[0] if failure != 'saved' else Tensor(1.000001, 'float16')
                if failure in ('arithmetic', 'scale_bytes'):
                    original_decode = fmt.decode
                    def corrupt_decode(blob, profile):
                        streams, scales, shape = original_decode(blob, profile)
                        if failure == 'arithmetic': streams = (b'\0'*len(streams[0]),)+streams[1:]
                        else: scales = b'\0'*len(scales)
                        return streams, scales, shape
                    stack.enter_context(patch.object(fmt, 'decode', side_effect=corrupt_decode))
                if failure == 'inverse':
                    stack.enter_context(patch.object(fmt, 'domains_from_streams', return_value=(b'\0'*320,)*2))
                if failure:
                    with self.assertRaises(ValueError): h.evaluate_pair(ctx, block, model, root/'out.bin')
                    self.assertEqual(check.call_count, int(failure == 'saved'))
                else:
                    row = h.evaluate_pair(ctx, block, model, root/'out.bin')
                    self.assertTrue(all(row[k] for k in h.CORRECTNESS))
                    self.assertEqual(row['actual_bitstream_bytes'], (root/'out.bin').stat().st_size)
                    check.assert_called_once()
                if failure in (None, 'saved'):
                    reconstruct.assert_called_once_with(encoded, device=CONTRACT['reconstruction_device'])
                else: reconstruct.assert_not_called()

    def test_A_reference_reads_matching_T10_rows_and_reconciles_population_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'full_storage').mkdir()
            evaluation = blocks()[2:]
            for mode, filename in h.shared_harness.PROFILE_FILES.items():
                (root/filename).write_bytes(b'PROFILE'+mode.encode())
            raw_rows, summary = [], []
            for bb in evaluation:
                for mode in core.MODES:
                    raw_rows.append(dict(block_id=bb['block_id'], dataset=bb['dataset'], token_group_size=10,
                        compression_mode=mode, **h.raw_bytes(10, hidden_dim=1), encoded_payload_bytes=512,
                        local_metadata_bytes=2651, status='COMPLETED', symbol_roundtrip_exact=True,
                        saved_c1_reconstruction_exact=True))
            for dataset, stratum in [('ALL', 'T=10'), ('snips', 'SNIPS/T10'), ('multiwoz', 'MultiWOZ/T10')]:
                n = sum(dataset == 'ALL' or bb['dataset'] == dataset for bb in evaluation)
                raw = h.raw_bytes(10, hidden_dim=1)
                for mode in ('UNIFORM_INT8', *core.MODES):
                    payload = n*(raw['raw_kv_bytes']//2 if mode == 'UNIFORM_INT8' else 512)
                    local = n*(2560 if mode == 'UNIFORM_INT8' else 2651)
                    shared = 0 if mode == 'UNIFORM_INT8' else (root/h.shared_harness.PROFILE_FILES[mode]).stat().st_size
                    summary.append(dict(stratum=stratum, compression_mode=mode, block_count=n,
                        raw_kv_pool_bytes=n*raw['raw_kv_bytes'], raw_semcache_pool_bytes=n*raw['raw_qkv_bytes'],
                        encoded_payload_pool_bytes=payload, local_metadata_pool_bytes=local, shared_profile_bytes=shared,
                        encoded_kv_pool_bytes=payload+local+shared,
                        encoded_semcache_pool_bytes=n*raw['raw_q_bytes']+payload+local+shared))
            # A deliberately different mixed-T overall pool must never be selected.
            summary.append(dict(summary[0], stratum='ALL', encoded_kv_pool_bytes=9999999))
            sp, rp = root/'full_storage/c15_full_summary.csv', root/'full_storage/c15_full_block_raw.csv'
            h.atomic_csv(sp, summary); h.atomic_csv(rp, raw_rows)
            refs = h.reference_inputs(root, evaluation)
            self.assertEqual(len(refs['source_rows']), 9)
            self.assertEqual(refs['source_sha256'], h.file_hash(sp))
            self.assertEqual(len(refs['blocks']), 6)
            raw_rows[0]['encoded_payload_bytes'] += 1
            h.atomic_csv(rp, raw_rows)
            with self.assertRaisesRegex(ValueError, 'reconciliation'): h.reference_inputs(root, evaluation)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; no installation or download')
class RealTensorTests(unittest.TestCase):
    def test_original_C1_scales_and_tensor_symbols_preserved_all_modes(self):
        import torch
        from semcache.experiments.cachegen.codecs import Baseline
        generator = torch.Generator().manual_seed(82)
        qkv = {n: torch.randn(32, 10, 3, generator=generator).half() for n in 'qkv'}
        q = qkv['q'].clone()
        original = Baseline('UNIFORM_INT8').encode(qkv['k'], qkv['v'])
        domains, scales, shape = h.pack_encoded(original)
        for mode in fmt.MODES:
            inverse = fmt.domains_from_streams(mode, fmt.streams_from_domains(mode, domains, shape), shape)
            recovered = h.restored_encoded(inverse, scales, shape)
            for (s, c), (rs, rc) in zip(original, recovered):
                self.assertTrue(torch.equal(s, rs))
                self.assertTrue(torch.equal(c, rc))
            for a, b in zip(Baseline('UNIFORM_INT8').decode(original), Baseline('UNIFORM_INT8').decode(recovered)):
                self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(q, qkv['q']))
        self.assertEqual(qkv['q'].dtype, torch.float16)


class JobTests(unittest.TestCase):
    @contextmanager
    def job(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root, a, b, out = (Path(tmp)/n for n in ('c1', 'a', 'b1', 'b2'))
            for p in (root, a, b, out): p.mkdir()
            for p in (root/'capture_manifest.json', a/'manifest.json', a/'shared_cdf_global.bin',
                      a/'smoke.json', b/'manifest.json', b/'b1_roundtrip.json'):
                p.write_bytes(b'FROZEN_INPUT')
            frozen = {str(p): h.file_hash(p) for parent in (root, a, b) for p in parent.iterdir()}
            bs = blocks()
            refs = dict(source_sha256='reference', source_rows=[], blocks=[])
            for dataset, stratum in (('ALL', 'T=10'), ('snips', 'SNIPS/T10'), ('multiwoz', 'MultiWOZ/T10')):
                for mode in ('UNIFORM_INT8', *core.MODES):
                    refs['source_rows'].append(dict(stratum=stratum, compression_mode=mode,
                        shared_profile_bytes='0' if mode == 'UNIFORM_INT8' else '2060' if mode == core.MODES[0] else '6156'))
            for bb in bs[2:]:
                for mode in ('UNIFORM_INT8', *core.MODES):
                    refs['blocks'].append(dict(block_id=bb['block_id'], dataset=bb['dataset'], mode=mode,
                        **h.raw_bytes(10, hidden_dim=1), total_payload_bytes=320, local_metadata_bytes=2560))
            base = dict(config=h.CONFIG, config_sha256=h.digest(h.CONFIG), device_contract=CONTRACT,
                input_sha256=frozen, implementation_sha256={}, runtime={'test': 'fixed'},
                calibration_block_ids=[bb['block_id'] for bb in bs[:2]], evaluation_block_ids=[bb['block_id'] for bb in bs[2:]],
                populations={s: dict(snips=1, multiwoz=1, ALL=2) for s in ('calibration', 'evaluation')})
            ctx = dict(root=root, calibration=bs[:2], evaluation=bs[2:], rec={}, contract=CONTRACT, base=base, references=refs)
            args = SimpleNamespace(capture_manifest=root/'capture_manifest.json', c15a_dir=a, b1_dir=b,
                                   output_dir=out, command='profile-fit')
            stack.enter_context(patch.object(h, 'bind', return_value=ctx))
            stack.enter_context(patch.object(h, 'FULL_PAIRS', 6))
            stack.enter_context(patch.object(h, 'verify_tensor_files'))
            stack.enter_context(patch.object(h.device_contract, 'resolve_contract', return_value=CONTRACT))
            stack.enter_context(patch.object(h.shared_harness, 'measure', side_effect=AssertionError('No timing repeats')))
            loaded = []
            def fixture(root, bb, contract):
                loaded.append(bb['block_id'])
                self.assertEqual(bb['token_group_size'], 10)
                self.assertEqual(contract, CONTRACT)
                return packed(), SCALES, SHAPE, None
            stack.enter_context(patch.object(h, 'quantized_fixture', side_effect=fixture))
            calls = []
            def pair(ctx, bb, model, path):
                calls.append((bb['block_id'], model.mode))
                domains = packed()
                streams = fmt.streams_from_domains(model.mode, domains, SHAPE)
                h.atomic_bytes(path, fmt.encode(model, streams, SCALES, SHAPE))
                decoded, scales, shape = fmt.decode(path.read_bytes(), model)
                self.assertEqual((decoded, scales, shape), (streams, SCALES, SHAPE))
                self.assertEqual(fmt.domains_from_streams(model.mode, decoded, shape), domains)
                return dict(block_id=bb['block_id'], dataset=bb['dataset'], query_id=bb['query_id'], token_group_size=10,
                    **h.raw_bytes(10, hidden_dim=1), **fmt.inspect(path.read_bytes(), model)[3], mode=model.mode,
                    profile_bytes_reference=len(model.to_bytes()), symbol_count=640,
                    **{k: True for k in h.CORRECTNESS}, **fmt.labels(model.mode), status='COMPLETED', Q_storage='FP16',
                    bitstream_sha256=h.file_hash(path), actual_bitstream_bytes=path.stat().st_size)
            spy = stack.enter_context(patch.object(h, 'evaluate_pair', side_effect=pair))
            stack.enter_context(patch('builtins.print'))
            yield args, ctx, out, calls, loaded, spy, pair
            h.assert_unchanged(frozen)

    def fit_smoke(self, args):
        h.main(['profile-fit', '--capture-manifest', str(args.capture_manifest), '--c15a-dir', str(args.c15a_dir),
                '--b1-dir', str(args.b1_dir), '--output-dir', str(args.output_dir)])
        args.command = 'smoke'
        h.evaluate(args, args.output_dir)

    def test_fit_smoke_full_single_pass_resume_accounting_provenance_immutability(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            with patch.object(core, 'arithmetic_encode', wraps=core.arithmetic_encode) as enc, \
                 patch.object(core, 'arithmetic_decode', wraps=core.arithmetic_decode) as dec:
                self.fit_smoke(args)
                self.assertEqual(loaded, ctx['base']['calibration_block_ids'])
                self.assertEqual((enc.call_count, dec.call_count), (24, 24))
                smoke = h.read_json(out/'smoke/manifest.json')
                self.assertFalse(smoke['primary_result_eligible'])
                self.assertEqual(smoke['run_classification'], 'DIAGNOSTIC_ONLY')
                args.command = 'storage-full'; h.evaluate(args, out)
                self.assertEqual((enc.call_count, dec.call_count), (48, 48))
                h.evaluate(args, out)
                self.assertEqual((enc.call_count, dec.call_count), (48, 48))
            state = h.read_json(out/'full_storage/manifest.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertTrue(state['primary_result_eligible'])
            self.assertEqual(state['completed_block_mode_pairs'], 6)
            self.assertEqual(calls[:6], calls[6:])
            rows = h.load_completed(out/'full_storage', state['run_contract_sha256'], ctx['evaluation'], profiles())
            summary = h.summaries(list(rows.values()), ctx['references'], profiles(), smoke=False)
            self.assertEqual(len(summary), 18)
            for mode in fmt.MODES:
                overall = next(r for r in summary if r['mode'] == mode and r['dataset'] == 'ALL')
                pool = [r for r in rows.values() if r['mode'] == mode]
                self.assertEqual(overall['shared_profile_bytes'], 4108)
                self.assertEqual(overall['encoded_kv_pool_bytes'], 4108+sum(r['actual_bitstream_bytes'] for r in pool))
                self.assertEqual(overall['encoded_semcache_pool_bytes'], overall['encoded_kv_pool_bytes']+sum(r['raw_q_bytes'] for r in pool))
                self.assertEqual(overall['k_payload_bytes']+overall['v_payload_bytes'], overall['total_payload_bytes'])
                self.assertEqual(overall['primary_result_eligible'], mode == fmt.MODES[1])
                strata = [r for r in summary if r['mode'] == mode and r['dataset'] != 'ALL']
                for field in ('block_count', 'raw_kv_pool_bytes', 'raw_semcache_pool_bytes',
                              'total_payload_bytes', 'local_metadata_bytes', 'k_payload_bytes', 'v_payload_bytes'):
                    self.assertEqual(overall[field], sum(r[field] for r in strata))
                self.assertEqual(overall['encoded_kv_pool_bytes'],
                    sum(r['encoded_kv_pool_bytes']-r['shared_profile_bytes'] for r in strata)+4108)
            self.assertEqual(h.read_json(out/'full_storage/run_diagnostics.json')['classification'], 'SINGLE_PASS_OPERATIONAL_TIMING')
            with self.assertRaisesRegex(ValueError, 'refit'): h.fit_profiles(args, out)

    def test_interrupted_resume_skips_only_atomic_commits(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            self.fit_smoke(args); calls.clear(); args.command = 'storage-full'
            def interrupted(*a):
                if len(calls) == 2: raise KeyboardInterrupt()
                return pair(*a)
            spy.side_effect = interrupted
            with self.assertRaises(KeyboardInterrupt): h.evaluate(args, out)
            state = h.read_json(out/'full_storage/manifest.json')
            self.assertEqual((state['status'], state['completed_block_mode_pairs']), ('INCOMPLETE', 2))
            self.assertFalse(state['primary_result_eligible'])
            self.assertFalse((out/'full_storage/b2_summary.csv').exists())
            spy.side_effect = pair; h.evaluate(args, out)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(set(calls)), 6)

    def test_commit_before_progress_crash_does_not_recompute(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            self.fit_smoke(args); calls.clear(); args.command = 'storage-full'
            original = h.atomic_json
            once = [False]
            def crash(path, data):
                if path.name == 'progress.json' and data['completed_block_mode_pairs'] == 1 and not once[0]:
                    once[0] = True
                    raise KeyboardInterrupt()
                original(path, data)
            with patch.object(h, 'atomic_json', side_effect=crash), self.assertRaises(KeyboardInterrupt):
                h.evaluate(args, out)
            h.evaluate(args, out)
            self.assertEqual(len(calls), 6)

    def test_incompatible_resume_cannot_mix_or_overwrite_completed_contract(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            self.fit_smoke(args); args.command = 'storage-full'; h.evaluate(args, out)
            path = out/'full_storage/manifest.json'; before = path.read_bytes()
            old_profiles = h.load_profiles
            def changed(*a):
                ctx['base']['runtime'] = {'different': True}
                return old_profiles(*a)
            with patch.object(h, 'load_profiles', side_effect=changed), self.assertRaisesRegex(ValueError, 'Incompatible'):
                h.evaluate(args, out)
            self.assertEqual(path.read_bytes(), before)

    def test_bad_correctness_seals_run_primary_false(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            self.fit_smoke(args); args.command = 'storage-full'
            def bad(*a):
                row = pair(*a); row['saved_c1_reconstruction_exact'] = False
                return row
            spy.side_effect = bad
            with self.assertRaisesRegex(ValueError, 'exactness'): h.evaluate(args, out)
            state = h.read_json(out/'full_storage/manifest.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertFalse(state['primary_result_eligible'])
            self.assertFalse(h.read_json(out/'manifest.json')['primary_result_eligible'])
            with self.assertRaisesRegex(ValueError, 'sealed'): h.evaluate(args, out)

    def test_duplicate_or_hybrid_primary_checkpoint_rejected(self):
        for corruption in ('duplicate', 'hybrid_primary', 'bitstream'):
            with self.subTest(corruption=corruption), self.job() as (args, ctx, out, calls, loaded, spy, pair):
                self.fit_smoke(args); args.command = 'storage-full'; h.evaluate(args, out)
                full = out/'full_storage'
                p = next(p for p in (full/'checkpoints').glob('*.json') if
                         h.read_json(p)['row']['mode'] == fmt.MODES[2])
                data = h.read_json(p)
                if corruption == 'duplicate':
                    (p.parent/'duplicate.json').write_bytes(p.read_bytes())
                elif corruption == 'hybrid_primary':
                    data['row']['result_classification'] = 'PRIMARY'
                    data['row_sha256'] = h.digest(data['row'])
                    h.atomic_json(p, data)
                else:
                    stream = full/data['row']['bitstream_file']; stream.write_bytes(stream.read_bytes()+b'x')
                with self.assertRaises(ValueError): h.evaluate(args, out)
                self.assertFalse(h.read_json(full/'manifest.json')['primary_result_eligible'])
                self.assertFalse((full/'b2_summary.csv').exists())

    def test_full_requires_matching_smoke(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            h.fit_profiles(args, out); args.command = 'storage-full'
            with self.assertRaises(FileNotFoundError): h.evaluate(args, out)
            self.assertEqual(calls, [])
            self.assertFalse(h.read_json(out/'full_storage/manifest.json')['primary_result_eligible'])

    def test_exclusive_writer_and_protected_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h.exclusive_run(root):
                with self.assertRaisesRegex(ValueError, 'writer'):
                    with h.exclusive_run(root): pass
            c1, a, b1dir = root/'c1', root/'a', root/'b1'
            for p in (c1, a, b1dir): p.mkdir()
            for output in (c1, a, b1dir, root):
                args = SimpleNamespace(output_dir=output, capture_manifest=c1/'capture_manifest.json', c15a_dir=a, b1_dir=b1dir)
                with self.assertRaises(ValueError): h.safe_output(args)

    def test_population_full_count_and_success_rule_datasets_secondary(self):
        self.assertEqual(h.FULL_PAIRS, 236*3)
        self.assertEqual(h.SMOKE_PAIRS, 2*3)
        self.assertEqual(h.EXPECTED['calibration'], dict(snips=103, multiwoz=138, ALL=241))
        def summary(gain):
            return [dict(mode=fmt.MODES[1], dataset=d, storage_reduction_vs_RAW_ROLE_SPLIT_percent=g)
                    for d, g in [('ALL', gain), ('snips', -1), ('multiwoz', 5)]]
        self.assertEqual(h.primary_decision(summary(3))['status'], 'B2_PRIMARY_SUCCESS')
        self.assertFalse(h.primary_decision(summary(3))['both_datasets_positive'])
        self.assertEqual(h.primary_decision(summary(2.999))['status'], 'B2_PRIMARY_NO_SUCCESS')
        self.assertEqual(h.RULE['provenance'], 'REPRODUCTION_CHOICE')

    def test_full_scheduler_requires_every_one_of_708_unique_pairs(self):
        with self.job() as (args, ctx, out, calls, loaded, spy, pair):
            evaluation = [dict(blocks()[2 if i < 93 else 3], block_id=f'evaluation_{i:03}', query_id=f'q{i}')
                          for i in range(236)]
            ctx['evaluation'] = evaluation
            ctx['base']['evaluation_block_ids'] = [b['block_id'] for b in evaluation]
            templates = list(ctx['references']['blocks'])
            ctx['references']['blocks'] = [dict(r, block_id=b['block_id']) for b in evaluation for r in templates
                                          if r['dataset'] == b['dataset']]
            with patch.object(h, 'FULL_PAIRS', 708):
                self.fit_smoke(args); calls.clear(); args.command = 'storage-full'
                # Tiny real arithmetic streams, rather than any real model fixtures.
                h.evaluate(args, out)
                expected = {(b['block_id'], m) for b in evaluation for m in fmt.MODES}
                self.assertEqual(len(calls), 708)
                self.assertEqual(set(calls), expected)
                state = h.read_json(out/'full_storage/manifest.json')
                self.assertEqual(state['evaluation_block_count'], 236)
                self.assertEqual(state['completed_block_mode_pairs'], 708)
                self.assertTrue(state['primary_result_eligible'])
                # A missing transaction cannot yield a final primary summary.
                victim = next((out/'full_storage/checkpoints').glob('*.json'))
                victim.unlink()
                spy.side_effect = KeyboardInterrupt()
                with self.assertRaises(KeyboardInterrupt): h.evaluate(args, out)
                state = h.read_json(out/'full_storage/manifest.json')
                self.assertEqual(state['completed_block_mode_pairs'], 707)
                self.assertFalse(state['primary_result_eligible'])
                self.assertFalse((out/'full_storage/b2_summary.csv').exists())

    def test_no_repeat_sampling_flags(self):
        for flag in ('--warmup-runs', '--measured-runs', '--max-blocks', '--device'):
            with patch('sys.stderr'), self.assertRaises(SystemExit): h.main(['storage-full', flag, '1'])


if __name__ == '__main__': unittest.main()
