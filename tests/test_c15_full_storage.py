"""Full-storage orchestration and fail-closed checks; no models, downloads or GPU."""
import csv
import hashlib
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen.common import digest, raw_bytes, read_json
from semcache.experiments.cachegen.shared import full_storage as fs
from semcache.experiments.cachegen.shared.core import MODES, Profile, cdf_from_counts
from semcache.experiments.cachegen.shared.device_contract import make_contract

CONTRACT = make_contract('cuda', cuda_available=True, cuda_device_count=1)


def models():
    cdf = cdf_from_counts([0]*127+[10000]+[0]*127)
    return {m: Profile(m, (cdf,)*(2 if m == MODES[0] else 6)) for m in MODES}


def small_blocks():
    return [dict(block_id=f'e{i}', dataset=d, token_group_size=t, partition='evaluation', hidden_dim=2)
            for i, (d, t) in enumerate((('snips', 3), ('snips', 10), ('multiwoz', 3), ('multiwoz', 10)))]


def small_raw(t):
    return raw_bytes(t, hidden_dim=2)


def reference_rows(blocks):
    return {(b['block_id'], m): dict(block_id=b['block_id'], dataset=b['dataset'],
        token_group_size=b['token_group_size'], compression_mode=m, **small_raw(b['token_group_size']),
        encoded_payload_bytes=small_raw(b['token_group_size'])['raw_kv_bytes']//(2 if m == 'UNIFORM_INT8' else 1),
        local_metadata_bytes=64*b['token_group_size']*4 if m == 'UNIFORM_INT8' else 0)
        for b in blocks for m in ('FP16_RAW', 'UNIFORM_INT8')}


class FullStorageTests(unittest.TestCase):
    def test_full_population_selection_no_sampling_no_calibration(self):
        blocks = [dict(block_id=f'e{i}', partition='evaluation', dataset='snips' if i % 2 else 'multiwoz',
                       token_group_size=3 if i < 1072 else 10) for i in range(1308)]
        calibration = dict(block_id='cal', partition='calibration')
        self.assertEqual(fs.selection([calibration]+blocks), blocks)
        for broken in (blocks[:-1], blocks+[dict(blocks[0])],
                       [dict(b, token_group_size=3) for b in blocks]):
            with self.assertRaises(ValueError):
                fs.selection([calibration]+broken)
        self.assertEqual(MODES, ('SHARED_CDF_GLOBAL', 'SHARED_CDF_LAYERGROUP'))

    @contextmanager
    def job(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root, out = Path(tmp)/'c1', Path(tmp)/'c15'
            root.mkdir()
            out.mkdir()
            for name in ('profile_manifest.json', 'shared_cdf_global.bin', 'shared_cdf_layergroup.bin',
                         'uniform_compatibility_diagnostic.json', 'smoke/manifest.json'):
                path = out/name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'frozen evidence')
            (root/'capture_manifest.json').write_text('frozen C1')
            frozen = {p: p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            blocks, profiles = small_blocks(), models()
            args = SimpleNamespace(output_dir=out, capture_manifest=root/'capture_manifest.json')
            @contextmanager
            def inputs(args):
                yield root, out, dict(blocks=[dict(block_id='cal', partition='calibration')]+blocks), {}, {}, [], {
                    'capture_manifest.json': 'capture', 'reconstruction_manifest.json': 'reconstruction'}
            for name, value in (('inputs', inputs), ('resolve_contract', lambda root: CONTRACT),
                                ('load_profiles', lambda *args: (profiles, {}))):
                stack.enter_context(patch.object(fs.h, name, value))
            stack.enter_context(patch.object(fs, 'EXPECTED_BLOCKS', 4))
            stack.enter_context(patch.object(fs, 'EXPECTED_T', {3: 2, 10: 2}))
            stack.enter_context(patch.object(fs, 'raw_bytes', small_raw))
            stack.enter_context(patch.object(fs, 'references', return_value=reference_rows(blocks)))
            stack.enter_context(patch.object(fs, 'verify_tensor_files'))
            stack.enter_context(patch.object(fs, 'environment', return_value={'test': 'stable'}))
            stack.enter_context(patch.object(fs.h, 'measure', side_effect=AssertionError('No timing loop')))
            calls = []
            def evaluate(root, rec, b, model, contract):
                calls.append((b['block_id'], model.mode))
                self.assertEqual(contract, CONTRACT)
                t = b['token_group_size']
                streams = tuple(bytes([127]) * ((end-start)*t*2)
                                for start, end in model.groups for _ in 'kv')
                prepared = streams, struct.pack('<'+'f'*(64*t), *([.25]*(64*t))), (32, t, 2)
                blob = fs.encode(model, *prepared)
                self.assertEqual(fs.decode(blob, model), prepared)
                return blob, dict(block_id=b['block_id'], dataset=b['dataset'], query_id=None,
                    token_group_size=t, compression_mode=model.mode, **small_raw(t),
                    **fs.inspect_block(blob, model)[3], shared_profile_bytes_reference=len(model.to_bytes()),
                    symbol_count=128*t, symbol_roundtrip_exact=True, saved_c1_reconstruction_exact=True,
                    q_dtype='float16', status='COMPLETED', provenance=fs.PROVENANCE)
            evaluation = stack.enter_context(patch.object(fs, 'evaluate_pair', side_effect=evaluate))
            stack.enter_context(patch('builtins.print'))
            yield args, out/'full_storage', calls, evaluation, evaluate, profiles
            for path, content in frozen.items():
                self.assertEqual(path.read_bytes(), content, f'Immutable artifact modified: {path}')

    def test_complete_single_pass_resume_strata_and_artifact_isolation(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            with patch.object(fs, 'encode', wraps=fs.encode) as enc, patch.object(fs, 'decode', wraps=fs.decode) as dec:
                fs.storage_full(args)
                self.assertEqual((enc.call_count, dec.call_count), (8, 8))
                fs.storage_full(args)
                self.assertEqual((enc.call_count, dec.call_count), (8, 8))
            expected = [(b['block_id'], m) for b in small_blocks() for m in MODES]
            self.assertEqual(calls, expected)
            state = read_json(out/'manifest.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertTrue(state['primary_result_eligible'])
            self.assertEqual(state['completed_block_mode_pairs'], 8)
            with (out/'c15_full_block_raw.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 8)
            with (out/'c15_full_summary.csv').open() as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 36)
            for mode in (*MODES, 'FP16_RAW', 'UNIFORM_INT8'):
                rows = {r['stratum']: r for r in summary if r['compression_mode'] == mode}
                overall = rows['ALL']
                for key in ('raw_kv_pool_bytes', 'raw_semcache_pool_bytes', 'encoded_payload_pool_bytes',
                            'local_metadata_pool_bytes', 'block_count'):
                    self.assertEqual(int(overall[key]), sum(int(rows[d][key]) for d in ('SNIPS', 'MultiWOZ')))
                    self.assertEqual(int(overall[key]), sum(int(rows[d][key]) for d in
                        ('SNIPS/T3', 'SNIPS/T10', 'MultiWOZ/T3', 'MultiWOZ/T10')))
                shared = int(overall['shared_profile_bytes'])
                self.assertEqual(int(overall['encoded_kv_pool_bytes']),
                    int(overall['encoded_payload_pool_bytes'])+int(overall['local_metadata_pool_bytes'])+shared)
                self.assertEqual(int(overall['encoded_kv_pool_bytes']),
                    sum(int(rows[d]['encoded_kv_pool_bytes']) for d in ('SNIPS', 'MultiWOZ'))-shared)
                self.assertEqual(int(overall['encoded_semcache_pool_bytes'])-int(overall['encoded_kv_pool_bytes']),
                    int(overall['raw_semcache_pool_bytes'])-int(overall['raw_kv_pool_bytes']))
            self.assertEqual(read_json(out/'run_diagnostics.json')['latency_result'], 'NOT_PRIMARY_LATENCY_RESULT')

    def test_interrupted_resume_only_pending_pairs(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            def interrupt(*args):
                if len(calls) == 3:
                    raise KeyboardInterrupt()
                return evaluate(*args)
            evaluation.side_effect = interrupt
            with self.assertRaises(KeyboardInterrupt):
                fs.storage_full(args)
            state = read_json(out/'manifest.json')
            self.assertEqual((state['status'], state['completed_block_mode_pairs']), ('INCOMPLETE', 3))
            self.assertFalse(state['primary_result_eligible'])
            self.assertFalse((out/'c15_full_summary.csv').exists())
            evaluation.side_effect = evaluate
            fs.storage_full(args)
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(set(calls)), 8)

    def test_preflight_interruption_is_persisted_nonprimary_and_resumable(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            with patch.object(fs, 'verify_tensor_files', side_effect=KeyboardInterrupt()), self.assertRaises(KeyboardInterrupt):
                fs.storage_full(args)
            state = read_json(out/'manifest.json')
            self.assertEqual(state['status'], 'INCOMPLETE')
            self.assertTrue(state['initialization_pending'])
            self.assertFalse(state['primary_result_eligible'])
            fs.storage_full(args)
            self.assertEqual(len(calls), 8)
            self.assertTrue(read_json(out/'manifest.json')['primary_result_eligible'])

    def test_checkpoint_committed_before_progress_is_not_recomputed(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            original = fs.atomic_json
            def interrupt(path, value):
                if Path(path).name == 'progress.json' and value['completed_block_mode_pairs'] == 1:
                    raise KeyboardInterrupt()
                original(path, value)
            with patch.object(fs, 'atomic_json', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                fs.storage_full(args)
            self.assertEqual(len(list((out/'checkpoints').glob('*.json'))), 1)
            fs.storage_full(args)
            self.assertEqual(len(calls), 8)

    def test_incompatible_checkpoint_rejected_without_overwriting(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            fs.storage_full(args)
            before = (out/'manifest.json').read_bytes()
            with patch.object(fs, 'environment', return_value={'changed': True}), self.assertRaisesRegex(ValueError, 'Incompatible'):
                fs.storage_full(args)
            self.assertEqual((out/'manifest.json').read_bytes(), before)
            self.assertEqual(len(calls), 8)

    def test_duplicate_pair_rejected_and_sealed_failed(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            fs.storage_full(args)
            path = next((out/'checkpoints').glob('*.json'))
            (path.parent/'duplicate.json').write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                fs.storage_full(args)
            self.assertFalse(read_json(out/'manifest.json')['primary_result_eligible'])
            self.assertFalse((out/'c15_full_summary.csv').exists())

    def test_correctness_failure_has_no_completed_pair_or_primary_summary(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            evaluation.side_effect = ValueError('Exact saved-C1 reconstruction mismatch')
            with self.assertRaises(ValueError):
                fs.storage_full(args)
            state = read_json(out/'manifest.json')
            self.assertEqual((state['status'], state['failed_pairs']), ('FAILED', 1))
            self.assertFalse(state['primary_result_eligible'])
            self.assertEqual(state['completed_block_mode_pairs'], 0)
            self.assertFalse((out/'c15_full_summary.csv').exists())
            self.assertTrue((out/'failures.jsonl').exists())
            with self.assertRaisesRegex(ValueError, 'sealed'):
                fs.storage_full(args)

    def test_missing_pair_prevents_completion(self):
        with self.job() as (args, out, calls, evaluation, evaluate, profiles):
            real = fs.load_completed
            def missing(*args):
                rows = real(*args)
                if rows:
                    rows.pop(next(iter(rows)))
                return rows
            with patch.object(fs, 'load_completed', side_effect=missing), self.assertRaisesRegex(ValueError, 'Incomplete'):
                fs.storage_full(args)
            self.assertEqual(read_json(out/'manifest.json')['status'], 'FAILED')
            self.assertFalse((out/'c15_full_summary.csv').exists())

    def test_corrupt_bitstream_and_contract_bound_row_rejected(self):
        for corruption in ('stream', 'contract', 'exactness'):
            with self.subTest(corruption=corruption), self.job() as (args, out, calls, evaluation, evaluate, profiles):
                fs.storage_full(args)
                path = next((out/'checkpoints').glob('*.json'))
                record = read_json(path)
                if corruption == 'stream':
                    stream = out/record['row']['bitstream_file']
                    stream.write_bytes(stream.read_bytes()+b'x')
                elif corruption == 'contract':
                    record['run_contract_sha256'] = 'wrong'
                    fs.atomic_json(path, record)
                else:
                    record['row']['symbol_roundtrip_exact'] = False
                    record['row_sha256'] = digest(record['row'])
                    fs.atomic_json(path, record)
                with self.assertRaises(ValueError):
                    fs.storage_full(args)
                self.assertEqual(len(calls), 8)
                self.assertFalse(read_json(out/'manifest.json')['primary_result_eligible'])

    def test_exclusive_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with fs.exclusive_run(Path(tmp)):
                with self.assertRaisesRegex(ValueError, 'writer'):
                    with fs.exclusive_run(Path(tmp)):
                        self.fail('Concurrent writer was allowed')

    def test_c1_references_read_validate_and_never_retime(self):
        blocks = small_blocks()
        records = []
        for b in blocks:
            for mode in ('FP16_RAW', 'UNIFORM_INT8'):
                raw = raw_bytes(b['token_group_size'])
                payload = raw['raw_kv_bytes']//(2 if mode == 'UNIFORM_INT8' else 1)
                metadata = 64*b['token_group_size']*4 if mode == 'UNIFORM_INT8' else 0
                records.append(dict(block_id=b['block_id'], compression_mode=mode, dataset=b['dataset'],
                    token_group_size=b['token_group_size'], status='MEASURED', **raw,
                    encoded_kv_payload_bytes=payload, encoded_metadata_bytes=metadata,
                    encoded_kv_total_bytes=payload+metadata,
                    semcache_total_stored_bytes=raw['raw_q_bytes']+payload+metadata))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'c1_block_raw.csv'
            fs.atomic_csv(path, records)
            self.assertEqual(len(fs.references(root, blocks)), 8)
            fs.atomic_csv(path, records+records[:1])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                fs.references(root, blocks)
            fs.atomic_csv(path, records[:-1])
            with self.assertRaisesRegex(ValueError, 'Missing'):
                fs.references(root, blocks)
            records[0]['encoded_metadata_bytes'] = 1
            fs.atomic_csv(path, records)
            with self.assertRaisesRegex(ValueError, 'accounting'):
                fs.references(root, blocks)

    def test_cli_has_no_sampling_or_repeated_timing_options(self):
        with patch.object(fs, 'storage_full') as run:
            fs.h.main(['storage-full'])
            run.assert_called_once()
        for flag in ('--smoke', '--measured-runs', '--warmup-runs', '--max-blocks-per-group'):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                fs.h.main(['storage-full', flag, '1'])

    def test_exact_gates_and_one_encode_decode_real_pair_function(self):
        class Tensor:
            def __init__(self, value, dtype='float16'):
                self.value, self.dtype = value, dtype
            def numel(self):
                return 192
        torch = SimpleNamespace(Tensor=Tensor, float16='float16', int8='int8', float32='float32',
            equal=lambda a, b: a.dtype == b.dtype and a.value == b.value)
        block = small_blocks()[0]
        model = models()[MODES[0]]
        quantized = tuple((Tensor(1, 'int8'), Tensor(.25, 'float32')) for _ in 'kv')
        output = (Tensor(.25), Tensor(.5))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            (root/'saved.pt').write_bytes(b'saved')
            rec = {'reconstructed': [dict(block_id=block['block_id'], compression_mode='UNIFORM_INT8',
                file='saved.pt', sha256=hashlib.sha256(b'saved').hexdigest())]}
            torch.load = lambda *a, **k: dict(zip('kv', output))
            stack.enter_context(patch.dict('sys.modules', {'torch': torch}))
            stack.enter_context(patch.object(fs, 'load_fixture', return_value=dict(q=Tensor(7), k=Tensor(8), v=Tensor(9))))
            quant = stack.enter_context(patch.object(fs, 'quantize', return_value=quantized))
            stack.enter_context(patch.object(fs, 'prepare', return_value=((), b'', (32, 3, 2))))
            enc = stack.enter_context(patch.object(fs, 'encode', return_value=b'abc'))
            dec = stack.enter_context(patch.object(fs, 'decode', return_value=((), b'', (32, 3, 2))))
            restore = stack.enter_context(patch.object(fs, 'restore', return_value=quantized))
            reconstruct = stack.enter_context(patch.object(fs, 'reconstruct', return_value=output))
            stack.enter_context(patch.object(fs, 'inspect_block', return_value=(None, None, None,
                dict(encoded_payload_bytes=1, local_metadata_bytes=2))))
            blob, row = fs.evaluate_pair(root, rec, block, model, CONTRACT)
            self.assertEqual((enc.call_count, dec.call_count), (1, 1))
            self.assertEqual(row['q_dtype'], 'float16')
            self.assertEqual(quant.call_args.kwargs['device'], 'cuda:0')
            self.assertEqual(reconstruct.call_args.kwargs['device'], 'cuda:0')
            self.assertTrue(row['saved_c1_reconstruction_exact'])
            for bad in (Tensor(2, 'int8'), Tensor(1, 'float32')):
                restore.return_value = ((bad, quantized[0][1]), quantized[1])
                with self.assertRaisesRegex(ValueError, 'roundtrip'):
                    fs.evaluate_pair(root, rec, block, model, CONTRACT)
            restore.return_value = quantized
            # Exercise real existing_uniform torch.equal gate; stub only the verbose failure report.
            reconstruct.return_value = (Tensor(.251), output[1])
            with patch.object(fs.h, 'failure_report', return_value={'components': {}, 'block_id': 'e0'}), self.assertRaises(fs.h.UniformCompatibilityError):
                fs.evaluate_pair(root, rec, block, model, CONTRACT)


if __name__ == '__main__':
    unittest.main()
