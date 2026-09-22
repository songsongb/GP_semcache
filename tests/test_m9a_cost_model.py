"""Fast M9-A tests: arithmetic, local files and stub resources only; no models/torch."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from semcache.system_cost.common import (Dimensions, PROVENANCE, read_dimensions,
                                         tagged_ms, write_json)
from semcache.system_cost.calibration import calibration_latency, cpu_resources, timing_summary
from semcache.system_cost.memory import UD_CAPACITY, ud_memory
from semcache.system_cost.network import transfer_ms, directional_cost, communication
from semcache.system_cost.profiles import load_rows, pair_rows, reuse_gate, validate_row
from semcache.system_cost.model import compare_request

ROOT = Path(__file__).resolve().parents[1]
DIMS = Dimensions('facebook/opt-2.7b', 2560, 32)


def calibration():
    return dict(**DIMS.record(), provenance='CALIBRATED', dtype='float32', num_threads=4,
        parameter_element_bytes=4, temporary_element_bytes=4, measured_count=3,
        hostname='fixture-only', cpu_model='fixture-only', samples=[
            dict(sequence_length=n, raw_timings_ms=raw, **timing_summary(raw), provenance='CALIBRATED')
            for n, raw in [(32, [.1, .2, .3]), (16, [.05, .1, .15])]])


def profiles():
    base = dict(experiment_id='fixture', model_id=DIMS.model_id, query_id='same_user_exact',
        user_id='user_a', adapter_name='user_a', prompt_token_ids_sha256='fixture-token-hash',
        repeat_index=0, dtype='torch.float16', model_revision='fixture-model-rev',
        tokenizer_revision='fixture-tokenizer-rev', hostname='fixture-SERAPH', gpu_name='RTX 3090',
        seed=42, attention_implementation='eager', measured_or_analytical='MEASURED',
        execution_scope='prefill_only', prompt_tokens=32, reused_tokens=0, recomputed_tokens=32,
        token_reuse_ratio=0., physical_reuse_used=False, projection_skip_used=False,
        prefill_wall_ms=10., request_wall_ms=12., tokenization_ms=1., mixed_qkv_execution_ms=4.,
        saved_communication_bytes=0, correctness_validation_status='controlled_exact_parity_passed',
        controlled_exact_parity_passed=True, impact_reducer_type='paper_row_l2_sum',
        cluster_update_interval_queries=100)
    native = dict(base, mode='NATIVE_NO_CACHE', correctness_validation_status='native_self_parity')
    lookup = dict(base, mode='SEMCACHE_LOOKUP_NO_REUSE')
    physical = dict(base, mode='SEMCACHE_PHYSICAL_REUSE', reused_tokens=16, recomputed_tokens=16,
        token_reuse_ratio=.5, physical_reuse_used=True, projection_skip_used=True,
        prefill_wall_ms=5., request_wall_ms=7., mixed_qkv_execution_ms=2.,
        saved_communication_bytes=16*2560*32*8)
    return native, lookup, physical


class NetworkMemoryTest(unittest.TestCase):
    def test_200_mbps_and_other_supported_rates(self):
        self.assertEqual(transfer_ms(25_000_000, 200), 1000.)
        self.assertEqual(transfer_ms(25_000_000, 500), 400.)
        self.assertEqual(transfer_ms(25_000_000, 1000), 200.)
        self.assertEqual(transfer_ms(0), 0.)
        for bandwidth in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                transfer_ms(10, bandwidth)
        with self.assertRaises(ValueError):
            transfer_ms(1.5)

    def test_directional_layer_request_and_boundary_accounting(self):
        d = Dimensions('facebook/opt-125m', 768, 12)
        net = communication(d, 10, 4, hidden_element_bytes=2, delta_element_bytes=4)
        layer = net['per_layer']
        self.assertEqual(layer['ud_to_es_bytes'], 3*6*768*4)
        self.assertEqual(layer['es_to_ud_bytes'], 6*768*2)
        request = net['request']
        self.assertEqual(request['ud_to_es_bytes'], layer['ud_to_es_bytes']*12 + 10*768*2)
        self.assertEqual(request['es_to_ud_bytes'], layer['es_to_ud_bytes']*12 + 10*768*2)
        self.assertAlmostEqual(request['total_network_ms'], request['ud_to_es_ms']+request['es_to_ud_ms'])
        all_hits = communication(d, 10, 10)
        self.assertEqual(all_hits['per_layer']['total_network_bytes'], 0)
        self.assertEqual(all_hits['request']['total_network_bytes'], 2*10*768*2)
        self.assertEqual(communication(d, 10, 10, boundary_transfers=False)['request']['total_network_bytes'], 0)

    def test_logical_8_gib_capacity_and_no_allocation(self):
        memory = ud_memory(DIMS, 32)
        self.assertEqual(memory['ud_memory_capacity_bytes'], 8*1024**3)
        self.assertEqual(memory['adapter_parameter_bytes'], 6*32*2560*8*4)
        self.assertTrue(memory['fits'])
        fill = UD_CAPACITY-memory['estimated_usage_bytes']
        self.assertTrue(ud_memory(DIMS, 32, user_local_cache_bytes=fill)['fits'])
        self.assertFalse(ud_memory(DIMS, 32, user_local_cache_bytes=fill+1)['fits'])
        self.assertEqual(memory['provenance'], 'SIMULATED')


class ProvenanceProfileTest(unittest.TestCase):
    def test_provenance_validation(self):
        for kind in PROVENANCE:
            self.assertEqual(tagged_ms(1., kind, 'test')['provenance'], kind)
        for kind in ('MEASURED_OR_SIMULATED', 'CALIBRATED_ON_SERAPH_CPU', None):
            with self.assertRaises(ValueError):
                tagged_ms(1., kind, 'test')
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                tagged_ms(value, 'MEASURED', 'test')
        row = profiles()[0]
        row['measured_or_analytical'] = 'PAPER_REFERENCE'
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_correctness_rejection_and_explicit_unsafe_override(self):
        for status, parity in [('correctness_not_validated', True),
                ('correctness_not_validated_for_lengths_ge_64', True),
                ('controlled_exact_parity_failed', False), ('controlled_exact_parity_passed', False),
                ('diagnostic_recorded', None)]:
            row = dict(profiles()[2], correctness_validation_status=status,
                       controlled_exact_parity_passed=parity)
            with self.assertRaises(ValueError):
                reuse_gate(row)
            gated = reuse_gate(row, True)
            self.assertTrue(gated['invalid_reuse_override'])
            self.assertFalse(gated['correctness_gate_passed'])
            self.assertFalse(gated['safe_reuse_claimed'])
        self.assertTrue(reuse_gate(profiles()[2])['correctness_gate_passed'])

    def test_matching_identity_and_missing_or_duplicate_rows(self):
        rows = profiles()
        self.assertEqual(len(pair_rows(list(rows), DIMS.model_id)), 1)
        with self.assertRaises(ValueError):
            pair_rows(list(rows[:2]), DIMS.model_id)
        with self.assertRaises(ValueError):
            pair_rows(list(rows)+[rows[0]], DIMS.model_id)
        mismatch = copy.deepcopy(rows)
        mismatch[2]['adapter_name'] = 'other_user'
        with self.assertRaises(ValueError):
            pair_rows(list(mismatch), DIMS.model_id)
        mismatch = copy.deepcopy(rows)
        mismatch[2]['impact_reducer_type'] = 'reproduction_frobenius_mean'
        with self.assertRaises(ValueError):
            pair_rows(list(mismatch), DIMS.model_id)
        with self.assertRaises(ValueError):
            compare_request(*mismatch, calibration(), DIMS, es_compute_policy='peft-prefill-proxy')

    def test_json_and_jsonl_input_and_local_config_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for payload in (list(profiles()), {'rows': list(profiles())}, profiles()[0]):
                write_json(root/'rows.json', payload)
                self.assertGreater(len(load_rows(root/'rows.json')), 0)
            (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in profiles()))
            self.assertEqual(len(load_rows(root/'rows.jsonl')), 3)
            write_json(root/'config.json', dict(model_type='opt', hidden_size=2560, num_hidden_layers=32))
            dims, source = read_dimensions(DIMS.model_id, root/'config.json')
            self.assertEqual(dims, DIMS)
            self.assertFalse(source['model_weights_loaded'])
            with self.assertRaises(ValueError):
                read_dimensions('facebook/opt-6.7b', root/'config.json')
            with self.assertRaises(ValueError):
                read_dimensions('facebook/opt-125m', root/'config.json')


class CalibrationTest(unittest.TestCase):
    def test_exact_length_calibration_and_zero_work(self):
        cal = calibration()
        self.assertAlmostEqual(calibration_latency(cal, DIMS, 32)['value_ms'], .2*32)
        self.assertEqual(calibration_latency(cal, DIMS, 0)['value_ms'], 0)
        with self.assertRaises(ValueError):
            calibration_latency(cal, DIMS, 17)
        cal['samples'][0]['mean_ms'] = 999
        with self.assertRaises(ValueError):
            calibration_latency(cal, DIMS, 32)
        cal['provenance'] = 'MEASURED'
        with self.assertRaises(ValueError):
            calibration_latency(cal, DIMS, 0)

    def test_thread_affinity_limit_and_restore_on_error(self):
        class TorchStub:
            threads = 8
            def get_num_threads(self):
                return self.threads
            def set_num_threads(self, n):
                self.threads = n
        torch = TorchStub()
        affinity = set(range(8))
        def set_affinity(pid, cpus):
            nonlocal affinity
            affinity = set(cpus)
        with patch('os.sched_getaffinity', side_effect=lambda _: affinity, create=True), \
             patch('os.sched_setaffinity', side_effect=set_affinity, create=True):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with cpu_resources(torch) as record:
                    self.assertEqual(torch.threads, 4)
                    self.assertEqual(affinity, {0, 1, 2, 3})
                    self.assertTrue(record['applied'])
                    raise RuntimeError('fixture')
        self.assertEqual(torch.threads, 8)
        self.assertEqual(affinity, set(range(8)))


class SystemModelTest(unittest.TestCase):
    def cost(self, rows=None, **kwargs):
        return compare_request(*(rows or profiles()), calibration(), DIMS,
                               es_compute_policy='peft-prefill-proxy', **kwargs)

    def test_positive_negative_and_tie_decisions(self):
        result = self.cost()
        row = result['row']
        self.assertGreater(row['system_delta_ms'], 0)
        self.assertEqual(row['selected_action'], 'REUSE')
        self.assertEqual(row['decision_provenance'], 'RESEARCH_EXTENSION')
        self.assertAlmostEqual(row['system_delta_ms'], row['compute_saved_ms']+
                               row['communication_saved_ms']-row['reuse_overhead_ms'])
        rows = profiles()
        rows[2]['request_wall_ms'] = 10_000.
        slower = self.cost(rows)['row']
        self.assertLess(slower['system_delta_ms'], 0)
        self.assertEqual(slower['selected_action'], 'RECOMPUTE')
        # Exact equal-cost case: no token reuse, same ES timing, zero control.
        rows = profiles()
        rows[2].update(reused_tokens=0, recomputed_tokens=32, token_reuse_ratio=0.,
            physical_reuse_used=False, projection_skip_used=False, saved_communication_bytes=0,
            prefill_wall_ms=10., request_wall_ms=11.)
        tied = self.cost(rows)['row']
        self.assertEqual(tied['system_delta_ms'], 0.)
        self.assertEqual(tied['selected_action'], 'RECOMPUTE')

    def test_explicit_proxy_and_base_only_measurement(self):
        rows = profiles()
        with self.assertRaisesRegex(ValueError, 'base-only'):
            compare_request(*rows, calibration(), DIMS)
        proxy = self.cost()['row']
        self.assertEqual(proxy['es_compute_ms_provenance'], 'SIMULATED')
        self.assertEqual(proxy['ud_lora_ms_provenance'], 'CALIBRATED')
        for row in (rows[0], rows[2]):
            row.update(es_base_compute_ms=3., es_base_compute_provenance='MEASURED',
                       es_base_compute_scope='prefill_base_only_excluding_lora_control')
        # M9-A.1 rejects legacy self-declared fields without strict profile evidence.
        with self.assertRaisesRegex(ValueError, 'strict-base-only'):
            compare_request(*rows, calibration(), DIMS)

    def test_every_latency_is_tagged_and_timers_not_double_counted(self):
        result = self.cost()
        row = result['row']
        for key in row:
            if key.endswith('_ms'):
                self.assertIn(row[key+'_provenance'], PROVENANCE)
        self.assertEqual(row['semcache_control_ms'], 1.)
        self.assertEqual(result['es_source_timings']['lookup_no_reuse']['mixed_qkv_execution_ms']['provenance'], 'MEASURED')
        altered = profiles()
        altered[1]['mixed_qkv_execution_ms'] = 500.
        self.assertEqual(self.cost(altered)['row']['system_delta_ms'], row['system_delta_ms'])
        altered[2]['saved_communication_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'saved_communication_bytes'):
            self.cost(altered)

    def test_no_input_mutation_or_inference_behavior_changes(self):
        rows, cal = profiles(), calibration()
        snapshot = copy.deepcopy((rows, cal))
        compare_request(*rows, cal, DIMS, es_compute_policy='peft-prefill-proxy')
        self.assertEqual((rows, cal), snapshot)
        core = ['src/semcache/semcache_engine.py', 'src/semcache/edgelora/mixed_projection.py',
                'scripts/30_run_m8_inference_timing.py']
        previous_engine_module = sys.modules.get('semcache.semcache_engine')
        before = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in core}
        self.cost()
        self.assertEqual(before, {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in core})
        self.assertIs(sys.modules.get('semcache.semcache_engine'), previous_engine_module)

    def test_cli_end_to_end_only_fixture_artifacts_and_no_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'config.json', dict(model_type='opt', hidden_size=2560, num_hidden_layers=32))
            write_json(root/'calibration.json', calibration())
            write_json(root/'m8.json', list(profiles()))
            cmd = [sys.executable, str(ROOT/'scripts/34_run_m9a_single_user_cost.py'),
                '--model-config', str(root/'config.json'), '--es-input', str(root/'m8.json'),
                '--ud-calibration', str(root/'calibration.json'), '--es-compute-policy', 'peft-prefill-proxy',
                '--bandwidth-mbps', '200', '500', '1000', '--output-dir', str(root/'out')]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads((root/'out/m9a_summary.json').read_text())
            self.assertEqual(summary['comparison_count'], 3)
            self.assertFalse(summary['decisions_applied'])
            with (root/'out/single_user_cost_breakdown.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)
            env = json.loads((root/'out/m9a_environment.json').read_text())
            self.assertEqual(env['paper_reference']['provenance'], 'PAPER_REFERENCE')
            self.assertFalse(env['model_execution_performed'])
            rows = list(profiles())
            rows[2]['controlled_exact_parity_passed'] = False
            write_json(root/'m8.json', rows)
            cmd[-1] = str(root/'rejected')
            rejected = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
