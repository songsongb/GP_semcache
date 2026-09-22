"""No model construction/downloads. Optional tensor tests use synthetic attention."""
import copy
import json
import tempfile
from unittest.mock import patch
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

from semcache.cache.attention_impact import make_impact_reducer, PaperRowL2SumReducer
from semcache.cache.global_cache import GlobalCache
from semcache.diagnostics.impact import MODES, PARTS, parity, measure_impact
from semcache.diagnostics.impact_policy import replay_policy, compare_policy
from semcache.diagnostics.impact_cost import recompose, break_even
from semcache.diagnostics.impact_runtime import summarize
from semcache.semantic.subsequence import SubsequenceExtractor, Subsequence
from semcache.semcache_engine import SemCacheEngine
from semcache.system_cost.model import compare_request
from test_m9a1_base_profile import strict_profiles
from test_m9a_cost_model import calibration, DIMS

HAS_TORCH = importlib.util.find_spec('torch') is not None


def policy_queries(values):
    windows = SubsequenceExtractor(3).extract([1, 2, 3, 4, 5, 6])
    return [dict(windows=windows, values=list(values), cluster=0, query_id=q,
                 token_count=6, sizes=[10]*4) for q in ('cold_miss', 'same_user_exact')]


class ImpactAuditContractTest(unittest.TestCase):
    def test_default_engine_and_reducer_unchanged(self):
        engine = SemCacheEngine(None, None, None, None, None, GlobalCache(100))
        self.assertIs(type(engine.impact_reducer), PaperRowL2SumReducer)
        self.assertEqual(make_impact_reducer().name, 'paper_row_l2_sum')
        self.assertEqual(make_impact_reducer('reproduction_frobenius_mean').name,
                         'reproduction_frobenius_mean')
        self.assertEqual(engine.metrics.updater.rho, .8)

    def test_empty_parity_and_nonfinite_rejection(self):
        self.assertTrue(parity([], [])['impact_values_within_tolerance'])
        self.assertFalse(parity([1.], [float('nan')])['impact_values_within_tolerance'])
        with self.assertRaises(ValueError):
            parity([1.], [])

    def test_policy_parity_with_eviction_and_chu(self):
        queries = policy_queries([1., 2., 3., 4.])
        before = copy.deepcopy(queries)
        baseline = replay_policy(queries, capacity_bytes=20)
        result = compare_policy(baseline, replay_policy(queries, capacity_bytes=20))
        self.assertTrue(result['eviction_occurred'])
        self.assertTrue(result['admission_decisions_identical'])
        self.assertTrue(result['eviction_decisions_identical'])
        self.assertTrue(result['eviction_scores_within_tolerance'])
        self.assertEqual(before, queries)

    def test_policy_drift_is_detected(self):
        baseline = replay_policy(policy_queries([1., 2., 3., 4.]), capacity_bytes=20)
        candidate = replay_policy(policy_queries([4., 3., 2., 1.]), capacity_bytes=20)
        result = compare_policy(baseline, candidate)
        self.assertFalse(result['eviction_decisions_identical'] and result['eviction_scores_within_tolerance'])

    def test_no_candidates_policy(self):
        query = dict(windows=[], values=[], sizes=[], cluster=0, query_id='empty', token_count=2)
        result = replay_policy([query], capacity_bytes=20)
        self.assertEqual(result['admissions'], [])
        self.assertEqual(result['resident_keys'], [])

    def test_recomposition_only_replaces_impact_and_preserves_original(self):
        original = compare_request(*strict_profiles(), calibration(), DIMS)
        before = copy.deepcopy(original)
        same = recompose(original, original['row']['attention_impact_ms'], MODES[0])
        self.assertAlmostEqual(same['semcache_total_ms'], original['row']['semcache_total_ms'])
        faster = recompose(original, 0., MODES[1])
        self.assertAlmostEqual(faster['system_delta_ms']-same['system_delta_ms'], original['row']['attention_impact_ms'])
        self.assertEqual(original, before)
        self.assertEqual(faster['result_label'], 'RESEARCH_EXTENSION_DIAGNOSTIC')
        self.assertEqual(faster['provenance'], 'SIMULATED_RESEARCH_EXTENSION')
        self.assertFalse(faster['decision_applied_to_inference'])

    def test_bandwidth_and_break_even_sign(self):
        original = compare_request(*strict_profiles(), calibration(), DIMS)
        low = recompose(original, 100., MODES[2], bandwidth_mbps=200)
        high = recompose(original, 100., MODES[2], bandwidth_mbps=1000)
        self.assertAlmostEqual(low['edge_network_ms'], high['edge_network_ms']*5)
        self.assertGreater(low['system_delta_ms'], high['system_delta_ms'])
        threshold = low['analytical_break_even_bandwidth_mbps']
        self.assertIsNotNone(threshold)
        tie = recompose(original, 100., MODES[2], bandwidth_mbps=threshold)
        self.assertAlmostEqual(tie['system_delta_ms'], 0., places=10)
        self.assertIsNone(break_even(0, 0)['analytical_break_even_bandwidth_mbps'])
        self.assertEqual(break_even(0, -1)['break_even_regime'], 'RECOMPUTE_at_all_bandwidths')
        self.assertEqual(break_even(1, 0)['break_even_regime'], 'REUSE_at_all_finite_positive_bandwidths')

    def test_recomposition_rejects_proxy_unsafe_and_bad_replacement(self):
        original = compare_request(*strict_profiles(), calibration(), DIMS)
        for field, value in [('result_label', 'SIMULATED_PROXY_DOUBLE_COUNTS_LORA'),
                             ('correctness_gate_passed', False)]:
            broken = copy.deepcopy(original)
            broken['row'][field] = value
            with self.assertRaises(ValueError):
                recompose(broken, 1., MODES[1])
        with self.assertRaises(ValueError):
            recompose(original, -1., MODES[1])
        with self.assertRaises(ValueError):
            recompose(original, 1., MODES[1], bandwidth_mbps=0)

    def test_driver_writes_separate_outputs_and_gates_parity_without_models(self):
        from semcache.system_cost.common import file_sha256, write_json
        spec = importlib.util.spec_from_file_location('impact_driver',
            Path(__file__).resolve().parents[1]/'scripts/36_audit_m9a2_semantic_impact.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root/'fresh_m85/inference_environment.json'
            write_json(env_path, {'model': 'stub test only'})
            rows = list(strict_profiles())
            for r in (rows[0], rows[2]):
                r['es_base_profile']['source_environment_sha256'] = file_sha256(env_path)
            es_path = root/'strict_es_input.jsonl'
            es_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            original = compare_request(*rows, calibration(), DIMS)
            cost_path = root/'costs/m9a_summary.json'
            write_json(cost_path, dict(result_label='STRICT_BASE_ONLY_SYSTEM_MODEL', comparisons=[original]))
            write_json(cost_path.parent/'m9a_environment.json', dict(input_files={str(es_path): file_sha256(es_path)}))
            before = (es_path.read_bytes(), cost_path.read_bytes())
            summaries = [dict(implementation=mode, uninstrumented_impact_total_ms={'mean': .01}) for mode in MODES]
            for allowed in (True, False):
                output = root/('audit_pass' if allowed else 'audit_fail')
                with patch('semcache.diagnostics.impact_runtime.capture_fixture', return_value=([], [], {'capacity_bytes': 1}, ())), \
                     patch('semcache.diagnostics.impact_runtime.benchmark', return_value=[{'diagnostic_composition_eligible': allowed}]), \
                     patch('semcache.diagnostics.impact_runtime.summarize', return_value=summaries):
                    argv = ['--es-input', str(es_path), '--system-summary', str(cost_path), '--output-dir', str(output)]
                    if allowed:
                        driver.main(argv)
                    else:
                        with self.assertRaises(SystemExit):
                            driver.main(argv)
                manifest = json.loads((output/'audit_manifest.json').read_text())
                self.assertEqual(manifest['state'], 'COMPLETE' if allowed else 'PARITY_FAILED')
                system = json.loads((output/'system_diagnostic.json').read_text())
                self.assertEqual(len(system['rows']), 9 if allowed else 0)
            self.assertEqual(before, (es_path.read_bytes(), cost_path.read_bytes()))
            # An inconsistent original artifact is rejected before capture/model loading.
            broken = copy.deepcopy(original)
            broken['row']['attention_impact_ms'] += 1
            write_json(cost_path, dict(result_label='STRICT_BASE_ONLY_SYSTEM_MODEL', comparisons=[broken]))
            with self.assertRaises(ValueError):
                driver.load_compositions(cost_path, es_path, [rows])

    def test_summary_excludes_warmup(self):
        rows = []
        for phase in ('warmup', 'measured'):
            for mode in MODES:
                rows.append(dict(phase=phase, implementation=mode, provenance='MEASURED',
                    impact_total_ms=1000. if phase == 'warmup' else 2., uninstrumented_impact_total_ms=1.,
                    impact_value_max_absolute_error=0., impact_value_relative_l2_error=0.,
                    diagnostic_composition_eligible=True, policy_parity=dict(admission_decisions_identical=True,
                        eviction_decisions_identical=True, eviction_scores_within_tolerance=True, eviction_occurred=False)))
        summary = summarize(rows)
        self.assertEqual(summary[0]['impact_total_ms']['mean'], 2.)
        self.assertEqual(summary[1]['impact_total_ms_speedup_vs_current'], 1.)


@unittest.skipUnless(HAS_TORCH, 'optional torch synthetic-tensor tests; no models')
class ImpactTensorTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        generator = torch.Generator().manual_seed(12)
        # Nonzero future keys exercise explicit causal exclusion, not an already
        # triangular input that could hide a missing mask in an implementation.
        self.attentions = tuple(torch.rand((1, 3, 6, 6), generator=generator, dtype=torch.float16) for _ in range(2))
        self.windows = SubsequenceExtractor(3).extract([1, 2, 3, 4, 5, 6])

    def all_modes(self, attentions=None, windows=None, mask=None):
        attentions = self.attentions if attentions is None else attentions
        windows = self.windows if windows is None else windows
        original = [make_impact_reducer().reduce(attentions, w.start, w.end, mask) for w in windows]
        outputs = [measure_impact(attentions, windows, mode, mask) for mode in MODES]
        for output in outputs:
            self.assertTrue(parity(original, output['impact_values'])['impact_values_within_tolerance'])
            self.assertAlmostEqual(sum(output[k] for k in PARTS), output['impact_total_ms'], places=9)
        return outputs

    def test_overlapping_windows_multiple_layers_heads_and_fp16_unchanged(self):
        before = [a.clone() for a in self.attentions]
        outputs = self.all_modes()
        for a, b in zip(before, self.attentions):
            self.assertTrue(self.torch.equal(a, b))
        self.assertEqual(outputs[0]['operation_counts']['layer_visits'], 8)
        self.assertEqual(outputs[1]['operation_counts']['layer_visits'], 2)
        self.assertEqual(outputs[0]['operation_counts']['impact_scalar_host_reads'], 8)
        self.assertEqual(outputs[2]['operation_counts']['batched_host_materializations'], 1)
        self.assertNotIn('impact_scalar_host_reads', outputs[2]['operation_counts'])

    def test_causal_and_padding_masks(self):
        mask = self.torch.tensor([[1, 1, 0, 1, 1, 0]])
        output = self.all_modes(mask=mask)
        expected = []
        for w in self.windows:
            total = 0.
            for a in self.attentions:
                for t in range(w.start, w.end):
                    if mask[0, t]:
                        total += sum(sum(float(a[0, h, t, k])**2 for k in range(t+1) if mask[0, k])**.5
                                     for h in range(3))/3
            expected.append(total)
        self.assertTrue(parity(expected, output[0]['impact_values'])['impact_values_within_tolerance'])

    def test_empty_candidates(self):
        outputs = self.all_modes(windows=[])
        self.assertTrue(all(r['impact_values'] == [] for r in outputs))
        self.assertTrue(all(not r['operation_counts'] for r in outputs))

    def test_nonfinite_masked_cells_and_outside_windows(self):
        tensors = tuple(a.clone() for a in self.attentions)
        for a in tensors:
            a[:, :, 0, 5] = float('nan')  # Future causal cell.
            a[:, :, 5, :] = float('nan')  # Query outside selected windows.
        self.all_modes(attentions=tensors, windows=[Subsequence((1, 2, 3), 0, 3)])
        tensors[0][0, 0, 0, 0] = float('nan')
        for mode in MODES:
            with self.assertRaises(ValueError):
                measure_impact(tensors, self.windows, mode)

    def test_admission_and_eviction_parity_for_all_modes(self):
        outputs = self.all_modes()
        reference = replay_policy(policy_queries(outputs[0]['impact_values']), capacity_bytes=20)
        for output in outputs:
            result = compare_policy(reference, replay_policy(policy_queries(output['impact_values']), capacity_bytes=20))
            self.assertTrue(result['admission_decisions_identical'])
            self.assertTrue(result['eviction_decisions_identical'])
            self.assertTrue(result['eviction_scores_within_tolerance'])

    def test_ragged_and_repeated_token_windows(self):
        windows = [Subsequence((1,), 0, 1), Subsequence((1,), 4, 5), Subsequence((1, 2, 3, 4), 1, 5)]
        outputs = self.all_modes(windows=windows)
        self.assertTrue(all(r['per_query_impact_count'] == 2 for r in outputs))

    def test_invalid_windows_and_masks(self):
        for mode in MODES:
            with self.assertRaises(ValueError):
                measure_impact(self.attentions, self.windows, mode, self.torch.ones(2))
            with self.assertRaises(ValueError):
                measure_impact(self.attentions, [Subsequence((1, 2, 3), 5, 8)], mode)

    @unittest.skipUnless(HAS_TORCH, 'torch required')
    def test_cuda_when_available(self):
        if not self.torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        previous = self.torch.are_deterministic_algorithms_enabled()
        try:
            self.torch.use_deterministic_algorithms(True)
            outputs = self.all_modes(attentions=tuple(a.cuda() for a in self.attentions))
        finally:
            self.torch.use_deterministic_algorithms(previous)
        self.assertEqual(outputs[0]['scalar_or_result_cuda_sync_points'], 16)
        self.assertEqual(outputs[1]['scalar_or_result_cuda_sync_points'], 1)
        self.assertGreaterEqual(outputs[1]['impact_row_l2_reduction_device_ms'], 0.)


if __name__ == '__main__':
    unittest.main()
