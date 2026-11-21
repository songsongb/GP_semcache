"""CPU observation parity, rejection ordering, and unchanged tamper boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_r5_subsequence_validation as fixtures
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora import r5_subsequence_physical as physical
from semcache.edgelora.r5_subsequence_integrity import SCOPED
from semcache.edgelora.r5_subsequence_profile import CURRENT
from semcache.edgelora.r5_d5_owner_diagnostics import CPUOwnerDiagnostics, COMPONENTS

spec = importlib.util.spec_from_file_location('owner_cpu_harness',
    Path(__file__).resolve().parents[1] / 'scripts/161_r5_d5_owner_diagnostics.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class OwnerObservation(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='')
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_all_scenarios_preserve_outputs_metadata_proofs_policy_and_occupancy(self):
        for scenario in harness.SCENARIOS:
            with self.subTest(scenario=scenario):
                reference = harness.run_fixture(scenario)
                observed = harness.run_fixture(scenario, CPUOwnerDiagnostics())
                self.assertEqual(reference['semantic_sha256'], observed['semantic_sha256'])
                report = observed['diagnostics']
                self.assertEqual(set(report['completion_ms']), set(COMPONENTS))
                self.assertTrue(all(v >= 0 for v in report['completion_ms'].values()))
                self.assertAlmostEqual(sum(report['completion_ms'].values()) +
                    report['boundary_start_unassigned_ms'], report['completion_total_ms'])
                self.assertEqual(report['counts']['gpu_to_cpu_host_transfer_calls'], 0)
                self.assertEqual(report['profile']['eligibility'], [])

    def test_admit_and_both_reject_reasons_count_input_proof_before_decision(self):
        for scenario, reason in (('ADMIT', 'RESIDENT'), ('CAPACITY_REJECT', 'CAPACITY'),
                                 ('POLICY_REJECT', 'POLICY')):
            with self.subTest(scenario=scenario):
                result = harness.run_fixture(scenario, CPUOwnerDiagnostics())
                report = result['diagnostics']
                counts = report['counts']
                admitted = scenario == 'ADMIT'
                self.assertEqual(counts['admission_candidates'], 4)
                self.assertEqual(counts['admission_approved'], 4 if admitted else 0)
                self.assertEqual(counts['admission_rejected'], 0 if admitted else 4)
                self.assertEqual(counts['materialization_calls'], 4 if admitted else 0)
                self.assertEqual(counts['tensor_proof_calls'], 38 if admitted else 14)
                self.assertEqual(counts['serialized_tensor_bytes'], 2112 if admitted else 960)
                self.assertEqual(counts['entry_proof_all_layers_calls'], 4 if admitted else 0)
                self.assertEqual(counts['admission_entry_full_calls'], 4 if admitted else 0)
                self.assertEqual(result['integrity']['full_entry_proof_count'], 0)
                self.assertEqual(result['integrity']['consumed_tensor_proof_count'], 0)
                self.assertEqual(result['integrity']['admission_entry_proof_count'], 4 if admitted else 0)
                for row in report['candidates']:
                    self.assertEqual(row['outcome'], reason)
                    self.assertEqual(row['input_proof']['tensor_proofs'], 2)
                    self.assertEqual(row['input_proof']['serialized_bytes'], 96)

    def test_equal_content_does_not_imply_same_tensor_object(self):
        tensor = ArrayTensor([1., 2., 3., 4.], [1, 1, 4])
        observer = CPUOwnerDiagnostics()
        with observer.observe('FINGERPRINTS'):
            physical.tensor_proof(tensor)
            physical.tensor_proof(tensor)
            physical.tensor_proof(tensor.clone())
            tensor.values[0] += 1
            physical.tensor_proof(tensor)
        counts = observer.report()['counts']
        self.assertEqual(counts['tensor_proof_calls'], 4)
        self.assertEqual(counts['same_live_object_same_fingerprint_repeats'], 1)
        self.assertEqual(counts['same_content_fingerprint_repeats'], 2)

    def test_self_eviction_is_approval_and_materialization_but_not_resident_sealing(self):
        result = harness.run_fixture('SELF_EVICTION', CPUOwnerDiagnostics())
        counts = result['diagnostics']['counts']
        self.assertEqual(counts['admission_approved'], 4)
        self.assertEqual(counts['admission_rejected'], 0)
        self.assertEqual(counts['materialization_calls'], 4)
        self.assertEqual(counts['evictions'], 3)
        self.assertEqual(counts['admission_entry_full_calls'], 1)
        self.assertEqual([r['outcome'] for r in result['diagnostics']['candidates']].count('APPROVED_THEN_EVICTED'), 3)

    def test_materialized_entry_owns_storage_and_eviction_invalidates_proof_bindings(self):
        from semcache.cache.cache_entry import CacheEntry
        from semcache.edgelora.r5_sparse_qkv import gather
        source = ArrayTensor(list(range(24)), [1, 6, 4])
        gathered = tuple(gather(source, [0, 1, 2]) for _ in range(3))
        entry = CacheEntry.from_tensors(0, (1, 2, 3), (0, 3), {0: gathered})
        source.values[0] = -1
        gathered[0].values[0] = -2
        self.assertEqual(entry.tensors[0][0].values[0], 0)
        f = fixtures.Validation()
        f.setUp()
        owner = physical.PhysicalOwner(f.f.ident, f.f.registry, capacity=288, validation_mode=SCOPED)
        with CPUOwnerDiagnostics().observe('EVICTION_BINDING'):
            owner.begin(f.f.req('E', [1, 2, 3, 4, 5, 6]), {'cluster_id': 0})
            f.project_all(owner, 6)
            f.complete(owner)
        self.assertEqual(set(owner.proofs), set(owner.cache.entries))
        self.assertEqual(set(owner.admission_bindings), set(owner.cache.entries))
        self.assertEqual(len(owner.proofs), 1)
        owner.dispose()

    def test_capacity_rejection_still_raises_on_invalid_input_before_insert(self):
        for enabled in (False, True):
            f = fixtures.Validation()
            f.setUp()
            owner = physical.PhysicalOwner(f.f.ident, f.f.registry, capacity=1, validation_mode=SCOPED)
            owner.begin(f.f.req('INVALID', [1, 2, 3, 4, 5, 6]), {'cluster_id': 0})
            f.project_all(owner, 6)
            # Invalid captured source shape fails gather even though capacity rejects.
            owner.active['inputs'][0].shape = (2, 6, 4)
            observer = CPUOwnerDiagnostics()
            from contextlib import nullcontext
            with self.assertRaisesRegex(ValueError, 'invalid gather positions'):
                with observer.observe('REJECT_ERROR') if enabled else nullcontext():
                    f.complete(owner)
            self.assertFalse(owner.validation.open)
            self.assertFalse(owner.cache.entries)
            owner.abort()  # Existing caller cleanup contract, no silent fallback.
            self.assertIsNone(owner.active)
            if enabled:
                self.assertEqual(observer.report()['counts']['admission_candidates'], 0)

    def test_wrappers_restore_after_failure_without_masking_primary(self):
        original = physical.blob
        observer = CPUOwnerDiagnostics()
        with self.assertRaisesRegex(RuntimeError, 'primary'):
            with observer.observe('FAILURE'):
                raise RuntimeError('primary')
        self.assertIs(physical.blob, original)
        self.assertIsNone(CURRENT.get())
        with self.assertRaisesRegex(ValueError, 'one diagnostic session'):
            with observer.observe('REUSE_OBSERVER'):
                pass

    def test_cpu_guard_precedes_serialization_and_no_raw_detail_export(self):
        tensor = ArrayTensor([1.], [1, 1, 1])
        tensor.device = 'cuda:0'
        observer = CPUOwnerDiagnostics()
        with observer.observe('CPU_GUARD'):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                physical.tensor_proof(tensor)
        self.assertEqual(observer.report()['counts']['serialized_tensor_bytes'], 0)
        result = harness.run_fixture('REUSE', CPUOwnerDiagnostics())
        exported = json.dumps(result['diagnostics'])
        for forbidden in ('source_token_ids', 'input_text', 'fresh_positions', 'target_input_proof'):
            self.assertNotIn(forbidden, exported)

    def test_reuse_keeps_three_validation_counters_separate(self):
        result = harness.run_fixture('REUSE', CPUOwnerDiagnostics())
        integrity = result['integrity']
        counts = result['diagnostics']['counts']
        self.assertEqual(integrity['full_entry_proof_count'], counts['full_entry_full_calls'])
        self.assertEqual(integrity['admission_entry_proof_count'], counts['admission_entry_full_calls'])
        self.assertGreater(integrity['consumed_tensor_proof_count'], 0)
        self.assertGreater(integrity['full_entry_proof_count'], 0)
        self.assertGreater(integrity['admission_entry_proof_count'], 0)

    def test_scalar_autoregressive_tokens_text_and_logits_under_observer(self):
        # Existing test checks byte-identical logits and generated IDs across modes.
        # Also compare all generated text/IDs/logit bytes off/on via an in-memory spy.
        from semcache.edgelora import r5_subsequence_generation as generation
        original = generation.generate_loop
        snapshots = []
        def generate(*args, **kwargs):
            result = original(*args, **kwargs)
            snapshots.append((result['generated_token_ids'], result['generated_text'], result['logits_blob']))
            return result
        for enabled in (False, True):
            f = fixtures.Validation()
            f.setUp()
            from contextlib import nullcontext
            with CPUOwnerDiagnostics().observe('SCALAR_GENERATION') if enabled else nullcontext():
                with patch.object(generation, 'generate_loop', generate):
                    f.test_scalar_autoregressive_fixture_logits_and_tokens_unchanged()
        self.assertEqual(snapshots[:4], snapshots[4:])


# Dynamically expose only the selected existing safety tests, preserving their
# assertions and exceptions. No native tests are converted to skips or passes.
def safety_case(name):
    def run(self):
        f = fixtures.Validation()
        f.setUp()
        with CPUOwnerDiagnostics().observe(name):
            getattr(f, name)()
    return run


class ObservedIntegrity(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='')
        env.start()
        self.addCleanup(env.stop)


for _name in (
    'test_corruption_before_authorization_all_roles', 'test_corruption_after_authorization_all_roles',
    'test_tensor_replacement_even_equal_values_rejected',
    'test_equal_byte_replacement_before_request_verification_rejected',
    'test_metadata_policy_registry_and_physical_tampering',
    'test_eviction_disposal_and_abort_expire_records', 'test_failure_cannot_resurrect_authorization',
    'test_old_verification_record_cannot_grant_a_new_request',
    'test_new_request_revalidates_resident_entry', 'test_private_snapshot_has_no_cache_alias',
    'test_chu_pbr_remain_legal_after_records_expire',
):
    setattr(ObservedIntegrity, _name, safety_case(_name))


if __name__ == '__main__':
    unittest.main()
