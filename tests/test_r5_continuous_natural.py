"""Explicit CPU fixtures; never evidence about original NATURAL opportunities."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest
from semcache.edgelora.r5_continuous_natural import (
    DATASETS, CONDITIONS, CPUProbe, build_sessions, runtime_task, execute_sessions,
    validate_lifecycle, static_analysis, resource_assessment, cpu_validate)
from semcache.edgelora.r5_multi_artifacts import registry_for, sharing
from semcache.edgelora.r5_i5_contracts import request
from semcache.edgelora.r4c_workload import policy
from semcache.edgelora.r5_subsequence_natural_io import verify

ROOT = Path(__file__).resolve().parents[1]


def reseal(w):
    for b in w['blocks']:
        b.pop('block_id', None); b['block_id'] = digest(b)
    w.pop('experiment_id', None); w['experiment_id'] = digest(w)
    return w


def fixture_workload():
    """Only tests construct requests. Production command only imports sealed data."""
    blocks = []
    for dataset in DATASETS:
        for rep in range(2):
            for cohort in range(2):
                tasks = []
                for j in range(5):
                    n = 5*cohort+j; aid = 'Adapter_'+'ABC'[n % 3]
                    user = f'User_{10*(n % 3)+1:02d}'
                    ids = [1, 2, 3, 4, 10+n, 40, 41, 42]
                    text = f'EXPLICIT_CPU_FIXTURE {dataset} {n}'
                    p = policy(dict(bos_token_id=2, eos_token_id=2, pad_token_id=1), 'quality', 32)
                    tasks.append(dict(request_id=f'P{n:06d}T', pair_id=f'P{n:06d}',
                        sample_id=f'{dataset}_sample_{n}', dataset=dataset, group_id=str(n),
                        input_text=text, token_ids=ids, policy=p, user_id=user, adapter_id=aid,
                        role='SCORED_TARGET', scored=True, strict=j == 0, request_sequence_number=j,
                        prepared=dict(sample=dict(id=f'{dataset}_sample_{n}', context=text),
                                      token_ids=ids, input_text=text, max_new_tokens=32)))
                blocks.append(dict(dataset=dataset, repetition=rep, track='NATURAL', priming_requests=0, tasks=tasks))
    return reseal(dict(blocks=blocks, clusters=1, TEST_FIXTURE_ONLY=True))


def identities():
    return {aid: dict(schema='R5_PHYSICAL_IDENTITY_V1', model='fixture', model_revision='fixture',
            base_fingerprint='a'*64, tokenizer={'fixture': 'integer'}, adapter_id=aid,
            adapter_matrix_fingerprint=str(n+1)*64, dtype='float32', layers=2, hidden_size=4,
            rank=8, device='cpu', gpu_uuid='CPU_FIXTURE_ONLY', opt_contract='b'*64,
            attention_context='standalone causal; all-ones mask; positions from zero')
            for n, aid in enumerate(('Adapter_A', 'Adapter_B', 'Adapter_C'))}


class ContinuousNatural(unittest.TestCase):
    def setUp(self):
        self.w = fixture_workload()
        self.registry = registry_for(['Adapter_A', 'Adapter_B', 'Adapter_C'], 'expanded')
        self.sessions = build_sessions(self.w, self.registry)

    def test_exact_counts_payloads_order_and_original_unchanged(self):
        original = deepcopy(self.w)
        rebuilt = build_sessions(self.w, self.registry)
        self.assertEqual(rebuilt, self.sessions)
        self.assertEqual(self.w, original)
        self.assertEqual(len(rebuilt), 3)
        for si, s in enumerate(rebuilt):
            inv = s['invocations']; self.assertEqual(len(inv), 20)
            self.assertEqual(set(Counter(i['original_sample_id'] for i in inv).values()), {2})
            self.assertEqual(len({i['original_sample_id'] for i in inv}), 10)
            self.assertEqual([i['original_block_index'] for i in inv], [b for b in range(4*si, 4*si+4) for _ in range(5)])
            self.assertEqual([i['task'] for i in inv], [t for b in self.w['blocks'][4*si:4*si+4] for t in b['tasks']])
            inv[0]['task']['token_ids'][0] = 999
        self.assertEqual(self.w, original)

    def test_identifier_collisions_resolved_and_strict_request_schema_compatible(self):
        ids = []; pairs = []
        for s in self.sessions:
            self.assertEqual(len({i['original_request_id'] for i in s['invocations']}), 10)
            self.assertEqual(len({i['original_pair_id'] for i in s['invocations']}), 10)
            for i in s['invocations']:
                pairs.append(i['invocation_pair_id'])
                for condition in CONDITIONS:
                    t = runtime_task(i, condition); ids.append(t['request_id'])
                    req = request(t, t['token_ids'], t['policy'], identities()[t['adapter_id']], fixture=True)
                    self.assertEqual(req['sample_id'], i['original_sample_id'])
                    self.assertEqual(req['request_sequence_number'], i['session_sequence_number'])
                    self.assertEqual(t['token_ids'], i['task']['token_ids'])
        self.assertEqual(len(ids), len(set(ids))); self.assertEqual(len(ids), 120)
        self.assertEqual(len(pairs), len(set(pairs))); self.assertEqual(len(pairs), 60)

    def test_lifecycle_owns_real_empty_objects_and_separates_every_condition_dataset(self):
        r = validate_lifecycle(self.sessions, self.registry, identities())
        self.assertEqual(r['fresh_lifecycle_initializations'], 6)
        self.assertEqual(r['block_checkpoints'], 24)
        self.assertEqual(r['internal_cache_resets'], 0)
        self.assertEqual(r['actual_model_initializations'], 'NOT_MEASURED')
        self.assertEqual(len(r['trace']), 120)

    def test_real_policy_clock_and_frequency_persist_across_blocks_and_repeats(self):
        # Exercise actual arrival accounting only: no attention, admissions or HITs.
        class ArrivalProbe(CPUProbe):
            def __init__(self, *args):
                super().__init__(*args); self.expected = {}; self.clocks = []
            def execute(p, state, a, invocation, task, condition):
                if a.owner is not None:
                    k = id(a.owner); expected = p.expected.get(k, 0)
                    self.assertEqual(a.owner.cache.now, expected)
                    windows = a.owner.bridge.extractor.extract(task['token_ids'])
                    keys = [a.owner.bridge.matcher.key(0, w) for w in windows]
                    a.owner.bridge.metrics.arrive(keys)
                    p.expected[k] = expected+1
                    self.assertEqual(a.owner.cache.now, expected+1)
                    self.assertEqual(a.owner.bridge.metrics.frequencies[keys[0]], expected+1)
                    self.assertFalse(a.owner.cache.entries)
            def checkpoint(p, state, block):
                super().checkpoint(state, block)
                p.clocks.append([a.owner.cache.now for a in state.adapters.values() if a.owner])
        p = ArrivalProbe(self.registry, identities()); execute_sessions(self.sessions, p)
        cstates = [s for _, c, s in p.initializations if c == CONDITIONS[1]]
        for state in cstates:
            self.assertEqual(sum(a.owner.cache.now for a in state.adapters.values()), 20)
            self.assertTrue(all(not a.owner.cache.entries for a in state.adapters.values()))
        self.assertEqual(len(p.closed), 6)

    def test_state_reconstruction_detected_and_failure_disposes_owner(self):
        class Bad(CPUProbe):
            def checkpoint(p, state, block):
                a = next(iter(state.adapters.values()))
                a.channel = object()
        p = Bad(self.registry, identities())
        with self.assertRaisesRegex(ValueError, 'reconstructed'):
            execute_sessions(self.sessions, p)
        self.assertEqual(len(p.closed), 1)

    def test_execution_failure_always_closes_and_stops_without_retry(self):
        class Bad(CPUProbe):
            def execute(p, *args): raise ValueError('intentional fixture failure')
        p = Bad(self.registry, identities())
        with self.assertRaisesRegex(ValueError, 'intentional'):
            execute_sessions(self.sessions, p)
        self.assertEqual(len(p.initializations), 1); self.assertEqual(len(p.closed), 1)

    def test_no_policy_encoder_or_reuse_called_by_cpu_probe(self):
        with patch('semcache.edgelora.r5_subsequence_physical.PhysicalOwner.begin', side_effect=AssertionError('begin forbidden')), \
             patch('semcache.edgelora.r5_continuous_natural.NoInferenceEncoder.encode', side_effect=AssertionError('encode forbidden')):
            validate_lifecycle(self.sessions, self.registry, identities())

    def test_malformed_or_rewritten_workloads_fail_closed(self):
        changes = [lambda w: w['blocks'].pop(),
                   lambda w: w['blocks'][1].update(dataset='multi_news'),
                   lambda w: w['blocks'][2].update(repetition=0),
                   lambda w: w['blocks'][0]['tasks'][0].update(role='PRIMING_SOURCE'),
                   lambda w: w['blocks'][0]['tasks'][0].update(user_id='User_11'),
                   lambda w: w['blocks'][2]['tasks'][0].update(sample_id='extra'),
                   lambda w: w['blocks'][0]['tasks'][0].update(request_sequence_number=2),
                   lambda w: w['blocks'][0]['tasks'][0].update(input_text='changed')]
        for change in changes:
            with self.subTest(change=change):
                w = deepcopy(self.w); change(w); reseal(w)
                with self.assertRaises(ValueError): build_sessions(w, self.registry)
        w = deepcopy(self.w); w['blocks'][0]['tasks'][0]['token_ids'][0] = 999
        with self.assertRaises(ValueError): build_sessions(w, self.registry)

    def static(self, left, right, user='User_01', adapter='Adapter_A', same_sample=False):
        a, b = deepcopy(self.sessions[0]['invocations'][:2])
        a['task'].update(token_ids=left, input_text='identical text', user_id='User_01', adapter_id='Adapter_A')
        b['task'].update(token_ids=right, input_text='identical text', user_id=user, adapter_id=adapter)
        if same_sample: b['task']['sample_id'] = a['task']['sample_id']
        return static_analysis(dict(dataset='qasper', invocations=[a, b]), self.registry)

    def test_static_prefix_same_position_nonprefix_and_shifted(self):
        r = self.static([1,2,3,4,5,6], [1,2,3,9,5,6])
        self.assertEqual(r['occurrence_counts']['shared_prefix_windows'], 1)
        r = self.static([1,2,3,4,5,6], [9,2,3,4,5,6])
        self.assertEqual(r['occurrence_counts']['same_position_nonprefix_windows'], 3)
        self.assertEqual(r['occurrence_counts']['shared_prefix_windows'], 0)
        r = self.static([1,2,3,4,5,6], [9,1,2,3,4,5,6])
        self.assertEqual(r['occurrence_counts']['shifted_position_nonprefix_windows'], 4)
        self.assertEqual(r['gpu_physical_hits'], 'NOT_MEASURED')
        self.assertEqual(r['resident_entries'], 'NOT_MEASURED')

    def test_static_occurrence_denominators_and_prefix_shift_are_disjoint(self):
        r = self.static([1]*5, [1]*5)
        self.assertEqual(r['occurrence_counts']['shared_prefix_windows'], 3)
        self.assertEqual(r['occurrence_counts']['shifted_prefix_target_windows'], 6)
        self.assertEqual(sum(r['occurrence_counts'].values()), 9)
        self.assertEqual(r['matching_target_window_union'], 3)

    def test_experiment_repeats_are_separate_from_distinct_sample_identical_requests(self):
        r = static_analysis(self.sessions[0], self.registry)
        self.assertEqual(r['ordered_request_pairs'], 190)
        self.assertEqual(r['exact_requests']['experiment_repeated_sample_pairs'], 10)
        self.assertEqual(r['exact_requests']['distinct_sample_identical_request_pairs'], 0)
        r = self.static([1,2,3], [1,2,3])
        self.assertEqual(r['exact_requests']['distinct_sample_identical_request_pairs'], 1)
        self.assertEqual(r['exact_requests']['experiment_repeated_sample_pairs'], 0)

    def test_sharing_cross_adapter_and_explicit_user_denial(self):
        r = self.static([1,2,3], [1,2,3], 'User_11', 'Adapter_B')
        self.assertEqual(r['sharing_eligible_occurrence_counts']['shared_prefix_windows'], 0)
        self.assertEqual(r['sharing_pair_counts']['CROSS_ADAPTER_REUSE_FORBIDDEN'], 1)
        r = self.static([1,2,3], [1,2,3], 'User_02')
        self.assertEqual(r['sharing_eligible_occurrence_counts']['shared_prefix_windows'], 1)
        self.registry['users']['User_02']['allow_cross_user_reuse'] = False
        r = self.static([1,2,3], [1,2,3], 'User_02')
        self.assertEqual(r['sharing_pair_counts']['SHARING_NOT_AUTHORIZED'], 1)
        self.assertIsNone(sharing(self.registry, 'User_01', 'User_01', 'Adapter_A', 'Adapter_A'))

    def test_resource_overflow_is_reported_without_shrinking_or_raising_limits(self):
        tasks = [deepcopy(i['task']) for i in self.sessions[0]['invocations']]
        for t in tasks:
            t['token_ids'] = list(range(512)); t['policy']['max_new_tokens'] = 96
        r = resource_assessment(tasks)
        self.assertEqual(r['existing_gates']['status'], 'FAIL')
        self.assertGreater(r['condition_evidence_forecast_bytes'], r['condition_limit_bytes'])
        self.assertGreater(r['estimated_run_files'], r['file_limit'])
        self.assertEqual(len(tasks), 20)

    def test_missing_real_inputs_write_sealed_failure_and_never_fixture_counts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            out = Path(tmp)/'audit'
            r = cpu_validate(Path(tmp)/'missing', Path(tmp)/'multi', out)
            self.assertEqual(r['phase1_status'], 'BLOCKED'); self.assertEqual(r['real_data_validation'], 'BLOCKED')
            self.assertNotIn('sessions', r); self.assertIn('blocker', r)
            self.assertEqual(r['gpu_physical_hits'], 'NOT_MEASURED'); verify(out)
            with self.assertRaises(ValueError): cpu_validate(Path(tmp)/'missing', Path(tmp)/'multi', out)

    def test_strict_real_loader_rejects_sealed_fixture(self):
        from semcache.edgelora.r5_i5_evidence import seal as old_seal
        from semcache.edgelora.r5_subsequence_natural_io import put
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            original = Path(tmp)/'original'; original.mkdir()
            multi = Path(tmp)/'multi'; multi.mkdir()
            put(original/'i5_workload_manifest.json', self.w)
            put(original/'i5_dataset_manifest.json', {})
            old_seal(original)
            r = cpu_validate(original, multi, Path(tmp)/'audit')
            self.assertEqual(r['phase1_status'], 'FAIL')
            self.assertIn('real I5 plan', r['blocker'])

    def test_report_serialization_with_explicit_mocked_validated_input_plumbing(self):
        # Only tests bypass the real-assets loader. These temporary artifacts are
        # deleted, and provide no real-data PASS or opportunity counts.
        args = (self.w, {}, {'TEST_FIXTURE_ONLY': True}, {}, {}, {},
                {aid: {} for aid in identities()}, self.registry,
                SimpleNamespace(identity={'TEST_FIXTURE_ONLY': True}))
        def predicted(task, *args):
            return {'physical_identity': identities()[task['adapter_id']]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            original = Path(tmp)/'original'; original.mkdir()
            multi = Path(tmp)/'multi'; multi.mkdir()
            out = Path(tmp)/'audit'
            with patch('semcache.edgelora.r5_subsequence_natural.inputs', return_value=args), \
                 patch('semcache.edgelora.r5_i5r3b_transport.predicted_request', side_effect=predicted), \
                 patch('semcache.edgelora.r5_subsequence_natural.transport', return_value={'TEST_FIXTURE_ONLY': True}), \
                 patch('semcache.edgelora.r5_continuous_natural.Evidence.verify', return_value={'TEST_FIXTURE_ONLY': True}):
                r = cpu_validate(original, multi, out)
            self.assertEqual(r['phase1_status'], 'PASS', r.get('blocker')); self.assertEqual(len(r['sessions']), 3)
            self.assertEqual(r['gpu_physical_hits'], 'NOT_MEASURED')
            self.assertEqual(r['lifecycle']['fresh_lifecycle_initializations'], 6)
            self.assertTrue((out/'multi_news/static_opportunities.json').exists())
            verify(out)

    def test_cpu_environment_and_cli_exclude_gpu_actions(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
                cpu_validate('missing', 'missing_multi', 'unused')
        p = subprocess.run(['python3', str(ROOT/'scripts/154_r5_continuous_natural.py'), 'run'], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0); self.assertIn('invalid choice', p.stderr)


if __name__ == '__main__':
    unittest.main()
