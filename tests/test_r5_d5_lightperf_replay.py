"""Memory-only replay/preflight/analysis tests. Synthetic IDs are not real data."""
from copy import deepcopy
from collections import Counter
import os
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest
from semcache.edgelora.r5_d5_lightperf_workload import DATASETS, CONDITIONS, stream, preflight, knobs
from semcache.edgelora.r5_d5_lightperf_replay import replay_stream, validate_replay_stream, replay_enabled
from semcache.edgelora.r5_d5_lightperf_replay_analysis import run_segments, compare_segments
from semcache.edgelora.r5_d5_lightperf_receipts import bind_replay_provenance
from semcache.edgelora.r5_d5_lightperf_runtime import launch_scope
from test_r5_d5_lightperf import workload


def replay_fixture(dataset='qasper', count=20, pool_size=10):
    w, original = workload(dataset, n=pool_size)
    selected = replay_stream(w, dataset, count, enable_continuous_replay=True,
                             baseline_tasks=original[:5], baseline_seed=w['seed'])
    return w, original, selected


class ReplayWorkload(unittest.TestCase):
    def test_frozen_baseline_and_pilot_are_protected_output_inputs(self):
        from semcache.edgelora.r5_d5_lightperf_runtime import protected
        p = dict(original_experiment='/data/khuss/original', multi='/data/khuss/multi',
                 base_prepared='/data/khuss/base', dataset_paths={'qasper': '/data/khuss/qasper'},
                 baseline_plan='/data/khuss/frozen_five', strict_pilot='/data/khuss/strict_five')
        paths = protected(p, '/data/khuss/new_plan', '/data/khuss/new_audit')
        self.assertIn(p['baseline_plan'], paths)
        self.assertIn(p['strict_pilot'], paths)
        self.assertEqual(len(paths), len(set(paths)))

    def test_preflight_report_is_static_and_never_gpu_approval(self):
        from semcache.edgelora import r5_d5_lightperf as api
        _, _, selected = replay_fixture()
        p = dict(dataset='qasper', count=20, workload_mode='CONTINUOUS_REPLAY', stream=selected,
                 resources=preflight(selected['tasks'], knobs()))
        with patch.object(api, 'load_plan', return_value=(p, {})):
            report = api.preflight_report('TEST_ONLY')
        self.assertEqual(report['status'], 'PASS')
        self.assertFalse(report['gpu_execution_approved'])
        self.assertEqual(report['statistics']['repeated_requests'], 10)
        self.assertEqual(report['GPU_status'], 'GPU_NOT_RUN')

    def test_extended_job_missing_approval_is_fail_closed(self):
        import subprocess
        from pathlib import Path
        job = Path(__file__).resolve().parents[1] / 'jobs/r5_d5_lightperf_replay.sh'
        p = subprocess.run(['bash', str(job)], env={'PATH': os.environ['PATH']}, text=True, capture_output=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('D5L_REPO_ROOT', p.stderr)
        text = job.read_text()
        self.assertIn('--enable-extended-experiment --enable-continuous-replay', text)
        self.assertIn('#SBATCH --time=04:30:00', text)
        self.assertIn('#SBATCH --mem=64G', text)
        self.assertNotIn('#SBATCH --array', text)
    def test_source_review_is_still_required(self):
        from semcache.edgelora.r5_d5_lightperf_io import source_identity
        with self.assertRaises(ValueError):
            source_identity(reviewed_commit=None, reviewed_fingerprint=None)
    def test_three_datasets_all_scales_and_statistics(self):
        for dataset in DATASETS:
            for count in (5, 20, 50):
                with self.subTest(dataset=dataset, count=count):
                    w, source, selected = replay_fixture(dataset, count)
                    self.assertTrue(validate_replay_stream(selected))
                    self.assertEqual(len(selected['tasks']), count)
                    self.assertEqual(len({t['request_id'] for t in selected['tasks']}), count)
                    stats = selected['statistics']
                    self.assertEqual(stats['original_request_count'], 10)
                    self.assertEqual(stats['unique_samples'], min(10, count))
                    self.assertEqual(stats['repeated_requests'], max(0, count - 10))
                    self.assertEqual(stats['repeat_ratio'], max(0, count - 10) / count)
                    self.assertEqual(stats['complete_cycles'], count // 10)
                    self.assertEqual(selected['first_five_match']['status'], 'MATCHED_SELECTION_ONLY')

    def test_deterministic_prefixes_preserve_first_cycle_and_payload(self):
        w, source, five = replay_fixture(count=5)
        before = deepcopy(w)
        twenty = replay_stream(w, 'qasper', 20, enable_continuous_replay=True)
        fifty = replay_stream(w, 'qasper', 50, enable_continuous_replay=True)
        self.assertEqual(w, before)
        self.assertEqual(five['tasks'], twenty['tasks'][:5])
        self.assertEqual(twenty['tasks'], fifty['tasks'][:20])
        self.assertEqual(twenty['tasks'][:10], source)
        self.assertEqual(twenty, replay_stream(w, 'qasper', 20, enable_continuous_replay=True))
        for i, task in enumerate(fifty['tasks']):
            restored = dict(task, request_id=source[i % 10]['request_id'])
            self.assertEqual(restored, source[i % 10])
            self.assertEqual(fifty['occurrences'][i]['occurrence_index'], i // 10)

    def test_original_pool_not_reselected_for_hits_and_nat_default_preserved(self):
        w, source, _ = replay_fixture()
        self.assertEqual(stream(w, 'qasper', 5)['tasks'], source[:5])
        with self.assertRaises(ValueError):
            stream(w, 'qasper', 20)
        with self.assertRaises(ValueError):
            replay_stream(w, 'qasper', 20)
        self.assertFalse(replay_enabled({}))
        for p in ({'workload_mode': 'CONTINUOUS_REPLAY'}, {'enable_continuous_replay': True}, {'workload_mode': 'unknown'}):
            with self.assertRaises(ValueError):
                replay_enabled(p)

    def test_original_duplicates_remain_forbidden(self):
        w, _, _ = replay_fixture()
        w['blocks'][0]['tasks'][1]['sample_id'] = w['blocks'][0]['tasks'][0]['sample_id']
        block = w['blocks'][0]
        block['block_id'] = digest({k: v for k, v in block.items() if k != 'block_id'})
        with self.assertRaises(ValueError):
            replay_stream(w, 'qasper', 50, enable_continuous_replay=True)

    def test_duplicate_execution_ids_and_resealed_mutations_rejected(self):
        for change in ('id', 'route', 'tokens', 'policy', 'occurrence', 'block', 'stats'):
            _, _, selected = replay_fixture()
            if change == 'id':
                selected['tasks'][10]['request_id'] = selected['tasks'][0]['request_id']
            elif change == 'route':
                selected['tasks'][10]['user_id'] = 'User_99'
            elif change == 'tokens':
                selected['tasks'][10]['token_ids'][0] = 999
            elif change == 'policy':
                selected['tasks'][10]['policy']['max_new_tokens'] = 2
            elif change == 'occurrence':
                selected['occurrences'][10]['occurrence_index'] = 0
            elif change == 'block':
                selected['occurrences'][10]['original_block_id'] = 'changed'
            else:
                selected['statistics']['repeated_requests'] = 0
            selected['tasks_sha256'] = digest(selected['tasks'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_replay_stream(selected)

    def test_seed_or_five_payload_mismatch_labeled(self):
        w, original, _ = replay_fixture()
        changed = deepcopy(original[:5])
        changed[0]['input_text'] += 'different'
        selected = replay_stream(w, 'qasper', 20, enable_continuous_replay=True,
                                  baseline_tasks=changed, baseline_seed=123)
        self.assertEqual(selected['first_five_match']['status'], 'NOT_MATCHED')
        self.assertIn('BASELINE_SEED_DIFFERS', selected['first_five_match']['reasons'])
        self.assertIn('FIRST_FIVE_TASK_PAYLOAD_OR_ORDER_DIFFERS', selected['first_five_match']['reasons'])

    def test_group_boundary_split_rejected_without_reordering(self):
        w, _, _ = replay_fixture()
        a, b = w['blocks'][0]['tasks'][-1], w['blocks'][1]['tasks'][0]
        b['group_id'] = b['prepared']['sample']['group_id'] = a['group_id']
        block = w['blocks'][1]
        block['block_id'] = digest({k: v for k, v in block.items() if k != 'block_id'})
        with self.assertRaisesRegex(ValueError, 'GROUP_SPLIT'):
            replay_stream(w, 'qasper', 5, enable_continuous_replay=True)

    def test_small_pool_and_partial_cycle_are_explicit(self):
        w, _, _ = replay_fixture(pool_size=3)
        selected = replay_stream(w, 'qasper', 20, enable_continuous_replay=True)
        self.assertEqual(selected['statistics']['unique_samples'], 3)
        self.assertEqual(selected['statistics']['repeated_requests'], 17)
        self.assertEqual(selected['statistics']['partial_cycle_requests'], 2)
        self.assertEqual(selected['first_five_match']['status'], 'NOT_MATCHED')
        self.assertTrue(validate_replay_stream(selected))

    def test_explicit_launch_approvals_and_baseline_knobs(self):
        _, _, selected = replay_fixture()
        p = dict(workload_mode='CONTINUOUS_REPLAY', enable_continuous_replay=True, stream=selected,
                 count=20, performance_knobs=knobs(), source=dict(commit='reviewed'))
        for extended, replay in ((False, False), (True, False), (False, True)):
            with self.assertRaises(ValueError):
                launch_scope(p, CONDITIONS[0], 'reviewed', 'GPU-TEST', False, extended, replay)
        launch_scope(p, CONDITIONS[0], 'reviewed', 'GPU-TEST', False, True, True)
        with self.assertRaises(ValueError):
            launch_scope(p, CONDITIONS[2], 'reviewed', 'GPU-TEST', False, True, True)
        # Generic future sensitivity interface remains; L2 job template pins baseline.
        launch_scope(dict(p, performance_knobs=knobs(50)), CONDITIONS[0], 'reviewed', 'GPU-TEST', False, True, True)
        from semcache.edgelora.r5_d5_lightperf import run
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            run(experiment='none', cpu_audit='none', condition=CONDITIONS[0], output_dir='none',
                source_commit='reviewed', expected_gpu_uuid='GPU-TEST', enable_gpu_experiment=False,
                enable_extended_experiment=True, enable_continuous_replay=True)

    def test_memory_deadline_bounds_and_no_hit_discount(self):
        for dataset in DATASETS:
            for count in (5, 20, 50):
                _, _, selected = replay_fixture(dataset, count)
                tasks = deepcopy(selected['tasks'])
                budget = 96 if dataset == 'multi_news' else 32
                for task in tasks:
                    task['token_ids'] = [1] * 512  # Resource fixture only, never a production task.
                    task['policy']['max_new_tokens'] = budget
                p = preflight(tasks, knobs())
                self.assertEqual(p['max_forwards'], count * budget)
                self.assertEqual(p['forecast_seconds'], 600 + 5 * count * budget)
                self.assertEqual(p['worker_deadline_seconds'], 7200)
                self.assertEqual(p['gpu_allocation_ceiling_bytes'], 22 * 2**30)
                blocked = count == 50 or dataset == 'multi_news' and count == 20
                self.assertEqual(p['status'] == 'BLOCKED_INFEASIBLE', blocked)
        self.assertIn('INSUFFICIENT_CPUS', preflight(tasks[:5], knobs(), available={'cpus': 1})['reasons'])

    def test_receipt_provenance_binding_and_approval_coverage(self):
        from semcache.edgelora.r5_d5_lightperf import approval_modules, TEST_MODULES
        _, _, selected = replay_fixture()
        occurrence = selected['occurrences'][10]
        row = {k: occurrence[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id', 'session_ordinal')}
        bind_replay_provenance(row, occurrence)
        self.assertEqual(row['replay_provenance'], occurrence)
        with self.assertRaises(ValueError):
            bind_replay_provenance(dict(row, adapter_id='FOREIGN'), occurrence)
        self.assertEqual(approval_modules({}), list(TEST_MODULES))
        self.assertIn('test_r5_d5_lightperf_replay', approval_modules(dict(workload_mode='CONTINUOUS_REPLAY', enable_continuous_replay=True)))


class ReplayPhysical(unittest.TestCase):
    def test_repeated_rows_reuse_and_condition_reset_with_original_operator(self):
        from semcache.edgelora.r5_subsequence_physical import PhysicalOwner, EXACT, APPROX
        from semcache.edgelora.r5_subsequence_integrity import SCOPED
        from semcache.edgelora.r5_sparse_qkv import gather
        from semcache.edgelora.r5_i5_contracts import request
        from semcache.edgelora.r3b_fixture import ArrayTensor
        from semcache.edgelora.user_device import cpu_deltas
        import test_r5_sparse_physical as fixture
        f = fixture.Physical()
        f.setUp()
        _, _, selected = replay_fixture()
        history = []
        for mode in (EXACT, APPROX):
            owners = {aid: PhysicalOwner(dict(f.ident, adapter_id=aid), f.registry, capacity=4096, validation_mode=SCOPED)
                      for aid in ('Adapter_A', 'Adapter_B', 'Adapter_C')}
            self.assertTrue(all(not o.cache.entries for o in owners.values()))
            skipped = []
            class Session:
                def correction(self, layer, role, x, positions):
                    if not positions:
                        return None
                    data = cpu_deltas(gather(x, positions).values, f.matrices, layer, 4, torch_backend=False)
                    n = len(positions) * 4
                    i = 'qkv'.index(role)
                    return ArrayTensor(data[i * n:(i + 1) * n], [1, len(positions), 4])
            for task in selected['tasks']:
                owner = owners[task['adapter_id']]
                req = request(task, task['token_ids'], task['policy'], owner.bridge.identity, fixture=True)
                owner.begin(req, {'cluster_id': 0}, mode=mode, allow_approximate=mode == APPROX)
                x = ArrayTensor([float(t) for t in task['token_ids'] for _ in range(4)], [1, len(task['token_ids']), 4])
                for layer in range(2):
                    for role in 'qkv':
                        owner.project(layer, role, x, lambda t: ArrayTensor([2 * v for v in t.values], t.shape), Session())
                with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact', return_value=2.):
                    result = owner.complete(completed=True, attentions=['SCALAR_TEST_ONLY'] * 2)
                skipped.append(result['physical_projection_rows_reused'])
                for record in result['records']:
                    for source in record['consumed']:
                        source_task = next(t for t in selected['tasks'] if t['request_id'] == source['source_request_id'])
                        self.assertEqual(source_task['adapter_id'], task['adapter_id'])
            self.assertEqual(skipped[0], 0)
            self.assertTrue(any(n > 0 for n in skipped[10:]))
            self.assertTrue(all(o.cache.physical_tensor_bytes <= 4096 for o in owners.values()))
            history.append(skipped)
            for owner in owners.values():
                owner.dispose()
        self.assertEqual(history[0], history[1])

    def test_full_continuous_lifecycle_and_no_cross_adapter_claim(self):
        from semcache.edgelora.r5_d5_lightperf_timing import SessionLifecycle
        from semcache.edgelora.r5_multi_artifacts import registry_for, sharing
        _, _, selected = replay_fixture(count=50)
        life = SessionLifecycle(selected['tasks'])
        for task in selected['tasks']:
            life.begin(task)
            if life.ack(task['request_id']):
                life.stop_after_ack(task['adapter_id'])
                life.bye(task['adapter_id'])
        life.finish()
        registry = registry_for(['Adapter_A', 'Adapter_B', 'Adapter_C'], 'expanded')
        self.assertEqual(sharing(registry, 'User_01', 'User_11', 'Adapter_A', 'Adapter_B'), 'CROSS_ADAPTER_REUSE_FORBIDDEN')


def analysis_fixture(count=20):
    _, _, selected = replay_fixture(count=count)
    rows = []
    for i, (t, o) in enumerate(zip(selected['tasks'], selected['occurrences'])):
        rows.append(dict(request_id=t['request_id'], sample_id=t['sample_id'], user_id=t['user_id'], adapter_id=t['adapter_id'],
            session_ordinal=i, replay_provenance=o, input_tokens=4, output_tokens=1, generated_token_ids=[3],
            application_ttft_ms=2., application_e2e_ms=10., arrival_ns=i * 20_000_000, completion_ns=i * 20_000_000 + 10_000_000,
            fresh_qkv_rows=300, skipped_qkv_rows=84, physical_hit=True, cache_occupancy_bytes=1000,
            policy_counts=dict(INSERT=1, EVICT=0, CHU=1, PBR=0), cache_completion_costs=dict(owner_completion_ms=3.)))
    return dict(dataset='qasper', scale=count, condition=CONDITIONS[0], tasks_sha256=selected['tasks_sha256'],
        workload_mode='CONTINUOUS_REPLAY', replay_stream_sha256=digest(selected), source={}, performance_knobs=knobs(),
        backend={}, gpu=dict(uuid='TEST'), precision_policy='TEST', cpu_concurrency={}, warmup_requests=0,
        measurement_scope='CPU_SYNTHETIC_TEST_NOT_GPU', sequence_span_ms=count * 20., requests=rows)


class ReplayAnalysis(unittest.TestCase):
    def test_offline_receipt_provenance_and_five_request_compatibility(self):
        """Mock byte-store controls field binding; not filesystem/native approval."""
        from pathlib import Path
        from semcache.edgelora import r5_d5_lightperf_analysis as audit
        from semcache.edgelora.r5_d5_lightperf import approval_modules
        from semcache.edgelora.r5_d5_lightperf_io import ENGINE, SCHEMA, MODE, BRANCH, BASE
        from semcache.edgelora.r5_d5_lightperf_receipts import WireSummary
        for replay in (False, True):
            n = 20 if replay else 5
            _, _, selected = replay_fixture(count=n)
            root = Path('/data/khuss/UNIT_TEST_ONLY_REPLAY')
            plan, cpu = root / 'plan', root / 'cpu'
            source = dict(branch=BRANCH, strict_baseline=BASE, files={}, content_fingerprint=digest({}),
                          approval_scope='CLEAN_REVIEWED_SOURCE', commit='TEST_ONLY')
            p = dict(engine=ENGINE, schema=SCHEMA, mode=MODE, source=source, dataset='qasper', count=n,
                     stream=selected if replay else dict(tasks=selected['tasks'], tasks_sha256=selected['tasks_sha256']),
                     conditions=list(CONDITIONS), performance_knobs=knobs())
            if replay:
                p.update(workload_mode='CONTINUOUS_REPLAY', enable_continuous_replay=True, seed=selected['seed'])
            p['plan_id'] = digest(p)
            pi, ai, ri = {'test': 'plan'}, {'test': 'cpu'}, {'test': 'run'}
            a = dict(engine=ENGINE, versions={}, modules=approval_modules(p), status='PASS', errors=0, failed=0, skipped=0,
                     scope='CLEAN_NATIVE_CPU_APPROVAL', source=source, plan_inventory=pi, native_parity='TINY_OPT_STRICT_LIGHT_CPU_PASS')
            docs, inventories, rows = {}, {str(plan): pi, str(cpu): ai, str(root): ri}, []
            for i, task in enumerate(selected['tasks']):
                row = dict(**{k: task[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')},
                    session_ordinal=i, condition=CONDITIONS[0], input_tokens=4, output_tokens=1, generated_token_ids=[3],
                    generated_token_ids_sha256=digest([3]), generated_text='TEST',
                    generated_text_sha256=__import__('hashlib').sha256(b'TEST').hexdigest(),
                    arrival_ns=i * 20_000_000, first_available_ns=i * 20_000_000 + 1_000_000,
                    completion_ns=i * 20_000_000 + 10_000_000, application_ttft_ms=1., application_e2e_ms=10.,
                    skipped_qkv_rows=0, fresh_qkv_rows=384, categories=dict.fromkeys('ABCD', 0), physical_hit=False,
                    cache_occupancy_bytes=0, quality=dict(metric='TEST_ONLY', score=1.), scorer_version='TEST_ONLY',
                    wire=WireSummary.empty(), ud_compute={})
                if replay:
                    bind_replay_provenance(row, selected['occurrences'][i])
                name = f'request_{i:03d}.json'
                docs[str(root / name)] = dict(row, receipt_scope='PRIMARY_BODY_FINAL_TIMING_IN_RUN_JSON')
                row.update(primary_receipt=name, primary_receipt_sha256='TEST_HASH')
                rows.append(row)
            r = dict(engine=ENGINE, schema=SCHEMA, mode=MODE, source=source, plan_id=p['plan_id'], plan_inventory=pi,
                cpu_inventory=ai, status='COMPLETE', condition=CONDITIONS[0], performance_knobs=knobs(), dataset='qasper', scale=n,
                tasks_sha256=selected['tasks_sha256'], versions={}, measurement_scope='NATIVE_LIGHT', gpu={'uuid': 'TEST_ONLY'},
                backend={'test': True}, precision_policy='TEST_ONLY', cpu_concurrency={'test': True}, warmup_requests=0,
                independent_full_vocabulary_audit='NOT_EVALUATED', correctness_qualified_speedup='NOT_ESTABLISHED',
                resources=dict(gpu_free_after_load_bytes=12 * 2**30, peak_gpu_allocated_bytes=2**30,
                               host_available_bytes=32 * 2**30, cpu_count=4), errors=[], requests=rows, ud={}, sequence_span_ms=n * 20.)
            if replay:
                r.update(workload_mode='CONTINUOUS_REPLAY', replay_stream_sha256=digest(selected), replay_statistics=selected['statistics'])
            for aid in {t['adapter_id'] for t in selected['tasks']}:
                path = Path(str(root) + '__ud_' + aid)
                members = [x for x in rows if x['adapter_id'] == aid]
                inventory = {'test': aid}
                inventories[str(path)] = inventory
                r['ud'][aid] = dict(path=str(path), inventory=inventory)
                docs[str(path / 'ud.json')] = dict(status='COMPLETE', source=source, plan_inventory=pi, cpu_inventory=ai,
                    completed_request_ids=[x['request_id'] for x in members], final_bye_sent=True,
                    user_device_cuda_used=False, full_model_loaded=False, wire={x['request_id']: x['wire'] for x in members})
                for i, row in enumerate(members):
                    docs[str(path / f'request_{i:03d}.json')] = dict(request_id=row['request_id'], cpu_projection_rows=384, compute={})
            docs.update({str(plan / 'plan.json'): p, str(cpu / 'cpu_validation.json'): a, str(root / 'run.json'): r})
            with patch.object(audit, 'verify', side_effect=lambda path: inventories[str(path)]), \
                 patch.object(audit, 'read_json', side_effect=lambda path: deepcopy(docs[str(path)])), \
                 patch.object(audit, 'file_hash', return_value='TEST_HASH'):
                self.assertEqual(audit.verify_run(root, plan, cpu)[0]['scale'], n)
                if replay:
                    r['requests'][10]['replay_provenance']['occurrence_index'] = 0
                    with self.assertRaises(ValueError):
                        audit.verify_run(root, plan, cpu)

    def test_five_request_bins_and_missing_metrics_are_not_zero(self):
        for count in (20, 50):
            result = run_segments(analysis_fixture(count))
            self.assertEqual(len(result['windows']), count // 5)
            self.assertEqual(result['windows'][0]['repeated_occurrences'], 0)
            self.assertEqual(result['windows'][2]['repeated_occurrences'], 5)
            self.assertEqual(result['whole_session']['qkv_skip_ratio'], 84 / 384)
            timing = result['windows'][0]['timings']['semantic_encoding_ms']
            self.assertEqual(timing['status'], 'NOT_MEASURED')
            self.assertIsNone(timing['statistics'])
            self.assertEqual(result['windows'][0]['timings']['owner_completion_ms']['statistics']['mean'], 3.)

    def test_speedup_token_agreement_and_comparability(self):
        full = analysis_fixture()
        other = deepcopy(full)
        other['condition'] = CONDITIONS[1]
        for row in other['requests']:
            row['application_e2e_ms'] = 20.
        result = compare_segments(full, other)
        self.assertEqual(result['whole_session']['speedup'], .5)
        self.assertEqual(result['whole_session']['measured_outcome'], 'FULL_FASTER')
        self.assertEqual(result['whole_session']['generated_token_agreement_rate'], 1.)
        self.assertEqual(result['observed_break_even'], 'NO_OBSERVED_BREAK_EVEN')
        self.assertIsNone(result['threshold_estimate'])
        other['gpu']['uuid'] = 'OTHER'
        self.assertEqual(compare_segments(full, other)['whole_session']['status'], 'NOT_COMPARABLE')


if __name__ == '__main__':
    unittest.main()
