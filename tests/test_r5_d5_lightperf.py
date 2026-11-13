"""No-skips stdlib contracts. These are never native OPT/GPU fidelity approval."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest, header, listener, connect, encode
from semcache.edgelora.r5_d5_lightperf_io import (SummaryWriter, fresh, verify, source_identity,
                                               SCHEMA, MODE, BASE, BRANCH)
from semcache.edgelora.r5_d5_lightperf_workload import stream, preflight, knobs, matrix, CONDITIONS
from semcache.edgelora.r5_d5_lightperf_emulation import (BytePacer, PacedSocket, LightChannel,
                                                      UDServiceEmulator, serve_light_sparse)
from semcache.edgelora.r5_d5_lightperf_timing import generate_loop, LightProfiler, SessionLifecycle
from semcache.edgelora.r5_d5_lightperf_analysis import categories, describe, break_even, compare, verify_run
from semcache.edgelora.r5_i5_workload import TEMPLATE, routing
from semcache.edgelora.r4c_workload import policy
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r5_sparse_qkv import SparseSession, finish_request
from semcache.edgelora.r5_subsequence_generation import generate_loop as strict_loop
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner, EXACT, APPROX
from semcache.edgelora.r5_subsequence_integrity import SCOPED
from semcache.edgelora.user_device import cpu_deltas
import test_r5_sparse_physical as physical_fixture


def workload(dataset='qasper', n=50):
    special = dict(bos_token_id=2, eos_token_id=2, pad_token_id=1)
    tasks, blocks = [], []
    for i in range(n):
        sample = dict(dataset=dataset, version='LongBench-v1', id=f'S{i}', group_id=f'G{i}')
        ids = [1, 2, 3, 4 + i]
        prepared = dict(sample=sample, token_ids=ids, template_version=TEMPLATE,
                        max_new_tokens=1, input_text='fixture')
        tasks.append(dict(request_id=f'P{i:06d}T', sample_id=sample['id'], dataset=dataset,
                          group_id=sample['group_id'], **routing(sample), token_ids=ids, prepared=prepared,
                          input_text='fixture', policy=policy(special, 'quality', 1), scored=True,
                          role='SCORED_TARGET', request_sequence_number=0))
    # Preserve original local request sequence numbers, with independent blocks.
    start = 0
    for width in (5, 15, 20, 10):
        chunk = tasks[start:start + width]
        if not chunk:
            break
        for j, t in enumerate(chunk):
            t['request_sequence_number'] = j
        b = dict(dataset=dataset, track='NATURAL', repetition=0, priming_requests=0,
                 routing_seed=42, tasks=chunk, continuity_breaks=[])
        b['block_id'] = digest(b)
        blocks.append(b)
        start += width
    return dict(template_version=TEMPLATE, clusters=1, seed=42, special_tokens=special, blocks=blocks), tasks


def contract(w, tasks, count):
    selected, covered = [], 0
    for b in w['blocks']:
        if covered >= count:
            break
        selected.append(b['block_id'])
        covered += len(b['tasks'])
    return dict(dataset=tasks[0]['dataset'], request_count=count, source_blocks=selected,
                tasks_sha256=digest(tasks[:count]), session_ordinals=list(range(count)),
                reset='EMPTY_ONCE_PER_CONDITION', cache_lifetime_requests=count,
                session_contract='NEW_CONTINUOUS_SESSION_FROM_ORIGINAL_ORDER_V1',
                acknowledgement='REVIEWED_NEW_CONTINUOUS_IDENTITY')


class Clock:
    def __init__(self):
        self.now = 10.
        self.sleeps = []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds


class IOTests(unittest.TestCase):
    def test_fresh_protection_symlink_parent_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = Path(tmp) / 'd5_lightperf'
            protected = ns / 'sealed'
            protected.mkdir(parents=True)
            for p in (ns, ns / 'sealed/child', Path(tmp) / 'strict'):
                with self.assertRaises(ValueError):
                    fresh(p, [protected], namespace=ns)
            (ns / 'link').symlink_to(protected, target_is_directory=True)
            with self.assertRaises(ValueError):
                fresh(ns / 'link/new', namespace=ns)
            target = fresh(ns / 'new', namespace=ns)
            with self.assertRaises(ValueError):
                fresh(target, namespace=ns)

    def test_summary_roundtrip_no_tensor_dump_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = SummaryWriter(tmp)
            self.assertFalse(hasattr(writer, 'save'))
            self.assertFalse(hasattr(writer, 'blob'))
            writer.write('request.json', dict(tokens=[1, 2], numerical='NOT_EVALUATED'))
            with self.assertRaises(ValueError):
                writer.write('request.json', {})
            for name in ('../a.json', 'x.bin', 'light_inventory.json'):
                with self.assertRaises(ValueError):
                    writer.write(name, {})
            receipt = writer.seal()
            self.assertEqual(receipt, verify(tmp))
            self.assertEqual(receipt['files'], 2)
            with self.assertRaises(ValueError):
                writer.seal()
            Path(tmp, 'request.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify(tmp)

    def test_writer_bounds_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = SummaryWriter(tmp, max_bytes=100, max_files=2)
            with self.assertRaises(ValueError):
                writer.write('large.json', {'x': 'x' * 101})
            writer.write('a.json', {})
            with self.assertRaises(ValueError):
                writer.write('b.json', {})
            Path(tmp, 'extra').write_text('x')
            with self.assertRaises(ValueError):
                writer.seal()

    def test_source_identity_fail_closed_and_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'src').mkdir()
            (root / 'src/a.py').write_text('reviewed')
            state = dict(branch=BRANCH, head='a' * 40, dirty='', tracked='src/a.py', top=tmp)
            def fake_git(root, *args):
                return {('rev-parse', '--show-toplevel'): state['top'], ('branch', '--show-current'): state['branch'],
                        ('rev-parse', 'HEAD'): state['head'], ('status', '--porcelain', '--untracked-files=no'): state['dirty'],
                        ('ls-files',): state['tracked']}[args]
            with patch('semcache.edgelora.r5_d5_lightperf_io.git', side_effect=fake_git), \
                 patch('semcache.edgelora.r5_d5_lightperf_io.subprocess.run', return_value=SimpleNamespace(returncode=0)), \
                 patch('semcache.edgelora.r5_d5_lightperf_io.subprocess.check_output', return_value=b'reviewed'):
                src = source_identity(development=True, expected_root=root)
                self.assertEqual(src['approval_scope'], 'DEVELOPMENT_NOT_GPU_APPROVED')
                kwargs = dict(expected_root=root, reviewed_commit=state['head'], reviewed_fingerprint=src['content_fingerprint'])
                self.assertEqual(source_identity(**kwargs)['approval_scope'], 'CLEAN_REVIEWED_SOURCE')
                with self.assertRaises(ValueError):
                    source_identity(**dict(kwargs, reviewed_commit=BASE))
                with self.assertRaises(ValueError):
                    source_identity(**dict(kwargs, reviewed_fingerprint='bad'))
                for field, bad in (('branch', 'strict'), ('dirty', ' M src/a.py'), ('top', '/wrong'), ('tracked', '')):
                    old = state[field]
                    state[field] = bad
                    with self.assertRaises(ValueError):
                        source_identity(**kwargs)
                    state[field] = old
                (root / 'src/a.py').write_text('changed')
                with self.assertRaises(ValueError):
                    source_identity(**kwargs)


class WorkloadTests(unittest.TestCase):
    def test_three_datasets_stable_prefix_explicit_continuous_contract(self):
        for dataset in ('qasper', '2wikimqa', 'multi_news'):
            w, tasks = workload(dataset)
            five = stream(w, dataset, 5, strict_pilot_tasks=tasks[:5])
            self.assertEqual(five['strict_five_parity'], 'MATCHED_SELECTION_ONLY')
            for count in (20, 50):
                with self.assertRaises(ValueError):
                    stream(w, dataset, count)
                result = stream(w, dataset, count, continuous_contract=contract(w, tasks, count))
                self.assertEqual(result['tasks'][:5], five['tasks'])
                self.assertEqual(result['cache_lifetime_requests'], count)
                self.assertEqual(result['tasks'], tasks[:count])
            self.assertEqual(stream(w, dataset, 5)['strict_five_parity'], 'NOT_MATCHED')

    def test_duplicate_mutation_group_and_insufficient_data_rejected(self):
        w, tasks = workload(n=5)
        with self.assertRaises(ValueError):
            stream(w, 'qasper', 50)
        for kind in ('routing', 'duplicate', 'policy', 'order', 'blockhash'):
            w, tasks = workload()
            if kind == 'routing':
                tasks[0]['user_id'] = 'User_99'
            elif kind == 'duplicate':
                tasks[1]['sample_id'] = tasks[0]['sample_id']
            elif kind == 'policy':
                tasks[0]['policy']['max_new_tokens'] = 2
            elif kind == 'order':
                tasks[0]['request_sequence_number'] = 1
            else:
                w['blocks'][0]['block_id'] = 'bad'
            if kind != 'blockhash':
                b = w['blocks'][0]
                b['block_id'] = digest({k: v for k, v in b.items() if k != 'block_id'})
            with self.assertRaises(ValueError, msg=kind):
                stream(w, 'qasper', 5)
        w, tasks = workload()
        tasks[5]['group_id'] = tasks[0]['group_id']
        tasks[5]['prepared']['sample']['group_id'] = tasks[0]['group_id']
        b = w['blocks'][1]
        b['block_id'] = digest({k: v for k, v in b.items() if k != 'block_id'})
        with self.assertRaisesRegex(ValueError, 'GROUP_SPLIT'):
            stream(w, 'qasper', 5)

    def test_static_bounds_all_scales_and_blocked_50(self):
        _, tasks = workload()
        for n in (5, 20, 50):
            p = preflight(tasks[:n], knobs())
            self.assertEqual(p['requests'], n)
            self.assertEqual(p['status'], 'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING')
            self.assertIsNone(p['measured_resources'])
            self.assertLess(p['gpu_estimate_bytes'], 22 * 2**30)
        heavy = deepcopy(tasks)
        for t in heavy:
            t['token_ids'] = [1] * 512
            t['policy']['max_new_tokens'] = 96
        plan = preflight(heavy, knobs(50, .25))
        self.assertEqual(plan['status'], 'BLOCKED_INFEASIBLE')
        self.assertIn('UD_OPERATION_CALIBRATION_REQUIRED', plan['reasons'])
        plan = preflight(heavy, knobs(50, .25), ud_operation_seconds=.1)
        self.assertIn('DEADLINE_OR_SLURM_BOUND', plan['reasons'])
        self.assertIn('INSUFFICIENT_CPUS', preflight(tasks[:5], knobs(), available={'cpus': 1})['reasons'])

    def test_invalid_options_and_matrix_never_submits(self):
        for value in (0, -1, float('nan'), float('inf'), True, 49):
            with self.assertRaises(ValueError):
                knobs(value)
        for factor in (0, .75, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                knobs(None, factor)
        m = matrix()
        self.assertEqual(m['selected_jobs'], [])
        self.assertEqual(m['full_grid_condition_count'], 324)


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = physical_fixture.Physical()
        self.fixture.setUp()
        self.req = self.fixture.req('GEN', [1, 2, 3, 4, 5, 6])

    def test_same_greedy_tokens_eos_ties_and_no_logits_dump(self):
        for values in ([0., 3., 3.], [0., 0., 3.]):
            strict = strict_loop(self.req, lambda *args: (values, None, {}), lambda ids: {}, str)
            light = generate_loop(self.req, lambda *args: (values, None), lambda ids: {}, str)
            self.assertEqual(strict['generated_token_ids'], light['generated_token_ids'])
            self.assertEqual(strict['termination'], light['termination'])
            self.assertNotIn('logits_blob', light)
            self.assertEqual(light['independent_full_vocabulary_audit'], 'NOT_EVALUATED')
            self.assertEqual(light['application_e2e_ms'], (light['completion_ns'] - light['arrival_ns']) / 1e6)
            self.assertEqual(light['application_ttft_ms'], (light['first_available_ns'] - light['arrival_ns']) / 1e6)

    def test_nonfinite_and_pipeline_exceptions_propagate(self):
        with self.assertRaises(ValueError):
            generate_loop(self.req, lambda *args: ([float('nan')], None), lambda ids: self.fail(), str)
        def fail(*args):
            raise RuntimeError('completion failed')
        with self.assertRaisesRegex(RuntimeError, 'completion failed'):
            generate_loop(self.req, lambda *args: ([1.], None), fail, str)

    def test_profiler_exclusive_scopes_not_double_counted(self):
        p = LightProfiler()
        with p.activate('R'), p.scope('outer'):
            with p.scope('inner'):
                time.sleep(.001)
        r = p.finish()
        stats = {x['component']: x for x in r['components']}
        self.assertAlmostEqual(sum(x['exclusive_wall_ms'] for x in stats.values()), stats['outer']['inclusive_wall_ms'])
        self.assertGreater(sum(x['inclusive_wall_ms'] for x in stats.values()), stats['outer']['inclusive_wall_ms'])
        self.assertIsNone(r['profiler_overhead_ms'])
        self.assertIsNone(r['gpu_kernel_ms'])

    def test_final_ack_stop_bye_and_no_session_restart(self):
        _, tasks = workload()
        s = SessionLifecycle(tasks)
        with self.assertRaises(ValueError):
            s.stop_after_ack(tasks[0]['adapter_id'])
        for t in tasks:
            s.begin(t)
            final = s.ack(t['request_id'])
            if final:
                s.stop_after_ack(t['adapter_id'])
                with self.assertRaises(ValueError):
                    s.stop_after_ack(t['adapter_id'])
                s.bye(t['adapter_id'])
        s.finish()
        with self.assertRaises(ValueError):
            s.begin(tasks[0])

    def test_missing_bye_wrong_ack_and_order_fail_closed(self):
        _, tasks = workload(n=5)
        s = SessionLifecycle(tasks)
        with self.assertRaises(ValueError):
            s.begin(tasks[1])
        s.begin(tasks[0])
        with self.assertRaises(ValueError):
            s.ack(tasks[1]['request_id'])
        with self.assertRaises(ValueError):
            s.finish()


class EmulationTests(unittest.TestCase):
    def test_byte_scheduling_disabled_all_rates_zero_and_deadline(self):
        for rate in (None, 200, 100, 50):
            clock = Clock()
            p = BytePacer(rate, clock=clock, sleep=clock.sleep)
            p.reserve(0, 20)
            self.assertIsNone(p.report()['achieved_mbps'])
            start = clock()
            p.reserve(10000, 20)
            p.commit(10000, 10000, start)
            self.assertEqual(p.bytes, 10000)
            self.assertAlmostEqual(sum(clock.sleeps), 0 if rate is None else 10000 * 8 / rate / 1e6)
            if rate:
                self.assertAlmostEqual(p.report()['achieved_mbps'], rate)
                with self.assertRaises(TimeoutError):
                    p.reserve(65536, clock() + 1e-9)

    def test_partial_sends_and_no_busy_spin(self):
        clock = Clock()
        p = BytePacer(50, clock=clock, sleep=clock.sleep)
        raw = SimpleNamespace(settimeout=lambda x: None, send=lambda data: len(data) // 2)
        sock = PacedSocket(raw, p)
        sock.settimeout(2)
        self.assertEqual(sock.send(b'x' * 1000), 500)
        self.assertEqual(p.bytes, 500)
        self.assertGreater(len(clock.sleeps), 0)
        self.assertEqual(sock.send(b''), 0)
        broken = BytePacer(50, clock=lambda: 10., sleep=lambda x: None)
        with self.assertRaisesRegex(ValueError, 'clock'):
            broken.reserve(1, 20.)

    def test_ud_measured_operation_cost_output_parity_factors(self):
        fixture = physical_fixture.Physical()
        fixture.setUp()
        x = [1.] * 24
        expected = cpu_deltas(x, fixture.matrices, 0, 4, torch_backend=False)
        for factor in (1., .5, .25):
            clock = Clock()
            service = UDServiceEmulator(factor, clock=clock, sleep=clock.sleep)
            def compute():
                result = cpu_deltas(x, fixture.matrices, 0, 4, torch_backend=False)
                clock.now += .004
                return result
            result = service.run_lora(compute, deadline=20)
            self.assertEqual(result, expected)
            self.assertAlmostEqual(service.report()['achieved_service_factor'], factor)
            self.assertAlmostEqual(service.report()['pacing_ms'], 4 * (1 / factor - 1))
            self.assertEqual(service.report()['operations'], 1)
        with self.assertRaises(TimeoutError):
            service.run_lora(compute, deadline=clock() + .001)

    def test_real_loopback_both_directions_acks_payloads_all_knobs(self):
        f = physical_fixture.Physical()
        f.setUp()
        expected = None
        for rate, factor in ((None, 1.), (200, 1.), (100, .5), (50, .25)):
            server = listener()
            client = connect(server.getsockname()[1])
            peer, _ = server.accept()
            server.close()
            es = LightChannel(client, 'fixture_session', f.ident['adapter_matrix_fingerprint'], bandwidth_mbps=rate)
            ud = LightChannel(peer, 'fixture_session', f.ident['adapter_matrix_fingerprint'], bandwidth_mbps=rate)
            service = UDServiceEmulator(factor)
            errors, rows = [], []
            def serve():
                try:
                    serve_light_sparse(ud, f.ident, f.registry, f.matrices, service=service,
                                       deadline=time.monotonic() + 20, fixture=True, trace=rows.append)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    ud.close()
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                req = f.req('R', [1, 2, 3, 4, 5, 6])
                session = SparseSession(es, req, 'R_s000', fixture=True)
                x = ArrayTensor([1.] * 24, [1, 6, 4])
                values = []
                for layer in range(2):
                    for role in 'qkv':
                        values.extend(session.correction(layer, role, x, list(range(6))).values)
                session.complete()
                finish_request(es, req, [3])
                es.send(header('STOP', 'shutdown', -1))
                es.recv(kind='BYE')
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len([r for r in rows if r.get('kind') == 'COMPLETE_REQUEST']), 1)
                self.assertEqual(es.summary['send']['wire_bytes'], ud.summary['recv']['wire_bytes'])
                self.assertEqual(es.summary['recv']['wire_bytes'], ud.summary['send']['wire_bytes'])
                self.assertEqual(es.pacer.bytes, es.summary['send']['wire_bytes'])
                self.assertEqual(ud.pacer.bytes, ud.summary['send']['wire_bytes'])
                self.assertEqual(es.records, [])
                self.assertEqual(ud.records, [])
                if rate:
                    self.assertGreater(es.pacer.pacing_seconds, 0)
                    self.assertGreater(ud.pacer.pacing_seconds, 0)
                expected = values if expected is None else expected
                self.assertEqual(values, expected)
            finally:
                es.close()
                ud.close()
                thread.join(2)

    def test_corrupt_frame_still_rejected(self):
        f = physical_fixture.Physical()
        f.setUp()
        server = listener()
        raw = connect(server.getsockname()[1])
        peer, _ = server.accept()
        server.close()
        channel = LightChannel(peer, 'fixture_session', f.ident['adapter_matrix_fingerprint'])
        try:
            from semcache.edgelora.r3a_artifacts import pack
            h = dict(header('HIDDEN', 'R', 0, 'float32', [1, 1, 4]), session_id='fixture_session',
                     adapter_sha256=f.ident['adapter_matrix_fingerprint'], sequence=0)
            packet, _ = encode(h, pack([1., 2., 3., 4.], 'float32'))
            raw.sendall(packet[:-1] + bytes([packet[-1] ^ 1]))
            with self.assertRaises(ValueError):
                channel.recv()
            self.assertTrue(channel.failed)
        finally:
            raw.close()
            channel.close()


class PhysicalParityTests(unittest.TestCase):
    def execute_sequence(self, light):
        f = physical_fixture.Physical()
        f.setUp()
        events = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        from semcache.edgelora.r4c_evidence import Evidence
        evidence = None if light else Evidence(tmp.name)
        owner = PhysicalOwner(f.ident, f.registry, capacity=1024, validation_mode=SCOPED,
                              evidence=evidence, trace=lambda r: events.append(r['kind']))
        # Explicit fixture-only trigger exercises original PBR arithmetic.
        owner.bridge.pbr_interval = 2
        class Session:
            def correction(self, layer, role, x, positions):
                from semcache.edgelora.r5_sparse_qkv import gather
                if not positions:
                    return None
                values = cpu_deltas(gather(x, positions).values, f.matrices, layer, 4, torch_backend=False)
                n = len(positions) * 4
                index = 'qkv'.index(role)
                return ArrayTensor(values[index * n:(index + 1) * n], [1, len(positions), 4])
        outputs = []
        for rid, ids, mode in (('S', [1, 2, 3, 4, 5, 6], EXACT),
                                ('T', [1, 2, 3, 4, 5, 6, 7], EXACT),
                                ('X', [9, 2, 3, 4, 5, 6, 7], EXACT),
                                ('Z', [0, 1, 2, 3, 4, 5, 6], APPROX)):
            req = f.req(rid, ids)
            session = Session()
            p = LightProfiler()
            context = p.activate(rid) if light else __import__('contextlib').nullcontext()
            with context:
                owner.begin(req, {'cluster_id': 0}, mode=mode, allow_approximate=mode == APPROX)
                for layer in range(2):
                    x = ArrayTensor([float(t) for t in ids for _ in range(4)], [1, len(ids), 4])
                    for role in 'qkv':
                        owner.project(layer, role, x, lambda t: ArrayTensor([2 * v for v in t.values], t.shape), session)
                with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact', return_value=2.):
                    result = owner.complete(completed=True, attentions=['FIXTURE_ONLY'] * 2)
            outputs.append(dict(mode=result['mode'], skipped=result['physical_projection_rows_reused'],
                                hashes=[r['output']['sha256'] for r in result['records']],
                                fresh=[r['fresh_positions'] for r in result['records']],
                                reused=[r['reused_positions'] for r in result['records']],
                                occupancy=owner.cache.physical_tensor_bytes, entries=len(owner.cache.entries), events=list(events)))
        entry = next(iter(owner.cache.entries.values()))
        owner.begin(f.req('BAD', list(entry.token_ids)), {'cluster_id': 0}, mode=APPROX, allow_approximate=True)
        entry.tensors[0][0].values[0] += 1
        with self.assertRaises(ValueError):
            owner.valid(entry)
        owner.dispose()
        self.assertEqual(bool(list(Path(tmp.name).rglob('*.bin'))), not light)
        return outputs

    def test_same_physical_decisions_rows_classification_policy_and_hashes(self):
        strict, light = self.execute_sequence(False), self.execute_sequence(True)
        self.assertEqual(strict, light)
        self.assertGreater(light[1]['skipped'], 0)
        self.assertEqual(light[2]['skipped'], 0)
        self.assertEqual(light[-1]['mode'], APPROX)
        self.assertTrue(all(x['occupancy'] <= 1024 for x in light))
        self.assertIn('EVICT', light[-1]['events'])
        self.assertIn('PBR', light[-1]['events'])


class AnalysisTests(unittest.TestCase):
    def receipts(self, root, *, mutate_run=None, mutate_audit=None):
        """Synthetic test receipts only; never returned as real CPU approval."""
        _, tasks = workload(n=5)
        src = dict(branch=BRANCH, strict_baseline=BASE, files={}, content_fingerprint=digest({}),
                   commit='TEST_FIXTURE_ONLY', approval_scope='CLEAN_REVIEWED_SOURCE')
        p = dict(schema=SCHEMA, mode=MODE, source=src, dataset='qasper', count=5,
                 conditions=list(CONDITIONS), performance_knobs=knobs(),
                 stream=dict(tasks=tasks, tasks_sha256=digest(tasks)))
        p['plan_id'] = digest(p)
        plan_root, cpu_root, run_root = (root / name for name in ('plan', 'cpu', 'run'))
        for path in (plan_root, cpu_root, run_root):
            path.mkdir()
        writer = SummaryWriter(plan_root)
        writer.write('plan.json', p)
        pi = writer.seal()
        a = dict(status='PASS', failed=0, skipped=0, scope='CLEAN_NATIVE_CPU_APPROVAL', source=src, plan_inventory=pi)
        if mutate_audit:
            mutate_audit(a)
        writer = SummaryWriter(cpu_root)
        writer.write('cpu_validation.json', a)
        ai = writer.seal()
        rows = []
        for i, t in enumerate(tasks):
            ids, text = [3], 'fixture'
            arrival, first, end = i * 20_000_000, i * 20_000_000 + 1_000_000, i * 20_000_000 + 10_000_000
            rows.append(dict(**{k: t[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')},
                             session_ordinal=i, condition=CONDITIONS[0], input_tokens=len(t['token_ids']), output_tokens=1,
                             generated_token_ids=ids, generated_text=text, generated_token_ids_sha256=digest(ids),
                             generated_text_sha256=hashlib.sha256(text.encode()).hexdigest(), arrival_ns=arrival,
                             first_available_ns=first, completion_ns=end, application_e2e_ms=10., application_ttft_ms=1.,
                             skipped_qkv_rows=0, fresh_qkv_rows=384, physical_hit=False, categories=dict.fromkeys('ABCD', 0),
                             cache_occupancy_bytes=0, scorer_version='TEST_FIXTURE_ONLY', quality=dict(metric='fixture', score=1.)))
        r = dict(schema=SCHEMA, mode=MODE, source=src, plan_id=p['plan_id'], plan_inventory=pi,
                 cpu_inventory=ai, status='COMPLETE', condition=CONDITIONS[0], performance_knobs=knobs(),
                 dataset='qasper', scale=5, tasks_sha256=digest(tasks), measurement_scope='NATIVE_LIGHT',
                 gpu=dict(uuid='GPU_TEST_FIXTURE_ONLY'), backend=dict(version='TEST_FIXTURE_ONLY'),
                 precision_policy='TEST_FIXTURE_ONLY', cpu_concurrency=4, warmup_requests=0,
                 independent_full_vocabulary_audit='NOT_EVALUATED', correctness_qualified_speedup='NOT_ESTABLISHED',
                 resources=dict(gpu_free_after_load_bytes=12 * 2**30, peak_gpu_allocated_bytes=2**30,
                                host_available_bytes=32 * 2**30, cpu_count=4), errors=[], requests=rows, sequence_span_ms=90.)
        if mutate_run:
            mutate_run(r)
        writer = SummaryWriter(run_root)
        writer.write('run.json', r)
        writer.seal()
        return run_root, plan_root, cpu_root

    def test_receipt_binding_and_generated_hashes_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.receipts(Path(tmp))
            r, binding = verify_run(*paths)
            self.assertEqual(len(r['requests']), 5)
            self.assertEqual(set(binding), {'run', 'plan', 'cpu'})
            Path(paths[0], 'run.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_run(*paths)

    def test_resealed_bad_tokens_resources_counts_and_stale_approvals_rejected(self):
        def bad_tokens(r):
            r['requests'][0]['generated_token_ids'] = [99]
        def bad_rows(r):
            r['requests'][0]['fresh_qkv_rows'] += 1
        def bad_resources(r):
            r['resources']['cpu_count'] = 1
        def bad_plan(r):
            r['plan_id'] = 'old'
        def bad_claim(r):
            r['independent_full_vocabulary_audit'] = 'PASS'
        for mutation in (bad_tokens, bad_rows, bad_resources, bad_plan, bad_claim):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    verify_run(*self.receipts(Path(tmp), mutate_run=mutation))
        for mutation in (lambda a: a.update(skipped=1), lambda a: a.update(scope='STRICT_APPROVAL'),
                         lambda a: a.update(plan_inventory={}), lambda a: a.update(source={})):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    verify_run(*self.receipts(Path(tmp), mutate_audit=mutation))

    def test_changed_prepared_input_rejected_by_plan_reconstruction(self):
        from semcache.edgelora import r5_d5_lightperf as api
        from semcache.edgelora.r3a_artifacts import THRESHOLDS
        with tempfile.TemporaryDirectory() as tmp:
            src = dict(approval_scope='DEVELOPMENT_NOT_GPU_APPROVED', commit='fixture', content_fingerprint='fixture')
            p = dict(schema=SCHEMA, mode=MODE, source=src, conditions=list(CONDITIONS),
                     validation_mode='REQUEST_SCOPED_VERIFIED', thresholds=THRESHOLDS, warmup_requests=0,
                     approx_requires_explicit_run_opt_in=True, correctness_qualified_speedup='NOT_ESTABLISHED',
                     original_experiment='fixture', multi='fixture', dataset='qasper', count=5,
                     performance_knobs=knobs(), continuous_contract=None, strict_pilot=None,
                     ud_operation_seconds=None, dataset_inventories={'qasper': 'old_input_sha'})
            p['plan_id'] = digest(p)
            writer = SummaryWriter(tmp)
            writer.write('plan.json', p)
            writer.seal()
            with patch.object(api, 'source_identity', return_value=src), \
                 patch.object(api, 'material', return_value=dict(dataset_inventories={'qasper': 'new_input_sha'})):
                with self.assertRaisesRegex(ValueError, 'changed prepared input'):
                    api.load_plan(tmp)

    def test_stats_and_measured_crossings_no_extrapolation(self):
        stats = describe([1, 2, 3, 4, 5])
        self.assertEqual(stats['p95'], 5)
        self.assertEqual(stats['inference'], 'EXPLORATORY_SINGLE_TRIAL')
        self.assertEqual(break_even([dict(x=50, delta_e2e_ms=2), dict(x=100, delta_e2e_ms=3)])['status'], 'NO_OBSERVED_BREAK_EVEN')
        crossing = break_even([dict(x=50, delta_e2e_ms=-2), dict(x=100, delta_e2e_ms=2)])
        self.assertEqual(crossing['brackets'][0]['estimated_linear_crossing'], 75)
        self.assertIsNone(crossing['extrapolation'])

    def test_categories_and_bad_w3(self):
        tasks = [dict(request_id='S', token_ids=[1, 2, 3, 4, 5, 6]), dict(request_id='T', token_ids=[1, 2, 3, 9, 4, 5, 6])]
        physical = dict(records=[dict(consumed=[dict(source_request_id='S', source_positions=[0, 3], target_positions=[0, 3]),
                                               dict(source_request_id='S', source_positions=[3, 6], target_positions=[4, 7])])],
                        physical_projection_rows_reused=6)
        counts, sources = categories(physical, tasks, 'T')
        self.assertEqual(counts, dict(A=3, B=0, C=3, D=0))
        self.assertEqual(sources, ['S'])
        physical['physical_projection_rows_reused'] = 99
        with self.assertRaises(ValueError):
            categories(physical, tasks, 'T')

    def test_comparison_device_mismatch_and_approx_separate(self):
        r = dict(condition=CONDITIONS[0], dataset='qasper', scale=5, tasks_sha256='x', source={}, performance_knobs=knobs(),
                 gpu={'uuid': 'GPU_A'}, backend={'torch': 'x'}, precision_policy='fp16', cpu_concurrency=4,
                 warmup_requests=0, measurement_scope='NATIVE_LIGHT', sequence_span_ms=10,
                 requests=[dict(request_id='R', application_e2e_ms=10, generated_token_ids=[3])])
        other = deepcopy(r)
        other['condition'] = CONDITIONS[1]
        other['requests'][0]['application_e2e_ms'] = 5
        c = compare(r, other)
        self.assertEqual(c['delta_e2e_ms'], -5)
        self.assertEqual(c['speedup'], 2)
        other['gpu']['uuid'] = 'GPU_B'
        self.assertEqual(compare(r, other)['status'], 'NOT_COMPARABLE')
        other['gpu']['uuid'] = 'GPU_A'
        other['condition'] = CONDITIONS[2]
        self.assertIn('APPROX_SEPARATE', compare(r, other)['quality_curve'])

    def test_gpu_run_is_hard_blocked_even_with_opt_in(self):
        from semcache.edgelora.r5_d5_lightperf import run
        with self.assertRaisesRegex(ValueError, 'BLOCKED_NATIVE_WORKER'):
            run(enable_gpu_experiment=True, enable_approximate_experiment=True)


if __name__ == '__main__':
    unittest.main()
