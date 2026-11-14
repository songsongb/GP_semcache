"""No-skips stdlib worker/approval/transport failure contracts, not native parity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import connect, digest, header, listener
from semcache.edgelora.r5_d5_lightperf_io import ENGINE, MODE, SCHEMA, SummaryWriter, read_json
from semcache.edgelora.r5_d5_lightperf_workload import CONDITIONS, knobs
from semcache.edgelora.r5_d5_lightperf_emulation import LightChannel, UDServiceEmulator, serve_light_sparse
from semcache.edgelora.r5_d5_lightperf_receipts import WireSummary, wire_equal, compact_profile
from semcache.edgelora import r5_d5_lightperf_runtime as runtime
import test_r5_sparse_physical as physical_fixture


class GPUUUIDValidation(unittest.TestCase):
    UUID = '880f065c-37fc-f142-05e1-f043a8fb49ff'
    OTHER = '880f065c-37fc-f142-05e1-f043a8fb49fe'

    def test_same_uuid_with_and_without_prefix(self):
        for actual, expected in ((self.UUID, 'GPU-' + self.UUID), ('GPU-' + self.UUID, self.UUID),
                                 (self.UUID, self.UUID), ('GPU-' + self.UUID, 'GPU-' + self.UUID),
                                 (self.UUID.upper(), 'GPU-' + self.UUID)):
            with self.subTest(actual=actual, expected=expected):
                self.assertTrue(runtime.gpu_uuid_matches(actual, expected))

    def test_different_uuid_rejected(self):
        self.assertFalse(runtime.gpu_uuid_matches(self.UUID, 'GPU-' + self.OTHER))
        with self.assertRaisesRegex(ValueError, 'GPU UUID/device/context/12GiB'):
            runtime.validate_execution_gpu(dict(uuid=self.UUID), 'GPU-' + self.OTHER,
                device_count=1, free=18 * 2**30, max_position_embeddings=2048)

    def test_invalid_and_unavailable_values_rejected_on_either_side(self):
        for value in (None, '', 'UNAVAILABLE', 'UNKNOWN', 'None', 'not-a-uuid', 123, b'GPU-' + self.UUID.encode(),
                      'GPU-' + self.UUID[:-1] + 'z'):
            for actual, expected in ((value, self.UUID), (self.UUID, value)):
                with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                    runtime.gpu_uuid_matches(actual, expected)

    def test_parseable_but_disallowed_uuid_representations_rejected(self):
        for value in (self.UUID.replace('-', ''), '{' + self.UUID + '}', 'urn:uuid:' + self.UUID,
                      runtime.uuid.UUID(self.UUID)):
            runtime.uuid.UUID(str(value))  # Valid for Python's permissive parser.
            for actual, expected in ((value, self.UUID), (self.UUID, value)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runtime.gpu_uuid_matches(actual, expected)
        for value in (' ' + self.UUID, self.UUID + '\n', 'gpu-' + self.UUID, 'GPU-GPU-' + self.UUID):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.gpu_uuid_matches(value, self.UUID)

    def test_matching_uuid_does_not_bypass_resource_guards(self):
        valid = dict(device_count=1, free=12 * 2**30, max_position_embeddings=2048)
        runtime.validate_execution_gpu(dict(uuid=self.UUID), 'GPU-' + self.UUID, **valid)
        for key, bad_values in (('device_count', (0, 2)), ('free', (0, 12 * 2**30 - 1)),
                                ('max_position_embeddings', (1024, 4096))):
            for value in bad_values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runtime.validate_execution_gpu(dict(uuid=self.UUID), 'GPU-' + self.UUID,
                                                   **dict(valid, **{key: value}))

    def test_execution_identity_and_receipt_stay_unmodified(self):
        from semcache.edgelora.gpu_identity import validate_gpu, SAME_COMPATIBLE_GPU_CLASS
        prepared = dict(uuid='GPU-' + self.OTHER, name='NVIDIA GeForce RTX 3090')
        gpu = dict(uuid=self.UUID, name='NVIDIA GeForce RTX 3090', device='cuda:0')
        before = deepcopy(gpu)
        receipt = validate_gpu(prepared, gpu, 'cuda:0', SAME_COMPATIBLE_GPU_CLASS)
        runtime.validate_execution_gpu(gpu, 'GPU-' + self.UUID, device_count=1,
                                       free=18 * 2**30, max_position_embeddings=2048)
        self.assertEqual(gpu, before)
        self.assertEqual(validate_gpu(prepared, gpu, 'cuda:0', SAME_COMPATIBLE_GPU_CLASS), receipt)
        self.assertEqual(receipt['execution_gpu_uuid'], self.UUID)


class WorkerContracts(unittest.TestCase):
    def allocation(self):
        return dict(SLURM_JOB_ID='TEST_ONLY', R5_ENABLE_GPU='1', R5_I5_ENABLE_GPU='1', D5L_ENABLE_GPU='1',
                    D5N_NODE='moana-y1', SLURMD_NODENAME='moana-y1', SLURM_CPUS_PER_TASK='4',
                    SLURM_MEM_PER_NODE='65536', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='42', D5L_ALLOCATION_REMAINING_SECONDS='16200')

    def test_allocation_requires_every_guard_without_gpu_import(self):
        env = self.allocation()
        with patch.object(runtime.os, 'sched_getaffinity', return_value={0, 1, 2, 3}):
            self.assertEqual(runtime.allocation_guard(env), 16200)
            for key in ('SLURM_JOB_ID', 'D5L_ENABLE_GPU', 'R5_ENABLE_GPU', 'HF_HUB_OFFLINE', 'PYTHONHASHSEED',
                        'D5L_ALLOCATION_REMAINING_SECONDS', 'SLURM_MEM_PER_NODE', 'SLURMD_NODENAME'):
                e = dict(env)
                e.pop(key)
                with self.assertRaises((ValueError, KeyError)):
                    runtime.allocation_guard(e)
            with self.assertRaises(ValueError):
                runtime.allocation_guard(dict(env, SLURM_ARRAY_TASK_ID='1'))

    def test_approx_extended_count_and_gpu_uuid_gates(self):
        p = dict(count=5, performance_knobs=knobs(), source=dict(commit='current'))
        runtime.launch_scope(p, CONDITIONS[0], 'current', 'GPU-TEST', False, False)
        for condition, commit, gpu, approximate in ((CONDITIONS[2], 'current', 'GPU-TEST', False),
                                                  (CONDITIONS[0], 'old', 'GPU-TEST', False),
                                                  (CONDITIONS[0], 'current', 'unknown', False)):
            with self.assertRaises(ValueError):
                runtime.launch_scope(p, condition, commit, gpu, approximate, False)
        for changed in (dict(p, count=20), dict(p, performance_knobs=knobs(50, .25))):
            with self.assertRaises(ValueError):
                runtime.launch_scope(changed, CONDITIONS[0], 'current', 'GPU-TEST', False, False)
            runtime.launch_scope(changed, CONDITIONS[0], 'current', 'GPU-TEST', False, True)

    def approval_files(self, root, mutation=None):
        from semcache.edgelora.r5_d5_lightperf import TEST_MODULES
        source = dict(approval_scope='CLEAN_REVIEWED_SOURCE', commit='CURRENT_TEST', content_fingerprint='TEST_FIXTURE')
        p = dict(engine=ENGINE, mode=MODE, schema=SCHEMA, source=source,
                 resources=dict(status='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING'))
        p['plan_id'] = digest(p)
        plan, audit = root / 'plan', root / 'audit'
        plan.mkdir()
        audit.mkdir()
        w = SummaryWriter(plan)
        w.write('plan.json', p)
        pi = w.seal()
        a = dict(engine=ENGINE, status='PASS', scope='CLEAN_NATIVE_CPU_APPROVAL', failed=0, errors=0, skipped=0,
                 tests=1, source=source, plan_inventory=pi, modules=list(TEST_MODULES),
                 native_parity='TINY_OPT_STRICT_LIGHT_CPU_PASS', versions={})
        if mutation:
            mutation(a)
        w = SummaryWriter(audit)
        w.write('cpu_validation.json', a)
        w.seal()
        return plan, audit, source

    def test_fresh_native_approval_binding_and_stale_or_skipped_rejection(self):
        mutations = (None, lambda a: a.update(engine='old'), lambda a: a.update(errors=1),
                     lambda a: a.update(skipped=1), lambda a: a.update(modules=['strict_only']),
                     lambda a: a.update(native_parity='NOT_VERIFIED'), lambda a: a.update(plan_inventory={}),
                     lambda a: a.update(versions={'torch': 'old'}), lambda a: a.update(scope='DEVELOPMENT_NOT_GPU_APPROVED'))
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                plan, audit, source = self.approval_files(Path(tmp), mutation)
                with patch.object(runtime, 'source_identity', return_value=source), patch.object(runtime, 'native_versions', return_value={}):
                    if mutation is None:
                        self.assertEqual(runtime.bound_receipts(plan, audit)[0]['engine'], ENGINE)
                    else:
                        with self.assertRaises(ValueError):
                            runtime.bound_receipts(plan, audit)

    def test_source_changes_rejected_by_ud_before_matrix_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, audit, source = self.approval_files(Path(tmp))
            with patch.object(runtime, 'source_identity', return_value=dict(source, commit='changed')):
                with self.assertRaises(ValueError):
                    runtime.bound_receipts(plan, audit)

    def test_cpu_approval_requires_zero_skip_and_clean_source(self):
        from semcache.edgelora import r5_d5_lightperf as api
        for clean, skip in ((True, False), (False, False), (True, True)):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'new_audit'
                p = dict(source=dict(approval_scope='CLEAN_REVIEWED_SOURCE' if clean else 'DEVELOPMENT_NOT_GPU_APPROVED'),
                         resources=dict(status='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING'),
                         original_experiment='original', multi='multi', base_prepared='base', dataset_paths={})
                def test_body():
                    if skip:
                        raise unittest.SkipTest('injected missing coverage')
                suite = unittest.TestSuite([unittest.FunctionTestCase(test_body)])
                def fresh_output(*args):
                    output.mkdir()
                    return output
                with patch.object(api, 'load_plan', return_value=(p, {})), patch.object(api, 'dependencies', return_value={'torch': True}), \
                     patch.dict('sys.modules', torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))), \
                     patch.object(api.unittest.defaultTestLoader, 'loadTestsFromNames', return_value=suite), \
                     patch.object(api, 'fresh', side_effect=fresh_output), patch.object(runtime, 'native_versions', return_value={}):
                    if skip:
                        with self.assertRaisesRegex(ValueError, 'failed/skipped'):
                            api.cpu_validate('plan', output)
                        self.assertFalse(output.exists())
                    else:
                        record = api.cpu_validate('plan', output)
                        self.assertEqual(record['status'], 'PASS' if clean else 'DEVELOPMENT_TESTS_PASS')
                        self.assertEqual(record['scope'], 'CLEAN_NATIVE_CPU_APPROVAL' if clean else 'DEVELOPMENT_NOT_GPU_APPROVED')

    def test_job_template_missing_approval_exits_before_python_gpu(self):
        job = Path(__file__).resolve().parents[1] / 'jobs/r5_d5_lightperf.sh'
        result = subprocess.run(['bash', str(job)], env={'PATH': os.environ['PATH']}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('D5L_REPO_ROOT', result.stderr)

    def test_finalization_never_masks_primary_failure(self):
        primary = TimeoutError('original timeout')
        writer = SimpleNamespace(write=lambda *args: (_ for _ in ()).throw(OSError('disk full')))
        runtime.seal_after(writer, primary, [])
        with self.assertRaises(OSError):
            runtime.seal_after(writer, None, ['cleanup failed'])

    def test_reap_escalates_only_its_own_child(self):
        calls = []
        class Process:
            def poll(self):
                return None
            def terminate(self):
                calls.append('terminate')
            def kill(self):
                calls.append('kill')
            def wait(self, timeout):
                calls.append(('wait', timeout))
                if timeout == 30:
                    raise subprocess.TimeoutExpired('own child', timeout)
                return -9
        runtime.reap(Process())
        self.assertEqual(calls, ['terminate', ('wait', 30), 'kill', ('wait', 10)])

    def test_wait_file_deadline_and_dead_child_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'missing.json'
            with self.assertRaises(TimeoutError):
                runtime.wait_file(path, SimpleNamespace(poll=lambda: None), time.monotonic() - 1)
            with self.assertRaises(ValueError):
                runtime.wait_file(path, SimpleNamespace(poll=lambda: 1), time.monotonic() + 10)

    def test_stop_requires_ack_and_duplicate_bye_fails(self):
        from semcache.edgelora.r5_d5_lightperf_timing import SessionLifecycle
        task = dict(request_id='R', adapter_id='Adapter_A')
        req = dict(task)
        events = []
        ch = SimpleNamespace(send=lambda h: events.append(h['kind']), close=lambda: events.append('close'),
                             recv=lambda **k: (header('BYE', 'shutdown', -1), b'', {}))
        s = dict(ids=['R'], completed=0, stopped=False, channel=ch)
        lifecycle = SessionLifecycle([task])
        with self.assertRaises(ValueError):
            runtime.stop_after_ack(s, req, {'request': req}, lifecycle)
        lifecycle.begin(task)
        runtime.stop_after_ack(s, req, {'request': req}, lifecycle)
        lifecycle.finish()
        self.assertEqual(events, ['STOP', 'close'])
        with self.assertRaises((ValueError, IndexError)):
            runtime.stop_after_ack(s, req, {'request': req}, lifecycle)

    def test_invalid_bye_cannot_report_success(self):
        from semcache.edgelora.r5_d5_lightperf_timing import SessionLifecycle
        t = dict(request_id='R', adapter_id='Adapter_A')
        life = SessionLifecycle([t])
        life.begin(t)
        ch = SimpleNamespace(send=lambda h: None, recv=lambda **k: (header('BYE', 'shutdown', -1), b'bad', {}))
        s = dict(ids=['R'], completed=0, stopped=False, channel=ch)
        with self.assertRaises(ValueError):
            runtime.stop_after_ack(s, t, {'request': t}, life)
        self.assertFalse(s['stopped'])
        with self.assertRaises(ValueError):
            life.finish()

    def test_wire_byte_and_hash_chains_detect_endpoint_mismatch(self):
        f = physical_fixture.Physical()
        f.setUp()
        sock = listener()
        raw = connect(sock.getsockname()[1])
        peer, _ = sock.accept()
        sock.close()
        es, ud = WireSummary(['R']), WireSummary(['R'])
        a = LightChannel(raw, 'S', f.ident['adapter_matrix_fingerprint'], trace=es)
        b = LightChannel(peer, 'S', f.ident['adapter_matrix_fingerprint'], trace=ud)
        try:
            a.send(header('HELLO', 'R_s000', -1))
            b.recv()
            wire_equal(es.request('R'), ud.request('R'))
            bad = ud.request('R')
            bad['recv']['frame_hash_chain'] = 'bad'
            with self.assertRaises(ValueError):
                wire_equal(es.request('R'), bad)
            self.assertEqual(a.records, [])
        finally:
            a.close()
            b.close()

    def test_native_session_protocol_mismatch_and_timeout_no_hang(self):
        f = physical_fixture.Physical()
        f.setUp()
        for timeout in (False, True):
            sock = listener()
            raw = connect(sock.getsockname()[1])
            peer, _ = sock.accept()
            sock.close()
            client = LightChannel(raw, 'S', f.ident['adapter_matrix_fingerprint'])
            server = LightChannel(peer, 'S', f.ident['adapter_matrix_fingerprint'], timeout=.1)
            server.condition_deadline = time.monotonic() + .15
            errors = []
            def serve():
                try:
                    serve_light_sparse(server, f.ident, f.registry, f.matrices, service=UDServiceEmulator(),
                                       deadline=time.monotonic() + .15, fixture=True, expected_requests=[f.req('R', [1, 2, 3])])
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    server.close()
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                if not timeout:
                    client.send(header('STOP', 'shutdown', -1))
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], (ValueError, TimeoutError))
            finally:
                client.close()
                server.close()

    def test_logit_observer_is_bounded_memory_hook_without_result_blob(self):
        from semcache.edgelora.r5_d5_lightperf_timing import generate_loop
        f = physical_fixture.Physical()
        f.setUp()
        req = f.req('R', [1, 2, 3])
        observed = []
        r = generate_loop(req, lambda *a: ([0., 3., 1.], None), lambda ids: {}, str,
                          on_logits=lambda step, history, values: observed.append((step, history, values)))
        self.assertEqual(observed, [(0, (), (0., 3., 1.))])
        self.assertNotIn('logits_blob', r)
        self.assertEqual(r['generated_token_ids'], [1])

    def test_serial_stream_orchestration_receipts_and_failure_boundary(self):
        """Injected orchestration fixture only; native tests exercise real OPT."""
        from test_r5_d5_lightperf import workload
        from semcache.edgelora.r5_d5_lightperf_receipts import PolicyCounter
        from semcache.edgelora.r5_d5_lightperf_timing import LightProfiler
        from semcache.edgelora.r5_i5_contracts import request
        from semcache.edgelora.r5_multi_artifacts import runtime_view
        from semcache.edgelora.r5_i5_contracts import namespace
        _, tasks = workload(n=5)
        f = physical_fixture.Physical()
        f.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'es').mkdir()
            writer = SummaryWriter(root / 'es')
            sessions = {}
            for aid in {t['adapter_id'] for t in tasks}:
                identity = dict(f.ident, adapter_id=aid)
                out = root / aid
                out.mkdir()
                events = []
                ch = SimpleNamespace(send=lambda h, events=events: events.append(h['kind']),
                     recv=lambda **kw: (header('BYE', 'shutdown', -1), b'', {}), close=lambda: None)
                sessions[aid] = dict(identity=identity, ids=[t['request_id'] for t in tasks if t['adapter_id'] == aid],
                    owner=None, index=None, wire=WireSummary([t['request_id'] for t in tasks if t['adapter_id'] == aid]),
                    policy=PolicyCounter(), channel=ch, process=SimpleNamespace(poll=lambda: None),
                    ud_out=out, completed=0, stopped=False, events=events)

            def generate(model, adapter, tokenizer, req, channel, owner, index, **kw):
                s = sessions[req['adapter_id']]
                s['wire'].rows[req['request_id']] = WireSummary.empty()
                kw['on_tcp_complete'](dict(request=req))
                fresh = 3 * 2 * len(req['token_ids'])
                (s['ud_out'] / f'request_{s["completed"] - 1:03d}.json').write_text(__import__('json').dumps(
                    dict(request_id=req['request_id'], wire=WireSummary.empty(), cpu_projection_rows=fresh, compute={})))
                return dict(completion=dict(physical=None, impact=None), row_summary=dict(fresh_qkv_rows=fresh, skipped_qkv_rows=0),
                    generated_token_ids=[3], generated_token_ids_sha256=digest([3]), generated_text='3', generated_text_sha256='hash',
                    termination='LENGTH_CAP', first_available_ns=time.monotonic_ns(), model_execution_e2e_ms=1., model_ttft_ms=1.,
                    prefill_ms=1., decode_ms=0., admission_completion_ms=0., condition=CONDITIONS[0], application_e2e_ms=1.,
                    matching_candidate_count=None, components=LightProfiler().finish())

            req_fn = lambda t, ids, p, ident: request(t, ids, p, ident, fixture=True)
            rows = runtime.execute_stream(tasks, sessions, None, None, None, CONDITIONS[0], writer, time.monotonic() + 10,
                request_fn=req_fn, generate_fn=generate, score_fn=lambda t, text: ({'metric': 'TEST', 'score': 1.}, 'TEST'))
            self.assertEqual([r['request_id'] for r in rows], [t['request_id'] for t in tasks])
            self.assertTrue(all(s['stopped'] and s['events'] == ['STOP'] for s in sessions.values()))
            self.assertTrue(all(r['application_e2e_ms'] >= r['application_ttft_ms'] for r in rows))
            self.assertEqual(writer.files, 5)
            with self.assertRaises(TimeoutError):
                runtime.execute_stream(tasks, sessions, None, None, None, CONDITIONS[0], writer, time.monotonic() - 1,
                    request_fn=req_fn, generate_fn=generate, score_fn=lambda *a: self.fail('quality after timeout'))


if __name__ == '__main__':
    unittest.main()
