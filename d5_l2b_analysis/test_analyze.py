"""Small SYNTHETIC fixtures only. No GPU, downloads, network or real approvals."""
import csv
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import analyze as a


def sample(condition=a.FULL, **changes):
    row = dict(dataset='qasper', condition=condition, scope='SESSION', group='all', requests=20,
               e2e_ms=1000. if condition == a.FULL else 1200., ttft_ms=100., es_bytes=1000000,
               ud_bytes=7000000, compute_ms=400., input_integrity='SYNTHETIC_TEST_ONLY')
    row.update(changes)
    return row


class FormulaTests(unittest.TestCase):
    def test_baseline_does_not_require_wire_or_compute(self):
        value = a.predict(sample(es_bytes=None, ud_bytes=None, compute_ms=None), None, 1., 'SERIAL_BLOCKING_REFERENCE')
        self.assertEqual(value['e2e_ms'], 1000.)
        self.assertEqual(value['ttft_ms'], 100.)

    def test_mbps_units_and_both_directions(self):
        value = a.predict(sample(), 100, 1., 'SERIAL_BLOCKING_REFERENCE')
        self.assertEqual(value['e2e_ms'], 1640.)
        self.assertEqual(value['ttft_lower_ms'], 100.)
        self.assertEqual(value['ttft_upper_ms'], 740.)
        self.assertIsNone(value['ttft_ms'])
        duplex = a.predict(sample(), 100, 1., 'DUPLEX_MAX_STRESS_ASSUMED')
        self.assertEqual(duplex['e2e_ms'], 1560.)

    def test_ud_uses_compute_not_service_or_wait(self):
        row = sample(service_ms=900., es_recv_ms=999999.)
        self.assertEqual(a.predict(row, None, .5, 'SERIAL_BLOCKING_REFERENCE')['e2e_ms'], 1400.)
        self.assertEqual(a.predict(row, None, .25, 'SERIAL_BLOCKING_REFERENCE')['e2e_ms'], 2200.)

    def test_missing_is_not_zero_or_a_socket_fallback(self):
        row = sample(es_bytes=None, compute_ms=None, es_recv_ms=999.)
        self.assertEqual(a.predict(row, 50, 1., 'SERIAL_BLOCKING_REFERENCE')['status'], 'NOT_ESTIMABLE')
        self.assertEqual(a.predict(row, None, .5, 'SERIAL_BLOCKING_REFERENCE')['status'], 'NOT_ESTIMABLE')
        self.assertIsNone(a.complete_sum([{'x': 1}, {}], 'x'))
        self.assertEqual(a.predict(sample(es_bytes=0, ud_bytes=0, compute_ms=0), 50, .25, 'SERIAL_BLOCKING_REFERENCE')['e2e_ms'], 1000.)

    def test_ud_root_and_strict_speedup(self):
        f, c = sample(), sample(a.EXACT, compute_ms=100.)
        t = a.thresholds(f, c, None, .5, 'SERIAL_BLOCKING_REFERENCE')
        self.assertEqual(t['ud_break_even_factor'], .6)
        self.assertEqual(t['required_lora_saving_ms_for_strict_speedup'], 200.)
        root = a.estimate_pair(f, c, None, .6, 'SERIAL_BLOCKING_REFERENCE')
        self.assertAlmostEqual(root['speedup'], 1.)
        self.assertFalse(root['cache_e2e_benefit'])
        self.assertGreater(a.estimate_pair(f, c, None, .5, 'SERIAL_BLOCKING_REFERENCE')['speedup'], 1.)

    def test_bandwidth_root_and_combination(self):
        f, c = sample(compute_ms=400.), sample(a.EXACT, es_bytes=0, ud_bytes=4000000, compute_ms=100.)
        t = a.thresholds(f, c, None, 1., 'SERIAL_BLOCKING_REFERENCE')
        self.assertEqual(t['bandwidth_break_even_mbps'], 160.)
        self.assertAlmostEqual(a.estimate_pair(f, c, 160., 1., 'SERIAL_BLOCKING_REFERENCE')['speedup'], 1.)
        self.assertGreater(a.estimate_pair(f, c, 100., .5, 'SERIAL_BLOCKING_REFERENCE')['speedup'], 1.)

    def test_no_crossing_zero_negative_savings_and_reverse_region(self):
        self.assertIn('ZERO_SAVING', a.crossing(200, 0, 'ud')[0])
        self.assertIsNone(a.crossing(200, -10, 'ud')[1])
        self.assertEqual(a.crossing(-200, -300, 'ud')[1:], (.6, 'ABOVE_ROOT'))
        self.assertEqual(a.crossing(-200, -32000, 'bandwidth')[1:], (160., 'ABOVE_ROOT'))

    def test_roots_missing_still_emit_required_saving_not_fake_crossing(self):
        f, c = sample(compute_ms=None, es_bytes=None), sample(a.EXACT, compute_ms=None)
        t = a.thresholds(f, c, None, .25, 'SERIAL_BLOCKING_REFERENCE')
        self.assertAlmostEqual(t['required_lora_saving_ms_for_strict_speedup'], 200 / 3)
        self.assertIsNone(t['ud_break_even_factor'])
        self.assertEqual(t['ud_crossing_status'], 'NOT_ESTIMABLE')

    def test_invalid_knobs_and_numbers(self):
        for value in ('NaN', 'inf', '-1', '', None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.number(value, 'bad')
        for rate, factor in ((0, 1), (-1, 1), (50, 0), (50, 1.1)):
            with self.subTest(rate=rate, factor=factor), self.assertRaises(ValueError):
                a.predict(sample(), rate, factor, 'SERIAL_BLOCKING_REFERENCE')
        with self.assertRaises(ValueError):
            a.grid('1,1')

    def test_hypothetical_values_remain_separate_and_labeled(self):
        values = a.hypothetical([sample(compute_ms=None), sample(a.APPROX, compute_ms=None)], [.5], [.25], [.5])
        self.assertEqual(values[0]['assumed_full_compute_ms'], 500.)
        self.assertEqual(values[0]['assumed_cache_compute_ms'], 375.)
        self.assertEqual(values[0]['evidence_status'], 'ASSUMED_MODELED_NOT_MEASURED')
        self.assertEqual(values[0]['correctness_qualified_speedup'], 'NOT_ESTABLISHED')

    def test_pairs_reject_routing_and_group_mutation(self):
        f = sample(scope='REQUEST', request_id='R0', adapter_id='A', cycle=0)
        c = sample(a.EXACT, scope='REQUEST', request_id='R0', adapter_id='B', cycle=0)
        with self.assertRaises(ValueError):
            list(a.pairs([f, c]))

    def test_partial_exposure_is_assumed_equal_across_conditions(self):
        full, cache = sample(), sample(a.EXACT, compute_ms=100.)
        reference = a.estimate_pair(full, cache, None, .5, 'SERIAL_BLOCKING_REFERENCE')
        partial = a.estimate_pair(full, cache, None, .5, 'PARTIAL_EXPOSURE_50PCT_ASSUMED')
        self.assertEqual(reference['full_e2e_ms'] - 1000, 400.)
        self.assertEqual(partial['full_e2e_ms'] - 1000, 200.)
        self.assertEqual(partial['cache_e2e_ms'] - 1200, 50.)

    def test_overflow_is_rejected_and_no_epsilon_speedup_claim(self):
        with self.assertRaises(ValueError):
            a.predict(sample(), None, 1e-309, 'SERIAL_BLOCKING_REFERENCE')
        value = a.estimate_pair(sample(), sample(a.EXACT, e2e_ms=1000.-1e-12), None, 1., 'SERIAL_BLOCKING_REFERENCE')
        self.assertEqual(value['speedup'], 1.)
        self.assertFalse(value['cache_e2e_benefit'])

    def test_row_budget_rejects_before_a_large_sweep(self):
        rows = []
        for i in range(100):
            rows.extend([sample(group=str(i)), sample(a.EXACT, group=str(i))])
        with self.assertRaises(ValueError):
            a.sensitivity(rows, list(range(1, 33)), [.5] * 32)


def fixture_documents():
    index, registry, cycles, comparison = [], [], [], []
    for dataset in a.DATASETS:
        for condition in a.CONDITIONS:
            # Explicit invented TEST paths, not inferred or real SERAPH assets.
            row = dict(dataset=dataset, condition=condition, plan=f'/TEST/{dataset}/plan', cpu_audit=f'/TEST/{dataset}/audit', run=f'/TEST/{dataset}/{condition}')
            index.append(row)
            registry.append({k: row[k] for k in ('run', 'plan', 'cpu_audit')})
            for cycle in (1, 2):
                cycles.append(dict(dataset=dataset, condition=condition, cycle=f'cycle{cycle}', requests=10,
                                   mean_e2e_ms=100 if condition == a.FULL else 120, mean_ttft_ms=50))
        for name in ('EXACT', 'APPROX'):
            comparison.append(dict(dataset=dataset, comparison='FULL_vs_' + name, status='DESCRIPTIVE_MATCHED_IDENTITY',
                                   mismatches='', full_mean_e2e_ms=100, cache_mean_e2e_ms=120))
    def csv_data(rows):
        output = io.StringIO()
        w = csv.DictWriter(output, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        return output.getvalue().encode()
    tsv = '\n'.join('\t'.join(str(r[k]) for k in ('dataset', 'condition', 'plan', 'cpu_audit', 'run')) for r in index).encode()
    return dict(zip(a.MEMBERS, [csv_data(comparison), csv_data(cycles),
        csv_data([dict(dataset='qasper', condition=a.FULL, phase='PREFILL', component='tcp_receive_wait_inclusive',
                       measured_requests=20, exclusive_mean_ms=99999, measurement_status='MEASURED')]), tsv,
        json.dumps(registry).encode()]))


class InputOutputTests(unittest.TestCase):
    def test_headerless_tsv_keeps_first_run_and_explicit_header(self):
        d = fixture_documents()
        rows = a.index_rows(d['run_index.tsv'])
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]['condition'], a.FULL)
        header = b'dataset\tcondition\tplan\tcpu_audit\trun\n'
        self.assertEqual(a.index_rows(header + d['run_index.tsv']), rows)

    def test_summary_cannot_turn_wait_into_compute_or_bytes(self):
        rows, index, info = a.summary_inputs(fixture_documents())
        self.assertEqual(len(rows), 27)
        transport, compute = a.measurement_exports(rows)
        self.assertTrue(all(r['compute_ms'] is None and r['evidence_status'] == a.MISSING for r in compute))
        self.assertTrue(all(r['es_bytes'] is None for r in transport))
        self.assertEqual(info['identity_status'], 'SOURCE_GPU_REQUEST_IDS_NOT_AVAILABLE_IN_ATTACHMENTS')

    def test_csv_ragged_duplicate_headers_nan_and_mismatch(self):
        for data in (b'a,a\n1,2\n', b'a,b\n1,2,3\n', b'a,b\n1\n'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                a.csv_rows(data, ['a'])
        d = fixture_documents()
        d['comparison_summary.csv'] = d['comparison_summary.csv'].replace(b',100,120', b',999,120', 1)
        with self.assertRaises(ValueError):
            a.summary_inputs(d)
        with self.assertRaises(ValueError):
            a.strict_json(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            a.strict_json(b'{"x":NaN}')

    def test_duplicate_runs_and_path_mismatch(self):
        d = fixture_documents()
        registry = a.registry_rows(d['registry.json'])
        registry[0]['plan'] = '/TEST/wrong'
        with self.assertRaises(ValueError):
            a.bind_index(registry, a.index_rows(d['run_index.tsv']))
        with self.assertRaises(ValueError):
            a.registry_rows(json.dumps([registry[0], registry[0]]).encode())

    def test_archive_path_traversal_or_links_rejected_without_extracting(self):
        for name, kind in (('../bad', tarfile.REGTYPE), ('registry.json', tarfile.SYMTYPE)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bad.tar.gz'
                with tarfile.open(path, 'w:gz') as out:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.size = 1 if kind == tarfile.REGTYPE else 0
                    out.addfile(info, io.BytesIO(b'x'))
                with self.assertRaises(ValueError):
                    a.bundle(path)

    def test_fresh_output_and_protected_paths_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'sealed'
            source.mkdir()
            with self.assertRaises(ValueError):
                a.output_tables(source / 'child', {}, {}, [source])
            out = root / 'out'
            a.output_tables(out, {'data.csv': [dict(value=None, evidence_status=a.MISSING)]}, {}, [source])
            inv = json.loads((out / 'analysis_inventory.json').read_text())
            self.assertTrue(all(a.sha((out / k).read_bytes()) == v for k, v in inv['files'].items()))
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with self.assertRaises(ValueError):
                a.output_tables(out, {}, {}, [source])
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})
            link = root / 'link'
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                a.output_tables(link / 'other', {}, {}, [])

    def test_cli_summary_and_no_cuda_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'fixture.tar.gz'
            with tarfile.open(archive, 'w:gz') as out:
                for name, data in fixture_documents().items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    out.addfile(info, io.BytesIO(data))
            command = [sys.executable, str(Path(a.__file__)), 'summary', '--bundle', str(archive), '--output-dir', str(root / 'out')]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            p = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)['status'], 'ANALYSIS_COMPLETE_GPU_NOT_RUN')
            env['CUDA_VISIBLE_DEVICES'] = '0'
            p = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(p.returncode, 2)


def official_fixture(root, dataset, condition):
    """Synthetic sealed data accepted by official OFFLINE verifier, not execution proof."""
    from semcache.edgelora.communication import digest
    from semcache.edgelora.r5_d5_lightperf_io import SummaryWriter, SCHEMA, MODE, ENGINE, BRANCH, BASE
    from semcache.edgelora.r5_d5_lightperf import TEST_MODULES
    from semcache.edgelora.r5_d5_lightperf_workload import knobs
    from semcache.edgelora.r5_d5_lightperf_receipts import WireSummary
    source = dict(branch=BRANCH, strict_baseline=BASE, files={}, content_fingerprint=digest({}),
                  commit='SYNTHETIC_TEST_ONLY', approval_scope='CLEAN_REVIEWED_SOURCE')
    tasks = [dict(request_id=f'R{i}', sample_id=f'S{i}', user_id='User_01', adapter_id='Adapter_A',
                  token_ids=[1, 2, 3, 4], policy={'max_new_tokens': 1}, dataset=dataset) for i in range(20)]
    paths = {k: root / k for k in ('run', 'plan', 'cpu_audit')}
    for p in paths.values():
        p.mkdir(parents=True)
    plan = dict(schema=SCHEMA, mode=MODE, engine=ENGINE, source=source, dataset=dataset, count=20,
                conditions=list(a.CONDITIONS), performance_knobs=knobs(), resources={'worker_deadline_seconds': 7200},
                stream=dict(tasks=tasks, tasks_sha256=digest(tasks)))
    plan['plan_id'] = digest(plan)
    writer = SummaryWriter(paths['plan']); writer.write('plan.json', plan); pi = writer.seal()
    audit = dict(status='PASS', failed=0, errors=0, skipped=0, engine=ENGINE, modules=list(TEST_MODULES), versions={},
                 native_parity='TINY_OPT_STRICT_LIGHT_CPU_PASS', scope='CLEAN_NATIVE_CPU_APPROVAL', source=source, plan_inventory=pi)
    writer = SummaryWriter(paths['cpu_audit']); writer.write('cpu_validation.json', audit); ai = writer.seal()
    cache = condition != a.FULL
    rows = []
    for i, task in enumerate(tasks):
        wire = WireSummary.empty()
        for op, value in (('send', 500 if cache else 1000), ('recv', 3500 if cache else 7000)):
            wire[op].update(frames=1, wire_bytes=value, payload_bytes=value-20, frame_overhead_bytes=20, socket_call_wall_ms=2.)
        comp = dict(measured_compute_ms=.5 if cache else 1., measured_service_ms=.5 if cache else 1., pacing_ms=0.,
                    operations=1, serialization_ms=.1, deserialization_ms=.1, hash_ms=.1)
        arrival = i * 30_000_000
        end = arrival + (20_000_000 if cache else 10_000_000)
        row = dict(**{k: task[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')},
            session_ordinal=i, condition=condition, input_tokens=4, output_tokens=1, generated_token_ids=[3], generated_text='fixture',
            generated_token_ids_sha256=digest([3]), generated_text_sha256=a.sha(b'fixture'), arrival_ns=arrival,
            first_available_ns=arrival+1_000_000, completion_ns=end, application_e2e_ms=20. if cache else 10., application_ttft_ms=1.,
            skipped_qkv_rows=9 if cache else 0, fresh_qkv_rows=375 if cache else 384, physical_hit=cache,
            categories=dict(A=9 if cache else 0, B=0, C=0, D=0), cache_occupancy_bytes=0,
            scorer_version='SYNTHETIC_TEST_ONLY', quality=dict(metric='fixture', score=1.), wire=wire, ud_compute=comp)
        rows.append(row)
    control = WireSummary.empty()
    for v in control.values():
        v.update(frames=1, wire_bytes=10, frame_overhead_bytes=10)
    peer_wire = {r['request_id']: dict(send=deepcopy(r['wire']['recv']), recv=deepcopy(r['wire']['send'])) for r in rows}
    peer_wire['CONTROL'] = control
    service = {k: sum(r['ud_compute'][k] for r in rows) for k in ('measured_compute_ms', 'measured_service_ms', 'pacing_ms', 'operations')}
    udpath = Path(str(paths['run']) + '__ud_Adapter_A'); udpath.mkdir()
    writer = SummaryWriter(udpath)
    for i, row in enumerate(rows):
        writer.write(f'request_{i:03d}.json', dict(request_id=row['request_id'], cpu_projection_rows=row['fresh_qkv_rows'], compute=row['ud_compute']))
    pace = dict(actual_wire_bytes=sum(r['wire']['recv']['wire_bytes'] for r in rows)+10)
    ud = dict(status='COMPLETE', source=source, plan_inventory=pi, cpu_inventory=ai,
              completed_request_ids=[r['request_id'] for r in rows], final_bye_sent=True, user_device_cuda_used=False,
              full_model_loaded=False, wire=peer_wire, service=service, pacing=pace)
    writer.write('ud.json', ud); ui = writer.seal()
    run = dict(schema=SCHEMA, mode=MODE, engine=ENGINE, versions={}, source=source, plan_id=plan['plan_id'], plan_inventory=pi,
        cpu_inventory=ai, status='COMPLETE', condition=condition, performance_knobs=knobs(), dataset=dataset, scale=20,
        tasks_sha256=digest(tasks), measurement_scope='NATIVE_LIGHT', gpu=dict(uuid='GPU_SYNTHETIC_ONLY', name='TEST', device='cuda:0', total_bytes=24*2**30),
        backend={'version': 'TEST'}, precision_policy='TEST', cpu_concurrency=4, warmup_requests=0,
        independent_full_vocabulary_audit='NOT_EVALUATED', correctness_qualified_speedup='NOT_ESTABLISHED',
        resources=dict(gpu_free_after_load_bytes=12*2**30, peak_gpu_allocated_bytes=2**30, host_available_bytes=32*2**30, cpu_count=4),
        errors=[], requests=rows, sequence_span_ms=600., ud={'Adapter_A': dict(path=str(udpath), inventory=ui,
              es_pacing=dict(actual_wire_bytes=sum(r['wire']['send']['wire_bytes'] for r in rows)+10))})
    writer = SummaryWriter(paths['run'])
    for i, row in enumerate(rows):
        name = f'request_{i:03d}.json'; writer.write(name, dict(row, receipt_scope='PRIMARY_BODY_FINAL_TIMING_IN_RUN_JSON'))
        row.update(primary_receipt=name, primary_receipt_sha256=writer.hashes[name])
    writer.write('run.json', run); writer.seal()
    return {k: str(p) for k, p in paths.items()}


class OfficialExtractionTests(unittest.TestCase):
    def test_nine_fixture_runs_official_verification_and_endpoint_totals(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); registry = []; index = []
            for dataset in a.DATASETS:
                for condition in a.CONDITIONS:
                    p = official_fixture(root / dataset / condition, dataset, condition)
                    registry.append(p)
                    index.append('\t'.join((dataset, condition, p['plan'], p['cpu_audit'], p['run'])))
            rp, ip = root / 'registry.json', root / 'run_index.tsv'
            rp.write_text(json.dumps(registry)); ip.write_text('\n'.join(index))
            before = {str(p): a.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
            rows, _, info, protected = a.receipt_inputs(repo, rp, ip, 'SYNTHETIC_TEST_ONLY')
            self.assertEqual(info['identity_status'], 'ALL_NINE_OFFICIALLY_VERIFIED_AND_IDENTITY_MATCHED')
            self.assertEqual(sum(r['scope'] == 'REQUEST' for r in rows), 180)
            full = next(r for r in rows if r['scope'] == 'SESSION' and r['dataset'] == 'qasper' and r['condition'] == a.FULL)
            self.assertEqual(full['es_bytes'], 20010)
            self.assertEqual(full['ud_bytes'], 140010)
            self.assertEqual(full['compute_ms'], 20.)
            self.assertEqual(before, {str(p): a.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()})
            # Tampering never falls back to trusting attached summaries.
            primary = Path(registry[0]['run']) / 'request_000.json'
            primary.write_text('{}')
            with self.assertRaises(ValueError):
                a.receipt_inputs(repo, rp, ip)


if __name__ == '__main__':
    unittest.main()
