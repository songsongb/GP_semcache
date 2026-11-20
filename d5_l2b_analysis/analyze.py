#!/usr/bin/env python3
"""Standalone, stdlib-only, read-only D5 receipt extraction and MODELED sensitivity.

No model loading, launch, source edits, network access, or approval generation.
Summary mode NEVER substitutes CSV summaries for authenticated wire/compute data.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
FULL = 'FULL_RECOMPUTE'
EXACT = 'SEMCACHE_D5_EXACT'
APPROX = 'SEMCACHE_D5_APPROX'
CONDITIONS = (FULL, EXACT, APPROX)
DATASETS = ('qasper', '2wikimqa', 'multi_news')
MEMBERS = ('comparison_summary.csv', 'cycle_summary.csv', 'observed_profile_components.csv', 'run_index.tsv', 'registry.json')
# Exposures are assumptions about incremental delay, not changed hardware factors.
SCENARIOS = {
    'SERIAL_BLOCKING_REFERENCE': (1., 'SUM'),
    'PARTIAL_EXPOSURE_50PCT_ASSUMED': (.5, 'SUM'),
    'DUPLEX_MAX_STRESS_ASSUMED': (1., 'MAX'),
}
LIMIT = 64 * 2**20
MISSING = 'NOT_MEASURED'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value, label, *, integer=False, positive=False):
    require(not isinstance(value, bool), 'boolean numeric input: ' + label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid number: ' + label) from exc
    require(math.isfinite(result) and (result > 0 if positive else result >= 0), 'invalid range: ' + label)
    if integer:
        require(result.is_integer(), 'noninteger: ' + label)
        return int(result)
    return result


def metric(mapping, name, *, integer=False):
    value = mapping.get(name)
    return None if value is None or value in ('NOT_MEASURED', 'NOT_AVAILABLE') else number(value, name, integer=integer)


def strict_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key: ' + k)
            result[k] = v
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    require(len(data) <= LIMIT, 'JSON size limit')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_bytes(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink input forbidden: ' + str(path))
    require(path.is_file() and path.stat().st_size <= LIMIT, 'missing/oversized input: ' + str(path))
    return path.read_bytes()


def csv_rows(data, required):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')), strict=True)
    fields = reader.fieldnames
    require(fields and len(fields) == len(set(fields)) and set(required) <= set(fields), 'missing/duplicate CSV columns')
    rows = list(reader)
    require(len(rows) <= 100000 and all(None not in r and all(v is not None for v in r.values()) for r in rows), 'ragged/oversized CSV')
    return rows


def registry_rows(data):
    result = strict_json(data)
    require(type(result) is list and 0 < len(result) <= 9, 'explicit bounded registry required')
    require(all(type(r) is dict and set(r) == {'run', 'plan', 'cpu_audit'} and
                all(isinstance(v, str) and v for v in r.values()) for r in result), 'invalid registry triple')
    require(len({r['run'] for r in result}) == len(result), 'duplicate registry run')
    return result


def index_rows(data):
    """Both headerless official TSV and an explicit five-column header are valid."""
    raw = list(csv.reader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t', strict=True))
    if raw and raw[0] == ['dataset', 'condition', 'plan', 'cpu_audit', 'run']:
        raw = raw[1:]
    require(0 < len(raw) <= 9 and all(len(r) == 5 for r in raw), 'invalid run_index TSV')
    rows = [dict(zip(('dataset', 'condition', 'plan', 'cpu_audit', 'run'), r)) for r in raw]
    require(all(r['dataset'] in DATASETS and r['condition'] in CONDITIONS and all(r.values()) for r in rows), 'invalid index identity')
    require(len({(r['dataset'], r['condition']) for r in rows}) == len(rows) and
            len({r['run'] for r in rows}) == len(rows), 'duplicate index run/condition')
    return rows


def bind_index(registry, index):
    require(len(registry) == len(index), 'registry/index count mismatch')
    by_run = {r['run']: r for r in index}
    for item in registry:
        require(item['run'] in by_run and all(by_run[item['run']][k] == v for k, v in item.items()), 'registry/index path mismatch')
    return by_run


def bundle(path):
    data = read_bytes(path)
    documents = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for item in archive:
            require(item.isfile() and item.name in MEMBERS and item.name not in documents and item.size <= LIMIT,
                    'unexpected/duplicate/unsafe archive member')
            require(sum(map(len, documents.values())) + item.size <= LIMIT, 'expanded archive byte budget')
            documents[item.name] = archive.extractfile(item).read()
    require(set(documents) == set(MEMBERS), 'five named attachments required')
    return documents, {'bundle_sha256': sha(data), 'members': {k: sha(v) for k, v in documents.items()}}


def complete_sum(rows, field):
    values = [r.get(field) for r in rows]
    return None if any(v is None for v in values) else sum(values)


def aggregate(rows, scope, group):
    result = {k: rows[0].get(k) for k in ('dataset', 'condition', 'input_integrity', 'gpu_uuid', 'source_commit')}
    result.update(scope=scope, group=group, requests=sum(r['requests'] for r in rows))
    for field in ('e2e_ms', 'ttft_ms', 'es_bytes', 'ud_bytes', 'es_frames', 'ud_frames', 'compute_ms',
                  'service_ms', 'pacing_ms', 'serialization_ms', 'deserialization_ms', 'hash_ms',
                  'fresh_rows', 'skipped_rows', 'es_send_ms', 'es_recv_ms', 'ud_send_ms', 'ud_recv_ms',
                  'framing_serialization_ms', 'framing_deserialization_ms', 'es_payload_bytes', 'ud_payload_bytes',
                  'es_frame_overhead_bytes', 'ud_frame_overhead_bytes', 'input_tokens', 'output_tokens',
                  'token_agreements_vs_full', 'ud_operations', 'ud_framing_serialization_ms', 'ud_framing_deserialization_ms'):
        result[field] = complete_sum(rows, field)
    metrics = {r.get('quality_metric') for r in rows}
    require(len(metrics) == 1, 'unrelated quality metrics cannot be pooled')
    result['quality_metric'] = next(iter(metrics))
    result['quality_score_mean'] = (None if any(r.get('quality_score_mean') is None for r in rows) else
        sum(r['quality_score_mean'] * r['requests'] for r in rows) / result['requests'])
    result['worker_deadline_ms'] = rows[0].get('worker_deadline_ms')
    return result


def with_groups(requests):
    rows = list(requests)
    by_run = {}
    for row in requests:
        by_run.setdefault((row['dataset'], row['condition']), []).append(row)
    for records in by_run.values():
        rows.append(aggregate(records, 'SESSION', 'all'))
        for field, scope in (('cycle', 'CYCLE'), ('adapter_id', 'ADAPTER')):
            for value in sorted({r[field] for r in records}, key=str):
                if value is not None:
                    rows.append(aggregate([r for r in records if r[field] == value], scope, str(value)))
    return rows


def summary_inputs(documents):
    registry = registry_rows(documents['registry.json'])
    index = index_rows(documents['run_index.tsv'])
    bind_index(registry, index)
    require({(r['dataset'], r['condition']) for r in index} == {(d, c) for d in DATASETS for c in CONDITIONS}, 'nine-run matrix incomplete')
    cycles = csv_rows(documents['cycle_summary.csv'], ('dataset', 'condition', 'cycle', 'requests', 'mean_e2e_ms', 'mean_ttft_ms'))
    rows, seen = [], set()
    for row in cycles:
        key = row['dataset'], row['condition'], row['cycle']
        require(key not in seen and key[:2] in {(i['dataset'], i['condition']) for i in index}, 'duplicate/unknown cycle')
        seen.add(key)
        n = number(row['requests'], 'cycle requests', integer=True, positive=True)
        rows.append(dict(dataset=key[0], condition=key[1], scope='CYCLE', group=key[2], requests=n,
                         e2e_ms=number(row['mean_e2e_ms'], 'mean E2E', positive=True) * n,
                         ttft_ms=number(row['mean_ttft_ms'], 'mean TTFT') * n,
                         es_bytes=None, ud_bytes=None, compute_ms=None,
                         output_tokens=None if not row.get('mean_output_tokens') else number(row['mean_output_tokens'], 'output tokens') * n,
                         quality_score_mean=None if not row.get('mean_quality') else number(row['mean_quality'], 'quality'),
                         input_integrity='ATTACHED_SUMMARY_ONLY_NOT_RECEIPT_VERIFIED'))
    require(len(rows) == 18, 'two cycle summaries per run required')
    for item in index:
        subset = [r for r in rows if (r['dataset'], r['condition']) == (item['dataset'], item['condition'])]
        require(len(subset) == 2 and sum(r['requests'] for r in subset) == 20, 'expected two cycles / twenty requests')
        rows.append(aggregate(subset, 'SESSION', 'all'))
    comparisons = csv_rows(documents['comparison_summary.csv'], ('dataset', 'comparison', 'status', 'mismatches', 'full_mean_e2e_ms', 'cache_mean_e2e_ms'))
    seen = set()
    sessions = {(r['dataset'], r['condition']): r for r in rows if r['scope'] == 'SESSION'}
    for row in comparisons:
        require(row['comparison'] in ('FULL_vs_EXACT', 'FULL_vs_APPROX') and row['dataset'] in DATASETS, 'unknown comparison')
        key = row['dataset'], EXACT if row['comparison'] == 'FULL_vs_EXACT' else APPROX
        require(key not in seen and row['status'] == 'DESCRIPTIVE_MATCHED_IDENTITY' and not row['mismatches'], 'unmatched/duplicate summary comparison')
        seen.add(key)
        sessions[key]['token_agreements_vs_full'] = (None if not row.get('token_agreement_count') else
            number(row['token_agreement_count'], 'token agreements', integer=True))
        for condition in (FULL, key[1]):
            sessions[key[0], condition]['quality_metric'] = row.get('quality_metric')
        for cond, field in ((FULL, 'full_mean_e2e_ms'), (key[1], 'cache_mean_e2e_ms')):
            record = sessions[key[0], cond]
            require(abs(record['e2e_ms'] / record['requests'] - number(row[field], field)) <= .003,
                    'cycle/comparison latency mismatch beyond CSV rounding')
    require(len(seen) == 6, 'six comparisons required')
    profiles = csv_rows(documents['observed_profile_components.csv'], ('dataset', 'condition', 'phase', 'component', 'measured_requests', 'exclusive_mean_ms', 'measurement_status'))
    seen = set()
    for row in profiles:
        key = tuple(row[k] for k in ('dataset', 'condition', 'phase', 'component'))
        require(key not in seen and (key[0], key[1]) in sessions, 'duplicate/unknown profile group')
        seen.add(key)
        number(row['exclusive_mean_ms'], 'profile time')
        number(row['measured_requests'], 'profile requests', integer=True)
    return rows, index, dict(registry=registry, comparisons=comparisons, profile_rows=len(profiles),
        summary_latency_origin='WEIGHTED_CYCLE_MEANS_ROUNDED_CSV; comparison means cross-checked within .003ms',
        identity_status='SOURCE_GPU_REQUEST_IDS_NOT_AVAILABLE_IN_ATTACHMENTS',
        missing='No request/UD receipts in bundle; profile socket wait is NOT LoRA compute; no phase wire/compute allocation')


def equal_sum(actual, expected, name):
    require(actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-5), name + ' total mismatch')


def extract_verified(run, plan, read_json):
    """Called only AFTER official verify_run; never an alternative verifier."""
    rows = []
    by_adapter = {}
    for raw in run['requests']:
        by_adapter.setdefault(raw['adapter_id'], []).append(raw)
    for aid, records in by_adapter.items():
        item = run['ud'][aid]
        ud = read_json(Path(item['path']) / 'ud.json')
        control = ud['wire']['CONTROL']
        service_rows = [r['ud_compute'] for r in records]
        for name in ('measured_compute_ms', 'measured_service_ms', 'pacing_ms', 'operations'):
            # Missing metrics remain missing, never socket-time substitutes.
            values = [metric(r, name) for r in service_rows]
            if all(v is not None for v in values) and metric(ud['service'], name) is not None:
                equal_sum(sum(values), metric(ud['service'], name), 'UD ' + name)
        for op, pacer in (('send', item['es_pacing']), ('recv', ud['pacing'])):
            endpoint_op = 'recv' if op == 'send' else 'send'
            total = sum(r['wire'][op]['wire_bytes'] for r in records) + control[endpoint_op]['wire_bytes']
            equal_sum(total, pacer['actual_wire_bytes'], 'wire/pacer')
        for index, raw in enumerate(records):
            wire, comp = raw['wire'], raw['ud_compute']
            peer = ud['wire'][raw['request_id']]
            last = index == len(records) - 1
            es_control = control['recv'] if last else {}
            ud_control = control['send'] if last else {}
            def plus(op, name, extra):
                value = metric(wire[op], name, integer=name.endswith('bytes') or name == 'frames')
                other = metric(extra, name, integer=name.endswith('bytes') or name == 'frames') if extra else 0
                return None if value is None or other is None else value + other
            provenance = raw.get('replay_provenance', {})
            row = dict(dataset=run['dataset'], condition=run['condition'], scope='REQUEST', group=str(raw['session_ordinal']),
                requests=1, request_id=raw['request_id'], sample_id=raw['sample_id'], user_id=raw['user_id'], adapter_id=aid,
                cycle=provenance.get('cycle_index'), original_request_id=provenance.get('original_request_id'),
                cycle_index_zero_based=provenance.get('cycle_index'),
                gpu_uuid=run['gpu']['uuid'], source_commit=run['source']['commit'], input_integrity='LIGHT_RECEIPT_INTEGRITY_PASS',
                input_tokens=number(raw['input_tokens'], 'input tokens', integer=True),
                output_tokens=number(raw['output_tokens'], 'output tokens', integer=True),
                generated_tokens_sha256=raw['generated_token_ids_sha256'],
                quality_metric=raw['quality']['metric'], quality_score_mean=number(raw['quality']['score'], 'quality score'),
                e2e_ms=number(raw['application_e2e_ms'], 'E2E', positive=True), ttft_ms=number(raw['application_ttft_ms'], 'TTFT'),
                es_bytes=plus('send', 'wire_bytes', es_control), ud_bytes=plus('recv', 'wire_bytes', ud_control),
                es_frames=plus('send', 'frames', es_control), ud_frames=plus('recv', 'frames', ud_control),
                es_payload_bytes=plus('send', 'payload_bytes', es_control), ud_payload_bytes=plus('recv', 'payload_bytes', ud_control),
                es_frame_overhead_bytes=plus('send', 'frame_overhead_bytes', es_control), ud_frame_overhead_bytes=plus('recv', 'frame_overhead_bytes', ud_control),
                allocated_control_es_bytes=es_control.get('wire_bytes', 0), allocated_control_ud_bytes=ud_control.get('wire_bytes', 0),
                control_allocation='STOP_BYE_ON_LAST_ADAPTER_REQUEST' if last else 'NONE',
                compute_ms=metric(comp, 'measured_compute_ms'), service_ms=metric(comp, 'measured_service_ms'),
                ud_operations=metric(comp, 'operations', integer=True),
                pacing_ms=metric(comp, 'pacing_ms'), serialization_ms=metric(comp, 'serialization_ms'),
                deserialization_ms=metric(comp, 'deserialization_ms'), hash_ms=metric(comp, 'hash_ms'),
                fresh_rows=number(raw['fresh_qkv_rows'], 'fresh rows', integer=True), skipped_rows=number(raw['skipped_qkv_rows'], 'skipped rows', integer=True),
                es_send_ms=metric(wire['send'], 'socket_call_wall_ms'), es_recv_ms=metric(wire['recv'], 'socket_call_wall_ms'),
                ud_send_ms=metric(peer['send'], 'socket_call_wall_ms'), ud_recv_ms=metric(peer['recv'], 'socket_call_wall_ms'),
                framing_serialization_ms=metric(wire['send'], 'framing_serialization_ms'),
                framing_deserialization_ms=metric(wire['recv'], 'framing_deserialization_ms'),
                ud_framing_serialization_ms=metric(peer['send'], 'framing_serialization_ms'),
                ud_framing_deserialization_ms=metric(peer['recv'], 'framing_deserialization_ms'),
                worker_deadline_ms=number(plan['resources']['worker_deadline_seconds'], 'deadline', positive=True) * 1000)
            if row['compute_ms'] is not None:
                require(row['compute_ms'] <= row['e2e_ms'] + 1, 'LoRA compute exceeds enclosing sequential E2E')
            rows.append(row)
    rows.sort(key=lambda r: int(r['group']))
    return rows


def receipt_inputs(repo, registry_path, index_path, expected_commit=None):
    repo = Path(repo).resolve()
    require(Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=repo, text=True).strip()).resolve() == repo, 'wrong repository root')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip(),
            'clean tracked source required for official offline verifier')
    sys.path.insert(0, str(repo / 'src'))
    from semcache.edgelora.r5_d5_lightperf_analysis import verify_run, compare
    from semcache.edgelora.r5_d5_lightperf_io import read_json
    registry_data, index_data = read_bytes(registry_path), read_bytes(index_path)
    registry, index = registry_rows(registry_data), index_rows(index_data)
    mapping = bind_index(registry, index)
    require({(r['dataset'], r['condition']) for r in index} == {(d, c) for d in DATASETS for c in CONDITIONS}, 'nine-run matrix required')
    rows, runs, bindings, plans = [], {}, [], {}
    for paths in registry:
        run, binding = verify_run(paths['run'], paths['plan'], paths['cpu_audit'])
        entry = mapping[paths['run']]
        require(run['dataset'] == entry['dataset'] and run['condition'] == entry['condition'] and run['scale'] == 20, 'index/run identity mismatch')
        knobs = run['performance_knobs']
        require(knobs['bandwidth_mbps'] is None and knobs['ud_throughput_factor'] == 1. and
                knobs['link_scope'] == 'APPLICATION_LOOPBACK_PER_CHANNEL_PER_DIRECTION' and
                knobs['ud_scope'] == 'EMULATED_UD_SERVICE_TIME' and knobs['rtt_ms'] is None, 'requires unlimited/factor1/no-RTT measured baseline')
        require(expected_commit is None or run['source']['commit'] == expected_commit, 'receipt source commit differs')
        plan = read_json(Path(paths['plan']) / 'plan.json')
        rows.extend(extract_verified(run, plan, read_json))
        runs[run['dataset'], run['condition']] = run
        plans[paths['run']] = plan
        bindings.append(dict(paths=paths, binding=binding, source=run['source'], gpu=run['gpu'], backend=run['backend'],
            precision_policy=run['precision_policy'], cpu_concurrency=run['cpu_concurrency'], tasks_sha256=run['tasks_sha256']))
    for dataset in DATASETS:
        for condition in CONDITIONS[1:]:
            full, cache = runs[dataset, FULL], runs[dataset, condition]
            require(compare(full, cache)['status'] == 'DESCRIPTIVE_MATCHED_IDENTITY', 'NOT_COMPARABLE: device/source/workload/knobs differ')
            require([(r[k]) for r in full['requests'] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')] ==
                    [(r[k]) for r in cache['requests'] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')], 'routing/request identity differs')
    # No current-HEAD requirement for frozen evidence: official offline verifier
    # validates its original source fingerprints/CPU approvals without rewriting them.
    for paths, expected in ((b['paths'], b['binding']) for b in bindings):
        require(verify_run(paths['run'], paths['plan'], paths['cpu_audit'])[1] == expected, 'inputs changed during extraction')
    require(read_bytes(registry_path) == registry_data and read_bytes(index_path) == index_data, 'registry/index changed')
    protected = [registry_path, index_path]
    for paths in registry:
        protected.extend(paths.values())
        plan = plans[paths['run']]
        protected.extend(plan[k] for k in ('original_experiment', 'multi', 'base_prepared', 'baseline_plan', 'strict_pilot') if plan.get(k))
        protected.extend(plan.get('dataset_paths', {}).values())
        protected.extend(item['path'] for item in runs[mapping[paths['run']]['dataset'], mapping[paths['run']]['condition']]['ud'].values())
    full_tokens = {(r['dataset'], r['group']): r['generated_tokens_sha256'] for r in rows if r['condition'] == FULL}
    for row in rows:
        row['token_agreements_vs_full'] = int(row['generated_tokens_sha256'] == full_tokens[row['dataset'], row['group']])
    verifier_path = repo / 'src/semcache/edgelora/r5_d5_lightperf_analysis.py'
    return with_groups(rows), index, dict(bindings=bindings, registry_sha256=sha(registry_data), index_sha256=sha(index_data),
        official_verifier_sha256=sha(verifier_path.read_bytes()),
        identity_status='ALL_NINE_OFFICIALLY_VERIFIED_AND_IDENTITY_MATCHED'), protected


def net_units(row, scenario):
    a, b = row.get('es_bytes'), row.get('ud_bytes')
    if a is None or b is None:
        return None
    return a + b if SCENARIOS[scenario][1] == 'SUM' else max(a, b)


def predict(row, bandwidth, factor, scenario):
    if bandwidth is not None:
        number(bandwidth, 'Mbps', positive=True)
    require(0 < number(factor, 'factor', positive=True) <= 1 and scenario in SCENARIOS, 'invalid throughput/scenario')
    exposure = SCENARIOS[scenario][0]
    wire, cpu = net_units(row, scenario), row.get('compute_ms')
    reasons = []
    if bandwidth is not None and wire is None:
        reasons.append('ACTUAL_WIRE_BYTES_NOT_MEASURED')
    if factor != 1 and cpu is None:
        reasons.append('PURE_LORA_WALL_COST_NOT_MEASURED')
    if reasons:
        return dict(status='NOT_ESTIMABLE', reason=';'.join(reasons), e2e_ms=None, ttft_ms=None, ttft_lower_ms=None, ttft_upper_ms=None)
    increment = exposure * ((0 if bandwidth is None else .008 * wire / bandwidth) + (0 if factor == 1 else cpu * (1 / factor - 1)))
    require(math.isfinite(increment) and math.isfinite(row['e2e_ms'] + increment), 'nonfinite model prediction')
    baseline = bandwidth is None and factor == 1
    return dict(status='MEASURED_BASELINE' if baseline else 'MODELED', reason='', e2e_ms=row['e2e_ms'] + increment,
                incremental_delay_ms=increment, ttft_ms=row.get('ttft_ms') if baseline else None,
                ttft_lower_ms=row.get('ttft_ms'), ttft_upper_ms=None if row.get('ttft_ms') is None else row['ttft_ms'] + increment,
                ttft_status='MEASURED_BASELINE' if baseline else 'MODELED_BOUNDS_ONLY_PHASE_ALLOCATION_NOT_MEASURED')


def pairs(rows):
    by_key = {(r['dataset'], r['condition'], r['scope'], r['group']): r for r in rows}
    require(len(by_key) == len(rows), 'duplicate analytical group')
    for row in rows:
        if row['condition'] != FULL:
            key = row['dataset'], FULL, row['scope'], row['group']
            require(key in by_key, 'missing matched FULL analytical group')
            full = by_key[key]
            require(full['requests'] == row['requests'], 'group request counts differ')
            if row['scope'] == 'REQUEST':
                require(all(full.get(k) == row.get(k) for k in ('request_id', 'sample_id', 'user_id', 'adapter_id', 'cycle')), 'request routing/group mismatch')
            yield full, row


def estimate_pair(full, cache, bandwidth, factor, scenario):
    a, b = predict(full, bandwidth, factor, scenario), predict(cache, bandwidth, factor, scenario)
    ready = a['e2e_ms'] is not None and b['e2e_ms'] is not None
    baseline = bandwidth is None and factor == 1
    delta = None if not ready else b['e2e_ms'] - a['e2e_ms']
    tolerance = None if not ready else max(1e-9, 1e-12 * max(a['e2e_ms'], b['e2e_ms']))
    equal = ready and abs(delta) <= tolerance
    result = dict(dataset=cache['dataset'], condition=cache['condition'], scope=cache['scope'], group=cache['group'],
        requests=cache['requests'], bandwidth_mbps='UNLIMITED' if bandwidth is None else bandwidth, ud_throughput_factor=factor,
        scenario=scenario, assumption_exposed_fraction=SCENARIOS[scenario][0], direction_model=SCENARIOS[scenario][1],
        input_integrity=cache['input_integrity'], evidence_status='MEASURED_BASELINE' if baseline else 'MODELED' if ready else 'NOT_ESTIMABLE',
        missing_reason=';'.join(sorted(set(filter(None, (a['reason'], b['reason']))))),
        full_e2e_ms=a['e2e_ms'], cache_e2e_ms=b['e2e_ms'],
        full_mean_request_e2e_ms=None if a['e2e_ms'] is None else a['e2e_ms'] / full['requests'],
        cache_mean_request_e2e_ms=None if b['e2e_ms'] is None else b['e2e_ms'] / cache['requests'],
        latency_scope='SUM_REQUEST_E2E; means use the group request count',
        delta_e2e_ms=0. if equal else delta, speedup=None if not ready else 1. if equal else a['e2e_ms'] / b['e2e_ms'],
        cache_e2e_benefit=None if not ready else delta < -tolerance, floating_equality_tolerance_ms=tolerance,
        full_ttft_ms=a['ttft_ms'], cache_ttft_ms=b['ttft_ms'],
        full_mean_request_ttft_ms=None if a['ttft_ms'] is None else a['ttft_ms'] / full['requests'],
        cache_mean_request_ttft_ms=None if b['ttft_ms'] is None else b['ttft_ms'] / cache['requests'],
        full_ttft_lower_ms=a['ttft_lower_ms'], full_ttft_upper_ms=a['ttft_upper_ms'],
        cache_ttft_lower_ms=b['ttft_lower_ms'], cache_ttft_upper_ms=b['ttft_upper_ms'],
        ttft_status='MEASURED_BASELINE' if baseline else 'NOT_ESTIMABLE_POINT_ESTIMATE_PHASE_SPLIT_MISSING',
        ttft_aggregation_scope='SUM_REQUEST_TTFT; mean_request columns divide by group request count',
        runtime_knob_supported=bandwidth in (None, 200, 100, 50) and factor in (1, .5, .25),
        correctness_qualified_speedup='NOT_ESTABLISHED', independent_numerical_audit='NOT_EVALUATED',
        fidelity_curve='APPROX_SEPARATE_UNQUALIFIED' if cache['condition'] == APPROX else 'EXACT_UNQUALIFIED')
    result.update(token_agreements_vs_full=cache.get('token_agreements_vs_full'), quality_metric=cache.get('quality_metric'),
                  full_quality_mean=full.get('quality_score_mean'), cache_quality_mean=cache.get('quality_score_mean'),
                  full_output_tokens=full.get('output_tokens'), cache_output_tokens=cache.get('output_tokens'))
    deadline = cache.get('worker_deadline_ms')
    result['execution_feasibility'] = ('MODELED_REQUEST_SUM_EXCEEDS_WORKER_DEADLINE' if ready and deadline and cache['scope'] == 'SESSION' and
                                      max(a['e2e_ms'], b['e2e_ms']) > deadline else 'NOT_ESTABLISHED_INIT_AND_FRAME_TIMEOUTS_UNMODELED')
    return result


def crossing(residual, saving, axis):
    """D = residual-saving*x, x=1/B or 1/f-1. Exact model root, never observed."""
    if saving == 0:
        return 'NO_MODELED_CROSSING_ZERO_SAVING', None, 'ALL' if residual < 0 else 'NONE'
    if axis == 'bandwidth':
        root = None if residual == 0 else saving / residual
        valid = root is not None and root > 0
    else:
        root = None if saving + residual == 0 else saving / (saving + residual)
        valid = root is not None and 0 < root <= 1
    if not valid:
        return 'NO_FINITE_MODELED_CROSSING_IN_DOMAIN', None, 'ALL' if residual < 0 and saving > 0 else 'NONE_OR_BOUNDARY_EQUALITY'
    return 'MODELED_CROSSING_NOT_OBSERVED', root, 'BELOW_ROOT' if saving > 0 else 'ABOVE_ROOT'


def thresholds(full, cache, bandwidth, factor, scenario):
    exposure = SCENARIOS[scenario][0]
    base = cache['e2e_ms'] - full['e2e_ms']
    fn, cn = net_units(full, scenario), net_units(cache, scenario)
    cpu_known = full.get('compute_ms') is not None and cache.get('compute_ms') is not None
    wire_known = fn is not None and cn is not None
    saving_cpu = full['compute_ms'] - cache['compute_ms'] if cpu_known else None
    saving_wire = fn - cn if wire_known else None
    net_saved_ms = 0 if bandwidth is None else None if not wire_known else exposure * .008 * saving_wire / bandwidth
    cpu_saved_ms = 0 if factor == 1 else None if not cpu_known else exposure * saving_cpu * (1 / factor - 1)
    ud_residual = None if net_saved_ms is None else base - net_saved_ms
    bw_residual = None if cpu_saved_ms is None else base - cpu_saved_ms
    ud = ('NOT_ESTIMABLE', None, 'NOT_ESTIMABLE') if not cpu_known or ud_residual is None else crossing(ud_residual, exposure * saving_cpu, 'ud')
    bw = ('NOT_ESTIMABLE', None, 'NOT_ESTIMABLE') if not wire_known or bw_residual is None else crossing(bw_residual, exposure * .008 * saving_wire, 'bandwidth')
    required = None if factor == 1 or ud_residual is None else max(0., ud_residual) / (exposure * (1 / factor - 1))
    row = estimate_pair(full, cache, bandwidth, factor, scenario)
    row.update(baseline_delta_e2e_ms=base, measured_lora_saving_ms=saving_cpu, measured_effective_wire_saving_bytes=saving_wire,
        required_lora_saving_ms_for_strict_speedup=required, threshold_inequality='SAVING_MUST_EXCEED_THRESHOLD; equality is not speedup>1',
        required_mean_lora_saving_ms_per_request=None if required is None else required / cache['requests'],
        measured_mean_lora_saving_ms_per_request=None if saving_cpu is None else saving_cpu / cache['requests'],
        bandwidth_crossing_status=bw[0], bandwidth_break_even_mbps=bw[1], bandwidth_win_region=bw[2],
        ud_crossing_status=ud[0], ud_break_even_factor=ud[1], ud_win_region=ud[2],
        measured_break_even='NO_OBSERVED_BREAK_EVEN_BASELINE_ONLY',
        lora_saving_upper_bound_ms=full['e2e_ms'], upper_bound_status='ASSUMED_EXTREME_ALL_FULL_E2E_IS_LORA_AND_CACHE_LORA_ZERO',
        ud_win_possible_under_extreme_serial_upper_bound=None if required is None else full['e2e_ms'] > required)
    return row


def measurement_exports(rows):
    transport, compute = [], []
    keys = ('dataset', 'condition', 'scope', 'group', 'requests', 'request_id', 'sample_id', 'user_id', 'adapter_id',
            'cycle', 'cycle_index_zero_based', 'original_request_id', 'gpu_uuid', 'source_commit', 'input_integrity',
            'input_tokens', 'output_tokens', 'quality_metric', 'quality_score_mean', 'token_agreements_vs_full')
    full_by = {(r['dataset'], r['scope'], r['group']): r for r in rows if r['condition'] == FULL}
    for row in rows:
        identity = {k: row.get(k) for k in keys}
        a = dict(identity)
        for field in ('es_bytes', 'ud_bytes', 'es_frames', 'ud_frames', 'es_payload_bytes', 'ud_payload_bytes',
                      'es_frame_overhead_bytes', 'ud_frame_overhead_bytes', 'es_send_ms', 'es_recv_ms', 'ud_send_ms', 'ud_recv_ms',
                      'framing_serialization_ms', 'framing_deserialization_ms', 'ud_framing_serialization_ms',
                      'ud_framing_deserialization_ms', 'allocated_control_es_bytes', 'allocated_control_ud_bytes', 'control_allocation'):
            a[field] = row.get(field)
        a.update(prefill_es_bytes=None, prefill_ud_bytes=None, decode_es_bytes=None, decode_ud_bytes=None,
                 phase_status='NOT_MEASURED_AGGREGATED_WIRE_SUMMARY',
                 evidence_status='MEASURED' if row.get('es_bytes') is not None and row.get('ud_bytes') is not None else MISSING,
                 wire_scope='APPLICATION_FRAMING_BYTES_NOT_TCP_IP_PACKET_OVERHEAD; request totals include allocated STOP/BYE',
                 socket_time_scope='REQUEST_TAGGED_FRAMES_ONLY_EXCLUDES_CONTROL; ENDPOINT_INCLUSIVE_WAIT; do not add mirrored endpoint times')
        f = full_by[row['dataset'], row['scope'], row['group']]
        for direction in ('es_bytes', 'ud_bytes'):
            a['saving_' + direction + '_vs_full'] = None if row.get(direction) is None or f.get(direction) is None else f[direction] - row[direction]
        transport.append(a)
        b = dict(identity)
        for field in ('compute_ms', 'service_ms', 'pacing_ms', 'serialization_ms', 'deserialization_ms', 'hash_ms', 'fresh_rows', 'skipped_rows', 'ud_operations'):
            b[field] = row.get(field)
        b.update(prefill_compute_ms=None, decode_compute_ms=None, phase_status='NOT_MEASURED_AGGREGATED_UD_SUMMARY',
                 evidence_status='MEASURED' if row.get('compute_ms') is not None else MISSING,
                 compute_scope='WALL_OF_cpu_deltas_QKV_function_NOT_CPU_CYCLES; serialization/hash/TCP outside; service includes pacing',
                 cpu_lora_saving_ms_vs_full=None if row.get('compute_ms') is None or f.get('compute_ms') is None else f['compute_ms'] - row['compute_ms'])
        compute.append(b)
    return transport, compute


def sensitivity(rows, bandwidths, factors):
    paired = list(pairs(rows))
    estimated_rows = len(SCENARIOS) * (len(paired) * (len(bandwidths) + len(factors) + len(bandwidths) * len(factors)) +
        sum(c['scope'] != 'REQUEST' for f, c in paired) * len(bandwidths) * len(factors))
    require(estimated_rows <= 200000, 'analysis row budget exceeded before allocation')
    tables = {k: [] for k in ('bandwidth_sensitivity.csv', 'ud_sensitivity.csv', 'combined_sensitivity.csv', 'break_even_summary.csv')}
    for full, cache in paired:
        for scenario in SCENARIOS:
            for bandwidth in bandwidths:
                tables['bandwidth_sensitivity.csv'].append(estimate_pair(full, cache, bandwidth, 1., scenario))
            for factor in factors:
                tables['ud_sensitivity.csv'].append(estimate_pair(full, cache, None, factor, scenario))
            for bandwidth in bandwidths:
                for factor in factors:
                    tables['combined_sensitivity.csv'].append(estimate_pair(full, cache, bandwidth, factor, scenario))
                    if cache['scope'] != 'REQUEST':
                        tables['break_even_summary.csv'].append(thresholds(full, cache, bandwidth, factor, scenario))
    require(sum(map(len, tables.values())) <= 200000, 'analysis row budget exceeded')
    return tables


def hypothetical(rows, compute_fractions, saving_fractions, factors):
    require(sum(c['scope'] == 'SESSION' for f, c in pairs(rows)) * len(compute_fractions) * len(saving_fractions) * len(factors) <= 50000,
            'hypothetical row budget exceeded before allocation')
    result = []
    for full, cache in pairs(rows):
        if cache['scope'] != 'SESSION':
            continue
        for cf in compute_fractions:
            for sf in saving_fractions:
                f = dict(full, compute_ms=full['e2e_ms'] * cf)
                c = dict(cache, compute_ms=f['compute_ms'] * (1 - sf))
                for factor in factors:
                    row = estimate_pair(f, c, None, factor, 'SERIAL_BLOCKING_REFERENCE')
                    row.update(evidence_status='ASSUMED_MODELED_NOT_MEASURED', assumed_full_compute_fraction_of_e2e=cf,
                        assumed_compute_saving_fraction=sf, assumed_full_compute_ms=f['compute_ms'], assumed_cache_compute_ms=c['compute_ms'])
                    result.append(row)
    return result


LIMITATIONS = """# Assumptions and limitations

All changed-bandwidth/throughput results are MODELED, never GPU observations.
Official receipt verification is required in receipts mode. Summary mode checks
attachment consistency only; it does not reissue LIGHT integrity or CPU approval.
Missing wire bytes/LoRA wall costs remain NOT_MEASURED and dependent predictions
are NOT_ESTIMABLE. TCP receive wait and generic exclusive profile times are never
used as LoRA compute; planned wire bounds and skip ratios never replace wire data.

T(B,f) = T0 + exposure * [0.008 * effective_bytes / B_Mbps
                         + measured_cpu_deltas_wall_ms * (1/f - 1)].
Unlimited has zero incremental pacing. T0 already includes baseline loopback,
serialization/hash, observer and ES computation; none is subtracted or rescaled.
This models extra pacing, not replacement of measured socket wait. The baseline
must be unlimited/factor1/no separate RTT. No GPU/model/semantic/quality term is
slowed by the UD factor. UDServiceEmulator executes the same LoRA function once,
measures each call, and targets cost/factor with observable sleep. Hardware
equivalence is NOT_ESTABLISHED; CPU wall cost includes function checks/stacking.

SERIAL_BLOCKING_REFERENCE sums both directions: the actual sparse RPC receives
the complete hidden frame before LoRA and waits for deltas before native Linear.
BytePacer reserves <=64KiB chunks before sending with no idle credit; ideal total
reserved time is bytes/rate, but OS oversleep, partial sends/refunds, buffering,
framing work, frame-size scheduling and contention are not reconstructed from
aggregate receipts. PARTIAL_EXPOSURE_50PCT_ASSUMED uses the same assumed exposed
fraction in every condition. DUPLEX_MAX_STRESS_ASSUMED uses max(direction bytes)
as a hypothetical overlap stress case, not the observed request/response schedule.
These scenarios are assumptions, not calibrated center/confidence bounds or a
guarantee that actual E2E lies within the range. RTT is not silently introduced.

Wire bytes mean application header/framing/payload bytes, excluding TCP/IP packet
headers/retransmissions. STOP/BYE CONTROL bytes are included once, assigned to
the last request of their adapter, where existing runtime performs shutdown.
Endpoint socket times are inclusive and mirrored; never sum them as E2E or CPU.
Per-request phase wire/compute totals are not retained by existing summaries.
TTFT at changed knobs has no point estimate: only explicit 0..all-extra-delay
allocation bounds are emitted where E2E can be modeled. These are conditional
model bounds, not phase measurements or rigorous real-system bounds.
Aggregate TTFT columns are sums of request TTFT; explicit mean_request columns
divide by the group request count. Repeated baseline rows across scenarios are
not independent trials. Summary mode uses weighted, rounded cycle means and
cross-checks comparison means; sub-millisecond CSV rounding differences remain.

Session E2E is sum(request E2E), matching the official comparison; it is not
initialization/sealing-inclusive run wall or request sequence span. Cycles use
authenticated cycle_index, not an assumed split at request10. Adapter groupings
retain routing. Predictions freeze tokens/cache decisions/quality and cannot
model deadline-induced aborts, changed contention, thermals or CPU clock effects.
Aggregate worker bounds are flagged; initialization and per-frame timeout
feasibility remain NOT_ESTABLISHED. No run is automatically launched.

Analytic roots are MODELED crossings, never observed break-even. Equality is not
Speedup>1. Nonpositive savings can yield no crossing or the opposite win region.
Below50Mbps / factors other than1,.5,.25 are outside existing runtime knobs even
if requested as model search points. The extreme UD bound assumes all FULL E2E
is LoRA and cache LoRA is zero; it is ASSUMED, not a measurement. Optional
hypothetical compute-fraction grids are separate ASSUMED_MODELED tables and are
never merged into measured extractions. Input summary rounding is retained.

APPROX remains a separate fidelity-unqualified curve; unrelated quality metrics
are not pooled. Receipt integrity and token agreement do not establish independent
full-vocabulary numerical fidelity. correctness_qualified_speedup=NOT_ESTABLISHED.
"""


def output_tables(output, tables, metadata, protected, references=None):
    path = Path(output).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink output forbidden')
    path = path.resolve()
    for source in protected:
        p = Path(source).resolve()
        require(not path.is_relative_to(p) and not p.is_relative_to(path), 'output overlaps protected input: ' + str(p))
    require(not path.exists(), 'fresh output required; existing output is never reused')
    require(path.parent.is_dir(), 'create an approved output parent first')
    path.mkdir(exist_ok=False)
    # Exclusive creation only. Inventory is completion marker; failed output is
    # preserved, not retried, removed, overwritten or presented as complete.
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ['evidence_status', 'missing_reason']
        with (path / name).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for name, data in (references or {}).items():
        require(name in ('registry.json', 'run_index.tsv'), 'unexpected reference filename')
        with (path / name).open('xb') as handle:
            handle.write(data)
    with (path / 'assumptions_and_limitations.md').open('x') as handle:
        handle.write(LIMITATIONS)
    with (path / 'analysis_metadata.json').open('x') as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    inventory = {p.name: sha(p.read_bytes()) for p in path.iterdir()}
    require(sum(p.stat().st_size for p in path.iterdir()) <= LIMIT, 'analysis output byte budget exceeded')
    with (path / 'analysis_inventory.json').open('x') as handle:
        json.dump(dict(schema='D5_L2B_OFFLINE_MODEL_V1_NOT_LIGHT_RECEIPT', files=inventory,
                       status='ANALYSIS_COMPLETE_GPU_NOT_RUN'), handle, indent=2)


def grid(value, *, bandwidth=False):
    result = []
    for item in value.split(','):
        v = None if bandwidth and item.lower() == 'unlimited' else number(item, 'grid value', positive=True)
        require(bandwidth or v <= 1, 'throughput/fraction must be <=1')
        require(v not in result, 'duplicate grid point')
        result.append(v)
    require(0 < len(result) <= 32, 'grid size bound')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('summary', 'receipts'))
    parser.add_argument('--bundle')
    parser.add_argument('--repo-root')
    parser.add_argument('--registry')
    parser.add_argument('--run-index')
    parser.add_argument('--expected-source-commit')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bandwidths', default='unlimited,200,100,50')
    parser.add_argument('--factors', default='1,.5,.25')
    parser.add_argument('--assumed-compute-fractions', default='')
    parser.add_argument('--assumed-saving-fractions', default='')
    args = parser.parse_args(argv)
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only command requires CUDA_VISIBLE_DEVICES empty')
        bandwidths, factors = grid(args.bandwidths, bandwidth=True), grid(args.factors)
        metadata = dict(schema='D5_L2B_OFFLINE_MODEL_V1_NOT_LIGHT_RECEIPT', analysis_mode=args.mode,
                        analyzer_sha256=sha(Path(__file__).read_bytes()), GPU_status='GPU_NOT_RUN',
                        correctness_qualified_speedup='NOT_ESTABLISHED', scenarios=SCENARIOS)
        if args.mode == 'summary':
            require(args.bundle and not any((args.registry, args.run_index, args.repo_root, args.expected_source_commit)), 'summary requires only a bundle')
            documents, identity = bundle(args.bundle)
            rows, index, info = summary_inputs(documents)
            metadata.update(identity, **info)
            protected = [args.bundle]
            references = {k: documents[k] for k in ('registry.json', 'run_index.tsv')}
        else:
            require(args.repo_root and args.registry and args.run_index and not args.bundle, 'receipts requires repo-root, registry and run-index')
            output = Path(args.output_dir).resolve()
            require(output.is_relative_to(Path('/data/khuss/tmp')), 'SERAPH receipt-analysis output must be a fresh path under /data/khuss/tmp')
            rows, index, info, protected = receipt_inputs(args.repo_root, args.registry, args.run_index, args.expected_source_commit)
            metadata.update(info)
            metadata['analyzer_repo_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo_root, text=True).strip()
            references = None
        transport, compute = measurement_exports(rows)
        tables = sensitivity(rows, bandwidths, factors)
        tables.update({'measured_transport.csv': transport, 'measured_ud_compute.csv': compute, 'input_mapping.csv': index})
        candidates = [dict(row) for row in tables['combined_sensitivity.csv'] if row['scope'] == 'SESSION' and
                      row['scenario'] == 'SERIAL_BLOCKING_REFERENCE' and row['runtime_knob_supported'] and
                      row['evidence_status'] == 'MODELED' and row['cache_e2e_benefit']]
        candidates.sort(key=lambda r: (r['condition'] != EXACT, (r['bandwidth_mbps'] != 'UNLIMITED') + (r['ud_throughput_factor'] != 1),
                                      -r['ud_throughput_factor'], -(float('inf') if r['bandwidth_mbps'] == 'UNLIMITED' else r['bandwidth_mbps']),
                                      -r['speedup'], r['dataset']))
        for index, row in enumerate(candidates):
            row.update(review_priority=index + 1, approval='MANUAL_GPU_VALIDATION_REQUIRED_NO_AUTOMATIC_JOB',
                       robustness='REFERENCE_SCENARIO_ONLY; compare other exposure assumptions before GPU selection')
        tables['gpu_validation_candidates.csv'] = candidates
        if args.assumed_compute_fractions or args.assumed_saving_fractions:
            require(args.assumed_compute_fractions and args.assumed_saving_fractions, 'both explicit hypothetical fraction grids required')
            tables['hypothetical_ud_assumptions.csv'] = hypothetical(rows, grid(args.assumed_compute_fractions), grid(args.assumed_saving_fractions), factors)
        metadata['table_rows'] = {k: len(v) for k, v in tables.items()}
        metadata['candidate_status'] = 'MODELED_CANDIDATES_REQUIRE_REVIEW' if candidates else 'NO_SUPPORTED_MODELED_CANDIDATES_OR_MISSING_MEASUREMENTS'
        metadata['missing_measurements'] = 'Existing summaries do not retain PREFILL/DECODE wire or pure LoRA split. Point TTFT predictions unavailable.'
        output_tables(args.output_dir, tables, metadata, protected, references)
        print(json.dumps(dict(status='ANALYSIS_COMPLETE_GPU_NOT_RUN', mode=args.mode, table_rows=metadata['table_rows'],
                              output_dir=str(Path(args.output_dir).resolve()), correctness_qualified_speedup='NOT_ESTABLISHED')))
        return 0
    except (ValueError, OSError, KeyError, TypeError, csv.Error, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print(json.dumps(dict(status='BLOCKED', error_type=type(exc).__name__, reason=str(exc), GPU_status='GPU_NOT_RUN')), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
