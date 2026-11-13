"""Receipt-bound descriptive statistics; no independent numerical PASS claims."""
import math
import statistics

from .communication import digest, require
from .r5_d5_lightperf_io import MODE, SCHEMA, ENGINE, BASE, BRANCH, read_json, verify, file_hash
from .r5_d5_lightperf_workload import CONDITIONS


def describe(values):
    require(values and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), 'invalid latency sample')
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[math.ceil(.95 * len(values)) - 1], p95_method='nearest rank',
                inference='EXPLORATORY_SINGLE_TRIAL' if len(values) <= 5 else 'DESCRIPTIVE_SINGLE_TRIAL')


def categories(physical, tasks, target_id):
    """Exactly the historical A/B/C/D w=3 definition, counted per layer/role."""
    by_id = {t['request_id']: t for t in tasks}
    target = by_id[target_id]['token_ids']
    counts = {c: 0 for c in 'ABCD'}
    sources = set()
    if physical is None:
        return counts, []
    for row in physical['records']:
        for item in row['consumed']:
            source = by_id[item['source_request_id']]['token_ids']
            lo, hi = item['source_positions']
            start, end = item['target_positions']
            require(hi - lo == end - start == 3 and source[lo:hi] == target[start:end], 'invalid consumed w=3')
            prefix = 0
            for a, b in zip(source, target):
                if a != b:
                    break
                prefix += 1
            category = ('A' if end <= prefix else 'B') if [lo, hi] == [start, end] else ('D' if end <= prefix else 'C')
            counts[category] += 3
            sources.add(item['source_request_id'])
    require(sum(counts.values()) == physical['physical_projection_rows_reused'], 'row/category accounting differs')
    return counts, sorted(sources)


def verify_run(root, plan_root, cpu_root):
    """Hash integrity and plan/audit binding, not a signed attestation of execution."""
    from pathlib import Path
    run_inventory, plan_inventory, cpu_inventory = verify(root), verify(plan_root), verify(cpu_root)
    p = read_json(Path(plan_root) / 'plan.json')
    a = read_json(Path(cpu_root) / 'cpu_validation.json')
    r = read_json(Path(root) / 'run.json')
    require(not (Path(root) / 'failure.json').exists(), 'failed finalization is not a successful run')
    body = dict(p)
    pid = body.pop('plan_id')
    require(p['schema'] == r['schema'] == SCHEMA and p['mode'] == r['mode'] == MODE and digest(body) == pid,
            'unknown/changed LIGHT plan/schema')
    require(p['engine'] == r['engine'] == a['engine'] == ENGINE, 'obsolete Light receipt engine')
    from .r5_d5_lightperf import TEST_MODULES
    require(a['modules'] == list(TEST_MODULES) and a['native_parity'] == 'TINY_OPT_STRICT_LIGHT_CPU_PASS' and
            a['errors'] == 0 and a['versions'] == r['versions'], 'native approval coverage/versions differ')
    require(p['source']['branch'] == BRANCH and p['source']['strict_baseline'] == BASE and
            p['source']['content_fingerprint'] == digest(p['source']['files']), 'source fingerprint contract differs')
    require(a['status'] == 'PASS' and a['failed'] == a['skipped'] == 0 and a['scope'] == 'CLEAN_NATIVE_CPU_APPROVAL' and
            a['source'] == p['source'] and a['plan_inventory'] == plan_inventory, 'fresh native CPU approval required')
    require(p['source']['approval_scope'] == 'CLEAN_REVIEWED_SOURCE' and r['source'] == p['source'] and
            r['plan_id'] == pid and r['plan_inventory'] == plan_inventory and
            r['cpu_inventory'] == cpu_inventory and r['status'] == 'COMPLETE', 'run/source/approval binding differs')
    require(r['condition'] in p['conditions'] and r['performance_knobs'] == p['performance_knobs'] and
            r['dataset'] == p['dataset'] and r['scale'] == p['count'] and
            r['tasks_sha256'] == p['stream']['tasks_sha256'], 'run/workload/knobs differ')
    require(r['measurement_scope'] == 'NATIVE_LIGHT' and r['gpu']['uuid'] and r['backend'] and
            r['precision_policy'] and r['cpu_concurrency'] and r['warmup_requests'] == 0 and
            r['independent_full_vocabulary_audit'] == 'NOT_EVALUATED' and
            r['correctness_qualified_speedup'] == 'NOT_ESTABLISHED', 'measurement metadata/claim differs')
    require(r['resources']['gpu_free_after_load_bytes'] >= 12 * 2**30 and
            r['resources']['peak_gpu_allocated_bytes'] < 22 * 2**30 and
            r['resources']['host_available_bytes'] >= 32 * 2**30 and
            r['resources']['cpu_count'] >= 4 and r['errors'] == [], 'actual run resource/error guard failed')
    requests = r['requests']
    require(len(requests) == len(p['stream']['tasks']) == p['count'], 'incomplete run request count')
    previous_end = None
    for index, (row, task) in enumerate(zip(requests, p['stream']['tasks'])):
        require(all(row[k] == task[k] for k in ('request_id', 'sample_id', 'user_id', 'adapter_id')) and
                row['session_ordinal'] == index and row['condition'] == r['condition'] and
                row['input_tokens'] == len(task['token_ids']), 'request order/identity differs')
        ids, text = row['generated_token_ids'], row['generated_text']
        require(ids and len(ids) <= task['policy']['max_new_tokens'] and all(type(t) is int and t >= 0 for t in ids) and
                row['output_tokens'] == len(ids) and row['generated_token_ids_sha256'] == digest(ids) and
                row['generated_text_sha256'] == __import__('hashlib').sha256(text.encode()).hexdigest(), 'generated output hash/budget differs')
        arrival, first, end = (row[k] for k in ('arrival_ns', 'first_available_ns', 'completion_ns'))
        require(arrival <= first <= end and (previous_end is None or arrival >= previous_end), 'overlapping/nonmonotonic request times')
        previous_end = end
        require(math.isclose(row['application_e2e_ms'], (end - arrival) / 1e6, abs_tol=1e-6) and
                math.isclose(row['application_ttft_ms'], (first - arrival) / 1e6, abs_tol=1e-6), 'E2E/TTFT definition differs')
        describe([row['application_e2e_ms'], row['application_ttft_ms']])
        require(type(row['skipped_qkv_rows']) is int and type(row['fresh_qkv_rows']) is int and
                row['skipped_qkv_rows'] >= 0 and row['fresh_qkv_rows'] >= 0 and
                row['skipped_qkv_rows'] + row['fresh_qkv_rows'] == 3 * 32 * (row['input_tokens'] + len(ids) - 1) and
                set(row['categories']) == set('ABCD') and sum(row['categories'].values()) == row['skipped_qkv_rows'] and
                row['physical_hit'] == (row['skipped_qkv_rows'] > 0), 'row accounting differs')
        require(0 <= row['cache_occupancy_bytes'] <= 512 * 2**20 and row['scorer_version'] and
                isinstance(row['quality'], dict) and isinstance(row['quality']['metric'], str) and
                type(row['quality']['score']) in (int, float) and math.isfinite(row['quality']['score']),
                'cache/quality metadata missing/invalid')
        if r['condition'] == CONDITIONS[0]:
            require(row['skipped_qkv_rows'] == 0, 'FULL cannot skip QKV rows')
        core_path = Path(root) / row['primary_receipt']
        require(row['primary_receipt'] == f'request_{index:03d}.json' and
                file_hash(core_path) == row['primary_receipt_sha256'], 'request primary receipt differs')
        core = read_json(core_path)
        require(core.pop('receipt_scope') == 'PRIMARY_BODY_FINAL_TIMING_IN_RUN_JSON' and
                all(row[k] == v for k, v in core.items()), 'primary/final request fields differ')
    from .r5_d5_lightperf_receipts import wire_equal
    by_adapter = {}
    for row in requests:
        by_adapter.setdefault(row['adapter_id'], []).append(row)
    require(set(r['ud']) == set(by_adapter), 'UD adapter coverage differs')
    for aid, rows in by_adapter.items():
        item = r['ud'][aid]
        require(Path(item['path']).absolute() == Path(str(Path(root).resolve()) + '__ud_' + aid), 'unexpected UD receipt path')
        require(verify(item['path']) == item['inventory'] and not (Path(item['path']) / 'failure.json').exists(), 'UD receipt changed/failed')
        u = read_json(Path(item['path']) / 'ud.json')
        require(u['status'] == 'COMPLETE' and u['source'] == p['source'] and u['plan_inventory'] == plan_inventory and
                u['cpu_inventory'] == cpu_inventory and u['completed_request_ids'] == [x['request_id'] for x in rows] and
                u['final_bye_sent'] is True and u['user_device_cuda_used'] is False and u['full_model_loaded'] is False,
                'UD final approval/lifecycle/isolation differs')
        for index, row in enumerate(rows):
            wire_equal(row['wire'], u['wire'][row['request_id']])
            ur = read_json(Path(item['path']) / f'request_{index:03d}.json')
            require(ur['request_id'] == row['request_id'] and ur['cpu_projection_rows'] == row['fresh_qkv_rows'] and
                    ur['compute'] == row['ud_compute'], 'UD compute/request summary differs')
    require(r['sequence_span_ms'] >= (requests[-1]['completion_ns'] - requests[0]['arrival_ns']) / 1e6 and
            r['sequence_span_ms'] > 0, 'invalid measured sequence span')
    return r, dict(run=run_inventory, plan=plan_inventory, cpu=cpu_inventory)


def summarize(r):
    rows = r['requests']
    span = r['sequence_span_ms'] / 1000
    quality_names = {x['quality']['metric'] for x in rows}
    require(len(quality_names) == 1, 'unrelated quality metrics cannot be pooled')
    return dict(dataset=r['dataset'], scale=r['scale'], condition=r['condition'],
                performance_knobs=r['performance_knobs'], gpu_uuid=r['gpu']['uuid'],
                ttft_ms=describe([x['application_ttft_ms'] for x in rows]),
                e2e_ms=describe([x['application_e2e_ms'] for x in rows]),
                total_request_e2e_ms=sum(x['application_e2e_ms'] for x in rows), sequence_span_ms=r['sequence_span_ms'],
                requests_per_second=len(rows) / span,
                output_tokens_per_second=sum(x['output_tokens'] for x in rows) / span,
                physical_hit_requests=sum(x['physical_hit'] for x in rows),
                first_hit_index=next((i for i, x in enumerate(rows) if x['physical_hit']), None),
                skipped_qkv_rows=sum(x['skipped_qkv_rows'] for x in rows),
                quality=dict(metric=next(iter(quality_names)), mean=statistics.mean(x['quality']['score'] for x in rows)),
                independent_full_vocabulary_audit='NOT_EVALUATED', correctness_qualified_speedup='NOT_ESTABLISHED')


def compare(full, other):
    require(full['condition'] == CONDITIONS[0] and other['condition'] in CONDITIONS[1:], 'invalid comparison conditions')
    keys = ('dataset', 'scale', 'tasks_sha256', 'source', 'performance_knobs', 'backend',
            'precision_policy', 'cpu_concurrency', 'warmup_requests', 'measurement_scope')
    mismatches = [key for key in keys if full[key] != other[key]]
    mismatches.extend('gpu.' + k for k in ('uuid', 'name', 'device', 'total_bytes')
                      if full['gpu'].get(k) != other['gpu'].get(k))
    if mismatches:
        return dict(status='NOT_COMPARABLE', mismatches=mismatches, delta_e2e_ms=None, speedup=None)
    require([x['request_id'] for x in full['requests']] == [x['request_id'] for x in other['requests']], 'request list differs')
    f = sum(x['application_e2e_ms'] for x in full['requests'])
    o = sum(x['application_e2e_ms'] for x in other['requests'])
    return dict(status='DESCRIPTIVE_MATCHED_IDENTITY', condition=other['condition'], delta_e2e_ms=o - f,
                speedup=None if o <= 0 else f / o,
                per_request_delta_e2e_ms=[b['application_e2e_ms'] - a['application_e2e_ms'] for a, b in zip(full['requests'], other['requests'])],
                sequence_span_delta_ms=other['sequence_span_ms'] - full['sequence_span_ms'],
                generated_token_agreements=sum(a['generated_token_ids'] == b['generated_token_ids'] for a, b in zip(full['requests'], other['requests'])),
                quality_curve='APPROX_SEPARATE_FIDELITY_UNESTABLISHED' if other['condition'] == CONDITIONS[2] else 'EXACT_FIDELITY_UNESTABLISHED',
                confidence='SINGLE_TRIAL_NO_STATISTICAL_SPEEDUP_CLAIM; resource contention may remain',
                correctness_qualified_speedup='NOT_ESTABLISHED')


def break_even(points):
    """Already identity-matched EXACT-FULL observations on ONE varying axis.

    Callers must group dataset/stream/device/source/other knob first. No estimate
    or extrapolation is represented as measured; unlimited is not numeric zero.
    """
    require(all(type(p['x']) in (int, float) and math.isfinite(p['x']) and
                type(p['delta_e2e_ms']) in (int, float) and math.isfinite(p['delta_e2e_ms']) for p in points), 'finite measured points required')
    ordered = sorted(points, key=lambda p: p['x'])
    require(len({p['x'] for p in ordered}) == len(ordered), 'duplicate ungrouped observations')
    brackets = []
    for a, b in zip(ordered, ordered[1:]):
        if a['delta_e2e_ms'] * b['delta_e2e_ms'] < 0:
            estimate = a['x'] - a['delta_e2e_ms'] * (b['x'] - a['x']) / (b['delta_e2e_ms'] - a['delta_e2e_ms'])
            brackets.append(dict(measured_bracket=[a['x'], b['x']], estimated_linear_crossing=estimate,
                                 estimate_scope='INTERPOLATION_ASSUMING_LINEARITY_NOT_MEASURED'))
    zeros = [p['x'] for p in ordered if p['delta_e2e_ms'] == 0]
    return dict(status='OBSERVED_SIGN_CROSSING_BRACKET' if brackets else ('OBSERVED_ZERO_SAMPLE' if zeros else 'NO_OBSERVED_BREAK_EVEN'),
                brackets=brackets, observed_zero_samples=zeros, extrapolation=None)
