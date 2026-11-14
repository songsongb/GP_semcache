"""CPU descriptive replay analysis; inputs must first pass the Light verifier."""
from collections import Counter
import math

from .communication import require
from .r5_d5_lightperf_analysis import describe, compare


def measured(values):
    present = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    return dict(status='MEASURED' if len(present) == len(values) else 'PARTIALLY_MEASURED' if present else 'NOT_MEASURED',
                measured_requests=len(present), statistics=describe(present) if present else None)


def profile_components(rows):
    names = sorted({(v['phase'], v['component']) for r in rows for v in r.get('profile', {}).get('components', [])})
    result = []
    for phase, name in names:
        inclusive, exclusive = [], []
        for row in rows:
            matches = [v for v in row.get('profile', {}).get('components', []) if v['phase'] == phase and v['component'] == name]
            require(len(matches) <= 1, 'duplicate compact profile component')
            inclusive.append(matches[0]['inclusive_wall_ms'] if matches else None)
            exclusive.append(matches[0]['exclusive_wall_ms'] if matches else None)
        result.append(dict(phase=phase, component=name, inclusive_wall_ms=measured(inclusive),
                           observed_exclusive_wall_ms=measured(exclusive)))
    return result


def segment(rows):
    require(rows, 'empty analysis segment')
    total_rows = sum(r['skipped_qkv_rows'] + r['fresh_qkv_rows'] for r in rows)
    hits = sum(r['physical_hit'] for r in rows)
    adapters, users = {}, {}
    for field, dest in (('adapter_id', adapters), ('user_id', users)):
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            skipped = sum(r['skipped_qkv_rows'] for r in subset)
            fresh = sum(r['fresh_qkv_rows'] for r in subset)
            dest[value] = dict(requests=len(subset), physical_hits=sum(r['physical_hit'] for r in subset),
                hit_rate=sum(r['physical_hit'] for r in subset) / len(subset), skipped_qkv_rows=skipped,
                fresh_qkv_rows=fresh, qkv_skip_ratio=None if not skipped + fresh else skipped / (skipped + fresh))
    names = ('prefill_ms', 'decode_ms', 'semantic_encoding_ms', 'cluster_assignment_ms',
             'admission_completion_ms', 'request_persistence_ms', 'quality_ms')
    timing = {name: measured([r.get(name) for r in rows]) for name in names}
    for field in ('owner_completion_ms', 'impact_history_chu_ms', 'insert_eviction_inclusive_ms',
                  'materialization_nested_ms', 'tensor_integrity_sealing_ms'):
        timing[field] = measured([(r.get('cache_completion_costs') or {}).get(field) for r in rows])
    for field in ('measured_compute_ms', 'measured_service_ms', 'pacing_ms', 'serialization_ms', 'deserialization_ms', 'hash_ms'):
        timing['ud_' + field] = measured([r.get('ud_compute', {}).get(field) for r in rows])
    for operation in ('send', 'recv'):
        timing['tcp_' + operation + '_inclusive_ms'] = measured([
            r.get('wire', {}).get(operation, {}).get('socket_call_wall_ms') for r in rows])
    hit_miss = {}
    for flag, label in ((True, 'HIT'), (False, 'MISS')):
        group = [r for r in rows if r['physical_hit'] == flag]
        hit_miss[label] = None if not group else dict(requests=len(group),
            e2e_ms=describe([r['application_e2e_ms'] for r in group]),
            owner_completion_ms=measured([(r.get('cache_completion_costs') or {}).get('owner_completion_ms') for r in group]),
            input_tokens=describe([r['input_tokens'] for r in group]), output_tokens=describe([r['output_tokens'] for r in group]))
    return dict(requests=len(rows), input_tokens=describe([r['input_tokens'] for r in rows]),
        output_tokens=describe([r['output_tokens'] for r in rows]),
        ttft_ms=describe([r['application_ttft_ms'] for r in rows]), e2e_ms=describe([r['application_e2e_ms'] for r in rows]),
        physical_hits=hits, hit_rate=hits / len(rows), skipped_qkv_rows=sum(r['skipped_qkv_rows'] for r in rows),
        fresh_qkv_rows=sum(r['fresh_qkv_rows'] for r in rows), qkv_skip_ratio=None if not total_rows else sum(r['skipped_qkv_rows'] for r in rows) / total_rows,
        unique_execution_requests=len({r['request_id'] for r in rows}), unique_samples=len({r['sample_id'] for r in rows}),
        unique_original_requests=len({r.get('replay_provenance', {}).get('original_request_id', r['request_id']) for r in rows}),
        repeated_occurrences=sum(r.get('replay_provenance', {}).get('occurrence_index', 0) > 0 for r in rows),
        per_adapter=adapters, per_user=users,
        cache_occupancy_bytes=measured([r.get('cache_occupancy_bytes') for r in rows]),
        policy_counts={k: sum(r['policy_counts'][k] for r in rows) if all(k in r.get('policy_counts', {}) for r in rows) else None
                       for k in ('INSERT', 'EVICT', 'CHU', 'PBR')},
        wire_bytes={op: sum(r['wire'][op]['wire_bytes'] for r in rows)
                    if all('wire_bytes' in r.get('wire', {}).get(op, {}) for r in rows) else None for op in ('send', 'recv')},
        timings=timing, hit_miss=hit_miss, profile_components=profile_components(rows),
        timing_scope='inclusive components overlap; observed exclusive scopes are partial; never sum to reconstruct E2E')


def run_segments(run):
    rows = run['requests']
    return dict(dataset=run['dataset'], scale=run['scale'], condition=run['condition'],
        workload_mode=run.get('workload_mode', 'NATURAL'), whole_session=segment(rows),
        windows=[dict(start=i, stop_exclusive=min(i + 5, len(rows)), **segment(rows[i:i + 5])) for i in range(0, len(rows), 5)],
        interpretation='position bins differ in tokens/output/user/adapter mix; no causal warm-up or statistical speedup claim')


def compare_segments(full, other):
    whole = compare(full, other)
    if whole['status'] == 'NOT_COMPARABLE':
        return dict(whole_session=whole, windows=[], observed_break_even='NOT_COMPARABLE')
    windows = []
    for i in range(0, len(full['requests']), 5):
        a, b = dict(full), dict(other)
        for current in (a, b):
            current['requests'] = current['requests'][i:i + 5]
            current['sequence_span_ms'] = (current['requests'][-1]['completion_ns'] - current['requests'][0]['arrival_ns']) / 1e6
        value = compare(a, b)
        windows.append(dict(start=i, stop_exclusive=i + len(a['requests']), **value))
    for value in [whole, *windows]:
        speed = value['speedup']
        value['measured_outcome'] = 'NOT_MEASURED' if speed is None else 'CACHE_FASTER' if speed > 1 else 'FULL_FASTER' if speed < 1 else 'EQUAL_AT_MEASURED_SAMPLE'
        value['generated_token_agreement_rate'] = value['generated_token_agreements'] / len(value['per_request_delta_e2e_ms'])
    return dict(whole_session=whole, windows=windows,
        observed_break_even='OBSERVED_EQUALITY_AT_MEASURED_SAMPLE' if any(v['speedup'] == 1 for v in [whole, *windows]) else 'NO_OBSERVED_BREAK_EVEN',
        threshold_estimate=None, note='no interpolation/extrapolation across heterogeneous position windows')
