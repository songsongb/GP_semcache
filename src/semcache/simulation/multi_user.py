"""M9-B logical, paired simulation. No model, tokenizer, or dataset downloads."""
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from statistics import mean

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.metric_manager import CacheMetricManager
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.experiments.user_assignment import assign_users
from semcache.system_cost.common import Dimensions, nonnegative
from semcache.system_cost.network import communication
from semcache.diagnostics.impact_cost import recompose

CAPACITY = 20 * 1024**3
SCENARIOS = ('CURRENT_BASELINE', 'TOKEN_PRECOMPUTE_DIAGNOSTIC')
PROVENANCE = dict(window='PAPER_DEFINED', rho='PAPER_DEFINED', history_lambda='PAPER_DEFINED',
    capacity='PAPER_REFERENCE', assignment='REPRODUCTION_CHOICE',
    cluster_schedule='REPRODUCTION_CHOICE', pbr_schedule='REPRODUCTION_CHOICE',
    es='MEASURED', ud='CALIBRATED', ud_measurement_label='CALIBRATED_ON_SERAPH_CPU',
    network='SIMULATED', totals='SIMULATED', decision='RESEARCH_EXTENSION',
    diagnostic='SIMULATED_RESEARCH_EXTENSION')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_workload(path, dataset):
    """Consume pre-tokenized semantic JSONL. Never infer intent or token IDs."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty workload')
    namespace = None
    for i, row in enumerate(rows):
        if row.get('dataset', dataset).lower() != dataset:
            raise ValueError('Dataset mismatch')
        ids = row.get('token_ids')
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('Need actual nonempty token_ids; raw text requires an existing semantic artifact')
        if type(row.get('cluster_id')) is not int:
            raise ValueError('Need existing integer cluster_id, not invented TinyBERT assignments')
        identity = tuple(row.get(k) for k in ('model_id', 'tokenizer_id', 'semantic_assignment_source'))
        if not all(identity) or (namespace is not None and identity != namespace):
            raise ValueError('Need one explicit model/tokenizer/semantic assignment namespace')
        namespace = identity
        if row['model_id'] != 'facebook/opt-2.7b':
            raise ValueError('Workload byte accounting requires OPT-2.7B token namespace')
        row.update(dataset=dataset, source_id=str(row.get('source_id', i)))
    return rows


def safety_eligible(source, target, start, evidence):
    """Exact full prompt, owner, adapter and occurrence plus external fixture evidence."""
    if not source or source['user_id'] != target['user_id']:
        return False
    if not source.get('adapter_id') or source.get('adapter_id') != target.get('adapter_id'):
        return False
    if source['prompt_hash'] != target['prompt_hash'] or source['start'] != start:
        return False
    # Production CLI passes no evidence: workload metadata is never correctness proof.
    return (target['user_id'], target['adapter_id'], target['prompt_hash'], start) in evidence


def execute_reuse(safe, reuse_ms, recompute_ms):
    if reuse_ms is None or recompute_ms is None:
        return False
    nonnegative(reuse_ms, 'reuse_ms')
    nonnegative(recompute_ms, 'recompute_ms')
    return bool(safe and reuse_ms < recompute_ms)


def simulate(rows, users, seed=42, capacity=CAPACITY, dims=None):
    dims = dims or Dimensions('facebook/opt-2.7b', 2560, 32)
    assignment = assign_users(rows, users, seed, 'seeded_round_robin')
    cache = GlobalCache(capacity)
    manager = CacheMetricManager(cache, rho=.8, history_lambda=100, frequency_window=100)
    extractor = SubsequenceExtractor(3)
    trace = []
    for index, (row, user) in enumerate(zip(rows, assignment)):
        windows = extractor.extract(row['token_ids'])
        keys = [(row['cluster_id'], w.token_ids) for w in windows]
        manager.arrive(keys)
        target = dict(user_id=user, adapter_id=row.get('adapter_id'), prompt_hash=digest(row['token_ids']))
        events, hit_windows, safe_windows, candidate_tokens = [], [], [], set()
        # All lookups precede admission: no within-query self-hits.
        for w, key in zip(windows, keys):
            entry = cache.lookup(key, record_reuse=False)
            hit = entry is not None
            safe = hit and safety_eligible(entry.qkv_metadata, target, w.start, set())
            hit_windows.append(hit)
            safe_windows.append(bool(safe))
            if hit:
                candidate_tokens.update(range(w.start, w.end))
        admissions = rejections = evictions = 0
        for w, key in zip(windows, keys):
            if key in cache.entries:
                continue
            entry = CacheEntry(row['cluster_id'], w.token_ids, (w.start, w.end),
                3 * 3 * dims.hidden_size * dims.layers * 2,
                qkv_metadata=dict(target, start=w.start), impact=None)
            emitted = []
            cache.insert(entry, manager.frequencies[key], manager.frequencies,
                         on_event=lambda kind, e, score: emitted.append(kind))
            admissions += emitted.count('INSERT')
            evictions += emitted.count('EVICT')
            rejections += int('INSERT' not in emitted)
            events.extend(emitted or ['REJECT'])
        # Missing attention remains None. No synthetic I or candidate-as-actual-reuse F.
        manager.history.append(row['cluster_id'], index, index, {})
        if (index + 1) % 100 == 0:
            for cluster in sorted({e.cluster_id for e in cache.entries.values()}):
                manager.pbr(cluster)
        n = len(row['token_ids'])
        baseline = communication(dims, n)['request']['total_network_bytes']
        potential = baseline - communication(dims, n, len(candidate_tokens))['request']['total_network_bytes']
        trace.append(dict(query_index=index, source_id=row['source_id'], user_id=user,
            prompt_tokens=n, candidate_window_count=len(windows), lookup_count=len(windows),
            candidate_hit_count=sum(hit_windows), safe_candidate_count=sum(safe_windows),
            candidate_hit_mask=hit_windows, safety_eligible_mask=safe_windows,
            cost_effective_reuse_count=0, reused_token_count=0, fresh_token_count=n,
            admission_count=admissions, rejection_count=rejections, eviction_count=evictions,
            cache_bytes=cache.logical_cache_bytes, events=events,
            communication_baseline_bytes=baseline, communication_reuse_bytes=baseline,
            communication_saved_bytes=0, candidate_potential_saved_bytes=potential,
            actual_attention_impact_available=False, latency_supported=False,
            latency_unavailable_reason='No workload-specific strict ES timing and correctness evidence',
            selected_action='RECOMPUTE'))
    return dict(trace=trace, query_order_hash=digest(rows), user_assignment_hash=digest(assignment),
        cache_trace_hash=digest(trace), user_count=users, seed=seed, cache_capacity_bytes=capacity,
        cache_peak_bytes=cache.peak_logical_cache_bytes, cache_final_bytes=cache.logical_cache_bytes)


COUNTERS = ('candidate_window_count', 'lookup_count', 'candidate_hit_count', 'safe_candidate_count',
    'cost_effective_reuse_count', 'reused_token_count', 'fresh_token_count', 'admission_count',
    'rejection_count', 'eviction_count', 'communication_baseline_bytes', 'communication_reuse_bytes',
    'communication_saved_bytes', 'candidate_potential_saved_bytes')


def aggregate(trace):
    result = {key: sum(row[key] for row in trace) for key in COUNTERS}
    result['query_count'] = len(trace)
    for name, count in (('candidate_hit_rate', 'candidate_hit_count'),
                        ('safety_eligible_hit_rate', 'safe_candidate_count'),
                        ('cost_effective_reuse_rate', 'cost_effective_reuse_count')):
        result[name] = result[count] / result['lookup_count'] if result['lookup_count'] else 0.
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lo = int(position)
    return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(position-lo)


def fixture_costs(summary_path, impact_path):
    """Read M9 outputs; diagnostic recomposition validates existing arithmetic."""
    summary = json.loads(Path(summary_path).read_text())
    impacts = json.loads(Path(impact_path).read_text())
    selected = [x for x in impacts if x['implementation'] == 'TOKEN_PRECOMPUTE']
    if len(selected) != 1:
        raise ValueError('Need one TOKEN_PRECOMPUTE uninstrumented summary')
    replacement = selected[0]['uninstrumented_impact_total_ms']['mean']
    comparisons = summary['comparisons']
    if not comparisons:
        raise ValueError('Empty strict cost summary')
    result = {}
    # Select one original bandwidth per repeat; avoid tripling paired samples.
    seen = set()
    unique = []
    for c in comparisons:
        r = c['row']
        if r['rank'] != 8 or r.get('es_compute_policy') not in ('strict-base-only', 'require-base-only'):
            raise ValueError('Need rank-8 strict-base-only cost evidence')
        if r.get('es_compute_ms_provenance') != 'MEASURED' or r.get('ud_lora_ms_provenance') != 'CALIBRATED':
            raise ValueError('Invalid ES/UD cost provenance')
        identity = (r['model'], r['source_prompt_hash'], r['reused_tokens'], r['repeat_index'])
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(c)
    if len({(c['row']['model'], c['row']['source_prompt_hash'], c['row']['reused_tokens']) for c in unique}) != 1:
        raise ValueError('Normalized latency analysis requires one fixture across repeats')
    for scenario in SCENARIOS:
        for bandwidth in (200, 500, 1000):
            costs = [recompose(c, c['row']['attention_impact_ms'] if scenario == SCENARIOS[0] else replacement,
                              'CURRENT_BLOCKWISE' if scenario == SCENARIOS[0] else 'TOKEN_PRECOMPUTE',
                              bandwidth_mbps=bandwidth) for c in unique]
            totals = [c['semcache_total_ms'] for c in costs]
            result[scenario, bandwidth] = dict(normalized_fixture_mean_modeled_latency_ms=mean(totals),
                normalized_fixture_p50_ms=percentile(totals, .5), normalized_fixture_p95_ms=percentile(totals, .95),
                normalized_fixture_edgelora_ms=mean(c['edge_lora_total_ms'] for c in costs),
                normalized_fixture_system_delta_ms=mean(c['system_delta_ms'] for c in costs),
                normalized_fixture_reuse_decision_fraction=mean(c['selected_action']=='REUSE' for c in costs),
                normalized_fixture_prompt_tokens=unique[0]['row']['prompt_tokens'],
                normalized_fixture_reused_tokens=unique[0]['row']['reused_tokens'],
                normalized_fixture_sample_count=len(costs))
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in r.items()} for r in rows)


def run_matrix(workloads, output, seed=42, costs=None, input_paths=(), smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, per_user, traces, raw = [], [], [], []
    for dataset, rows in workloads.items():
        for users in (10, 25, 50):
            simulation = simulate(rows, users, seed)
            common = dict(dataset=dataset, user_count=users, seed=seed,
                query_order_hash=simulation['query_order_hash'], user_assignment_hash=simulation['user_assignment_hash'],
                cache_trace_hash=simulation['cache_trace_hash'])
            traces.extend(dict(common, **r) for r in simulation['trace'])
            for bandwidth in (200, 500, 1000):
                for scenario in SCENARIOS:
                    identity = dict(common, bandwidth_mbps=bandwidth, cost_scenario=scenario)
                    row = dict(identity, **aggregate(simulation['trace']),
                        cache_peak_bytes=simulation['cache_peak_bytes'], cache_final_bytes=simulation['cache_final_bytes'],
                        cache_capacity_bytes=CAPACITY, cache_capacity_provenance='PAPER_REFERENCE',
                        cache_occupancy_ratio=simulation['cache_final_bytes']/CAPACITY,
                        safe_reuse_claimed=False, result_provenance='SIMULATED' if scenario == SCENARIOS[0] else 'SIMULATED_RESEARCH_EXTENSION',
                        workload_mean_modeled_latency_ms=None, workload_p50_ms=None, workload_p95_ms=None,
                        reuse_decision_fraction=0., recompute_decision_fraction=1.,
                        normalized_fixture_status='AVAILABLE' if costs else 'MISSING_LOCAL_COST_ARTIFACTS',
                        synthetic_smoke=smoke)
                    row.update((costs or {}).get((scenario, bandwidth), {}))
                    summaries.append(row)
                    raw.append(dict(row, provenance=PROVENANCE, per_user_query_count={f'user_{u:03d}': sum(r['user_id']==f'user_{u:03d}' for r in simulation['trace']) for u in range(users)}))
                    for u in range(users):
                        user = f'user_{u:03d}'
                        per_user.append(dict(identity, user_id=user, **aggregate([r for r in simulation['trace'] if r['user_id']==user])))
    write_csv(output/'summary.csv', summaries)
    write_csv(output/'per_user.csv', per_user)
    write_csv(output/'cache_trace.csv', traces)
    (output/'run_raw.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in raw))
    plots = {'user_count_candidate_hit_rate': 'candidate_hit_rate',
        'user_count_safety_eligible_hit_rate': 'safety_eligible_hit_rate',
        'user_count_cost_effective_reuse_rate': 'cost_effective_reuse_rate',
        'bandwidth_system_delta': 'normalized_fixture_system_delta_ms',
        'bandwidth_reuse_decision_fraction': 'normalized_fixture_reuse_decision_fraction',
        'communication_saved_user_count': 'communication_saved_bytes'}
    for name, metric in plots.items():
        write_csv(output/(name+'.csv'), [{k: r.get(k) for k in ('dataset','user_count','bandwidth_mbps','cost_scenario',metric,'candidate_potential_saved_bytes','synthetic_smoke')} for r in summaries])
    write_csv(output/'cache_occupancy_query_index.csv', [dict(dataset=r['dataset'], user_count=r['user_count'], query_index=r['query_index'], cache_bytes=r['cache_bytes'], cache_occupancy_ratio=r['cache_bytes']/CAPACITY) for r in traces])
    environment = dict(python=platform.python_version(), hostname=platform.node(), model_execution_performed=False,
        distributed_execution=False, downloads_performed=False, provenance=PROVENANCE,
        cost_artifacts_available=bool(costs), synthetic_smoke=smoke)
    manifest = dict(schema='m9b_v1', seed=seed, run_count=len(raw), safe_reuse_claimed=False,
        provenance=PROVENANCE, window=3, rank=8, cache_capacity_bytes=CAPACITY, rho=.8,
        history_lambda=100, cluster_update_interval=100, pbr_interval=100,
        cluster_policy='Replay artifact assignments; no online re-clustering, schedule unavailable',
        impact_policy='Actual attention unavailable; I=None mapped to zero only by existing policy scorer; no CHU',
        eviction_frequency_policy='Actual reuse F stays zero; candidate lookups do not mutate reuse frequency',
        latency_policy='Separate normalized measured fixture only; never transfer latency to workload lengths',
        safety_policy='No CLI evidence importer: all unvalidated workload candidates fail closed',
        workload_policy='Fixed source order/corpus for all user counts; seeded round robin logical users',
        byte_policy='OPT-2.7B d=2560 L=32; QKV FP16 logical storage; h0/hL retained; no protocol bytes',
        input_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in input_paths},
        output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.suffix in ('.csv','.jsonl')})
    for filename, value in (('environment.json', environment), ('manifest.json', manifest)):
        (output/filename).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return summaries
