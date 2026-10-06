"""C9-A provenance and synchronized local prompt-latency reporting. No quality/network evaluation."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
import time
from types import SimpleNamespace

from . import c7b3_q_freeze as provenance
from . import c8a_capacity as capacity
from . import c8b_quality as c8

require = provenance.require
BUDGETS, POLICIES, REFERENCE = c8.BUDGETS, c8.POLICIES, c8.REFERENCE
CONDITIONS = (REFERENCE, *(c8.condition(b, p) for b in BUDGETS for p in POLICIES))
REPEATS, WARMUP = 3, 5
COMPONENTS = ('target_total_ms', 'lookup_ms', 'storage_decode_ms', 'model_forward_ms',
              'projection_or_mixed_projection_ms', 'control_overhead_ms')
PAIRED_COMPONENTS = ('target_total_ms', 'lookup_ms', 'storage_decode_ms', 'model_forward_ms')
REVIEW_RULE = ('READY means hash-bound provenance, exact C8 residency/HIT vectors, synchronized GPU timing, '
    'zero runtime fitting/transport, and consistent decomposition. It does not require Q24 to be fastest '
    'and does not freeze a system policy.')
TIMING_METHOD = 'perf_counter_ns wall time with CUDA synchronization at request boundaries and decode/model boundaries; nested QKV CUDA events'


def validate_options(warmup, repeats):
    require(type(warmup) is int and warmup == WARMUP and type(repeats) is int and repeats == REPEATS,
        'Bounded C9-A requires exactly5 warmup forwards and3 repetitions')


def tie_tolerance_ms():
    return 10*time.get_clock_info('perf_counter').resolution*1000


def verify_c8b(args, prepared):
    root = args.c8b_root
    expected = dict(stage='C8-B', status='COMPLETE', recommendation='C8_B_SUPPORTS_Q24_KV_FOR_C9',
        c8a_residency_and_hit_vectors_reproduced=True, miss_paths_match_full=True,
        budgets_raw_entry_equivalent=list(BUDGETS), policies=list(POLICIES), reference_mode=REFERENCE,
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0,
        transport_encode_calls=0, transport_decode_calls=0, transport_compression_enabled=False,
        c7b3_freeze_decision_sha256=provenance.sha(args.freeze_decision),
        frozen32_selection_sha256=provenance.SELECTION_SHA, q24_profile_sha256=provenance.Q_SHA,
        kv_profile_sha256=provenance.KV_SHA, **prepared.frozen['model_namespace'])
    for name in ('manifest.json', 'summary.json', 'residency_trace.json', 'per_event.csv'):
        expected['c8a_'+name.rsplit('.', 1)[0]+'_sha256'] = provenance.sha(args.c8a_root/name)
    manifest, hashes = provenance.load_manifest(root, expected, required=('hit_audit.json',))
    provenance.verify_files(manifest['input_hashes'])
    require(all(manifest['input_hashes'].get(p) == h for p, h in prepared.input_hashes.items()), 'C8-B upstream chain differs')
    hashes.update(manifest['input_hashes'])
    rows = provenance.read(root/'hit_audit.json')['cases']
    require(len(rows) == 6*32, 'C8-B requires192 audited target conditions')
    by_pair = provenance.unique([dict(r, pair=r['condition']+'/'+r['episode_id']) for r in rows], 'pair')
    for name, saved in prepared.expected.items():
        for episode, event in zip(prepared.episodes, saved['lookups']):
            pair = name+'/'+episode['episode_id']
            require(pair in by_pair, 'Missing C8-B HIT audit: '+pair)
            row = by_pair[pair]
            require(all(row[k] == event[k] for k in event), 'C8-B HIT/resident source differs: '+pair)
            provenance.fields(row, dict(c8a_hit_exact_match=True, same_reuse_mask_qkv=True,
                target_admission_performed=False, reused_projection_rows_per_role_per_layer=3 if row['hit'] else 0,
                native_rows_skipped_per_role_per_layer=3 if row['hit'] else 0), 'C8-B HIT execution')
            require(len(row['reuse_audits']) == (2 if row['hit'] else 0), 'C8-B missing mixed reuse evidence')
            for audit in row['reuse_audits']:
                provenance.fields(audit, dict(same_reuse_mask_qkv=True,
                    hit_positions=list(range(episode['target_start'], episode['target_start']+3))), 'C8-B reuse mask')
                layers = provenance.unique(audit['per_layer'], 'layer')
                require(set(layers) == set(range(32)), 'C8-B reuse must cover all32 layers')
                for layer in layers.values():
                    provenance.fields(layer, {role+'_'+suffix: 3 for role in 'qkv'
                        for suffix in ('reused_projection_rows', 'native_projection_rows_skipped')}, 'C8-B row reuse')
    prepared.input_hashes.update(hashes)
    prepared.c8b_manifest = manifest
    provenance.verify_files(prepared.input_hashes)


def prepare(args):
    # Do NOT call C8-B prepare: it recalculates BLEU. C9 reads quality evidence only.
    prepared = c8.verify_capacity(args)
    verify_c8b(args, prepared)
    from . import c6b3_2_multiwoz as b3
    from . import c7b_q_profiles as qcodec
    from .c7b2_runtime import forbid_fitting, runtime_counts
    manifest = prepared.c8b_manifest
    for path in (args.source, args.semantic, args.adapter_freeze_decision, args.kv_profile,
                 args.adapter_root/'training_manifest.json'):
        key = str(path.resolve())
        require(key in prepared.input_hashes, 'Unbound C8-B runtime input: '+key)
        provenance.check_hash(path, prepared.input_hashes[key])
    for user in ('user_a', 'user_b'):
        for path in sorted((args.adapter_root/user).rglob('*')):
            if path.is_file():
                require(str(path.resolve()) in prepared.input_hashes, 'Unbound frozen adapter file: '+str(path))
                provenance.check_hash(path, prepared.input_hashes[str(path.resolve())])
    bargs = SimpleNamespace(**vars(args))
    bargs.freeze_decision = args.adapter_freeze_decision
    bargs.baseline_root = Path(manifest['teacher_forced_canonical_source']['manifest_path']).parent
    bargs.profile_path = args.kv_profile
    rows, episodes, _, _, chain = b3.prepare(bargs)  # Static provenance only; no BLEU/model forward.
    require(episodes == prepared.episodes, 'Canonical C9 selection differs from C8')
    for key in ('plan_manifest_sha256', 'training_manifest_sha256', 'freeze_decision_sha256',
                'canonical_frozen32_manifest_sha256', 'semantic_workload_sha256', 'adapter_hashes'):
        require(chain[key] == manifest['canonical_full_provenance'][key], 'C9 model/workload namespace differs: '+key)
    with forbid_fitting(runtime_counts(), 'q'):
        backend = qcodec.load_backend(args.storage_src)
    calibration = provenance.read(Path(prepared.frozen['b1_manifest_path']))
    require(backend['provenance']['source_hashes'] == calibration['transform_provenance']['source_hashes'], 'Frozen Q backend differs')
    prepared.input_hashes.update(backend['provenance']['source_hashes'])
    prepared.rows, prepared.provenance, prepared.q_backend = rows, chain, backend
    provenance.verify_files(prepared.input_hashes)
    return prepared


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered)-1)*fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)


def stats(values):
    if not values: return dict(mean=None, p50=None, p95=None, stddev=None)
    require(all(math.isfinite(v) for v in values), 'Nonfinite timing statistic')
    return dict(mean=mean(values), p50=median(values), p95=percentile(values, .95), stddev=pstdev(values))


def validate_timing(row, tolerance):
    require(row['condition'] in CONDITIONS and type(row['repeat']) is int and row['repeat'] in range(REPEATS), 'Invalid timing condition/repetition')
    require(all(type(row[k]) in (float, int) and math.isfinite(row[k]) and row[k] >= 0 for k in COMPONENTS), 'Invalid timing component')
    total = row['lookup_ms']+row['storage_decode_ms']+row['model_forward_ms']+row['control_overhead_ms']
    require(abs(total-row['target_total_ms']) <= tolerance, 'Timing decomposition is inconsistent')
    require(row['projection_or_mixed_projection_ms'] <= row['model_forward_ms']+tolerance, 'Nested projection timing exceeds enclosing forward')
    require(row['hit'] or row['storage_decode_ms'] == 0, 'MISS must not decode cache storage')
    if row['condition'] == REFERENCE:
        require(row['hit'] is False and row['lookup_ms'] == row['storage_decode_ms'] == 0, 'FULL must not use cache')
    if row['condition'].endswith('_RAW_QKV'):
        require(row['storage_decode_ms'] == 0, 'Raw cache must not run a codec')


def per_case(raw, episodes, expected, tolerance):
    ids = [r['episode_id'] for r in episodes]
    require(len(raw) == len(CONDITIONS)*len(ids)*REPEATS, 'Incomplete raw target timing')
    groups = {}
    for row in raw:
        validate_timing(row, tolerance)
        require(row['episode_id'] in ids, 'Unknown target timing episode')
        groups.setdefault((row['condition'], row['episode_id']), []).append(row)
    rows = []
    for name in CONDITIONS:
        for index, episode_id in enumerate(ids):
            group = groups.get((name, episode_id), [])
            require(len(group) == REPEATS and {r['repeat'] for r in group} == set(range(REPEATS)), 'Missing/duplicate repeat')
            hit = False if name == REFERENCE else expected[name]['lookups'][index]['hit']
            source = None if name == REFERENCE else expected[name]['lookups'][index]['resident_source_episode_id']
            require(all(r['hit'] is hit and r['retained_source_episode_id'] == source for r in group), 'Repeated HIT/source decision differs from C8')
            row = dict(condition=name, episode_id=episode_id, episode_index=index, hit=hit,
                retained_source_episode_id=source, repetitions=REPEATS, aggregation='per-component median of3 synchronized repetitions',
                **{k: median(r[k] for r in group) for k in COMPONENTS})
            # Independent component medians need not sum to the median total.
            # Consistency is checked on raw repeats; NEVER add component medians.
            rows.append(row)
    full = {r['episode_id']: r for r in rows if r['condition'] == REFERENCE}
    for row in rows:
        reference = full[row['episode_id']]['target_total_ms']
        require(row['target_total_ms'] > 0, 'Positive measured request latency required')
        row.update(delta_vs_full_ms=row['target_total_ms']-reference, speedup_vs_full=reference/row['target_total_ms'])
    return rows


def aggregate(rows):
    return dict(cases=len(rows), **{k: stats([r[k] for r in rows]) for k in COMPONENTS},
        delta_vs_full_ms=stats([r['delta_vs_full_ms'] for r in rows]),
        speedup_vs_full_ratio_of_means=(mean(r['target_total_ms']-r['delta_vs_full_ms'] for r in rows)/mean(r['target_total_ms'] for r in rows)) if rows else None)


def summaries(rows):
    result = []
    for name in CONDITIONS:
        group = [r for r in rows if r['condition'] == name]
        result.append(dict(condition=name, ALL=aggregate(group),
            HIT=aggregate([r for r in group if r['hit']]), MISS=aggregate([r for r in group if not r['hit']])))
    return result


def paired(rows, tolerance):
    result = []
    by_key = {(r['condition'], r['episode_id']): r for r in rows}
    ids = [r['episode_id'] for r in rows if r['condition'] == REFERENCE]
    for budget in BUDGETS:
        pairs = [(c8.condition(budget, before), c8.condition(budget, after)) for before, after in
            (('RAW_QKV', 'KV_COMP'), ('KV_COMP', 'Q24_KV_COMP'), ('RAW_QKV', 'Q24_KV_COMP'))]
        pairs += [(REFERENCE, c8.condition(budget, p)) for p in POLICIES]
        for before, after in pairs:
            differences = [{k: by_key[after, i][k]-by_key[before, i][k] for k in PAIRED_COMPONENTS} for i in ids]
            result.append(dict(budget_raw_entry_equivalent=budget, before=before, after=after,
                comparison=after+' - '+before, tie_tolerance_ms=tolerance,
                components={k: dict(**stats([d[k] for d in differences]),
                    faster=sum(d[k] < -tolerance for d in differences), slower=sum(d[k] > tolerance for d in differences),
                    effectively_tied=sum(abs(d[k]) <= tolerance for d in differences)) for k in PAIRED_COMPONENTS},
                per_target=[dict(episode_id=i, **d) for i, d in zip(ids, differences)]))
    return result


def source_summary(rows):
    groups = []
    for name in CONDITIONS[1:]:
        group = [r for r in rows if r['condition'] == name]
        require(len(group) == 32, 'Incomplete one-time source build timing')
        totals = {k: sum(r[k] for r in group) for k in ('source_forward_ms', 'source_capture_ms', 'storage_encode_ms', 'cache_admission_ms')}
        for r in group:
            require(all(math.isfinite(r[k]) and r[k] >= 0 for k in totals), 'Invalid source build timing')
        total = sum(totals.values())
        groups.append(dict(condition=name, source_admission_events=32, **totals,
            total_source_build_ms=total, mean_source_build_ms_per_admission=total/32))
    return dict(source_build_cost_in_primary_target_latency=False, conditions=groups,
        accounting='one-cache construction scenarios; source forward/capture shared across policies; encoded objects shared across budgets; do not sum scenario totals')


def expected_execution_counts(expected):
    """Counts include explicit first-real-block codec warmup, never MISS decode."""
    hits = {name: sum(event['hit'] for event in saved['lookups']) for name, saved in expected.items()}
    require(set(hits) == set(CONDITIONS[1:]), 'Incomplete C8 conditions')
    compressed_hits = sum(value for name, value in hits.items() if not name.endswith('_RAW_QKV'))
    q_hits = sum(value for name, value in hits.items() if name.endswith('_Q24_KV_COMP'))
    return dict(source_forward_count=32, target_prompt_forward_count=WARMUP+32*len(CONDITIONS)*REPEATS,
        q_encode_count=32+1, q_decode_count=q_hits*REPEATS+1,
        storage_decode_count=compressed_hits*REPEATS+2,
        target_greedy_forward_count=0, teacher_forced_forward_count=0, **{k: 0 for k in c8.SAFETY})


def write_reports(root, raw, source, prepared, manifest):
    complete = manifest['status'] == 'COMPLETE'
    rows = per_case(raw, prepared.episodes, prepared.expected, manifest['tie_tolerance_ms']) if complete else []
    summary = summaries(rows) if complete else []
    pairs = paired(rows, manifest['tie_tolerance_ms']) if complete else []
    source_data = source_summary(source) if complete else dict(status='NOT_COMPLETED', partial_rows=len(source))
    def csv_file(name, values):
        # The root belongs exclusively to this new run; allow failure reporting
        # to replace its own partially written reports without masking the cause.
        path = root/name
        import csv
        columns = list(dict.fromkeys(k for r in values for k in r))
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            writer.writerows(c8.csv_form(r) for r in values)
    csv_file('source_build_latency.csv', source); csv_file('target_latency_raw.csv', raw)
    csv_file('target_latency_per_case.csv', rows)
    for name, value in (('source_build_summary.json', source_data),
        ('target_latency_summary.json', dict(status=manifest['status'], primary_metric='target_total_ms',
            aggregation='per-target median, then unweighted cross-target statistics', conditions=summary)),
        ('hit_miss_latency.json', summary), ('paired_latency.json', pairs)):
        (root/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    lines = ['# C9-A measured local prompt latency', '', 'Status: '+manifest['status'],
        'Primary scope: cache lookup through next-token logits. Source build, model/tokenizer initialization, disk I/O, and correctness-only validation are excluded.',
        'Transport compression is OFF; no network latency, distributed speedup, or new quality evaluation is claimed.',
        'Frozen32 remains a selection-based controlled trace, not an untouched final test cohort.',
        'QKV CUDA event timing is nested in model_forward_ms. Never add it to model_forward_ms or target_total_ms.',
        'Component medians do not necessarily sum to the median total; additive consistency is checked on each raw repetition.', '',
        'Review rule: '+REVIEW_RULE, '']
    if complete:
        lines += ['| Condition | Hits /32 | Total mean /p50 /p95 ms | HIT mean ms | MISS mean ms | MISS delta vs FULL ms | HIT decode mean ms | Speedup vs FULL |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
        for row in summary:
            a, h, m = row['ALL'], row['HIT'], row['MISS']
            t = a['target_total_ms']
            lines.append(f"| {row['condition']} | {h['cases']} | {t['mean']:.6f} /{t['p50']:.6f} /{t['p95']:.6f} | {h['target_total_ms']['mean']} | {m['target_total_ms']['mean']} | {m['delta_vs_full_ms']['mean']} | {h['storage_decode_ms']['mean']} | {a['speedup_vs_full_ratio_of_means']:.6f} |")
        lines += ['', 'Q24 versus KV_COMP (negative deltas are faster; matched all-target comparisons include the changed HIT coverage):']
        for row in pairs:
            if row['before'].endswith('_KV_COMP') and not row['before'].endswith('_Q24_KV_COMP') and row['after'].endswith('_Q24_KV_COMP'):
                lines.append(f"B{row['budget_raw_entry_equivalent']}: mean total delta {row['components']['target_total_ms']['mean']:.6f} ms; mean decode delta {row['components']['storage_decode_ms']['mean']:.6f} ms.")
                verdict = 'offset its additional decode cost in the measured all-target mean' if row['components']['target_total_ms']['mean'] < -manifest['tie_tolerance_ms'] else 'did not show an all-target mean benefit beyond the declared tie tolerance'
                lines.append('Q24 '+verdict+'. This is a descriptive local measurement, not a statistical significance claim.')
        lines += ['', 'One-time source costs (separate from steady-state target latency):']
        for r in source_data['conditions']:
            lines.append(f"{r['condition']}: source forward {r['source_forward_ms']:.6f} ms; capture {r['source_capture_ms']:.6f} ms; storage preparation/encode {r['storage_encode_ms']:.6f} ms; admission {r['cache_admission_ms']:.6f} ms; total {r['total_source_build_ms']:.6f} ms ({r['mean_source_build_ms_per_admission']:.6f} ms/admission).")
        lines += ['', 'Local timing is internally valid for proceeding to network accounting; this does not establish Q24 as the fastest system policy.']
    else:
        lines += ['Timing/HIT decomposition: NOT_COMPLETED.', 'Primary failure: '+manifest.get('failure_message', '')]
    lines += ['', manifest['recommendation'], '']
    (root/'summary.md').write_text('\n'.join(lines))
    manifest['output_hashes'] = {p.name: provenance.sha(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(adapter_root='results/cachegen/c6b3/b1_full', adapter_freeze_decision='results/cachegen/c6b3/b1_full_freeze_decision.json',
        plan_dir='results/cachegen/c6b3/multiwoz_plan_v2', source='results/workloads/multiwoz.jsonl',
        semantic='results/workloads/c6b3_multiwoz_history_semantic.jsonl', freeze_decision='results/cachegen/c7/q24_freeze/freeze_decision.json',
        c8a_root='results/cachegen/c8/a_capacity_replay', c8b_root='results/cachegen/c8/b_quality_b2_b8',
        kv_profile='results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin',
        storage_src='/data/khuss/repos/GP_semcache/.c6_storage_src/src', output_root='results/cachegen/c9/a_local_latency')
    for name, default in defaults.items(): parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(default))
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=WARMUP); parser.add_argument('--repeats', type=int, default=REPEATS)
    args = parser.parse_args(argv); validate_options(args.warmup, args.repeats)
    require(args.seed == 42 and args.device.startswith('cuda:'), 'Frozen seed42 and one GPU required')
    require(Path('/data/khuss').is_dir() and args.output_root.resolve().is_relative_to(Path('/data/khuss')),
        'Real execution requires SERAPH with outputs under /data/khuss')
    require(not args.output_root.exists(), 'New C9-A output root required; no overwrite')
    from . import c7b_q_capture as b0
    b0.seraph_paths(args.output_root)
    args.profile_path = args.kv_profile; args.max_sequence_length = 384; args.max_new_tokens = 160
    prepared = prepare(args)
    from .c9a_runtime import run
    run(args, prepared)
