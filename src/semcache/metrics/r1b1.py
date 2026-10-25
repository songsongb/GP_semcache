"""R1-B1 extension: production policy evidence and paired impact A/B reports.

Tensor-free inspection/comparison runs after closed request timers. Failed
decisions remain in raw artifacts; floating tolerance never excuses them.
"""
from collections import defaultdict
import csv
import json
import math
from statistics import mean, pstdev

from semcache.cache.impact_execution import EXECUTIONS
from semcache.diagnostics.impact import ATOL, RTOL, parity
from .r1 import atomic_json, percentile


def snapshot(engine, result):
    """Copy scalars/identities only; never retain GPU tensors or mutate history."""
    events = {"CACHE_LOOKUP", "HIT", "MISS", "NON_OVERLAP_RESOLVE", "ADMISSION_SCORE",
              "ADMIT", "DENY", "INSERT", "EVICT", "CHU", "PBR", "DEDUPLICATE"}
    history = [dict(cluster=cluster, observations=[dict(query_id=q['query_id'], order=q['order'],
        impacts=[dict(key=k, value=v) for k,v in sorted(q['impacts'].items())]) for q in observations])
        for cluster, observations in sorted(engine.metrics.history.history.items())]
    entries = [dict(key=e.key, positions=e.positions, frequency=e.frequency, impact=e.impact,
        size_bytes=e.size_bytes, created_at=e.created_at, updated_at=e.updated_at,
        last_access=e.last_access, provenance=e.qkv_metadata)
        for _,e in sorted(engine.cache.entries.items())]
    value = dict(windows=result['summary']['impact_windows'], history=history,
        clock=engine.cache.now, history_lambda=engine.metrics.history.history_lambda,
        entries=entries, events=[e for e in result['events'] if e['event_type'] in events],
        reuse_provenance=result['summary']['reuse_block_provenance'],
        projection_records=list(result['projection_audit'].values()),
        reused_tokens=result['summary']['reused_unique_token_count'],
        cache_hits=engine.cache.hits, cache_misses=engine.cache.misses,
        candidate_order=engine._admission_candidate_order, admission_order=engine._admission_order,
        pbr_updates=result['summary']['pbr_updates_this_query'])
    return json.loads(json.dumps(value))  # detach containers + canonical tuple keys


def compare_snapshots(reference, candidate):
    differences = []
    maximum = 0.
    def walk(a, b, path):
        nonlocal maximum
        if isinstance(a, bool) or isinstance(b, bool):
            if type(a) is not type(b) or a != b:
                differences.append(dict(path=path, reference=a, candidate=b, kind="exact_decision"))
        elif isinstance(a, float) or isinstance(b, float):
            if not isinstance(a, (int,float)) or not isinstance(b,(int,float)):
                differences.append(dict(path=path, reference=a, candidate=b, kind="type"))
            else:
                error = abs(a-b)
                maximum = max(maximum, error) if math.isfinite(error) else maximum
                if not (math.isfinite(a) and math.isfinite(b) and error <= ATOL+RTOL*abs(a)):
                    differences.append(dict(path=path, reference=a, candidate=b, kind="numeric_tolerance"))
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append(dict(path=path, reference=sorted(a), candidate=sorted(b), kind="keys"))
            for key in sorted(set(a) & set(b)):
                walk(a[key], b[key], path + "/" + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(dict(path=path, reference=len(a), candidate=len(b), kind="length"))
            for i,(x,y) in enumerate(zip(a,b)):
                walk(x,y,path+f"/{i}")
        elif type(a) is not type(b) or a != b:
            differences.append(dict(path=path, reference=a, candidate=b, kind="exact_identity"))
    walk(reference,candidate,"")
    return dict(passed=not differences, differences=differences, numeric_max_absolute_error=maximum,
                atol=ATOL, rtol=RTOL)


def pair_key(row):
    return tuple(row[k] for k in ('phase','repeat','storage','instrumentation',
                                 'requested_prompt_tokens','condition','mode'))


def pair_checks(rows):
    legacy = {pair_key(r): r for r in rows if r['impact_execution'] == 'legacy_blockwise'}
    checks = []
    for row in rows:
        if row['impact_execution'] == 'legacy_blockwise':
            continue
        reference = legacy.get(pair_key(row))
        if reference is None:
            checks.append(dict(pair_id=row['pair_id'], mode=row['mode'], impact_execution=row['impact_execution'],
                               phase=row['phase'], passed=None, status='PENDING_LEGACY_PAIR'))
            continue
        stages = {}
        for name in ('source_policy', 'target_policy'):
            stages[name] = compare_snapshots(reference[name], row[name])
        identities = compare_snapshots({k: reference[k] for k in ('token_ids_sha256','adapter','reuse_provenance',
            'reused_tokens','projection_audit','resident_bytes_before','resident_bytes_after')},
            {k: row[k] for k in ('token_ids_sha256','adapter','reuse_provenance','reused_tokens',
            'projection_audit','resident_bytes_before','resident_bytes_after')})
        impacts = {}
        for name in ('source_policy', 'target_policy'):
            a,b = reference[name],row[name]
            if a is not None and b is not None:
                x,y = [w['value'] for w in a['windows']], [w['value'] for w in b['windows']]
                impacts[name] = parity(x,y) if len(x)==len(y) else dict(impact_values_within_tolerance=False, reason='window_count_mismatch')
        passed = identities['passed'] and all(v['passed'] for v in stages.values()) and all(v['impact_values_within_tolerance'] for v in impacts.values())
        checks.append(dict(pair_id=row['pair_id'], mode=row['mode'], impact_execution=row['impact_execution'],
            phase=row['phase'], passed=passed, status='PASS' if passed else 'POLICY_PARITY_FAILED',
            stages=stages, identities=identities, impact_errors=impacts,
            logit_allclose_to_full=row['correctness']['allclose'],
            request_delta_vs_legacy_ms=row['timing']['request_wall_ms']-reference['timing']['request_wall_ms'],
            impact_delta_vs_legacy_ms=(row['timing']['attention_impact_ms']-reference['timing']['attention_impact_ms']
                if row['mode'] != 'FULL_RECOMPUTE' else None)))
    return checks


def condition_checks(rows):
    """Also compare CPU/GPU and minimal/breakdown policy evidence, never timing."""
    references, checks = {}, []
    for row in rows:
        key=tuple(row[k] for k in ('phase','repeat','requested_prompt_tokens','condition','mode','impact_execution'))
        if key not in references:
            references[key]=row
            continue
        reference=references[key]
        result=compare_snapshots({k:reference[k] for k in ('source_policy','target_policy','token_ids_sha256')},
                                 {k:row[k] for k in ('source_policy','target_policy','token_ids_sha256')})
        checks.append(dict(pair_id=row['pair_id'],mode=row['mode'],
                           reference_pair_id=reference['pair_id'],**result))
    return checks


def stats(values):
    return dict(mean=mean(values), p50=percentile(values,.5), p95=percentile(values,.95), std=pstdev(values)) if values else dict.fromkeys(('mean','p50','p95','std'))


def summarize(rows, checks):
    groups = defaultdict(list)
    full = {r['pair_id']:r for r in rows if r['mode']=='FULL_RECOMPUTE'}
    legacy = {pair_key(r):r for r in rows if r['impact_execution']=='legacy_blockwise'}
    checked = {(c['pair_id'],c['mode']):c for c in checks}
    fields = ('storage','instrumentation','requested_prompt_tokens','condition','mode','impact_execution')
    for row in rows:
        if row['phase']=='measured':
            groups[tuple(row[k] for k in fields)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        output = dict(zip(fields,key), count=len(group), primary_latency=key[1]=='minimal', safe_reuse_claimed=False)
        for metric in ('request_wall_ms','attention_impact_ms'):
            values = [r['timing'][metric] for r in group if r['timing'].get(metric) is not None]
            output.update({metric+'_'+k:v for k,v in stats(values).items()})
            deltas = [r['timing'][metric]-legacy[pair_key(r)]['timing'][metric] for r in group
                if pair_key(r) in legacy and r['timing'].get(metric) is not None]
            output.update({metric+'_delta_vs_legacy_'+k:v for k,v in stats(deltas).items()})
        paired = [r for r in group if r['pair_id'] in full]
        output['observed_full_request_ratio'] = (mean(full[r['pair_id']]['timing']['request_wall_ms'] for r in paired)
            /mean(r['timing']['request_wall_ms'] for r in paired)) if paired else None
        output.update({"request_delta_vs_full_"+k:v for k,v in stats([
            r['timing']['request_wall_ms']-full[r['pair_id']]['timing']['request_wall_ms'] for r in paired]).items()})
        results = [checked.get((r['pair_id'],r['mode'])) for r in group]
        output['policy_parity'] = ('REFERENCE' if key[-1]=='legacy_blockwise' else
            'FAIL' if any(c and c['passed'] is False for c in results) else
            'PASS' if all(c and c['passed'] is True for c in results) else 'PENDING')
        errors = [e.get('impact_value_max_absolute_error',0.) for c in results if c
                  for e in c.get('impact_errors',{}).values()]
        output['impact_max_absolute_error'] = max(errors,default=0.) if results and all(results) else None
        output.update(mean_coverage=mean(r['coverage'] for r in group),
            mean_skipped_rows=mean(r['projection_audit']['skipped_projection_rows_all_layers_roles'] for r in group),
            logit_parity_passes=sum(r['correctness']['allclose'] for r in group),
            strict_nonzero_passes=sum(r['correctness']['nonzero_reuse_validated'] for r in group))
        output['validated_strict_reuse_ratio'] = (output['observed_full_request_ratio']
            if output['policy_parity']=='PASS' and output['strict_nonzero_passes']==len(group) else None)
        summaries.append(output)
    return summaries


class R1B1Extension:
    name = 'R1-B1'

    def initialize_manifest(self, manifest):
        manifest['policy_equivalent_optimization'] = False
        manifest['primary_latency'] = 'minimal/request_wall_ms'
        manifest['impact_formula'] = 'paper_row_l2_sum'
        manifest['reduction_dtype'] = 'torch.float64'

    def add_arguments(self, parser):
        parser.add_argument('--impact-executions', default='legacy_blockwise,token_precompute',
            help='Paired algorithms; legacy_blockwise plus token_precompute and/or token_prefix_sum')
        parser.set_defaults(prompt_lengths='32,64', dtype='float32')

    def validate(self, args, parser):
        variants = self.variants(args)
        if (len(variants)<2 or len(set(variants))!=len(variants)
                or 'legacy_blockwise' not in variants or any(v not in EXECUTIONS for v in variants)):
            parser.error('Choose unique legacy_blockwise plus token_precompute and/or token_prefix_sum')
        if args.impact_reducer != 'paper_row_l2_sum':
            parser.error('R1-B1 algorithms require paper_row_l2_sum')

    def variants(self, args):
        return args.impact_executions.split(',')

    def provenance_files(self, root):
        return [root/'scripts/70_run_r1b1_impact.py', root/'jobs/r1b1_impact.sbatch']

    def engine_options(self, variant):
        return dict(impact_execution=variant, capture_impact_values=True)

    snapshot = staticmethod(snapshot)

    def decorate(self, row, variant, engine, result):
        row['impact_execution'] = variant
        row['source_policy'] = row['source_build'].get('policy_snapshot') if row['source_build'] else None
        row['target_policy'] = snapshot(engine,result) if engine else None
        if engine:
            # Reuse the existing engine CPU timer; no extra reduction/events.
            row['timing']['attention_impact_ms'] = result['summary']['impact_wall_ms']
            row['timing_scopes']['attention_impact_ms'] = dict(parent='request_wall_ms',
                kind='inclusive', additive_to_parent=False, clock='cpu_wall', available=True)
        # Natural cross-user/shifted hits are never counted as safe successes.
        row['correctness']['nonzero_reuse_validated'] = bool(row['correctness']['allclose']
            and row['reused_tokens'] and row['condition']=='same_user_exact'
            and row['mode']=='POSITION_ALIGNED_DIAGNOSTIC')

    def write_reports(self, root, rows):
        # Write evidence before interpreting mismatches; no failed row is filtered.
        (root/'raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
        checks = pair_checks(rows)
        atomic_json(root/'policy_parity.json',checks)
        atomic_json(root/'storage_instrumentation_parity.json',condition_checks(rows))
        summary = summarize(rows,checks)
        fields = sorted({k for r in summary for k in r})
        with (root/'summary.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fields)
            writer.writeheader()
            writer.writerows(summary)
        atomic_json(root/'correctness.json',[dict(pair_id=r['pair_id'],mode=r['mode'],
            impact_execution=r['impact_execution'],phase=r['phase'],**r['correctness']) for r in rows])
        minimal = {(r['phase'],r['repeat'],r['storage'],r['requested_prompt_tokens'],r['condition'],r['mode'],r['impact_execution']):r
                   for r in rows if r['instrumentation']=='minimal'}
        overhead = []
        for r in rows:
            k=(r['phase'],r['repeat'],r['storage'],r['requested_prompt_tokens'],r['condition'],r['mode'],r['impact_execution'])
            if r['instrumentation']=='breakdown' and k in minimal:
                overhead.append(dict(pair_id=r['pair_id'],mode=r['mode'],impact_execution=r['impact_execution'],phase=r['phase'],
                    breakdown_minus_minimal_request_ms=r['timing']['request_wall_ms']-minimal[k]['timing']['request_wall_ms']))
        atomic_json(root/'instrumentation_overhead.json',overhead)
        lines=['# R1-B1 measured integration', '',
            'Minimal request latency is primary. Ratios are observed timings, not safety claims.',
            'Natural reuse logit mismatch is not a validated reuse success. Nested timings are not additive.', '',
            '|Storage|Pass|Length|Case|Mode|Impact algorithm|Request mean ms|Impact mean ms|FULL ratio|Policy|Logit passes/n|',
            '|---|---|---:|---|---|---|---:|---:|---:|---|---|']
        for r in summary:
            lines.append('|'+ '|'.join(str(r[k]) for k in ('storage','instrumentation','requested_prompt_tokens',
                'condition','mode','impact_execution','request_wall_ms_mean','attention_impact_ms_mean',
                'observed_full_request_ratio','policy_parity'))+f"|{r['logit_parity_passes']}/{r['count']}|")
        (root/'summary.md').write_text('\n'.join(lines)+'\n')

    def finalize(self, args, manifest, rows):
        checks = pair_checks(rows)
        failures = sum(c['passed'] is False for c in checks)
        condition_failures=sum(not c['passed'] for c in condition_checks(rows))
        pending = sum(c['passed'] is None for c in checks)
        strict = {v:sum(r['phase']=='measured' and r['impact_execution']==v
            and r['correctness']['nonzero_reuse_validated'] for r in rows) for v in self.variants(args)}
        manifest.update(policy_parity_failures=failures, storage_instrumentation_parity_failures=condition_failures,
            unpaired_algorithm_rows=pending,
            strict_validated_cases_by_algorithm=strict, impact_tolerance=dict(atol=ATOL,rtol=RTOL),
            pbr_exercised=any(r['target_policy'] and r['target_policy']['pbr_updates'] for r in rows),
            eviction_exercised=any(r['cache_event_counts']['EVICT'] for r in rows),
            policy_equivalent_optimization=False)
        if failures or condition_failures:
            manifest['status']='POLICY_PARITY_FAILED'
        elif pending:
            manifest['status']='INCOMPLETE_PAIRING'
        elif not all(strict.values()) and manifest['status']=='COMPLETE':
            manifest['status']='NO_VALIDATED_REUSE'
        elif manifest['status']=='COMPLETE':
            manifest['policy_equivalent_optimization']=True
        manifest['validation_failures'] = [name for name, failed in (
            ('policy_parity', failures), ('storage_instrumentation_parity', condition_failures),
            ('incomplete_pairing', pending), ('strict_logits', manifest.get('strict_parity_failures',0)),
            ('missing_nonzero_strict_reuse', not all(strict.values()))) if failed]
        self.write_reports(args.output_dir,rows)
