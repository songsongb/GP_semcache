"""Scalar R1-B2 contracts, diagnostic coverage, and conservative aggregation.

No model imports at collection. Never changes production selection or policies.
"""
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter_ns
from unittest.mock import patch

from .r1 import atomic_json, digest, percentile, timing_scopes

MODES = ('FULL_RECOMPUTE', 'LOOKUP_NO_REUSE', 'NATURAL_RAW_REUSE', 'POSITION_ALIGNED_REUSE')
GROUP = ('comparison_identity', 'length', 'storage', 'instrumentation', 'mode', 'requested_coverage')


def coverage_value(value):
    if str(value) == 'max':
        return 'max'
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError('Coverage must be a fraction in [0,1] or max')
    return number


def filter_coverage(selected, sequence_length, requested):
    """Earliest aligned nonoverlapping windows, capped at floor(n*f/3)."""
    requested = coverage_value(requested)
    ordered = sorted(selected, key=lambda h: h.window.start)
    used = set()
    for hit in ordered:
        w, entry = hit.window, hit.entry
        positions = set(range(w.start, w.end))
        if (w.end-w.start != 3 or not 0 <= w.start < w.end <= sequence_length
                or positions & used or tuple(w.token_ids) != tuple(entry.token_ids)
                or tuple(entry.positions) != (w.start, w.end)):
            raise ValueError('Invalid position-aligned exact-w3 candidate')
        used.update(positions)
    count = len(ordered) if requested == 'max' else min(len(ordered), math.floor(sequence_length*requested/3))
    chosen = ordered[:count]
    mask = [False]*sequence_length
    for hit in chosen:
        mask[hit.window.start:hit.window.end] = [True]*3
    return chosen, mask


@contextmanager
def coverage_override(requested):
    """Single-threaded runner-only binding; original aligned selector still runs."""
    import semcache.semcache_engine as engine_module
    original = engine_module.select_position_aligned_diagnostic
    evidence = dict(filter_host_ms=None, candidate_windows=None)
    def select(*args, **kwargs):
        candidates, _ = original(*args, **kwargs)
        start = perf_counter_ns()
        chosen, mask = filter_coverage(candidates, args[1], requested)
        evidence.update(filter_host_ms=(perf_counter_ns()-start)/1e6,
                        candidate_windows=len(candidates))
        return chosen, mask
    with patch.object(engine_module, 'select_position_aligned_diagnostic', select):
        yield evidence


def correctness_status(reused, allclose, aligned, adapter_match, context_match):
    if not reused:
        return 'NO_REUSE'
    if allclose is None:
        return 'UNVERIFIED'
    return ('STRICT_CORRECTNESS_PASS' if allclose and aligned and adapter_match and context_match
            else 'STRICT_CORRECTNESS_FAIL')


def stats(values):
    return dict(mean=mean(values), p50=percentile(values,.5), p95=percentile(values,.95), std=pstdev(values)) if values else dict.fromkeys(('mean','p50','p95','std'))


def breakdown(timing, mode, instrumentation):
    """GPU residual includes launch gaps; never a sum of CPU + CUDA clocks."""
    t = dict(timing)
    mixed = mode != 'FULL_RECOMPUTE'
    enclosing = t.get('prefill_gpu_ms')
    child = t.get('mixed_executor_gpu_ms' if mixed else 'native_qkv_gpu_ms')
    t['remaining_transformer_gpu_ms'] = enclosing-child if enclosing is not None and child is not None else None
    names = ('tokenization_ms','semantic_encode_ms','cluster_assign_update_ms','subsequence_extract_ms',
             'cache_lookup_ms','hit_selection_ms','attention_impact_ms','chu_update_ms','pbr_update_ms',
             'cache_policy_ms','cache_materialization_ms','request_wall_ms','prefill_wall_ms','prefill_gpu_ms',
             'cache_fetch_host_ms','cache_fetch_gpu_ms','index_transfer_host_ms','index_transfer_gpu_ms',
             'tensor_assembly_host_ms','tensor_assembly_gpu_ms','native_qkv_host_ms','native_qkv_gpu_ms',
             'mixed_executor_host_ms','mixed_executor_gpu_ms','remaining_transformer_gpu_ms')
    scopes = timing_scopes(t)
    result = []
    for name in names:
        value = t.get(name)
        # Existing helper exports synthetic zero for absent mixed/native regions.
        if mode == 'FULL_RECOMPUTE' and name.startswith(('mixed_executor','cache_fetch','index_transfer','tensor_assembly')):
            value = None
        scope = scopes.get(name,{})
        if mode == 'FULL_RECOMPUTE' and name.startswith('native_qkv'):
            scope = dict(scope, parent='prefill_gpu_ms' if name.endswith('gpu_ms') else 'prefill_wall_ms')
        derived = name == 'remaining_transformer_gpu_ms'
        result.append(dict(metric=name, value_ms=value,
            clock='cuda_stream_event' if name.endswith('gpu_ms') else 'cpu_wall',
            status='UNAVAILABLE' if value is None else 'INVALID_NEGATIVE_DERIVED' if derived and value<0 else 'DERIVED' if derived else 'MEASURED',
            parent='prefill_gpu_ms' if derived else scope.get('parent'), additive_to_parent=False,
            definition=('prefill GPU minus disjoint mixed executor GPU (native QKV for FULL); includes stream idle/launch gaps'
                        if derived else 'existing R1 interval; see timing contract')))
    return result


def summarize(rows):
    measured = [r for r in rows if r['phase']=='measured']
    full = {}
    failed_comparisons = set()
    for row in rows:
        check = row.get('comparison_parity', {})
        if check.get('status') == 'FAIL':
            failed_comparisons.update((row['row_id'], check.get('reference')))
    seen = set()
    for row in measured:
        identity = tuple(str(row[k]) for k in (*GROUP, 'repeat'))
        if identity in seen:
            raise ValueError('Duplicate measured row: '+str(identity))
        seen.add(identity)
        if row['mode']=='FULL_RECOMPUTE':
            full[row['pair_id']] = row
    groups = defaultdict(list)
    for row in measured:
        groups[tuple(str(row[k]) for k in GROUP)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        pairs = [(r,full[r['pair_id']]) for r in group if r['pair_id'] in full]
        for r,f in pairs:
            for field in ('comparison_identity','length','storage','instrumentation','token_ids_sha256','adapter'):
                if r[field] != f[field]:
                    raise ValueError('FULL pairing identity mismatch: '+field)
        reuse = [r['request_wall_ms'] for r,f in pairs]
        baseline = [f['request_wall_ms'] for r,f in pairs]
        out = dict(zip(GROUP,key), count=len(group), paired_count=len(pairs),
            measurement_profile=group[0].get('measurement_profile','UNSPECIFIED'))
        for prefix, values in [('full',baseline),('request',reuse),('paired_delta',[a-b for a,b in zip(reuse,baseline)])]:
            out.update({prefix+'_'+k+'_ms':v for k,v in stats(values).items()})
        speedup = mean(baseline)/mean(reuse) if reuse else None
        eligible = (key[3]=='minimal' and len(pairs)==len(group) and key[4] in MODES[2:]
                    and all(r['reused_tokens']>0 for r in group))
        observed = bool(eligible and speedup is not None and speedup>1)
        qualified = observed and all(r['correctness']['status']=='STRICT_CORRECTNESS_PASS'
            and f['correctness']['logit_allclose'] and r['row_id'] not in failed_comparisons
            and f['row_id'] not in failed_comparisons for r,f in pairs)
        out.update(speedup=speedup, observed_break_even=observed,
            correctness_qualified_break_even=qualified, primary_latency=key[3]=='minimal',
            correctness_status=('STRICT_CORRECTNESS_FAIL' if any(r['correctness']['status']=='STRICT_CORRECTNESS_FAIL' for r in group)
                else group[0]['correctness']['status'] if len({r['correctness']['status'] for r in group})==1 else 'UNVERIFIED'),
            actual_coverage_mean=mean(r['actual_coverage'] for r in group),
            reused_tokens_mean=mean(r['reused_tokens'] for r in group),
            native_projection_rows_mean=mean(r['native_projection_rows'] for r in group),
            skipped_projection_rows_mean=mean(r['skipped_projection_rows'] for r in group),
            source_build_mean_ms=mean([r['source_build_ms'] for r in group if r['source_build_ms'] is not None]) if any(r['source_build_ms'] is not None for r in group) else None,
            source_build_in_warm_request=False)
        summaries.append(out)
    return summaries


def comparison_audits(rows):
    """Compact independent-run equality proof; differing hashes are UNVERIFIED.

    Runtime paired allclose/policy-tolerance checks are recorded separately.
    Exact hashes prove equality; inequality cannot prove tolerance failure.
    """
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(str(r[k]) for k in ('comparison_identity','phase','repeat','length','mode','requested_coverage'))].append(r)
    checks=[]
    for group in grouped.values():
        for r in group[1:]:
            a=group[0]
            fields=('source_signature','selection_signature','native_projection_rows','skipped_projection_rows','policy_signature','source_policy_signature','logits_sha256')
            checks.append(dict(reference=a['row_id'],candidate=r['row_id'],
                status='EXACT_PASS' if all(a[k]==r[k] for k in fields) else 'UNVERIFIED_HASH_DIFFERENCE',
                differing_fields=[k for k in fields if a[k]!=r[k]]))
    return checks


def overhead(rows):
    key=lambda r: tuple(str(r[k]) for k in ('comparison_identity','phase','repeat','length','storage','mode','requested_coverage'))
    minimal={key(r):r for r in rows if r['instrumentation']=='minimal' and r['phase']=='measured'}
    return [dict(row_id=r['row_id'],reference_row_id=minimal[key(r)]['row_id'],
        breakdown_minus_minimal_ms=r['request_wall_ms']-minimal[key(r)]['request_wall_ms'])
        for r in rows if r['instrumentation']=='breakdown' and r['phase']=='measured' and key(r) in minimal]


def csv_write(path, rows):
    fields=sorted({k for r in rows for k in r}) or ['status']
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def reports(root, rows, aggregate=False):
    root=Path(root)
    summary=summarize(rows)
    csv_write(root/('overall_summary.csv' if aggregate else 'break_even_summary.csv'),summary)
    detail=[dict(row_id=r['row_id'],phase=r['phase'],length=r['length'],storage=r['storage'],mode=r['mode'],
        requested_coverage=r['requested_coverage'],instrumentation=r['instrumentation'],**s)
        for r in rows for s in r['breakdown']]
    csv_write(root/'latency_breakdown.csv',detail)
    atomic_json(root/'correctness.json',[dict(row_id=r['row_id'],**r['correctness']) for r in rows])
    atomic_json(root/'coverage_audit.json',dict(rows=[{k:r[k] for k in ('row_id','requested_coverage','actual_coverage',
        'reused_tokens','recomputed_tokens','selected_windows','native_projection_rows','skipped_projection_rows',
        'selection_signature','source_signature','coverage_filter_host_ms','spans')} for r in rows],
        storage_instrumentation_checks=comparison_audits(rows)))
    atomic_json(root/'instrumentation_overhead.json',dict(pairs=overhead(rows),
        status='PAIRED' if overhead(rows) else 'UNPAIRED',
        note='Includes detailed executor, event, prefill synchronization overhead; not pure profiler cost'))
    if aggregate:
        csv_write(root/'break_even_matrix.csv',summary)
        # Rank only disjoint top-level CPU regions, separately from nested GPU costs.
        ranked=[]
        for s in summary:
            group=[r for r in rows if r['phase']=='measured' and all(str(r[k])==s[k] for k in GROUP)]
            if s['instrumentation']!='breakdown': continue
            for clock, metrics in [('cpu_wall',('tokenization_ms','semantic_encode_ms','cluster_assign_update_ms',
                'subsequence_extract_ms','cache_lookup_ms','hit_selection_ms','prefill_wall_ms',
                'attention_impact_ms','cache_policy_ms','chu_update_ms','pbr_update_ms')),
                ('cuda_stream_event',('cache_fetch_gpu_ms','index_transfer_gpu_ms','tensor_assembly_gpu_ms','native_qkv_gpu_ms','remaining_transformer_gpu_ms'))]:
                entries=[]
                for metric in metrics:
                    values=[x['value_ms'] for r in group for x in r['breakdown'] if x['metric']==metric and x['value_ms'] is not None and x['status']!='INVALID_NEGATIVE_DERIVED']
                    if values: entries.append(dict(metric=metric,mean_ms=mean(values)))
                for rank,item in enumerate(sorted(entries,key=lambda x:-x['mean_ms']),1):
                    ranked.append(dict(**{k:s[k] for k in GROUP},clock=clock,rank=rank,**item))
        csv_write(root/'bottleneck_ranking.csv',ranked)
    atomic_json(root/'break_even_status.json',dict(
        observed='OBSERVED' if any(s['observed_break_even'] for s in summary) else 'NOT_OBSERVED_IN_TESTED_RANGE' if any(s['primary_latency'] and s['mode'] in MODES[2:] for s in summary) else 'UNVERIFIED',
        correctness_qualified='OBSERVED' if any(s['correctness_qualified_break_even'] for s in summary) else 'NOT_OBSERVED_IN_TESTED_RANGE' if any(s['primary_latency'] and s['mode'] in MODES[2:] for s in summary) else 'UNVERIFIED',
        measured_lengths=sorted({r['length'] for r in rows if r['phase']=='measured'}),
        extrapolation=False))


def load_checkpoints(root, identity):
    rows=[]
    for p in sorted((Path(root)/'checkpoints').glob('*.json')):
        bundle=json.loads(p.read_text())
        if bundle['identity']!=identity or bundle['status']!='COMPLETE':
            raise ValueError('Invalid/incomplete checkpoint '+str(p))
        if digest(bundle['rows'])!=bundle['rows_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        pairs=defaultdict(list)
        for row in bundle['rows']: pairs[row['pair_id']].append(row)
        for group in pairs.values():
            if sum(r['mode']=='FULL_RECOMPUTE' for r in group)!=1:
                raise ValueError('Checkpoint group must contain exactly one FULL baseline')
        rows.extend(bundle['rows'])
    summarize(rows)  # reject duplicates, mismatched FULL pairing
    return rows
