"""Scalar paired session statistics, without mixing prefix-KV or profiler evidence."""
from collections import defaultdict
import math
import random
import statistics
from semcache.metrics.r1b2 import stats
from semcache.diagnostics.r2c1_workloads import MODES

DELTAS=('delta_full_raw_ms','delta_miss_raw_ms','delta_full_miss_ms')


def summarize(rows):
    groups=defaultdict(list); seen=set()
    for r in rows:
        if r['phase']!='measured': continue
        if r['instrumentation']!='minimal' or set(r['wall_ms'])!=set(MODES): raise ValueError('not a minimal three-mode pair')
        if any(v<=0 or not math.isfinite(v) for v in r['wall_ms'].values()): raise ValueError('invalid latency')
        f,m,q=[r['wall_ms'][k] for k in MODES]
        if any(not math.isclose(r[k],v,abs_tol=1e-9) for k,v in zip(DELTAS,(f-q,m-q,f-m))): raise ValueError('paired sign mismatch')
        key=(r['independent_run_id'],r['case_id']); pair=(*key,r['pair_index'])
        if pair in seen: raise ValueError('duplicate sample')
        seen.add(pair); groups[key].append(r)
    output=[]
    for (run,case),block in groups.items():
        output.append(dict(independent_run_id=run,case_id=case,count=len(block),length=block[0]['length'],requested_coverage=block[0]['requested_coverage'],
            walls={m:stats([r['wall_ms'][m] for r in block]) for m in MODES},
            differences={k:stats([r[k] for r in block]) for k in DELTAS},
            full_raw_speedup=statistics.mean(r['wall_ms']['FULL_NATIVE'] for r in block)/statistics.mean(r['wall_ms']['RAW_SEMCACHE_QKV'] for r in block),
            wins=sum(r['delta_full_raw_ms']>0 for r in block),win_rate=sum(r['delta_full_raw_ms']>0 for r in block)/len(block),
            actual_reused_tokens=sorted({r['coverage']['reused_tokens'] for r in block}),
            strict_pass=all(r['strict_pass'] for r in block),outliers_removed=False))
    return output


def aggregate(sessions,seed=4101,draws=2000):
    if not sessions: raise ValueError('no sessions')
    runs=set(); processes=set(); environment=None; hardware=None; expected=None; grouped=defaultdict(list)
    for manifest,rows in sessions:
        config=manifest['benchmark_config']; run=config['independent_run_id']
        if manifest['experiment']!='R2-C1' or config['phase']!='latency' or manifest['execution_status']!='COMPLETE': raise ValueError('only COMPLETE C1 latency runs')
        if run in runs or manifest['process_instance'] in processes: raise ValueError('duplicate independent process/run')
        runs.add(run); processes.add(manifest['process_instance'])
        if environment is None: environment=manifest['environment_identity']; hardware=manifest['hardware_identity']; expected=set(manifest['conditions'])
        elif hardware!=manifest['hardware_identity']: raise ValueError('HARDWARE_MISMATCH: separate physical GPU sessions')
        elif environment!=manifest['environment_identity'] or expected!=set(manifest['conditions']): raise ValueError('MODEL_CODE_ENDPOINT_OR_CASE_MISMATCH')
        if not rows or any(r['benchmark_phase']!='latency' or r['independent_run_id']!=run for r in rows): raise ValueError('phase/run mixing')
        summaries=summarize(rows)
        if {r['case_id'] for r in summaries}!=expected: raise ValueError('missing conditions')
        for s in summaries:
            c=manifest['conditions'][s['case_id']]; block=[r for r in rows if r['case_id']==s['case_id']]
            measured=[r for r in block if r['phase']=='measured']
            if c['status']!='VALID' or sorted(r['pair_index'] for r in measured)!=list(range(config['measured_runs'])) or len(measured)<20: raise ValueError('invalid/incomplete case')
            if any(r['condition_identity']!=c['condition_identity'] or not r['structural_valid'] or not r['cache_integrity_valid'] or r['length']!=c['length'] or r['requested_coverage']!=c['requested_coverage'] or
                r['model']!=config['model'] or r['dtype']!=config['dtype'] or r['endpoint_policy']!='all_token_logits_use_cache_false_attention_true' for r in block): raise ValueError('condition/output/gate mismatch')
            if len(s['actual_reused_tokens'])!=1: raise ValueError('coverage changed within session')
            grouped[s['case_id']].append((s,c,manifest))
    rng=random.Random(seed); results=[]
    for case,group in sorted(grouped.items()):
        if len({c['condition_identity'] for _,c,_ in group})!=1 or len({tuple(s['actual_reused_tokens']) for s,_,_ in group})!=1: raise ValueError('cross-session source/selection/coverage mismatch')
        means=[s['differences']['delta_full_raw_ms']['mean'] for s,_,_ in group]; n=len(means)
        bootstrap=sorted(statistics.mean(rng.choices(means,k=n)) for _ in range(draws)) if n>=3 else []
        ci=[bootstrap[int(.025*(draws-1))],bootstrap[int(.975*(draws-1))]] if bootstrap else None
        verified=all(m['environment_verified'] for _,_,m in group); reused=group[0][0]['actual_reused_tokens'][0]
        strict=all(s['strict_pass'] and c['strict_pass'] for s,c,_ in group)
        label=('NO_REUSE_CONTROL' if not reused else 'NO_OBSERVED_BREAK_EVEN' if statistics.mean(means)<=0 else
            'REPRODUCED_POSITIVE_DIFFERENCE' if n>=3 and all(v>0 for v in means) and ci and ci[0]>0 and verified else 'POSITIVE_MEAN_ONLY')
        results.append(dict(case_id=case,length=group[0][1]['length'],requested_coverage=group[0][1]['requested_coverage'],
            reused_tokens=reused,actual_coverage=reused/group[0][1]['length'],sessions=n,
            paired_difference=stats(means),session_bootstrap_95_ci=ci,between_session_variance=statistics.variance(means) if n>1 else None,
            full_raw_speedup=statistics.mean(s['walls']['FULL_NATIVE']['mean'] for s,_,_ in group)/statistics.mean(s['walls']['RAW_SEMCACHE_QKV']['mean'] for s,_,_ in group),
            conclusion=label,strict_status='STRICT_PASS' if strict else 'STRICT_FAIL',
            environment_verified=verified,
            correctness_qualified_speedup=strict and label=='REPRODUCED_POSITIVE_DIFFERENCE',
            classification='ORACLE_DIAGNOSTIC',production_safe_reuse_claimed=False,session_results=[s for s,_,_ in group],
            bootstrap_seed=seed,bootstrap_draws=draws,resampling_unit='independent process session means, equal weights; exploratory with three runs',
            scope='conditional R1 request boundary; Engine cloning/harness validation excluded; no service or Prefix-KV speedup claim'))
    return results


def hypotheses(results):
    usable=[r for r in results if r['reused_tokens']]
    reproduced=[r['case_id'] for r in usable if r['conclusion']=='REPRODUCED_POSITIVE_DIFFERENCE']
    positive=[r['case_id'] for r in usable if r['paired_difference']['mean']>0]
    growth=[]; coverage_gains=[]
    def paired_session_change(a,b,mode):
        left={s['independent_run_id']:s['walls'][mode]['mean'] for s in a['session_results']}
        right={s['independent_run_id']:s['walls'][mode]['mean'] for s in b['session_results']}
        return [right[k]-left[k] for k in sorted(left)] if set(left)==set(right) else []
    for coverage in {str(r['requested_coverage']) for r in results}:
        group=sorted([r for r in results if str(r['requested_coverage'])==coverage],key=lambda r:r['length'])
        for a,b in zip(group,group[1:]):
            values=paired_session_change(a,b,'FULL_NATIVE')
            growth.append(dict(short=a['case_id'],long=b['case_id'],session_long_minus_short_ms=values,
                verified=a['environment_verified'] and b['environment_verified']))
    for length in {r['length'] for r in results}:
        group=sorted([r for r in results if r['length']==length],key=lambda r:r['reused_tokens'])
        for a,b in zip(group,group[1:]):
            if b['reused_tokens']==a['reused_tokens']: continue
            values=[-x for x in paired_session_change(a,b,'RAW_SEMCACHE_QKV')]
            coverage_gains.append(dict(low=a['case_id'],high=b['case_id'],session_low_minus_high_raw_ms=values,
                verified=a['environment_verified'] and b['environment_verified']))
    def direction(evidence,key):
        vectors=[r[key] for r in evidence]
        if not vectors or any(len(v)<3 for v in vectors) or not all(r['verified'] for r in evidence): return 'UNRESOLVED'
        if all(all(v>0 for v in x) for x in vectors): return 'SUPPORTED'
        if all(all(v<=0 for v in x) for x in vectors): return 'CONTRADICTED'
        return 'UNRESOLVED'
    return dict(H1=dict(status=direction(growth,'session_long_minus_short_ms'),observations=growth,
            reason='observed Full scaling only; compute/attention/head/dispatch attribution requires separate trace'),
        H2=dict(status=direction(coverage_gains,'session_low_minus_high_raw_ms'),observations=coverage_gains,
            reason='net request association with actual reuse; not a QKV GEMM-only causal saving estimate'),
        H3=dict(status='SUPPORTED' if reproduced else 'CONTRADICTED' if usable and not positive else 'UNRESOLVED',
            result='REPRODUCED_MEASURED_REGION_CANDIDATE' if reproduced else 'POSITIVE_MEAN_ONLY' if positive else 'NO_OBSERVED_BREAK_EVEN' if usable else 'UNVERIFIED',
            reproduced_cases=reproduced,positive_mean_cases=positive,interpretation='only tested C1 conditions; no threshold interpolation/extrapolation'),
        historical_prefix_kv_reference=dict(scope='historical only, not pooled',comparable_paired_speedup=False,
            reason='R2-B3.5 uses use_cache=True, attention OFF, one-token head and full-prefix decoder skip; C1 skips QKV rows only'))
