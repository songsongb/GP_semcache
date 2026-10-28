"""Four-path ablation statistics with independent sessions as the resampling unit."""
from collections import defaultdict
import math
import random
import statistics
from semcache.metrics.r1b2 import stats
from semcache.diagnostics.r2b35_executor import MODES

DIFFERENCES=('delta_legacy_ms','delta_matched_ms','delta_difference_ms','full_head_policy_effect_ms','warm_head_policy_effect_ms')


def summarize(rows):
    groups=defaultdict(list); seen=set()
    for r in rows:
        if r['phase']!='measured': continue
        if r['instrumentation']!='minimal' or set(r['wall_ms'])!=set(MODES): raise ValueError('not a complete minimal four-path sample')
        if any(not math.isfinite(v) or v<=0 for v in r['wall_ms'].values()): raise ValueError('invalid latency')
        from semcache.diagnostics.r2b35_benchmark import comparisons
        computed=comparisons({m:dict(wall_ms=v) for m,v in r['wall_ms'].items()},r['P'])
        if any(not math.isclose(r[k],computed[k],abs_tol=1e-9) for k in DIFFERENCES): raise ValueError('paired sign/ablation mismatch')
        key=(r['independent_run_id'],r['case_id'],r['attention_arm']); pair=(*key,r['pair_index'])
        if pair in seen: raise ValueError('duplicate pair')
        seen.add(pair); groups[key].append(r)
    result=[]
    for (run,case,arm),block in groups.items():
        result.append(dict(independent_run_id=run,case_id=case,attention_arm=arm,count=len(block),
            walls={m:stats([r['wall_ms'][m] for r in block]) for m in MODES},
            differences={k:stats([r[k] for r in block]) for k in DIFFERENCES},
            matched_wins=sum(r['delta_matched_ms']>0 for r in block),matched_win_rate=sum(r['delta_matched_ms']>0 for r in block)/len(block),
            matched_speedup=statistics.mean(r['wall_ms']['MATCHED_FULL'] for r in block)/statistics.mean(r['wall_ms']['MATCHED_WARM_NORMAL'] for r in block),
            native_endpoint_equivalent=all(r['native_endpoint_equivalent'] for r in block),
            full_warm_strict=all(r['full_warm_strict'] for r in block)))
    return result


def performance_label(means,ci,valid,equivalent,verified):
    if not means or not valid or not equivalent: return 'MATCHED_RESULT_UNRESOLVED'
    if statistics.mean(means)<=0: return 'MATCHED_NO_OBSERVED_SPEEDUP'
    if len(means)>=3 and all(x>0 for x in means) and ci and ci[0]>0 and verified: return 'MATCHED_REPRODUCED_POSITIVE_DIFFERENCE'
    return 'MATCHED_POSITIVE_MEAN_ONLY'


def aggregate(sessions,seed=3501,draws=2000):
    if not sessions: raise ValueError('no sessions')
    env=None; hardware=None; runs=set(); processes=set(); conditions=None; groups=defaultdict(list)
    for m,rows in sessions:
        if m['experiment']!='R2-B3.5' or m['execution_status']!='COMPLETE' or m['benchmark_config']['phase']!='latency': raise ValueError('only COMPLETE B3.5 latency sessions')
        run=m['benchmark_config']['independent_run_id']
        if run in runs or m['process_instance'] in processes: raise ValueError('duplicate session/process')
        runs.add(run); processes.add(m['process_instance'])
        if env is None: env=m['environment_identity']; hardware=m['hardware_identity']; conditions=set(m['conditions'])
        elif env!=m['environment_identity'] or hardware!=m['hardware_identity'] or conditions!=set(m['conditions']): raise ValueError('incompatible GPU/model/adapter/code/endpoint/cases')
        if not rows or any(r['benchmark_phase']!='latency' or r['independent_run_id']!=run for r in rows): raise ValueError('wrong phase/run identity')
        summaries=summarize(rows)
        if {s['case_id']+'_'+s['attention_arm'] for s in summaries}!=conditions: raise ValueError('missing/unlisted condition')
        for s in summaries:
            key=s['case_id']+'_'+s['attention_arm']; c=m['conditions'][key]
            block=[r for r in rows if r['case_id']+'_'+r['attention_arm']==key]
            measured=[r for r in block if r['phase']=='measured']
            if c['status']!='VALID' or sorted(r['pair_index'] for r in measured)!=list(range(m['benchmark_config']['measured_runs'])) or len(measured)<20: raise ValueError('invalid/incomplete case')
            if any(r['condition_identity']!=c['condition_identity'] or not r['structural_valid'] or not r['cache_integrity_valid'] or r['output_matching']!='OUTPUT_MATCHED_VALID' or
                (r['L'],r['P'],r['S'])!=(c['L'],c['P'],c['S']) or r['model']!=m['benchmark_config']['model'] or r['dtype']!=m['benchmark_config']['dtype'] for r in block): raise ValueError('raw condition/namespace mismatch')
            if any(r['endpoints']!=['legacy_all_logits','matched_last_hidden_head'] or r['backend']!='eager' or
                r['namespace']!=c['namespace'] or r['source_snapshot_id']!=c['source_snapshot_id'] for r in block):
                raise ValueError('raw adapter/source/backend/endpoint mismatch')
            groups[key].append((s,c,m))
    rng=random.Random(seed); result=[]
    for key,group in sorted(groups.items()):
        if len({c['condition_identity'] for _,c,_ in group})!=1: raise ValueError('source/token/adapter/endpoint mismatch across sessions')
        means={k:[s['differences'][k]['mean'] for s,_,_ in group] for k in DIFFERENCES}; n=len(group)
        ci={}
        for k,values in means.items():
            b=sorted(statistics.mean(rng.choices(values,k=n)) for _ in range(draws)) if n>=3 else []
            ci[k]=[b[int(.025*(draws-1))],b[int(.975*(draws-1))]] if b else None
        equivalent=all(s['native_endpoint_equivalent'] and c['native_endpoint_equivalent'] for s,c,_ in group)
        strict=all(s['full_warm_strict'] and c['full_warm_strict'] for s,c,_ in group)
        verified=all(m['environment_verified'] for _,_,m in group)
        label=performance_label(means['delta_matched_ms'],ci['delta_matched_ms'],True,equivalent,verified)
        c=group[0][1]
        result.append(dict(case_id=key,L=c['L'],P=c['P'],S=c['S'],attention_arm=c['attention_arm'],sessions=n,
            output_matching='OUTPUT_MATCHED_VALID',native_endpoint_equivalence='NATIVE_ENDPOINT_EQUIVALENCE_PASS' if equivalent else 'NATIVE_ENDPOINT_EQUIVALENCE_FAIL',
            numerical_correctness='FULL_WARM_STRICT_PASS' if strict else 'FULL_WARM_STRICT_FAIL',performance=label,
            correctness_qualified_speedup=strict and equivalent and label=='MATCHED_REPRODUCED_POSITIVE_DIFFERENCE',
            equal_session_differences={k:stats(v) for k,v in means.items()},session_bootstrap_95_ci=ci,
            between_session_variance=statistics.variance(means['delta_matched_ms']) if n>1 else None,
            matched_speedup=statistics.mean(s['walls']['MATCHED_FULL']['mean'] for s,_,_ in group)/statistics.mean(s['walls']['MATCHED_WARM_NORMAL']['mean'] for s,_,_ in group),
            session_results=[s for s,_,_ in group],bootstrap_seed=seed,bootstrap_draws=draws,
            resampling_unit='independent process session mean; equal session weights; exploratory with three sessions',
            inference='same-session output-policy sensitivity, not exact causal head-cost decomposition; no amortized service claim'))
    return result
