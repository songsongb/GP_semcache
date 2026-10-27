"""Session-level statistics. Pairs are nested within independent processes."""
from collections import defaultdict
import math
import random
import statistics
from semcache.metrics.r1b2 import stats


def summarize(rows):
    groups=defaultdict(list); seen=set()
    for r in rows:
        if r['phase']!='measured': continue
        if r['instrumentation']!='minimal': raise ValueError('instrumented timing cannot enter latency summary')
        key=(r['independent_run_id'],r['case_id'],r['attention_arm'],r['endpoint_policy'])
        sample=(*key,r['pair_index'])
        if sample in seen: raise ValueError('duplicate pair')
        seen.add(sample)
        for field in ('full_wall_ms','normal_warm_wall_ms'):
            if not math.isfinite(r[field]) or r[field]<=0: raise ValueError('invalid request latency')
        if not math.isclose(r['full_minus_normal_ms'],r['full_wall_ms']-r['normal_warm_wall_ms'],abs_tol=1e-9):
            raise ValueError('paired sign or latency mismatch')
        groups[key].append(r)
    output=[]
    for key,block in groups.items():
        f=[r['full_wall_ms'] for r in block]; n=[r['normal_warm_wall_ms'] for r in block]
        d=[r['full_minus_normal_ms'] for r in block]
        output.append(dict(independent_run_id=key[0],case_id=key[1],attention_arm=key[2],endpoint_policy=key[3],
            count=len(block),full=stats(f),normal=stats(n),paired_difference=stats(d),
            cache_hit=all(r.get('cache_hit',False) for r in block),
            within_session_variance=statistics.variance(d) if len(d)>1 else None,
            full_normal_ratio=statistics.mean(f)/statistics.mean(n),wins=sum(x>0 for x in d),win_rate=sum(x>0 for x in d)/len(d),
            all_strict_pass=all(r['correctness_status']=='STRICT_PASS' for r in block),
            correctness_qualified_speedup=False,performance_status='SINGLE_SESSION_DESCRIPTIVE',
            prepared_classification='IDEAL_DIAGNOSTIC_ONLY',
            prepared=stats([r['prepared_wall_ms'] for r in block if r['prepared_wall_ms'] is not None]),
            normal_minus_prepared=stats([r['normal_minus_prepared_ms'] for r in block if r['normal_minus_prepared_ms'] is not None])))
    return output


def aggregate(sessions,seed=901,draws=2000):
    """Bootstrap entire session means, equally weighted; not pooled pair resampling."""
    if not sessions: raise ValueError('no sessions')
    run_ids=set(); processes=set(); groups=defaultdict(list); expected=None; identity=None
    qualification_eligible=all(m.get('qualification_eligible',False) for m,_ in sessions)
    for manifest,rows in sessions:
        if manifest['execution_status']!='COMPLETE' or manifest['config']['phase']!='latency':
            raise ValueError('only complete minimal latency sessions can aggregate')
        if not rows: raise ValueError('empty latency evidence')
        run=manifest['config']['independent_run_id']; process=manifest['process_instance']
        if run in run_ids or process in processes: raise ValueError('duplicate run/process; independent processes required')
        run_ids.add(run); processes.add(process)
        if identity is None: identity=manifest['comparison_identity']
        elif identity!=manifest['comparison_identity']: raise ValueError('incompatible model/code/hardware/workload/endpoint identity')
        summary=summarize(rows); keys={(s['case_id'],s['attention_arm'],s['endpoint_policy']) for s in summary}
        for s in summary:
            block=[r for r in rows if r['phase']=='measured' and (r['case_id'],r['attention_arm'],r['endpoint_policy'])==
                (s['case_id'],s['attention_arm'],s['endpoint_policy'])]
            if sorted(r['pair_index'] for r in block)!=list(range(s['count'])):
                raise ValueError('missing pair indices')
            if s['count']!=manifest['config']['measured_runs']: raise ValueError('manifest/sample repeat count differs')
        if expected is None: expected=keys
        elif expected!=keys: raise ValueError('missing condition in independent session')
        for s in summary:
            if s['independent_run_id']!=run or s['count']<20: raise ValueError('run identity/repeat protocol mismatch')
            groups[(s['case_id'],s['attention_arm'],s['endpoint_policy'])].append(s)
    rng=random.Random(seed); output=[]
    for key,block in groups.items():
        means=[s['paired_difference']['mean'] for s in block]; k=len(means)
        boot=sorted(statistics.mean(rng.choices(means,k=k)) for _ in range(draws)) if k>=3 else []
        ci=[boot[int(.025*(draws-1))],boot[int(.975*(draws-1))]] if boot else None
        consistent=all(d>0 for d in means)
        strict=all(s['all_strict_pass'] for s in block)
        # Strict correctness and all independent sessions must support the sign.
        hit=all(s['cache_hit'] for s in block)
        qualified=qualification_eligible and hit and k>=3 and strict and consistent and ci is not None and ci[0]>0
        output.append(dict(case_id=key[0],attention_arm=key[1],endpoint_policy=key[2],sessions=k,
            run_results=block,run_mean_difference=stats(means),between_session_variance=statistics.variance(means) if k>1 else None,
            full_normal_ratio_equal_session_means=statistics.mean(s['full']['mean'] for s in block)/statistics.mean(s['normal']['mean'] for s in block),
            run_bootstrap_95_ci=ci,bootstrap_seed=seed,bootstrap_draws=draws,
            resampling_unit='independent process session mean, equal weight; NOT pooled pairs',
            uncertainty='exploratory; three sessions give limited confidence and cannot exclude unmeasured environment effects',
            LATENCY_REPEATED=k>=3,STRICT_CORRECTNESS_PASS=strict,CORRECTNESS_QUALIFIED_SPEEDUP=qualified,
            cache_hit=hit,hardware_and_structural_qualification_eligible=qualification_eligible,
            all_session_differences_positive=consistent,
            conclusion='CONSISTENT_OBSERVED_WARM_ADVANTAGE' if k>=3 and consistent else 'MIXED_OR_INSUFFICIENT_SESSION_EVIDENCE',
            scope='conditional warm-hit only; source construction excluded; no amortized service claim'))
    return output
