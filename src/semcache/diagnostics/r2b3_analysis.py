"""Length-wise, session-balanced observations with explicit endpoint qualification."""
from collections import defaultdict
import random
import statistics
from semcache.diagnostics.r2b2_analysis import summarize
from semcache.metrics.r1b2 import stats


def classify(means,ci,strict,hit,verified=True,output_matched=False):
    if not hit: return 'BREAK_EVEN_UNRESOLVED'
    if not means or not verified: return 'BREAK_EVEN_UNRESOLVED'
    if statistics.mean(means)<=0: return 'NO_OBSERVED_SPEEDUP'
    repeated=len(means)>=3 and all(x>0 for x in means) and ci is not None and ci[0]>0
    if repeated and strict and output_matched: return 'CORRECTNESS_QUALIFIED_SPEEDUP'
    return 'REPRODUCED_POSITIVE_LATENCY_DIFFERENCE' if repeated else 'POSITIVE_MEAN_ONLY'


def aggregate(sessions,seed=901,draws=2000):
    if not sessions: raise ValueError('input sessions required')
    seen_runs=set(); seen_process=set(); environment=None; hardware=None
    groups=defaultdict(list); identities={}; requested=set(); manifests=[]
    for manifest,rows in sessions:
        if manifest['experiment']!='R2-B3' or manifest['benchmark_config']['phase']!='latency' or manifest['execution_status'] not in ('COMPLETE','COMPLETE_WITH_FAILURES'):
            raise ValueError('only finished B3 latency inputs; pilot/profile/failed sessions cannot enter minimal statistics')
        config=manifest['benchmark_config']; run=config['independent_run_id']
        process=manifest['process_instance']
        if run in seen_runs or process in seen_process: raise ValueError('duplicate run/process identity')
        seen_runs.add(run); seen_process.add(process); manifests.append(manifest)
        if hardware is None: hardware=manifest['hardware_identity']
        elif hardware!=manifest['hardware_identity']: raise ValueError('HARDWARE_MISMATCH: separate physical GPU sessions')
        if environment is None: environment=manifest['environment_identity']
        elif environment!=manifest['environment_identity']: raise ValueError('MODEL_CODE_OR_ENVIRONMENT_MISMATCH')
        expected=set(manifest['conditions'])
        requested|=expected
        if any(r['benchmark_phase']!='latency' or r['instrumentation']!='minimal' for r in rows): raise ValueError('instrumentation/phase mixing')
        if any(r['independent_run_id']!=run for r in rows): raise ValueError('raw run identity mismatch')
        for key,condition in manifest['conditions'].items():
            block=[r for r in rows if r['case_id']+'_'+r['attention_arm']==key]
            if condition['status']!='VALID': continue
            if not block: raise ValueError('missing valid condition rows')
            if condition['P']+condition['S']!=condition['L'] or not 0<=condition['P']<condition['L'] or key!=f'L{condition["L"]}_P{condition["P"]}_{condition["attention_arm"]}':
                raise ValueError('invalid manifest geometry')
            if key in identities and identities[key]!=condition['condition_identity']: raise ValueError('WORKLOAD_NAMESPACE_MISMATCH')
            identities[key]=condition['condition_identity']
            if any(not r['structural_reuse_valid'] or not r['cache_integrity_valid'] or r['condition_identity']!=identities[key]
                or r['endpoint_policy']!='b1_all_logits' or r['output_matched'] or not r['materialization_confounded'] for r in block):
                raise ValueError('invalid gate or endpoint evidence')
            if any((r['L'],r['P'],r['S'])!=(condition['L'],condition['P'],condition['S']) or
                r['model']!=config['model'] or r['dtype']!=config['dtype'] or r['backend_effective']!='eager' for r in block):
                raise ValueError('raw model/case/backend does not match manifest')
            measured=[r for r in block if r['phase']=='measured']
            if sorted(r['pair_index'] for r in measured)!=list(range(config['measured_runs'])) or config['measured_runs']<20:
                raise ValueError('incomplete/duplicate measured pairs')
            s=summarize(block)
            if len(s)!=1: raise ValueError('mixed condition')
            for scope in ('suffix','last_position'):
                quality=[r.get('quality',{}).get(scope) for r in measured]
                available=[q for q in quality if q and q.get('max_abs_error') is not None]
                s[0][scope+'_quality']=dict(status='MEASURED' if len(available)==len(quality) else 'UNAVAILABLE_OR_PARTIAL',
                    max_abs_error_worst=max(q['max_abs_error'] for q in available) if available else None,
                    violation_count_worst=max(q['tolerance_violations'] for q in available) if available else None)
            groups[key].append((s[0],condition,manifest))
        if any(r['case_id']+'_'+r['attention_arm'] not in expected for r in rows): raise ValueError('unlisted raw condition')
    rng=random.Random(seed); result=[]
    for key in sorted(requested):
        block=groups[key]; means=[s['paired_difference']['mean'] for s,_,_ in block]
        n=len(means); missing=sorted(seen_runs-{s['independent_run_id'] for s,_,_ in block})
        ci=None
        if n>=3:
            boot=sorted(statistics.mean(rng.choices(means,k=n)) for _ in range(draws))
            ci=[boot[int(.025*(draws-1))],boot[int(.975*(draws-1))]]
        condition=next(m['conditions'][key] for m in manifests if key in m['conditions'])
        strict=bool(block) and all(s['all_strict_pass'] and c['strict_correctness']=='STRICT_PASS' for s,c,_ in block)
        verified=bool(block) and not missing and all(m['environment_verified'] for _,_,m in block)
        label=classify(means,ci,strict,condition['P']>0,verified,False)
        result.append(dict(case_id=key,L=condition['L'],P=condition['P'],S=condition['S'],
            prefix_ratio=condition['P']/condition['L'],suffix_ratio=condition['S']/condition['L'],
            attention_arm=condition['attention_arm'],sessions=n,missing_sessions=missing,
            mean_paired_difference_ms=statistics.mean(means) if means else None,
            run_mean_difference=stats(means),session_bootstrap_95_ci=ci,
            between_session_variance=statistics.variance(means) if n>1 else None,
            full_mean_ms=statistics.mean(s['full']['mean'] for s,_,_ in block) if block else None,
            normal_mean_ms=statistics.mean(s['normal']['mean'] for s,_,_ in block) if block else None,
            speedup=statistics.mean(s['full']['mean'] for s,_,_ in block)/statistics.mean(s['normal']['mean'] for s,_,_ in block) if block else None,
            strict_correctness='STRICT_PASS' if strict else 'STRICT_FAIL' if block else 'UNVERIFIED',
            endpoint_policy='b1_all_logits',output_matched=False,materialization_confounded=True,
            endpoint_classification='B1_COMPATIBLE_LATENCY_OBSERVATION',
            conclusion=label,correctness_qualified_speedup=False,service_speedup_claimed=False,
            no_hit_control=condition['P']==0,environment_verified=verified,
            session_results=[s for s,_,_ in block],
            uncertainty='exploratory session-mean bootstrap; three sessions give limited confidence; no outliers discarded',
            bootstrap_seed=seed,bootstrap_draws=draws,resampling_unit='independent process session mean, equal weight'))
    # A candidate region requires adjacent tested members of the same fixed-S
    # family with reproduced positive differences. Never interpolate a threshold.
    for arm in ('off','on'):
        for suffix in (1,64):
            family=sorted([r for r in result if r['attention_arm']==arm and r['S']==suffix],key=lambda r:r['L'])
            for a,b in zip(family,family[1:]):
                if b['L']==2*a['L'] and all(r['conclusion'] in ('REPRODUCED_POSITIVE_LATENCY_DIFFERENCE','BREAK_EVEN_REGION_CANDIDATE') for r in (a,b)):
                    for r in (a,b):
                        r['conclusion']='BREAK_EVEN_REGION_CANDIDATE'
                        r['region_scope']='adjacent measured positive endpoints under B1-compatible output; interval interior unmeasured'
    return result


def experimental_config(manifests):
    """Derive aggregate experiment from inputs, never analysis argparse defaults."""
    return dict(model=sorted({m['benchmark_config']['model'] for m in manifests}),
        dtype=sorted({m['benchmark_config']['dtype'] for m in manifests}),
        cases=sorted({(c['L'],c['P'],c['S']) for m in manifests for c in m['conditions'].values()}),
        attention_arms=sorted({c['attention_arm'] for m in manifests for c in m['conditions'].values()}),
        backend='eager',endpoint_policy='b1_all_logits',output_matched=False,
        input_environment_identities=sorted({m['environment_identity'] for m in manifests}))
