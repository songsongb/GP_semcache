"""Fail-closed, CPU-only five-arm statistics; process sessions have equal weight."""
from collections import defaultdict
import math
import random
import statistics
from semcache.metrics.r1b2 import stats
from semcache.metrics.r1 import digest
from semcache.diagnostics.r2c2_profile_memory import uuid
from semcache.diagnostics.r2c3_impact_replay import ARMS,BASES,SETTINGS

COMPARISONS={
    'no_reuse_on_minus_replay_ms':(ARMS[1],ARMS[2]),
    'raw_on_minus_replay_ms':(ARMS[3],ARMS[4]),
    'no_reuse_on_minus_full_ms':(ARMS[1],ARMS[0]),
    'raw_on_minus_full_ms':(ARMS[3],ARMS[0]),
    'raw_replay_minus_full_ms':(ARMS[4],ARMS[0])}


def differences(wall): return {key:wall[a]-wall[b] for key,(a,b) in COMPARISONS.items()}


def validate_row(row):
    if row['instrumentation']!='minimal' or set(row['wall_ms'])!=set(ARMS):
        raise ValueError('not a minimal five-arm pair')
    if sorted(row['execution_order'])!=sorted(ARMS): raise ValueError('missing/duplicate paired arm')
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in row['wall_ms'].values()):
        raise ValueError('invalid request wall latency')
    if set(row['differences'])!=set(COMPARISONS) or any(not math.isclose(row['differences'][k],v,abs_tol=1e-9,rel_tol=1e-12) for k,v in differences(row['wall_ms']).items()):
        raise ValueError('paired difference sign/value mismatch')
    if any(row.get(k) is not True for k in ('ablation_equivalence_pass','structural_valid','cache_integrity_valid')):
        raise ValueError('ablation/structural/cache gate failed')
    if row['impact_replay_reduction_calls']!=0 or row.get('correctness_qualified_speedup') is not False:
        raise ValueError('invalid reduction/service qualification')
    if set(row['arms'])!=set(ARMS) or set(row['full_relative_quality'])!=set(ARMS):
        raise ValueError('missing execution/correctness evidence')
    length,layers=row['length'],row['projection_layers']
    if not isinstance(length,int) or not isinstance(layers,int) or min(length,layers)<1:
        raise ValueError('invalid physical row dimensions')
    for arm,r in row['arms'].items():
        reused=r['reused_tokens']
        if not isinstance(reused,int) or not 0<=reused<=length or r['native_projection_rows']!=(length-reused)*3*layers or r['skipped_projection_rows']!=reused*3*layers:
            raise ValueError('projection coverage/rows mismatch')
        if not math.isclose(r['actual_coverage'],reused/length,abs_tol=1e-12): raise ValueError('actual coverage mismatch')
        expected=1 if arm.endswith('_ON') else 0
        if r['real_reduction_calls']!=expected or r['fallback_calls']!=0:
            raise ValueError('unexpected real reduction calls')
        if r['dispatch_calls']!=(0 if arm=='FULL_NATIVE' else 1): raise ValueError('impact dispatch missing/repeated')
        if arm in ARMS[:3] and reused: raise ValueError('control reused QKV rows')
        quality=row['full_relative_quality'][arm]
        if (not isinstance(quality['allclose'],bool) or not isinstance(quality.get('tolerance_violations'),int) or
            quality['tolerance_violations']<0 or quality['allclose']!=(quality['tolerance_violations']==0)):
            raise ValueError('missing/inconsistent strict result')
    for base in BASES:
        on,replay=[row['arms'][base+suffix] for suffix in ('_IMPACT_ON','_IMPACT_REPLAY')]
        for key in ('reused_tokens','actual_coverage','skipped_projection_rows','native_projection_rows',
                    'selection_hash','selected_provenance_digest','policy','impact_values_digest','reference_digest'):
            if on.get(key)!=replay.get(key): raise ValueError('ON/REPLAY evidence mismatch: '+key)
        if not on.get('reference_digest') or not on.get('policy'): raise ValueError('missing Replay provenance/policy proof')
        parity=row['on_replay_parity'][base]
        if any(parity.get(k) is not True for k in ('exact_logits','policy_equal','coverage_and_rows_equal','impact_values_equal')) or not parity['strict_metrics']['allclose'] or parity['impact_replay_reduction_calls']!=0:
            raise ValueError('missing exact output/policy parity')


def summarize(rows):
    groups=defaultdict(list); seen=set()
    for row in rows:
        validate_row(row)
        key=(row['independent_run_id'],row['case_id'],row['phase'],row['pair_index'])
        if key in seen: raise ValueError('duplicate paired observation')
        seen.add(key)
        if row['phase']=='measured': groups[key[:2]].append(row)
        elif row['phase']!='warmup': raise ValueError('unknown sample phase')
    summaries=[]
    for (run,case),block in sorted(groups.items()):
        first=block[0]; reused={r['arms'][ARMS[3]]['reused_tokens'] for r in block}
        if len(reused)!=1: raise ValueError('coverage changed within session')
        summaries.append(dict(independent_run_id=run,case_id=case,count=len(block),length=first['length'],
            requested_coverage=first['requested_coverage'],reused_tokens=next(iter(reused)),
            actual_coverage=next(iter(reused))/first['length'],
            walls={a:stats([r['wall_ms'][a] for r in block]) for a in ARMS},
            differences={k:stats([r['differences'][k] for r in block]) for k in COMPARISONS},
            wins={k:sum(r['differences'][k]>0 for r in block) for k in COMPARISONS},
            full_relative_strict={a:all(r['full_relative_quality'][a]['allclose'] for r in block) for a in ARMS},
            ablation_equivalence_pass=True,impact_replay_reduction_calls=0,
            correctness_qualified_speedup=False,outliers_removed=False))
    return summaries


def csv_summaries(rows):
    output=[]
    for s in summarize(rows):
        flat={k:s[k] for k in ('independent_run_id','case_id','count','length','requested_coverage','reused_tokens','actual_coverage')}
        for arm,v in s['walls'].items(): flat.update({arm+'_'+k+'_ms':x for k,x in v.items()})
        for key,v in s['differences'].items(): flat.update({key+'_'+k:x for k,x in v.items()})
        flat.update(correctness_qualified_speedup=False,ablation_equivalence_pass=True)
        output.append(flat)
    return output


def approve_pilot(pilot_manifest,pilot,current,key):
    for value in (pilot_manifest,pilot):
        if value.get('experiment')!='R2-C3' or value.get('execution_status')!='COMPLETE' or value.get('environment_identity')!=current['environment_identity']:
            raise ValueError('C3 completed compatible Pilot required')
    if pilot_manifest['benchmark_config']['phase']!='pilot' or pilot_manifest.get('GPU_VALIDATION')!='EXECUTED':
        raise ValueError('Pilot GPU phase unverified')
    if (pilot_manifest.get('settings')!=SETTINGS or pilot_manifest.get('classification')!='ORACLE_DIAGNOSTIC' or
        pilot_manifest.get('safe_reuse_claimed') is not False or pilot_manifest.get('correctness_qualified_speedup') is not False):
        raise ValueError('Pilot fixed settings/diagnostic classification changed')
    if uuid(pilot_manifest['hardware_identity']['uuid'])!=uuid(current['hardware_identity']['uuid']): raise ValueError('Pilot physical GPU UUID mismatch')
    for name in ('model_weight_identity','implementation_sha256','settings'):
        if not current.get(name) or pilot_manifest.get(name)!=current[name]: raise ValueError('Pilot identity mismatch: '+name)
    if pilot_manifest['code']['sha256']!=current['code']['sha256']: raise ValueError('Pilot code changed')
    for name in ('model','dtype','capacity_mib','cache_device','seed'):
        if pilot_manifest['benchmark_config'].get(name)!=current['benchmark_config'].get(name):
            raise ValueError('Pilot benchmark setting mismatch: '+name)
    if not pilot_manifest.get('final_weights_verified') or not pilot.get('model_weights_verified'):
        raise ValueError('Pilot model/adapter weights unverified')
    condition=pilot['conditions'].get(key)
    if not condition or condition!=pilot_manifest['conditions'].get(key): raise ValueError('Pilot condition missing/inconsistent')
    if any(condition.get(k) is not True for k in ('structural_valid','cache_integrity_valid','ablation_equivalence_pass','memory_valid','source_token_integrity_valid')) or condition.get('status')!='CAPACITY_VALIDATED' or condition.get('impact_replay_reduction_calls')!=0:
        raise ValueError('Pilot capacity/ablation approval invalid')
    if not condition.get('memory_after',{}).get('gpu',{}).get('peak_allocated_bytes'):
        raise ValueError('Pilot actual GPU memory evidence missing')
    return condition


def aggregate(sessions,seed=4101,draws=2000):
    if len(sessions)!=3: raise ValueError('exactly three independent latency sessions required')
    if draws<100: raise ValueError('insufficient bootstrap draws')
    seen_runs=set(); seen_processes=set(); seen_order_seeds=set(); grouped=defaultdict(list); expected=None
    for manifest,rows in sessions:
        config=manifest['benchmark_config']; run=config['independent_run_id']
        if (manifest['experiment']!='R2-C3' or manifest['execution_status']!='COMPLETE' or
            config['phase']!='latency' or manifest.get('GPU_VALIDATION')!='EXECUTED' or
            manifest.get('classification')!='ORACLE_DIAGNOSTIC' or manifest.get('safe_reuse_claimed') is not False or
            manifest.get('correctness_qualified_speedup') is not False or manifest.get('settings')!=SETTINGS or
            not manifest.get('final_weights_verified')):
            raise ValueError('only fully verified COMPLETE C3 GPU latency sessions accepted')
        if run in seen_runs or manifest['process_instance'] in seen_processes: raise ValueError('duplicated independent run/process')
        if config['order_seed'] in seen_order_seeds: raise ValueError('independent latency sessions require distinct order seeds')
        seen_runs.add(run); seen_processes.add(manifest['process_instance'])
        seen_order_seeds.add(config['order_seed'])
        signature=(manifest['environment_identity'],uuid(manifest['hardware_identity']['uuid']),
            manifest['code']['sha256'],manifest['implementation_sha256'],manifest['model_weight_identity'],
            manifest['settings'],set(manifest['conditions']),{k:config[k] for k in ('model','dtype','capacity_mib','cache_device','seed')})
        if expected is None: expected=signature
        elif expected!=signature: raise ValueError('incompatible GPU/model/adapter/code/settings/cases')
        if config['warmup_runs']<3 or config['measured_runs']<20: raise ValueError('insufficient repeated protocol')
        summaries=summarize(rows)
        if {s['case_id'] for s in summaries}!=set(manifest['conditions']): raise ValueError('missing/extra session cases')
        for s in summaries:
            c=manifest['conditions'][s['case_id']]; block=[r for r in rows if r['case_id']==s['case_id']]
            if (c['status']!='VALID' or c.get('measured_pairs')!=config['measured_runs'] or
                any(c.get(k) is not True for k in ('structural_valid','cache_integrity_valid','ablation_equivalence_pass','memory_valid','source_token_integrity_valid')) or
                c['impact_replay_reduction_calls']!=0): raise ValueError('case failed required gates')
            if not math.isclose(c['actual_coverage'],c['actual_reused_tokens']/c['length'],abs_tol=1e-12):
                raise ValueError('condition actual coverage mismatch')
            for phase,count in [('warmup',config['warmup_runs']),('measured',config['measured_runs'])]:
                if sorted(r['pair_index'] for r in block if r['phase']==phase)!=list(range(count)):
                    raise ValueError('incomplete paired warmup/measured samples')
            for r in block:
                if (r['independent_run_id']!=run or r['benchmark_phase']!='latency' or
                    r['condition_identity']!=c['condition_identity'] or r['model']!=config['model'] or r['dtype']!=config['dtype'] or
                    r['length']!=c['length'] or r['requested_coverage']!=c['requested_coverage'] or
                    r['endpoint_policy']!='all_token_logits_use_cache_false_attention_true' or
                    r['arms'][ARMS[3]]['reused_tokens']!=c['actual_reused_tokens']):
                    raise ValueError('sample identity/coverage/configuration mismatch')
                for base in BASES:
                    for suffix in ('_IMPACT_ON','_IMPACT_REPLAY'):
                        arm=r['arms'][base+suffix]
                        if arm['reference_digest']!=c['reference_digests'][base] or arm['policy']!=c['policy_proofs'][base] or arm['impact_values_digest']!=c['impact_values_digests'][base]:
                            raise ValueError('missing/changed reference impact or policy proof')
            grouped[s['case_id']].append((s,c,manifest))
    rng=random.Random(seed); output=[]
    for key,group in sorted(grouped.items()):
        if len({c['condition_identity'] for _,c,_ in group})!=1 or len({digest(c['reference_digests']) for _,c,_ in group})!=1:
            raise ValueError('source or actual reference values changed across sessions')
        comparisons={}
        environment_verified=all(m.get('environment_verified') is True for _,_,m in group)
        for name in COMPARISONS:
            means=[s['differences'][name]['mean'] for s,_,_ in group]
            bootstrap=sorted(statistics.mean(rng.choices(means,k=3)) for _ in range(draws))
            ci=[bootstrap[int(.025*(draws-1))],bootstrap[int(.975*(draws-1))]]
            comparisons[name]=dict(session_means_ms=means,session_mean_statistics=stats(means),
                left_arm=COMPARISONS[name][0],right_arm=COMPARISONS[name][1],
                equal_session_mean_ms=statistics.mean(means),between_session_variance=statistics.variance(means),
                session_bootstrap_95_ci_ms=ci,session_win_counts=[s['wins'][name] for s,_,_ in group],
                session_win_rates=[s['wins'][name]/s['count'] for s,_,_ in group],
                direction='REPRODUCED_POSITIVE_DIFFERENCE' if all(v>0 for v in means) and ci[0]>0 and environment_verified else
                    'NONPOSITIVE_MEAN' if statistics.mean(means)<=0 else 'MIXED_OR_POSITIVE_MEAN_ONLY')
        output.append(dict(case_id=key,length=group[0][1]['length'],requested_coverage=group[0][1]['requested_coverage'],
            actual_reused_tokens=group[0][1]['actual_reused_tokens'],actual_coverage=group[0][1]['actual_coverage'],
            comparisons=comparisons,sessions=3,session_results=[s for s,_,_ in group],
            arm_equal_session_mean_ms={a:statistics.mean(s['walls'][a]['mean'] for s,_,_ in group) for a in ARMS},
            full_relative_strict={a:all(s['full_relative_strict'][a] and c['full_relative_strict'][a] for s,c,_ in group) for a in ARMS},
            environment_verified=environment_verified,
            ablation_equivalence_pass=True,impact_replay_reduction_calls=0,
            correctness_qualified_speedup=False,production_speedup_claimed=False,classification='ORACLE_DIAGNOSTIC',
            resampling_unit='independent process session means, equal weights; exploratory with only three sessions',
            bootstrap_seed=seed,bootstrap_draws=draws,outliers_removed=False,
            scope='conditional request impact-reduction ablation; no GPU-kernel-sum subtraction, no service-speedup inference'))
    return output


def coverage_changes(results):
    """Same-length high-minus-low measured contrasts, no interpolation."""
    output=[]
    for n in sorted({r['length'] for r in results}):
        block=sorted((r for r in results if r['length']==n),key=lambda r:r['actual_reused_tokens'])
        for low,high in zip(block,block[1:]):
            if low['actual_reused_tokens']==high['actual_reused_tokens']: continue
            output.append(dict(length=n,low_case=low['case_id'],high_case=high['case_id'],
                low_actual_coverage=low['actual_coverage'],high_actual_coverage=high['actual_coverage'],
                high_minus_low_arm_wall_ms={a:high['arm_equal_session_mean_ms'][a]-low['arm_equal_session_mean_ms'][a] for a in ARMS},
                high_minus_low_comparison_ms={k:high['comparisons'][k]['equal_session_mean_ms']-low['comparisons'][k]['equal_session_mean_ms'] for k in COMPARISONS},
                interpretation='observed same-length coverage sensitivity; not a kernel cost decomposition'))
    return output
