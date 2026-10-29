"""CPU-only, fail-closed independent-session cache-path analysis."""
from collections import defaultdict
import math
import random
import statistics
from semcache.metrics.r1 import digest
from semcache.metrics.r1b2 import stats
from semcache.diagnostics.r2c2_profile_memory import uuid
from semcache.diagnostics.r2c3_impact_replay import SETTINGS,BASES
from semcache.diagnostics.r2c4_cache_path import ARMS,RAW,SNAPSHOT,MAPPING

COMPARISONS={'baseline_minus_snapshot_ms':(ARMS[2],ARMS[3]),'baseline_minus_mapping_ms':(ARMS[2],ARMS[4]),
    'baseline_minus_both_ms':(ARMS[2],ARMS[5]),'baseline_minus_no_reuse_ms':(ARMS[2],ARMS[1]),
    'baseline_minus_full_ms':(ARMS[2],ARMS[0])}


def differences(wall):
    result={name:wall[a]-wall[b] for name,(a,b) in COMPARISONS.items()}
    result['interaction_ms']=result['baseline_minus_both_ms']-result['baseline_minus_snapshot_ms']-result['baseline_minus_mapping_ms']
    return result


def validate_row(r):
    if r['instrumentation']!='minimal' or set(r['wall_ms'])!=set(ARMS) or sorted(r['execution_order'])!=sorted(ARMS):
        raise ValueError('not a complete six-arm minimal observation')
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in r['wall_ms'].values()):
        raise ValueError('invalid request latency')
    expected=differences(r['wall_ms'])
    if set(r['differences'])!=set(expected) or any(not math.isclose(r['differences'][k],v,abs_tol=1e-9) for k,v in expected.items()):
        raise ValueError('incorrect paired differences/interaction')
    if any(r.get(k) is not True for k in ('ablation_equivalence_pass','structural_valid','cache_integrity_valid')) or r['impact_replay_reduction_calls']!=0 or r['correctness_qualified_speedup'] is not False:
        raise ValueError('failed ablation/structural/replay gates')
    if set(r['arms'])!=set(ARMS) or set(r['full_relative_quality'])!=set(ARMS):raise ValueError('missing arm execution or correctness evidence')
    n,layers=r['length'],r['projection_layers']
    for arm,a in r['arms'].items():
        reused=a['reused_tokens']; p=a['cache_path']; calls=p['calls']['request_internal']
        if not isinstance(reused,int) or not 0<=reused<=n or a['native_projection_rows']!=(n-reused)*layers*3 or a['skipped_projection_rows']!=reused*layers*3:
            raise ValueError('physical QKV row accounting mismatch')
        if not math.isclose(a['actual_coverage'],reused/n,abs_tol=1e-12) or a['real_reduction_calls']!=0 or a['fallback_calls']!=0 or a['dispatch_calls']!=int(arm!='FULL_NATIVE'):
            raise ValueError('invalid coverage or real Attention Impact calls')
        if arm in ARMS[:2] and reused:raise ValueError('Full/No-Reuse control skipped QKV')
        if p['timing_boundary_verified'] is not True:raise ValueError('unverified internal timing boundary')
        positive=bool(reused);ss=positive and arm in SNAPSHOT;ms=positive and arm in MAPPING
        if p['snapshot_bypassed']!=ss or p['mapping_bypassed']!=ms:raise ValueError('unexpected intervention')
        if p['snapshot_intervention']!=('APPLICABLE' if positive else 'NOT_APPLICABLE') or p['mapping_intervention']!=('APPLICABLE' if positive else 'NOT_APPLICABLE'):
            raise ValueError('zero-reuse ablation must be NOT_APPLICABLE')
        for name,count in [('source_snapshot_validate',int(positive)-int(ss)),('source_snapshot_oracle_dispatch',int(ss)),
            ('prepared_mapping_get',int(positive)-int(ms)),('prepared_mapping_oracle_dispatch',int(ms)),
            ('request_mapping_select',int(arm!='FULL_NATIVE')),('executor_setup',int(arm!='FULL_NATIVE')),('executor_cleanup',int(arm!='FULL_NATIVE'))]:
            if calls.get(name,0)!=count:raise ValueError('cache-path invocation mismatch: '+name)
        if ms and (p['mapping_lookup_kind']!='PREBOUND_NOT_A_CACHE_LOOKUP' or p['mapping_cache_hit'] is not None):
            raise ValueError('prebound dispatch misreported as a measured lookup hit')
        quality=r['full_relative_quality'][arm]
        if (not isinstance(quality['allclose'],bool) or not isinstance(quality['tolerance_violations'],int) or quality['tolerance_violations']<0 or
            quality['allclose']!=(quality['tolerance_violations']==0)):raise ValueError('inconsistent strict correctness')
    base=r['arms'][ARMS[2]]
    for arm in RAW:
        a=r['arms'][arm];proof=r['raw_equivalence'][arm]
        if any(proof.get(k) is not True for k in ('exact_logits','exact_qkv','policy_equal','impact_values_equal','source_and_rows_equal')):
            raise ValueError('missing exact baseline/oracle equivalence')
        if not proof['strict_metrics']['allclose'] or proof['impact_replay_reduction_calls']!=0:raise ValueError('oracle parity failed')
        if proof['qkv_digest']!=a['qkv_digest']:raise ValueError('assembled QKV proof digest inconsistent')
        for key in ('reused_tokens','actual_coverage','native_projection_rows','skipped_projection_rows',
            'selection_hash','selected_provenance_digest','policy','reference_digest','impact_values_digest','qkv_digest'):
            if not base.get('qkv_digest') or a[key]!=base[key]:raise ValueError('RAW source/QKV/policy/provenance mismatch: '+key)


def summarize(rows):
    groups=defaultdict(list);seen=set()
    for r in rows:
        validate_row(r)
        key=(r['independent_run_id'],r['case_id'],r['phase'],r['pair_index'])
        if key in seen:raise ValueError('duplicate paired sample')
        seen.add(key)
        if r['phase']=='measured':groups[key[:2]].append(r)
        elif r['phase']!='warmup':raise ValueError('unknown sample phase')
    output=[]
    for (run,case),block in sorted(groups.items()):
        reused={r['arms'][ARMS[2]]['reused_tokens'] for r in block}
        if len(reused)!=1:raise ValueError('coverage changed within condition')
        output.append(dict(independent_run_id=run,case_id=case,count=len(block),length=block[0]['length'],
            requested_coverage=block[0]['requested_coverage'],reused_tokens=next(iter(reused)),
            walls={a:stats([r['wall_ms'][a] for r in block]) for a in ARMS},
            differences={k:stats([r['differences'][k] for r in block]) for k in differences(block[0]['wall_ms'])},
            wins={k:sum(r['differences'][k]>0 for r in block) for k in differences(block[0]['wall_ms'])},
            full_relative_strict={a:all(r['full_relative_quality'][a]['allclose'] for r in block) for a in ARMS},outliers_removed=False))
    return output


def csv_summary(rows):
    output=[]
    for s in summarize(rows):
        row={k:s[k] for k in ('independent_run_id','case_id','count','length','requested_coverage','reused_tokens')}
        for a,values in s['walls'].items():row.update({a+'_'+k+'_ms':v for k,v in values.items()})
        for a,values in s['differences'].items():row.update({a+'_'+k:v for k,v in values.items()})
        output.append(row)
    return output


def signature(m,require_complete=True):
    c=m['benchmark_config']
    if (m['experiment']!='R2-C4' or (require_complete and (m['execution_status']!='COMPLETE' or m['GPU_VALIDATION']!='EXECUTED' or not m.get('final_weights_verified'))) or
        m['settings']!=SETTINGS or m.get('classification')!='ORACLE_DIAGNOSTIC' or
        any(m.get(k) is not False for k in ('safe_reuse_claimed','correctness_qualified_speedup','production_speedup_claimed'))):
        raise ValueError('only fully validated C4 GPU inputs accepted')
    return (m['environment_identity'],uuid(m['hardware_identity']['uuid']),m['code']['sha256'],
        m['implementation_sha256'],m['model_weight_identity'],{k:c[k] for k in ('model','dtype','capacity_mib','cache_device','seed')})


def approve_pilot(manifest,pilot,current,key):
    if signature(manifest)!=signature(current,require_complete=False):
        raise ValueError('C4 Pilot GPU/model/code/settings mismatch')
    if manifest['benchmark_config']['phase']!='pilot' or pilot.get('experiment')!='R2-C4' or pilot.get('execution_status')!='COMPLETE' or not pilot.get('model_weights_verified') or pilot.get('environment_identity')!=current['environment_identity']:
        raise ValueError('completed compatible C4 Pilot required')
    c=pilot['conditions'].get(key)
    if not c or c!=manifest['conditions'].get(key) or c['status']!='CAPACITY_VALIDATED':raise ValueError('Pilot condition missing/invalid')
    expected=current['conditions'].get(key)
    if not expected or (c['length'],c['requested_coverage'])!=(expected['length'],expected['requested_coverage']):raise ValueError('Pilot requested length/coverage mismatch')
    if any(c.get(k) is not True for k in ('structural_valid','cache_integrity_valid','memory_valid','ablation_equivalence_pass')) or c['impact_replay_reduction_calls']!=0 or not c.get('memory_after',{}).get('gpu',{}).get('peak_allocated_bytes'):
        raise ValueError('Pilot structural/oracle/memory evidence missing')
    return c


def aggregate(sessions,breakdown,seed=4101,draws=2000):
    if len(sessions)!=3 or draws<100:raise ValueError('three independent sessions and >=100 bootstrap draws required')
    expected=None;seen_run=set();seen_pid=set();seen_seed=set();cases=None;grouped=defaultdict(list)
    for m,rows in sessions:
        c=m['benchmark_config'];sig=signature(m)
        if c['phase']!='latency' or c['warmup_runs']<3 or c['measured_runs']<20:raise ValueError('not repeated minimal latency')
        if c['independent_run_id'] in seen_run or m['process_instance'] in seen_pid or c['order_seed'] in seen_seed:raise ValueError('duplicate run/process/order seed')
        seen_run.add(c['independent_run_id']);seen_pid.add(m['process_instance']);seen_seed.add(c['order_seed'])
        if expected is None:expected=sig;cases=set(m['conditions'])
        elif sig!=expected or set(m['conditions'])!=cases:raise ValueError('incompatible session identity/cases')
        summaries=summarize(rows)
        if {s['case_id'] for s in summaries}!=cases:raise ValueError('missing cases')
        for s in summaries:
            case=m['conditions'][s['case_id']];block=[r for r in rows if r['case_id']==s['case_id']]
            if (case['status']!='VALID' or case['measured_pairs']!=c['measured_runs'] or
                any(case.get(k) is not True for k in ('ablation_equivalence_pass','structural_valid','cache_integrity_valid','memory_valid')) or
                case['impact_replay_reduction_calls']!=0):raise ValueError('invalid case')
            if not math.isclose(case['actual_coverage'],case['actual_reused_tokens']/case['length'],abs_tol=1e-12):raise ValueError('condition actual coverage mismatch')
            for phase,count in [('warmup',c['warmup_runs']),('measured',c['measured_runs'])]:
                if sorted(r['pair_index'] for r in block if r['phase']==phase)!=list(range(count)):raise ValueError('incomplete warmup/measured sample set')
            for r in block:
                if (r['benchmark_phase']!='latency' or r['independent_run_id']!=c['independent_run_id'] or
                    r['condition_identity']!=case['condition_identity'] or r['length']!=case['length'] or r['requested_coverage']!=case['requested_coverage'] or
                    r['model']!=c['model'] or r['dtype']!=c['dtype'] or r['endpoint_policy']!='all_token_logits_use_cache_false_attention_true' or
                    r['arms'][ARMS[2]]['reused_tokens']!=case['actual_reused_tokens']):raise ValueError('sample/source/config mismatch')
                for arm in ARMS[1:]:
                    base=BASES[0] if arm==ARMS[1] else BASES[1];a=r['arms'][arm]
                    if a['reference_digest']!=case['reference_digests'][base] or a['impact_values_digest']!=case['impact_values_digests'][base] or a['policy']!=case['policy_proofs'][base]:
                        raise ValueError('changed Impact reference/policy')
            grouped[s['case_id']].append((s,case,m))
    bm,diagnostics=breakdown
    if signature(bm)!=expected or bm['benchmark_config']['phase']!='breakdown':raise ValueError('incompatible breakdown identity/phase')
    expected_diag={k for k in cases if sessions[0][0]['conditions'][k]['length']==1024}
    if not expected_diag or set(bm['conditions'])!=expected_diag:raise ValueError('breakdown must cover the measured L1024 conditions')
    for key,c in bm['conditions'].items():
        if c['status']!='DIAGNOSED' or c.get('timing_boundary_verified') is not True or not c.get('ablation_equivalence_pass') or c['condition_identity']!=sessions[0][0]['conditions'][key]['condition_identity']:
            raise ValueError('breakdown boundary/equivalence/source missing')
        ref=sessions[0][0]['conditions'][key]
        if c['reference_digests']!=ref['reference_digests'] or c['actual_reused_tokens']!=ref['actual_reused_tokens']:
            raise ValueError('breakdown Impact reference or actual coverage mismatch')
        block=[r for r in diagnostics if r['case_id']==key]
        if len(block)!=len(ARMS)*bm['benchmark_config']['measured_runs'] or any(r['instrumentation']!='breakdown' or r['timing_boundary_verified'] is not True for r in block):
            raise ValueError('breakdown observations incomplete or mixed with minimal')
        pairs={(r['arm'],r['repeat']) for r in block}
        if pairs!={(a,i) for a in ARMS for i in range(bm['benchmark_config']['measured_runs'])}:raise ValueError('duplicate/missing diagnostic arm')
    rng=random.Random(seed);output=[]
    for key,group in sorted(grouped.items()):
        if len({c['condition_identity'] for _,c,_ in group})!=1 or len({digest(c['reference_digests']) for _,c,_ in group})!=1:raise ValueError('cross-session source/reference changed')
        comparisons={};positive=group[0][1]['actual_reused_tokens']>0
        for name in differences({a:1. for a in ARMS}):
            means=[s['differences'][name]['mean'] for s,_,_ in group]
            boot=sorted(statistics.mean(rng.choices(means,k=3)) for _ in range(draws))
            applicable=positive or name in ('baseline_minus_no_reuse_ms','baseline_minus_full_ms')
            comparisons[name]=dict(status='APPLICABLE' if applicable else 'NOT_APPLICABLE',session_means_ms=means,
                observed_equal_session_mean_ms=statistics.mean(means),attributable_difference_ms=statistics.mean(means) if applicable else None,
                session_bootstrap_95_ci_ms=[boot[int(.025*(draws-1))],boot[int(.975*(draws-1))]],
                session_win_counts=[s['wins'][name] for s,_,_ in group],between_session_variance=statistics.variance(means),
                interpretation='signed observed request contrast' if applicable else 'zero-reuse negative-control timing noise; no validation/mapping work was removed')
        output.append(dict(case_id=key,length=group[0][1]['length'],requested_coverage=group[0][1]['requested_coverage'],
            actual_reused_tokens=group[0][1]['actual_reused_tokens'],actual_coverage=group[0][1]['actual_coverage'],
            sessions=3,session_results=[s for s,_,_ in group],comparisons=comparisons,
            full_relative_strict={a:all(s['full_relative_strict'][a] and c['full_relative_strict'][a] for s,c,_ in group) for a in ARMS},
            resampling_unit='independent process session means, equal weights; exploratory with only three sessions',
            bootstrap_seed=seed,bootstrap_draws=draws,outliers_removed=False,
            classification='ORACLE_DIAGNOSTIC',correctness_qualified_speedup=False,production_speedup_claimed=False,
            interpretation='interaction is observed delta_both - delta_snapshot - delta_mapping; not an additive CPU/GPU cost decomposition'))
    return output
