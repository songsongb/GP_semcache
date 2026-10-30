"""The predeclared six primary arms and two auxiliary blocks; no grid search."""
import random

from .communication import digest,require
from .r3c_analysis import budget as resource_budget


def flags(fixture=False):
    return dict(classification='R3C_FOCUSED_APPLICATION_LAYER_DIAGNOSTIC',PRIMARY_PERFORMANCE_ELIGIBLE=False,
        correctness_qualified_reuse_speedup=False,actual_remote_network_measured=False,
        semantic_generalization_validated=False,cachegen_enabled=False,compression_enabled=False,
        trained_adapter=False,r2d_break_even_validated=False,TEST_FIXTURE_ONLY=fixture)


def make_schedule(seed=42):
    require(type(seed) is int and 0<=seed<2**32,'bounded integer seed required')
    profiles=[dict(id='C00',rtt_ms=0,es_to_ud_mbps=None,ud_to_es_mbps=None),
              dict(id='C03',rtt_ms=10,es_to_ud_mbps=200,ud_to_es_mbps=200)]
    rng=random.Random(seed);tasks=[]
    for profile,counts in zip(profiles,((None,2,4),(None,0,1))):
        arms=list(counts);rng.shuffle(arms)
        for hits in arms:
            modes=['NO_REUSE']*16 if hits is None else ['WARM_HIT']*hits+['COLD_MISS']*(16-hits)
            rng.shuffle(modes)
            tasks.append(dict(condition=profile,arm='NO_REUSE' if hits is None else f'CACHE_{hits:02d}_OF_16',
                role='PRIMARY',hit_fraction=None if hits is None else hits/16,
                warmup_phase='warmup',measured_phase='measured',
                warmup_modes=['NO_REUSE' if hits is None else 'COLD_MISS'],measured_modes=modes))
        tasks.append(dict(condition=profile,arm='AUX_RESIDENT_HIT',role='AUXILIARY',hit_fraction=1.,
            warmup_phase='aux_fill',measured_phase='aux_measured',
            warmup_modes=['COLD_MISS'],measured_modes=['WARM_HIT']*8))
    body=dict(schema='r3cf_fixed_schedule_v1',seed=seed,profiles=profiles,schedule=tasks,
        conditions_order='proxy baseline first, then C03',auxiliary_order='after randomized primary arms in each condition',
        scope='controlled full-request exact-prompt HIT counts; not natural hit probabilities')
    return dict(body,schedule_sha256=digest(body))


def events(manifest):
    result=[]
    for profile in manifest['profiles']:
        result.extend(dict(condition_id=profile['id'],arm='GATE',phase='gate',ordinal=i,mode=m,role='GATE')
            for i,m in enumerate(('NO_REUSE','COLD_MISS','WARM_HIT')))
        for task in (t for t in manifest['schedule'] if t['condition']['id']==profile['id']):
            for phase,modes in ((task['warmup_phase'],task['warmup_modes']),(task['measured_phase'],task['measured_modes'])):
                result.extend(dict(condition_id=profile['id'],arm=task['arm'],phase=phase,ordinal=i,mode=m,role=task['role'])
                    for i,m in enumerate(modes))
    return result


def validate_schedule(manifest):
    require(manifest==make_schedule(manifest['seed']),'changed/noncanonical focused schedule')
    realized=events(manifest)
    count=lambda phase:sum(e['phase']==phase for e in realized)
    accounting=dict(primary_measured=count('measured'),primary_warmup=count('warmup'),condition_gates=count('gate'),
        auxiliary_measured=count('aux_measured'),auxiliary_fills=count('aux_fill'),total=len(realized))
    require(accounting==dict(primary_measured=96,primary_warmup=6,condition_gates=6,
        auxiliary_measured=16,auxiliary_fills=2,total=126),'126-request accounting changed')
    return accounting


def budget(manifest):
    accounting=validate_schedule(manifest);result=resource_budget(manifest)
    require(result['model_request_bound']==accounting['total'],'resource/model count mismatch')
    actual=events(manifest);nonhits=sum(e['mode']!='WARM_HIT' for e in actual)
    return dict(result,accounting=accounting,planned_full_exchange_requests=nonhits,
        planned_full_hit_requests=len(actual)-nonhits,planned_hidden_frames=32*nonhits,planned_delta_frames=32*nonhits,
        count_provenance='PREDECLARED_SCHEDULE_ONLY; actual socket ledgers must independently match')


def audit_schedule(rows,manifest):
    accounting=validate_schedule(manifest);expected=events(manifest)
    keys=('condition_id','arm','phase','ordinal','mode')
    require(len(rows)==len(expected) and len({r['request_id'] for r in rows})==len(rows),'missing/duplicate/extra request IDs')
    require([tuple(r[k] for k in keys) for r in rows]==[tuple(e[k] for k in keys) for e in expected],
        'realized focused schedule differs from immutable plan')
    for i,(row,event) in enumerate(zip(rows,expected)):
        require(row['execution_order']==i and row['analysis_role']==event['role'] and
            row['schedule_sha256']==manifest['schedule_sha256'],'request role/order/schedule identity mismatch')
    return accounting
