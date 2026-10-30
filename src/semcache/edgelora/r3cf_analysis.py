"""Fixed-count conditional inference; primary arms and auxiliary support stay separate."""
import csv
import math
import random
import statistics

from .communication import require
from .r3a_artifacts import write
from .r3c_analysis import percentile
from .r3cf_schedule import audit_schedule,flags

RESAMPLES=10000


def validate_values(values):
    require(values and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values),
        'finite nonnegative actual request walls required')


def fixed_count(hits,misses,control,seed=42,resamples=RESAMPLES):
    """Draw exactly n_hit HITs and n_miss MISSes in every bootstrap replicate."""
    require(type(resamples) is int and 10000<=resamples<=50000,'10000..50000 predeclared resamples required')
    validate_values(hits+misses);validate_values(control)
    require(len(control)>=8,'at least eight same-condition controls required')
    nh,nm,nc=len(hits),len(misses),len(control);n=nh+nm;rng=random.Random(seed)
    draws=[]
    for _ in range(resamples):
        total=(sum(rng.choices(hits,k=nh)) if nh else 0)+(sum(rng.choices(misses,k=nm)) if nm else 0)
        draws.append(total/n-sum(rng.choices(control,k=nc))/nc)
    lo,hi=percentile(draws,.025),percentile(draws,.975)
    warnings=[]
    if nh==1:warnings.append('SINGLE_HIT_SAMPLE_DEGENERACY: HIT variability is unidentifiable from this arm')
    if nm==1:warnings.append('SINGLE_MISS_SAMPLE_DEGENERACY: MISS variability is unidentifiable from this arm')
    return dict(delta_mean_ms=sum(hits+misses)/n-sum(control)/nc,ci95_low_ms=lo,ci95_high_ms=hi,
        status='CACHE_FASTER' if hi<0 else 'CACHE_SLOWER' if lo>0 else 'UNRESOLVED',
        n_hit=nh,n_miss=nm,n_control=nc,hit_fraction=nh/n,resamples=resamples,seed=seed,
        resampled_hit_count=nh,resampled_miss_count=nm,single_hit_degenerate=nh==1,
        uncertainty_adequate_for_diagnostic_bracket=not warnings,warnings=warnings,
        method='fixed within-arm HIT/MISS counts; independent same-condition NO_REUSE resampling; percentile95',
        caveat='conditional single-cohort small-n uncertainty, not independent deployments; shared controls across contrasts')


def describe(rows):
    values=[r['request_wall_ms'] for r in rows];validate_values(values)
    return dict(samples=len(rows),mean_ms=statistics.mean(values),median_ms=statistics.median(values),p90_ms=percentile(values,.9),
        request_ids=[r['request_id'] for r in rows],raw_walls_ms=values,
        hidden_frames=sum(r['hidden_frames'] for r in rows),delta_frames=sum(r['delta_frames'] for r in rows),
        application_wire_bytes=sum(r['application_wire_bytes'] for r in rows),
        hidden_payload_bytes=sum(r['hidden_payload_bytes'] for r in rows),delta_payload_bytes=sum(r['delta_payload_bytes'] for r in rows),
        calibration_status='VALID' if all(r['calibration_status']=='VALID' for r in rows) else 'INVALID',
        cache_bitwise_all_passed=all(r['cache_bitwise_passed'] for r in rows),
        native_strict_passes=sum(r['native_parity']['passed'] for r in rows))


def bracket(points,retrospective=False):
    ordered=sorted(points,key=lambda p:p['hit_fraction']);candidates=[]
    for a,b in zip(ordered,ordered[1:]):
        if {a['status'],b['status']}!={'CACHE_FASTER','CACHE_SLOWER'}:continue
        adequate=all(p['uncertainty_adequate_for_diagnostic_bracket'] for p in (a,b))
        x,y=a['hit_fraction'],b['hit_fraction'];u,v=a['delta_mean_ms'],b['delta_mean_ms']
        candidates.append(dict(lower_hit_fraction=x,upper_hit_fraction=y,
            status='RETROSPECTIVE_EXPLORATORY_ONLY' if retrospective else
                'CONFIRMED_DIAGNOSTIC_BRACKET' if adequate else 'UNRESOLVED',
            uncertainty_adequate=adequate,exploratory_linear_point=x-u*(y-x)/(v-u),
            root_interpretation='interpolation of conditional mean contrasts, not an exactly measured root',
            caveat='single-HIT/MISS variability missing; auxiliary support cannot silently repair primary uncertainty' if not adequate else
                'small cohort, controlled schedule; no deployment/correctness-qualified inference'))
    status=('RETROSPECTIVE_EXPLORATORY_ONLY' if retrospective and candidates else
        'CONFIRMED_DIAGNOSTIC_BRACKET' if any(c['status']=='CONFIRMED_DIAGNOSTIC_BRACKET' for c in candidates) else 'UNRESOLVED')
    return dict(status=status,candidates=candidates,measured_domain=[p['hit_fraction'] for p in ordered],
        point_states=[p['status'] for p in ordered],no_grid_expansion=True)


def auxiliary_sensitivity(hits,misses,control,aux,seed=42,resamples=RESAMPLES):
    """Model-assisted replacement sensitivity, never an enlarged primary arm."""
    require(len(hits)==1 and len(aux)==8,'single primary HIT and exactly eight independent auxiliary HIT observations required')
    validate_values(aux);rng=random.Random(seed);n=len(hits)+len(misses);nc=len(control)
    draws=[(sum(rng.choices(misses,k=len(misses)))+rng.choice(aux))/n-sum(rng.choices(control,k=nc))/nc
        for _ in range(resamples)]
    return dict(classification='SECONDARY_AUXILIARY_REPLACEMENT_SENSITIVITY',primary_point_unchanged=True,
        primary_delta_mean_ms=sum(hits+misses)/n-statistics.mean(control),
        model_assisted_delta_mean_ms=(sum(misses)+statistics.mean(aux))/n-statistics.mean(control),
        ci95_low_ms=percentile(draws,.025),ci95_high_ms=percentile(draws,.975),
        primary_hit_ms=hits[0],auxiliary_hit_walls_ms=aux,auxiliary_samples=8,
        delta_at_aux_min_ms=(sum(misses)+min(aux))/n-statistics.mean(control),
        delta_at_aux_max_ms=(sum(misses)+max(aux))/n-statistics.mean(control),
        resamples=resamples,seed=seed,permits_primary_bracket_confirmation=False,
        assumption='replace one primary HIT draw with same-condition auxiliary resident-HIT distribution; time/order/fill exchangeability is assumed',
        caveat='auxiliary rows are not target-arm replications; does not identify the single-HIT primary distribution')


def mixture_sensitivity(misses,hits,controls,seed=42,resamples=RESAMPLES):
    """Different-arm mean-mixture sensitivity; never a naturally observed rate."""
    if not misses or not hits or not controls:return dict(status='NOT_ESTIMABLE',reason='missing same-condition components')
    c,h,n=[r['request_wall_ms'] for r in misses],[r['request_wall_ms'] for r in hits],[r['request_wall_ms'] for r in controls]
    for values in (c,h,n):validate_values(values)
    cm,hm,nm=map(statistics.mean,(c,h,n));den=cm-hm
    if den==0:return dict(status='NOT_ESTIMABLE',reason='equal MISS/HIT means; no finite mixture root')
    point=(cm-nm)/den;rng=random.Random(seed);roots=[]
    for _ in range(resamples):
        cc,hh,nn=[sum(rng.choices(v,k=len(v)))/len(v) for v in (c,h,n)]
        if cc!=hh:
            root=(cc-nn)/(cc-hh)
            if math.isfinite(root):roots.append(root)
    return dict(status='EXPLORATORY_MEAN_MIXTURE_ONLY',hit_fraction_point=point,in_unit_interval=0<=point<=1,
        finite_bootstrap_roots=len(roots),resamples=resamples,
        exploratory_root_ci95=None if not roots else [percentile(roots,.025),percentile(roots,.975)],
        raw_components=dict(miss_ms=c,hit_ms=h,no_reuse_ms=n),
        component_request_ids=dict(miss=[r['request_id'] for r in misses],hit=[r['request_id'] for r in hits],no_reuse=[r['request_id'] for r in controls]),
        different_arms_combined=True,reused_control_observations=True,
        assumptions='linear mean mixture; exchangeable same-condition components despite order/arm/fill differences; not natural hit probability',
        scope='model-assisted sensitivity, not measured bracket or exact root; no R2-D timing pooling')


def focused_analysis(rows,manifest,fixture=False,resamples=RESAMPLES):
    accounting=audit_schedule(rows,manifest)
    require(all(r['cache_bitwise_passed'] and r['calibration_status']=='VALID' and r['impairment_status']=='VALID' for r in rows),
        'quality/calibration failure blocks statistical conclusions')
    points=[];decisions={};residents={};models={}
    for p in manifest['profiles']:
        cid=p['id'];block=[r for r in rows if r['condition_id']==cid]
        controls=[r for r in block if r['phase']=='measured' and r['arm']=='NO_REUSE']
        require(len(controls)==16,'sixteen primary controls per condition required')
        control=[r['request_wall_ms'] for r in controls]
        aux=[r for r in block if r['phase']=='aux_measured'];require(len(aux)==8,'eight auxiliary HITs required')
        auxfill=[r for r in block if r['phase']=='aux_fill'];require(len(auxfill)==1,'one auxiliary fill required')
        current=[]
        for task in (t for t in manifest['schedule'] if t['condition']['id']==cid and t['role']=='PRIMARY' and t['arm']!='NO_REUSE'):
            arm=[r for r in block if r['phase']=='measured' and r['arm']==task['arm']]
            hits=[r['request_wall_ms'] for r in arm if r['cache_status']=='HIT'];misses=[r['request_wall_ms'] for r in arm if r['cache_status']=='MISS']
            require(len(arm)==16 and len(hits)==int(task['hit_fraction']*16) and len(hits)+len(misses)==16,'fixed primary count mismatch')
            stat=fixed_count(hits,misses,control,manifest['seed'],resamples)
            warm=[r for r in block if r['phase']=='warmup' and r['arm']==task['arm']]
            baseline_warm=[r for r in block if r['phase']=='warmup' and r['arm']=='NO_REUSE']
            cost=lambda rs:sum(r['request_wall_ms']+r['cache_reset_ms_outside_request']+r['arm_reset_ms_outside_request'] for r in rs)
            point=dict(condition_id=cid,condition=p,arm=task['arm'],**stat,cache=describe(arm),no_reuse=describe(controls),
                raw_hit_ms=hits,raw_miss_ms=misses,observed_wire_bytes_saved=sum(r['application_wire_bytes'] for r in controls)-sum(r['application_wire_bytes'] for r in arm),
                reset_inclusive_delta_mean_ms=(cost(arm)-cost(controls))/16,
                controlled_warmup_fill_reset_amortized_delta_ms=(cost(arm+warm)-cost(controls+baseline_warm))/16)
            if len(hits)==1:point['auxiliary_sensitivity']=auxiliary_sensitivity(hits,misses,control,
                [r['request_wall_ms'] for r in aux],manifest['seed'],resamples)
            current.append(point);points.append(point)
        decisions[cid]=bracket(current)
        resident_stat=fixed_count([r['request_wall_ms'] for r in aux],[],control,manifest['seed'],resamples)
        residents[cid]=dict(auxiliary_resident_hits=describe(aux),matched_primary_control=describe(controls),contrast=resident_stat,
            cold_fill_excluded_from_resident_request_wall=True,fill=describe(auxfill),
            auxiliary_fill_and_reset_amortized_delta_ms=cost(aux+auxfill)/8-statistics.mean(control),
            caveat='auxiliary diagnostic arm, not part of primary mixture counts; amortization includes one fill plus eight requests')
        misses=[r for r in block if r['phase']=='measured' and r['mode']=='COLD_MISS']
        models[cid]=mixture_sensitivity(misses,aux,controls,manifest['seed'],resamples)
    return points,dict(execution_status='COMPLETE',accounting=accounting,hit_fraction_brackets=decisions,
        resident_hit_only=residents,mean_mixture_sensitivity=models,
        primary_measurement_rows=96,auxiliary_measurement_rows=16,resamples=resamples,
        auxiliary_rows_excluded_from_primary=True,single_hit_policy='naive primary interval cannot confirm a bracket, even with auxiliary support',
        full_deployment_lifecycle='NOT_ESTIMABLE',endpoint='all-token prefill; not TTFT/full generation',**flags(fixture))


def export(output,points,decision):
    write(output/'r3cf_break_even_decision.json',decision)
    fields=('condition_id','arm','hit_fraction','n_hit','n_miss','n_control','delta_mean_ms','ci95_low_ms','ci95_high_ms',
        'status','single_hit_degenerate','uncertainty_adequate_for_diagnostic_bracket')
    with (output/'r3cf_break_even.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for point in points:writer.writerow({k:point.get(k) for k in fields})
    write(output/'r3cf_arm_statistics.json',points)
