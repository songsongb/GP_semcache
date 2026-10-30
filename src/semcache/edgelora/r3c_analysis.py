"""Bounded schedules, evidence audit and signed, uncertainty-gated crossovers."""
import csv
import math
from pathlib import Path
import random
import statistics

from .communication import digest,require
from .r3a_artifacts import read,sha,THRESHOLDS,write
from .r3b_analysis import q90,validate_row as validate_b_row
from .r3c_network_proxy import condition

FRACTIONS=(0.,.125,.25,.5,1.)


def flags(fixture=False):
    return dict(classification='R3C_APPLICATION_LAYER_NETWORK_DIAGNOSTIC',PRIMARY_PERFORMANCE_ELIGIBLE=False,
        correctness_qualified_reuse_speedup=False,actual_remote_network_measured=False,
        semantic_generalization_validated=False,cachegen_enabled=False,compression_enabled=False,
        trained_adapter=False,r2d_break_even_validated=False,TEST_FIXTURE_ONLY=fixture)


def profiles(rtts,bandwidths):
    require(rtts and bandwidths and len(rtts)==len(set(rtts)) and len(bandwidths)==len(set(bandwidths)),'duplicate/empty grid axes')
    return validate_profiles([dict(id=f'C{i:02d}',rtt_ms=r,es_to_ud_mbps=b,ud_to_es_mbps=b)
        for i,(r,b) in enumerate((r,b) for r in rtts for b in bandwidths)])


def validate_profiles(values):
    require(1<=len(values)<=8,'one to eight bounded conditions required')
    result=[condition(v) for v in values]
    signatures=[(p['rtt_ms'],p['es_to_ud_mbps'],p['ud_to_es_mbps']) for p in result]
    require(len(set(signatures))==len(result) and len({p['id'] for p in result})==len(result),'duplicate conditions/IDs')
    require((0,None,None) in signatures,'include the required zero-delay/unlimited proxy control')
    base=signatures.index((0,None,None));result.insert(0,result.pop(base));return result


def schedule(values,fractions=FRACTIONS,n=8,warmup=1,seed=42):
    values=validate_profiles(values)
    require(type(n) is int and n in (8,16) and type(warmup) is int and 1<=warmup<=2,'requests 8 or 16; warmups 1 or 2')
    require(fractions and len(fractions)<=5 and len(set(fractions))==len(fractions),'unique one to five fractions required')
    require(all(type(h) in (float,int) and math.isfinite(h) and 0<=h<=1 and h*n==int(h*n) for h in fractions),
            'fractions must yield integral actual HIT counts')
    rng=random.Random(seed);order=list(values[1:]);rng.shuffle(order);order=[values[0],*order];result=[]
    for p in order:
        arms=[dict(id='NO_REUSE',fraction=None),*[dict(id=f'CACHE_{int(h*n):02d}_OF_{n}',fraction=float(h)) for h in fractions]]
        rng.shuffle(arms)
        for arm in arms:
            hits=0 if arm['fraction'] is None else int(arm['fraction']*n)
            measured=['WARM_HIT']*hits+['COLD_MISS']*(n-hits) if arm['fraction'] is not None else ['NO_REUSE']*n
            rng.shuffle(measured)
            result.append(dict(condition=p,arm=arm['id'],hit_fraction=arm['fraction'],
                warmup_modes=['NO_REUSE' if arm['fraction'] is None else 'COLD_MISS']*warmup,
                measured_modes=measured,reset_before_arm=True,
                schedule_scope='controlled deterministic exact-prompt HIT/MISS schedule, not natural hit probability'))
    return result


def budget(plan):
    tasks=plan['schedule'];conditions=plan['profiles'];requests=3*len(conditions)+sum(len(a['warmup_modes'])+len(a['measured_modes']) for a in tasks)
    # Upper-bound wire allowance for resource admission only, never a measured ledger.
    up=32*32*2560*2+32*4096;down=3*32*32*2560*4+32*4096
    traffic=0;network_seconds=0.
    for p in conditions:
        jobs=[a for a in tasks if a['condition']['id']==p['id']]
        misses=2+sum(sum(mode!='WARM_HIT' for mode in a['warmup_modes']+a['measured_modes']) for a in jobs)
        traffic+=misses*(up+down)
        network_seconds+=misses*(32*p['rtt_ms']/1000+sum(size*8/(rate*1e6) if rate else 0
            for size,rate in ((up,p['es_to_ud_mbps']),(down,p['ud_to_es_mbps']))))
    duration=network_seconds+requests*5+900  # conservative assumed compute/startup allowance, not a measurement
    require(requests<=512 and traffic<=12*2**30 and duration<=7200,'grid exceeds request/traffic/assumed duration budget')
    return dict(model_request_bound=requests,estimated_framed_bytes_upper_allowance=traffic,
        assumed_duration_budget_seconds=duration,max_duration_seconds=7200,
        provenance='RESOURCE_ADMISSION_ASSUMPTION_ONLY; no measured latency or traffic substituted')


def percentile(values,q):
    ordered=sorted(values);x=(len(ordered)-1)*q;i=int(x);j=min(i+1,len(ordered)-1)
    return ordered[i]+(ordered[j]-ordered[i])*(x-i)


def contrast(cache,control,seed=42,resamples=2000):
    require(len(cache)==len(control)>=8,'at least eight matched-condition samples required')
    require(all(math.isfinite(x) and x>=0 for x in cache+control),'nonfinite/negative request walls')
    rng=random.Random(seed);n=len(cache)
    bootstrap=[statistics.mean(rng.choices(cache,k=n))-statistics.mean(rng.choices(control,k=n)) for _ in range(resamples)]
    lo,hi=percentile(bootstrap,.025),percentile(bootstrap,.975)
    differences=[c-b for c,b in zip(cache,control)]
    return dict(delta_mean_ms=statistics.mean(cache)-statistics.mean(control),
        delta_median_ordinal_ms=statistics.median(differences),raw_ordinal_contrasts_ms=differences,
        ci95_low_ms=lo,ci95_high_ms=hi,
        status='CACHE_FASTER' if hi<0 else 'CACHE_SLOWER' if lo>0 else 'UNRESOLVED',
        samples=n,resamples=resamples,bootstrap='independent resampling of matched-condition arms; not a paired randomized trial',
        uncertainty='small-n conditional bootstrap; no definitive deployment inference')


def crossover(points,axis):
    if not points:return dict(status='NOT_ESTIMABLE',brackets=[])
    points=sorted(points,key=lambda p:p[axis]);brackets=[]
    for a,b in zip(points,points[1:]):
        if {a['status'],b['status']}=={'CACHE_FASTER','CACHE_SLOWER'}:
            x,y=a[axis],b[axis];u,v=a['delta_mean_ms'],b['delta_mean_ms']
            estimate=x-u*(y-x)/(v-u)
            brackets.append(dict(lower=x,upper=y,exploratory_linear_estimate=estimate,
                provenance='within-domain interpolation between measured opposite-sign CI-qualified points; not measured exact root'))
    if brackets:return dict(status='R3C_DIAGNOSTIC_CROSSOVER',brackets=brackets)
    states={p['status'] for p in points}
    status=('ALWAYS_FASTER_IN_TESTED_DOMAIN' if states=={'CACHE_FASTER'} else
            'ALWAYS_SLOWER_IN_TESTED_DOMAIN' if states=={'CACHE_SLOWER'} else
            'UNRESOLVED' if 'UNRESOLVED' in states else 'NO_CROSSOVER_IN_TESTED_DOMAIN')
    return dict(status=status,brackets=[],domain=[points[0][axis],points[-1][axis]])


def analyze(rows,plan):
    expected=[(a['condition']['id'],a['arm'],i,mode) for a in plan['schedule'] for i,mode in enumerate(a['measured_modes'])]
    measured=[r for r in rows if r['phase']=='measured']
    require([(r['condition_id'],r['arm'],r['ordinal'],r['mode']) for r in measured]==expected,'missing/reordered/duplicate measured requests')
    require(len({r['request_id'] for r in rows})==len(rows),'duplicate request IDs')
    output=[];hit_cross={};band_cross={};rtt_cross={}
    for p in plan['profiles']:
        cid=p['id'];control=[r for r in measured if r['condition_id']==cid and r['arm']=='NO_REUSE']
        require(control,'missing matched physical NO_REUSE arm')
        points=[]
        for arm in (a for a in plan['schedule'] if a['condition']['id']==cid and a['hit_fraction'] is not None):
            cache=[r for r in measured if r['condition_id']==cid and r['arm']==arm['arm']]
            require(all(r['cache_bitwise_passed'] and r['calibration_status']=='VALID' for r in cache+control),'invalid cache/calibration evidence')
            stats=contrast([r['request_wall_ms'] for r in cache],[r['request_wall_ms'] for r in control],plan['seed'])
            warm=[r for r in rows if r['condition_id']==cid and r['arm']==arm['arm'] and r['phase']=='warmup']
            base_warm=[r for r in rows if r['condition_id']==cid and r['arm']=='NO_REUSE' and r['phase']=='warmup']
            lifecycle=(sum(r['request_wall_ms']+r['cache_reset_ms_outside_request']+r['arm_reset_ms_outside_request'] for r in cache+warm)-
                sum(r['request_wall_ms']+r['cache_reset_ms_outside_request']+r['arm_reset_ms_outside_request'] for r in control+base_warm))/len(cache)
            point=dict(condition_id=cid,**{k:v for k,v in p.items() if k!='id'},arm=arm['arm'],hit_fraction=arm['hit_fraction'],**stats,
                cache_mean_ms=statistics.mean(r['request_wall_ms'] for r in cache),cache_median_ms=statistics.median(r['request_wall_ms'] for r in cache),
                cache_p90_ms=q90([r['request_wall_ms'] for r in cache]),no_reuse_mean_ms=statistics.mean(r['request_wall_ms'] for r in control),
                no_reuse_median_ms=statistics.median(r['request_wall_ms'] for r in control),no_reuse_p90_ms=q90([r['request_wall_ms'] for r in control]),
                observed_application_wire_bytes_saved=sum(r['application_wire_bytes'] for r in control)-sum(r['application_wire_bytes'] for r in cache),
                realized_hit_fraction=sum(r['cache_status']=='HIT' for r in cache)/len(cache),
                reset_inclusive_delta_mean_ms=statistics.mean(r['request_wall_ms']+r['cache_reset_ms_outside_request'] for r in cache)-statistics.mean(r['request_wall_ms'] for r in control),
                controlled_warmup_fill_and_reset_amortized_delta_ms=lifecycle,
                lifecycle_scope='declared controlled batch including warmups and resets; not deployment arrival/refresh amortization')
            points.append(point);output.append(point)
        hit_cross[cid]=crossover(points,'hit_fraction')
    for h in sorted({r['hit_fraction'] for r in output}):
        group=[r for r in output if r['hit_fraction']==h]
        for rtt in sorted({r['rtt_ms'] for r in group}):
            finite=[dict(r,bandwidth_mbps=r['es_to_ud_mbps']) for r in group if r['rtt_ms']==rtt and
                    r['es_to_ud_mbps'] is not None and r['es_to_ud_mbps']==r['ud_to_es_mbps']]
            band_cross[f'h{h}_rtt{rtt}']=crossover(finite,'bandwidth_mbps') if len(finite)>=2 else dict(status='NOT_ESTIMABLE',reason='no two finite equal-direction bandwidth points; unlimited is not a finite bracket')
        for pair in {(r['es_to_ud_mbps'],r['ud_to_es_mbps']) for r in group}:
            selected=[r for r in group if (r['es_to_ud_mbps'],r['ud_to_es_mbps'])==pair]
            rtt_cross[f'h{h}_bw{pair}']=crossover(selected,'rtt_ms') if len(selected)>=2 else dict(status='NOT_ESTIMABLE')
    return output,dict(execution_status='COMPLETE',hit_fraction_crossovers=hit_cross,bandwidth_crossovers=band_cross,
        rtt_crossovers=rtt_cross,full_deployment_lifecycle='NOT_ESTIMABLE: realistic arrival/refresh/residency distribution unavailable',
        resident_h1_scope='cold warmup/fill excluded from measured resident request contrast; controlled amortized value reported separately',
        primary_classification='R3C_DIAGNOSTIC_CROSSOVER only if brackets exist',**flags(plan['fixture']))


def r2d_comparison(directory=None):
    result=dict(status='NOT_DIRECTLY_COMPARABLE',r2d_break_even_validated=False,
        matrix=dict(model='same family must be verified',prompt_length='R3-C fixed L32; R2-D has separate historical/C4 cohorts',
            widths='physical hidden FP16 / bundled DELTAS FP32; R2-D widths configurable',
            boundaries='physical ES owns input/output, no h0/hL socket messages; M9-A may retain them',
            schedule='physical measured 32 HIDDEN/32 DELTAS on MISS, zero on HIT; R2-D RTT counts hypothetical',
            compute_endpoint='physical all-token FP16 base+CPU LoRA vs historical strict base/profile/calibrated totals',
            residency_fill='actual GPU cache insert/reset/guard vs analytical resident reuse assumptions',
            correctness='physical cache bitwise vs NO_REUSE; Native strict separate and may FAIL'),
        historical_timings_pooled=False,qualitative_direction_only=True,optional_artifacts_status='NOT_SUPPLIED')
    if directory is not None:
        path=Path(directory).resolve();manifest=read(path/'r2d_manifest.json')
        require(manifest['execution_status']=='COMPLETE' and manifest['classification']=='ANALYTICAL_DIAGNOSTIC','invalid optional R2-D manifest')
        required={'r2d_evidence_audit.json','r2d_bandwidth_rtt_sweep.csv','r2d_break_even_summary.json','r2d_decision.json'}
        require(required.issubset(manifest['output_hashes']),'missing R2-D output hashes')
        hashes={}
        for name,expected in manifest['output_hashes'].items():
            require(Path(name).name==name and sha(path/name)==expected,'R2-D output SHA mismatch');hashes[name]=expected
        audit=read(path/'r2d_evidence_audit.json');summary=read(path/'r2d_break_even_summary.json');decision=read(path/'r2d_decision.json')
        require(audit['execution_status']=='COMPLETE' and summary['schema']=='r2d_break_even_v1' and
            decision['execution_status']=='COMPLETE' and decision['classification']=='ANALYTICAL_DIAGNOSTIC','invalid R2-D export schemas')
        result['recorded_m9a_vs_c4_matrix']=audit.get('compatibility',{}).get('matrix',[])
        result['recorded_historical_payload_assumptions']=[dict(condition_id=r['condition_id'],prompt_tokens=r['prompt_tokens'],
            hidden_element_bytes=r['ledger']['hidden_element_bytes'],delta_element_bytes=r['ledger']['delta_element_bytes'],
            boundary_policy=r['ledger']['boundary_policy'],message_counts=r['ledger']['actual_message_counts'],
            compute_endpoint=r['historical_components_scope'],residency=r['ledger']['residency_assumption']) for r in summary['historical_m9a']]
        result.update(optional_artifacts_status='OUTPUT_SHA_VALIDATED_READ_ONLY',manifest_sha256=sha(path/'r2d_manifest.json'),output_hashes=hashes,
            original_input_rehash_status='NOT_REVERIFIED; historical input files are not required or fetched')
    return result


def export_analysis(output,points,decision):
    write(output/'r3c_break_even_decision.json',decision)
    fields=sorted({k for r in points for k in r}) or ['status','reason']
    with (output/'r3c_break_even.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in points:writer.writerow(row)


def audit_model(output,plan,result):
    """Reconcile original ES/UD and relay bytes independently, plus raw output SHA."""
    from .r3a_evidence import jsonl
    from .r3a_artifacts import check_parity,unpack,validate_prepared
    from .communication import canonical,PREFIX,validate
    from .r3b_analysis import flags as bflags
    output=Path(output);es=output/'es';rows=jsonl(es/'r3c_requests.jsonl')
    pm,native,am=validate_prepared(plan['prepared'])
    require(plan['reference_identity']==dict(identity=pm['identity'],identity_sha256=pm['identity_sha256'],
        prepared_manifest_sha256=sha(Path(plan['prepared'])/'r3a_prepare_manifest.json'),adapter_matrix_sha256=am['matrix_sha256'],
        gpu_uuid=pm.get('gpu',{}).get('uuid','TEST_FIXTURE_ONLY')),'cohort prepared/adapter/GPU identity mismatch')
    require(rows and len(rows)==plan['budget']['model_request_bound'],'missing/extra model requests')
    require(result['proxy_pid']==plan['proxy']['pid'] and len({result['pid'],result['ud_pid'],result['proxy_pid']})==3,'ES/UD/proxy PID identity failure')
    calibration=plan['calibration']
    require(read(output/'r3c_network_calibration.json')==calibration,'calibration report copies differ')
    require(read(es/'r3c_condition_gates.json')==result['condition_gates'],'condition gate copies differ')
    require(calibration['execution_status']=='COMPLETE' and calibration['independent_source_relay_sink_sha_and_bytes_passed'] and
            all(c['status']=='VALID' for c in calibration['conditions']),'INVALID_IMPAIRMENT calibration evidence')
    physical=read(es/'r3b_no_reuse_reference.json');physical_raw=(es/'r3b_no_reuse_logits.f32.bin').read_bytes()
    require(sha(es/'r3b_no_reuse_logits.f32.bin')==physical['logits_sha256'] and
        len(physical_raw)==4*math.prod(native['logits_shape']) and
        set(physical['qkv_sha256'])=={f'{l}.{r}' for l in range(pm['identity']['layers']) for r in 'qkv'},
        'physical reference SHA, shape or all-QKV set corrupted')
    expected_native=check_parity(unpack(physical_raw,'float32'),unpack((Path(plan['prepared'])/native['logits_file']).read_bytes(),'float32'),native['logits_shape'])
    source=jsonl(es/'es_wire.jsonl');sink=jsonl(es/'r3a_ud_wire_trace.jsonl')
    require(len(source)==len(sink),'ES/UD framing counts differ')
    proxy=[f for f in jsonl(output/'proxy/proxy_frames.jsonl') if f.get('header',{}).get('session_id')==result['session_id']]
    relay={(f['direction'],f['header']['sequence']):f for f in proxy}
    require(len(relay)==len(source)==len(proxy),'relay framing count/sequence duplicates')
    for a,b in zip(source,sink):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and
            all(a[k]==b[k] for k in ('header','frame_sha256','wire_bytes')),'independent ES/UD byte/hash mismatch')
        direction='es_to_ud' if a['operation']=='send' else 'ud_to_es';f=relay[direction,a['header']['sequence']]
        h=a['header'];validate(h,h['payload_bytes'])
        require(h['session_id']==result['session_id'] and h['adapter_sha256']==am['matrix_sha256'] and
            a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],'wire identity/framing byte ledger incorrect')
        require(f['complete'] and f['frame_sha256']==a['frame_sha256'] and f['header']==a['header'] and
                f['wire_bytes_read']==f['wire_bytes_forwarded']==a['wire_bytes'],'relay byte/hash mismatch')
    for operation in ('send','recv'):
        require([r['header']['sequence'] for r in source if r['operation']==operation]==
            list(range(sum(r['operation']==operation for r in source))),'wire sequence discontinuity')
    ud=read(es/'r3a_ud_result.json')
    require(ud['execution_status']=='COMPLETE' and ud['pid']==result['ud_pid'] and ud['user_device_cuda_used'] is False and
        ud['full_model_loaded'] is False and ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='','UD isolation failed')
    n=plan['reference_identity']['identity']['layers'];order=[]
    for task in plan['schedule']:
        if task['condition']['id'] not in order:order.append(task['condition']['id'])
    expected=[]
    for cid in order:
        expected.extend((cid,'GATE','gate',i,m) for i,m in enumerate(('NO_REUSE','COLD_MISS','WARM_HIT')))
        for task in (t for t in plan['schedule'] if t['condition']['id']==cid):
            expected.extend((cid,task['arm'],phase,i,mode) for phase,modes in
                (('warmup',task['warmup_modes']),('measured',task['measured_modes'])) for i,mode in enumerate(modes))
    require([(r['condition_id'],r['arm'],r['phase'],r['ordinal'],r['mode']) for r in rows]==expected,'realized workload schedule mismatch')
    for i,row in enumerate(rows):
        require(row['execution_order']==i and row['reference_identity']==plan['reference_identity'] and
            all(row.get(k)==v for k,v in flags(plan['fixture']).items()),'request identity/order/qualification mismatch')
        validate_b_row(dict(row,**bflags(plan['fixture'])),n)
        require(row['logits_sha256']==physical['logits_sha256'] and row['qkv_sha256']==physical['qkv_sha256'] and
                len(row['qkv_sha256'])==3*n and row['native_parity']==expected_native,'cache/Native output gate mismatch')
        require(row['calibration_status']==row['impairment_status']=='VALID','invalid condition realization')
        if row['phase']!='gate':
            require(result['condition_gates'][row['condition_id']]['status']=='PASS' and row['request_start_monotonic_ns']>=
                    result['condition_gates'][row['condition_id']]['completed_monotonic_ns'],'measurement before condition gate')
        frames=[r for r in source if r['header']['request_id']==row['request_id']]
        observed=[f for f in proxy if f['header']['request_id']==row['request_id']]
        require(observed==row['proxy_frames'] and len(frames)==len(observed)==row['hidden_frames']+row['delta_frames'],
                'per-request relay ledger mismatch')
        require(sum(f['wire_bytes_forwarded'] for f in observed)==row['application_wire_bytes'],'request wire total mismatch')
        for kind,field in (('HIDDEN','hidden_payload_bytes'),('DELTAS','delta_payload_bytes')):
            selected=[f for f in frames if f['header']['kind']==kind]
            require(sum(f['header']['payload_bytes'] for f in selected)==row[field] and
                [f['header']['layer_id'] for f in selected]==([] if row['mode']=='WARM_HIT' else list(range(n))),
                'directional payload or actual layer message schedule mismatch')
        layers=[l for l in ud['layers'] if l['request_id']==row['request_id']]
        require(len(layers)==row['hidden_frames'] and sum(l['metrics']['cpu_lora_ms'] for l in layers)==row['ud_cpu_lora_ms'],'UD did unexpected work')
        require(all(f['condition_id']==row['condition_id'] for f in observed),'mid-request profile change')
        for f in frames:
            require(row['request_start_monotonic_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=row['request_end_monotonic_ns'],
                    'data IO outside timed request')
        if i:require(rows[i-1]['request_end_monotonic_ns']<=row['request_start_monotonic_ns'],'overlapping/reordered requests')
    require(sum(r['hidden_frames'] for r in rows)==len(ud['layers']),'unaccounted UD computation')
    require(ud['completed_requests']==sum(r['hidden_frames']>0 for r in rows),'UD request total mismatch')
    controls=[f for f in proxy if f['header']['kind'] not in ('HIDDEN','DELTAS')]
    require(len(controls)==4 and [r['header']['kind'] for r in source[:2]]==['HELLO','READY'] and
            [r['header']['kind'] for r in source[-2:]]==['STOP','BYE'],'model handshake/shutdown missing')
    return dict(status='PASS',rows=len(rows),independent_es_ud_relay_sha_bytes_reconciled=True,
        native_peft_strict_parity_passed=expected_native['passed'],native_parity=expected_native,
        model_session_control_frames=4,model_session_control_wire_bytes=sum(f['wire_bytes_forwarded'] for f in controls),
        controls_outside_request_wall=True,original_ud_requests=ud['completed_requests'])
