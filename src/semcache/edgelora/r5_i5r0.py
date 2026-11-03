"""I5R-0 CPU diagnostics. Independent checks are observations, never reuse grants."""
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from .r5_i5r0_io import UNKNOWN,CONDITIONS,ROOT,Files,canonical,sha,require,load_blocks,provenance

TIMINGS=('application_ttft_ms','model_ttft_ms','application_e2e_ms','model_execution_e2e_ms','semantic_encoding_ms',
    'cluster_assignment_ms','candidate_lookup_ms','eligibility_validation_ms','cache_lookup_ms','kv_reconstruction_ms',
    'cache_admission_eviction_ms','evidence_logging_io_ms','prefill_ms','decode_ms',
    'es_to_ud_wire_bytes','ud_to_es_wire_bytes','total_request_wire_bytes')


def stats(values):
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values) if values else None,median=statistics.median(values) if values else None,
                p95=values[math.ceil(.95*len(values))-1] if values else None)


def conjunction(values):
    return False if any(x is False for x in values) else True if all(x is True for x in values) else UNKNOWN


def candidate_diagnostic(source,target,*,source_length=None,target_length=None,resident=UNKNOWN,registry=None,actual_reason=UNKNOWN,windows=UNKNOWN):
    """No short circuit between individual safety checks, including length/prefix."""
    si=source.get('token_ids');ti=target.get('token_ids')
    sl=len(si) if isinstance(si,list) else source_length;tl=len(ti) if isinstance(ti,list) else target_length
    length_known=isinstance(sl,int) and isinstance(tl,int)
    tokens_known=isinstance(si,list) and isinstance(ti,list)
    lcp=0
    if tokens_known:
        for a,b in zip(si,ti):
            if a!=b:break
            lcp+=1
    sa=source.get('physical_identity',{}).get('adapter_id',source.get('adapter_id',UNKNOWN));ta=target.get('adapter_id',target.get('physical_identity',{}).get('adapter_id',UNKNOWN))
    adapter=sa==ta if sa!=UNKNOWN and ta!=UNKNOWN else UNKNOWN
    physical=source['physical_identity']==target['physical_identity'] if 'physical_identity' in source and 'physical_identity' in target else UNKNOWN
    su=source.get('source_user',source.get('user_id',UNKNOWN));tu=target.get('user_id',UNKNOWN)
    authorization=UNKNOWN
    if adapter is False:authorization=False
    elif registry is not None:
        a=registry.get('users',{}).get(su);b=registry.get('users',{}).get(tu)
        authorization=bool(a and b and a['adapter_id']==sa and b['adapter_id']==ta and adapter and
            (su==tu or a['share_group']==b['share_group'] and a['allow_cross_user_reuse'] and b['allow_cross_user_reuse']))
    def compatible(field):
        return source[field]==target[field][:sl] if field in source and field in target and isinstance(sl,int) else UNKNOWN
    checks=dict(resident=resident,physical_identity_compatible=physical,source_completed=source.get('completed',UNKNOWN),
        source_strictly_shorter=sl<tl if length_known else UNKNOWN,exact_full_source_prefix=ti[:sl]==si if tokens_known else UNKNOWN,
        positions_compatible=compatible('positions'),mask_compatible=compatible('attention_mask'),adapter_compatible=adapter,sharing_authorized=authorization)
    names=dict(resident='STALE_EVICTED_ENTRY',physical_identity_compatible='INCOMPATIBLE_PHYSICAL_IDENTITY',source_completed='INCOMPLETE_SOURCE',
        source_strictly_shorter='TARGET_NOT_LONGER_THAN_SOURCE',exact_full_source_prefix='EXACT_PREFIX_TOKEN_MISMATCH',
        positions_compatible='POSITION_MISMATCH',mask_compatible='MASK_MISMATCH',adapter_compatible='CROSS_ADAPTER_REUSE_FORBIDDEN',sharing_authorized='AUTHORIZATION_DENIED')
    return dict(source_request_id=source.get('source_request_id',UNKNOWN),source_user=su,target_user=tu,source_adapter=sa,target_adapter=ta,
        source_length=sl if sl is not None else UNKNOWN,target_length=tl if tl is not None else UNKNOWN,
        source_length_equal_target=sl==tl if length_known else UNKNOWN,source_length_greater_target=sl>tl if length_known else UNKNOWN,
        exact_full_sequence_equal=si==ti if tokens_known else UNKNOWN,longest_common_prefix_tokens=lcp if tokens_known else UNKNOWN,
        matching_3_token_windows=windows,checks=checks,actual_rejection_reason=actual_reason,
        independent_diagnostic_reasons=[names[k] for k,v in checks.items() if v is False],
        unobservable_checks=[k for k,v in checks.items() if v==UNKNOWN],
        could_pass_unchanged_safety_policy=conjunction(checks.values()),authorization_to_reuse=False)


def indexed(rows,key):
    result={}
    for row in rows:
        value=key(row);require(value not in result,'duplicate request/record identity: '+str(value));result[value]=row
    return result


def prepare_block(b,allow_partial):
    config=b['config'];block=config['block'];tasks=indexed(block['tasks'],lambda t:t['request_id'])
    rows=indexed(b['rows'],lambda r:(r['condition'],r['request_id']))
    require(set(rows)=={(c,rid) for c in CONDITIONS for rid in tasks},'missing/extra condition requests')
    for (c,rid),row in rows.items():
        require(row['status']=='COMPLETE' and row['adapter_id']==tasks[rid]['adapter_id'] and row['user_id']==tasks[rid]['user_id'],'request completion/routing differs')
        require(row.get('dataset',block['dataset'])==block['dataset'],'dataset identity differs')
        require(row.get('block_id',block['block_id'])==block['block_id'],'row block identity differs')
        require(row['scored']==tasks[rid]['scored'],'scored/source role differs')
        for metric in TIMINGS:
            if metric in row:require(isinstance(row[metric],(int,float)) and math.isfinite(row[metric]) and row[metric]>=0,'invalid/nonfinite timing or wire metric')
        require(type(row['reused_prefix_tokens']) is int and 0<=row['reused_prefix_tokens']<row['input_tokens'],'invalid reused-token count')
        require((row.get('source') is not None)==(row['reused_prefix_tokens']>0),'HIT receipt and reused token count disagree')
        phys=row.get('physical_identity',{})
        if 'adapter_matrix_fingerprint' in phys and 'tokenizer' in phys:
            require(row.get('matched_identity_sha256')==sha(canonical(dict(task=tasks[rid],adapter_sha256=phys['adapter_matrix_fingerprint'],tokenizer=phys['tokenizer']))),'prepared/observed immutable request hash differs')
        require(row['input_tokens']==len(tasks[rid]['token_ids']),'measured/prepared token lengths differ')
    for rid in tasks:
        matches=[rows[c,rid].get('matched_identity_sha256') for c in CONDITIONS]
        require(all(matches) and len(set(matches))==1,'unmatched A/B/C immutable request identity')
        physical=[rows[c,rid].get('physical_identity') for c in CONDITIONS]
        require(all(x==physical[0] for x in physical),'unmatched A/B/C physical execution identity')
    metadata={};discoveries={};indices={c:defaultdict(dict) for c in CONDITIONS};residency={};sequences=defaultdict(list)
    for wrapper in b['events']:
        c=wrapper['condition'];require(c in CONDITIONS,'unknown event condition');e=wrapper['event'];aid=wrapper['adapter_id']
        sequences[c].append(wrapper['sequence'])
        if e['kind']=='INDEX_INSERT':
            m=e['metadata'];metadata[c,m['source_request_id']]=m;indices[c][aid][m['entry_id']]=m
        elif e['kind']=='INDEX_INVALIDATE':indices[c][aid].pop(e['entry_id'],None)
        elif e['kind']=='DISCOVERY':
            key=(c,e['request_id']);require(key not in discoveries,'duplicate discovery event')
            require(key in rows,'orphan discovery');discoveries[key]=e
            residency[key]=set(indices[c][aid])
            for candidate in e['candidates']:metadata[c,candidate['metadata']['source_request_id']]=candidate['metadata']
    for c,seq in sequences.items():require(seq==list(range(len(seq))),'event sequence gap/order differs')
    candidates=[];request_records=[];gaps=[]
    if b.get('mode')=='RAW_RUNS' and len(b['requests'])!=len(rows):
        require(allow_partial,'missing original request arrivals')
        gaps.append('RAW_REQUEST_ARRIVALS_MISSING')
    for key,row in sorted(rows.items()):
        c,rid=key;task=tasks[rid]
        target=dict(task,physical_identity=row.get('physical_identity',{}))
        m=metadata.get(key,{})
        for name in ('positions','attention_mask'):
            if name in m:target[name]=m[name]
        if key in b['requests']:
            req=b['requests'][key];require(req['token_ids']==task['token_ids'],'arrival token evidence differs');target.update(req)
            if 'request_sha256' in req:
                sealed=dict(req);request_seal=sealed.pop('request_sha256')
                require(sha(canonical(sealed))==request_seal==row['request_sha256'],'raw arrival seal differs')
        if m:require(m['token_ids']==target['token_ids'],'source/target metadata token evidence differs')
        e=discoveries.get(key)
        if c!='FULL_RECOMPUTE':
            if e is None:
                require(allow_partial,'missing semantic discovery evidence');gaps.append(str(key)+':DISCOVERY')
        actual=row.get('candidate_reasons',[])
        n=row['candidate_count'];require(n==len(actual),'candidate count/reason count differs')
        if e is not None:require(n==len(e['candidates']) and actual==e['candidate_reasons'],'discovery/row candidates differ')
        reasons={x['entry_id']:x for x in actual};require(len(reasons)==len(actual),'duplicate candidate entry')
        these=[]
        observed=e['candidates'] if e is not None else [dict(metadata=dict(entry_id=x['entry_id'],source_request_id=x['source_request_id'])) for x in actual]
        for rank,x in enumerate(observed):
            source=x['metadata'];reason=reasons[source['entry_id']];sid=source['source_request_id']
            if (c,sid) in rows:
                require(rows[c,sid]['completion_ns']<row['arrival_ns'],'candidate source not completed before target')
                if 'token_ids' in source:require(source['token_ids']==tasks[sid]['token_ids'],'candidate source tokens differ from source request')
                if 'request_sha256' in source:require(source['request_sha256']==rows[c,sid]['request_sha256'],'candidate source receipt differs')
            if c=='SEMCACHE_INTEGRATED' and 'cache_before' in row:
                resident=source['entry_id'] in row['cache_before']['lru_to_mru']
            else:resident=source['entry_id'] in residency[key] if key in residency else UNKNOWN
            d=candidate_diagnostic(source,target,source_length=reason.get('prefix_length'),target_length=row['input_tokens'],resident=resident,
                registry=b['registry'],actual_reason=reason['rejection'],windows=len(x['matching_windows']) if 'matching_windows' in x else UNKNOWN)
            reason_check={'TARGET_NOT_LONGER_THAN_SOURCE':'source_strictly_shorter','EXACT_PREFIX_TOKEN_MISMATCH':'exact_full_source_prefix','INCOMPLETE_SOURCE':'source_completed','INCOMPATIBLE_PHYSICAL_IDENTITY':'physical_identity_compatible','STALE_EVICTED_ENTRY':'resident','CROSS_ADAPTER_REUSE_FORBIDDEN':'adapter_compatible'}.get(reason['rejection'])
            if reason_check:require(d['checks'][reason_check] is not True,'actual rejection contradicts observable diagnostic')
            if d['unobservable_checks']:gaps.append(str(key)+':'+','.join(d['unobservable_checks']))
            d.update(block_id=block['block_id'],experiment_id=config['experiment_id'],block_index=b['index'],dataset=block['dataset'],track=block['track'],
                repetition=block['repetition'],condition=c,request_id=rid,entry_id=source['entry_id'],candidate_rank=rank,
                scored=row['scored'],ranking_rule='LONGEST_SOURCE_THEN_ENTRY_ID; limit 64',source_order_before_target=True if (c,sid) in rows else UNKNOWN,
                source_cluster_id=source.get('cluster_id',UNKNOWN),target_cluster_id=e.get('assignment',{}).get('cluster_id',UNKNOWN) if e else UNKNOWN,
                token_evidence='RECORDED_METADATA_AND_RAW_REQUEST' if key in b['requests'] else 'RECORDED_SOURCE_METADATA_AND_PREPARED_TARGET_WITH_MATCHED_IDENTITY_RECEIPTS',
                condition_physical_reuse_enabled=c=='SEMCACHE_INTEGRATED')
            these.append(d);candidates.append(d)
        if e and all('token_ids' in x['metadata'] for x in e['candidates']):
            keys=[(-len(x['metadata']['token_ids']),x['metadata']['entry_id']) for x in e['candidates']]
            require(keys==sorted(keys) and len(keys)<=64,'candidate ranking/limit differs from recorded policy')
        request_records.append(dict(block_id=block['block_id'],block_index=b['index'],experiment_id=config['experiment_id'],dataset=block['dataset'],
            track=block['track'],repetition=block['repetition'],condition=c,request_id=rid,user_id=row['user_id'],adapter_id=row['adapter_id'],
            group_id=task.get('group_id',UNKNOWN),sample_id=task['sample_id'],scored=row['scored'],input_tokens=row['input_tokens'],
            truncated=row.get('truncated',UNKNOWN),original_context_tokens=row.get('original_context_tokens',UNKNOWN),
            retained_context_tokens=row.get('retained_context_tokens',UNKNOWN),removed_context_tokens=row.get('removed_context_tokens',UNKNOWN),
            candidate_count=n,has_candidates=n>0,has_safety_eligible_candidate=any(x['could_pass_unchanged_safety_policy'] is True for x in these),
            eligibility_unknown=any(x['could_pass_unchanged_safety_policy']==UNKNOWN for x in these),
            actual_hit=row.get('source') is not None,reused_prefix_tokens=row['reused_prefix_tokens'],
            admissions=int(row.get('admission') is not None),evictions=row.get('eviction_count',0),
            cross_adapter_rejections=row.get('rejected_cross_adapter_candidates',0),
            no_candidates=n==0,adapter_failure_candidates=sum(x['checks']['adapter_compatible'] is False for x in these),
            length_failure_candidates=sum(x['checks']['source_strictly_shorter'] is False for x in these),
            token_prefix_failure_candidates=sum(x['checks']['exact_full_source_prefix'] is False for x in these),
            exact_prefix_other_safety_failures=sum(x['checks']['exact_full_source_prefix'] is True and bool(x['independent_diagnostic_reasons']) for x in these),
            original=row,target=target))
    b.update(tasks=tasks,rows_by_id=rows,metadata=metadata,diagnostic_requests=request_records,diagnostic_candidates=candidates)
    return request_records,candidates,gaps


def aggregate_requests(rows,candidates):
    ids={(r['block_id'],r['condition'],r['request_id']) for r in rows}
    cs=[x for x in candidates if (x['block_id'],x['condition'],x['request_id']) in ids]
    count=len(rows);input_tokens=sum(r['input_tokens'] for r in rows);reused=sum(r['reused_prefix_tokens'] for r in rows)
    return dict(scored_requests=count,input_length_histogram=dict(sorted(Counter(r['input_tokens'] for r in rows).items())),
        truncation=dict(numerator=sum(r.get('truncated') is True for r in rows),denominator=count,unknown=sum(r.get('truncated',UNKNOWN)==UNKNOWN for r in rows)),requests_with_semantic_candidates=sum(r['has_candidates'] for r in rows),
        candidate_occurrences=sum(r['candidate_count'] for r in rows),distinct_candidate_cache_entries=len({(x['block_id'],x['condition'],x['source_adapter'],x['entry_id']) for x in cs}),
        requests_with_observable_safety_eligible_candidate=sum(r['has_safety_eligible_candidate'] for r in rows),requests_with_eligibility_unknown=sum(r['eligibility_unknown'] for r in rows),
        actual_physical_hit_requests=sum(r['actual_hit'] for r in rows),total_reused_prefix_tokens=reused,total_input_tokens=input_tokens,
        candidate_request_rate=dict(numerator=sum(r['has_candidates'] for r in rows),denominator=count,unit='requests'),
        physical_hit_rate=dict(numerator=sum(r['actual_hit'] for r in rows),denominator=count,unit='requests'),
        input_reuse_fraction=dict(numerator=reused,denominator=input_tokens,ratio=reused/input_tokens if input_tokens else None,unit='input_tokens'),
        admissions=sum(r['admissions'] for r in rows),evictions=sum(r['evictions'] for r in rows),cross_adapter_rejections=sum(r['cross_adapter_rejections'] for r in rows),
        no_candidate_requests=sum(r['no_candidates'] for r in rows),candidate_actual_rejection_reasons=dict(Counter(str(x['actual_rejection_reason']) for x in cs)),
        overlapping_diagnostic_candidate_failures=dict(Counter(reason for x in cs for reason in x['independent_diagnostic_reasons'])),
        equal_length_candidates=sum(x['source_length_equal_target'] is True for x in cs),exact_sequence_candidates=sum(x['exact_full_sequence_equal'] is True for x in cs),
        exact_full_prefix_candidates=sum(x['checks']['exact_full_source_prefix'] is True for x in cs),
        longest_common_prefix_tokens=stats([x['longest_common_prefix_tokens'] for x in cs if isinstance(x['longest_common_prefix_tokens'],int)]))


def latency(requests):
    groups=defaultdict(list)
    for r in requests:
        if r['scored']:groups[r['condition']].append(r)
    distributions={c:{k:stats([r['original'][k] for r in rs if isinstance(r['original'].get(k),(float,int))]) for k in TIMINGS} for c,rs in sorted(groups.items())}
    by={(r['block_id'],r['condition'],r['request_id']):r for r in requests if r['scored']};paired={}
    for a,b in ((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2])):
        records=[]
        for block,c,rid in sorted(by):
            if c!=a:continue
            x=by[block,a,rid];y=by[block,b,rid];metrics={}
            for k in TIMINGS:
                xv=x['original'].get(k);yv=y['original'].get(k)
                if isinstance(xv,(int,float)) and isinstance(yv,(int,float)):
                    metrics[k]=dict(reference=xv,candidate=yv,delta=yv-xv,percentage_delta=100*(yv-xv)/xv if xv else None)
            records.append(dict(block_id=block,request_id=rid,dataset=x['dataset'],repetition=x['repetition'],metrics=metrics))
        summaries={}
        for k in TIMINGS:
            vals=[r['metrics'][k] for r in records if k in r['metrics']]
            baseline=sum(v['reference'] for v in vals)
            summaries[k]=dict(paired_n=len(vals),absolute_delta=stats([v['delta'] for v in vals]),
                percentage_delta=stats([v['percentage_delta'] for v in vals if v['percentage_delta'] is not None]),
                aggregate_percentage_delta=100*sum(v['delta'] for v in vals)/baseline if baseline else None)
        paired[a+'_vs_'+b]=dict(summary=summaries,requests=records)
    return dict(distributions=distributions,paired=paired,timing_warning='Instrumented wall timings overlap/nest; do not add timers or label model_execution_e2e as pure kernel time. Admission duration is NOT isolated GPU-copy time.',
        network_scope='Actual application framing and payload bytes over localhost TCP, not a geographically distributed network',confidence_intervals='NOT_COMPUTED: descriptive paired observations only; conversation/repetition/block dependence retained')


def opportunity_scan(requests):
    """Token-only counterfactual scan; never count these as measured HITs."""
    c=CONDITIONS[2];rs=sorted([r for r in requests if r['condition']==c],key=lambda r:(r['block_index'],r['original']['arrival_ns']))
    results=[]
    for i,target in enumerate(rs):
        for source in rs[:i]:
            if (source['dataset'],source['track'],source['repetition'])!=(target['dataset'],target['track'],target['repetition']):continue
            st=source['target']['token_ids'];tt=target['target']['token_ids']
            if not (len(st)<len(tt) and tt[:len(st)]==st):continue
            cross_block=source['block_id']!=target['block_id']
            admission=source['original'].get('admission');snapshot=target['original'].get('cache_before',{})
            resident=admission['entry_id'] in snapshot['lru_to_mru'] if admission and 'lru_to_mru' in snapshot and not cross_block else UNKNOWN
            results.append(dict(source_block=source['block_id'],target_block=target['block_id'],source_request_id=source['request_id'],target_request_id=target['request_id'],
                dataset=target['dataset'],source_adapter=source['adapter_id'],target_adapter=target['adapter_id'],adapter_compatible=source['adapter_id']==target['adapter_id'],
                category='TOKEN_PREFIX_SEPARATED_BY_BLOCK_BOUNDARY' if cross_block else 'ADMITTED_EXACT_TOKEN_PREFIX_NOT_RESIDENT' if admission and resident is False else 'WITHIN_BLOCK_TOKEN_PREFIX_OPPORTUNITY',
                registry_positions_physical_safety='NOT_AUTHORIZED_BY_THIS_SCAN',source_admitted=bool(admission),source_resident=resident,
                cache_eviction_proven=False,interpretation='Token opportunity only; missing residency is not proof of eviction without an EVICT event. Cross-block order is plan index, not comparable wall clocks.'))
    return results


def diagnose(blocks,*,allow_partial=False,input_mode='AUDIT_ZIP'):
    requests=[];candidates=[];gaps=[]
    for b in blocks:
        rs,cs,gs=prepare_block(b,allow_partial);requests.extend(rs);candidates.extend(cs);gaps.extend(gs+b['missing'])
    for r in requests:
        for metric in TIMINGS:
            if not isinstance(r['original'].get(metric),(int,float)):gaps.append(str((r['block_id'],r['condition'],r['request_id']))+':'+metric)
    require(allow_partial or not gaps,'optional diagnostic evidence missing; use --allow-partial to label unavailable checks explicitly')
    scored=[r for r in requests if r['scored']];groupings={}
    for dimensions in (('condition',),('dataset','condition'),('block_index','condition'),('adapter_id','condition'),('dataset','block_index','adapter_id','condition')):
        grouped=defaultdict(list)
        for r in scored:grouped[tuple(r[k] for k in dimensions)].append(r)
        groupings['by_'+'_'.join(dimensions)]=[dict(zip(dimensions,key),metrics=aggregate_requests(rs,candidates)) for key,rs in sorted(grouped.items(),key=lambda x:str(x[0]))]
    integrated=[r for r in scored if r['condition']==CONDITIONS[2]]
    cs=[x for x in candidates if x['condition']==CONDITIONS[2] and x['scored']]
    work=dict(scope='SCORED_REQUESTS_PER_CONDITION; repetitions retained, not unique independent documents',
        experiment_id=blocks[0]['config']['experiment_id'],blocks=len(blocks),condition_observations=len(scored),integrated=aggregate_requests(integrated,candidates),
        groups=groupings,scored_input_lengths={c:stats([r['input_tokens'] for r in scored if r['condition']==c]) for c in CONDITIONS},
        integrated_truncation=dict(numerator=sum(r['truncated'] is True for r in integrated),denominator=len(integrated)),
        integrated_context_lengths={k:stats([r[k] for r in integrated if isinstance(r[k],(int,float))]) for k in ('original_context_tokens','retained_context_tokens','removed_context_tokens')},
        distinct_samples_by_dataset={d:len({r['sample_id'] for r in integrated if r['dataset']==d}) for d in sorted({r['dataset'] for r in integrated})},
        workload_blocks=[dict(block_index=b['index'],block_id=b['config']['block']['block_id'],dataset=b['config']['block']['dataset'],
            track=b['config']['block']['track'],repetition=b['config']['block']['repetition'],
            cache_lifetime_requests=b['config']['block'].get('cache_lifetime_requests',UNKNOWN),reset=b['config']['block'].get('reset',UNKNOWN),
            continuity_breaks=b['config']['block'].get('continuity_breaks',UNKNOWN),
            cluster_ids=sorted({str(x['target_cluster_id']) for x in candidates if x['block_id']==b['config']['block']['block_id']}),
            scale_semantics=b['config']['block'].get('scale_semantics',UNKNOWN)) for b in blocks],
        routing=dict(logical_users_observed=sorted({r['user_id'] for r in integrated}),adapters_observed=sorted({r['adapter_id'] for r in integrated}),
            physical_device_inference='Logical users do not imply thirty physical devices'),
        priming_requests_by_condition={c:sum(not r['scored'] and r['condition']==c for r in requests) for c in CONDITIONS})
    opportunities=opportunity_scan(requests)
    # Confirm eviction from chronological EVICT receipts; missing entries alone do not prove it.
    for op in opportunities:
        if op['category']=='ADMITTED_EXACT_TOKEN_PREFIX_NOT_RESIDENT':
            b=next(b for b in blocks if b['config']['block']['block_id']==op['target_block'])
            target=b['rows_by_id'][CONDITIONS[2],op['target_request_id']]
            op['cache_eviction_proven']=any(e['condition']==CONDITIONS[2] and e['event']['kind']=='EVICT' and
                e['event'].get('source_request_id')==op['source_request_id'] and e['event'].get('monotonic_ns',float('inf'))<target['arrival_ns'] for e in b['events'])
    spans=defaultdict(set)
    for r in integrated:spans[(r['dataset'],r['track'],r['repetition'],r['group_id'])].add(r['block_index'])
    work['observed_groups_spanning_blocks']=[dict(dataset=k[0],track=k[1],repetition=k[2],group_id=k[3],block_indices=sorted(v)) for k,v in sorted(spans.items()) if len(v)>1]
    work['input_limit_setting']=dict(user_reported=512,recorded_config_values=[b['config'].get('input_limit',UNKNOWN) for b in blocks],interpretation='Uniform observed lengths do not independently recover a missing preparation CLI flag')
    work['observed_cluster_counts']=sorted({e['event']['assignment']['cluster_before']['num_clusters'] for b in blocks for e in b['events'] if e['event']['kind']=='DISCOVERY' and 'cluster_before' in e['event']['assignment']})
    work['independent_cache_resets_all_conditions']=3*len(blocks)
    work['exported_quality_fidelity_gates']=[dict(block_index=b['index'],decision=b['reported_decision']) for b in blocks]
    work['integrated_admission_eviction_by_dataset']={d:dict(admissions=sum(r['admissions'] for r in integrated if r['dataset']==d),evictions=sum(r['evictions'] for r in integrated if r['dataset']==d)) for d in sorted({r['dataset'] for r in integrated})}
    work['cache_occupancy_by_block_condition_adapter']=[]
    for b in blocks:
        for c in CONDITIONS:
            for aid in sorted({r['adapter_id'] for r in b['rows'] if r['condition']==c}):
                rs=sorted([r for r in b['rows'] if r['condition']==c and r['adapter_id']==aid],key=lambda r:r['arrival_ns'])
                snapshots=[r['cache_after'] for r in rs if 'cache_after' in r]
                work['cache_occupancy_by_block_condition_adapter'].append(dict(block_index=b['index'],condition=c,adapter_id=aid,
                    final_occupancy_bytes=snapshots[-1].get('occupancy_bytes',UNKNOWN) if snapshots else UNKNOWN,
                    peak_occupancy_bytes=max((s['peak_occupancy_bytes'] for s in snapshots if 'peak_occupancy_bytes' in s),default=UNKNOWN),
                    evidence='RECORDED_REQUEST_CACHE_SNAPSHOTS; not a fresh GPU allocation check'))
    work['opportunity_categories_by_dataset_block_adapter']=[dict(dataset=k[0],block_id=k[1],adapter_id=k[2],counts=dict(v)) for k,v in sorted(((key,Counter(o['category'] for o in opportunities if (o['dataset'],o['target_block'],o['target_adapter'])==key)) for key in sorted({(r['dataset'],r['block_id'],r['adapter_id']) for r in integrated})),key=lambda x:str(x[0]))]
    work['opportunity_scan']=dict(token_prefix_pairs=len(opportunities),eviction_proven_pairs=sum(o['cache_eviction_proven'] for o in opportunities),
        cross_block_token_prefix_pairs=sum(o['category']=='TOKEN_PREFIX_SEPARATED_BY_BLOCK_BOUNDARY' for o in opportunities),
        warning='Pair counts are counterfactual token observations, not additional requests or authorized HITs; repetitions never bridged.')
    timings=latency(requests)
    timings['by_dataset']={d:latency([r for r in requests if r['dataset']==d]) for d in sorted({r['dataset'] for r in requests})}
    timings['by_repetition']={str(rep):latency([r for r in requests if r['repetition']==rep]) for rep in sorted({r['repetition'] for r in requests})}
    timings['per_block_startup_warmup']=[dict(block_index=b['index'],condition=c,initialization=v.get('initialization',UNKNOWN),warmup_total_ms=v.get('warmup_total_ms',UNKNOWN)) for b in blocks for c,v in b.get('reported_audit',{}).get('conditions',{}).items()]
    prov=provenance(blocks)
    require(not any(b['verification']=='RECORDED_SOURCE_MISMATCH' for b in prov['blocks']),'recorded source inventory contradicts available historical Git blobs')
    inventories={c:{p:d for p,d in next(b['config']['sources'] for b in blocks if str(b['config'].get('source_commit'))==c).items()} for c in prov['historical_groups']}
    if len(inventories)>1:
        first=next(iter(inventories.values()))
        prov['recorded_inventory_differences_from_first']={c:sorted(p for p in set(first)|set(inv) if first.get(p)!=inv.get(p)) for c,inv in inventories.items()}
    provided=dict(integrated_requests=60,requests_with_candidates=28,candidate_occurrences=34,actual_hits=0,reused_tokens=0,
                  app_e2e_delta_percent=5.46,admission_eviction_mean_ms=1900)
    observed=dict(integrated_requests=len(integrated),requests_with_candidates=sum(r['has_candidates'] for r in integrated),candidate_occurrences=len(cs),
        actual_hits=sum(r['actual_hit'] for r in integrated),reused_tokens=sum(r['reused_prefix_tokens'] for r in integrated),
        app_e2e_delta_percent=timings['paired'][CONDITIONS[0]+'_vs_'+CONDITIONS[2]]['summary']['application_e2e_ms']['aggregate_percentage_delta'],
        admission_eviction_mean_ms=timings['distributions'][CONDITIONS[2]]['cache_admission_eviction_ms']['mean'])
    work['historical_handoff_comparison']=dict(user_reported=provided,recalculated=observed,
        exact_count_discrepancies={k:dict(reported=provided[k],observed=observed[k]) for k in list(provided)[:5] if provided[k]!=observed[k]},
        approximate_timing_deltas={k:observed[k]-provided[k] if observed[k] is not None else UNKNOWN for k in list(provided)[5:]})
    coverage=dict(raw_GPU_tensors_available_for_fresh_verification=False,registry_permissions_available=all(b['registry'] is not None for b in blocks),input_mode=input_mode,optional_partial_mode_requested=allow_partial,partial_observability=bool(gaps),
        evidence_gaps=sorted(set(gaps)),candidate_unknown_checks=dict(Counter(k for x in candidates for k in x['unobservable_checks'])),
        input_manifests={b['root']:b['integrity'] for b in blocks},
        original_arrivals_available=sum(len(b['requests']) for b in blocks),
        historical_audit_pass_claims=[b.get('reported_audit',{}).get('status',UNKNOWN) for b in blocks],
        fresh_independent_GPU_evidence_audit_performed=False,TEST_FIXTURE_ONLY=all(b['config'].get('fixture',False) for b in blocks))
    recommendations=dict(stage='R5-I5R-1_DESIGN_CONSTRAINTS_ONLY',GPU_execution_required_for_R0=False,
        GPU_execution_required_for_future_measured_benefit=True,verified=dict(
            equal_length_integrated_candidates=sum(x['source_length_equal_target'] is True for x in cs),
            full_prefix_integrated_candidates=sum(x['checks']['exact_full_source_prefix'] is True for x in cs),
            length_failure_integrated_candidates=sum(x['checks']['source_strictly_shorter'] is False for x in cs)),
        hypotheses=['Fixed-length preparation can remove shorter-source opportunities; this is not a measured intervention on the same requests.',
            'clusters=1 plus common instruction token windows can produce candidates without useful context-prefix similarity.',
            'Independent resets and evictions can reduce opportunities only where the token/ordering/residency scan supports them.'],
        constraints=['Keep Natural and Controlled tracks separate and report zero natural HITs honestly.',
            'Use actual tokenizer IDs and genuine completed standalone priming prefixes; target must have a nonempty suffix.',
            'Reserve generation/context budget; use variable eligible input lengths or explicit controlled prefixes, never truncate gold/question/instructions.',
            'Do not authorize equal-length, partial-LCP, cross-adapter, evicted or cross-block reuse based on this report.',
            'Before GPU scheduling, CPU-enumerate exact-prefix, adapter, completion, positions/mask, ownership, residency and capacity constraints.',
            'For 64/128/256/512 (1024 only if feasible), include priming/admission costs, model startup/warmup and matched A/B/C instrumented measurements.',
            'Profile admission hashing, validation, serialization, synchronization, TCP control and copies separately before attributing its whole duration to GPU copying.',
            'Use existing generated-quality, token/text and frozen numerical gates separately. ROOT_CAUSE_UNRESOLVED remains; no correctness-qualified speedup claim.'])
    return dict(candidates=candidates,requests=requests,workload=work,latency=timings,provenance=prov,coverage=coverage,
                opportunities=opportunities,recommendations=recommendations)


def write_outputs(output,result):
    output=Path(output)
    for name,key in [('workload_summary','workload'),('latency_diagnostics','latency'),('source_provenance','provenance'),('evidence_coverage','coverage'),('recommendations','recommendations')]:
        (output/f'i5r0_{name}.json').write_text(json.dumps(result[key],sort_keys=True,indent=2,allow_nan=False)+'\n')
    (output/'i5r0_rejection_breakdown.json').write_text(json.dumps(result['workload']['groups'],sort_keys=True,indent=2)+'\n')
    for name,rows in [('candidate_diagnostics',result['candidates']),('opportunity_scan',result['opportunities'])]:
        with (output/f'i5r0_{name}.jsonl').open('x') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    def csv_rows(name,rows):
        with (output/f'i5r0_{name}.csv').open('x',newline='') as f:
            keys=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    csv_rows('candidate_diagnostics',result['candidates'])
    csv_rows('request_diagnostics',[dict({k:v for k,v in r.items() if k not in ('original','target')},**{k:r['original'].get(k,UNKNOWN) for k in TIMINGS}) for r in result['requests']])
    w=result['workload'];o=w['historical_handoff_comparison']['recalculated'];c=w['integrated']
    report=f'''# R5-I5R-0 read-only diagnosis

Evidence: {result['coverage']['input_mode']}; {w['blocks']} blocks; partial observability: {result['coverage']['partial_observability']}.
Input SHA inventories verified where supplied; this is NOT a fresh independent GPU tensor audit.
Confidence is high for hash-verified exported counts and observable token comparisons; authorization without the registry and fresh physical GPU proof remain unavailable.
CPU fixture input: {result['coverage']['TEST_FIXTURE_ONLY']}. Source-object availability and all hash mismatches are explicit in source_provenance.json.

## Reproduced observations

{json.dumps(o,indent=2)}

Scored observations across conditions: {w['condition_observations']}. Candidate requests and candidate occurrences have distinct denominators.
Equal-length integrated candidate occurrences: {c['equal_length_candidates']}; full-source prefix matches: {c['exact_full_prefix_candidates']}; full-sequence equality: {c['exact_sequence_candidates']}.
Longest common prefix tokens: {json.dumps(c['longest_common_prefix_tokens'])}.
Truncation: {json.dumps(w['integrated_truncation'])}; input lengths: {json.dumps(w['scored_input_lengths'])}.

## Findings and limits

Length and token-prefix conditions were evaluated independently, not inferred from the first rejection code.
A strict length failure alone is sufficient to deny reuse under the unchanged rule. A non-prefix candidate would still fail if that length rule alone were relaxed.
Equal fixed input lengths are a structural constraint on completed full-source reuse; these records do not establish a causal performance improvement from changing truncation.
Common short token prefixes and w=3 candidates do not prove useful document-prefix similarity; cluster IDs and candidate ranks are reported.
Counterfactual token opportunities, eviction evidence and block boundaries: {json.dumps(w['opportunity_scan'])}.
No-candidate requests, overlapping failure categories and actual rejections are broken down by dataset/block/adapter/condition.
A field marked {UNKNOWN} is not an affirmative eligibility result. No speculative reuse grant is issued.

Foreign cross-adapter probe rejections: {c['cross_adapter_rejections']}; these are privileged aggregate probes, not the {c['candidate_occurrences']} ordinary integrated candidate occurrences. Per-source foreign tokens/identities are not supplied by those aggregate probes.
Observed cluster counts: {w['observed_cluster_counts']}; independent condition cache resets: {w['independent_cache_resets_all_conditions']}. Groups split across blocks (excluding separate repetitions): {json.dumps(w['observed_groups_spanning_blocks'])}.
The preparation input_limit flag itself is {UNKNOWN} where not persisted; uniform 512-token dispatch and truncation are independently counted.

## Exported quality/fidelity gates and source limitations

{json.dumps(w['exported_quality_fidelity_gates'],indent=2)}

Recorded source snapshot groups: {json.dumps(result['provenance']['historical_groups'])}. Historical Git blob verification: {json.dumps(dict(Counter(x['verification'] for x in result['provenance']['blocks'])))}.
Only recorded source-inventory differences are observable when a Git object is absent; a current working file is not substituted for that historical object.

## Latency interpretation

Application and model timings are separate instrumented wall measurements, not pure kernel execution time.
Paired candidate-minus-reference deltas (A-B, B-C, A-C) use matched block/request identities; positive E2E delta means slower.
Nested timers are not additive. Admission/eviction time is not isolated GPU-copy time. TCP bytes are localhost application framing/payload, not WAN measurements.
Per-dataset and repetition/block observations remain available; no independent-sample confidence interval is asserted.

## Next stage

R0 requires no GPU execution. R1 constraints are in recommendations.json: genuine completed prefixes, nonempty suffixes, same-adapter authorization, exact positions/masks, capacity, isolated reset semantics, and separately counted priming costs.
Future latency break-even and task/fidelity claims require new matched GPU measurements. No R1 success or SemCache speedup is claimed here.
Exported numerical/token PASS on a zero-HIT workload does not establish physical-HIT numerical fidelity. Historical numerical status remains ROOT_CAUSE_UNRESOLVED. Original raw tensors, wire ledgers, source Git objects or registry permissions absent from a ZIP require additional original artifacts; exported audit PASS does not replace their fresh verification.
'''
    (output/'i5r0_report.md').write_text(report)
    hashes={p.name:sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'i5r0_output_manifest.json').write_text(json.dumps(dict(files=hashes),sort_keys=True,indent=2)+'\n')


def run(*,runs_root=None,audit_zip=None,output_dir,allow_partial=False,expected_blocks=12):
    require(bool(runs_root)!=bool(audit_zip),'select exactly one input mode')
    inp=Path(runs_root or audit_zip).resolve();out=Path(output_dir).resolve()
    require(not out.exists(),'output must be a fresh directory')
    require(not out.is_relative_to(inp) and not inp.is_relative_to(out),'output overlaps input')
    require(not out.is_relative_to(ROOT) or out.is_relative_to(ROOT/'results/r5/i5r'),'repository output must be under results/r5/i5r')
    files=Files(inp,zip_mode=bool(audit_zip))
    try:
        blocks=load_blocks(files,'AUDIT_ZIP' if audit_zip else 'RAW_RUNS',allow_partial,expected_blocks)
        result=diagnose(blocks,allow_partial=allow_partial,input_mode='AUDIT_ZIP' if audit_zip else 'RAW_RUNS')
        result['coverage']['input_path']=str(inp)
        result['coverage']['zip_sha256']=sha(inp.read_bytes()) if audit_zip else None
        out.mkdir(parents=True,mode=0o700);write_outputs(out,result)
        return result
    finally:files.close()
