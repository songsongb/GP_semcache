"""Read-only I5 matched identity, TCP, sampled tensors and cache lifecycle audit."""
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
import csv
import math
import random
import statistics

from .communication import digest, require
from .r3a_artifacts import read, write, THRESHOLDS
from .r4c_analysis import clean
from .r5_i5_evidence import verify, jsonl, sources
from .r5_i5_workload import CONDITIONS
from .r5_i5_quality import score, aggregate
from .r5_i5_audit import audit_adapter
from .r5_i5_contracts import validate_request, prompt
from .r5_i5_cache import I5Owner
from .r5_i5_controller import DisabledIndex
from .r5_multi_artifacts import validate, runtime_view
from .r5_multi_controller import reason,public_result
from .r5_multi_analysis import candidates, fidelity, unique
from .r5_perf_analysis import accounting_audit

HISTORY=dict(provenance='USER_VERIFIED_I4A_EXPANDED_HANDOFF',hits=12,requests=30,reused_tokens=132,
    frozen_numerical_hit_failures=12,root_cause='ROOT_CAUSE_UNRESOLVED',native_peft_strict_parity='UNESTABLISHED',
    correctness_qualified_speedup='UNESTABLISHED',I3F_B='FUTURE_WORK')
METRICS=('model_ttft_ms','application_ttft_ms','prefill_ms','decode_ms','model_execution_e2e_ms',
         'application_e2e_ms','semantic_encoding_ms','cluster_assignment_ms','candidate_lookup_ms',
         'eligibility_validation_ms','cache_lookup_ms','kv_reconstruction_ms','cache_admission_eviction_ms',
         'evidence_logging_io_ms','generated_tokens_per_second')


def distribution(values):
    v=sorted(values)
    if not v:return dict(n=0,mean=None,median=None,p95=None,variance=None)
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),p95=v[max(0,math.ceil(.95*len(v))-1)],
                variance=statistics.variance(v) if len(v)>1 else None)


def paired(a,b,key):
    absolute=b[key]-a[key]
    return dict(reference=a[key],candidate=b[key],absolute_delta=absolute,
                percentage_delta=100*absolute/a[key] if a[key] else None)


def grouped_ci(rows, value='absolute_delta', group='independent_block_group', seed=42):
    groups=defaultdict(list)
    for r in rows: groups[r[group]].append(r[value])
    if len(groups)<3:return dict(available=False,reason='fewer than three independent block groups',groups=len(groups))
    # Resample whole independent blocks, retaining all dependent requests and repetitions.
    units=list(groups.values());rng=random.Random(seed);means=[]
    for _ in range(2000):
        draw=[rng.choice(units) for _ in units];means.append(statistics.mean(x for u in draw for x in u))
    means.sort();return dict(available=True,method='paired whole-block cluster bootstrap; 2000 draws',groups=len(groups),
                             lower=means[49],upper=means[1949],level=.95,seed=seed)


def replay_cache(path, plan, registry, adapters, pm, m, reqs, rows, steps):
    condition=plan['condition'];indices={a:{} for a in adapters};resident={a:OrderedDict() for a in adapters}
    peak={a:0 for a in adapters};owners={};discovered={};global_peak=0;ownership=unique(jsonl(path/'ownership.jsonl'),'source_request_id')
    for aid,spec in adapters.items():
        identity=read(path/aid/'identity.json')
        owners[aid]=I5Owner(runtime_view(pm,spec),digest(dict(preparation=m,adapter=spec)),identity['physical_identity']['gpu_uuid'],plan['device'],
            plan['block']['resources']['per_adapter_capacity_bytes'],index=DisabledIndex(),tokenizer=m['tokenizer'],sources=plan['sources'],fixture=plan['fixture'],
            contract=identity['contract'],observed_cache_type=None,spec=spec,registry=registry)
    cluster={};isolation=unique(jsonl(path/'isolation.jsonl'),'request_id');hit_events=set();reuse_events=set();admissions=set()
    for sequence,item in enumerate(jsonl(path/'events.jsonl')):
        require(item['sequence']==sequence and item['adapter_id'] in adapters,'event sequence differs')
        aid=item['adapter_id'];e=item['event'];kind=e['kind'];idx=indices[aid];state=resident[aid];key=e.get('context_sha256')
        if item['stream']=='semantic':
            require(condition!='FULL_RECOMPUTE','A performed semantic computation')
            if kind=='INDEX_INSERT':
                meta=e['metadata'];rid=meta['source_request_id'];req=reqs[rid]
                require(key is None and meta['entry_id'] not in idx and meta['completed'] and meta['token_ids']==req['token_ids'] and
                    meta['physical_identity']==req['physical_identity'] and meta['source_user']==req['user_id'] and meta['request_sha256']==req['request_sha256'], 'index source identity differs')
                require(rows[rid]['source'] is None and rows[rid]['model_end_ns']<=item['observed_ns'], 'index source incomplete or not standalone')
                idx[meta['entry_id']]=meta
            elif kind=='INDEX_INVALIDATE':require(idx.pop(e['entry_id'],None) is not None, 'stale index invalidation')
            else:
                require(kind=='DISCOVERY','unexpected semantic event');rid=e['request_id'];req=reqs[rid]
                a=e['assignment'];before=a['cluster_before']
                from semcache.semantic.intent_clusterer import IntentClusterer
                if aid not in cluster:
                    c=IntentClusterer(1,before['update_interval'],update_mode=before['update_mode']);c.initialize(before['centroids'],counts=before['counts']);cluster[aid]=c
                c=cluster[aid];require(c.queries==before['queries'] and c.centroids==before['centroids'] and c.counts==before['counts'],'semantic state discontinuity')
                observed=c.observe_with_diagnostics(a['embedding'])
                require(all(a[k]==v for k,v in observed.items()) and c.centroids==a['cluster_after']['centroids'] and c.counts==a['cluster_after']['counts'], 'semantic assignment replay failed')
                cs=candidates(idx,req);reasons=[dict(entry_id=x['metadata']['entry_id'],source_request_id=x['metadata']['source_request_id'],
                    prefix_length=len(x['metadata']['token_ids']),rejection=reason(x['metadata'],req,state,registry)) for x in cs]
                eligible=[x['entry_id'] for x in reasons if x['rejection'] is None];chosen=eligible[0] if eligible else None
                require(e['candidates']==cs and e['candidate_reasons']==reasons and rows[rid]['candidate_reasons']==reasons and
                    e['virtual_selected_entry_id']==chosen and rows[rid]['virtual_selected_entry_id']==chosen and
                    rows[rid]['candidate_count']==len(cs) and rows[rid]['exact_prefix_eligible']==bool(eligible),'semantic candidate/eligibility mismatch')
                if chosen:
                    if condition=='SEMCACHE_INTEGRATED':require(rows[rid]['source']['entry_id']==chosen,'physical candidate differs')
                    else:require(rows[rid]['source'] is None,'shadow consumed tensors');state.move_to_end(chosen)
                foreign=[dict(foreign_adapter_id=other,rejected_count=len(candidates(indices[other],req)),reason='CROSS_ADAPTER_REUSE_FORBIDDEN')
                    for other in sorted(adapters) if other!=aid and candidates(indices[other],req)]
                require(isolation[rid]['rejected']==foreign and isolation[rid]['foreign_tensor_reads']==isolation[rid]['foreign_wire_frames']==0 and
                    rows[rid]['rejected_cross_adapter_candidates']==sum(x['rejected_count'] for x in foreign),'cross-adapter isolation replay differs')
                discovered[rid]=e
        else:
            require(item['stream']=='cache','unknown cache event stream');rid=e['request_id']
            if kind in ('EVICT','INVALIDATE','REJECT','SHADOW_EVICT'):
                require(key in state and key not in idx,'untracked release')
                if kind=='EVICT':require(key==next(iter(state)), 'LRU victim mismatch')
                if kind=='SHADOW_EVICT' and key!=digest(owners[aid].context_for(prompt(reqs[rid]))):require(key==next(iter(state)), 'shadow LRU victim mismatch')
                state.pop(key)
            elif kind=='SHADOW_ADMISSION':
                require(condition=='SEMANTIC_NO_REUSE' and key in idx and key not in state,'invalid shadow admission')
                expected=len(reqs[rid]['token_ids'])*3*pm['identity']['layers']*pm['identity']['hidden_size']*(4 if plan['fixture'] else 2)
                require(e['simulated_bytes']==expected,'shadow capacity differs');state[key]=dict(bytes=expected,source_request_id=rid)
            elif kind=='ADMISSION' and e['allowed']:
                require(condition=='SEMCACHE_INTEGRATED' and rid in ownership,'unowned physical admission')
                o=ownership[rid];req=reqs[rid];source=steps[rid][0];expected=owners[aid].context_for(prompt(req))
                expected_bytes=len(req['token_ids'])*3*pm['identity']['layers']*pm['identity']['hidden_size']*(4 if plan['fixture'] else 2)
                require(o['context']==expected and key==o['entry_id']==digest(expected) and o['adapter_id']==aid and o['source_user']==req['user_id'] and
                    o['registry_sha256']==registry['sha256'] and o['sharing_authorization']==registry['users'][req['user_id']] and
                    o['proof']==source['raw_kv_proof'] and o['physical_bytes']==e['candidate_bytes']==expected_bytes and
                    o['tensor_specs']==o['seal']['specs']=={k:clean(v) for k,v in source['qkv'].items()} and
                    rows[rid]['model_end_ns']<=e['monotonic_ns'] and digest(o['proof'])==rows[rid]['admission']['proof_sha256'], 'invalid physical ownership/proof')
                for l,proof in enumerate(o['proof']['layers']):
                    require(proof['bitwise_equal'] and proof['layer']==l,'invalid raw proof layer')
                    for role in 'kv':require(proof['raw_'+role]==clean(source['qkv'][f'{l}.{role}']) and proof['actual_'+role]==proof['derived_'+role]==clean(source['kv'][l][role]),'raw/consumed proof differs')
                require(key not in state,'duplicate physical entry');state[key]=dict(bytes=expected_bytes,source_request_id=rid);admissions.add(rid)
            elif kind in ('HIT','PHYSICAL_REUSE','INSERT'):
                require(condition=='SEMCACHE_INTEGRATED' and key in state and e['source_request_id']==state[key]['source_request_id'],'stale/foreign cache access')
                if kind=='HIT':require(rows[rid]['source']['entry_id']==key,'HIT grant differs');state.move_to_end(key);hit_events.add(rid)
                if kind=='PHYSICAL_REUSE':reuse_events.add(rid)
            else:require(kind=='ADMISSION_DENIED' or kind=='ADMISSION' and not e['allowed'],'unexpected cache event')
            if condition=='SEMCACHE_INTEGRATED':
                require(e['occupancy_bytes']==sum(v['bytes'] for v in state.values()) and e['lru_to_mru']==list(state),'physical occupancy replay differs')
            else:require(e['occupancy_bytes']==0, 'baseline owns physical cache tensors')
        size=sum(v['bytes'] for v in state.values());peak[aid]=max(peak[aid],size)
        require(size<=plan['block']['resources']['per_adapter_capacity_bytes'] and len(state)<=64,'cache exceeds capacity')
        global_peak=max(global_peak,sum(sum(v['bytes'] for v in x.values()) for x in resident.values()))
    actual_hits={rid for rid,r in rows.items() if r['source'] is not None}
    require(actual_hits==hit_events==reuse_events and admissions==set(ownership),'cache receipts incomplete')
    require(set(discovered)==(set() if condition=='FULL_RECOMPUTE' else set(rows)), 'missing or unexpected semantic discovery')
    for aid in adapters:
        final=read(path/aid/'final.json')['cache'];expected=0 if condition!='SEMCACHE_INTEGRATED' else sum(v['bytes'] for v in resident[aid].values())
        require(final['occupancy_bytes']==expected,'final occupancy mismatch')
    physical=condition=='SEMCACHE_INTEGRATED'
    return dict(status='PASS',actual_hits=len(actual_hits),strict_hit_requests=sum(rows[r]['evidence_profile']=='STRICT_AUDIT' for r in actual_hits),
                global_peak_occupancy_bytes=global_peak if physical else 0,per_adapter_peak_bytes=peak if physical else {a:0 for a in adapters},
                shadow_simulated_global_peak_bytes=global_peak if condition=='SEMANTIC_NO_REUSE' else 0,admissions=len(admissions),
                shadow_index_equivalence_scope='metadata/LRU replay; compared with physical condition separately')


def analyze(root):
    root=Path(root);verify(root);require(read(root/'i5_block_status.json')['status']=='COMPLETE','partial/failed block rejected')
    config=read(root/'i5_execution_config.json');block=config['block'];local_workload=read(root/'i5_workload_manifest.json')
    require(config['sources']==sources(),'execution source provenance differs; audit with the recorded source checkout')
    from .r3a_artifacts import sha
    exp_path=Path(config['experiment_prepared']);verify(exp_path)
    require(sha(exp_path/'i5_output_manifest.json')==config['experiment_manifest_sha256'] and
        local_workload==dict(schema='I5_BLOCK_WORKLOAD_V1',experiment_id=config['experiment_id'],block=block),'experiment provenance chain differs')
    experiment=read(exp_path/'i5_workload_manifest.json')
    b=dict(block);identity=b.pop('block_id');require(digest(b)==identity and block in experiment['blocks'],'sealed block identity differs')
    ex=dict(experiment);eid=ex.pop('experiment_id');require(digest(ex)==eid==config['experiment_id'],'experiment identity differs')
    m,pm,adapters,registry=validate(config['multi_prepared']);require(read(root/'i5_adapter_registry.json')==dict(adapters=adapters,registry=registry),'adapter registry changed')
    from .r5_i5_data import load_dataset
    from .r3a_artifacts import sha
    dataset_manifest=read(root/'i5_dataset_manifest.json')[block['dataset']]
    ds_path=Path(dataset_manifest['prepared_path']);originals,_=load_dataset(ds_path,fixture=config['fixture'])
    require(sha(ds_path/'i5_output_manifest.json')==dataset_manifest['prepared_manifest_sha256'],'dataset provenance chain differs')
    original_by_id={s['id']:s for s in originals}
    for task in block['tasks']:
        require(task['prepared']['sample']==original_by_id[task['sample_id']], 'task references differ from immutable dataset')
        require(task['prepared']['token_ids_sha256']==digest(task['prepared']['token_ids']),'prepared token digest differs')
    tasks={t['request_id']:t for t in block['tasks']};allrows={};qualities=[];events=[];audits={};pids=[];sessions=[];allsteps={}
    for condition in CONDITIONS:
        path=root/condition;plan=read(path/'plan.json');require(plan['block']==block and plan['condition']==condition and plan['sources']==config['sources'],'condition plan mismatch')
        require(accounting_audit(path)['budget_bytes']==config['condition_budget_bytes'],'accounting budget differs');life=read(path/'lifecycle.json');result=read(path/'result.json')
        require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'] and result['status']=='COMPLETE' and result['requests']==len(tasks), 'condition lifecycle incomplete')
        require(result['base_models_loaded']==1 and result['es_lora_parameters']==0 and result['sequential'] and result['gpu_inference_performed']==(not config['fixture']), 'base model/runtime execution scope differs')
        negative=read(path/'negative_controls.json')
        require(negative['wire_frames']==0 and len(negative['probes'])==3 and all(p['outcome']=='EXPECTED_REJECTION' and not p['generated_request'] for p in negative['probes']), 'authorization negative controls missing')
        rows=unique(jsonl(path/'generation.jsonl'),'request_id');arrivals=[x for x in jsonl(path/'requests.jsonl') if x['kind']=='ARRIVAL']
        reqs={x['request']['request_id']:validate_request(x['request']) for x in arrivals}
        completions=[x['request_id'] for x in jsonl(path/'requests.jsonl') if x['kind']=='COMPLETION']
        require(list(rows)==list(reqs)==completions==list(tasks) and len(arrivals)==len(tasks),'missing/reordered/duplicate requests')
        warmups=jsonl(path/'warmup.jsonl');require(len(warmups)==3,'warmup coverage differs')
        reqs.update({w['request']['request_id']:validate_request(w['request']) for w in warmups})
        q=unique(jsonl(path/'quality.jsonl'),'request_id');require(set(q)=={rid for rid,t in tasks.items() if t['scored']},'task score completeness failure')
        stats={};steps={}
        for aid,spec in adapters.items():
            subset={rid:r for rid,r in rows.items() if r['adapter_id']==aid}
            st,ss=audit_adapter(path,aid,plan,registry,spec,pm,m['tokenizer'],reqs,subset);stats[aid]=st;steps.update(ss)
            pids.append(st['ud_pid']);sessions.append(st['session_id'])
        pids.append(result['es_pid']);allsteps[condition]=steps
        for rid,row in rows.items():
            task=tasks[rid];req=reqs[rid]
            require(read(path/'client_results'/(rid+'.json'))==public_result(row),'logical client result exposes invalid/private metadata')
            require(req['token_ids']==task['token_ids'] and req['generation_policy']==task['policy'] and req['user_id']==task['user_id'] and
                    req['adapter_id']==task['adapter_id'] and req['input_text']==task['input_text'] and row['scored']==task['scored'] and
                    row['matched_identity_sha256']==digest(dict(task=task,adapter_sha256=adapters[task['adapter_id']]['matrix_sha256'],tokenizer=m['tokenizer'])), 'matched immutable request differs')
            require(row['evidence_profile']==('STRICT_AUDIT' if task['strict'] else 'SCALABLE_EVALUATION'),'strict sampling changed after execution')
            if condition=='FULL_RECOMPUTE':require(row['semantic_disabled'] and row['semantic_encoding_ms']==row['candidate_count']==0 and not row['source'],'A semantics violated')
            if condition!='SEMCACHE_INTEGRATED':require(not row['source'] and not row['admission'] and row['reused_prefix_tokens']==0,'baseline consumed cache tensors')
            if task['scored']:
                expected=dict(score(task['prepared']['sample'],row['generated_text'],task['prepared']['template_version']),request_id=rid,condition=condition,truncated=row['truncated'],hit=row['source'] is not None)
                require(q[rid]==expected,'reference score differs');qualities.append(dict(expected,block_id=identity))
        cache=replay_cache(path,plan,registry,adapters,pm,m,reqs,rows,steps)
        audits[condition]=dict(adapters=stats,cache=cache,initialization=read(path/'initialization.json'),warmup_total_ms=sum(w['elapsed_ms'] for w in warmups))
        allrows[condition]=rows;events.extend(dict(e,condition=condition,block_id=identity) for e in jsonl(path/'events.jsonl'))
    require(len(set(pids))==len(pids) and len(set(sessions))==len(sessions),'process/session state reused between conditions')
    comparisons=[];numerical=[];quality_delta=[]
    qmap={(r['condition'],r['request_id']):r for r in qualities}
    for a,b in ((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2])):
        for rid,t in tasks.items():
            ra,rb=allrows[a][rid],allrows[b][rid]
            require(ra['matched_identity_sha256']==rb['matched_identity_sha256'],'unmatched comparison')
            if not t['scored']:continue
            match=dict(comparison=a+'_vs_'+b,request_id=rid,block_id=identity,dataset=block['dataset'],track=block['track'],
                prefix_length=block['prefix_length'],scale=block['scale'],repetition=block['repetition'],group_id=t['group_id'],
                hit=rb['source'] is not None,independent_block_group=digest([config['experiment_id'],block['dataset'],block['track'],block['prefix_length'],list(tasks)]))
            comparisons.append(dict(match,metrics={k:paired(ra,rb,k) for k in METRICS},wire=paired(ra,rb,'total_request_wire_bytes')))
            quality_delta.append(dict(match,absolute_delta=qmap[b,rid]['score']-qmap[a,rid]['score']))
            numerical.append(dict(match,**fidelity(ra,rb,root/a/ra['adapter_id']/'es',root/b/rb['adapter_id']/'es')))
    shadow_equivalent=all(allrows[CONDITIONS[1]][rid]['virtual_selected_entry_id']==allrows[CONDITIONS[2]][rid]['virtual_selected_entry_id'] for rid in tasks)
    request_rows=[dict(row,block_id=identity,scale=block['scale']) for c in CONDITIONS for row in allrows[c].values()]
    latency={};quality={};network={};cache={}
    for c in CONDITIONS:
        rows=list(allrows[c].values());targets=[r for r in rows if r['scored']];priming=[r for r in rows if not r['scored']]
        latency[c]=dict(metrics={k:distribution([r[k] for r in targets]) for k in METRICS},
            hit_miss={kind:{k:distribution([r[k] for r in targets if bool(r['source'])==hit]) for k in ('application_e2e_ms','application_ttft_ms')} for kind,hit in (('HIT',True),('MISS',False))},
            target_requests=len(targets),priming_requests=len(priming),target_wall_ms=sum(r['application_e2e_ms'] for r in targets),
            priming_wall_ms=sum(r['application_e2e_ms'] for r in priming),
            amortized_ms_per_target=(sum(r['application_e2e_ms'] for r in rows)+audits[c]['warmup_total_ms'])/len(targets),
            requests_per_second=len(targets)/(sum(r['application_e2e_ms'] for r in targets)/1000),
            workload_requests_per_second=len(targets)/(sum(r['application_e2e_ms'] for r in rows)/1000),
            initialization=audits[c]['initialization'],warmup_ms=audits[c]['warmup_total_ms'],
            gpu_peak_bytes=max((r['gpu_memory']['peak_allocated_bytes'] for r in rows if r['gpu_memory']),default=None),
            model_timing_includes_evidence=True,application_e2e_includes_admission=True,
            quality_scoring_and_final_request_serialization_in_e2e=False)
        cq=[r for r in qualities if r['condition']==c];quality[c]=dict(overall=aggregate(cq))
        for label,predicate in (('HIT',lambda r:r['hit']),('MISS',lambda r:not r['hit']),('TRUNCATED',lambda r:r['truncated']),('UNTRUNCATED',lambda r:not r['truncated'])):
            subset=[r for r in cq if predicate(r)];quality[c][label]=aggregate(subset) if subset else None
        network[c]=dict(localhost_tcp=True,wire_scope='application TCP framing including JSON/payload; excludes IP headers/retransmission',
            all_requests={k:sum(r[k] for r in rows) for k in ('es_to_ud_wire_bytes','ud_to_es_wire_bytes','total_request_wire_bytes','control_wire_bytes')},
            scored_targets={k:sum(r[k] for r in targets) for k in ('es_to_ud_wire_bytes','ud_to_es_wire_bytes','total_request_wire_bytes')},
            prefill_wire_bytes=sum(r['prefill']['application_wire_bytes'] for r in rows),decode_wire_bytes=sum(r['decode']['application_wire_bytes'] for r in rows))
        lifecycle_frames=[f for aid in adapters for f in jsonl(root/c/aid/'es/wire.jsonl')]
        network[c]['lifecycle_all_wire_bytes']=sum(f['wire_bytes'] for f in lifecycle_frames)
        network[c]['lifecycle_directions']={op:sum(f['wire_bytes'] for f in lifecycle_frames if f['operation']==op) for op in ('send','recv')}
        network[c]['warmup_handshake_shutdown_wire_bytes']=network[c]['lifecycle_all_wire_bytes']-sum(r['total_request_wire_bytes'] for r in rows)
        cache[c]=dict(audits[c]['cache'],semantic_candidate_rate=statistics.mean(r['candidate_count']>0 for r in targets),
            physical_hit_rate=statistics.mean(r['source'] is not None for r in targets),miss_rate=statistics.mean(r['source'] is None for r in targets),
            exact_prefix_eligibility_rate=statistics.mean(r['exact_prefix_eligible'] for r in targets),
            reused_prefix_tokens=distribution([r['reused_prefix_tokens'] for r in targets if r['source']]),
            total_reused_tokens=sum(r['reused_prefix_tokens'] for r in targets),input_reuse_fraction=sum(r['reused_prefix_tokens'] for r in targets)/sum(r['input_tokens'] for r in targets),
            evictions=sum(r['eviction_count'] for r in rows),cross_adapter_rejections=sum(r['rejected_cross_adapter_candidates'] for r in rows),
            per_adapter={aid:dict(hits=sum(r['source'] is not None for r in rows if r['adapter_id']==aid),final_occupancy_bytes=read(root/c/aid/'final.json')['cache']['occupancy_bytes']) for aid in adapters},
            rejection_reasons=dict(__import__('collections').Counter(x['rejection'] for r in rows for x in r['candidate_reasons'] if x['rejection'])),
            invalid_or_unauthorized_reuse_attempts=0)
    gates=dict(runtime_completion='PASS',cache_correctness='PASS' if cache[CONDITIONS[2]]['actual_hits'] else 'NOT_EXERCISED',
        adapter_isolation='PASS',task_score_completeness='PASS',token_text_fidelity='PASS' if all(n['generated_tokens_equal'] and n['generated_text_equal'] for n in numerical) else 'FAIL',
        numerical_tolerance='PASS' if all(n['frozen_tolerance_pass'] for n in numerical) else 'FAIL',
        strict_sampled_kv_audit='PASS',strict_sampled_hit_audit='PASS' if cache[CONDITIONS[2]]['strict_hit_requests'] else 'NOT_EXERCISED',
        timing_measurement_validity='CPU_FIXTURE_ONLY' if config['fixture'] else 'PASS',evidence_integrity='PASS',
        full_tensor_all_request_audit='NOT_SUPPORTED_IN_SCALABLE_EVALUATION',native_peft_strict_parity='UNESTABLISHED',
        correctness_qualified_speedup=False,GPU_NOT_RUN=config['fixture'],GPU_VERIFIED=not config['fixture'])
    for summary in (quality,latency,network,cache):summary['TEST_FIXTURE_ONLY']=config['fixture']
    return dict(config=config,request_metrics=request_rows,cache_events=events,quality_per_sample=qualities,quality_summary=quality,
        latency_summary=latency,network_summary=network,cache_summary=cache,quality_deltas=quality_delta,paired_latency=comparisons,
        numerical_fidelity=dict(thresholds=THRESHOLDS,comparisons=numerical,history=HISTORY),decision=gates,
        independent_audit=dict(status='PASS',TEST_FIXTURE_ONLY=config['fixture'],scope='TCP/metadata all requests; full Raw-QKV/consumed KV on predetermined strict sample',
            conditions=audits,shadow_index_equivalent=shadow_equivalent,
            logical_users_configured=len(registry['users']),logical_users_observed=sorted({r['user_id'] for r in request_rows}),physical_UD_processes_per_condition=3,
            limitation=None if shadow_equivalent else 'B-C is not a perfectly isolated physical-reuse comparison',
            source_files_match_current_checkout=config['sources']==sources(),strict_all_requests=False))


def export(out,report):
    out=Path(out)
    for name in ('quality_summary','latency_summary','network_summary','cache_summary','numerical_fidelity','decision','independent_audit'):
        write(out/f'i5_{name}.json',report[name])
    write(out/'i5_execution_config.json',report['config'])
    for name in ('request_metrics','cache_events','quality_per_sample','quality_deltas','paired_latency'):
        with (out/f'i5_{name}.jsonl').open('x') as f:
            import json
            for row in report[name]:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    fields=('block_id','condition','dataset','track','scale','repetition','request_id','sample_id','group_id','adapter_id','scored','TEST_FIXTURE_ONLY',
        'prefix_length_target','input_tokens','generated_tokens','reused_prefix_tokens','reused_input_fraction','truncated','total_request_wire_bytes',*METRICS)
    with (out/'i5_plot_data.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in report['request_metrics'])
