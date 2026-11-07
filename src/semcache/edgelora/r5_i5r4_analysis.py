"""Independent CPU replay. Evidence validity is separate from benefit/fidelity."""
from collections import OrderedDict,defaultdict
from copy import deepcopy
from pathlib import Path
import json,math,statistics,time

from .communication import digest,digest_bytes,require
from .r3a_artifacts import read,write,sha,THRESHOLDS
from .r5_i5_evidence import seal,jsonl
from .r5_i5r2_tensors import Evidence,safe_path,file_sha
from .r5_i5r4 import load,cpu_receipt,output
from .r5_i5r4_policy import Engine,CAPACITY
from .r5_i5_contracts import validate_request,prompt
from .r5_i5_audit import audit_adapter
from .r5_i5_analysis import replay_cache,distribution,grouped_ci,METRICS
from .r5_multi_analysis import unique,candidates,fidelity
from .r5_multi_artifacts import validate,runtime_view
from .r5_multi_controller import reason,public_result
from .r5_i5_cache import I5Owner
from .r5_i5_controller import DisabledIndex
from .r4c_analysis import clean
from .r5_perf_analysis import accounting_audit
from .r5_i5r3bbe import sequence

def audit_json(ev,name):
    # NF1's historical JSON reader remains frozen at 16MiB. I4's bounded compact
    # whole-campaign tables may be larger; all tensor files stay in child seals.
    require(name in ev.files and ev.files[name]<=128*2**20,'campaign JSON exceeds 128MiB bound')
    from .r5_i5r0_io import parse
    return parse(safe_path(ev.root,name).read_bytes())

def seal_bundle(root):
    """Small coordinator inventory references separately sealed <=16GiB children.
    Never build one recursive >128GiB Evidence object for a campaign.
    """
    root=Path(root);require(not (root/'i4_bundle_manifest.json').exists(),'bundle already sealed')
    children={}
    for p in sorted((root/'children').iterdir()):
        require(p.is_dir() and not p.is_symlink(),'unsafe child directory')
        if not (p/'i5_output_manifest.json').exists():seal(p) # Fresh failed partial child only.
        children[p.name]=dict(manifest_sha256=sha(p/'i5_output_manifest.json'))
    files={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    require(not any(p.is_symlink() for p in root.iterdir()),'bundle symlink')
    write(root/'i4_bundle_manifest.json',dict(schema='I4_CHILD_INVENTORY_V1',files=files,children=children))

def bundle(root):
    root=Path(root);require(root.is_dir() and not root.is_symlink(),'missing/unsafe campaign')
    require(not any(p.is_symlink() for p in root.iterdir()),'bundle symlink')
    m=read(root/'i4_bundle_manifest.json');require(m['schema']=='I4_CHILD_INVENTORY_V1','bundle schema')
    require(set(p.name for p in root.iterdir())==set(m['files'])|{'children','i4_bundle_manifest.json'},'unexpected/missing coordinator file')
    for name,h in m['files'].items():require(file_sha(safe_path(root,name))==h,'campaign metadata SHA mismatch')
    require(set(p.name for p in (root/'children').iterdir())==set(m['children']),'missing/extra policy child')
    receipts={}
    for name,c in m['children'].items():
        path=safe_path(root,'children/'+name);ev=Evidence(path,max_input_bytes=17*2**30);r=ev.verify()
        require(r['manifest_sha256']==c['manifest_sha256'],'child seal changed');receipts[name]=r
    return dict(manifest_sha256=sha(root/'i4_bundle_manifest.json'),children=receipts)

def policy_replay(path,plan,requests,rows):
    """Recompute causal policy from arrivals and actual physical selection only.
    Independent physical/index/semantic proof replay below validates selection.
    """
    logs=jsonl(path/'i4_policy.jsonl');by={rid:[] for rid in rows}
    for row in logs:
        require(row['request_id'] in by,'orphan policy receipt');by[row['request_id']].append(row)
    engines={};expected_removals={};overhead=[]
    events=jsonl(path/'events.jsonl');cache=[x['event'] for x in events if x['stream']=='cache']
    for rid,row in rows.items():
        req=requests[rid];aid=req['adapter_id'];e=engines.setdefault(aid,Engine(plan['i4']['policy'],plan['i4']['parameters'],CAPACITY));rs=by[rid]
        require(all(r['policy']==e.policy and r['adapter_id']==aid for r in rs),'policy receipt namespace differs')
        cursor=0
        def observe(kind,fields=()):
            nonlocal cursor
            require(cursor<len(rs) and rs[cursor]['kind']==kind and rs[cursor]['state']==e.snapshot(),'policy causal state/receipt differs at '+rid+'/'+kind)
            r=rs[cursor];cursor+=1
            for k,v in fields:require(r[k]==v,'policy decision differs: '+k)
            return r
        require(row['cache_before']['lru_to_mru']==list(e.entries),'cache-before / policy state differs')
        expired=e.begin(rid);observe('BEGIN',[('expired',expired)])
        for key in expired:e.remove(key)
        retention=observe('RETENTION_DONE',[('expired',expired)]);overhead.append(retention['incremental_policy_ms'])
        if row['source']:e.hit(row['source']['entry_id'])
        observe('LOOKUP_DONE',[('selected',None if row['source'] is None else row['source']['entry_id'])])
        selected=row['source'] is not None;size=len(req['token_ids'])*3*req['physical_identity']['layers']*req['physical_identity']['hidden_size']*(4 if req['TEST_FIXTURE_ONLY'] else 2)
        expected_admission=False;victims=[];replacement=None
        if e.policy!='P5' and not selected and len(req['token_ids'])<2048:
            d=e.admission(digest(req['token_ids']),size)
            observe('ADMISSION_DECISION',[('decision',d),('candidate_bytes',size),('token_hash',digest(req['token_ids']))])
            expected_admission=d['allowed']
        if expected_admission:
            require(row['admission'] is not None,'expected physical admission missing')
            key=row['admission']['entry_id']
            if key in e.entries:replacement=key;e.remove(key)
            victims=e.victims(size)
            for k in victims:e.remove(k)
            e.admitted(key,digest(req['token_ids']),size)
        else:require(row['admission'] is None,'unexpected cost-policy admission')
        observe('COMPLETE',[('admitted',None if row['admission'] is None else row['admission']['entry_id'])]);require(cursor==len(rs),'extra policy decisions')
        actual=[x for x in cache if x['request_id']==rid and x['kind'] in ('EVICT','INVALIDATE','REJECT')]
        expected=[(k,'EVICT','I4_IDLE_TTL_EXPIRED') for k in expired]
        if replacement:expected.append((replacement,'INVALIDATE','explicit cold source replacement'))
        expected.extend((k,'EVICT',('capacity_lru' if e.policy=='P0' else 'I4_PRESELECTED_'+e.policy)) for k in victims)
        require([(x['context_sha256'],x['kind'],x['reason']) for x in actual]==expected,'TTL/victim/replacement differs or hidden lower-level eviction')
        require(row['cache_after']['lru_to_mru']==list(e.entries) and row['cache_after']['occupancy_bytes']==sum(x['bytes'] for x in e.entries.values()),'policy cache-after differs')
        expected_removals[rid]=[x[0] for x in expected]
    return dict(status='PASS',expected_removals=expected_removals,policy_retention_including_expiration_ms=distribution(overhead),
                policy_engine_compute_ms=distribution([x.get('engine_compute_ms',x.get('decision_compute_ms',0.)) for x in logs if x['kind'] in ('BEGIN','LOOKUP_DONE','ADMISSION_DECISION')]),
                overhead_scope='Compute timers omit receipt IO; retention includes actual TTL release/TCP. All are nested in application E2E and must not be added to it.',baseline='ORIGINAL_I5_OWNER' if plan['i4']['policy']=='P0' else None)

def physical_replay(path,plan,registry,adapters,pm,m,reqs,rows,steps,allowed):
    """Original strict I5 structural checks, with explicitly verified victims.
    P0/P5 additionally run the frozen legacy replayer without any normalization.
    """
    resident={a:OrderedDict() for a in adapters};indices={a:{} for a in adapters};owners={};cluster={};discovered=set();hits=set();reuses=set();admissions=set();peak=0
    ownership=unique(jsonl(path/'ownership.jsonl'),'source_request_id');isolation=unique(jsonl(path/'isolation.jsonl'),'request_id')
    for aid,spec in adapters.items():
        ident=read(path/aid/'identity.json')
        owners[aid]=I5Owner(runtime_view(pm,spec),digest(dict(preparation=m,adapter=spec)),ident['physical_identity']['gpu_uuid'],plan['device'],CAPACITY,index=DisabledIndex(),tokenizer=m['tokenizer'],sources=plan['sources'],fixture=plan['fixture'],contract=ident['contract'],observed_cache_type=None,spec=spec,registry=registry)
    removals=defaultdict(list);residencies=[]
    for sequence_number,item in enumerate(jsonl(path/'events.jsonl')):
        require(item['sequence']==sequence_number and item['adapter_id'] in adapters,'cache event order/namespace differs')
        aid=item['adapter_id'];state=resident[aid];idx=indices[aid];e=item['event'];kind=e['kind'];key=e.get('context_sha256')
        if item['stream']=='semantic':
            require(plan['condition']!='FULL_RECOMPUTE','P5 executed semantics')
            if kind=='INDEX_INSERT':
                meta=e['metadata'];rid=meta['source_request_id'];req=reqs[rid]
                require(meta['entry_id'] in state and meta['entry_id'] not in idx and meta['completed'] and meta['token_ids']==req['token_ids'] and meta['positions']==req['positions'] and meta['attention_mask']==req['attention_mask'] and meta['physical_identity']==req['physical_identity'] and meta['source_user']==req['user_id'] and meta['request_sha256']==req['request_sha256'] and meta['proof_sha256']==rows[rid]['admission']['proof_sha256'],'index metadata/proof differs')
                require(rows[rid]['source'] is None and rows[rid]['model_end_ns']<=item['observed_ns'],'incomplete/nonstandalone indexed source');idx[meta['entry_id']]=meta
            elif kind=='INDEX_INVALIDATE':require(idx.pop(e['entry_id'],None) is not None,'stale index invalidation')
            else:
                require(kind=='DISCOVERY','unknown semantic event');rid=e['request_id'];req=reqs[rid];a=e['assignment'];before=a['cluster_before']
                from semcache.semantic.intent_clusterer import IntentClusterer
                if aid not in cluster:
                    c=IntentClusterer(1,before['update_interval'],update_mode=before['update_mode']);c.initialize(before['centroids'],counts=before['counts']);cluster[aid]=c
                c=cluster[aid];require(c.queries==before['queries'] and c.centroids==before['centroids'] and c.counts==before['counts'],'cluster state discontinuity')
                observed=c.observe_with_diagnostics(a['embedding']);require(all(a[k]==v for k,v in observed.items()) and c.centroids==a['cluster_after']['centroids'] and c.counts==a['cluster_after']['counts'],'actual clustering replay differs')
                cs=candidates(idx,req);rs=[dict(entry_id=x['metadata']['entry_id'],source_request_id=x['metadata']['source_request_id'],prefix_length=len(x['metadata']['token_ids']),rejection=reason(x['metadata'],req,state,registry)) for x in cs];eligible=[r['entry_id'] for r in rs if r['rejection'] is None];chosen=eligible[0] if eligible else None
                require(e['candidates']==cs and e['candidate_reasons']==rs and rows[rid]['candidate_reasons']==rs and e['virtual_selected_entry_id']==chosen and rows[rid]['virtual_selected_entry_id']==chosen and rows[rid]['candidate_count']==len(cs) and rows[rid]['exact_prefix_eligible']==bool(eligible),'semantic/exact-prefix/physical selection differs')
                require((rows[rid]['source'] is None if chosen is None else rows[rid]['source']['entry_id']==chosen),'false or unconsumed candidate HIT')
                foreign=[dict(foreign_adapter_id=other,rejected_count=len(candidates(indices[other],req)),reason='CROSS_ADAPTER_REUSE_FORBIDDEN') for other in sorted(adapters) if other!=aid and candidates(indices[other],req)]
                require(isolation[rid]['rejected']==foreign and isolation[rid]['foreign_tensor_reads']==isolation[rid]['foreign_wire_frames']==0 and rows[rid]['rejected_cross_adapter_candidates']==sum(f['rejected_count'] for f in foreign),'cross-adapter isolation differs')
                discovered.add(rid)
        else:
            require(item['stream']=='cache','unknown event stream');rid=e['request_id']
            if kind in ('EVICT','INVALIDATE','REJECT'):
                require(key in state and key not in idx,'release did not invalidate index');removals[rid].append(key);old=state.pop(key)
                require(e['monotonic_ns']>=old['born_ns'],'negative residency interval')
                residencies.append(dict(entry_id=key,adapter_id=aid,source_request_id=old['source_request_id'],wall_residency_ms=(e['monotonic_ns']-old['born_ns'])/1e6,right_censored=False,release_reason=e['reason']))
            elif kind=='ADMISSION' and e['allowed']:
                require(rid in ownership,'unowned cache admission');o=ownership[rid];req=reqs[rid];s=steps[rid][0];expected=owners[aid].context_for(prompt(req));size=len(req['token_ids'])*3*pm['identity']['layers']*pm['identity']['hidden_size']*(4 if plan['fixture'] else 2)
                require(o['context']==expected and o['entry_id']==key==digest(expected) and o['source_user']==req['user_id'] and o['adapter_id']==aid and o['registry_sha256']==registry['sha256'] and o['sharing_authorization']==registry['users'][req['user_id']] and o['physical_bytes']==e['candidate_bytes']==size and o['tensor_specs']==o['seal']['specs']=={k:clean(v) for k,v in s['qkv'].items()} and o['proof']==s['raw_kv_proof'] and digest(o['proof'])==rows[rid]['admission']['proof_sha256'] and rows[rid]['model_end_ns']<=e['monotonic_ns'],'physical ownership/source proof differs')
                for l,proof in enumerate(o['proof']['layers']):
                    require(proof['bitwise_equal'] and proof['layer']==l,'invalid proof layer')
                    for role in 'kv':require(proof['raw_'+role]==clean(s['qkv'][f'{l}.{role}']) and proof['actual_'+role]==proof['derived_'+role]==clean(s['kv'][l][role]),'Raw/consumed proof differs')
                require(key not in state,'duplicate physical admission');state[key]=dict(bytes=size,source_request_id=rid,born_ns=e['monotonic_ns']);admissions.add(rid)
            elif kind in ('HIT','PHYSICAL_REUSE','INSERT'):
                require(key in state and e['source_request_id']==state[key]['source_request_id'],'stale/foreign physical entry')
                if kind=='HIT':require(rows[rid]['source']['entry_id']==key,'HIT grant differs');state.move_to_end(key);hits.add(rid)
                if kind=='PHYSICAL_REUSE':reuses.add(rid)
            else:require(kind=='ADMISSION_DENIED' or kind=='ADMISSION' and not e['allowed'],'unknown physical cache event')
            require(e['occupancy_bytes']==sum(s['bytes'] for s in state.values()) and e['lru_to_mru']==list(state),'occupancy/order replay differs')
        require(sum(v['bytes'] for v in state.values())<=CAPACITY and len(state)<=64,'physical capacity/entries exceeded');peak=max(peak,sum(sum(v['bytes'] for v in s.values()) for s in resident.values()))
    require(hits==reuses=={r for r in rows if rows[r]['source']} and admissions==set(ownership),'physical receipts incomplete')
    require(discovered==(set() if plan['condition']=='FULL_RECOMPUTE' else set(rows)),'discovery coverage differs')
    require(all(removals.get(rid,[])==ks for rid,ks in allowed.items()),'policy release mismatch')
    for aid in adapters:
        controls=jsonl(path/aid/'es/controls.jsonl');actual=[x['entry_id'] for x in controls if x['action']=='EVICT_PREFIX']
        released=[x['event']['context_sha256'] for x in jsonl(path/'events.jsonl') if x['adapter_id']==aid and x['stream']=='cache' and x['event']['kind'] in ('EVICT','INVALIDATE','REJECT')]
        require(actual==released,'proof/UD grant invalidation chain differs')
    for aid in adapters:
        require(read(path/aid/'final.json')['cache']['occupancy_bytes']==sum(x['bytes'] for x in resident[aid].values()),'final cache state differs')
        for key,x in resident[aid].items():residencies.append(dict(entry_id=key,adapter_id=aid,source_request_id=x['source_request_id'],wall_residency_ms=(max(r['completion_ns'] for r in rows.values())-x['born_ns'])/1e6,right_censored=True,release_reason='RESIDENT_AT_LAST_REQUEST; process exit disposes session/cache'))
    if plan['i4']['policy'] in ('P0','P5'):replay_cache(path,plan,registry,adapters,pm,m,reqs,rows,steps)
    releases=[e['event'] for e in jsonl(path/'events.jsonl') if e['stream']=='cache']
    return dict(status='PASS',actual_hits=len(hits),admissions=len(admissions),peak_bytes=peak,negative_misses=sum(r['role'] in ('NEGATIVE_TARGET','CROSS_ADAPTER_NEGATIVE') and not r['source'] for r in rows.values()),policy_releases=dict(removals),
                actual_evictions=sum(e['kind']=='EVICT' for e in releases),actual_TTL_expirations=sum(e.get('reason')=='I4_IDLE_TTL_EXPIRED' for e in releases),
                actual_admission_denials=sum(e['kind']=='ADMISSION_DENIED' for e in releases),wall_residencies=residencies,wall_residency_scope='Per-entry admission/release monotonic observations; descriptive only, never a TTL operand')

def audit_child(path,j,p):
    path=Path(path);status=read(path/'i4_child_status.json');require(status==dict(status='COMPLETE',child_id=j['child_id'],policy=j['policy'],plan_id=p['plan_id']),'incomplete/wrong child')
    c=path/'condition';plan=read(c/'plan.json');life=read(c/'lifecycle.json');result=read(c/'result.json')
    require(plan['block']==j['block'] and plan['sources']==p['source']['files'] and plan['source_commit']==p['source_commit'] and plan['i4']==dict(schema=p['schema'],source=p['source'],plan_id=p['plan_id'],policy=j['policy'],parameters=p['policy_parameters'],seed=p['seed']),'policy/runtime source/plan differs')
    require(read(c/'i4_policy_configuration.json')==dict(policy=j['policy'],parameters=p['policy_parameters'],base_configuration_scope='Original I5 configuration.json is preserved; actual opt-in owner/controller policy is specified here',native_arithmetic='UNCHANGED',capacity_per_adapter_bytes=CAPACITY,max_entries=64),'active opt-in policy configuration differs')
    require(life['returncode']==0 and not life['forced_termination'] and not life['pipe_errors'] and result['status']=='COMPLETE' and result['base_models_loaded']==1 and result['es_lora_parameters']==0 and result['sequential'] and result['gpu_inference_performed']==(not plan['fixture']),'execution lifecycle differs')
    account=accounting_audit(c);require(account['budget_bytes']==j['block']['resources']['evidence_budget_bytes'],'child accounting budget differs')
    m,pm,adapters,registry=validate(p['multi_prepared']);tasks={t['request_id']:t for t in j['block']['tasks']};rows=unique(jsonl(c/'generation.jsonl'),'request_id')
    arrivals=[x for x in jsonl(c/'requests.jsonl') if x['kind']=='ARRIVAL'];completions=[x['request_id'] for x in jsonl(c/'requests.jsonl') if x['kind']=='COMPLETION'];reqs={x['request']['request_id']:validate_request(x['request']) for x in arrivals}
    require(list(rows)==list(reqs)==completions==list(tasks) and len(arrivals)==len(tasks),'request order/coverage differs')
    warm=jsonl(c/'warmup.jsonl');require(len(warm)==3,'warmup coverage differs');reqs.update({w['request']['request_id']:validate_request(w['request']) for w in warm})
    stats={};steps={};sessions=[]
    from .r5_i5_gpu import verify_receipt
    gpu=read(c/'gpu_identity.json')
    backend=read(c/'i4_backend.json') if not plan['fixture'] else {'TEST_FIXTURE_ONLY':True}
    if not plan['fixture']:require(backend['deterministic'] and not backend['allow_tf32'] and not backend['cudnn_allow_tf32'] and backend['matmul_precision']=='highest' and backend['cpu_threads']==4 and backend['CUBLAS_WORKSPACE_CONFIG']==':4096:8','supported effective runtime backend differs')
    for aid,spec in adapters.items():
        st,ss=audit_adapter(c,aid,plan,registry,spec,pm,m['tokenizer'],reqs,{rid:r for rid,r in rows.items() if r['adapter_id']==aid});stats[aid]=st;steps.update(ss);sessions.append(st['session_id'])
        verify_receipt(gpu,pm,read(c/aid/'identity.json')['gpu'],plan['gpu_identity_policy'],fixture=plan['fixture'])
    require(gpu['execution_gpu_uuid']==plan['expected_execution_gpu_uuid'],'actual GPU mismatch')
    q=unique(jsonl(c/'quality.jsonl'),'request_id');require(set(q)=={rid for rid,t in tasks.items() if t['scored']},'quality sample completeness differs')
    from .r5_i5_quality import score
    for rid,t in tasks.items():
        r=rows[rid];req=reqs[rid]
        require(req['token_ids']==t['token_ids'] and req['input_text']==t['input_text'] and req['generation_policy']==t['policy'] and req['user_id']==t['user_id'] and req['adapter_id']==t['adapter_id'] and r['scored']==t['scored'] and req['sample_id']==t['sample_id'] and read(c/'client_results'/(rid+'.json'))==public_result(r),'matched original request differs')
        require(r['matched_identity_sha256']==digest(dict(task=t,adapter_sha256=adapters[t['adapter_id']]['matrix_sha256'],tokenizer=m['tokenizer'])),'matched request seal differs')
        require(r['evidence_profile']==('STRICT_AUDIT' if t['strict'] else 'SCALABLE_EVALUATION'),'instrumentation scope differs')
        if r['source']:require(r['reused_prefix_tokens']==512 and not r['cached_prefix_q_consumed'],'invalid physical prefix/Q consumption')
        if t['role'] in ('NEGATIVE_TARGET','CROSS_ADAPTER_NEGATIVE'):require(r['source'] is None,'negative control consumed a prefix')
        if j['policy']=='P5':require(r['semantic_disabled'] and r['candidate_count']==0 and r['semantic_encoding_ms']==0 and r['source'] is r['admission'] is None,'P5 not full target-only')
        if t['scored']:require(q[rid]==dict(score(t['prepared']['sample'],r['generated_text'],t['prepared']['template_version']),request_id=rid,condition=plan['condition'],truncated=r['truncated'],hit=r['source'] is not None),'Qasper score differs')
    negative=read(c/'negative_controls.json');require(negative['wire_frames']==0 and len(negative['probes'])==3 and all(x['outcome']=='EXPECTED_REJECTION' and not x['generated_request'] for x in negative['probes']),'unauthorized routing probes missing')
    replay=policy_replay(c,plan,reqs,rows);physical=physical_replay(c,plan,registry,adapters,pm,m,reqs,rows,steps,replay['expected_removals'])
    # Scalable evidence is an integrity/receipt audit, never an all-tensor proof.
    return dict(child_id=j['child_id'],policy=j['policy'],segment=j['block']['segment'],trace_unit=j['block']['trace_unit'],rotation=j['block']['repetition'],
                rows=list(rows.values()),quality=list(q.values()),physical=physical,policy_replay=replay,adapters=stats,
                sessions=sessions,gpu=gpu,backend=backend,lifecycle=life,initialization=read(c/'initialization.json'),warmup_ms=sum(w['elapsed_ms'] for w in warm),
                NONEMPTY_GENERATED_OUTPUTS='PASS' if all(r['generated_token_ids'] and r['generated_text'].strip() for r in rows.values() if r['scored']) else 'FAIL',
                statistical_unit='Trace/context group, not repeated request',path=str(path.resolve()))

def comparisons(p,children):
    units=defaultdict(dict);sentinels={};numerical=[];pairs=[]
    for c in children:
        if c['segment']=='MEASUREMENT':units[c['trace_unit'],c['rotation']][c['policy']]=c
        else:sentinels[c['policy']]=c
    for key,group in list(units.items())+[(('SENTINEL',0),sentinels)]:
        require(set(group)==set(p['policies']),'missing policy in matched comparison')
        for reference in ('P0','P5'):
            base=group[reference];br={r['request_id']:r for r in base['rows'] if r['scored']}
            for policy,c in group.items():
                if policy==reference:continue
                cr={r['request_id']:r for r in c['rows'] if r['scored']};require(set(br)==set(cr),'target denominator differs')
                for rid,r in cr.items():
                    a=br[rid];require(a['sample_id']==r['sample_id'] and a['user_id']==r['user_id'] and a['adapter_id']==r['adapter_id'],'unmatched target original route')
                    n=fidelity(a,r,Path(base['path'])/'condition'/a['adapter_id']/'es',Path(c['path'])/'condition'/r['adapter_id']/'es')
                    numerical.append(dict(unit=key,reference=reference,policy=policy,request_id=rid,**n))
                if key[0]=='SENTINEL':continue
                bs=sequence(base['rows']);cs=sequence(c['rows']);aq={q['request_id']:q for q in base['quality']};cq={q['request_id']:q for q in c['quality']}
                deltas=[100*(cq[rid]['score']-aq[rid]['score']) for rid in cr]
                # Equal weight per genuine sample within context; repeated
                # requests remain in workload cost, not extra QA sample weight.
                by_sample=defaultdict(list);sample_context={}
                task_context={t['request_id']:digest_bytes(t['prepared']['sample']['context'].encode()) for j in p['jobs'] if j['child_id']==base['child_id'] for t in j['block']['tasks']}
                for rid in cr:
                    sample=cr[rid]['sample_id'];by_sample[sample].append(100*(cq[rid]['score']-aq[rid]['score']));sample_context[sample]=task_context[rid]
                by_context=defaultdict(list)
                for sample,xs in by_sample.items():by_context[sample_context[sample]].append(statistics.mean(xs))
                unique_quality_delta=statistics.mean(statistics.mean(xs) for xs in by_context.values())
                positive=[rid for rid,r in cr.items() if r['role'] not in ('NEGATIVE_TARGET','CROSS_ADAPTER_NEGATIVE')]
                target_delta=statistics.mean(cr[rid]['application_e2e_ms']-br[rid]['application_e2e_ms'] for rid in positive)
                source_delta=sum(r['application_e2e_ms'] for r in c['rows'] if not r['scored'])-sum(r['application_e2e_ms'] for r in base['rows'] if not r['scored'])
                positive_saving=sum(br[rid]['application_e2e_ms']-cr[rid]['application_e2e_ms'] for rid in positive)
                negative_overhead=sum(cr[rid]['application_e2e_ms']-br[rid]['application_e2e_ms'] for rid in cr if rid not in positive)
                require(abs((cs['span_ms']-bs['span_ms'])-(source_delta-positive_saving+negative_overhead+cs['gaps_ms']-bs['gaps_ms']))<1e-5,'source-inclusive economic decomposition differs')
                pairs.append(dict(trace_unit=key[0],rotation=key[1],reference=reference,policy=policy,
                                  source_inclusive_sequence_ms=cs['span_ms'],reference_sequence_ms=bs['span_ms'],D_ms=cs['span_ms']-bs['span_ms'],
                                  source_ms=sum(r['application_e2e_ms'] for r in c['rows'] if not r['scored']),
                                  source_admission_nested_ms=sum(r['cache_admission_eviction_ms'] for r in c['rows'] if not r['scored']),
                                  reference_source_ms=sum(r['application_e2e_ms'] for r in base['rows'] if not r['scored']),
                                  source_cost_delta_ms=source_delta,positive_target_savings_ms=positive_saving,negative_target_overhead_ms=negative_overhead,
                                  scored_target_intervals_ms=sum(r['application_e2e_ms'] for r in cr.values()),
                                  reference_target_intervals_ms=sum(r['application_e2e_ms'] for r in br.values()),
                                  negative_intervals_ms=sum(r['application_e2e_ms'] for r in cr.values() if r['role'] in ('NEGATIVE_TARGET','CROSS_ADAPTER_NEGATIVE')),
                                  gap_delta_ms=cs['gaps_ms']-bs['gaps_ms'],scored_requests=len(cr),unique_scored_samples=len({r['sample_id'] for r in cr.values()}),
                                  amortized_ms_per_scored_request=cs['span_ms']/len(cr),warm_target_mean_delta_ms=target_delta,
                                  warm_target_scope='All scored positive-arrival targets, including actual misses; negatives excluded, HIT rate separate',
                                  physical_hit_target_delta_ms=distribution([cr[rid]['application_e2e_ms']-br[rid]['application_e2e_ms'] for rid in cr if cr[rid]['source']]),
                                  process_lifecycle_ms=(c['lifecycle']['execution_end_ns']-c['lifecycle']['execution_start_ns'])/1e6,
                                  model_initialization_ms=c['initialization']['model_and_session_initialization_ms'],warmup_ms=c['warmup_ms'],
                                  quality_delta_f1_points=unique_quality_delta,quality_request_weighted_delta_f1_points=statistics.mean(deltas),quality_per_request_deltas=dict(zip(cr,deltas)),
                                  request_metrics={k:distribution([r[k] for r in cr.values()]) for k in METRICS},
                                  total_wire_bytes=sum(r['total_request_wire_bytes'] for r in c['rows']),
                                  source_inclusive_vs_target_only=reference=='P5',localhost_TCP=True,overlapping_timers_not_added=True))
    return pairs,numerical

def aggregate(p,pairs,numerical):
    groups={}
    for reference in ('P0','P5'):
        for policy in p['policies']:
            rs=[r for r in pairs if r['reference']==reference and r['policy']==policy]
            if not rs:continue
            # Collapse dependent rotations first; equal weight per independent trace.
            collapsed=[]
            for trace in sorted({r['trace_unit'] for r in rs}):
                xs=[r for r in rs if r['trace_unit']==trace];collapsed.append(dict(independent_block_group=trace,absolute_delta=statistics.mean(x['D_ms'] for x in xs),quality_delta=statistics.mean(x['quality_delta_f1_points'] for x in xs)))
            ci=grouped_ci(collapsed);qi=grouped_ci(collapsed,value='quality_delta');n=[x for x in numerical if x['reference']==reference and x['policy']==policy]
            quality_status='DESCRIPTIVE_ONLY' if len(collapsed)<p['quality']['minimum_independent_traces_for_confirmatory'] else 'PASS' if qi['available'] and qi['lower']>=-p['quality']['margin_f1_points'] else 'NOT_ESTABLISHED'
            groups[reference+'_vs_'+policy]=dict(independent_traces=len(collapsed),independent_positive_contexts=3*len(collapsed),raw_trace_means=collapsed,
                mean_D_ms=statistics.mean(r['absolute_delta'] for r in collapsed),CI95_D_ms=ci,CI95_quality_f1=qi,
                quality_noninferiority=quality_status,quality_margin_f1_points=p['quality']['margin_f1_points'],
                WARM_TARGET_BENEFIT='OBSERVED_DESCRIPTIVE' if statistics.mean(r['warm_target_mean_delta_ms'] for r in rs)<0 else 'FAIL',
                AMORTIZED_E2E_BENEFIT='OBSERVED_DESCRIPTIVE' if statistics.mean(r['D_ms'] for r in rs)<0 else 'FAIL',
                NUMERICAL_FIDELITY='PASS' if n and all(x['frozen_tolerance_pass'] for x in n) else 'FAIL',
                TOKEN_TEXT_FIDELITY='PASS' if n and all(x['generated_tokens_equal'] and x['generated_text_equal'] for x in n) else 'FAIL',
                CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED')
    return groups

def policy_totals(p,children):
    out={}
    for policy in p['policies']:
        cs=[c for c in children if c['policy']==policy and c['segment']=='MEASUREMENT'];rows=[r for c in cs for r in c['rows']];scored=[r for r in rows if r['scored']]
        qs=defaultdict(list);contexts={}
        for c in cs:
            tasks=next(j['block']['tasks'] for j in p['jobs'] if j['child_id']==c['child_id']);byid={t['request_id']:t for t in tasks}
            for q in c['quality']:
                t=byid[q['request_id']];qs[t['sample_id']].append(q['score']*100);contexts[t['sample_id']]=digest_bytes(t['prepared']['sample']['context'].encode())
        cq=defaultdict(list)
        for sample,values in qs.items():cq[contexts[sample]].append(statistics.mean(values))
        out[policy]=dict(total_requests=len(rows),scored_request_observations=len(scored),unique_scored_samples=len(qs),
                         independent_trace_units=len({c['trace_unit'] for c in cs}),rotations=p['rotations'],
                         repeated_request_observations=len(scored)-len(qs),independent_positive_contexts=p['independent_positive_contexts'],
                         TASK_QUALITY=dict(metric='Qasper F1',scale='0..100',unique_sample_mean=statistics.mean(statistics.mean(x) for x in qs.values()),
                                           context_mean=statistics.mean(statistics.mean(x) for x in cq.values()),per_context={k:statistics.mean(v) for k,v in cq.items()},adequacy='DESCRIPTIVE_ONLY'),
                         FUNCTIONAL_PHYSICAL_REUSE='PASS' if any(r['source'] for r in rows) else 'NOT_EXERCISED',
                         CACHE_INTEGRITY='PASS_SAMPLED_STRICT_PLUS_ALL_REQUEST_RECEIPTS',actual_HITs=sum(bool(r['source']) for r in scored),actual_MISSes=sum(not r['source'] for r in scored),
                         actual_reused_tokens=sum(r['reused_prefix_tokens'] for r in rows),actual_skipped_projection_rows=sum(r['skipped_prefix_projection_token_rows'] for r in rows),
                         source_inclusive_wire_bytes=sum(r['total_request_wire_bytes'] for r in rows),CPU_LoRA_ms=sum(r['cpu_lora_ms'] for r in rows),
                         observed_evictions=sum(c['physical']['actual_evictions'] for c in cs),TTL_expirations=sum(c['physical']['actual_TTL_expirations'] for c in cs),
                         admission_denials=sum(c['physical']['actual_admission_denials'] for c in cs),
                         GPU_peak_bytes=max((r['gpu_memory']['peak_allocated_bytes'] for r in rows if r['gpu_memory']),default=None),
                         metrics={k:distribution([r[k] for r in scored]) for k in METRICS},
                         CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED')
    return out

def compute(experiment,run,cpu_audit):
    p,pr=load(experiment);cr=cpu_receipt(p,pr,cpu_audit);receipt=bundle(run);root=Path(run);cfg=read(root/'i4_run_config.json');status=read(root/'i4_run_status.json')
    require(cfg['plan_inventory']==pr and cfg['source']==p['source'] and cfg['cpu_audit']==cr and status['status']=='COMPLETE' and status['completed_children']==[j['child_id'] for j in p['jobs']] and status['plan_id']==p['plan_id'] and status['source']==p['source'] and not status['GPU_NOT_RUN'],'incomplete/unmatched campaign')
    require(set(receipt['children'])=={j['child_id'] for j in p['jobs']},'campaign child coverage differs')
    cs=[audit_child(root/'children'/j['child_id'],j,p) for j in p['jobs']]
    sessions=[s for c in cs for s in c['sessions']];require(len(sessions)==len(set(sessions)),'UD session state reused between policies')
    require(all(c['gpu']['execution_gpu_uuid']==status['execution_gpu_uuid'] for c in cs),'matched campaign GPU changed')
    require(all(c['backend']==cs[0]['backend'] for c in cs),'backend changed between policies')
    require(all(a['lifecycle']['execution_end_ns']<=b['lifecycle']['execution_start_ns'] for a,b in zip(cs,cs[1:])),'parallel/reordered GPU policy execution')
    require(all(c['policy']!='P5' or c['physical']['actual_hits']==0 for c in cs),'P5 physical reuse prohibited')
    pairs,numerical=comparisons(p,cs);summary=aggregate(p,pairs,numerical)
    sent=[c for c in cs if c['segment']=='STRICT_SENTINEL'];require(any(c['policy']=='P0' and c['physical']['actual_hits']>=1 for c in sent),'P0 strict HIT sentinel not exercised')
    return dict(schema='I5R4_RESULTS_V1',source=p['source'],plan_id=p['plan_id'],plan_inventory=pr,cpu_audit=cr,run_inventory=receipt,
                children=cs,matched_comparisons=pairs,numerical=dict(thresholds=THRESHOLDS,comparisons=numerical),summary=summary,policy_totals=policy_totals(p,cs),
                quality_contract=p['quality'],measurement_scope=p['instrumentation'],
                independent_audit=dict(status='PASS',GPU_NOT_RUN=False,scope='Actual adapter/TCP/position/proof and causal policy replay; full tensor validation only isolated strict sentinels',
                                       all_request_full_tensor_audit='NOT_SUPPORTED_IN_SCALABLE_EVALUATION',strict_sentinel_hits=sum(c['physical']['actual_hits'] for c in sent),
                                       P0_original_replay='PASS',policy_benefit_independent_of_audit=True),
                ROOT_CAUSE='UNRESOLVED',NATIVE_PEFT_STRICT_PARITY='UNESTABLISHED',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED')

def analyze(experiment,run,cpu_audit,output_dir):
    r=compute(experiment,run,cpu_audit);out=output(output_dir,(experiment,run,cpu_audit))
    require(len(json.dumps(r,allow_nan=False).encode())<=64*2**20,'compact campaign report exceeds 64MiB logical bound')
    for name,value in (('i4_results',r),('i4_independent_audit',r['independent_audit']),('i4_summary',r['summary']),('i4_policy_totals',r['policy_totals']),('i4_numerical_fidelity',r['numerical'])):write(out/(name+'.json'),value)
    with (out/'i4_request_metrics.jsonl').open('x') as f:
        for c in r['children']:
            for row in c['rows']:f.write(json.dumps(dict(policy=c['policy'],segment=c['segment'],trace=c['trace_unit'],rotation=c['rotation'],**row),sort_keys=True,allow_nan=False)+'\n')
    seal(out);return dict(status='EVIDENCE_AUDIT_PASS',policy_results=r['summary'])

def verify(experiment,run,audit,cpu_audit,output_dir):
    # Recompute from raw bytes, not exported pass flags or the simulator.
    actual=compute(experiment,run,cpu_audit);ev=Evidence(audit,max_input_bytes=256*2**20);receipt=ev.verify()
    require(audit_json(ev,'i4_results.json')==actual and ev.json('i4_independent_audit.json')==actual['independent_audit'] and ev.json('i4_summary.json')==actual['summary'] and audit_json(ev,'i4_numerical_fidelity.json')==actual['numerical'],'independent audit/report tampering')
    require(ev.json('i4_policy_totals.json')==actual['policy_totals'],'policy total/task-score export differs')
    expected=[dict(policy=c['policy'],segment=c['segment'],trace=c['trace_unit'],rotation=c['rotation'],**row) for c in actual['children'] for row in c['rows']]
    require(ev.jsonl('i4_request_metrics.jsonl')==expected,'exported request metrics differ from raw evidence')
    out=output(output_dir,(experiment,run,audit,cpu_audit));write(out/'i4_verification.json',dict(status='PASS',source=actual['source'],plan_id=actual['plan_id'],audit_inventory=receipt,run_inventory=actual['run_inventory'],cpu_audit=actual['cpu_audit'],result_sha256=digest(actual),summary=actual['summary'],numerical_failures_preserved=True));seal(out);return dict(status='PASS')

def report(experiment,run,audit,verification,output_dir):
    p,pr=load(experiment);ae=Evidence(audit,max_input_bytes=256*2**20);ar=ae.verify();r=audit_json(ae,'i4_results.json');ve=Evidence(verification);vr=ve.verify();v=ve.json('i4_verification.json')
    require(v['status']=='PASS' and v['source']==r['source']==p['source'] and v['plan_id']==r['plan_id']==p['plan_id'] and v['audit_inventory']==ar and v['result_sha256']==digest(r) and v['summary']==r['summary'],'verification/report binding differs')
    require(bundle(run)==v['run_inventory'] and r['plan_inventory']==pr,'run or plan changed after verification')
    out=output(output_dir,(experiment,run,audit,verification));write(out/'i4_final_decision.json',dict(source=p['source'],plan_id=p['plan_id'],EVIDENCE_VALIDITY='PASS',GPU_NOT_RUN=False,summary=r['summary'],policy_totals=r['policy_totals'],verification_inventory=vr,ROOT_CAUSE='UNRESOLVED',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED'))
    lines=['# I5R-4 integrated campaign','','Evidence validity PASS is not numerical fidelity, quality adequacy or policy benefit.','',json.dumps(r['summary'],indent=2),'',json.dumps(r['policy_totals'],indent=2),'','Scalable timing and strict sentinels are separate. Repeated questions/rotations are dependent observations. P5 has no proactive sources and different arrival traffic. Localhost application TCP bytes are not a geographically distributed network. Historical numerical FAIL remains visible. Kernel cause and native PEFT strict parity remain unresolved.']
    (out/'i4_report.md').write_text('\n'.join(lines)+'\n');seal(out);return dict(status='REPORT_COMPLETE',output=str(out))
