"""Read-only, artifact-derived R5 audit. Never trusts a saved success flag."""
from collections import OrderedDict
from pathlib import Path
import math

from .communication import PREFIX, canonical, digest, digest_bytes, require, validate
from .r3a_artifacts import read, sha, unpack, validate_prepared, write
from .r3a_evidence import jsonl
from .r4b2_analysis import raw_head_bytes, tensor_data, token_slice
from .r4c_analysis import append_head_bytes, clean
from .r4c_metrics import numerical
from .r4c_session import ForwardOrder, context
from .r4c_workload import termination
from .r5_contracts import gates, namespace, validate_request
from .r5_evidence import verify
from .r5_semantic_index import eligibility


def audit(directory):
    root=Path(directory);es=root/'es';plan=read(es/'es_plan.json');worker=read(es/'es_result.json')
    identity=read(root/'r5_identity.json');physical=identity['physical_identity'];model=identity['model_identity']
    fixture=plan['fixture'];layers=model['layers'];hidden=model['hidden_size'];element=4 if model['hidden_dtype']=='float32' else 2
    session=read(es/'r5_session.json');cohort=session['cohort_sha256'];capacity=session['capacity_bytes'];contract=identity['contract']
    require(worker['execution_status']=='COMPLETE' and worker['pid']!=worker['ud_pid'] and worker['user_device_cuda_used'] is False and
        worker['gpu_inference_performed']==(not fixture) and identity['TEST_FIXTURE_ONLY']==fixture,'incomplete worker or promoted fixture')
    pm,_,am=validate_prepared(plan['base_prepared'])
    require(sha(Path(plan['base_prepared'])/'r3a_prepare_manifest.json')==plan['base_manifest_sha256'] and pm['identity']==model and
        pm.get('TEST_FIXTURE_ONLY',False)==fixture,'prepared scope/base fingerprint differs')
    require(physical==namespace(pm,identity['tokenizer'],plan['device'],physical['gpu_uuid'],contract),'physical namespace differs')
    if fixture:require(identity['semantic_encoder']['checkpoint']=='controlled_vectors_v1','fixture semantic provenance differs')
    else:
        sem=identity['semantic_encoder']
        require(sem['family']=='TinyBERT' and sem['device']=='cpu' and sem['resolved_revision']==plan['semantic_revision'] and
            len(sem['weights_sha256'])==64 and len(sem['tokenizer_vocab_sha256'])==64,'genuine TinyBERT evidence missing')
        require('RTX 3090' in identity['gpu']['name'] and identity['gpu']['uuid']==pm['gpu']['uuid']==physical['gpu_uuid'],'GPU allocation differs')
        require((es/'gpu_execution.json').exists(),'actual GPU forward evidence missing')
        for method in contract['methods'].values():require(digest_bytes(method['source'].encode())==method['source_sha256'],'installed OPT source digest differs')
    ledger=jsonl(root/'r5_requests.jsonl');arrivals=[e for e in ledger if e['kind']=='ARRIVAL'];completions=[e for e in ledger if e['kind']=='COMPLETION']
    rows=jsonl(root/'r5_generation_outputs.jsonl');by={r['request_id']:r for r in rows}
    requests={e['request']['request_id']:validate_request(e['request']) for e in arrivals}
    require(len(by)==len(rows)==len(arrivals)==len(completions)==20 and len(requests)==len(rows),'missing/duplicate completion or request')
    require([e['request_id'] for e in completions]==list(by)==list(requests),'completion ordering differs')
    for arrival,done,row in zip(arrivals,completions,rows):
        req=arrival['request'];require(req['request_id']==row['request_id']==done['request_id'] and
            req['request_sha256']==row['request_sha256']==done['request_sha256'] and
            row['physical_identity']==req['physical_identity']==physical and row['TEST_FIXTURE_ONLY']==fixture and
            row['user_id']==req['user_id'] and row['adapter_id']==req['adapter_id'],'request identity mismatch')
        require(row['application_e2e_ms']==(row['completion_ns']-row['arrival_ns'])/1e6 and
            row['application_ttft_ms']==(row['first_token_available_ns']-row['arrival_ns'])/1e6 and
            row['arrival_ns']==arrival['arrival_ns'] and row['completion_ns']==done['completion_ns'],'application clocks differ')
    # Independently reconstruct index discovery and every cluster transition.
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.semantic.matcher import ExactTokenMatcher
    from semcache.semantic.subsequence import SubsequenceExtractor
    index={};discoveries={};extractor=SubsequenceExtractor(3);matcher=ExactTokenMatcher();cluster=None
    for event in jsonl(root/'r5_semantic_events.jsonl'):
        kind=event['kind']
        if kind=='INDEX_INSERT':
            m=event['metadata'];key=m['entry_id'];require(key not in index and m['source_request_id'] in by,'duplicate/foreign index source')
            require(m['completed'] and m['physical_identity']==physical and m['source_user']==requests[m['source_request_id']]['user_id'],'index identity differs')
            index[key]=m
        elif kind=='INDEX_INVALIDATE':require(index.pop(event['entry_id'],None) is not None,'stale index invalidation')
        else:
            require(kind=='DISCOVERY','unknown semantic event');rid=event['request_id'];req=requests[rid];assignment=event['assignment']
            before=assignment['cluster_before'];after=assignment['cluster_after']
            if cluster is None:
                cluster=IntentClusterer(before['num_clusters'],before['update_interval'],update_mode=before['update_mode'])
                cluster.initialize(before['centroids'],counts=before['counts'])
            require(cluster.centroids==before['centroids'] and cluster.counts==before['counts'] and cluster.queries==before['queries'],'cluster transition discontinuity')
            observed=cluster.observe_with_diagnostics(assignment['embedding'])
            require(all(assignment[k]==v for k,v in observed.items()) and cluster.centroids==after['centroids'] and cluster.counts==after['counts'],'cluster assignment/update differs')
            require(assignment['encoder']==identity['semantic_encoder'],'semantic encoder provenance changed')
            windows={matcher.key(assignment['cluster_id'],w) for w in extractor.extract(req['token_ids'])}
            expected=sorted([k for k,m in index.items() if any(matcher.key(m['cluster_id'],w) in windows for w in extractor.extract(m['token_ids']))],
                key=lambda k:(-len(index[k]['token_ids']),k))[:64]
            candidates=event['candidates'];require([c['metadata']['entry_id'] for c in candidates]==expected and event['index_entry_ids']==sorted(index),'candidate discovery differs from actual index')
            for candidate in candidates:
                m=candidate['metadata'];require(m==index[m['entry_id']],'stale/altered candidate metadata')
                expected_windows=[dict(start=w.start,end=w.end,tokens=list(w.token_ids)) for w in extractor.extract(req['token_ids'])
                    if matcher.key(m['cluster_id'],w) in {matcher.key(m['cluster_id'],x) for x in extractor.extract(m['token_ids'])}]
                require(candidate['matching_windows']==expected_windows,'candidate token windows differ')
            expected_reasons=[dict(entry_id=index[k]['entry_id'],source_request_id=index[k]['source_request_id'],
                prefix_length=len(index[k]['token_ids']),rejection=eligibility(index[k],req,index)) for k in expected]
            require(event['candidate_reasons']==expected_reasons,'physical eligibility reasons differ')
            eligible=[r['entry_id'] for r in expected_reasons if r['rejection'] is None]
            selected=eligible[0] if eligible and event['physical_reuse_enabled'] else None
            outcome='PHYSICAL_PREFIX_HIT' if selected else 'SEMANTIC_CANDIDATE_REJECTED' if candidates and not eligible else 'CACHE_MISS'
            require(event['selected_entry_id']==selected and event['outcome']==outcome==by[rid]['decision'],'automatic dispatch differs')
            require(by[rid]['candidate_count']==len(candidates) and by[rid]['cluster_id']==assignment['cluster_id'],'semantic metrics differ')
            require(rid not in discoveries,'duplicate discovery');discoveries[rid]=event
    require(set(discoveries)==set(by),'missing semantic path')
    steps=jsonl(es/'r5_tokens.jsonl');bysteps={rid:[s for s in steps if s['request_id']==rid] for rid in by}
    require([r['context'] for r in jsonl(es/'forward_attempts.jsonl')]==[s['context'] for s in steps]==
        [r['context'] for r in jsonl(es/'forward_completed.jsonl')],'hidden/incomplete full-prefill forwards')
    require(jsonl(es/'r4c_logit_diagnostics.jsonl')==[s['logit_diagnostic'] for s in steps],'missing logit diagnostics')
    frames=jsonl(es/'es_wire.jsonl');other=jsonl(es/'ud_wire.jsonl');ud=read(es/'ud_result.json')
    require(len(frames)==len(other) and ud['execution_status']=='COMPLETE' and ud['pid']==worker['ud_pid'] and not ud['user_device_cuda_used'] and
        not ud['full_model_loaded'] and ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='','CPU UD process isolation differs')
    order=ForwardOrder(cohort,layers,contract['context_limit']);pending=None;forward_frames={};controls=[]
    for a,b in zip(frames,other):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and all(a[k]==b[k] for k in ('header','frame_sha256','wire_bytes')),'independent TCP ledgers differ')
        h=a['header'];validate(h,h['payload_bytes'])
        require(h['session_id']==worker['session_id'] and h['adapter_sha256']==am['matrix_sha256'] and
            a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],'TCP identity/bytes differ')
        if h['kind']=='HELLO' and h['request_id']!='handshake':
            require(pending is None,'control during exchange');order.control(h['metadata']);controls.append(dict(h['metadata'],request_id=h['request_id']))
        if h['kind']=='HIDDEN':require(pending is None,'overlapping forwards');order.accept(h,model);pending=h
        if h['kind']=='DELTAS':
            require(pending is not None and h['request_id']==pending['request_id'] and h['layer_id']==pending['layer_id'] and
                h['metadata']['hidden_payload_sha256']==pending['payload_sha256'] and h['metadata']['r4c_forward_sha256']==digest(pending['metadata']['r4c_forward']) and
                h['metadata']['user_device_cuda_used'] is False,'delta/CPU receipt differs')
            order.advance();pending=None
        if h['kind'] in ('HIDDEN','DELTAS'):forward_frames.setdefault(h['request_id'],[]).append(a)
    require(pending is None,'incomplete TCP exchange');order.finish()
    require(controls==jsonl(es/'es_prefix_controls.jsonl')==jsonl(es/'ud_prefix_controls.jsonl'),'source/completion controls differ')
    for operation in ('send','recv'):require([f['header']['sequence'] for f in frames if f['operation']==operation]==list(range(sum(f['operation']==operation for f in frames))),'TCP sequence gap')
    ud_layers=jsonl(es/'ud_layers.jsonl');require(len(ud_layers)==len(steps)*layers and set(forward_frames)=={s['forward_id'] for s in steps},'orphan/missing TCP forwards')
    strict_files=set();kv_checks=prefix_checks=0
    for row in rows:
        rid=row['request_id'];req=requests[rid];ss=bysteps[rid];g=row['generated_token_ids'];target=req['token_ids'];p=row['reused_prefix_tokens'];grant=row['source'];policy=req['generation_policy']
        require(g and len(g)==row['generated_tokens']==len(ss) and row['termination']==termination(policy,g) and order.finished[rid]['generated_token_ids']==g,'incomplete generation/receipt')
        require(row['mode']==('PARTIAL_PREFIX_HIT' if grant else 'COLD_MISS' if row['condition']=='INTEGRATED' else 'NO_REUSE'),'execution differs from decision')
        require((row['decision']=='PHYSICAL_PREFIX_HIT')==(grant is not None),'false physical HIT')
        require(not row['cached_prefix_q_consumed'] and row['projection_kernels_entirely_skipped']==0,'unsupported compute claim')
        if grant:
            source=requests[grant['source_request_id']]
            require(grant['entry_id']==discoveries[rid]['selected_entry_id'] and source['token_ids']==target[:p] and
                source['physical_identity']==req['physical_identity'] and by[source['request_id']]['completion_ns']<row['arrival_ns'],'invalid source prefix/completion')
        blob=(es/row['logits_file']).read_bytes();vv=unpack(blob,'float32');width=contract['vocabulary_size']
        require(digest_bytes(blob)==row['logits_sha256'] and row['logits_shape']==[1,len(g),width] and len(vv)==len(g)*width and all(math.isfinite(v) for v in vv),'incomplete/nonfinite logits')
        reconstructed=None
        for i,s in enumerate(ss):
            old=p if i==0 else len(target)+i-1;ids=target[p:] if i==0 else [g[i-1]]
            require(s['context']==context(rid,i,ids,target,old,policy,row['mode'],cohort,grant,context_limit=contract['context_limit']) and
                s['executed_projection_calls']==3*layers and s['executed_projection_token_rows']==len(ids)*3*layers and
                s['prefix_projection_token_rows_skipped']==(p*3*layers if i==0 else 0),'hidden full-prefill / wrong skipped rows')
            require(s['past']['sequence_length']==len(target)+i and (s['input_past'] is None if old==0 else s['input_past']['sequence_length']==old),'KV continuation length differs')
            kvshape=[1,contract['heads'],len(target)+i,hidden//contract['heads']]
            require(len(s['kv'])==layers and all(t['shape']==kvshape and t['dtype']==model['hidden_dtype'] and t['device']==plan['device'] for pair in s['kv'] for t in pair.values()),'KV shape/dtype/device differs')
            require(len(s['qkv'])==3*layers and all(t['shape']==[1,len(ids),hidden] and t['dtype']==model['hidden_dtype'] and t['device']==plan['device'] for t in s['qkv'].values()),'QKV dimensions differ')
            require(s['position_observation']['positions']==s['context']['positions'] and s['position_observation']['past_length']==old,'position observation differs')
            if not fixture:require(s['position_observation']['lookup_output_equal'] and len(s['components'])==3*layers and all(t['shape']==[1,len(ids),hidden] for t in s['components'].values()),'hidden OPT prefix processing')
            v=vv[i*width:(i+1)*width]
            require(g[i]==s['selected_token_id']==max(range(width),key=v.__getitem__) and s['logits_sha256']==digest_bytes(blob[i*width*4:(i+1)*width*4]),'greedy token/logit differs')
            require(all(s['logit_diagnostic'][k]['nonfinite']==0 for k in ('all_logits','valid_positions','last_position')),'nonfinite diagnostics')
            fs=forward_frames[s['forward_id']];uls=[u for u in ud_layers if u['forward_id']==s['forward_id']]
            require(len(fs)==2*layers and [u['layer'] for u in uls]==list(range(layers)) and all(u['context']==s['context'] for u in uls),'missing UD layer evidence')
            for u in uls:
                a,b=fs[2*u['layer']:2*u['layer']+2]
                require(u['hidden_frame_sha256']==a['frame_sha256'] and u['delta_frame_sha256']==b['frame_sha256'] and u['metrics']==b['header']['metadata'],'UD tensor receipt differs')
            require(s['hidden_payload_bytes']==sum(f['header']['payload_bytes'] for f in fs if f['header']['kind']=='HIDDEN') and
                s['delta_payload_bytes']==sum(f['header']['payload_bytes'] for f in fs if f['header']['kind']=='DELTAS') and
                s['application_wire_bytes']==sum(f['wire_bytes'] for f in fs),'projection payload/wire metrics differ')
            for op,key in (('send','es_to_ud_wire_bytes'),('recv','ud_to_es_wire_bytes')):require(s[key]==sum(f['wire_bytes'] for f in fs if f['operation']==op),'directional forward bytes differ')
            for t in [*s['qkv'].values(),*s['components'].values(),*(t for pair in s['kv'] for t in pair.values())]:
                if 'file' in t:tensor_data(es,t);strict_files.add(t['file'])
            if row['evidence_profile']=='STRICT_AUDIT':
                if i==0:reconstructed=[[tensor_data(es,s['kv'][l][r]) for r in 'kv'] for l in range(layers)]
                for l in range(layers):
                    for j,r in enumerate('kv'):
                        raw=s['qkv'][f'{l}.{r}'];transformed=raw_head_bytes(tensor_data(es,raw),raw['shape'],contract['heads'],element)
                        if i==0:require(token_slice(reconstructed[l][j],s['kv'][l][r],old,len(target))==transformed,'source Raw/HF KV differs')
                        else:reconstructed[l][j]=append_head_bytes(reconstructed[l][j],transformed,heads=contract['heads'],old_length=old,head_dim=hidden//contract['heads'],element_bytes=element)
                        require(digest_bytes(reconstructed[l][j])==s['kv'][l][r]['sha256'],'reconstructed Decode KV differs');kv_checks+=1
            if i:require(s['input_kv']==[[clean(ss[i-1]['kv'][l][r]) for r in 'kv'] for l in range(layers)],'prior Decode history mutated')
            if i==0 and grant:
                source=bysteps[grant['source_request_id']][0];proof=source['raw_kv_proof']
                require(digest(proof)==grant['proof_sha256'],'source proof digest differs')
                for l in range(layers):
                    for j,r in enumerate('kv'):
                        require(s['input_kv'][l][j]==proof['layers'][l]['actual_'+r],'initial HIT KV differs from source')
                        if row['evidence_profile']=='STRICT_AUDIT':
                            require(token_slice(tensor_data(es,s['kv'][l][r]),s['kv'][l][r],0,p)==tensor_data(es,source['kv'][l][r]),'cached prefix bytes differ');prefix_checks+=1
            require(row['model_start_ns']<=s['start_monotonic_ns']<=s['available_monotonic_ns']<=row['model_end_ns'],'step timing outside model')
        require(row['model_execution_e2e_ms']==(row['model_end_ns']-row['model_start_ns'])/1e6 and
            row['model_ttft_ms']==(ss[0]['available_monotonic_ns']-row['model_start_ns'])/1e6 and row['first_token_available_ns']==ss[0]['available_monotonic_ns'],'model timing differs')
        require(row['skipped_prefix_projection_token_rows']==p*3*layers and row['executed_prefill_projection_token_rows']==(len(target)-p)*3*layers,'request skipped work differs')
        for key in row['prefill']:require(row['prefill'][key]==ss[0][key] and row['decode'][key]==sum(s[key] for s in ss[1:]),'request traffic summary differs')
        rf=[f for f in frames if row['arrival_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=row['completion_ns']]
        for op,key in (('send','es_to_ud_wire_bytes'),('recv','ud_to_es_wire_bytes')):
            require(row[key]==sum(f['wire_bytes'] for f in rf if f['operation']==op),'application directional traffic differs')
        require(row['total_request_wire_bytes']==sum(f['wire_bytes'] for f in rf) and
            row['request_control_wire_bytes']==row['control_wire_bytes']==sum(f['wire_bytes'] for f in rf if f['header']['kind'] not in ('HIDDEN','DELTAS')),
            'request control/wire byte totals differ')
        require(row['decode_ms']==sum(s['wall_ms'] for s in ss[1:]) and row['cache_admission_eviction_ms']==(row['generation_end_ns']-row['model_end_ns'])/1e6 and
            row['generated_tokens_per_second']==len(g)/((row['model_end_ns']-row['model_start_ns'])/1e9),'request timing summary differs')
        for key in ('candidate_lookup_ms','eligibility_validation_ms'):
            require(row[key]==discoveries[rid][key],'semantic timing summary differs')
        for key in ('semantic_encoding_ms','cluster_assignment_ms'):
            require(row[key]==discoveries[rid]['assignment'][key],'encoder timing summary differs')
    # Ownership evidence is captured from sealed resident tensors, after completion.
    ownership=jsonl(root/'r5_ownership.jsonl');owned_by_source={o['source_request_id']:o for o in ownership}
    require(len(owned_by_source)==len(ownership),'duplicate ownership record')
    for o in ownership:
        rid=o['source_request_id'];req=requests[rid];s=bysteps[rid][0];n=len(req['token_ids'])
        require(by[rid]['admission']['entry_id']==o['entry_id']==digest(o['context']) and by[rid]['mode']=='COLD_MISS' and
            o['context']['token_ids']==req['token_ids'] and o['context']['adapter_matrix_sha256']==physical['adapter_matrix_fingerprint'] and
            o['context']['identity']['base_weight_sha256']==physical['base_fingerprint'],'owner context identity differs')
        specs={k:clean(v) for k,v in s['qkv'].items()}
        require(o['tensor_specs']==o['seal']['specs']==specs and o['physical_bytes']==o['seal']['size_bytes']==3*layers*n*hidden*element,'actual owner QKV byte accounting differs')
        require(o['proof']==s['raw_kv_proof'] and o['proof']['prefix_length']==n and not o['proof']['q_consumed'],'owner source proof differs')
        for l,item in enumerate(o['proof']['layers']):
            for r in 'kv':require(item['raw_'+r]==specs[f'{l}.{r}'] and item['derived_'+r]==item['actual_'+r]==clean(s['kv'][l][r]) and item['bitwise_equal'],'proof disconnected from actual tensors')
    state=OrderedDict();peak=evictions=hits=0;events=jsonl(root/'r5_cache_events.jsonl');admitted_sources=set()
    for i,e in enumerate(events):
        kind=e['kind'];key=e.get('context_sha256');rid=e['request_id']
        require(e['sequence']==i and e['capacity_bytes']==capacity,'cache sequence/cap differs')
        if kind in ('EVICT','REJECT','INVALIDATE'):
            require(key in state and e['released_bytes']==state[key]['bytes'],'invalid release')
            if kind=='EVICT':require(key==next(iter(state)) and e['reason']=='capacity_lru','incorrect LRU victim');evictions+=1
            del state[key]
        elif kind=='ADMISSION' and e['allowed']:
            o=owned_by_source[rid];require(key not in state and key==o['entry_id'] and e['candidate_bytes']==o['physical_bytes'] and
                by[rid]['model_end_ns']<=e['monotonic_ns'],'incomplete/foreign admission')
            state[key]=dict(context_sha256=key,bytes=o['physical_bytes'],source_request_id=rid);admitted_sources.add(rid)
        elif kind in ('INSERT','HIT','PHYSICAL_REUSE'):
            require(key in state and e['source_request_id']==state[key]['source_request_id'],'HIT uses evicted/foreign entry')
            if kind=='HIT':
                require(discoveries[rid]['selected_entry_id']==key and by[rid]['source']['entry_id']==key,'HIT bypassed candidate selection');state.move_to_end(key);hits+=1
        else:require(kind in ('MISS','ADMISSION_DENIED') or kind=='ADMISSION' and not e['allowed'],'unknown/integrity cache event in successful run')
        total=sum(v['bytes'] for v in state.values());peak=max(peak,total)
        require(e['occupancy_bytes']==total<=capacity and e['lru_to_mru']==list(state) and e['resident_entries']==list(state.values()) and e['peak_occupancy_bytes']==peak,'cache replay occupancy differs')
    require(admitted_sources==set(owned_by_source) and worker['cache_final']['lru_to_mru']==list(state) and worker['cache_final']['occupancy_bytes']==sum(e['bytes'] for e in state.values()),'final owner state differs')
    for row in rows:
        ev=[e for e in events if e['request_id']==row['request_id']]
        require(row['cache_before']['occupancy_bytes']==(ev[0]['occupancy_bytes'] if ev and ev[0]['kind'] in ('MISS','HIT') else row['cache_before']['occupancy_bytes']) and row['cache_after']=={k:ev[-1][k] for k in row['cache_after']} and row['eviction_count']==sum(e['kind']=='EVICT' for e in ev),'request cache state summary differs')
    neg=read(root/'r5_negative_controls.json');require(neg['adapter_probe']=='EXPECTED_REJECTION' and neg['tensor_frames']==0 and not neg['generated_request'],'unprepared adapter probe unsafe')
    for name in ('lifecycle.json','ud_lifecycle.json'):
        life=read(es/name);require(life['returncode']==0 and life['all_children_exited'] and not life['forced_termination'],'unclean worker shutdown')
    integrated=[r for r in rows if r['condition']=='INTEGRATED']
    lifecycle=bool(hits and evictions and any(r['decision']=='SEMANTIC_CANDIDATE_REJECTED' for r in integrated))
    if plan['lifecycle_capacity']:
        require(integrated[0]['decision']=='CACHE_MISS' and integrated[0]['admission'] and integrated[1]['source']['source_request_id']==integrated[0]['request_id'] and
            integrated[1]['user_id']!=integrated[0]['user_id'] and integrated[3]['source']['source_request_id']==integrated[0]['request_id'] and
            any(e['kind']=='EVICT' and e['source_request_id']==integrated[2]['request_id'] and e['request_id']==integrated[4]['request_id'] for e in events) and
            integrated[5]['source'] is None and integrated[6]['decision']=='SEMANTIC_CANDIDATE_REJECTED','mandatory lifecycle absent')
    decode_hit=any(len(bysteps[r['request_id']])>1 for r in rows if r['source'])
    decode_miss=any(len(bysteps[r['request_id']])>1 for r in rows if r['source'] is None)
    return dict(schema='R5_AUDIT_V1',status='PASS',scope='TEST_FIXTURE_ONLY' if fixture else 'PHYSICAL_OPT_RTX3090_LOCALHOST',requests=len(rows),
        generated_steps=len(steps),hits=hits,evictions=evictions,admissions=len(ownership),peak_occupancy_bytes=peak,
        strict_tensor_files=len(strict_files),strict_kv_reconstruction_checks=kv_checks,strict_prefix_byte_checks=prefix_checks,
        lifecycle_demonstrated=lifecycle,hit_and_miss_decode_observed=decode_hit and decode_miss,
        independent_tcp_replay=True,physical_gpu_verified=not fixture,
        compact_evidence_full_tensor_verification=False)


def analyze(directory,*,sealed=False):
    root=Path(directory)
    if sealed:verify(root)
    audited=audit(root);rows=jsonl(root/'r5_generation_outputs.jsonl');by={r['request_id']:r for r in rows};es=root/'es'
    pairs=[];steps=jsonl(es/'r5_tokens.jsonl');bysteps={rid:[s for s in steps if s['request_id']==rid] for rid in by}
    for row in rows:
        if row['condition']!='INTEGRATED':continue
        baseline=by['N_'+row['request_id']];rid=row['request_id']
        a=(es/baseline['logits_file']).read_bytes();b=(es/row['logits_file']).read_bytes();width=row['logits_shape'][-1]
        av=unpack(a,'float32');bv=unpack(b,'float32')
        metrics=numerical([av[i:i+width] for i in range(0,len(av),width)],[bv[i:i+width] for i in range(0,len(bv),width)],baseline['generated_token_ids'],row['generated_token_ids'])
        full_strict=row['evidence_profile']==baseline['evidence_profile']=='STRICT_AUDIT'
        tensor_checks=[]
        if full_strict:
            for i,(actual,base) in enumerate(zip(bysteps[rid],bysteps[baseline['request_id']])):
                for key,t in {**actual['qkv'],**{f'component_{k}':v for k,v in actual['components'].items()}}.items():
                    bt=base['qkv'][key] if not key.startswith('component_') else base['components'][key[len('component_'):]]
                    expected=token_slice(tensor_data(es,bt),bt,row['reused_prefix_tokens'] if i==0 else 0,bt['shape'][1])
                    tensor_checks.append(tensor_data(es,t)==expected)
                tensor_checks.extend(actual['kv'][l][r]['sha256']==base['kv'][l][r]['sha256'] for l in range(len(actual['kv'])) for r in 'kv')
        pairs.append(dict(request_id=rid,baseline_request_id=baseline['request_id'],physical_hit=row['source'] is not None,
            generated_token_agreement=baseline['generated_token_ids']==row['generated_token_ids'],generated_text_agreement=baseline['generated_text']==row['generated_text'],
            strict_bitwise_logits_equal=a==b,numerical=metrics,strict_tensor_comparison_available=full_strict,
            strict_tensor_comparisons=len(tensor_checks),strict_all_compared_tensors_equal=all(tensor_checks) if full_strict else None,
            application_ttft_delta_ms=row['application_ttft_ms']-baseline['application_ttft_ms'],application_e2e_delta_ms=row['application_e2e_ms']-baseline['application_e2e_ms'],
            model_e2e_delta_ms=row['model_execution_e2e_ms']-baseline['model_execution_e2e_ms'],
            prefill_wire_bytes_delta=row['prefill']['application_wire_bytes']-baseline['prefill']['application_wire_bytes'],
            prefill_hidden_payload_bytes_delta=row['prefill']['hidden_payload_bytes']-baseline['prefill']['hidden_payload_bytes'],
            prefill_delta_payload_bytes_delta=row['prefill']['delta_payload_bytes']-baseline['prefill']['delta_payload_bytes'],
            total_wire_bytes_delta=row['es_to_ud_wire_bytes']+row['ud_to_es_wire_bytes']-baseline['es_to_ud_wire_bytes']-baseline['ud_to_es_wire_bytes']))
    fixture=read(es/'es_plan.json')['fixture'];decision=gates(fixture)
    decision.update(R5_I0_CONTRACT_PASS=True,R5_I1_SEMANTIC_PATH_PASS=True,R5_I2_CACHE_LIFECYCLE_PASS=audited['lifecycle_demonstrated'],
        R5_I3_DISTRIBUTED_GENERATION_PASS=audited['hits']>0 and audited['hit_and_miss_decode_observed'],
        R5_CPU_FIXTURE_PASS=fixture and audited['lifecycle_demonstrated'] and audited['hit_and_miss_decode_observed'],
        R5_GPU_PHYSICAL_EXECUTION_PASS=not fixture,R5_INDEPENDENT_AUDIT_PASS=True,
        R5_TOKEN_FIDELITY_PASS=all(p['generated_token_agreement'] for p in pairs),
        R5_STRICT_BITWISE_FIDELITY_PASS=all(p['strict_bitwise_logits_equal'] and p['strict_all_compared_tensors_equal'] is True for p in pairs))
    if sealed:require(read(root/'r5_decision.json')==decision,'stored decision promoted or differs from independent evidence')
    summary={}
    for condition in ('INTEGRATED','DISTRIBUTED_NO_REUSE'):
        group=[r for r in rows if r['condition']==condition]
        summary[condition]=dict(requests=len(group),candidates=sum(r['candidate_count']>0 for r in group),physical_hits=sum(r['source'] is not None for r in group),
            candidate_rate=sum(r['candidate_count']>0 for r in group)/len(group),physical_hit_rate=sum(r['source'] is not None for r in group)/len(group),
            application_e2e_ms=sum(r['application_e2e_ms'] for r in group),model_execution_e2e_ms=sum(r['model_execution_e2e_ms'] for r in group))
    tcp={condition:{key:sum(r[key] for r in rows if r['condition']==condition) for key in ('es_to_ud_wire_bytes','ud_to_es_wire_bytes','request_control_wire_bytes')}
        for condition in summary}
    frames=jsonl(es/'es_wire.jsonl')
    tcp['schema']='R5_TCP_SUMMARY_V1'
    tcp['whole_session']=dict(es_to_ud_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='send'),
        ud_to_es_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='recv'),
        hidden_payload_bytes=sum(f['header']['payload_bytes'] for f in frames if f['header']['kind']=='HIDDEN'),
        delta_payload_bytes=sum(f['header']['payload_bytes'] for f in frames if f['header']['kind']=='DELTAS'),
        control_wire_bytes=sum(f['wire_bytes'] for f in frames if f['header']['kind'] not in ('HIDDEN','DELTAS')),
        scope='includes handshake, condition-reset eviction controls and shutdown')
    return dict(audit=audited,decision=decision,fidelity=dict(schema='R5_FIDELITY_V1',pairs=pairs,gold_quality='UNAVAILABLE: integration texts have no genuine reference answers'),
        latency=dict(schema='R5_LATENCY_SUMMARY_V1',conditions=summary,scope='instrumented localhost; nested timers overlap; no correctness-qualified speedup'),tcp=tcp,
        limitations=dict(schema='R5_LIMITATIONS_V1',GPU_NOT_RUN=fixture,workload='controlled ordinary real-text relationships; not natural traffic',
            fixture='ControlledEncoder/TinyBackend/TinyTokenizer only' if fixture else False,network='localhost TCP',
            native_peft='HISTORICAL_STRICT_PARITY_FAILED_UNRESOLVED',historical_bitwise='R4-B2/C1 FP16 differences preserved; R5 measured separately',
            raw_q='stored/sealed/accounted; not consumed by prefix K/V construction',adapter='single physically prepared untrained rank8 user_a',
            impact='unavailable output_attentions=False; only physical LRU; no full paper policy',
            evidence='STRICT_AUDIT full QKV/prefill + reconstructed Decode KV; SCALABLE digests/logits only',
            performance='instrumented only; small sequential pilot; no statistical speedup qualification'))


def export(output,report):
    names=dict(audit='r5_audit.json',decision='r5_decision.json',fidelity='r5_fidelity.json',latency='r5_latency_summary.json',tcp='r5_tcp_summary.json',limitations='r5_limitations.json')
    for key,name in names.items():write(Path(output)/name,report[key])
