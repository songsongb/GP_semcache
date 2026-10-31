"""Read-only adapter, protocol, semantic, cache and generation replay for I4A."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import statistics

from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.subsequence import SubsequenceExtractor
from .communication import PREFIX,canonical,digest,digest_bytes,require,validate as validate_header
from .r3a_artifacts import check_parity,read,unpack,write
from .r3a_evidence import jsonl
from .r4b2_analysis import raw_head_bytes,tensor_data,token_slice
from .r4c_analysis import append_head_bytes,clean
from .r4c_session import ForwardOrder,context
from .r4c_workload import termination
from .r5_contracts import validate_request
from .r5_evidence import verify
from .r5_multi_artifacts import validate as prepared,physical,runtime_view,resolve,sharing
from .r5_multi_controller import reason,public_result
from .r5_multi_runtime import workload
from .r5_multi_session import VERSION,SessionGuard
from .r5_perf_analysis import accounting_audit

HISTORY=dict(provenance='USER_VERIFIED_R5_I3F_GPU_HANDOFF',gpu_forensics='PASS',independent_audit='PASS',physical_cache_integrity='PASS',
    source_reconstructed_consumed_kv='BITWISE_IDENTICAL_IN_INSPECTED_COMPARISONS',generated_tokens_text='PASS',first_observed_divergence='IDENTIFIED',
    frozen_numerical_tolerance='FAIL_R5_03_R5_08',root_cause='ROOT_CAUSE_UNRESOLVED',native_peft_strict_parity='UNESTABLISHED',
    correctness_qualified_e2e_speedup='UNESTABLISHED',I3F_B='EXPLICITLY_DEFERRED_TO_FUTURE_WORK')


def gates():
    return {key:False for key in ('R5_I4A_ADAPTER_PREPARATION_PASS','R5_I4A_ADAPTER_IDENTITY_PASS',
        'R5_I4A_SAME_ADAPTER_CROSS_USER_HIT_PASS','R5_I4A_CROSS_ADAPTER_ISOLATION_PASS','R5_I4A_UD_SESSION_ISOLATION_PASS',
        'R5_I4A_GENERATION_COMPLETE_PASS','R5_I4A_GPU_EXECUTION_PASS','R5_I4A_INDEPENDENT_AUDIT_PASS',
        'R5_I4A_FROZEN_NUMERICAL_TOLERANCE_PASS','R5_I4A_GENERATED_TOKEN_FIDELITY_PASS','NATIVE_PEFT_STRICT_PARITY_PASS',
        'correctness_qualified_speedup')}


def unique(rows,key):
    result={r[key]:r for r in rows};require(len(result)==len(rows),'duplicate '+key);return result


def candidates(index,req):
    extractor=SubsequenceExtractor(3);matcher=ExactTokenMatcher();windows={matcher.key(0,w) for w in extractor.extract(req['token_ids'])}
    keys=sorted([k for k,m in index.items() if any(matcher.key(m['cluster_id'],w) in windows for w in extractor.extract(m['token_ids']))],
                key=lambda k:(-len(index[k]['token_ids']),k))[:64]
    return [dict(metadata=index[k],matching_windows=[dict(start=w.start,end=w.end,tokens=list(w.token_ids)) for w in extractor.extract(req['token_ids'])
        if matcher.key(0,w) in {matcher.key(index[k]['cluster_id'],x) for x in extractor.extract(index[k]['token_ids'])}]) for k in keys]


def fidelity(a,b,es_a,es_b):
    require(a['adapter_id']==b['adapter_id'] and a['physical_identity']==b['physical_identity'],'unmatched adapter fidelity identity')
    av=unpack((es_a/a['logits_file']).read_bytes(),'float32');bv=unpack((es_b/b['logits_file']).read_bytes(),'float32');width=a['logits_shape'][-1]
    comparisons=[];skipped=[]
    for step in range(max(len(a['generated_token_ids']),len(b['generated_token_ids']))):
        if step>=min(len(a['generated_token_ids']),len(b['generated_token_ids'])) or a['generated_token_ids'][:step]!=b['generated_token_ids'][:step]:
            skipped.append(dict(step=step,reason='UNMATCHED_GENERATION_HISTORY'));continue
        reference=av[step*width:(step+1)*width];actual=bv[step*width:(step+1)*width]
        comparisons.append(dict(step=step,**check_parity(actual,reference,[1,1,width])))
    return dict(request_id=b['request_id'],adapter_id=b['adapter_id'],steps=comparisons,skipped=skipped,
        generated_tokens_equal=a['generated_token_ids']==b['generated_token_ids'],generated_text_equal=a['generated_text']==b['generated_text'],
        bitwise_logits_equal=(es_a/a['logits_file']).read_bytes()==(es_b/b['logits_file']).read_bytes(),
        frozen_tolerance_pass=bool(comparisons) and not skipped and all(s['passed'] for s in comparisons))


def audit_adapter(path,aid,plan,registry,spec,pm,tok,requests,rows):
    base=path/aid;es=base/'es';ud=base/'ud';identity=read(base/'identity.json');phys=identity['physical_identity'];model=identity['model_identity'];contract=identity['contract']
    require(phys==physical(runtime_view(pm,spec),tok,plan['device'],phys['gpu_uuid'],{k:v for k,v in contract.items() if k!='r5_multi'},spec,registry)[0] and
            identity['adapter']==spec and model==runtime_view(pm,spec)['identity'],'physical adapter/model/tokenizer contract differs')
    require(identity['tokenizer']==tok and identity['TEST_FIXTURE_ONLY']==plan['fixture'],'tokenizer/scope mismatch')
    if plan['fixture']:require(identity['semantic_encoder']['checkpoint']=='controlled_vectors_v1','fixture encoder mismatch')
    else:
        from .r5_perf_schedule import TINYBERT_REVISION
        require(identity['semantic_encoder']['family']=='TinyBERT' and identity['semantic_encoder']['resolved_revision']==TINYBERT_REVISION and
                identity['semantic_encoder']['device']=='cpu' and 'RTX 3090' in identity['gpu']['name'] and
                identity['gpu']['uuid']==pm['gpu']['uuid']==phys['gpu_uuid'] and (es/'gpu_execution.json').is_file(),'real GPU/encoder proof missing')
        for method in contract['methods'].values():require(digest_bytes(method['source'].encode())==method['source_sha256'],'runtime method digest differs')
    initial=read(base/'initial.json');ui=read(ud/'initial.json');result=read(ud/'result.json');up=read(ud/'plan.json');life=read(ud/'lifecycle.json')
    require(initial['cache']['occupancy_bytes']==0 and initial['cache']['entry_count']==0 and not initial['index_entries'] and not initial['seen'] and
            not initial['active_past'] and not initial['index']['centroids'] and initial['index']['queries']==0,'cache/index state leaked')
    require(ui['grants']==ui['parents']==ui['finished']==[] and ui['layer']==0 and ui['current'] is None and
            ui['adapter_id']==aid and ui['matrix_sha256']==spec['matrix_sha256'] and ui['session_id']==initial['session_id']==up['session_id'],'UD startup identity/state mismatch')
    require(result['status']=='COMPLETE' and result['pid']==ui['pid']!=initial['es_pid'] and result['adapter_id']==aid and
            result['matrix_sha256']==spec['matrix_sha256'] and result['session_id']==initial['session_id'] and not result['user_device_cuda_used'] and
            not result['full_model_loaded'] and result['environment']['variables']['CUDA_VISIBLE_DEVICES']=='' and
            life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'UD lifecycle/CPU isolation failure')
    steps=jsonl(es/'r5_tokens.jsonl');by_steps={rid:[s for s in steps if s['request_id']==rid] for rid in rows}
    require(sum(len(v) for v in by_steps.values())==len(steps),'orphan forward evidence')
    require([s['context'] for s in jsonl(es/'forward_attempts.jsonl')]==[s['context'] for s in steps]==[s['context'] for s in jsonl(es/'forward_completed.jsonl')],
            'hidden/incomplete model forward')
    require(jsonl(es/'r4c_logit_diagnostics.jsonl')==[s['logit_diagnostic'] for s in steps],'missing logit diagnostics')
    frames=jsonl(es/'wire.jsonl');other=jsonl(ud/'wire.jsonl');require(len(frames)==len(other),'TCP ledger lengths differ')
    order=ForwardOrder(up['cohort_sha256'],model['layers'],contract['context_limit']);guard=SessionGuard(registry,phys)
    pending=None;forwards={};controls=[];hello=dict(application=VERSION,adapter_id=aid,matrix_sha256=spec['matrix_sha256'],registry_sha256=registry['sha256'],
        physical_identity=phys,cohort_sha256=up['cohort_sha256'])
    require(frames[0]['header']['metadata']==hello and frames[1]['header']['metadata']==dict(hello,pid=ui['pid'],user_device_cuda_used=False),'handshake identity mismatch')
    for a,b in zip(frames,other):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and all(a[k]==b[k] for k in ('header','frame_sha256','wire_bytes')),'TCP endpoint records disagree')
        h=a['header'];validate_header(h,h['payload_bytes'])
        require(h['adapter_sha256']==spec['matrix_sha256'] and h['session_id']==ui['session_id'] and a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],
                'TCP adapter/session/byte mismatch')
        if h['kind']=='HELLO' and h['request_id']!='handshake':
            require(pending is None and h['metadata']['application']==VERSION,'invalid control ordering/version')
            message=h['metadata']['control'];order.control(message);controls.append(dict(message,request_id=h['request_id']))
        if h['kind']=='READY' and h['request_id']!='handshake':
            require(controls and h['metadata']['control_sha256']==digest({k:v for k,v in controls[-1].items() if k!='request_id'}),'control digest differs')
        if h['kind']=='HIDDEN':
            require(pending is None,'overlapping TCP projections');req=guard.accept(h['metadata']['r5_multi'],h['metadata']['r4c_forward'])
            require(req==requests[req['request_id']],'wire request mapping differs');order.accept(h,model);pending=h
        if h['kind']=='DELTAS':
            require(pending is not None and h['request_id']==pending['request_id'] and h['layer_id']==pending['layer_id'] and
                    h['metadata']['hidden_payload_sha256']==pending['payload_sha256'] and h['metadata']['r4c_forward_sha256']==digest(pending['metadata']['r4c_forward']) and
                    h['metadata']['route_sha256']==digest(pending['metadata']['r5_multi']) and h['metadata']['adapter_id']==aid and
                    h['metadata']['matrix_sha256']==spec['matrix_sha256'] and h['metadata']['pid']==ui['pid'] and
                    h['metadata']['user_id']==pending['metadata']['r5_multi']['request']['user_id'] and not h['metadata']['user_device_cuda_used'],
                    'wrong adapter/user/CPU correction receipt')
            order.advance();pending=None
        if h['kind'] in ('HIDDEN','DELTAS'):forwards.setdefault(h['request_id'],[]).append(a)
    require(pending is None,'incomplete exchange');order.finish()
    require(controls==jsonl(es/'controls.jsonl')==jsonl(ud/'controls.jsonl') and result['finished']==order.finished,'incomplete/mismatched source completion controls')
    for operation in ('send','recv'):require([f['header']['sequence'] for f in frames if f['operation']==operation]==list(range(sum(f['operation']==operation for f in frames))),'TCP sequence gap')
    require(set(forwards)=={s['forward_id'] for s in steps},'orphan/missing TCP forward')
    ud_layers=jsonl(ud/'layers.jsonl');require(len(ud_layers)==len(steps)*model['layers']==result['layers'],'UD layer coverage differs')
    kv_checks=0
    for rid,row in rows.items():
        req=requests[rid];ss=by_steps[rid];target=req['token_ids'];grant=row['source'];p=row['reused_prefix_tokens'];generated=row['generated_token_ids']
        require(req['adapter_id']==aid and req['physical_identity']==phys and resolve(registry,req['user_id'],aid)==aid and
                row['request_sha256']==req['request_sha256'] and row['physical_identity']==phys and row['user_id']==req['user_id'] and
                row['ud_pid']==ui['pid'] and row['es_pid']==initial['es_pid'] and row['TEST_FIXTURE_ONLY']==plan['fixture'],'request identity/user/adapter mismatch')
        require(row['status']=='COMPLETE' and len(ss)==len(generated)==row['generated_tokens']>0 and
                row['termination']==termination(req['generation_policy'],generated) and order.finished[rid]['generated_token_ids']==generated,'incomplete generation')
        require(row['mode']==('PARTIAL_PREFIX_HIT' if grant else 'COLD_MISS' if plan['condition']=='INTEGRATED' else 'NO_REUSE') and
                (row['decision']=='PHYSICAL_PREFIX_HIT')==(grant is not None) and not row['cached_prefix_q_consumed'],'false HIT/compute claim')
        if grant:
            source=requests[grant['source_request_id']]
            require(source['adapter_id']==aid and source['token_ids']==target[:p] and source['physical_identity']==phys and
                    sharing(registry,source['user_id'],req['user_id'],aid,aid) is None and rows[source['request_id']]['completion_ns']<row['arrival_ns'],
                    'incompatible/incomplete source or unauthorized sharing')
        blob=(es/row['logits_file']).read_bytes();values=unpack(blob,'float32');width=contract['vocabulary_size']
        require(digest_bytes(blob)==row['logits_sha256'] and row['logits_shape']==[1,len(generated),width] and len(values)==len(generated)*width,'logit evidence mismatch')
        reconstructed=None
        for step,s in enumerate(ss):
            old=p if not step else len(target)+step-1;inputs=target[p:] if not step else [generated[step-1]]
            require(s['context']==context(rid,step,inputs,target,old,req['generation_policy'],row['mode'],up['cohort_sha256'],grant,context_limit=contract['context_limit']) and
                    s['executed_projection_calls']==3*model['layers'] and s['executed_projection_token_rows']==len(inputs)*3*model['layers'] and
                    s['prefix_projection_token_rows_skipped']==(p*3*model['layers'] if not step else 0),'hidden recomputation or invalid suffix/decode')
            require(s['position_observation']['positions']==s['context']['positions'] and s['position_observation']['past_length']==old and
                    s['past']['sequence_length']==len(target)+step and (s['input_past'] is None if old==0 else s['input_past']['sequence_length']==old),'positions/past length mismatch')
            require(len(s['qkv'])==3*model['layers'] and len(s['kv'])==model['layers'],'incomplete raw/KV layers')
            for raw in s['qkv'].values():
                require(raw['shape']==[1,len(inputs),model['hidden_size']] and raw['dtype']==model['hidden_dtype'] and raw['device']==plan['device'],
                        'QKV shape/dtype/device mismatch')
            for component in s['components'].values():
                require(component['shape']==[1,len(inputs),model['hidden_size']] and component['dtype']==model['hidden_dtype'] and component['device']==plan['device'],
                        'actual component processed incompatible token rows')
            if not plan['fixture']:require(s['position_observation']['lookup_output_equal'] and len(s['components'])==3*model['layers'],'actual OPT position/components missing')
            lv=values[step*width:(step+1)*width]
            require(generated[step]==s['selected_token_id']==max(range(width),key=lv.__getitem__) and s['logits_sha256']==digest_bytes(blob[step*width*4:(step+1)*width*4]),'greedy output mismatch')
            require(all(s['logit_diagnostic'][k]['nonfinite']==0 for k in ('all_logits','last_position','valid_positions')),'nonfinite logits')
            fs=forwards[s['forward_id']];uls=[u for u in ud_layers if u['forward_id']==s['forward_id']]
            require(len(fs)==2*model['layers'] and [u['layer'] for u in uls]==list(range(model['layers'])),'missing TCP layer evidence')
            for u in uls:
                sent,received=fs[2*u['layer']:2*u['layer']+2]
                require(u['context']==s['context'] and u['hidden_frame_sha256']==sent['frame_sha256'] and u['delta_frame_sha256']==received['frame_sha256'] and u['metrics']==received['header']['metadata'],'UD trace mismatch')
            for key,kind in (('hidden_payload_bytes','HIDDEN'),('delta_payload_bytes','DELTAS')):
                require(s[key]==sum(f['header']['payload_bytes'] for f in fs if f['header']['kind']==kind),'projection byte accounting mismatch')
            if not step:reconstructed=[[tensor_data(es,s['kv'][l][r]) for r in 'kv'] for l in range(model['layers'])]
            for l in range(model['layers']):
                for j,r in enumerate('kv'):
                    raw=s['qkv'][f'{l}.{r}'];kv=s['kv'][l][r]
                    require(raw['shape']==[1,len(inputs),model['hidden_size']] and raw['dtype']==model['hidden_dtype'] and raw['device']==plan['device'] and
                            kv['shape']==[1,contract['heads'],len(target)+step,model['hidden_size']//contract['heads']] and kv['dtype']==model['hidden_dtype'] and kv['device']==plan['device'],'raw/KV layout/dtype mismatch')
                    transformed=raw_head_bytes(tensor_data(es,raw),raw['shape'],contract['heads'],2 if raw['dtype']=='float16' else 4)
                    if not step:require(token_slice(reconstructed[l][j],kv,old,len(target))==transformed,'raw-derived actual KV mismatch')
                    else:reconstructed[l][j]=append_head_bytes(reconstructed[l][j],transformed,heads=contract['heads'],old_length=old,head_dim=model['hidden_size']//contract['heads'],element_bytes=2 if raw['dtype']=='float16' else 4)
                    require(digest_bytes(reconstructed[l][j])==kv['sha256'],'Decode KV integrity mismatch');kv_checks+=1
                    if not step and grant:
                        source_step=by_steps[grant['source_request_id']][0];sourcekv=source_step['kv'][l][r]
                        require(s['input_kv'][l][j]==clean(sourcekv) and token_slice(reconstructed[l][j],kv,0,p)==tensor_data(es,sourcekv),
                                'cached source/constructed/consumed prefix differs')
                tensor_data(es,s['qkv'][f'{l}.q'])
            for t in s['components'].values():tensor_data(es,t)
            if step:require(s['input_kv']==[[clean(ss[step-1]['kv'][l][r]) for r in 'kv'] for l in range(model['layers'])],'prior Decode KV mutated')
        rf=[f for f in frames if row['arrival_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=row['completion_ns']]
        require(row['total_request_wire_bytes']==sum(f['wire_bytes'] for f in rf) and row['cpu_lora_ms']==sum(f['header']['metadata'].get('cpu_lora_ms',f['header']['metadata'].get('cpu_lora_and_serialization_ms',0.)) for f in rf if f['operation']=='recv'),'request wire/CPU metrics mismatch')
        for op,key in (('send','es_to_ud_wire_bytes'),('recv','ud_to_es_wire_bytes')):require(row[key]==sum(f['wire_bytes'] for f in rf if f['operation']==op),'directional bytes mismatch')
        require(row['application_e2e_ms']==(row['completion_ns']-row['arrival_ns'])/1e6 and row['application_ttft_ms']==(row['first_token_available_ns']-row['arrival_ns'])/1e6,'request timing mismatch')
    return dict(status='PASS',physical_identity=phys,ud_pid=ui['pid'],es_pid=initial['es_pid'],session_id=ui['session_id'],requests=len(rows),
                forwards=len(steps),strict_kv_checks=kv_checks,first_layer_hidden=frames[2]['header']['payload_sha256'],
                first_layer_delta=frames[3]['header']['payload_sha256']),by_steps


def summarize(rows,adapter_stats):
    fields=('reused_prefix_tokens','eviction_count','es_to_ud_wire_bytes','ud_to_es_wire_bytes','cpu_lora_ms','semantic_encoding_ms',
            'cluster_assignment_ms','candidate_lookup_ms','eligibility_validation_ms','cache_lookup_ms','rejected_cross_adapter_candidates')
    result=dict(requests=len(rows),logical_users=len({r['user_id'] for r in rows}),actual_ud_processes=len(adapter_stats),
        semantic_candidates=sum(r['candidate_count'] for r in rows),physical_hits=sum(r['source'] is not None for r in rows),
        physical_misses=sum(r['source'] is None for r in rows),admissions=sum(r['admission'] is not None for r in rows),
        **{k:sum(r.get(k,0.) for r in rows) for k in fields})
    result['physical_hit_rate']=result['physical_hits']/len(rows)
    for key in ('model_ttft_ms','application_ttft_ms','model_execution_e2e_ms','application_e2e_ms'):
        values=[r[key] for r in rows];result[key]=dict(mean=statistics.mean(values),median=statistics.median(values),total=sum(values))
    result['prefill_payload_bytes']=sum(r['prefill']['hidden_payload_bytes']+r['prefill']['delta_payload_bytes'] for r in rows)
    result['decode_payload_bytes']=sum(r['decode']['hidden_payload_bytes']+r['decode']['delta_payload_bytes'] for r in rows)
    for key in ('hidden_payload_bytes','delta_payload_bytes'):
        result[key]=sum(r['prefill'][key]+r['decode'][key] for r in rows)
    result['gpu_memory_observations']=[r['gpu_memory'] for r in rows if r['gpu_memory'] is not None]
    return result


def analyze(root,*,sealed=True):
    root=Path(root)
    if sealed:
        verify(root);require(read(root/'multi_results.json')['status']=='COMPLETE','failed/incomplete multi-adapter run')
    config=read(root/'multi_config.json');m,pm,adapters,registry=prepared(config['multi_prepared'])
    require(read(root/'multi_historical.json')==HISTORY,'historical numerical findings changed')
    all_data={};audits={};metrics={};pids=[];sessions=[];per_adapter_stats={};allpairs=[];global_peaks={}
    for condition in ('DISTRIBUTED_NO_REUSE','INTEGRATED'):
        path=root/condition;account=accounting_audit(path);result=read(path/'result.json');life=read(path/'lifecycle.json');plan=read(path/'plan.json')
        require(result['status']=='COMPLETE' and result['condition']==condition and result['sequential'] and result['base_models_loaded']==1 and
                result['es_lora_parameters']==0 and result['gpu_inference_performed']==(not config['fixture']) and plan['sources']==read(root/'multi_sources.json') and
                all(plan[k]==v for k,v in config.items()) and life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'ES lifecycle/config mismatch')
        require(read(path/'registry.json')==registry and read(path/'workload.json')==workload(registry,fixture=config['fixture']),'registry/workload changed')
        ledger=jsonl(path/'requests.jsonl');arrivals=[r for r in ledger if r['kind']=='ARRIVAL'];done=[r for r in ledger if r['kind']=='COMPLETION']
        reqs=unique([validate_request(r['request']) for r in arrivals],'request_id');rows=unique(jsonl(path/'generation.jsonl'),'request_id')
        raw=workload(registry,fixture=config['fixture']);expected=[('N_' if condition=='DISTRIBUTED_NO_REUSE' else '')+r['request_id'] for r in raw]
        require(list(reqs)==list(rows)==[r['request_id'] for r in done]==expected and len(rows)==result['requests'],'missing/reordered requests or completions')
        for arrival,completion,row,original in zip(arrivals,done,rows.values(),raw):
            req=reqs[row['request_id']]
            require(all(req[k]==original[k] for k in ('user_id','adapter_id','input_text','request_sequence_number','sample_id','dataset')) and
                    row['arrival_ns']==arrival['arrival_ns'] and row['completion_ns']==completion['completion_ns'] and
                    (row['request_id']==expected[0] or list(rows.values())[expected.index(row['request_id'])-1]['completion_ns']<row['arrival_ns']),
                    'workload/order/sequential interval differs')
            require(read(path/'client_results'/(row['request_id']+'.json'))==public_result(row),'client result leaked privileged metadata')
        adapter_audits={};steps={}
        for aid,spec in adapters.items():
            rr={rid:r for rid,r in rows.items() if r['adapter_id']==aid};rq={rid:reqs[rid] for rid in rr}
            audit,ss=audit_adapter(path,aid,plan,registry,spec,pm,m['tokenizer'],rq,rr);adapter_audits[aid]=audit;steps.update(ss)
            pids.append(audit['ud_pid']);sessions.append(audit['session_id'])
        pids.append(result['es_pid'])
        ownership=unique(jsonl(path/'ownership.jsonl'),'source_request_id') if (path/'ownership.jsonl').exists() else {}
        index={aid:{} for aid in adapters};resident={aid:OrderedDict() for aid in adapters};cluster={};discoveries={};peak={aid:0 for aid in adapters};global_peak=0
        last_cache_event={};started_requests=set()
        isolation=unique(jsonl(path/'isolation.jsonl'),'request_id');events=jsonl(path/'events.jsonl')
        for sequence,item in enumerate(events):
            require(item['sequence']==sequence and item['adapter_id'] in adapters,'event sequence/adapter mismatch')
            aid=item['adapter_id'];event=item['event'];kind=event['kind'];state=resident[aid];idx=index[aid]
            if item['stream']=='semantic':
                if kind=='INDEX_INSERT':
                    meta=event['metadata'];rid=meta['source_request_id'];key=meta['entry_id']
                    require(key not in idx and key in state and rid in ownership and meta['source_user']==reqs[rid]['user_id'] and meta['physical_identity']==reqs[rid]['physical_identity'] and
                            meta['completed'] and meta['proof_sha256']==rows[rid]['admission']['proof_sha256'] and meta['request_sha256']==reqs[rid]['request_sha256'],'unauthorized/incomplete index source')
                    idx[key]=meta
                elif kind=='INDEX_INVALIDATE':require(idx.pop(event['entry_id'],None) is not None,'stale index invalidation')
                else:
                    require(kind=='DISCOVERY','unknown semantic event');rid=event['request_id'];req=reqs[rid];assignment=event['assignment'];before=assignment['cluster_before']
                    if aid not in cluster:
                        c=IntentClusterer(before['num_clusters'],before['update_interval'],update_mode=before['update_mode']);c.initialize(before['centroids'],counts=before['counts']);cluster[aid]=c
                    c=cluster[aid];require(c.centroids==before['centroids'] and c.counts==before['counts'] and c.queries==before['queries'],'semantic state discontinuity')
                    observed=c.observe_with_diagnostics(assignment['embedding']);require(all(assignment[k]==v for k,v in observed.items()) and c.centroids==assignment['cluster_after']['centroids'] and c.counts==assignment['cluster_after']['counts'],'semantic transition mismatch')
                    require(assignment['encoder']==read(path/aid/'identity.json')['semantic_encoder'],'encoder identity mismatch')
                    cs=candidates(idx,req);reasons=[dict(entry_id=r['metadata']['entry_id'],source_request_id=r['metadata']['source_request_id'],prefix_length=len(r['metadata']['token_ids']),rejection=reason(r['metadata'],req,state,registry)) for r in cs]
                    eligible=[r['entry_id'] for r in reasons if r['rejection'] is None];chosen=eligible[0] if eligible and condition=='INTEGRATED' else None
                    outcome='PHYSICAL_PREFIX_HIT' if chosen else 'SEMANTIC_CANDIDATE_REJECTED' if cs and not eligible else 'CACHE_MISS'
                    require(event['physical_reuse_enabled']==(condition=='INTEGRATED') and event['candidates']==cs and event['candidate_reasons']==reasons and event['index_entry_ids']==sorted(idx) and event['selected_entry_id']==chosen and event['outcome']==rows[rid]['decision']==outcome and rows[rid]['candidate_count']==len(cs),'automatic eligibility/dispatch mismatch')
                    rejected=[dict(foreign_adapter_id=other,rejected_count=len(candidates(index[other],req)),reason='CROSS_ADAPTER_REUSE_FORBIDDEN') for other in sorted(adapters) if other!=aid and candidates(index[other],req)]
                    require(isolation[rid]==dict(request_id=rid,adapter_id=aid,rejected=rejected,foreign_tensor_reads=0,foreign_wire_frames=0,scope='PRIVILEGED_ADMIN_AUDIT_ONLY') and rows[rid]['rejected_cross_adapter_candidates']==sum(r['rejected_count'] for r in rejected),'cross-adapter rejection probe differs')
                    discoveries[rid]=event
            else:
                require(item['stream']=='cache','unknown event stream');key=event.get('context_sha256');rid=event['request_id']
                if rid not in started_requests:
                    before=rows[rid]['cache_before']
                    require(before['occupancy_bytes']==sum(v['bytes'] for v in state.values()) and before['lru_to_mru']==list(state) and
                            before['resident_entries']==list(state.values()) and before['peak_occupancy_bytes']==peak[aid],'initial request cache snapshot mismatch')
                    started_requests.add(rid)
                if kind in ('EVICT','INVALIDATE','REJECT'):
                    require(key in state and event['released_bytes']==state[key]['bytes'] and key not in idx,'unsynchronized cache/index release')
                    if kind=='EVICT':require(key==next(iter(state)),'LRU victim mismatch')
                    del state[key]
                elif kind=='ADMISSION' and event['allowed']:
                    o=ownership[rid];req=reqs[rid];source=steps[rid][0];element=2 if pm['identity']['hidden_dtype']=='float16' else 4
                    expected_identity=deepcopy(runtime_view(pm,adapters[aid])['identity'])
                    expected_identity.update(length=len(req['token_ids']),token_ids=req['token_ids'],token_ids_sha256=digest(req['token_ids']),
                        endpoint=dict(batch_size=1,prompt_tokens=len(req['token_ids']),attention='eager',use_cache=True,output_attentions=False,output_head='native_all_token_logits'))
                    expected_multi=dict(schema='R5_MULTI_CACHE_NAMESPACE_V1',adapter_spec_sha256=digest(adapters[aid]),registry_sha256=registry['sha256'],
                        sharing_policy=registry['policy'],scale=adapters[aid]['scale'],layout=adapters[aid]['layout'],cache_format='UNCHANGED_R3B_TOTAL_RAW_QKV')
                    require(o['context']['identity']==expected_identity and o['context']['r5_multi']==expected_multi and
                            o['context']['token_ids']==req['token_ids'] and o['context']['positions']==req['positions'] and o['context']['attention_mask']==req['attention_mask'] and
                            o['context']['tokenizer']==m['tokenizer'] and o['context']['execution_sources']==plan['sources'] and
                            o['context']['storage_device']==plan['device'] and o['context']['opt_contract']==read(path/aid/'identity.json')['contract'] and
                            o['context']['prepared_manifest_sha256']==digest(dict(preparation=m,adapter=adapters[aid])),
                            'cache context model/adapter/tokenizer/runtime namespace mismatch')
                    require(key not in state and rows[rid]['mode']=='COLD_MISS' and o['adapter_id']==aid and o['source_user']==req['user_id'] and
                            o['registry_sha256']==registry['sha256'] and o['sharing_authorization']==registry['users'][req['user_id']] and
                            o['proof']==source['raw_kv_proof'] and digest(o['context'])==o['entry_id']==key and
                            o['context']['identity']['adapter_matrix_sha256']==adapters[aid]['matrix_sha256'] and
                            o['context']['r5_multi']['registry_sha256']==registry['sha256'] and o['context']['r5_multi']['adapter_spec_sha256']==digest(adapters[aid]) and
                            o['tensor_specs']==o['seal']['specs']=={k:clean(v) for k,v in source['qkv'].items()} and
                            o['physical_bytes']==o['seal']['size_bytes']==event['candidate_bytes']==3*pm['identity']['layers']*len(req['token_ids'])*pm['identity']['hidden_size']*element and
                            rows[rid]['model_end_ns']<=event['monotonic_ns'],'invalid adapter source ownership/admission')
                    require(o['proof']['prefix_length']==len(req['token_ids']) and len(o['proof']['layers'])==pm['identity']['layers'] and
                            not o['proof']['q_consumed'] and digest(o['proof'])==rows[rid]['admission']['proof_sha256'],'source proof identity mismatch')
                    for l,proof in enumerate(o['proof']['layers']):
                        require(proof['layer']==l and proof['bitwise_equal'],'source proof layer mismatch')
                        for r in 'kv':
                            require(proof['raw_'+r]==clean(source['qkv'][f'{l}.{r}']) and
                                    proof['derived_'+r]==proof['actual_'+r]==clean(source['kv'][l][r]),'source Raw/actual KV proof binding mismatch')
                    state[key]=dict(context_sha256=key,bytes=o['physical_bytes'],source_request_id=rid)
                elif kind in ('INSERT','HIT','PHYSICAL_REUSE'):
                    require(key in state and event['source_request_id']==state[key]['source_request_id'],'foreign/stale physical cache access')
                    if kind=='HIT':
                        require(rows[rid]['source']['entry_id']==key and sharing(registry,reqs[event['source_request_id']]['user_id'],reqs[rid]['user_id'],aid,aid) is None,'unauthorized physical HIT')
                        state.move_to_end(key)
                else:require(kind in ('MISS','ADMISSION_DENIED') or kind=='ADMISSION' and not event['allowed'],'unexpected cache/integrity event')
                size=sum(s['bytes'] for s in state.values());peak[aid]=max(peak[aid],size)
                require(event['occupancy_bytes']==size<=event['capacity_bytes']<=2**30 and event['lru_to_mru']==list(state) and event['resident_entries']==list(state.values()) and event['peak_occupancy_bytes']==peak[aid],'cache occupancy/ownership accounting mismatch')
                global_peak=max(global_peak,sum(sum(v['bytes'] for v in r.values()) for r in resident.values()))
                last_cache_event[rid]=event
        require(set(discoveries)==set(rows),'missing semantic discovery')
        for rid,row in rows.items():
            require(row['cache_after']=={k:last_cache_event[rid][k] for k in row['cache_after']},'final request cache snapshot mismatch')
        for aid in adapters:
            final=read(path/aid/'final.json');require(final['cache']['lru_to_mru']==list(resident[aid]) and final['cache']['occupancy_bytes']==sum(v['bytes'] for v in resident[aid].values()) and set(resident[aid])==set(index[aid]),'final cache/index state mismatch')
            stats=summarize([r for r in rows.values() if r['adapter_id']==aid],{aid:adapter_audits[aid]})
            stats.update(cache_occupancy_bytes=final['cache']['occupancy_bytes'],peak_cache_bytes=peak[aid],cache_capacity_bytes=final['cache']['capacity_bytes'])
            expected_counts=(0,0,0) if condition=='DISTRIBUTED_NO_REUSE' else (1,1,0) if config['stage']=='pilot' else (4,4,1)
            require((stats['physical_hits'],stats['admissions'],stats['eviction_count'])==expected_counts,'controlled automatic lifecycle missing')
            require(stats['logical_users']==(2 if config['stage']=='pilot' else 10),'logical user coverage mismatch')
            if config['stage']=='expanded' and condition=='INTEGRATED':
                prefix='M_'+aid[-1]+'_'
                require(rows[prefix+'05']['source'] is None and rows[prefix+'06']['decision']=='SEMANTIC_CANDIDATE_REJECTED' and
                        rows[prefix+'08']['source']['source_request_id']==prefix+'07','post-eviction/re-admission sequence missing')
                source_lengths=[len(reqs[prefix+f'{i:02d}']['token_ids']) for i in (0,2,4)]
                expected_capacity=max(source_lengths[i]+source_lengths[j] for i in range(3) for j in range(i))*3*pm['identity']['layers']*pm['identity']['hidden_size']*(4 if config['fixture'] else 2)
                require(final['cache']['capacity_bytes']==expected_capacity,'token-derived isolated cache capacity mismatch')
            per_adapter_stats[condition,aid]=stats
        neg=read(path/'negative_controls.json');require(neg['wire_frames']==0 and len(neg['probes'])==3 and all(r['outcome']=='EXPECTED_REJECTION' and not r['generated_request'] for r in neg['probes']),'negative mapping controls missing')
        metrics[condition]=dict(global_metrics=summarize(list(rows.values()),adapter_audits),per_adapter={aid:per_adapter_stats[condition,aid] for aid in adapters})
        metrics[condition]['global_metrics'].update(peak_cache_bytes=global_peak,cache_occupancy_bytes=sum(s['cache_occupancy_bytes'] for s in metrics[condition]['per_adapter'].values()))
        audits[condition]=dict(status='PASS',accounting=account,adapters=adapter_audits);all_data[condition]=(rows,reqs,path)
    require(len(set(pids))==len(pids) and len(set(sessions))==len(sessions),'ES/UD sessions or PIDs reused')
    ar,aq,ap=all_data['DISTRIBUTED_NO_REUSE'];br,bq,bp=all_data['INTEGRATED']
    for rid,b in br.items():
        a=ar['N_'+rid];x=aq['N_'+rid];y=bq[rid]
        require(all(x[k]==y[k] for k in ('input_text','token_ids','attention_mask','positions','generation_policy','physical_identity','user_id','adapter_id')),'unmatched no-reuse baseline')
        allpairs.append(fidelity(a,b,ap/a['adapter_id']/'es',bp/b['adapter_id']/'es'))
    decision=gates();same_user_hits=[r for r in br.values() if r['source'] and bq[r['source']['source_request_id']]['user_id']!=r['user_id']]
    for key in ('R5_I4A_ADAPTER_PREPARATION_PASS','R5_I4A_ADAPTER_IDENTITY_PASS','R5_I4A_CROSS_ADAPTER_ISOLATION_PASS',
                'R5_I4A_UD_SESSION_ISOLATION_PASS','R5_I4A_GENERATION_COMPLETE_PASS','R5_I4A_INDEPENDENT_AUDIT_PASS'):decision[key]=True
    decision['R5_I4A_SAME_ADAPTER_CROSS_USER_HIT_PASS']=bool(same_user_hits) and {r['adapter_id'] for r in same_user_hits}==set(adapters)
    decision['R5_I4A_CROSS_ADAPTER_ISOLATION_PASS']=metrics['INTEGRATED']['global_metrics']['rejected_cross_adapter_candidates']>0
    decision['R5_I4A_GPU_EXECUTION_PASS']=not config['fixture']
    decision['R5_I4A_FROZEN_NUMERICAL_TOLERANCE_PASS']=all(p['frozen_tolerance_pass'] for p in allpairs)
    decision['R5_I4A_GENERATED_TOKEN_FIDELITY_PASS']=all(p['generated_tokens_equal'] and p['generated_text_equal'] for p in allpairs)
    for condition in metrics:
        for aid,stats in metrics[condition]['per_adapter'].items():
            pairs=[p for p in allpairs if p['adapter_id']==aid];stats['generated_token_fidelity']=all(p['generated_tokens_equal'] for p in pairs);stats['frozen_tolerance_pass']=all(p['frozen_tolerance_pass'] for p in pairs)
        metrics[condition]['global_metrics'].update(generated_token_fidelity=decision['R5_I4A_GENERATED_TOKEN_FIDELITY_PASS'],frozen_tolerance_pass=decision['R5_I4A_FROZEN_NUMERICAL_TOLERANCE_PASS'])
    return dict(decision=decision,audit=dict(status='PASS',TEST_FIXTURE_ONLY=config['fixture'],conditions=audits,independent_tcp_cache_generation_replay=True),
        metrics=metrics,fidelity=allpairs,adapter_validation={aid:dict(matrix_sha256=s['matrix_sha256'],source_seed=s['source_seed'],trained_adapter=False,validation='PASS') for aid,s in adapters.items()},
        limitations=dict(TEST_FIXTURE_ONLY=config['fixture'],historical=HISTORY,network='localhost TCP',
            users='logical registry identities, not physical devices',adapters='controlled synthetic untrained research matrices',
            performance='instrumented descriptive metrics only; no statistical speedup or correctness-qualified improvement',
            privacy='private per-adapter indexes/caches; privileged operator evidence; client outputs contain only their own generation',
            numerical='current matched tolerances reported separately; historical R5_03/R5_08 failures preserved; I3F-B deferred'))


def export(path,report):
    for key,value in report.items():write(Path(path)/('multi_'+key+'.json'),value)
