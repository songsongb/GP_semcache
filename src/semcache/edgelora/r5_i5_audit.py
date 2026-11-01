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
from .r5_i5_protocol import ForwardOrder,context
from .r4c_workload import termination
from .r5_i5_contracts import validate_request
from .r5_i5_evidence import verify
from .r5_multi_artifacts import validate as prepared,physical,runtime_view,resolve,sharing
from .r5_multi_controller import reason,public_result

from .r5_i5_session import VERSION,SessionGuard
from .r5_perf_analysis import accounting_audit


def audit_adapter(path,aid,plan,registry,spec,pm,tok,requests,rows):
    base=path/aid;es=base/'es';ud=base/'ud';identity=read(base/'identity.json');phys=identity['physical_identity'];model=identity['model_identity'];contract=identity['contract']
    require(phys==physical(runtime_view(pm,spec),tok,plan['device'],phys['gpu_uuid'],{k:v for k,v in contract.items() if k!='r5_multi'},spec,registry)[0] and
            identity['adapter']==spec and model==runtime_view(pm,spec)['identity'],'physical adapter/model/tokenizer contract differs')
    require(identity['tokenizer']==tok and identity['TEST_FIXTURE_ONLY']==plan['fixture'],'tokenizer/scope mismatch')
    if not plan['fixture']:
        require(identity['gpu']['uuid']==phys['gpu_uuid']==plan['expected_execution_gpu_uuid'] and (es/'gpu_execution.json').is_file(),'execution GPU identity/forward evidence differs')
    if plan['condition']=='FULL_RECOMPUTE':require(identity['semantic_encoder'] is None,'A performed semantic computation')
    elif plan['fixture']:require(identity['semantic_encoder']['checkpoint']=='controlled_vectors_v1','fixture encoder mismatch')
    else:
        from .r5_perf_schedule import TINYBERT_REVISION
        require(identity['semantic_encoder']['family']=='TinyBERT' and identity['semantic_encoder']['resolved_revision']==TINYBERT_REVISION and
                identity['semantic_encoder']['device']=='cpu' and 'RTX 3090' in identity['gpu']['name'] and
                identity['gpu']['uuid']==phys['gpu_uuid']==plan['expected_execution_gpu_uuid'] and (es/'gpu_execution.json').is_file(),'real GPU/encoder proof missing')
        for method in contract['methods'].values():require(digest_bytes(method['source'].encode())==method['source_sha256'],'runtime method digest differs')
    initial=read(base/'initial.json');ui=read(ud/'initial.json');result=read(ud/'result.json');up=read(ud/'plan.json');life=read(ud/'lifecycle.json')
    require(initial['cache']['occupancy_bytes']==0 and initial['cache']['entry_count']==0 and not initial['index_entries'] and not initial['seen'] and
            not initial['active_past'] and not initial['index'].get('centroids') and initial['index']['queries']==0,'cache/index state leaked')
    require(ui['grants']==ui['parents']==ui['finished']==[] and ui['layer']==0 and ui['current'] is None and
            ui['adapter_id']==aid and ui['matrix_sha256']==spec['matrix_sha256'] and ui['session_id']==initial['session_id']==up['session_id'],'UD startup identity/state mismatch')
    require(result['status']=='COMPLETE' and result['pid']==ui['pid']!=initial['es_pid'] and result['adapter_id']==aid and
            result['matrix_sha256']==spec['matrix_sha256'] and result['session_id']==initial['session_id'] and not result['user_device_cuda_used'] and
            not result['full_model_loaded'] and result['environment']['variables']['CUDA_VISIBLE_DEVICES']=='' and
            life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'UD lifecycle/CPU isolation failure')
    steps=jsonl(es/'r5_tokens.jsonl');by_steps={rid:[s for s in steps if s['request_id']==rid] for rid in rows}
    require(sum(len(v) for v in by_steps.values())+sum(s['request_id'].startswith('W_') for s in steps)==len(steps),'orphan forward evidence')
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
        require(ss and row['arrival_ns']<=row['model_start_ns']<=ss[0]['start_monotonic_ns'] and
                ss[-1]['available_monotonic_ns']<=row['model_end_ns']<=row['generation_end_ns']<=row['completion_ns'],'timing interval order differs')
        require(row['model_ttft_ms']==(ss[0]['available_monotonic_ns']-row['model_start_ns'])/1e6 and
                row['model_execution_e2e_ms']==(row['model_end_ns']-row['model_start_ns'])/1e6 and
                row['cache_admission_eviction_ms']==(row['generation_end_ns']-row['model_end_ns'])/1e6 and
                row['decode_ms']==sum(s['wall_ms'] for s in ss[1:]) and 0<row['prefill_ms']<=row['model_ttft_ms'], 'timing measurement differs')
        for part,subset in (('prefill',ss[:1]),('decode',ss[1:])):
            for field,value in row[part].items():require(value==sum(s[field] for s in subset), 'prefill/decode wire totals differ')
        require(req['adapter_id']==aid and req['physical_identity']==phys and resolve(registry,req['user_id'],aid)==aid and
                row['request_sha256']==req['request_sha256'] and row['physical_identity']==phys and row['user_id']==req['user_id'] and
                row['ud_pid']==ui['pid'] and row['es_pid']==initial['es_pid'] and row['TEST_FIXTURE_ONLY']==plan['fixture'],'request identity/user/adapter mismatch')
        require(row['status']=='COMPLETE' and len(ss)==len(generated)==row['generated_tokens']>0 and
                row['termination']==termination(req['generation_policy'],generated) and order.finished[rid]['generated_token_ids']==generated,'incomplete generation')
        require(row['mode']==('PARTIAL_PREFIX_HIT' if grant else 'COLD_MISS' if plan['condition']=='SEMCACHE_INTEGRATED' else 'NO_REUSE') and
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
            require(s['wall_ms']==(s['available_monotonic_ns']-s['start_monotonic_ns'])/1e6 and
                    (s['cuda_event_elapsed_ms'] is None if plan['fixture'] else s['cuda_event_elapsed_ms']>=0),'forward timing invalid')
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
            if row['evidence_profile']=='STRICT_AUDIT':
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
                            cached=row['strict_cached_qkv'][f'{l}.{r}']
                            require(clean(cached)==clean(source_step['qkv'][f'{l}.{r}']) and s['input_kv'][l][j]==clean(sourcekv) and token_slice(reconstructed[l][j],kv,0,p)==raw_head_bytes(tensor_data(es,cached),cached['shape'],contract['heads'],2 if cached['dtype']=='float16' else 4),
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

