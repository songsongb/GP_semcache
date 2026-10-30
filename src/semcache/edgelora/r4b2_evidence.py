"""Independent B2 protocol, prefix-KV, residency and suffix-only evidence audit."""
from collections import OrderedDict
import math
from pathlib import Path

from .communication import PREFIX,canonical,digest,digest_bytes,require,validate
from .r3a_artifacts import ROOT,read,sha,unpack
from .r3a_evidence import jsonl
from .r4a_evidence import verify_sources
from .r4b2_analysis import raw_head_bytes,tensor_data,token_slice
from .r4b2_cache import PrefixCache
from .r4b2_prefix_runtime import flags,load_inputs
from .r4b2_session import ForwardOrder,context
from .r4b2_workload import schedule


def audit_run(directory,plan,worker):
    directory=Path(directory);es=directory/'es';m,suite,native,pm,ref,am=load_inputs(plan['prepared']);verify_sources(plan['sources'])
    require(m['sources']==plan['sources'] and worker['execution_status']=='COMPLETE','incomplete or changed B2 worker')
    require(worker['pid']!=worker['ud_pid'] and worker['adapter_parameters_on_es']==0 and worker['user_device_cuda_used'] is False and
        worker['gpu_inference_performed']==(not plan['fixture']) and all(worker.get(k)==v for k,v in flags(plan['fixture']).items()),'B2 process/qualification gate failed')
    if not plan['fixture']:require(worker['gpu']['uuid']==pm['gpu']['uuid']!='UNAVAILABLE' and '3090' in worker['gpu']['name'],'actual GPU identity mismatch')
    contract=read(es/'r4b2_opt_contract.json')
    require(contract['tokenizer']==worker['tokenizer'] and worker['tokenizer']['vocab_sha256']==native['native_result']['tokenizer_vocab_sha256'] and
        worker['tokenizer']['special_tokens']==suite['special_tokens'] and worker['tokenizer']['configuration']==suite['tokenizer_configuration'],
        'actual tokenizer configuration changed')
    builder=PrefixCache(pm,native['base_manifest_sha256'],contract['gpu_uuid'],plan['device'],plan['capacity_bytes'],
        tokenizer=contract['tokenizer'],sources=plan['sources'],fixture=plan['fixture'],contract=contract['implementation'],observed_cache_type=None)
    require(builder.cohort==worker['cohort_sha256']==contract['cohort_sha256'],'OPT/cache/source namespace differs')
    if not plan['fixture']:
        require(contract['implementation']['scope']=='ACTUAL_LOADED_OPT_IMPLEMENTATION' and
            contract['implementation']['versions']==pm['identity']['versions'] and len(contract['implementation']['cache_module_sha256'])==64,
            'installed OPT/cache implementation was not inspected')
        for method in contract['implementation']['methods'].values():require(digest_bytes(method['source'].encode())==method['source_sha256'],'loaded OPT source digest changed')
    rows=jsonl(es/'r4b2_requests.jsonl');tokens=jsonl(es/'r4b2_tokens.jsonl');tasks=schedule(suite)
    require(len(rows)==len(tasks) and all(all(r[k]==v for k,v in task.items()) for r,task in zip(rows,tasks)) and
        rows==jsonl(es/'r4b2_generation_outputs.jsonl') and read(es/'r4b2_schedule.json')==tasks,'B2 request schedule incomplete/reordered')
    by={r['request_id']:r for r in rows};steps={rid:[s for s in tokens if s['request_id']==rid] for rid in by};prompts={p['sample_id']:p for p in suite['prompts']}
    require(sum(map(len,steps.values()))==len(tokens),'orphan token rows')
    triage=jsonl(es/'triage_passes.jsonl');require(len(triage)==2,'both original strict triages required')
    all_steps=triage+tokens
    require(jsonl(es/'r4b2_logit_diagnostics.jsonl')==[s['logit_diagnostic'] for s in all_steps],'missing/reordered nonfinite diagnostics')
    attempts=jsonl(es/'forward_attempts.jsonl');completed=jsonl(es/'forward_completed.jsonl')
    require(len(attempts)==len(completed)==len(all_steps) and [a['forward_id'] for a in attempts]==[s['forward_id'] for s in all_steps] and
        [c['context'] for c in completed]==[s['context'] for s in all_steps],'hidden recomputation or incomplete model forwards')
    frames=jsonl(es/'es_wire.jsonl');sink=jsonl(es/'ud_wire.jsonl');layers=jsonl(es/'ud_layers.jsonl');ud=read(es/'ud_result.json')
    require(len(frames)==len(sink) and ud['execution_status']=='COMPLETE' and ud['pid']==worker['ud_pid'] and ud['user_device_cuda_used'] is False and
        ud['full_model_loaded'] is False and ud['adapter_matrix_sha256']==am['matrix_sha256'] and ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='',
        'UD isolation or endpoint frame counts differ')
    order=ForwardOrder(builder.cohort,pm['identity']['layers']);by_forward={};controls=[];pending=None
    for a,b in zip(frames,sink):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and
            all(a[k]==b[k] for k in ('header','wire_bytes','frame_sha256')),'independent ES/UD frame disagreement')
        h=a['header'];validate(h,h['payload_bytes'])
        require(h['session_id']==worker['session_id'] and h['adapter_sha256']==am['matrix_sha256'] and
            a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],'B2 framing/identity byte ledger differs')
        if h['kind']=='HELLO' and h['request_id']!='handshake':
            require(pending is None,'prefix control interrupted a layer exchange')
            order.control(h['metadata']);controls.append(dict(h['metadata'],request_id=h['request_id']))
        if h['kind']=='HIDDEN':
            require(pending is None,'overlapping B2 layer exchange');order.accept(h,pm['identity']);pending=h
        if h['kind']=='DELTAS':
            require(pending is not None and h['request_id']==pending['request_id'] and h['layer_id']==pending['layer_id'] and
                h['metadata']['hidden_payload_sha256']==pending['payload_sha256'],'delta is not tied to the actual preceding hidden frame')
            order.advance();pending=None
        if h['kind'] in ('HIDDEN','DELTAS'):by_forward.setdefault(h['request_id'],[]).append(a)
    require(pending is None,'incomplete terminal exchange');order.finish()
    require(controls==jsonl(es/'es_prefix_controls.jsonl')==jsonl(es/'ud_prefix_controls.jsonl'),'UD prefix registration ledger differs')
    for op in ('send','recv'):require([r['header']['sequence'] for r in frames if r['operation']==op]==list(range(sum(r['operation']==op for r in frames))),
        'B2 wire sequence discontinuity')
    ud_by={}
    for l in layers:ud_by.setdefault(l['forward_id'],[]).append(l)
    require(set(by_forward)==set(ud_by)=={s['forward_id'] for s in all_steps},'orphan or missing suffix/decode traffic')
    source_contexts={};source_proofs={};prefix_integrity_checks=0
    for row in rows:
        rid=row['request_id'];ss=steps[rid];p=prompts[row['sample_id']];n=len(p['token_ids']);g=suite['policy']['max_new_tokens']
        require(row['target_token_ids']==p['token_ids'] and row['prompt_sha256']==p['prompt_sha256'] and row['policy']==suite['policy'] and
            row['generated_tokens']==len(row['generated_token_ids'])==len(ss)==g and row['decode_forward_count']==g-1,'B2 prompt/policy/generation differs')
        require(all(row.get(k)==v for k,v in flags(plan['fixture']).items()) and row['es_pid']==worker['pid'] and row['ud_pid']==worker['ud_pid'],
            'row process/qualification changed')
        blob=(es/row['logits_file']).read_bytes();require(digest_bytes(blob)==row['logits_sha256'] and row['logits_shape']==[1,g,ss[0]['logit_diagnostic']['logits_shape'][-1]],
            'generation logit bytes/shape changed');values=unpack(blob,'float32');width=row['logits_shape'][-1];require(len(values)==g*width,'truncated generation logits')
        grant=row['source'];prefix=row['reused_prefix_tokens'];require(prefix==(0 if grant is None else grant['prefix_length']) and
            row['newly_computed_prefill_tokens']==n-prefix and row['cached_derived_kv_bytes']==0 and row['prefix_q_consumed'] is False,
            'Raw/KV accounting or suffix-only claim differs')
        if row['mode']=='PARTIAL_PREFIX_HIT':
            source=by[grant['source_request_id']];require(source['mode']=='COLD_MISS' and source['target_token_ids']==p['token_ids'][:prefix] and
                source['request_end_monotonic_ns']<=row['request_start_monotonic_ns'] and prefix==row['prefix_length'],
                'partial HIT source is not a completed standalone exact prefix')
        for i,s in enumerate(ss):
            ids=p['token_ids'][prefix:] if i==0 else [row['generated_token_ids'][i-1]];old=prefix if i==0 else n+i-1
            expected=context(rid,i,ids,p['token_ids'],old,g,row['mode'],builder.cohort,grant)
            require(s['context']==expected and s['step']==i and s['executed_projection_calls']==3*pm['identity']['layers'] and
                s['executed_projection_token_rows']==len(ids)*3*pm['identity']['layers'] and
                s['prefix_projection_token_rows_skipped']==(prefix*3*pm['identity']['layers'] if i==0 else 0),'hidden full prefill or wrong suffix operation ledger')
            require(s['past']['sequence_length']==n+i and len(s['past']['layers'])==pm['identity']['layers'] and
                (s['input_past'] is None if old==0 else s['input_past']['sequence_length']==old),'actual HF KV lengths differ')
            kv_shape=[1,worker['heads'],n+i,pm['identity']['hidden_size']//worker['heads']]
            require(all(x['k_shape']==x['v_shape']==kv_shape and x['dtype']==pm['identity']['hidden_dtype'] and x['device']==plan['device'] for x in s['past']['layers']),
                'actual per-layer KV shape/dtype/device differs')
            require(len(s['kv'])==pm['identity']['layers'] and all(kv[role]['shape']==kv_shape and
                kv[role]['dtype']==pm['identity']['hidden_dtype'] and kv[role]['device']==plan['device'] for kv in s['kv'] for role in 'kv'),
                'actual KV contents ledger has inconsistent head layout/dtype/device')
            if i:
                previous=[[{k:v for k,v in ss[i-1]['kv'][l][role].items() if k!='file'} for role in 'kv'] for l in range(pm['identity']['layers'])]
                require(s['input_kv']==previous,'decode input KV contents differ from actual prior model output')
            observed=s['position_observation'];require(observed['positions']==expected['positions'] and observed['past_length']==old,'observed positions differ')
            if not plan['fixture']:require(observed['source']=='ACTUAL_OPT_POSITION_MODULE' and observed['lookup_output_equal'] is True,'actual OPT positions not verified')
            d=s['logit_diagnostic'];require(d['all_logits']['nonfinite']==d['last_position_logits']['nonfinite']==0 and
                d['total_input_length']==n+i and d['input_length']==len(ids) and d['padding_tokens']==0,'nonfinite/incomplete per-step diagnostics')
            piece=blob[i*width*4:(i+1)*width*4];vv=unpack(piece,'float32');require(s['logits_sha256']==digest_bytes(piece) and
                s['selected_token_id']==row['generated_token_ids'][i]==max(range(width),key=vv.__getitem__),'greedy selected token/logit evidence differs')
            require(row['request_start_monotonic_ns']<=s['start_monotonic_ns']<=s['available_monotonic_ns']<=row['request_end_monotonic_ns'],
                'forward clocks outside request')
            fs=by_forward[s['forward_id']];uls=ud_by[s['forward_id']]
            require(len(fs)==2*pm['identity']['layers'] and len(uls)==pm['identity']['layers'],'missing/extra actual suffix projection exchange')
            for kind,field,shape,dtype in (('HIDDEN','hidden_payload_bytes',[1,len(ids),pm['identity']['hidden_size']],pm['identity']['hidden_dtype']),
                    ('DELTAS','delta_payload_bytes',[3,1,len(ids),pm['identity']['hidden_size']],pm['identity']['delta_dtype'])):
                selected=[f for f in fs if f['header']['kind']==kind]
                require([f['header']['layer_id'] for f in selected]==list(range(pm['identity']['layers'])) and
                    sum(f['header']['payload_bytes'] for f in selected)==s[field] and all(f['header']['shape']==shape and f['header']['dtype']==dtype for f in selected),
                    'suffix layer shapes/payload bytes differ')
            for f in fs:
                h=f['header'];require((h['metadata']==dict(r4b2_forward=expected)) if h['kind']=='HIDDEN' else h['metadata']['r4b2_forward_sha256']==digest(expected),
                    'wire prefix identity differs')
                require(s['start_monotonic_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=s['available_monotonic_ns'],'wire outside step wall')
            require([l['layer'] for l in uls]==list(range(pm['identity']['layers'])) and all(l['context']==expected for l in uls),'UD ordering/context differs')
            for op,field in (('send','es_to_ud_wire_bytes'),('recv','ud_to_es_wire_bytes')):
                require(s[field]==sum(f['wire_bytes'] for f in fs if f['operation']==op),'directional suffix bytes differ')
            require(s['application_wire_bytes']==sum(f['wire_bytes'] for f in fs) and len(s['qkv'])==3*pm['identity']['layers'],'QKV/network evidence incomplete')
            for q in s['qkv'].values():require(q['shape']==[1,len(ids),pm['identity']['hidden_size']] and q['dtype']==pm['identity']['hidden_dtype'] and
                q['device']==plan['device'] and tensor_data(es,q),'actual suffix QKV shape/storage differs')
            if i==0:
                for op in s['components'].values():require(op['shape']==[1,len(ids),pm['identity']['hidden_size']] and tensor_data(es,op),'suffix attention/FFN skipped')
                if not plan['fixture']:require(len(s['components'])==3*pm['identity']['layers'],'missing actual suffix attention/FFN/layer observations')
                for layer,kvs in enumerate(s['kv']):
                    for role in 'kv':
                        data=tensor_data(es,kvs[role]);q=s['qkv'][f'{layer}.{role}'];raw=tensor_data(es,q)
                        require(token_slice(data,kvs[role],old,n)==raw_head_bytes(raw,q['shape'],worker['heads'],{'float16':2,'float32':4}[q['dtype']]),
                            'Raw K/V transform differs from actual appended HF K/V')
                        if grant is not None:
                            source_step=steps[grant['source_request_id']][0];base=source_step['kv'][layer][role]
                            require(token_slice(data,kvs[role],0,prefix)==tensor_data(es,base) and s['input_kv'][layer]['kv'.index(role)]==
                                {k:v for k,v in base.items() if k!='file'},'cached prefix K/V mutated or incorrect initial HF cache')
                            prefix_integrity_checks+=1
            if row['mode']=='COLD_MISS' and i==0:
                ctx=builder.context_for(p);key=digest(ctx);proof=s['raw_kv_proof']
                require(proof['prefix_length']==n and proof['heads']==worker['heads'] and proof['q_consumed'] is False and len(proof['layers'])==pm['identity']['layers'],
                    'standalone source proof incomplete')
                for l,item in enumerate(proof['layers']):
                    for role in 'kv':
                        raw_spec={k:v for k,v in s['qkv'][f'{l}.{role}'].items() if k!='file'}
                        actual={k:v for k,v in s['kv'][l][role].items() if k!='file'}
                        require(item['raw_'+role]==raw_spec and item['derived_'+role]==item['actual_'+role]==actual and item['bitwise_equal'] is True,
                            'source proof not backed by captured actual Raw/HF K/V')
                source_contexts[key]=ctx;source_proofs[key]=dict(request_id=rid,proof=proof)
        for k in row['prefill_accounting']:
            require(row['prefill_accounting'][k]==ss[0][k] and row['decode_accounting'][k]==sum(s[k] for s in ss[1:]),'Prefill/Decode accounting differs')
        require(row['ttft_ms']==(ss[0]['available_monotonic_ns']-row['request_start_monotonic_ns'])/1e6 and
            row['e2e_wall_ms']==(row['request_end_monotonic_ns']-row['request_start_monotonic_ns'])/1e6 and
            row['prefix_projection_token_rows_skipped']==prefix*3*pm['identity']['layers'],'request clocks/projection savings differ')
    events=jsonl(es/'r4b2_cache_events.jsonl');state=OrderedDict();peak=0
    for index,e in enumerate(events):
        require(e['sequence']==index and e['duration_ms']>=0 and e['capacity_bytes']==plan['capacity_bytes'],'cache event version/clock/capacity differs')
        key=e['context_sha256'];kind=e['kind']
        if kind in ('EVICT','REJECT','INVALIDATE'):
            require(key in state and e['released_bytes']==state[key]['bytes'] and e['source_request_id']==state[key]['source_request_id'],'invalid cache eviction/release')
            if kind=='EVICT':require(key==next(iter(state)),'not actual LRU victim')
            del state[key]
        elif kind=='ADMISSION' and e['allowed']:
            require(key in source_contexts and key not in state and e['request_id']==source_proofs[key]['request_id'],'uncompleted/foreign prefix admission')
            r=by[e['request_id']];n=len(r['target_token_ids']);size=3*pm['identity']['layers']*n*pm['identity']['hidden_size']*{'float16':2,'float32':4}[pm['identity']['hidden_dtype']]
            require(e['candidate_bytes']==size and r['final_decode_available_monotonic_ns']<=e['monotonic_ns']<=r['request_end_monotonic_ns'],
                'wrong byte size or admission before completed generation')
            state[key]=dict(context_sha256=key,bytes=size,source_request_id=e['request_id'])
        elif kind in ('HIT','PHYSICAL_REUSE','INSERT'):
            require(key in state and e['source_request_id']==state[key]['source_request_id'],'HIT after eviction or wrong cache source')
            if kind=='HIT':state.move_to_end(key)
            if kind=='PHYSICAL_REUSE':require(e['raw_kv_projections_consumed']==2*pm['identity']['layers'] and
                e['cached_q_consumed'] is False and 'qkv_outputs' not in e,'cached Q was incorrectly claimed as physically consumed')
        elif kind=='MISS':require(key is None or key not in state,'false MISS on valid entry')
        elif kind=='ADMISSION':require(not e['allowed'],'invalid admission')
        else:require(False,'unsupported cache event')
        total=sum(x['bytes'] for x in state.values());peak=max(peak,total)
        require(total<=plan['capacity_bytes'] and e['occupancy_bytes']==total and e['entry_count']==len(state) and e['lru_to_mru']==list(state) and
            e['resident_entries']==list(state.values()) and e['peak_occupancy_bytes']==peak,'physical residency/LRU accounting differs')
    for g in controls:
        if g['action']=='REGISTER_PREFIX':
            source=source_proofs[g['entry_id']];require(g['source_request_id']==source['request_id'] and g['proof_sha256']==digest(source['proof']) and
                g['prefix_token_ids']==by[g['source_request_id']]['target_token_ids'],'UD registered invalid prefix source proof')
    for row in rows:
        selected=[e for e in events if e['request_id']==row['request_id']]
        require(row['cache_events']==selected and row['cached_raw_qkv_bytes']==row['cache_after']['occupancy_bytes'],'request cache accounting differs')
        if row['mode']=='PARTIAL_PREFIX_HIT':
            hits=[e for e in selected if e['kind']=='HIT'];require(len(hits)==1 and hits[0]['context_sha256']==row['source']['entry_id'] and
                hits[0]['source_request_id']==row['source']['source_request_id'] and row['source']['proof_sha256']==digest(source_proofs[row['source']['entry_id']]['proof']),
                'partial HIT not backed by actual resident/source proof lookup')
        if row['mode']=='NO_REUSE':require(not selected and row['cache_before']==row['cache_after'],'NO_REUSE mutated cross-request cache')
        for clock,key in (('request_start_monotonic_ns','cache_before'),('request_end_monotonic_ns','cache_after')):
            preceding=[e for e in events if e['monotonic_ns']<=row[clock]]
            snapshot=({k:preceding[-1][k] for k in ('occupancy_bytes','entry_count','capacity_bytes','peak_occupancy_bytes','lru_to_mru','resident_entries')}
                if preceding else dict(occupancy_bytes=0,entry_count=0,capacity_bytes=plan['capacity_bytes'],peak_occupancy_bytes=0,lru_to_mru=[],resident_entries=[]))
            require(row[key]==snapshot,'cache before/after differs from independent event replay')
        request_frames=[f for f in frames if row['request_start_monotonic_ns']<=f['monotonic_start_ns'] and f['monotonic_end_ns']<=row['request_end_monotonic_ns']]
        require(row['request_total_wire_bytes']==sum(f['wire_bytes'] for f in request_frames) and
            row['cache_control_wire_bytes']==sum(f['wire_bytes'] for f in request_frames if f['header']['kind'] not in ('HIDDEN','DELTAS')),
            'cache control/total request communication bytes differ')
    last=events[-1]
    require(worker['cache_final']=={k:last[k] for k in ('occupancy_bytes','entry_count','capacity_bytes','peak_occupancy_bytes','lru_to_mru','resident_entries')},
        'worker final cache occupancy differs from audited events')
    negatives=read(es/'r4b2_negative_controls.json');names={n['case'] for n in negatives}
    required={'no_matching_prefix','one_token_difference','nonprefix_offset','adapter_mismatch','model_mismatch','mask_mismatch','position_mismatch',
        'bad_digest','unsupported_length','corrupt_proof','insufficient_capacity','evicted_prefix'}
    if len(suite['conditions'])>1:required.add('corrupt_tensor')
    require(required.issubset(names) and all(n['status']=='EXPECTED_REJECTION' and n['data_frames']==0 and n['recomputed'] is False for n in negatives),
        'missing/accepted invalid HIT negative controls')
    failures=jsonl(es/'r4b2_failed_requests.jsonl');require({n['request_id'] for n in negatives if 'request_id' in n}=={f['request_id'] for f in failures},
        'failed requests lost diagnostic evidence')
    failed={f['request_id']:f for f in failures};first=suite['conditions'][0];original=prompts[first['targets'][0]]
    all_prefixes=[ctx['token_ids'] for ctx in source_contexts.values()]
    for negative in negatives:
        name=negative['case']
        if name=='insufficient_capacity':
            require(negative['capacity_bytes']==1 and negative['occupancy_bytes']==0 and
                any(e['kind']=='ADMISSION' and not e['allowed'] and e['candidate_bytes']>1 for e in negative['events']),
                'capacity negative did not exercise actual denied byte admission');continue
        f=failed[negative['request_id']];require(f['generation_step'] is None and f['status']=='MISS_OR_REJECT' and
            f['cache_cohort_sha256']==builder.cohort,'invalid HIT reached model execution or lost request identity')
        ids=f['target_token_ids']
        if name in ('no_matching_prefix','one_token_difference','nonprefix_offset'):
            require(not any(ids[:len(prefix)]==prefix for prefix in all_prefixes),'negative token control actually matches a prefix')
        if name=='one_token_difference':require(sum(a!=b for a,b in zip(ids,original['token_ids']))==1,'one-token negative malformed')
        if name=='nonprefix_offset':require(ids[1:first['prefix_length']+1]==original['token_ids'][:first['prefix_length']],'non-prefix-offset control malformed')
        if name in ('adapter_mismatch','model_mismatch'):require(f['requested_cohort_sha256']!=builder.cohort,'identity negative did not differ')
        if name=='mask_mismatch':require(f['attention_mask']!=[1]*len(ids),'mask negative did not differ')
        if name=='position_mismatch':require(f['positions']!=list(range(len(ids))),'position negative did not differ')
        if name=='bad_digest':require(f['prompt_sha256']!=digest(dict(tokens=ids,mask=f['attention_mask'])),'digest negative did not differ')
        if name=='unsupported_length':require(f['expected_prefix'] not in (32,64,96),'unsupported-boundary control malformed')
        if name=='corrupt_proof':
            proof=source_proofs[negative['entry_id']]['proof'];altered=dict(proof,prefix_length=proof['prefix_length']+1)
            require(negative['original_proof_sha256']==digest(proof) and negative['corrupted_proof_sha256']==digest(altered),'proof corruption evidence differs')
        if name=='corrupt_tensor':
            source_id=source_proofs[negative['entry_id']]['request_id'];q=steps[source_id][0]['qkv']['0.q']
            require(negative['original_q']=={k:v for k,v in q.items() if k!='file'} and tensor_data(es,negative['corrupted_q'])!=tensor_data(es,q),
                'corrupted cache tensor was not independently evidenced')
    for name in ('lifecycle.json','ud_lifecycle.json'):
        life=read(es/name);require(life['returncode']==0 and life['all_children_exited'] and not life['forced_termination'],'unclean B2 child lifecycle')
    require(read(es/'lifecycle.json')['tracked_descendants']['all_stopped'],'orphan B2 descendants')
    for use in (False,True):
        from .r4a_parity import compare
        d=es/f'triage_es_{int(use)}';record=read(d/'forward.json');data=(d/record['logits_file']).read_bytes()
        actual=compare(Path(m['native_prepared'])/f'triage_native_{int(use)}',d,unpack(data,'float32'),record['shape'],fixture=plan['fixture'])
        saved=read(es/'r4b2_parity_diagnostics.json')
        require(saved['original_input_use_cache_true' if use else 'original_input_use_cache_false']==actual and
            saved['NATIVE_PEFT_STRICT_PARITY_PASS'] is False and saved['frozen_thresholds_unchanged'] is True,'deferred Native gate/metrics changed')
    return dict(status='PASS',requests=len(rows),generation_steps=len(tokens),standalone_prefix_sources=len(source_proofs),
        partial_hits=sum(r['mode']=='PARTIAL_PREFIX_HIT' for r in rows),prefix_kv_integrity_checks=prefix_integrity_checks,
        peak_raw_qkv_bytes=peak,persistent_derived_kv_bytes=0,negative_controls=len(negatives),independent_wire_agreement=True,
        suffix_only_forwards=True,cache_factory_runtime_verified=True)


def load_run(directory):
    directory=Path(directory).resolve();r=read(directory/'r4b2_results.json')
    require(r['execution_status']=='COMPLETE' and r.get('audit',{}).get('status')=='PASS' and not r.get('analysis_only'),'completed audited B2 run required')
    required={'es/es_plan.json','es/es_result.json','es/es_wire.jsonl','es/ud_wire.jsonl','es/ud_layers.jsonl','es/r4b2_requests.jsonl','es/r4b2_tokens.jsonl',
        'es/r4b2_cache_events.jsonl','es/r4b2_negative_controls.json','es/r4b2_opt_contract.json','r4b2_decision.json','r4b2_fidelity.json','r4b2_quality.json'}
    require(required.issubset(r['output_hashes']),'missing required B2 output fingerprints')
    for name,h in r['output_hashes'].items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and sha(p)==h,'B2 artifact fingerprint differs: '+name)
    verify_sources(r['sources']);plan=read(directory/'es/es_plan.json');worker=read(directory/'es/es_result.json')
    require(all(r.get(k)==v for k,v in flags(plan['fixture']).items()),'B2 report qualification/scope was promoted')
    decision=read(directory/'r4b2_decision.json')
    require(decision==r['decision'] and all(decision.get(k)==v for k,v in flags(plan['fixture']).items()),
        'B2 decision qualification/scope changed')
    if plan['fixture']:
        require(not any(v for k,v in decision.items() if k.endswith('_PASS') or k in ('R4B2_TTFT_BENEFIT_OBSERVED','R4B2_E2E_BENEFIT_OBSERVED')),
            'CPU fixture cannot promote physical/strict/performance GPU flags')
    require(plan['sources']==r['sources'] and audit_run(directory,plan,worker)==r['audit'],'independent B2 audit differs')
    return r,plan,worker
