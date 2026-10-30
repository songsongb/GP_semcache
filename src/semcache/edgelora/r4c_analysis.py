"""Read-only C1 audit and distinct gold, fidelity, and cost analyses."""
from collections import OrderedDict
from pathlib import Path

from .communication import PREFIX, canonical, digest, digest_bytes, require, validate
from .r3a_artifacts import read, unpack, write, write_csv, sha, validate_prepared
from .r3a_evidence import jsonl
from .r4a_evidence import verify_sources
from .r4b2_analysis import raw_head_bytes, tensor_data, token_slice, tensor_metrics
from .r4c_evidence import verify_directory
from .r4c_metrics import gold_score, numerical
from .r4c_prepare import load_prepared
from .r4c_runtime import flags
from .r4c_session import ForwardOrder, context
from .r4c_workload import termination
from .r4c_cache import PrefixCache


def clean(s): return {k:v for k,v in s.items() if k!='file'}


def append_head_bytes(previous,raw_projection,*,heads,old_length,head_dim,element_bytes):
    """Append new per-head K/V rows, preserving head-major HF layout."""
    old_width=old_length*head_dim*element_bytes;new_width=head_dim*element_bytes
    require(len(previous)==heads*old_width and len(raw_projection)==heads*new_width,'Decode KV reconstruction dimensions differ')
    return b''.join(previous[h*old_width:(h+1)*old_width]+raw_projection[h*new_width:(h+1)*new_width] for h in range(heads))


def audit(directory):
    directory=Path(directory);es=directory/'es';plan=read(es/'es_plan.json');worker=read(es/'es_result.json')
    verify_sources(plan['sources']);manifest,prompts,tasks=load_prepared(plan['prepared'])
    for name in ('r4c_dataset_manifest.json','r4c_prepared_prompts.jsonl','r4c_prefix_coverage.json','r4c_schedule.json'):
        require((directory/name).read_bytes()==(Path(plan['prepared'])/name).read_bytes(),'run C0 artifact differs from immutable preparation')
    require(worker['execution_status']=='COMPLETE' and worker['pid']!=worker['ud_pid'] and worker['user_device_cuda_used'] is False and
            worker['gpu_inference_performed']==(not plan['fixture']), 'incomplete/unqualified C1 worker')
    contract=read(es/'r4c_opt_contract.json');identity=contract['identity'];cohort=contract['cohort_sha256'];layers=identity['layers']
    pm,_,am=validate_prepared(plan['base_prepared'])
    require(sha(Path(plan['base_prepared'])/'r3a_prepare_manifest.json')==plan['base_manifest_sha256'] and pm['identity']==identity,'base/model/adapter identity differs')
    builder=PrefixCache(pm,plan['base_manifest_sha256'],'TEST_FIXTURE_ONLY' if plan['fixture'] else contract['gpu']['uuid'],plan['device'],plan['capacity_bytes'],
        tokenizer=contract['tokenizer'],sources=plan['sources'],fixture=plan['fixture'],contract=contract['implementation'],observed_cache_type=None)
    require(builder.cohort==cohort,'actual source/model/adapter/device/tokenizer namespace differs')
    require(contract['tokenizer']==manifest['model_tokenizer'] and contract['implementation']['context_limit']==manifest['context_limit'], 'model/tokenizer/config identity differs')
    if not plan['fixture']:
        require(contract['gpu']['uuid']!='UNAVAILABLE' and '3090' in contract['gpu']['name'],'allocated RTX3090 evidence missing')
        for method in contract['implementation']['methods'].values():require(digest_bytes(method['source'].encode())==method['source_sha256'],'installed OPT source differs')
    rows=jsonl(es/'r4c_requests.jsonl');steps=jsonl(es/'r4c_tokens.jsonl');by={r['request_id']:r for r in rows}
    require(len(rows)==len(tasks) and len(by)==len(rows) and [r['request_id'] for r in rows]==[t['request_id'] for t in tasks], 'missing/reordered/duplicate cohort requests; failures cannot be dropped')
    require([r['context'] for r in jsonl(es/'forward_attempts.jsonl')]==[s['context'] for s in steps]==[r['context'] for r in jsonl(es/'forward_completed.jsonl')], 'hidden/incomplete full-prefill forwards')
    require(jsonl(es/'r4c_logit_diagnostics.jsonl')==[s['logit_diagnostic'] for s in steps], 'missing nonfinite diagnostics')
    frames=jsonl(es/'es_wire.jsonl');other=jsonl(es/'ud_wire.jsonl');ud=read(es/'ud_result.json')
    require(len(frames)==len(other) and ud['execution_status']=='COMPLETE' and ud['pid']==worker['ud_pid'] and not ud['full_model_loaded'] and not ud['user_device_cuda_used'] and
        ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='', 'UD isolation/independent frame evidence differs')
    order=ForwardOrder(cohort,layers,manifest['context_limit']);pending=None;byforward={};controls=[]
    for a,b in zip(frames,other):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and all(a[k]==b[k] for k in ('header','frame_sha256','wire_bytes')), 'independent TCP ledger disagreement')
        h=a['header'];validate(h,h['payload_bytes'])
        require(h['session_id']==worker['session_id'] and h['adapter_sha256']==identity['adapter_matrix_sha256'] and a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'], 'frame identity/byte ledger differs')
        if h['kind']=='HELLO' and h['request_id']!='handshake':
            require(pending is None,'control during layer exchange');order.control(h['metadata']);controls.append(dict(h['metadata'],request_id=h['request_id']))
        if h['kind']=='HIDDEN':
            require(pending is None,'overlapping layer forwards');order.accept(h,identity);pending=h
        if h['kind']=='DELTAS':
            require(pending is not None and h['request_id']==pending['request_id'] and h['layer_id']==pending['layer_id'] and
                h['metadata']['hidden_payload_sha256']==pending['payload_sha256'] and h['metadata']['r4c_forward_sha256']==digest(pending['metadata']['r4c_forward']), 'delta not tied to actual hidden/context')
            order.advance();pending=None
        if h['kind'] in ('HIDDEN','DELTAS'):byforward.setdefault(h['request_id'],[]).append(a)
    require(pending is None,'incomplete wire exchange');order.finish()
    require(controls==jsonl(es/'es_prefix_controls.jsonl')==jsonl(es/'ud_prefix_controls.jsonl'),'completion/grant controls differ')
    for op in ('send','recv'):require([r['header']['sequence'] for r in frames if r['operation']==op]==list(range(sum(r['operation']==op for r in frames))), 'wire sequence discontinuity')
    ud_layers=jsonl(es/'ud_layers.jsonl');require(len(ud_layers)==len(steps)*layers and set(byforward)=={s['forward_id'] for s in steps},'orphan/missing projection traffic')
    bysteps={rid:[s for s in steps if s['request_id']==rid] for rid in by};strict_checks=0;kv_reconstruction_checks=0;sources={};tensor_files=set()
    for row,task in zip(rows,tasks):
        rid=row['request_id'];ss=bysteps[rid];target=task['prompt']['token_ids'];p=row['reused_prefix_tokens'];g=row['generated_token_ids'];policy=task['policy'];grant=row['source']
        reconstructed=None
        require(all(row[k]==task[k] for k in ('sample_id','mode','pair_id','dataset','policy','evidence_profile','workload_label')) and row['target_token_ids']==target and
                row['prompt_sha256']==task['prompt']['prompt_sha256'] and len(g)==row['generated_tokens']==len(ss) and row['termination']==termination(policy,g), 'request prompt/policy/termination differs')
        require(order.finished[rid]['generated_token_ids']==g and order.finished[rid]['termination']==row['termination'],'UD completion receipt differs from generated output')
        require(all(row.get(k)==v for k,v in flags(plan['fixture']).items()), 'qualification flags promoted')
        require(row['model_identity_sha256']==pm['identity_sha256'] and row['adapter_matrix_sha256']==am['matrix_sha256'] and
            row['tokenizer_sha256']==digest(contract['tokenizer']) and row['evaluation_policy_sha256']==digest(task['policy']), 'request model/adapter/tokenizer/policy fingerprint differs')
        require(p==(0 if grant is None else grant['prefix_length']) and row['newly_computed_prefill_tokens']==len(target)-p and not row['cached_prefix_q_consumed'] and
                row['projection_kernels_entirely_skipped']==0 and row['cached_derived_kv_bytes']==0,'physical skipped rows/storage claims differ')
        if grant is not None:
            require(grant['source_request_id']==task['source_request_id'] and grant['source_request_id'] in sources and sources[grant['source_request_id']]['target_token_ids']==target[:p], 'HIT not backed by completed standalone exact token source')
        blob=(es/row['logits_file']).read_bytes();require(digest_bytes(blob)==row['logits_sha256'],'logit file digest differs')
        width=row['logits_shape'][-1];vv=unpack(blob,'float32');require(width==contract['implementation']['vocabulary_size'] and len(vv)==len(g)*width and row['logits_shape']==[1,len(g),width], 'truncated/model-inconsistent logits')
        for i,s in enumerate(ss):
            old=p if i==0 else len(target)+i-1;ids=target[p:] if i==0 else [g[i-1]]
            c=context(rid,i,ids,target,old,policy,row['mode'],cohort,grant,context_limit=manifest['context_limit'])
            require(s['context']==c and s['executed_projection_calls']==3*layers and s['executed_projection_token_rows']==len(ids)*3*layers and
                    s['prefix_projection_token_rows_skipped']==(p*3*layers if i==0 else 0), 'hidden full-prefill or skipped suffix operations')
            require(s['past']['sequence_length']==len(target)+i and (s['input_past'] is None if old==0 else s['input_past']['sequence_length']==old), 'actual KV length differs')
            heads=contract['implementation']['heads'];kvshape=[1,heads,len(target)+i,identity['hidden_size']//heads]
            require(len(s['past']['layers'])==len(s['kv'])==layers and all(x['k_shape']==x['v_shape']==kvshape and x['dtype']==identity['hidden_dtype'] and x['device']==plan['device'] for x in s['past']['layers']), 'per-layer actual KV layout/dtype/device differs')
            require(all(x['shape']==kvshape and x['dtype']==identity['hidden_dtype'] and x['device']==plan['device'] for kv in s['kv'] for x in kv.values()),'KV contents ledger shape differs')
            require(s['position_observation']['positions']==c['positions'] and s['position_observation']['past_length']==old, 'actual positional alignment differs')
            if not plan['fixture']:require(s['position_observation']['lookup_output_equal'] is True and len(s['components'])==3*layers,'actual OPT position/attention/FFN observation missing')
            d=s['logit_diagnostic'];require(all(d[k]['nonfinite']==0 for k in ('all_logits','valid_positions','padded_positions','last_position')) and d['padding_tokens']==0,'nonfinite/padded evidence')
            v=vv[i*width:(i+1)*width];require(s['selected_token_id']==g[i]==max(range(width),key=v.__getitem__) and s['logits_sha256']==digest_bytes(blob[i*width*4:(i+1)*width*4]), 'greedy/logit evidence differs')
            fs=byforward[s['forward_id']];uls=[u for u in ud_layers if u['forward_id']==s['forward_id']]
            require(len(fs)==2*layers and [u['layer'] for u in uls]==list(range(layers)) and all(u['context']==c for u in uls), 'per-layer communication missing/extra')
            require(s['hidden_payload_bytes']==sum(f['header']['payload_bytes'] for f in fs if f['header']['kind']=='HIDDEN') and
                s['delta_payload_bytes']==sum(f['header']['payload_bytes'] for f in fs if f['header']['kind']=='DELTAS') and s['application_wire_bytes']==sum(f['wire_bytes'] for f in fs), 'payload/wire accounting differs')
            for op,key in (('send','es_to_ud_wire_bytes'),('recv','ud_to_es_wire_bytes')):require(s[key]==sum(f['wire_bytes'] for f in fs if f['operation']==op),'directional wire bytes differ')
            require(len(s['qkv'])==3*layers,'missing QKV evidence')
            require(all(v['shape']==[1,len(ids),identity['hidden_size']] and v['dtype']==identity['hidden_dtype'] and v['device']==plan['device'] for v in s['qkv'].values()), 'suffix QKV shape/dtype/device differs')
            require(all(v['shape']==[1,len(ids),identity['hidden_size']] for v in s['components'].values()),'attention/FFN/layer processed incorrect token rows')
            require(row['request_start_monotonic_ns']<=s['start_monotonic_ns']<=s['available_monotonic_ns']<=row['request_end_monotonic_ns'], 'step clocks outside request')
            for value in [*s['qkv'].values(),*s['components'].values(),*(q for kv in s['kv'] for q in kv.values())]:
                if 'file' in value:
                    tensor_data(es,value);tensor_files.add(value['file'])
            if row['evidence_profile']=='STRICT_AUDIT':
                width_bytes={'float16':2,'float32':4}[identity['hidden_dtype']]
                if i==0:reconstructed=[[tensor_data(es,s['kv'][l][r]) for r in 'kv'] for l in range(layers)]
                for l in range(layers):
                    for j,r in enumerate('kv'):
                        raw=s['qkv'][f'{l}.{r}'];transformed=raw_head_bytes(tensor_data(es,raw),raw['shape'],heads,width_bytes)
                        if i==0:
                            require(token_slice(reconstructed[l][j],s['kv'][l][r],old,len(target))==transformed,'prefill Raw K/V / HF KV bytes differ')
                        else:
                            reconstructed[l][j]=append_head_bytes(reconstructed[l][j],transformed,heads=heads,old_length=old,
                                head_dim=identity['hidden_size']//heads,element_bytes=width_bytes)
                        require(digest_bytes(reconstructed[l][j])==s['kv'][l][r]['sha256'],'reconstructed Decode KV digest differs from actual HF KV')
                        kv_reconstruction_checks+=1
            if i:
                require(s['input_kv']==[[clean(bysteps[rid][i-1]['kv'][l][r]) for r in 'kv'] for l in range(layers)],'decode past changed prior history')
            if i==0 and grant is not None:
                source_step=bysteps[grant['source_request_id']][0]
                proof=source_step['raw_kv_proof'];require(grant['proof_sha256']==digest(proof),'prefix proof identity differs')
                for l in range(layers):
                    for j,r in enumerate('kv'):
                        require(s['input_kv'][l][j]==proof['layers'][l]['actual_'+r], 'initial target KV differs from admitted source')
                        if row['evidence_profile']=='STRICT_AUDIT':
                            data=tensor_data(es,s['kv'][l][r]);base=tensor_data(es,source_step['kv'][l][r]);raw=tensor_data(es,s['qkv'][f'{l}.{r}'])
                            require(token_slice(data,s['kv'][l][r],0,p)==base and token_slice(data,s['kv'][l][r],old,len(target))==raw_head_bytes(raw,s['qkv'][f'{l}.{r}']['shape'],proof['heads'],{'float16':2,'float32':4}[identity['hidden_dtype']]), 'physical prefix/suffix K/V bytes differ')
                            strict_checks+=1
            for field in row['prefill']:
                require(row['prefill'][field]==ss[0][field] and row['decode'][field]==sum(x[field] for x in ss[1:]),'Prefill/Decode request accounting differs')
        require(row['ttft_ms']==(ss[0]['available_monotonic_ns']-row['request_start_monotonic_ns'])/1e6 and row['e2e_ms']==(row['request_end_monotonic_ns']-row['request_start_monotonic_ns'])/1e6,'TTFT/E2E clocks differ')
        if row['mode']=='COLD_MISS':
            proof=ss[0]['raw_kv_proof']; require(proof['prefix_length']==len(target) and len(proof['layers'])==layers and not proof['q_consumed'],'source proof missing')
            for l,item in enumerate(proof['layers']):
                for r in 'kv':require(item['raw_'+r]==clean(ss[0]['qkv'][f'{l}.{r}']) and item['derived_'+r]==item['actual_'+r]==clean(ss[0]['kv'][l][r]) and item['bitwise_equal'], 'source Raw/HF KV proof disconnected')
            sources[rid]=row
    # Replay owned bytes and LRU independently; cached raw storage counted once.
    state=OrderedDict();peak=0;events=jsonl(es/'r4c_cache_events.jsonl')
    for index,e in enumerate(events):
        key=e.get('context_sha256');kind=e['kind'];require(e['sequence']==index and e['capacity_bytes']==plan['capacity_bytes'],'cache ledger sequence/cap differs')
        if kind in ('EVICT','REJECT','INVALIDATE'):
            require(key in state and e['released_bytes']==state[key]['bytes'],'cache release differs')
            if kind=='EVICT' and e['reason']=='capacity_lru':require(key==next(iter(state)),'wrong LRU victim')
            del state[key]
        elif kind=='ADMISSION' and e['allowed']:
            source=sources[e['request_id']];require(key not in state and source['mode']=='COLD_MISS','foreign/incomplete cache admission')
            expected=3*layers*len(source['target_token_ids'])*identity['hidden_size']*{'float16':2,'float32':4}[identity['hidden_dtype']]
            require(e['candidate_bytes']==expected,'physical owned byte size differs')
            state[key]=dict(context_sha256=key,bytes=expected,source_request_id=e['request_id'])
        elif kind in ('INSERT','HIT','PHYSICAL_REUSE'):
            require(key in state and e['source_request_id']==state[key]['source_request_id'],'HIT after eviction / wrong source')
            if kind=='HIT':state.move_to_end(key)
        elif kind=='MISS':require(key is None or key not in state,'false MISS')
        else:require(kind=='ADMISSION' and not e['allowed'],'unknown cache event')
        total=sum(v['bytes'] for v in state.values());peak=max(peak,total)
        require(e['occupancy_bytes']==total<=plan['capacity_bytes'] and e['lru_to_mru']==list(state) and e['resident_entries']==list(state.values()) and e['peak_occupancy_bytes']==peak,'cache physical occupancy differs')
    negatives=read(es/'r4c_negative_controls.json');failures=jsonl(es/'r4c_failed_requests.jsonl')
    require(len(negatives)==8 and {n['request_id'] for n in negatives}=={f['request_id'] for f in failures} and
        all(n['status']=='EXPECTED_REJECTION' and n['tensor_frames']==0 for n in negatives) and
        all(f['generation_step'] is None for f in failures), 'invalid HIT did not fail before forward or lost diagnostic evidence')
    for n in negatives:require(not any(f['header']['request_id'].startswith(n['request_id']+'.') for f in frames),'invalid HIT sent tensor network data')
    require(worker['cache_final']['occupancy_bytes']==sum(v['bytes'] for v in state.values()) and worker['cache_final']['lru_to_mru']==list(state), 'final cache differs from event replay')
    for name in ('lifecycle.json','ud_lifecycle.json'):
        life=read(es/name);require(life['returncode']==0 and life['all_children_exited'] and not life['forced_termination'],'unclean process shutdown')
    return dict(status='PASS',scope='CPU_FIXTURE_ONLY' if plan['fixture'] else 'PHYSICAL_LOCALHOST_OPT_C1',requests=len(rows),generation_steps=len(steps),
        strict_prefix_kv_byte_checks=strict_checks,strict_tensor_files=len(tensor_files),compact_evidence_is_full_tensor_verification=False,
        strict_hf_kv_reconstruction_checks=kv_reconstruction_checks,
        physical_gpu_verified=not plan['fixture'],independent_wire_replay=True,peak_raw_cache_bytes=peak,invalid_hit_refusals=len(negatives))


def analyze(directory, *, sealed=False, amortization_counts=(1,2,4,8)):
    directory=Path(directory)
    original=None
    if sealed:
        verify_directory(directory);original=read(directory/'r4c_results.json')
        require(original['execution_status']=='COMPLETE' and not original.get('analysis_only'), 'completed primary C1 run required')
        require(all(original.get(k)==v for k,v in flags(original['TEST_FIXTURE_ONLY']).items()), 'stored source scope/qualification flags changed')
    audit_result=audit(directory);plan=read(directory/'es/es_plan.json');fixture=plan['fixture'];manifest,prompts,tasks=load_prepared(plan['prepared'])
    rows=jsonl(directory/'es/r4c_requests.jsonl');steps=jsonl(directory/'es/r4c_tokens.jsonl');by={r['request_id']:r for r in rows};promptby={p['sample_id']:p for p in prompts}
    pairs=[];fidelity=[];latency=[];communication=[];costs=[]
    for hit in (r for r in rows if r['mode']=='PARTIAL_PREFIX_HIT'):
        nr=by[hit['pair_id']+'_N'];source=by[hit['pair_id']+'_P'];prompt=promptby[hit['sample_id']]
        require(nr['policy']==hit['policy'],'matched generation policies differ')
        gold=prompt['original']['answers'];qn=gold_score(hit['dataset'],nr['generated_text'],gold,fixture=fixture);qh=gold_score(hit['dataset'],hit['generated_text'],gold,fixture=fixture)
        pairs.append(dict(pair_id=hit['pair_id'],sample_id=hit['sample_id'],dataset=hit['dataset'],policy=hit['policy'],evidence_profile=hit['evidence_profile'],
            quality_no_reuse=qn,quality_hit=qh,quality_delta=qh['score']-qn['score'],gold_answers=gold,original_id=prompt['original']['_id'],
            no_reuse_text=nr['generated_text'],hit_text=hit['generated_text'],no_reuse_length=nr['generated_tokens'],hit_length=hit['generated_tokens'],
            no_reuse_termination=nr['termination'],hit_termination=hit['termination']))
        def vectors(r):
            v=unpack((directory/'es'/r['logits_file']).read_bytes(),'float32');w=r['logits_shape'][-1];return [v[i:i+w] for i in range(0,len(v),w)]
        strict=(directory/'es'/nr['logits_file']).read_bytes()==(directory/'es'/hit['logits_file']).read_bytes()
        f=dict(pair_id=hit['pair_id'],strict_serialized_logits_equal=strict,generated_ids_equal=nr['generated_token_ids']==hit['generated_token_ids'],
               generated_text_equal=nr['generated_text']==hit['generated_text'],full_tensor_verified=hit['evidence_profile']=='STRICT_AUDIT',
               **numerical(vectors(nr),vectors(hit),nr['generated_token_ids'],hit['generated_token_ids']))
        ns=[s for s in steps if s['request_id']==nr['request_id']];hs=[s for s in steps if s['request_id']==hit['request_id']]
        first_operation=None;all_qkv=True;intermediate=[]
        for i,(a,b) in enumerate(zip(ns,hs)):
            same_history=nr['generated_token_ids'][:i]==hit['generated_token_ids'][:i]
            for key in sorted(b['qkv'],key=lambda k:(int(k.split('.')[0]),'qkv'.index(k.split('.')[1]))):
                actual=b['qkv'][key]
                expected=a['qkv'][key]
                if i==0 and hit['evidence_profile']=='STRICT_AUDIT':
                    expected_bytes=token_slice(tensor_data(directory/'es',expected),expected,hit['reused_prefix_tokens'],len(hit['target_token_ids']))
                    actual_bytes=tensor_data(directory/'es',actual);equal=expected_bytes==actual_bytes
                    intermediate.append(dict(step=i,layer_role=key,same_history=same_history,**tensor_metrics(expected_bytes,actual_bytes,actual['dtype'])))
                elif i:
                    equal=actual['sha256']==expected['sha256']
                    if hit['evidence_profile']=='STRICT_AUDIT':intermediate.append(dict(step=i,layer_role=key,same_history=same_history,**tensor_metrics(tensor_data(directory/'es',expected),tensor_data(directory/'es',actual),actual['dtype'])))
                else:equal=None
                if equal is False:
                    all_qkv=False
                    if first_operation is None:first_operation=dict(step=i,operation='raw_projection',layer_role=key)
        f.update(first_observed_divergent_operation=first_operation,suffix_qkv_strict_equal=all_qkv if f['full_tensor_verified'] else None,
            suffix_qkv_numerical_diagnostics=intermediate,changed_generated_tokens=sum(a!=b for a,b in zip(nr['generated_token_ids'],hit['generated_token_ids']))+abs(len(nr['generated_token_ids'])-len(hit['generated_token_ids'])))
        prefix_diagnostics=[]
        if f['full_tensor_verified']:
            source_step=next(s for s in steps if s['request_id']==source['request_id'] and s['step']==0)
            for l in range(len(source_step['kv'])):
                for role in 'kv':
                    a=ns[0]['kv'][l][role];b=source_step['kv'][l][role]
                    reference=token_slice(tensor_data(directory/'es',a),a,0,hit['reused_prefix_tokens']);actual=tensor_data(directory/'es',b)
                    prefix_diagnostics.append(dict(layer=l,role=role,bitwise_equal=reference==actual,**tensor_metrics(reference,actual,b['dtype'])))
        f['cached_prefix_vs_no_reuse_kv']=prefix_diagnostics
        f['prefix_vs_no_reuse_strict_equal']=all(p['bitwise_equal'] for p in prefix_diagnostics) if prefix_diagnostics else None
        f['first_divergent_prefix_kv']=next((dict(layer=p['layer'],role=p['role']) for p in prefix_diagnostics if not p['bitwise_equal']),None)
        f['strict_fidelity_pass']=strict and f['generated_ids_equal'] and f['generated_text_equal'] and f['suffix_qkv_strict_equal'] is True and f['prefix_vs_no_reuse_strict_equal'] is True
        fidelity.append(f)
        def delta(a,b):return dict(no_reuse=a,hit=b,improvement=a-b,improvement_percent=100*(a-b)/a if a else None)
        latency.append(dict(pair_id=hit['pair_id'],dataset=hit['dataset'],policy=hit['policy']['mode'],profile=hit['evidence_profile'],
            input_tokens=len(hit['target_token_ids']),prefix_tokens=hit['reused_prefix_tokens'],no_reuse_generated=nr['generated_tokens'],hit_generated=hit['generated_tokens'],
            ttft_no_reuse_ms=nr['ttft_ms'],ttft_hit_ms=hit['ttft_ms'],ttft_improvement_percent=delta(nr['ttft_ms'],hit['ttft_ms'])['improvement_percent'],
            ttft_improvement_ms=nr['ttft_ms']-hit['ttft_ms'],
            e2e_no_reuse_ms=nr['e2e_ms'],e2e_hit_ms=hit['e2e_ms'],e2e_improvement_percent=delta(nr['e2e_ms'],hit['e2e_ms'])['improvement_percent'],
            e2e_improvement_ms=nr['e2e_ms']-hit['e2e_ms'],prefill_no_reuse_ms=nr['prefill_ms'],prefill_hit_ms=hit['prefill_ms'],
            decode_no_reuse_ms=nr['decode_ms'],decode_hit_ms=hit['decode_ms'],decode_no_reuse_tokens_per_second=nr['decode_tokens_per_second'],decode_hit_tokens_per_second=hit['decode_tokens_per_second'],
            equal_generation_counts=nr['generated_tokens']==hit['generated_tokens']))
        pn,ph=nr['prefill'],hit['prefill'];payloadn=pn['hidden_payload_bytes']+pn['delta_payload_bytes'];payloadh=ph['hidden_payload_bytes']+ph['delta_payload_bytes']
        communication.append(dict(pair_id=hit['pair_id'],policy=hit['policy']['mode'],prefill_tensor_payload=delta(payloadn,payloadh),
            prefill_application_wire=delta(pn['application_wire_bytes'],ph['application_wire_bytes']),prefill_es_to_ud_wire=delta(pn['es_to_ud_wire_bytes'],ph['es_to_ud_wire_bytes']),
            prefill_ud_to_es_wire=delta(pn['ud_to_es_wire_bytes'],ph['ud_to_es_wire_bytes']),decode_no_reuse=nr['decode'],decode_hit=hit['decode'],
            protocol_overhead_no_reuse=pn['application_wire_bytes']-payloadn,protocol_overhead_hit=ph['application_wire_bytes']-payloadh,
            source_wire_bytes=source['total_request_wire_bytes'],control_wire_hit=hit['control_wire_bytes']))
        require(all(type(k) is int and k>0 for k in amortization_counts),'invalid explicit amortization reuse counts')
        costs.append(dict(pair_id=hit['pair_id'],source_creation_e2e_ms=source['e2e_ms'],source_admission_ms=source['cache_insert_ms'],
            hit_lookup_validation_and_kv_construction_ms=hit['cache_lookup_validation_and_kv_construction_ms'],
            cache_lookup_validation_ms=hit['cache_lookup_validation_ms'],kv_construction_and_validation_ms=hit['kv_construction_and_validation_ms'],
            occupancy=hit['cache_after'],temporary_derived_kv_bytes=hit['temporary_derived_kv_bytes'],cuda_memory=hit.get('cuda_memory'),
            eviction_ms=sum(e['duration_ms'] for r in (source,nr,hit) for e in r['cache_events'] if e['kind']=='EVICT'),
            source_cost_amortization=[dict(reuse_count=k,analytical_measured_cost_model=True,no_natural_hit_rate_assumed=True,
                amortized_hit_e2e_ms=hit['e2e_ms']+source['e2e_ms']/k,improvement_percent=100*(nr['e2e_ms']-hit['e2e_ms']-source['e2e_ms']/k)/nr['e2e_ms']) for k in amortization_counts]))
    grouped={}
    for pair in pairs:
        key=pair['dataset'];items=[p for p in pairs if p['dataset']==key]
        grouped[key]=dict(examples=len(items),no_reuse_mean=sum(p['quality_no_reuse']['score'] for p in items)/len(items),hit_mean=sum(p['quality_hit']['score'] for p in items)/len(items),
            quality_delta_mean=sum(p['quality_delta'] for p in items)/len(items),metric=pair['quality_no_reuse']['metric'])
    try:
        from sacrebleu.metrics import BLEU
        import sacrebleu
        metric=BLEU();bleu=dict(score=metric.corpus_score([p['hit_text'] for p in pairs],[[p['no_reuse_text'] for p in pairs]]).score,signature=str(metric.get_signature()),package_version=sacrebleu.__version__)
    except ImportError:bleu=dict(status='MISSING_DEPENDENCY',package='sacrebleu',score=None)
    measured=not fixture;strict_items=[f for f in fidelity if f['full_tensor_verified']]
    decision=dict(**flags(fixture),R4C_DATASET_INTEGRITY_PASS=measured,R4C_GOLD_REFERENCE_VALID_PASS=measured,
        R4C_PHYSICAL_E2E_PASS=measured,R4C_EXACT_PREFIX_REUSE_PASS=measured,R4C_INDEPENDENT_AUDIT_PASS=measured,
        R4C_GOLD_METRICS_MEASURED=measured,R4C_QUALITY_PRESERVATION_MEASURED=measured,
        R4C_STRICT_BITWISE_FIDELITY_PASS=measured and len(strict_items)==len(fidelity) and all(f['strict_fidelity_pass'] for f in fidelity),
        R4C_NUMERICAL_FIDELITY_DIAGNOSTIC=dict(pairs=len(fidelity),strict_logit_equal=sum(f['strict_serialized_logits_equal'] for f in fidelity),
            strict_tensor_pairs=len(strict_items),cpu_fixture_only=fixture),R4C_OUTPUT_FIDELITY_MEASURED=measured,
        R4C_TTFT_BENEFIT_OBSERVED=measured and all(p['ttft_improvement_percent']>0 for p in latency),
        R4C_E2E_BENEFIT_OBSERVED=measured and all(p['e2e_improvement_percent']>0 for p in latency),
        R4C_COMMUNICATION_REDUCTION_MEASURED=measured,R4C_AMORTIZED_BENEFIT_MEASURED=measured,
        paired_measurements=len(pairs),eight_pair_benefit_gate_satisfied=False,confidence='SMALL_DIAGNOSTIC_PILOT; no statistical qualification')
    performance_groups=[]
    conditions=sorted({(p['dataset'],p['policy'],p['profile'],p['input_tokens'],p['prefix_tokens']) for p in latency})
    for condition in conditions:
        ps=[p for p in latency if (p['dataset'],p['policy'],p['profile'],p['input_tokens'],p['prefix_tokens'])==condition]
        performance_groups.append(dict(dataset=condition[0],policy=condition[1],profile=condition[2],input_tokens=condition[3],prefix_tokens=condition[4],
            prefix_fraction=condition[4]/condition[3],pairs=len(ps),cohort=manifest['cohort'],seed=manifest['seed'],model_tokenizer=manifest['model_tokenizer'],
            means={k:sum(p[k] for p in ps)/len(ps) for k in ('ttft_no_reuse_ms','ttft_hit_ms','ttft_improvement_ms','e2e_no_reuse_ms','e2e_hit_ms','e2e_improvement_ms')},
            confidence_interval=None,independent_repeat_qualification=False,limitation='Initial diagnostic cohort; C3 independent repeated inference and confidence analysis not run'))
    if original is not None:
        require(original['decision']==read(directory/'r4c_decision.json')==decision and original['audit']==audit_result,'stored decision/audit changed or source run promoted')
    return dict(audit=audit_result,decision=decision,quality_pairs=pairs,quality_by_dataset=grouped,performance_groups=performance_groups,fidelity=dict(pairs=fidelity,output_fidelity_sacrebleu=bleu),
                latency=latency,communication=communication,cache_costs=costs)


def export(output,analysis):
    from .r3a_artifacts import append
    output=Path(output)
    for key,name in (('audit','audit'),('decision','decision'),('quality_by_dataset','quality_by_dataset'),('fidelity','fidelity'),('communication','communication'),('cache_costs','cache_costs')):
        write(output/f'r4c_{name}.json',analysis[key])
    for row in analysis['quality_pairs']:append(output/'r4c_quality_pairs.jsonl',row)
    write_csv(output/'r4c_latency_contrasts.csv',analysis['latency'])
    write(output/'r4c_performance_groups.json',analysis['performance_groups'])
