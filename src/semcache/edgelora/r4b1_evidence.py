"""Bounded B1 schedule and independent capacity, provenance and wire audits."""
from collections import OrderedDict
from pathlib import Path

from .communication import PREFIX,canonical,digest,digest_bytes,require,validate
from .r3a_artifacts import ROOT,read,sha,unpack,write
from .r3a_evidence import jsonl
from .r4a_evidence import audit_request,load_prepared,verify_sources
from .r4a_session import context,positions
from .r4b1_cache import MultiPromptCache


def flags(fixture=False):
    return dict(classification='R4B1_MULTI_ENTRY_EXACT_FULL_PROMPT_DIAGNOSTIC',TEST_FIXTURE_ONLY=fixture,
        PRIMARY_PERFORMANCE_ELIGIBLE=False,correctness_qualified_speedup=False,semantic_generalization_validated=False,
        partial_prefix_reuse=False,actual_remote_network_measured=False,trained_adapter=False,native_parity_deferred=True)


def schedule(suite):
    cell=suite['cells'][0];prompts=[p for p in suite['prompts'] if p['length']==32][:2]
    require(len(prompts)==2 and len({p['prompt_sha256'] for p in prompts})==2,'two distinct exact L32 prompts required')
    a,b=[p['sample_id'] for p in prompts];rows=[]
    for block in (-1,101,202):
        for sample,mode in ((a,'NO_REUSE'),(b,'NO_REUSE'),(a,'COLD_MISS'),(b,'COLD_MISS'),(a,'WARM_HIT'),(b,'WARM_HIT')):
            rows.append(dict(sample_id=sample,mode=mode,phase='warmup' if block==-1 else 'measured',block=block,pool_id='main'))
    for sample,mode in ((a,'COLD_MISS'),(b,'COLD_MISS'),(a,'COLD_MISS'),(a,'WARM_HIT'),(b,'COLD_MISS'),(b,'WARM_HIT')):
        rows.append(dict(sample_id=sample,mode=mode,phase='capacity_pressure',block=303,pool_id='pressure'))
    for i,row in enumerate(rows):
        row.update(request_id=f'B{i:04d}',cell_id=cell['id'],comparison_id=f'{cell["id"]}:{row["block"]}:{row["sample_id"]}')
    require(len(rows)==24 and len(rows)*cell['new_tokens']<=192,'bounded B1 pilot only')
    return rows


def entry_bytes(identity,length=32):
    return 3*identity['layers']*length*identity['hidden_size']*({'float16':2,'float32':4}[identity['hidden_dtype']])


def replay_events(events,capacities,contexts,rows):
    """Reconstruct LRU/residency rather than trusting worker occupancy claims."""
    states={p:OrderedDict() for p in capacities};seq={p:0 for p in capacities};peaks={p:0 for p in capacities};hits=evictions=0
    row_by={r['request_id']:r for r in rows}
    for e in events:
        pool=e['pool_id'];require(pool in states,'unknown cache pool');state=states[pool]
        require(e['sequence']==seq[pool] and e['duration_ms']>=0,'cache event sequence/timing invalid');seq[pool]+=1
        key=e['context_sha256'];require(key in contexts,'unknown full-prompt identity in cache ledger')
        kind=e['kind']
        if kind in ('EVICT','INVALIDATE','REJECT'):
            require(key in state and e['released_bytes']==state[key]['bytes'] and e['source_request_id']==state[key]['source_request_id'],
                'eviction provenance/bytes missing')
            if kind=='EVICT':require(key==next(iter(state)) and e['reason']=='capacity_lru','victim is not actual LRU');evictions+=1
            del state[key]
        elif kind=='ADMISSION' and e['allowed']:
            require(key not in state and e['request_id'] in row_by,'duplicate or uncompleted admission')
            row=row_by[e['request_id']]
            require(row['mode']=='COLD_MISS' and row['cache_context_sha256']==key and
                row['last_decode_available_ns']<=e['monotonic_ns']<=row['request_end_monotonic_ns'],'admission precedes completed generation')
            size=entry_bytes(contexts[key]['identity'],len(contexts[key]['token_ids']))
            require(e['candidate_bytes']==size,'candidate byte estimate differs from actual shape/dtype')
            primitive=e['primitive_events'];require(any(p['kind']=='ADMISSION' and p['allowed'] for p in primitive) and
                any(p['kind']=='INSERT' for p in primitive),'missing actual R3-B admission/materialization')
            state[key]=dict(context_sha256=key,bytes=size,source_request_id=e['request_id'])
        elif kind in ('HIT','PHYSICAL_REUSE','INSERT'):
            require(key in state and e['source_request_id']==state[key]['source_request_id'],'HIT/reuse/insert after eviction or wrong source')
            if kind=='HIT':
                require(e['validation_read_bytes']==state[key]['bytes'],'missing full content validation');state.move_to_end(key);hits+=1
        elif kind=='MISS':require(key not in state,'MISS while a valid entry is resident')
        elif kind=='ADMISSION':require(not e['allowed'],'unknown admission outcome')
        else:require(False,'unsupported cache event')
        total=sum(v['bytes'] for v in state.values());peaks[pool]=max(peaks[pool],total)
        require(total<=capacities[pool] and e['capacity_bytes']==capacities[pool] and e['occupancy_bytes']==total and
            e['entry_count']==len(state) and e['lru_to_mru']==list(state) and e['resident_entries']==list(state.values()) and
            e['peak_occupancy_bytes']==peaks[pool],'global cache occupancy/LRU ledger inconsistent')
        require(sum(v['bytes'] for s in states.values() for v in s.values())<=capacities['main'],
            'pressure experiment retained main cache storage beyond global pilot budget')
    require(evictions>=3 and peaks['main']>=2*entry_bytes(next(iter(contexts.values()))['identity']),
        'multi-entry residency or capacity-pressure eviction not demonstrated')
    return dict(hits=hits,evictions=evictions,peak_bytes=peaks,final_occupancy={p:sum(v['bytes'] for v in s.values()) for p,s in states.items()})


def fidelity(rows,tokens,directory):
    directory=Path(directory);steps={r['request_id']:[t for t in tokens if t['request_id']==r['request_id']] for r in rows}
    refs={(r['block'],r['sample_id']):r for r in rows if r['mode']=='NO_REUSE'};checks=[]
    for r in rows:
        data=(directory/r['logits_file']).read_bytes();require(digest_bytes(data)==r['logits_sha256'],'serialized generation logits corrupted')
        require(r['logits_shape']==[1,r['generated_tokens'],r['logits_shape'][-1]],'generation logit batch/sequence shape differs')
        values=unpack(data,'float32');require(len(values)==r['generated_tokens']*r['logits_shape'][-1],'logit dimensions differ')
        if r['mode']=='NO_REUSE':continue
        ref=refs[(101 if r['phase']=='capacity_pressure' else r['block'],r['sample_id'])]
        equal=(r['policy']==ref['policy'] and r['prompt_sha256']==ref['prompt_sha256'] and r['generated_token_ids']==ref['generated_token_ids'] and
            r['generated_text']==ref['generated_text'] and data==(directory/ref['logits_file']).read_bytes() and
            [s['qkv_sha256'] for s in steps[r['request_id']]]==[s['qkv_sha256'] for s in steps[ref['request_id']]])
        checks.append(dict(request_id=r['request_id'],reference_request_id=ref['request_id'],bitwise_equal=equal))
        require(equal,'R4B1 distributed NO_REUSE bitwise fidelity failed')
    return checks


def audit_run(directory,plan,worker):
    directory=Path(directory);es=directory/'es';prepared,suite,pm,ref,am=load_prepared(plan['prepared']);verify_sources(plan['sources'])
    require(prepared['sources']==plan['sources'],'prepared/run source identity differs')
    require(worker['execution_status']=='COMPLETE' and worker['pid']!=worker['ud_pid'] and worker['adapter_parameters_on_es']==0 and
        worker['gpu_inference_performed']==(not plan['fixture']) and worker['user_device_cuda_used'] is False,'physical process/GPU/isolation gate failed')
    require(all(worker.get(k)==v for k,v in flags(plan['fixture']).items()),'B1 qualification gate changed')
    if not plan['fixture']:require(worker['gpu']['uuid']==prepared['native_result']['gpu']['uuid']!='UNAVAILABLE','GPU identity changed')
    rows=jsonl(es/'r4b1_requests.jsonl');tokens=jsonl(es/'r4b1_tokens.jsonl');expected=schedule(suite)
    require(len(rows)==len(expected) and all(all(r[k]==v for k,v in e.items()) for r,e in zip(rows,expected)),
        'B1 schedule incomplete/reordered')
    require(rows==jsonl(es/'r4b1_generation_outputs.jsonl') and len({r['request_id'] for r in rows})==len(rows),'output ledger differs/duplicate request')
    require(jsonl(es/'r4b1_logit_diagnostics.jsonl')==[t['logit_diagnostic'] for t in tokens],'missing independent logit diagnostics')
    require(worker['tokenizer']==dict(vocab_sha256=prepared['native_result']['tokenizer_vocab_sha256'],special_tokens=suite['special_tokens'],
        source=pm['identity'].get('tokenizer_source'),revision=pm['identity'].get('tokenizer_revision')),'actual tokenizer differs from Native preparation')
    contexts={};builder=MultiPromptCache(pm,prepared['base_manifest_sha256'],worker['gpu_uuid'],plan['device'],plan['capacity_bytes'],
        tokenizer=worker['tokenizer'],sources=plan['sources'],fixture=plan['fixture'])
    for p in suite['prompts'][:2]:
        c=builder.context_for(p);contexts[digest(c)]=c
    require(read(es/'r4b1_contexts.json')==contexts,'full token/model/adapter/source identities changed')
    require(plan['capacity_bytes']>=2*entry_bytes(pm['identity']) and worker['capacities']==dict(main=plan['capacity_bytes'],pressure=entry_bytes(pm['identity'])),
        'pilot capacity contract differs')
    by_id={r['request_id']:r for r in rows};by_step={r['request_id']:[t for t in tokens if t['request_id']==r['request_id']] for r in rows}
    require(sum(map(len,by_step.values()))==len(tokens),'orphan token records')
    frames=jsonl(es/'es_wire.jsonl');sink=jsonl(es/'ud_wire.jsonl');ud=read(es/'ud_result.json');layers=jsonl(es/'ud_layers.jsonl')
    require(len(frames)==len(sink) and ud['execution_status']=='COMPLETE' and ud['pid']==worker['ud_pid'] and
        ud['user_device_cuda_used'] is False and ud['full_model_loaded'] is False and
        ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='' and ud['adapter_matrix_sha256']==am['matrix_sha256'],'UD/frame isolation failed')
    for a,b in zip(frames,sink):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and
            all(a[k]==b[k] for k in ('header','wire_bytes','frame_sha256')),'independent ES/UD wire mismatch')
        h=a['header'];validate(h,h['payload_bytes']);require(h['session_id']==worker['session_id'] and h['adapter_sha256']==am['matrix_sha256'] and
            a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],'wire identity/byte ledger invalid')
    for op in ('send','recv'):
        require([r['header']['sequence'] for r in frames if r['operation']==op]==list(range(sum(r['operation']==op for r in frames))),
            'wire sequence discontinuity')
    by_forward={};ud_forward={}
    for f in frames:
        if f['header']['kind'] in ('HIDDEN','DELTAS'):by_forward.setdefault(f['header']['request_id'],[]).append(f)
    for l in layers:ud_forward.setdefault(l['forward_id'],[]).append(l)
    expected_forwards=[]
    for r in rows:
        steps=by_step[r['request_id']];audit_request(r,steps,pm['identity'],worker['heads'],plan['device'])
        require(all(r.get(k)==v for k,v in flags(plan['fixture']).items()),'row qualifications changed')
        p=next(p for p in suite['prompts'] if p['sample_id']==r['sample_id']);mask=list(p['attention_mask'])
        require(r['policy']==suite['cells'][0]['policy'] and r['prompt_sha256']==p['prompt_sha256'],'prompt/policy differs from preparation')
        require(r['last_decode_available_ns']==steps[-1]['available_monotonic_ns'],'completed generation timestamp differs')
        if r['mode']!='NO_REUSE':require(r['cache_context_sha256']==digest(builder.context_for(p)),'cache prompt context differs')
        blob=(es/r['logits_file']).read_bytes();width=r['logits_shape'][-1]
        for i,s in enumerate(steps):
            ids=p['token_ids'] if i==0 else [r['generated_token_ids'][i-1]]
            c=context(r['request_id'],i,ids,mask,0 if i==0 else len(mask)-1,r['policy']['max_new_tokens'],hit=r['mode']=='WARM_HIT')
            require(s['context']==c and s['positions']==positions(mask)[len(mask)-len(ids):] and s['input_token_ids']==ids,'actual decode/mask/position differs')
            piece=blob[i*width*4:(i+1)*width*4];vals=unpack(piece,'float32')
            require(digest_bytes(piece)==s['logits_sha256'] and s['selected_token_id']==max(range(width),key=vals.__getitem__),'last-token evidence differs')
            f=by_forward.get(s['forward_id'],[]);u=ud_forward.get(s['forward_id'],[]);hit=i==0 and r['mode']=='WARM_HIT'
            require(len(f)==(0 if hit else 64) and len(u)==(0 if hit else 32),'prefill/decode actual frame counts differ')
            require(sum(x['wire_bytes'] for x in f)==s['application_wire_bytes'],'step network bytes differ')
            for op,field in (('send','es_to_ud_wire_bytes'),('recv','ud_to_es_wire_bytes')):
                require(s[field]==sum(x['wire_bytes'] for x in f if x['operation']==op),'directional step network bytes differ')
            for kind,field in (('HIDDEN','hidden_payload_bytes'),('DELTAS','delta_payload_bytes')):
                selected=[x for x in f if x['header']['kind']==kind]
                require([x['header']['layer_id'] for x in selected]==([] if hit else list(range(32))) and
                    s[field]==sum(x['header']['payload_bytes'] for x in selected),'payload bytes/layer sequence differs')
            for x in f:
                h=x['header'];require(s['start_monotonic_ns']<=x['monotonic_start_ns']<=x['monotonic_end_ns']<=s['available_monotonic_ns'],'wire outside step clock')
                require((h['metadata']==dict(r4a_forward=c) and h['shape']==[1,len(ids),pm['identity']['hidden_size']] and h['dtype']==pm['identity']['hidden_dtype'])
                    if h['kind']=='HIDDEN' else (h['metadata']['r4a_forward_sha256']==digest(c) and
                    h['shape']==[3,1,len(ids),pm['identity']['hidden_size']] and h['dtype']==pm['identity']['delta_dtype']), 'wire context/shape/dtype differs')
            for j,l in enumerate(u):require(l['layer']==j and l['context']==c,'UD layer context differs')
            if not hit:expected_forwards.append(s['forward_id'])
            mask.append(1)
        for field in r['total_accounting']:
            prefill=steps[0][field];decode=sum(s[field] for s in steps[1:])
            require(r['prefill_accounting'][field]==prefill and r['decode_accounting'][field]==decode and
                r['total_accounting'][field]==prefill+decode,'Prefill/Decode ledger differs')
        require(r['ttft_ms']==(r['first_token_available_monotonic_ns']-r['request_start_monotonic_ns'])/1e6 and
            r['e2e_wall_ms']==(r['request_end_monotonic_ns']-r['request_start_monotonic_ns'])/1e6,'request wall clocks differ')
        require(r['result_sha256']==digest(dict(tokens=r['generated_token_ids'],text=r['generated_text'],prompt=r['prompt_sha256'],policy=r['policy'])),
            'generated output identity changed')
        if r['mode']=='WARM_HIT':
            source=by_id[r['cache_source_request_id']]
            require(source['mode']=='COLD_MISS' and source['prompt_sha256']==r['prompt_sha256'] and
                source['request_end_monotonic_ns']<=r['request_start_monotonic_ns'],'HIT source is not completed matching generation')
    for use in (False,True):expected_forwards.append(f'ES_TRIAGE_{int(use)}.s00')
    require(set(by_forward)==set(expected_forwards) and set(ud_forward)==set(expected_forwards),'orphan/missing physical forwards')
    checks=fidelity(rows,tokens,es);require(checks==jsonl(es/'fidelity_checks.jsonl'),'worker fidelity evidence differs')
    events=jsonl(es/'r4b1_cache_events.jsonl');cache_audit=replay_events(events,worker['capacities'],contexts,rows)
    for pool in worker['capacities']:
        last=next(e for e in reversed(events) if e['pool_id']==pool)
        require(worker['cache_final'][pool]=={k:last[k] for k in ('occupancy_bytes','entry_count','capacity_bytes','peak_occupancy_bytes','lru_to_mru','resident_entries')},
            'worker final residency differs from independent event replay')
    for r in rows:
        pool_events=[e for e in events if e['pool_id']==r['pool_id']]
        for clock,snapshot_name,bytes_name in (('request_start_monotonic_ns','cache_before','cache_bytes_before'),
                ('request_end_monotonic_ns','cache_after','cache_bytes_after')):
            preceding=[e for e in pool_events if e['monotonic_ns']<=r[clock]]
            expected_snapshot=({k:preceding[-1][k] for k in ('occupancy_bytes','entry_count','capacity_bytes','peak_occupancy_bytes','lru_to_mru','resident_entries')}
                if preceding else dict(occupancy_bytes=0,entry_count=0,capacity_bytes=worker['capacities'][r['pool_id']],
                    peak_occupancy_bytes=0,lru_to_mru=[],resident_entries=[]))
            require(r[snapshot_name]==expected_snapshot and r[bytes_name]==expected_snapshot['occupancy_bytes'],
                'request before/after occupancy differs from independent cache event replay')
        selected=[e for e in events if e['request_id']==r['request_id'] and e['pool_id']==r['pool_id']]
        require(r['cache_events']==[{k:v for k,v in e.items() if k!='pool_id'} for e in selected],'request cache events differ')
        if r['mode']=='NO_REUSE':require(not selected and r['cache_before']==r['cache_after'],'NO_REUSE changed resident cache')
        else:
            require(selected[-1]['occupancy_bytes']==r['cache_bytes_after']==r['cache_after']['occupancy_bytes'],'request occupancy differs')
            require(r['cache_eviction_ms']==sum(e['duration_ms'] for e in selected if e['kind'] in ('EVICT','INVALIDATE','REJECT')) and
                r['cache_insert_ms']==sum(e['duration_ms'] for e in selected if e['kind']=='ADMISSION') and
                r['cache_lookup_validation_ms']==sum(e['duration_ms'] for e in selected if e['kind'] in ('HIT','MISS')),'cache operation time differs')
    refused=read(es/'r4b1_hit_refusal.json')
    require(refused['status']=='EXPECTED_HIT_REJECTED' and refused['new_wire_frames']==0 and refused['recomputed'] is False and
        sum(e['kind']=='MISS' and e['request_id']=='REFUSED_A' for e in events)==1,'requested HIT silently fell back')
    from .r4a_parity import compare
    diagnostics=read(es/'r4b1_parity_diagnostics.json');triage=jsonl(es/'triage_passes.jsonl');native_gates=[]
    require(len(triage)==2,'both original-input use_cache triages required')
    for use in (False,True):
        native_dir=Path(plan['prepared'])/f'triage_native_{int(use)}';es_dir=es/f'triage_es_{int(use)}'
        er=read(es_dir/'forward.json');blob=(es_dir/er['logits_file']).read_bytes()
        require(sha(es_dir/er['logits_file'])==er['logits_sha256'],'ES triage logits corrupted')
        actual=compare(native_dir,es_dir,unpack(blob,'float32'),er['shape'],fixture=plan['fixture'])
        require(diagnostics['use_cache_true' if use else 'use_cache_false']==actual,'Native strict triage metrics changed')
        native_gates.append(actual['strict_metrics']['passed'] and actual.get('matched_model_logits_dtype',False))
        fid=f'ES_TRIAGE_{int(use)}.s00';require(len(by_forward[fid])==64 and len(ud_forward[fid])==32,'missing actual ES triage work')
    original=read(Path(plan['prepared'])/'triage_native_0/forward.json')
    native_ok=not plan['fixture'] and all(native_gates) and original.get('original_bitwise_equal') is True and original.get('dtype_matches_original') is True
    require(worker['native_equivalent_prefill']==diagnostics['native_equivalent_prefill']==native_ok and
        diagnostics['frozen_thresholds_unchanged'] is True,'Native parity gate promoted or altered')
    for name in ('lifecycle.json','ud_lifecycle.json'):
        life=read(es/name);require(life['returncode']==0 and life['all_children_exited'] and not life['forced_termination'],'unclean child lifecycle')
    require(read(es/'lifecycle.json')['tracked_descendants']['all_stopped'],'orphan descendants')
    return dict(status='PASS',requests=len(rows),generation_steps=len(tokens),fidelity_comparisons=len(checks),
        cache=cache_audit,independent_wire_agreement=True,requested_hit_after_eviction_rejected=True)


def load_run(directory):
    directory=Path(directory).resolve();report=read(directory/'r4b1_results.json')
    require(report['execution_status']=='COMPLETE' and report['audit']['status']=='PASS' and not report.get('analysis_only'),'completed audited B1 run required')
    required={'es/es_result.json','es/es_plan.json','es/es_wire.jsonl','es/ud_wire.jsonl','es/ud_layers.jsonl','es/r4b1_requests.jsonl',
        'es/r4b1_tokens.jsonl','es/r4b1_cache_events.jsonl','es/r4b1_contexts.json','es/r4b1_logit_diagnostics.jsonl',
        'r4b1_quality.json','r4b1_decision.json'}
    require(required.issubset(report['output_hashes']),'missing required B1 artifacts')
    for name,h in report['output_hashes'].items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and sha(p)==h,'B1 artifact SHA mismatch: '+name)
    verify_sources(report['sources']);plan=read(directory/'es/es_plan.json');worker=read(directory/'es/es_result.json')
    for name in ('r4b1_requests.jsonl','r4b1_tokens.jsonl','r4b1_generation_outputs.jsonl','r4b1_cache_events.jsonl','r4b1_parity_diagnostics.json'):
        require(name in report['output_hashes'] and (directory/name).read_bytes()==(directory/'es'/name).read_bytes(),
            'root and worker evidence copies differ')
    require(report['sources']==plan['sources'] and audit_run(directory,plan,worker)==report['audit'],'independent B1 audit differs')
    return report,plan,worker
