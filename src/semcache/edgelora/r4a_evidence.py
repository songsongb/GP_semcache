"""Immutable suite, prepared identities and independent R4 generation audits."""
from pathlib import Path
import random
import time

from .communication import digest,require
from .r3a_artifacts import ROOT,read,sha,validate_prepared
from .r3a_evidence import jsonl
from .r4a_session import positions

TEXTS=(
    'The lighthouse keeper watched the sea and wrote a short note about the approaching storm.',
    'A student planted a small garden beside the library and described what happened in spring.',
    'The engineer inspected the bridge before explaining the repair plan to the nearby residents.',
    'Two musicians arrived at the station carrying instruments and discussed their evening concert.')
MODES=('NO_REUSE','COLD_MISS','WARM_HIT')
CELLS=((32,8),(32,16),(128,8),(128,16))


def flags(fixture=False):
    return dict(classification='R4A_PHYSICAL_E2E_DIAGNOSTIC',PRIMARY_PERFORMANCE_ELIGIBLE=False,
        correctness_qualified_speedup=False,actual_remote_network_measured=False,semantic_generalization_validated=False,
        natural_cache_hit_rate_validated=False,trained_adapter=False,cachegen_enabled=False,compression_enabled=False,
        concurrent_serving_validated=False,TEST_FIXTURE_ONLY=fixture)


def make_suite(tokenizer=None,*,fixture=False,scope='pilot',policy_mode='fixed'):
    require(scope in ('pilot','matrix') and policy_mode in ('fixed','natural-eos'),'declared bounded scope/policy required')
    from .r4a_generation import generation_policy
    cells=([dict(id='L32_G3_FIXTURE',length=32,new_tokens=3)] if fixture else
        [dict(id=f'L{l}_G{g}',length=l,new_tokens=g) for l,g in (CELLS if scope=='matrix' else CELLS[:1])])
    special=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0) if fixture else {
        k:getattr(tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    prompts=[]
    for length in sorted({c['length'] for c in cells}):
        for i,text in enumerate(TEXTS[:2] if fixture else TEXTS):
            tokenization_start=time.monotonic_ns()
            if fixture:ids=[1+(j+i)%5 for j in range(length)];mask=[1]*length
            else:
                tokenizer.padding_side='left'
                encoded=tokenizer(text,add_special_tokens=True,truncation=True,padding='max_length',max_length=length)
                ids=list(encoded['input_ids']);mask=list(encoded['attention_mask'])
            require(len(ids)==len(mask)==length,'tokenized prompt length changed');positions(mask)
            prompts.append(dict(sample_id=f'text_{i}_L{length}',source_text=text,token_ids=ids,attention_mask=mask,
                length=length,prompt_sha256=digest(dict(tokens=ids,mask=mask)),tokenization_outside_request_wall=True,
                tokenization_ms=None if fixture else (time.monotonic_ns()-tokenization_start)/1e6))
    require(len({p['prompt_sha256'] for p in prompts})==len(prompts),'distinct prompt suite collapsed after tokenization')
    for c in cells:c['policy']=generation_policy(special,policy_mode,c['new_tokens'])
    return dict(schema='r4a_suite_v1',scope='TEST_FIXTURE_ONLY' if fixture else scope,cells=cells,prompts=prompts,
        repeat_blocks=[101,202],special_tokens=special,padding='left',position_rule='cumsum(attention_mask)*attention_mask-1',
        gold_references='NOT_SUPPLIED',adapter_training='untrained_controlled_fixture')


def schedule(suite,cell):
    prompts=[p for p in suite['prompts'] if p['length']==cell['length']];result=[]
    for mode in MODES:result.append(dict(sample_id=prompts[0]['sample_id'],block=-1,phase='warmup',mode=mode))
    for block in suite['repeat_blocks']:
        rng=random.Random(block);order=list(prompts);rng.shuffle(order)
        for p in order:
            # Cold must precede Warm. Counterbalance the NR position; a triad is
            # validated as a group before any following triad can run.
            modes=MODES if rng.randrange(2)==0 else ('COLD_MISS','WARM_HIT','NO_REUSE')
            result.extend(dict(sample_id=p['sample_id'],block=block,phase='measured',mode=m) for m in modes)
    return result


def planned_counts(suite):
    rows=sum(len(schedule(suite,c)) for c in suite['cells'])
    steps=sum(len(schedule(suite,c))*c['new_tokens'] for c in suite['cells'])
    require(rows<=108 and steps<=1296,'bounded generation experiment budget exceeded')
    work=sum(len(schedule(suite,c))*(c['length']+c['new_tokens']-1) for c in suite['cells'])
    pilot=suite['cells'][0];pilot_work=len(schedule(suite,pilot))*(pilot['length']+pilot['new_tokens']-1)
    return dict(generation_requests=rows,max_generation_forward_passes=steps,triage_recompute_prefills=2,
        expansion_admission_scale=work/pilot_work,
        evidence_logit_bytes_upper=steps*50272*4,resource_assumption_only=True)


def verify_sources(sources):
    require(sources,'source hashes required')
    for name,h in sources.items():
        p=(ROOT/name).resolve();require(p.is_relative_to(ROOT) and sha(p)==h,'R4 source SHA mismatch: '+name)


def load_prepared(directory):
    directory=Path(directory).resolve();m=read(directory/'r4a_prepared_manifest.json')
    require(m['execution_status']=='COMPLETE' and m['native_process_exited'] is True,'completed R4 native preparation required');verify_sources(m['sources'])
    life=m['native_lifecycle'];require(life['returncode']==0 and not life['forced_termination'] and life['all_children_exited'],
        'Native model process did not exit before distributed execution')
    for name,h in m['output_hashes'].items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and sha(p)==h,'R4 prepared artifact SHA mismatch')
    pm,ref,am=validate_prepared(m['base_prepared'])
    require(sha(Path(m['base_prepared'])/'r3a_prepare_manifest.json')==m['base_manifest_sha256'],'original prepared identity changed')
    require(pm.get('TEST_FIXTURE_ONLY',False)==m['TEST_FIXTURE_ONLY'],'prepared fixture scope mismatch')
    require(m['native_result']['identity_sha256']==pm['identity_sha256'] and
        m['native_result']['adapter_matrix_sha256']==am['matrix_sha256'] and
        m['native_result']['native_peft_executed']==(not m['TEST_FIXTURE_ONLY']), 'Native reference matrix/model identity mismatch')
    suite=read(directory/'r4a_suite.json');planned_counts(suite)
    return m,suite,pm,ref,am


def audit_request(row,token_rows,identity,heads=None,device=None):
    from .r4a_generation import validate_generated
    validate_generated(row['policy'],row['generated_token_ids'])
    require(row['in_request_kv_enabled'] is True and row['generated_tokens']==len(row['generated_token_ids']) and
        row['decode_forward_count']==row['generated_tokens']-1,'unfair/missing in-request KV or decode counts')
    require(row['policy']['mode'] in ('fixed','natural-eos') and row['eos_ignored']==(row['policy']['mode']=='fixed'),'missing/changed EOS policy')
    if row['policy']['mode']=='fixed':require(row['generated_tokens']==row['policy']['max_new_tokens'],'fixed output count mismatch')
    require(len(token_rows)==row['generated_tokens'] and [s['step'] for s in token_rows]==list(range(row['generated_tokens'])),'missing/reordered token rows')
    for i,s in enumerate(token_rows):
        hit=i==0 and row['mode']=='WARM_HIT';count=0 if hit else identity['layers']
        require(s['hidden_frames']==s['delta_frames']==count and s['reused_qkv_outputs']==(96 if hit else 0),'dynamic HIT/MISS frame/projection counts wrong')
        require(s['context']['input_length']==(row['prompt_tokens'] if i==0 else 1) and s['past']['sequence_length']==row['prompt_tokens']+i,
            'prefill/decode KV length mismatch')
        if heads is not None:
            expected=[1,heads,row['prompt_tokens']+i,identity['hidden_size']//heads]
            require(len(s['past']['layers'])==32 and all(l['layer']==j and l['k_shape']==l['v_shape']==expected and
                l['dtype']==identity['hidden_dtype'] and l['device']==device for j,l in enumerate(s['past']['layers'])),
                'actual KV layer shape/dtype/device ledger changed')
        require(s['selected_token_id']==row['generated_token_ids'][i] and len(s['qkv_sha256'])==96,'token selection/QKV evidence incomplete')
        if hit:require(s['application_wire_bytes']==s['ud_cpu_lora_ms']==0,'prefill HIT performed UD/socket work')
        else:require(s['application_wire_bytes']>s['hidden_payload_bytes']+s['delta_payload_bytes']>0,'wire/payload ledger invalid')
        require(row['request_start_monotonic_ns']<=s['start_monotonic_ns']<=s['available_monotonic_ns']<=row['request_end_monotonic_ns'],'token outside wall')
        if i:require(token_rows[i-1]['available_monotonic_ns']<=s['start_monotonic_ns'],'reordered token clocks')
    require(row['prefill_reused_qkv_outputs']==(96 if row['mode']=='WARM_HIT' else 0),'wrong full-prefix reuse count')
    if row['mode']=='COLD_MISS':require(row['cache_admitted'] is True and row['cold_insert_ms']>=0,'missing real cold admission')
    return True


def audit_run(directory,plan,worker):
    from .communication import PREFIX,canonical,validate
    from .r3a_artifacts import check_parity,unpack
    from .r4a_session import context
    directory=Path(directory);es=directory/'es';m,suite,pm,ref,am=load_prepared(plan['prepared'])
    verify_sources(plan['sources']);fixture=plan['fixture'];identity=pm['identity']
    require(worker['execution_status']=='COMPLETE' and worker['pid']!=worker['ud_pid'] and
        worker['adapter_parameters_on_es']==0 and worker['user_device_cuda_used'] is False,'physical worker/isolation gate failed')
    require(worker['gpu_inference_performed']==(not fixture),'GPU/fixture execution provenance inconsistent')
    require(worker['heads']==m['native_result']['heads'] and all(worker.get(k)==v for k,v in flags(fixture).items()),'worker head/qualification identity mismatch')
    require(m['TEST_FIXTURE_ONLY']==fixture and plan['base_prepared']==m['base_prepared'],'cohort input mismatch')
    if not fixture:require(worker['gpu']['uuid']==m['native_result']['gpu']['uuid']!='UNAVAILABLE','Native/ES GPU UUID mismatch')
    rows=jsonl(es/'r4a_requests.jsonl');tokens=jsonl(es/'r4a_tokens.jsonl');outputs=jsonl(es/'r4a_generation_outputs.jsonl')
    require(rows==outputs and rows and len({r['request_id'] for r in rows})==len(rows)==worker['requests'],'missing/duplicate output rows')
    cells=[c['id'] for c in suite['cells']]
    require(worker['completed_cells']==cells[:len(worker['completed_cells'])] and worker['completed_cells'],'completed cells not a valid prefix')
    if len(worker['completed_cells'])<len(cells):
        gate=worker['cell_gates']['expansion'];mem=gate['observed_memory']
        require(gate['status']=='SKIPPED_RESOURCE_CAP' and (mem['peak_allocated_bytes']>20*2**30 or mem['free_bytes']<2*2**30 or
            gate['elapsed_seconds']*planned_counts(suite)['expansion_admission_scale']+900>7200),'unjustified omitted matrix cells')
    require(read(es/'cell_gates.json')==worker['cell_gates'],'cell gate copies differ')
    expected=[dict(s,cell_id=c['id']) for c in suite['cells'] if c['id'] in worker['completed_cells'] for s in schedule(suite,c)]
    require(len(expected)==len(rows) and all(all(r[k]==e[k] for k in ('cell_id','sample_id','phase','block','mode')) for r,e in zip(rows,expected)),
        'generation schedule changed/incomplete')
    for i in range(0,len(rows),3):
        triad=rows[i:i+3];cold=next(r for r in triad if r['mode']=='COLD_MISS');warm=next(r for r in triad if r['mode']=='WARM_HIT')
        require(warm['cache_source_request_id']==cold['request_id'] and warm['cache_context_sha256']==cold['cache_context_sha256'],
            'cross-request cache fill/source provenance mismatch')
    by_request={r['request_id']:[s for s in tokens if s['request_id']==r['request_id']] for r in rows}
    require(sum(len(v) for v in by_request.values())==len(tokens),'orphan token records')
    source=jsonl(es/'es_wire.jsonl');sink=jsonl(es/'ud_wire.jsonl');ud=read(es/'ud_result.json');layers=jsonl(es/'ud_layers.jsonl')
    require(ud['execution_status']=='COMPLETE' and ud['pid']==worker['ud_pid'] and ud['user_device_cuda_used'] is False and
        ud['full_model_loaded'] is False and ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='' and
        ud['adapter_matrix_sha256']==am['matrix_sha256'],'UD identity/isolation mismatch')
    require(len(source)==len(sink),'endpoint frame counts differ')
    by_forward={};ud_forward={}
    for a,b in zip(source,sink):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and
            all(a[k]==b[k] for k in ('header','wire_bytes','frame_sha256')),'ES/UD independent wire SHA mismatch')
        h=a['header'];validate(h,h['payload_bytes']);require(h['session_id']==worker['session_id'] and h['adapter_sha256']==am['matrix_sha256'] and
            a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],'framing/identity byte ledger incorrect')
        if h['kind'] in ('HIDDEN','DELTAS'):by_forward.setdefault(h['request_id'],[]).append(a)
    for l in layers:ud_forward.setdefault(l['forward_id'],[]).append(l)
    for op in ('send','recv'):
        require([r['header']['sequence'] for r in source if r['operation']==op]==list(range(sum(r['operation']==op for r in source))),'wire sequence discontinuity')
    audited_forward_ids=[]
    for row in rows:
        steps=by_request[row['request_id']];audit_request(row,steps,identity,worker['heads'],'cpu' if fixture else plan['device'])
        require(all(row.get(k)==v for k,v in flags(fixture).items()) and row['es_pid']==worker['pid'] and
            row['ud_pid']==worker['ud_pid'] and row['adapter_matrix_sha256']==am['matrix_sha256'],'request qualifications/adapter identity changed')
        prompt=next(p for p in suite['prompts'] if p['sample_id']==row['sample_id'])
        cell=next(c for c in suite['cells'] if c['id']==row['cell_id'])
        require(row['policy']==cell['policy'] and row['prompt_sha256']==prompt['prompt_sha256'],'prompt/policy identity mismatch')
        require(row['result_sha256']==digest(dict(tokens=row['generated_token_ids'],text=row['generated_text'],prompt=row['prompt_sha256'],policy=row['policy'])),
            'generated token/text result identity corrupted')
        data=(es/row['logits_file']).read_bytes();require(sha(es/row['logits_file'])==row['logits_sha256'] and len(data)==4*__import__('math').prod(row['logits_shape']),
            'generation logit bytes/hash corrupted')
        mask=list(prompt['attention_mask'])
        for i,s in enumerate(steps):
            ids=prompt['token_ids'] if i==0 else [row['generated_token_ids'][i-1]]
            expected_context=context(row['request_id'],i,ids,mask,0 if i==0 else row['prompt_tokens']+i-1,row['policy']['max_new_tokens'],
                hit=row['mode']=='WARM_HIT',natural=row['policy']['mode']=='natural-eos')
            require(s['context']==expected_context and s['input_token_ids']==ids and s['positions']==positions(mask)[len(mask)-len(ids):],
                'decode token/position/mask identity mismatch')
            width=row['logits_shape'][-1];blob=data[i*width*4:(i+1)*width*4];vals=unpack(blob,'float32')
            require(s['logits_sha256']==__import__('hashlib').sha256(blob).hexdigest() and
                s['selected_token_id']==max(range(width),key=vals.__getitem__),'next-token argmax/logit evidence mismatch')
            frames=by_forward.get(s['forward_id'],[]);work=ud_forward.get(s['forward_id'],[]);hit=row['mode']=='WARM_HIT' and i==0
            require(s['forward_id']==f'{row["request_id"]}.s{i:02d}','forward ID does not belong to generation step')
            require(len(frames)==(0 if hit else 64) and len(work)==(0 if hit else 32),'missing/extra per-pass physical frames/UD work')
            require(sum(f['wire_bytes'] for f in frames)==s['application_wire_bytes'] and
                sum(l['metrics']['cpu_lora_ms'] for l in work)==s['ud_cpu_lora_ms'],'per-pass byte/UD timing mismatch')
            require(s['es_to_ud_wire_bytes']==sum(f['wire_bytes'] for f in frames if f['operation']=='send') and
                s['ud_to_es_wire_bytes']==sum(f['wire_bytes'] for f in frames if f['operation']=='recv'),'directional application wire bytes differ')
            for kind,field in (('HIDDEN','hidden_payload_bytes'),('DELTAS','delta_payload_bytes')):
                selected=[f for f in frames if f['header']['kind']==kind]
                require([f['header']['layer_id'] for f in selected]==([] if hit else list(range(32))) and
                    sum(f['header']['payload_bytes'] for f in selected)==s[field],'layer/directional payload schedule differs')
            for f in frames:
                require(s['start_monotonic_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=s['available_monotonic_ns'],'socket IO outside step timing')
                shape=[1,len(ids),identity['hidden_size']]
                if f['header']['kind']=='HIDDEN':
                    require(f['header']['metadata']==dict(r4a_forward=expected_context) and f['header']['shape']==shape and
                        f['header']['dtype']==identity['hidden_dtype'],'wire hidden shape/context changed')
                else:require(f['header']['metadata']['r4a_forward_sha256']==digest(expected_context) and
                    f['header']['shape']==[3]+shape and f['header']['dtype']==identity['delta_dtype'],'returned delta identity/shape wrong')
            require(all(l['context']==expected_context and l['layer']==j for j,l in enumerate(work)),'UD context/layer ledger differs')
            if frames:audited_forward_ids.append(s['forward_id'])
            mask.append(1)
        for field in row['total_accounting']:
            # Match the declared aggregation order exactly. Float component
            # timers can differ at the last bit if regrouped; bytes stay exact.
            prefix=steps[0][field];decode=sum(s[field] for s in steps[1:])
            require(row['prefill_accounting'][field]==prefix and row['decode_accounting'][field]==decode and
                row['total_accounting'][field]==prefix+decode,'request prefill/decode accounting mismatch')
        require(row['ttft_ms']==(row['first_token_available_monotonic_ns']-row['request_start_monotonic_ns'])/1e6 and
            row['e2e_wall_ms']==(row['request_end_monotonic_ns']-row['request_start_monotonic_ns'])/1e6,'wall/TTFT not derived from actual clocks')
    diagnostics=read(es/'r4a_parity_diagnostics.json');triage=jsonl(es/'triage_passes.jsonl')
    require(len(triage)==2,'both use_cache endpoint triages required')
    for use_cache in (False,True):
        directory_native=Path(plan['prepared'])/f'triage_native_{int(use_cache)}';directory_es=es/f'triage_es_{int(use_cache)}'
        nr=read(directory_native/'forward.json');er=read(directory_es/'forward.json')
        actual=check_parity(unpack((directory_es/er['logits_file']).read_bytes(),'float32'),
            unpack((directory_native/nr['logits_file']).read_bytes(),'float32'),nr['shape'])
        require(diagnostics['use_cache_true' if use_cache else 'use_cache_false']['strict_metrics']==actual,'Native strict triage metrics changed')
        step=triage[int(use_cache)];frames=by_forward.get(step['forward_id'],[])
        require(len(frames)==64 and len(ud_forward.get(step['forward_id'],[]))==32,'triage missing actual distributed recompute')
        audited_forward_ids.append(step['forward_id'])
    expected_native=(not fixture and m['native_result']['native_peft_executed'] is True and
        all(diagnostics[k]['strict_metrics']['passed'] and diagnostics[k]['matched_model_logits_dtype'] for k in ('use_cache_false','use_cache_true')) and
        read(Path(plan['prepared'])/'triage_native_0/forward.json').get('original_bitwise_equal') is True and
        read(Path(plan['prepared'])/'triage_native_0/forward.json').get('dtype_matches_original') is True)
    require(worker['native_equivalent_prefill']==expected_native,'Native parity promoted without independent strict evidence')
    require(set(audited_forward_ids)==set(by_forward)==set(ud_forward) and len(audited_forward_ids)==len(set(audited_forward_ids))==ud['completed_forwards'],
        'unaccounted physical forward or duplicated generation work')
    controls=[r for r in source if r['header']['kind'] not in ('HIDDEN','DELTAS')]
    require([r['header']['kind'] for r in controls]==['HELLO','READY','STOP','BYE'],'control framing mismatch')
    for path in (es/'ud_lifecycle.json',es/'lifecycle.json'):
        life=read(path);require(life['returncode']==0 and not life['forced_termination'] and life['all_children_exited'],'unclean child lifecycle')
    require(read(es/'lifecycle.json')['tracked_descendants']['all_stopped'],'orphan UD descendants')
    return dict(status='PASS',requests=len(rows),generation_steps=len(tokens),real_socket_forward_passes=len(audited_forward_ids),
        independent_es_ud_wire_sha_agree=True,completed_cells=worker['completed_cells'],native_prefill_equivalent=worker['native_equivalent_prefill'])


def load_run(directory):
    directory=Path(directory).resolve();report=read(directory/'r4a_results.json')
    require(report['execution_status']=='COMPLETE' and report.get('audit',{}).get('status')=='PASS' and not report.get('analysis_only'),'completed audited R4 run required')
    required={'r4a_requests.jsonl','r4a_tokens.jsonl','r4a_generation_outputs.jsonl','r4a_quality.json','r4a_decision.json',
        'es/es_result.json','es/es_plan.json','es/es_wire.jsonl','es/ud_wire.jsonl','es/ud_layers.jsonl'}
    require(required.issubset(report['output_hashes']),'missing required R4 output hashes')
    for name,h in report['output_hashes'].items():
        path=(directory/name).resolve();require(path.is_relative_to(directory) and sha(path)==h,'R4 artifact SHA mismatch: '+name)
    verify_sources(report['sources']);plan=read(directory/'es/es_plan.json');worker=read(directory/'es/es_result.json')
    require(report['sources']==plan['sources'],'run/worker source manifests differ')
    for name in ('r4a_requests.jsonl','r4a_tokens.jsonl','r4a_generation_outputs.jsonl','r4a_parity_diagnostics.json'):
        require((directory/name).read_bytes()==(directory/'es'/name).read_bytes(),'root/worker export copies differ')
    require(audit_run(directory,plan,worker)==report['audit'],'independent R4 audit differs')
    return report,plan,worker
