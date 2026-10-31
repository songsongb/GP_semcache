"""Independent, deterministic, read-only replay of R5-I3F tensor evidence."""
from pathlib import Path
import math
import struct

from .communication import digest, digest_bytes, require
from .r3a_artifacts import check_parity, read, unpack, write
from .r3a_evidence import jsonl
from .r4b2_analysis import raw_head_bytes
from .r5_analysis import audit as audit_r5
from .r5_evidence import verify
from .r5_fidelity import (TARGETS, SELECTED, ORDER, align, compare, compare_bytes, first_divergence,
                          gates, historical_id, operation_order, tensor_blob)
from .r5_perf_analysis import accounting_audit


def indexed(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    require(len(result) == len(rows), 'duplicate forensic evidence index')
    return result


def validate_observations(step, layers, fixture):
    ts=step['tensors'];c=step['context'];n=c['input_length']
    for i in range(layers):
        required=['layer_input','layer_output','raw_q','raw_k','raw_v','attention_kv_k','attention_kv_v']
        if not fixture:
            required+=['norm1','projection_input','base_q','base_k','base_v','delta_q','delta_k','delta_v',
                       'attention_context','attention_output','attention_residual','norm2','fc1','activation','ffn','ffn_residual',
                       'used_attention_kv_k','used_attention_kv_v']
        require(all(f'L{i}.{op}' in ts for op in required),'missing layer/operation capture')
        for op in required:
            item=ts[f'L{i}.{op}'];axis=item['token_axis']
            expected=c['past_length']+n if 'attention_kv' in op else n
            require(item['shape'][axis]==expected,'layer tensor positions/shape differ')
    indices={int(k.split('.')[0][1:]) for k in ts if k.startswith('L')}
    require(indices==set(range(layers)),'incorrect layer indexing')
    if not fixture:
        require(set(step['actual_layer_context'])=={str(i) for i in range(layers)},'missing actual attention contexts')
    # Check the actual additive masks, not just requested masks. No execution
    # relies on these diagnostic checks; they run after inference has exited.
    for layer,context in step['actual_layer_context'].items():
        require(int(layer) in range(layers),'foreign attention layer')
        mask=context.get('attention_mask')
        if mask is None:
            require(c['input_length']==1, 'actual full/suffix causal mask unavailable')
            continue
        q,k=c['input_length'],len(c['attention_mask'])
        require(mask['shape']==[1,1,q,k] and len(mask['values'])==q*k,'actual attention-mask shape differs')
        for qi in range(q):
            for ki in range(k):
                value=mask['values'][qi*k+ki];visible=ki<=c['positions'][qi]
                require(value==0 if visible else value=='-inf' or isinstance(value,(int,float)) and value<=-1e4,
                        'actual attention-mask causal semantics differ')


def validate_masked_scores(es,step):
    for name,item in step['tensors'].items():
        if not item.get('allow_negative_infinity'):continue
        require(name.endswith('.attention_masked_scores'),'nonfinite allowance outside masked attention')
        blob=tensor_blob(es,item);q,k=item['shape'][-2:];c=step['context']
        require(q==c['input_length'] and k==len(c['attention_mask']),'masked score context shape differs')
        fmt='<e' if item['dtype']=='float16' else '<f'
        for i,(value,) in enumerate(struct.iter_unpack(fmt,blob)):
            require(math.isfinite(value) or value==-math.inf and i%k>c['positions'][(i//k)%q],
                    'nonfinite score outside future masked positions')


def compare_steps(es_a,a,es_b,b,*,kind,prefix,history_a=(),history_b=()):
    qa,qb=align(a['context'],b['context'],kind=kind,prefix=prefix,history_a=history_a,history_b=history_b)
    records=[]
    keys=sorted(set(a['tensors']) & set(b['tensors']),key=operation_order)
    for key in keys:
        if '.input_past_' in key or 'attention_kv_' in key: continue
        if key=='logits' and kind=='source_prefix': continue
        left,right=(None,None) if key=='logits' else (qa,qb)
        metrics=compare(es_a,a['tensors'][key],es_b,b['tensors'][key],left,right,
                        key_stop=prefix if kind=='source_prefix' else None)
        records.append(dict(comparison=kind,step=a['step'],operation=key,metrics=metrics))
    require(records,'no comparable layer evidence')
    return records


def wire_audit(path, fsteps, layers, fixture):
    """Link serialized bytes, CPU inputs/results and actual ES correction bytes."""
    es=path/'es';eswire=indexed(jsonl(es/'fidelity_wire.jsonl'),('forward_id','layer','kind'))
    udwire=indexed(jsonl(es/'fidelity_ud_wire.jsonl'),('forward_id','layer','kind'))
    math=indexed(jsonl(es/'fidelity_ud_math.jsonl'),('forward_id','layer'))
    frames=indexed([dict(f,forward_id=f['header']['request_id'],layer=f['header']['layer_id'],kind=f['header']['kind'])
                    for f in jsonl(es/'es_wire.jsonl') if f['header']['kind'] in ('HIDDEN','DELTAS') and
                    historical_id(f['header']['request_id'].split('.s')[0]) in SELECTED],('forward_id','layer','kind'))
    expected={(s['forward_id'],i,k) for s in fsteps.values() for i in range(layers) for k in ('HIDDEN','DELTAS')}
    require(set(eswire)==set(udwire)==set(frames)==expected and
            set(math)=={(fid,i) for fid,i,k in expected},'missing/orphan wire/math observations')
    checks=0
    byfid={s['forward_id']:s for s in fsteps.values()}
    for key in sorted(expected):
        a,b=eswire[key]['tensor'],udwire[key]['tensor'];blob=tensor_blob(es,a)
        require(blob==tensor_blob(es,b) and digest_bytes(blob)==frames[key]['header']['payload_sha256'],
                'TCP serialization or endpoint reconstruction differs')
        fid,i,kind=key;step=byfid[fid];actual=math[fid,i]['input' if kind=='HIDDEN' else 'output']
        require(a['dtype']==actual['dtype'] and a['shape']==actual['shape'] and blob==tensor_blob(es,actual),
                'CPU arithmetic input/output serialization differs')
        if not fixture:
            if kind=='HIDDEN':
                require(blob==tensor_blob(es,step['tensors'][f'L{i}.projection_input']), 'ES hidden serialization differs')
            else:
                width=len(blob)//3
                for j,r in enumerate('qkv'):
                    t=step['tensors'][f'L{i}.delta_{r}']
                    require(t['dtype']==a['dtype'] and blob[j*width:(j+1)*width]==tensor_blob(es,t),
                            'ES delta CPU-to-GPU reconstruction differs')
        checks+=1
    return dict(status='PASS',payload_checks=checks,arithmetic_replay=False,
                scope='actual pre-serialization CPU results vs actual TCP payloads vs actual ES inputs/deltas; arithmetic across shapes is compared separately')


def reconstruction(es, source, target, full_es, full, evidence, layers, heads, hidden):
    checks=[];prefix=evidence['grant']['prefix_length']
    require(evidence['grant']['source_request_id']==source['request_id'] and
            digest(evidence['source_proof'])==evidence['proof_sha256']==evidence['grant']['proof_sha256'], 'source reconstruction proof differs')
    for i in range(layers):
        for r in 'kv':
            raw=evidence['tensors'][f'L{i}.cached_raw_{r}']
            srcraw=source['tensors'][f'L{i}.raw_{r}']
            require(raw['shape']==[1,prefix,hidden] and raw['dtype']==srcraw['dtype'],'cached Raw-KV shape/dtype differs')
            transformed=raw_head_bytes(tensor_blob(es,raw),raw['shape'],heads,2 if raw['dtype']=='float16' else 4)
            derived=evidence['tensors'][f'L{i}.derived_{r}'];built=evidence['tensors'][f'L{i}.reconstructed_{r}']
            require(derived['shape']==built['shape']==[1,heads,prefix,hidden//heads] and
                    derived['dtype']==built['dtype']==raw['dtype'],'reconstructed KV layout differs')
            require(transformed==tensor_blob(es,derived),'cached Raw-K/V transform corrupted')
            src=source['tensors'][f'L{i}.attention_kv_{r}'];consumed=target['tensors'][f'L{i}.input_past_{r}']
            used=source['tensors'].get(f'L{i}.used_attention_kv_{r}')
            if used is not None:
                require(compare(es,used,es,src)['bitwise_equal'],'used/source Attention KV differs')
            pairs=[('source_raw_vs_owned_raw',srcraw,raw),('source_attention_vs_derived',src,derived),
                   ('source_attention_vs_reconstructed',src,built),('derived_vs_reconstructed',derived,built),
                   ('reconstructed_vs_consumed',built,consumed)]
            for name,a,b in pairs:
                m=compare(es,a,es,b);require(m['bitwise_equal'],'corrupted cached K/V: '+name)
                checks.append(dict(layer=i,role=r,comparison=name,metrics=m))
            m=compare(es,src,full_es,full['tensors'][f'L{i}.attention_kv_{r}'],(0,prefix),(0,prefix))
            checks.append(dict(layer=i,role=r,comparison='source_attention_vs_full_target_prefix',metrics=m))
    return checks


def hypotheses(prefix_records,matched,recon,fixture):
    pre=any(not r['metrics']['bitwise_equal'] for r in prefix_records)
    suffix=any(not r['metrics']['bitwise_equal'] for r in matched)
    base=[];delta=[]
    for records,dest,branch in ((prefix_records,base,'base_'),(matched,base,'base_'),
                                (prefix_records,delta,'delta_'),(matched,delta,'delta_')):
        inputs={(r['step'],r['operation'].split('.')[0]):r['metrics']['bitwise_equal']
                for r in records if r['operation'].endswith('.projection_input')}
        dest.extend(dict(step=r['step'],operation=r['operation'],metrics=r['metrics']) for r in records
                    if '.'+branch in r['operation'] and not r['metrics']['bitwise_equal'] and
                    inputs.get((r['step'],r['operation'].split('.')[0])) is True)
    return dict(
        H1=dict(status='OBSERVED' if pre else 'NOT_OBSERVED',
                claim='source/full-target corresponding prefix differs' if pre else 'captured corresponding prefix values agree',
                causal_kernel_attribution='UNRESOLVED; shape and kernel selection are not independently intervened on'),
        H2=dict(status='EXCLUDED_FOR_CAPTURED_KV',claim='independent byte transform, source attention KV, derived KV and reconstructed KV are identical'),
        H3=dict(status='CONTENTS_PRESERVED_CONSUMPTION_CAUSE_UNRESOLVED',
                claim='actual consumed past is identical; equal bytes alone do not isolate the effect of the cache API/attention execution'),
        H4=dict(status='OBSERVED_PATH_DIFFERENCE' if suffix else 'NOT_OBSERVED',
                claim='matched full-target vs suffix/decode comparison; prefix-origin effects and suffix-kernel effects may coexist'),
        H5=dict(status='NO_CONTEXT_ERROR_OBSERVED',claim='requested and observed positions, cache positions, masks, layout and physical identity validated'),
        H6=dict(status='SERIALIZATION_EXCLUDED_ARITHMETIC_MEASURED' if not fixture else 'FIXTURE_ONLY',
                equal_input_delta_differences=delta,
                claim='actual CPU results preserved across both TCP endpoints; equal-input different-shape deltas are a separate arithmetic observation'),
        H7=dict(status='SUPPORTED_ARITHMETIC_SHAPE_SENSITIVITY' if base or delta else 'UNRESOLVED',
                equal_input_base_differences=base,production_dtype='float32 fixture' if fixture else 'float16',
                claim='same-weight equal-input slices with different surrounding execution shapes support arithmetic shape sensitivity; no FP16-vs-higher-precision intervention performed'))


def analyze(root, *, sealed=True):
    root=Path(root)
    if sealed:
        verify(root)
        require(read(root/'fidelity_results.json')['execution_status']=='COMPLETE','incomplete or failed forensic input')
    config=read(root/'fidelity_config.json');sources=read(root/'fidelity_sources.json');fixture=config['fixture']
    require(config['targets']==list(TARGETS) and config['generation_mode']=='fixed' and config['warmup_requests']==0,
            'unexpected diagnostic scope/settings')
    blocks={};audits={};identities=[];invariants=[];all_fsteps={};wire={};pids=[];sessions=[]
    for condition in ('A','B'):
        path=root/'blocks'/condition;es=path/'es';plan=read(es/'es_plan.json')
        require(plan['condition']==condition and plan['sources']==sources and all(plan[k]==v for k,v in config.items()),
                'condition configuration/source identity differs')
        audits[condition]=dict(physical=audit_r5(path,performance_block=True),accounting=accounting_audit(path))
        life=read(es/'lifecycle.json');require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'unclean ES exit')
        require(read(path/'r5_final_reconciliation.json')['status']=='PASS','final reconciliation missing')
        identity=read(path/'r5_identity.json');identities.append(identity)
        invariant={k:v for k,v in identity.items() if k!='gpu'}
        invariant['gpu_hardware']=None if identity['gpu'] is None else {k:identity['gpu'][k] for k in ('name','uuid')}
        invariants.append(invariant)
        initial=read(path/'r5_initial_state.json');worker=read(es/'es_result.json')
        require(initial['cache']['occupancy_bytes']==0 and initial['cache']['entry_count']==0 and not initial['index_entries'] and
                not initial['seen'] and not initial['cluster']['centroids'] and initial['cluster']['queries']==0 and
                initial['driver_cache_type_is_none'] and not initial['active_kv_history'],'diagnostic state leaked')
        require(read(es/'ud_order_initial.json')==dict(grants=[],parents=[],finished=[],layer=0,current=None),'UD history leaked')
        pids.extend([worker['pid'],worker['ud_pid']]);sessions.append(worker['session_id'])
        layers=identity['model_identity']['layers']
        rows={historical_id(r['request_id']):r for r in jsonl(path/'r5_generation_outputs.jsonl')}
        reqs={historical_id(e['request']['request_id']):e['request'] for e in jsonl(path/'r5_requests.jsonl') if e['kind']=='ARRIVAL'}
        from .r5_workload import workload
        original=workload(fixture=fixture)
        require(list(reqs)==[r['request_id'] for r in original] and
                all(r['input_text']==reqs[r['request_id']]['input_text'] and r['request_sequence_number']==reqs[r['request_id']]['request_sequence_number'] for r in original),
                'original workload/dependency order changed')
        counts=audits[condition]['physical']
        require((counts['hits'],counts['admissions'],counts['evictions'])==((0,0,0) if condition=='A' else (4,4,1)),
                'original automatic cache lifecycle did not occur')
        fsteps=indexed(jsonl(es/'fidelity_steps.jsonl'),('request_id','step'))
        expected={(rows[rid]['request_id'],i) for rid in SELECTED for i in range(len(rows[rid]['generated_token_ids']))}
        require(set(fsteps)==expected,'missing/orphan selected forensic forwards')
        for s in fsteps.values():
            validate_observations(s,layers,fixture)
            validate_masked_scores(es,s)
            for item in s['tensors'].values(): tensor_blob(es,item)
        wire[condition]=wire_audit(path,fsteps,layers,fixture)
        blocks[condition]=(path,rows,reqs);all_fsteps[condition]=fsteps
    require(invariants[0]==invariants[1],'model/adapter/tokenizer/encoder/GPU/numerical settings differ')
    require(len(set(pids))==4 and len(set(sessions))==2,'ES/UD processes or sessions reused')
    identity=identities[0];layers=identity['model_identity']['layers'];heads=identity['contract']['heads'];hidden=identity['model_identity']['hidden_size']
    a,ar,aq=blocks['A'];b,br,bq=blocks['B'];ea=a/'es';eb=b/'es';ast=all_fsteps['A'];bst=all_fsteps['B']
    recon=indexed(jsonl(eb/'fidelity_reconstruction.jsonl'),('request_id',))
    require(set(recon)=={(t,) for t in TARGETS},'missing/unexpected diagnostic reconstructions')
    reports=[];dependencies=[];comparisons=[];kv=[];tolerances=[];contexts=[]
    for rid in TARGETS:
        ra,rb=ar[rid],br[rid];grant=rb['source'];require(grant is not None,'target did not physically HIT: '+rid)
        source=grant['source_request_id'];p=grant['prefix_length'];source_row=br[source];src=bst[source,0]
        require(source_row['completion_ns']<rb['arrival_ns'] and source_row['admission'] and source_row['mode']=='COLD_MISS',
                'source is not a completed standalone admission')
        require(aq[rid]['token_ids']==bq[rid]['token_ids'] and aq[rid]['positions']==bq[rid]['positions'] and
                aq[rid]['attention_mask']==bq[rid]['attention_mask'] and
                aq[rid]['generation_policy']==bq[rid]['generation_policy'],'target A/B identity/context mismatch')
        full=ast[ra['request_id'],0];hit=bst[rid,0]
        prefix_records=compare_steps(eb,src,ea,full,kind='source_prefix',prefix=p)
        matched=[];skipped=[];gates_for_request=[]
        for step in range(max(len(ra['generated_token_ids']),len(rb['generated_token_ids']))):
            if (step>=min(len(ra['generated_token_ids']),len(rb['generated_token_ids'])) or
                ra['generated_token_ids'][:step]!=rb['generated_token_ids'][:step]):
                skipped.append(dict(step=step,reason='UNMATCHED_DECODE_HISTORY_OR_LENGTH'));continue
            sa,sb=ast[ra['request_id'],step],bst[rid,step]
            matched.extend(compare_steps(ea,sa,eb,sb,kind='matched',prefix=p,
                history_a=ra['generated_token_ids'],history_b=rb['generated_token_ids']))
            av=unpack(tensor_blob(ea,sa['tensors']['logits']),sa['tensors']['logits']['dtype'])
            bv=unpack(tensor_blob(eb,sb['tensors']['logits']),sb['tensors']['logits']['dtype'])
            m=check_parity(bv,av,[1,1,len(av)])
            gates_for_request.append(dict(request_id=rid,step=step,**m))
        rk=reconstruction(eb,src,hit,ea,full,recon[rid,],layers,heads,hidden)
        prefix_first=first_divergence(prefix_records);matched_first=first_divergence(matched)
        logits_differ=any(not r['metrics']['bitwise_equal'] for r in matched if r['operation']=='logits')
        hypotheses_for_request=hypotheses(prefix_records,matched,rk,fixture)
        reports.append(dict(request_id=rid,source_request_id=source,source_prefix_first_divergence=prefix_first,
            matched_suffix_decode_first_divergence=matched_first,
            first_divergent_layer=matched_first['layer'],first_divergent_operation=matched_first['operation'],
            first_divergence_type=matched_first.get('type'),magnitude=matched_first.get('metrics'),
            before_reconstruction_prefix_difference=prefix_first['status']=='FIRST_OBSERVED_DIVERGENCE',
            reconstruction_introduced_difference=False,matched_divergence_observed_after_reconstruction=True if matched_first['operation'] else None,
            causal_origin='UNRESOLVED; source-prefix and suffix observations are separate claims',
            logits_differ=logits_differ,propagated_to_logits='CO_OCCURS_CAUSAL_PROPAGATION_NOT_ISOLATED' if logits_differ else 'NOT_OBSERVED',
            generated_tokens_changed=ra['generated_token_ids']!=rb['generated_token_ids'],
            generated_text_agrees=ra['generated_text']==rb['generated_text'],
            baseline_tokens=ra['generated_token_ids'],reuse_tokens=rb['generated_token_ids'],
            frozen_tolerance_pass=not skipped and all(g['passed'] for g in gates_for_request),
            skipped_decode=skipped,hypotheses=hypotheses_for_request,root_cause='ROOT_CAUSE_UNRESOLVED'))
        dep=dict(target_request_id=rid,source_request_id=source,grant=grant,source_completion_ns=source_row['completion_ns'],
                 target_arrival_ns=rb['arrival_ns'],source_admission=source_row['admission'],source_completion_proof_sha256=digest(source_row))
        dependencies.append(dep)
        contexts.append(dict(request_id=rid,source_request=bq[source],no_reuse_request=aq[rid],reuse_request=bq[rid],
            exact_prefix_token_ids=bq[source]['token_ids'],source_completion=source_row,source_prefill_context=src['context'],
            target_full_context=full['context'],target_suffix_context=hit['context']))
        comparisons.extend(dict(request_id=rid,**r) for r in prefix_records+matched)
        kv.extend(dict(request_id=rid,**r) for r in rk);tolerances.extend(gates_for_request)
    decision=gates()
    decision.update(R5_I3F_GPU_FORENSICS_COMPLETE=not fixture,R5_I3F_CACHE_INTEGRITY_PASS=True,
        R5_I3F_FIRST_DIVERGENCE_IDENTIFIED=not fixture and any(r['logits_differ'] for r in reports) and
            all(r['first_divergent_operation'] is not None for r in reports if r['logits_differ']),
        R5_I3F_FROZEN_NUMERICAL_TOLERANCE_PASS=all(r['frozen_tolerance_pass'] for r in reports),
        R5_I3F_INDEPENDENT_AUDIT_PASS=True,R5_I3F_GENERATION_TOKEN_FIDELITY_PASS=all(not r['generated_tokens_changed'] for r in reports))
    limitations=dict(TEST_FIXTURE_ONLY=fixture,root_cause='ROOT_CAUSE_UNRESOLVED',
        historical_failures='NOT_REPRODUCED_BY_CPU_FIXTURE' if fixture else 'new captured run; compare measured reports to historical values',
        cpu_tests_gate='unit-test execution is reported separately; fixture replay does not certify the unit suite',
        native_peft='UNESTABLISHED; distributed comparisons are not Native PEFT parity',performance='historical qualification remains false; forensic capture is not a timing benchmark',
        controls='A/B are one isolated block each; no repetition, precision sweep, cache-container intervention or kernel profiler',
        attention='Python locals captured only when obtainable from installed eager code; output_attentions remains False',
        first_divergence='first observed operation; parallel branches retained as peers; hooks cannot exclude unobserved internal operations',
        fixture='scalar decoder has no OPT norms/FFN or eager attention-local evidence' if fixture else False,
        fallback='unchanged controller rejects unsupported identities and aborts corrupted-cache use; operator can run distributed no-reuse',
        correction='none: no implementation bug established',handoff='do not begin R5-I4A until operator reviews real GPU evidence, frozen failures, independent audit and unresolved causal controls')
    return dict(decision=decision,audit=dict(status='PASS',scope='CPU_FIXTURE_ONLY' if fixture else 'ACTUAL_GPU_TENSOR_FORENSICS',
        blocks=audits,wire=wire,inventory_hashes_verified=True,requests=20,diagnostic_targets=4),identity=identity,
        dependencies=dependencies,contexts=contexts,comparisons=comparisons,reconstruction=kv,divergence=reports,tolerances=tolerances,limitations=limitations)


def export(output, report):
    output=Path(output)
    for key,value in report.items(): write(output/f'fidelity_{key}.json',value)
