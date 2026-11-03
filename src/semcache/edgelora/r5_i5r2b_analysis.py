"""Independent Layer-0 arithmetic/transfer analysis around the unchanged I5 audit."""
import json
from pathlib import Path

from .communication import require
from .r3a_artifacts import write,pack,unpack
from .r4b2_analysis import raw_head_bytes
from .r5_i5_analysis import analyze as audit_i5
from .r5_i5_evidence import fresh
from .r5_i5r2_tensors import Evidence,metrics,DTYPES
from .r5_i5r2b_plan import validate_plan,VERSION

ROLES='qkv'
ES_REQUIRED=['layer_input','norm_input','norm_output','token_embedding','position_embedding','actual_attention_mask','attention_output',
    'wire_send_HIDDEN','wire_recv_DELTAS']+[f'{name}_{r}' for r in ROLES for name in ('projection_input','base','delta_on_ES','sum_before_cast','total','raw_qkv')]
UD_REQUIRED=['CPU_reconstructed_hidden','CPU_returned_bundle','wire_recv_HIDDEN','wire_send_DELTAS']+[
    f'{name}_{r}' for r in ROLES for name in ('CPU_input','CPU_A','CPU_B_input','CPU_B','CPU_scaled')]
ORDER={'token_embedding':0,'position_embedding':0,'layer_input':1,'norm_input':2,'norm_output':3,
    'actual_attention_mask':1,'wire_send_HIDDEN':4.1,'wire_recv_HIDDEN':4.2,'CPU_reconstructed_hidden':4.3,
    'CPU_returned_bundle':8.1,'wire_send_DELTAS':8.2,'wire_recv_DELTAS':8.3,
    **{f'{name}_{r}':rank for r in ROLES for name,rank in (('projection_input',4),('base',10),('CPU_input',5),('CPU_A',6),('CPU_B_input',6),('CPU_B',7),
        ('CPU_scaled',8),('delta_on_ES',9),('sum_before_cast',11),('total',12),('raw_qkv',12))},'attention_output':13}


def classify(comparisons,missing=()):
    """Only aligned equal input values justify localizing an operation boundary."""
    by={r['boundary']:r['metrics']['bitwise_equal'] for r in comparisons}
    all_equal=lambda names: all(by.get(n) is True for n in names)
    any_diff=lambda names: any(by.get(n) is False for n in names)
    inputs=[f'projection_input_{r}' for r in ROLES];bases=[f'base_{r}' for r in ROLES]
    deltas=[f'delta_on_ES_{r}' for r in ROLES];totals=[f'total_{r}' for r in ROLES]
    cases=[];localized=[]
    if any_diff(inputs):
        cases.append('CASE_A_INPUT_HIDDEN_DIFFERS');localized.append(dict(boundary='BEFORE_QKV_PROJECTION',evidence='Input values already differ; do not attribute differing outputs to Linear'))
        if by.get('norm_input') is True and by.get('norm_output') is False:
            localized.append(dict(boundary='LAYER_0_PRE_ATTENTION_NORMALIZATION',evidence='Equal aligned normalization inputs, different observed outputs; mechanism still unresolved'))
    if all_equal(inputs) and any_diff(bases):
        cases.append('CASE_B_GPU_BASE_CANDIDATE');localized.append(dict(boundary='NATIVE_GPU_LINEAR',evidence='Equal aligned input values and differing base outputs; original sequence shapes differ. Kernel/precision/nondeterminism alternatives remain.'))
    if all_equal(bases) and any_diff(deltas):cases.append('CASE_C_CPU_OR_TRANSFER_CANDIDATE')
    for role in ROLES:
        for upstream,downstream in ((f'CPU_input_{role}',f'CPU_A_{role}'),(f'CPU_B_input_{role}',f'CPU_B_{role}'),(f'CPU_B_{role}',f'CPU_scaled_{role}')):
            if by.get(upstream) is True and by.get(downstream) is False:
                localized.append(dict(boundary=downstream,evidence='Equal recorded aligned inputs and different output; immutable adapter matrices verified by I5 audit'))
    if all_equal(bases+deltas) and any_diff(totals):
        cases.append('CASE_D_ADDITION_OR_CAST_CANDIDATE')
        for role in ROLES:
            if by.get('base_'+role) is True and by.get('delta_on_ES_'+role) is True:
                if by.get('sum_before_cast_'+role) is False:localized.append(dict(boundary='sum_before_cast_'+role,evidence='Equal base/delta values; differing actual add output'))
                elif by.get('sum_before_cast_'+role) is True and by.get('total_'+role) is False:
                    localized.append(dict(boundary='total_'+role,evidence='Equal actual sum values; different final-cast output'))
    if all_equal(totals) and by.get('attention_output') is False:cases.append('CASE_E_ATTENTION_CANDIDATE')
    if missing or not all(n in by for n in inputs+bases+deltas+totals):cases.append('CASE_F_INCOMPLETE_CAPTURE')
    ordered=sorted(comparisons,key=lambda r:(ORDER.get(r['boundary'],100),r['boundary']))
    observed=next((r for r in ordered if r['metrics']['bitwise_equal'] is False),None)
    return dict(cases=cases or ['NO_DIFFERENCE_IN_OBSERVED_BOUNDARIES'],FIRST_OBSERVED_DIFFERENCE=observed,
        FIRST_CAUSALLY_LOCALIZED_BOUNDARY=localized or 'UNRESOLVED',ROOT_CAUSE_CONFIRMED=False,ROOT_CAUSE='UNRESOLVED',
        peers=[r['boundary'] for r in ordered if observed and ORDER.get(r['boundary'],100)==ORDER.get(observed['boundary'],100) and not r['metrics']['bitwise_equal']],
        interpretation='Localization is not a mechanism. Same-shape repeats, controlled identical-input shape tests and instrumentation-off GPU comparisons are NOT_RUN.')


def analyze(experiment,run_dir,output):
    plan=validate_plan(experiment);root=Path(run_dir);cfg=plan['i5r2b'];legacy=audit_i5(root)
    require(legacy['config']['experiment_id']==plan['experiment_id'] and legacy['config']['source_commit']==cfg['source']['commit'] and
        legacy['config'].get('i5r2b')==cfg,'arithmetic execution/source identity differs')
    ev=Evidence(root,max_input_bytes=4*2**30);ev.verify();inventories={};backends={};entries={}
    source,target=cfg['source_request_id'],cfg['target_request_id'];missing=[];comparisons=[];integrity=[]
    conditions=('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED')
    for c in conditions:
        for endpoint,directory in (('ES','es'),('UD','ud')):
            base=c+'/Adapter_A/'+directory;cap=ev.json(base+'/i5r2b_capture.json');backends[c,endpoint]=ev.json(base+'/i5r2b_backend.json')
            require(cap['version']==VERSION and cap['status']=='COMPLETE' and cap['TEST_FIXTURE_ONLY']==plan['TEST_FIXTURE_ONLY'] and
                cap['byte_limit']==cfg['capture_bytes'] and cap['file_limit']==cfg['max_files'],'diagnostic capture identity/status differs')
            captured=[item for item in cap['entries'].values() if item['status']=='CAPTURED']
            require(cap['files']==len(captured)<=cfg['max_files'] and cap['bytes']==sum(ev.files[base+'/'+item['file']] for item in captured)<=cfg['capture_bytes'],
                    'diagnostic capture accounting differs')
            for key,item in cap['entries'].items():
                entries[c,endpoint,key]=(base,item)
                if item['status']=='CAPTURED':
                    require(item['semantic_positions']==[0,64] and item['shape'][item['token_axis']]==64,'diagnostic captured suffix or incorrect positions')
                    ev.validate_tensor_reference(base,item,required_file=True)
            inventories[c+'/'+endpoint]=cap
            # Bind whole-payload receipts to the original independently audited
            # protocol ledger; selected bytes alone cannot authenticate suffixes.
            ledger=ev.jsonl(base+'/wire.jsonl')
            for key,record in cap['entries'].items():
                if record['status']!='CAPTURED' or '/wire_' not in key:continue
                rid,name=key.split('/',1);_,direction,kind=name.split('_')
                matching=[f['header'] for f in ledger if f['operation']==direction and f['header']['kind']==kind and
                    f['header']['layer_id']==0 and f['header']['request_id']==rid+'.s00']
                require(len(matching)==1,'diagnostic wire receipt missing or ambiguous')
                header=matching[0]
                require(record['full_payload_sha256']==header['payload_sha256'] and record['full_payload_shape']==header['shape'] and
                        record['full_payload_bytes']==header['payload_bytes'],'diagnostic whole-payload receipt differs from TCP ledger')
                integrity.append(dict(check=c+'/'+endpoint+'/'+key+'/whole_payload_receipt',status='PASS'))
    def item(c,endpoint,rid,name):
        value=entries.get((c,endpoint,rid+'/'+name))
        return value if value and value[1]['status']=='CAPTURED' else None
    def blob(record):return ev.tensor(*record) if record else None
    def compare_pair(endpoint,name):
        left=item(conditions[0],endpoint,source,name);right=item(conditions[0],endpoint,target,name)
        if not left or not right:
            missing.append(dict(endpoint=endpoint,boundary=name,source_missing=not left,target_missing=not right));return
        a,b=left[1],right[1];require(a['dtype']==b['dtype'],'corresponding arithmetic boundary dtype differs')
        x,y=blob(left),blob(right)
        # Actual additive masks have different key lengths. Align keys 0:64;
        # validate that all extra target keys are masked for these prefix queries.
        if name=='actual_attention_mask':
            from .r5_i5r2_tensors import slice_axis
            width=DTYPES[a['dtype']][1];x,sx=slice_axis(x,a['shape'],len(a['shape'])-1,0,64,width)
            y,sy=slice_axis(y,b['shape'],len(b['shape'])-1,0,64,width)
            if b['shape'][-1]>64:
                extra,_=slice_axis(blob(right),b['shape'],len(b['shape'])-1,64,b['shape'][-1],width)
                import struct,math
                require(all(v==-math.inf or v<=-1e4 for (v,) in struct.iter_unpack(DTYPES[b['dtype']][0],extra)),'full-target prefix can attend to future suffix')
        else:require(a['shape']==b['shape'],'corresponding arithmetic boundary shapes differ')
        m=metrics(x,y,a['dtype'],allow_negative_infinity=name=='actual_attention_mask')
        m.update(frozen_max_abs_verdict='PASS' if m['max_abs'] is not None and m['max_abs']<=legacy['numerical_fidelity']['thresholds']['max_abs'] else 'FAIL',
                 frozen_relative_l2_verdict='PASS' if m['relative_l2'] is not None and m['relative_l2']<=legacy['numerical_fidelity']['thresholds']['relative_l2'] else 'FAIL')
        comparisons.append(dict(boundary=name,endpoint=endpoint,layer=0,phase='PREFILL',semantic_positions=[0,64],source=a,target=b,metrics=m))
    for endpoint,names in (('ES',ES_REQUIRED),('UD',UD_REQUIRED)):
        for name in names:compare_pair(endpoint,name)
    def check(label,left,right):
        if not left or not right:
            integrity.append(dict(check=label,status='NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE'));return
        require(left==right,'diagnostic transfer/model-boundary integrity failed: '+label)
        integrity.append(dict(check=label,status='PASS'))
    for c in conditions:
        observed_steps=ev.jsonl(c+'/Adapter_A/es/r5_tokens.jsonl')
        for endpoint in ('ES','UD'):
            for rid,ctx in inventories[c+'/'+endpoint]['contexts'].items():
                actual=[s['context'] for s in observed_steps if s['request_id']==rid and s['context']['pass_index']==0]
                require(len(actual)==1,'diagnostic prefill context missing/ambiguous')
                context=actual[0]
                require(ctx['origin']==context['past_length'] and ctx['input_length']==context['input_length'],'diagnostic position context differs')
                if endpoint=='UD' and c==conditions[2] and rid==target:
                    continue  # No selected prefix CPU arithmetic on a legitimate HIT.
                require(ctx.get('actual_context')==context and ctx.get('backend_before') is not None and ctx.get('backend_after') is not None,
                        'diagnostic context/backend receipt differs from audited forward')
                if endpoint=='ES':
                    require(ctx['token_ids']==context['input_token_ids'] and ctx['positions']==context['positions'] and
                            ctx['attention_mask']==context['attention_mask'],'diagnostic token/position/mask alignment differs')
        for rid in (source,target):
            # Prefix arithmetic on a HIT is intentionally absent, not missing work.
            if c==conditions[2] and rid==target:continue
            check(c+'/'+rid+'/hidden_wire',blob(item(c,'ES',rid,'wire_send_HIDDEN')),blob(item(c,'UD',rid,'wire_recv_HIDDEN')))
            check(c+'/'+rid+'/hidden_reconstruction',blob(item(c,'UD',rid,'wire_recv_HIDDEN')),blob(item(c,'UD',rid,'CPU_reconstructed_hidden')))
            bundle=blob(item(c,'UD',rid,'CPU_returned_bundle'));sent=blob(item(c,'UD',rid,'wire_send_DELTAS'));received=blob(item(c,'ES',rid,'wire_recv_DELTAS'))
            check(c+'/'+rid+'/delta_serialization',bundle,sent);check(c+'/'+rid+'/delta_wire',sent,received)
            for index,role in enumerate(ROLES):
                check(c+'/'+rid+'/ES_hidden_serialization_'+role,blob(item(c,'ES',rid,'projection_input_'+role)),blob(item(c,'ES',rid,'wire_send_HIDDEN')))
                cpu_input=item(c,'UD',rid,'CPU_input_'+role);hidden=item(c,'UD',rid,'CPU_reconstructed_hidden')
                if cpu_input and hidden:
                    cast=pack(unpack(blob(hidden),hidden[1]['dtype']),cpu_input[1]['dtype'])
                    check(c+'/'+rid+'/CPU_input_cast_'+role,cast,blob(cpu_input))
                check(c+'/'+rid+'/CPU_A_to_B_'+role,blob(item(c,'UD',rid,'CPU_A_'+role)),blob(item(c,'UD',rid,'CPU_B_input_'+role)))
                scaled=item(c,'UD',rid,'CPU_scaled_'+role)
                if scaled and bundle:
                    size=len(blob(scaled));check(c+'/'+rid+'/CPU_scaled_bundle_'+role,blob(scaled),bundle[index*size:(index+1)*size])
                check(c+'/'+rid+'/total_raw_'+role,blob(item(c,'ES',rid,'total_'+role)),blob(item(c,'ES',rid,'raw_qkv_'+role)))
                delta=item(c,'ES',rid,'delta_on_ES_'+role)
                if delta and received:
                    width=DTYPES[delta[1]['dtype']][1];size=64*delta[1]['shape'][-1]*width
                    check(c+'/'+rid+'/received_delta_'+role,received[index*size:(index+1)*size],blob(delta))
    rows={(r['condition'],r['request_id']):r for r in legacy['request_metrics']};hit=rows[conditions[2],target]
    require(hit['source'] is not None and hit['source']['source_request_id']==source and hit['reused_prefix_tokens']==64 and not hit['cached_prefix_q_consumed'],
            'secondary legitimate physical HIT not exercised')
    for role in 'kv':
        check('cached_source_vs_consumed_'+role,blob(item(conditions[2],'ES',source,'returned_prefix_'+role)),blob(item(conditions[2],'ES',target,'consumed_prefix_'+role)))
        check('consumed_vs_preserved_'+role,blob(item(conditions[2],'ES',target,'consumed_prefix_'+role)),blob(item(conditions[2],'ES',target,'returned_prefix_'+role)))
        raw_source=item(conditions[2],'ES',source,'raw_qkv_'+role);kv=item(conditions[2],'ES',source,'returned_prefix_'+role)
        if raw_source and kv:
            check('raw_head_transform_'+role,raw_head_bytes(blob(raw_source),raw_source[1]['shape'],kv[1]['shape'][1],DTYPES[raw_source[1]['dtype']][1]),blob(kv))
    for c in conditions:
        for endpoint in ('ES','UD'):
            cap=inventories[c+'/'+endpoint]
            for rid,ctx in cap['contexts'].items():
                require(ctx.get('backend_before')==ctx.get('backend_after'),'backend settings changed during diagnostic forward')
            if c==conditions[0]:
                require(cap['contexts'][source].get('backend_before')==cap['contexts'][target].get('backend_before'),'S/F backend configuration confounded')
    result=classify(comparisons,missing)
    result.update(IMPLEMENTATION_COMPLETE=True,GPU_NOT_RUN=plan['TEST_FIXTURE_ONLY'],TEST_FIXTURE_ONLY=plan['TEST_FIXTURE_ONLY'],
        GPU_EVIDENCE_COMPLETE='CPU_FIXTURE_ONLY' if plan['TEST_FIXTURE_ONLY'] else 'PASS' if not missing and all(x['status']=='PASS' for x in integrity) else 'FAIL',
        NUMERICAL_FIDELITY_RESTORED='NOT_ASSERTED',capture_layer_scope=[0],all_layer_full_tensor_audit='NOT_RUN: normal I5 scalable metadata/TCP audit plus targeted Layer-0 bytes',
        frozen_numerical_fidelity=legacy['numerical_fidelity'],token_text_gate=legacy['decision']['token_text_fidelity'],
        independent_cache_hit_receipts=1,same_shape_repeat_variability='NOT_RUN',different_shape_variation='OBSERVED_BOUNDARIES_ONLY',
        instrumentation_induced_variation='NOT_CONTROLLED',performance_claims='PROHIBITED_DIAGNOSTIC_ONLY')
    out=fresh(output,(experiment,run_dir))
    write(out/'i5r2b_capture_inventory.json',dict(config=cfg,inventories=inventories,missing=missing,integrity=integrity,
        independent_I5_audit=legacy['independent_audit'],source_commit=legacy['config']['source_commit']))
    write(out/'i5r2b_tensor_comparisons.json',comparisons);write(out/'i5r2b_first_divergence.json',result)
    write(out/'i5r2b_gpu_base_vs_lora.json',[r for r in comparisons if any(r['boundary'].startswith(n) for n in ('base_','delta_','CPU_','sum_','total_'))])
    write(out/'i5r2b_dtype_and_backend.json',{c+'/'+e:v for (c,e),v in backends.items()})
    write(out/'i5r2b_hypothesis_verdicts.json',dict(cases=result['cases'],localized=result['FIRST_CAUSALLY_LOCALIZED_BOUNDARY'],
        ROOT_CAUSE='UNRESOLVED',next_action='Review first observed and matched-input boundaries. Only then design a same-shape repeat/identical-input shape/FP32 diagnostic. No automatic follow-up.'))
    (out/'i5r2b_report.md').write_text('# R5-I5R-2B arithmetic diagnosis\n\n'+json.dumps(result,indent=2)+'\n\nA localized boundary is not a confirmed numerical mechanism. Only Layer 0 positions 0:64 are captured. Cached Q is not consumed. Diagnostic latency is invalid for speedup claims.\n')
    from .r5_i5r2_tensors import file_sha
    write(out/'i5r2b_output_manifest.json',dict(files={p.name:file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}));return result
