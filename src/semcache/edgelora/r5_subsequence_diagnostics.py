"""Read-only historical sparse-QKV forensics. No Torch, CUDA or inference.

Historical identity is verified against Git blobs, never substituted with HEAD.
Only bounded tensor slices/metrics are exported; raw evidence is not copied.
"""
from pathlib import Path
import copy,math,subprocess
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,write,sha,pack,unpack,THRESHOLDS,check_parity
from .r3a_tiny import f32
from .r5_i5r2_tensors import Evidence,metrics,DTYPES
from .r5_i5_evidence import seal
from .r5_subsequence_audit import rows,replay_projection,replay_wire
from .r5_i5_contracts import validate_request
from .r5_multi_artifacts import sharing,runtime_view

REFERENCE_COMMIT='a6fae7d3137935583de38e563eb0d46a396f7eba'
CONDITIONS=('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_SUBSEQUENCE')
UNKNOWN='NOT_OBSERVABLE'

def current_source():
    files=[ROOT/'src/semcache/edgelora'/name for name in ('r5_subsequence_diagnostics.py','r5_subsequence_profile.py','r5_subsequence_integrity.py','r5_subsequence_audit.py',
        'r5_i5r2_tensors.py','r5_i5r2_analysis.py','r5_i5_evidence.py','r5_multi_artifacts.py','communication.py','r3a_artifacts.py','r3a_tiny.py')]
    files.append(ROOT/'scripts/152_r5_subsequence_diagnostics.py')
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True),files={str(p.relative_to(ROOT)):sha(p) for p in files})

def fresh(path,protected):
    path=Path(path).resolve();namespace=ROOT/'results/r5/i5r/parallel/subsequence_diagnostics'
    require(not path.is_relative_to(ROOT) or path.is_relative_to(namespace),'diagnostic namespace required')
    require(not path.exists() and all(not path.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(path) for x in protected),'output exists/overlaps historical input')
    path.mkdir(parents=True);return path

def prepared_identity(p,configs):
    """Existing CPU validators; no tokenizer/model instantiation or downloads."""
    from .r5_multi_artifacts import validate
    multi=Evidence(p['multi']);require(multi.verify('r5_output_manifest.json')==p['multi_inventory'],'historical adapter inventory differs')
    require(Evidence(p['dataset']).verify()==p['dataset_inventory'],'historical dataset inventory differs')
    m,pm,specs,registry=validate(p['multi']);spec=specs[p['adapter_id']]
    require(not m['TEST_FIXTURE_ONLY'] and pm['identity']['model']=='facebook/opt-2.7b' and pm['identity']['layers']==32,'real historical OPT GPU preparation required')
    require(p['thresholds']==THRESHOLDS and m['tokenizer']==p['tokenizer_identity'],'frozen tokenizer/threshold differs')
    for c in configs:
        identity=c['physical'];require(c['model_identity']==runtime_view(pm,spec)['identity'] and c['registry']==registry,'historical runtime model/adapter/registry differs')
        require(identity['base_fingerprint']==pm['identity']['base_weight_sha256'] and identity['adapter_matrix_fingerprint']==spec['matrix_sha256'] and identity['tokenizer']==m['tokenizer'],'historical physical fingerprints differ')
        require(digest(c['contract'])==identity['opt_contract'],'historical OPT contract hash differs')
        require(c['contract']['versions']==pm['identity']['versions'],'historical native library versions differ')
        from .r5_i5_gpu import verify_receipt
        verify_receipt(c['gpu_identity_receipt'],pm,c['gpu'],'SAME_COMPATIBLE_GPU_CLASS')
        require(identity['gpu_uuid']==c['gpu']['uuid'],'physical/recorded GPU identity differs')

def history(experiment,run,audit,expected_source_commit=REFERENCE_COMMIT):
    pe=Evidence(experiment);pr=pe.verify();p=pe.json('subsequence_plan.json')
    require(p['schema']=='R5_SUBSEQUENCE_GENERATION_PILOT_V1' and p['mode']=='EXACT_CONTEXT_VERIFIED' and not p['allow_approximate'],'strict historical pilot required')
    require(p['source']['commit']==expected_source_commit and p['source']['branch']=='exp/raw-semcache-r5-integration','historical commit/branch differs')
    from .r5_i5r2_analysis import source_provenance
    provenance=source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=expected_source_commit,fixture=False)
    provenance['current_diagnostic']=current_source()
    if 'diagnostics' in p:
        extra=source_provenance(dict(source_commit=p['source']['commit'],sources=p['diagnostics']['instrumentation_files']),expected_commit=expected_source_commit,fixture=False)
        provenance['additional_instrumentation_blob_checks']=extra['blob_checks']
    re=Evidence(run,max_input_bytes=128*2**30);rr=re.verify();status=re.json('status.json')
    require(status['status']=='COMPLETE' and status['source']==p['source'] and status['plan_inventory']==pr,'run/plan provenance differs')
    ce=Evidence(status['cpu_validation_path']);require(ce.verify()==status['cpu_validation'],'historical CPU audit inventory differs')
    cpu=ce.json('cpu_validation.json');require(cpu['status']=='PASS' and cpu['failed']==cpu['skipped']==0 and cpu['source']==p['source'] and cpu['plan_inventory']==pr,'historical CPU approval differs')
    ae=Evidence(audit,max_input_bytes=128*2**30);ar=ae.verify();export=ae.json('subsequence_audit.json')
    require(export['status']=='INDEPENDENT_EVIDENCE_AUDIT_PASS' and export['thresholds']==THRESHOLDS,'historical independent audit invalid')
    records={}
    for condition in CONDITIONS:
        ev=Evidence(Path(run)/condition,max_input_bytes=128*2**30,max_tensor_bytes=128*2**20);inventory=ev.verify()
        config=ev.json('identity.json');reqs=ev.json('requests.json');results=ev.json('results.json')
        require(ev.json('status.json')['status']=='COMPLETE' and config['condition']==condition and config['source']==p['source'] and config['plan_inventory']==pr,'child identity differs')
        recorded=export['reports'][condition]
        require(recorded['inventory']==inventory and recorded['config']==config and recorded['requests']==reqs and recorded['results']==results,'audit not linked to these raw records')
        require(len(reqs)==len(results)==len(p['tasks'])==2 and reqs[0]['request_id']!=reqs[1]['request_id'],'bounded source/target order required')
        require(results[0]['completion_ns']<=results[1]['arrival_ns'],'source incomplete at target arrival')
        for req,row,task in zip(reqs,results,p['tasks']):
            validate_request(req)
            require(req['physical_identity']==config['physical'] and row['condition']==condition and row['context_mode']==p['mode'],'physical/context mode differs')
            require(all(req[k]==task[k] for k in ('request_id','sample_id','user_id','adapter_id','token_ids','input_text','request_sequence_number')) and req['generation_policy']==task['policy'] and row['request_id']==req['request_id'],'sealed request/order differs')
        require(reqs[0]['token_ids']==reqs[1]['token_ids'][:len(reqs[0]['token_ids'])] and len(reqs[0]['token_ids'])<len(reqs[1]['token_ids']),'source/target prefix differs')
        ev.registry=config['registry'];network=replay_wire(ev.jsonl('es/wire.jsonl'),ev.jsonl('ud/wire.jsonl'))
        records[condition]=dict(ev=ev,config=config,requests=reqs,results=results,network=network)
    baseline=records[CONDITIONS[0]]
    require(all(x['config']['physical']==baseline['config']['physical'] for x in records.values()),'A/B/C physical identity differs')
    prepared_identity(p,[x['config'] for x in records.values()])
    return p,records,export,dict(plan=pr,run=rr,audit=ar,source=provenance,process_lifecycle=status.get('children',UNKNOWN),scope='Historical exported independent audit linked by SHA256; new bounded CPU forensics; no fresh original-source GPU audit')

def projections(row):
    physical=row['completion']['physical']
    records=row['prefill_records'] if physical is None else physical['records']
    result={(r['layer'],r['role']):r for r in records}
    require(len(result)==len(records),'duplicate projection record');return result

def aligned(ev,spec,positions):
    return rows(ev.tensor('es',spec),spec['shape'],spec['dtype'],positions)

def comparison(left,right,dtype,positions,hidden,**labels):
    result=metrics(left,right,dtype);result['differing_fraction']=result['differing_values']/result['elements']
    first=result['first_bitwise_difference_element']
    return dict(**labels,dtype=dtype,positions_count=len(positions),positions_sha256=digest(positions),
        first_absolute_token_position=None if first is None else positions[first//hidden],first_hidden_index=None if first is None else first%hidden,**result)

def forensics(records):
    a,b,c=[records[k] for k in CONDITIONS];arec=projections(a['results'][1]);brec=projections(b['results'][1]);crec=projections(c['results'][1]);srec=projections(c['results'][0])
    a_source=projections(a['results'][0]);b_source=projections(b['results'][0])
    req=c['requests'][1];source=c['requests'][0];n=len(req['token_ids']);p=len(source['token_ids']);h=req['physical_identity']['hidden_size'];dtype=req['physical_identity']['dtype'];table=[]
    def compare(name,l,r,positions,layer,role,**extra):table.append(comparison(l,r,dtype,positions,h,comparison=name,layer=layer,role=role,**extra))
    ud=c['ev'].jsonl('ud/layers.jsonl');source_map={source['request_id']:(source,srec)}
    for layer in range(req['physical_identity']['layers']):
        for role in 'qkv':
            aa,bb,cc,ss=[rs[layer,role] for rs in (arec,brec,crec,srec)]
            require(all(r['input']['shape']==[1,n,h] and r['output']['shape']==[1,n,h] for r in (aa,bb,cc)) and ss['input']['shape']==ss['output']['shape']==[1,p,h],'unaligned full projection evidence')
            frame=[r for r in ud if r.get('context',{}).get('pass_id')==req['request_id']+'_s000' and r['context']['layer']==layer]
            require(len(frame)==1,'missing UD prefill operand');replay_projection(c['ev'],cc,frame[0],req,source_map)
            if layer==0:
                for condition,src,ev in (('A',a_source,a['ev']),('B',b_source,b['ev'])):
                    for boundary in ('input','fresh_base','fresh_delta','output'):
                        original=src[layer,role][boundary];current=ss[boundary]
                        require(original['shape']==current['shape']==[1,p,h] and original['dtype']==current['dtype'],'source comparison shape/dtype differs')
                        table.append(comparison(ev.tensor('es',original),c['ev'].tensor('es',current),current['dtype'],list(range(p)),h,comparison=condition+'_source_vs_C_source_'+boundary,layer=layer,role=role))
            for boundary in ('input','output'):
                compare('A_vs_B_'+boundary,a['ev'].tensor('es',aa[boundary]),b['ev'].tensor('es',bb[boundary]),list(range(n)),layer,role)
                compare('A_vs_C_'+boundary,a['ev'].tensor('es',aa[boundary]),c['ev'].tensor('es',cc[boundary]),list(range(n)),layer,role)
                compare('C_source_vs_A_target_prefix_'+boundary,c['ev'].tensor('es',ss[boundary]),aligned(a['ev'],aa[boundary],list(range(p))),list(range(p)),layer,role,source_shape=ss[boundary]['shape'],target_shape=aa[boundary]['shape'])
            for boundary in ('fresh_base','fresh_delta'):
                for name,other in (('A',a),('B',b)):
                    other_spec=(aa if name=='A' else bb)[boundary];source_spec=ss[boundary]
                    require(source_spec['dtype']==other_spec['dtype'],'source/target projection dtype differs')
                    table.append(comparison(c['ev'].tensor('es',source_spec),aligned(other['ev'],other_spec,list(range(p))),source_spec['dtype'],list(range(p)),h,
                        comparison='C_source_vs_'+name+'_target_prefix_'+boundary,layer=layer,role=role,source_shape=source_spec['shape'],target_shape=other_spec['shape']))
            cached=cc['reused_positions'];fresh_positions=cc['fresh_positions']
            if cached:compare('cached_rows_A_vs_C',aligned(a['ev'],aa['output'],cached),aligned(c['ev'],cc['output'],cached),cached,layer,role)
            for item in cc['consumed']:
                lo,hi=item['source_positions'];start,end=item['target_positions'];require(item['source_request_id']==source['request_id'],'unexpected cached source')
                compare('source_cache_vs_C_scatter',aligned(c['ev'],ss['output'],list(range(lo,hi))),c['ev'].tensor('es',item['tensor']),list(range(start,end)),layer,role)
            if fresh_positions:
                for boundary in ('fresh_base','fresh_delta'):
                    spec=cc[boundary];require(spec['shape']==[1,len(fresh_positions),h] and spec['dtype']==aa[boundary]['dtype'],'sparse native operand shape/dtype differs')
                    table.append(comparison(aligned(a['ev'],aa[boundary],fresh_positions),c['ev'].tensor('es',spec),spec['dtype'],fresh_positions,h,comparison='fresh_rows_A_vs_C_'+boundary,layer=layer,role=role,source_shape=aa[boundary]['shape'],target_shape=spec['shape']))
                compare('fresh_rows_A_vs_C_total',aligned(a['ev'],aa['output'],fresh_positions),aligned(c['ev'],cc['output'],fresh_positions),fresh_positions,layer,role)
                total=pack([f32(x+y) for x,y in zip(unpack(c['ev'].tensor('es',cc['fresh_base']),dtype),unpack(c['ev'].tensor('es',cc['fresh_delta']),'float32'))],dtype)
                compare('fresh_add_cast_vs_C_scatter',total,aligned(c['ev'],cc['output'],fresh_positions),fresh_positions,layer,role)
    return table

def eligibility(records):
    c=records[CONDITIONS[2]];req=c['requests'][1];source=c['requests'][0];row=c['results'][1];rs=projections(row);ss=projections(c['results'][0]);match=row['matching']
    selected={tuple(x) for x in match['selected_spans']};candidates=match['candidates'];selected_candidates=[x for x in candidates if tuple(x['target_positions']) in selected]
    require(len(selected_candidates)==len(selected),'ambiguous historical selection')
    exact=[]
    for x in selected_candidates:
        lo,hi=x['source_positions'];start,end=x['target_positions']
        require(x['source_request_id']==source['request_id'],'selected source provenance unavailable')
        if (lo,hi)==(start,end) and source['token_ids'][:hi]==req['token_ids'][:end] and sharing(c['config']['registry'],x['source_user'],req['user_id'],x['source_adapter'],req['adapter_id']) is None:exact.append(x)
    baseline_a=records[CONDITIONS[0]];baseline_b=records[CONDITIONS[1]]
    ar=projections(baseline_a['results'][1]);br=projections(baseline_b['results'][1]);results=[]
    for layer in range(req['physical_identity']['layers']):
        mismatched=[];equal=[];preexisting_a=[];preexisting_b=[];additional_c=[];reuse_specific=[]
        for x in exact:
            lo,hi=x['source_positions'];start,end=x['target_positions']
            left=aligned(c['ev'],ss[layer,'q']['input'],list(range(lo,hi)));right=aligned(c['ev'],rs[layer,'q']['input'],list(range(start,end)))
            av=aligned(baseline_a['ev'],ar[layer,'q']['input'],list(range(start,end)));bv=aligned(baseline_b['ev'],br[layer,'q']['input'],list(range(start,end)))
            if left!=av:preexisting_a.append([start,end])
            if left!=bv:preexisting_b.append([start,end])
            if right!=av:additional_c.append([start,end])
            if left==av and left!=right:reuse_specific.append([start,end])
            (equal if left==right else mismatched).append([start,end])
        actual=rs[layer,'q']['reused_positions'];predicted=sorted(i for start,end in equal for i in range(start,end))
        require(actual==predicted,'saved consumption disagrees with independently aligned exact-input eligibility')
        results.append(dict(layer=layer,semantic_candidates=len(candidates),selected_nonoverlap_windows=len(selected),exact_context_windows=len(exact),
            not_selected_windows=len(candidates)-len(selected),exact_context_rejections=len(selected)-len(exact),input_equal_windows=equal,input_mismatch_windows=mismatched,
            observed_input_mismatch_count=len(mismatched),actual_consumed_positions=actual,fresh_qkv_rows=sum(len(rs[layer,r]['fresh_positions']) for r in 'qkv'),reused_qkv_rows=sum(len(rs[layer,r]['reused_positions']) for r in 'qkv'),
            source_vs_A_input_mismatch_windows=preexisting_a,source_vs_B_input_mismatch_windows=preexisting_b,C_vs_A_input_mismatch_windows=additional_c,reuse_specific_rejection_windows=reuse_specific,
            ownership_integrity='EXPORTED_INDEPENDENT_AUDIT_AND_NEW_CONSUMED_VALUE_REPLAY',historical_rejection_event_count=UNKNOWN,policy_rejections=UNKNOWN,
            reason_scope='input mismatch is reconstructed from saved selected-source inputs; original runtime did not record a per-layer rejection reason'))
    return results

def logits(records,export):
    a=records[CONDITIONS[0]];results=[]
    for condition,data in records.items():
        for request_index,(left,right) in enumerate(zip(a['results'],data['results'])):
            av=unpack(a['ev'].tensor('es',left['logits']),'float32');bv=unpack(data['ev'].tensor('es',right['logits']),'float32');width=left['logits_shape'][-1]
            require(width==right['logits_shape'][-1] and len(av)%width==len(bv)%width==0,'invalid full-vocabulary logit alignment')
            for row,values,req in ((left,av,a['requests'][request_index]),(right,bv,data['requests'][request_index])):
                require(row['logits_shape']==row['logits']['shape']==[1,len(row['generated_token_ids']),width] and len(row['steps'])==len(row['generated_token_ids']),'incomplete generation/logit shape')
                require(req['TEST_FIXTURE_ONLY'] or width==50272,'missing full OPT vocabulary')
                for step,token in enumerate(row['generated_token_ids']):
                    chunk=values[step*width:(step+1)*width];stored=row['steps'][step]
                    require(token==stored['selected_token_id']==max(range(width),key=chunk.__getitem__) and stored['logits_sha256']==digest_bytes(pack(chunk,'float32')),'stored generated token/logit hash differs')
            steps=[check_parity(bv[i:i+width],av[i:i+width],[1,1,width]) for i in range(0,min(len(av),len(bv)),width)]
            old=[r for r in export['comparisons'] if r['condition']==condition and r['request_id']==right['request_id']]
            require(len(old)==1 and old[0]['numerical']['per_step']==steps,'historical per-step frozen gate differs')
            results.append(dict(condition=condition,request_id=right['request_id'],per_step=steps,lengths_equal=len(av)==len(bv),
                tokens_equal=left['generated_token_ids']==right['generated_token_ids'],text_equal=left['generated_text']==right['generated_text']))
    return results

def diagnose(experiment,run,audit,output_dir,expected_source_commit=REFERENCE_COMMIT):
    p,records,export,provenance=history(experiment,run,audit,expected_source_commit)
    comparisons=forensics(records);layers=eligibility(records);fidelity=logits(records,export)
    first=next((x for x in comparisons if x['comparison']=='A_vs_C_output' and not x['bitwise_equal']),None)
    costs={}
    for k,v in records.items():
        costs[k]=[]
        for row in v['results']:
            item={key:row.get(key,UNKNOWN) for key in ('request_id','application_e2e_ms','application_ttft_ms','model_ttft_ms','prefill_ms','decode_ms','admission_completion_ms','semantic_policy_setup_ms','matching_policy_setup_ms')}
            physical=row['completion']['physical'];item['owner_costs']=UNKNOWN if physical is None else physical.get('costs',UNKNOWN)
            receipts=row.get('sparse_receipts',[])
            item['projection_tcp_bytes']=sum(r['sent']['wire_bytes']+r['received']['wire_bytes'] for r in receipts) if receipts else UNKNOWN
            item['fresh_projection_rows']=sum(3*r['fresh_rows_per_role'] for r in receipts) if receipts else UNKNOWN
            item['cpu_lora_ms']=sum(r['received']['header']['metadata'].get('cpu_lora_ms',0.) for r in receipts) if receipts else UNKNOWN
            item['reused_projection_rows']=0 if physical is None else sum(len(r['reused_positions']) for r in physical['records'])
            prof=row.get('diagnostics');item['profile_components']=UNKNOWN if prof is None else prof['components']
            costs[k].append(item)
    timing_differences=[]
    for left,right in ((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2])):
        for index in range(2):
            differences={}
            for field in ('application_e2e_ms','application_ttft_ms','model_ttft_ms','prefill_ms','decode_ms'):
                x,y=costs[left][index][field],costs[right][index][field]
                differences[field]=y-x if isinstance(x,(float,int)) and isinstance(y,(float,int)) else UNKNOWN
            timing_differences.append(dict(reference=left,condition=right,request_id=records[right]['requests'][index]['request_id'],condition_minus_reference_ms=differences,scope='matched request diagnostic wall scopes; do not sum nested timers'))
    validations=sum(x['exact_context_windows'] for x in layers)+sum(x['reused_qkv_rows']//3 for x in layers)
    identity=records[CONDITIONS[2]]['config']['physical'];entry_bytes=3*3*identity['layers']*identity['hidden_size']*DTYPES[identity['dtype']][1]
    hotpath=dict(measurement=UNKNOWN,reconstructed_valid_calls=validations,reconstructed_entry_proof_tensor_serialization_bytes=validations*entry_bytes,
        scope='static reconstruction for original valid() implementation and selected exact-mode windows; NOT measured time/transfer traffic; additional evidence/protocol serialization excluded',provenance=provenance['source']['historical_source_verification'])
    hotpath['validation_mode']=records[CONDITIONS[2]]['results'][1].get('validation_mode','EAGER_FULL_PROOF')
    hotpath['reconstruction_scope']='original eager behavior only; not a scoped-mode measurement'
    target_profile=records[CONDITIONS[2]]['results'][1].get('diagnostics')
    if target_profile is not None:
        from .r5_subsequence_profile import validate_receipt
        validate_receipt(target_profile,records[CONDITIONS[2]]['requests'][1]['request_id'],records[CONDITIONS[2]]['results'][1]['completion']['physical'])
        hotpath['measurement']=[x for x in target_profile['components'] if x['phase']=='PREFILL' and x['component'] in ('owner_valid','entry_proof_all_layers','tensor_proof','tensor_serialization_host_transfer_inclusive','tensor_sha256')]
        hotpath['measurement_scope']='observed target PREFILL counters/wall; nested components overlap; Evidence.save internals are separate inclusive scopes'
    different=lambda name:any(not x['bitwise_equal'] for x in comparisons if x['comparison']==name and x['layer']==0)
    propagation=bool(layers[0]['reused_qkv_rows'] and all(x['observed_input_mismatch_count']==x['exact_context_windows'] and x['reused_qkv_rows']==0 for x in layers[1:]))
    hypotheses=dict(cached_value_divergence=dict(observed=different('cached_rows_A_vs_C'),causality='UNRESOLVED; source value preservation and dense-target equivalence are separate'),
        sparse_base_divergence=dict(observed=different('fresh_rows_A_vs_C_fresh_base'),projection_inputs_equal=not different('A_vs_C_input'),status='SUPPORTED_HYPOTHESIS' if different('fresh_rows_A_vs_C_fresh_base') and not different('A_vs_C_input') else 'UNRESOLVED',mechanism='shape/layout/backend/repeat variability not distinguished by one historical execution'),
        lora_delta_divergence=dict(observed=different('fresh_rows_A_vs_C_fresh_delta')),
        scatter_or_add_mismatch=dict(observed=any(not x['bitwise_equal'] for x in comparisons if x['comparison'] in ('source_cache_vs_C_scatter','fresh_add_cast_vs_C_scatter'))),
        propagated_input_rejection=dict(observed=propagation,status='SUPPORTED_HYPOTHESIS' if propagation else 'UNRESOLVED',
            preexisting_source_vs_dense_mismatch_windows=sum(len(x['source_vs_A_input_mismatch_windows']) for x in layers),
            additional_reuse_specific_rejection_windows=sum(len(x['reuse_specific_rejection_windows']) for x in layers),
            limitation='saved input inequality explains rejection; source/dense mismatch may predate reuse. Additional C-vs-A drift does not isolate the upstream attention/residual/normalization operation'),
        repeated_integrity_hashing=dict(status='SUPPORTED_HYPOTHESIS',time_attribution=UNKNOWN,scope='statically present; duration requires opt-in profiling'))
    metadata=[]
    for condition,data in records.items():
        for row in data['results']:
            for record in projections(row).values():
                for boundary in ('input','fresh_base','fresh_delta','output'):
                    spec=record.get(boundary)
                    if spec is not None:metadata.append(dict(condition=condition,request_id=row['request_id'],layer=record['layer'],role=record['role'],boundary=boundary,
                        recorded=spec,observed_stride=spec.get('observed_stride',spec.get('stride',UNKNOWN)),original_contiguous=spec.get('original_contiguous',UNKNOWN)))
    result=dict(schema='R5_SUBSEQUENCE_FORENSICS_V1',status='SEALED_HISTORICAL_FORENSICS_COMPLETE',ROOT_CAUSE='UNRESOLVED',comparisons=comparisons,layer_eligibility=layers,logits=fidelity,
        provenance=provenance,tensor_metadata=metadata,costs=costs,matched_timing_differences=timing_differences,hotpath=hotpath,availability=dict(per_layer_rejection_events=UNKNOWN if target_profile is None else 'AVAILABLE_IN_NEW_PROFILE',CUDA_kernel_timings=UNKNOWN,hashing_wall_time=UNKNOWN if target_profile is None else 'AVAILABLE_IN_NEW_PROFILE',attention_internal_boundaries=UNKNOWN,historical_strides=UNKNOWN,full_saved_projection_rows='AVAILABLE'),
        hypotheses=hypotheses,
        next_action='Profile unchanged strong validation before optimizing. Smallest arithmetic control: dense target QKV versus sparse rows on byte-identical layer-0 inputs and unchanged weights; capture base/delta separately. DIAGNOSTIC_ONLY, no speedup claim.',
        A0_break_even='NOT_ESTABLISHED',GPU_NOT_RUN=True)
    import json
    payload=json.dumps(result,sort_keys=True,indent=2).encode();require(len(payload)<=16*2**20,'diagnostic JSON exceeds unchanged16MiB reader bound')
    out=fresh(output_dir,(experiment,run,audit));(out/'subsequence_diagnostics.json').write_bytes(payload)
    text=['# Historical sparse QKV diagnosis','', 'ROOT_CAUSE=UNRESOLVED. Historical exported audit linked and Git blobs verified; no inference or new GPU measurement.',
          '',f'First observed target projection output difference: {first}', '', '## Layer 0 partitions','', '| Role | Comparison | Differing elements | Max abs | Relative L2 |','|---|---|---:|---:|---:|']
    names={'cached_rows_A_vs_C','fresh_rows_A_vs_C_fresh_base','fresh_rows_A_vs_C_fresh_delta','fresh_rows_A_vs_C_total','source_cache_vs_C_scatter','fresh_add_cast_vs_C_scatter'}
    text.extend(f"| {x['role']} | {x['comparison']} | {x['differing_values']}/{x['elements']} | {x['max_abs']} | {x['relative_l2']} |" for x in comparisons if x['layer']==0 and x['comparison'] in names)
    text+=['','## Layer eligibility','', '| Layer | Exact windows | Source/C mismatch | Source/A mismatch | C/A mismatch | Reused QKV rows |','|---:|---:|---:|---:|---:|---:|']
    text.extend(f"| {x['layer']} | {x['exact_context_windows']} | {x['observed_input_mismatch_count']} | {len(x['source_vs_A_input_mismatch_windows'])} | {len(x['C_vs_A_input_mismatch_windows'])} | {x['reused_qkv_rows']} |" for x in layers)
    text+=['','## Target wall timing','', '| Condition | App E2E ms | TTFT ms | Prefill ms | Reused projection rows |','|---|---:|---:|---:|---:|']
    text.extend(f"| {name} | {costs[name][1]['application_e2e_ms']} | {costs[name][1]['application_ttft_ms']} | {costs[name][1]['prefill_ms']} | {costs[name][1]['reused_projection_rows']} |" for name in CONDITIONS)
    text+=['','Cached and fresh row errors are reported independently in JSON; scatter/add checks are value replays. Saved operand differences do not identify a CUDA kernel cause.',
           '',f'Static integrity-work reconstruction (not measured duration): {hotpath}',
           '', 'Original A/B/C order, GPU warm state and single-run variability remain confounded. Profiled runs have additional observer cost; CUDA stream-span time is not kernel-only time. No fidelity restoration, A0 break-even or speedup claim.']
    (out/'subsequence_diagnosis.md').write_text('\n'.join(text)+'\n');seal(out)
    return dict(status=result['status'],ROOT_CAUSE=result['ROOT_CAUSE'],first_difference=first,layer_count=len(layers),output_dir=str(out))

def prepare_revalidation(experiment,run,audit,output_dir,source_commit,expected_source_commit=REFERENCE_COMMIT,cuda_events=False,validation_mode=None):
    p,records,export,provenance=history(experiment,run,audit,expected_source_commit)
    from .r5_i5r1_plan import source_identity
    from .r5_subsequence_profile import VERSION
    from .r5_i5r_parallel import output
    current=source_identity();require(current['commit']==source_commit and current['commit']!=p['source']['commit'],'reviewed NEW revalidation source differs')
    new=copy.deepcopy(p);new['source']=current;new['validation_test_sha256']=sha(ROOT/'tests/test_r5_subsequence_generation.py')
    # Do not carry an earlier D5 mode into an otherwise ordinary D1-D4 plan.
    require('integrity_validation' not in new or validation_mode is not None,'D5 input requires an explicit new validation mode')
    new['diagnostics']=dict(version=VERSION,cuda_events=cuda_events,historical_inputs=provenance,classification='DIAGNOSTIC_ONLY_UNCHANGED_ARITHMETIC',instrumentation_files=diagnostic_source_files(source_commit))
    if validation_mode is not None:
        from .r5_subsequence_integrity import MODES,SCHEMA,THREAT_MODEL
        require(validation_mode in MODES,'invalid validation mode')
        name='tests/test_r5_subsequence_validation.py';blob=subprocess.run(['git','show',source_commit+':'+name],cwd=ROOT,capture_output=True)
        require(blob.returncode==0 and digest_bytes(blob.stdout)==sha(ROOT/name),'commit reviewed D5 tests')
        new['integrity_validation']=dict(schema=SCHEMA,mode=validation_mode,threat_model=THREAT_MODEL,test_file=name,test_sha256=sha(ROOT/name))
        new['diagnostics']['instrumentation_files'][name]=sha(ROOT/name)
    out=output(output_dir,(experiment,run,audit,p['multi'],p['dataset']));write(out/'subsequence_plan.json',new);seal(out)
    return dict(status='SEALED_CPU_PREPARED_GPU_NOT_RUN',source=source_commit,workload='identical historical sealed tasks',validation_mode=validation_mode or 'EAGER_FULL_PROOF',output_dir=str(out))

def diagnostic_source_files(commit):
    names=('scripts/151_r5_subsequence_generation.py','scripts/152_r5_subsequence_diagnostics.py','jobs/r5_subsequence_generation.sh','tests/test_r5_subsequence_diagnostics.py')
    result={}
    for name in names:
        blob=subprocess.run(['git','show',commit+':'+name],cwd=ROOT,capture_output=True)
        require(blob.returncode==0 and digest_bytes(blob.stdout)==sha(ROOT/name),'commit reviewed diagnostic CLI/job/tests: '+name)
        result[name]=sha(ROOT/name)
    return result
