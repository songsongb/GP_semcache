"""Independent streaming value audit for one NATURAL A/C block; no model forward."""
from pathlib import Path
import json,math,tempfile
from .communication import require,digest
from .r3a_artifacts import unpack,check_parity,THRESHOLDS
from .r5_i5r2_tensors import Evidence
from .r5_subsequence_natural import load,approval,CONDITIONS
from .r5_subsequence_natural_io import verify as verify_tree,output,put,seal,load_result
from .r5_subsequence_audit import audit_request,replay_wire,replay_policy,request_costs
from .r5_sparse_qkv import SparseOrder
from .r4c_workload import termination


def ledger(ev,endpoint,kind,rid):
    prefix=endpoint+'/'+kind+'/'+rid+'/'
    records=[];size=0
    for name in sorted(n for n in ev.files if n.startswith(prefix)):
        if name.endswith('.jsonl'):
            size+=ev.files[name];require(size<=256*2**20,'request ledger exceeds256MiB');records.extend(ev.jsonl(name))
    # Wire sequence is per direction; pass names sort lexically COMPLETE before
    # s000, therefore use actual protocol sequence for wire pairing.
    if kind=='wire':records.sort(key=lambda r:(r['operation'],r['header']['sequence']))
    else:records.sort(key=lambda r:(len(r['context']['generated_token_ids']),r['context']['layer']) if 'context' in r else (1000,0))
    return records


def policy_counts_by_request(events,request_ids):
    """Attribute already replayed policy events to their existing request boundaries."""
    counts={};active=None;names=('INSERT','EVICT','CHU','PBR')
    for event in events:
        kind=event['kind']
        if kind=='MSTAGE_MATCH':
            rid=event['request_id'];require(active is None and rid in request_ids and rid not in counts,'policy request boundary differs')
            active=rid;counts[rid]={k:0 for k in names}
        elif kind in names:
            require(active is not None,'policy update outside request');counts[active][kind]+=1
        elif kind=='PHYSICAL_SUBSEQUENCE_COMPLETE':
            require(active is not None,'policy completion outside request');active=None
    require(active is None and list(counts)==request_ids,'incomplete/reordered request policy boundaries')
    return counts


def child(path,p,pr,block,tok,*,_conditions=None,_context_mode=None,_allow_approximate=False,_request_policy=False):
    root=Path(path);config=json.loads((root/'identity.json').read_bytes());status=json.loads((root/'status.json').read_bytes())
    require(status['status']=='COMPLETE' and config['source']==p['implementation']['source'] and config['plan_inventory']==pr,'failed/source/plan child differs')
    c=config['condition'];require(c in (CONDITIONS if _conditions is None else _conditions),'condition differs');sequence=json.loads((root/'sequence.json').read_bytes())
    require([r['request_id'] for r in sequence]==[t['request_id'] for t in block['tasks']],'original global order differs')
    require(all(a['completion_ns']<=b['arrival_ns'] for a,b in zip(sequence,sequence[1:])),'overlapping/noncompleted NATURAL requests')
    summaries={};identities={};costs=[];all_policy=[]
    from .r5_i5_quality import score
    for aid in sorted({t['adapter_id'] for t in block['tasks']}):
        ev=Evidence(root/aid,max_input_bytes=32*2**30,max_tensor_bytes=128*2**20);cfg=ev.json('identity.json');ev.registry=cfg['registry'];identities[aid]=cfg
        reqs=ev.json('requests.json');tasks=[t for t in block['tasks'] if t['adapter_id']==aid]
        require(len(reqs)==len(tasks) and cfg['condition']==c and cfg['source']==config['source'] and cfg['plan_inventory']==pr,'adapter source/condition differs')
        initial=ev.json('ud/initial.json');udconfig=ev.json('ud_config.json')
        require(ev.json('ud/status.json')['status']=='COMPLETE' and initial['identity']==cfg['physical'] and initial['source']==cfg['source'] and initial['session']==udconfig['session'] and initial['matrix_sha256']==cfg['physical']['adapter_matrix_fingerprint'] and not initial['user_device_cuda_used'] and not initial['full_model_loaded'],'UD isolation/identity differs')
        require(initial['backend']['cuda_initialized'] is False and initial['backend']['torch']==config['backend']['torch'] and initial['backend']['cpu_threads']==4,'UD backend/isolation differs')
        sources={};completed=set();order=SparseOrder(cfg['physical'],ev.registry)
        for req,task in zip(reqs,tasks):
            require(all(req[k]==task[k] for k in ('request_id','sample_id','adapter_id','user_id','token_ids','input_text','request_sequence_number')) and req['generation_policy']==task['policy'],'original dataset request/routing changed')
            rid=req['request_id'];r=load_result(ev.root/'es',rid)
            require(r['natural_condition']==c and r['condition']==('FULL_RECOMPUTE' if c=='FULL_RECOMPUTE' else 'SEMCACHE_SUBSEQUENCE') and r['context_mode']==(p['context_mode'] if _context_mode is None else _context_mode) and r['validation_mode']==p['validation_mode'],'execution mode differs')
            require((r['completion']['physical'] is None)==(c=='FULL_RECOMPUTE') and (r['completion']['impact'] is None)==(c=='FULL_RECOMPUTE') and (r['assignment'] is None)==(c=='FULL_RECOMPUTE'),'A performed cache work or C omitted impact/policy')
            if _context_mode is not None and r['completion']['physical'] is not None:
                phys=r['completion']['physical']
                require(phys['mode']==_context_mode and phys['integrity_validation']['validation_mode']==p['validation_mode'] and
                    all(x['mode']==_context_mode for x in phys['records']),'explicit pilot projection/integrity mode differs')
            ud=ledger(ev,'ud','layers',rid);wire=ledger(ev,'es','wire',rid);network=replay_wire(wire,ledger(ev,'ud','wire',rid))
            for row in ud:
                if 'context' in row:
                    require(rid not in completed and row['context']['request']['request_id']==rid,'wrong/finished UD parent');order.accept(row['context'])
                else:
                    control=row['control'];previous=order.parents.get(rid)
                    require(rid not in completed and order.layer==0 and previous is not None and previous[0]==req and control['request']==req and control['generated_token_ids'][:-1]==previous[1] and termination(req['generation_policy'],control['generated_token_ids'])==control['termination'],'UD completion differs');completed.add(rid)
            frames={(x['operation'],x['header']['sequence']):x for x in wire}
            require(len(frames)==len(wire),'duplicate wire sequence')
            reused=audit_request(ev,cfg,req,r,ud,frames,sources,completed,allow_approximate=_allow_approximate)
            require(r['generated_text']==tok.tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False) and r['quality']==(score(task['prepared']['sample'],r['generated_text'],task['prepared']['template_version']) if task['scored'] else None) and r['sample_id']==task['sample_id'] and r['scored']==task['scored'],'text/quality/sample differs')
            timing=ev.json('es/'+rid+'/timing.json');recorded=next(x for x in sequence if x['request_id']==rid)
            require(recorded==dict(adapter_id=aid,**timing) and timing['generation_scope_arrival_ns']==r['arrival_ns'] and timing['arrival_ns']<=r['arrival_ns'] and timing['completion_ns']>=r['completion_ns'] and timing['application_e2e_ms']==(timing['completion_ns']-timing['arrival_ns'])/1e6 and timing['application_ttft_ms']==(r['steps'][0]['available_ns']-timing['arrival_ns'])/1e6,'primary evidence-inclusive timing differs')
            cost=request_costs(r);costs.append(cost)
            phys=r['completion']['physical'];records=r['prefill_records'] if phys is None else phys['records']
            by_layer_role=[dict(layer=x['layer'],role=x['role'],fresh_rows=len(x.get('fresh_positions',list(range(len(req['token_ids']))))),reused_rows=len(x.get('reused_positions',[])),consumed_windows=[dict(source_request_id=a['source_request_id'],source_positions=a['source_positions'],target_positions=a['target_positions']) for a in x.get('consumed',[])]) for x in records]
            profile=r['diagnostics']
            summaries[rid]=dict(request_id=rid,adapter_id=aid,sample_id=task['sample_id'],timing=timing,generated_token_ids=r['generated_token_ids'],generated_text=r['generated_text'],quality=r['quality'],logits=r['logits'],logits_shape=r['logits_shape'],network=network,costs=cost,physical_hit=reused>0,reused_projection_rows=reused,by_layer_role=by_layer_role,cache_bytes=0 if phys is None else phys['physical_tensor_bytes'],matching=r['matching'],model_timings={k:r[k] for k in ('model_execution_e2e_ms','model_ttft_ms','prefill_ms','decode_ms','admission_completion_ms','semantic_policy_setup_ms','matching_policy_setup_ms','generated_tokens_per_second')},diagnostics=profile,steps=len(r['steps']))
            if req==reqs[0]:require(reused==0,'first adapter request consumed preloaded tensors')
        # Reuse unchanged M score/CHU replay, streaming one request at a time.
        policy_replay=replay_policy(ev,reqs,(load_result(ev.root/'es',req['request_id']) for req in reqs))
        policy=ev.jsonl('es/policy.jsonl',optional=True)
        if _request_policy:
            request_ids=[t['request_id'] for t in tasks]
            per_request=({rid:{k:0 for k in ('INSERT','EVICT','CHU','PBR')} for rid in request_ids}
                if c=='FULL_RECOMPUTE' else policy_counts_by_request(policy,request_ids))
            for rid in request_ids:summaries[rid]['cache_policy_counts']=per_request[rid]
        counts={k:sum(r.get('kind')==k for r in policy) for k in ('INSERT','EVICT','CHU','PBR')}
        require(counts['PBR']==0,'PBR cannot trigger within <=20 original adapter requests at interval100')
        all_policy.append(dict(adapter_id=aid,counts=counts,occupancy=policy_replay['occupancy']))
        replay_wire(ledger(ev,'es','wire','CONTROL'),ledger(ev,'ud','wire','CONTROL'))
    return dict(config=config,identities=identities,requests=summaries,policy=all_policy,sequence=sequence)


def analyze(experiment,run,cpu_audit,output_dir):
    p,pr,rest,ar=approval(experiment,cpu_audit);block,m,pm,specs,users,tok=rest
    inventory=verify_tree(run);status=json.loads((Path(run)/'status.json').read_bytes())
    require(status['status']=='COMPLETE' and status['source']==p['implementation']['source'] and status['plan_inventory']==pr and status['cpu_audit_inventory']==ar and status['cpu_audit_path']==str(Path(cpu_audit).resolve()),'incomplete/unapproved run')
    require([r['condition'] for r in status['children']]==p['condition_order'] and all(r['returncode']==0 for r in status['children']),'condition lifecycle differs')
    reports={c:child(Path(run)/c,p,pr,block,tok) for c in CONDITIONS}
    a,c=[reports[k] for k in CONDITIONS]
    require(a['config']['gpu']['uuid']==c['config']['gpu']['uuid'] and a['identities'].keys()==c['identities'].keys(),'A/C GPU/adapter set differs')
    from .r5_i5_gpu import verify_receipt
    for report in (a,c):verify_receipt(report['config']['gpu_identity_receipt'],pm,report['config']['gpu'],'SAME_COMPATIBLE_GPU_CLASS',fixture=False)
    require(a['config']['backend']==c['config']['backend'],'A/C effective precision/backend differs')
    for aid in a['identities']:
        require(all(a['identities'][aid][k]==c['identities'][aid][k] for k in ('physical','contract','model_identity','registry')),'A/C native identity/backend contract differs')
        for cfg in (a['identities'][aid],c['identities'][aid]):
            from .r5_multi_artifacts import physical,runtime_view
            native=dict(cfg['contract']);native.pop('r5_multi')
            expected,contract=physical(runtime_view(pm,specs[aid]),p['tokenizer_identity'],cfg['physical']['device'],cfg['physical']['gpu_uuid'],native,specs[aid],users)
            require(cfg['physical']==expected and cfg['contract']==contract and cfg['model_identity']==runtime_view(pm,specs[aid])['identity'] and cfg['registry']==users,'prepared physical model/adapter/registry binding differs')
        ident=a['identities'][aid]['physical'];require(ident['adapter_matrix_fingerprint']==specs[aid]['matrix_sha256'] and ident['base_fingerprint']==pm['identity']['base_weight_sha256'] and ident['tokenizer']==p['tokenizer_identity'],'actual model/adapter fingerprints differ')
    comparisons=[]
    for t in block['tasks']:
        rid=t['request_id'];arow=a['requests'][rid];crow=c['requests'][rid];width=arow['logits_shape'][-1]
        av=unpack(Evidence(Path(run)/CONDITIONS[0]/t['adapter_id'],max_input_bytes=32*2**30).tensor('es',arow['logits']),'float32')
        cv=unpack(Evidence(Path(run)/CONDITIONS[1]/t['adapter_id'],max_input_bytes=32*2**30).tensor('es',crow['logits']),'float32')
        gates=[check_parity(cv[i:i+width],av[i:i+width],[1,1,width]) for i in range(0,min(len(av),len(cv)),width)]
        same_length=len(av)==len(cv);tokens=arow['generated_token_ids']==crow['generated_token_ids'];text=arow['generated_text']==crow['generated_text']
        histories=[arow['generated_token_ids'][:i]==crow['generated_token_ids'][:i] for i in range(len(gates))]
        comparisons.append(dict(request_id=rid,numerical_fidelity='PASS' if same_length and gates and all(histories) and all(g['passed'] for g in gates) else 'FAIL',full_vocabulary_per_step=gates,per_step_input_histories_equal=histories,generation_length_equal=same_length,token_equal=tokens,text_equal=text))
    totals={k:sum(r['timing']['application_e2e_ms'] for r in reports[k]['requests'].values()) for k in CONDITIONS};delta=totals[CONDITIONS[1]]-totals[CONDITIONS[0]]
    numerical=all(r['numerical_fidelity']=='PASS' for r in comparisons)
    result=dict(status='INDEPENDENT_EVIDENCE_AUDIT_PASS',plan_id=p['plan_id'],plan_inventory=pr,run_inventory=inventory,cpu_audit_inventory=ar,thresholds=THRESHOLDS,
        reports=reports,comparisons=comparisons,Total_E2E_ms=totals,Delta_E2E_ms=delta,Delta_E2E_percent=100*delta/totals[CONDITIONS[0]],
        physical_hit_requests=sum(r['physical_hit'] for r in c['requests'].values()),reused_qkv_projection_rows=sum(r['reused_projection_rows'] for r in c['requests'].values()),
        numerical_fidelity='PASS' if numerical else 'FAIL',CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',ROOT_CAUSE='UNRESOLVED',
        phase3='PENDING_EXPLICIT_REVIEW_APPROVAL',measurement='sum evidence-inclusive request intervals; init/warmup0/process lifecycle/final seals separate; instrumented diagnostic, one fixed-order block, no statistical speedup claim',
        request_sequence_span_ms={k:(reports[k]['sequence'][-1]['completion_ns']-reports[k]['sequence'][0]['arrival_ns'])/1e6 for k in CONDITIONS},process_lifecycle=status['children'],policy='actual D5 F/A/I/S events independently replayed; source cost is original NATURAL requests, no constructed source; PBR interval100 cannot trigger',independent_contexts=len({digest(t['prepared']['sample']['context']) for t in block['tasks']}))
    # Summaries may be large: one report per condition/request, small index.
    out=output(output_dir,(experiment,run,cpu_audit));report_index={}
    for label,report in result.pop('reports').items():
        req_reports=report.pop('requests');names={}
        for rid,row in req_reports.items():name=label+'/'+rid+'.json';put(out/name,row);names[rid]=name
        report['request_reports']=names;name=label+'/condition.json';put(out/name,report);report_index[label]=name
    result['condition_reports']=report_index;put(out/'natural_audit.json',result)
    (out/'natural_report.md').write_text(f"# D5 NATURAL one-block audit\n\nEvidence: PASS. Physical HIT requests: {result['physical_hit_requests']}; reused QKV projection rows: {result['reused_qkv_projection_rows']}.\n\nTotal A E2E: {totals[CONDITIONS[0]]:.3f} ms; C: {totals[CONDITIONS[1]]:.3f} ms. Delta (C−A): {delta:.3f} ms ({result['Delta_E2E_percent']:.3f}%). Frozen numerical fidelity: {result['numerical_fidelity']}.\n\nFull attention/FFN/head always execute. Strict capture and wall profiling overhead included; initialization and final seals separate. Zero HIT is a valid negative outcome. Historical exact-prefix timings are separate references. No correctness-qualified speedup, root cause, production benefit or long-session claim. Phase3 requires explicit review.\n")
    seal(out);return {k:result[k] for k in ('status','physical_hit_requests','reused_qkv_projection_rows','Delta_E2E_ms','Delta_E2E_percent','numerical_fidelity')}


def verify(experiment,run,cpu_audit,audit,output_dir):
    old=verify_tree(audit)
    with tempfile.TemporaryDirectory() as tmp:
        fresh=Path(tmp)/'audit';analyze(experiment,run,cpu_audit,fresh)
        require(verify_tree(fresh)==old,'independent recomputation differs')
    out=output(output_dir,(experiment,run,cpu_audit,audit));put(out/'natural_verification.json',dict(status='PASS',audit_inventory=old));seal(out);return dict(status='PASS')
