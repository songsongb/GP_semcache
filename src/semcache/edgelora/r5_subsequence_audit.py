"""Independent sealed sparse-generation value replay. CPU only; no inference."""
from pathlib import Path
import struct,math
from .communication import require,digest,digest_bytes
from .r3a_artifacts import unpack,pack,check_parity,THRESHOLDS,write
from .r3a_tiny import f32
from .r5_i5r2_tensors import Evidence,DTYPES,safe_path,metrics
from .r5_i5_contracts import validate_request
from .r5_multi_artifacts import sharing
from .r5_sparse_qkv import SparseOrder,envelope,audit_sparse_receipts
from .r4c_workload import termination

def replay_wire(es,ud):
    require(es and ud and all(r['complete'] for r in es+ud),'missing/incomplete wire frames')
    def frames(rs,op):return [(r['frame_sha256'],r['wire_bytes'],r['header']) for r in rs if r['operation']==op]
    require(frames(es,'send')==frames(ud,'recv') and frames(es,'recv')==frames(ud,'send'),'ES/UD frame payload/sequence identity differs')
    return dict(es_to_ud_wire_bytes=sum(r['wire_bytes'] for r in es if r['operation']=='send'),ud_to_es_wire_bytes=sum(r['wire_bytes'] for r in es if r['operation']=='recv'),frames=len(es))

def request_costs(result):
    """Actual per-pass receipts; projection row counts sum layers and roles."""
    receipts=result['sparse_receipts'];physical=result['completion']['physical']
    fresh=sum(3*r['fresh_rows_per_role'] for r in receipts);reused=sum(3*r['skipped_rows_per_role'] for r in receipts)
    if physical is not None:require(reused==physical['physical_projection_rows_reused'],'projection reuse summary differs')
    return dict(request_id=result['request_id'],native_projection_rows=fresh,cpu_lora_projection_rows=fresh,physically_reused_projection_rows=reused,
        es_to_ud_projection_wire_bytes=sum(r['sent']['wire_bytes'] for r in receipts),ud_to_es_projection_wire_bytes=sum(r['received']['wire_bytes'] for r in receipts),
        projection_frame_count=2*len(receipts),cpu_lora_ms=sum(r['received']['header']['metadata'].get('cpu_lora_ms',0.) for r in receipts),
        owner_costs=None if physical is None else physical['costs'],integrity_validation=None if physical is None else {k:v for k,v in physical.get('integrity_validation',{}).items() if k!='authorizations'},scope='actual prefill+decode projection frames; completion controls in condition wire totals; summed rows are not unique tokens')

def rows(data,shape,dtype,positions):
    require(len(shape)==3 and shape[0]==1,'batch-one projection required');stride=shape[2]*DTYPES[dtype][1]
    require(all(type(p) is int and 0<=p<shape[1] for p in positions),'invalid absolute tensor slice')
    return b''.join(data[p*stride:(p+1)*stride] for p in positions)

def replay_projection(ev,record,ud,req,sources):
    """Verify actual gather/CPU payload/add/scatter and saved source consumption."""
    inp,out=record['input'],record['output'];x=ev.tensor('es',inp);y=ev.tensor('es',out);n=inp['shape'][1];h=inp['shape'][2]
    require(inp['shape']==out['shape'] and h==req['physical_identity']['hidden_size'] and inp['dtype']==out['dtype']==req['physical_identity']['dtype'],'projection shape/dtype differs')
    fresh=ud['context']['fresh_positions'];fresh_data=rows(x,inp['shape'],inp['dtype'],fresh)
    require(record.get('fresh_positions',fresh)==fresh,'fresh positions differ')
    output_stride=h*DTYPES[out['dtype']][1];used=set(fresh)
    if fresh:
        require(record['fresh_base']['shape']==record['fresh_delta']['shape']==[1,len(fresh),h],'computed more than fresh rows')
        hidden=safe_path(ev.root,'ud/'+ud['hidden_file']).read_bytes();delta=safe_path(ev.root,'ud/'+ud['delta_file']).read_bytes()
        require(hidden==fresh_data and digest_bytes(hidden)==ud['hidden_sha256'] and digest_bytes(delta)==ud['delta_sha256'],'CPU TCP input/output payload differs')
        base=ev.tensor('es',record['fresh_base']);received=ev.tensor('es',record['fresh_delta']);count=len(fresh)*h;role='qkv'.index(record['role'])
        require(received==delta[role*count*4:(role+1)*count*4],'CPU delta consumed differs from transmitted delta')
        values=[f32(a+b) for a,b in zip(unpack(base,record['fresh_base']['dtype']),unpack(received,'float32'))]
        require(pack(values,out['dtype'])==rows(y,out['shape'],out['dtype'],fresh),'fresh total-QKV construction differs')
    for consumed in record.get('consumed',[]):
        sid=consumed['source_request_id'];require(sid in sources,'uncompleted/unknown tensor source');source_req,source_records=sources[sid]
        require(sharing(ev.registry,source_req['user_id'],req['user_id'],source_req['adapter_id'],req['adapter_id']) is None and source_req['physical_identity']==req['physical_identity'],'unauthorized physical source')
        a,b=consumed['source_positions'];c,d=consumed['target_positions'];require(b-a==d-c and 0<=a<b<=len(source_req['token_ids']) and 0<=c<d<=len(req['token_ids']),'source/target positions differ')
        require(source_req['token_ids'][a:b]==req['token_ids'][c:d],'source/target token IDs differ')
        original=source_records[record['layer'],record['role']];source_bytes=rows(ev.tensor('es',original['output']),original['output']['shape'],original['output']['dtype'],list(range(a,b)))
        cached=ev.tensor('es',consumed['tensor']);require(cached==source_bytes==rows(y,out['shape'],out['dtype'],list(range(c,d))),'cached/scattered tensor values differ')
        if record['mode']=='EXACT_CONTEXT_VERIFIED':
            require((a,b)==(c,d) and source_req['token_ids'][:b]==req['token_ids'][:d],'exact context claim differs')
            require(rows(ev.tensor('es',original['input']),original['input']['shape'],original['input']['dtype'],list(range(a,b)))==rows(x,inp['shape'],inp['dtype'],list(range(c,d))),'exact per-layer input differs')
        require(not used&set(range(c,d)),'overlapping physical scatter');used.update(range(c,d))
    require(used==set(range(n)),'incomplete physical output coverage')
    return sum(c['target_positions'][1]-c['target_positions'][0] for c in record.get('consumed',[]))

def replay_validation(ev,req,result,sources,*,allow_approximate=False):
    """D5 verify admission fingerprints from completed source bytes, not counters."""
    from .r5_subsequence_integrity import SCHEMA,EAGER,SCOPED,THREAT_MODEL
    physical=result['completion']['physical']
    if physical is None:return
    approximate=allow_approximate and physical['mode']=='APPROXIMATE_SUBSEQUENCE_TOTAL_QKV'
    if allow_approximate:require(approximate and result['context_mode']==physical['mode'],'explicit approximate audit mode differs')
    report=physical.get('integrity_validation')
    if allow_approximate:require(report is not None,'new approximate evidence requires integrity validation')
    if report is None:return  # Original sealed evidence predates D5.
    require(report['schema']==SCHEMA and report['validation_mode'] in (EAGER,SCOPED) and report['request_sha256']==req['request_sha256'] and report['request_id']==req['request_id'],'validation receipt/request differs')
    require(report['records_expired'] and report['integrity_check_failures']==0 and report['threat_model']==THREAT_MODEL,'failed/live/unknown validation scope')
    expected_keys=set();exact_windows=0
    if result['condition']=='SEMCACHE_SUBSEQUENCE':
        selected={tuple(x) for x in result['matching']['selected_spans']}
        for candidate in result['matching']['candidates']:
            start,end=candidate['target_positions'];sid=candidate['source_request_id']
            if (start,end) not in selected:continue
            require(sid in sources,'selected validation source incomplete')
            source=sources[sid][0]
            if (approximate or list(candidate['source_positions'])==[start,end] and source['token_ids'][:end]==req['token_ids'][:end]) and sharing(ev.registry,source['user_id'],req['user_id'],source['adapter_id'],req['adapter_id']) is None:
                exact_windows+=1;expected_keys.add(digest([result['assignment']['cluster_id'],req['token_ids'][start:end]]))
    for key,value in report.items():
        if key.endswith(('_count','_entries','_bytes','_ms')):
            require(isinstance(value,(int,float)) and math.isfinite(value) and value>=0,'invalid integrity count/timing')
    authorized={};auth_bytes=0
    for auth in report['authorizations']:
        original=dict(auth);aid=original.pop('authorization_id');require(digest(original)==aid and aid not in authorized,'authorization digest/uniqueness differs')
        require(auth['request_sha256']==req['request_sha256'] and auth['request_epoch']==report['request_epoch'],'stale request authorization')
        h=auth['header'];m=h['metadata'];sid=m['source_request_id'];require(sid in sources,'authorization references incomplete/future source')
        sr,records=sources[sid];lo,hi=h['positions'];positions=list(range(lo,hi));key=auth['key']
        require(hi-lo==3 and 0<=lo<hi<=len(sr['token_ids']) and list(h['tokens'])==sr['token_ids'][lo:hi] and list(key[1])==list(h['tokens']) and key[0]==result['assignment']['cluster_id'],'authorization window/key differs')
        require(sharing(ev.registry,sr['user_id'],req['user_id'],sr['adapter_id'],req['adapter_id']) is None and sr['physical_identity']==req['physical_identity'],'authorization source ownership differs')
        layers=req['physical_identity']['layers'];hidden=req['physical_identity']['hidden_size'];dtype=req['physical_identity']['dtype'];tag=auth['tensors']['0'][0]['dtype']
        require(tag in (dtype,'torch.'+dtype),'authorization tensor dtype differs')
        inputs={};tensors={}
        for layer in range(layers):
            rec=records[layer,'q'];data=rows(ev.tensor('es',rec['input']),rec['input']['shape'],dtype,positions)
            inputs[str(layer)]=dict(shape=[1,3,hidden],dtype=tag,sha256=digest_bytes(data));tensors[str(layer)]=[]
            for role in 'qkv':
                spec=records[layer,role]['output'];data=rows(ev.tensor('es',spec),spec['shape'],dtype,positions)
                tensors[str(layer)].append(dict(shape=[1,3,hidden],dtype=tag,sha256=digest_bytes(data)))
        metadata=dict(component_scope='total_qkv',completed=True,provenance_mode=physical['mode'],physical_identity=sr['physical_identity'],source_user=sr['user_id'],source_adapter=sr['adapter_id'],source_request_id=sid,source_token_ids=sr['token_ids'],input_proofs=inputs)
        size=3*3*layers*hidden*DTYPES[dtype][1];expected=dict(metadata=metadata,positions=[lo,hi],tokens=sr['token_ids'][lo:hi],size=size)
        require(digest(h)==digest(expected) and auth['tensors']==tensors and auth['admission_proof']==digest(dict(expected,tensors=tensors)),'authorization does not reproduce source admission strong proof')
        auth_bytes+=size;authorized[aid]=auth
    consumed=0;consumed_bytes=0
    for record in physical['records']:
        for item in record['consumed']:
            if report['validation_mode']==EAGER:
                require('integrity_validation' not in item,'eager path silently changed validation semantics');continue
            check=item['integrity_validation'];aid=check['authorization_id'];require(aid in authorized,'unauthenticated consumed tensor')
            auth=authorized[aid];expected=auth['tensors'][str(record['layer'])]['qkv'.index(record['role'])]
            require(check['private_snapshot'] is True and check['expected']==check['actual']==expected and digest(auth['key'])==digest(item['key']) and list(auth['header']['positions'])==list(item['source_positions']),'consumption snapshot/source differs')
            spec=item['tensor'];require(spec['sha256']==expected['sha256'] and spec['shape']==expected['shape'] and spec['dtype']==expected['dtype'].removeprefix('torch.'),'consumed evidence differs from authenticated snapshot')
            consumed+=1;consumed_bytes+=math.prod(spec['shape'])*DTYPES[spec['dtype']][1]
    require(report['request_validation_wall_ms']==sum(report[k] for k in ('owner_valid_wall_ms','consumed_operation_wall_ms','request_scope_setup_wall_ms','request_scope_cleanup_wall_ms')),'validation wall accounting differs')
    if report['validation_mode']==SCOPED:
        require({digest(a['key']) for a in authorized.values()}==expected_keys,'selected-entry authorization coverage differs')
        require(report['request_authorized_entries']==report['full_entry_proof_count']==len(authorized) and report['full_entry_serialized_bytes']==auth_bytes,'request full-proof accounting differs')
        require(report['consumed_tensor_proof_count']==consumed and report['consumed_tensor_serialized_bytes']==consumed_bytes,'consumed-proof accounting differs')
    else:
        require(not authorized and report['request_authorized_entries']==report['consumed_tensor_proof_count']==report['consumed_tensor_serialized_bytes']==0,'eager receipt has scoped authorizations')
        expected_full=exact_windows*req['physical_identity']['layers']+sum(len(r['consumed']) for r in physical['records'])
        size=3*3*req['physical_identity']['layers']*req['physical_identity']['hidden_size']*DTYPES[req['physical_identity']['dtype']][1]
        require(report['full_entry_proof_count']==expected_full and report['full_entry_serialized_bytes']==expected_full*size,'eager original full-proof accounting differs')

def audit_request(ev,config,req,result,ud,wire_frames,sources,completed,*,allow_approximate=False):
    reused=0
    validate_request(req);require(req['physical_identity']==config['physical'] and req['request_id']==result['request_id'] and req['request_id'] in completed,'request identity/completion differs')
    if 'diagnostics' in result:
        from .r5_subsequence_profile import validate_receipt
        validate_receipt(result['diagnostics'],req['request_id'],result['completion']['physical'])
        if 'integrity_validation' in result['diagnostics']:
            physical=result['completion']['physical'];expected=None if physical is None else {k:v for k,v in physical['integrity_validation'].items() if k!='authorizations'}
            require(result['diagnostics']['integrity_validation']==expected and result['diagnostics']['validation_mode']==result['validation_mode'],'profile/physical validation summary differs')
    step_receipts=[receipt for step in result['steps'] for receipt in step['record']['sparse']]
    require(step_receipts==result['sparse_receipts'],'aggregate sparse receipts differ')
    for receipt in step_receipts:
        for name in ('sent','received'):
            frame=receipt[name]
            require(wire_frames.get((frame['operation'],frame['header']['sequence']))==frame,'claimed projection receipt differs from actual wire evidence')
    values=unpack(ev.tensor('es',result['logits']),'float32');width=result['logits_shape'][-1]
    if not req['TEST_FIXTURE_ONLY']:require(width==50272,'missing full OPT vocabulary')
    require(len(values)==len(result['steps'])*width and len(result['steps'])==len(result['generated_token_ids']),'full-vocabulary logits incomplete')
    for i,step in enumerate(result['steps']):
        logits=values[i*width:(i+1)*width];require(step['selected_token_id']==max(range(width),key=logits.__getitem__)==result['generated_token_ids'][i] and digest_bytes(pack(logits,'float32'))==step['logits_sha256'],'generation/logit selection differs')
        obs=step['record']['position_observation'];past_length=len(req['token_ids'])+i-1 if i else 0;input_length=1 if i else len(req['token_ids'])
        require(obs['lookup_output_equal'] and obs['positions']==list(range(past_length,past_length+input_length)) and obs['past_length']==past_length,'actual native positional lookup differs')
        require(step['record']['input_past']==(None if not i else result['steps'][i-1]['record']['past']) and step['record']['past']['sequence_length']==past_length+input_length,'native target KV continuation differs')
        if i==0:
            phys=result['completion']['physical'];records=result['prefill_records'] if phys is None else phys['records']
            if phys is not None:
                pass_ud=[r for r in ud if r.get('context',{}).get('pass_id')==req['request_id']+'_s000']
                replayed=audit_sparse_receipts(records,step['record']['sparse'],pass_ud)
                require(phys['physical_projection_rows_reused']==replayed['reused_projection_rows'] and phys['kv_prefix_reuse'] is False,'false physical reuse/forward-skip count')
        else:records=step['record']['fresh_projection_records']
        require(len(records)==3*req['physical_identity']['layers'] and len({(r['layer'],r['role']) for r in records})==len(records),'incomplete projection evidence')
        for record in records:
            matches=[r for r in ud if r.get('context',{}).get('pass_id')==req['request_id']+f'_s{i:03d}' and r['context']['layer']==record['layer']]
            require(len(matches)==1,'UD projection count differs');frame=matches[0]
            require(frame['context']==envelope(req,record['layer'],frame['context']['fresh_positions'],frame['context']['pass_id'],result['generated_token_ids'][:i]),'decode position/history differs')
            reused+=replay_projection(ev,record,frame,req,sources)
        if i==0:prefill_records={(r['layer'],r['role']):r for r in records}
    require(result['completion']['control']['generated_token_ids']==result['generated_token_ids'] and result['completion']['control']['request']==req and result['termination']==result['completion']['control']['termination'],'completed generation differs')
    impact=result['completion']['impact']
    if impact is not None:
        require(len(impact['records'])==req['physical_identity']['layers'],'missing actual impact layers')
        require([r['layer'] for r in impact['records']]==list(range(req['physical_identity']['layers'])),'impact layer order differs')
        # Strict independent use of original reducer over saved real matrices.
        from .r3a_artifacts import tensor_from_bytes
        from semcache.cache.metric_manager import actual_attention_impact
        for record in impact['records']:
            spec=record['probabilities'];prob=tensor_from_bytes(ev.tensor('es',spec),spec['dtype'],spec['shape'])
            import torch
            n=len(req['token_ids']);require(tuple(prob.shape[2:])==(n,n) and bool(((prob>=0)&(prob<=1)).all()) and bool((prob.triu(1)==0).all()),'invalid causal probabilities')
            prob=prob.double()  # One bounded conversion per layer, same as capture.
            require(set(map(int,record['span_values']))==set(range(n-2)),'missing impact windows')
            for start,value in record['span_values'].items():require(math.isclose(actual_attention_impact([prob],int(start),int(start)+3),value,rel_tol=1e-12,abs_tol=1e-12),'attention impact replay differs')
    replay_validation(ev,req,result,sources,allow_approximate=allow_approximate)
    sources[req['request_id']]=(req,prefill_records)
    return reused

def condition(path):
    ev=Evidence(path,max_input_bytes=128*2**30,max_tensor_bytes=128*2**20);inventory=ev.verify()
    require(ev.json('status.json')['status']=='COMPLETE','failed condition')
    config=ev.json('identity.json');ev.registry=config['registry'];reqs=ev.json('requests.json');results=ev.json('results.json')
    wire=ev.jsonl('es/wire.jsonl');network=replay_wire(wire,ev.jsonl('ud/wire.jsonl'))
    wire_frames={(r['operation'],r['header']['sequence']):r for r in wire}
    initial=ev.json('ud/initial.json');require(initial['identity']==config['physical'] and initial['source']==config['source'] and not initial['user_device_cuda_used'] and not initial['full_model_loaded'] and initial['matrix_sha256']==config['physical']['adapter_matrix_fingerprint'],'UD isolation/identity differs')
    require(len(reqs)==len(results)==2,'bounded source/target pilot required')
    ud=ev.jsonl('ud/layers.jsonl');sources={};reused=0;order=SparseOrder(config['physical'],ev.registry);completed=set()
    for r in ud:
        if 'context' in r:
            require(r['context']['request']['request_id'] not in completed,'projection after completion');order.accept(r['context'])
        else:
            control=r['control'];req=control['request'];previous=order.parents.get(req['request_id'])
            require(req['request_id'] not in completed and order.layer==0 and previous is not None and previous[0]==req and control['generated_token_ids'][:-1]==previous[1] and termination(req['generation_policy'],control['generated_token_ids'])==control['termination'],'UD completion evidence differs');completed.add(req['request_id'])
    for req,result in zip(reqs,results):
        reused+=audit_request(ev,config,req,result,ud,wire_frames,sources,completed)
    require(results[0]['completion_ns']<=results[1]['arrival_ns'],'requests overlap or source not completed first')
    replay_policy(ev,reqs,results)
    return dict(inventory=inventory,config=config,results=results,reused_projection_rows=reused,requests=reqs,network=network)

def replay_policy(ev,reqs,results):
    from .r5_mstage_bridge import BoundedCache
    from semcache.cache.cache_entry import CacheEntry
    from semcache.cache.metric_manager import CacheMetricManager
    from semcache.semantic.subsequence import SubsequenceExtractor
    cache=BoundedCache(512*2**20);metrics=CacheMetricManager(cache);expected=[];occupancy=[]
    for req,result in zip(reqs,results):
        impact=result['completion']['impact']
        if impact is None:continue
        require(result['assignment']['cluster_id']==0,'one-cluster pilot differs')
        windows=SubsequenceExtractor(3).extract(req['token_ids']);keys=[(0,w.token_ids) for w in windows];metrics.arrive(keys)
        values={w.start:sum(layer['span_values'][str(w.start)] for layer in impact['records']) for w in windows}
        grouped={}
        for w,key in zip(windows,keys):grouped.setdefault(key,[]).append(values[w.start])
        metrics.history.append(0,req['request_id'],cache.now,{k:sum(v)/len(v) for k,v in grouped.items()})
        executed={}
        for record in result['completion']['physical']['records']:
            for c in record['consumed']:executed[(tuple([c['key'][0],tuple(c['key'][1])]),c['target_positions'][0])]=c
        for (key,start),c in sorted(executed.items(),key=lambda item:item[0][1]):
            require(key in cache.entries,'consumed nonresident policy entry');entry=cache.entries[key]
            old_f,old_i=entry.frequency,entry.impact;metrics.reused(entry,values[start])
            expected.append(dict(kind='CHU',key=key,old_F=old_f,F=entry.frequency,old_I=old_i,I=entry.impact,actual_impact=values[start]))
        seen=set()
        for w,key in zip(windows,keys):
            if key in seen or key in cache.entries:continue
            seen.add(key);size=3*3*req['physical_identity']['layers']*req['physical_identity']['hidden_size']*DTYPES[req['physical_identity']['dtype']][1]
            entry=CacheEntry(0,w.token_ids,(w.start,w.end),size,impact=values[w.start])
            cache.insert(entry,metrics.frequencies[key],metrics.frequencies,on_event=lambda kind,e,score:expected.append(dict(kind=kind,key=e.key,score=score,component_scope='physical_total_qkv')))
        require(cache.logical_cache_bytes<=512*2**20 and len(cache.entries)<=64,'policy capacity differs')
        require(result['completion']['physical'].get('physical_tensor_bytes',cache.logical_cache_bytes)==cache.logical_cache_bytes,'physical/policy occupancy differs')
        occupancy.append(dict(request_id=req['request_id'],entries=len(cache.entries),bytes=cache.logical_cache_bytes))
    actual=[r for r in ev.jsonl('es/policy.jsonl',optional=True) if r.get('kind') in ('CHU','INSERT','EVICT','PBR')]
    require(digest(expected)==digest(actual),'independent F/A/I/S/CHU replay differs')
    return dict(occupancy=occupancy,events=actual)

def analyze(experiment,run,output_dir):
    from .r5_subsequence_pilot import load,CONDITIONS
    from .r5_i5r_parallel import output
    from .r5_i5_evidence import seal
    p,pr=load(experiment);root=Evidence(run,max_input_bytes=128*2**30);rr=root.verify();status=root.json('status.json')
    require(status['status']=='COMPLETE' and status['source']==p['source'] and status['plan_inventory']==pr,'incomplete/mismatched pilot')
    ce=Evidence(status['cpu_validation_path']);require(ce.verify()==status['cpu_validation'],'CPU approval inventory differs')
    ca=ce.json('cpu_validation.json');require(ca['source']==p['source'] and ca['plan_inventory']==pr and ca['status']=='PASS' and ca['failed']==ca['skipped']==0,'CPU approval differs')
    from .r5_i5r_parallel import assets
    from .r5_i5_quality import score
    _,_,_,_,_,tokenizer=assets(p['registry_path'],p['multi'])
    reports={c:condition(Path(run)/c) for c in CONDITIONS}
    require(len({r['config']['physical']['gpu_uuid'] for r in reports.values()})==1,'unmatched actual GPU')
    baseline=reports['FULL_RECOMPUTE'];comparisons=[];prefill_diagnostics=[]
    for c,r in reports.items():
        require(r['config']['source']==p['source'] and r['config']['plan_inventory']==pr,'source/plan differs')
        require(r['config']['condition']==c and all(row['condition']==c and row['context_mode']==p['mode'] for row in r['results']),'condition/context-mode evidence differs')
        if 'integrity_validation' in p:
            expected=p['integrity_validation']['mode'];require(r['config'].get('validation_mode')==expected and all(row.get('validation_mode')==expected for row in r['results']),'sealed validation mode differs')
            require(all(row['completion']['physical'] is None or row['completion']['physical']['integrity_validation']['validation_mode']==expected for row in r['results']),'physical validation mode differs')
        require(r['config']['physical']==baseline['config']['physical'],'model/adapter/position implementation differs across conditions')
        if c=='FULL_RECOMPUTE':require(all(row['completion']['physical'] is None and row['completion']['impact'] is None and row['assignment'] is None for row in r['results']),'Full Recompute performed semantic/cache work')
        else:require(all(row['completion']['physical'] is not None and row['completion']['impact'] is not None for row in r['results']),'missing semantic/policy/impact path')
        if c=='SEMANTIC_NO_REUSE':require(r['reused_projection_rows']==0,'shadow baseline consumed physical tensors')
        for req,task in zip(r['requests'],p['tasks']):require(all(req[k]==task[k] for k in ('request_id','sample_id','adapter_id','user_id','token_ids','input_text')) and req['generation_policy']==task['policy'],'dataset request/order differs')
        for row,task in zip(r['results'],p['tasks']):
            require(row['generated_text']==tokenizer.tokenizer.decode(row['generated_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),'generated text differs')
            expected=score(task['prepared']['sample'],row['generated_text'],task['prepared']['template_version']) if task['scored'] else None
            require(row['quality']==expected and row['sample_id']==task['sample_id'] and row['scored']==task['scored'],'quality/sample evidence differs')
        for a,b in zip(baseline['results'],r['results']):
            ae=Evidence(Path(run)/'FULL_RECOMPUTE',max_input_bytes=128*2**30);be=Evidence(Path(run)/c,max_input_bytes=128*2**30)
            av=unpack(ae.tensor('es',a['logits']),'float32');bv=unpack(be.tensor('es',b['logits']),'float32');n=min(len(av),len(bv));width=a['logits_shape'][-1]
            gate=check_parity(bv[:n],av[:n],[1,n//width,width]);equal_tokens=a['generated_token_ids']==b['generated_token_ids']
            gate['per_step']=[check_parity(bv[i:i+width],av[i:i+width],[1,1,width]) for i in range(0,n,width)]
            gate['passed']=all(x['passed'] for x in gate['per_step']) and len(av)==len(bv);gate['status']='PASS' if gate['passed'] else 'PARITY_FAIL'
            comparisons.append(dict(condition=c,request_id=b['request_id'],numerical=gate,token_equal=equal_tokens,text_equal=a['generated_text']==b['generated_text'],quality=b['quality']))
            if c!='FULL_RECOMPUTE':
                left=a['prefill_records'];right=b['completion']['physical']['records'];right={(x['layer'],x['role']):x for x in right}
                for item in left:
                    other=right[item['layer'],item['role']]
                    for boundary in ('input','output'):
                        diff=metrics(ae.tensor('es',item[boundary]),be.tensor('es',other[boundary]),item[boundary]['dtype'])
                        prefill_diagnostics.append(dict(condition=c,request_id=b['request_id'],layer=item['layer'],role=item['role'],boundary=boundary,**diff))
    a=baseline['results'];c=reports['SEMCACHE_SUBSEQUENCE']['results']
    span=lambda rs:(rs[-1]['completion_ns']-rs[0]['arrival_ns'])/1e6
    economy=dict(A_source_inclusive_ms=span(a),C_source_inclusive_ms=span(c),C_minus_A_source_inclusive_ms=span(c)-span(a),warm_A_target_e2e_ms=a[1]['application_e2e_ms'],warm_C_target_e2e_ms=c[1]['application_e2e_ms'],source_C_cost_ms=c[0]['application_e2e_ms'],C_minus_warm_A_target_ms=span(c)-a[1]['application_e2e_ms'],reference='A target after A source; NOT an independently executed A0 reference',true_target_only_break_even='NOT_ESTABLISHED')
    first=next((r for r in prefill_diagnostics if not r['bitwise_equal']),None)
    costs={c:[request_costs(row) for row in r['results']] for c,r in reports.items()}
    out=output(output_dir,(experiment,run));result=dict(status='INDEPENDENT_EVIDENCE_AUDIT_PASS',reports=reports,request_costs=costs,comparisons=comparisons,prefill_diagnostics=prefill_diagnostics,first_observed_projection_difference=first,ROOT_CAUSE='UNRESOLVED',economics=economy,thresholds=THRESHOLDS,CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',statistical_scope='one context; descriptive only')
    write(out/'subsequence_audit.json',result)
    lines=['# Sparse generation pilot','','Evidence valid; numerical/quality/performance outcomes are separate. One context and at most2 target tokens: descriptive only.','','| Condition | Source app E2E ms | Target app E2E ms | Target TTFT ms | Reused projection rows | Frozen numerical gate |','|---|---:|---:|---:|---:|---|']
    for c,r in reports.items():
        rs=r['results'];gate='PASS' if all(x['numerical']['passed'] for x in comparisons if x['condition']==c) else 'FAIL'
        lines.append(f"| {c} | {rs[0]['application_e2e_ms']} | {rs[1]['application_e2e_ms']} | {rs[1]['application_ttft_ms']} | {r['reused_projection_rows']} | {gate} |")
    if 'integrity_validation' in p:
        lines+=['','Validation mode: '+p['integrity_validation']['mode'],
            '| Condition | Full validation proofs | Consumed proofs | Validation wall ms | Full serialized bytes | Consumed bytes |',
            '|---|---:|---:|---:|---:|---:|']
        for c,r in reports.items():
            physical=r['results'][1]['completion']['physical']
            if physical is not None:
                v=physical['integrity_validation'];lines.append(f"| {c} | {v['full_entry_proof_count']} | {v['consumed_tensor_proof_count']} | {v['request_validation_wall_ms']} | {v['full_entry_serialized_bytes']} | {v['consumed_tensor_serialized_bytes']} |")
        lines+=['Admission proofs are separate and remain in application E2E. Full/consumed proof subscopes are inclusive; do not add them to validation wall again. Unconsumed tensor content is not continuously rehashed in scoped mode.']
    lines+=['','CORRECTNESS_QUALIFIED_SPEEDUP=NOT_ESTABLISHED. ROOT_CAUSE=UNRESOLVED.','Economic reference is the measured warm A target interval, not a separately executed A0 workload. True target-only break-even is NOT_ESTABLISHED.','All probability capture/reduction, policy, tensor materialization and request evidence costs are instrumented. Initialization/process lifecycle are separate. No production latency or statistical quality claim.']
    (out/'subsequence_report.md').write_text('\n'.join(lines)+'\n');seal(out);return dict(status=result['status'],comparisons=comparisons)

def verify(experiment,run,audit,output_dir):
    import tempfile
    from .r5_i5r_parallel import output
    from .r5_i5_evidence import seal
    old=Evidence(audit,max_input_bytes=128*2**30);receipt=old.verify()
    with tempfile.TemporaryDirectory() as tmp:
        recomputed=Path(tmp)/'audit';analyze(experiment,run,recomputed);new=Evidence(recomputed,max_input_bytes=128*2**30);new.verify()
        require(old.inventory==new.inventory,'independent full replay differs')
    out=output(output_dir,(experiment,run,audit));write(out/'subsequence_verification.json',dict(status='PASS',audit_inventory=receipt));seal(out);return {'status':'PASS'}
