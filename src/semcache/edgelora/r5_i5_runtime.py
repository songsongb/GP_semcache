"""Sequential multi-user serving: one bare model and one CPU UD per adapter."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import uuid

from .communication import Channel,connect,digest,header,require
from .r3a_artifacts import ROOT,full_base_fingerprint,read
from .r4a_runtime import setup
from .r5_i5_driver import QualityDriver
from .r4c_prepare import TinyTokenizer
from .r5_i5_contracts import request
from .r5_i5_generation import generate
from .r5_runtime import encoder_for
from .r5_semantic_index import SemanticIndex
from .r5_multi_artifacts import validate,runtime_view,physical,resolve,tokenizer_identity
from .r5_i5_cache import I5Owner as MultiOwner
from .r5_i5_controller import I5Controller,DisabledIndex
from .r5_multi_controller import reject_foreign,public_result
from .r5_i5_session import VERSION,RoutedChannel
from .r5_perf_evidence import PerformanceEvidence,spawn,finish


def execute(plan,root,writer):
    root=Path(root);m,pm,adapters,registry=validate(plan['multi_prepared'])
    require(m['TEST_FIXTURE_ONLY']==plan['fixture'] and m['base_prepared']==plan['base_prepared'],'multi/base preparation scope mismatch')
    initialized=time.monotonic_ns()
    _,_,_,old,tokenizer,gpu=setup(plan,native=False)
    if plan['fixture']:old.tiny.matrices=None;tokenizer=TinyTokenizer()
    tok=tokenizer_identity(pm,tokenizer);require(tok==m['tokenizer'],'runtime tokenizer differs from adapter preparation')
    raw=deepcopy(plan['block']['tasks']);encoder=None if plan['condition']=='FULL_RECOMPUTE' else encoder_for(plan,raw);config=read(ROOT/'configs/r5_integration.json')
    require(config['clusters']==1 and config['physical_policy']=='LRU_FIT_COMPLETE_SOURCE_V1','unsupported semantic/cache policy')
    writer.write(root/'configuration.json',config);writer.write(root/'registry.json',registry);writer.write(root/'workload.json',raw)
    states={};children=[];events=[];rows=[]
    def emit(aid,stream,event):
        row=dict(sequence=len(events),adapter_id=aid,stream=stream,event=event,observed_ns=time.monotonic_ns())
        events.append(row);writer.append(root/'events.jsonl',row)
    try:
        for aid,spec in adapters.items():
            base=root/aid;es=base/'es';ud=base/'ud';es.mkdir(parents=True);ud.mkdir()
            view=runtime_view(pm,spec)
            driver=QualityDriver(view['identity'],old.device,model=old.model,adapter=old.adapter,tiny=old.tiny,native=False)
            if not plan['fixture']:driver.model_tokenizer=tokenizer
            contract=driver.configure(es,budget_bytes=writer.budget)
            ident,contract=physical(view,tok,driver.device,'TEST_FIXTURE_ONLY' if plan['fixture'] else gpu['uuid'],contract,spec,registry)
            driver.contract=contract;driver.evidence=PerformanceEvidence(es,writer,profile='SCALABLE_EVALUATION',reconcile_every=32)
            index=DisabledIndex() if encoder is None else SemanticIndex(encoder,clusters=config['clusters'],update_mode=config['cluster_update_mode'],
                update_interval=config['cluster_update_interval'],limit=config['candidate_limit'],
                trace=lambda e,aid=aid:emit(aid,'semantic',e))
            capacity=plan['block']['resources']['per_adapter_capacity_bytes']
            capacity_plan=plan['block']['resources']
            state=dict(index=index,driver=driver,control_count=0,base=base,physical=ident,view=view)
            def control(message,state=state,aid=aid):
                rid=f'MC_{aid}_{state["control_count"]:04d}';state['control_count']+=1;channel=state['channel']
                channel.send(header('HELLO',rid,-1,application=VERSION,control=message))
                h,_,_=channel.recv(kind='READY',request_id=rid,layer_id=-1)
                require(h['metadata']['control_sha256']==digest(message) and not h['metadata']['user_device_cuda_used'],'invalid control receipt')
                writer.append(state['base']/'es/controls.jsonl',dict(message,request_id=rid))
            owner=MultiOwner(view,digest(dict(preparation=m,adapter=spec)),ident['gpu_uuid'],driver.device,capacity,
                index=index,tokenizer=tok,sources=plan['sources'],fixture=plan['fixture'],contract=contract,observed_cache_type=None,
                spec=spec,registry=registry,synchronize=driver.synchronize,control=control,trace=lambda e,aid=aid:emit(aid,'cache',e))
            controller=I5Controller(owner,index,ident,condition=plan['condition'],registry=registry)
            state.update(owner=owner,controller=controller);states[aid]=state;session=uuid.uuid4().hex
            writer.write(base/'identity.json',dict(physical_identity=ident,model_identity=view['identity'],adapter=spec,tokenizer=tok,
                contract=contract,gpu=gpu,semantic_encoder=None if encoder is None else encoder.metadata,TEST_FIXTURE_ONLY=plan['fixture']))
            writer.write(base/'initial.json',dict(cache=owner.snapshot(),index=index.state(),index_entries=list(index.entries),
                seen=list(controller.seen),active_past=False,session_id=session,es_pid=os.getpid()))
            writer.write(base/'capacity.json',dict(capacity_plan,actual_capacity_bytes=capacity))
            child_plan=dict(plan,adapter_id=aid,physical_identity=ident,session_id=session,cohort_sha256=owner.cohort,
                context_limit=driver.context_limit,output=str(ud));writer.write(ud/'plan.json',child_plan)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            child=spawn([sys.executable,'-u',str(ROOT/'scripts/119_run_r5_i5.py'),'--worker','ud','--plan',str(ud/'plan.json')],
                writer,ud/'stdout.log',ud/'stderr.log',env=env);children.append(child);state['child']=child
            deadline=time.monotonic()+120
            while not (ud/'ready.json').exists():
                require(child[0].poll() is None and time.monotonic()<deadline,'UD startup failed/timed out');time.sleep(.01)
            info=read(ud/'ready.json');require(info['pid']==child[0].pid!=os.getpid(),'UD is not independent')
            channel=Channel(connect(info['port']),session,spec['matrix_sha256'],timeout=plan['runtime_budget_seconds'],trace=lambda f,es=es:writer.append(es/'wire.jsonl',f))
            state['channel']=channel;driver.channel=RoutedChannel(channel,registry,ident)
            hello=dict(application=VERSION,adapter_id=aid,matrix_sha256=spec['matrix_sha256'],registry_sha256=registry['sha256'],
                physical_identity=ident,cohort_sha256=owner.cohort)
            channel.send(header('HELLO','handshake',-1,**hello));h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
            require(h['metadata']==dict(hello,pid=info['pid'],user_device_cuda_used=False),'wrong adapter handshake')
        writer.write(root/'initialization.json',dict(model_and_session_initialization_ms=(time.monotonic_ns()-initialized)/1e6))
        from .r5_i5_limits import wire_limits
        for t in raw:
            state=states[t['adapter_id']]
            req=request(t,t['token_ids'],t['policy'],state['physical'],fixture=plan['fixture'])
            limits=wire_limits(req,registry,state['owner'].cohort,state['driver'].context_limit)
            writer.append(root/'wire_limits.jsonl',dict(request_id=t['request_id'],**limits))
        for aid,state in states.items():
            task=next(t for t in raw if t['adapter_id']==aid) if any(t['adapter_id']==aid for t in raw) else raw[0]
            warm=dict(task,request_id='W_'+aid,user_id=next(u for u,v in registry['users'].items() if v['adapter_id']==aid),adapter_id=aid,input_text='model warmup')
            req=request(warm,task['token_ids'][:min(16,len(task['token_ids']))],dict(task['policy'],mode='fixed',stop_at_eos=False,max_new_tokens=1),state['physical'],fixture=plan['fixture'])
            state['driver'].channel.request=req;controller=state['controller'];condition=controller.condition;controller.condition='FULL_RECOMPUTE';controller.reuse=False
            begin=time.monotonic_ns();discovery,found=controller.decide(req);row,steps=generate(state['driver'],req,controller,discovery,found)
            row.pop('logits_blob');writer.append(root/'warmup.jsonl',dict(request=req,elapsed_ms=(time.monotonic_ns()-begin)/1e6,generated_token_ids=row['generated_token_ids']))
            controller.condition=condition;controller.reuse=condition=='SEMCACHE_INTEGRATED'
            require(not state['owner'].entries and not state['index'].entries,'warmup admitted cache state')
        for original in raw:
            arrival=time.monotonic_ns();r=deepcopy(original)
            aid=resolve(registry,r['user_id'],r['adapter_id']);state=states[aid];driver=state['driver'];owner=state['owner'];index=state['index']
            tick=time.monotonic_ns();ids=list(r['token_ids']);policy=r['policy']
            if r['scored']:require(ids==list(tokenizer.encode(r['input_text'],add_special_tokens=False)),'prepared/runtime input IDs differ')
            req=request(r,ids,policy,state['physical'],fixture=plan['fixture'])
            require(len(ids)+policy['max_new_tokens']<=driver.context_limit and max(ids)<driver.vocab,'token/context limit')
            token_ms=(time.monotonic_ns()-tick)/1e6;driver.channel.request=req
            writer.append(root/'requests.jsonl',dict(kind='ARRIVAL',request=req,arrival_ns=arrival,tokenization_ms=token_ms))
            before=owner.snapshot();first=len(state['channel'].records);event_first=len(owner.events)
            foreign=[] if plan['condition']=='FULL_RECOMPUTE' else reject_foreign(req,states,registry);writer.append(root/'isolation.jsonl',dict(request_id=req['request_id'],adapter_id=aid,
                rejected=foreign,foreign_tensor_reads=0,foreign_wire_frames=0,scope='PRIVILEGED_ADMIN_AUDIT_ONLY'))
            driver.evidence.profile='STRICT_AUDIT' if r['strict'] else 'SCALABLE_EVALUATION'
            io_before=writer.io_ms
            discovery,found=state['controller'].decide(req)
            cached_capture={}
            if found is not None and r['strict']:
                for layer,block in found[0].tensors.items():
                    for role,tensor in zip('qkv',block):cached_capture[f'{layer}.{role}']=driver.evidence.save(tensor,req['request_id']+'_cached_'+str(layer)+role)
            row,steps=generate(driver,req,state['controller'],discovery,found)
            if row['admission'] is not None:
                key=row['admission']['entry_id'];owned=owner.entries[key];entry=next(iter(owned.pool.entries.values()))
                writer.append(root/'ownership.jsonl',dict(entry_id=key,source_request_id=req['request_id'],source_user=req['user_id'],adapter_id=aid,
                    registry_sha256=registry['sha256'],sharing_authorization=registry['users'][req['user_id']],context=owned.context,
                    proof=owner.proofs[key],seal=deepcopy(owned.seals[entry.key]),tensor_specs=owned.validate(entry,owned.context),
                    physical_bytes=owned.bytes,admission_ns=time.monotonic_ns()))
            blob=row.pop('logits_blob');row['logits_file']=driver.evidence.blob(blob,'logits/'+req['request_id']+'.f32.bin')
            driver.evidence.check();driver.synchronize();completed=time.monotonic_ns();frames=state['channel'].records[first:]
            row.update(condition=plan['condition'],user_id=req['user_id'],adapter_id=aid,request_sha256=req['request_sha256'],physical_identity=state['physical'],
                arrival_ns=arrival,completion_ns=completed,application_e2e_ms=(completed-arrival)/1e6,application_ttft_ms=(row['first_token_available_ns']-arrival)/1e6,
                tokenization_ms=token_ms,semantic_encoding_ms=discovery['assignment']['semantic_encoding_ms'],
                cluster_assignment_ms=discovery['assignment']['cluster_assignment_ms'],cache_lookup_ms=discovery['cache_lookup_and_kv_ms'],
                candidate_lookup_ms=discovery['candidate_lookup_ms'],eligibility_validation_ms=discovery['eligibility_validation_ms'],
                candidate_count=len(discovery['candidates']),candidate_reasons=discovery['candidate_reasons'],exact_prefix_eligible=any(c['rejection'] is None for c in discovery['candidate_reasons']),
                semantic_disabled=discovery.get('semantic_disabled',False),decision=discovery['outcome'],cache_before=before,cache_after=owner.snapshot(),
                eviction_count=sum(e['kind']=='EVICT' for e in owner.events[event_first:]),
                rejected_cross_adapter_candidates=sum(e['rejected_count'] for e in foreign),
                es_to_ud_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='send'),
                ud_to_es_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='recv'),
                cpu_lora_ms=sum(f['header']['metadata'].get('cpu_lora_ms',0.) for f in frames if f['operation']=='recv'),
                es_pid=os.getpid(),ud_pid=state['child'][0].pid,TEST_FIXTURE_ONLY=plan['fixture'])
            if plan['fixture']:row['gpu_memory']=None
            else:
                import torch
                row['gpu_memory']=dict(allocated_bytes=torch.cuda.memory_allocated(driver.device),reserved_bytes=torch.cuda.memory_reserved(driver.device),
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(driver.device))
            row.update(role=r['role'],scored=r['scored'],dataset=r['dataset'],sample_id=r['sample_id'],group_id=r['group_id'],pair_id=r['pair_id'],
                track=plan['block']['track'],prefix_length_target=plan['block']['prefix_length'],repetition=plan['block']['repetition'],
                evidence_profile=driver.evidence.profile,strict_cached_qkv=cached_capture,shadow=state['controller'].shadow_state(),
                virtual_selected_entry_id=discovery.get('virtual_selected_entry_id'),
                kv_reconstruction_ms=0. if found is None else owner.last_lookup_timings['kv_construction_and_validation_ms'],
                evidence_logging_io_ms=writer.io_ms-io_before,
                evidence_logging_scope='ES accounted file IO; nested in model/app wall, excludes UD IO and tensor hash time',
                matched_identity_sha256=digest(dict(task=r,adapter_sha256=adapters[aid]['matrix_sha256'],tokenizer=tok)),
                input_tokens=len(ids),reused_input_fraction=row['reused_prefix_tokens']/len(ids),
                cuda_event_ms=sum(s['cuda_event_elapsed_ms'] or 0. for s in steps),
                app_e2e_includes_admission=True,model_timing_scope='synchronized instrumented distributed forward and evidence; completion receipt included',
                original_context_tokens=r['prepared']['original_context_tokens'],retained_context_tokens=r['prepared']['retained_context_tokens'],
                removed_context_tokens=r['prepared']['removed_context_tokens'],truncated=r['prepared']['truncated'])
            from .r5_i5_quality import score
            if r['scored']:writer.append(root/'quality.jsonl',dict(score(r['prepared']['sample'],row['generated_text'],r['prepared']['template_version']),request_id=req['request_id'],condition=plan['condition'],truncated=row['truncated'],hit=row['source'] is not None))
            writer.append(root/'generation.jsonl',row);writer.append(root/'requests.jsonl',dict(kind='COMPLETION',request_id=req['request_id'],completion_ns=completed))
            writer.write(root/'client_results'/(req['request_id']+'.json'),public_result(row));rows.append(row)
        # Actual fail-closed mapping probes occur before tokenization/TCP.
        wire_before=sum(len(s['channel'].records) for s in states.values());probes=[]
        for user,claimed in ((next(iter(registry['users'])),'unprepared'),('Unknown_User',next(iter(adapters))),
                             (next(iter(registry['users'])),list(adapters)[1])):
            try:resolve(registry,user,claimed)
            except ValueError as exc:probes.append(dict(outcome='EXPECTED_REJECTION',reason=str(exc),generated_request=False))
            else:raise ValueError('invalid user/adapter request accepted')
        require(wire_before==sum(len(s['channel'].records) for s in states.values()),'unauthorized probe sent frames')
        writer.write(root/'negative_controls.json',dict(probes=probes,wire_frames=0))
        for aid,state in states.items():
            channel=state['channel'];channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();state['channel']=None
            life=finish(state['child'],timeout=30);writer.write(state['base']/'ud/lifecycle.json',life)
            require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'UD shutdown failed')
            writer.write(state['base']/'final.json',dict(cache=state['owner'].snapshot(),ud_pid=state['child'][0].pid))
        if not plan['fixture']:
            from semcache.system_cost.es_base_profile import assert_no_lora
            assert_no_lora(old.model,old.adapter);require(full_base_fingerprint(old.model)['sha256']==pm['identity']['base_weight_sha256'],'base weights mutated')
        writer.write(root/'result.json',dict(status='COMPLETE',condition=plan['condition'],es_pid=os.getpid(),requests=len(rows),
            gpu_inference_performed=not plan['fixture'],base_models_loaded=1,es_lora_parameters=0,adapter_ids=list(adapters),sequential=True))
    finally:
        for state in states.values():
            if state.get('channel') is not None:state['channel'].close()
        for child in children:
            if child[0].poll() is None:child[0].terminate();finish(child,timeout=10)
