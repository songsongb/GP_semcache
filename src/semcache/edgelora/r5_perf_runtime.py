"""One isolated measured block; original semantic/controller/generator/OPT path."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import uuid

from .communication import Channel,connect,digest,header,require
from .r3a_artifacts import ROOT,full_base_fingerprint,read
from .r4a_runtime import setup
from .r4c_prepare import TinyTokenizer,opt_tokenizer
from .r4c_session import VERSION
from .r5_contracts import namespace,request
from .r5_controller import Controller,RawOwner
from .r5_generation import generate
from .r5_runtime import encoder_for
from .r5_semantic_index import SemanticIndex
from .r5_workload import workload,lifecycle_capacity,generation_policy
from .r5_perf_timing import ProfiledDriver
from .r5_perf_evidence import PerformanceEvidence,spawn,finish


def execute(plan,root,writer):
    root=Path(root);setup_start=time.monotonic_ns()
    pm,_,am,old,tokenizer,gpu=setup(plan,native=False)
    driver=ProfiledDriver(old.identity,old.device,model=old.model,adapter=old.adapter,tiny=old.tiny,native=False)
    if plan['fixture']:
        driver.tiny.matrices=None;tokenizer=TinyTokenizer();tok=dict(scope='TEST_FIXTURE_ONLY_MOD5_CHAR_TOKENIZER',vocab_sha256=digest(list(range(7))))
    else:driver.model_tokenizer=tokenizer;tok=opt_tokenizer(pm,tokenizer).identity
    (root/'es').mkdir(exist_ok=True);driver.configure(root/'es',budget_bytes=writer.budget)
    raw=workload(fixture=plan['fixture']);encoder=encoder_for(plan,raw);configuration=read(ROOT/'configs/r5_integration.json')
    capacity,capacity_plan=lifecycle_capacity(tokenizer,raw,3*pm['identity']['layers']*pm['identity']['hidden_size']*(4 if plan['fixture'] else 2))
    ident=namespace(pm,tok,driver.device,'TEST_FIXTURE_ONLY' if plan['fixture'] else gpu['uuid'],driver.contract)
    identity=dict(physical_identity=ident,model_identity=pm['identity'],model_identity_sha256=pm['identity_sha256'],tokenizer=tok,
        contract=driver.contract,gpu=gpu,semantic_encoder=encoder.metadata,TEST_FIXTURE_ONLY=plan['fixture'])
    writer.write(root/'r5_identity.json',identity);writer.write(root/'r5_configuration.json',configuration)
    writer.write(root/'r5_capacity_plan.json',dict(capacity_plan,actual_capacity_bytes=capacity))
    setup_ms=(time.monotonic_ns()-setup_start)/1e6;phases=[]
    for warmup in ([True,False] if plan['warmup_requests'] else [False]):
        phase_start=time.monotonic_ns();dest=root/'warmup' if warmup else root;es=dest/'es';es.mkdir(parents=True,exist_ok=True)
        phase_plan=dict(plan,output=str(es),performance_block=True)
        writer.write(es/'es_plan.json',phase_plan)
        for name in ('r5_requests','r5_semantic_events','r5_cache_events','r5_generation_outputs','r5_ownership'):
            writer.bytes(dest/(name+'.jsonl'),b'',exclusive=True)
        condition='DISTRIBUTED_NO_REUSE' if warmup or plan['condition']=='A' else 'INTEGRATED'
        driver.output=es;driver.diagnostics_path=es/'r4c_logit_diagnostics.jsonl';driver.cache_type=None
        driver.perf.events=[];driver.evidence=PerformanceEvidence(es,writer,profile=plan['evidence_profile'],
            reconcile_every=plan['reconcile_every'],legacy_scan=plan['legacy_scan'],perf=driver.perf)
        controls=0;channel=child=None;rows=[]
        def control(message):
            nonlocal controls
            rid=f'R5P_CONTROL_{controls:05d}';controls+=1
            channel.send(header('HELLO',rid,-1,**message));h,_,_=channel.recv(kind='READY',request_id=rid,layer_id=-1)
            require(h['metadata']['control_sha256']==digest(message) and h['metadata']['user_device_cuda_used'] is False,'control receipt differs')
            writer.append(es/'es_prefix_controls.jsonl',dict(message,request_id=rid))
        index=SemanticIndex(encoder,clusters=configuration['clusters'],update_mode=configuration['cluster_update_mode'],
            update_interval=configuration['cluster_update_interval'],limit=configuration['candidate_limit'],
            trace=lambda e:writer.append(dest/'r5_semantic_events.jsonl',dict(e,schema='R5_SEMANTIC_EVENT_V1',condition=condition)))
        owner=RawOwner(pm,plan['base_manifest_sha256'],ident['gpu_uuid'],driver.device,capacity,index=index,tokenizer=tok,
            sources=plan['sources'],fixture=plan['fixture'],contract=driver.contract,observed_cache_type=None,
            synchronize=driver.synchronize,control=control,trace=lambda e:writer.append(dest/'r5_cache_events.jsonl',dict(e,schema='R5_CACHE_EVENT_V1',condition=condition)))
        controller=Controller(owner,index,ident,reuse=condition=='INTEGRATED');session=uuid.uuid4().hex
        initial=dict(cache=owner.snapshot(),index_entries=list(index.entries),cluster=index.state(),seen=list(controller.seen),
            driver_cache_type_is_none=driver.cache_type is None,active_kv_history=False,session_id=session)
        writer.write(dest/'r5_initial_state.json',initial)
        writer.write(es/'r5_session.json',dict(cohort_sha256=owner.cohort,capacity_bytes=capacity,es_pid=os.getpid()))
        child_plan=dict(phase_plan,session_id=session,cohort_sha256=owner.cohort,context_limit=driver.context_limit)
        writer.write(es/'ud_plan.json',child_plan)
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            child=spawn([sys.executable,'-u',str(ROOT/'scripts/112_run_r5_balanced_benchmark.py'),'--worker','ud','--plan',str(es/'ud_plan.json')],
                writer,es/'ud.stdout.log',es/'ud.stderr.log',env=env)
            deadline=time.monotonic()+120
            while not (es/'r4c_ud_ready.json').exists():
                require(child[0].poll() is None and time.monotonic()<deadline,'UD startup failed/timed out');time.sleep(.01)
            info=read(es/'r4c_ud_ready.json');require(info['pid']==child[0].pid!=os.getpid(),'independent UD PID differs')
            channel=Channel(connect(info['port']),session,am['matrix_sha256'],timeout=180.,trace=lambda e:writer.append(es/'es_wire.jsonl',e));driver.channel=channel
            channel.send(header('HELLO','handshake',-1,application=VERSION,cohort_sha256=owner.cohort,identity_sha256=pm['identity_sha256']))
            h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
            require(h['metadata']['pid']==child[0].pid and not h['metadata']['user_device_cuda_used'],'UD handshake differs')
            writer.write(dest/'r5_ud_initial_state.json',dict(fresh_pid=child[0].pid,session_id=session,prefix_grants=[],
                completed_parents=0,handshake=h['metadata'],proof='new process; ForwardOrder constructed empty before handshake'))
            policy=generation_policy(tokenizer,plan['generation_mode'],plan['generation_budget'])
            for i,original in enumerate(raw[:plan['warmup_requests']] if warmup else raw):
                arrival=time.monotonic_ns();driver.perf.request_index=i;driver.perf.step=None
                r=deepcopy(original)
                if warmup:r['request_id']='W_'+r['request_id']
                elif condition=='DISTRIBUTED_NO_REUSE':r['request_id']='N_'+r['request_id']
                driver.perf.request_id=r['request_id'];event_start=len(driver.perf.events);io_before=writer.io_ms;scan_before=driver.evidence.diagnostics()
                with driver.perf.span('request_preparation'):
                    t=time.monotonic_ns();ids=list(tokenizer.encode(r['input_text'],add_special_tokens=False));token_ms=(time.monotonic_ns()-t)/1e6
                    req=request(r,ids,policy,ident,fixture=plan['fixture'])
                    require(len(ids)+policy['max_new_tokens']<=driver.context_limit and max(ids)<driver.vocab,'token/context limit')
                    writer.append(dest/'r5_requests.jsonl',dict(kind='ARRIVAL',request=req,condition=condition,arrival_ns=arrival,tokenization_ms=token_ms))
                if not plan['fixture']:
                    import torch
                    torch.cuda.reset_peak_memory_stats(driver.device)
                before=owner.snapshot();event_index=len(owner.events)
                with driver.perf.span('semantic_discovery_eligibility_and_lookup'):
                    discovery,found=controller.decide(req)
                row,steps=generate(driver,req,controller,discovery,found)
                if row['admission'] is not None:
                    key=row['admission']['entry_id'];owned=owner.entries[key];entry=next(iter(owned.pool.entries.values()))
                    specs=owned.validate(entry,owned.context)
                    writer.append(dest/'r5_ownership.jsonl',dict(entry_id=key,source_request_id=req['request_id'],source_user=req['user_id'],
                        context=owned.context,proof=owner.proofs[key],seal=deepcopy(owned.seals[entry.key]),tensor_specs=specs,
                        physical_bytes=owned.bytes,admission_ns=time.monotonic_ns(),condition=condition))
                blob=row.pop('logits_blob');row['logits_file']=driver.evidence.blob(blob,'logits/'+req['request_id']+'.f32.bin')
                driver.evidence.check();driver.synchronize();completed=time.monotonic_ns()
                row.update(condition=condition,user_id=req['user_id'],adapter_id=req['adapter_id'],request_sha256=req['request_sha256'],physical_identity=ident,
                    cluster_id=discovery['assignment']['cluster_id'],decision=discovery['outcome'],candidate_count=len(discovery['candidates']),
                    arrival_ns=arrival,completion_ns=completed,application_e2e_ms=(completed-arrival)/1e6,
                    application_ttft_ms=(row['first_token_available_ns']-arrival)/1e6,tokenization_ms=token_ms,
                    semantic_encoding_ms=discovery['assignment']['semantic_encoding_ms'],cluster_assignment_ms=discovery['assignment']['cluster_assignment_ms'],
                    candidate_lookup_ms=discovery['candidate_lookup_ms'],eligibility_validation_ms=discovery['eligibility_validation_ms'],
                    cache_lookup_and_kv_ms=discovery['cache_lookup_and_kv_ms'],cache_before=before,cache_after=owner.snapshot(),
                    eviction_count=sum(e['kind']=='EVICT' for e in owner.events[event_index:]),
                    es_to_ud_wire_bytes=row['prefill']['es_to_ud_wire_bytes']+row['decode']['es_to_ud_wire_bytes']+sum(f['wire_bytes'] for f in channel.records if
                        f['operation']=='send' and f['header']['kind'] not in ('HIDDEN','DELTAS') and arrival<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=completed),
                    ud_to_es_wire_bytes=row['prefill']['ud_to_es_wire_bytes']+row['decode']['ud_to_es_wire_bytes']+sum(f['wire_bytes'] for f in channel.records if
                        f['operation']=='recv' and f['header']['kind'] not in ('HIDDEN','DELTAS') and arrival<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=completed),
                    request_control_wire_bytes=row['control_wire_bytes'],evidence_profile=plan['evidence_profile'],es_pid=os.getpid(),ud_pid=child[0].pid,
                    TEST_FIXTURE_ONLY=plan['fixture'],block_id=plan['block_id'],request_index=i,
                    evidence_scan_ms=driver.evidence.diagnostics()['cumulative_scan_ms']-scan_before['cumulative_scan_ms'],
                    evidence_writer_ms=writer.io_ms-io_before,perf_nested_ms=driver.perf.totals(event_start))
                row['gpu_memory']=None if plan['fixture'] else dict(allocated=torch.cuda.memory_allocated(driver.device),
                    reserved=torch.cuda.memory_reserved(driver.device),peak=torch.cuda.max_memory_allocated(driver.device))
                writer.append(dest/'r5_generation_outputs.jsonl',row)
                writer.append(dest/'r5_requests.jsonl',dict(kind='COMPLETION',request_id=req['request_id'],condition=condition,completion_ns=completed,request_sha256=req['request_sha256']))
                rows.append(row)
            # Actual probe, outside request timings, before shutdown.
            before_probe=len(channel.records)
            bad=deepcopy(req);bad['request_id']='NEG_ADAPTER';bad['adapter_id']='unsupported';bad['physical_identity']['adapter_id']='unsupported';bad.pop('request_sha256');bad['request_sha256']=digest(bad)
            try:controller.decide(bad)
            except ValueError as exc:require('UNSUPPORTED_ADAPTER_OR_MODEL' in str(exc),'unexpected probe failure')
            else:raise ValueError('unsupported adapter accepted')
            require(len(channel.records)==before_probe,'unsupported adapter sent traffic')
            writer.write(dest/'r5_negative_controls.json',dict(adapter_probe='EXPECTED_REJECTION',tensor_frames=0,generated_request=False,
                validation='actual unsupported-adapter probe after measured requests'))
            channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();channel=None
            life=finish(child,timeout=30);writer.write(es/'ud_lifecycle.json',life);require(life['returncode']==0 and not life['pipe_errors'],'UD failed')
            result=dict(execution_status='COMPLETE',pid=os.getpid(),ud_pid=child[0].pid,session_id=session,requests=len(rows),cache_final=owner.snapshot(),
                user_device_cuda_used=False,gpu_inference_performed=not plan['fixture'],block_id=plan['block_id'],condition=plan['condition'])
            writer.write(es/'es_result.json',result)
            writer.write(dest/'r5_accounting_diagnostics.json',dict(**driver.evidence.diagnostics(),reconciliations=writer.scans,legacy_scans=driver.evidence.legacy_scans))
            phases.append(dict(warmup=warmup,requests=len(rows),elapsed_ms=(time.monotonic_ns()-phase_start)/1e6,es_pid=os.getpid(),ud_pid=child[0].pid,
                cache_final=owner.snapshot(),session_id=session))
            if warmup:require(owner.bytes==0 and not index.entries,'warmup populated cache')
        finally:
            if channel is not None:channel.close()
            if child is not None and child[0].poll() is None:child[0].terminate();finish(child,timeout=10)
    if not plan['fixture']:
        from semcache.system_cost.es_base_profile import assert_no_lora
        assert_no_lora(driver.model,driver.adapter);require(full_base_fingerprint(driver.model)['sha256']==pm['identity']['base_weight_sha256'],'base mutated')
    writer.write(root/'r5_block_execution.json',dict(setup_ms=setup_ms,phases=phases,execution_status='COMPLETE'))
    return result
