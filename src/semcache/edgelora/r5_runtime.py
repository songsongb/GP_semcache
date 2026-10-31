"""Persistent ES, separate CPU UD, actual semantic → generation application path."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from semcache.semantic.encoder import ControlledEncoder, TinyBERTSemanticEncoder
from .communication import Channel, connect, digest, header, require
from .r3a_artifacts import ROOT, append, full_base_fingerprint, read, sha, write
from .r3c_process import ready, reap
from .r4a_runtime import setup
from .r4c_generation import QualityDriver
from .r4c_prepare import TinyTokenizer, opt_tokenizer
from .r4c_session import VERSION
from .r5_contracts import namespace, request
from .r5_controller import Controller, RawOwner
from .r5_generation import generate
from .r5_evidence import TimedEvidence
from .r5_semantic_index import SemanticIndex
from .r5_workload import workload, lifecycle_capacity, generation_policy


def encoder_for(plan,raw):
    if plan['fixture']:
        return ControlledEncoder({r['input_text']:[1.,0.] for r in raw})
    import re
    revision=plan['semantic_revision']
    require(isinstance(revision,str) and re.fullmatch('[0-9a-f]{40}',revision),
        'MISSING_TINYBERT: explicit immutable --semantic-revision required; legacy config main is unpinned')
    try:
        encoder=TinyBERTSemanticEncoder(revision=revision,device='cpu',dtype='float32',local_files_only=True)
    except (OSError,ImportError) as exc:
        raise ValueError('MISSING_TINYBERT: offline checkpoint/revision required; no fixture fallback: '+str(exc)) from exc
    require(encoder.metadata['resolved_revision']==revision,'TinyBERT resolved revision differs')
    encoder._metadata['weights_sha256']=full_base_fingerprint(encoder.model)['sha256']
    encoder._metadata['tokenizer_vocab_sha256']=digest(encoder.tokenizer.get_vocab())
    return encoder


def launch_ud(plan,output):
    write(output/'ud_plan.json',dict(plan,output=str(output)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    out=(output/'ud.stdout.log').open('w');err=(output/'ud.stderr.log').open('w')
    try:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/110_run_r5_semcache_e2e.py'),'--worker','ud','--plan',str(output/'ud_plan.json')],
            cwd=ROOT,env=env,stdout=out,stderr=err)
    except BaseException:out.close();err.close();raise
    return p,out,err


def execute(plan,output):
    output=Path(output);root=output.parent;channel=ud=None;result=dict(execution_status='RUNNING',stage='SETUP');count=0
    try:
        pm,ref,am,old,tokenizer,gpu=setup(plan,native=False)
        driver=QualityDriver(old.identity,old.device,model=old.model,adapter=old.adapter,tiny=old.tiny,native=False)
        if plan['fixture']:
            driver.tiny.matrices=None;tokenizer=TinyTokenizer();tok=dict(scope='TEST_FIXTURE_ONLY_MOD5_CHAR_TOKENIZER',vocab_sha256=digest(list(range(7))))
        else:
            driver.model_tokenizer=tokenizer;tok=opt_tokenizer(pm,tokenizer).identity
        driver.configure(output,budget_bytes=plan['evidence_budget_bytes'])
        driver.evidence=TimedEvidence(output,budget_bytes=plan['evidence_budget_bytes'])
        configuration=read(ROOT/'configs/r5_integration.json')
        require(configuration['window_size']==3 and configuration['physical_policy']=='LRU_FIT_COMPLETE_SOURCE_V1',
            'unsupported policy; attention impact is unavailable with output_attentions=False')
        write(root/'r5_configuration.json',configuration)
        raw=workload(fixture=plan['fixture']);encoder=encoder_for(plan,raw)
        bytes_per_token=3*pm['identity']['layers']*pm['identity']['hidden_size']*(4 if plan['fixture'] else 2)
        lifecycle_budget,capacity_evidence=lifecycle_capacity(tokenizer,raw,bytes_per_token)
        capacity=lifecycle_budget if plan['lifecycle_capacity'] else plan['capacity_bytes']
        write(root/'r5_capacity_plan.json',dict(capacity_evidence,actual_capacity_bytes=capacity))
        ident=namespace(pm,tok,driver.device,'TEST_FIXTURE_ONLY' if plan['fixture'] else gpu['uuid'],driver.contract)
        write(root/'r5_identity.json',dict(physical_identity=ident,model_identity=pm['identity'],
            model_identity_sha256=pm['identity_sha256'],tokenizer=tok,contract=driver.contract,gpu=gpu,
            semantic_encoder=encoder.metadata,TEST_FIXTURE_ONLY=plan['fixture']))
        control_count=0;active_condition='INTEGRATED'
        def control(message):
            nonlocal control_count
            rid=f'R5_CONTROL_{control_count:05d}';control_count+=1
            channel.send(header('HELLO',rid,-1,**message));h,_,_=channel.recv(kind='READY',request_id=rid,layer_id=-1)
            require(h['metadata']['control_sha256']==digest(message) and h['metadata']['user_device_cuda_used'] is False,'invalid UD control receipt')
            append(output/'es_prefix_controls.jsonl',dict(message,request_id=rid))
        index=SemanticIndex(encoder,clusters=configuration['clusters'],update_mode=configuration['cluster_update_mode'],
            update_interval=configuration['cluster_update_interval'],limit=configuration['candidate_limit'],
            trace=lambda e:append(root/'r5_semantic_events.jsonl',dict(e,schema='R5_SEMANTIC_EVENT_V1',condition=active_condition)))
        owner=RawOwner(pm,plan['base_manifest_sha256'],ident['gpu_uuid'],driver.device,capacity,
            index=index,tokenizer=tok,sources=plan['sources'],fixture=plan['fixture'],contract=driver.contract,
            observed_cache_type=None,synchronize=driver.synchronize,control=control,
            trace=lambda e:append(root/'r5_cache_events.jsonl',dict(e,schema='R5_CACHE_EVENT_V1',condition=active_condition)))
        cohort=owner.cohort
        write(output/'r5_session.json',dict(cohort_sha256=cohort,capacity_bytes=capacity,es_pid=os.getpid()))
        result['stage']='UD_START';session=uuid.uuid4().hex
        ud=launch_ud(dict(plan,session_id=session,cohort_sha256=cohort,context_limit=driver.context_limit),output)
        info=ready(ud,output/'r4c_ud_ready.json');require(info['pid']==ud[0].pid!=os.getpid(),'UD is not an independent PID')
        channel=Channel(connect(info['port']),session,am['matrix_sha256'],timeout=180.,trace=lambda e:append(output/'es_wire.jsonl',e));driver.channel=channel
        channel.send(header('HELLO','handshake',-1,application=VERSION,cohort_sha256=cohort,identity_sha256=pm['identity_sha256']))
        h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(h['metadata']['pid']==ud[0].pid and h['metadata']['cohort_sha256']==cohort and h['metadata']['user_device_cuda_used'] is False,'invalid handshake')
        p=generation_policy(tokenizer,plan['generation_mode'],plan['generation_budget'])
        global_seen=set()
        for condition in ('INTEGRATED','DISTRIBUTED_NO_REUSE'):
            active_condition=condition
            if condition=='DISTRIBUTED_NO_REUSE':
                owner.request_id='CONDITION_RESET'
                for key in list(owner.entries):owner.remove(key,'INVALIDATE','end integrated condition')
            controller=Controller(owner,index,ident,reuse=condition=='INTEGRATED')
            for original in raw:
                arrival=time.monotonic_ns();r=deepcopy(original)
                if condition=='DISTRIBUTED_NO_REUSE':r['request_id']='N_'+r['request_id']
                result.update(stage='APPLICATION_REQUEST',request_id=r['request_id'],condition=condition)
                write(output/'r5_stage.json',result)
                require(r['request_id'] not in global_seen,'duplicate application request');global_seen.add(r['request_id'])
                driver.evidence.profile=plan['evidence_profile'];tick=time.monotonic_ns()
                write_before=driver.evidence.write_ms
                ids=list(tokenizer.encode(r['input_text'],add_special_tokens=False));tokenized=time.monotonic_ns()
                req=request(r,ids,p,ident,fixture=plan['fixture'])
                require(len(ids)+p['max_new_tokens']<=driver.context_limit and max(ids)<driver.vocab,'input context/vocabulary limit')
                append(root/'r5_requests.jsonl',dict(schema='R5_REQUEST_EVENT_V1',kind='ARRIVAL',request=req,condition=condition,arrival_ns=arrival,tokenization_ms=(tokenized-tick)/1e6))
                if not plan['fixture']:
                    import torch
                    torch.cuda.reset_peak_memory_stats(driver.device)
                before=owner.snapshot();events_start=len(owner.events);wire_start=len(channel.records)
                discovery,found=controller.decide(req)
                row,steps=generate(driver,req,controller,discovery,found)
                # Audit the actual authoritative owner, including all three roles.
                if row['admission'] is not None:
                    key=row['admission']['entry_id'];owned=owner.entries[key];entry=next(iter(owned.pool.entries.values()))
                    specs=owned.validate(entry,owned.context)
                    append(root/'r5_ownership.jsonl',dict(schema='R5_OWNERSHIP_V1',entry_id=key,source_request_id=req['request_id'],
                        source_user=req['user_id'],context=owned.context,proof=owner.proofs[key],
                        seal=deepcopy(owned.seals[entry.key]),tensor_specs=specs,physical_bytes=owned.bytes,
                        admission_ns=time.monotonic_ns(),condition=condition))
                blob=row.pop('logits_blob');row['logits_file']=driver.evidence.blob(blob,'logits/'+req['request_id']+'.f32.bin')
                driver.evidence.check();driver.synchronize();completed=time.monotonic_ns()
                request_frames=channel.records[wire_start:]
                row.update(schema='R5_GENERATION_V1',condition=condition,user_id=req['user_id'],adapter_id=req['adapter_id'],
                    request_sha256=req['request_sha256'],physical_identity=ident,cluster_id=discovery['assignment']['cluster_id'],
                    decision=discovery['outcome'],candidate_count=len(discovery['candidates']),
                    arrival_ns=arrival,completion_ns=completed,application_e2e_ms=(completed-arrival)/1e6,
                    application_ttft_ms=(row['first_token_available_ns']-arrival)/1e6,
                    tokenization_ms=(tokenized-tick)/1e6,semantic_encoding_ms=discovery['assignment']['semantic_encoding_ms'],
                    cluster_assignment_ms=discovery['assignment']['cluster_assignment_ms'],candidate_lookup_ms=discovery['candidate_lookup_ms'],
                    eligibility_validation_ms=discovery['eligibility_validation_ms'],cache_lookup_and_kv_ms=discovery['cache_lookup_and_kv_ms'],
                    cache_before=before,cache_after=owner.snapshot(),eviction_count=sum(e['kind']=='EVICT' for e in owner.events[events_start:]),
                    es_to_ud_wire_bytes=sum(f['wire_bytes'] for f in request_frames if f['operation']=='send'),
                    ud_to_es_wire_bytes=sum(f['wire_bytes'] for f in request_frames if f['operation']=='recv'),
                    request_control_wire_bytes=sum(f['wire_bytes'] for f in request_frames if f['header']['kind'] not in ('HIDDEN','DELTAS')),
                    evidence_profile=driver.evidence.profile,es_pid=os.getpid(),ud_pid=ud[0].pid,TEST_FIXTURE_ONLY=plan['fixture'],
                    timing_scope='instrumented; application includes semantic/cache/evidence; model excludes discovery and admission; nested timers overlap',
                    evidence_write_overhead='included in forward host_forward_and_evidence_ms; no uninstrumented speedup claim')
                row['tensor_logits_evidence_write_ms']=driver.evidence.write_ms-write_before
                row['cache_lookup_validation_ms']=0. if found is None else owner.last_lookup_timings['cache_lookup_validation_ms']
                row['kv_construction_and_validation_ms']=0. if found is None else owner.last_lookup_timings['kv_construction_and_validation_ms']
                if not plan['fixture']:
                    row['gpu_memory']=dict(allocated_bytes=torch.cuda.memory_allocated(driver.device),reserved_bytes=torch.cuda.memory_reserved(driver.device),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(driver.device))
                else:row['gpu_memory']=None
                append(root/'r5_generation_outputs.jsonl',row)
                append(root/'r5_requests.jsonl',dict(schema='R5_REQUEST_EVENT_V1',kind='COMPLETION',request_id=req['request_id'],condition=condition,completion_ns=completed,request_sha256=req['request_sha256']))
                count+=1
        # A separately labeled probe is never counted as a generated request.
        before=len(channel.records);probe=deepcopy(req);probe['request_id']='NEG_UNPREPARED_ADAPTER';probe['adapter_id']='other_adapter'
        probe['physical_identity']['adapter_id']='other_adapter';probe.pop('request_sha256');probe['request_sha256']=digest(probe)
        try:controller.decide(probe)
        except ValueError as exc:
            require('UNSUPPORTED_ADAPTER_OR_MODEL' in str(exc),'unexpected negative-control error')
            write(root/'r5_negative_controls.json',dict(adapter_probe='EXPECTED_REJECTION',reason=str(exc),generated_request=False,
                tensor_frames=len(channel.records)-before,request_id=probe['request_id']))
        else:raise ValueError('unsupported adapter accepted')
        result['stage']='SHUTDOWN';channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
        channel.close();channel=None;ud[0].wait(timeout=10);require(ud[0].returncode==0,'UD failed')
        if not plan['fixture']:
            from semcache.system_cost.es_base_profile import assert_no_lora
            assert_no_lora(driver.model,driver.adapter)
            require(full_base_fingerprint(driver.model)['sha256']==pm['identity']['base_weight_sha256'],'ES base weights mutated')
        result.update(execution_status='COMPLETE',stage='COMPLETE',pid=os.getpid(),ud_pid=ud[0].pid,session_id=session,
            requests=count,cache_final=owner.snapshot(),user_device_cuda_used=False,gpu_inference_performed=not plan['fixture'])
        return result
    except BaseException as exc:
        result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}')
        write(output/'r5_failure.json',result);raise
    finally:
        if channel is not None:channel.close()
        if ud is not None:result['ud_lifecycle']=reap(ud,output/'ud_lifecycle.json',group=False)
        result['completed_requests']=count;write(output/'r5_partial.json',result)
