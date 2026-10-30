"""OPT-only C1 backend; generic datasets/metrics do not depend on this module."""
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from copy import deepcopy

from .communication import Channel, connect, digest, digest_bytes, header, require
from .r3a_artifacts import ROOT, append, full_base_fingerprint, read, validate_prepared, write
from .r3c_process import ready, reap
from .r4a_runtime import setup
from .r4c_cache import PrefixCache
from .r4c_generation import QualityDriver, generate
from .r4c_prepare import load_prepared, opt_tokenizer
from .r4c_session import VERSION


def flags(fixture):
    return dict(TEST_FIXTURE_ONLY=fixture,NATIVE_PEFT_STRICT_PARITY_PASS=False,PRIMARY_PERFORMANCE_ELIGIBLE=False,
                correctness_qualified_speedup=False,native_parity_deferred=True,trained_adapter=False,
                actual_remote_network_measured=False,network_scope='LOCALHOST_TCP',qwen_physical_implemented=False)


def launch_ud(plan,output):
    write(output/'ud_plan.json',dict(plan,output=str(output)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    out=(output/'ud.stdout.log').open('w');err=(output/'ud.stderr.log').open('w')
    try:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/106_run_r4c_quality_e2e.py'),'--worker','ud','--plan',str(output/'ud_plan.json')],
                                 cwd=ROOT,env=env,stdout=out,stderr=err)
    except BaseException:out.close();err.close();raise
    return process,out,err


def negative_controls(driver,cache,task):
    """Requested invalid HITs must fail before any tensor network frame."""
    controls=[]
    def reject(name,change=None):
        altered=deepcopy(task);altered['request_id']='NEG_'+name
        if change is not None:change(altered['prompt'])
        if name!='digest':
            p=altered['prompt'];p['prompt_sha256']=digest(dict(tokens=p['token_ids'],mask=p['attention_mask']))
        before=len(driver.channel.records)
        try:generate(driver,altered,cache)
        except ValueError as exc:
            frames=driver.channel.records[before:]
            require(not any(f['header']['kind'] in ('HIDDEN','DELTAS') for f in frames),'invalid HIT performed hidden recomputation')
            controls.append(dict(case=name,request_id=altered['request_id'],status='EXPECTED_REJECTION',tensor_frames=0,reason=str(exc)))
        else:raise ValueError('invalid C1 HIT accepted: '+name)
    reject('no_match',lambda p:p.update(token_ids=[driver.vocab-1]*len(p['token_ids'])))
    reject('one_token',lambda p:p['token_ids'].__setitem__(task['prefix_length']//2,(p['token_ids'][task['prefix_length']//2]+1)%driver.vocab))
    reject('mask',lambda p:p['attention_mask'].__setitem__(0,0))
    reject('positions',lambda p:p['positions'].__setitem__(0,99))
    reject('digest',lambda p:p.update(prompt_sha256='0'*64))
    saved=cache._pm['identity']['adapter'];cache._pm['identity']['adapter']='NEGATIVE_OTHER_ADAPTER'
    try:reject('adapter_identity')
    finally:cache._pm['identity']['adapter']=saved
    key=next(k for k,owner in cache.entries.items() if next(iter(owner.pool.entries.values())).qkv_metadata['source_request_id']==task['source_request_id'])
    saved=deepcopy(cache.proofs[key]);cache.proofs[key]['prefix_length']+=1
    try:reject('proof_corruption')
    finally:cache.proofs[key]=saved
    cache.request_id='NEG_EVICTION';cache.remove(key,'EVICT','explicit invalid-HIT negative control')
    reject('evicted')
    write(driver.output/'r4c_negative_controls.json',controls)
    return controls


def execute(plan,output):
    output=Path(output);result=dict(execution_status='RUNNING',stage='PREPARED_INPUT_VALIDATION',**flags(plan['fixture']));rows=[];channel=ud=None
    try:
        manifest,prompts,tasks=load_prepared(plan['prepared']);require(manifest['TEST_FIXTURE_ONLY']==plan['fixture'],'C0/C1 fixture scope differs')
        pm,ref,am,old,tokenizer,gpu=setup(plan,native=False)
        driver=QualityDriver(old.identity,old.device,model=old.model,adapter=old.adapter,tiny=old.tiny,native=False)
        if driver.fixture:driver.tiny.matrices=None;tok=manifest['model_tokenizer']
        else:
            driver.model_tokenizer=tokenizer;tok=opt_tokenizer(pm,tokenizer).identity
        require(tok==manifest['model_tokenizer'],'actual model/tokenizer fingerprint differs from prepared dataset')
        if not driver.fixture:require({k:getattr(tokenizer,k) for k in manifest['special_tokens']}==manifest['special_tokens'],'actual tokenizer special IDs differ from generation policy')
        driver.configure(output,budget_bytes=plan['evidence_budget_bytes'])
        require(driver.context_limit==manifest['context_limit'],'actual model positional limit differs from C0 config')
        for task in tasks:
            require(len(task['prompt']['token_ids'])+task['policy']['max_new_tokens']<=driver.context_limit,'actual model positional limit exceeded')
        fixture=driver.fixture;control_count=0
        def control(message):
            nonlocal control_count
            rid=f'C_CONTROL_{control_count:05d}';control_count+=1
            channel.send(header('HELLO',rid,-1,**message));h,_,_=channel.recv(kind='READY',request_id=rid,layer_id=-1)
            require(h['metadata']['control_sha256']==digest(message) and h['metadata']['user_device_cuda_used'] is False,'UD C1 control receipt differs')
            append(output/'es_prefix_controls.jsonl',dict(message,request_id=rid))
        cache=PrefixCache(pm,plan['base_manifest_sha256'],'TEST_FIXTURE_ONLY' if fixture else gpu['uuid'],driver.device,plan['capacity_bytes'],
            tokenizer=tok,sources=plan['sources'],fixture=fixture,contract=driver.contract,observed_cache_type=None,
            synchronize=driver.synchronize,control=control,trace=lambda e:append(output/'r4c_cache_events.jsonl',e))
        write(output/'r4c_opt_contract.json',dict(implementation=driver.contract,cohort_sha256=cache.cohort,tokenizer=tok,
            identity=pm['identity'],identity_sha256=pm['identity_sha256'],gpu=gpu))
        result['stage']='UD_START';session=uuid.uuid4().hex
        ud=launch_ud(dict(plan,session_id=session,cohort_sha256=cache.cohort,context_limit=driver.context_limit),output)
        info=ready(ud,output/'r4c_ud_ready.json');require(info['pid']==ud[0].pid!=os.getpid(),'independent UD PID mismatch')
        channel=Channel(connect(info['port']),session,am['matrix_sha256'],timeout=180.,trace=lambda r:append(output/'es_wire.jsonl',r));driver.channel=channel
        channel.send(header('HELLO','handshake',-1,application=VERSION,cohort_sha256=cache.cohort,identity_sha256=pm['identity_sha256']))
        h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(h['metadata']['application']==VERSION and h['metadata']['pid']==ud[0].pid and h['metadata']['cohort_sha256']==cache.cohort and h['metadata']['user_device_cuda_used'] is False,'invalid C1 handshake')
        start=time.monotonic()
        for task in tasks:
            require(time.monotonic()-start<7200,'bounded C1 worker wall deadline')
            result.update(stage='REQUEST_'+task['mode'],request_id=task['request_id'],dataset=task['dataset'],sample_id=task['sample_id']);write(output/'r4c_stage.json',result)
            if not fixture:
                import torch
                torch.cuda.reset_peak_memory_stats(driver.device)
            row=generate(driver,task,cache)
            if not fixture:
                row['cuda_memory']=dict(allocated_bytes=torch.cuda.memory_allocated(driver.device),peak_allocated_bytes=torch.cuda.max_memory_allocated(driver.device),reserved_bytes=torch.cuda.memory_reserved(driver.device))
            blob=row.pop('logits_blob');row['logits_file']=driver.evidence.blob(blob,'logits/'+row['request_id']+'.f32.bin')
            steps=row.pop('steps');row.update(es_pid=os.getpid(),ud_pid=ud[0].pid,cohort_sha256=cache.cohort,
                model_identity_sha256=pm['identity_sha256'],adapter_matrix_sha256=am['matrix_sha256'],
                tokenizer_sha256=digest(tok),evaluation_policy_sha256=digest(task['policy']),
                logits_sha256=digest_bytes(blob),source_creation_cost_ms=row['e2e_ms'] if task['mode']=='COLD_MISS' else None,**flags(fixture))
            append(output/'r4c_requests.jsonl',row);rows.append(row);driver.evidence.check()
        result['stage']='INVALID_HIT_NEGATIVE_CONTROLS';negatives=negative_controls(driver,cache,tasks[-1]);driver.evidence.check()
        result['stage']='SHUTDOWN';channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
        channel.close();channel=None;ud[0].wait(timeout=10);require(ud[0].returncode==0,'C1 UD exited unsuccessfully')
        if not fixture:
            from semcache.system_cost.es_base_profile import assert_no_lora
            assert_no_lora(driver.model,driver.adapter)
            require(full_base_fingerprint(driver.model)['sha256']==pm['identity']['base_weight_sha256'],'base weights mutated')
        result.update(execution_status='COMPLETE',stage='COMPLETE',pid=os.getpid(),ud_pid=ud[0].pid,session_id=session,requests=len(rows),
            generation_steps=sum(r['generated_tokens'] for r in rows),gpu_inference_performed=not fixture,user_device_cuda_used=False,
            cache_final=cache.snapshot(),tensor_evidence_bytes=driver.evidence.bytes,evidence_directory_bytes=driver.evidence.check(),gpu=gpu,negative_controls=len(negatives))
        return result
    except BaseException as exc:
        result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}');write(output/'r4c_failure.json',result);raise
    finally:
        if channel is not None:channel.close()
        if ud is not None:result['ud_lifecycle']=reap(ud,output/'ud_lifecycle.json',group=False)
        result['partial_requests']=len(rows);write(output/'r4c_partial.json',result)
