"""Physical B1 ES GPU / independent CPU UD using frozen generation and math."""
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .communication import Channel,ProtocolError,connect,digest,digest_bytes,header,require
from .r3a_artifacts import append,full_base_fingerprint,pack,process_environment,read,tensor_bytes,unpack,write
from .r3c_process import ready,reap
from .r4a_evidence import load_prepared
from .r4a_parity import capture,compare
from .r4a_runtime import setup
from .r4b1_cache import MultiPromptCache
from .r4b1_evidence import entry_bytes,fidelity,flags,schedule
from .r4b1_generation import generate


def launch_ud(plan,output):
    write(output/'ud_plan.json',dict(plan,output=str(output)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    out=(output/'ud.stdout.log').open('w');err=(output/'ud.stderr.log').open('w')
    try:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve().parents[3]/'scripts/103_run_r4b1_multi_entry.py'),
        '--worker','ud','--plan',str(output/'ud_plan.json'),'--test-fixture-only' if plan['fixture'] else '--enable-gpu-experiment'],
        cwd=Path(__file__).resolve().parents[3],env=env,stdout=out,stderr=err)
    except BaseException:out.close();err.close();raise
    return p,out,err


def store(row,output,meta,fixture):
    blob=row.pop('logits_blob');steps=row.pop('steps');rid=row['request_id'];name=f'logits/{rid}.f32.bin'
    (output/'logits').mkdir(exist_ok=True);(output/name).write_bytes(blob)
    row.update(meta,logits_file=name,last_decode_available_ns=steps[-1]['available_monotonic_ns'],**flags(fixture))
    row['result_sha256']=digest(dict(tokens=row['generated_token_ids'],text=row['generated_text'],prompt=row['prompt_sha256'],policy=row['policy']))
    for s in steps:append(output/'r4b1_tokens.jsonl',dict(s,request_id=rid))
    append(output/'r4b1_requests.jsonl',row);append(output/'r4b1_generation_outputs.jsonl',row)
    return steps


def execute(plan,output):
    output=Path(output);prepared,suite,pm,ref,am=load_prepared(plan['prepared'])
    require(prepared['sources']==plan['sources'],'prepared and serving source revisions differ')
    pm,ref,am,driver,tokenizer,gpu=setup(plan,native=False);fixture=driver.fixture
    if fixture:driver.tiny.matrices=None  # Serving fixture also keeps all correction math on the independent UD.
    driver.diagnostics_path=output/'r4b1_logit_diagnostics.jsonl'
    if not fixture:driver.model_tokenizer=tokenizer
    tok=dict(vocab_sha256='TEST_FIXTURE_ONLY' if fixture else digest(tokenizer.get_vocab()),special_tokens=suite['special_tokens'],
        source=pm['identity'].get('tokenizer_source'),revision=pm['identity'].get('tokenizer_revision'))
    require(tok['vocab_sha256']==prepared['native_result']['tokenizer_vocab_sha256'],'actual ES tokenizer differs from Native')
    if not fixture:require({k:getattr(tokenizer,k) for k in suite['special_tokens']}==suite['special_tokens'],'tokenizer special IDs differ')
    gpu_uuid='TEST_FIXTURE_ONLY' if fixture else gpu['uuid'];size=entry_bytes(pm['identity'])
    require(plan['capacity_bytes']>=2*size,'main pilot capacity must hold both full-prompt entries')
    capacities=dict(main=plan['capacity_bytes'],pressure=size)
    pools={name:MultiPromptCache(pm,prepared['base_manifest_sha256'],gpu_uuid,driver.device,cap,tokenizer=tok,sources=plan['sources'],
        fixture=fixture,synchronize=driver.synchronize,trace=lambda e,pool=name:append(output/'r4b1_cache_events.jsonl',dict(e,pool_id=pool)))
        for name,cap in capacities.items()}
    contexts={digest(c):c for p in suite['prompts'][:2] for c in [pools['main'].context_for(p)]};write(output/'r4b1_contexts.json',contexts)
    rows=[];all_tokens=[];channel=ud=None;result={};triage={};started=time.monotonic()
    try:
        session=uuid.uuid4().hex;ud=launch_ud(dict(plan,session_id=session),output);info=ready(ud,output/'r4a_ud_ready.json')
        require(info['pid']==ud[0].pid!=os.getpid(),'independent UD PID required')
        channel=Channel(connect(info['port']),session,am['matrix_sha256'],timeout=120.,trace=lambda r:append(output/'es_wire.jsonl',r));driver.channel=channel
        channel.send(header('HELLO','handshake',-1,application='R4A_GENERATION_V1',identity_sha256=pm['identity_sha256']))
        h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(h['metadata']['pid']==ud[0].pid and h['metadata']['user_device_cuda_used'] is False and
            h['metadata']['identity_sha256']==pm['identity_sha256'],'UD identity/isolation handshake differs')
        for use_cache in (False,True):
            d=output/f'triage_es_{int(use_cache)}';d.mkdir();ids=pm['identity']['token_ids']
            if not fixture:write(output/'gpu_attempt.json',dict(gpu_inference_attempted=True))
            out,record,_=driver.forward(f'ES_TRIAGE_{int(use_cache)}',0,ids,[1]*len(ids),None,1,use_cache=use_cache,triage=True,
                observer=None if fixture else lambda a,d=d:capture(a,d))
            if not fixture:write(output/'gpu_execution.json',dict(gpu_inference_performed=True))
            raw=pack(out.logits,'float32') if fixture else tensor_bytes(out.logits.float());shape=out.shape if fixture else list(out.logits.shape)
            del out;(d/'logits.f32.bin').write_bytes(raw)
            write(d/'forward.json',dict(logits_file='logits.f32.bin',logits_sha256=digest_bytes(raw),shape=shape,
                use_cache=use_cache,model_logits_dtype=record['model_logits_dtype']))
            triage[str(use_cache)]=compare(Path(plan['prepared'])/f'triage_native_{int(use_cache)}',d,unpack(raw,'float32'),shape,fixture=fixture)
            append(output/'triage_passes.jsonl',record)
        native_ok=not fixture and all(v['strict_metrics']['passed'] and v['matched_model_logits_dtype'] for v in triage.values()) and \
            read(Path(plan['prepared'])/'triage_native_0/forward.json').get('original_bitwise_equal') is True and \
            read(Path(plan['prepared'])/'triage_native_0/forward.json').get('dtype_matches_original') is True
        write(output/'r4b1_parity_diagnostics.json',dict(use_cache_false=triage['False'],use_cache_true=triage['True'],
            native_equivalent_prefill=native_ok,frozen_thresholds_unchanged=True,native_parity_deferred=True))
        tasks=schedule(suite);write(output/'r4b1_schedule.json',tasks)
        for index,item in enumerate(tasks):
            require(time.monotonic()-started<7200,'bounded B1 worker time exhausted')
            if index==18:
                pools['main'].request_id='POOL_TRANSITION'
                for key in list(pools['main'].entries):pools['main'].remove(key,'INVALIDATE','release main cache before independent pressure experiment')
            if item['pool_id']=='pressure' and index==20:
                # A has actually been evicted by B. A requested HIT must stop
                # before a single projection or data frame, never recompute.
                p=next(p for p in suite['prompts'] if p['sample_id']==tasks[18]['sample_id']);before=len(channel.records)
                try:generate(driver,p,suite['cells'][0]['policy'],'REFUSED_A',pools['pressure'],'WARM_HIT')
                except ProtocolError as exc:
                    require('invalid HIT/MISS; no fallback counted as HIT' in str(exc),'unexpected pressure probe failure')
                    write(output/'r4b1_hit_refusal.json',dict(status='EXPECTED_HIT_REJECTED',reason=str(exc),recomputed=False,
                        new_wire_frames=len(channel.records)-before,context_sha256=digest(pools['pressure'].context_for(p))))
                else:require(False,'requested evicted HIT unexpectedly succeeded')
                require(len(channel.records)==before,'requested HIT miss performed physical recomputation')
            prompt=next(p for p in suite['prompts'] if p['sample_id']==item['sample_id']);pool=pools[item['pool_id']]
            row=generate(driver,prompt,suite['cells'][0]['policy'],item['request_id'],pool,item['mode'])
            steps=store(row,output,dict(item,es_pid=os.getpid(),ud_pid=ud[0].pid,identity_sha256=pm['identity_sha256'],
                adapter_matrix_sha256=am['matrix_sha256'],gpu_uuid=gpu_uuid),fixture)
            rows.append(row);all_tokens.extend(dict(s,request_id=row['request_id']) for s in steps)
            checks=fidelity(rows,all_tokens,output)  # Stop immediately on changed output, before following requests.
        for c in checks:append(output/'fidelity_checks.jsonl',c)
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();channel=None
        ud[0].wait(timeout=10.);require(ud[0].returncode==0,'UD process failed')
        if not fixture:
            from semcache.system_cost.es_base_profile import assert_no_lora
            from .r3a_runtime import memory
            assert_no_lora(driver.model,driver.adapter);require(full_base_fingerprint(driver.model)['sha256']==pm['identity']['base_weight_sha256'],'ES weights changed')
            gpu=memory(driver.device)
        result=dict(execution_status='COMPLETE',pid=os.getpid(),ud_pid=ud[0].pid,session_id=session,requests=len(rows),heads=driver.heads,
            gpu=gpu,gpu_uuid=gpu_uuid,tokenizer=tok,capacities=capacities,cache_final={p:c.snapshot() for p,c in pools.items()},
            native_equivalent_prefill=native_ok,completed_cells=[suite['cells'][0]['id']],environment=process_environment(),
            adapter_parameters_on_es=0,user_device_cuda_used=False,gpu_inference_performed=not fixture,**flags(fixture))
        return result
    finally:
        if channel is not None:channel.close()
        if ud is not None:result['ud_lifecycle']=reap(ud,output/'ud_lifecycle.json',group=False)
        result['partial_rows']=len(rows);write(output/'r4b1_partial.json',result)
