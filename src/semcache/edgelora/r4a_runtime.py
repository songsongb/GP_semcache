"""Separate Native preparation and persistent bare ES / independent dynamic UD."""
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .communication import Channel,connect,digest,digest_bytes,header,require
from .r3a_artifacts import (append,check_parity,full_base_fingerprint,load_matrices,pack,process_environment,read,sha,
    tensor_bytes,unpack,validate_prepared,versions,write)
from .r3c_process import ready,reap
from .r4a_cache import PromptCache
from .r4a_evidence import flags,load_prepared,make_suite,planned_counts,schedule,verify_sources
from .r4a_generation import ForwardDriver,generate
from .r4a_fixture import TinyBackend
from .r4a_parity import capture,compare


def launch(kind,plan,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);write(output/(kind+'_plan.json'),dict(plan,output=str(output)))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    if kind=='ud':env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
    out=(output/(kind+'.stdout.log')).open('w');err=(output/(kind+'.stderr.log')).open('w')
    try:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve().parents[3]/'scripts/102_run_r4a_opt_full_e2e.py'),
        '--worker',kind,'--plan',str(output/(kind+'_plan.json')),'--test-fixture-only' if plan['fixture'] else '--enable-gpu-experiment'],
        cwd=Path(__file__).resolve().parents[3],env=env,stdout=out,stderr=err,start_new_session=False)
    except BaseException:out.close();err.close();raise
    return p,out,err


def setup(plan,native=False):
    pm,ref,am=validate_prepared(plan['base_prepared']);fixture=plan['fixture']
    require(pm.get('TEST_FIXTURE_ONLY',False)==fixture,'base preparation fixture scope mismatch')
    identity=pm['identity'];gpu=None;model=adapter=tokenizer=tiny=None
    if fixture:
        base=read(Path(plan['base_prepared'])/'r3a_tiny_base.json');require(digest(base)==identity['base_weight_sha256'],'fixture base SHA differs')
        matrices=load_matrices(plan['base_prepared'],am,torch_backend=False);tiny=TinyBackend(base,matrices)
    else:
        require(identity['model']=='facebook/opt-2.7b' and identity['layers']==32 and identity['hidden_size']==2560 and
            identity['dtype']=='float16' and identity['seed']==42,'R4-A requires the frozen OPT-2.7B FP16 rank8 seed42 cohort')
        import torch
        from .r3a_runtime import runtime_setup,memory,load_bare
        runtime_setup(plan['device']);torch.manual_seed(identity['seed'])
        available=sorted(os.sched_getaffinity(0))[:4];os.sched_setaffinity(0,available);torch.set_num_threads(len(available));torch.set_num_interop_threads(1)
        gpu=memory(plan['device']);require(gpu['uuid']!='UNAVAILABLE' and gpu['uuid']==pm['gpu']['uuid'] and '3090' in gpu['name'],'fresh allocated RTX3090 UUID required')
        require(versions()==identity['versions'],'prepared/runtime versions differ')
        if native:
            from .parity_forensics_probe import load_native
            model,adapter,matrices=load_native(dict(prepared=plan['base_prepared'],device=plan['device']),pm,am)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(identity['tokenizer_source'],revision=identity['tokenizer_revision'],local_files_only=True)
        else:
            from semcache.models.model_adapter import OPTModelAdapter
            from semcache.system_cost.es_base_profile import assert_no_lora
            model,tokenizer,meta=load_bare(identity,plan['device']);adapter=OPTModelAdapter(model);assert_no_lora(model,adapter)
            require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'bare model fingerprint differs')
    driver=ForwardDriver(identity,'cpu' if fixture else plan['device'],model=model,adapter=adapter,tiny=tiny,native=native)
    return pm,ref,am,driver,tokenizer,gpu


def store_generation(row,driver,output,meta):
    blob=row.pop('logits_blob');steps=row.pop('steps');rid=row['request_id'];name=f'logits/{rid}.f32.bin'
    (output/'logits').mkdir(exist_ok=True);(output/name).write_bytes(blob)
    row.update(meta,logits_file=name,**flags(driver.fixture))
    row['result_sha256']=digest(dict(tokens=row['generated_token_ids'],text=row['generated_text'],prompt=row['prompt_sha256'],policy=row['policy']))
    if not driver.fixture:
        from .r3a_runtime import memory
        row['gpu_memory']=memory(driver.device)
    for step in steps:append(output/'r4a_tokens.jsonl',dict(step,request_id=rid))
    append(output/'r4a_generation_outputs.jsonl',row);append(output/'r4a_requests.jsonl',row)
    return row,steps


def native_prepare(plan,output):
    pm,ref,am,driver,tokenizer,gpu=setup(plan,native=True);output=Path(output)
    if not driver.fixture:driver.model_tokenizer=tokenizer
    suite=make_suite(tokenizer,fixture=driver.fixture,scope=plan['scope'],policy_mode=plan['policy']);write(output/'r4a_suite.json',suite)
    original_prompt=dict(token_ids=pm['identity']['token_ids'],attention_mask=[1]*pm['identity']['length'])
    controls={}
    for use_cache in (False,True):
        directory=output/f'triage_native_{int(use_cache)}';directory.mkdir();probes=[]
        observer=None if driver.fixture else lambda a,d=directory:capture(a,d)
        if driver.fixture:probe_context=nullcontext()
        else:
            from .parity_forensics_probe import same_hidden_probes
            matrices=load_matrices(plan['base_prepared'],am,torch_backend=True)
            probe_context=same_hidden_probes(driver.adapter,matrices,pm['identity'],{0},directory)
        # Install probes inside observer by combining contexts; neither replaces
        # native outputs. Same-hidden probes are only for this unmeasured triage.
        if not driver.fixture:
            from contextlib import contextmanager
            @contextmanager
            def observer(a,d=directory,pc=probe_context):
                with capture(a,d),pc as result:
                    probes_ref=result[0];yield
                probes.extend(probes_ref)
        if not driver.fixture:write(output/'gpu_attempt.json',dict(gpu_inference_attempted=True))
        out,record,_=driver.forward(f'NATIVE_TRIAGE_{int(use_cache)}',0,original_prompt['token_ids'],original_prompt['attention_mask'],None,1,
            use_cache=use_cache,triage=True,observer=observer)
        if not driver.fixture:write(output/'gpu_execution.json',dict(gpu_inference_performed=True,completed_native_forward=True))
        raw=pack(out.logits,'float32') if driver.fixture else tensor_bytes(out.logits.float())
        shape=out.shape if driver.fixture else list(out.logits.shape);del out
        (directory/'logits.f32.bin').write_bytes(raw)
        value=dict(logits_file='logits.f32.bin',logits_sha256=digest_bytes(raw),shape=shape,use_cache=use_cache,
            model_logits_dtype=record['model_logits_dtype'],probes=probes)
        if not use_cache and not driver.fixture:
            original=unpack((Path(plan['base_prepared'])/ref['logits_file']).read_bytes(),'float32')
            value['vs_original_95']=check_parity(unpack(raw,'float32'),original,shape)
            value['original_bitwise_equal']=value['logits_sha256']==ref['logits_sha256']
            value['dtype_matches_original']=record['model_logits_dtype']==ref['native_logits_dtype'].removeprefix('torch.')
        write(directory/'forward.json',value);controls[str(use_cache)]=value
    native_outputs=[]
    for cell in suite['cells']:
        for prompt in (p for p in suite['prompts'] if p['length']==cell['length']):
            rid=f'N_{cell["id"]}_{prompt["sample_id"]}'
            row=generate(driver,prompt,cell['policy'],rid)
            row,steps=store_generation(row,driver,output,dict(cell_id=cell['id'],sample_id=prompt['sample_id'],comparison_id=rid,
                phase='native_reference',block=None,source='TEST_FIXTURE_ONLY_CPU_CONTROL' if driver.fixture else 'MATCHED_NATIVE_PEFT'))
            native_outputs.append(row)
    write(output/'native_index.json',dict(controls=controls,generations=native_outputs))
    if not driver.fixture:
        from .r3a_runtime import memory
        require(full_base_fingerprint(driver.model.get_base_model(),peft_wrapped=True)['sha256']==pm['identity']['base_weight_sha256'],
            'Native base weights changed during references')
        original={(r['layer'],r['role'],r['matrix']):r for r in am['matrices']}
        for layer in range(32):
            for role,module in driver.adapter.projection_modules(layer).items():
                for name,part in (('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                    require(digest_bytes(tensor_bytes(part.weight))==original[layer,role,name]['sha256'],'Native A/B changed during reference execution')
        gpu=memory(driver.device)
    return dict(execution_status='COMPLETE',pid=os.getpid(),gpu=gpu,identity_sha256=pm['identity_sha256'],
        adapter_matrix_sha256=am['matrix_sha256'],heads=driver.heads,environment=process_environment(),native_peft_executed=not driver.fixture,
        tokenizer_vocab_sha256='TEST_FIXTURE_ONLY' if driver.fixture else digest(tokenizer.get_vocab()),
        gpu_inference_performed=not driver.fixture,**flags(driver.fixture))


def execute(plan,output):
    prepared,suite,pm,ref,am=load_prepared(plan['prepared']);pm,ref,am,driver,tokenizer,gpu=setup(plan,native=False)
    output=Path(output)
    if not driver.fixture:
        driver.model_tokenizer=tokenizer
        require(digest(tokenizer.get_vocab())==prepared['native_result']['tokenizer_vocab_sha256'] and
            {k:getattr(tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}==suite['special_tokens'],
            'Native/ES tokenizer vocabulary or special IDs differ')
    router=PromptCache(pm,prepared['base_manifest_sha256'],'TEST_FIXTURE_ONLY' if driver.fixture else gpu['uuid'],driver.device,fixture=driver.fixture)
    ud=None;channel=None;rows=[];all_steps={};completed=[];gates={};triage={};started=time.monotonic();result={}
    try:
        session=uuid.uuid4().hex;ud=launch('ud',dict(plan,session_id=session),output)
        ready_data=ready(ud,output/'r4a_ud_ready.json');require(ready_data['pid']==ud[0].pid!=os.getpid(),'UD independent PID mismatch')
        channel=Channel(connect(ready_data['port']),session,am['matrix_sha256'],timeout=120.,trace=lambda r:append(output/'es_wire.jsonl',r));driver.channel=channel
        channel.send(header('HELLO','handshake',-1,application='R4A_GENERATION_V1',identity_sha256=pm['identity_sha256']))
        h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(h['metadata']['pid']==ud[0].pid and h['metadata']['user_device_cuda_used'] is False and
            h['metadata']['identity_sha256']==pm['identity_sha256'],'UD handshake/identity isolation mismatch')
        for use_cache in (False,True):
            directory=output/f'triage_es_{int(use_cache)}';directory.mkdir()
            observer=None if driver.fixture else lambda a,d=directory:capture(a,d)
            ids=pm['identity']['token_ids'];mask=[1]*len(ids)
            if not driver.fixture:write(output/'gpu_attempt.json',dict(gpu_inference_attempted=True))
            out,record,_=driver.forward(f'ES_TRIAGE_{int(use_cache)}',0,ids,mask,None,1,use_cache=use_cache,triage=True,observer=observer)
            if not driver.fixture:write(output/'gpu_execution.json',dict(gpu_inference_performed=True,completed_distributed_forward=True))
            raw=pack(out.logits,'float32') if driver.fixture else tensor_bytes(out.logits.float());shape=out.shape if driver.fixture else list(out.logits.shape);del out
            (directory/'logits.f32.bin').write_bytes(raw)
            write(directory/'forward.json',dict(logits_file='logits.f32.bin',logits_sha256=digest_bytes(raw),shape=shape,use_cache=use_cache,
                model_logits_dtype=record['model_logits_dtype']))
            triage[str(use_cache)]=compare(Path(plan['prepared'])/f'triage_native_{int(use_cache)}',directory,unpack(raw,'float32'),shape,fixture=driver.fixture)
            append(output/'triage_passes.jsonl',record)
        native_ok=(not driver.fixture and all(v['strict_metrics']['passed'] and v['matched_model_logits_dtype'] for v in triage.values()) and
            read(Path(plan['prepared'])/'triage_native_0/forward.json').get('original_bitwise_equal') is True and
            read(Path(plan['prepared'])/'triage_native_0/forward.json').get('dtype_matches_original') is True)
        write(output/'r4a_parity_diagnostics.json',dict(use_cache_false=triage['False'],use_cache_true=triage['True'],native_equivalent_prefill=native_ok,
            frozen_thresholds_unchanged=True,instrumentation_outside_latency_trials=True))
        for ci,cell in enumerate(suite['cells']):
            if ci:
                if not driver.fixture:
                    from .r3a_runtime import memory
                    measured=memory(driver.device);elapsed=time.monotonic()-started
                    scale=planned_counts(suite)['expansion_admission_scale']
                    if measured['peak_allocated_bytes']>20*2**30 or measured['free_bytes']<2*2**30 or elapsed*scale+900>7200:
                        gates['expansion']=dict(status='SKIPPED_RESOURCE_CAP',observed_memory=measured,elapsed_seconds=elapsed,
                            resource_admission_scale=scale,scope='conservative declared token-work ratio; not latency/speedup evidence')
                        write(output/'cell_gates.json',gates);break
            cell_rows=[];expected=schedule(suite,cell)
            for index,item in enumerate(expected):
                require(time.monotonic()-started<7200,'bounded R4 serving time exhausted')
                prompt=next(p for p in suite['prompts'] if p['sample_id']==item['sample_id'])
                rid=f'R{len(rows):04d}';comparison=f'{cell["id"]}:{item["block"]}:{item["sample_id"]}'
                row=generate(driver,prompt,cell['policy'],rid,router=router,mode=item['mode'])
                row,steps=store_generation(row,driver,output,dict(item,cell_id=cell['id'],comparison_id=comparison,es_pid=os.getpid(),ud_pid=ud[0].pid,
                    tokenization_ms_outside_wall=prompt['tokenization_ms'],
                    identity_sha256=pm['identity_sha256'],adapter_matrix_sha256=am['matrix_sha256'],gpu_uuid='TEST_FIXTURE_ONLY' if driver.fixture else gpu['uuid']))
                rows.append(row);all_steps[rid]=steps;cell_rows.append(row)
                if (index+1)%3==0:
                    triad=cell_rows[-3:];baseline=next(r for r in triad if r['mode']=='NO_REUSE')
                    cold=next(r for r in triad if r['mode']=='COLD_MISS');warm=next(r for r in triad if r['mode']=='WARM_HIT')
                    require(warm['cache_source_request_id']==cold['request_id'] and warm['cache_context_sha256']==cold['cache_context_sha256'],
                        'resident HIT did not use this triad real completed cold fill')
                    for candidate in triad:
                        equal=(candidate['generated_token_ids']==baseline['generated_token_ids'] and candidate['generated_text']==baseline['generated_text'] and
                            candidate['logits_sha256']==baseline['logits_sha256'] and
                            [s['qkv_sha256'] for s in all_steps[candidate['request_id']]]==[s['qkv_sha256'] for s in all_steps[baseline['request_id']]])
                        check=dict(request_id=candidate['request_id'],reference_request_id=baseline['request_id'],bitwise_equal=equal)
                        if not equal and candidate['logits_shape']==baseline['logits_shape']:
                            check['quantified_strict_metrics']=check_parity(
                                unpack((output/candidate['logits_file']).read_bytes(),'float32'),
                                unpack((output/baseline['logits_file']).read_bytes(),'float32'),candidate['logits_shape'])
                        append(output/'fidelity_checks.jsonl',check)
                        require(equal,'R4A_RAW_QKV_E2E_FIDELITY_FAIL; no following triad/expansion')
                    append(output/'triad_gates.jsonl',dict(comparison_id=comparison,status='PASS',requests=[r['request_id'] for r in triad]))
            gates[cell['id']]=dict(status='PASS',requests=len(cell_rows));completed.append(cell['id']);write(output/'cell_gates.json',gates)
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();channel=None
        ud[0].wait(timeout=10.);require(ud[0].returncode==0,'UD process failed')
        if not driver.fixture:
            from semcache.system_cost.es_base_profile import assert_no_lora
            assert_no_lora(driver.model,driver.adapter);require(full_base_fingerprint(driver.model)['sha256']==pm['identity']['base_weight_sha256'],'ES weights mutated')
            from .r3a_runtime import memory
            gpu=memory(driver.device)
        result=dict(execution_status='COMPLETE',pid=os.getpid(),ud_pid=ud[0].pid,session_id=session,completed_cells=completed,cell_gates=gates,
            native_equivalent_prefill=native_ok,requests=len(rows),gpu=gpu,heads=driver.heads,environment=process_environment(),adapter_parameters_on_es=0,
            user_device_cuda_used=False,gpu_inference_performed=not driver.fixture,**flags(driver.fixture))
        return result
    finally:
        if channel is not None:channel.close()
        if ud is not None:result['ud_lifecycle']=reap(ud,output/'ud_lifecycle.json',group=False)
        result.update(partial_rows=len(rows),completed_cells=completed);write(output/'es_partial.json',result)
