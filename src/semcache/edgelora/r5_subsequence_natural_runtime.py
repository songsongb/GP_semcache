"""One original NATURAL block. Original D5 arithmetic and protocol are reused."""
from pathlib import Path
import os,sys,time,subprocess,uuid,json,shutil
from .communication import require,Channel,connect,header
from .r3a_artifacts import ROOT,read,append
from .r5_subsequence_natural_io import output,put,seal,verify,RequestWriter,save_result
from .r5_subsequence_natural import approval,CONDITIONS
from .r5_subsequence_pilot import wait_worker

SCRIPT=ROOT/'scripts/153_r5_subsequence_natural.py'

def ledger(root,record,request_ids,*,kind='wire'):
    unbound=False
    if kind=='wire':
        h=record.get('header');pid=h.get('request_id') if isinstance(h,dict) else None
        if not isinstance(pid,str) or not pid or Path(pid).name!=pid or pid in ('.','..'):
            require(record.get('complete') is False and record.get('error'),'missing/unsafe request ID in successful wire record')
            # File routing only. Keep the incomplete header and original error
            # unchanged; this is never a synthesized protocol request/header.
            pid='FAILED_FRAME';unbound=True
    else:pid=record.get('context',{}).get('pass_id',record.get('control',{}).get('request',{}).get('request_id','CONTROL'))
    rid=None if unbound else next((r for r in request_ids if pid==r or pid.startswith(r+'_')),None)
    name='CONTROL' if rid is None else rid
    # One pass/frame group per file; no giant cross-request JSONL.
    d=Path(root)/kind/name;d.mkdir(parents=True,exist_ok=True)
    append(d/(pid+'.jsonl'),record)


def stop_adapter(session,control):
    """Close after this adapter's final validated completion ACK, before admission."""
    req=control['request']
    require(not session.get('stop_started') and session['requests'] and req==session['requests'][-1] and
        [r['request_id'] for r in session['requests']]==session['ids'],'premature/duplicate adapter stop')
    session['stop_started']=True
    channel=session['channel'];channel.send(header('STOP','shutdown',-1))
    h,b,_=channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
    require(not b and h['dtype']=='none' and h['shape']==[] and h['metadata']=={},'invalid adapter shutdown acknowledgement')
    channel.close();session['stopped']=True


def join_adapter(session):
    require(session.get('stopped') is True,'adapter completion/STOP acknowledgement missing')
    session['process'].wait(timeout=60)
    require(session['process'].returncode==0,'UD failed after STOP acknowledgement')


def adapter_completion_callback(session,req):
    return (lambda control:stop_adapter(session,control)) if req['request_id']==session['ids'][-1] else None

def ud(config,output_dir,*,_implementation=None):
    from .r5_multi_artifacts import validate,load as load_matrices
    from .user_device import configure_cpu
    from .communication import listener
    from .r5_sparse_qkv import serve_sparse
    from .r5_subsequence_natural import implementation
    c=read(config);out=output(output_dir,(config,c['multi']));writer=RequestWriter(out,budget_bytes=c['budget'],request_ids=c['request_ids'])
    primary=None;connection=None
    try:
        m,pm,specs,registry=validate(c['multi']);cpu=configure_cpu(True);spec=specs[c['identity']['adapter_id']]
        require(c['implementation']==(implementation() if _implementation is None else _implementation()),'UD source differs')
        require(spec['matrix_sha256']==c['identity']['adapter_matrix_fingerprint'] and c['identity']['base_fingerprint']==pm['identity']['base_weight_sha256'] and c['identity']['tokenizer']==m['tokenizer'] and c['identity']['model_revision']==pm['identity']['base_revision'],'UD model/adapter differs')
        matrices=load_matrices(c['multi'],spec,torch_backend=True)
        import torch
        require(not torch.cuda.is_initialized() and 'transformers' not in sys.modules and all(t.device.type=='cpu' for t in matrices.values()),'UD isolation differs')
        from .r5_i5r2b_capture import backend
        put(out/'initial.json',dict(backend=backend(),pid=os.getpid(),cpu=cpu,source=c['implementation']['source'],identity=c['identity'],session=c['session'],matrix_sha256=spec['matrix_sha256'],user_device_cuda_used=False,full_model_loaded=False))
        sock=listener();put(out/'ready.json',dict(port=sock.getsockname()[1]));connection,address=sock.accept();sock.close()
        require(address[0]=='127.0.0.1','localhost required')
        channel=Channel(connection,c['session'],spec['matrix_sha256'],timeout=1800,trace=lambda r:ledger(out,r,c['request_ids']))
        serve_sparse(channel,c['identity'],registry,matrices,evidence=writer,trace=lambda r:ledger(out,r,c['request_ids'],kind='layers'))
        writer.check();put(out/'status.json',{'status':'COMPLETE'})
    except BaseException as exc:primary=exc;raise
    finally:
        if connection:connection.close()
        try:
            if primary:put(out/'status.json',dict(status='FAILED',reason=repr(primary)))
            seal(out)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note('UD finalization: '+repr(exc))

def worker(experiment,output_dir,condition,cpu_audit,expected_gpu_uuid=None,*,_profile=None):
    p,pr,rest,ar=(approval(experiment,cpu_audit) if _profile is None else _profile.approval(experiment,cpu_audit));block,m,pm,specs,registry,tok=rest
    require(condition in (CONDITIONS if _profile is None else _profile.conditions),'unknown NATURAL condition')
    mode,allow_approximate=('EXACT_CONTEXT_VERIFIED',False) if _profile is None else _profile.execution(condition)
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID') and os.environ.get('R5_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')=='1','manual opted-in allocation required')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536 and len(os.sched_getaffinity(0))>=4,'4CPU/64GiB required')
    require(os.environ.get('D5N_NODE') in ('moana-y3','moana-y4') and os.environ.get('SLURMD_NODENAME',os.environ.get('SLURM_JOB_NODELIST'))==os.environ['D5N_NODE'],'node differs')
    require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','offline/backend differs')
    protected=(experiment,cpu_audit,p['multi'],p['original_experiment'],*([] if _profile is None else p['dataset_paths'].values()))
    out=output(output_dir,protected);status=dict(status='FAILED',condition=condition);primary=None;sessions={};begin=time.monotonic_ns()
    try:
        from .r4a_runtime import setup
        from .r4b2_kv import installed_contract
        from .r5_multi_artifacts import runtime_view,physical,tokenizer_identity
        from .r5_i5_gpu import receipt
        from .r5_i5_contracts import request
        from .r5_runtime import encoder_for
        from .r5_semantic_index import SemanticIndex
        from .r5_perf_schedule import TINYBERT_REVISION
        from .r5_subsequence_physical import PhysicalOwner,EXACT
        from .r5_subsequence_integrity import SCOPED
        from .r5_sparse_qkv import VERSION
        from .r5_subsequence_generation import generate
        from .r5_subsequence_natural import NaturalProfiler
        from .r5_i5_quality import score
        import importlib.util
        loader=importlib.util.spec_from_file_location('natural_host_preflight',ROOT/'scripts/119_run_r5_i5.py');legacy=importlib.util.module_from_spec(loader);loader.loader.exec_module(legacy)
        require(legacy.cpu_mem_available_bytes()>=p['resources']['host_available_required_bytes'],'32GiB available host RAM required')
        _,_,_,driver,tokenizer,gpu=setup(dict(base_prepared=p['base_prepared'],fixture=False,device='cuda:0',i5_gpu_policy_scope='R5_I5'),native=False,gpu_identity_policy='SAME_COMPATIBLE_GPU_CLASS')
        import torch
        require(expected_gpu_uuid is None or gpu['uuid']==expected_gpu_uuid,'A/C GPU changed')
        gpu_receipt=receipt(pm,gpu,'SAME_COMPATIBLE_GPU_CLASS',fixture=False)
        require(torch.cuda.device_count()==1 and driver.model.config.max_position_embeddings==2048 and torch.cuda.mem_get_info(0)[0]>=12*2**30,'one2048-context GPU with12GiB free reserve required')
        require(tokenizer_identity(pm,tokenizer)==p['tokenizer_identity'],'actual tokenizer differs')
        torch.cuda.reset_peak_memory_stats(0)
        native_contract=installed_contract(driver);native_contract.update(sparse_protocol=VERSION,impact='streamed_original_M_row_l2_sum',generation_scope='native full prefill and native decode')
        tasks=block['tasks'];encoder=None if condition=='FULL_RECOMPUTE' else encoder_for({'fixture':False,'semantic_revision':TINYBERT_REVISION},tasks)
        for aid in sorted({t['adapter_id'] for t in tasks}):
            spec=specs[aid];identity,contract=physical(runtime_view(pm,spec),p['tokenizer_identity'],driver.device,gpu['uuid'],dict(native_contract),spec,registry)
            d=out/aid;d.mkdir();(d/'es').mkdir();ids=[t['request_id'] for t in tasks if t['adapter_id']==aid]
            config=dict(multi=p['multi'],identity=identity,implementation=p['implementation'],session=uuid.uuid4().hex,budget=p['resources']['per_adapter_ud_budget_bytes'],request_ids=ids)
            put(d/'ud_config.json',config);log=(d/'ud.log').open('w')
            proc=subprocess.Popen([sys.executable,str(SCRIPT if _profile is None else _profile.script),'_ud','--config',str(d/'ud_config.json'),'--output-dir',str(d/'ud')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
            s=dict(process=proc,log=log,channel=None,owner=None,identity=identity,directory=d,ids=ids,requests=[]);sessions[aid]=s
            deadline=time.monotonic()+120
            while not (d/'ud/ready.json').exists():require(proc.poll() is None and time.monotonic()<deadline,'UD startup failed: '+str(d/'ud.log'));time.sleep(.05)
            writer=RequestWriter(d/'es',budget_bytes=p['resources']['per_adapter_es_budget_bytes'],request_ids=ids)
            channel=Channel(connect(read(d/'ud/ready.json')['port']),config['session'],spec['matrix_sha256'],timeout=1800,trace=lambda r,d=d,ids=ids:ledger(d/'es',r,ids))
            s.update(writer=writer,channel=channel,index=None if encoder is None else SemanticIndex(encoder,clusters=1))
            s['owner']=None if encoder is None else PhysicalOwner(identity,registry,evidence=writer,trace=lambda r,d=d:append(d/'es/policy.jsonl',r),validation_mode=SCOPED)
            put(d/'identity.json',dict(physical=identity,contract=contract,model_identity=runtime_view(pm,spec)['identity'],registry=registry,source=p['implementation']['source'],plan_inventory=pr,condition=condition,validation_mode=SCOPED,semantic_encoder=None if encoder is None else encoder.metadata))
        from .r5_i5r2b_capture import backend
        effective_backend=backend();initialized=time.monotonic_ns();sequence=[]
        for task in tasks:
            request_arrival=time.monotonic_ns()
            s=sessions[task['adapter_id']];req=request(task,task['token_ids'],task['policy'],s['identity']);s['requests'].append(req);s['writer'].current=req['request_id']
            # No future sparse use on this connection after its final request
            # ACK. Do not wait for local admission, evidence I/O or other adapters.
            on_tcp_complete=adapter_completion_callback(s,req)
            result=generate(driver.model,driver.adapter,tokenizer,req,s['channel'],s['owner'],s['index'],condition='FULL_RECOMPUTE' if condition=='FULL_RECOMPUTE' else 'SEMCACHE_SUBSEQUENCE',mode=mode,allow_approximate=allow_approximate,evidence=s['writer'],profiler=NaturalProfiler(cuda_events=False,device=driver.device),on_tcp_complete=on_tcp_complete)
            result.update(validation_mode=SCOPED,natural_condition=condition)
            result['diagnostics']['validation_mode']=SCOPED;phys=result['completion']['physical']
            result['diagnostics']['integrity_validation']=None if phys is None else {k:v for k,v in phys['integrity_validation'].items() if k!='authorizations'}
            data=result.pop('logits_blob');result['logits']=dict(file=s['writer'].blob(data,req['request_id']+'_logits.bin'),dtype='float32',shape=result['logits_shape'],sha256=__import__('hashlib').sha256(data).hexdigest())
            result.update(sample_id=task['sample_id'],scored=task['scored'],quality=score(task['prepared']['sample'],result['generated_text'],task['prepared']['template_version']) if task['scored'] else None)
            # Persistence is inside the primary wall scope. The small timing sidecar
            # and final inventory sealing are outside and separately stated.
            generated_end=result['completion_ns'];first_available=result['steps'][0]['available_ns']
            save_result(s['directory']/'es',result);s['writer'].check()
            require(sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<p['resources']['condition_limit_bytes'],'condition disk budget exceeded')
            if _profile is not None:_profile.check_run_budget(out.parent,p['resources'])
            end=time.monotonic_ns()
            timing=dict(request_id=req['request_id'],arrival_ns=request_arrival,generation_scope_arrival_ns=result['arrival_ns'],completion_ns=end,application_e2e_ms=(end-request_arrival)/1e6,post_generation_persistence_quality_ms=(end-generated_end)/1e6,application_ttft_ms=(first_available-request_arrival)/1e6)
            put(s['directory']/'es'/req['request_id']/'timing.json',timing);sequence.append(dict(adapter_id=task['adapter_id'],**timing))
        for s in sessions.values():
            join_adapter(s)
            put(s['directory']/'requests.json',s['requests']);s['writer'].check()
        require(torch.cuda.max_memory_allocated(0)<22*2**30,'actual22GiB allocation bound exceeded')
        require(backend()==effective_backend,'backend changed during condition')
        put(out/'identity.json',dict(backend=effective_backend,gpu=gpu,gpu_identity_receipt=gpu_receipt,source=p['implementation']['source'],plan_inventory=pr,condition=condition,initialization_ms=(initialized-begin)/1e6,warmup_requests=0,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),ordering=p.get('ordering','original I5 tasks; no priming; one model and per-adapter CPU UD/cache/index'),measurement='strict observer overhead included; model init, final inventory sealing and small timing sidecars separate'))
        put(out/'sequence.json',sequence);status['status']='COMPLETE'
    except BaseException as exc:primary=exc;status['reason']=repr(exc);raise
    finally:
        for s in sessions.values():
            try:
                if s['owner'] is not None:s['owner'].dispose()
                if s['channel'] is not None:s['channel'].sock.close()
                if s['process'].poll() is None:s['process'].terminate();s['process'].wait(timeout=30)
                s['log'].close()
            except BaseException as exc:
                if primary is None:primary=exc;status.update(status='FAILED',reason=repr(exc))
                elif hasattr(primary,'add_note'):primary.add_note('cleanup: '+repr(exc))
        try:put(out/'status.json',status);seal(out)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note('finalization: '+repr(exc))
        if status['status']=='FAILED' and primary is not None and sys.exc_info()[0] is None:raise primary

def run(experiment,output_dir,source_commit,cpu_audit,enable_gpu_experiment=False):
    p,pr,rest,ar=approval(experiment,cpu_audit)
    require(enable_gpu_experiment and source_commit==p['implementation']['source']['commit'],'explicit pinned GPU opt-in required')
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID') and os.environ.get('R5_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')=='1','manual non-array allocation required')
    require(shutil.disk_usage(Path(output_dir).parent).free>=p['resources']['required_free_disk_bytes'],'96GiB free disk required')
    out=output(output_dir,(experiment,cpu_audit,p['multi'],p['original_experiment']));status=dict(status='FAILED',source=p['implementation']['source'],plan_inventory=pr,cpu_audit_path=str(Path(cpu_audit).resolve()),cpu_audit_inventory=ar,children=[]);primary=None;gpu=None
    try:
        for c in p['condition_order']:
            start=time.monotonic_ns();argv=[sys.executable,str(SCRIPT),'_worker','--experiment',str(experiment),'--cpu-audit',str(cpu_audit),'--condition',c,'--output-dir',str(out/c)]
            if gpu:argv+=['--expected-gpu-uuid',gpu]
            rc=wait_worker(argv,timeout=p['resources']['worker_timeout_seconds']);status['children'].append(dict(condition=c,returncode=rc,process_lifecycle_ms=(time.monotonic_ns()-start)/1e6))
            require(rc==0,'NATURAL child failed; no retry/fallback');gpu=read(out/c/'identity.json')['gpu']['uuid']
        status['status']='COMPLETE'
    except BaseException as exc:primary=exc;status['reason']=repr(exc);raise
    finally:
        try:put(out/'status.json',status);seal(out)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note('run finalization: '+repr(exc))
    return status
