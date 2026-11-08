"""Bounded Qasper sparse-generation pilot; separate from frozen I5 runners."""
from pathlib import Path
import os,sys,time,subprocess,uuid
from .communication import require,digest,Channel,connect,header
from .r3a_artifacts import ROOT,read,write,sha,THRESHOLDS,append
from .r5_i5_evidence import seal,sources
from .r5_i5r1_plan import source_identity
from .r5_i5r_parallel import output,assets,provenance
from .r5_i5_workload import select,requests
from .r5_subsequence_physical import EXACT,APPROX,PhysicalOwner
from .r5_sparse_qkv import VERSION,serve_sparse
from .r5_i5r2_tensors import Evidence

CONDITIONS=('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_SUBSEQUENCE')

def wait_worker(command,*,timeout=3600):
    """Give a timed-out/interrupted worker a chance to seal and reap its UD."""
    child=subprocess.Popen(command,cwd=ROOT)
    try:return child.wait(timeout=timeout)
    except BaseException as primary:
        try:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        except BaseException as exc:
            if hasattr(primary,'add_note'):primary.add_note('worker cleanup: '+repr(exc))
        raise

def prepare(registry,multi,source_commit,output_dir,approximate=False):
    src=source_identity();require(src['commit']==source_commit,'reviewed source differs')
    paths,m,pm,specs,users,tok=assets(registry,multi);require('qasper' in paths,'DATASET_NOT_READY: qasper')
    samples,dm=provenance(paths['qasper'],'qasper')
    from .r5_i5_preflight import readiness
    ready=readiness({'qasper':paths['qasper']},base_prepared=m['base_prepared'],multi_prepared=multi,count=1,budgets={'qasper':2},input_limit=256,allow_truncation=True)
    require(ready['status']=='READY_FOR_MANUAL_GPU_SMOKE','offline CPU readiness failed: '+str(ready))
    prompts,_=select(samples,tok.tokenizer,count=1,seed=42,context_limit=2048,budget=2,prefix=64,allow_truncation=True,input_limit=256)
    p=prompts[0];p.update(strict=True,source_semantic_text=tok.tokenizer.decode(p['token_ids'][:64],skip_special_tokens=False,clean_up_tokenization_spaces=False))
    special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    tasks=requests(prompts,track='CONTROLLED',prefix=64,special=special)
    aid=tasks[0]['adapter_id'];require(tasks[0]['token_ids']==tasks[1]['token_ids'][:64],'exact source tokens differ')
    plan=dict(schema='R5_SUBSEQUENCE_GENERATION_PILOT_V1',source=src,registry_path=str(Path(registry).resolve()),registry_sha256=sha(registry),multi=str(Path(multi).resolve()),multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),
              dataset=str(Path(paths['qasper']).resolve()),dataset_inventory=Evidence(paths['qasper']).verify(),base_prepared=m['base_prepared'],adapter_id=aid,
              tasks=tasks,conditions=list(CONDITIONS),mode=APPROX if approximate else EXACT,allow_approximate=approximate,tokenizer_identity=tok.identity,
              classification='CONTROLLED_REAL_QASPER_PREFIX_DERIVED_SOURCE_W3_PROJECTION_REUSE',source_tokens=64,target_tokens=len(tasks[1]['token_ids']),target_generation_limit=2,
              budget_bytes_per_condition=8*2**30,readiness=ready,GPU_NOT_RUN=True,independent_contexts=1,warmup_requests=0,statistical_scope='DESCRIPTIVE_ONLY')
    from .r5_i5_workload import resources
    plan.update(thresholds=THRESHOLDS,impact_replay_fp64_tolerance=dict(atol=1e-12,rtol=1e-12),resources=resources(tasks,evidence_budget=8*2**30),validation_test_sha256=sha(ROOT/'tests/test_r5_subsequence_generation.py'))
    from .r5_i5r3b_transport import preflight
    # Original full request framing plus immutable headroom, then sparse native
    # framing remains independently enforced by Channel's unchanged64KiB cap.
    plan['transport']=preflight({'pilot':{'conditions':list(CONDITIONS),'blocks':[{'tasks':tasks}]}},pm,specs,m['tokenizer'],users)
    out=output(output_dir,(registry,multi,*paths.values()));write(out/'subsequence_plan.json',plan);seal(out)
    return dict(status='SEALED_CPU_PREPARED_GPU_NOT_RUN',mode=plan['mode'],source_tokens=64,target_tokens=plan['target_tokens'])

def load(experiment):
    ev=Evidence(experiment);receipt=ev.verify();p=ev.json('subsequence_plan.json')
    require(p['schema']=='R5_SUBSEQUENCE_GENERATION_PILOT_V1' and p['GPU_NOT_RUN'] is True and p['budget_bytes_per_condition']==8*2**30 and p['warmup_requests']==0,'pilot schema/resource profile differs')
    require(p['source']==source_identity(),'pinned source differs')
    require(Evidence(p['multi']).verify('r5_output_manifest.json')==p['multi_inventory'] and Evidence(p['dataset']).verify()==p['dataset_inventory'],'prepared input differs')
    require(p['conditions']==list(CONDITIONS) and p['source_tokens']==64 and 64<p['target_tokens']<=256 and p['target_generation_limit']==2,'bounded pilot differs')
    require(p['thresholds']==THRESHOLDS and p['impact_replay_fp64_tolerance']==dict(atol=1e-12,rtol=1e-12) and p['validation_test_sha256']==sha(ROOT/'tests/test_r5_subsequence_generation.py'),'threshold/test source differs')
    if 'diagnostics' in p:
        from .r5_subsequence_profile import VERSION as PROFILE_VERSION
        d=p['diagnostics'];require(d['version']==PROFILE_VERSION and type(d['cuda_events']) is bool and d['classification']=='DIAGNOSTIC_ONLY_UNCHANGED_ARITHMETIC','unknown diagnostic configuration')
        from .r5_subsequence_diagnostics import diagnostic_source_files
        expected_files=diagnostic_source_files(p['source']['commit'])
        if 'integrity_validation' in p:
            from .r5_subsequence_integrity import SCHEMA,MODES,THREAT_MODEL
            validation=p['integrity_validation'];name='tests/test_r5_subsequence_validation.py'
            require(validation==dict(schema=SCHEMA,mode=validation['mode'],threat_model=THREAT_MODEL,test_file=name,test_sha256=sha(ROOT/name)) and validation['mode'] in MODES,'sealed validation configuration differs')
            expected_files[name]=sha(ROOT/name)
            blob=subprocess.run(['git','show',p['source']['commit']+':'+name],cwd=ROOT,capture_output=True)
            require(blob.returncode==0 and __import__('hashlib').sha256(blob.stdout).hexdigest()==expected_files[name],'D5 test source differs')
        require(d['instrumentation_files']==expected_files,'diagnostic CLI/job/test source changed')
    require('integrity_validation' not in p or 'diagnostics' in p,'D5 requires explicit diagnostic plan')
    require(sha(p['registry_path'])==p['registry_sha256'],'dataset registry changed')
    paths,m,pm,specs,users,tok=assets(p['registry_path'],p['multi']);samples,_=provenance(p['dataset'],'qasper')
    require(m['base_prepared']==p['base_prepared'],'base model preparation differs')
    prompts,_=select(samples,tok.tokenizer,count=1,seed=42,context_limit=2048,budget=2,prefix=64,allow_truncation=True,input_limit=256)
    prompts[0].update(strict=True,source_semantic_text=tok.tokenizer.decode(prompts[0]['token_ids'][:64],skip_special_tokens=False,clean_up_tokenization_spaces=False))
    rebuilt=requests(prompts,track='CONTROLLED',prefix=64,special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')})
    require(rebuilt==p['tasks'] and tok.identity==p['tokenizer_identity'] and p['mode']==(APPROX if p['allow_approximate'] else EXACT),'immutable prompt/workload reconstruction differs')
    from .r5_i5_workload import resources
    require(p['resources']==resources(rebuilt,evidence_budget=8*2**30),'resource forecast differs')
    return p,receipt

def ud(config,output_dir):
    from .r5_multi_artifacts import validate,load as load_matrices
    from .user_device import configure_cpu
    from .communication import listener
    from .r4c_evidence import Evidence as Writer
    c=read(config);out=output(output_dir,(c['multi'],c['dataset'],config));writer=Writer(out,budget_bytes=c['budget'])
    m,pm,specs,registry=validate(c['multi']);cpu=configure_cpu(True);spec=specs[c['identity']['adapter_id']]
    require(c['source']==source_identity(),'UD source differs')
    require(spec['matrix_sha256']==c['identity']['adapter_matrix_fingerprint'] and c['identity']['base_fingerprint']==pm['identity']['base_weight_sha256'] and c['identity']['tokenizer']==m['tokenizer'] and c['identity']['model_revision']==pm['identity']['base_revision'],'UD model/adapter identity differs')
    matrices=load_matrices(c['multi'],spec,torch_backend=True)
    import torch
    require(not torch.cuda.is_initialized() and 'transformers' not in sys.modules and all(t.device.type=='cpu' for t in matrices.values()),'UD isolation differs')
    write(out/'initial.json',dict(pid=os.getpid(),cpu=cpu,source=c['source'],identity=c['identity'],session=c['session'],matrix_sha256=spec['matrix_sha256'],user_device_cuda_used=torch.cuda.is_initialized(),full_model_loaded=False))
    sock=listener();write(out/'ready.json',{'port':sock.getsockname()[1]});connection,address=sock.accept();sock.close()
    require(address[0]=='127.0.0.1','nonlocal sparse connection')
    channel=Channel(connection,c['session'],spec['matrix_sha256'],timeout=1800,trace=lambda r:append(out/'wire.jsonl',r))
    try:serve_sparse(channel,c['identity'],registry,matrices,evidence=writer,trace=lambda r:append(out/'layers.jsonl',r))
    finally:connection.close();seal(out)

def worker(experiment,output_dir,condition,cpu_audit,expected_gpu_uuid=None):
    p,pr=load(experiment);require(condition in CONDITIONS,'unsupported condition')
    approval=Evidence(cpu_audit);approval.verify();record=approval.json('cpu_validation.json')
    require(record['status']=='PASS' and record['source']==p['source'] and record['plan_inventory']==pr and record['failed']==record['skipped']==0,'worker CPU validation differs')
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID') and os.environ.get('R5_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')=='1','manual opted-in worker allocation required')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536 and len(os.sched_getaffinity(0))>=4,'4CPU/64GiB worker allocation required')
    require(os.environ.get('SSG_NODE') in ('moana-y3','moana-y4') and os.environ.get('SLURMD_NODENAME',os.environ.get('SLURM_JOB_NODELIST'))==os.environ['SSG_NODE'],'worker node differs')
    require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','offline/backend environment differs')
    out=output(output_dir,(experiment,p['multi'],p['dataset']));(out/'es').mkdir();primary=None;process=None;channel=None
    status={'status':'FAILED','condition':condition};begin=time.monotonic_ns();log=None
    try:
        from .r5_multi_artifacts import validate,runtime_view,physical
        from .r4a_runtime import setup
        from .r4b2_kv import installed_contract
        from .r5_perf_schedule import TINYBERT_REVISION
        from .r4c_evidence import Evidence as Writer
        from .r5_runtime import encoder_for
        from .r5_semantic_index import SemanticIndex
        from .r5_i5_contracts import request
        from .r5_subsequence_generation import generate
        from .r5_i5_quality import score
        m,pm,specs,registry=validate(p['multi']);spec=specs[p['adapter_id']]
        import importlib.util
        loader=importlib.util.spec_from_file_location('sparse_host_preflight',ROOT/'scripts/119_run_r5_i5.py');legacy=importlib.util.module_from_spec(loader);loader.loader.exec_module(legacy)
        require(legacy.cpu_mem_available_bytes()>=32*2**30,'32GiB available host RAM required')
        _,_,_,driver,tokenizer,gpu=setup(dict(base_prepared=p['base_prepared'],fixture=False,device='cuda:0',i5_gpu_policy_scope='R5_I5'),native=False,gpu_identity_policy='SAME_COMPATIBLE_GPU_CLASS')
        require(expected_gpu_uuid is None or expected_gpu_uuid==gpu['uuid'],'execution GPU changed between conditions')
        from .r5_i5_gpu import receipt
        gpu_identity_receipt=receipt(pm,gpu,'SAME_COMPATIBLE_GPU_CLASS',fixture=False)
        import torch
        require(torch.cuda.device_count()==1 and driver.model.config.max_position_embeddings==2048,'one GPU/context required')
        require(torch.cuda.mem_get_info(0)[0]>=12*2**30,'12GiB runtime free-memory reserve required')
        gpu_loaded_bytes=torch.cuda.memory_allocated(0);torch.cuda.reset_peak_memory_stats(0)
        from .r5_multi_artifacts import tokenizer_identity
        require(tokenizer_identity(pm,tokenizer)==p['tokenizer_identity'],'actual tokenizer differs')
        contract=installed_contract(driver);contract.update(sparse_protocol=VERSION,impact='streamed_original_M_row_l2_sum',generation_scope='native full prefill and native decode')
        identity,contract=physical(runtime_view(pm,spec),p['tokenizer_identity'],driver.device,gpu['uuid'],contract,spec,registry)
        session=uuid.uuid4().hex;config=out/'ud_config.json';write(config,dict(multi=p['multi'],dataset=p['dataset'],identity=identity,source=p['source'],session=session,budget=2*2**30))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='');log=(out/'ud.log').open('w')
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/151_r5_subsequence_generation.py'),'_ud','--config',str(config),'--output-dir',str(out/'ud')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+120
        while not (out/'ud/ready.json').exists():
            require(process.poll() is None and time.monotonic()<deadline,'CPU UD startup failed; see ud.log');time.sleep(.05)
        writer=Writer(out/'es',budget_bytes=p['budget_bytes_per_condition'])
        channel=Channel(connect(read(out/'ud/ready.json')['port']),session,spec['matrix_sha256'],timeout=1800,trace=lambda r:append(out/'es/wire.jsonl',r))
        index=None if condition=='FULL_RECOMPUTE' else SemanticIndex(encoder_for({'fixture':False,'semantic_revision':TINYBERT_REVISION},p['tasks']),clusters=1)
        validation_mode=p.get('integrity_validation',{}).get('mode','EAGER_FULL_PROOF')
        owner=None if index is None else PhysicalOwner(identity,registry,evidence=writer,trace=lambda r:append(out/'es/policy.jsonl',r),validation_mode=validation_mode)
        initialized=time.monotonic_ns();rows=[];reqs=[]
        for task in p['tasks']:
            req=request(task,task['token_ids'],task['policy'],identity);reqs.append(req)
            from .r5_subsequence_profile import Profiler
            profiler=None if 'diagnostics' not in p else Profiler(cuda_events=p['diagnostics']['cuda_events'],device=driver.device)
            result=generate(driver.model,driver.adapter,tokenizer,req,channel,owner,index,condition=condition,mode=p['mode'],allow_approximate=p['allow_approximate'],evidence=writer,profiler=profiler)
            result['validation_mode']=validation_mode
            if 'diagnostics' in result:
                result['diagnostics']['validation_mode']=validation_mode
                result['diagnostics']['integrity_validation']=None if owner is None else {k:v for k,v in owner.validation.report().items() if k!='authorizations'}
            data=result.pop('logits_blob');result['logits']=dict(file=writer.blob(data,req['request_id']+'_logits.bin'),dtype='float32',shape=result['logits_shape'],sha256=__import__('hashlib').sha256(data).hexdigest())
            result.update(sample_id=task['sample_id'],scored=task['scored'],quality=score(task['prepared']['sample'],result['generated_text'],task['prepared']['template_version']) if task['scored'] else None)
            end=time.monotonic_ns();result['post_generation_evidence_quality_ms']=(end-result['completion_ns'])/1e6
            result['completion_ns']=end;result['application_e2e_ms']=(end-result['arrival_ns'])/1e6
            result['generated_tokens_per_second']=len(result['generated_token_ids'])/(result['application_e2e_ms']/1000);rows.append(result)
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE');process.wait(timeout=60);require(process.returncode==0,'CPU UD failed')
        write(out/'requests.json',reqs);write(out/'results.json',rows);writer.check()
        require(torch.cuda.max_memory_allocated(0)<22*2**30,'actual GPU allocation peak exceeded22GiB')
        write(out/'identity.json',dict(physical=identity,contract=contract,model_identity=runtime_view(pm,spec)['identity'],gpu=gpu,gpu_identity_receipt=gpu_identity_receipt,gpu_loaded_bytes=gpu_loaded_bytes,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),registry=registry,source=p['source'],plan_inventory=pr,condition=condition,validation_mode=validation_mode,semantic_encoder=None if index is None else index.encoder.metadata,initialization_ms=(initialized-begin)/1e6,warmup_requests=0))
        status['status']='COMPLETE'
    except BaseException as exc:primary=exc;status['reason']=repr(exc);raise
    finally:
        try:
            if channel:channel.sock.close()
            if process and process.poll() is None:process.terminate();process.wait(timeout=30)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note(repr(exc))
        if log:log.close()
        try:write(out/'status.json',status);seal(out)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note(repr(exc))

def run(experiment,output_dir,source_commit,cpu_audit,enable_gpu_experiment=False):
    p,pr=load(experiment);require(enable_gpu_experiment and p['source']['commit']==source_commit,'explicit pinned GPU opt-in required')
    ce=Evidence(cpu_audit);cr=ce.verify();ca=ce.json('cpu_validation.json');require(ca['status']=='PASS' and ca['source']==p['source'] and ca['plan_inventory']==pr and ca['failed']==ca['skipped']==0,'native CPU validation required')
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID') and os.environ.get('R5_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')=='1','manual non-array GPU allocation required')
    require(os.environ.get('SSG_NODE') in ('moana-y3','moana-y4') and os.environ.get('SLURMD_NODENAME',os.environ.get('SLURM_JOB_NODELIST'))==os.environ['SSG_NODE'],'allocated node differs')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'4CPU/64GiB required')
    import shutil
    require(shutil.disk_usage(Path(experiment).parent).free>=32*2**30,'32GiB evidence reserve required')
    out=output(output_dir,(experiment,cpu_audit,p['multi'],p['dataset']));status={'status':'FAILED','source':p['source'],'plan_inventory':pr,'cpu_validation':cr,'cpu_validation_path':str(Path(cpu_audit).resolve()),'children':[]};primary=None
    expected_gpu=None
    try:
        for condition in CONDITIONS:
            start=time.monotonic_ns()
            command=[sys.executable,str(ROOT/'scripts/151_r5_subsequence_generation.py'),'_worker','--experiment',str(experiment),'--cpu-audit',str(cpu_audit),'--output-dir',str(out/condition),'--condition',condition]
            if expected_gpu is not None:command+=['--expected-gpu-uuid',expected_gpu]
            returncode=wait_worker(command)
            status['children'].append(dict(condition=condition,returncode=returncode,process_lifecycle_ms=(time.monotonic_ns()-start)/1e6))
            require(returncode==0,'sparse condition failed; no retry or fallback')
            expected_gpu=read(out/condition/'identity.json')['physical']['gpu_uuid']
        status['status']='COMPLETE'
    except BaseException as exc:primary=exc;status['reason']=repr(exc);raise
    finally:
        try:write(out/'status.json',status);seal(out)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note(repr(exc))
    return status

def cpu_validate(experiment,output_dir):
    p,pr=load(experiment)
    # Installed APIs may not be silently skipped at the GPU approval boundary.
    import torch,transformers,numpy,unittest
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU validation requires hidden/uninitialized CUDA')
    sys.path.insert(0,str(ROOT/'tests'))
    from test_r5_subsequence_generation import Generation
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Generation)
    if 'diagnostics' in p:
        from test_r5_subsequence_diagnostics import Diagnostics
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Diagnostics))
    if 'integrity_validation' in p:
        from test_r5_subsequence_validation import Validation
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Validation))
    result=unittest.TestResult();suite.run(result)
    failed=len(result.failures)+len(result.errors);skipped=len(result.skipped)
    require(not failed and not skipped,'native CPU generation validation failed/skipped: '+str(result.errors+result.failures+result.skipped))
    require(not torch.cuda.is_initialized(),'CPU validation initialized CUDA')
    out=output(output_dir,(experiment,));receipt=dict(status='PASS',source=p['source'],plan_inventory=pr,tests=result.testsRun,failed=failed,skipped=skipped)
    write(out/'cpu_validation.json',receipt);seal(out);return receipt
