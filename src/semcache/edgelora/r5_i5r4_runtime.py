"""One allocation, sequential independent policy processes; original I5 forwards."""
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy
import importlib.util,os,random,shutil,sys,time,subprocess,json

from .communication import require,digest
from .r3a_artifacts import ROOT,read,write,process_environment,versions
from .r5_i5_evidence import seal,verify_sources
from .r5_i5r4 import load,cpu_receipt,output,GPU_SECONDS
from .r5_i5r4_policy import factories
from .r5_perf_evidence import Writer,accounted_modules,spawn,finish
from .r5_perf_schedule import TINYBERT_REVISION

@contextmanager
def injection(policy,params,receipt):
    # Explicit opt-in process-local factories, restored even on failure. Normal
    # workers never import this module or change their factory bindings.
    from . import r5_i5_runtime as runtime
    old=runtime.MultiOwner,runtime.I5Controller
    runtime.MultiOwner,runtime.I5Controller=factories(policy,params,receipt)
    try:yield runtime.execute
    finally:runtime.MultiOwner,runtime.I5Controller=old

def execute(plan,root,writer):
    require(plan['i4']['schema']=='I5R4_CAMPAIGN_V1' and plan['condition']==('FULL_RECOMPUTE' if plan['i4']['policy']=='P5' else 'SEMCACHE_INTEGRATED'),'policy runtime condition differs')
    require(not plan.get('i5r2b') and not plan.get('i5r2c2'),'diagnostic projection arithmetic prohibited')
    verify_sources(plan['sources'])
    writer.write(Path(root)/'i4_policy_configuration.json',dict(policy=plan['i4']['policy'],parameters=plan['i4']['parameters'],
                 base_configuration_scope='Original I5 configuration.json is preserved; actual opt-in owner/controller policy is specified here',
                 native_arithmetic='UNCHANGED',capacity_per_adapter_bytes=512*2**20,max_entries=64))
    log=lambda row:writer.append(Path(root)/'i4_policy.jsonl',row)
    with injection(plan['i4']['policy'],plan['i4']['parameters'],log) as original:
        with accounted_modules(writer):original(plan,root,writer)
    if not plan['fixture']:
        import torch
        writer.write(Path(root)/'i4_backend.json',dict(observation='AFTER_EXECUTION; setup source fingerprint records preforward assignments',
                     torch=torch.__version__,CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                     allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                     fp16_reduced_precision_reduction=getattr(torch.backends.cuda.matmul,'allow_fp16_reduced_precision_reduction','UNKNOWN'),
                     deterministic=torch.are_deterministic_algorithms_enabled(),matmul_precision=torch.get_float32_matmul_precision(),
                     cpu_threads=torch.get_num_threads(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))

def worker(plan_path):
    p=read(plan_path);require(os.environ.get('I4_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')==os.environ.get('R5_ENABLE_GPU')=='1','I4 worker GPU opt-in missing')
    require(not p['fixture'] and os.environ.get('SLURM_JOB_ID'),'GPU worker rejects fixtures/nonallocated execution')
    from .r5_i5r4 import reviewed
    require(reviewed(p['source_commit'])==p['i4']['source'],'worker source differs')
    random.seed(p['i4']['seed']);import torch
    torch.manual_seed(p['i4']['seed']);torch.set_num_threads(4)
    writer=Writer(p['account_root'],p['condition_budget_bytes'])
    writer.write(Path(p['account_root'])/'i4_environment.json',dict(environment=process_environment(),versions=versions(),seed=p['i4']['seed'],
                  CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),instrumentation=p['block']['segment'],source=p['i4']['source']))
    execute(p,p['account_root'],writer)

def finalize(path,status,primary=None):
    """Preserve primary failure and completed children; never overwrite attempt."""
    from .r5_i5r4_analysis import seal_bundle
    try:
        status['GPU_NOT_RUN']=not any((Path(path)/'children').rglob('gpu_execution.json'))
        write(Path(path)/'i4_run_status.json',status);seal_bundle(path)
    except BaseException as secondary:
        if primary is None:raise
        if hasattr(primary,'add_note'):primary.add_note('I4 failure evidence finalization: '+str(secondary))
        else:print('I4 secondary finalization failure (primary preserved): '+str(secondary),file=sys.stderr)

def preflight(p,out,remaining):
    require(os.environ.get('I4_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')==os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'manual I4/I5/R5 GPU allocation required')
    require(not os.environ.get('SLURM_ARRAY_TASK_ID') and os.environ.get('I4_NODE') in ('moana-y3','moana-y4') and os.environ.get('SLURMD_NODENAME',os.environ.get('SLURM_JOB_NODELIST'))==os.environ['I4_NODE'],'one permitted non-array node required')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'4 CPUs /64GiB allocation required')
    require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','pinned offline/backend environment required')
    from .r5_i5_preflight import installed_opt_check
    installed_opt_check() # Mandatory installed CPU API, never silently skipped.
    from .r5_i5_gpu import diagnostics
    from .r5_multi_artifacts import validate
    from .r5_runtime import encoder_for
    from .r4c_prepare import opt_tokenizer
    m,pm,_,_=validate(p['multi_prepared']);tok=opt_tokenizer(pm)
    require(tok.identity==p['tokenizer_identity'] and tok.supported_context_length==2048,'tokenizer/model snapshot changed')
    e=encoder_for(dict(fixture=False,semantic_revision=TINYBERT_REVISION),[]);del e
    d=diagnostics(pm);require(d['eligible'],'RTX3090 GPU preflight: '+str(d))
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.mem_get_info(0)[0]>p['resources']['GPU_peak_estimate_bytes'],'one GPU / memory reserve unavailable')
    spec=importlib.util.spec_from_file_location('i4_legacy_preflight',ROOT/'scripts/119_run_r5_i5.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    require(legacy.cpu_mem_available_bytes()>=p['resources']['host_available_reserve_bytes'],'32GiB available CPU RAM reserve required')
    require(shutil.disk_usage(out.parent).free>=p['resources']['required_free_disk_bytes'],'campaign evidence disk/reserve unavailable')
    require(p['resources']['GPU_children_seconds']<=remaining,'sealed campaign cannot fit remaining budget; no partial launch')
    return d['execution_gpu_uuid']

def run(experiment,source_commit,output_dir,runtime_budget_seconds=GPU_SECONDS,enable_gpu_experiment=False,cpu_audit=None):
    require(enable_gpu_experiment and runtime_budget_seconds==GPU_SECONDS and cpu_audit,'explicit GPU opt-in, CPU audit and exact 68400s bound required')
    begin=time.monotonic();p,pr=load(experiment,source_commit);cr=cpu_receipt(p,pr,cpu_audit)
    require(runtime_budget_seconds-(time.monotonic()-begin)>=p['resources']['GPU_children_seconds'],'provenance/preflight consumed reserved campaign bound')
    out=output(output_dir,(experiment,cpu_audit,p['multi_prepared'],*p['baseline']['paths'].values()));(out/'children').mkdir()
    status=dict(status='INCOMPLETE',plan_id=p['plan_id'],source=p['source'],cpu_audit=cr,completed_children=[],GPU_NOT_RUN=True,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write(out/'i4_run_config.json',dict(plan_inventory=pr,cpu_audit_path=str(Path(cpu_audit).resolve()),cpu_audit=cr,source=p['source'],runtime_budget_seconds=runtime_budget_seconds,experiment=str(Path(experiment).resolve())))
    primary=None
    try:
        # Allocation checks run in a separate short-lived process. Its CUDA
        # context exits BEFORE any ES child starts; the coordinator is CPU-only.
        # Allocation preflight needs identities/resources, never all child prompts.
        write(out/'i4_preflight_input.json',dict(plan={k:p[k] for k in ('multi_prepared','tokenizer_identity','resources')},output=str(out),remaining=runtime_budget_seconds-(time.monotonic()-begin)))
        checked=subprocess.run([sys.executable,str(ROOT/'scripts/148_r5_i5r4.py'),'_gpu-preflight','--plan',str(out/'i4_preflight_input.json')],cwd=ROOT,capture_output=True,timeout=300)
        require(len(checked.stdout)+len(checked.stderr)<=2**20,'preflight output exceeds bound')
        (out/'i4_preflight.stdout').write_bytes(checked.stdout);(out/'i4_preflight.stderr').write_bytes(checked.stderr)
        require(checked.returncode==0,'GPU preflight rejected; see i4_preflight.stderr')
        gpu=json.loads(checked.stdout)['execution_gpu_uuid'];status['execution_gpu_uuid']=gpu
        from .r5_multi_artifacts import validate
        m,_,_,_=validate(p['multi_prepared'])
        bounds={x['child_id']:x for x in p['resources']['children']}
        for j in p['jobs']:
            remaining=runtime_budget_seconds-(time.monotonic()-begin)
            require(remaining>=bounds[j['child_id']]['duration_seconds'],'insufficient sealed child budget; campaign INCOMPLETE, no task drop')
            dest=out/'children'/j['child_id'];dest.mkdir();cond=dest/'condition';cond.mkdir()
            plan=dict(schema='I5_EXECUTION_V1',fixture=False,gpu_identity_policy='SAME_COMPATIBLE_GPU_CLASS',i5_gpu_policy_scope='R5_I5',
                      expected_execution_gpu_uuid=gpu,multi_prepared=p['multi_prepared'],base_prepared=m['base_prepared'],device='cuda:0',
                      condition_budget_bytes=j['block']['resources']['evidence_budget_bytes'],semantic_revision=TINYBERT_REVISION,
                      source_commit=p['source_commit'],sources=p['source']['files'],block=j['block'],runtime_budget_seconds=math_ceil(bounds[j['child_id']]['duration_seconds']),
                      experiment_id=p['plan_id'],experiment_prepared=str(Path(experiment).resolve()),experiment_manifest_sha256=pr['manifest_sha256'],
                      condition='FULL_RECOMPUTE' if j['policy']=='P5' else 'SEMCACHE_INTEGRATED',account_root=str(cond),output=str(cond),
                      i4=dict(schema=p['schema'],source=p['source'],plan_id=p['plan_id'],policy=j['policy'],parameters=p['policy_parameters'],seed=p['seed']))
            writer=Writer(cond,plan['condition_budget_bytes'],create=True);writer.write(cond/'plan.json',plan)
            started=time.monotonic_ns();child=spawn([sys.executable,'-u',str(ROOT/'scripts/148_r5_i5r4.py'),'_worker','--plan',str(cond/'plan.json')],writer,cond/'stdout.log',cond/'stderr.log',env=dict(os.environ))
            life=finish(child,timeout=min(remaining,plan['runtime_budget_seconds']));life.update(execution_start_ns=started,execution_end_ns=time.monotonic_ns())
            writer.write(cond/'lifecycle.json',life)
            ok=life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination']
            if ok:writer.reconcile();writer.write(cond/'reconciliation.json',dict(status='PASS'));writer.reconcile()
            write(dest/'i4_child_status.json',dict(status='COMPLETE' if ok else 'FAILED',child_id=j['child_id'],policy=j['policy'],plan_id=p['plan_id']))
            seal(dest);require(ok,'I4 condition failed: '+str(cond/'stderr.log'))
            status['GPU_NOT_RUN']=False;status['completed_children'].append(j['child_id'])
        verify_sources(p['source']['files']);status['status']='COMPLETE'
    except BaseException as exc:
        primary=exc;status.update(status='FAILED',reason=type(exc).__name__+': '+str(exc)[:2048]);raise
    finally:finalize(out,status,primary)
    return dict(status=status['status'],output=str(out),GPU_NOT_RUN=False)

def math_ceil(value):
    import math
    return max(60,min(10800,math.ceil(value)))
