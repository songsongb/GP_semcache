#!/usr/bin/env python3
"""One sealed I5 matched A/B/C block; no submission, download or partial-cache restart."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import read,write,sha,source_identity
from semcache.edgelora.r5_i5_evidence import fresh,seal,verify,sources,verify_sources
from semcache.edgelora.r5_perf_evidence import Writer,accounted_modules,spawn,finish
from semcache.edgelora.r5_perf_schedule import TINYBERT_REVISION

def cpu_mem_available_bytes():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")

def preflight(a,experiment,block,m,pm):
    require(os.environ.get('R5_I5_ENABLE_GPU')=='1' and os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'), 'manual Slurm allocation and I5/R5 GPU opt-in required')
    require(not experiment['TEST_FIXTURE_ONLY'] and not m['TEST_FIXTURE_ONLY'], 'GPU rejects fixture data/adapters')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='exp/raw-semcache-r5-integration', 'required branch mismatch')
    require(a.source_commit and source_identity()['git_head']==a.source_commit, 'explicit committed source identity required')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(), 'tracked working tree dirty')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    require(set(sources())<=tracked, 'commit I5 execution sources before GPU run')
    require(a.semantic_revision==TINYBERT_REVISION and m['stage']=='expanded', 'pinned TinyBERT/three adapters required')
    from semcache.edgelora.r5_i5_data import load_dataset
    from semcache.edgelora.r5_i5_preflight import installed_opt_check
    installed_opt_check()  # Required real CPU OPT contract, never optional in GPU preflight.
    datasets=read(a.experiment/'i5_dataset_manifest.json')
    from semcache.edgelora.r5_i5_quality import validate_scorers
    validate_scorers(datasets)
    for name,manifest in datasets.items():
        path=Path(manifest['prepared_path']);load_dataset(path)
        require(sha(path/'i5_output_manifest.json')==manifest['prepared_manifest_sha256'], 'dataset snapshot changed since selection')
    from semcache.edgelora.r4c_prepare import opt_tokenizer
    tok=opt_tokenizer(pm)
    require(tok.identity==experiment['tokenizer_identity'] and tok.supported_context_length==experiment['context_limit'], 'model/tokenizer/config differs')
    from semcache.edgelora.r5_runtime import encoder_for
    encoder=encoder_for(dict(fixture=False,semantic_revision=a.semantic_revision),[]);del encoder
    from semcache.edgelora.r5_i5_gpu import diagnostics
    diagnostic=diagnostics(pm,policy=a.gpu_identity_policy)
    require(diagnostic['eligible'],'allocated GPU ineligible: '+json.dumps(diagnostic))
    a.execution_gpu_uuid=diagnostic['execution_gpu_uuid']
    import torch
    r=block['resources']
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'RTX 3090' in torch.cuda.get_device_name(0), 'one RTX 3090 required')
    require(torch.cuda.mem_get_info(0)[0] > r['gpu_estimate_bytes'], 'estimated GPU memory unavailable')
    require(shutil.disk_usage(a.output_dir.parent).free>3*r['evidence_budget_bytes']+2**30, 'insufficient block worst-case disk')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4, 'four CPU cores required')

    available = cpu_mem_available_bytes()
    required = r['cpu_memory_reserve_bytes']

    require(
      available >= required,
      f"CPU memory reserve unavailable: "
      f"available={available / 2**30:.2f} GiB, "
      f"required={required / 2**30:.2f} GiB"
    )

    require(r['duration_estimate_seconds']*3<a.runtime_budget_seconds, 'estimated block duration exceeds bound; reduce block/generation or use measured smoke estimate')


def main(argv=None):
    os.umask(0o077);p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path);p.add_argument('--block-index',type=int,default=0);p.add_argument('--multi-prepared',type=Path)
    p.add_argument('--output-dir',type=Path);p.add_argument('--source-commit');p.add_argument('--semantic-revision',default=TINYBERT_REVISION)
    p.add_argument('--runtime-budget-seconds',type=int,default=9000);p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--test-fixture-only',action='store_true');p.add_argument('--worker',choices=('es','ud'));p.add_argument('--plan',type=Path)
    p.add_argument('--gpu-identity-policy',choices=('STRICT_PREPARATION_GPU','SAME_COMPATIBLE_GPU_CLASS'),default='SAME_COMPATIBLE_GPU_CLASS')
    a=p.parse_args(argv)
    if a.worker:
        plan=read(a.plan);writer=Writer(plan['account_root'],plan['condition_budget_bytes']);verify_sources(plan['sources'])
        if not plan['fixture']:
            require(os.environ.get('R5_I5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID') and source_identity()['git_head']==plan['source_commit'], 'GPU worker opt-in/source mismatch')
        from semcache.edgelora.r5_i5_runtime import execute
        from semcache.edgelora.r5_i5_session import serve
        with accounted_modules(writer):
            if a.worker=='es':execute(plan,Path(plan['account_root']),writer)
            else:serve(plan,Path(plan['output']),writer)
        verify_sources(plan['sources']);return 0
    if not all((a.experiment,a.multi_prepared,a.output_dir)):p.error('--experiment --multi-prepared --output-dir required')
    require(a.test_fixture_only != a.enable_gpu_experiment, 'explicit fixture OR GPU opt-in required')
    require(60<=a.runtime_budget_seconds<=10800, 'runtime bound 60..10800 seconds')
    out=fresh(a.output_dir,(a.experiment,a.multi_prepared));status=dict(status='FAILED',GPU_NOT_RUN=True)
    try:
        verify(a.experiment);experiment=read(a.experiment/'i5_workload_manifest.json')
        require(0<=a.block_index<len(experiment['blocks']), 'invalid block index')
        block=experiment['blocks'][a.block_index]
        from semcache.edgelora.r5_multi_artifacts import validate
        m,pm,adapters,registry=validate(a.multi_prepared)
        require(m['stage']=='expanded' and experiment['TEST_FIXTURE_ONLY']==a.test_fixture_only==m['TEST_FIXTURE_ONLY'], 'scope/expanded preparation mismatch')
        if a.enable_gpu_experiment:preflight(a,experiment,block,m,pm)
        config=dict(schema='I5_EXECUTION_V1',gpu_identity_policy=a.gpu_identity_policy,i5_gpu_policy_scope='R5_I5',expected_execution_gpu_uuid=getattr(a,'execution_gpu_uuid',None),fixture=a.test_fixture_only,multi_prepared=str(a.multi_prepared.resolve()),
            base_prepared=m['base_prepared'],device='cpu' if a.test_fixture_only else 'cuda:0',condition_budget_bytes=block['resources']['evidence_budget_bytes'],
            semantic_revision=a.semantic_revision,source_commit=a.source_commit,sources=sources(),block=block,runtime_budget_seconds=a.runtime_budget_seconds,
            experiment_id=experiment['experiment_id'],experiment_prepared=str(a.experiment.resolve()),experiment_manifest_sha256=sha(a.experiment/'i5_output_manifest.json'),
            model_initialization_measured_separately=True,localhost_tcp=True,numerical_history='ROOT_CAUSE_UNRESOLVED; I4A Expanded 12/12 HIT tolerances failed; I3F-B FUTURE_WORK')
        write(out/'i5_execution_config.json',config);write(out/'i5_workload_manifest.json',dict(schema='I5_BLOCK_WORKLOAD_V1',experiment_id=experiment['experiment_id'],block=block))
        write(out/'i5_dataset_manifest.json',read(a.experiment/'i5_dataset_manifest.json'));write(out/'i5_adapter_registry.json',dict(adapters=adapters,registry=registry))
        deadline=time.monotonic()+a.runtime_budget_seconds
        for condition in block['condition_order']:
            dest=out/condition;dest.mkdir();writer=Writer(dest,config['condition_budget_bytes'],create=True)
            plan=dict(config,condition=condition,account_root=str(dest),output=str(dest));writer.write(dest/'plan.json',plan)
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            remaining=deadline-time.monotonic();require(remaining>1, 'block runtime exhausted')
            child=spawn([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(dest/'plan.json')],writer,dest/'stdout.log',dest/'stderr.log',env=env)
            life=finish(child,timeout=remaining);writer.write(dest/'lifecycle.json',life)
            require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'], 'condition failed: '+str(dest/'stderr.log'))
            from semcache.edgelora.r5_i5_gpu import match_execution
            match_execution(config['expected_execution_gpu_uuid'],read(dest/'gpu_identity.json')['execution_gpu_uuid'])
            writer.reconcile();writer.write(dest/'reconciliation.json',dict(status='PASS'));writer.reconcile()
        verify_sources(config['sources']);status=dict(status='COMPLETE',GPU_NOT_RUN=a.test_fixture_only,block_id=block['block_id'])
    except BaseException as exc:
        status.update(reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    status['gpu_attempted']=a.enable_gpu_experiment and any((out/c/'plan.json').exists() for c in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED'))
    status['gpu_forward_evidence_present']=any(out.rglob('gpu_execution.json'))
    status['GPU_NOT_RUN']=not status['gpu_forward_evidence_present']
    write(out/'i5_block_status.json',status);seal(out)
    if status['status']=='COMPLETE':
        from semcache.edgelora.r5_i5_analysis import analyze,export
        # Audit always writes into a new directory, including on audit failure.
        audit=fresh(out.with_name(out.name+'_audit'))
        try:report=analyze(out);export(audit,report)
        except BaseException as exc:
            write(audit/'i5_independent_audit.json',dict(status='FAIL',reason=str(exc)));seal(audit);raise
        seal(audit)
    print(json.dumps(dict(status=status['status'],output=str(out))));return 0 if status['status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));sys.exit(main())

