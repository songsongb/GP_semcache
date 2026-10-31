#!/usr/bin/env python3
"""R5 integrated application: CPU fixture, opt-in GPU pilot, read-only analysis."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.r4a_evidence import verify_sources
from semcache.edgelora.r5_contracts import gates
from semcache.edgelora.r5_evidence import fresh,seal,sources


def gpu_preflight(a):
    require(os.environ.get('R5_ENABLE_GPU')=='1','explicit independent R5_ENABLE_GPU=1 required')
    identity=source_identity()
    require(identity['git_branch']=='exp/raw-semcache-r5-integration','R5 development branch required')
    require(a.source_commit and a.source_commit==identity['git_head'],'explicit matching immutable --source-commit required')
    status=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,capture_output=True,text=True,check=True)
    require(not status.stdout.strip(),'commit tracked implementation before GPU execution')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,'one explicit Slurm RTX3090 allocation required')
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline model and tokenizer mode required')
    require(os.environ.get('HF_HOME') and Path(os.environ['HF_HOME']).is_dir() and os.environ.get('HF_HUB_CACHE') and Path(os.environ['HF_HUB_CACHE']).is_dir(),'explicit existing local Hugging Face cache required')
    missing=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None]
    require(not missing,'MISSING_DEPENDENCY: '+', '.join(missing))
    require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timing')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'RTX 3090' in torch.cuda.get_device_name(0),'exactly one allocated RTX 3090 required')
    require(shutil.disk_usage(a.output_dir).free>a.evidence_budget_bytes+2**30,'insufficient disk/evidence budget')
    from semcache.edgelora.r3a_artifacts import validate_prepared
    pm,_,_=validate_prepared(a.base_prepared)
    from semcache.edgelora.r3a_runtime import memory
    gpu=memory('cuda:0')
    require(gpu['uuid']!='UNAVAILABLE' and pm['gpu']['uuid']==gpu['uuid'],'fresh R3-A preparation for allocated RTX3090 UUID required')
    from semcache.edgelora.r5_runtime import encoder_for
    encoder_for(dict(fixture=False,semantic_revision=a.semantic_revision),[])


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','fixture','run','analyze'));p.add_argument('--output-dir',type=Path)
    p.add_argument('--test-fixture-only',action='store_true');p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--base-prepared',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--semantic-revision');p.add_argument('--source-commit')
    p.add_argument('--capacity-bytes',type=int,default=256*2**20);p.add_argument('--lifecycle-capacity',action='store_true')
    p.add_argument('--evidence-budget-bytes',type=int,default=2*2**30)
    p.add_argument('--generation-budget',type=int,default=3);p.add_argument('--generation-mode',choices=('fixed','quality'),default='fixed')
    p.add_argument('--evidence-profile',choices=('STRICT_AUDIT','SCALABLE_EVALUATION'),default='STRICT_AUDIT')
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan:p.error('worker plan required')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.test_fixture_only!=(a.phase=='fixture'):p.error('fixture requires explicit --test-fixture-only')
    if a.enable_gpu_experiment!=(a.phase=='run'):p.error('run requires explicit --enable-gpu-experiment')
    if a.phase=='run' and not a.base_prepared:p.error('run requires --base-prepared')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs belongs to analyze')
    if not 0<a.capacity_bytes<=2**30 or not 0<a.evidence_budget_bytes<=16*2**30 or not 1<=a.generation_budget<=16:p.error('bounded capacity/evidence/generation required')
    return a


def worker(a):
    plan=read(a.plan);out=Path(plan['output']);result={}
    try:
        verify_sources(plan['sources'])
        if not plan['fixture']:require(os.environ.get('R5_ENABLE_GPU')=='1','GPU worker opt-in missing')
        if a.worker=='ud':
            from semcache.edgelora.r4c_dynamic_ud import serve
            serve(plan,out);result=read(out/'ud_result.json')
        else:
            from semcache.edgelora.r5_runtime import execute
            result=execute(plan,out)
    except BaseException as exc:
        result=dict(execution_status='FAILED',stage='WORKER_START',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if (out/'r5_failure.json').exists():result['stage']=read(out/'r5_failure.json')['stage']
        traceback.print_exc()
    result.update(pid=os.getpid(),TEST_FIXTURE_ONLY=plan['fixture']);write(out/(a.worker+'_result.json'),result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def main(argv=None):
    a=parse(argv)
    if a.worker:return worker(a)
    try:out=fresh(a.output_dir,(a.inputs,a.base_prepared))
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';result=dict(execution_status='RUNNING',stage='PREFLIGHT',created_at=now(),
        sources=sources(),source_identity=source_identity(),gpu_status='GPU_NOT_RUN',gpu_inference_performed=False)
    try:
        if a.phase=='dry-run':
            write(out/'r5_preflight.json',dict(missing_dependencies=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None],
                required_gpu_optins=['--enable-gpu-experiment','R5_ENABLE_GPU=1','allocated RTX3090','--source-commit','--semantic-revision'],
                GPU_NOT_RUN=True,downloads=False,job_submitted=False))
        elif a.phase=='analyze':
            from semcache.edgelora.r5_analysis import analyze,export
            report=analyze(a.inputs,sealed=True);export(out,report);result['analysis_only']=True
            result['input_manifest_sha256']=sha(a.inputs/'r5_output_manifest.json')
        else:
            if fixture:
                from semcache.edgelora.r3a_tiny import prepare_tiny
                base=prepare_tiny(out/'base_prepare')
            else:gpu_preflight(a);base=a.base_prepared.resolve()
            for name in ('r5_requests','r5_semantic_events','r5_cache_events','r5_generation_outputs','r5_ownership'):(out/(name+'.jsonl')).touch()
            plan=dict(fixture=fixture,base_prepared=str(base),base_manifest_sha256=sha(base/'r3a_prepare_manifest.json'),
                device='cpu' if fixture else 'cuda:0',sources=result['sources'],capacity_bytes=a.capacity_bytes,
                lifecycle_capacity=a.lifecycle_capacity,evidence_budget_bytes=a.evidence_budget_bytes,
                semantic_revision=a.semantic_revision,generation_mode=a.generation_mode,generation_budget=a.generation_budget,evidence_profile=a.evidence_profile)
            es=out/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
            from semcache.edgelora.r3cf_evidence import legacy_runner
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            result['stage']='ES_EXECUTION'
            legacy_runner().process_runner()([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json')],es,timeout=7500.,env=env)
            result.update(gpu_inference_performed=not fixture,gpu_status='GPU_NOT_RUN' if fixture else 'GPU_EXECUTION_COMPLETE_WITH_PENDING_AUDIT')
            result['stage']='INDEPENDENT_AUDIT'
            from semcache.edgelora.r5_analysis import analyze,export
            report=analyze(out);export(out,report)
            from semcache.edgelora.r5_evidence import directory_bytes
            require(directory_bytes(out)<=a.evidence_budget_bytes,'EVIDENCE_BUDGET_EXCEEDED: final reports exceed budget')
            if not fixture:result['gpu_status']='GPU_EXECUTION_COMPLETE_WITH_AUDIT'
            verify_sources(result['sources'])
        result.update(execution_status='COMPLETE',stage='COMPLETE')
    except BaseException as exc:
        result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if (out/'es/r5_failure.json').exists():result.update(worker_failure=read(out/'es/r5_failure.json'))
        write(out/'r5_failure.json',result);traceback.print_exc()
        if not (out/'r5_decision.json').exists():write(out/'r5_decision.json',gates(fixture))
    finally:
        if (out/'es/gpu_execution.json').exists():result.update(gpu_inference_performed=True)
        if result['execution_status']=='FAILED' and result['gpu_inference_performed']:result['gpu_status']='GPU_RUN_FAILED'
        if not (out/'r5_limitations.json').exists():write(out/'r5_limitations.json',dict(GPU_NOT_RUN=not result['gpu_inference_performed'],native_parity='HISTORICAL_FAILED_UNRESOLVED',network='localhost TCP',adapter='controlled untrained user_a'))
        result['finished_at']=now();write(out/'r5_results.json',result);seal(out)
    print(json.dumps(dict(status=result['execution_status'],results=str(out/'r5_results.json'))))
    return 0 if result['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));sys.exit(main())
