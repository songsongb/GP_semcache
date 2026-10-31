#!/usr/bin/env python3
"""R5-I4A versioned preparation, CPU pilot, opt-in GPU serving and audit."""
import argparse
import importlib.util
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
from semcache.edgelora.communication import digest_bytes,require
from semcache.edgelora.r3a_artifacts import read,write,sha,source_identity
from semcache.edgelora.r4a_evidence import verify_sources
from semcache.edgelora.r5_evidence import seal
from semcache.edgelora.r5_multi_artifacts import fresh,prepare,validate
from semcache.edgelora.r5_multi_analysis import HISTORY,gates,analyze,export
from semcache.edgelora.r5_perf_evidence import Writer,accounted_modules,spawn,finish
from semcache.edgelora.r5_perf_schedule import TINYBERT_REVISION


def sources():
    paths=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'configs/r5_integration.json',
           ROOT/'scripts/116_run_r5_multiuser.py',ROOT/'scripts/117_analyze_r5_multiuser.py',
           ROOT/'jobs/r5_multiuser_pilot.sh',ROOT/'jobs/r5_multiuser_expanded.sh']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','dry-run','fixture','run','analyze'));p.add_argument('--stage',choices=('pilot','expanded'),default='pilot')
    p.add_argument('--output-dir',type=Path);p.add_argument('--base-prepared',type=Path);p.add_argument('--multi-prepared',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--source-commit');p.add_argument('--semantic-revision',default=TINYBERT_REVISION)
    p.add_argument('--generation-budget',type=int,default=2);p.add_argument('--condition-budget-bytes',type=int,default=4*2**30)
    p.add_argument('--runtime-budget-seconds',type=int,default=7200)
    p.add_argument('--test-fixture-only',action='store_true');p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan:p.error('--plan required')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.test_fixture_only!=(a.phase=='fixture'):p.error('fixture requires --test-fixture-only')
    if a.enable_gpu_experiment!=(a.phase=='run'):p.error('run requires --enable-gpu-experiment')
    if a.phase=='prepare' and not a.base_prepared:p.error('prepare requires --base-prepared')
    if a.phase=='run' and not a.multi_prepared:p.error('run requires --multi-prepared')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs belongs to analyze')
    if not 2<=a.generation_budget<=8 or not 128*2**20<=a.condition_budget_bytes<=8*2**30 or not 60<=a.runtime_budget_seconds<=7200:p.error('bounded generation/evidence/runtime required')
    return a


def preflight(a,m):
    require(os.environ.get('R5M_ENABLE_GPU')=='1' and os.environ.get('R5_ENABLE_GPU')=='1','R5M_ENABLE_GPU=1 and R5_ENABLE_GPU=1 required')
    require(a.semantic_revision==TINYBERT_REVISION and not m['TEST_FIXTURE_ONLY'],'real prepared adapters and verified TinyBERT revision required')
    spec=importlib.util.spec_from_file_location('r5m_preflight',ROOT/'scripts/110_run_r5_semcache_e2e.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a.base_prepared=Path(m['base_prepared']);a.evidence_budget_bytes=2*a.condition_budget_bytes;module.gpu_preflight(a)
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines());require(set(sources()).issubset(tracked),'commit all I4A execution sources')
    require(shutil.disk_usage(a.output_dir).free>2*a.condition_budget_bytes+2**30,'insufficient worst-case evidence disk budget')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4,'four allocated CPU cores required; serving is sequential')
    import torch
    require(torch.cuda.mem_get_info(0)[0]>8*2**30,'at least 8 GiB available GPU memory required before model load')


def worker(a):
    plan=read(a.plan);root=Path(plan['account_root']);writer=Writer(root,plan['condition_budget_bytes'])
    try:
        verify_sources(plan['sources'])
        if not plan['fixture']:
            require(os.environ.get('R5M_ENABLE_GPU')=='1' and os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID') and
                    source_identity()['git_head']==plan['source_commit'],'GPU worker identity/opt-in mismatch')
        from semcache.edgelora.r5_multi_runtime import execute
        from semcache.edgelora.r5_multi_session import serve
        with accounted_modules(writer):
            if a.worker=='es':execute(plan,root,writer)
            else:serve(plan,Path(plan['output']),writer)
        verify_sources(plan['sources']);return 0
    except BaseException:traceback.print_exc();return 1


def main(argv=None):
    os.umask(0o077);a=parse(argv)
    if a.worker:return worker(a)
    if a.phase=='prepare':
        try:out=prepare(a.base_prepared,a.output_dir,a.stage);print(json.dumps(dict(status='PREPARED',output=str(out))));return 0
        except BaseException as exc:print(str(exc),file=sys.stderr);return 1
    try:out=fresh(a.output_dir,(a.inputs,a.base_prepared,a.multi_prepared))
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);return 1
    result=dict(status='RUNNING',stage='PREFLIGHT',GPU_NOT_RUN=True,source_identity=source_identity());decision=gates()
    try:
        if a.phase=='analyze':
            report=analyze(a.inputs);export(out,report);decision=report['decision'];result.update(analysis_only=True,input_manifest_sha256=sha(a.inputs/'r5_output_manifest.json'))
        elif a.phase=='dry-run':
            write(out/'multi_preflight.json',dict(stage=a.stage,adapters=2 if a.stage=='pilot' else 3,logical_users=4 if a.stage=='pilot' else 30,sequential=True,gpu_jobs_submitted=0,
                semantic_revision=TINYBERT_REVISION,condition_budget_bytes=a.condition_budget_bytes,history=HISTORY,
                missing_dependencies=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None]))
        else:
            if a.phase=='fixture':
                from semcache.edgelora.r3a_tiny import prepare_tiny
                base=prepare_tiny(out/'base_prepare');multi=prepare(base,out/'multi_prepare',a.stage)
            else:multi=a.multi_prepared.resolve()
            m,_,_,_=validate(multi)
            require(m.get('stage','pilot')==a.stage,'prepared adapter stage mismatch')
            if a.phase=='run':preflight(a,m)
            config=dict(fixture=a.phase=='fixture',stage=a.stage,multi_prepared=str(multi),base_prepared=m['base_prepared'],
                generation_budget=a.generation_budget,condition_budget_bytes=a.condition_budget_bytes,semantic_revision=a.semantic_revision,source_commit=a.source_commit,
                device='cpu' if a.phase=='fixture' else 'cuda:0',runtime_budget_seconds=a.runtime_budget_seconds)
            write(out/'multi_config.json',config);write(out/'multi_sources.json',sources());write(out/'multi_historical.json',HISTORY)
            deadline=time.monotonic()+a.runtime_budget_seconds
            for condition in ('DISTRIBUTED_NO_REUSE','INTEGRATED'):
                result.update(stage='SEQUENTIAL_GENERATION',condition=condition);write(out/'multi_status.json',result)
                dest=out/condition;dest.mkdir();writer=Writer(dest,a.condition_budget_bytes,create=True)
                plan=dict(config,condition=condition,account_root=str(dest),sources=sources(),output=str(dest))
                writer.write(dest/'plan.json',plan)
                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                remaining=deadline-time.monotonic();require(remaining>1,'bounded runtime exhausted before worker start')
                child=spawn([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(dest/'plan.json')],writer,dest/'stdout.log',dest/'stderr.log',env=env)
                life=finish(child,timeout=remaining);writer.write(dest/'lifecycle.json',life)
                require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'condition failed; preserved '+str(dest/'stderr.log'))
                writer.reconcile();writer.write(dest/'reconciliation.json',dict(status='PASS',scans=writer.scans));writer.reconcile()
            result['stage']='INDEPENDENT_AUDIT';report=analyze(out,sealed=False);export(out,report);decision=report['decision']
            verify_sources(read(out/'multi_sources.json'))
            from semcache.edgelora.r4c_evidence import directory_bytes
            require(directory_bytes(out)<2*a.condition_budget_bytes+2**30,'total evidence/setup budget exceeded')
            result['GPU_NOT_RUN']=config['fixture']
        result.update(status='COMPLETE',stage='COMPLETE')
    except BaseException as exc:
        result.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());decision=gates()
        write(out/'multi_failure.json',result);traceback.print_exc()
    finally:
        if a.phase=='run':result.update(gpu_attempted=(out/'INTEGRATED').exists() or (out/'DISTRIBUTED_NO_REUSE').exists(),gpu_forward_evidence_present=any(out.rglob('gpu_execution.json')))
        write(out/'multi_decision.json',decision);write(out/'multi_results.json',result);seal(out)
    print(json.dumps(dict(status=result['status'],output=str(out))));return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));sys.exit(main())
