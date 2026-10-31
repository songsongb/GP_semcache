#!/usr/bin/env python3
"""Bounded R5-I3F: CPU fixture, explicitly opted-in GPU capture, independent analysis."""
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
from semcache.edgelora.communication import digest_bytes, require
from semcache.edgelora.r3a_artifacts import read, write, sha, source_identity
from semcache.edgelora.r4a_evidence import verify_sources
from semcache.edgelora.r5_evidence import fresh, seal
from semcache.edgelora.r5_fidelity import TARGETS, gates
from semcache.edgelora.r5_perf_schedule import TINYBERT_REVISION
from semcache.edgelora.r5_perf_evidence import Writer, accounted_modules, spawn, finish


def sources():
    paths=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'configs/r5_integration.json',
        *[ROOT/'scripts'/name for name in ('95_run_r3a_distributed_no_reuse.py','110_run_r5_semcache_e2e.py',
            '112_run_r5_balanced_benchmark.py','114_run_r5_fidelity_forensics.py','115_analyze_r5_fidelity_forensics.py')],
        ROOT/'jobs/r5_opt_fidelity_forensics.sh']
    return {str(p.relative_to(ROOT)):digest_bytes(p.read_bytes()) for p in paths}


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','fixture','run','analyze'));p.add_argument('--output-dir',type=Path)
    p.add_argument('--inputs',type=Path);p.add_argument('--base-prepared',type=Path);p.add_argument('--source-commit')
    p.add_argument('--test-fixture-only',action='store_true');p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--semantic-revision',default=TINYBERT_REVISION)
    p.add_argument('--generation-budget',type=int,default=8)
    p.add_argument('--block-budget-bytes',type=int,default=4*2**30)
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan:p.error('--plan required')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.test_fixture_only!=(a.phase=='fixture'):p.error('fixture requires --test-fixture-only')
    if a.enable_gpu_experiment!=(a.phase=='run'):p.error('run requires --enable-gpu-experiment')
    if a.phase=='run' and not a.base_prepared:p.error('--base-prepared required')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs belongs to analyze')
    if not 2<=a.generation_budget<=8 or not 96*2**20<=a.block_budget_bytes<=8*2**30:p.error('bounded generation/evidence budget required')
    return a


def preflight(a):
    require(os.environ.get('R5F_ENABLE_GPU')=='1' and os.environ.get('R5_ENABLE_GPU')=='1',
            'explicit R5_ENABLE_GPU=1 and R5F_ENABLE_GPU=1 required')
    require(a.semantic_revision==TINYBERT_REVISION,'verified immutable TinyBERT revision required')
    spec=importlib.util.spec_from_file_location('r5f_validated_preflight',ROOT/'scripts/110_run_r5_semcache_e2e.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a.evidence_budget_bytes=2*a.block_budget_bytes;m.gpu_preflight(a)
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    require(set(sources()).issubset(tracked),'commit all forensic execution sources before GPU execution')
    require(shutil.disk_usage(a.output_dir).free>2*a.block_budget_bytes+2**30,'insufficient bounded evidence storage')


def worker(a):
    plan=read(a.plan);root=Path(plan['account_root']);output=Path(plan['output']);writer=Writer(root,plan['block_budget_bytes'])
    try:
        verify_sources(plan['sources'])
        if not plan['fixture']:
            require(os.environ.get('R5F_ENABLE_GPU')=='1' and os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),
                    'GPU worker opt-ins/allocation missing')
            require(source_identity()['git_head']==plan['source_commit'],'GPU worker committed identity differs')
        from semcache.edgelora.r5_fidelity_capture import execute, serve_ud
        # Import runtime before scoped IO injection; no frozen file is edited.
        from semcache.edgelora import r5_perf_runtime
        with accounted_modules(writer):
            if a.worker=='ud':serve_ud(plan,output,writer)
            else:execute(plan,root,writer)
        verify_sources(plan['sources']);return 0
    except BaseException:
        traceback.print_exc();return 1


def main(argv=None):
    a=parse(argv)
    if a.worker:return worker(a)
    try:
        candidate=a.output_dir.resolve()
        require(not candidate.is_relative_to(ROOT) or candidate.is_relative_to(ROOT/'results/r5/fidelity'),
                'forensic repository outputs belong under results/r5/fidelity')
        out=fresh(candidate,(a.inputs,a.base_prepared))
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);return 1
    result=dict(execution_status='RUNNING',stage='PREFLIGHT',GPU_NOT_RUN=True,source_identity=source_identity())
    decision=gates()
    try:
        if a.phase=='analyze':
            from semcache.edgelora.r5_fidelity_analysis import analyze, export
            report=analyze(a.inputs);export(out,report);decision=report['decision']
            result.update(analysis_only=True,input_manifest_sha256=sha(a.inputs/'r5_output_manifest.json'))
        elif a.phase=='dry-run':
            write(out/'fidelity_preflight.json',dict(targets=TARGETS,blocks=['A','B'],requests=20,
                generation_budget=a.generation_budget,semantic_revision=TINYBERT_REVISION,
                missing_dependencies=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None],
                required_gpu_optins=['--enable-gpu-experiment','R5_ENABLE_GPU=1','R5F_ENABLE_GPU=1','committed --source-commit','Slurm RTX3090'],
                root_cause='ROOT_CAUSE_UNRESOLVED',job_submitted=False,downloads=False))
        else:
            if a.phase=='fixture':
                from semcache.edgelora.r3a_tiny import prepare_tiny
                base=prepare_tiny(out/'base_prepare')
            else:preflight(a);base=a.base_prepared.resolve()
            config=dict(fixture=a.phase=='fixture',targets=list(TARGETS),generation_budget=a.generation_budget,generation_mode='fixed',
                semantic_revision=a.semantic_revision,evidence_profile='STRICT_AUDIT',warmup_requests=0,legacy_scan=False,
                seed=42,reconcile_every=32,block_budget_bytes=a.block_budget_bytes,source_commit=a.source_commit)
            write(out/'fidelity_config.json',config);write(out/'fidelity_sources.json',sources())
            for condition in ('A','B'):
                result.update(stage='BLOCK_EXECUTION',block=condition);write(out/'fidelity_status.json',result)
                dest=out/'blocks'/condition;es=dest/'es';es.mkdir(parents=True)
                writer=Writer(dest,a.block_budget_bytes,create=True)
                plan=dict(config,condition=condition,block_id=condition,device='cpu' if config['fixture'] else 'cuda:0',
                    base_prepared=str(base),base_manifest_sha256=sha(base/'r3a_prepare_manifest.json'),sources=sources(),
                    account_root=str(dest),output=str(es),evidence_budget_bytes=a.block_budget_bytes,lifecycle_capacity=True)
                writer.write(es/'es_plan.json',plan)
                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                child=spawn([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json')],
                    writer,es/'stdout.log',es/'stderr.log',env=env)
                life=finish(child);writer.write(es/'lifecycle.json',life)
                require(life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],
                        'forensic block failed; preserved '+str(es/'stderr.log'))
                writer.reconcile();writer.write(dest/'r5_final_reconciliation.json',dict(status='PASS',scans=writer.scans));writer.reconcile()
            result.update(stage='INDEPENDENT_AUDIT')
            from semcache.edgelora.r5_fidelity_analysis import analyze,export
            report=analyze(out,sealed=False);export(out,report);decision=report['decision']
            verify_sources(read(out/'fidelity_sources.json'))
            from semcache.edgelora.r4c_evidence import directory_bytes
            require(directory_bytes(out)<2*a.block_budget_bytes+2**30,'EVIDENCE_BUDGET_EXCEEDED: final reports/setup allowance')
            result['GPU_NOT_RUN']=config['fixture']
        result.update(execution_status='COMPLETE',stage='COMPLETE')
    except BaseException as exc:
        decision=gates();result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        write(out/'fidelity_failure.json',result);traceback.print_exc()
    finally:
        if a.phase=='run':
            result['gpu_forward_evidence_present']=any(out.rglob('gpu_execution.json'))
            result['gpu_attempted']=(out/'blocks').exists()
            result['GPU_NOT_RUN']=not result['gpu_attempted']
            result['gpu_status']='GPU_COMPLETE_WITH_AUDIT' if decision['R5_I3F_GPU_FORENSICS_COMPLETE'] else (
                'GPU_ATTEMPTED_INCOMPLETE' if result['gpu_attempted'] else 'GPU_NOT_RUN')
        write(out/'fidelity_decision.json',decision);write(out/'fidelity_results.json',result);seal(out)
    print(json.dumps(dict(status=result['execution_status'],output=str(out))))
    return 0 if result['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));sys.exit(main())
