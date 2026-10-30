#!/usr/bin/env python3
"""One-allocation physical Raw-QKV B1 implementation, B2 gates, B3 measurement."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.parity_forensics_audit import Evidence,verify_prepared
from semcache.edgelora.r3b_analysis import audit_run,flags


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('run','dry-run','fixture'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--test-fixture-only',action='store_true')
    p.add_argument('--warmup-runs',type=int,default=1)
    p.add_argument('--measured-runs',type=int,default=5)
    p.add_argument('--cache-capacity-mib',type=int,default=64)
    p.add_argument('--worker',choices=('es',),help=argparse.SUPPRESS)
    p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or not (a.enable_gpu_experiment or a.test_fixture_only):p.error('worker needs plan and explicit execution mode')
    else:
        if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
        if a.phase=='run' and (not a.enable_gpu_experiment or a.test_fixture_only or not a.device.startswith('cuda')):
            p.error('run requires allocated CUDA and --enable-gpu-experiment; no fixture fallback')
        if a.phase=='fixture' and (not a.test_fixture_only or a.enable_gpu_experiment):p.error('fixture requires --test-fixture-only')
        if a.phase=='dry-run' and (a.enable_gpu_experiment or a.test_fixture_only):p.error('dry-run executes no models/sockets')
        if not 1<=a.warmup_runs<=20 or not 2<=a.measured_runs<=20 or not 1<=a.cache_capacity_mib<=512:
            p.error('warmup 1..20, measured 2..20, capacity 1..512 MiB required')
    return a


def process_runner():
    """Reuse tested descendant-aware process cleanup, without invoking experiment98."""
    spec=importlib.util.spec_from_file_location('r3b_frozen_lifecycle',ROOT/'scripts/98_explore_r3a_cpu_precision.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.run_process


def new_output(path):
    path=path.resolve();base=ROOT/'results/r3/b'
    require(path.is_relative_to(base) and path.relative_to(base).parts and
            path.relative_to(base).parts[0].startswith('raw_qkv_one_shot_'),'new results/r3/b/raw_qkv_one_shot_* directory required')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty/existing-file output')
    path.mkdir(parents=True,exist_ok=True);return path


def sources():
    paths=[Path(__file__).resolve(),ROOT/'scripts/98_explore_r3a_cpu_precision.py',
        ROOT/'src/semcache/edgelora/parity_causal_process.py',ROOT/'src/semcache/edgelora/parity_forensics_audit.py',
        ROOT/'src/semcache/edgelora/cpu_precision_study.py',ROOT/'src/semcache/edgelora/parity_forensics_probe.py',
        ROOT/'jobs/r3b_raw_qkv_one_shot.sh',*sorted((ROOT/'src/semcache/edgelora').glob('r3b_*.py')),
        *sorted((ROOT/'src/semcache/cache').glob('*.py')),ROOT/'src/semcache/semantic/matcher.py',ROOT/'src/semcache/semantic/subsequence.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def worker(a):
    plan=read(a.plan);output=Path(plan['output']);fixture=plan['fixture']
    result={}
    try:
        require(fixture is a.test_fixture_only and (fixture or a.enable_gpu_experiment),'worker opt-in/scope mismatch')
        for name,expected in plan['sources'].items():require(sha(ROOT/name)==expected,'R3-B source changed during execution')
        from semcache.edgelora.r3b_runtime import execute
        result=execute(plan,output)
    except BaseException as exc:
        partial=read(output/'r3b_es_partial.json') if (output/'r3b_es_partial.json').is_file() else {}
        result=dict(partial,execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
    result.update(**flags(fixture))
    write(output/'r3b_worker_result.json',result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def human(report):
    lines=['R3B_PHYSICAL_RAW_QKV_DIAGNOSTIC','PRIMARY_PERFORMANCE_ELIGIBLE=false',
        'Execution: '+report['execution_status'],'GPU: '+report['gpu_findings_status'],
        'Reason: '+report.get('reason','none'),'Reuse scope: exact full-prompt replay only']
    for stage,value in report['stages'].items():lines.append(stage+': '+value['status'])
    if report.get('audit'):
        lines.append('Native strict failures (separate from cache parity): '+str(report['audit']['native_strict_failures']))
        lines.append(json.dumps(report['audit']['summary']['by_mode'],sort_keys=True))
    lines+=['No semantic generalization, correctness-qualified reuse speedup, R2-D/R3-C break-even or remote-network claim.',
        'HIT integrity copy/hash and clone overhead is included in request wall; nested timings must not be summed.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker(a)
    try:output=new_output(a.output_dir)
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';evidence=Evidence()
    report=dict(execution_status='RUNNING',created_at=now(),phase=a.phase,**flags(fixture),
        production_source_identity=source_identity(),sources=sources(),gpu_findings_status='NOT_RUN',
        gpu_inference_performed=False,actual_network_socket_used=False,
        stages={s:dict(status='NOT_RUN') for s in ('B1','B2','B3')},
        cli=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
    try:
        for name,expected in report['sources'].items():evidence.file(ROOT/name,expected)
        missing=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None]
        if a.phase=='dry-run':report['readiness']=dict(missing_dependencies=missing,no_model_or_socket_executed=True)
        else:
            prepared=output/'prepare'
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING incompatible with measurement scope')
            runner=process_runner()
            if fixture:
                from semcache.edgelora.r3a_tiny import prepare_tiny
                prepare_tiny(prepared)
            else:
                require(report['production_source_identity']['git_branch']=='exp/raw-semcache-r3','wrong experiment branch')
                require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),'one explicit Slurm GPU allocation required')
                require(not missing,'GPU experiment NOT_RUN; missing dependencies: '+', '.join(missing))
                runner([sys.executable,'-u',str(ROOT/'scripts/95_run_r3a_distributed_no_reuse.py'),
                    '--phase','prepare','--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','32',
                    '--device',a.device,'--adapter','user_a','--seed','42','--output-dir',str(prepared)],output/'prepare_process',env=env)
            pm,ref,am=verify_prepared(prepared,evidence)
            if not fixture:
                require(read(prepared/'r3a_preflight.json')['allocation_environment']['SLURM_JOB_ID']==os.environ['SLURM_JOB_ID'],
                        'fresh prepare must use the same allocation')
            report['prepared']=dict(manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),identity=pm['identity'],
                adapter_matrix_sha256=am['matrix_sha256'],reference_logits_sha256=ref['logits_sha256'],gpu=pm.get('gpu'))
            es=output/'es';es.mkdir()
            plan=dict(prepared=str(prepared),prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),
                output=str(es),fixture=fixture,device='cpu' if fixture else a.device,warmup=a.warmup_runs,
                measured=a.measured_runs,capacity_bytes=a.cache_capacity_mib*1024*1024,sources=report['sources'])
            write(es/'r3b_plan.json',plan)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'r3b_plan.json'),
                '--test-fixture-only' if fixture else '--enable-gpu-experiment']
            pid=runner(command,es,timeout=2400.,env=env)
            result=read(es/'r3b_worker_result.json');require(result['execution_status']=='COMPLETE' and result['pid']==pid,'ES worker status/PID mismatch')
            require(result['cleanup']['all_children_exited'] and all(c['returncode']==0 and not c['forced_termination']
                for c in result['cleanup']['children']),'unclean original UD lifecycle')
            report['worker_result']=result;report['stages']=result['stages']
            report['audit']=audit_run(es,pm,result,a.warmup_runs,a.measured_runs)
            report['actual_network_socket_used']=True
            report['gpu_findings_status']='TEST_FIXTURE_ONLY_NO_GPU' if fixture else 'COMPLETE'
            report['gpu_inference_performed']=not fixture
        evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        path=output/'es/r3b_worker_result.json'
        if path.is_file():
            failure=read(path);report['worker_failure']=failure;report['stages']=failure.get('stages',report['stages'])
            report['reason']+='; ES: '+failure.get('reason','unknown')
        elif (output/'es/r3b_stage_gates.json').is_file():
            report['stages']=read(output/'es/r3b_stage_gates.json')
            for value in report['stages'].values():
                if value['status']=='RUNNING':value.update(status='INCOMPLETE_FAILED',reason='worker exited before final report')
        traceback.print_exc()
    finally:
        raw_rows=output/'es/r3b_requests.jsonl'
        (output/'r3b_requests.jsonl').write_bytes(raw_rows.read_bytes() if raw_rows.is_file() else b'')
        report['evidence_files']=list(evidence.files.values());report['finished_at']=now()
        report['actual_network_socket_used']=report['actual_network_socket_used'] or any(p.stat().st_size>0 for p in output.glob('es/r3b_wire_trace.jsonl'))
        actual_gpu=(output/'prepare/r3a_prepare_execution.json').is_file()
        report['gpu_inference_performed']=report['gpu_inference_performed'] or actual_gpu
        if actual_gpu and report['execution_status']=='FAILED':report['gpu_findings_status']='INCOMPLETE_FAILED'
        report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and
            p.name not in ('r3b_one_shot_results.json','r3b_one_shot_summary.txt')}
        write(output/'r3b_one_shot_results.json',report)
        (output/'r3b_one_shot_summary.txt').write_text(human(report))
    print(json.dumps(dict(report=str(output/'r3b_one_shot_results.json'),execution_status=report['execution_status'],
        gpu_findings_status=report['gpu_findings_status'],stages={s:v['status'] for s,v in report['stages'].items()}),sort_keys=True))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    sys.exit(main())
