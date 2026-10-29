#!/usr/bin/env python3
"""Fresh one-allocation native preparation and exactly three CPU precision diagnostics."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import THRESHOLDS,now,read,sha,source_identity,write
from semcache.edgelora.parity_forensics_audit import Evidence,verify_prepared
from semcache.edgelora.cpu_precision_study import CANDIDATES,CLASSIFICATION,decide


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('run','dry-run'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--enable-gpu-probes',action='store_true')
    p.add_argument('--worker',choices=('allocation','native','candidate'),help=argparse.SUPPRESS)
    p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or not a.enable_gpu_probes:p.error('private worker requires explicit opt-in and plan')
    else:
        if not a.phase or not a.output_dir:p.error('--phase and --output-dir required; no historical prepared inputs accepted')
        if a.phase=='run' and not a.enable_gpu_probes:p.error('run requires --enable-gpu-probes')
        if a.phase=='dry-run' and a.enable_gpu_probes:p.error('dry-run does not enable GPU inference')
        if not a.device.startswith('cuda'):p.error('study requires allocated CUDA base model')
    return a


def new_output(path):
    path=path.resolve();base=ROOT/'results/r3/a'
    require(path.is_relative_to(base) and path.relative_to(base).parts and
            path.relative_to(base).parts[0].startswith('precision_study_'), 'new results/r3/a/precision_study_* output required')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty/existing-file output')
    path.mkdir(parents=True,exist_ok=True);return path


def _process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19],fields[0]  # start ticks, state (avoid reused PIDs)
    except FileNotFoundError:return None


def _descendants(pid):
    found={}
    try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:return found
    for child in children:
        child=int(child);identity=_process_identity(child)
        if identity:found[child]=identity[0];found.update(_descendants(child))
    return found


def run_process(command,directory,*,timeout=1500.,env=None):
    """Track/reap before the next model, including prepare95's separate GPU child."""
    directory.mkdir(parents=True,exist_ok=True)
    stdout=(directory/'stdout.log').open('w');stderr=(directory/'stderr.log').open('w')
    process=None;descendants={};reason=None;code=None;forced=False
    try:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        deadline=time.monotonic()+timeout
        while True:
            descendants.update(_descendants(process.pid))
            try:process.wait(timeout=min(.1,max(.001,deadline-time.monotonic())));break
            except subprocess.TimeoutExpired:
                if time.monotonic()>=deadline:raise TimeoutError('diagnostic child deadline exceeded')
        code=process.returncode
        require(code==0,f'child exited {code}; see {directory}/stderr.log')
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}';raise
    finally:
        cleanup=[]
        if process is not None:
            descendants.update(_descendants(process.pid));forced=process.poll() is None
            # Production prepare uses a separate process group. Kill tracked
            # descendants explicitly on failure rather than orphaning its model.
            for pid,start in descendants.items():
                current=_process_identity(pid)
                if current and current[0]==start and current[1]!='Z':
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    cleanup.append(dict(pid=pid,start_ticks=start,termination_required=True))
            from semcache.edgelora.parity_causal_process import finish
            life=finish((process,stdout,stderr))
            for item in cleanup:
                current=_process_identity(item['pid'])
                if current and current[0]==item['start_ticks'] and current[1]!='Z':
                    try:os.kill(item['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
            limit=time.monotonic()+3.
            while time.monotonic()<limit and any((identity:=_process_identity(p)) and identity[0]==s and identity[1]!='Z'
                                                 for p,s in descendants.items()):time.sleep(.01)
            life['tracked_descendants']=dict(pid_start_ticks=descendants,cleanup=cleanup,
                all_stopped=not any((identity:=_process_identity(p)) and identity[0]==s and identity[1]!='Z' for p,s in descendants.items()))
        else:
            stdout.close();stderr.close();life=dict(all_children_exited=True,not_started=True)
        life.update(command=command,reason=reason)
        write(directory/'lifecycle.json',life)
    require(life['all_children_exited'] and not forced and not life.get('descendant_group_cleanup_applied') and
            life['tracked_descendants']['all_stopped'] and not cleanup,'unclean model process lifecycle')
    return process.pid


def child(kind,plan,root,env):
    name=plan.get('candidate') if kind=='candidate' else kind
    output=root/name;output.mkdir()
    write(output/'diagnostic_plan.json',dict(plan,output=str(output)))
    try:
        pid=run_process([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',kind,
            '--plan',str(output/'diagnostic_plan.json'),'--enable-gpu-probes'],output,env=env)
    except BaseException as exc:
        failure=read(output/'worker_result.json') if (output/'worker_result.json').is_file() else {}
        raise RuntimeError(f"{name}: {failure.get('reason',str(exc))}; evidence: {output}") from exc
    result=read(output/'worker_result.json')
    require(result['execution_status']=='COMPLETE' and result['process']['pid']==pid,'worker identity/status mismatch')
    return result


def worker(a):
    plan=read(a.plan);output=Path(plan['output'])
    try:
        from semcache.edgelora.cpu_precision_study import configure,native_worker,candidate_worker
        if a.worker=='allocation':result=dict(execution_status='COMPLETE',process=configure(plan,reference_required=False))
        else:result=native_worker(plan,output) if a.worker=='native' else candidate_worker(plan,output)
    except BaseException as exc:
        result=dict(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',pid=os.getpid(),traceback=traceback.format_exc())
        traceback.print_exc()
    result.update(classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
        actual_network_socket_used=False,physical_tcp_parity_validated=False,original_151730_evidence_loaded=False,
        gpu_inference_performed=(output/'gpu_execution.json').is_file())
    if plan.get('prepared'):
        result['reference_identity']=dict(prepared_manifest_sha256=plan['prepared_manifest_sha256'],
            allocated_gpu_uuid=plan['allocated_gpu_uuid'],prepared=str(plan['prepared']))
    write(output/'worker_result.json',result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def human(report):
    lines=[CLASSIFICATION,'PRIMARY_PERFORMANCE_ELIGIBLE=false','Execution: '+report['execution_status'],
        'GPU findings: '+report['gpu_findings_status'],'Conclusion: '+report['decision']['conclusion'],
        'Reason: '+report.get('reason','none'),'Original job151730 evidence loaded/verified: false',
        'New cohort GPU UUID: '+str(report.get('prepared',{}).get('gpu_uuid','NOT_RUN')),
        'Native exact: '+str(report.get('native_control',{}).get('native_exact','NOT_RUN'))]
    for name,r in report['candidates'].items():
        parity=r.get('parity')
        lines.append(name+': '+r['execution_status']+' '+json.dumps({k:parity[k] for k in
            ('passed','max_abs','relative_l2','tolerance_violations','last_token_kl','last_token_argmax_agreement')}
            if parity else None,sort_keys=True))
        if parity:
            lines.append('  full logits SHA256: '+r['logits']['sha256'])
            lines.append('  actual QKV accounting: '+json.dumps(r['accounting'],sort_keys=True))
    lines += ['A local candidate PASS does not validate physical CPU-UD/TCP parity or performance.',
              'Three strict failures end this study; no automatic sweep or production modification.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker(a)
    try:output=new_output(a.output_dir)
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    source=source_identity();sources=[Path(__file__).resolve(),ROOT/'src/semcache/edgelora/cpu_precision_study.py',
        ROOT/'src/semcache/edgelora/parity_forensics_probe.py',ROOT/'src/semcache/edgelora/parity_forensics_audit.py',
        ROOT/'src/semcache/edgelora/parity_causal_process.py',
        ROOT/'jobs/r3a_cpu_precision_one_shot.sbatch']
    evidence=Evidence()
    report=dict(execution_status='RUNNING',classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
        created_at=now(),phase=a.phase,production_source_identity=source,
        diagnostic_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},thresholds=dict(THRESHOLDS),
        gpu_findings_status='NOT_RUN',gpu_inference_performed=False,actual_network_socket_used=False,
        physical_tcp_parity_validated=False,cache_reuse_enabled=False,performance_measurement_performed=False,
        original_151730_evidence_loaded=False,original_151730_evidence_verified=False,
        cohort_scope='fresh reference on this allocation; no historical UUID or expected numeric result constraint',
        cli=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
        candidates={name:dict(execution_status='NOT_RUN') for name in CANDIDATES},
        decision=dict(conclusion='NOT_RUN',candidate_local_parity_pass=[]))
    try:
        for p in sources:evidence.file(p,report['diagnostic_source_hashes'][str(p.relative_to(ROOT))])
        missing=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None]
        if a.phase=='dry-run':
            report['readiness']=dict(missing_dependencies=missing,no_model_or_socket_executed=True,
                prepare_required=True,branch=source['git_branch'],slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        else:
            require(source['git_branch']=='exp/raw-semcache-r3','branch mismatch: '+str(source['git_branch']))
            require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'), 'one Slurm GPU allocation required')
            require(not missing,'GPU study NOT_RUN; missing dependencies: '+', '.join(missing))
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            plan=dict(enable_gpu_probes=True,device=a.device,diagnostic_source_hashes=report['diagnostic_source_hashes'])
            allocation=child('allocation',plan,output,env);report['allocation']=allocation
            uuid=allocation['process']['gpu']['uuid'];print('Allocated GPU UUID: '+uuid,flush=True)
            prepared=output/'prepare'
            prepare_command=[sys.executable,'-u',str(ROOT/'scripts/95_run_r3a_distributed_no_reuse.py'),
                '--phase','prepare','--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','32',
                '--device',a.device,'--adapter','user_a','--seed',str(a.seed),'--output-dir',str(prepared)]
            # Wrapper logs are outside the immutable prepare artifact set.
            prepare_pid=run_process(prepare_command,output/'prepare_process',env=env)
            pm,ref,am=verify_prepared(prepared,evidence)
            require(pm['gpu']['uuid']==uuid and pm['identity']['versions']==allocation['process']['versions'] and
                pm['code']['files']==source['files'],'fresh prepared GPU/version/source mismatch')
            preflight=read(prepared/'r3a_preflight.json')
            require(preflight['allocation_environment']['SLURM_JOB_ID']==os.environ['SLURM_JOB_ID'], 'prepare left original allocation')
            report['prepared']=dict(directory=str(prepared),manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),
                identity=pm['identity'],identity_sha256=pm['identity_sha256'],gpu_uuid=uuid,
                adapter_sha256=am['matrix_sha256'],reference_logits_sha256=ref['logits_sha256'],
                reference_logits_shape=ref['logits_shape'],reference_logits_dtype=ref['logits_dtype'])
            plan.update(prepared=str(prepared),prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),
                allocated_gpu_uuid=uuid,cpu_affinity=allocation['process']['affinity'])
            native=child('native',plan,output,env);report['native_control']=native
            require(native['native_exact'] is True,'REFERENCE_NONREPRODUCIBLE: stop before candidates')
            plan.update(native_report=str(output/'native/worker_result.json'),native_report_sha256=sha(output/'native/worker_result.json'))
            pids=[prepare_pid,pm['preparation_pid'],native['process']['pid']]
            for name in CANDIDATES:
                try:result=child('candidate',dict(plan,candidate=name),output,env)
                except BaseException as exc:
                    path=output/name/'worker_result.json'
                    report['candidates'][name]=read(path) if path.is_file() else dict(execution_status='FAILED',reason=str(exc))
                    raise
                report['candidates'][name]=result;pids.append(result['process']['pid'])
            require(len(set(pids))==len(pids),'independent sequential process PIDs required')
            report['decision']=decide(True,report['candidates'])
            report['gpu_findings_status']='COMPLETE'
        evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        report['decision']=dict(conclusion='REFERENCE_NONREPRODUCIBLE' if 'REFERENCE_NONREPRODUCIBLE' in str(exc) else 'STUDY_FAILED',
            candidate_local_parity_pass=[],no_imputation=True)
        traceback.print_exc()
    finally:
        inferred=any(output.glob('*/gpu_execution.json')) or (output/'prepare/r3a_prepare_execution.json').is_file()
        report['gpu_inference_performed']=inferred
        if inferred and report['execution_status']=='FAILED':report['gpu_findings_status']='INCOMPLETE_FAILED'
        report['evidence_files']=list(evidence.files.values());report['finished_at']=now()
        report['worker_failures']={}
        for path in output.glob('*/worker_result.json'):
            try:item=read(path)
            except (ValueError,OSError) as exc:item=dict(execution_status='INVALID_ARTIFACT',reason=str(exc))
            if item.get('execution_status')!='COMPLETE':report['worker_failures'][str(path.relative_to(output))]=item
        report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and
            p.name not in ('r3a_cpu_precision_results.json','r3a_cpu_precision_summary.txt')}
        write(output/'r3a_cpu_precision_results.json',report)
        (output/'r3a_cpu_precision_summary.txt').write_text(human(report))
    print(json.dumps(dict(execution_status=report['execution_status'],conclusion=report['decision']['conclusion'],
        gpu_findings_status=report['gpu_findings_status'],report=str(output/'r3a_cpu_precision_results.json')),sort_keys=True))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    sys.exit(main())
