#!/usr/bin/env python3
"""One calibrated localhost TCP proxy grid; actual cache workload and signed contrasts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.parity_forensics_audit import Evidence,verify_prepared
from semcache.edgelora.r3c_analysis import (FRACTIONS,analyze,audit_model,budget,export_analysis,flags,
    profiles,r2d_comparison,schedule,validate_profiles)
from semcache.edgelora.r3c_network_proxy import FrameProxy,control
from semcache.edgelora.r3c_process import spawn,ready,reap


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('run','fixture','calibrate','dry-run','analyze'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--r2d-dir',type=Path,help='optional read-only R2-D exported artifact bundle; never pooled timings')
    p.add_argument('--device',default='cuda:0');p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--test-fixture-only',action='store_true')
    p.add_argument('--rtt-ms',default=None);p.add_argument('--bandwidth-mbps',default=None)
    p.add_argument('--profiles-json',type=Path,help='bounded asymmetric profile list; requires zero/unlimited control')
    p.add_argument('--hit-fractions',default='0,0.125,0.25,0.5,1')
    p.add_argument('--requests-per-arm',type=int,default=8);p.add_argument('--warmup-runs',type=int,default=1)
    p.add_argument('--seed',type=int,default=42,help='predeclared workload/bootstrap seed; fresh adapter seed remains 42')
    p.add_argument('--worker',choices=('proxy','echo','ud','es'),help=argparse.SUPPRESS)
    p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or not (a.enable_gpu_experiment or a.test_fixture_only):p.error('worker requires plan and explicit scope')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.phase=='run' and (not a.enable_gpu_experiment or a.test_fixture_only or not a.device.startswith('cuda')):
        p.error('run requires explicit GPU experiment opt-in; no synthetic fallback')
    if a.phase=='fixture' and (not a.test_fixture_only or a.enable_gpu_experiment):p.error('fixture requires --test-fixture-only')
    if a.phase in ('dry-run','calibrate','analyze') and (a.enable_gpu_experiment or a.test_fixture_only):p.error('CPU phase does not enable model workers')
    if a.phase=='analyze' and not a.inputs:p.error('--inputs required for read-only analysis')
    if a.phase!='analyze' and a.inputs:p.error('--inputs is only for analysis')
    try:
        if a.profiles_json:
            require(a.rtt_ms is None and a.bandwidth_mbps is None,'profiles-json and grid axes are exclusive')
            a.profiles=validate_profiles(read(a.profiles_json))
        else:
            rtts=[float(v) for v in (a.rtt_ms or '0,10').split(',')]
            bandwidth=[None if v.strip().lower()=='unlimited' else float(v) for v in (a.bandwidth_mbps or 'unlimited,200').split(',')]
            a.profiles=profiles(rtts,bandwidth)
        fractions=[float(v) for v in a.hit_fractions.split(',')]
        a.schedule=schedule(a.profiles,fractions,a.requests_per_arm,a.warmup_runs,a.seed)
        a.budget=budget(dict(profiles=a.profiles,schedule=a.schedule))
    except (ValueError,TypeError,KeyError) as exc:p.error(str(exc))
    return a


def sources():
    paths=[Path(__file__).resolve(),ROOT/'scripts/99_run_r3b_raw_qkv_one_shot.py',ROOT/'scripts/98_explore_r3a_cpu_precision.py',
        ROOT/'jobs/r3c_network_one_shot.sh',ROOT/'src/semcache/edgelora/parity_causal_process.py',
        ROOT/'src/semcache/edgelora/cpu_precision_study.py',ROOT/'src/semcache/edgelora/parity_forensics_probe.py',
        ROOT/'src/semcache/edgelora/parity_forensics_audit.py',*[ROOT/n for n in source_identity()['files']],
        *sorted((ROOT/'src/semcache/edgelora').glob('r3c_*.py')),*sorted((ROOT/'src/semcache/edgelora').glob('r3b_*.py')),
        *sorted((ROOT/'src/semcache/cache').glob('*.py')),ROOT/'src/semcache/semantic/matcher.py',ROOT/'src/semcache/semantic/subsequence.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def process_runner():
    spec=importlib.util.spec_from_file_location('r3c_frozen_runner',ROOT/'scripts/99_run_r3b_raw_qkv_one_shot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.process_runner()


def new_output(path,protected=None):
    path=path.resolve();base=ROOT/'results/r3/c'
    require(protected is None or not path.is_relative_to(protected.resolve()),'output must not modify input cohort')
    require(path.is_relative_to(base) and path.relative_to(base).parts and path.relative_to(base).parts[0].startswith('network_one_shot_'),
            'new results/r3/c/network_one_shot_* directory required')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty/existing-file output')
    path.mkdir(parents=True,exist_ok=True);return path


def worker(a):
    plan=read(a.plan);output=Path(plan['output']);proxy=None;result={}
    try:
        require(plan['fixture']==a.test_fixture_only,'worker cohort scope mismatch')
        for name,expected in plan['sources'].items():require(sha(ROOT/name)==expected,'source changed during R3-C')
        if a.worker=='proxy':
            proxy=FrameProxy(plan['proxy_token'],output);write(output/'proxy_ready.json',proxy.start())
            while not proxy.stop_event.wait(.1):pass
            life=proxy.close();proxy=None
            require(not life['errors'],'proxy transport failed: '+str(life['errors']))
            result=dict(execution_status='COMPLETE',pid=os.getpid(),lifecycle=life,model_inference_performed=False)
        elif a.worker=='echo':
            from semcache.edgelora.r3c_calibration import echo_worker
            result=echo_worker(output)
        elif a.worker=='ud':
            from functools import partial
            from semcache.edgelora import user_device
            original=user_device.Channel
            try:
                # Only a constructor timeout override in this NEW worker. Frame
                # code and original FP32 CPU math remain completely unchanged.
                user_device.Channel=partial(original,timeout=120.)
                user_device.serve(plan['prepared'],output,plan['session_id'])
            finally:user_device.Channel=original
            result=dict(execution_status='COMPLETE',pid=os.getpid(),model_inference_performed=False,
                user_device_cuda_used=False,channel_timeout_seconds=120.)
        else:
            require(plan['fixture'] or a.enable_gpu_experiment,'GPU model worker opt-in missing')
            from semcache.edgelora.r3c_runtime import execute
            result=execute(plan,output)
    except BaseException as exc:
        partial=read(output/'r3c_es_partial.json') if (output/'r3c_es_partial.json').is_file() else {}
        result=dict(partial,execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if proxy is not None:result['proxy_lifecycle']=proxy.close()
    result.update(flags(plan['fixture']));result.update(pid=os.getpid(),worker=a.worker,finished_at=now())
    write(output/(a.worker+'_result.json'),result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def load_run(directory):
    directory=directory.resolve();report=read(directory/'r3c_one_shot_results.json')
    require(report['execution_status']=='COMPLETE' and report.get('audit',{}).get('status')=='PASS' and not report.get('analysis_only'),
            'a completed full model run with audit is required')
    required={'r3c_requests.jsonl','r3c_network_calibration.json','es/es_result.json','proxy/proxy_frames.jsonl',
        'proxy/proxy_lifecycle.json','es/es_wire.jsonl','es/r3a_ud_wire_trace.jsonl','es/r3a_ud_result.json'}
    require(required.issubset(report['output_hashes']),'missing required R3-C artifacts/hashes')
    for name,expected in report['output_hashes'].items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and sha(p)==expected,'R3-C artifact SHA mismatch: '+name)
    require((directory/'r3c_requests.jsonl').read_bytes()==(directory/'es/r3c_requests.jsonl').read_bytes(),'request export copies differ')
    for name,expected in report['sources'].items():require(sha(ROOT/name)==expected,'unsupported changed analysis source')
    plan=read(directory/'es/es_plan.json');result=read(directory/'es/es_result.json')
    audit_model(directory,plan,result)
    return report,plan


def human(report):
    return '\n'.join(['R3C_APPLICATION_LAYER_NETWORK_DIAGNOSTIC','PRIMARY_PERFORMANCE_ELIGIBLE=false',
        'Execution: '+report['execution_status'],'GPU findings: '+report['gpu_findings_status'],
        'Reason: '+report.get('reason','none'),
        'Native PEFT strict parity passed: '+str(report.get('native_peft_strict_parity_passed')),
        'Real remote network measured: false; userspace framed TCP emulation over localhost.',
        'Hit-fraction decisions: '+json.dumps(report.get('decision',{}).get('hit_fraction_crossovers',{}),sort_keys=True),
        'Controlled exact-prompt schedules are not natural HIT rates. No R2-D numerical validation or qualified speedup.'])+'\n'


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker(a)
    try:output=new_output(a.output_dir,a.inputs)
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';evidence=Evidence();children=[];proxy_info=None;token=uuid.uuid4().hex
    report=dict(execution_status='RUNNING',created_at=now(),phase=a.phase,**flags(fixture),sources=sources(),
        production_source_identity=source_identity(),gpu_findings_status='NOT_RUN',gpu_inference_performed=False,
        actual_network_socket_used=False,native_peft_strict_parity_passed=None,
        profiles=a.profiles,schedule=a.schedule,budget=a.budget,seed=a.seed,
        user_reported_151857_evidence_accessed=False,cli=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
    points=[];decision=dict(execution_status='NOT_RUN',status='NOT_ESTIMABLE',**flags(fixture))
    try:
        for name,expected in report['sources'].items():evidence.file(ROOT/name,expected)
        if a.profiles_json:evidence.file(a.profiles_json)
        report['r2d_comparison']=r2d_comparison(a.r2d_dir)
        missing=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None]
        if a.phase=='dry-run':report['readiness']=dict(missing_dependencies=missing,no_model_or_socket_executed=True)
        elif a.phase=='analyze':
            old,plan=load_run(a.inputs)
            from semcache.edgelora.r3a_evidence import jsonl
            rows=jsonl(a.inputs/'r3c_requests.jsonl');points,decision=analyze(rows,plan)
            report.update(flags(plan['fixture']));report.update(analysis_only=True,input_run_sha256=sha(a.inputs/'r3c_one_shot_results.json'),
                input_gpu_inference_performed=old['gpu_inference_performed'],native_peft_strict_parity_passed=old['native_peft_strict_parity_passed'],
                profiles=plan['profiles'],schedule=plan['schedule'],budget=plan['budget'],seed=plan['seed'])
            (output/'r3c_requests.jsonl').write_bytes((a.inputs/'r3c_requests.jsonl').read_bytes())
            write(output/'r3c_network_calibration.json',plan['calibration'])
        else:
            if a.phase=='run':
                require(report['production_source_identity']['git_branch']=='exp/raw-semcache-r3','wrong experiment branch')
                require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),'one explicit Slurm GPU allocation required')
                require(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,'exactly one allocated visible GPU required')
                require(not missing,'GPU experiment NOT_RUN; missing dependencies: '+', '.join(missing))
                from semcache.edgelora.r3c_runtime import offline_model_preflight
                report['model_preflight']=offline_model_preflight();write(output/'r3c_preflight.json',report['model_preflight'])
            require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates measurement scope')
            plan=dict(fixture=fixture,proxy_token=token,sources=report['sources'],profiles=a.profiles,schedule=a.schedule,
                seed=a.seed,budget=a.budget,device='cpu' if fixture else a.device)
            echo=spawn('echo',plan,output/'calibration');children.append(('echo',echo,output/'calibration'))
            echo_info=ready(echo,output/'calibration/echo_ready.json')
            require(echo_info['pid']==echo[0].pid,'calibration sink PID mismatch')
            relay=spawn('proxy',plan,output/'proxy');children.append(('proxy',relay,output/'proxy'))
            proxy_info=ready(relay,output/'proxy/proxy_ready.json')
            require(proxy_info['pid']==relay[0].pid,'proxy PID mismatch')
            from semcache.edgelora.r3c_calibration import calibrate
            calibration=calibrate(a.profiles,proxy_info,token,echo_info,output/'calibration')
            write(output/'r3c_network_calibration.json',calibration);report['actual_network_socket_used']=True
            echo[0].wait(timeout=10.);require(echo[0].returncode==0,'calibration sink failed')
            require(calibration['execution_status']=='COMPLETE','INVALID_IMPAIRMENT: calibration failed; no GPU model experiment')
            if a.phase!='calibrate':
                prepared=output/'prepare';runner=process_runner()
                if fixture:
                    from semcache.edgelora.r3a_tiny import prepare_tiny
                    prepare_tiny(prepared)
                else:
                    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                    runner([sys.executable,'-u',str(ROOT/'scripts/95_run_r3a_distributed_no_reuse.py'),'--phase','prepare',
                        '--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','32','--device',a.device,
                        '--adapter','user_a','--seed','42','--output-dir',str(prepared)],output/'prepare_process',env=env)
                pm,ref,am=verify_prepared(prepared,evidence)
                if not fixture:require(read(prepared/'r3a_preflight.json')['allocation_environment']['SLURM_JOB_ID']==os.environ['SLURM_JOB_ID'],
                                        'prepare left this allocation')
                identity=dict(identity=pm['identity'],identity_sha256=pm['identity_sha256'],prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),
                    adapter_matrix_sha256=am['matrix_sha256'],gpu_uuid=pm.get('gpu',{}).get('uuid','TEST_FIXTURE_ONLY'))
                plan.update(prepared=str(prepared),prepared_manifest_sha256=identity['prepared_manifest_sha256'],reference_identity=identity,
                    proxy=proxy_info,calibration=calibration)
                es=output/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
                runner([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json'),
                    '--test-fixture-only' if fixture else '--enable-gpu-experiment'],es,timeout=7500.,
                    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                result=read(es/'es_result.json');require(result['execution_status']=='COMPLETE','ES worker failed')
                require(result['ud_lifecycle']['returncode']==0 and not result['ud_lifecycle']['forced_termination'],'unclean CPU UD exit')
                report['audit']=audit_model(output,plan,result);report['model_result']=result
                report['native_peft_strict_parity_passed']=report['audit']['native_peft_strict_parity_passed']
                from semcache.edgelora.r3a_evidence import jsonl
                rows=jsonl(es/'r3c_requests.jsonl');points,decision=analyze(rows,plan)
                report['gpu_findings_status']='TEST_FIXTURE_ONLY_NO_GPU' if fixture else 'COMPLETE'
                report['gpu_inference_performed']=not fixture
            control(proxy_info['control_port'],token,'STOP');relay[0].wait(timeout=10.);require(relay[0].returncode==0,'proxy failed')
        evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if (output/'es/es_result.json').is_file():
            report['failed_model_worker']=read(output/'es/es_result.json')
            if report['failed_model_worker'].get('reason'):
                report['reason']+='; ES: '+report['failed_model_worker']['reason']
        decision=dict(execution_status='FAILED',status='NOT_ESTIMABLE',reason=report['reason'],**flags(fixture));points=[]
        traceback.print_exc()
    finally:
        for kind,child,directory in reversed(children):
            lifecycle=reap(child,directory/(kind+'_parent_lifecycle.json'))
            if report['execution_status']=='COMPLETE' and (lifecycle['returncode']!=0 or lifecycle['forced_termination']):
                report.update(execution_status='FAILED',reason='unclean '+kind+' lifecycle')
                decision=dict(execution_status='FAILED',status='NOT_ESTIMABLE',reason=report['reason'],**flags(fixture));points=[]
        if (output/'es/r3c_requests.jsonl').is_file():(output/'r3c_requests.jsonl').write_bytes((output/'es/r3c_requests.jsonl').read_bytes())
        elif not (output/'r3c_requests.jsonl').exists():(output/'r3c_requests.jsonl').touch()
        if not (output/'r3c_network_calibration.json').exists():
            partial=output/'calibration/r3c_network_calibration.json'
            write(output/'r3c_network_calibration.json',read(partial) if partial.is_file() else dict(
                execution_status='INCOMPLETE' if (output/'calibration/calibration_probes.jsonl').is_file() else 'NOT_RUN',
                reason=report.get('reason'),conditions=[dict(condition=p,status='NOT_ESTIMABLE') for p in a.profiles]))
        if (output/'prepare/r3a_prepare_execution.json').is_file():
            report['gpu_inference_performed']=True
            if report['execution_status']=='FAILED':report['gpu_findings_status']='INCOMPLETE_FAILED'
        if report['execution_status']=='FAILED' and (output/'r3c_requests.jsonl').is_file():
            from semcache.edgelora.r3a_evidence import jsonl
            available=[r['native_parity'] for r in jsonl(output/'r3c_requests.jsonl') if 'native_parity' in r]
            if available:report['native_peft_strict_parity_passed']=all(p['passed'] for p in available)
        report['decision']=decision;report['evidence_files']=list(evidence.files.values());report['finished_at']=now()
        export_analysis(output,points,decision)
        report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and
            p.name not in ('r3c_one_shot_results.json','r3c_one_shot_summary.txt')}
        write(output/'r3c_one_shot_results.json',report);(output/'r3c_one_shot_summary.txt').write_text(human(report))
    print(json.dumps(dict(report=str(output/'r3c_one_shot_results.json'),execution_status=report['execution_status'],
        gpu_findings_status=report['gpu_findings_status']),sort_keys=True))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);sys.exit(main())
