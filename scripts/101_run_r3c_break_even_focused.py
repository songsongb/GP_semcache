#!/usr/bin/env python3
"""Fixed-count retrospective analysis and one 126-request physical focused run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.parity_forensics_audit import Evidence,verify_prepared
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r3c_process import spawn,ready,reap
from semcache.edgelora.r3c_network_proxy import control
from semcache.edgelora.r3cf_schedule import make_schedule,budget,flags
from semcache.edgelora.r3cf_analysis import export,focused_analysis
from semcache.edgelora.r3cf_evidence import audit_physical,legacy_runner,legacy_reanalysis,load_focused,verify_sources


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('run','fixture','reanalyze','analyze','dry-run'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--inputs',type=Path);p.add_argument('--legacy-input',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resamples',type=int,default=10000)
    p.add_argument('--enable-gpu-experiment',action='store_true');p.add_argument('--test-fixture-only',action='store_true')
    p.add_argument('--worker',choices=('es',),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or a.enable_gpu_experiment==a.test_fixture_only:p.error('worker requires plan and exactly one scope opt-in')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.phase=='run' and (not a.enable_gpu_experiment or a.test_fixture_only or not a.device.startswith('cuda')):p.error('explicit GPU opt-in required')
    if a.phase=='fixture' and (not a.test_fixture_only or a.enable_gpu_experiment):p.error('fixture requires explicit TEST_FIXTURE_ONLY')
    if a.phase in ('reanalyze','analyze','dry-run') and a.enable_gpu_experiment:p.error('analysis phases never execute GPU inference')
    if a.test_fixture_only and a.phase not in ('fixture','reanalyze'):p.error('fixture flag is only for fixture or explicitly labeled legacy fixtures')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs required only for analyze')
    if (a.legacy_input is not None)!=(a.phase=='reanalyze'):p.error('--legacy-input required only for reanalyze')
    if not 10000<=a.resamples<=50000:p.error('10000..50000 resamples required')
    try:a.schedule_manifest=make_schedule(a.seed)
    except ValueError as exc:p.error(str(exc))
    return a


def sources():
    result=legacy_runner().sources()
    paths=[Path(__file__),ROOT/'jobs/r3c_focused_break_even.sh',ROOT/'docs/R3C_FOCUSED_BREAK_EVEN.md',
        *sorted((ROOT/'src/semcache/edgelora').glob('r3cf_*.py'))]
    result.update({str(p.relative_to(ROOT)):sha(p) for p in paths});return result


def new_output(path,protected=None):
    path=path.resolve();base=ROOT/'results/r3/c'
    require(protected is None or not path.is_relative_to(protected.resolve()),'output cannot modify input cohort')
    require(path.is_relative_to(base) and path.relative_to(base).parts and path.relative_to(base).parts[0].startswith('focused_break_even_'),
        'new results/r3/c/focused_break_even_* directory required')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty output')
    path.mkdir(parents=True,exist_ok=True);return path


def worker(a):
    plan=read(a.plan);output=Path(plan['output'])
    try:
        require(plan['fixture']==a.test_fixture_only,'worker fixture/production scope mismatch')
        verify_sources(plan['sources'])
        from semcache.edgelora.r3cf_runtime import execute
        result=execute(plan,output)
    except BaseException as exc:
        partial=read(output/'r3c_es_partial.json') if (output/'r3c_es_partial.json').exists() else {}
        result=dict(partial,execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
    result.update(flags(plan['fixture']));result.update(pid=os.getpid(),finished_at=now())
    write(output/'es_result.json',result);return 0 if result['execution_status']=='COMPLETE' else 1


def summary(report):
    return '\n'.join([report['classification'],'PRIMARY_PERFORMANCE_ELIGIBLE=false',
        'Execution: '+report['execution_status'],'GPU findings: '+report['gpu_findings_status'],
        'Reason: '+report.get('reason','none'),
        'Native strict parity passed: '+str(report.get('native_peft_strict_parity_passed')),
        'Brackets: '+json.dumps(report.get('decision',{}).get('hit_fraction_brackets',{}),sort_keys=True),
        'Conditional controlled full-request HIT counts; not natural rates. One-HIT primary bootstrap cannot confirm a bracket.',
        'All-token prefill; no remote-network, TTFT, full-generation, R2-D-root or qualified-speedup validation.'])+'\n'


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker(a)
    try:output=new_output(a.output_dir,a.inputs or a.legacy_input)
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';evidence=Evidence();children=[];relay_info=None;token=uuid.uuid4().hex
    report=dict(execution_status='RUNNING',phase=a.phase,program_version='r3cf-v1',created_at=now(),**flags(fixture),
        sources=sources(),production_source_identity=source_identity(),gpu_findings_status='NOT_RUN',gpu_inference_performed=False,
        actual_network_socket_used=False,native_peft_strict_parity_passed=None,seed=a.seed,resamples=a.resamples,
        new_model_requests_completed=0,new_model_request_rows=0,read_only=a.phase in ('reanalyze','analyze'),
        schedule_manifest=a.schedule_manifest,budget=budget(a.schedule_manifest),
        cli=[sys.executable,str(Path(__file__)),*(sys.argv[1:] if argv is None else argv)])
    points=[];decision=dict(execution_status='NOT_RUN',status='NOT_ESTIMABLE',**flags(fixture))
    try:
        for name,h in report['sources'].items():evidence.file(ROOT/name,h)
        if a.phase=='dry-run':
            report['readiness']=dict(no_model_or_socket_executed=True,
                missing_dependencies=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None])
            write(output/'r3cf_schedule_manifest.json',a.schedule_manifest)
        elif a.phase=='reanalyze':
            legacy=legacy_reanalysis(a.legacy_input,allow_fixture=a.test_fixture_only,seed=a.seed,resamples=a.resamples)
            write(output/'r3cf_legacy_reanalysis.json',legacy);report['legacy_reanalysis']=legacy
            report['historical_model_requests_audited']=legacy.get('independent_request_ids',0)
            report.update(flags(legacy.get('TEST_FIXTURE_ONLY',False)))
            if legacy['execution_status']=='NOT_RUN_MISSING_INPUT':
                report.update(execution_status='NOT_RUN_MISSING_INPUT',reason=legacy['reason'])
                decision=dict(execution_status='NOT_RUN_MISSING_INPUT',status='NOT_ESTIMABLE',**flags())
            else:
                points=legacy['arms'];report['native_peft_strict_parity_passed']=legacy['native_parity_preserved']
                decision=dict(execution_status='COMPLETE',interpretation='RETROSPECTIVE_EXPLORATORY_ONLY',
                    hit_fraction_brackets=legacy['candidate_intervals'],full_deployment_lifecycle='NOT_ESTIMABLE',**flags(legacy['TEST_FIXTURE_ONLY']))
        elif a.phase=='analyze':
            old,plan=load_focused(a.inputs);rows=jsonl(a.inputs/'r3cf_requests.jsonl')
            points,decision=focused_analysis(rows,plan['schedule_manifest'],plan['fixture'],a.resamples)
            report.update(flags(plan['fixture']));report.update(analysis_only=True,input_result_sha256=sha(a.inputs/'r3cf_results.json'),
                input_gpu_inference_performed=old['gpu_inference_performed'],native_peft_strict_parity_passed=old['native_peft_strict_parity_passed'],
                schedule_manifest=plan['schedule_manifest'],budget=plan['budget'],seed=plan['seed'])
            (output/'r3cf_requests.jsonl').write_bytes((a.inputs/'r3cf_requests.jsonl').read_bytes())
            write(output/'r3cf_network_calibration.json',plan['calibration']);write(output/'r3cf_schedule_manifest.json',plan['schedule_manifest'])
            report['input_audit']=old['audit']
        else:
            if not fixture:
                require(report['production_source_identity']['git_branch']=='exp/raw-semcache-r3','wrong branch')
                require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') and
                    len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,'exactly one Slurm GPU allocation required')
                missing=[n for n in ('torch','numpy','transformers','peft') if importlib.util.find_spec(n) is None]
                require(not missing,'GPU NOT_RUN; missing dependencies: '+', '.join(missing))
                from semcache.edgelora.r3c_runtime import offline_model_preflight
                report['model_preflight']=offline_model_preflight();write(output/'r3cf_preflight.json',report['model_preflight'])
            require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates measurement scope')
            manifest=a.schedule_manifest;write(output/'r3cf_schedule_manifest.json',manifest)
            plan=dict(fixture=fixture,sources=report['sources'],proxy_token=token,seed=a.seed,resamples=a.resamples,
                profiles=manifest['profiles'],schedule=manifest['schedule'],schedule_manifest=manifest,
                budget=report['budget'],device='cpu' if fixture else a.device)
            echo=spawn('echo',plan,output/'calibration');children.append(('echo',echo,output/'calibration'))
            echo_info=ready(echo,output/'calibration/echo_ready.json');require(echo_info['pid']==echo[0].pid,'echo PID mismatch')
            relay=spawn('proxy',plan,output/'proxy');children.append(('proxy',relay,output/'proxy'))
            relay_info=ready(relay,output/'proxy/proxy_ready.json');require(relay_info['pid']==relay[0].pid,'proxy PID mismatch')
            from semcache.edgelora.r3c_calibration import calibrate
            calibration=calibrate(manifest['profiles'],relay_info,token,echo_info,output/'calibration')
            write(output/'r3cf_network_calibration.json',calibration);report['actual_network_socket_used']=True
            echo[0].wait(timeout=10.);require(echo[0].returncode==0,'calibration echo failed')
            require(calibration['execution_status']=='COMPLETE','INVALID_IMPAIRMENT: stop before GPU preparation')
            prepared=output/'prepare';runner=legacy_runner().process_runner()
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            if fixture:
                from semcache.edgelora.r3a_tiny import prepare_tiny
                prepare_tiny(prepared)
            else:
                runner([sys.executable,'-u',str(ROOT/'scripts/95_run_r3a_distributed_no_reuse.py'),'--phase','prepare',
                    '--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','32','--device',a.device,
                    '--adapter','user_a','--seed','42','--output-dir',str(prepared)],output/'prepare_process',env=env)
            pm,ref,am=verify_prepared(prepared,evidence)
            if not fixture:require(read(prepared/'r3a_preflight.json')['allocation_environment']['SLURM_JOB_ID']==os.environ['SLURM_JOB_ID'],
                'fresh prepare left allocation')
            identity=dict(identity=pm['identity'],identity_sha256=pm['identity_sha256'],prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),
                adapter_matrix_sha256=am['matrix_sha256'],gpu_uuid=pm.get('gpu',{}).get('uuid','TEST_FIXTURE_ONLY'))
            plan.update(prepared=str(prepared),prepared_manifest_sha256=identity['prepared_manifest_sha256'],reference_identity=identity,
                proxy=relay_info,calibration=calibration)
            es=output/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
            runner([sys.executable,'-u',str(Path(__file__)),'--worker','es','--plan',str(es/'es_plan.json'),
                '--test-fixture-only' if fixture else '--enable-gpu-experiment'],es,env=env,timeout=7500.)
            result=read(es/'es_result.json');require(result['execution_status']=='COMPLETE','ES failed')
            control(relay_info['control_port'],token,'STOP');relay[0].wait(timeout=10.)
            while children:
                kind,child,directory=children.pop();life=reap(child,directory/(kind+'_parent_lifecycle.json'))
                require(life['returncode']==0 and not life['forced_termination'],'unclean '+kind)
            report['audit']=audit_physical(output,plan,result);report['model_result']=result
            report['native_peft_strict_parity_passed']=report['audit']['native_peft_strict_parity_passed']
            rows=jsonl(es/'r3cf_requests.jsonl');points,decision=focused_analysis(rows,manifest,fixture,a.resamples)
            report['gpu_inference_performed']=not fixture;report['gpu_findings_status']='TEST_FIXTURE_ONLY_NO_GPU' if fixture else 'COMPLETE'
        evidence.unchanged()
        if report['execution_status']=='RUNNING':report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if (output/'es/es_result.json').is_file():
            report['failed_worker']=read(output/'es/es_result.json')
            report['reason']+='; ES: '+report['failed_worker'].get('reason','see worker traces')
        points=[];decision=dict(execution_status='FAILED',status='NOT_ESTIMABLE',reason=report['reason'],**flags(fixture));traceback.print_exc()
    finally:
        while children:
            kind,child,directory=children.pop();reap(child,directory/(kind+'_parent_lifecycle.json'))
        if (output/'es/r3cf_requests.jsonl').is_file():(output/'r3cf_requests.jsonl').write_bytes((output/'es/r3cf_requests.jsonl').read_bytes())
        if not (output/'r3cf_requests.jsonl').exists():(output/'r3cf_requests.jsonl').touch()
        if not (output/'r3cf_network_calibration.json').exists():
            write(output/'r3cf_network_calibration.json',dict(execution_status='NOT_RUN',reason=report.get('reason')))
        if (output/'prepare/r3a_prepare_execution.json').is_file():
            report['gpu_inference_performed']=True
            if report['execution_status']=='FAILED':report['gpu_findings_status']='INCOMPLETE_FAILED'
        available=[r['native_parity'] for r in jsonl(output/'r3cf_requests.jsonl') if 'native_parity' in r]
        if a.phase in ('run','fixture'):
            recorded=jsonl(output/'r3cf_requests.jsonl');report['new_model_request_rows']=len(recorded)
            report['new_model_requests_completed']=sum(r.get('observed_qkv_outputs')==96 and 'logits_sha256' in r for r in recorded)
        if available:report['native_peft_strict_parity_passed']=all(p['passed'] for p in available)
        report.update(decision=decision,evidence_files=list(evidence.files.values()),finished_at=now())
        export(output,points,decision)
        report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and
            p.name not in ('r3cf_results.json','r3cf_summary.txt')}
        write(output/'r3cf_results.json',report);(output/'r3cf_summary.txt').write_text(summary(report))
    print(json.dumps(dict(report=str(output/'r3cf_results.json'),execution_status=report['execution_status'],gpu_findings_status=report['gpu_findings_status'])))
    return 0 if report['execution_status'] in ('COMPLETE','NOT_RUN_MISSING_INPUT') else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);sys.exit(main())
