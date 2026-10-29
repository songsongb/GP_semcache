#!/usr/bin/env python3
"""Read-only R3-A causal intervention: all-layer local CPU deltas versus real TCP replay."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import THRESHOLDS,now,read,sha,source_identity,write
from semcache.edgelora.parity_forensics_audit import CLASSIFICATION,Evidence,audit_failure
from semcache.edgelora.parity_causal_evidence import (assess,audit_prior_forensics,identity_record,
    validate_identity,_controls_exact,control_failure)


def args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','audit','gpu-intervention'))
    p.add_argument('--prepared',type=Path)
    p.add_argument('--pilot-inputs',type=Path)
    p.add_argument('--forensic-inputs',type=Path,help='immutable COMPLETE script-96 diagnostic for the same prepare/pilot')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--enable-gpu-probes',action='store_true')
    p.add_argument('--capture-logits',action='store_true',help='opt-in bounded raw FP32 logits; hashes/strict metrics are always recorded')
    p.add_argument('--capture-comparison-tensors',action='store_true',help='opt-in <=128 MiB local tensors for first-difference magnitudes')
    p.add_argument('--cpu-fp64-control',action='store_true',help='same-native-hidden FP64 counterfactual only; never changes the CPU oracle')
    p.add_argument('--worker',choices=('native-a','native-b','local-cpu','tcp-replay','ud'),help=argparse.SUPPRESS)
    p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or not a.enable_gpu_probes:p.error('private worker requires plan and GPU opt-in')
    elif not all((a.phase,a.prepared,a.pilot_inputs,a.forensic_inputs,a.output_dir)):
        p.error('--phase, --prepared, --pilot-inputs, --forensic-inputs, --output-dir are required')
    elif a.phase!='gpu-intervention' and any((a.enable_gpu_probes,a.capture_logits,a.capture_comparison_tensors,a.cpu_fp64_control)):
        p.error('GPU/capture/precision options require --phase gpu-intervention')
    return a


def new_output(path,protected=()):
    path=path.resolve();base=ROOT/'results/r3/a'
    require(not any(path.is_relative_to(p.resolve()) for p in protected), 'output must not be inside immutable input evidence')
    require(path.is_relative_to(base) and path.relative_to(base).parts and
            path.relative_to(base).parts[0].startswith('parity_'), 'new diagnostic output required under results/r3/a/parity_*')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()), 'refusing nonempty/existing-file output')
    path.mkdir(parents=True,exist_ok=True);return path


def worker(a):
    plan=read(a.plan);output=Path(plan['output'])
    try:
        require(plan.get('enable_gpu_probes') is True,'GPU intervention plan not explicitly enabled')
        if a.worker=='ud':
            from semcache.edgelora.parity_causal_probe import exact_affinity,cpu_runtime
            exact_affinity(plan['comparison_identity'])
            from semcache.edgelora.user_device import serve
            serve(plan['prepared'],output,plan['session_id'])
            import torch
            require(not torch.cuda.is_initialized(),'diagnostic UD created CUDA context')
            write(output/'causal_ud_metadata.json',dict(pid=os.getpid(),cuda_initialized=False,cpu_runtime=cpu_runtime()))
            return 0
        from semcache.edgelora.parity_causal_probe import native,bare
        result=native(plan,output) if a.worker.startswith('native-') else bare(plan,output,physical=a.worker=='tcp-replay')
    except BaseException as exc:
        result=dict(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',pid=os.getpid(),traceback=traceback.format_exc())
        traceback.print_exc()
        if a.worker=='ud':
            write(output/'causal_ud_failure.json',result)
            return 1
    result.update(classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
                  thresholds=dict(THRESHOLDS),original_artifacts_modified=False)
    write(output/'worker_result.json',result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def run_child(kind,plan,output):
    from semcache.edgelora.parity_causal_process import launch,finish
    directory=output/kind;directory.mkdir();worker_plan=dict(plan,output=str(directory))
    write(directory/'diagnostic_plan.json',worker_plan);child=None
    try:
        child=launch(kind,directory/'diagnostic_plan.json',directory)
        child[0].wait(timeout=1800.)
        require(child[0].returncode==0,f'{kind} failed; see {directory}/worker_result.json and {kind}.stderr.log')
    finally:
        lifecycle=finish(child) if child is not None else dict(all_children_exited=True,not_started=True)
        write(directory/'lifecycle.json',lifecycle)
    result=read(directory/'worker_result.json')
    require(result['execution_status']=='COMPLETE' and result['process_metadata']['pid']==child[0].pid and
            lifecycle['all_children_exited'] and not lifecycle['forced_termination'] and
            not lifecycle['descendant_group_cleanup_applied'],'diagnostic child/PID/exit gate failed')
    return result


def human(report):
    assessment=report['assessment']
    return '\n'.join([CLASSIFICATION,'PRIMARY_PERFORMANCE_ELIGIBLE=false',
        'Execution: '+report['execution_status'],'GPU findings: '+report['gpu_findings_status'],
        'Conclusion: '+assessment['conclusion'],'Reason: '+report.get('reason','none'),
        'Sufficiency established: '+str(assessment['sufficiency_established']).lower(),
        'Full serialized logits matched: '+str(assessment.get('full_logits_bitwise_matched','NOT_ESTIMABLE')),
        'QKV hashes matched / 96: '+str(assessment.get('qkv_hashes_matched','NOT_ESTIMABLE')),
        assessment.get('interpretation','No causal numerical result without verified original evidence and explicit GPU execution.'),
        'Old replay without a full-logit SHA cannot be proved bitwise identical from strict metrics.',
        'Local CPU oracle is a diagnostic intervention; no benchmark, production fix or threshold relaxation.',
        'Layer>=1 arithmetic attribution requires matching actual hidden bytes.'])+'\n'


def main(argv=None):
    a=args(argv)
    if a.worker:return worker(a)
    try:output=new_output(a.output_dir,(a.prepared,a.pilot_inputs,a.forensic_inputs))
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);return 1
    evidence=Evidence();sources=[Path(__file__).resolve(),*sorted((ROOT/'src/semcache/edgelora').glob('parity_causal_*.py'))]
    report=dict(execution_status='RUNNING',classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
        phase=a.phase,created_at=now(),analysis_pid=os.getpid(),host=os.uname().nodename,
        cli=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
        prepared=str(a.prepared.resolve()),pilot=str(a.pilot_inputs.resolve()),prior_forensics=str(a.forensic_inputs.resolve()),
        production_source_identity=source_identity(),diagnostic_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        frozen_thresholds=dict(THRESHOLDS),gpu_findings_status='NOT_RUN',gpu_inference_performed=False,
        original_evidence_audited=False,original_artifacts_modified=False,cache_reuse_enabled=False,
        actual_remote_network_measured=False,performance_measurement_performed=False,
        actual_network_socket_used=False,
        raw_logit_capture=a.capture_logits,raw_comparison_tensor_capture=a.capture_comparison_tensors,
        assessment=dict(conclusion='UNRESOLVED',sufficiency_established=False))
    try:
        for path in sources:evidence.file(path,report['diagnostic_source_hashes'][str(path.relative_to(ROOT))])
        if a.phase=='dry-run':
            report['readiness']=dict(paths={k:str(p) for k,p in (('prepared',a.prepared),('pilot',a.pilot_inputs),('forensics',a.forensic_inputs))},
                exists={k:p.is_dir() for k,p in (('prepared',a.prepared),('pilot',a.pilot_inputs),('forensics',a.forensic_inputs))},
                dependencies={n:importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft','numpy')},
                no_model_or_socket_executed=True)
        else:
            offline=audit_failure(a.prepared,a.pilot_inputs,evidence);report['offline_audit']=offline
            pm=read(a.prepared/'r3a_prepare_manifest.json');ref=read(a.prepared/'r3a_reference.json')
            prior=audit_prior_forensics(a.forensic_inputs,evidence,offline,pm);report['prior_forensic_audit']=prior
            report['original_evidence_audited']=True
            expected=identity_record(a.prepared,a.pilot_inputs,pm,ref,offline);validate_identity(expected,expected)
            report['comparison_identity']=expected
            if a.phase=='gpu-intervention':
                require(a.enable_gpu_probes,'--enable-gpu-probes required; no automatic GPU execution')
                require(pm.get('TEST_FIXTURE_ONLY') is not True,'synthetic artifacts cannot establish GPU causal attribution')
                missing=[n for n in ('torch','transformers','peft','numpy') if importlib.util.find_spec(n) is None]
                require(not missing,'GPU findings NOT_RUN; dependencies unavailable: '+', '.join(missing))
                plan=dict(enable_gpu_probes=True,prepared=str(a.prepared.resolve()),pilot=str(a.pilot_inputs.resolve()),offline=offline,
                    comparison_identity=expected,device=a.device,capture_logits=a.capture_logits,
                    capture_comparison_tensors=a.capture_comparison_tensors,cpu_fp64_control=a.cpu_fp64_control,
                    diagnostic_source_hashes=report['diagnostic_source_hashes'])
                controls=[]
                for kind in ('native-a','native-b'):
                    result=run_child(kind,plan,output);report[kind.replace('-','_')]=result;controls.append(result)
                    if not _controls_exact(result):
                        report['assessment']=dict(conclusion=control_failure(result),sufficiency_established=False,
                            reason='exact native/original/GPU reconstruction gates failed; no CPU intervention or TCP replay executed')
                        break
                if len(controls)==2 and all(_controls_exact(c) for c in controls):
                    local=run_child('local-cpu',plan,output);report['local_cpu_oracle']=local
                    replay=run_child('tcp-replay',dict(plan,local_report=str(output/'local-cpu/worker_result.json')),output)
                    report['tcp_replay']=replay
                    report['actual_network_socket_used']=True
                    report['assessment']=assess(*controls,local,replay,expected,prior['replay'])
                    report['replay_vs_original_pilot_strict_metrics_equal']=all(replay['vs_original_logits'][k]==offline['parity'][k]
                        for k in ('max_abs','relative_l2','tolerance_violations','last_token_kl','last_token_argmax_agreement','passed'))
                    report['replay_vs_prior_strict_metrics_equal']=replay['vs_original_logits']==prior['replay']['vs_original_logits']
                report['gpu_findings_status']='COMPLETE';report['gpu_inference_performed']=True
            evidence.unchanged()
        report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}')
        print(report['reason'],file=sys.stderr)
        if any(output.glob('*/gpu_execution.json')):
            report.update(gpu_findings_status='INCOMPLETE_FAILED_PROBES',gpu_inference_performed=True)
    finally:
        report['evidence_files']=list(evidence.files.values());report['finished_at']=now()
        # Failed probes may already have used the physical socket. Retain that
        # fact even if no successful worker result was returned to the parent.
        report['actual_network_socket_used']=report['actual_network_socket_used'] or any(
            p.stat().st_size>0 for name in ('diagnostic_es_wire.jsonl','r3a_ud_wire_trace.jsonl') for p in output.rglob(name))
        report['diagnostic_output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and
            p.name not in ('r3a_cpu_delta_causal.json','r3a_cpu_delta_causal_summary.txt')}
        write(output/'r3a_cpu_delta_causal.json',report)
        (output/'r3a_cpu_delta_causal_summary.txt').write_text(human(report))
    print(json.dumps(dict(report=str(output/'r3a_cpu_delta_causal.json'),execution_status=report['execution_status'],
        gpu_findings_status=report['gpu_findings_status'],assessment=report['assessment']['conclusion'],
        PRIMARY_PERFORMANCE_ELIGIBLE=False),sort_keys=True))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':sys.exit(main())
