#!/usr/bin/env python3
"""Isolated read-only R3-A parity forensics. No benchmark admission or source changes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import THRESHOLDS,now,read,sha,source_identity,write
from semcache.edgelora.parity_forensics_audit import CLASSIFICATION,Evidence,audit_failure,conclude


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('audit','dry-run','gpu-probe'))
    p.add_argument('--prepared',type=Path)
    p.add_argument('--pilot-inputs',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--enable-gpu-probes',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--layers',default='all',help='all, or unique comma-separated layer IDs including 0')
    p.add_argument('--cpu-fp64-control',action='store_true',help='additional diagnostic CPU FP64 A/B comparison; never changes production')
    p.add_argument('--capture-qkv-reference',action='store_true',help='opt-in bounded layer0/1 native QKV reference arrays in new output only')
    p.add_argument('--worker',choices=('native-a','native-b','es'),help=argparse.SUPPRESS)
    p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or not a.enable_gpu_probes:p.error('private worker requires plan and explicit GPU opt-in')
    elif not a.phase or not a.prepared or not a.pilot_inputs or not a.output_dir:
        p.error('--phase, --prepared, --pilot-inputs, --output-dir required')
    if a.phase!='gpu-probe' and not a.worker and (a.enable_gpu_probes or a.cpu_fp64_control or a.capture_qkv_reference):
        p.error('GPU/precision/capture options require gpu-probe phase')
    return a


def new_output(path):
    path=path.resolve();base=ROOT/'results/r3/a'
    require(path.is_relative_to(base) and path.relative_to(base).parts and
            path.relative_to(base).parts[0].startswith('parity_forensics_'),
            'diagnostic output must be under results/r3/a/parity_forensics_<new tag>')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty/existing-file output')
    path.mkdir(parents=True,exist_ok=True)
    return path


def diagnostic_sources():
    paths=[Path(__file__).resolve(),ROOT/'src/semcache/edgelora/parity_forensics_audit.py',
           ROOT/'src/semcache/edgelora/parity_forensics_probe.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def child(kind,plan,output):
    """Wait/reap one GPU child before starting the next; no overlapping models."""
    from semcache.edgelora.collaborative_inference import cleanup
    directory=output/kind;directory.mkdir()
    worker_plan=dict(plan,output=str(directory))
    write(directory/'diagnostic_plan.json',worker_plan)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    stdout=(directory/'stdout.log').open('w');stderr=(directory/'stderr.log').open('w');process=None
    try:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',kind,
            '--plan',str(directory/'diagnostic_plan.json'),'--enable-gpu-probes'],cwd=ROOT,env=env,
            stdout=stdout,stderr=stderr,start_new_session=True)
        process.wait(timeout=1800.)
        require(process.returncode==0,f'{kind} diagnostic failed; see {directory}/worker_result.json and stderr.log')
    finally:
        if process is not None:lifecycle=cleanup([(kind,(process,stdout,stderr))])
        else:
            stdout.close();stderr.close();lifecycle=dict(all_children_exited=True,children=[])
        write(directory/'lifecycle.json',lifecycle)
    result=read(directory/'worker_result.json')
    require(result['execution_status']=='COMPLETE' and result['process_metadata']['pid']==process.pid,
            'worker result/PID mismatch')
    result['independent_process_cleanup']=lifecycle
    return result


def worker(a):
    from semcache.edgelora.parity_forensics_probe import native_worker,es_worker
    plan=read(a.plan);output=Path(plan['output'])
    print(f'R3A_PARITY_DIAGNOSTIC {a.worker} pid={os.getpid()} started',flush=True)
    try:
        result=es_worker(plan,output) if a.worker=='es' else native_worker(plan,output,components=a.worker=='native-a')
    except BaseException as exc:
        result=dict(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',pid=os.getpid(),traceback=traceback.format_exc())
        traceback.print_exc()
    result.update(classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
                  original_artifacts_modified=False,production_thresholds=dict(THRESHOLDS))
    write(output/'worker_result.json',result)
    print(f"R3A_PARITY_DIAGNOSTIC {a.worker} {result['execution_status']}",flush=True)
    return 0 if result['execution_status']=='COMPLETE' else 1


def summary(report):
    offline=report.get('offline_audit',{})
    parity=offline.get('parity')
    lines=[CLASSIFICATION,'PRIMARY_PERFORMANCE_ELIGIBLE=false',f"Execution: {report['execution_status']}",
        f"GPU findings: {report['gpu_findings_status']}",f"Conclusions: {', '.join(report['assessment']['conclusions'])}",
        f"Original audit: {offline.get('status','NOT_RUN')}",f"Reason: {report.get('reason','none')}",
        f"First original hidden bitwise divergence: {offline.get('first_bitwise_hidden_divergence','NOT_ESTIMABLE')}",
        f"Original root failure: {offline.get('root_failure','NOT_ESTIMABLE')}",
        f"Original strict parity metrics: {json.dumps(parity,sort_keys=True) if parity is not None else 'NOT_ESTIMABLE_NOT_AUDITED'}",
        f"Original schedule: {json.dumps(offline.get('actual_schedule'),sort_keys=True)}",
        f"Shutdown symptom: {offline.get('shutdown',{}).get('status','NOT_ESTIMABLE')}",
        'SHA differences have no error magnitude. Matched argmax/low KL do not satisfy the strict gate.',
        'No original artifacts, adapter weights, thresholds, production semantics or benchmark eligibility were changed.',
        'GPU NOT_RUN means no numerical H1-H5 conclusion has been established. See JSON for measured-versus-inferred scope.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker(a)
    try:output=new_output(a.output_dir)
    except (OSError,ValueError) as exc:
        print(str(exc),file=sys.stderr);return 1
    evidence=Evidence()
    report=dict(execution_status='RUNNING',classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
        created_at=now(),phase=a.phase,prepared=str(a.prepared.resolve()),pilot=str(a.pilot_inputs.resolve()),
        cli=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
        diagnostic_source_hashes=diagnostic_sources(),production_source_identity=source_identity(),
        frozen_thresholds=dict(THRESHOLDS),gpu_findings_status='NOT_RUN',gpu_inference_performed=False,
        original_artifacts_modified=False,raw_tensor_dumps=False,assessment=conclude())
    prepared_manifest=a.prepared/'r3a_prepare_manifest.json'
    report['prepared_manifest_sha256']=sha(prepared_manifest) if prepared_manifest.is_file() else None
    report['analysis_process']=dict(pid=os.getpid(),parent_pid=os.getppid(),hostname=os.uname().nodename)
    try:
        if a.phase=='dry-run':
            report['readiness']=dict(prepared_path_exists=a.prepared.is_dir(),pilot_path_exists=a.pilot_inputs.is_dir(),
                dependencies={n:importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft','numpy')},
                original_evidence_audited=False,no_model_or_network_executed=True)
            report['execution_status']='COMPLETE'
        else:
            offline=audit_failure(a.prepared,a.pilot_inputs,evidence);report['offline_audit']=offline
            report['TEST_FIXTURE_ONLY']=offline['TEST_FIXTURE_ONLY']
            if a.phase=='gpu-probe':
                require(a.enable_gpu_probes,'gpu-probe requires explicit --enable-gpu-probes')
                missing=[n for n in ('torch','transformers','peft','numpy') if importlib.util.find_spec(n) is None]
                require(not missing,'GPU probes NOT_RUN; missing local dependencies: '+', '.join(missing))
                n=offline['reference_identity']['layers']
                selected=list(range(n)) if a.layers=='all' else [int(v) for v in a.layers.split(',')]
                require(selected and 0 in selected and len(set(selected))==len(selected) and all(0<=v<n for v in selected),
                        'unique in-range layers including layer0 required')
                plan=dict(enable_gpu_probes=True,prepared=str(a.prepared.resolve()),pilot=str(a.pilot_inputs.resolve()),
                    offline=offline,device=a.device,layers=selected,cpu_fp64_control=a.cpu_fp64_control,
                    capture_qkv_reference=a.capture_qkv_reference)
                native_a=child('native-a',plan,output);report['native_a']=native_a
                # First GPU process has exited and its CUDA model is gone.
                native_b=child('native-b',plan,output);report['native_b']=native_b
                controls=conclude(native_a,native_b)
                if 'REFERENCE_NONREPRODUCIBLE' in controls['conclusions']:
                    es=None
                    report['es_replay_status']='NOT_RUN_FIX_REFERENCE_REPRODUCIBILITY_FIRST'
                else:
                    es=child('es',dict(plan,native_report=str(output/'native-a/worker_result.json')),output);report['es_replay']=es
                    report['es_replay_status']='COMPLETE'
                report.update(gpu_findings_status='COMPLETE',gpu_inference_performed=True,raw_tensor_dumps=a.capture_qkv_reference,
                    assessment=conclude(native_a,native_b,es))
                report['cross_process_native_logit_sha_equal']=(native_a['controls'][0]['logits_sha256']==native_b['controls'][0]['logits_sha256'])
                report['first_numerical_projection_differences']={name:next((dict(layer=p['layer'],role=p['role'],metrics=p['comparisons'][name])
                    for p in native_a['projections'] if p['comparisons'][name]['unequal_elements']),None) for name in
                    ('native_vs_gpu_reconstruction','gpu_A_vs_cpu_A','gpu_delta_vs_cpu_delta','native_vs_cpu_assembled')}
            evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}')
        print(report['reason'],file=sys.stderr)
        if any(output.glob('*/gpu_execution.json')):
            report.update(gpu_inference_performed=True,gpu_findings_status='INCOMPLETE_FAILED_PROBES')
    finally:
        report['evidence_files']=list(evidence.files.values());report['finished_at']=now()
        report['diagnostic_output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and
            p.name not in ('r3a_parity_forensics.json','r3a_parity_forensics_summary.txt')}
        write(output/'r3a_parity_forensics.json',report)
        (output/'r3a_parity_forensics_summary.txt').write_text(summary(report))
    print(json.dumps(dict(report=str(output/'r3a_parity_forensics.json'),execution_status=report['execution_status'],
        classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,gpu_findings_status=report['gpu_findings_status'],
        conclusions=report['assessment']['conclusions'],
        original_parity=report.get('offline_audit',{}).get('parity'),
        first_bitwise_hidden_divergence=report.get('offline_audit',{}).get('first_bitwise_hidden_divergence'),
        original_schedule=report.get('offline_audit',{}).get('actual_schedule')),sort_keys=True))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':sys.exit(main())
