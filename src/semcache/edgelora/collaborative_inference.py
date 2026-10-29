"""R3-A isolated reference, ES/UD process lifecycle and phase orchestration."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from .communication import digest, require
from .r3a_artifacts import ROOT, THRESHOLDS, flags, now, read, sha, source_identity, validate_prepared, versions, write, write_csv
from .r3a_evidence import aggregate, jsonl, validate_run


def boundary_audit(identity=None):
    ledger=None
    if identity is not None:
        n,h,l=identity['length'],identity['hidden_size'],identity['layers']
        ledger=dict(es_to_ud_projection_bytes=l*n*h*2,ud_to_es_projection_bytes=3*l*n*h*4,
            es_to_ud_retained_boundary_bytes=n*h*2,ud_to_es_retained_boundary_bytes=n*h*2,
            es_to_ud_total_bytes=(l+1)*n*h*2,ud_to_es_total_bytes=(3*l*4+2)*n*h,
            provenance='SIMULATED_M9A_EQUATIONS_EXPLICIT_WIDTHS',is_observed_network_traffic=False)
    return dict(h0_transported=False,hL_transported=False,m9a_directional_payload_ledger=ledger,
        ownership='ES owns tokenizer/prepared input IDs, embedding, whole decoder and LM head; UD owns only user_a QKV A/B',
        omitted_messages=['M9-A full h0 UD->ES','M9-A full hL ES->UD'],
        m9a_comparison_assumptions=dict(hidden_element_bytes=2,delta_element_bytes=4,full_h0_hL_boundaries=True,
            scope='explicit R2-D comparison scenario; historical network.py delta default is configurable'),
        actual_tensor_shape=None if identity is None else [1,identity['length'],identity['hidden_size']],
        actual_hidden_element_bytes=None if identity is None else (2 if identity['hidden_dtype']=='float16' else 4),
        actual_delta_element_bytes=None if identity is None else (2 if identity['delta_dtype']=='float16' else 4),
        actual_message_schedule='observed per-layer HIDDEN and bundled DELTAS; control HELLO/READY and STOP/BYE outside requests',
        wire_byte_scope='bytes accepted/read through application TCP socket including framing and JSON; excludes kernel TCP/IP headers/retransmissions',
        synchronized_wall='ES monotonic interval includes request coordinator, patch setup/teardown and full native eager forward with final CUDA synchronization',
        reference_inside_distributed_request_wall=False,nested_timers_may_be_summed=False,
        cuda_interval='event elapsed includes stream idle/host socket gaps; not kernel self time',
        timing_not_summed=['UD process CPU LoRA','ES send/receive blocking','copy/cast','CUDA interval','framing'],
        unmeasured_exclusive_spans=['native base GEMM/add self-time','Python dispatch','allocation',
            'UD identity/safety checks after recorded matmuls','host bookkeeping; retained in request wall'],
        excluded=['model load/fingerprint','adapter load','tokenization and input preparation','socket handshake','post-wall logit export/parity','shutdown'],
        overlap_policy='serialized dependency exchanges; no engineered compute/communication overlap',
        no_reuse=True,cache_admission_executed=False,source_qkv_reuse=False,
        r2d_totals_directly_comparable=False,ttft_measured=False,full_generation_measured=False)


def launch(role,plan,output):
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    if role=='ud':
        env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    stdout=(output/f'r3a_{role}.stdout.log').open('w')
    stderr=(output/f'r3a_{role}.stderr.log').open('w')
    try:
        process=subprocess.Popen([sys.executable,'-u','-m','semcache.edgelora.collaborative_inference',
            '--worker',role,'--plan',str(plan)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    except BaseException:
        stdout.close();stderr.close();raise
    return process,stdout,stderr


def cleanup(children):
    result=[]
    for role,(process,stdout,stderr) in children:
        forced=process.poll() is None
        if forced:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3.)
        stdout.close();stderr.close()
        result.append(dict(role=role,pid=process.pid,returncode=process.returncode,forced_termination=forced))
    return dict(all_children_exited=all(r['returncode'] is not None for r in result),children=result)


def wait_ready(process,path,timeout=30.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if path.exists(): return read(path)
        require(process.poll() is None,'UD exited before listener readiness')
        time.sleep(.02)
    raise TimeoutError('UD readiness deadline exceeded')


def run_processes(plan,output):
    children=[];report={}
    try:
        ud=launch('ud',output/'r3a_plan.json',output);children.append(('ud',ud))
        ready=wait_ready(ud[0],output/'r3a_ud_ready.json')
        require(ready['pid']==ud[0].pid,'UD ready PID mismatch')
        es=launch('es',output/'r3a_plan.json',output);children.append(('es',es))
        es[0].wait(timeout=600.)
        require(es[0].returncode==0,'ES worker failed; see r3a_es.stderr.log / r3a_es_failure.json')
        ud[0].wait(timeout=10.)
        require(ud[0].returncode==0,'UD worker failed; see r3a_ud.stderr.log / r3a_ud_failure.json')
    finally:
        report=cleanup(children)
        write(output/'r3a_lifecycle.json',report)
    require(all(not c['forced_termination'] and c['returncode']==0 for c in report['children']), 'unclean worker exit')
    return report


def _artifacts(output,exclude):
    return {p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in exclude and not p.name.endswith('.tmp')}


def prepare(args,output):
    manifest=dict(execution_status='RUNNING',created_at=now(),code=source_identity(),
        configuration=vars(args),preparation_process_exited=False,**flags())
    # vars(args) is converted by the CLI to plain JSON values before this call.
    write(output/'r3a_prepare_manifest.json',manifest)
    preflight=dict(execution_status='RUNNING',versions=versions(),
        offline_required=True,synthetic_untrained_adapter=True,thresholds=THRESHOLDS,
        allocation_environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','SLURM_JOB_ID','SLURM_JOB_GPUS')})
    write(output/'r3a_preflight.json',preflight)
    children=[]
    try:
        missing=[n for n in ('torch','transformers','peft') if importlib.util.find_spec(n) is None]
        require(not missing,'missing local runtime dependencies: '+', '.join(missing)+'; GPU prepare not executed')
        require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timing')
        write(output/'r3a_plan.json',dict(configuration=vars(args),output=str(output)))
        child=launch('prepare',output/'r3a_plan.json',output);children.append(('prepare',child))
        child[0].wait(timeout=900.)
        require(child[0].returncode==0,'isolated reference failed; see r3a_prepare.stderr.log / r3a_prepare_failure.json')
        result=read(output/'r3a_prepare_result.json')
        preflight.update(execution_status='COMPLETE',gpu=result['gpu'],native_reference_executed=True)
        write(output/'r3a_preflight.json',preflight)
        manifest.update(result,execution_status='COMPLETE',preparation_process_exited=True,prepared_at=now(),
            preparation_pid=child[0].pid)
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}'
        manifest.update(execution_status='FAILED',reason=reason)
        if (output/'r3a_prepare_execution.json').exists():
            manifest['gpu_inference_performed']=read(output/'r3a_prepare_execution.json')['gpu_inference_performed']
        preflight.update(execution_status='FAILED',reason=reason)
        write(output/'r3a_preflight.json',preflight)
        for name in ('r3a_reference.json','r3a_adapter_manifest.json'):
            if not (output/name).exists(): write(output/name,dict(execution_status='FAILED',reason=reason))
        raise
    finally:
        manifest['cleanup']=cleanup(children)
        manifest['artifact_hashes']=_artifacts(output,{'r3a_prepare_manifest.json'})
        write(output/'r3a_prepare_manifest.json',manifest)


def trial(args,output):
    m=dict(execution_status='RUNNING',configuration=vars(args),created_at=now(),code=source_identity(),
           parity_passed=False,**flags())
    environment={};observations=[]
    for name in ('r3a_wire_trace.jsonl','r3a_ud_wire_trace.jsonl','r3a_iterations.jsonl'):
        (output/name).touch()
    write(output/'r3a_manifest.json',m)
    write(output/'r3a_boundary_audit.json',boundary_audit())
    try:
        pm,ref,am=validate_prepared(args.prepared)
        fixture=pm.get('TEST_FIXTURE_ONLY') is True
        require((args.device=='cpu') if fixture else args.device.startswith('cuda'), 'device/backend mismatch')
        require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timing')
        pilot=None
        if args.phase=='latency':
            pilot=validate_run(args.pilot_inputs,args.prepared,phase='pilot')
            require(pilot['manifest']['configuration']['device']==args.device,'pilot device mismatch')
            m['pilot_manifest_sha256']=sha(Path(args.pilot_inputs)/'r3a_manifest.json')
        m.update(identity_sha256=pm['identity_sha256'],prepared_manifest_sha256=sha(Path(args.prepared)/'r3a_prepare_manifest.json'),
                 **flags(fixture=fixture))
        write(output/'r3a_boundary_audit.json',boundary_audit(pm['identity']))
        plan=dict(prepared=str(Path(args.prepared).resolve()),output=str(output),session_id=uuid.uuid4().hex,
            device=args.device,warmup=args.warmup_runs,measured=args.measured_runs)
        write(output/'r3a_plan.json',plan)
        m['cleanup']=run_processes(plan,output)
        environment=dict(es=read(output/'r3a_es_result.json'),ud=read(output/'r3a_ud_result.json'))
        require(environment['es']['pid']!=environment['ud']['pid'] and
                {c['pid'] for c in m['cleanup']['children']}=={environment['es']['pid'],environment['ud']['pid']},
                'worker/report PID mismatch')
        if pilot and not fixture:
            require(environment['es']['gpu']['uuid']==pilot['environment']['es']['gpu']['uuid'],'pilot accelerator mismatch')
        observations=jsonl(output/'r3a_iterations.jsonl')
        boundaries=boundary_audit(pm['identity'])
        modeled=boundaries['m9a_directional_payload_ledger']
        boundaries['observed_directional_comparison']=[dict(request_id=o['row']['request_id'],
            es_to_ud_payload_bytes=o['row']['es_to_ud_payload_bytes'],ud_to_es_payload_bytes=o['row']['ud_to_es_payload_bytes'],
            es_to_ud_wire_bytes=o['row']['es_to_ud_wire_bytes'],ud_to_es_wire_bytes=o['row']['ud_to_es_wire_bytes'],
            payload_delta_from_m9a_es_to_ud=o['row']['es_to_ud_payload_bytes']-modeled['es_to_ud_total_bytes'],
            payload_delta_from_m9a_ud_to_es=o['row']['ud_to_es_payload_bytes']-modeled['ud_to_es_total_bytes'],
            interpretation='directional reconciliation only; ownership/wire precision and endpoint differ; no equality assertion or historical wall comparison') for o in observations]
        wire=jsonl(output/'r3a_wire_trace.jsonl')
        boundaries['observed_session_wire_totals']={direction:dict(
            frames=sum(f['operation']==operation for f in wire),
            wire_bytes=sum(f['wire_bytes'] for f in wire if f['operation']==operation),
            payload_bytes=sum(f['header']['payload_bytes'] for f in wire if f['operation']==operation),
            control_frames_outside_request=sum(f['operation']==operation and f['header']['layer_id']==-1 for f in wire),
            control_wire_bytes_outside_request=sum(f['wire_bytes'] for f in wire if f['operation']==operation and f['header']['layer_id']==-1))
            for direction,operation in (('ES_TO_UD','send'),('UD_TO_ES','recv'))}
        write(output/'r3a_boundary_audit.json',boundaries)
        m.update(execution_status='VALIDATING',**flags(socket_used=True,gpu=not fixture,fixture=fixture))
        write(output/'r3a_environment.json',environment)
        write_csv(output/'r3a_latency_raw.csv',[o['row'] for o in observations])
        write(output/'r3a_parity.json',dict(parity_passed=False,thresholds=THRESHOLDS,observations=[o['parity'] for o in observations]))
        decision=dict(m,conclusion='Validation pending; not an eligible functional baseline yet.')
        write(output/'r3a_decision.json',decision)
        m['artifact_hashes']=_artifacts(output,{'r3a_manifest.json'})
        write(output/'r3a_manifest.json',m)
        validate_run(output,args.prepared,_candidate=True)
        m.update(execution_status='PILOT_VALIDATED' if args.phase=='pilot' else 'COMPLETE',parity_passed=True)
        write(output/'r3a_parity.json',dict(parity_passed=True,thresholds=THRESHOLDS,observations=[o['parity'] for o in observations]))
        write(output/'r3a_decision.json',dict(execution_status=m['execution_status'],parity_passed=True,
            conclusion='Independent ES/UD no-reuse loopback prefill passed the recorded gates. '+
                ('TEST_FIXTURE_ONLY tiny CPU decoder; OPT-2.7B remains unvalidated.' if fixture else
                 'Controlled untrained adapter; no remote hardware, full generation, reuse speedup or R2-D break-even claim.'),
            **flags(socket_used=True,gpu=not fixture,fixture=fixture)))
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}';m.update(execution_status='FAILED',reason=reason,parity_passed=False)
        environment={r:read(output/f'r3a_{r}_result.json') for r in ('es','ud') if (output/f'r3a_{r}_result.json').exists()}
        observations=jsonl(output/'r3a_iterations.jsonl')
        m['actual_network_socket_used']=any(r.get('actual_network_socket_used') is True for r in environment.values())
        m['gpu_inference_performed']=any(r.get('gpu_inference_performed') is True for r in environment.values())
        if (output/'r3a_lifecycle.json').exists(): m['cleanup']=read(output/'r3a_lifecycle.json')
        write(output/'r3a_environment.json',environment)
        write_csv(output/'r3a_latency_raw.csv',[o['row'] for o in observations] or [dict(status='FAILED',error=reason)])
        write(output/'r3a_parity.json',dict(parity_passed=False,thresholds=THRESHOLDS,observations=[o['parity'] for o in observations]))
        failures={p.name:read(p) for p in output.glob('r3a_*_failure.json')}
        write(output/'r3a_decision.json',dict(execution_status='FAILED',parity_passed=False,reason=reason,worker_failures=failures,
            **{k:m[k] for k in flags()}))
        raise
    finally:
        m['finished_at']=now();m['artifact_hashes']=_artifacts(output,{'r3a_manifest.json'})
        write(output/'r3a_manifest.json',m)


def analyze(args,output):
    try:
        result,audit=aggregate(args.prepared,args.pilot_inputs,args.inputs)
        decision=dict(result,conclusion='Verified warm localhost no-reuse prefill only; R3-B requires physical reuse and independent correctness gates. R2-D break-even, remote UD hardware and full generation remain unvalidated.')
    except BaseException as exc:
        result=decision=dict(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',parity_passed=False,**flags())
        audit=dict(execution_status='FAILED',reason=result['reason'],no_imputation=True)
        raise
    finally:
        write(output/'r3a_aggregate_summary.json',result)
        write(output/'r3a_decision.json',decision)
        write(output/'r3a_evidence_audit.json',audit)


def worker(role,plan_path):
    plan=read(plan_path);output=Path(plan['output'])
    print(f'R3-A worker={role} pid={os.getpid()} started',flush=True)
    try:
        if role=='prepare':
            from .r3a_runtime import prepare_reference
            write(output/'r3a_prepare_result.json',prepare_reference(plan['configuration'],output))
        elif role=='ud':
            from .user_device import serve
            serve(plan['prepared'],output,plan['session_id'])
        else:
            from .edge_server import execute
            execute(plan['prepared'],output,plan['session_id'],plan['device'],plan['warmup'],plan['measured'])
        print(f'R3-A worker={role} pid={os.getpid()} COMPLETE',flush=True)
        return 0
    except BaseException as exc:
        failure=dict(execution_status='FAILED',pid=os.getpid(),reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        write(output/f'r3a_{role}_failure.json',failure)
        traceback.print_exc();return 1


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('prepare','ud','es'),required=True);p.add_argument('--plan',required=True)
    a=p.parse_args();sys.exit(worker(a.worker,a.plan))
