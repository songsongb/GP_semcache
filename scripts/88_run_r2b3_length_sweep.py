#!/usr/bin/env python3
"""R2-B3 explicit capacity pilot, exact-prefix length sweep and session analysis."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r2b3_workloads import cases,validate_case,ordered_conditions
from semcache.diagnostics.r2b3_benchmark import Runtime,gate,run_pairs,enrich
from semcache.diagnostics.r2b3_capacity import memory,failure_record,release_failed_case,approved
from semcache.diagnostics.r2b3_analysis import aggregate,experimental_config
from semcache.diagnostics.r2b2_analysis import summarize
from semcache.diagnostics.r2b2_benchmark import modes
from semcache.diagnostics.r2b2_profiler import monitor,collect,ProfilerUnavailable
from semcache.diagnostics.r2b15_cost import prepare_only,invoke
from semcache.diagnostics.r1b3_break_even import code_identity
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','pilot','latency','profile','analyze'),default='smoke')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'))
    p.add_argument('--dtype',choices=('float32','float16'))
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cache-device',default=None)
    p.add_argument('--adapter',choices=('user_a','user_b'),default='user_a')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--allow-download',action='store_true')
    p.add_argument('--cases',help='explicit L:P list; overrides case group; maximum L=1024')
    p.add_argument('--case-group',choices=('A','B','AB','C','ABC','pilot'))
    p.add_argument('--attention-arm',choices=('off','on','both'),default='off')
    p.add_argument('--endpoint-policy',choices=('b1',),default='b1',help='frozen B1-compatible all-logits endpoint')
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--independent-run-id',default='run01'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--order-seed',type=int,default=1001)
    p.add_argument('--pilot-inputs',type=Path,nargs='+',help='exact per-case capacity approvals; mandatory for latency/profile')
    p.add_argument('--inputs',type=Path,nargs='+',help='B3 latency directories for analyze')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.phase=='analyze':
            if not a.inputs: raise ValueError('analyze requires --inputs')
            if any(v is not None for v in (a.model,a.dtype,a.cases,a.case_group,a.pilot_inputs)):
                raise ValueError('analyze derives experimental configuration from input manifests; omit model/dtype/cases')
            return a
        if a.inputs: raise ValueError('--inputs only valid for analyze')
        a.model=a.model or ('facebook/opt-125m' if a.phase=='smoke' else 'facebook/opt-2.7b')
        a.dtype=a.dtype or ('float32' if a.phase=='smoke' else 'float16')
        a.case_group=a.case_group or ('pilot' if a.phase=='pilot' else 'AB')
        a.case_values=cases(a.cases,a.case_group,a.phase)
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.phase=='pilot' else 3)
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (20 if a.phase=='latency' else 3)
        if a.warmup_runs<(1 if a.phase=='pilot' else 3) or a.measured_runs<(20 if a.phase=='latency' else 1 if a.phase=='pilot' else 3):
            raise ValueError('pilot >=1 warmup/measured; latency >=3 warmups/20 pairs; smoke >=3/3')
        if a.phase=='smoke' and (a.model!='facebook/opt-125m' or a.dtype!='float32' or any(n!=32 for n,_ in a.case_values)):
            raise ValueError('smoke requires OPT-125M FP32 at 32 tokens')
        if a.phase in ('latency','profile') and not a.pilot_inputs: raise ValueError('verified --pilot-inputs required before latency/profile')
        if a.phase=='profile' and (not a.cases or len(a.case_values)>3): raise ValueError('profile requires 1..3 explicitly selected cases')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': raise ValueError('CUDA_LAUNCH_BLOCKING invalidates timing')
    except ValueError as exc: p.error(str(exc))
    return a


def read_json(path): return json.loads(Path(path).read_text())


def initialize(a):
    a.output_dir.mkdir(parents=True,exist_ok=False)
    files=[ROOT/'scripts/88_run_r2b3_length_sweep.py',*sorted((ROOT/'src/semcache/diagnostics').glob('r2b3_*.py')),
        *sorted((ROOT/'tests').glob('test_r2b3_*.py')),ROOT/'docs/R2B3_LENGTH_SWEEP.md']
    def version(name):
        try: return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: return None
    def serial(value):
        if isinstance(value,Path): return str(value)
        if isinstance(value,(list,tuple)): return [serial(x) for x in value]
        return value
    start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19] if Path('/proc/self/stat').exists() else datetime.now(timezone.utc).isoformat()
    m=dict(experiment='R2-B3',execution_status='RUNNING',implementation_status='IMPLEMENTED_LOCAL_ONLY',
        created_at_utc=datetime.now(timezone.utc).isoformat(),process_instance=f'{socket.gethostname()}:{os.getpid()}:{start}',
        code=code_identity(ROOT),implementation_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()},
        analysis_process_environment=dict(python=platform.python_version(),versions={n:version(n) for n in ('torch','transformers','peft')}),
        classification='EXACT_PREFIX_BASELINE_DIAGNOSTIC',safe_semantic_reuse_claimed=False,
        GPU_VALIDATION='NOT_RUN',correctness_qualified_speedup=False,service_speedup_claimed=False,
        output_matched_control='OUTPUT_MATCHED_CONTROL_UNAVAILABLE',
        output_matched_reason='No custom decoder/head control validated; primary deliberately retains native B1 output semantics',
        timing=dict(request='frozen B2 synchronized CPU wall; includes normal namespace, clone/container, input/positions, adapter/eval and native Forward to device logits+KV',
            excluded='source/snapshot, hashes, diagnostic hooks, CPU logits copies, output audits/cleanup and serialization',
            prepared='IDEAL_DIAGNOSTIC_ONLY; one-use prepare outside timer; never primary claim',
            prepare_only='separate synchronized component sample after paired timing; not added to request wall'))
    if a.phase=='analyze':
        m['analysis_config']=dict(phase='analyze',inputs=[str(p) for p in a.inputs],bootstrap_seed=a.order_seed)
    else: m['benchmark_config']={k:serial(v) for k,v in vars(a).items() if k!='inputs'}
    atomic_json(a.output_dir/'manifest.json',m)
    environment=dict(process=m['analysis_process_environment'],environment={k:os.environ.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')})
    atomic_json(a.output_dir/'environment.json',environment)
    return m,environment


def analyze(a,m):
    sessions=[]; evidence=[]
    for directory in a.inputs:
        manifest=read_json(directory/'manifest.json'); rows=read_json(directory/'latency_raw_pairs.json')
        sessions.append((manifest,rows))
        evidence.append(dict(directory=str(directory),manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
            samples_sha256=hashlib.sha256((directory/'latency_raw_pairs.json').read_bytes()).hexdigest()))
    result=aggregate(sessions,a.order_seed)
    m['aggregated_experimental_config']=experimental_config([v for v,_ in sessions])
    m['input_benchmark_environments']=[dict(directory=str(p),hardware=s[0]['hardware_identity'],
        environment_identity=s[0]['environment_identity']) for p,s in zip(a.inputs,sessions)]
    m['input_evidence']=evidence; m['execution_status']='COMPLETE'
    atomic_json(a.output_dir/'latency_aggregate_summary.json',result)
    csv_write(a.output_dir/'break_even_summary.csv',[{k:v for k,v in r.items() if k!='session_results'} for r in result])
    atomic_json(a.output_dir/'break_even_decision.json',dict(conditions=result,numeric_threshold=None,
        interpolation='NOT_PERFORMED',extrapolation='NOT_PERFORMED',amortized_break_even='DEFERRED_R2_B4',
        scope='conditional warm hits under B1-compatible output; not output-matched causal benefit',correctness_qualified_speedup=False))
    atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='COMPLETE',conditions=result,
        correctness_qualified_speedup=False,service_speedup_claimed=False))


def run(a,m,environment):
    if a.phase=='analyze': return analyze(a,m)
    pilots=[read_json(p/'capacity_pilot.json') for p in (a.pilot_inputs or [])]
    runtime=Runtime(a); environment['runtime']=runtime.provenance
    environment['runtime_defaults_only']=True
    initial=monitor(runtime.device); environment['monitoring']=[dict(phase='initial',state=initial)]
    m['hardware_identity']=dict(device=runtime.device,name=runtime.provenance.get('gpu_name'),uuid=initial.get('torch_uuid'))
    m['environment_verified']=initial['status']=='GPU_MONITOR_VERIFIED'
    m['environment_identity']=digest(dict(namespace=asdict(runtime.identity()),versions=m['analysis_process_environment']['versions'],
        cuda=runtime.provenance.get('cuda'),hardware=m['hardware_identity'],code=m['code']['sha256'],
        implementation=m['implementation_sha256'],seed=a.seed,endpoint='b1_all_logits',backend='eager',
        api=runtime.provenance.get('api'),installed_api_source=runtime.provenance.get('installed_api_source'),
        tf32=runtime.provenance.get('tf32'),matmul_precision=runtime.provenance.get('float32_matmul_precision')))
    m['conditions']={f'L{n}_P{p}_{arm}':dict(**validate_case(n,p),attention_arm=arm,status='NOT_RUN')
        for n,p,arm in ordered_conditions(a.case_values,a.attention_arm,a.order_seed,a.phase)}
    m['execution_order']=list(m['conditions'])
    artifact={n:{} for n in ('correctness_and_gates','cache_integrity','forward_skip_audit','attention_backend_audit','endpoint_parity','source_costs')}
    rows=[]; profiles={}; cpu=[]; cuda=[]; kernels=[]; components=[]
    capacity=dict(experiment='R2-B3',phase=a.phase,environment_identity=m['environment_identity'],conditions=m['conditions'])
    def persist(context=None):
        if context: m['active_context']=context
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'environment.json',environment)
        for name,data in artifact.items(): atomic_json(a.output_dir/(name+'.json'),data)
        atomic_json(a.output_dir/'workload_config.json',dict(conditions=m['conditions'],execution_order=m['execution_order'],
            input_policy='existing independent per-length tokenizer fixture; no nested-length prefix assertion',seed=a.seed,
            token_ids=getattr(runtime,'ids',None)))
        if a.phase=='pilot':
            capacity.update(execution_status=m['execution_status'],model_weights_verified=bool(m.get('final_weight_check')),
                hardware=m['hardware_identity'],environment_verified=m['environment_verified'])
            atomic_json(a.output_dir/'capacity_pilot.json',capacity)
        if a.phase in ('smoke','pilot','latency'):
            csv_write(a.output_dir/'latency_raw_pairs.csv',rows); atomic_json(a.output_dir/'latency_raw_pairs.json',rows)
            csv_write(a.output_dir/'latency_session_summary.csv',summarize(rows))
        if a.phase=='profile':
            for name,data in [('profiling_cpu_summary',cpu),('profiling_cuda_summary',cuda),('profiling_kernel_counts',kernels),('profiling_components',components)]: csv_write(a.output_dir/(name+'.csv'),data)
            atomic_json(a.output_dir/'profiling_status.json',profiles)
    persist()
    if a.phase in ('latency','profile'):
        # Fail before any source/target request when the selected plan contains
        # unapproved cases. The user explicitly selects a reduced validated plan.
        for key,c in m['conditions'].items():
            persist(dict(case=key,phase='pilot_approval'))
            approved(pilots,f'L{c["L"]}_P{c["P"]}',c['attention_arm'],m['environment_identity'])
    for key,c in m['conditions'].items():
        n,p,arm=c['L'],c['P'],c['attention_arm']
        try:
            validate_case(n,p,runtime.model.config.max_position_embeddings)
            approval=approved(pilots,f'L{n}_P{p}',arm,m['environment_identity']) if a.phase in ('latency','profile') else None
            runtime.snapshot=None; runtime.source_logits=None
            persist(dict(case=key,phase='source_prepare'))
            c['memory_before']=memory(runtime.device,reset=True)
            runtime.configure(arm,'b1'); runtime.prepare(n,p)
            source=runtime.source_report
            condition_identity=digest(dict(namespace=asdict(runtime.identity()),target_token_ids=runtime.ids,
                snapshot=runtime.snapshot.identity_hash if runtime.snapshot else None,arm=arm,endpoint=runtime.endpoint))
            if approval and approval['condition_identity']!=condition_identity: raise ValueError('PILOT_WORKLOAD_IDENTITY_MISMATCH')
            c['condition_identity']=condition_identity
            c.update(token_ids=list(runtime.ids),actual_length=len(runtime.ids),target_token_hash=digest(runtime.ids))
            artifact['source_costs'][key]=dict(source=source,preflight=runtime.preflight,source_before_target=True,
                lookup_selection='NOT_PRESENT',transfer='SAME_DEVICE_NO_OFFLOAD')
            artifact['attention_backend_audit'][key]=runtime.backend
            persist(dict(case=key,phase='structural_gate'))
            g=gate(runtime); artifact['correctness_and_gates'][key]=g
            artifact['forward_skip_audit'][key]={k:g[k] for k in ('full','normal','cache_hit')}
            artifact['endpoint_parity'][key]=dict(policy=runtime.endpoint,output=g['output'],output_matched=False,
                materialization_confounded=True,output_matched_control='OUTPUT_MATCHED_CONTROL_UNAVAILABLE')
            c.update(structural_valid=True,cache_integrity_valid=True,strict_correctness=g['FULL_RELATIVE_STRICT_CORRECTNESS'])
            identity=dict(independent_run_id=a.independent_run_id,case_id=f'L{n}_P{p}',attention_arm=arm,
                endpoint_policy='b1_all_logits',backend_requested='eager',backend_effective=runtime.backend['effective'],
                condition_identity=condition_identity,structural_strict_pass=g['quality']['strict'],
                total_length=n,prefix_length=p,suffix_length=n-p,model=a.model,dtype=a.dtype,
                adapter_identity=asdict(runtime.identity()),source_snapshot_id=runtime.snapshot.identity_hash if runtime.snapshot else 'NO_PREFIX',
                target_token_hash=digest(runtime.ids),seed=a.seed,order_seed=a.order_seed)
            persist()
            if a.phase!='profile':
                start_index=len(rows)
                def save(context=None,rows=None):
                    if rows is not None:
                        # Replaces only this condition's checkpoint, retaining prior cases.
                        all_rows[start_index:]=enrich(rows,runtime,a.phase,source)
                    persist(context)
                all_rows=rows
                run_pairs(runtime,a,identity,save)
                if any(r['correctness_status']=='STRICT_FAIL' for r in rows[start_index:]): c['strict_correctness']='STRICT_FAIL'
                artifact['source_costs'][key]['prepare_only']=prepare_only(runtime) if p else dict(status='NOT_APPLICABLE_NO_PREFIX',wall_ms=None)
            else:
                for _ in range(a.warmup_runs):
                    for mode in modes(runtime): invoke(runtime,mode)
                trace_dir=a.output_dir/'profiler_trace'; trace_dir.mkdir(exist_ok=True)
                for mode in (*modes(runtime),'PREPARE_ONLY'):
                    persist(dict(case=key,phase='profile',mode=mode))
                    try: detail,cr,kr,br=collect(runtime,mode,trace_dir/f'{key}_{mode}.json')
                    except ProfilerUnavailable as exc:
                        profiles[key+'_'+mode]=dict(status='PROFILER_UNAVAILABLE',reason=str(exc)); continue
                    profiles[key+'_'+mode]=detail
                    cpu.extend(dict(identity,mode=mode,**r) for r in cr); kernels.extend(dict(identity,mode=mode,**r) for r in kr)
                    cuda.extend(dict(identity,mode=mode,**r) for r in kr if r['gpu_activity'])
                    components.extend(dict(identity,mode=mode,**r) for r in br); persist()
            artifact['cache_integrity'][key]=runtime.check()
            c['memory_after']=memory(runtime.device)
            observation=monitor(runtime.device); environment['monitoring'].append(dict(case=key,phase='after',state=observation))
            m['environment_verified']=m['environment_verified'] and observation['status']=='GPU_MONITOR_VERIFIED'
            c['status']='CAPACITY_VALIDATED' if a.phase=='pilot' else 'VALID'
            c['valid_performance_evidence']=a.phase=='latency'
            persist()
        except Exception as exc:
            record=failure_record(exc,m.get('active_context',dict(case=key)))
            try: record['memory_at_failure']=memory(runtime.device)
            except Exception as memory_exc: record['memory_at_failure']=dict(status='UNAVAILABLE',reason=str(memory_exc))
            c.update(record); persist()
            if record['status'] not in ('MEMORY_PREFLIGHT_FAILED','OOM'): raise
            # Catch boundary holds only scalar traceback text before allocator cleanup.
            exc.__traceback__=None
            release_failed_case(runtime)
    m['final_weight_check']=runtime.verify_weight_hashes()
    failed=any(c['status'] not in ('VALID','CAPACITY_VALIDATED') for c in m['conditions'].values())
    m.update(execution_status='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE',implementation_status='EXECUTED_MODEL_RUNTIME',
        GPU_VALIDATION='EXECUTED' if runtime.device.startswith('cuda') else 'NOT_RUN',
        correctness_status='STRICT_FAIL' if any(c.get('strict_correctness')=='STRICT_FAIL' for c in m['conditions'].values()) else 'STRICT_PASS' if not failed else 'UNVERIFIED')
    persist()
    atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],conditions=m['conditions'],
        pilot_timings_primary=False,correctness_qualified_speedup=False,service_speedup_claimed=False,
        next_step='Analyze >=3 compatible independent latency sessions; profile selected cases only; amortization deferred R2-B4'))


def main(argv=None):
    a=parse_args(argv); m,environment=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m,environment)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt) as exc:
        failure=failure_record(exc,m.get('active_context'))
        m.update(execution_status='FAILED',failure=failure,correctness_qualified_speedup=False)
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='FAILED',**failure,correctness_qualified_speedup=False))
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
