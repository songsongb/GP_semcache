#!/usr/bin/env python3
"""R2-B3.5 real one-token LM Head: same-session four-path output-policy ablation."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r2b35_executor import Runtime,MODES
from semcache.diagnostics.r2b35_endpoint_audit import gate
from semcache.diagnostics.r2b35_benchmark import PRIMARY,PILOT,run_pairs,profile
from semcache.diagnostics.r2b35_analysis import summarize,aggregate
from semcache.diagnostics.r2b3_workloads import validate_case
from semcache.diagnostics.r2b3_capacity import memory,failure_record,release_failed_case
from semcache.diagnostics.r2b2_profiler import monitor,ProfilerUnavailable
from semcache.diagnostics.r2b15_cost import prepare_only
from semcache.diagnostics.r1b3_break_even import code_identity
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','pilot','latency','analyze','profile'),default='smoke')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'))
    p.add_argument('--dtype',choices=('float32','float16'))
    p.add_argument('--cases'); p.add_argument('--case-group',choices=('primary','pilot','extended'))
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cache-device',default=None)
    p.add_argument('--adapter',choices=('user_a','user_b'),default='user_a')
    p.add_argument('--attention-arm',choices=('off','on'),default='off')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--allow-download',action='store_true')
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--order-seed',type=int,default=3501)
    p.add_argument('--independent-run-id',default='run01')
    p.add_argument('--smoke-inputs',type=Path,nargs='+',help='reviewed B3.5 GPU FP32 smoke; required before pilot')
    p.add_argument('--pilot-inputs',type=Path,nargs='+',help='exact per-case B3.5 approvals for latency/profile')
    p.add_argument('--inputs',type=Path,nargs='+',help='complete B3.5 latency sessions to aggregate')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.phase=='analyze':
            if not a.inputs or any(v is not None for v in (a.model,a.dtype,a.cases,a.case_group,a.pilot_inputs,a.smoke_inputs)):
                raise ValueError('analyze requires --inputs and derives experiment settings from manifests')
            return a
        if a.inputs: raise ValueError('--inputs only for analyze')
        a.model=a.model or ('facebook/opt-125m' if a.phase=='smoke' else 'facebook/opt-2.7b')
        a.dtype=a.dtype or ('float32' if a.phase=='smoke' else 'float16')
        a.case_group=a.case_group or ('pilot' if a.phase=='pilot' else 'primary')
        defaults=[(32,16),(32,31)] if a.phase=='smoke' else PILOT if a.case_group=='pilot' else PRIMARY+([(256,255),(256,192)] if a.case_group=='extended' else [])
        a.case_values=[tuple(map(int,c.split(':'))) for c in a.cases.split(',')] if a.cases else defaults
        allowed=set(PRIMARY+PILOT+[(32,16),(32,31),(256,255),(256,192)])
        if not a.case_values or len(set(a.case_values))!=len(a.case_values) or any(c not in allowed for c in a.case_values): raise ValueError('only compact B3.5 cases supported')
        for c in a.case_values: validate_case(*c)
        if a.phase=='smoke' and (a.model!='facebook/opt-125m' or a.dtype!='float32' or any(n!=32 for n,_ in a.case_values)): raise ValueError('smoke is OPT-125M FP32, 32 tokens')
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.phase=='pilot' else 3)
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (20 if a.phase=='latency' else 3)
        if a.warmup_runs<(1 if a.phase=='pilot' else 3) or a.measured_runs<(20 if a.phase=='latency' else 3): raise ValueError('pilot >=1 warmup/3 pairs; latency >=3/20; smoke >=3/3')
        if a.phase=='pilot' and not a.smoke_inputs: raise ValueError('pilot requires --smoke-inputs')
        if a.phase in ('latency','profile') and not a.pilot_inputs: raise ValueError('latency/profile require --pilot-inputs')
        if a.phase=='profile' and (not a.cases or len(a.case_values)>3): raise ValueError('profile requires 1..3 explicitly selected cases')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': raise ValueError('CUDA_LAUNCH_BLOCKING invalidates request timing')
    except ValueError as exc: p.error(str(exc))
    return a


def load(path): return json.loads(Path(path).read_text())


def initialize(a):
    a.output_dir.mkdir(parents=True,exist_ok=False)
    def version(n):
        try: return importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: return None
    def serial(v):
        if isinstance(v,Path): return str(v)
        if isinstance(v,(list,tuple)): return [serial(x) for x in v]
        return v
    paths=[ROOT/'scripts/89_run_r2b35_output_matched.py',*sorted((ROOT/'src/semcache/diagnostics').glob('r2b35_*.py')),
        *sorted((ROOT/'tests').glob('test_r2b35_*.py')),ROOT/'docs/R2B35_OUTPUT_MATCHED_LATENCY.md']
    start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19] if Path('/proc/self/stat').exists() else datetime.now(timezone.utc).isoformat()
    m=dict(experiment='R2-B3.5',execution_status='RUNNING',implementation_status='IMPLEMENTED_LOCAL_ONLY',
        created_at_utc=datetime.now(timezone.utc).isoformat(),process_instance=f'{socket.gethostname()}:{os.getpid()}:{start}',
        code=code_identity(ROOT),implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
        process_environment=dict(python=platform.python_version(),versions={n:version(n) for n in ('torch','transformers','peft')}),
        classification='EXACT_PREFIX_BASELINE_DIAGNOSTIC',GPU_VALIDATION='NOT_RUN',output_matching='OUTPUT_MATCHED_UNAVAILABLE',
        correctness_qualified_speedup=False,safe_semantic_reuse_claimed=False,
        timing=dict(primary='frozen B2 synchronized CPU request wall: adapter/eval, inputs, namespace, cloning/container, native PEFT invocation, device logits and final KV',
            excluded='source prefill/snapshot, matched context install/restore, audit hooks/hashes, CPU output copies, cache checks, cleanup and serialization',
            matched='real PEFT wrapper -> validated native decoder -> last hidden -> actual head; setup outside timer; decoder/head routing overhead inside',
            inference='conditional warm request; policy sensitivity not exact head-cost subtraction; amortization deferred R2-B4'))
    if a.phase=='analyze': m['analysis_config']=dict(inputs=[str(p) for p in a.inputs],bootstrap_seed=a.order_seed)
    else: m['benchmark_config']={k:serial(v) for k,v in vars(a).items() if k!='inputs'}
    environment=dict(process=m['process_environment'],environment={k:os.environ.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')})
    atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'environment.json',environment)
    atomic_json(a.output_dir/'output_matched_capability.json',dict(status='UNVERIFIED',reason='runtime not initialized'))
    return m,environment


def analyze(a,m):
    sessions=[(load(p/'manifest.json'),load(p/'latency_raw_pairs.json')) for p in a.inputs]
    result=aggregate(sessions,a.order_seed)
    m['aggregated_experimental_config']=dict(model=sorted({s[0]['benchmark_config']['model'] for s in sessions}),
        dtype=sorted({s[0]['benchmark_config']['dtype'] for s in sessions}),
        conditions=sorted({k for s in sessions for k in s[0]['conditions']}),
        attention_arms=sorted({s[0]['benchmark_config']['attention_arm'] for s in sessions}),
        endpoints=['legacy_all_logits','matched_last_hidden_head'],backend='eager')
    m['input_evidence']=[dict(directory=str(p),hardware=s[0]['hardware_identity'],environment_identity=s[0]['environment_identity'],
        manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),
        samples_sha256=hashlib.sha256((p/'latency_raw_pairs.json').read_bytes()).hexdigest()) for p,s in zip(a.inputs,sessions)]
    atomic_json(a.output_dir/'latency_aggregate_summary.json',result)
    csv_write(a.output_dir/'endpoint_ablation_summary.csv',[{k:v for k,v in r.items() if k!='session_results'} for r in result])
    atomic_json(a.output_dir/'output_matched_decision.json',dict(conditions=result,
        interpretation='evaluate measured matched delta and sensitivity; do not assign speculative fractions to head or prefix costs',
        amortized_speedup='UNVERIFIED_DEFERRED_R2_B4'))
    m.update(execution_status='COMPLETE',output_matching='OUTPUT_MATCHED_VALID',
        correctness_qualified_speedup=any(r['correctness_qualified_speedup'] for r in result))
    atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='COMPLETE',conditions=result))


def run(a,m,environment):
    if a.phase=='analyze': return analyze(a,m)
    if a.phase=='pilot':
        smokes=[load(p/'manifest.json') for p in a.smoke_inputs]
        if not any(s.get('experiment')=='R2-B3.5' and s.get('execution_status')=='COMPLETE' and s.get('benchmark_config',{}).get('phase')=='smoke' and
            s.get('GPU_VALIDATION')=='EXECUTED' and s.get('native_endpoint_equivalent') and s['code']['sha256']==m['code']['sha256'] and
            s['process_environment']['versions']==m['process_environment']['versions'] for s in smokes):
            raise ValueError('REVIEWED_SMOKE_REQUIRED: complete GPU FP32 endpoint-equivalent smoke on matching code/software')
    runtime=Runtime(a); cap=runtime.setup_matched()
    atomic_json(a.output_dir/'output_matched_capability.json',cap)
    environment['runtime_defaults_only']=True; environment['runtime']=runtime.provenance
    initial=monitor(runtime.device); environment['monitoring']=[dict(phase='initial',state=initial)]
    m['hardware_identity']=dict(name=runtime.provenance.get('gpu_name'),uuid=initial.get('torch_uuid'),device=runtime.device)
    m['environment_verified']=initial['status']=='GPU_MONITOR_VERIFIED'
    m['environment_identity']=digest(dict(namespace=asdict(runtime.identity()),code=m['code']['sha256'],implementation=m['implementation_sha256'],
        versions=m['process_environment']['versions'],cuda=runtime.provenance.get('cuda'),hardware=m['hardware_identity'],seed=a.seed,
        backend='eager',endpoints=['legacy_all_logits','matched_last_hidden_head'],capability=cap,
        tf32=runtime.provenance.get('tf32'),matmul=runtime.provenance.get('float32_matmul_precision')))
    plan=list(a.case_values)
    if a.phase=='pilot': plan.sort()
    else: random.Random(a.order_seed).shuffle(plan)
    m['conditions']={f'L{n}_P{p}_{a.attention_arm}':dict(**validate_case(n,p),attention_arm=a.attention_arm,status='NOT_RUN') for n,p in plan}
    m['execution_order']=list(m['conditions'])
    pilots=[load(p/'capacity_pilot.json') for p in (a.pilot_inputs or [])]
    def approval(key):
        for p in pilots:
            if p.get('experiment')=='R2-B3.5' and p.get('execution_status') in ('COMPLETE','COMPLETE_WITH_FAILURES') and p.get('model_weights_verified') and p.get('environment_identity')==m['environment_identity']:
                c=p['conditions'].get(key)
                if c and c['status']=='CAPACITY_VALIDATED' and c['output_matching']=='OUTPUT_MATCHED_VALID': return c
        raise ValueError('MATCHED_PILOT_REQUIRED: '+key)
    artifacts={n:{} for n in ('model_adapter_audit','endpoint_shape_audit','native_matched_parity','cache_integrity','forward_skip_audit','correctness_and_gates','source_costs')}
    rows=[]; cpu=[]; kernels=[]; components=[]; profiles={}
    def persist(context=None):
        if context: m['active_context']=context
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'environment.json',environment)
        atomic_json(a.output_dir/'workload_config.json',dict(conditions=m['conditions'],execution_order=m['execution_order'],input_policy='B1 deterministic per-length exact token IDs'))
        for name,data in artifacts.items(): atomic_json(a.output_dir/(name+'.json'),data)
        if a.phase in ('smoke','pilot','latency'):
            atomic_json(a.output_dir/'latency_raw_pairs.json',rows); csv_write(a.output_dir/'latency_raw_pairs.csv',rows)
            csv_write(a.output_dir/'latency_session_summary.csv',summarize(rows))
        if a.phase=='pilot': atomic_json(a.output_dir/'capacity_pilot.json',dict(experiment='R2-B3.5',execution_status=m['execution_status'],
            environment_identity=m['environment_identity'],conditions=m['conditions'],model_weights_verified=bool(m.get('final_weight_check'))))
        if a.phase=='profile':
            for name,data in [('profiling_cpu_summary',cpu),('profiling_kernel_counts',kernels),('profiling_cuda_summary',[r for r in kernels if r['gpu_activity']]),('profiling_components',components)]: csv_write(a.output_dir/(name+'.csv'),data)
            atomic_json(a.output_dir/'profiling_status.json',profiles)
    persist()
    if a.phase in ('latency','profile'):
        for key in m['conditions']: approval(key)
    runtime.configure(a.attention_arm,'b1')
    for key,c in m['conditions'].items():
        try:
            persist(dict(case=key,phase='source_prepare'))
            runtime.snapshot=None; runtime.source_logits=None
            c['memory_before']=memory(runtime.device,reset=True)
            validate_case(c['L'],c['P'],runtime.model.config.max_position_embeddings)
            runtime.prepare(c['L'],c['P'])
            c['condition_identity']=digest(dict(ids=runtime.ids,snapshot=runtime.snapshot.identity_hash,
                namespace=asdict(runtime.identity()),arm=runtime.arm,endpoints=['legacy_all_logits','matched_last_hidden_head']))
            if a.phase in ('latency','profile') and approval(key)['condition_identity']!=c['condition_identity']: raise ValueError('PILOT_WORKLOAD_MISMATCH')
            c['token_ids']=list(runtime.ids)
            c['namespace']=asdict(runtime.identity()); c['source_snapshot_id']=runtime.snapshot.identity_hash
            artifacts['source_costs'][key]=dict(source=runtime.source_report,preflight=runtime.preflight)
            persist(dict(case=key,phase='endpoint_and_structural_gate'))
            g=gate(runtime); artifacts['correctness_and_gates'][key]=g
            artifacts['endpoint_shape_audit'][key]={mode:v['endpoint'] for mode,v in g['audits'].items()}
            artifacts['model_adapter_audit'][key]={mode:v['endpoint']['lora'] for mode,v in g['audits'].items()}
            artifacts['forward_skip_audit'][key]={mode:v['physical'] for mode,v in g['audits'].items()}
            artifacts['native_matched_parity'][key]=g['axis_a']; artifacts['cache_integrity'][key]=g['canonical']
            c.update(output_matching=g['output_matching'],native_endpoint_equivalent=g['native_endpoint_equivalent'],
                full_warm_strict=g['axis_b']['matched']['allclose'] and g['axis_b']['legacy_suffix']['allclose'])
            m['output_matching']='OUTPUT_MATCHED_VALID'
            persist()
            if a.phase=='smoke' and not c['native_endpoint_equivalent']: raise PrefixError('NATIVE_ENDPOINT_EQUIVALENCE_FAIL','FP32 smoke native/matched gate failed')
            identity=dict(independent_run_id=a.independent_run_id,case_id=f'L{c["L"]}_P{c["P"]}',L=c['L'],P=c['P'],S=c['S'],
                attention_arm=a.attention_arm,backend='eager',condition_identity=c['condition_identity'],model=a.model,dtype=a.dtype,
                namespace=asdict(runtime.identity()),source_snapshot_id=runtime.snapshot.identity_hash,
                endpoints=['legacy_all_logits','matched_last_hidden_head'],gate_native_equivalent=c['native_endpoint_equivalent'],
                gate_full_warm_strict=c['full_warm_strict'],seed=a.seed,order_seed=a.order_seed)
            if a.phase!='profile':
                start=len(rows); all_rows=rows
                def save(context=None,rows=None):
                    if rows is not None: all_rows[start:]=rows
                    persist(context)
                run_pairs(runtime,a,identity,save)
                c['native_endpoint_equivalent']=c['native_endpoint_equivalent'] and all(r['native_endpoint_equivalent'] for r in rows[start:])
                c['full_warm_strict']=c['full_warm_strict'] and all(r['full_warm_strict'] for r in rows[start:])
                artifacts['source_costs'][key]['prepare_only']=prepare_only(runtime)
            else:
                for _ in range(a.warmup_runs):
                    for mode in MODES: runtime.execute(mode)
                trace=a.output_dir/'profiler_trace'; trace.mkdir(exist_ok=True)
                for mode in MODES:
                    persist(dict(case=key,phase='profile',mode=mode))
                    try: detail,cr,kr,br=profile(runtime,mode,trace/f'{key}_{mode}.json')
                    except ProfilerUnavailable as exc: profiles[key+'_'+mode]=dict(status='PROFILER_UNAVAILABLE',reason=str(exc)); continue
                    profiles[key+'_'+mode]=detail
                    cpu.extend(dict(identity,mode=mode,**r) for r in cr); kernels.extend(dict(identity,mode=mode,**r) for r in kr)
                    components.extend(dict(identity,mode=mode,**r) for r in br); persist()
            artifacts['cache_integrity'][key]=runtime.check(); c['memory_after']=memory(runtime.device)
            observation=monitor(runtime.device); environment['monitoring'].append(dict(case=key,phase='after',state=observation))
            m['environment_verified']=m['environment_verified'] and observation['status']=='GPU_MONITOR_VERIFIED'
            c['status']='CAPACITY_VALIDATED' if a.phase=='pilot' else 'VALID'; persist()
        except Exception as exc:
            c.update(failure_record(exc,m.get('active_context'))); c['output_matching']='OUTPUT_MATCHED_INVALID'; persist()
            if c['status'] not in ('OOM','MEMORY_PREFLIGHT_FAILED'): raise
            exc.__traceback__=None; release_failed_case(runtime)
    m['final_weight_check']=runtime.verify_weight_hashes()
    valid=all(c['status'] in ('VALID','CAPACITY_VALIDATED') for c in m['conditions'].values())
    m.update(execution_status='COMPLETE' if valid else 'COMPLETE_WITH_FAILURES',implementation_status='EXECUTED_MODEL_RUNTIME',
        GPU_VALIDATION='EXECUTED' if runtime.device.startswith('cuda') else 'NOT_RUN',output_matching='OUTPUT_MATCHED_VALID' if valid else 'OUTPUT_MATCHED_INVALID',
        native_endpoint_equivalent=valid and all(c['native_endpoint_equivalent'] for c in m['conditions'].values()))
    persist(); atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],conditions=m['conditions'],
        correctness_qualified_speedup=False,interpretation='independent latency sessions required; shape-validity and native numerical equivalence are separate'))


def main(argv=None):
    a=parse_args(argv); m,environment=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m,environment)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt) as exc:
        failure=failure_record(exc,m.get('active_context')); m.update(execution_status='FAILED',failure=failure,correctness_qualified_speedup=False)
        if getattr(exc,'code',None)=='OUTPUT_MATCHED_UNAVAILABLE':
            m['output_matching']='OUTPUT_MATCHED_UNAVAILABLE'
            atomic_json(a.output_dir/'output_matched_capability.json',dict(status='OUTPUT_MATCHED_UNAVAILABLE',reason=str(exc)))
        elif getattr(exc,'code',None)=='OUTPUT_MATCHED_INVALID': m['output_matching']='OUTPUT_MATCHED_INVALID'
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='FAILED',**failure,correctness_qualified_speedup=False))
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
