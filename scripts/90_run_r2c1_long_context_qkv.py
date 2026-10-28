#!/usr/bin/env python3
"""R2-C1: frozen RAW SemCache long-context conditional request diagnosis."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r2c1_workloads import MODES,lengths,coverages,memory_estimate
from semcache.diagnostics.r2c1_benchmark import Runtime,BOUNDARIES,validate,paired
from semcache.diagnostics.r2c1_profiler import collect,ProfilerUnavailable
from semcache.diagnostics.r2c1_analysis import summarize,aggregate,hypotheses
from semcache.diagnostics.r2b2_profiler import monitor
from semcache.diagnostics.r2b2_attention_ablation import backend_audit
from semcache.diagnostics.r2b3_capacity import memory,failure_record
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write
from semcache.diagnostics.r1b3_break_even import code_identity
from semcache.diagnostics.r2c2_profile_memory import (GIB, PilotPeakRecorder,
    ProfileMemoryRejected, pilot_evidence, preflight as profile_preflight, failure_status)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','pilot','latency','analyze','profile'),default='smoke')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'))
    p.add_argument('--dtype',choices=('float32','float16'))
    p.add_argument('--prompt-lengths'); p.add_argument('--coverages',default='0,.5,max')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cache-device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--capacity-mib',type=int,default=512)
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--encoder-revision',default='main')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--order-seed',type=int,default=4101)
    p.add_argument('--independent-run-id',default='run01')
    p.add_argument('--pilot-inputs',type=Path,nargs='+',help='C1 exact length/coverage approvals; also required for >1024 capacity extension')
    p.add_argument('--inputs',type=Path,nargs='+',help='complete independent C1 latency run directories')
    p.add_argument('--max-trace-mib',type=int,default=64)
    p.add_argument('--profile-memory-policy',choices=('conservative','pilot_guarded'),default='conservative',
        help='profile-only opt-in: use compatible completed Pilot allocated-memory evidence')
    p.add_argument('--profile-memory-allowance-gib',type=float,default=1.,help='provisional additional profiler budget (default 1 GiB)')
    p.add_argument('--profile-memory-safety-gib',type=float,default=2.,help='provisional free-memory safety reserve (default 2 GiB)')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if (a.profile_memory_policy!='conservative' or a.profile_memory_allowance_gib!=1. or a.profile_memory_safety_gib!=2.) and a.phase!='profile':
            raise ValueError('profile memory options are profile-only')
        if any(not math.isfinite(v) or v<=0 for v in (a.profile_memory_allowance_gib,a.profile_memory_safety_gib)):
            raise ValueError('profile memory margins must be finite and positive')
        if a.phase=='analyze':
            if not a.inputs or any(v is not None for v in (a.model,a.dtype,a.prompt_lengths,a.pilot_inputs)): raise ValueError('analyze derives experimental configuration from --inputs')
            return a
        if a.inputs: raise ValueError('--inputs only for analyze')
        a.model=a.model or ('facebook/opt-125m' if a.phase=='smoke' else 'facebook/opt-2.7b')
        a.dtype=a.dtype or ('float32' if a.phase=='smoke' else 'float16')
        a.lengths=lengths(a.prompt_lengths,a.phase); a.coverage_values=coverages(a.coverages)
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.phase=='pilot' else 3)
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (20 if a.phase=='latency' else 3)
        if a.warmup_runs<(1 if a.phase=='pilot' else 3) or a.measured_runs<(20 if a.phase=='latency' else 3) or min(a.capacity_mib,a.max_trace_mib)<1: raise ValueError('invalid counts/capacity; latency >=3 warmups/20 pairs')
        if a.phase=='smoke' and (a.model!='facebook/opt-125m' or a.dtype!='float32'): raise ValueError('smoke requires OPT-125M FP32')
        if a.cache_device=='cuda' and not a.device.startswith('cuda'): raise ValueError('CUDA cache requires CUDA model')
        if a.profile_memory_policy=='pilot_guarded' and not a.device.startswith('cuda'): raise ValueError('pilot_guarded requires actual CUDA memory evidence')
        if (a.phase in ('latency','profile') or max(a.lengths)>1024) and not a.pilot_inputs: raise ValueError('matching --pilot-inputs required')
        if a.phase=='profile' and (len(a.lengths)>2 or len(a.coverage_values)>2): raise ValueError('targeted profile only: <=2 lengths/coverages')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': raise ValueError('CUDA_LAUNCH_BLOCKING invalidates timing')
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
    paths=[ROOT/'scripts/90_run_r2c1_long_context_qkv.py',*sorted((ROOT/'src/semcache/diagnostics').glob('r2c1_*.py')),
        ROOT/'src/semcache/diagnostics/r2c2_profile_memory.py',
        *sorted((ROOT/'tests').glob('test_r2c1_*.py')),ROOT/'docs/R2C1_LONG_CONTEXT_QKV_BOTTLENECK.md']
    start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19] if Path('/proc/self/stat').exists() else datetime.now(timezone.utc).isoformat()
    m=dict(experiment='R2-C1',execution_status='RUNNING',implementation_status='IMPLEMENTED_LOCAL_ONLY',
        created_at_utc=datetime.now(timezone.utc).isoformat(),process_instance=f'{socket.gethostname()}:{os.getpid()}:{start}',
        code=code_identity(ROOT),implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
        process_environment=dict(python=platform.python_version(),versions={n:version(n) for n in ('torch','transformers','peft')}),
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,correctness_qualified_speedup=False,
        timing_boundaries=BOUNDARIES,settings=dict(use_cache=False,output_attentions=True,attention_backend='eager',head='native_all_token_logits',adapter='user_a',atol=1e-3,rtol=1e-3),
        historical_prefix_reference=dict(source='user-reported R2-B3.5',fresh_measurement=False,pooled_with_c1=False,cross_harness_speedup_supported=False))
    if a.phase=='analyze':
        m['analysis_config']=dict(inputs=[str(p) for p in a.inputs],bootstrap_seed=a.order_seed)
        m['analysis_process']=dict(gpu_inference_performed=False)
    else:
        m['benchmark_config']={k:serial(v) for k,v in vars(a).items() if k!='inputs'}
        m['GPU_VALIDATION']='NOT_RUN'
    environment=dict(process=m['process_environment'],environment={k:os.environ.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')})
    atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'environment.json',environment)
    atomic_json(a.output_dir/'timing_boundary_audit.json',BOUNDARIES)
    return m,environment


def analyze(a,m):
    sessions=[(load(p/'manifest.json'),load(p/'latency_raw_pairs.json')) for p in a.inputs]
    m['input_evidence']=[dict(directory=str(p),hardware=s[0]['hardware_identity'],environment_identity=s[0]['environment_identity'],
        manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),samples_sha256=hashlib.sha256((p/'latency_raw_pairs.json').read_bytes()).hexdigest()) for p,s in zip(a.inputs,sessions)]
    results=aggregate(sessions,a.order_seed); conclusions=hypotheses(results)
    m['aggregated_experimental_config']=dict(model=sorted({s[0]['benchmark_config']['model'] for s in sessions}),
        dtype=sorted({s[0]['benchmark_config']['dtype'] for s in sessions}),conditions=sorted({k for s in sessions for k in s[0]['conditions']}),
        settings=sessions[0][0]['settings'])
    statuses=[s[0]['GPU_VALIDATION'] for s in sessions]
    m['input_run_validation']=dict(execution_statuses=[s[0]['execution_status'] for s in sessions],gpu_validation=statuses,
        completed_gpu_sessions=sum(v=='EXECUTED' for v in statuses))
    m['GPU_VALIDATION']='AGGREGATED_GPU_RUNS' if all(v=='EXECUTED' for v in statuses) else 'AGGREGATED_NON_GPU_OR_UNVERIFIED_INPUTS'
    m['execution_status']='COMPLETE'; m['correctness_qualified_speedup']=any(r['correctness_qualified_speedup'] for r in results)
    atomic_json(a.output_dir/'latency_aggregate_summary.json',results)
    atomic_json(a.output_dir/'break_even_decision.json',dict(conditions=results,hypotheses=conclusions,numeric_threshold=None,
        interpolation='NOT_PERFORMED',service_speedup_claimed=False))
    atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='COMPLETE',hypotheses=conclusions,conditions=results))


def release_source(runtime,failed=False):
    for name in ('engine','catalog','index','descriptor','caches','hits','plan','source_hashes','policy_state'):
        if hasattr(runtime,name): delattr(runtime,name)
    gc.collect()
    if failed and runtime.a.device.startswith('cuda'):
        import torch
        torch.cuda.synchronize(runtime.a.device); torch.cuda.empty_cache()


def run(a,m,environment):
    if a.phase=='analyze': return analyze(a,m)
    runtime=Runtime(a); model_hash=runtime.setup_identity(); runtime.arm='on'; backend=backend_audit(runtime)
    m['model_weight_identity']=model_hash; m['opt_instance_identity']=id(runtime.model)
    environment['runtime']=runtime.provenance; environment['backend_audit']=backend
    m['hardware_identity']=dict(uuid=runtime.provenance['gpu'].get('gpu_uuid'),device=a.device)
    m['environment_identity']=digest(dict(code=m['code']['sha256'],implementation=m['implementation_sha256'],model_hash=model_hash,
        model=runtime.meta,encoder=runtime.encoder.metadata,storage=runtime.storage,capacity_mib=a.capacity_mib,seed=a.seed,
        versions=m['process_environment']['versions'],cuda=runtime.provenance.get('cuda'),hardware=m['hardware_identity'],backend=backend,settings=m['settings']))
    initial=monitor(a.device); environment['monitoring']=[dict(phase='initial',state=initial)]
    m['environment_verified']=initial['status']=='GPU_MONITOR_VERIFIED'
    guarded=a.phase=='profile' and a.profile_memory_policy=='pilot_guarded'
    pilots=[]
    for directory in a.pilot_inputs or []:
        try: pilots.append(load(directory/'capacity_pilot.json'))
        except (OSError,ValueError,TypeError):
            if not guarded: raise
            # Guarded mode records missing/malformed input files with both
            # memory budgets in the per-case rejection rather than using zero.
    def approved(length,coverage):
        key=f'L{length}_cov{coverage}'
        for p in pilots:
            if p.get('experiment')=='R2-C1' and p.get('environment_identity')==m['environment_identity'] and p.get('execution_status') in ('COMPLETE','COMPLETE_WITH_FAILURES') and p.get('model_weights_verified'):
                c=p['conditions'].get(key)
                if c and c['status']=='CAPACITY_VALIDATED' and c.get('structural_valid'): return c
        raise ValueError('C1_PILOT_REQUIRED: '+key)
    plan=[(n,c) for n in a.lengths for c in a.coverage_values]
    if a.phase=='pilot': plan.sort(key=lambda v:(v[0],str(v[1])))
    else: random.Random(a.order_seed).shuffle(plan)
    m['conditions']={f'L{n}_cov{cov}':dict(length=n,requested_coverage=cov,status='NOT_RUN') for n,cov in plan}
    m['execution_order']=list(m['conditions'])
    artifacts={name:{} for name in ('workload_audit','coverage_audit','correctness_and_gates')}
    rows=[]; cpu=[]; kernels=[]; profiles={}
    def persist(context=None):
        if context: m['active_context']=context
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'environment.json',environment)
        for name,data in artifacts.items(): atomic_json(a.output_dir/(name+'.json'),data)
        if a.phase in ('smoke','pilot','latency'):
            atomic_json(a.output_dir/'latency_raw_pairs.json',rows); csv_write(a.output_dir/'latency_raw_pairs.csv',rows)
            csv_write(a.output_dir/'latency_session_summary.csv',summarize(rows))
        if a.phase=='pilot': atomic_json(a.output_dir/'capacity_pilot.json',dict(experiment='R2-C1',environment_identity=m['environment_identity'],
            execution_status=m['execution_status'],conditions=m['conditions'],model_weights_verified=bool(m.get('final_weights_verified'))))
        if a.phase=='profile':
            csv_write(a.output_dir/'bottleneck_breakdown.csv',cpu); csv_write(a.output_dir/'profiler_kernel_summary.csv',kernels)
            atomic_json(a.output_dir/'profiling_status.json',profiles)
    persist()
    if a.phase in ('latency','profile') and not guarded:
        for n,cov in plan: approved(n,cov)
    if max(a.lengths)>1024:
        for n in (512,1024):
            for cov in a.coverage_values: approved(n,cov)
    current=None; length_baseline=None; length_peak=None
    for key,c in m['conditions'].items():
        peak_recorder=None
        try:
            n,cov=c['length'],c['requested_coverage']; persist(dict(case=key,phase='preflight'))
            lengths(str(n),a.phase,runtime.model.config.max_position_embeddings)
            needs_prepare=current!=n
            if needs_prepare:
                release_source(runtime); current=None
                c['memory_before']=memory(a.device,reset=True)
                length_baseline=c['memory_before']; length_peak=None
            elif guarded:
                c['memory_before']=memory(a.device)
            if needs_prepare or guarded:
                estimate=memory_estimate(runtime.model.config,n,next(runtime.model.parameters()).element_size(),a.capacity_mib*1024**2,a.phase=='profile')
                estimate['model_lora_resident_bytes']=sum(p.numel()*p.element_size() for p in runtime.model.parameters())
                estimate['semantic_encoder_resident_bytes']=sum(p.numel()*p.element_size() for p in runtime.encoder.model.parameters())
                c['preflight']=estimate
                if estimate['source_qkv_backing_bytes']>a.capacity_mib*1024**2: raise MemoryError('RAW source QKV exceeds requested Catalog capacity; explicit larger-capacity pilot required')
                gpu=c['memory_before']['gpu']
                if guarded:
                    evidence=None; evidence_errors=[]
                    for directory in a.pilot_inputs:
                        try:
                            evidence=pilot_evidence(directory,m,n,cov); break
                        except (OSError,ValueError,KeyError,TypeError) as exc: evidence_errors.append(f'{directory}: {exc}')
                    audit=profile_preflight('pilot_guarded',estimate,gpu,evidence,
                        int(a.profile_memory_allowance_gib*GIB),int(a.profile_memory_safety_gib*GIB),
                        evidence_error='; '.join(evidence_errors) if evidence is None else None)
                    c['memory_preflight_audit']=audit; persist()
                    if not audit['approved']: raise ProfileMemoryRejected(audit)
                elif a.device.startswith('cuda'):
                    c['memory_preflight_audit']=profile_preflight('conservative',estimate,gpu)
                    # Preserve C1's default estimate and exact rejection rule.
                    if estimate['estimated_additional_bytes']>gpu['free_bytes']+gpu['reserved_bytes']-gpu['allocated_bytes']: raise MemoryError('C1 estimated live allocations exceed GPU headroom')
            if a.phase=='pilot' and a.device.startswith('cuda'):
                peak_recorder=PilotPeakRecorder(a.device); peak_recorder.begin(length_baseline,length_peak)
            if needs_prepare:
                runtime.prepare(n); current=n
            runtime.coverage=cov
            semantic_ids=runtime.encoder.tokenizer(runtime.text,truncation=False)['input_ids']
            artifact=dict(token_ids=runtime.ids,token_hash=digest(runtime.ids),source_snapshot=runtime.snapshot,model_namespace=runtime.namespace,
                source_costs=runtime.costs,raw_cache_bytes=runtime.catalog.resident_bytes,cache_storage=runtime.storage,
                original_preflight=runtime.preflight,semantic_token_count=len(semantic_ids),semantic_effective_token_cap=runtime.encoder.max_length,
                semantic_truncated=len(semantic_ids)>runtime.encoder.max_length,model_token_count=len(runtime.ids),model_truncated=False,
                input_policy='existing exact tokenizer roundtrip per length, includes BOS/special tokens')
            artifacts['workload_audit'][key]=artifact
            c['condition_identity']=digest(dict(ids=runtime.ids,source_hashes=runtime.source_hashes,weights=runtime.weights,namespace=runtime.namespace,coverage=cov,storage=runtime.storage))
            if a.phase in ('latency','profile'):
                expected=(c['memory_preflight_audit']['pilot'] if guarded else approved(n,cov))
                if expected['condition_identity']!=c['condition_identity']: raise ValueError('pilot source/token identity mismatch')
            persist(dict(case=key,phase='untimed_gate'))
            g=validate(runtime); artifacts['correctness_and_gates'][key]=g
            artifacts['coverage_audit'][key]=g['audits']['RAW_SEMCACHE_QKV']['coverage']
            c.update(structural_valid=True,strict_pass=all(v['allclose'] for v in g['quality'].values()),
                actual_reused_tokens=g['audits']['RAW_SEMCACHE_QKV']['coverage']['reused_tokens'])
            identity=dict(independent_run_id=a.independent_run_id,case_id=key,length=n,requested_coverage=cov,condition_identity=c['condition_identity'],
                model=a.model,dtype=a.dtype,seed=a.seed,order_seed=a.order_seed,source_snapshot=runtime.snapshot,
                endpoint_policy='all_token_logits_use_cache_false_attention_true',cache_device=a.cache_device,adapter='user_a')
            persist()
            if a.phase!='profile':
                start=len(rows); all_rows=rows
                def save(context=None,rows=None):
                    if rows is not None: all_rows[start:]=rows
                    persist(context)
                paired(runtime,a,identity,save)
                c['strict_pass']=c['strict_pass'] and all(r['strict_pass'] for r in rows[start:])
            else:
                for _ in range(a.warmup_runs):
                    for mode in MODES: runtime.invoke(mode)
                directory=a.output_dir/'profiler_trace'; directory.mkdir(exist_ok=True)
                for mode in MODES:
                    persist(dict(case=key,phase='profile',mode=mode))
                    try: detail,cr,kr=collect(runtime,mode,directory/f'{key}_{mode}.json',a.max_trace_mib)
                    except ProfilerUnavailable as exc: profiles[key+'_'+mode]=dict(status='PROFILER_UNAVAILABLE',reason=str(exc)); continue
                    profiles[key+'_'+mode]=detail
                    cpu.extend(dict(identity,mode=mode,**r) for r in cr); kernels.extend(dict(identity,mode=mode,**r) for r in kr); persist()
            runtime.guard(True); c['memory_after']=memory(a.device)
            if peak_recorder:
                length_peak=peak_recorder.finish(); c['pilot_memory']=length_peak
            observation=monitor(a.device); environment['monitoring'].append(dict(case=key,phase='after',state=observation))
            m['environment_verified']=m['environment_verified'] and observation['status']=='GPU_MONITOR_VERIFIED'
            c['status']='CAPACITY_VALIDATED' if a.phase=='pilot' else 'VALID'; persist()
        except Exception as exc:
            c.update(failure_record(exc,m.get('active_context')))
            c['status']=failure_status(exc)
            if isinstance(exc,ProfileMemoryRejected): c['memory_preflight_audit']=exc.audit
            try: c['memory_at_failure']=memory(a.device)
            except Exception as memory_error: c['memory_at_failure']=dict(status='UNAVAILABLE',reason=str(memory_error))
            persist()
            if c['status'] not in ('MEMORY_PREFLIGHT_FAILED','OOM'): raise
            exc.__traceback__=None; release_source(runtime,failed=True); current=None
        finally:
            if peak_recorder: peak_recorder.close()
    runtime.model_guard(True); m['final_weights_verified']=True
    valid=all(c['status'] in ('VALID','CAPACITY_VALIDATED') for c in m['conditions'].values())
    any_valid=any(c['status'] in ('VALID','CAPACITY_VALIDATED') for c in m['conditions'].values())
    m.update(execution_status='COMPLETE' if valid else 'COMPLETE_WITH_FAILURES',implementation_status='EXECUTED_MODEL_RUNTIME',
        GPU_VALIDATION=('EXECUTED' if any_valid else 'FAILED_OR_CAPACITY_REJECTED') if a.device.startswith('cuda') else 'NOT_RUN')
    persist(); atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],conditions=m['conditions'],
        correctness_qualified_speedup=False,next_step='aggregate independent C1 sessions; Prefix KV reference remains separate',timing_boundary=BOUNDARIES))


def main(argv=None):
    a=parse_args(argv); m,environment=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m,environment)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt) as exc:
        failure=failure_record(exc,m.get('active_context')); failure['status']=failure_status(exc)
        m.update(execution_status='FAILED',failure=failure,correctness_qualified_speedup=False)
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='FAILED',**failure,correctness_qualified_speedup=False))
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
