#!/usr/bin/env python3
"""R2-B2: native exact-prefix paired latency; isolated eager ON/OFF and profiling."""
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
from semcache.diagnostics.r2b2_benchmark import B2Runtime,structural_gate,run_pairs,modes
from semcache.diagnostics.r2b2_analysis import summarize,aggregate
from semcache.diagnostics.r2b2_profiler import monitor,collect,ProfilerUnavailable
from semcache.diagnostics.r2b15_cost import invoke
from semcache.diagnostics.r1b3_break_even import code_identity
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','latency','profile','analyze'),default='smoke')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-125m')
    p.add_argument('--dtype',choices=('float32','float16'),default='float32')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cache-device',default=None)
    p.add_argument('--adapter',choices=('user_a','user_b'),default='user_a')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--allow-download',action='store_true')
    p.add_argument('--cases',help='small explicit L:P list; defaults 32:16,32:31 or 128:0,128:64,128:127')
    p.add_argument('--attention-arm',choices=('on','off','both'),default='both')
    p.add_argument('--attention-backend',choices=('eager',),default='eager')
    p.add_argument('--endpoint-policy',choices=('b1','auto','native-last'),default='auto')
    p.add_argument('--warmup-runs',type=int,default=3); p.add_argument('--measured-runs',type=int)
    p.add_argument('--independent-run-id',default='run01'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--order-seed',type=int,default=901)
    p.add_argument('--inputs',type=Path,nargs='+',help='complete latency directories; analyze only')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        a.case_values=[tuple(map(int,v.split(':'))) for v in (a.cases or ('32:16,32:31' if a.model.endswith('125m') else '128:0,128:64,128:127')).split(',')]
        allowed={(32,0),(32,16),(32,31),(128,0),(128,64),(128,127),(256,192)}
        if not 1<=len(a.case_values)<=4 or len(set(a.case_values))!=len(a.case_values) or any(v not in allowed for v in a.case_values): raise ValueError('unsupported/duplicate cases; no broad sweep')
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (3 if a.phase=='smoke' else 20)
        if a.warmup_runs<3 or a.measured_runs<(20 if a.phase=='latency' else 3): raise ValueError('>=3 warmups, latency >=20 pairs (smoke >=3) required')
        if a.phase=='smoke' and (a.model!='facebook/opt-125m' or a.dtype!='float32'): raise ValueError('smoke requires OPT-125M FP32')
        if a.phase=='analyze' and not a.inputs: raise ValueError('--inputs required for analyze')
        if a.phase!='analyze' and a.inputs: raise ValueError('--inputs only for analyze')
        if a.phase in ('smoke','latency') and os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': raise ValueError('CUDA_LAUNCH_BLOCKING=1 invalidates minimal timings')
    except ValueError as exc: p.error(str(exc))
    return a


def initialize(a):
    a.output_dir.mkdir(parents=True,exist_ok=False)
    files=[ROOT/'scripts/87_run_r2b2_paired_latency.py',*sorted((ROOT/'src/semcache/diagnostics').glob('r2b2_*.py')),
        *sorted((ROOT/'tests').glob('test_r2b2_*.py')),ROOT/'docs/R2B2_PAIRED_LATENCY_BENCHMARK.md']
    def version(n):
        try: return importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: return None
    def serial(v):
        if isinstance(v,Path): return str(v)
        if isinstance(v,list): return [serial(x) for x in v]
        return v
    process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19] if Path('/proc/self/stat').exists() else datetime.now(timezone.utc).isoformat()
    m=dict(experiment='R2-B2',created_at_utc=datetime.now(timezone.utc).isoformat(),execution_status='RUNNING',
        implementation_status='IMPLEMENTED_LOCAL_ONLY',GPU_SMOKE_VALIDATED=False,LATENCY_REPEATED=False,
        PROFILE_COLLECTED=False,PERFORMANCE_MEASURED=False,PERFORMANCE_UNVERIFIED=True,CORRECTNESS_QUALIFIED_SPEEDUP=False,
        correctness_status='UNVERIFIED',classification='EXACT_PREFIX_BASELINE_DIAGNOSTIC',safe_semantic_reuse_claimed=False,
        process_instance=f'{socket.gethostname()}:{os.getpid()}:{process_start}',
        code=code_identity(ROOT),implementation_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()},
        config={k:serial(v) for k,v in vars(a).items()},versions={n:version(n) for n in ('torch','transformers','peft')},
        python=platform.python_version(),
        timing=dict(primary='synchronized CPU wall: adapter/eval, tensor inputs/mask/positions, namespace, clone/factory, native forward to device logits+final KV',
            excluded='source prefill/snapshot, heavy hashes, audit hooks, CPU logits transfer, cache inspection, serialization, frame cleanup',
            prepared='IDEAL_DIAGNOSTIC_ONLY: single-use materialize outside request; in-request guard retained',
            endpoint='pretokenized input to next-token logits and final KV on device; not generation TTFT or CPU response latency',
            source='independent Prefix-only forward; never slice target Full to build cache'),
        settings=dict(use_cache=True,attention_backend='eager',atol=1e-3,rtol=1e-3))
    atomic_json(a.output_dir/'manifest.json',m)
    atomic_json(a.output_dir/'environment.json',dict(versions=m['versions'],python=m['python'],environment={k:os.environ.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUBLAS_WORKSPACE_CONFIG')}))
    return m


def run(a,m):
    if a.phase=='analyze':
        sessions=[]
        m['input_evidence']=[]
        for path in a.inputs:
            manifest=json.loads((path/'manifest.json').read_text())
            # JSON companion preserves scalar types and avoids ambiguous CSV coercion.
            rows=json.loads((path/'latency_raw_pairs.json').read_text())
            sessions.append((manifest,rows))
            m['input_evidence'].append(dict(directory=str(path),manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),
                samples_sha256=hashlib.sha256((path/'latency_raw_pairs.json').read_bytes()).hexdigest()))
        result=aggregate(sessions,a.order_seed)
        atomic_json(a.output_dir/'latency_aggregate_summary.json',result)
        m.update(execution_status='COMPLETE',LATENCY_REPEATED=all(r['LATENCY_REPEATED'] for r in result),
            PERFORMANCE_MEASURED=True,PERFORMANCE_UNVERIFIED=False,
            CORRECTNESS_QUALIFIED_SPEEDUP=any(r['CORRECTNESS_QUALIFIED_SPEEDUP'] for r in result))
        atomic_json(a.output_dir/'decision_summary.json',dict(conditions=result,scope='conditional warm request only; Prepared excluded from speedup qualification'))
        return
    runtime=B2Runtime(a); m['runtime']=runtime.provenance
    # Initial B1 provenance describes loader defaults. Per-arm effective settings override it explicitly below.
    m['runtime_defaults_only']=True
    environment=json.loads((a.output_dir/'environment.json').read_text()); environment['runtime']=runtime.provenance
    environment['monitoring']=[]
    rows=[]; gates={}; backend={}; sources={}; integrity={}; endpoints={}; skips={}; cpu=[]; cuda=[]; counts=[]; components=[]; profiles={}
    arms=['on','off'] if a.attention_arm=='both' else [a.attention_arm]
    if a.order_seed%2: arms.reverse()
    m['arm_order']=arms
    def persist(context=None,rows_new=None):
        if context: m['active_context']=context
        for name,data in [('correctness_and_gates',gates),('cache_integrity',integrity),('forward_skip_audit',skips),
            ('attention_backend_audit',backend),('endpoint_parity',endpoints),('source_costs',sources)]: atomic_json(a.output_dir/(name+'.json'),data)
        atomic_json(a.output_dir/'environment.json',environment)
        atomic_json(a.output_dir/'manifest.json',m)
        if a.phase in ('smoke','latency'):
            current=rows+(rows_new or [])
            csv_write(a.output_dir/'latency_raw_pairs.csv',current); atomic_json(a.output_dir/'latency_raw_pairs.json',current)
            csv_write(a.output_dir/'latency_session_summary.csv',summarize(current))
        if a.phase=='profile':
            for name,data in [('profiling_cpu_summary',cpu),('profiling_cuda_summary',cuda),('profiling_kernel_counts',counts),('profiling_components',components)]: csv_write(a.output_dir/(name+'.csv'),data)
            atomic_json(a.output_dir/'profiling_status.json',profiles)
    conditions=[]
    for arm in arms:
        runtime.configure(arm,a.endpoint_policy)
        for n,p in a.case_values:
            key=f'L{n}_P{p}_{arm}'; persist(dict(case=key,phase='source_prepare'))
            runtime.prepare(n,p)
            namespace=asdict(runtime.identity())
            identity=dict(independent_run_id=a.independent_run_id,case_id=f'L{n}_P{p}',attention_arm=arm,
                backend_requested='eager',backend_effective=runtime.backend['effective'],endpoint_policy=runtime.endpoint['effective'],
                total_length=n,prefix_length=p,suffix_length=n-p,source_snapshot_id=runtime.snapshot.identity_hash if runtime.snapshot else 'NO_PREFIX',
                target_token_hash=digest(runtime.ids),namespace_hash=digest(namespace),model_revision=runtime.meta['resolved_model_revision'],
                seed=a.seed,order_seed=a.order_seed)
            conditions.append({k:v for k,v in identity.items() if k not in ('independent_run_id','order_seed')})
            sources[key]=dict(costs=runtime.source_report,preflight=runtime.preflight,lookup_selection='NOT_PRESENT_EXACT_PREFIX_FIXTURE',transfer='NO_OFFLOAD_SAME_DEVICE',namespace=namespace)
            backend[key]=runtime.backend
            persist(dict(case=key,phase='structural_gate'))
            gates[key]=structural_gate(runtime)
            identity['structural_strict_pass']=gates[key]['quality']['strict']
            skips[key]={k:gates[key][k] for k in ('full','normal','cache_hit')}
            integrity[key]=gates[key]['canonical']; endpoints[key]=dict(policy=runtime.endpoint,outputs=gates[key]['output'],parity=gates[key]['native_endpoint_parity'])
            environment['monitoring'].append(dict(case=key,phase='before',state=monitor(runtime.device)))
            persist()
            if a.phase in ('smoke','latency'):
                def save(context=None,rows=None): persist(context,rows)
                new=run_pairs(runtime,a,identity,save); rows.extend(new)
            else:
                for _ in range(a.warmup_runs):
                    for mode in modes(runtime): invoke(runtime,mode)
                for mode in (*modes(runtime),'PREPARE_ONLY'):
                    persist(dict(case=key,phase='profile',mode=mode))
                    directory=a.output_dir/'profiler_trace'; directory.mkdir(exist_ok=True)
                    try: detail,cr,kr,br=collect(runtime,mode,directory/f'{key}_{mode}.json')
                    except ProfilerUnavailable as exc:
                        profiles[key+'_'+mode]=dict(status='PROFILER_UNAVAILABLE',reason=str(exc)); persist(); continue
                    profiles[key+'_'+mode]=detail
                    cpu.extend(dict(identity,mode=mode,**r) for r in cr)
                    cuda.extend(dict(identity,mode=mode,**r) for r in kr if r['gpu_activity'])
                    counts.extend(dict(identity,mode=mode,**r) for r in kr)
                    components.extend(dict(identity,mode=mode,**r) for r in br); persist()
            integrity[key]=runtime.check()
            environment['monitoring'].append(dict(case=key,phase='after',state=monitor(runtime.device)))
            persist()
    m['final_weight_check']=runtime.verify_weight_hashes()
    # Identical immutable comparison key required when aggregating processes.
    hw=runtime.provenance
    m['comparison_identity']=digest(dict(code=m['code']['sha256'],implementation=m['implementation_sha256'],
        conditions=sorted(conditions,key=lambda r:(r['case_id'],r['attention_arm'])),
        versions=m['versions'],gpu_name=hw.get('gpu_name'),device=runtime.device,
        gpu_uuid=environment['monitoring'][0]['state'].get('torch_uuid'),seed=a.seed,
        matmul_precision=hw.get('float32_matmul_precision'),tf32=hw.get('tf32')))
    m['qualification_eligible']=runtime.device.startswith('cuda') and bool(environment['monitoring'][0]['state'].get('torch_uuid') not in (None,'None'))
    strict=all(g['quality']['strict'] for g in gates.values()) and all(r['correctness_status']=='STRICT_PASS' for r in rows)
    m.update(execution_status='COMPLETE',implementation_status='EXECUTED_MODEL_RUNTIME',
        GPU_SMOKE_VALIDATED=a.phase=='smoke' and runtime.device.startswith('cuda'),
        correctness_status='STRICT_CORRECTNESS_PASS' if strict else 'STRICT_CORRECTNESS_FAIL',
        PERFORMANCE_MEASURED=bool(rows),PERFORMANCE_UNVERIFIED=not bool(rows),
        PROFILE_COLLECTED=any(p['status']=='PROFILE_COLLECTED' for p in profiles.values()))
    persist()
    atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],
        STRUCTURAL_REUSE_VALID=True,CACHE_INTEGRITY_VALID=True,FULL_RELATIVE_STRICT_CORRECTNESS=m['correctness_status'],
        NORMAL_PREPARED_PARITY='EXACT_WHERE_PREFIX_NONEMPTY',PERFORMANCE_MEASURED=m['PERFORMANCE_MEASURED'],
        PERFORMANCE_UNVERIFIED=m['PERFORMANCE_UNVERIFIED'],GPU_SMOKE_VALIDATED=m['GPU_SMOKE_VALIDATED'],
        PROFILE_COLLECTED=m['PROFILE_COLLECTED'],LATENCY_REPEATED=False,CORRECTNESS_QUALIFIED_SPEEDUP=False,
        hypotheses=dict(cpu_dispatch='UNRESOLVED',small_kernel_launch_cost='UNRESOLVED',output_materialization='MEASURED_SHAPES_NOT_CAUSAL_COST'),
        next_step='Aggregate >=3 independent latency sessions; inspect profile separately; no pooled-pair independent CI',
        limits='conditional warm-hit; source cost separate; Prepared ideal diagnostic only; strict fail preserved; no kernel causal dominance inferred'))


def main(argv=None):
    a=parse_args(argv); m=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt) as exc:
        m.update(execution_status='FAILED',failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc(),
            failure_code=getattr(exc,'code',None),
            PERFORMANCE_UNVERIFIED=True,CORRECTNESS_QUALIFIED_SPEEDUP=False)
        atomic_json(a.output_dir/'manifest.json',m)
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='FAILED',failure=m['failure'],
            traceback=m['traceback'],context=m.get('active_context'),PERFORMANCE_UNVERIFIED=True,CORRECTNESS_QUALIFIED_SPEEDUP=False))
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
