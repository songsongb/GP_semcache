#!/usr/bin/env python3
"""R2-B1.5 separate numeric, minimal paired timing and component diagnostics."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r2b1_prefix_runtime import PrefixRuntime
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r1b3_break_even import code_identity,schedule
from semcache.diagnostics.r2b15_analysis import REQUEST_MODES,resolve_layers,timing_summary,cost_decisions
from semcache.diagnostics.r2b15_cost import invoke,structural_gate,CostBook,prepare_only,monitoring
from semcache.diagnostics.r2b15_numeric import numeric_case
from semcache.diagnostics.r2b1_prefix_audit import compare_suffix
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write

CSV=('numeric_first_divergence','cache_kv_parity','timing_paired_samples','timing_summary','component_boundaries','projection_shape_probe')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','numeric','timing','breakdown'),default='smoke')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-125m')
    p.add_argument('--dtype',choices=('float16','float32'),default='float32'); p.add_argument('--device',default='cuda:0')
    p.add_argument('--cache-device',default=None); p.add_argument('--adapter',choices=('user_a','user_b'),default='user_a')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--allow-download',action='store_true')
    p.add_argument('--cases',default=None,help='32:16,32:31 or 128:64,128:127; small fixed B1.5 cases')
    p.add_argument('--layers',default='all',help='all or comma-separated indices including optional last; partial scope disclosed')
    p.add_argument('--reference-limit-mib',type=int,default=1024)
    p.add_argument('--shape-probe',choices=('auto','off'),default='auto')
    p.add_argument('--warmup-runs',type=int,default=3); p.add_argument('--measured-runs',type=int,default=None)
    p.add_argument('--instrumented-repeats',type=int,choices=(1,2,3),default=1)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--order-seed',type=int,default=701)
    p.add_argument('--independent-run-id',default='run01'); p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        a.case_values=[tuple(map(int,x.split(':'))) for x in (a.cases or ('32:16,32:31' if a.model.endswith('125m') else '128:64,128:127')).split(',')]
        if not 1<=len(a.case_values)<=2 or len(set(a.case_values))!=len(a.case_values) or any(x not in ((32,16),(32,31),(128,64),(128,127)) for x in a.case_values):
            raise ValueError('Use at most two supported B1.5 cases; P=0 native control remains in frozen runner85')
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (3 if a.phase=='smoke' else 10)
        if a.phase=='smoke' and (a.model!='facebook/opt-125m' or a.dtype!='float32'): raise ValueError('Smoke phase is OPT-125M FP32; use numeric/timing/breakdown for FP16 pilot')
        if a.warmup_runs<3 or a.measured_runs<3 or a.reference_limit_mib<1: raise ValueError('Invalid repeats/reference budget')
        if a.phase=='timing' and a.dtype=='float16' and a.measured_runs<10: raise ValueError('FP16 timing pilot requires >=10 measured repeats')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1' and a.phase in ('timing','smoke'): raise ValueError('CUDA_LAUNCH_BLOCKING invalidates minimal request timing')
    except ValueError as exc: p.error(str(exc))
    return a


def initialize(a):
    a.output_dir.mkdir(parents=True,exist_ok=False)
    files=['scripts/86_diagnose_r2b15_prefix.py',*[f'src/semcache/diagnostics/r2b15_{n}.py' for n in ('numeric','cost','analysis')],
        'tests/test_r2b15_numeric.py','tests/test_r2b15_cost.py','docs/R2B15_DIAGNOSTIC.md']
    def version(n):
        try: return importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: return None
    m=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),experiment='R2-B1.5 prefix numeric and warm cost isolation',classification='EXACT_PREFIX_BASELINE_DIAGNOSTIC',
        safe_semantic_reuse_claimed=False,correctness_qualified_speedup=False,implementation_status='IMPLEMENTED_LOCAL_ONLY',
        execution_status='RUNNING',correctness_status='UNVERIFIED',physical_prefix_skip='UNVERIFIED',
        code=code_identity(ROOT),implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files if (ROOT/p).exists()},
        specs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'docs').glob('R2-B1.5_*.md')},
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},python=platform.python_version(),
        packages={n:version(n) for n in ('torch','transformers','peft')},
        settings=dict(use_cache=True,output_attentions=True,return_dict=True,attention='eager',atol=1e-3,rtol=1e-3),
        boundaries=dict(normal='frozen B1 EXACT_PREFIX_WARM incl namespace+clones+container+input/positions+native forward',
            prepared='same B1 call; fresh clones/factory prepared outside, namespace/shape checks retained inside; IDEAL_DIAGNOSTIC',
            endpoint='pretokenized prompt -> GPU logits and output Cache; B1 closes before cache audit/CPU logit export/frame cleanup',
            excluded='source/canonical build, heavy integrity hashes, added hooks, CPU output copies',
            prepare_only='isolated materialize without forward; component clock, never request latency',
            breakdown='separate instrumented requests; nested CPU/CUDA clocks never summed with minimal'),
        environment={k:os.environ.get(k) for k in ('HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES')})
    for name in CSV: csv_write(a.output_dir/(name+'.csv'),[])
    for name in ('numeric_summary','cache_integrity','decision_summary'): atomic_json(a.output_dir/(name+'.json'),dict(status='UNVERIFIED'))
    atomic_json(a.output_dir/'gpu_state.json',dict(monitoring_status='MONITORING_UNAVAILABLE',reason='runtime not initialized'))
    atomic_json(a.output_dir/'manifest.json',m); return m


def no_hooks(runtime):
    if any(mod._forward_hooks or mod._forward_pre_hooks for mod in runtime.model.modules()):
        raise PrefixError('REJECT_MINIMAL_INSTRUMENTATION','module hooks remain before minimal timing')


def timing_case(runtime,a,identity,rows,save):
    import torch
    no_hooks(runtime)
    orders=schedule(REQUEST_MODES,a.warmup_runs,a.order_seed+runtime.n+runtime.p)+schedule(REQUEST_MODES,a.measured_runs,a.order_seed+runtime.n+runtime.p+100003)
    for index,order in enumerate(orders):
        phase='warmup' if index<a.warmup_runs else 'measured'; repeat=index if phase=='warmup' else index-a.warmup_runs
        results={}; pending=[]
        for mode in order:
            save(dict(identity,phase=phase,repeat=repeat,mode=mode))
            runtime.check(); result,ticket=invoke(runtime,mode); runtime.check()
            results[mode]=result
            pending.append(dict(identity,phase=phase,repeat=repeat,mode=mode,order=order,instrumentation='minimal',
                wall_ms=result['wall_ms'],cache_prepare_host_ms=result['cache_prepare_host_ms'],
                explicit_in_request_clone_bytes=runtime.snapshot.bytes if mode=='WARM_NORMAL' else 0,
                baseline_declared_clone_bytes=result['request_clone_bytes'],prepared=ticket,query_tokens=result['query_tokens'],
                output_kv_length=result['output_kv_length'],cache_api=result['cache_api'],memory=result['memory']))
        if not torch.equal(results['WARM_NORMAL']['logits'],results['WARM_PREPARED_DIAGNOSTIC']['logits']):
            raise PrefixError('REJECT_PREPARED_PARITY','normal/prepared timed outputs differ')
        quality=compare_suffix(results['FULL_USE_CACHE']['logits'],results['WARM_NORMAL']['logits'],runtime.p)
        for r in pending:
            r['quality']=quality; r['correctness_status']=quality['suffix']['correctness_status']
            r['full_minus_normal_ms']=results['FULL_USE_CACHE']['wall_ms']-results['WARM_NORMAL']['wall_ms']
            r['normal_minus_prepared_ms']=results['WARM_NORMAL']['wall_ms']-results['WARM_PREPARED_DIAGNOSTIC']['wall_ms']
            r['full_minus_prepared_ms']=results['FULL_USE_CACHE']['wall_ms']-results['WARM_PREPARED_DIAGNOSTIC']['wall_ms']
        rows.extend(pending); results.clear(); del result; save()
    return orders


def breakdown_case(runtime,a,identity):
    import torch
    from semcache.metrics.r1_runtime import synchronize,memory_snapshot
    rows=[]
    # Uninstrumented warmups only. No minimal statistics derived from this phase.
    for _ in range(a.warmup_runs):
        for mode in REQUEST_MODES: invoke(runtime,mode)
    for repeat in range(a.instrumented_repeats):
        reference=runtime.forward('EXACT_PREFIX_WARM')['logits']
        full_reference=runtime.forward('FULL_USE_CACHE')['logits']
        for mode in REQUEST_MODES:
            runtime.check(); no_hooks(runtime)
            if runtime.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(runtime.device)
            book=CostBook(runtime.device); start=perf_counter_ns()
            result,ticket=invoke(runtime,mode,book=book)
            synchronize(runtime.device); invocation=(perf_counter_ns()-start)/1e6
            expected=full_reference if mode=='FULL_USE_CACHE' else reference
            if not torch.equal(expected,result['logits']): raise PrefixError('REJECT_INSTRUMENTATION_PARITY','breakdown output changed: '+mode)
            common=dict(identity,repeat=repeat,mode=mode,instrumentation='breakdown',instrumented_request_wall_ms=result['wall_ms'],memory=memory_snapshot(runtime.device),prepared=ticket)
            rows.extend(dict(common,**part) for part in book.export())
            rows.append(dict(common,component='invocation_wall',host_ms=invocation,parent=None,status='MEASURED',
                nesting='inclusive wrapper incl setup/audits/CPU output; Prepared also includes excluded prebuild; never minimal request'))
            runtime.check(); del result
        book=CostBook(runtime.device); measured=prepare_only(runtime,book)
        # Same merge as request components above: export owns component metadata.
        # Both instrumentation fields mean breakdown (not separate clock types).
        common=dict(identity,repeat=repeat,mode='PREPARE_ONLY',instrumentation='breakdown',component_wall_ms=measured['wall_ms'])
        rows.extend(dict(common,**part) for part in book.export())
        rows.append(dict(identity,repeat=repeat,mode='PREPARE_ONLY',component='prepare_only_enclosing',host_ms=measured['wall_ms'],
            instrumentation='breakdown',status='MEASURED',parent=None,nesting=measured['scope'],native_forward_calls=0))
    return rows


def run(a,m):
    runtime=PrefixRuntime(a); m['runtime']=runtime.provenance; m['implementation_status']='EXECUTED_MODEL_RUNTIME'
    numeric=[]; kv=[]; probes=[]; minimal=[]; components=[]; numeric_summary={}; integrity={}; gpu=[]
    def save(context=None):
        if context is not None: m['active_case']=context
        for name,data in [('numeric_first_divergence',numeric),('cache_kv_parity',kv),('projection_shape_probe',probes),('timing_paired_samples',minimal),('component_boundaries',components)]:
            csv_write(a.output_dir/(name+'.csv'),data)
        summary=timing_summary(minimal); csv_write(a.output_dir/'timing_summary.csv',summary)
        atomic_json(a.output_dir/'numeric_summary.json',numeric_summary); atomic_json(a.output_dir/'cache_integrity.json',integrity)
        atomic_json(a.output_dir/'gpu_state.json',gpu); atomic_json(a.output_dir/'manifest.json',m)
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],numeric=numeric_summary,cost=cost_decisions(summary),
            safe_semantic_reuse_claimed=False,correctness_qualified_speedup=False,
            limitations='first observable boundaries only; no kernel identity; Prepared is IDEAL_DIAGNOSTIC; one-run timings exploratory',
            next_step='B2 requires structural/parity gates, strict quality disclosure, normal clone costs, same cache/output endpoint, source-cost accounting and independent runs'))
    for n,p in a.case_values:
        key=f'L{n}_P{p}'; save(dict(case=key,phase='source_prepare')); runtime.prepare(n,p)
        identity=dict(case=key,total_length=n,prefix_length=p,suffix_length=n-p,model=a.model,dtype=a.dtype,adapter=a.adapter,
            code_identity=m['code']['sha256'],model_revision=runtime.meta['resolved_model_revision'],target_identity=digest(runtime.ids),
            source_identity=runtime.snapshot.identity_hash,namespace=asdict(runtime.identity()),seed=a.seed,order_seed=a.order_seed,independent_run_id=a.independent_run_id)
        m.setdefault('cases',{})[key]=dict(identity,source_costs=runtime.source_report,preflight=runtime.preflight,token_ids=runtime.ids)
        save(dict(case=key,phase='structural_gate')); integrity[key]=structural_gate(runtime); save()
        gpu.append(dict(case=key,phase='before_diagnostics',state=monitoring(runtime)))
        if a.phase in ('smoke','numeric'):
            save(dict(case=key,phase='numeric')); layers=resolve_layers(a.layers,len(runtime.adapter.layers))
            nr,kr,pr,ns,audit=numeric_case(runtime,layers,a.reference_limit_mib,a.shape_probe=='auto',
                progress=lambda data:csv_write(a.output_dir/'numeric_first_divergence.csv',numeric+[dict(identity,**r) for r in data]))
            numeric.extend(dict(identity,**r) for r in nr); kv.extend(dict(identity,**r) for r in kr); probes.extend(dict(identity,**r) for r in pr)
            numeric_summary[key]=dict(identity,**ns); integrity[key]['numeric_audits']=audit; save()
        if a.phase in ('smoke','timing'):
            save(dict(case=key,phase='timing')); m['cases'][key]['orders']=timing_case(runtime,a,identity,minimal,save); save()
        if a.phase in ('smoke','breakdown'):
            save(dict(case=key,phase='breakdown')); components.extend(breakdown_case(runtime,a,identity)); save()
        integrity[key]['final_canonical']=runtime.check(); integrity[key]['model_weights']=runtime.verify_weight_hashes()
        gpu.append(dict(case=key,phase='after_diagnostics',state=monitoring(runtime))); save()
    strict_fail=(any(not v['quality']['suffix']['allclose'] for v in integrity.values()) or any(r['correctness_status']=='STRICT_FAIL' for r in minimal)
        or any(not q['allclose'] for ns in numeric_summary.values() for q in ns['quality'].values()))
    m.update(execution_status='COMPLETE',physical_prefix_skip='PHYSICAL_PREFIX_SKIP_VERIFIED',correctness_status='STRICT_FAIL' if strict_fail else 'STRICT_PASS')
    save()


def main(argv=None):
    a=parse_args(argv); m=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt,SystemExit) as exc:
        status='OOM' if 'out of memory' in str(exc).lower() else 'MEMORY_PREFLIGHT_FAILED' if isinstance(exc,MemoryError) else 'UNSUPPORTED' if isinstance(exc,PrefixError) and exc.code.startswith('UNSUPPORTED') else 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        m.update(execution_status=status,failure_type=type(exc).__name__,failure=str(exc),failure_code=getattr(exc,'code',None),traceback=traceback.format_exc())
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'failure.json',dict(status=status,type=type(exc).__name__,reason=str(exc),context=m.get('active_case'),traceback=m['traceback']))
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=status,correctness_status='UNVERIFIED',physical_prefix_skip='UNVERIFIED',correctness_qualified_speedup=False,safe_semantic_reuse_claimed=False))
        traceback.print_exc(); return 1
if __name__=='__main__': raise SystemExit(main())
