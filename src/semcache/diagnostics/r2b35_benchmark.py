"""Same-session four-path comparisons; structural hooks only in separate gates."""
from semcache.diagnostics.r2b35_executor import MODES
from semcache.diagnostics.r2b35_endpoint_audit import metrics
from semcache.diagnostics.r1b3_break_even import schedule
from semcache.diagnostics.r2b2_attention_ablation import backend_audit
from semcache.diagnostics.r2b1_prefix_cache import PrefixError

PRIMARY=[(128,127),(512,511),(1024,1023),(128,64),(512,448),(1024,960)]
PILOT=[(128,127),(512,511),(1024,960),(1024,1023)]


def comparisons(results,prefix):
    lf,lw,mf,mw=[results[m] for m in MODES]
    a=lf['wall_ms']-lw['wall_ms']; b=mf['wall_ms']-mw['wall_ms']
    return dict(delta_legacy_ms=a,delta_matched_ms=b,delta_difference_ms=a-b,
        full_head_policy_effect_ms=lf['wall_ms']-mf['wall_ms'],warm_head_policy_effect_ms=lw['wall_ms']-mw['wall_ms'],
        matched_speedup=mf['wall_ms']/mw['wall_ms'],legacy_speedup=lf['wall_ms']/lw['wall_ms'])


def pair_quality(results,prefix):
    from semcache.diagnostics.r2b2_benchmark import exact_cache
    axis_a={}
    for kind in ('FULL','WARM_NORMAL'):
        a,b=results['LEGACY_'+kind],results['MATCHED_'+kind]
        if not exact_cache(a['cache'],b['cache']): raise PrefixError('OUTPUT_MATCHED_INVALID','paired native/matched KV mismatch')
        axis_a[kind]=metrics(a['logits'][:,-1:,:],b['logits'])
    return dict(axis_a=axis_a,matched=metrics(results['MATCHED_FULL']['logits'],results['MATCHED_WARM_NORMAL']['logits']),
        legacy_suffix=metrics(results['LEGACY_FULL']['logits'][:,prefix:,:],results['LEGACY_WARM_NORMAL']['logits']))


def run_pairs(runtime,args,identity,save):
    if any(m._forward_hooks or m._forward_pre_hooks for m in runtime.model.modules()):
        raise PrefixError('REJECT_MINIMAL_INSTRUMENTATION','hooks remain before minimal request')
    def binding(): return (tuple(runtime.ids),runtime.n,runtime.p,id(runtime.snapshot),runtime.snapshot.identity_hash,getattr(runtime,'source_forwards',None))
    expected=binding(); rows=[]
    for phase,count,offset in [('warmup',args.warmup_runs,0),('measured',args.measured_runs,100003)]:
        for index,order in enumerate(schedule(MODES,count,args.order_seed+runtime.n+runtime.p+offset)):
            results={}
            for mode in order:
                save(context=dict(case=identity['case_id'],arm=runtime.arm,phase=phase,pair_index=index,mode=mode))
                runtime.check(); backend_audit(runtime)
                if binding()!=expected: raise PrefixError('REJECT_PAIRED_IDENTITY','source/target changed')
                results[mode]=runtime.execute(mode)
                runtime.check()
                if binding()!=expected: raise PrefixError('REJECT_PAIRED_IDENTITY','source/target changed during request')
            quality=pair_quality(results,runtime.p)
            strict=quality['matched']['allclose'] and quality['legacy_suffix']['allclose'] and identity['gate_full_warm_strict']
            equivalent=all(q['allclose'] for q in quality['axis_a'].values()) and identity['gate_native_equivalent']
            row=dict(identity,phase=phase,pair_index=index,execution_order=list(order),instrumentation='minimal',benchmark_phase=args.phase,
                **comparisons(results,runtime.p),
                wall_ms={m:r['wall_ms'] for m,r in results.items()},cache_prepare_host_ms={m:r['cache_prepare_host_ms'] for m,r in results.items()},
                clone_bytes={m:r['request_clone_bytes'] for m,r in results.items()},
                logit_shapes={m:r['logit_shape'] for m,r in results.items()},memory={m:r['memory'] for m,r in results.items()},
                native_endpoint_equivalent=equivalent,full_warm_strict=strict,quality=quality,
                output_matching='OUTPUT_MATCHED_VALID',structural_valid=True,cache_integrity_valid=True,
                correctness_qualified_speedup=False)
            rows.append(row); results.clear(); save(rows=rows)
    return rows


def profile(runtime,mode,path):
    """Optional isolated trace. Never enters minimal statistics."""
    from contextlib import ExitStack
    import json
    from unittest.mock import patch
    import torch
    from semcache.diagnostics.r2b2_profiler import require_profiler,ProfileBook,cpu_costs,module_ranges,trace_rows
    from semcache.diagnostics.r2b15_cost import instrument
    from semcache.diagnostics.r2b2_benchmark import exact_cache
    from semcache.metrics.r1_runtime import synchronize
    p,activities=require_profiler(runtime.device)
    reference=runtime.execute(mode); book=ProfileBook(runtime.device)
    with p.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as prof:
        with ExitStack() as stack:
            stack.enter_context(instrument(runtime,book)); stack.enter_context(module_ranges(runtime))
            for owner,label in [(runtime.matched_decoder,'decoder'),(runtime.matched_head,'actual_lm_head')]:
                original=owner.forward
                def call(*args,_fn=original,_label=label,**kwargs):
                    with p.record_function('r2b2::'+_label): return _fn(*args,**kwargs)
                stack.enter_context(patch.object(owner,'forward',call))
            result=runtime.execute(mode); synchronize(runtime.device)
    if not torch.equal(reference['logits'],result['logits']) or not exact_cache(reference['cache'],result['cache']):
        raise PrefixError('PROFILE_PARITY_FAILED','profiling changes result')
    prof.export_chrome_trace(str(path)); trace=json.loads(path.read_text())
    adapted=dict(trace,traceEvents=[dict(e,name=e.get('name','').replace('r2b2::','r2a::',1)) for e in trace.get('traceEvents',[])])
    kernels,definitions=trace_rows(adapted)
    return dict(status='PROFILE_COLLECTED',instrumented_wall_ms=result['wall_ms'],reference_wall_ms=reference['wall_ms'],**definitions,
        limitation='separate instrumented request; nested clocks not summed; trace includes outside-timer audits/output copies'),cpu_costs(prof.events()),kernels,book.export()
