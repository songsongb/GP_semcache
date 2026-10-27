#!/usr/bin/env python3
"""R2-A5: static/offline boundary audit and opt-in immutable Oracle validation A/B."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_b3_runtime import common_arguments,config
from semcache.diagnostics.r1b3_break_even import create_run,failure,schedule,MissingEvidence
from semcache.diagnostics.r2a_runtime import CostRuntime
from semcache.diagnostics.r2a_overhead_audit import boundaries,source_contract,ImmutableFixture,compare_execution,audit_trace
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
from semcache.edgelora.batched_mapping_optimized import paired_statistics
from semcache.metrics.r1 import atomic_json,raw_only_guard,digest
from semcache.metrics.r1b2 import csv_write,stats

MODES=('ORACLE_VALIDATED','ORACLE_PRIOR_VALIDATED')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); common_arguments(p)
    p.add_argument('--phase',choices=('static','paired'),default='static')
    p.add_argument('--allow-validation-bypass',action='store_true',help='Controlled serial fixture only; no production safety claim')
    p.add_argument('--warmup-runs',type=int,default=3)
    p.add_argument('--measured-runs',type=int,default=10)
    p.add_argument('--order-seed',type=int,default=501)
    p.add_argument('--trace',type=Path,action='append',default=[],help='Existing Chrome JSON trace to audit offline; repeatable; never profiles a model')
    a=p.parse_args(argv); a.lengths=[int(x) for x in a.prompt_lengths.split(',')]
    if (not a.lengths or min(a.lengths)<4 or len(set(a.lengths))!=len(a.lengths) or a.warmup_runs<3 or a.measured_runs<2 or a.capacity_mib<1):
        p.error('Require distinct lengths >=4, warmup >=3, measured >=2, positive capacity')
    if a.phase=='paired' and not a.allow_validation_bypass: p.error('Paired intervention requires explicit --allow-validation-bypass')
    if a.cache_device=='cuda' and not a.device.startswith('cuda'): p.error('CUDA cache requires CUDA model')
    import os
    if a.phase=='paired' and os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': p.error('CUDA_LAUNCH_BLOCKING=1 invalidates paired timing')
    return a


def summary_rows(rows):
    result=[]
    for length in sorted({r['length'] for r in rows}):
        groups={mode:[r for r in rows if r['length']==length and r['phase']=='measured' and r['mode']==mode] for mode in MODES}
        if not any(groups.values()): continue
        a,b=groups.values()
        if [r['repeat'] for r in a]!=[r['repeat'] for r in b] or len({r['repeat'] for r in a})!=len(a): raise MissingEvidence('Incomplete/duplicate A5 measured pairs')
        pair=paired_statistics([r['wall_ms'] for r in a],[r['wall_ms'] for r in b],a[0]['order_seed'])
        ci=pair['bootstrap_mean_ci95']
        result.append(dict(length=length,validated=stats([r['wall_ms'] for r in a]),prior_validated=stats([r['wall_ms'] for r in b]),
            difference_definition='D=validated-prior_validated; positive means bypass request faster',paired=pair,
            status='SUPPORTED' if ci and ci[0]>0 else 'CONTRADICTED' if ci and ci[1]<0 else 'UNRESOLVED',
            interpretation='fixture-specific causal effect of one repeated descriptor.validate call; small dispatch guard retained',
            inference='one run, within-run bootstrap; no outlier removal; no production-safe or Full-relative speedup claim'))
    return result


def paired(a,m):
    runtime=CostRuntime(a); m['runtime']=runtime.provenance
    samples=[]; cases={}; validation={}; gpu=[]
    def write():
        csv_write(a.output_dir/'paired_samples.csv',samples)
        atomic_json(a.output_dir/'correctness_and_fixture.json',validation)
        atomic_json(a.output_dir/'gpu_state.json',gpu)
        atomic_json(a.output_dir/'manifest.json',m)
    for length in a.lengths:
        m['active_case']=dict(length=length,phase='source_prepare'); write()
        runtime.prepare(length)
        lease=ImmutableFixture(runtime)
        m.setdefault('cases',{})[str(length)]=dict(token_ids=runtime.ids,target_identity=digest(runtime.ids),
            snapshot=runtime.snapshot,source_costs=runtime.costs,preflight=runtime.preflight,
            source_backing=runtime.backing_check,fixture_prevalidation_setup_ms=getattr(lease,'setup_ms',None),model_revision=runtime.meta['resolved_model_revision'],
            model_namespace=runtime.namespace,weight_identity=runtime.weights,adapter='user_a')
        # Capture outside timing. Both conditions use exactly CostRuntime.invoke.
        m['active_case']=dict(length=length,phase='untimed_qkv_parity'); write()
        first,reference=runtime.invoke('ORACLE_PRESELECTED',capture=True)
        with lease.bypass(): actual,e=runtime.invoke('ORACLE_PRESELECTED',capture=True)
        parity=compare_execution(first,actual,reference,e)
        full=runtime.invoke('FULL_NATIVE')[0]
        validation[str(length)]=dict(parity=parity,full_relative=logit_metrics(full,actual),fixture_status='PENDING_POSTCHECK')
        del first,actual,full,reference,e
        order=schedule(MODES,a.warmup_runs,a.order_seed+length)+schedule(MODES,a.measured_runs,a.order_seed+length+100003)
        m['cases'][str(length)]['order']=order
        gpu.append(dict(length=length,phase='before_paired',state=runtime.gpu_state())); write()
        for i,modes in enumerate(order):
            phase='warmup' if i<a.warmup_runs else 'measured'; repeat=i if phase=='warmup' else i-a.warmup_runs
            outputs={}; records={}
            for mode in modes:
                m['active_case']=dict(length=length,phase=phase,repeat=repeat,mode=mode); write()
                # Each condition performs one prior and one post full check.
                if mode=='ORACLE_PRIOR_VALIDATED':
                    with lease.bypass(): outputs[mode],records[mode]=runtime.invoke('ORACLE_PRESELECTED')
                else:
                    lease.check()
                    try: outputs[mode],records[mode]=runtime.invoke('ORACLE_PRESELECTED')
                    finally: lease.check()
            compare_execution(outputs[MODES[0]],outputs[MODES[1]],records[MODES[0]],records[MODES[1]])
            for mode in modes:
                r=records[mode]
                samples.append(dict(length=length,phase=phase,repeat=repeat,mode=mode,order=modes,order_seed=a.order_seed,
                    wall_ms=r['request_ms'],reused_tokens=r['reused_tokens'],projection_audit=r['projection_audit'],
                    selection_hash=r['selection_hash'],profiler_enabled=False,precision=a.dtype,
                    source_snapshot=runtime.snapshot,target_identity=digest(runtime.ids),code_identity=m['code']['sha256'],
                    model_revision=runtime.meta['resolved_model_revision'],weight_identity=runtime.weights,
                    independent_run_id=a.independent_run_id,correctness_status=validation[str(length)]['full_relative']['correctness_status'],
                    execution_parity='EXACT',fixture_status='PENDING_POSTCHECK'))
            write()
        validation[str(length)]['fixture']=lease.close()
        validation[str(length)]['fixture_status']='VERIFIED_CONTROLLED_FIXTURE'
        for r in samples:
            if r['length']==length: r['fixture_status']='VERIFIED_CONTROLLED_FIXTURE'
        gpu.append(dict(length=length,phase='after_paired',state=runtime.gpu_state()))
        cases[str(length)]=summary_rows([r for r in samples if r['length']==length])[0]
        atomic_json(a.output_dir/'controlled_difference.json',cases); write()
    m['correctness_status']='STRICT_FAIL' if any(v['full_relative']['correctness_status']=='STRICT_FAIL' for v in validation.values()) else 'STRICT_PASS'
    return cases


def main(argv=None):
    a=parse_args(argv); cfg=config(a); cfg['trace']=[str(p) for p in a.trace]
    m=create_run(ROOT,a.output_dir,cfg,'R2-A5 final overhead audit')
    m.update(safe_reuse_claimed=False,fixture_status='UNVERIFIED',instrumentation='minimal only; offline trace analysis separate',
        boundary_contract=None,bypass_scope='one instance SourceBackingDescriptors.validate at executor entry ONLY',
        bypass_qualification='prior-validated serial immutable fixture; not production-safe; repeated validation removed only in explicit A5 condition',
        checks_outside_timer='full descriptor checks and identity stamps before/after requests; backing/index and all model-weight content hashes at block boundaries',
        model_hooks='no new profiling hooks',results_are_new_gpu_measurements=False)
    table=boundaries(ROOT)
    csv_write(a.output_dir/'boundary_audit.csv',table); csv_write(a.output_dir/'paired_samples.csv',[])
    atomic_json(a.output_dir/'correctness_and_fixture.json',dict(status='UNVERIFIED'))
    atomic_json(a.output_dir/'controlled_difference.json',dict(status='UNVERIFIED',reason='No controlled A/B executed'))
    atomic_json(a.output_dir/'trace_attribution_audit.json',dict(status='UNVERIFIED',reason='No original traces supplied'))
    try:
        m['boundary_contract']=source_contract(ROOT)
        traces=[]
        for path in a.trace:
            traces.append(dict(path=str(path),sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(),audit=audit_trace(json.loads(path.read_text()))))
        if traces: atomic_json(a.output_dir/'trace_attribution_audit.json',traces)
        with raw_only_guard() as guard:
            try: differences=paired(a,m) if a.phase=='paired' else {}
            finally: m['raw_only']=guard.evidence()
        m.update(execution_status='COMPLETE',results_are_new_gpu_measurements=a.phase=='paired' and a.device.startswith('cuda'),fixture_status='VERIFIED_CONTROLLED_FIXTURE' if a.phase=='paired' else 'UNVERIFIED')
        if a.phase=='paired':
            for length,delta in differences.items():
                table.append(dict(component='repeated_executor_snapshot_validation',exact_measured_boundary=m['bypass_scope'],nesting='paired request-level intervention',
                    status=delta['status'],evidence_kind='NEW_CONTROLLED_PAIRED_RUN',length=length,
                    measured_difference_ms=delta['paired']['mean'],uncertainty=delta['inference'],interpretation=delta['interpretation']))
        csv_write(a.output_dir/'boundary_audit.csv',table)
        atomic_json(a.output_dir/'manifest.json',m)
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='COMPLETE',static_conclusions=table,controlled=differences,
            trace_evidence='AUDITED' if traces else 'UNVERIFIED_NO_ORIGINAL_TRACE',safe_reuse_claimed=False,
            R2A_final='Mapping/validation microbench overlap confirmed; closest-tag attention reduction does not establish skipped attention. GPU causal cost pending paired evidence.',
            R2B_required='Measure native Full versus exact-prefix KV prefill with identical model/adapter/position and strict logits; account for source cost, prefix length, cold/warm KV residency and all remaining FFN/attention work. No arbitrary semantic-window safety claim.'))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        code=failure(a.output_dir,m,exc); m['fixture_status']='UNVERIFIED'
        atomic_json(a.output_dir/'manifest.json',m)
        atomic_json(a.output_dir/'failure.json',dict(type=type(exc).__name__,reason=str(exc),context=m.get('active_case'),traceback=m.get('traceback')))
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],fixture_status='UNVERIFIED',controlled_conclusion='INVALID_OR_INCOMPLETE',safe_reuse_claimed=False))
        return code
if __name__=='__main__': raise SystemExit(main())
