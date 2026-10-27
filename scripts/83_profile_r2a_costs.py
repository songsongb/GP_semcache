#!/usr/bin/env python3
"""R2-A frozen-executor profiling: minimal, trace and component clocks stay separate."""
import argparse
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_b3_runtime import common_arguments,config
from semcache.diagnostics.r1b3_break_even import create_run,failure,ParityFailure
from semcache.diagnostics.r2a_analysis import MODES,schedule,summaries,decisions
from semcache.diagnostics.r2a_runtime import CostRuntime,SEMANTICS,microbench
from semcache.diagnostics.r2a_profiler import profile_request,ProfilerUnavailable
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
from semcache.metrics.r1 import atomic_json,raw_only_guard,digest
from semcache.metrics.r1b2 import csv_write

CSV_FILES=('minimal_paired_samples','minimal_summary','profiler_operator_cost','profiler_gpu_kernel_cost','component_microbench','overhead_bridge')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); common_arguments(p)
    p.add_argument('--phase',choices=('smoke','minimal','trace','microbench'),default='smoke')
    p.add_argument('--warmup-runs',type=int,default=3)
    p.add_argument('--measured-runs',type=int,default=None)
    p.add_argument('--order-seed',type=int,default=101)
    p.add_argument('--trace-repeats',type=int,choices=(1,2,3),default=1)
    p.add_argument('--max-trace-mib',type=int,default=64)
    p.add_argument('--with-stack',action='store_true')
    p.add_argument('--microbench-repeats',type=int,default=10)
    a=p.parse_args(argv); a.lengths=[int(x) for x in a.prompt_lengths.split(',')]
    a.measured_runs=a.measured_runs if a.measured_runs is not None else (3 if a.phase=='smoke' else 10)
    if (not a.lengths or min(a.lengths)<4 or len(set(a.lengths))!=len(a.lengths) or a.warmup_runs<3 or
        a.measured_runs<(2 if a.phase=='smoke' else 10) or a.max_trace_mib<1 or a.microbench_repeats<1 or a.capacity_mib<1): p.error('Invalid counts/lengths; pilot >=3 warmups and >=10 measured')
    if a.cache_device=='cuda' and not a.device.startswith('cuda'): p.error('CUDA cache requires CUDA model')
    if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': p.error('CUDA_LAUNCH_BLOCKING=1 invalidates main timing; use separate debugging')
    return a


def initialize(a):
    m=create_run(ROOT,a.output_dir,config(a),'R2-A GPU cost diagnosis')
    m.update(safe_reuse_claimed=False,historical_r1_numbers_are_new_measurements=False,mode_semantics=SEMANTICS,
        timing=dict(primary='existing synchronized R1 request_wall_ms; Oracle includes setup/cleanup; excludes logits CPU transfer',
            wrapper_harness='R1 engine clone and post-request audit excluded as in 80; independently profiled but not called request wall',
            minimal='torch.profiler and added hooks disabled',trace='instrumented timings never pooled with minimal'),
        versions={name:version(name) for name in ('torch','transformers','peft')},python=platform.python_version(),
        environment={k:os.environ.get(k) for k in ('HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')},phases={})
    (a.output_dir/'profiler_trace').mkdir()
    for name in CSV_FILES: csv_write(a.output_dir/(name+'.csv'),[])
    for name in ('correctness_and_coverage','decision_summary'): atomic_json(a.output_dir/(name+'.json'),dict(status='UNVERIFIED'))
    atomic_json(a.output_dir/'manifest.json',m); return m


def version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def run(a,m):
    import torch
    rt=CostRuntime(a); m['runtime']=rt.provenance
    try:
        driver=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True,timeout=5,check=True)
        m['driver_inventory']=driver.stdout.strip().splitlines()
    except Exception as exc: m['driver_inventory']=dict(value=None,reason=str(exc))
    rows=[]; operators=[]; kernels=[]; micros=[]; validation={}; profiles=[]; gpu=[]
    def save():
        csv_write(a.output_dir/'minimal_paired_samples.csv',rows)
        summary,bridge=summaries(rows,a.order_seed)
        csv_write(a.output_dir/'minimal_summary.csv',summary); csv_write(a.output_dir/'overhead_bridge.csv',bridge)
        csv_write(a.output_dir/'profiler_operator_cost.csv',operators); csv_write(a.output_dir/'profiler_gpu_kernel_cost.csv',kernels)
        csv_write(a.output_dir/'component_microbench.csv',micros)
        atomic_json(a.output_dir/'correctness_and_coverage.json',validation)
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=m['execution_status'],minimal=decisions(bridge),
            profile_evidence=profiles,safe_reuse_claimed=False,historical_reference_reproduced=False,
            H_A_B_C='Review attributed component/trace tables; no automatic inference from overlapping clocks',
            next_step='R2-D if controlled host/executor removal closes the gap; investigate R2-B if QKV-only oracle still slower, without claiming prefix safety'))
        atomic_json(a.output_dir/'gpu_state.json',gpu); atomic_json(a.output_dir/'manifest.json',m)
    for length in a.lengths:
        m['active_case']=dict(length=length,phase='source_build'); atomic_json(a.output_dir/'manifest.json',m)
        rt.prepare(length)
        identity=dict(length=length,model=a.model,dtype=a.dtype,cache_device=a.cache_device,seed=a.seed,order_seed=a.order_seed,
            independent_run_id=a.independent_run_id,source_snapshot=rt.snapshot,target_identity=digest(rt.ids),
            model_revision=rt.meta['resolved_model_revision'],weight_identity=rt.weights,adapter='user_a',
            code_identity=m['code']['sha256'],hardware_identity=rt.provenance['gpu'].get('gpu_uuid'))
        m.setdefault('cases',{})[str(length)]=dict(identity,token_ids=rt.ids,source_costs=rt.costs,memory_preflight=rt.preflight,
            cache_bytes=rt.catalog.resident_bytes,oracle_preselected=True,source_capture_backing=rt.backing_check)
        orders=schedule(MODES,a.warmup_runs,a.order_seed+length)+schedule(MODES,a.measured_runs,a.order_seed+length+100003)
        m['cases'][str(length)]['execution_orders']=orders
        gpu.append(dict(length=length,phase='before_minimal',state=rt.gpu_state()))
        # Every minimal block finishes before profiler or component instrumentation starts.
        for iteration,order in enumerate(orders):
            phase='warmup' if iteration<a.warmup_runs else 'measured'
            repeat=iteration if phase=='warmup' else iteration-a.warmup_runs
            outputs={}; records={}
            for mode in order:
                m['active_case']=dict(length=length,phase=phase,repeat=repeat,mode=mode,instrumentation='minimal')
                atomic_json(a.output_dir/'manifest.json',m)
                outputs[mode],e=rt.invoke(mode); records[mode]=e
                row=dict(identity,phase=phase,repeat=repeat,mode=mode,execution_order=order,wall_ms=e['request_ms'],
                    cuda_event_ms=None,profiler_enabled=False,reused_tokens=e['reused_tokens'],actual_coverage=e['reused_tokens']/length,
                    projection_audit=e['projection_audit'],mapping_cache_hit=e.get('mapping_cache_hit'),
                    mapping_ms=e.get('mapping_ms'),selection_ms=e.get('selection_ms'),context_setup_ms=e.get('context_setup_ms'),memory=e.get('memory'),
                    strict_status='PENDING_UNTIMED_CHECK')
                rows.append(row); csv_write(a.output_dir/'minimal_paired_samples.csv',rows)
            for r in rows[-len(MODES):]:
                q=logit_metrics(outputs['FULL_NATIVE'],outputs[r['mode']]); r['strict_status']=q['correctness_status']; r['quality']=q
            if not torch.equal(outputs['CONTIGUOUS_WARM'],outputs['ORACLE_PRESELECTED']):
                csv_write(a.output_dir/'minimal_paired_samples.csv',rows); raise ParityFailure('Contiguous/Oracle logits differ in minimal pair')
            if (records['CONTIGUOUS_WARM']['projection_audit']!=records['ORACLE_PRESELECTED']['projection_audit'] or records['CONTIGUOUS_WARM']['selection_hash']!=records['ORACLE_PRESELECTED']['selection_hash']): raise ParityFailure('Physical row audit differs')
            if phase=='measured' and records['CONTIGUOUS_WARM']['reused_tokens'] and records['CONTIGUOUS_WARM']['mapping_cache_hit'] is not True: raise ParityFailure('Warm mapping expected a real cache hit')
            outputs.clear(); save()
        gpu.append(dict(length=length,phase='after_minimal',state=rt.gpu_state()))
        m['active_case']=dict(length=length,phase='untimed_validation'); validation[str(length)]=rt.validate_modes(); save()
        if a.phase in ('trace','smoke'):
            for mode in MODES:
                ref,reference=rt.invoke(mode)
                for repeat in range(a.trace_repeats):
                    m['active_case']=dict(length=length,phase='trace',mode=mode,repeat=repeat); save()
                    path=a.output_dir/'profiler_trace'/f'tokens_{length}_{mode}_{repeat}.json'
                    try: output,record,ops,raw,meta=profile_request(rt,mode,path,a.max_trace_mib,a.with_stack)
                    except ProfilerUnavailable as exc:
                        m['phases']['trace']=dict(status='UNSUPPORTED',reason=str(exc)); save(); raise
                    parity=torch.equal(ref,output) and reference['projection_audit']==record['projection_audit']
                    operators.extend(dict(identity,mode=mode,repeat=repeat,profiler_enabled=True,**r) for r in ops)
                    kernels.extend(dict(identity,mode=mode,repeat=repeat,profiler_enabled=True,**r) for r in raw)
                    baseline=[r['wall_ms'] for r in rows if r['length']==length and r['mode']==mode and r['phase']=='measured']
                    profiles.append(dict(length=length,mode=mode,repeat=repeat,profiler_request_ms=record['request_ms'],
                        minimal_mean_ms=sum(baseline)/len(baseline),instrumentation_overhead_ms=record['request_ms']-sum(baseline)/len(baseline),
                        instrumentation_parity=parity,scope='descriptive sequential-phase overhead, not a paired causal timing estimate',**meta))
                    save()
                    if not parity: raise ParityFailure('Profiler instrumentation changed logits or physical rows')
            m['phases']['trace']=dict(status='COMPLETE')
        if a.phase in ('microbench','smoke'):
            m['active_case']=dict(length=length,phase='microbench'); save()
            micros.extend(dict(identity,**r) for r in microbench(rt,a.microbench_repeats)); save()
            m['phases']['microbench']=dict(status='COMPLETE')
        gpu.append(dict(length=length,phase='after_diagnostics',state=rt.gpu_state()))
    m.update(execution_status='COMPLETE',correctness_status='STRICT_FAIL' if any(r['strict_status']=='STRICT_FAIL' for r in rows) else 'STRICT_PASS')
    m['phases']['minimal']=dict(status='COMPLETE'); save()


def main(argv=None):
    a=parse_args(argv); m=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt) as exc:
        code=failure(a.output_dir,m,exc)
        import json
        decision=json.loads((a.output_dir/'decision_summary.json').read_text())
        decision.update(execution_status=m['execution_status'],failure=str(exc),performance_conclusion='INVALID_OR_INCOMPLETE_RUN')
        atomic_json(a.output_dir/'decision_summary.json',decision)
        atomic_json(a.output_dir/'failure.json',dict(status=m['execution_status'],type=type(exc).__name__,reason=str(exc),context=m.get('active_case'),traceback=m.get('traceback')))
        return code
if __name__=='__main__': raise SystemExit(main())
