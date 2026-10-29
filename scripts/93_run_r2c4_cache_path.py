#!/usr/bin/env python3
"""R2-C4 controlled RAW cache-path oracles. No production configuration changes."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sys
import traceback
from time import perf_counter_ns
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write
from semcache.diagnostics.r1b3_break_even import code_identity,MissingEvidence
from semcache.diagnostics.r2c1_workloads import lengths,coverages,memory_estimate
from semcache.diagnostics.r2c2_profile_memory import PilotPeakRecorder,failure_status,uuid
from semcache.diagnostics.r2b3_capacity import memory,failure_record
from semcache.diagnostics.r2c1_benchmark import validate as physical_gate
from semcache.diagnostics.r2c3_benchmark import capture_references,validate_replay
from semcache.diagnostics.r2c3_impact_replay import BASES
from semcache.diagnostics.r2c4_cache_path import ARMS,RAW
from semcache.diagnostics.r2c4_benchmark import Runtime,BOUNDARIES,gate,paired
from semcache.diagnostics.r2c4_breakdown import collect as collect_breakdown
from semcache.diagnostics.r2c4_analysis import summarize,csv_summary,aggregate,approve_pilot


def helpers():
    spec=importlib.util.spec_from_file_location('c4_frozen_c3_runner',ROOT/'scripts/92_run_r2c3_impact_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('pilot','breakdown','latency','analyze'),required=True)
    p.add_argument('--model',choices=('facebook/opt-2.7b','facebook/opt-125m'))
    p.add_argument('--dtype',choices=('float16','float32'))
    p.add_argument('--prompt-lengths');p.add_argument('--coverages',default='0,.5,max')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cache-device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--capacity-mib',type=int,default=512)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--order-seed',type=int,default=4101)
    p.add_argument('--independent-run-id',default='run01')
    p.add_argument('--warmup-runs',type=int);p.add_argument('--measured-runs',type=int)
    p.add_argument('--pilot-inputs',type=Path,nargs='+')
    p.add_argument('--inputs',type=Path,nargs=3)
    p.add_argument('--breakdown-input',type=Path)
    p.add_argument('--revision');p.add_argument('--tokenizer-revision');p.add_argument('--encoder-revision',default='main')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.phase=='analyze':
            if not a.inputs or not a.breakdown_input or any(v is not None for v in (a.model,a.dtype,a.prompt_lengths,a.pilot_inputs,a.warmup_runs,a.measured_runs)):
                raise ValueError('analysis requires exactly three --inputs and completed --breakdown-input; configuration comes from inputs')
            return a
        if a.inputs or a.breakdown_input:raise ValueError('analysis inputs are analyze-only')
        a.model=a.model or 'facebook/opt-2.7b';a.dtype=a.dtype or 'float16'
        value=a.prompt_lengths or ('1024' if a.phase=='breakdown' else '32,64' if a.model.endswith('125m') else '512,1024')
        a.lengths=[int(v) for v in value.split(',')]
        if not a.lengths or len(set(a.lengths))!=len(a.lengths):raise ValueError('unique lengths required')
        for n in a.lengths:lengths(str(n),'smoke' if n in (32,64) and a.model.endswith('125m') else 'pilot',1024)
        a.coverage_values=coverages(a.coverages)
        if a.phase=='breakdown' and (a.lengths!=[1024] or set(a.coverage_values)!={0.,.5,'max'}):
            raise ValueError('breakdown requires L1024 coverages 0,.5,max')
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (3 if a.phase=='latency' else 1)
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (20 if a.phase=='latency' else 3)
        if a.capacity_mib<1 or a.warmup_runs<(3 if a.phase=='latency' else 1) or a.measured_runs<(20 if a.phase=='latency' else 1):raise ValueError('invalid counts/capacity')
        if a.phase=='latency' and not a.pilot_inputs:raise ValueError('matching C4 --pilot-inputs required for latency')
        if a.phase=='pilot' and a.pilot_inputs:raise ValueError('fresh C4 Pilot required, not historical approvals')
        if not a.device.startswith('cuda'):raise ValueError('GPU phase requires CUDA; analyze is CPU-only')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1':raise ValueError('CUDA_LAUNCH_BLOCKING invalidates timing')
        a.allow_download=False
    except ValueError as exc:p.error(str(exc))
    return a


def load(path):return __import__('json').loads(Path(path).read_text())


def initialize(a):
    m=helpers().initialize(a)  # stdlib provenance/artifact initialization only
    m.update(experiment='R2-C4',arms=list(ARMS),timing_boundaries=BOUNDARIES)
    paths=[ROOT/'scripts/93_run_r2c4_cache_path.py',*sorted((ROOT/'src/semcache/diagnostics').glob('r2c4_*.py'))]
    m['implementation_sha256'].update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if a.phase=='analyze':m['analysis_config']['breakdown_input']=str(a.breakdown_input)
    atomic_json(a.output_dir/'manifest.json',m);atomic_json(a.output_dir/'timing_boundary_audit.json',BOUNDARIES)
    return m


def analyze(a,m):
    sessions=[(load(p/'manifest.json'),load(p/'cache_path_raw_pairs.json')) for p in a.inputs]
    bm=load(a.breakdown_input/'manifest.json');bd=load(a.breakdown_input/'cache_path_breakdown.json')
    for directory,manifest in [(p,s[0]) for p,s in zip(a.inputs,sessions)]+[(a.breakdown_input,bm)]:
        if hashlib.sha256((directory/'impact_references.json').read_bytes()).hexdigest()!=manifest.get('impact_reference_artifact_sha256'):
            raise MissingEvidence('missing/altered reference artifact: '+str(directory))
    result=aggregate(sessions,(bm,bd),a.order_seed)
    m.update(execution_status='COMPLETE',GPU_VALIDATION='AGGREGATED_GPU_RUNS',
        analysis_process=dict(gpu_inference_performed=False))
    m['input_evidence']=[dict(directory=str(p),manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),
        raw_samples_sha256=hashlib.sha256((p/'cache_path_raw_pairs.json').read_bytes()).hexdigest(),
        execution_status=s[0]['execution_status'],GPU_VALIDATION=s[0]['GPU_VALIDATION']) for p,s in zip(a.inputs,sessions)]
    m['breakdown_evidence']=dict(directory=str(a.breakdown_input),manifest_sha256=hashlib.sha256((a.breakdown_input/'manifest.json').read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((a.breakdown_input/'cache_path_breakdown.json').read_bytes()).hexdigest(),pooled_with_minimal=False)
    config=sessions[0][0]['benchmark_config']
    m['aggregated_experimental_config']={k:config[k] for k in ('model','dtype','lengths','coverage_values','cache_device','capacity_mib','seed')}
    m['aggregated_experimental_config'].update(settings=sessions[0][0]['settings'],hardware=sessions[0][0]['hardware_identity'],
        order_seeds=[s[0]['benchmark_config']['order_seed'] for s in sessions],source_code_sha256=sessions[0][0]['code']['sha256'])
    m['conditions']={r['case_id']:dict(status='AGGREGATED',ablation_equivalence_pass=True,impact_replay_reduction_calls=0) for r in result}
    atomic_json(a.output_dir/'cache_path_aggregate_summary.json',result)
    atomic_json(a.output_dir/'cache_path_decision.json',dict(execution_status='COMPLETE',conditions=result,
        breakdown_not_pooled=True,classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,
        correctness_qualified_speedup=False,production_speedup_claimed=False,
        interpretation='controlled request-internal exclusions; interaction is measured, savings need not add; C3 181.10ms is historical residual, not independent costs'))


def run(a,m):
    if a.phase=='analyze':return analyze(a,m)
    h=helpers();runtime=Runtime(a);model_hash=runtime.setup_identity()
    runtime.model_weight_identity=model_hash;runtime.execution_code_sha256=m['code']['sha256'];runtime.arm='on'
    backend=h.backend_audit(runtime);m['model_weight_identity']=model_hash
    m['hardware_identity']=dict(uuid=runtime.provenance['gpu'].get('gpu_uuid'),device=a.device);uuid(m['hardware_identity']['uuid'])
    m['environment_identity']=digest(dict(code=m['code']['sha256'],implementation=m['implementation_sha256'],model_hash=model_hash,
        model=runtime.meta,encoder=runtime.encoder.metadata,storage=runtime.storage,capacity_mib=a.capacity_mib,seed=a.seed,
        versions=m['process_environment']['versions'],cuda=runtime.provenance.get('cuda'),hardware=m['hardware_identity'],backend=backend,settings=m['settings']))
    environment=dict(runtime=runtime.provenance,backend=backend,process=m['process_environment'],monitoring=[],
        variables={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')})
    plan=[(n,c) for n in a.lengths for c in a.coverage_values]
    if a.phase=='pilot':plan.sort(key=lambda x:(x[0],str(x[1])))
    else:random.Random(a.order_seed).shuffle(plan)
    m['conditions']={f'L{n}_cov{c}':dict(status='NOT_RUN',length=n,requested_coverage=c,ablation_equivalence_pass=False,
        impact_replay_reduction_calls=None,measured_pairs=0,timing_boundary_verified=False) for n,c in plan}
    m['execution_order']=list(m['conditions']);m['environment_verified']=True
    pilots=[(load(p/'manifest.json'),load(p/'capacity_pilot.json')) for p in a.pilot_inputs or []]
    rows=[];diagnostics=[];refs={};gates={};workloads={}
    def approval(key):
        errors=[]
        for pm,pc in pilots:
            try:return approve_pilot(pm,pc,m,key)
            except (ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
        raise MissingEvidence('C4_PILOT_REQUIRED '+key+': '+'; '.join(errors))
    def persist(context=None):
        if context:m['active_context']=context
        atomic_json(a.output_dir/'manifest.json',m);atomic_json(a.output_dir/'environment.json',environment)
        for name,value in [('impact_references',refs),('correctness_and_gates',gates),('workload_audit',workloads)]:atomic_json(a.output_dir/(name+'.json'),value)
        if a.phase=='breakdown':
            atomic_json(a.output_dir/'cache_path_breakdown.json',diagnostics)
            csv_write(a.output_dir/'cache_path_breakdown.csv',[dict(case_id=r['case_id'],arm=r['arm'],repeat=r['repeat'],**v) for r in diagnostics for v in r['components']])
        else:
            atomic_json(a.output_dir/'cache_path_raw_pairs.json',rows);csv_write(a.output_dir/'cache_path_session_summary.csv',csv_summary(rows))
        if a.phase=='pilot':atomic_json(a.output_dir/'capacity_pilot.json',dict(experiment='R2-C4',execution_status=m['execution_status'],
            environment_identity=m['environment_identity'],model_weights_verified=bool(m.get('final_weights_verified')),conditions=m['conditions']))
    def observe(label):
        state=h.monitor(a.device);environment['monitoring'].append(dict(phase=label,state=state))
        m['environment_verified']=m['environment_verified'] and state.get('status')=='GPU_MONITOR_VERIFIED' and state.get('other_processes') is not None
        if state.get('other_processes'):raise MissingEvidence('competing GPU processes observed')
    persist();observe('initial')
    use_pilot=a.phase=='latency' or bool(a.pilot_inputs)
    m['pilot_approval_policy']='REQUIRED_COMPATIBLE_C4_PILOT' if use_pilot else 'FRESH_PILOT' if a.phase=='pilot' else 'BREAKDOWN_SELF_VALIDATED_WITH_CONSERVATIVE_PREFLIGHT_AND_FULL_GATES'
    if use_pilot:
        for key in m['conditions']:approval(key)
    current=None;baseline=None;previous_peak=None
    for key,c in m['conditions'].items():
        recorder=None
        try:
            n,cov=c['length'],c['requested_coverage'];persist(dict(case=key,phase='preflight'))
            lengths(str(n),'smoke' if n in (32,64) else 'pilot',runtime.model.config.max_position_embeddings)
            if code_identity(ROOT)['sha256']!=m['code']['sha256']:raise MissingEvidence('execution code changed')
            if current!=n:
                h.release_source(runtime);current=None;c['memory_before']=memory(a.device,reset=True);baseline=c['memory_before'];previous_peak=None
                estimate=memory_estimate(runtime.model.config,n,next(runtime.model.parameters()).element_size(),a.capacity_mib*1024**2,False)
                c['preflight']=estimate;gpu=c['memory_before']['gpu']
                if estimate['source_qkv_backing_bytes']>a.capacity_mib*1024**2:raise MemoryError('requested Source QKV exceeds capacity')
                if estimate['estimated_additional_bytes']>gpu['free_bytes']+gpu['reserved_bytes']-gpu['allocated_bytes']:raise MemoryError('C1 conservative preflight rejected configuration')
                if a.phase=='pilot':recorder=PilotPeakRecorder(a.device);recorder.begin(baseline)
                runtime.prepare(n);current=n;m['GPU_VALIDATION']='EXECUTED'
            elif a.phase=='pilot':recorder=PilotPeakRecorder(a.device);recorder.begin(baseline,previous_peak)
            runtime.coverage=cov
            c['condition_identity']=digest(dict(ids=runtime.ids,source_hashes=runtime.source_hashes,weights=runtime.weights,namespace=runtime.namespace,coverage=cov,storage=runtime.storage))
            if use_pilot and approval(key)['condition_identity']!=c['condition_identity']:raise MissingEvidence('Pilot source/token mismatch')
            c['untimed_gate_costs_ms']={}
            def untimed(name,fn):
                started=perf_counter_ns()
                try:return fn()
                finally:c['untimed_gate_costs_ms'][name]=(perf_counter_ns()-started)/1e6
            persist(dict(case=key,phase='untimed_physical_gate'));physical=untimed('physical_gate',lambda:physical_gate(runtime))
            coverage=physical['audits']['RAW_SEMCACHE_QKV']['coverage'];gates[key]={'physical':physical}
            c.update(structural_valid=True,cache_integrity_valid=True,memory_valid=True,actual_reused_tokens=coverage['reused_tokens'],
                actual_coverage=coverage['actual_coverage'],coverage_audit=coverage)
            persist(dict(case=key,phase='untimed_impact_reference'));r,outputs,records=untimed('impact_references',lambda:capture_references(runtime));refs[key]=r
            c['reference_digests']=dict(runtime.reference_digests)
            c['policy_proofs']={b:records[b]['policy'] for b in BASES};c['impact_values_digests']={b:records[b]['impact_values_digest'] for b in BASES}
            if use_pilot and approval(key)['reference_digests']!=c['reference_digests']:raise MissingEvidence('Pilot actual Impact reference changed')
            gates[key]['impact_replay']=untimed('impact_replay_gate',lambda:validate_replay(runtime,outputs,records));del outputs,records
            persist(dict(case=key,phase='untimed_raw_oracle_gate'));gates[key]['cache_path']=untimed('cache_path_gate',lambda:gate(runtime))
            c.update(ablation_equivalence_pass=True,impact_replay_reduction_calls=0)
            workloads[key]=dict(token_ids=runtime.ids,token_hash=digest(runtime.ids),source_snapshot=runtime.snapshot,
                source_costs=runtime.costs,cache_resident_bytes=runtime.catalog.resident_bytes,original_source_preflight=runtime.preflight)
            identity=dict(independent_run_id=a.independent_run_id,case_id=key,length=n,requested_coverage=cov,projection_layers=len(runtime.adapter.layers),
                condition_identity=c['condition_identity'],model=a.model,dtype=a.dtype,source_snapshot=runtime.snapshot,
                endpoint_policy='all_token_logits_use_cache_false_attention_true',seed=a.seed,order_seed=a.order_seed)
            if a.phase=='breakdown':
                start=len(diagnostics);all_diagnostics=diagnostics
                def save(context=None,breakdown=None):
                    if breakdown is not None:all_diagnostics[start:]=breakdown
                    persist(context)
                collect_breakdown(runtime,a,identity,save)
                block=diagnostics[start:];c['timing_boundary_verified']=bool(block) and all(r['timing_boundary_verified'] for r in block)
                c['full_relative_strict']={arm:all(r['full_relative_quality']['allclose'] for r in block if r['arm']==arm) for arm in ARMS}
                if not c['timing_boundary_verified']:raise MissingEvidence('breakdown timing boundaries unverified')
            else:
                start=len(rows);all_rows=rows
                def save(context=None,rows=None):
                    if rows is not None:all_rows[start:]=rows
                    persist(context)
                paired(runtime,a,identity,save);summary=summarize(rows[start:])[0]
                c['measured_pairs']=summary['count'];c['full_relative_strict']=summary['full_relative_strict']
                if c['measured_pairs']!=a.measured_runs:raise MissingEvidence('incomplete paired observations')
            runtime.guard(True);c['memory_after']=memory(a.device)
            if recorder:previous_peak=recorder.finish();c['pilot_memory']=previous_peak
            if code_identity(ROOT)['sha256']!=m['code']['sha256']:raise MissingEvidence('code changed during case')
            observe(key+'_after');c['status']={'pilot':'CAPACITY_VALIDATED','latency':'VALID','breakdown':'DIAGNOSED'}[a.phase];persist()
        except (Exception,KeyboardInterrupt) as exc:
            c.update(failure_record(exc,m.get('active_context')));c['status']=failure_status(exc);c['ablation_equivalence_pass']=False
            c['impact_replay_reduction_calls']=getattr(exc,'impact_audit',{}).get('real_reduction_calls')
            c['cache_path_failure_evidence']=getattr(exc,'cache_path_evidence',None);c['failed_gate_evidence']=getattr(exc,'evidence',None)
            try:c['memory_at_failure']=memory(a.device)
            except Exception as error:c['memory_at_failure']={'status':'UNAVAILABLE','reason':str(error)}
            persist();raise
        finally:
            if recorder:recorder.close()
    runtime.model_guard(True)
    if code_identity(ROOT)['sha256']!=m['code']['sha256']:raise MissingEvidence('code changed before completion')
    m['impact_reference_artifact_sha256']=hashlib.sha256((a.output_dir/'impact_references.json').read_bytes()).hexdigest()
    m.update(final_weights_verified=True,execution_status='COMPLETE',GPU_VALIDATION='EXECUTED');persist()
    atomic_json(a.output_dir/'cache_path_decision.json',dict(execution_status='COMPLETE',conditions=m['conditions'],
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,correctness_qualified_speedup=False,production_speedup_claimed=False))


def main(argv=None):
    a=parse_args(argv);m=initialize(a)
    try:
        with raw_only_guard() as raw:
            try:run(a,m)
            finally:m['raw_only']=raw.evidence()
        atomic_json(a.output_dir/'manifest.json',m);return 0
    except (Exception,KeyboardInterrupt) as exc:
        m.update(execution_status='FAILED',failure=dict(failure_record(exc,m.get('active_context')),status=failure_status(exc)))
        atomic_json(a.output_dir/'manifest.json',m)
        if a.phase=='pilot':atomic_json(a.output_dir/'capacity_pilot.json',dict(experiment='R2-C4',execution_status='FAILED',
            model_weights_verified=False,conditions=m['conditions'],failure=m['failure']))
        atomic_json(a.output_dir/'cache_path_decision.json',dict(execution_status='FAILED',failure=m['failure'],correctness_qualified_speedup=False))
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
