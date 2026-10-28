#!/usr/bin/env python3
"""R2-C3: exact Attention Impact replay, ORACLE upper-bound request ablation."""
import argparse
from datetime import datetime,timezone
import gc
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write
from semcache.diagnostics.r1b3_break_even import code_identity,MissingEvidence,ParityFailure
from semcache.diagnostics.r2c1_workloads import coverages,memory_estimate,lengths
from semcache.diagnostics.r2c1_benchmark import validate as validate_physical
from semcache.diagnostics.r2c2_profile_memory import PilotPeakRecorder,failure_status,uuid
from semcache.diagnostics.r2b3_capacity import memory,failure_record
from semcache.diagnostics.r2b2_profiler import monitor
from semcache.diagnostics.r2b2_attention_ablation import backend_audit
from semcache.diagnostics.r2c3_benchmark import Runtime,BOUNDARIES,capture_references,validate_replay,paired
from semcache.diagnostics.r2c3_impact_replay import ARMS,BASES,SETTINGS
from semcache.diagnostics.r2c3_analysis import summarize,csv_summaries,aggregate,approve_pilot,coverage_changes


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('pilot','latency','analyze'),required=True)
    p.add_argument('--model',choices=('facebook/opt-2.7b','facebook/opt-125m'))
    p.add_argument('--dtype',choices=('float16','float32'))
    p.add_argument('--prompt-lengths'); p.add_argument('--coverages',default='0,.5,max')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cache-device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--capacity-mib',type=int,default=512)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--order-seed',type=int,default=4101)
    p.add_argument('--independent-run-id',default='run01')
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--pilot-inputs',nargs='+',type=Path)
    p.add_argument('--inputs',nargs=3,type=Path,help='exactly three COMPLETE independent C3 latency directories')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--encoder-revision',default='main')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.phase=='analyze':
            if not a.inputs or any(v is not None for v in (a.model,a.dtype,a.prompt_lengths,a.pilot_inputs,a.warmup_runs,a.measured_runs)):
                raise ValueError('analyze derives GPU experiment configuration from exactly three --inputs')
            return a
        if a.inputs: raise ValueError('--inputs only for analyze')
        a.model=a.model or 'facebook/opt-2.7b'; a.dtype=a.dtype or 'float16'
        value=a.prompt_lengths or ('32,64' if a.model.endswith('125m') else '512,1024')
        a.lengths=[int(v) for v in value.split(',')]
        if not a.lengths or len(set(a.lengths))!=len(a.lengths): raise ValueError('unique lengths required')
        for n in a.lengths:
            lengths(str(n),'smoke' if n in (32,64) and a.model.endswith('125m') else 'pilot',1024)
        a.coverage_values=coverages(a.coverages)
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.phase=='pilot' else 3)
        a.measured_runs=a.measured_runs if a.measured_runs is not None else (3 if a.phase=='pilot' else 20)
        if a.capacity_mib<1 or a.warmup_runs<(1 if a.phase=='pilot' else 3) or a.measured_runs<(1 if a.phase=='pilot' else 20):
            raise ValueError('invalid capacity/counts; latency requires >=3 warmups and >=20 measured pairs')
        if a.phase=='latency' and not a.pilot_inputs: raise ValueError('--pilot-inputs required for latency')
        if a.phase=='pilot' and a.pilot_inputs: raise ValueError('new C3 Pilot does not consume historical approvals')
        if not a.device.startswith('cuda'): raise ValueError('GPU phases require actual CUDA; analysis is CPU-only')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': raise ValueError('CUDA_LAUNCH_BLOCKING invalidates request comparison')
        a.allow_download=False
    except ValueError as exc: p.error(str(exc))
    return a


def load(path): return json.loads(Path(path).read_text())


def initialize(a):
    a.output_dir.mkdir(parents=True,exist_ok=False)
    def version(name):
        try: return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: return None
    def serial(v):
        if isinstance(v,Path): return str(v)
        if isinstance(v,(tuple,list)): return [serial(x) for x in v]
        return v
    paths=[ROOT/'scripts/92_run_r2c3_impact_ablation.py',*sorted((ROOT/'src/semcache/diagnostics').glob('r2c3_*.py'))]
    started=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19] if Path('/proc/self/stat').exists() else datetime.now(timezone.utc).isoformat()
    m=dict(experiment='R2-C3',execution_status='RUNNING',GPU_VALIDATION='NOT_RUN',
        classification='ORACLE_DIAGNOSTIC',ablation='ORACLE_UPPER_BOUND',safe_reuse_claimed=False,
        correctness_qualified_speedup=False,production_speedup_claimed=False,
        code=code_identity(ROOT),implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        process_instance=f'{socket.gethostname()}:{os.getpid()}:{started}',created_at_utc=datetime.now(timezone.utc).isoformat(),
        process_environment=dict(python=platform.python_version(),versions={n:version(n) for n in ('torch','transformers','peft')}),
        timing_boundaries=BOUNDARIES,arms=list(ARMS),
        settings=dict(SETTINGS),conditions={})
    if a.phase=='analyze':
        m['analysis_process']=dict(gpu_inference_performed=False)
        m['analysis_config']=dict(inputs=[str(v) for v in a.inputs],bootstrap_seed=a.order_seed)
    else:
        m['benchmark_config']={k:serial(v) for k,v in vars(a).items() if k!='inputs'}
        m['hardware_identity']=dict(uuid=None,device=a.device,status='UNVERIFIED_BEFORE_MODEL_LOAD')
        m['conditions']={f'L{n}_cov{c}':dict(status='NOT_RUN',length=n,requested_coverage=c,
            ablation_equivalence_pass=False,impact_replay_reduction_calls=None,measured_pairs=0)
            for n in a.lengths for c in a.coverage_values}
        m['active_context']=dict(phase='model_loading')
    atomic_json(a.output_dir/'manifest.json',m)
    atomic_json(a.output_dir/'timing_boundary_audit.json',BOUNDARIES)
    return m


def analyze(a,m):
    sessions=[(load(p/'manifest.json'),load(p/'ablation_raw_pairs.json')) for p in a.inputs]
    for directory,(manifest,_) in zip(a.inputs,sessions):
        observed=hashlib.sha256((directory/'impact_references.json').read_bytes()).hexdigest()
        if observed!=manifest.get('impact_reference_artifact_sha256'):
            raise MissingEvidence('input impact reference artifact missing/altered: '+str(directory))
    result=aggregate(sessions,a.order_seed)
    m['input_evidence']=[dict(directory=str(p),manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),
        samples_sha256=hashlib.sha256((p/'ablation_raw_pairs.json').read_bytes()).hexdigest(),
        impact_references_sha256=s[0]['impact_reference_artifact_sha256'],
        GPU_VALIDATION=s[0]['GPU_VALIDATION'],execution_status=s[0]['execution_status']) for p,s in zip(a.inputs,sessions)]
    m['aggregated_experimental_config']=dict(benchmark_config={k:sessions[0][0]['benchmark_config'][k] for k in
        ('model','dtype','lengths','coverage_values','device','cache_device','capacity_mib','seed')},
        settings=sessions[0][0]['settings'],hardware=sessions[0][0]['hardware_identity'],
        execution_code_sha256=sessions[0][0]['code']['sha256'],independent_sessions=3)
    m['aggregated_experimental_config']['independent_run_ids']=[s[0]['benchmark_config']['independent_run_id'] for s in sessions]
    m['aggregated_experimental_config']['order_seeds']=[s[0]['benchmark_config']['order_seed'] for s in sessions]
    m['execution_status']='COMPLETE'; m['GPU_VALIDATION']='AGGREGATED_GPU_RUNS'
    m['conditions']={r['case_id']:dict(status='AGGREGATED',ablation_equivalence_pass=True,
        impact_replay_reduction_calls=0,actual_coverage=r['actual_coverage']) for r in result}
    atomic_json(a.output_dir/'ablation_aggregate_summary.json',result)
    atomic_json(a.output_dir/'ablation_decision.json',dict(execution_status='COMPLETE',conditions=result,
        coverage_changes=coverage_changes(result),
        interpretation='upper-bound Attention Impact reduction exclusion; not production/E2E/service speedup',
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,correctness_qualified_speedup=False,production_speedup_claimed=False))


def release_source(runtime):
    for name in ('engine','catalog','index','descriptor','caches','hits','plan','source_hashes','policy_state',
                 'references','reference_digests','reference_policies','expected_binding'):
        if hasattr(runtime,name): delattr(runtime,name)
    gc.collect()


def run(a,m):
    if a.phase=='analyze': return analyze(a,m)
    runtime=Runtime(a); model_hash=runtime.setup_identity()
    runtime.model_weight_identity=model_hash; runtime.execution_code_sha256=m['code']['sha256']
    runtime.arm='on'
    backend=backend_audit(runtime)
    m['hardware_identity']=dict(uuid=runtime.provenance['gpu'].get('gpu_uuid'),device=a.device)
    uuid(m['hardware_identity']['uuid'])
    m['model_weight_identity']=model_hash
    m['environment_identity']=digest(dict(code=m['code']['sha256'],implementation=m['implementation_sha256'],
        model_hash=model_hash,model=runtime.meta,encoder=runtime.encoder.metadata,storage=runtime.storage,
        capacity_mib=a.capacity_mib,seed=a.seed,versions=m['process_environment']['versions'],
        cuda=runtime.provenance.get('cuda'),hardware=m['hardware_identity'],backend=backend,settings=m['settings']))
    environment=dict(runtime=runtime.provenance,backend=backend,process=m['process_environment'],
        variables={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')},monitoring=[])
    plan=[(n,c) for n in a.lengths for c in a.coverage_values]
    if a.phase=='pilot': plan.sort(key=lambda x:(x[0],str(x[1])))
    else: random.Random(a.order_seed).shuffle(plan)
    m['conditions']={f'L{n}_cov{cov}':dict(status='NOT_RUN',length=n,requested_coverage=cov,
        ablation_equivalence_pass=False,impact_replay_reduction_calls=None,measured_pairs=0) for n,cov in plan}
    m['execution_order']=list(m['conditions']); rows=[]; references={}; gates={}; workloads={}
    m['environment_verified']=True
    pilot_files=[]
    for p in a.pilot_inputs or []:
        pilot_files.append((load(p/'manifest.json'),load(p/'capacity_pilot.json')))
    def approval(key):
        errors=[]
        for pm,pc in pilot_files:
            try: return approve_pilot(pm,pc,m,key)
            except (ValueError,KeyError,TypeError) as exc: errors.append(str(exc))
        raise MissingEvidence('C3_PILOT_REQUIRED '+key+': '+'; '.join(errors))
    def persist(context=None):
        if context: m['active_context']=context
        atomic_json(a.output_dir/'manifest.json',m); atomic_json(a.output_dir/'environment.json',environment)
        atomic_json(a.output_dir/'ablation_raw_pairs.json',rows)
        csv_write(a.output_dir/'ablation_session_summary.csv',csv_summaries(rows))
        for name,data in [('impact_references',references),('correctness_and_gates',gates),('workload_audit',workloads)]:
            atomic_json(a.output_dir/(name+'.json'),data)
        if a.phase=='pilot': atomic_json(a.output_dir/'capacity_pilot.json',dict(experiment='R2-C3',
            execution_status=m['execution_status'],environment_identity=m['environment_identity'],
            model_weights_verified=bool(m.get('final_weights_verified')),conditions=m['conditions']))
    def observe(label):
        state=monitor(a.device); environment['monitoring'].append(dict(phase=label,state=state))
        m['environment_verified']=(m['environment_verified'] and state.get('status')=='GPU_MONITOR_VERIFIED'
            and state.get('other_processes') is not None)
        if state.get('other_processes'): raise MissingEvidence('competing physical-GPU processes observed')
        return state
    persist(); observe('initial')
    if a.phase=='latency':
        for key in m['conditions']: approval(key)
    current=None; baseline=None; previous_peak=None
    for key,c in m['conditions'].items():
        recorder=None
        try:
            n,cov=c['length'],c['requested_coverage']; persist(dict(case=key,phase='preflight'))
            lengths(str(n),'smoke' if n in (32,64) else 'pilot',runtime.model.config.max_position_embeddings)
            if code_identity(ROOT)['sha256']!=m['code']['sha256']: raise MissingEvidence('code identity changed during execution')
            if current!=n:
                release_source(runtime); current=None
                c['memory_before']=memory(a.device,reset=True); baseline=c['memory_before']; previous_peak=None
                estimate=memory_estimate(runtime.model.config,n,next(runtime.model.parameters()).element_size(),a.capacity_mib*1024**2,False)
                c['preflight']=estimate; gpu=c['memory_before']['gpu']
                if estimate['source_qkv_backing_bytes']>a.capacity_mib*1024**2: raise MemoryError('RAW QKV exceeds requested source capacity')
                if estimate['estimated_additional_bytes']>gpu['free_bytes']+gpu['reserved_bytes']-gpu['allocated_bytes']:
                    raise MemoryError('C1 conservative live-allocation screen rejected requested condition')
                if a.phase=='pilot':
                    recorder=PilotPeakRecorder(a.device); recorder.begin(baseline)
                runtime.prepare(n); current=n; m['GPU_VALIDATION']='EXECUTED'
            elif a.phase=='pilot':
                recorder=PilotPeakRecorder(a.device); recorder.begin(baseline,previous_peak)
            runtime.coverage=cov
            c['condition_identity']=digest(dict(ids=runtime.ids,source_hashes=runtime.source_hashes,weights=runtime.weights,
                namespace=runtime.namespace,coverage=cov,storage=runtime.storage))
            if a.phase=='latency' and approval(key)['condition_identity']!=c['condition_identity']:
                raise MissingEvidence('Pilot source/token/coverage identity mismatch')
            persist(dict(case=key,phase='physical_gate'))
            from time import perf_counter_ns
            gate_start=perf_counter_ns(); physical=validate_physical(runtime)
            c['untimed_physical_gate_wall_ms']=(perf_counter_ns()-gate_start)/1e6
            gates[key]={'physical':physical}
            coverage=physical['audits']['RAW_SEMCACHE_QKV']['coverage']
            c.update(structural_valid=True,cache_integrity_valid=True,memory_valid=True,
                source_token_integrity_valid=True,actual_reused_tokens=coverage['reused_tokens'],
                actual_coverage=coverage['actual_coverage'],skipped_projection_rows=coverage['skipped_projection_rows'],
                native_projection_rows=coverage['native_projection_rows'],coverage_audit=coverage)
            persist(dict(case=key,phase='untimed_reference'))
            refs,reference_outputs,reference_records=capture_references(runtime); references[key]=refs
            c['reference_digests']=dict(runtime.reference_digests)
            c['impact_reference_provenance']={b:dict(provenance=refs[b]['provenance'],digest=refs[b]['digest'],
                artifact='impact_references.json',artifact_case=key) for b in BASES}
            c['impact_values_digests']={b:reference_records[b]['impact_values_digest'] for b in BASES}
            c['policy_proofs']={b:reference_records[b]['policy'] for b in BASES}
            if a.phase=='latency' and approval(key)['reference_digests']!=c['reference_digests']:
                raise MissingEvidence('fresh ON reference differs from Pilot impact/provenance')
            persist(dict(case=key,phase='replay_equivalence_gate'))
            gates[key]['replay']=validate_replay(runtime,reference_outputs,reference_records)
            del reference_outputs,reference_records
            c.update(ablation_equivalence_pass=True,impact_replay_reduction_calls=0)
            semantic_ids=runtime.encoder.tokenizer(runtime.text,truncation=False)['input_ids']
            workloads[key]=dict(token_ids=runtime.ids,token_hash=digest(runtime.ids),source_snapshot=runtime.snapshot,
                namespace=runtime.namespace,adapter_weight_identity=runtime.weights,source_costs=runtime.costs,
                cache_resident_bytes=runtime.catalog.resident_bytes,original_source_preflight=runtime.preflight,
                semantic_token_count=len(semantic_ids),semantic_cap=runtime.encoder.max_length,
                semantic_truncated=len(semantic_ids)>runtime.encoder.max_length,model_truncated=False)
            identity=dict(independent_run_id=a.independent_run_id,case_id=key,length=n,requested_coverage=cov,
                projection_layers=len(runtime.adapter.layers),condition_identity=c['condition_identity'],model=a.model,dtype=a.dtype,
                source_snapshot=runtime.snapshot,endpoint_policy='all_token_logits_use_cache_false_attention_true',
                seed=a.seed,order_seed=a.order_seed)
            start=len(rows); all_rows=rows
            def save(context=None,rows=None):
                if rows is not None: all_rows[start:]=rows
                persist(context)
            paired(runtime,a,identity,save)
            block=rows[start:]; summary=summarize(block)[0]
            c['measured_pairs']=summary['count']; c['full_relative_strict']=summary['full_relative_strict']
            if c['measured_pairs']!=a.measured_runs: raise MissingEvidence('incomplete paired measurements')
            runtime.guard(True)
            if code_identity(ROOT)['sha256']!=m['code']['sha256']: raise MissingEvidence('implementation changed during case')
            c['memory_after']=memory(a.device)
            if recorder:
                previous_peak=recorder.finish(); c['pilot_memory']=previous_peak
            observe(key+'_after')
            c['status']='CAPACITY_VALIDATED' if a.phase=='pilot' else 'VALID'; persist()
        except (Exception,KeyboardInterrupt) as exc:
            c.update(failure_record(exc,m.get('active_context'))); c['status']=failure_status(exc)
            if hasattr(exc,'evidence'): c['failed_gate_evidence']=exc.evidence
            c['failed_impact_audit']=getattr(exc,'impact_audit',None)
            c['impact_replay_reduction_calls']=(exc.impact_audit['real_reduction_calls']
                if getattr(exc,'impact_audit',{}).get('replay') else None)
            c['ablation_equivalence_pass']=False
            try: c['memory_at_failure']=memory(a.device)
            except Exception as error: c['memory_at_failure']={'status':'UNAVAILABLE','reason':str(error)}
            persist(); raise
        finally:
            if recorder: recorder.close()
    runtime.model_guard(True)
    if code_identity(ROOT)['sha256']!=m['code']['sha256']: raise MissingEvidence('code changed before final completion')
    m['impact_reference_artifact_sha256']=hashlib.sha256((a.output_dir/'impact_references.json').read_bytes()).hexdigest()
    m.update(final_weights_verified=True,execution_status='COMPLETE',GPU_VALIDATION='EXECUTED')
    persist()
    atomic_json(a.output_dir/'ablation_decision.json',dict(execution_status='COMPLETE',conditions=m['conditions'],
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,correctness_qualified_speedup=False,
        interpretation='upper-bound conditional impact ablation; analyze exactly three independent sessions'))


def main(argv=None):
    a=parse_args(argv); m=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        m.update(execution_status='FAILED',failure=dict(failure_record(exc,m.get('active_context')),status=failure_status(exc)),
            correctness_qualified_speedup=False)
        atomic_json(a.output_dir/'manifest.json',m)
        if a.phase=='pilot':
            atomic_json(a.output_dir/'capacity_pilot.json',dict(experiment='R2-C3',execution_status='FAILED',
                environment_identity=m.get('environment_identity'),model_weights_verified=False,
                conditions=m['conditions'],failure=m['failure']))
        atomic_json(a.output_dir/'ablation_decision.json',dict(execution_status='FAILED',failure=m['failure'],
            correctness_qualified_speedup=False,safe_reuse_claimed=False))
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
