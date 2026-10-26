#!/usr/bin/env python3
"""R1-B3-A source occurrence audit and opt-in admission-bypassing RAW catalog."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard,audit_projection_records
from semcache.metrics.r1b2 import coverage_override,csv_write,stats
from semcache.diagnostics.r1b3_occurrences import (OccurrenceCatalog,source_evidence,
    occurrence_loss_audit,capture_completed_source,catalog_override,physical_bytes)


def b2():
    spec=importlib.util.spec_from_file_location('r1b2_runner',ROOT/'scripts/72_run_r1b2_latency.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def args(argv=None):
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--enable-catalog',action='store_true',help='Opt in to ORACLE_DIAGNOSTIC occurrence sidecar')
    extra.add_argument('--catalog-capacity-mib',type=int,default=256)
    extra.add_argument('--full-shape-control',action='store_true',help='Untimed D1 injection correctness control')
    extra.add_argument('--reference-b2-root',type=Path,help='Read-only B2 root for token/runtime/coverage comparison')
    x,remaining=extra.parse_known_args(argv)
    if '--help' in remaining:
        extra.print_help()
    if not any(x=='--instrumentation' or x.startswith('--instrumentation=') for x in remaining):
        remaining+=['--instrumentation','minimal']
    a=b2().parse_args(remaining)
    if a.instrumentation!='minimal': raise ValueError('R1-B3-A primary runner supports minimal only; use B2 for breakdown')
    if x.catalog_capacity_mib<1: raise ValueError('Positive catalog capacity required')
    for k,v in vars(x).items(): setattr(a,k,v)
    return a


def load_groups(root,identity):
    bundles=[]; seen=set()
    for path in sorted((root/'checkpoints').glob('*.json')):
        b=json.loads(path.read_text())
        if b['identity']!=identity or b['status']!='COMPLETE' or digest(b['rows'])!=b['checksum']:
            raise ValueError('Invalid checkpoint '+str(path))
        if digest(b['audits'])!=b['audit_checksum']: raise ValueError('Audit checkpoint checksum mismatch')
        for row in b['rows']:
            if row['row_id'] in seen: raise ValueError('Duplicate row')
            seen.add(row['row_id'])
        bundles.append(b)
    return bundles


def reports(root,bundles):
    rows=[r for b in bundles for r in b['rows']]
    audits={p.stem:json.loads(p.read_text()) for p in (root/'source_audits').glob('*.json')}
    audits.update({k:v for b in bundles for k,v in b['audits'].items()})
    atomic_json(root/'occurrence_loss_audit.json',audits)
    fields={
        'coverage_comparison.csv':('requested_coverage','actual_coverage','reused_tokens','recomputed_tokens','selected_windows','stored_occurrences','unique_keys','unique_source_keys','skipped_projection_rows','native_projection_rows','theoretical_upper_tokens','audit_id','selected_spans','selection_windows'),
        'correctness_comparison.csv':('correctness','projection_skip_audit','full_shape_control','storage_parity','reference_b2_comparison'),
        'storage_accounting.csv':('catalog_physical_bytes','legacy_physical_bytes','legacy_physical_bytes_before_target','combined_physical_bytes','catalog_copy_ms','catalog_build_ms','catalog_transfer_bytes','memory_before_source','memory_after_build','memory_after_target'),
        'latency_comparison.csv':('source_build_ms','source_request_ms','catalog_build_ms','target_warm_ms','catalog_selection_host_ms','source_capture_enabled')}
    common=('row_id','length','storage','phase','repeat','mode','classification')
    for filename,columns in fields.items(): csv_write(root/filename,[{k:r.get(k) for k in (*common,*columns)} for r in rows])
    groups={}
    for row in rows:
        if row['phase']=='measured': groups.setdefault((row['length'],row['storage'],row['mode'],str(row['requested_coverage'])),[]).append(row)
    summary=[]
    for k,group in groups.items():
        summary.append(dict(length=k[0],storage=k[1],mode=k[2],requested_coverage=k[3],count=len(group),
            warm_latency=stats([r['target_warm_ms'] for r in group]),source_build=stats([r['source_build_ms'] for r in group if r['source_build_ms'] is not None]),
            reused_tokens=sorted({r['reused_tokens'] for r in group}),logit_passes=sum(r['correctness']['logit_allclose'] for r in group),
            safe_reuse_claimed=False))
    atomic_json(root/'summary.json',dict(groups=summary,historical_54_token_root_cause_verified=False,
        note='Source provenance verifies current-run loss only; catalog is ORACLE_DIAGNOSTIC; no speedup or FP16-error fix inferred'))


def quality(reference,actual):
    import torch
    from semcache.evaluation.logit_metrics import compare_logits
    q=compare_logits(reference,actual,0)
    q={k:v if not isinstance(v,float) or math.isfinite(v) else None for k,v in q.items()}
    q.update(logit_allclose=bool(torch.isfinite(reference).all() and torch.isfinite(actual).all() and
        torch.allclose(reference,actual,atol=1e-3,rtol=1e-3)),atol=1e-3,rtol=1e-3,
        last_position_kl=q['last_position_kl_baseline_to_injected'])
    return q


def execute(a,manifest,guard):
    import torch
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.cache.global_cache import GlobalCache
    from semcache.semcache_engine import SemCacheEngine
    from semcache.metrics.scaling_workload import controlled_length_trace
    from semcache.metrics.r1_runtime import engine_request,native_request,synchronize,memory_preflight,memory_snapshot,raw_storage_audit
    from semcache.utils.seed import seed_everything
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; NOT EXECUTED')
    seed_everything(a.seed); start=perf_counter_ns()
    model,tokenizer,metadata=load_model(dict(name=a.model,tokenizer=a.model,revision=a.revision,
        tokenizer_revision=a.tokenizer_revision or a.revision,dtype=a.dtype,device=a.device,
        local_files_only=not a.allow_download,attention_implementation='eager'))
    if getattr(model,'is_quantized',False) or getattr(model.config,'quantization_config',None): raise ValueError('Quantization forbidden')
    model,lora=create_controlled_users(model); adapter=OPTModelAdapter(model)
    encoder=TinyBERTSemanticEncoder(device=a.device,dtype=a.dtype,revision=a.encoder_revision,local_files_only=not a.allow_download)
    synchronize(a.device)
    weights={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for n,p in model.named_parameters() if 'lora_' in n and 'user_a' in n}
    namespace=digest({k:v for k,v in metadata.items() if k not in ('device','dtype','torch_version','transformers_version')})
    weight_identity=digest(dict(model_revision=metadata.get('resolved_model_revision'),adapter_weights=weights))
    if not metadata.get('resolved_model_revision'): raise ValueError('Resolved model revision required for namespace identity')
    layout={l:tuple((adapter.projection_modules(l)[role].out_features,
        adapter.projection_modules(l)[role].get_base_layer().weight.dtype) for role in 'qkv') for l in range(len(adapter.layers))}
    runtime=dict(model=metadata,lora=lora,encoder=encoder.metadata,namespace=namespace,weight_identity=weight_identity,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,
        driver=b2().legacy().command_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    if manifest.get('runtime') and manifest['runtime']!=runtime: raise ValueError('Resume runtime mismatch')
    manifest.update(runtime=runtime,initialization_ms=(perf_counter_ns()-start)/1e6)
    bundles=load_groups(a.output_dir,manifest['identity'])
    schedule=[('warmup',i) for i in range(a.warmup_runs)]+[('measured',i) for i in range(a.measured_runs)]
    cases=[('FULL_RECOMPUTE',None),('LOOKUP_NO_REUSE',None),('NATURAL_RAW_REUSE',None)]
    cases += [('LEGACY_POSITION_ALIGNED',c) for c in a.coverage_values]
    if a.enable_catalog: cases += [('POSITIONAL_OCCURRENCE_DIAGNOSTIC',c) for c in a.coverage_values]
    for length in a.lengths:
        trace=controlled_length_trace(tokenizer,length); ids=trace[1]['token_ids']; text=trace[1]['text']; user='user_a'
        if len(ids)!=length: raise ValueError('TOKEN_LENGTH_MISMATCH')
        manifest.setdefault('fixtures',{})[str(length)]=dict(token_ids=ids,token_hash=digest(ids))
        if a.reference_b2_root:
            refroot=a.reference_b2_root/f'tokens_{length}'
            refmanifest=json.loads((refroot/'manifest.json').read_text())
            if refmanifest['fixture']['token_ids']!=ids: raise ValueError('Reference B2 tokens differ')
            matching=[]
            with (refroot/'raw.jsonl').open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['mode']=='POSITION_ALIGNED_REUSE' and r['requested_coverage']=='max' and r['phase']=='measured' and r['instrumentation']=='minimal': matching.append(dict(storage=r['storage'],repeat=r['repeat'],reused_tokens=r['reused_tokens'],selection_signature=r['selection_signature']))
            manifest.setdefault('reference_b2',{})[str(length)]=dict(manifest_sha256=hashlib.sha256((refroot/'manifest.json').read_bytes()).hexdigest(),
                model_metadata_equal=refmanifest['runtime']['model']==metadata,coverage_rows=matching,
                raw_sha256=hashlib.sha256((refroot/'raw.jsonl').read_bytes()).hexdigest(),
                runtime_mismatches=[k for k in runtime if k in refmanifest['runtime'] and runtime[k]!=refmanifest['runtime'][k]],
                code_mismatches=[k for k,v in refmanifest['code']['files'].items() if k in manifest['code']['files'] and manifest['code']['files'][k]!=v],
                seed_equal=a.seed==refmanifest['config']['seed'],
                admission_provenance_available=False,historical_reproduction='UNVERIFIED; B2 compact rows omit full admission/dedup evidence')
        anchors=encoder.encode([trace[0]['text'],trace[-1]['text']])
        def make_engine(storage):
            cluster=IntentClusterer(2,initialization='first_k',update_mode='buffered',update_interval=100); cluster.initialize(anchors)
            return SemCacheEngine(model,tokenizer,adapter,encoder,cluster,GlobalCache(a.cache_capacity_mib*1024**2),
                storage_device=storage,metadata=metadata,seed=a.seed,impact_reducer='paper_row_l2_sum',
                impact_execution='token_precompute',capture_impact_values=True,rho=.8,history_lambda=100,pbr_interval_queries=100)
        for phase,repeat in schedule:
            storage_checks={}
            for storage_name in (a.storages if repeat%2==0 else list(reversed(a.storages))):
                group=f'{length}-{phase}-{repeat}-{storage_name}'; path=a.output_dir/'checkpoints'/(group+'.json')
                if phase=='measured' and path.exists(): continue
                storage=str(next(model.parameters()).device) if storage_name=='cuda' else 'cpu'
                manifest['active_group']=group
                try:
                    manifest['preflight']=memory_preflight(model,length,(a.cache_capacity_mib+(a.catalog_capacity_mib if a.enable_catalog else 0))*1024**2,storage,a.device)
                except MemoryError:
                    manifest['status']='MEMORY_PREFLIGHT_FAILED'; raise
                atomic_json(a.output_dir/'manifest.json',manifest)
                batch=[]; audits={}
                order=cases[repeat%len(cases):]+cases[:repeat%len(cases)]
                if repeat%2: order=list(reversed(order))
                reference=native_request(model,tokenizer,adapter,text,user,False)
                for mode,coverage in order:
                    catalog=None; engine=None; audit_id=None; source_ms=None; build_ms=None; catalog_ms=0.; capture={}; selected=[]
                    oracle=mode=='POSITIONAL_OCCURRENCE_DIAGNOSTIC'
                    manifest['active_case']=dict(mode=mode,coverage=coverage,stage='source')
                    if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(a.device)
                    memory_before=memory_snapshot(a.device)
                    if mode!='FULL_RECOMPUTE':
                        engine=make_engine(storage)
                        with capture_completed_source() if oracle else nullcontext({}) as capture:
                            source=engine_request(engine,text,user,'source','SEMCACHE_LOOKUP_NO_REUSE',True)
                        source_ms=source['summary']['timing']['request_wall_ms']
                        ev=source_evidence(engine,source,namespace,weight_identity); ev['actual_execution']=True
                        audit=occurrence_loss_audit(ev,ids,user,namespace,weight_identity)
                        audit_id=digest(ev); audits.setdefault(audit_id,dict(source_evidence=ev,audit=audit))
                        raw_storage_audit(engine.cache,storage)
                        audit_path=a.output_dir/'source_audits'/(audit_id+'.json')
                        if not audit_path.exists(): atomic_json(audit_path,audits[audit_id])
                        if oracle:
                            manifest['active_case']['stage']='catalog_build'
                            start=perf_counter_ns()
                            catalog=OccurrenceCatalog(namespace,weight_identity,storage,a.catalog_capacity_mib*1024**2)
                            catalog.add_source('source',ids,source['summary']['cluster_id'],user,capture,layout,prefill_complete=True,
                                impacts={w['start']:w['value'] for w in source['summary']['impact_windows']},source_clock=engine.cache.now)
                            synchronize(a.device); catalog_ms=(perf_counter_ns()-start)/1e6
                        capture.clear(); del source
                        build_ms=source_ms+catalog_ms
                    legacy_before=engine.cache.physical_tensor_bytes if engine else 0
                    before_target=memory_snapshot(a.device)
                    scope=(catalog_override(catalog,engine,namespace,weight_identity,layout,coverage) if oracle else
                        coverage_override(coverage) if mode=='LEGACY_POSITION_ALIGNED' else nullcontext({}))
                    manifest['active_case']['stage']='target'
                    with scope as selection:
                        if mode=='FULL_RECOMPUTE':
                            result=native_request(model,tokenizer,adapter,text,user,False)
                            summary=dict(token_ids=result['ids'],reused_unique_token_count=0,reuse_block_provenance=[],timing=result['timing'])
                            records=[]
                        else:
                            execution='SEMCACHE_LOOKUP_NO_REUSE' if mode=='LOOKUP_NO_REUSE' else 'SEMCACHE_PHYSICAL_REUSE' if mode=='NATURAL_RAW_REUSE' else 'SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC'
                            result=engine_request(engine,text,user,'target',execution,False)
                            summary=result['summary']; records=list(result['projection_audit'].values())
                    if summary['token_ids']!=ids: raise AssertionError('Target token identity mismatch')
                    after_target=memory_snapshot(a.device)
                    prov=summary['reuse_block_provenance']; reused=summary['reused_unique_token_count']
                    positions=[i for p in prov for i in range(p['destination_start'],p['destination_end'])]
                    audit_rows=audit_projection_records(records,length,len(adapter.layers),reused,positions) if records else dict(
                        skipped_projection_rows_all_layers_roles=0,native_projection_rows_all_layers_roles=length*len(adapter.layers)*3,derived_native_full=True)
                    spans=[dict(source_start=p['source_start'],source_end=p['source_end'],destination_start=p['destination_start'],destination_end=p['destination_end'],
                        prefix_token_ids_equal=ids[:p['source_end']]==ids[:p['destination_end']],token_ids=p['token_ids']) for p in prov]
                    if engine and summary['cluster_id']!=ev['cluster']: raise AssertionError('Source/target cluster differs; audit must use actual target cluster')
                    if mode=='LEGACY_POSITION_ALIGNED' and coverage=='max':
                        if [[p['destination_start'],p['destination_end']] for p in prov]!=audit['predicted_legacy_max_spans']:
                            raise AssertionError('Observed legacy selection disagrees with source occurrence audit')
                        audits[audit_id]['audit']['current_run_legacy_max_selection_verified']=True
                    q=quality(reference['logits'],result['logits'])
                    q.update(position_mismatch_count=sum(p['destination_end']-p['destination_start'] for p in prov if not p['positions_identical']),
                        prefix_context_mismatch_count=sum(p['destination_end']-p['destination_start'] for p in spans if not p['prefix_token_ids_equal']),
                        adapter_identity_match=all(p['source_adapter']==user for p in prov))
                    control=None
                    if oracle and a.full_shape_control:
                        from semcache.diagnostics.r1b1_d1 import full_shape_injection
                        with full_shape_injection(adapter,selection['hits']):
                            output=native_request(model,tokenizer,adapter,text,user,False)
                        control=dict(classification='CORRECTNESS_ONLY_FULL_SHAPE_NO_SAVINGS',**quality(reference['logits'],output['logits']))
                        del output
                    selected_key=(mode,str(coverage)); parity=dict(status='UNPAIRED')
                    if selected_key in storage_checks:
                        prior=storage_checks[selected_key]
                        parity=dict(status='PASS' if prior['spans']==spans and torch.allclose(prior['logits'],result['logits'],atol=1e-3,rtol=1e-3) else 'FAIL')
                    storage_checks.setdefault(selected_key,dict(spans=spans,logits=result['logits']))
                    legacy_bytes=engine.cache.physical_tensor_bytes if engine else 0
                    cat_bytes=catalog.resident_bytes if catalog else 0
                    combined=physical_bytes(([t for e in engine.cache.entries.values() for qkv in e.tensors.values() for t in qkv] if engine else [])+(catalog.tensors if catalog else []))
                    refcheck=dict(status='UNVERIFIED')
                    if a.reference_b2_root and mode=='LEGACY_POSITION_ALIGNED' and coverage=='max' and phase=='measured':
                        info=manifest['reference_b2'][str(length)]
                        candidates=[r for r in info['coverage_rows'] if r['repeat']==repeat and r['storage']==('RAW_CPU' if storage_name=='cpu' else 'RAW_GPU')]
                        signature=digest([dict(source=[p['source_start'],p['source_end']],destination=[p['destination_start'],p['destination_end']],tokens_sha256=digest(p['token_ids'])) for p in prov])
                        if len(candidates)==1:
                            refcheck=dict(status='SELECTION_MATCH' if candidates[0]['selection_signature']==signature and candidates[0]['reused_tokens']==reused else 'SELECTION_MISMATCH',
                                runtime_compatible=not info['runtime_mismatches'],code_compatible=not info['code_mismatches'],seed_equal=info['seed_equal'])
                    row=dict(row_id=f'{group}:{mode}:{coverage}',length=length,storage=storage,phase=phase,repeat=repeat,mode=mode,
                        classification='ORACLE_DIAGNOSTIC' if oracle else 'NATIVE_BASELINE' if mode=='FULL_RECOMPUTE' else 'LEGACY',requested_coverage=coverage,actual_coverage=reused/length,
                        reused_tokens=reused,recomputed_tokens=length-reused,selected_windows=len(prov),
                        stored_occurrences=len(catalog.entries) if catalog else len(ev['entries']) if engine else 0,
                        unique_keys=len({e.key for e in catalog.entries.values()}) if catalog else len(ev['entries']) if engine else 0,
                        unique_source_keys=audit['unique_token_window_keys'] if engine else 0,
                        skipped_projection_rows=audit_rows['skipped_projection_rows_all_layers_roles'],native_projection_rows=audit_rows['native_projection_rows_all_layers_roles'],
                        theoretical_upper_tokens=length//3*3,audit_id=audit_id,selected_spans=spans,
                        selection_windows=selection.get('windows'),correctness=q,projection_skip_audit=audit_rows,full_shape_control=control,storage_parity=parity,reference_b2_comparison=refcheck,
                        catalog_physical_bytes=cat_bytes,legacy_physical_bytes=legacy_bytes,legacy_physical_bytes_before_target=legacy_before,combined_physical_bytes=combined,
                        catalog_copy_ms=catalog.copy_ms if catalog else None,catalog_transfer_bytes=catalog.transfer_bytes if catalog else None,
                        catalog_build_ms=catalog_ms,source_request_ms=source_ms,source_build_ms=build_ms,
                        target_warm_ms=summary['timing']['request_wall_ms'],catalog_selection_host_ms=selection.get('selection_host_ms'),
                        source_capture_enabled=oracle,memory_before_source=memory_before,memory_after_build=before_target,memory_after_target=after_target)
                    batch.append(row)
                    selection.pop('hits',None)  # release sidecar views before the next source build
                    del result,engine,catalog
                    guard.evidence()
                bundle=dict(identity=manifest['identity'],status='COMPLETE',rows=batch,audits=audits,checksum=digest(batch),audit_checksum=digest(audits))
                atomic_json(path,bundle)
                bundles=load_groups(a.output_dir,manifest['identity']); reports(a.output_dir,bundles)
                print('COMPLETE',group,flush=True)
    manifest.update(status='COMPLETE',raw_only=guard.evidence(),rows=sum(len(b['rows']) for b in bundles))
    if any(r['storage_parity']['status']=='FAIL' for b in bundles for r in b['rows']): manifest['status']='STORAGE_PARITY_FAILED'
    atomic_json(a.output_dir/'manifest.json',manifest)


def main(argv=None):
    a=args(sys.argv[1:] if argv is None else argv); cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','output_dir')}
    code=b2().provenance(); code['files']['scripts/74_diagnose_r1b3_occurrences.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity=digest(dict(config=cfg,files=code['files'])); path=a.output_dir/'manifest.json'
    if a.resume:
        manifest=json.loads(path.read_text())
        if manifest['identity']!=identity: raise ValueError('Resume rejected: changed code/config')
        load_groups(a.output_dir,identity)
    else:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        manifest=dict(experiment='R1-B3-A',identity=identity,config=cfg,code=code,
            historical_root_cause='UNVERIFIED',settings=dict(w=3,atol=1e-3,rtol=1e-3,use_cache=False,output_attentions=True,attention='eager'),
            catalog_classification='ORACLE_DIAGNOSTIC',timing='minimal R1 request; source detailed construction separate; catalog build outside target')
    manifest.setdefault('attempts',[]).append(dict(previous_status=manifest.get('status'),previous_failure=manifest.get('failure')))
    for k in ('failure','failure_type'): manifest.pop(k,None)
    manifest['status']='RUNNING'; atomic_json(path,manifest)
    try:
        with raw_only_guard() as guard:
            manifest['raw_only']=guard.evidence(); execute(a,manifest,guard)
    except (Exception,KeyboardInterrupt) as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'MEMORY_PREFLIGHT_FAILED' if manifest.get('status')=='MEMORY_PREFLIGHT_FAILED' else 'CATALOG_CAPACITY_EXCEEDED' if 'CATALOG_' in str(exc) and isinstance(exc,MemoryError) else 'OOM' if 'out of memory' in str(exc).lower() else 'FAILED'
        manifest.update(status=status,failure_type=type(exc).__name__,failure=str(exc)); atomic_json(path,manifest)
        reports(a.output_dir,load_groups(a.output_dir,identity))
        print(status,str(exc),file=sys.stderr); return 1
    return 0 if manifest['status']=='COMPLETE' else 2


if __name__=='__main__': raise SystemExit(main())
