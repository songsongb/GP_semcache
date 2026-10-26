#!/usr/bin/env python3
"""Paired B3-B1 occurrence selectors on shared source tensors; no executor edits."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns
from unittest.mock import patch
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard,audit_projection_records
from semcache.metrics.r1b2 import csv_write,stats
from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog,capture_completed_source
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex,HostBook,profile_legacy,compare_selection


def a3():
    spec=importlib.util.spec_from_file_location('r1b3_a',ROOT/'scripts/74_diagnose_r1b3_occurrences.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def args(argv=None):
    from semcache.metrics.r1b2 import coverage_value
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occurrence-selector',choices=('legacy','optimized','both'),default='legacy')
    p.add_argument('--selection-instrumentation',choices=('minimal','breakdown','both'),default='minimal')
    p.add_argument('--profile',choices=('smoke','pilot','full'),default='pilot')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-2.7b')
    p.add_argument('--dtype',choices=('float16','float32'),default='float16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision')
    p.add_argument('--encoder-revision',default='main')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--prompt-lengths',help='Default 128,256 (32 for smoke)')
    p.add_argument('--cache-devices',help='Default cuda for CUDA model, cpu otherwise')
    p.add_argument('--coverages',default='max')
    p.add_argument('--cache-capacity-mib',type=int,default=256)
    p.add_argument('--catalog-capacity-mib',type=int,default=256)
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    try:
        a.lengths=[int(x) for x in (a.prompt_lengths or ('32' if a.profile=='smoke' else '128,256')).split(',')]
        a.storages=(a.cache_devices or ('cuda' if a.device.startswith('cuda') else 'cpu')).split(',')
        a.coverage_values=[coverage_value(x) for x in a.coverages.split(',')]
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else 1 if a.profile=='smoke' else 3
        a.measured_runs=a.measured_runs if a.measured_runs is not None else 2 if a.profile=='smoke' else 10
        if (not a.lengths or min(a.lengths)<4 or len(set(a.lengths))!=len(a.lengths)
            or len(set(a.storages))!=len(a.storages) or any(x not in ('cpu','cuda') for x in a.storages)
            or len(set(a.coverage_values))!=len(a.coverage_values)
            or a.warmup_runs<0 or a.measured_runs<1 or min(a.cache_capacity_mib,a.catalog_capacity_mib)<1
            or 'cuda' in a.storages and not a.device.startswith('cuda')): raise ValueError('Invalid experiment arguments')
        if a.profile!='smoke' and (a.warmup_runs<3 or a.measured_runs<10): raise ValueError('Use >=3/10 runs or explicit smoke')
    except ValueError as exc: p.error(str(exc))
    return a


def clone_source_engine(engine):
    """Independent policy/cluster/history state, same model and raw source tensors."""
    memo={id(v):v for v in (engine.model,engine.tokenizer,engine.adapter,engine.encoder)}
    for entry in engine.cache.entries.values():
        for qkv in entry.tensors.values():
            for tensor in qkv: memo[id(tensor)]=tensor
    return copy.deepcopy(engine,memo)


def checkpoints(root,identity):
    bundles=[]; seen=set()
    for p in sorted((root/'checkpoints').glob('*.json')):
        b=json.loads(p.read_text())
        if b['identity']!=identity or b['status']!='COMPLETE' or digest(b['rows'])!=b['checksum']:
            raise ValueError('Invalid/incomplete checkpoint '+str(p))
        for r in b['rows']:
            if r['row_id'] in seen: raise ValueError('Duplicate row')
            seen.add(r['row_id'])
        bundles.append(b)
    return bundles


def reports(root,bundles):
    rows=[r for b in bundles for r in b['rows']]
    common=('row_id','source_snapshot_id','phase','repeat','length','storage','coverage','selector','instrumentation')
    for filename,fields in {
        'latency_comparison.csv':('selection_ms','target_warm_ms','full_request_ms','source_request_ms','catalog_build_ms','index_build_ms','target_index_validation_ms','breakdown_minus_minimal_request_ms','instrumentation_parity','instrumentation_logit_parity'),
        'coverage_comparison.csv':('actual_coverage','reused_tokens','projection_rows','selected_provenance'),
        'correctness_comparison.csv':('vs_full','vs_legacy','selection_parity','full_error_metrics_equal'),
        'index_storage_accounting.csv':('catalog_bytes','additional_index_memory_bytes','index_memory_estimated','additional_qkv_bytes','index_build_ms')}.items():
        csv_write(root/filename,[{k:r.get(k) for k in (*common,*fields)} for r in rows])
    csv_write(root/'selection_breakdown.csv',[dict(**{k:r[k] for k in common},stage=n,**v)
        for r in rows for n,v in r['breakdown'].items()])
    atomic_json(root/'selection_parity.json',[dict(row_id=r['row_id'],**r['selection_parity']) for r in rows])
    invalid_conditions={(r['length'],r['storage'],str(r['coverage'])) for r in rows
        if r.get('instrumentation_parity',{}).get('passed') is False
        or r.get('instrumentation_logit_parity') and not r['instrumentation_logit_parity']['logit_allclose']}
    groups={}
    for r in rows:
        if r['phase']=='measured': groups.setdefault((r['length'],r['storage'],str(r['coverage']),r['instrumentation']),[]).append(r)
    summary=[]
    for k,group in groups.items():
        legacy={r['repeat']:r for r in group if r['selector']=='legacy'}
        optimized={r['repeat']:r for r in group if r['selector']=='optimized'}
        paired=sorted(set(legacy)&set(optimized))
        item=dict(length=k[0],storage=k[1],coverage=k[2],instrumentation=k[3],primary=k[3]=='minimal',
            count_legacy=len(legacy),count_optimized=len(optimized),pair_count=len(paired))
        for selector,records in [('legacy',legacy),('optimized',optimized)]:
            for metric in ('selection_ms','target_warm_ms'):
                item[selector+'_'+metric]=stats([r[metric] for r in records.values()])
        for metric,name in [('selection_ms','selection_speedup'),('target_warm_ms','warm_request_speedup')]:
            item[name]=(sum(legacy[i][metric] for i in paired)/sum(optimized[i][metric] for i in paired)) if paired else None
            item[metric+'_paired_delta']=stats([optimized[i][metric]-legacy[i][metric] for i in paired])
        passed=k[:3] not in invalid_conditions and bool(paired) and len(paired)==len(legacy)==len(optimized) and all(
            optimized[i]['selection_parity']['passed'] and optimized[i]['vs_legacy']['logit_allclose']
            and legacy[i]['projection_rows']==optimized[i]['projection_rows']
            and legacy[i]['catalog_bytes']==optimized[i]['catalog_bytes'] for i in paired)
        item.update(parity_pass=passed,selection_optimization_success=bool(passed and item['primary'] and item['selection_speedup']>1),
            end_to_end_improvement=bool(passed and item['primary'] and item['warm_request_speedup']>1),
            strict_full_passes=sum(r['vs_full']['logit_allclose'] for r in group),
            full_speedup_legacy=stats([r['full_request_ms']/r['target_warm_ms'] for r in legacy.values()]),
            full_speedup_optimized=stats([r['full_request_ms']/r['target_warm_ms'] for r in optimized.values()]),
            no_safety_claim=True)
        summary.append(item)
    atomic_json(root/'summary.json',dict(groups=summary,classification='ORACLE_DIAGNOSTIC',
        historical_reference_reproduced=False,source_cost_not_amortized_into_warm_latency=True))


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
        driver=a3().b2().legacy().command_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    if manifest.get('runtime') and manifest['runtime']!=runtime: raise ValueError('Resume runtime mismatch')
    manifest.update(runtime=runtime,initialization_ms=(perf_counter_ns()-start)/1e6)
    import semcache.semcache_engine as engine_module
    bundles=checkpoints(a.output_dir,manifest['identity'])
    selectors=['legacy','optimized'] if a.occurrence_selector=='both' else [a.occurrence_selector]
    passes=['minimal','breakdown'] if a.selection_instrumentation=='both' else [a.selection_instrumentation]
    schedule=[('warmup',i) for i in range(a.warmup_runs)]+[('measured',i) for i in range(a.measured_runs)]
    for length in a.lengths:
        trace=controlled_length_trace(tokenizer,length); ids=trace[1]['token_ids']; text=trace[1]['text']; user='user_a'
        if len(ids)!=length: raise ValueError('TOKEN_LENGTH_MISMATCH')
        manifest.setdefault('fixtures',{})[str(length)]=dict(token_ids=ids,token_hash=digest(ids))
        anchors=encoder.encode([trace[0]['text'],trace[-1]['text']])
        for phase,repeat in schedule:
            for storage_name in (a.storages if repeat%2==0 else list(reversed(a.storages))):
                group=f'{length}-{phase}-{repeat}-{storage_name}'; path=a.output_dir/'checkpoints'/(group+'.json')
                if phase=='measured' and path.exists(): continue
                storage=str(next(model.parameters()).device) if storage_name=='cuda' else 'cpu'
                manifest['active_group']=group
                try:
                    manifest['preflight']=memory_preflight(model,length,(a.cache_capacity_mib+a.catalog_capacity_mib)*1024**2,storage,a.device)
                except MemoryError:
                    manifest['status']='MEMORY_PREFLIGHT_FAILED'; raise
                atomic_json(a.output_dir/'manifest.json',manifest)
                cluster=IntentClusterer(2,initialization='first_k',update_mode='buffered',update_interval=100); cluster.initialize(anchors)
                engine=SemCacheEngine(model,tokenizer,adapter,encoder,cluster,GlobalCache(a.cache_capacity_mib*1024**2),
                    storage_device=storage,metadata=metadata,seed=a.seed,impact_reducer='paper_row_l2_sum',
                    impact_execution='token_precompute',capture_impact_values=True,rho=.8,history_lambda=100,pbr_interval_queries=100)
                with capture_completed_source() as captured:
                    source=engine_request(engine,text,user,'source','SEMCACHE_LOOKUP_NO_REUSE',True)
                source_ms=source['summary']['timing']['request_wall_ms']
                start=perf_counter_ns()
                catalog=OccurrenceCatalog(namespace,weight_identity,storage,a.catalog_capacity_mib*1024**2)
                catalog.add_source('source',ids,source['summary']['cluster_id'],user,captured,layout,prefill_complete=True,
                    impacts={w['start']:w['value'] for w in source['summary']['impact_windows']},source_clock=engine.cache.now)
                synchronize(a.device); catalog_ms=(perf_counter_ns()-start)/1e6
                captured.clear(); del source
                index=FastOccurrenceIndex(catalog,layout) if 'optimized' in selectors else None
                # Guarded owners keep same QKV lifetime across all variants.
                state={k:(e.frequency,e.impact,e.created_at,e.last_access,e.updated_at) for k,e in catalog.entries.items()}
                catalog_bytes=catalog.resident_bytes
                reference=native_request(model,tokenizer,adapter,text,user,False)
                cases=[(p,c,s) for p in passes for c in a.coverage_values for s in selectors]
                if repeat%2: cases.reverse()
                batch=[]; results={}
                for instrumentation,coverage,selector in cases:
                    for k,values in state.items():
                        e=catalog.entries[k]; e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=values
                    target=clone_source_engine(engine)
                    book=HostBook() if instrumentation=='breakdown' else None
                    selected={}
                    def select(hits,n,target_ids,target_user):
                        c=next(e['cluster_id'] for e in reversed(target.events) if e['event_type']=='CLUSTER_ASSIGN')
                        start=perf_counter_ns()
                        if selector=='legacy':
                            with profile_legacy(catalog,book) if book else nullcontext():
                                answer=catalog.select(target_ids,c,target_user,namespace,weight_identity,layout,coverage)
                        else:
                            answer=index.select(target_ids,c,target_user,namespace,weight_identity,layout,coverage,book)
                        selected.update(result=answer,selection_ms=(perf_counter_ns()-start)/1e6)
                        return answer[0],answer[1]
                    with patch.object(engine_module,'select_position_aligned_diagnostic',select):
                        result=engine_request(target,text,user,'target','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                    s=result['summary']; hits,mask,windows=selected['result']
                    spans=[i for h in hits for i in range(h.window.start,h.window.end)]
                    projection=audit_projection_records(list(result['projection_audit'].values()),length,len(adapter.layers),sum(mask),spans)
                    if catalog.resident_bytes!=catalog_bytes: raise AssertionError('Catalog residency changed')
                    # Never hold a profile wrapper over model execution or parity.
                    if index: index.guard()
                    full_quality=a3().quality(reference['logits'],result['logits'])
                    row=dict(row_id=f'{group}:{instrumentation}:{coverage}:{selector}',source_snapshot_id=group,phase=phase,repeat=repeat,length=length,
                        storage=storage,coverage=coverage,selector=selector,instrumentation=instrumentation,
                        classification='ORACLE_DIAGNOSTIC',selection_ms=selected['selection_ms'],target_warm_ms=s['timing']['request_wall_ms'],
                        full_request_ms=reference['timing']['request_wall_ms'],source_request_ms=source_ms,catalog_build_ms=catalog_ms,
                        index_build_ms=index.build_ms if index else None,additional_index_memory_bytes=index.additional_index_memory_bytes if index else None,
                        index_memory_estimated=True,additional_qkv_bytes=index.additional_qkv_bytes if index else None,catalog_bytes=catalog_bytes,
                        actual_coverage=sum(mask)/length,reused_tokens=sum(mask),projection_rows=projection,
                        selected_provenance=[dict(source_query=h.entry.qkv_metadata['source_query_id'],source=list(h.entry.positions),
                            target=[h.window.start,h.window.end],token_ids=list(h.window.token_ids)) for h in hits],
                        target_index_validation_ms=(sum(book.values.get(k,0.) for k in ('snapshot_structure_validation','tensor_backing_validation','prefix_context_validation','model_adapter_identity_validation')) if book and selector=='optimized' else None),
                        vs_full=full_quality,vs_legacy=None,selection_parity=dict(passed=None,status='UNPAIRED'),
                        breakdown=book.export() if book else HostBook().export())
                    row['breakdown']['source_only_index_construction']=dict(host_ms=index.build_ms if index else None,
                        calls=1 if index else None,status='MEASURED_SOURCE_OUTSIDE_TARGET' if index else 'UNAVAILABLE')
                    results[(instrumentation,str(coverage),selector)]=(selected['result'],result['logits'],row)
                    batch.append(row); del target,result
                for instrumentation,coverage,selector in cases:
                    current=results[(instrumentation,str(coverage),selector)]
                    legacy=results.get((instrumentation,str(coverage),'legacy'))
                    if legacy:
                        current[2]['selection_parity']=dict(status='PAIRED',**compare_selection(legacy[0],current[0]))
                        current[2]['vs_legacy']=a3().quality(legacy[1],current[1])
                        current[2]['vs_legacy']['exact_equal']=bool(torch.equal(legacy[1],current[1]))
                        current[2]['full_error_metrics_equal']=legacy[2]['vs_full']==current[2]['vs_full']
                    if instrumentation=='breakdown':
                        minimal=results.get(('minimal',str(coverage),selector))
                        if minimal:
                            current[2]['instrumentation_parity']=compare_selection(minimal[0],current[0])
                            current[2]['instrumentation_logit_parity']=a3().quality(minimal[1],current[1])
                            current[2]['breakdown_minus_minimal_request_ms']=current[2]['target_warm_ms']-minimal[2]['target_warm_ms']
                bundle=dict(identity=manifest['identity'],status='COMPLETE',rows=batch,checksum=digest(batch))
                atomic_json(path,bundle); bundles=checkpoints(a.output_dir,manifest['identity']); reports(a.output_dir,bundles)
                guard.evidence(); print('COMPLETE',group,flush=True)
                results.clear(); selected.clear()
                hits=[]; current=None; legacy=None; e=None; select=None
                del catalog,index,engine,reference
    all_rows=[r for b in bundles for r in b['rows']]
    failures=[r['row_id'] for r in all_rows if r['selection_parity']['passed'] is False
        or r['vs_legacy'] and not r['vs_legacy']['logit_allclose']
        or r.get('instrumentation_parity',{}).get('passed') is False
        or r.get('instrumentation_logit_parity') and not r['instrumentation_logit_parity']['logit_allclose']]
    manifest.update(status='PARITY_FAILED' if failures else 'COMPLETE',parity_failures=failures,
        rows=len(all_rows),raw_only=guard.evidence(),
        selector_equivalence_validated=(a.occurrence_selector=='both' and not failures),full_correctness_claim=False)
    # Success is reported per minimal group; missing pairs never imply a pass.
    atomic_json(a.output_dir/'manifest.json',manifest)


def main(argv=None):
    a=args(sys.argv[1:] if argv is None else argv)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','output_dir')}
    code=a3().b2().provenance()
    for name in ('74_diagnose_r1b3_occurrences.py','75_benchmark_r1b3_selection.py'):
        code['files']['scripts/'+name]=hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
    identity=digest(dict(config=cfg,files=code['files'])); path=a.output_dir/'manifest.json'
    if a.resume:
        manifest=json.loads(path.read_text())
        if manifest['identity']!=identity: raise ValueError('Resume rejected: code/config changed')
        checkpoints(a.output_dir,identity)
    else:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        manifest=dict(experiment='R1-B3-B1',identity=identity,config=cfg,code=code,
            classification='ORACLE_DIAGNOSTIC',historical_reference_reproduced=False,
            settings=dict(use_cache=False,output_attentions=True,attention='eager',atol=1e-3,rtol=1e-3),
            timing='minimal request uses original mixed executor; breakdown instruments selector only; source/index costs separate')
    manifest.setdefault('attempts',[]).append(dict(previous_status=manifest.get('status'),failure=manifest.get('failure')))
    for k in ('failure','failure_type'): manifest.pop(k,None)
    manifest['status']='RUNNING'; atomic_json(path,manifest)
    try:
        with raw_only_guard() as guard:
            manifest['raw_only']=guard.evidence(); execute(a,manifest,guard)
    except (Exception,KeyboardInterrupt) as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'MEMORY_PREFLIGHT_FAILED' if manifest.get('status')=='MEMORY_PREFLIGHT_FAILED' else 'OOM' if 'out of memory' in str(exc).lower() else 'FAILED'
        manifest.update(status=status,failure_type=type(exc).__name__,failure=str(exc)); atomic_json(path,manifest)
        reports(a.output_dir,checkpoints(a.output_dir,identity))
        print(status,str(exc),file=sys.stderr); return 1
    return 0 if manifest['status']=='COMPLETE' else 2


if __name__=='__main__': raise SystemExit(main())
