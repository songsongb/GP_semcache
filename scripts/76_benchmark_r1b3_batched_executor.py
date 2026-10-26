#!/usr/bin/env python3
"""R1-B3-B2: fixed Fast Selector, paired original/batched assembly diagnostics."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
from time import perf_counter_ns
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard,audit_projection_records
from semcache.metrics.r1b2 import csv_write,stats
from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog,capture_completed_source
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex,compare_selection
from semcache.edgelora.batched_mixed_projection import (PositionMapping,ExecutorBook,
    batched_mixed_projection_path,observe_legacy,compare_projections)


def b1():
    spec=importlib.util.spec_from_file_location('b1_reference',ROOT/'scripts/75_benchmark_r1b3_selection.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def a3(): return b1().a3()

def args(argv=None):
    from semcache.metrics.r1b2 import coverage_value
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor',choices=('legacy','batched','both'),default='legacy')
    p.add_argument('--instrumentation',choices=('minimal','breakdown','both'),default='minimal')
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
    a.occurrence_selector='optimized'
    return a

checkpoints=b1().checkpoints


def reports(root,bundles):
    rows=[r for b in bundles for r in b['rows']]
    common=('row_id','source_snapshot_id','phase','repeat','length','storage','coverage','executor','instrumentation')
    for filename,fields in {
        'latency_comparison.csv':('selection_ms','target_warm_ms','full_request_ms','source_request_ms',
            'catalog_build_ms','index_build_ms','batch_mapping_ms','executor_context_setup_ms','executor_ms','executor_gpu_ms',
            'fetch_ms','assembly_ms','native_qkv_ms','breakdown_minus_minimal_request_ms'),
        'memory_accounting.csv':('catalog_bytes','legacy_source_cache_bytes','index_metadata_bytes','mapping','memory','gpu_memory'),
        'correctness_comparison.csv':('vs_full','vs_legacy','logits_exact_equal','executor_parity',
            'actual_coverage','reused_tokens')}.items():
        csv_write(root/filename,[{k:r.get(k) for k in (*common,*fields)} for r in rows])
    csv_write(root/'executor_breakdown.csv',[dict(**{k:r[k] for k in common},stage=k,**v)
        for r in rows for k,v in r['breakdown'].items()])
    atomic_json(root/'executor_parity.json',[dict(row_id=r['row_id'],**r['executor_parity']) for r in rows])
    atomic_json(root/'projection_audit.json',[dict(row_id=r['row_id'],audit=r['projection_rows'],
        selected=r['selected_provenance'],mapping=r.get('mapping'),qkv=r['qkv_parity']) for r in rows])
    groups={}
    for r in rows:
        if r['phase']=='measured': groups.setdefault((r['length'],r['storage'],str(r['coverage']),r['instrumentation']),[]).append(r)
    summary=[]
    for key,records in groups.items():
        sides={e:{r['repeat']:r for r in records if r['executor']==e} for e in ('legacy','batched')}
        pairs=sorted(set(sides['legacy'])&set(sides['batched']))
        item=dict(length=key[0],storage=key[1],coverage=key[2],instrumentation=key[3],paired_count=len(pairs),primary=key[3]=='minimal')
        for e,side in sides.items():
            for metric in ('target_warm_ms','selection_ms','executor_ms','executor_gpu_ms','batch_mapping_ms'):
                item[e+'_'+metric]=stats([r[metric] for r in side.values() if r.get(metric) is not None])
            item[e+'_full_speedup']=stats([r['full_request_ms']/r['target_warm_ms'] for r in side.values()])
        item['legacy_warm_ms']=item['legacy_target_warm_ms']
        item['batched_warm_ms']=item['batched_target_warm_ms']
        item['warm_speedup']=sum(sides['legacy'][i]['target_warm_ms'] for i in pairs)/sum(sides['batched'][i]['target_warm_ms'] for i in pairs) if pairs else None
        item['paired_warm_delta_ms']=stats([sides['batched'][i]['target_warm_ms']-sides['legacy'][i]['target_warm_ms'] for i in pairs])
        valid=bool(pairs) and len(pairs)==len(sides['legacy'])==len(sides['batched']) and all(r['executor_parity']['passed'] for r in records)
        item.update(parity_pass=valid,warm_improvement=bool(valid and item['primary'] and item['warm_speedup']>1),
            strict_full_passes=sum(r['vs_full']['logit_allclose'] for r in records),no_safety_claim=True)
        # Full projection timings exist only in the separately instrumented run.
        gpu_pairs=[i for i in pairs if all(sides[e][i].get('executor_gpu_ms') is not None for e in sides)]
        item['executor_gpu_speedup']=sum(sides['legacy'][i]['executor_gpu_ms'] for i in gpu_pairs)/sum(sides['batched'][i]['executor_gpu_ms'] for i in gpu_pairs) if gpu_pairs and sum(sides['batched'][i]['executor_gpu_ms'] for i in gpu_pairs)>0 else None
        summary.append(item)
    atomic_json(root/'summary.json',dict(groups=summary,classification='ORACLE_DIAGNOSTIC',
        historical_reference_reproduced=False,source_cost_not_in_warm=True,
        timing_note='minimal is primary; full_mixed_projection is nested inclusive projection time, not whole transformer'))


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
    import semcache.edgelora.mixed_projection as mixed_module
    original_mixed=mixed_module.mixed_projection_path
    bundles=checkpoints(a.output_dir,manifest['identity'])
    executors=['legacy','batched'] if a.executor=='both' else [a.executor]
    passes=['minimal','breakdown'] if a.instrumentation=='both' else [a.instrumentation]
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
                index=FastOccurrenceIndex(catalog,layout)
                state={k:(e.frequency,e.impact,e.created_at,e.last_access,e.updated_at) for k,e in catalog.entries.items()}
                catalog_bytes=catalog.resident_bytes
                reference=native_request(model,tokenizer,adapter,text,user,False)

                def run_case(executor,instrumentation,coverage,validation=False):
                    # Reset the identical source policy snapshot before EVERY target.
                    for k,values in state.items():
                        e=catalog.entries[k]; e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=values
                    target=b1().clone_source_engine(engine); selected={}; held={}
                    book=ExecutorBook(a.device) if instrumentation=='breakdown' else None
                    def select(hits,n,target_ids,target_user):
                        c=next(e['cluster_id'] for e in reversed(target.events) if e['event_type']=='CLUSTER_ASSIGN')
                        started=perf_counter_ns()
                        answer=index.select(target_ids,c,target_user,namespace,weight_identity,layout,coverage)
                        selected.update(result=answer,selection_ms=(perf_counter_ns()-started)/1e6)
                        return answer[:2]
                    @contextmanager
                    def executor_context(ad,ad_name,hits,n,**kwargs):
                        setup_started=perf_counter_ns()
                        plan=None
                        if executor=='batched':
                            plan=PositionMapping(index,hits,ids,ad_name,namespace,weight_identity)
                            context=batched_mixed_projection_path(ad,ad_name,hits,n,mapping=plan,book=book,**kwargs)
                        else: context=original_mixed(ad,ad_name,hits,n,**kwargs)
                        with observe_legacy(book if executor=='legacy' else None), context as audit:
                            held['setup_ms']=(perf_counter_ns()-setup_started)/1e6
                            yield audit
                        held['audit']=audit
                        held['plan']=plan
                    synchronize(a.device)
                    if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(a.device)
                    before=memory_snapshot(a.device)
                    with patch.object(engine_module,'select_position_aligned_diagnostic',select), patch.object(mixed_module,'mixed_projection_path',executor_context):
                        result=engine_request(target,text,user,'target','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                    after=memory_snapshot(a.device)
                    # Scalar accounting after request synchronization; no extra
                    # baseline per-window accounting loop in the timed region.
                    audit=held.pop('audit'); plan=held.pop('plan')
                    held['memory']=dict(audit.memory) if hasattr(audit,'memory') else dict(
                        temp_tensor_bytes=None,index_tensor_bytes=None,gather_calls=0,scatter_calls=None,kernel_calls=None,
                        window_assignment_calls=len(selected['result'][0])*len(adapter.layers)*3,
                        conversion_calls=len(selected['result'][0])*len(adapter.layers)*3,
                        conversion_copy_calls=sum(int(t.device!=next(model.parameters()).device or t.dtype!=getattr(torch,a.dtype))
                            for h in selected['result'][0] for q in h.entry.tensors.values() for t in q),
                        accounting='legacy copy count inferred from dtype/device; exclusive temporary bytes unavailable')
                    held['mapping']=plan.evidence() if plan else None
                    if validation: held['projections']=audit.projections
                    del audit,plan
                    hits,mask,windows=selected['result']
                    spans=[i for h in hits for i in range(h.window.start,h.window.end)]
                    projection=audit_projection_records(list(result['projection_audit'].values()),length,len(adapter.layers),sum(mask),spans)
                    index.guard()
                    if catalog.resident_bytes!=catalog_bytes: raise AssertionError('Catalog residency changed')
                    breakdown=book.export() if book else ExecutorBook(a.device).export()
                    def stage(name): return breakdown[name]['host_ms']
                    row=dict(row_id=f'{group}:{instrumentation}:{coverage}:{executor}',source_snapshot_id=group,
                        phase=phase,repeat=repeat,length=length,storage=storage,coverage=coverage,executor=executor,
                        selector='OPTIMIZED_OCCURRENCE_SELECTOR',instrumentation=instrumentation,classification='ORACLE_DIAGNOSTIC',
                        target_warm_ms=result['summary']['timing']['request_wall_ms'],selection_ms=selected['selection_ms'],
                        full_request_ms=reference['timing']['request_wall_ms'],source_request_ms=source_ms,
                        catalog_build_ms=catalog_ms,index_build_ms=index.build_ms,catalog_bytes=catalog_bytes,
                        index_metadata_bytes=index.additional_index_memory_bytes,legacy_source_cache_bytes=engine.cache.physical_tensor_bytes,mapping=held['mapping'],
                        batch_mapping_ms=held['mapping']['build_ms'] if held['mapping'] else 0.,
                        executor_context_setup_ms=held['setup_ms'],
                        executor_ms=stage('full_mixed_projection'),executor_gpu_ms=breakdown['full_mixed_projection']['cuda_ms'],
                        fetch_ms=stage('cached_gather'),assembly_ms=stage('cached_scatter'),native_qkv_ms=stage('fresh_native_qkv'),
                        memory=held['memory'],gpu_memory=dict(before=before,after=after),breakdown=breakdown,
                        actual_coverage=sum(mask)/length,reused_tokens=sum(mask),projection_rows=projection,
                        selected_provenance=[dict(source_query=h.entry.qkv_metadata['source_query_id'],source=list(h.entry.positions),
                            target=[h.window.start,h.window.end],tokens=list(h.window.token_ids)) for h in hits],
                        vs_full=a3().quality(reference['logits'],result['logits']),executor_parity=dict(passed=None,status='UNVERIFIED'))
                    return dict(row=row,selection=selected['result'],logits=result['logits'],projections=held.get('projections'))

                cases=[(p,c,e) for p in passes for c in a.coverage_values for e in executors]
                if repeat%2: cases.reverse()
                results={}
                for instrumentation,coverage,executor in cases:
                    results[(instrumentation,str(coverage),executor)]=run_case(executor,instrumentation,coverage)
                # Untimed extra forwards isolate QKV verification memory/hooks from
                # all performance samples. Never serialize projection tensors.
                for coverage in a.coverage_values:
                    validation={}
                    for executor in (['legacy','batched'] if repeat%2==0 else ['batched','legacy']):
                        validation[executor]=run_case(executor,'minimal',coverage,True)
                    left,right=validation['legacy'],validation['batched']
                    selection=compare_selection(left['selection'],right['selection'])
                    qkv=compare_projections(left['projections'],right['projections'],left['selection'][1])
                    quality=a3().quality(left['logits'],right['logits'])
                    exact=torch.equal(left['logits'],right['logits'])
                    validation_pass=bool(selection['passed'] and qkv['passed'] and exact
                        and left['row']['projection_rows']==right['row']['projection_rows'])
                    for instrumentation,c,executor in cases:
                        if c!=coverage: continue
                        current=results[(instrumentation,str(c),executor)]; row=current['row']; control=validation[executor]
                        same=compare_selection(control['selection'],current['selection'])
                        timed_exact=torch.equal(control['logits'],current['logits'])
                        row.update(vs_legacy=a3().quality(left['logits'],current['logits']),
                            logits_exact_equal=torch.equal(left['logits'],current['logits']),qkv_parity=qkv,
                            executor_parity=dict(passed=bool(validation_pass and same['passed'] and timed_exact
                                and control['row']['projection_rows']==row['projection_rows']),
                                validation_selection=selection,validation_qkv_exact=qkv['passed'],
                                validation_logits=quality,validation_logits_exact=exact,
                                timed_selection=same,timed_vs_validation_logits_exact=timed_exact,
                                projection_audit_equal=left['row']['projection_rows']==right['row']['projection_rows'],
                                scope='QKV checked in separate untimed forwards; timing outputs checked exactly'))
                        minimal=results.get(('minimal',str(c),executor))
                        row['breakdown_minus_minimal_request_ms']=row['target_warm_ms']-minimal['row']['target_warm_ms'] if instrumentation=='breakdown' and minimal else None
                    validation.clear(); left=None; right=None; control=None
                batch=[x['row'] for x in results.values()]
                # A failed detailed/validation case disqualifies the entire paired
                # coverage condition, including otherwise-faster minimal samples.
                for coverage in a.coverage_values:
                    failed=any(not r['executor_parity']['passed'] for r in batch if r['coverage']==coverage)
                    if failed:
                        for r in batch:
                            if r['coverage']==coverage: r['executor_parity'].update(passed=False,condition_failure=True)
                atomic_json(path,dict(identity=manifest['identity'],status='COMPLETE',rows=batch,checksum=digest(batch)))
                bundles=checkpoints(a.output_dir,manifest['identity']); reports(a.output_dir,bundles)
                guard.evidence(); print('COMPLETE',group,flush=True)
                results.clear(); current=None; minimal=None; run_case=None
                del catalog,index,engine,reference
    rows=[r for b in bundles for r in b['rows']]
    failures=[r['row_id'] for r in rows if not r['executor_parity']['passed']]
    manifest.update(status='PARITY_FAILED' if failures else 'COMPLETE',parity_failures=failures,
        rows=len(rows),raw_only=guard.evidence(),full_correctness_claim=False)
    atomic_json(a.output_dir/'manifest.json',manifest)


def main(argv=None):
    a=args(argv)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','output_dir')}
    code=a3().b2().provenance()
    for name in ('74_diagnose_r1b3_occurrences.py','75_benchmark_r1b3_selection.py','76_benchmark_r1b3_batched_executor.py'):
        code['files']['scripts/'+name]=hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
    # Explicit even if the inherited provenance glob changes in future.
    for rel in ('src/semcache/edgelora/batched_mixed_projection.py','src/semcache/diagnostics/r1b3_fast_selection.py'):
        code['files'][rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    identity=digest(dict(config=cfg,files=code['files'])); path=a.output_dir/'manifest.json'
    if a.resume:
        manifest=json.loads(path.read_text())
        if manifest['identity']!=identity: raise ValueError('Resume rejected: code/config changed')
        checkpoints(a.output_dir,identity)
    else:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        manifest=dict(experiment='R1-B3-B2',identity=identity,config=cfg,code=code,
            classification='ORACLE_DIAGNOSTIC',historical_reference_reproduced=False,
            settings=dict(use_cache=False,output_attentions=True,attention='eager',atol=1e-3,rtol=1e-3),
            timing='minimal primary; breakdown original legacy traced vs batched spans; validation forwards outside performance; no nested sum')
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
