#!/usr/bin/env python3
"""R1-B3-B2.1: paired setup/mapping reproducibility; ORACLE diagnostic only."""
import argparse, hashlib, importlib.util, itertools, json, math, os, random, subprocess, sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard,audit_projection_records
from semcache.metrics.r1b2 import csv_write,stats
from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog,capture_completed_source
from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex,compare_selection
from semcache.edgelora.batched_mixed_projection import (PositionMapping,ExecutorBook,observe_legacy,
    batched_mixed_projection_path,compare_projections)
from semcache.edgelora.batched_mapping_optimized import (SetupBook,SourceBackingDescriptors,
    PreparedMappingCache,batched_v2_projection_path,balanced_orders,paired_statistics)

def load_b2(name='r1b2_runtime'):
    path=ROOT/'scripts/76_benchmark_r1b3_batched_executor.py'
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def args(argv=None):
    from semcache.metrics.r1b2 import coverage_value
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor',choices=('legacy','v1','v2','contiguous','all'),default='legacy',
                   help='Default preserves the Legacy Window Loop; all runs four paired modes')
    p.add_argument('--instrumentation',choices=('minimal','breakdown'),default='minimal')
    p.add_argument('--profile',choices=('smoke','pilot'),default='pilot')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-2.7b')
    p.add_argument('--dtype',choices=('float16','float32'),default='float16'); p.add_argument('--device',default='cuda:0')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--encoder-revision',default='main')
    p.add_argument('--allow-download',action='store_true'); p.add_argument('--prompt-lengths',default=None)
    p.add_argument('--cache-device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--coverages',default='max')
    p.add_argument('--cache-capacity-mib',type=int,default=256); p.add_argument('--catalog-capacity-mib',type=int,default=256)
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--mapping-cache-capacity',type=int,default=4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--independent-run-id',default='run01'); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    a.lengths=[int(x) for x in (a.prompt_lengths or ('32' if a.profile=='smoke' else '128,256')).split(',')]
    a.modes=['legacy','v1','v2','contiguous'] if a.executor=='all' else [a.executor]
    a.scheduled_modes=[*a.modes,'full']
    a.coverage_values=[coverage_value(x) for x in a.coverages.split(',')]
    if len(a.coverage_values)!=1: p.error('Use one coverage per run; combine independent output directories for sweeps')
    a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.profile=='smoke' else 3)
    a.measured_runs=a.measured_runs if a.measured_runs is not None else (2 if a.profile=='smoke' else 30)
    if (not a.lengths or min(a.lengths)<4 or len(set(a.lengths))!=len(a.lengths) or not a.independent_run_id
        or len(set(a.coverage_values))!=len(a.coverage_values) or a.warmup_runs<0 or a.measured_runs<1
        or a.mapping_cache_capacity<1 or min(a.cache_capacity_mib,a.catalog_capacity_mib)<1
        or a.cache_device=='cuda' and not a.device.startswith('cuda')): p.error('Invalid R1-B3-B2.1 configuration')
    if a.profile=='pilot' and (a.warmup_runs<3 or a.measured_runs<30): p.error('Pilot requires >=3 warmup and >=30 paired iterations')
    return a

def checkpoint_rows(root,identity):
    rows=[]; seen=set()
    for file in sorted((root/'checkpoints').glob('*.json')):
        bundle=json.loads(file.read_text())
        if bundle.get('identity')!=identity or digest(bundle.get('rows',[]))!=bundle.get('checksum'):
            raise ValueError('Invalid B2.1 checkpoint identity/checksum')
        for row in bundle.get('rows',[]):
            if row['row_id'] in seen: raise ValueError('Duplicate B2.1 row id')
            seen.add(row['row_id']); rows.append(row)
    return rows

def csv_reports(root,rows):
    common=('row_id','source_snapshot_id','independent_run_id','phase','repeat','order','length','storage','coverage','mode','instrumentation')
    csv_write(root/'latency_comparison.csv',[{k:r.get(k) for k in (*common,'selection_ms','mapping_ms','context_setup_ms','mapping_setup_ms',
        'target_warm_ms','full_request_ms','speedup_vs_full','source_prefill_ms','catalog_build_ms','fast_index_build_ms',
        'descriptor_build_ms','mapping_cache_hit','mapping_cache_hits','mapping_cache_rate')} for r in rows])
    csv_write(root/'paired_differences.csv',[r for r in make_pairs(rows)])
    csv_write(root/'setup_breakdown.csv',[dict(**{k:r.get(k) for k in common},stage=stage,**value)
        for r in rows for stage,value in r.get('setup_breakdown',{}).items()])
    csv_write(root/'memory_accounting.csv',[{k:r.get(k) for k in (*common,'catalog_physical_bytes','source_policy_bytes','descriptor_python_bytes',
        'mapping_python_bytes','mapping_cpu_index_bytes','mapping_index_tensor_bytes','mapping_cache','temporary_memory','gpu_memory')} for r in rows])
    csv_write(root/'correctness_comparison.csv',[{k:r.get(k) for k in (*common,'full_correctness','executor_parity','actual_coverage','reused_tokens',
        'native_rows','skipped_rows')} for r in rows])

def make_pairs(rows):
    groups={}
    for r in rows:
        if r.get('phase')=='measured': groups.setdefault((r['source_snapshot_id'],r['repeat'],r['length'],r['storage'],str(r['coverage']),r['instrumentation']),{})[r['mode']]=r
    output=[]
    for key,modes in groups.items():
        base=modes.get('legacy')
        if base is None: continue
        for mode,row in modes.items():
            if mode=='legacy': continue
            output.append(dict(source_snapshot_id=key[0],repeat=key[1],length=key[2],storage=key[3],coverage=key[4],
                instrumentation=key[5],baseline_mode='legacy',optimized_mode=mode,baseline_ms=base['target_warm_ms'],
                optimized_ms=row['target_warm_ms'],paired_difference_ms=base['target_warm_ms']-row['target_warm_ms'],
                optimized_win=base['target_warm_ms']>row['target_warm_ms'],parity_pass=row.get('executor_parity',{}).get('passed')))
    return output

def common_group_source(baseline,target):
    values=list(target.values())
    return values[0]['source_snapshot_id'] if values else None

def _first_snapshot_rows(rows):
    seen=set(); result=[]
    for row in rows:
        key=row['source_snapshot_id']
        if key not in seen: seen.add(key); result.append(row)
    return result

def summarize(rows,seed):
    groups={}
    for row in rows:
        if row.get('phase')=='measured': groups.setdefault((row['length'],row['storage'],str(row['coverage']),row['instrumentation']),{}).setdefault(row['mode'],[]).append(row)
    summary=[]
    for key,modes in groups.items():
        by_mode={m:{r['repeat']:r for r in rr} for m,rr in modes.items()}
        item=dict(length=key[0],storage=key[1],coverage=key[2],instrumentation=key[3],independent_run_count=1,
                  classification='ORACLE_DIAGNOSTIC',statistical_conclusion='DESCRIPTIVE_SINGLE_RUN')
        for mode,values in by_mode.items():
            lat=[r['target_warm_ms'] for r in values.values()]
            item[mode+'_latency']=stats(lat)
            item[mode+'_latency'].update(min=min(lat) if lat else None,max=max(lat) if lat else None)
            vals=sorted(lat); q1=vals[len(vals)//4] if vals else None; q3=vals[(3*len(vals))//4] if vals else None
            lo=q1-1.5*(q3-q1) if vals else None; hi=q3+1.5*(q3-q1) if vals else None
            for r in values.values(): r['outlier_iqr']=bool(lo is not None and (r['target_warm_ms']<lo or r['target_warm_ms']>hi))
            item[mode+'_outlier_repeats']=[r['repeat'] for r in values.values() if r['outlier_iqr']]
        baseline=by_mode.get('legacy',{})
        for mode,target in by_mode.items():
            common=sorted(set(baseline)&set(target))
            if mode=='legacy' or not common: continue
            # FULL is a separately timed reference anchor in the same balanced
            # schedule, never an optimized executor or a parity-qualified path.
            if mode=='full':
                item['full_recompute_paired_against_legacy']=paired_statistics(
                    [baseline[i]['target_warm_ms'] for i in common],
                    [target[i]['target_warm_ms'] for i in common],seed+key[0]+len(mode))
                continue
            diffs=[baseline[i]['target_warm_ms']-target[i]['target_warm_ms'] for i in common]
            pair=paired_statistics([baseline[i]['target_warm_ms'] for i in common],[target[i]['target_warm_ms'] for i in common],seed+key[0]+len(mode))
            passes=all(baseline[i].get('executor_parity',{}).get('passed',True) and target[i].get('executor_parity',{}).get('passed',False) for i in common)
            item[mode+'_paired']=pair; item[mode+'_warm_speedup']=sum(baseline[i]['target_warm_ms'] for i in common)/sum(target[i]['target_warm_ms'] for i in common)
            item[mode+'_parity_pass']=passes; item[mode+'_warm_improvement']=bool(passes and pair['mean']>0)
            full=[target[i]['full_request_ms']/target[i]['target_warm_ms'] for i in common]
            item[mode+'_full_speedup']=stats(full)
            if mode in ('v2','contiguous') and 'v1' in by_mode:
                warm_saving=sum(by_mode['v1'][i]['target_warm_ms']-by_mode[mode][i]['target_warm_ms'] for i in common)/len(common)
                source_extra=max(r.get('descriptor_build_ms',0.) for r in target.values())
                cold_v1=next((r['mapping_ms'] for r in rows if r.get('phase')=='warmup' and r.get('source_snapshot_id')==common_group_source(by_mode,target) and r.get('mode')=='v1'),0.)
                cold_new=next((r['mapping_ms'] for r in rows if r.get('phase')=='warmup' and r.get('source_snapshot_id')==common_group_source(by_mode,target) and r.get('mode')==mode),0.)
                fixed=max(0.,cold_new-cold_v1)+source_extra
                item[mode+'_amortization']=dict(steady_paired_warm_saving_ms=warm_saving,additional_descriptor_build_ms=source_extra,
                    incremental_first_mapping_cost_ms=max(0.,cold_new-cold_v1),
                    break_even_target_count=(math.ceil(fixed/warm_saving) if warm_saving>0 else None),
                    status='DERIVED_FROM_MEASURED_RUN; ignores other common source costs' if warm_saving>0 else 'NO_POSITIVE_MEASURED_WARM_SAVING')
        summary.append(item)
    return dict(groups=summary,bootstrap_seed=seed,bootstrap_scope='paired repeats within each independent run; use aggregator for run-level clusters',
        source_build_not_in_warm=True,source_costs=[dict(source_snapshot_id=r['source_snapshot_id'],source_prefill_ms=r['source_prefill_ms'],
            catalog_build_ms=r['catalog_build_ms'],fast_index_build_ms=r['fast_index_build_ms'],descriptor_build_ms=r['descriptor_build_ms'],
            additional_descriptor_memory_bytes=r['descriptor_memory_bytes']) for r in _first_snapshot_rows(rows)],strict_full_tolerance=dict(atol=1e-3,rtol=1e-3),no_safety_claim=True)

def gpu_snapshot(device):
    timestamp=datetime.now(timezone.utc).isoformat()
    query='name,uuid,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw'
    try:
        result=subprocess.run(['nvidia-smi',f'--query-gpu={query}','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        procs=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=False)
        fields=[x.strip() for x in result.stdout.strip().splitlines()[0].split(',')]
        return dict(timestamp_utc=timestamp,gpu_name=fields[0],gpu_uuid=fields[1],utilization_percent=fields[2],memory_used_mib=fields[3],
            sm_clock_mhz=fields[4],temperature_c=fields[5],power_w=fields[6],other_processes=procs.stdout.strip().splitlines() if procs.returncode==0 else None,
            monitoring='nvidia-smi sampled outside target request')
    except Exception as exc: return dict(timestamp_utc=timestamp,gpu_uuid=None,utilization_percent=None,memory_used_mib=None,
        sm_clock_mhz=None,temperature_c=None,power_w=None,other_processes=None,reason=str(exc))

def load_context(): return load_b2()

@contextmanager
def measured_context(context,book):
    """Measure only context enter/exit; never time the model forward as setup."""
    import sys as _sys
    with book.span('module_forward_patch_setup'):
        audit=context.__enter__()
    try:
        yield audit
    except BaseException:
        exc=_sys.exc_info()
        with book.span('context_cleanup'): context.__exit__(*exc)
        raise
    else:
        with book.span('context_cleanup'): context.__exit__(None,None,None)


def execute(a,manifest,raw_guard):
    import torch
    b=load_context(); a3=b.a3()
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.cache.global_cache import GlobalCache
    from semcache.semcache_engine import SemCacheEngine
    from semcache.metrics.scaling_workload import controlled_length_trace
    from semcache.metrics.r1_runtime import engine_request,native_request,synchronize,memory_preflight,memory_snapshot
    from semcache.utils.seed import seed_everything
    import semcache.semcache_engine as em; import semcache.edgelora.mixed_projection as mm
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; NOT EXECUTED')
    seed_everything(a.seed); init=perf_counter_ns()
    model,tokenizer,meta=load_model(dict(name=a.model,tokenizer=a.model,revision=a.revision,tokenizer_revision=a.tokenizer_revision or a.revision,
        dtype=a.dtype,device=a.device,local_files_only=not a.allow_download,attention_implementation='eager'))
    if getattr(model,'is_quantized',False) or getattr(model.config,'quantization_config',None): raise ValueError('Quantization forbidden')
    model,lora=create_controlled_users(model); adapter=OPTModelAdapter(model)
    encoder=TinyBERTSemanticEncoder(device=a.device,dtype=a.dtype,revision=a.encoder_revision,local_files_only=not a.allow_download); synchronize(a.device)
    lora_hash={n:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,x in model.named_parameters() if 'lora_' in n and 'user_a' in n}
    namespace=digest({k:v for k,v in meta.items() if k not in ('device','dtype','torch_version','transformers_version')})
    weights=digest(dict(model_revision=meta.get('resolved_model_revision'),adapter_weights=lora_hash))
    if not meta.get('resolved_model_revision'): raise ValueError('Resolved model revision required')
    layout={l:tuple((adapter.projection_modules(l)[r].out_features,adapter.projection_modules(l)[r].get_base_layer().weight.dtype) for r in 'qkv') for l in range(len(adapter.layers))}
    runtime=dict(model=meta,lora=lora,encoder=encoder.metadata,namespace=namespace,weight_identity=weights,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,gpu_uuid=gpu_snapshot(a.device).get('gpu_uuid'),
        initialization_ms=(perf_counter_ns()-init)/1e6)
    if manifest.get('runtime') and manifest['runtime']!=runtime: raise ValueError('Resume runtime/provenance mismatch')
    manifest['runtime']=runtime
    gpu_records=[]
    for length in a.lengths:
        trace=controlled_length_trace(tokenizer,length); ids=trace[1]['token_ids']; text=trace[1]['text']; user='user_a'
        if len(ids)!=length: raise ValueError('TOKEN_LENGTH_MISMATCH')
        manifest.setdefault('fixtures',{})[str(length)]=dict(token_ids=ids,token_hash=digest(ids))
        anchors=encoder.encode([trace[0]['text'],trace[-1]['text']])
        for store in [a.cache_device]:
            storage=str(next(model.parameters()).device) if store=='cuda' else 'cpu'
            group=f"{a.independent_run_id}-{length}-{store}-{a.instrumentation}"
            checkpoint=a.output_dir/'checkpoints'/f'{group}.json'
            old=[]
            if a.resume and checkpoint.exists():
                saved=json.loads(checkpoint.read_text())
                if saved.get('identity')!=manifest['identity'] or digest(saved.get('rows',[]))!=saved.get('checksum'):
                    raise ValueError('Checkpoint identity/checksum invalid')
                if saved['status']=='COMPLETE': continue
                old=[r for r in saved['rows'] if r.get('phase')!='warmup']
                if saved['status']=='COMPLETE': old=saved['rows']
            try: manifest['memory_preflight']=memory_preflight(model,length,(a.cache_capacity_mib+a.catalog_capacity_mib)*1024**2,storage,a.device)
            except MemoryError: manifest['status']='MEMORY_PREFLIGHT_FAILED'; raise
            atomic_json(a.output_dir/'manifest.json',manifest)
            cluster=IntentClusterer(2,initialization='first_k',update_mode='buffered',update_interval=100); cluster.initialize(anchors)
            engine=SemCacheEngine(model,tokenizer,adapter,encoder,cluster,GlobalCache(a.cache_capacity_mib*1024**2),storage_device=storage,
                metadata=meta,seed=a.seed,impact_reducer='paper_row_l2_sum',impact_execution='token_precompute',capture_impact_values=True,rho=.8,history_lambda=100,pbr_interval_queries=100)
            with capture_completed_source() as captured: source=engine_request(engine,text,user,'source','SEMCACHE_LOOKUP_NO_REUSE',True)
            source_ms=source['summary']['timing']['request_wall_ms']; t=perf_counter_ns()
            catalog=OccurrenceCatalog(namespace,weights,storage,a.catalog_capacity_mib*1024**2)
            catalog.add_source('source',ids,source['summary']['cluster_id'],user,captured,layout,prefill_complete=True,
                impacts={w['start']:w['value'] for w in source['summary']['impact_windows']},source_clock=engine.cache.now)
            synchronize(a.device); catalog_ms=(perf_counter_ns()-t)/1e6; captured.clear(); del source
            t=perf_counter_ns(); index=FastOccurrenceIndex(catalog,layout); index_ms=(perf_counter_ns()-t)/1e6
            descriptor_book=SetupBook() if a.instrumentation=='breakdown' and any(m in ('v2','contiguous') for m in a.modes) else None
            descriptor=SourceBackingDescriptors(index,descriptor_book) if any(m in ('v2','contiguous') for m in a.modes) else None
            b_original=mm.mixed_projection_path
            map_cache=PreparedMappingCache(descriptor,a.mapping_cache_capacity) if descriptor else None
            source_state={k:(e.frequency,e.impact,e.created_at,e.last_access,e.updated_at) for k,e in catalog.entries.items()}
            resident_bytes=catalog.resident_bytes; reference=native_request(model,tokenizer,adapter,text,user,False)
            orders=balanced_orders(a.scheduled_modes,a.warmup_runs+a.measured_runs,a.seed+length)
            manifest.setdefault('execution_schedules',{})[group]=dict(order=[[x for x in order] for order in orders],seed=a.seed+length)
            gpu_records.append(dict(group=group,phase='before_measurement_block',snapshot=gpu_snapshot(a.device)))
            atomic_json(a.output_dir/'gpu_state.json',gpu_records)
            rows=list(old); done={('measured',r['repeat']) for r in old if r.get('phase')=='measured'}
            for global_repeat,order in enumerate(orders):
                phase='warmup' if global_repeat<a.warmup_runs else 'measured'; repeat=global_repeat if phase=='warmup' else global_repeat-a.warmup_runs
                if (phase,repeat) in done: continue
                pair=[]; pair_logits={}; pair_selections={}
                for mode in order:
                    if mode=='full':
                        full_result=native_request(model,tokenizer,adapter,text,user,False)
                        pair.append(dict(row_id=f'{group}:{phase}:{repeat}:full',source_snapshot_id=group,
                            independent_run_id=a.independent_run_id,phase=phase,repeat=repeat,order=order,length=length,
                            storage=storage,coverage=a.coverage_values[0],mode='full',instrumentation=a.instrumentation,
                            classification='FULL_RECOMPUTE',selection_ms=None,mapping_ms=None,context_setup_ms=None,mapping_setup_ms=None,
                            target_warm_ms=full_result['timing']['request_wall_ms'],full_request_ms=full_result['timing']['request_wall_ms'],
                            speedup_vs_full=1.,source_prefill_ms=source_ms,catalog_build_ms=catalog_ms,fast_index_build_ms=index_ms,
                            descriptor_build_ms=descriptor.build_ms if descriptor else None,descriptor_memory_bytes=descriptor.python_bytes_estimate if descriptor else None,
                            mapping_cache_hit=None,mapping_cache_hits=map_cache.hits if map_cache else 0,mapping_cache_rate=map_cache.stats()['hit_rate'] if map_cache else None,
                            mapping_cache=map_cache.stats() if map_cache else None,catalog_physical_bytes=catalog.resident_bytes,
                            source_policy_bytes=engine.cache.physical_tensor_bytes,descriptor_python_bytes=descriptor.python_bytes_estimate if descriptor else None,
                            mapping_python_bytes=None,mapping_cpu_index_bytes=None,mapping_index_tensor_bytes=None,temporary_memory=None,
                            gpu_memory=None,mapping=None,actual_coverage=0.,reused_tokens=0,native_rows=None,skipped_rows=0,
                            projection_audit=None,selected_provenance=[],full_correctness=dict(status='NATIVE_REFERENCE'),
                            executor_parity=dict(passed=None,status='FULL_RECOMPUTE_REFERENCE'),setup_breakdown={},executor_breakdown={}))
                        pair_logits['full']=full_result['logits']; pair_full_ms=full_result['timing']['request_wall_ms']
                        del full_result
                        continue
                    setup_book=SetupBook() if a.instrumentation=='breakdown' else None
                    validation=False
                    for key,values in source_state.items():
                        e=catalog.entries[key]; e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=values
                    target=b.clone_source_engine(engine); selected={}; held={}; executor_book=ExecutorBook(a.device) if a.instrumentation=='breakdown' else None
                    def select(hits,n,target_ids,target_user):
                        cid=next(x['cluster_id'] for x in reversed(target.events) if x['event_type']=='CLUSTER_ASSIGN')
                        st=perf_counter_ns(); answer=index.select(target_ids,cid,target_user,namespace,weights,layout,a.coverage_values[0])
                        selected['selection_ms']=(perf_counter_ns()-st)/1e6; selected['result']=answer
                        selected['mapping_started']=perf_counter_ns()
                        if mode in ('v2','contiguous'):
                            prepared,hit=map_cache.get(answer[0],target_ids,target_user,namespace,weights,a.device,torch.get_default_dtype() if a.dtype=='float32' else torch.float16,
                                setup_book,mode='contiguous' if mode=='contiguous' else 'gather',validate=False)
                            selected['prepared']=prepared; selected['mapping_cache_hit']=hit
                        return answer[:2]
                    @contextmanager
                    def chosen_executor(ad,ad_name,hits,n,**kwargs):
                        plan=None
                        if mode=='legacy': ctx=b_original(ad,ad_name,hits,n,**kwargs)
                        elif mode=='v1':
                            map_start=perf_counter_ns(); plan=PositionMapping(index,hits,ids,ad_name,namespace,weights)
                            selected['mapping_ms']=(perf_counter_ns()-map_start)/1e6
                            ctx=batched_mixed_projection_path(ad,ad_name,hits,n,mapping=plan,book=executor_book,**kwargs)
                        else:
                            plan=selected['prepared']; ctx=batched_v2_projection_path(ad,ad_name,hits,n,mapping=plan,
                                setup_book=setup_book,executor_book=executor_book,contiguous=(mode=='contiguous'))
                            selected['mapping_ms']=(perf_counter_ns()-selected['mapping_started'])/1e6
                        ctx_start=perf_counter_ns()
                        observed=observe_legacy(executor_book) if mode=='legacy' and executor_book else _null_context()
                        timed=measured_context(ctx,setup_book) if setup_book and mode in ('legacy','v1') else ctx
                        with observed:
                            with timed as audit:
                                selected['context_before_forward_ms']=(perf_counter_ns()-ctx_start)/1e6
                                yield audit
                                cleanup_start=perf_counter_ns()
                        selected['context_cleanup_ms']=(getattr(audit,'context_cleanup_ms',None) if mode in ('v2','contiguous') else (perf_counter_ns()-cleanup_start)/1e6)
                        if validation: held['projections']=audit.projections
                        held['memory']=getattr(audit,'memory',{})
                    synchronize(a.device)
                    if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(a.device)
                    mem_before=memory_snapshot(a.device)
                    with patch.object(em,'select_position_aligned_diagnostic',select),patch.object(mm,'mixed_projection_path',chosen_executor):
                        result=engine_request(target,text,user,f'target-{phase}-{repeat}','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                    mem_after=memory_snapshot(a.device)
                    hits,mask,window_rows=selected['result']; spans=[p for h in hits for p in range(h.window.start,h.window.end)]
                    row_audit=audit_projection_records(list(result['projection_audit'].values()),length,len(adapter.layers),sum(mask),spans)
                    index.guard()
                    if catalog.resident_bytes!=resident_bytes: raise AssertionError('Catalog physical residency changed')
                    breakdown=executor_book.export() if executor_book else ExecutorBook(a.device).export()
                    setup_export=setup_book.export() if setup_book else {k:dict(host_ms=None,calls=None,status='NOT_INSTRUMENTED') for k in SetupBook.STAGES}
                    if descriptor_book and mode in ('v2','contiguous') and repeat==0 and phase=='warmup':
                        setup_export['source_backing_descriptor_preparation']=dict(host_ms=descriptor.build_ms,cuda_ms=None,calls=1,status='SOURCE_ONLY_ONCE_PER_SNAPSHOT')
                    setup_export['warm_request']=dict(host_ms=result['summary']['timing']['request_wall_ms'],calls=1,status='ENCLOSING_REQUEST')
                    setup_export['mapping_setup_total']=dict(host_ms=selected.get('mapping_ms',0.)+selected.get('context_before_forward_ms',0.)+selected.get('context_cleanup_ms',0.),
                        calls=1,status='INCLUSIVE_TARGET_MAPPING_PLUS_CONTEXT; HOST STOPWATCH')
                    mapping=selected.get('prepared')
                    map_evidence=(mapping.evidence() if mode=='v1' else dict(selected_sequence=list(mapping.selected_sequence),
                        groups=[dict(query=g['query'],source=g['source'],target=g['target'],contiguous=g['is_contiguous']) for g in mapping.groups],
                        reused=mapping.reused,cache_key_exact=True) if mapping else None)
                    cache_state=map_cache.stats() if map_cache else dict(hits=0,misses=0,hit_rate=None,entries=0,estimated_memory_bytes=0,evictions=0)
                    row=dict(row_id=f'{group}:{phase}:{repeat}:{mode}',source_snapshot_id=group,independent_run_id=a.independent_run_id,
                        phase=phase,repeat=repeat,order=order,length=length,storage=storage,coverage=a.coverage_values[0],mode=mode,
                        instrumentation=a.instrumentation,classification='ORACLE_DIAGNOSTIC',selection_ms=selected['selection_ms'],
                        mapping_ms=selected.get('mapping_ms',0.),context_setup_ms=selected.get('context_before_forward_ms'),
                        mapping_setup_ms=setup_export['mapping_setup_total'].get('host_ms'),
                        target_warm_ms=result['summary']['timing']['request_wall_ms'],full_request_ms=reference['timing']['request_wall_ms'],
                        speedup_vs_full=reference['timing']['request_wall_ms']/result['summary']['timing']['request_wall_ms'],
                        source_prefill_ms=source_ms,catalog_build_ms=catalog_ms,fast_index_build_ms=index_ms,
                        descriptor_build_ms=descriptor.build_ms if descriptor else None,descriptor_memory_bytes=descriptor.python_bytes_estimate if descriptor else None,
                        mapping_cache_hit=selected.get('mapping_cache_hit'),mapping_cache_hits=cache_state['hits'],mapping_cache_rate=cache_state['hit_rate'],
                        mapping_cache=cache_state,catalog_physical_bytes=catalog.resident_bytes,source_policy_bytes=engine.cache.physical_tensor_bytes,
                        descriptor_python_bytes=descriptor.python_bytes_estimate if descriptor else None,mapping_python_bytes=mapping.python_bytes_estimate if mapping else None,
                        mapping_cpu_index_bytes=mapping.cpu_index_bytes if mapping else None,
                        mapping_index_tensor_bytes=sum(x.numel()*x.element_size() for x in mapping.index_tensors.values()) if mapping else None,
                        temporary_memory=held['memory'],gpu_memory=dict(before=mem_before,after=mem_after),mapping=map_evidence,
                        actual_coverage=sum(mask)/length,reused_tokens=sum(mask),native_rows=row_audit['native_projection_rows_all_layers_roles'],
                        skipped_rows=row_audit['skipped_projection_rows_all_layers_roles'],projection_audit=row_audit,
                        selected_provenance=[dict(query=h.entry.qkv_metadata['source_query_id'],source=list(h.entry.positions),target=[h.window.start,h.window.end],tokens=list(h.window.token_ids)) for h in hits],
                        full_correctness=a3.quality(reference['logits'],result['logits']),executor_parity=dict(passed=None,status='PENDING_PAIRED_GATE'),
                        setup_breakdown=setup_export,executor_breakdown=breakdown)
                    if validation: row['_validation_projections']=held['projections']
                    pair.append(row); pair_logits[mode]=result['logits']; pair_selections[mode]=selected['result']; del target,result
                # Paired exact comparison of selections/logits for each repeat.
                by={r['mode']:r for r in pair}; base=by.get('legacy')
                for r in pair:
                    if r['mode']=='full':
                        r['executor_parity']=dict(passed=None,status='FULL_RECOMPUTE_REFERENCE')
                        continue
                    r['full_request_ms']=pair_full_ms; r['speedup_vs_full']=pair_full_ms/r['target_warm_ms']
                    r['full_correctness']=a3.quality(pair_logits['full'],pair_logits[r['mode']])
                    parity=dict(passed=None,status='AWAITING_UNTIMED_VALIDATION')
                    if base and r is base: parity=dict(passed=True,status='LEGACY_REFERENCE')
                    if base and r is not base:
                        sel=compare_selection(pair_selections['legacy'],pair_selections[r['mode']])
                        logits_eq=torch.equal(pair_logits['legacy'],pair_logits[r['mode']])
                        row_eq=base['projection_audit']==r['projection_audit']
                        parity=dict(passed=bool(sel['passed'] and logits_eq and row_eq),selection=sel,
                            projection_row_audit_equal=row_eq,paired_logits_exact=logits_eq,status='PAIRED_SAMPLE_CHECK')
                    r['executor_parity']=parity
                    rows.append(r)
                pair_logits.clear(); pair_selections.clear()
                atomic_json(checkpoint,dict(identity=manifest['identity'],status='RUNNING',rows=rows,checksum=digest(rows)))
                live_rows=checkpoint_rows(a.output_dir,manifest['identity'])
                csv_reports(a.output_dir,live_rows); atomic_json(a.output_dir/'summary.json',summarize(live_rows,a.seed))
            gpu_records.append(dict(group=group,phase='after_measurement_block',snapshot=gpu_snapshot(a.device)))
            atomic_json(a.output_dir/'gpu_state.json',gpu_records)
            atomic_json(checkpoint,dict(identity=manifest['identity'],status='RUNNING',rows=rows,checksum=digest(rows)))
            # Exact QKV and final-logit validation is separate from all latency samples.
            validation_rows=[]; val_ids=[r['repeat'] for r in rows if r['phase']=='measured']; validation=False
            # Run one additional validation pass per mode using the same immutable source.
            # This closure reuses the already-built exact-target mapping cache.
            val_outputs={}; val_projections={}; val_selections={}; val_projection_audits={}; val_masks={}
            validation_repeat='validation'
            for mode in a.modes:
                setup_book=None
                for key,values in source_state.items():
                    e=catalog.entries[key]; e.frequency,e.impact,e.created_at,e.last_access,e.updated_at=values
                target=b.clone_source_engine(engine); selected={}; held={}; executor_book=None; validation=True
                def select_validation(hits,n,target_ids,target_user):
                    cid=next(x['cluster_id'] for x in reversed(target.events) if x['event_type']=='CLUSTER_ASSIGN')
                    answer=index.select(target_ids,cid,target_user,namespace,weights,layout,a.coverage_values[0]); selected['result']=answer
                    if mode in ('v2','contiguous'):
                        selected['prepared'],_=map_cache.get(answer[0],target_ids,target_user,namespace,weights,a.device,torch.float16 if a.dtype=='float16' else torch.float32,mode='contiguous' if mode=='contiguous' else 'gather',validate=False)
                    return answer[:2]
                @contextmanager
                def validation_executor(ad,ad_name,hits,n,**kwargs):
                    if mode=='legacy': ctx=b_original(ad,ad_name,hits,n,**kwargs)
                    elif mode=='v1':
                        plan=PositionMapping(index,hits,ids,ad_name,namespace,weights); ctx=batched_mixed_projection_path(ad,ad_name,hits,n,mapping=plan)
                    else: ctx=batched_v2_projection_path(ad,ad_name,hits,n,mapping=selected['prepared'],contiguous=mode=='contiguous')
                    with ctx as audit: yield audit
                    held['projections']=audit.projections
                with patch.object(em,'select_position_aligned_diagnostic',select_validation),patch.object(mm,'mixed_projection_path',validation_executor):
                    res=engine_request(target,text,user,'target-validation','SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC',False)
                val_outputs[mode]=res['logits']; val_projections[mode]=held.pop('projections'); val_selections[mode]=selected['result']; val_masks[mode]=selected['result'][1]
                val_audit=audit_projection_records(list(res['projection_audit'].values()),length,len(adapter.layers),sum(selected['result'][1]),
                    [p for h in selected['result'][0] for p in range(h.window.start,h.window.end)])
                val_projection_audits[mode]=val_audit
                val_summary=dict(mode=mode,coverage=sum(selected['result'][1])/length,projection_audit=val_audit,
                    full_correctness=a3.quality(reference['logits'],res['logits']))
                validation_rows.append(val_summary)
                del target,res
            executor_validation=[]; selection_validation={}
            if 'legacy' in val_outputs:
                for mode in a.modes:
                    if mode=='legacy': continue
                    selection=compare_selection(val_selections['legacy'],val_selections[mode])
                    qkv=compare_projections(val_projections['legacy'],val_projections[mode],val_masks['legacy'])
                    logits_equal=torch.equal(val_outputs['legacy'],val_outputs[mode])
                    rows_equal=val_projection_audits['legacy']==val_projection_audits[mode]
                    selection_validation[mode]=selection
                    executor_validation.append(dict(mode=mode,selection=selection,qkv=qkv,logits_equal=logits_equal,projection_rows_equal=rows_equal))
            # Per-repeat policy/run rows are gated by held validation equality, captured separately.
            validation_pass={x['mode']:bool(x['selection']['passed'] and x['qkv']['passed'] and x['logits_equal'] and x['projection_rows_equal']) for x in executor_validation}
            for r in rows:
                if r['phase']!='measured' or r['mode']=='full': continue
                if r['mode']=='legacy': r['executor_parity']=dict(passed=True,validation='self-reference',qkv_exact=True)
                elif 'legacy' in a.modes: r['executor_parity']=dict(passed=bool(validation_pass.get(r['mode'],False) and r['executor_parity'].get('passed',False)),validation='untimed source-snapshot validation',
                    selection=selection_validation.get(r['mode']),qkv_exact=next((x['qkv']['passed'] for x in executor_validation if x['mode']==r['mode']),False),
                    logits_exact=next((x['logits_equal'] for x in executor_validation if x['mode']==r['mode']),False),
                    projection_rows_equal=next((x['projection_rows_equal'] for x in executor_validation if x['mode']==r['mode']),False),
                    paired_sample_consistency=r['executor_parity'])
                else: r['executor_parity']=dict(passed=None,status='UNPAIRED_NO_LEGACY_REFERENCE')
            atomic_json(checkpoint,dict(identity=manifest['identity'],status='COMPLETE',rows=rows,checksum=digest(rows)))
            live_rows=checkpoint_rows(a.output_dir,manifest['identity'])
            atomic_json(a.output_dir/'executor_parity.json',dict(validation=executor_validation,coverage=validation_rows))
            atomic_json(a.output_dir/'mapping_parity.json',dict(validation=validation_rows,selector_parity=selection_validation,
                cache=map_cache.stats() if map_cache else None,descriptor_build_ms=descriptor.build_ms if descriptor else None,
                descriptor_memory_bytes=descriptor.python_bytes_estimate if descriptor else None))
            atomic_json(checkpoint,dict(identity=manifest['identity'],status='COMPLETE',rows=rows,checksum=digest(rows)))
            live_rows=checkpoint_rows(a.output_dir,manifest['identity'])
            atomic_json(a.output_dir/'gpu_state.json',gpu_records)
            csv_reports(a.output_dir,live_rows); atomic_json(a.output_dir/'summary.json',summarize(live_rows,a.seed))
            del catalog,index,descriptor,map_cache,engine,reference
    atomic_json(a.output_dir/'gpu_state.json',gpu_records)
    allrows=checkpoint_rows(a.output_dir,manifest['identity'])
    failures=[r['row_id'] for r in allrows if r['executor_parity'].get('passed') is False]
    manifest.update(status='PARITY_FAILED' if failures else 'COMPLETE',parity_failures=failures,rows=len(allrows),raw_only=raw_guard.evidence(),
        full_correctness_claim=False,cache_classification='ORACLE_DIAGNOSTIC')
    atomic_json(a.output_dir/'manifest.json',manifest)


@contextmanager
def _null_context(): yield

def main(argv=None):
    a=args(argv); cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','output_dir')}
    b=load_context(); prov=b.a3().b2().provenance()
    files=['75_benchmark_r1b3_selection.py','76_benchmark_r1b3_batched_executor.py','77_benchmark_r1b3_b21_setup.py']
    for name in files: prov['files']['scripts/'+name]=hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
    for rel in ('src/semcache/edgelora/batched_mapping_optimized.py','src/semcache/edgelora/batched_mixed_projection.py','src/semcache/diagnostics/r1b3_fast_selection.py'):
        prov['files'][rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    identity=digest(dict(config=cfg,files=prov['files'])); manifest_path=a.output_dir/'manifest.json'
    if a.resume:
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('identity')!=identity: raise ValueError('Resume rejected: code/config changed')
    else:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        manifest=dict(experiment='R1-B3-B2.1',identity=identity,config=cfg,code=prov,status='READY',
            classification='ORACLE_DIAGNOSTIC',settings=dict(use_cache=False,output_attentions=True,attention='eager',atol=1e-3,rtol=1e-3),
            historical_reference_reproduced=False,run_tag=a.output_dir.name)
    manifest.setdefault('attempts',[]).append(dict(previous_status=manifest.get('status'),failure=manifest.get('failure')))
    manifest.pop('failure',None); manifest.pop('failure_type',None)
    manifest['status']='RUNNING'; atomic_json(manifest_path,manifest)
    # Required empty/report schemas exist even if Torch or a model is missing.
    for name in ('mapping_parity.json','executor_parity.json','gpu_state.json','summary.json'): atomic_json(a.output_dir/name,{})
    for name in ('setup_breakdown.csv','latency_comparison.csv','paired_differences.csv','memory_accounting.csv','correctness_comparison.csv'):
        if not (a.output_dir/name).exists(): csv_write(a.output_dir/name,[])
    try:
        with raw_only_guard() as guard: execute(a,manifest,guard)
    except (Exception,KeyboardInterrupt) as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'MEMORY_PREFLIGHT_FAILED' if manifest.get('status')=='MEMORY_PREFLIGHT_FAILED' else 'OOM' if 'out of memory' in str(exc).lower() else 'FAILED'
        manifest.update(status=status,failure_type=type(exc).__name__,failure=str(exc)); atomic_json(manifest_path,manifest)
        atomic_json(a.output_dir/'summary.json',dict(status=status,failure_type=type(exc).__name__,failure=str(exc),classification='ORACLE_DIAGNOSTIC'))
        print(status,str(exc),file=sys.stderr); return 1
    return 0 if manifest['status']=='COMPLETE' else 2
if __name__=='__main__': raise SystemExit(main())
