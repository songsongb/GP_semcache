#!/usr/bin/env python3
"""R1-B2 RAW-only paired prefill breakdown and diagnostic coverage sweep."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json, digest, raw_only_guard, audit_projection_records
from semcache.metrics.r1b2 import (MODES, coverage_value, coverage_override, correctness_status,
    breakdown, reports, load_checkpoints)


def legacy():
    spec=importlib.util.spec_from_file_location('r1_setup',ROOT/'scripts/69_run_r1_raw_semcache.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','pilot','full'),default='full')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-2.7b')
    p.add_argument('--dtype',choices=('float16','float32'),default='float16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision')
    p.add_argument('--encoder-revision',default='main')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--prompt-lengths'); p.add_argument('--cache-devices')
    p.add_argument('--instrumentation',choices=('minimal','breakdown','both'),default='both')
    p.add_argument('--coverages',default='0,.25,.5,.75,max')
    p.add_argument('--cache-capacity-mib',type=int,default=256)
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',type=Path,required=True,help='New run root; tokens_<length> children are created')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    try:
        a.lengths=[int(x) for x in (a.prompt_lengths or {'smoke':'32','pilot':'32,64','full':'32,64,128,256'}[a.profile]).split(',')]
        a.storages=(a.cache_devices or ('cpu,cuda' if a.profile=='full' else 'cuda' if a.device.startswith('cuda') else 'cpu')).split(',')
        a.coverage_values=[coverage_value(x) for x in a.coverages.split(',')]
        a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else 1 if a.profile=='smoke' else 3
        a.measured_runs=a.measured_runs if a.measured_runs is not None else 2 if a.profile=='smoke' else 10
        if (not a.lengths or min(a.lengths)<4 or len(set(a.lengths))!=len(a.lengths)
            or not a.storages or any(x not in ('cpu','cuda') for x in a.storages)
            or len(set(a.storages))!=len(a.storages) or len(set(a.coverage_values))!=len(a.coverage_values)
            or a.warmup_runs<0 or a.measured_runs<1 or a.cache_capacity_mib<1
            or ('cuda' in a.storages and not a.device.startswith('cuda'))):
            raise ValueError('Invalid lengths, storage, coverage, capacity or repetitions')
        if a.profile!='smoke' and (a.warmup_runs<3 or a.measured_runs<10):
            raise ValueError('Pilot/full require warmup >=3, measured >=10')
    except ValueError as exc: p.error(str(exc))
    return a


def config(a):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()
            if k not in ('resume','output_dir','prompt_lengths','cache_devices','coverages')}


def provenance():
    code=legacy().code_provenance()
    for name in ('72_run_r1b2_latency.py','73_aggregate_r1b2.py'):
        p=ROOT/'scripts'/name
        code['files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return code


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def execute(a,root,manifest,guard):
    import torch
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.cache.global_cache import GlobalCache
    from semcache.semcache_engine import SemCacheEngine
    from semcache.metrics.scaling_workload import controlled_length_trace
    from semcache.metrics.r1_runtime import (native_request, engine_request, synchronize,
        memory_preflight, memory_snapshot, raw_storage_audit)
    from semcache.metrics.r1b1 import snapshot, compare_snapshots
    from semcache.evaluation.logit_metrics import compare_logits
    from semcache.utils.seed import seed_everything
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: GPU benchmark NOT EXECUTED')
    seed_everything(a.seed)
    start=perf_counter_ns()
    model,tokenizer,metadata=load_model(dict(name=a.model,tokenizer=a.model,revision=a.revision,
        tokenizer_revision=a.tokenizer_revision or a.revision,dtype=a.dtype,device=a.device,
        local_files_only=not a.allow_download,attention_implementation='eager'))
    if getattr(model,'is_quantized',False) or getattr(model.config,'quantization_config',None):
        raise ValueError('RAW-only requires an unquantized model')
    model,lora=create_controlled_users(model); adapter=OPTModelAdapter(model)
    synchronize(a.device)
    initialization=(perf_counter_ns()-start)/1e6
    start=perf_counter_ns()
    encoder=TinyBERTSemanticEncoder(device=a.device,dtype=a.dtype,revision=a.encoder_revision,local_files_only=not a.allow_download)
    synchronize(a.device)
    runtime=dict(model=metadata,lora=lora,encoder=encoder.metadata,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,
        driver=legacy().command_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    if manifest.get('runtime') and manifest['runtime']!=runtime:
        raise ValueError('Resume runtime/model mismatch')
    manifest.update(runtime=runtime,model_adapter_initialization_ms=initialization,
        encoder_initialization_ms=(perf_counter_ns()-start)/1e6)
    # Exclude only scheduling choices; numerical/model/hardware/code identity stays bound.
    comparable={k:v for k,v in manifest['config'].items() if k not in
        ('instrumentation','lengths','storages','coverage_values','warmup_runs','measured_runs')}
    comparison_identity=digest(dict(config=comparable,runtime=runtime,files=manifest['code']['files']))
    manifest['comparison_identity']=comparison_identity
    atomic_json(root/'manifest.json',manifest)
    length=manifest['length']; capacity=a.cache_capacity_mib*1024**2
    trace=controlled_length_trace(tokenizer,length); item=trace[1]
    ids=item['token_ids']; text=item['text']; user='user_a'
    manifest['fixture']=dict(requested_tokens=length,actual_tokens=len(ids),token_ids=ids,token_ids_sha256=digest(ids),adapter=user)
    if len(ids)!=length:
        atomic_json(root/'manifest.json',manifest)
        raise ValueError(f'TOKEN_LENGTH_MISMATCH: requested {length}, tokenizer produced {len(ids)}; no silent truncation')
    anchors=encoder.encode([trace[0]['text'],trace[-1]['text']])
    def engine(storage):
        clusterer=IntentClusterer(2,initialization='first_k',update_mode='buffered',update_interval=100)
        clusterer.initialize(anchors)
        return SemCacheEngine(model,tokenizer,adapter,encoder,clusterer,GlobalCache(capacity),
            storage_device=storage,metadata=metadata,seed=a.seed,impact_reducer='paper_row_l2_sum',
            impact_execution='token_precompute',capture_impact_values=True,rho=.8,history_lambda=100,pbr_interval_queries=100)
    rows=load_checkpoints(root,manifest['identity'])
    passes=['minimal','breakdown'] if a.instrumentation=='both' else [a.instrumentation]
    schedule=[('warmup',i) for i in range(a.warmup_runs)]+[('measured',i) for i in range(a.measured_runs)]
    configurations=[(s,p) for s in a.storages for p in passes]
    cases=[(m,None) for m in MODES[:3]]+[(MODES[3],c) for c in a.coverage_values]
    for phase,repeat in schedule:
        # Temporary CPU logits/policy for cross-storage/instrumentation checks only.
        comparisons={}
        for storage_name,instrumentation in (configurations if repeat%2==0 else list(reversed(configurations))):
            group=f'{phase}-{repeat}-{storage_name}-{instrumentation}'
            checkpoint=root/'checkpoints'/(group+'.json')
            if phase=='measured' and checkpoint.exists(): continue
            storage=str(next(model.parameters()).device) if storage_name=='cuda' else 'cpu'
            manifest['active_group']=group
            try:
                manifest['last_preflight']=memory_preflight(model,length,capacity,storage,a.device)
            except MemoryError:
                manifest['status']='MEMORY_PREFLIGHT_FAILED'
                raise
            atomic_json(root/'manifest.json',manifest)
            reference=native_request(model,tokenizer,adapter,text,user,False)
            if reference['ids']!=ids: raise AssertionError('Reference token IDs differ')
            offset=repeat%len(cases)
            order=cases[offset:]+cases[:offset]
            if repeat%2: order=list(reversed(order))
            batch=[]
            for order_index,(mode,coverage) in enumerate(order):
                semcache=None; build_ms=None; source_sig=digest([]); source_policy=None; source_policy_sig=digest(None)
                if mode!='FULL_RECOMPUTE':
                    semcache=engine(storage)
                    source=engine_request(semcache,text,user,'source','SEMCACHE_LOOKUP_NO_REUSE',True)
                    build_ms=source['summary']['timing']['request_wall_ms']
                    source_sig=legacy().source_signature(semcache)
                    source_policy=snapshot(semcache,source)
                    source_policy_sig=digest(source_policy)
                    del source
                source_contexts={k:tuple(e.qkv_metadata.get('source_query_token_ids',())) for k,e in semcache.cache.entries.items()} if semcache else {}
                resident_before=semcache.cache.physical_tensor_bytes if semcache else 0
                storage_evidence=raw_storage_audit(semcache.cache,storage) if semcache else None
                if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(a.device)
                before=memory_snapshot(a.device)
                detailed=instrumentation=='breakdown'
                scope=coverage_override(coverage) if mode==MODES[3] else nullcontext({})
                with scope as filter_evidence:
                    if mode=='FULL_RECOMPUTE':
                        result=native_request(model,tokenizer,adapter,text,user,detailed)
                        s=dict(token_ids=result['ids'],timing=result['timing'],reuse_block_provenance=[],
                            reused_unique_token_count=0,block_hit_count=0)
                        records=result['projection_records']
                    else:
                        execution={'LOOKUP_NO_REUSE':'SEMCACHE_LOOKUP_NO_REUSE',
                            'NATURAL_RAW_REUSE':'SEMCACHE_PHYSICAL_REUSE',
                            'POSITION_ALIGNED_REUSE':'SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC'}[mode]
                        result=engine_request(semcache,text,user,'target',execution,detailed)
                        s=result['summary']; records=list(result['projection_audit'].values())
                after=memory_snapshot(a.device)
                if s['token_ids']!=ids: raise AssertionError('Measured input identity differs')
                reused=s['reused_unique_token_count']; prov=s['reuse_block_provenance']
                positions=[i for p in prov for i in range(p['destination_start'],p['destination_end'])]
                if records:
                    audit=audit_projection_records(records,length,len(adapter.layers),reused,positions)
                    native_rows=audit['native_projection_rows_all_layers_roles']; skipped=audit['skipped_projection_rows_all_layers_roles']
                else:
                    if mode!='FULL_RECOMPUTE': raise AssertionError('Missing physical execution records')
                    native_rows=length*len(adapter.layers)*3; skipped=0
                if coverage==0 and skipped: raise AssertionError('Zero coverage skipped real rows')
                quality=compare_logits(reference['logits'],result['logits'],0)
                finite=bool(torch.isfinite(reference['logits']).all() and torch.isfinite(result['logits']).all())
                allclose=finite and bool(torch.allclose(reference['logits'],result['logits'],atol=1e-3,rtol=1e-3))
                quality={k:(v if not isinstance(v,float) or math.isfinite(v) else None) for k,v in quality.items()}
                mismatch=sum(p['destination_end']-p['destination_start'] for p in prov if not p['positions_identical'])
                adapter_match=all(p['source_adapter']==p['destination_adapter']==user for p in prov)
                # Inspect the actual selected source entry; source_query_id alone is not a context proof.
                context_match=all(source_contexts.get(p['cache_key'])==tuple(ids)
                    for p in prov) if prov else True
                quality.update(logit_allclose=allclose,last_position_kl=quality['last_position_kl_baseline_to_injected'],
                    actual_reuse_coverage=reused/length,position_mismatch_count=mismatch,
                    position_mismatch_windows=sum(not p['positions_identical'] for p in prov),adapter_identity_match=adapter_match,
                    source_target_context_match=context_match,atol=1e-3,rtol=1e-3,
                    status=correctness_status(reused,allclose,mismatch==0,adapter_match,context_match))
                policy=snapshot(semcache,result) if semcache else None
                key=(mode,str(coverage))
                parity=dict(status='UNPAIRED')
                if key in comparisons:
                    previous=comparisons[key]
                    pol=compare_snapshots(previous['policy'],policy)
                    source_pol=compare_snapshots(previous['source_policy'],source_policy)
                    logits_equal=bool(torch.allclose(previous['logits'],result['logits'],atol=1e-3,rtol=1e-3))
                    parity=dict(status='PASS' if pol['passed'] and source_pol['passed'] and logits_equal and previous['source']==source_sig else 'FAIL',
                        reference=previous['row_id'],policy_pass=pol['passed'],source_policy_pass=source_pol['passed'],logit_allclose=logits_equal,
                        source_equal=previous['source']==source_sig,policy_differences=pol['differences'][:10],source_policy_differences=source_pol['differences'][:10])
                row_id=f'{manifest["identity"]}:{group}:{mode}:{coverage}'
                comparisons.setdefault(key,dict(policy=policy,source_policy=source_policy,logits=result['logits'],source=source_sig,row_id=row_id))
                spans=[dict(source=[p['source_start'],p['source_end']],destination=[p['destination_start'],p['destination_end']],
                    tokens_sha256=digest(p['token_ids'])) for p in prov]
                row=dict(row_id=row_id,pair_id=f'{manifest["identity"]}:{group}',comparison_identity=comparison_identity,
                    phase=phase,repeat=repeat,measurement_profile=a.profile,order_index=order_index,length=length,actual_prompt_tokens=len(ids),
                    storage='RAW_GPU' if storage_name=='cuda' else 'RAW_CPU',instrumentation=instrumentation,mode=mode,
                    requested_coverage=coverage,actual_coverage=reused/length,reused_tokens=reused,recomputed_tokens=length-reused,
                    selected_windows=len(prov),raw_hit_windows=s['block_hit_count'],native_projection_rows=native_rows,
                    skipped_projection_rows=skipped,projection_audit_kind='executed' if records else 'native_structure_derived_no_inner_hooks',
                    spans=spans,selection_signature=digest(spans),source_signature=source_sig,source_policy_signature=source_policy_sig,
                    policy_signature=digest(policy),logits_sha256=tensor_digest(result['logits']),comparison_parity=parity,
                    token_ids_sha256=digest(ids),adapter=user,request_wall_ms=s['timing']['request_wall_ms'],
                    source_build_ms=build_ms,source_build_in_warm_request=False,
                    coverage_filter_host_ms=filter_evidence.get('filter_host_ms'),
                    feasible_candidate_windows=filter_evidence.get('candidate_windows'),
                    resident_bytes_before=resident_before,resident_bytes_after=semcache.cache.physical_tensor_bytes if semcache else 0,
                    memory_before=before,memory_after=after,raw_storage=storage_evidence,
                    cache_fetch_calls=result['cache_fetch_calls'],h2d_cache_bytes=result['h2d_cache_bytes'],
                    correctness=quality,breakdown=breakdown(s['timing'],mode,instrumentation),
                    diagnostic_override=mode==MODES[3],safe_reuse_claimed=False)
                batch.append(row)
                guard.evidence()
                del result,semcache
            atomic_json(checkpoint,dict(identity=manifest['identity'],status='COMPLETE',rows=batch,rows_sha256=digest(batch)))
            rows=[r for r in rows if r['pair_id']!=batch[0]['pair_id']]+batch
            save_reports(root,rows)
            print('COMPLETE',length,group,flush=True)
    manifest.update(status='COMPLETE',raw_only=guard.evidence(),rows=len(rows),
        correctness_failures=sum(r['correctness']['status']=='STRICT_CORRECTNESS_FAIL' for r in rows),
        full_or_no_reuse_logit_failures=sum(not r['correctness']['logit_allclose'] for r in rows if not r['reused_tokens']),
        instrumentation_storage_parity_failures=sum(r['comparison_parity']['status']=='FAIL' for r in rows))
    if manifest['instrumentation_storage_parity_failures']: manifest['status']='PARITY_FAILED'
    atomic_json(root/'manifest.json',manifest)
    save_reports(root,rows)
    return manifest['status']=='COMPLETE'


def save_reports(root,rows):
    temporary=root/'raw.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    temporary.replace(root/'raw.jsonl')
    reports(root,rows)


def main(argv=None):
    a=parse_args(argv); code=provenance(); cfg=config(a)
    if not a.resume: a.output_dir.mkdir(parents=True,exist_ok=False)
    failed=False
    for length in a.lengths:
        root=a.output_dir/f'tokens_{length}'; path=root/'manifest.json'
        identity=digest(dict(config=cfg,code=code['files'],length=length))
        if a.resume:
            manifest=json.loads(path.read_text())
            if manifest['identity']!=identity: raise SystemExit('Resume rejected: code/config changed; use a new output root')
        else:
            root.mkdir(exist_ok=False)
            manifest=dict(schema_version=1,experiment='R1-B2',identity=identity,length=length,config=cfg,code=code,
                python=platform.python_version(),platform=platform.platform(),status='INITIALIZING',
                reference_reproduced='UNVERIFIED; historical user numbers are not runtime evidence',
                settings=dict(use_cache=False,output_attentions=True,attention_implementation='eager',batch_size=1,
                    impact_execution='token_precompute',atol=1e-3,rtol=1e-3),
                timing_contract=dict(primary='minimal request_wall_ms',source='separate detailed LOOKUP_NO_REUSE build',
                    request='synchronize before request and before close; correctness outside',
                    breakdown='existing R1 detailed executor + deferred CUDA events; prefill end synchronizes',
                    minimal='original executor; existing CPU policy timers and audit retained; no inner CUDA events',
                    nesting='never sum CPU and GPU or parent and child; materialization is inside policy; CHU/PBR separate',
                    remaining_transformer='derived prefill GPU minus mixed executor GPU, or native QKV GPU for FULL; includes idle gaps'))
        manifest.setdefault('attempts',[]).append(dict(previous_status=manifest.get('status'),previous_failure=manifest.get('failure')))
        for key in ('failure','failure_type'): manifest.pop(key,None)
        manifest['status']='RUNNING'
        atomic_json(path,manifest)
        try:
            with raw_only_guard() as guard:
                manifest['raw_only']=guard.evidence()
                atomic_json(path,manifest)
                if not execute(a,root,manifest,guard): failed=True
        except Exception as exc:
            failed=True
            status=('MEMORY_PREFLIGHT_FAILED' if manifest.get('status')=='MEMORY_PREFLIGHT_FAILED' else
                'OOM' if 'out of memory' in str(exc).lower() else 'TOKEN_LENGTH_MISMATCH' if 'TOKEN_LENGTH_MISMATCH' in str(exc) else 'FAILED')
            manifest.update(status=status,failure_type=type(exc).__name__,failure=str(exc))
            atomic_json(path,manifest)
            save_reports(root,load_checkpoints(root,identity))
            print(status,length,str(exc),file=sys.stderr,flush=True)
        except KeyboardInterrupt:
            manifest.update(status='INTERRUPTED',failure='Only atomic completed groups are resumable')
            atomic_json(path,manifest)
            save_reports(root,load_checkpoints(root,identity))
            return 130
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
