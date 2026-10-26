#!/usr/bin/env python3
"""R1-B1-D1: artifact-bound correctness controls, NOT latency measurements."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import sys
import os
import subprocess
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json, digest, raw_only_guard
from semcache.diagnostics.r1b1_d1 import (provenance_audit,spans,guarded_selections,
    capture,forward,error,layer_errors,shape_probes,cache_errors)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,nargs='+',required=True,help='R1-B1 raw.jsonl files; sibling manifest.json required')
    p.add_argument('--lengths',default='32,64')
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--impact-execution',default='token_precompute')
    p.add_argument('--storage',choices=('RAW_CPU','RAW_GPU_DIAGNOSTIC'),default='RAW_CPU')
    p.add_argument('--instrumentation',choices=('minimal','breakdown'),default='minimal')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--audit-only',action='store_true',help='Provenance verification only; no tensor inference')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        args.lengths=[int(x) for x in args.lengths.split(',')]
    except ValueError:
        p.error('Invalid lengths')
    if not args.lengths or len(set(args.lengths))!=len(args.lengths) or any(x not in (32,64) for x in args.lengths):
        p.error('D1 supports unique requested lengths 32 and/or 64')
    return args


def read_cases(args):
    chosen={}
    bindings=[]
    for path in args.raw:
        manifest_path=path.parent/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                sha.update(chunk)
        bindings.append(dict(raw=str(path),raw_sha256=sha.hexdigest(),
            manifest=str(manifest_path),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
        with path.open() as f:
            for line in f:
                row=json.loads(line)
                if (row.get('phase')!='measured' or row.get('condition')!='same_user_exact'
                    or row.get('repeat')!=args.repeat or row.get('storage')!=args.storage
                    or row.get('instrumentation')!=args.instrumentation
                    or row.get('impact_execution')!=args.impact_execution
                    or row.get('requested_prompt_tokens') not in args.lengths
                    or row.get('mode') not in ('RAW_PHYSICAL_REUSE','POSITION_ALIGNED_DIAGNOSTIC')):
                    continue
                key=(row['requested_prompt_tokens'],row['mode'])
                if key in chosen:
                    raise ValueError('Ambiguous input row: '+str(key))
                if digest(row['input_token_ids'])!=row['token_ids_sha256']:
                    raise ValueError('Input token hash mismatch')
                chosen[key]=(row,manifest)
    cases=[]
    for length in args.lengths:
        natural,manifest=chosen[(length,'RAW_PHYSICAL_REUSE')]
        strict,other=chosen[(length,'POSITION_ALIGNED_DIAGNOSTIC')]
        if manifest!=other:
            raise ValueError('Pair must share the same experiment manifest')
        cases.append(dict(length=length,natural=natural,strict=strict,manifest=manifest,
                          audit=provenance_audit(natural,strict)))
    return cases,bindings


def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r}) or ['status']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def run_case(args,case):
    import torch
    import semcache.semcache_engine as engine_module
    import semcache.metrics.r1_runtime as r1_runtime
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.semantic.hit_selection import CacheHit
    from semcache.cache.global_cache import GlobalCache
    from semcache.metrics.scaling_workload import controlled_length_trace
    from semcache.metrics.r1_runtime import memory_preflight
    from semcache.utils.seed import seed_everything
    from semcache.evaluation.logit_metrics import compare_logits
    manifest=case['manifest']
    runtime,cfg=manifest['runtime'],manifest['config']
    natural,strict=case['natural'],case['strict']
    model_meta=runtime['model']
    ids=natural['input_token_ids']
    user=natural['adapter']
    atol,rtol=natural['correctness']['atol'],natural['correctness']['rtol']
    if (atol,rtol)!=(strict['correctness']['atol'],strict['correctness']['rtol']):
        raise ValueError('Pair tolerances differ')
    if natural['use_cache'] or not natural['output_attentions'] or natural['attention_implementation']!='eager':
        raise ValueError('Unsupported reference inference settings')
    seed_everything(cfg['seed'])
    model,tokenizer,metadata=load_model(dict(name=model_meta['model'],tokenizer=model_meta['tokenizer'],
        revision=model_meta['resolved_model_revision'],tokenizer_revision=model_meta['resolved_tokenizer_revision'],
        dtype=cfg['dtype'],device=args.device,local_files_only=True,attention_implementation='eager'))
    for field in ('resolved_model_revision','resolved_tokenizer_revision','dtype'):
        if metadata[field]!=model_meta[field] or metadata[field] is None:
            raise ValueError('Reference model identity not reproduced: '+field)
    if getattr(model,'is_quantized',False):
        raise ValueError('Quantized models forbidden')
    model,lora=create_controlled_users(model)
    if lora!=runtime['lora']:
        raise ValueError('Controlled adapter fixture differs from reference')
    adapter=OPTModelAdapter(model)
    enc=runtime['encoder']
    encoder=TinyBERTSemanticEncoder(model_id=enc['checkpoint'],revision=enc.get('resolved_revision') or enc['revision'],
        pooling=enc['pooling'],max_length=enc['max_length'],device=args.device,dtype=cfg['dtype'],local_files_only=True)
    if encoder.metadata.get('resolved_revision')!=enc.get('resolved_revision'):
        raise ValueError('Semantic encoder revision differs')
    trace=controlled_length_trace(tokenizer,case['length'])
    if trace[0]['token_ids']!=ids or user!='user_a':
        raise ValueError('Reference exact prompt/adapter not reproduced; no replacement input allowed')
    clusterer=IntentClusterer(2,initialization='first_k',update_mode=cfg['cluster_update_mode'],
                             update_interval=cfg['cluster_update_interval'])
    clusterer.initialize(encoder.encode([trace[0]['text'],trace[-1]['text']]))
    storage='cpu' if args.storage=='RAW_CPU' else str(next(model.parameters()).device)
    preflight=memory_preflight(model,len(ids),natural['cache_capacity_bytes'],storage,args.device)
    engine=engine_module.SemCacheEngine(model,tokenizer,adapter,encoder,clusterer,GlobalCache(natural['cache_capacity_bytes']),
        storage_device=storage,metadata=metadata,seed=cfg['seed'],impact_reducer=cfg['impact_reducer'],
        impact_execution=args.impact_execution,rho=cfg['rho'],history_lambda=cfg['history_lambda'],
        pbr_interval_queries=cfg['pbr_interval'] if cfg['pbr_mode']=='interval' else None)
    source_capture={}
    original_source_executor=r1_runtime.detailed_mixed
    @contextmanager
    def captured_source(*a,**kw):
        with original_source_executor(*a,**kw) as audit:
            with capture(adapter) as tensors:
                yield audit
            source_capture.update(tensors)
    # Temporary binding only in this standalone diagnostic process.
    # R1-B1 always constructs sources through detailed=True, even for minimal targets.
    with patch.object(r1_runtime,'detailed_mixed',captured_source):
        source=r1_runtime.engine_request(engine,trace[0]['text'],user,'source','SEMCACHE_LOOKUP_NO_REUSE',True)
    actual_source=json.loads(json.dumps([dict(key=e.key,positions=e.positions,size_bytes=e.size_bytes,
        provenance=e.qkv_metadata) for _,e in sorted(engine.cache.entries.items())]))
    for reference_row in (natural,strict):
        expected_source=[{k:e[k] for k in ('key','positions','size_bytes','provenance')}
                         for e in reference_row['source_policy']['entries']]
        if actual_source!=expected_source:
            raise ValueError('Source residency/context/adapter provenance differs from reference')
    cluster=clusterer.observe_with_diagnostics(encoder.encode([trace[0]['text']])[0])['cluster_id']
    hits=[]
    for window in engine.extractor.extract(ids):
        entry=engine.cache.entries.get(engine.matcher.key(cluster,window))
        if entry is not None:
            hits.append(CacheHit(window,entry,entry.impact or 0.))
    selections=guarded_selections(hits,len(ids),ids,user)
    for name,row in (('NATURAL',natural),('STRICT',strict)):
        reference=[{k:p[k] for k in ('source_start','source_end','destination_start','destination_end','token_ids')}
                   for p in row['reuse_provenance']]
        if spans(selections[name])!=reference:
            raise ValueError('Reconstructed '+name+' selection differs from artifact')
    base=forward(model,adapter,ids,user)
    repeat=forward(model,adapter,ids,user)
    full_repeat=error(base['logits'],repeat['logits'],atol,rtol)
    del repeat
    results={}
    strict_capture=None
    layer_rows,first_rows=[],[]
    for name,hits_selected in list(selections.items())+[
        ('STRICT_FULL_SHAPE_INJECTION',selections['STRICT']),('STRICT_FULL_SHAPE_ASSEMBLY',selections['STRICT']),
        ('NATURAL_FULL_SHAPE_INJECTION',selections['NATURAL'])]:
        control='injection' if name.endswith('INJECTION') else 'full_assembly' if name.endswith('ASSEMBLY') else 'mixed'
        output=forward(model,adapter,ids,user,hits_selected,control)
        results[name]=dict(logits=output['logits'],quality=compare_logits(base['logits'],output['logits'],0),
            errors=error(base['logits'],output['logits'],atol,rtol),selected=spans(hits_selected),
            full_shape_projection_used=control in ('injection','full_assembly'),projection_savings_claimed=False,
            projection_records=output['projection_records'])
        if name=='STRICT':
            strict_capture=output['captured']
            layer_rows,first_rows=layer_errors(base['captured'],strict_capture,atol,rtol)
        del output
    shape_rows=[]
    for origin,tensors in (('FULL',base['captured']),('STRICT',strict_capture)):
        shape_rows.extend(dict(hidden_origin=origin,**r) for r in shape_probes(adapter,tensors,selections['STRICT'],len(ids),user,atol,rtol))
    cache_rows=cache_errors(selections['STRICT'],source_capture,base['captured'],atol,rtol,strict_capture)
    misplaced=[h for h in selections['NATURAL'] if h.entry.positions[0]!=h.window.start]
    misplaced_rows=cache_errors(misplaced,source_capture,base['captured'],atol,rtol)
    source_vs_full,_=layer_errors(base['captured'],source_capture,atol,rtol)
    cache_exact=all(r['nonexact_windows']==0 for r in cache_rows if r['comparison']=='cache_vs_source_execution')
    source_exact=all(r['exact_equal'] for r in source_vs_full)
    assembly_exact=(all(r['nonexact_windows']==0 for r in cache_rows if r['comparison']=='cache_vs_mixed_assembled_rows')
        and all(r['observed_output_vs_recomputed_subset']['exact_equal'] for r in shape_rows
                if r['hidden_origin']=='STRICT' and r['component']=='total_peft'))
    shape_difference=any(not r['exact_equal'] for r in shape_rows)
    eliminated=(results['STRICT_FULL_SHAPE_ASSEMBLY']['errors']['exact_equal']
                and results['STRICT_FULL_SHAPE_INJECTION']['errors']['exact_equal'])
    guard_comparisons={name:dict(selections_equal=results[name]['selected']==results['STRICT']['selected'],
        logits=error(results['STRICT']['logits'],results[name]['logits'],atol,rtol))
        for name in ('POSITION_GUARD','SELECTED_POSITION_GUARD')}
    current_runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    try:
        current_runtime['driver']=subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        current_runtime['driver']=None
    runtime_match={k:dict(reference=runtime.get(k),current=v,matches=runtime.get(k)==v) for k,v in current_runtime.items()}
    reference_files=manifest.get('code',{}).get('files',{})
    execution_files=('src/semcache/semcache_engine.py','src/semcache/edgelora/mixed_projection.py',
        'src/semcache/metrics/r1_runtime.py',
        'src/semcache/cache/attention_impact.py','src/semcache/cache/impact_execution.py',
        'src/semcache/models/lora_fixtures.py','src/semcache/models/loader.py',
        'src/semcache/models/model_adapter.py','src/semcache/utils/seed.py',
        'src/semcache/semantic/hit_selection.py')
    code_match={p:dict(reference=reference_files.get(p),current=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
                for p in execution_files}
    for item in code_match.values():
        item['matches']=item['reference']==item['current']
    report=dict(reference_dtype=cfg['dtype'],requested_length=case['length'],actual_tokens=len(ids),
        adapter=user,token_ids_sha256=natural['token_ids_sha256'],tolerances=dict(atol=atol,rtol=rtol),
        runtime=dict(model=metadata,lora=lora,encoder=encoder.metadata,seed=cfg['seed'],**current_runtime,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        reference_runtime_comparison=runtime_match,
        reference_execution_code_comparison=code_match,
        preflight=preflight,full_repeat=full_repeat,position_guard_vs_strict=guard_comparisons,
        source_precondition_identity_verified=True,
        first_shape_difference=next((r for r in shape_rows if not r['exact_equal']),None),
        first_divergence_by_layer=first_rows,
        first_divergent_layer=next((r['layer'] for r in first_rows if r['first_nonzero'] is not None),None),
        first_tolerance_failure_layer=next((r['layer'] for r in first_rows if r['first_tolerance_failure'] is not None),None),
        reference_reproduction={name:dict(observed_original_max_diff=row['correctness']['max_abs_logit_diff'],
            reproduced_max_diff=results[name]['quality']['max_abs_logit_diff'],
            original_allclose=row['correctness']['allclose'],reproduced_allclose=results[name]['errors']['allclose'],
            exact_reported_max_diff_match=row['correctness']['max_abs_logit_diff']==results[name]['quality']['max_abs_logit_diff'])
            for name,row in (('NATURAL',natural),('STRICT',strict))},
        facts=dict(cache_equals_captured_source=cache_exact,source_nohit_equals_native_full=source_exact,
            strict_cache_and_fresh_assembly_exact=assembly_exact,
            same_input_projection_shape_difference_observed=shape_difference,
            full_shape_controls_equal_full=eliminated,
            position_guard_equals_strict=all(c['selections_equal'] and c['logits']['exact_equal'] for c in guard_comparisons.values()),
            misaligned_reuse_error_persists_without_subset_gemm=(bool(misplaced) and cache_exact and source_exact
                and not results['NATURAL_FULL_SHAPE_INJECTION']['errors']['allclose']
                and results['STRICT_FULL_SHAPE_INJECTION']['errors']['exact_equal']),
            shape_change_eliminates_strict_difference=(full_repeat['exact_equal'] and cache_exact and source_exact and assembly_exact
                and shape_difference and not results['STRICT']['errors']['exact_equal'] and eliminated)),
        hypotheses=['Historical FP16 error is not attributed unless reference selection/runtime and diagnostic outputs reproduce.',
                    'A shape difference on identical inputs establishes shape sensitivity; it alone does not prove the final logit error cause.',
                    'Cache-vs-native differences alone do not prove storage corruption; cache-vs-captured-source checks storage/assembly provenance.',
                    'Specific cuBLAS kernels, Tensor Core accumulation order and internal precision are hypotheses; no kernel profiler is used.'],
        controls={name:{k:v for k,v in result.items() if k!='logits'} for name,result in results.items()},
        latency_measured=False,safe_reuse_claimed=False)
    write_csv(args.output_dir/f'layers_{case["length"]}.csv',layer_rows)
    write_csv(args.output_dir/f'shape_probes_{case["length"]}.csv',shape_rows)
    write_csv(args.output_dir/f'cache_{case["length"]}.csv',cache_rows)
    write_csv(args.output_dir/f'misaligned_cache_{case["length"]}.csv',misplaced_rows)
    write_csv(args.output_dir/f'source_vs_full_{case["length"]}.csv',source_vs_full)
    atomic_json(args.output_dir/f'diagnosis_{case["length"]}.json',report)
    # Return no tensors: the next case can release this model and all captures.
    reproduced=(all(r['exact_reported_max_diff_match'] and r['original_allclose']==r['reproduced_allclose']
                    for r in report['reference_reproduction'].values())
                and all(r['matches'] for r in runtime_match.values()) and all(r['matches'] for r in code_match.values()))
    return dict(length=case['length'],facts=report['facts'],reference_reproduction=report['reference_reproduction'],
                historical_case_reproduced=reproduced)


def main(argv=None):
    args=parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    paths=list((ROOT/'src/semcache').rglob('*.py'))+[Path(__file__).resolve()]
    state=dict(status='RUNNING',gpu_executed=False,latency_measured=False,
        code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths})
    try:
        state['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        state['git_head']=None
    try:
        cases,bindings=read_cases(args)
        state['inputs']=bindings
        atomic_json(args.output_dir/'provenance_audit.json',[dict(length=c['length'],**c['audit']) for c in cases])
        if any(c['audit']['expected_64_case_verified'] is False for c in cases):
            raise ValueError('64-token artifact does not match reported 63/54 coverage and three position moves')
        if args.audit_only:
            state['status']='ARTIFACT_AUDIT_ONLY'
        else:
            with raw_only_guard() as guard:
                state['diagnoses']=[]
                for case in cases:
                    state['active_length']=case['length']
                    atomic_json(args.output_dir/'manifest.json',state)
                    state['diagnoses'].append(run_case(args,case))
                    state['gpu_executed']=args.device.startswith('cuda')
                state['raw_only']=guard.evidence()
            state['status']=('DIAGNOSTICS_EXECUTED_NOT_A_SAFETY_CLAIM' if all(d['historical_case_reproduced'] for d in state['diagnoses'])
                             else 'DIAGNOSTICS_EXECUTED_REFERENCE_NOT_REPRODUCED')
    except Exception as exc:
        state.update(status='FAILED',failure_type=type(exc).__name__,failure=str(exc))
        print(f'D1 FAILED: {exc}',file=sys.stderr)
    atomic_json(args.output_dir/'manifest.json',state)
    return 1 if state['status']=='FAILED' else 0


if __name__=='__main__':
    raise SystemExit(main())
