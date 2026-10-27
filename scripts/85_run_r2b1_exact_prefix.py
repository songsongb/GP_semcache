#!/usr/bin/env python3
"""R2-B1: independent native OPT exact-prefix KV correctness/Forward-skip baseline."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import importlib.metadata
import hashlib
import os
from pathlib import Path
import platform
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r2b1_prefix_runtime import PrefixRuntime
from semcache.diagnostics.r2b1_prefix_audit import geometry,compare_suffix
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r1b3_break_even import code_identity,schedule
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
from semcache.metrics.r1 import atomic_json,digest,raw_only_guard
from semcache.metrics.r1b2 import csv_write,stats

MODES=('FULL_USE_CACHE','EXACT_PREFIX_WARM')
ARTIFACTS=('correctness','cache_integrity','forward_skip_audit','negative_tests','source_costs')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','correctness'),default='smoke')
    p.add_argument('--model',choices=('facebook/opt-125m','facebook/opt-2.7b'),default='facebook/opt-125m')
    p.add_argument('--dtype',choices=('float32','float16'),default='float32')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--cache-device',default=None,help='Same device only; default model device')
    p.add_argument('--adapter',choices=('user_a','user_b'),default='user_a')
    p.add_argument('--total-length',type=int,default=32); p.add_argument('--prefix-length',type=int,default=16)
    p.add_argument('--cases',help='Explicit small case list L:P,L:P; overrides total/prefix options')
    p.add_argument('--revision'); p.add_argument('--tokenizer-revision'); p.add_argument('--allow-download',action='store_true')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--order-seed',type=int,default=601)
    p.add_argument('--warmup-runs',type=int,default=3); p.add_argument('--measured-runs',type=int,default=3)
    p.add_argument('--independent-run-id',default='run01'); p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        a.case_values=[tuple(int(v) for v in x.split(':')) for x in a.cases.split(',')] if a.cases else [(a.total_length,a.prefix_length)]
        for case in a.case_values:
            if len(case)!=2: raise ValueError('cases require L:P')
            geometry(*case)
        if len(set(a.case_values))!=len(a.case_values) or not 1<=len(a.case_values)<=6: raise ValueError('1..6 distinct explicit cases')
        if any(n<4 or n>256 for n,_ in a.case_values): raise ValueError('B1 cases restricted to 4..256 tokens; longer runs deferred')
        if a.warmup_runs<3 or not 3<=a.measured_runs<=5: raise ValueError('B1 requires >=3 warmups and 3..5 exploratory repeats')
        if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1': raise ValueError('CUDA_LAUNCH_BLOCKING=1 invalidates exploratory timing')
    except (ValueError,PrefixError) as exc: p.error(str(exc))
    return a


def package_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def initialize(a):
    a.output_dir.mkdir(parents=True,exist_ok=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    m=dict(experiment='R2-B1 Exact Prefix KV baseline',classification='EXACT_PREFIX_BASELINE',
        semantic_reuse_safety_claimed=False,cross_user_reuse=False,implementation_status='IMPLEMENTED_LOCAL_ONLY',
        execution_status='RUNNING',correctness_status='UNVERIFIED',forward_skip_status='UNVERIFIED',
        code=code_identity(ROOT),config=cfg,python=platform.python_version(),
        versions={n:package_version(n) for n in ('torch','transformers','peft')},
        environment={k:os.environ.get(k) for k in ('HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES')},
        settings=dict(use_cache=True,output_attentions=True,return_dict=True,attention='eager',atol=1e-3,rtol=1e-3),
        timing=dict(endpoint='pretokenized prompt -> logits; not generation TTFT',
            request='synchronized wall; adapter/eval, input/mask/positions, namespace lookup, per-request KV clones/container, native Forward and full output cache',
            excluded='model loading, fixture/tokenization, source prefill/snapshot, expensive hashes, cache integrity and logits CPU transfer',
            cache_prepare='host enqueue/container interval; GPU copy completion included in synchronized request wall',
            source='independent native PREFIX_SOURCE_PREFILL; separate hooked proof pass; never target Full cache slicing',
            exploratory=True,performance_inference='R2-B2/B3 deferred; no B1 statistical speedup conclusion'))
    implementation=['scripts/85_run_r2b1_exact_prefix.py',
        *[f'src/semcache/diagnostics/r2b1_prefix_{name}.py' for name in ('runtime','cache','audit')],
        'tests/test_r2b1_prefix_cache.py','tests/test_r2b1_prefix_runtime.py','docs/R2B1_EXACT_PREFIX_KV_BASELINE.md']
    m['implementation_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in implementation if (ROOT/name).exists()}
    m['specifications_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'docs').glob('R2-B1_*.md')}
    m['identity']=digest(dict(code=m['code']['sha256'],config=cfg))
    atomic_json(a.output_dir/'manifest.json',m)
    for name in ARTIFACTS: atomic_json(a.output_dir/(name+'.json'),dict(status='UNVERIFIED'))
    csv_write(a.output_dir/'paired_samples.csv',[]); csv_write(a.output_dir/'case_summary.csv',[])
    atomic_json(a.output_dir/'decision_summary.json',dict(status='UNVERIFIED',speedup_claim_qualified=False))
    return m


def summarize(samples,source_costs):
    rows=[]
    for key in sorted({r['case'] for r in samples}):
        block=[r for r in samples if r['case']==key and r['phase']=='measured']
        groups={mode:{r['pair_index']:r for r in block if r['mode']==mode} for mode in MODES}
        f,h=groups.values()
        if not f and not h: continue
        if set(f)!=set(h) or len(block)!=2*len(f): raise PrefixError('MISSING_PAIRED_EVIDENCE','incomplete/duplicate pair')
        keys=sorted(f); full=[f[i]['wall_ms'] for i in keys]; warm=[h[i]['wall_ms'] for i in keys]
        delta=[f[i]['wall_ms']-h[i]['wall_ms'] for i in keys]
        exemplar=h[keys[0]]
        rows.append(dict(case=key,total_length=exemplar['total_length'],prefix_length=exemplar['prefix_length'],suffix_length=exemplar['suffix_length'],
            full=stats(full),warm=stats(warm),paired_difference=stats(delta),warm_wins=sum(d>0 for d in delta),count=len(keys),
            difference_definition='D(F,H)=Full-Warm; positive favors conditional warm target',
            observed_ratio=sum(full)/sum(warm),speedup_claim_qualified=False,
            source_costs=source_costs[key],latency_classification='EXPLORATORY_ONLY; source costs excluded, copies included'))
    return rows


def run(a,m):
    import torch
    runtime=PrefixRuntime(a); m['runtime']=runtime.provenance; m['implementation_status']='EXECUTED_MODEL_RUNTIME'
    samples=[]; correctness={}; integrity={}; skip={}; negatives={}; sources={}; summaries=[]
    def write():
        for name,value in [('correctness',correctness),('cache_integrity',integrity),('forward_skip_audit',skip),('negative_tests',negatives),('source_costs',sources)]:
            atomic_json(a.output_dir/(name+'.json'),value)
        csv_write(a.output_dir/'paired_samples.csv',samples); csv_write(a.output_dir/'case_summary.csv',summaries)
        atomic_json(a.output_dir/'manifest.json',m)
    for n,p in a.case_values:
        key=f'L{n}_P{p}'; m['active_case']=dict(case=key,phase='source_fixture'); write()
        runtime.prepare(n,p)
        sources[key]=runtime.source_report
        m.setdefault('cases',{})[key]=dict(geometry=runtime.geometry,token_ids=runtime.ids,text=runtime.text,
            namespace=__import__('dataclasses').asdict(runtime.identity()),preflight=runtime.preflight,source_forwards=runtime.source_forwards,
            snapshot=runtime.snapshot.identity_hash if runtime.snapshot else None,source_cache_api=runtime.snapshot.mode if runtime.snapshot else None)
        m['active_case']=dict(case=key,phase='untimed_forward_and_position_audit'); write()
        f=runtime.forward('FULL_USE_CACHE',audit=True); full=f['logits']
        native_repeat=runtime.forward('FULL_USE_CACHE')['logits']
        h=runtime.forward('EXACT_PREFIX_WARM',audit=True)
        correctness[key]=dict(full_repeat=logit_metrics(full,native_repeat),**compare_suffix(full,h['logits'],p),
            prefix_boundary=logit_metrics(full[:,p-1:p,:],runtime.source_logits[:,-1:,:]) if p else None,
            prefix_boundary_definition='Source last logit vs Full at P-1; separate from suffix first logit')
        skip[key]=dict(full=f['audit'],target=h['audit'],source=runtime.source_report.get('source_forward_audit'),
            source_forwards_total=runtime.source_forwards,prefix_source_forwards_inside_target=0,
            full_layer_token_rows=n*len(runtime.adapter.layers),target_layer_token_rows=(n-p)*len(runtime.adapter.layers),
            skipped_prefix_layer_token_rows=p*len(runtime.adapter.layers),status='VERIFIED')
        integrity[key]=dict(before=runtime.check(),target_output_kv_length=h['output_kv_length'],
            request_input_before_lengths=h.get('request_input_before_lengths'),request_input_after_lengths=h['request_input_after_lengths'],prepared_input=h['prepared_input_audit'],
            output_prefix=h['output_prefix_check'],repeat_checks=[],cache_api=h['cache_api'],actual_cache_class=h.get('actual_cache_class'))
        reference=h['logits']; del f,h,native_repeat
        negatives[key]=runtime.negative_tests(); runtime.check()
        # Repeated suffix-only Forward evidence, not a claim inferred from one hook pass.
        for i in range(3):
            m['active_case']=dict(case=key,phase='repeat_integrity',repeat=i); write()
            runtime.check(); again=runtime.forward('EXACT_PREFIX_WARM',audit=True)
            if not torch.equal(reference,again['logits']):
                raise PrefixError('REJECT_REPEATABILITY','identical target logits changed across fresh request caches')
            integrity[key]['repeat_checks'].append(dict(repeat=i,output_kv_length=again['output_kv_length'],
                input_before_lengths=again.get('request_input_before_lengths'),input_after_lengths=again['request_input_after_lengths'],canonical=runtime.check(),logits_exact=True,
                target_audit=again['audit']))
            del again
        if p==0 and not torch.equal(full,reference): raise PrefixError('REJECT_NO_PREFIX_PARITY','P=0 must be the same native Full path')
        del full,reference
        orders=schedule(MODES,a.warmup_runs,a.order_seed+n+p)+schedule(MODES,a.measured_runs,a.order_seed+n+p+100003)
        m['cases'][key]['orders']=orders; m['cases'][key]['gpu_before']=runtime.gpu_state()
        for i,order in enumerate(orders):
            phase='warmup' if i<a.warmup_runs else 'measured'; index=i if phase=='warmup' else i-a.warmup_runs
            result={}; pending=[]
            for mode in order:
                m['active_case']=dict(case=key,phase=phase,pair_index=index,mode=mode); write()
                runtime.check(); result[mode]=runtime.forward(mode); runtime.check()
                r=result[mode]
                pending.append(dict(case=key,phase=phase,pair_index=index,order=order,order_seed=a.order_seed,mode=mode,
                    total_length=n,prefix_length=p,suffix_length=n-p,cache_hit=r['cache_hit'],wall_ms=r['wall_ms'],
                    cache_prepare_host_ms=r['cache_prepare_host_ms'],cache_prepare_scope=r['cache_prepare_scope'],clone_bytes=r['request_clone_bytes'],
                    query_tokens=r['query_tokens'],output_kv_length=r['output_kv_length'],cache_api=r['cache_api'],actual_cache_class=r.get('actual_cache_class'),
                    source_identity=runtime.snapshot.identity_hash if runtime.snapshot else None,target_identity=digest(runtime.ids),
                    code_identity=m['code']['sha256'],model_revision=runtime.meta['resolved_model_revision'],adapter=a.adapter,
                    independent_run_id=a.independent_run_id,profiler_enabled=False,instrumented=False,dtype=a.dtype,memory=r['memory']))
            quality=compare_suffix(result[MODES[0]]['logits'],result[MODES[1]]['logits'],p)
            for row in pending:
                row['suffix_correctness_status']=quality['suffix']['correctness_status']; row['pair_quality']=quality
            samples.extend(pending); summaries=summarize(samples,sources); write()
            result.clear(); del r
        integrity[key]['after']=runtime.check(); integrity[key]['model_weights']=runtime.verify_weight_hashes()
        m['cases'][key]['gpu_after']=runtime.gpu_state(); write()
    failures=[v for v in correctness.values() if not v['suffix']['allclose'] or not v['last_position']['allclose'] or not v['full_repeat']['allclose'] or (v['prefix_boundary'] and not v['prefix_boundary']['allclose'])]
    m.update(execution_status='COMPLETE',forward_skip_status='VERIFIED',correctness_status='STRICT_FAIL' if failures or any(r['suffix_correctness_status']=='STRICT_FAIL' for r in samples) else 'STRICT_PASS')
    write()
    atomic_json(a.output_dir/'decision_summary.json',dict(execution_status='COMPLETE',classification='EXACT_PREFIX_BASELINE',
        validation_status='STRUCTURAL_VERIFIED',correctness_status=m['correctness_status'],
        b1_gate='STRICT_PASS' if m['correctness_status']=='STRICT_PASS' else 'NUMERIC_STRICT_FAIL_REQUIRES_DIAGNOSIS',
        fp16_numerical_cause='UNRESOLVED; shape sensitivity is a hypothesis, not inferred from strict failure',
        semantic_reuse_safety_claimed=False,speedup_claim_qualified=False,exploratory_case_summary=summaries,
        next_step='R2-B2 requires independent paired runs, matching use_cache/output settings, actual clone/residency and source-cost accounting; no R1 latency pooling'))


def main(argv=None):
    a=parse_args(argv); m=initialize(a)
    try:
        with raw_only_guard() as guard:
            try: run(a,m)
            finally: m['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',m); return 0
    except (Exception,KeyboardInterrupt,SystemExit) as exc:
        status=('OOM' if 'out of memory' in str(exc).lower() else 'MEMORY_PREFLIGHT_FAILED' if isinstance(exc,MemoryError)
            else 'UNSUPPORTED' if isinstance(exc,PrefixError) and exc.code.startswith('UNSUPPORTED') else 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED')
        m.update(execution_status=status,failure_type=type(exc).__name__,failure=str(exc),failure_code=getattr(exc,'code',None),traceback=traceback.format_exc())
        atomic_json(a.output_dir/'manifest.json',m)
        atomic_json(a.output_dir/'failure.json',dict(status=status,type=type(exc).__name__,code=getattr(exc,'code',None),reason=str(exc),context=m.get('active_case'),traceback=m['traceback']))
        atomic_json(a.output_dir/'decision_summary.json',dict(execution_status=status,validation_status='UNVERIFIED',correctness_status='UNVERIFIED',
            implementation_status=m['implementation_status'],semantic_reuse_safety_claimed=False,speedup_claim_qualified=False))
        traceback.print_exc(); return 1
if __name__=='__main__': raise SystemExit(main())
