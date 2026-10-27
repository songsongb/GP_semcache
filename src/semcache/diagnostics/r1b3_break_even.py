"""B3 scalar-only workload, artifact and FULL-relative cost analysis (no Torch import)."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from semcache.metrics.r1 import atomic_json, digest
from semcache.metrics.r1b2 import stats

class MissingEvidence(RuntimeError): pass
class ParityFailure(RuntimeError): pass


def code_identity(root):
    root=Path(root)
    paths=sorted((root/'src/semcache').rglob('*.py'))+sorted((root/'scripts').glob('*.py'))
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*args):
        r=subprocess.run(['git',*args],cwd=root,text=True,capture_output=True)
        return r.stdout.strip() if r.returncode==0 else None
    return dict(git_head=git('rev-parse','HEAD'),git_branch=git('branch','--show-current'),
                git_status=git('status','--short'),files=hashes,sha256=digest(hashes))


def create_run(root,output,config,experiment):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    manifest=dict(experiment=experiment,code=code_identity(root),config=config,
        classification='ORACLE_DIAGNOSTIC',execution_status='RUNNING',correctness_status='UNVERIFIED',
        settings=dict(use_cache=False,output_attentions=True,attention='eager',atol=1e-3,rtol=1e-3))
    manifest['identity']=digest(dict(code=manifest['code']['sha256'],config=config))
    atomic_json(output/'manifest.json',manifest)
    return manifest


def failure(output,manifest,exc):
    import traceback
    status=('OOM' if 'out of memory' in str(exc).lower() else 'MEMORY_PREFLIGHT_FAILED' if isinstance(exc,MemoryError)
            else 'PARITY_FAILED' if isinstance(exc,ParityFailure) else 'MISSING_EVIDENCE' if isinstance(exc,MissingEvidence)
            else 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED')
    manifest.update(execution_status=status,failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc())
    atomic_json(Path(output)/'manifest.json',manifest)
    atomic_json(Path(output)/'summary.json',dict(execution_status=status,failure=str(exc),context=manifest.get('active_case')))
    traceback.print_exc()
    return 1


def schedule(modes,count,seed):
    """Randomly permuted cyclic Latin blocks: each mode occupies each slot per block."""
    if not modes or len(set(modes))!=len(modes) or count<1: raise ValueError('Invalid paired schedule')
    rng=random.Random(seed); result=[]
    while len(result)<count:
        base=list(modes); rng.shuffle(base)
        block=[base[i:]+base[:i] for i in range(len(base))]; rng.shuffle(block)
        result.extend(block)
    return result[:count]


def target_variants(tokenizer,source_ids,source_text,count,seed,accept=None):
    """Change only the last (unreused) token; require exact tokenizer round trip."""
    if len(source_ids)%3==0: raise ValueError('Cold suffix fixture requires an unreused final token')
    rng=random.Random(seed); vocabulary=list(range(len(tokenizer))); rng.shuffle(vocabulary)
    special=set(tokenizer.all_special_ids); result=[]; seen={tuple(source_ids)}
    for token in vocabulary:
        if token in special: continue
        ids=list(source_ids[:-1])+[token]
        if tuple(ids) in seen: continue
        text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if list(tokenizer(text)['input_ids'])!=ids or (accept and not accept(text,ids)): continue
        seen.add(tuple(ids)); result.append(dict(text=text,ids=ids,identity=digest(ids)))
        if len(result)==count: return result
    raise MissingEvidence(f'Only {len(result)}/{count} cold valid tokenizer/cluster fixtures available')


def workload_target(workload,repeat,warm,cold):
    if workload in ('warm','noncontiguous'): return warm
    if workload=='cold': return cold[repeat]
    if workload=='mixed': return warm if repeat%2==0 else cold[repeat]
    raise ValueError('Unknown workload')


def fallback_filter(hits,length):
    chosen=hits[::2]; mask=[False]*length
    for h in chosen: mask[h.window.start:h.window.end]=[True]*(h.window.end-h.window.start)
    return chosen,mask


def amortized(full,warm,cold,incremental):
    if any(not math.isfinite(x) or x<0 for x in (full,warm,incremental)) or (cold is not None and (not math.isfinite(cold) or cold<0)):
        raise ValueError('Finite nonnegative milliseconds required')
    delta=full-warm
    result=dict(full_ms=full,warm_ms=warm,cold_ms=cold,incremental_source_ms=incremental,delta_warm_ms=delta,
        formula='N*FULL > S_incremental + COLD + (N-1)*WARM',baseline='FULL_RECOMPUTE')
    if delta<=0: return dict(result,n=None,status='NO_FINITE_BREAK_EVEN_UNDER_MEASURED_STEADY_STATE')
    if cold is None: return dict(result,n=None,status='MISSING_COLD_MAPPING_EVIDENCE')
    # Strict inequality: equality is not a speedup. One source snapshot shared by N targets.
    n=max(1,math.floor((incremental+cold-warm)/delta)+1)
    while n*full<=incremental+cold+(n-1)*warm: n+=1
    return dict(result,n=n,status='DERIVED_MEASURED_COST_MODEL',extrapolation=True)


def summarize_pairs(rows):
    groups={}
    for row in rows:
        if row['phase']=='measured':
            key=(row['length'],str(row['requested_coverage']),row['workload'],row['executor'],row['instrumentation'])
            groups.setdefault(key,[]).append(row)
    result=[]
    for key,group in groups.items():
        full=[r['full_ms'] for r in group]; reuse=[r['reuse_ms'] for r in group]; diff=[a-b for a,b in zip(full,reuse)]
        valid=all(r['executor_parity'] for r in group)
        strict=all(r['correctness_status']=='STRICT_PASS' for r in group)
        minimal=key[4]=='minimal'; observed=minimal and valid and statistics.mean(diff)>0
        cold=[r['reuse_ms'] for r in rows if r['source_snapshot_id']==group[0]['source_snapshot_id'] and r['executor']==key[3]
              and r['workload']==key[2] and str(r['requested_coverage'])==key[1] and r['mapping_cache_hit'] is False]
        warm_samples=[r['reuse_ms'] for r in group if r['mapping_cache_hit'] is True or r['mapping_cache_hit'] is None]
        item=dict(length=key[0],requested_coverage=key[1],workload=key[2],executor=key[3],instrumentation=key[4],
            full=dict(stats(full),min=min(full),max=max(full)),reuse=dict(stats(reuse),min=min(reuse),max=max(reuse)),paired_difference=stats(diff),paired_win_count=sum(d>0 for d in diff),
            paired_win_rate=sum(d>0 for d in diff)/len(diff),count=len(diff),speedup=statistics.mean(full)/statistics.mean(reuse),
            executor_parity=valid,correctness_status='STRICT_PASS' if strict else 'STRICT_FAIL',
            observed_break_even=observed,correctness_qualified_break_even=observed and strict and any(r['reused_tokens'] for r in group),
            decision='OBSERVED' if observed else 'NO_BREAK_EVEN_OBSERVED' if minimal else 'BREAKDOWN_NOT_PERFORMANCE_EVIDENCE',
            actual_coverage=stats([r['actual_coverage'] for r in group]),mapping_hits=sum(r['mapping_cache_hit'] is True for r in group),
            mapping_misses=sum(r['mapping_cache_hit'] is False for r in group),cold_sample_count=len(cold))
        if key[2]=='warm' and minimal and warm_samples:
            source=group[0]['source_costs']; inc=source['incremental_build_ms']
            c=statistics.mean(cold) if cold else (statistics.mean(reuse) if key[3] in ('legacy','v1') or not any(r['reused_tokens'] for r in group) else None)
            item['amortized_existing_source']=amortized(statistics.mean(full),statistics.mean(warm_samples),c,inc)
            item['amortized_dedicated_source']=amortized(statistics.mean(full),statistics.mean(warm_samples),c,inc+source['source_prefill_ms'])
            item['existing_source_assumption']='prefill AND QKV capture already sunk; extra capture overhead of ordinary native source is not isolated'
            item['amortization_scope']='one source snapshot shared by N identical targets; cold observed during warmup; model-derived, not N-request experiment'
        else: item['amortization_status']='NOT_IDENTICAL_WARM_STEADY_STATE_OR_NOT_MINIMAL'
        result.append(item)
    return result


def hierarchical(pairs_by_run,seed=1,samples=5000):
    if samples<100 or not pairs_by_run or any(not v for v in pairs_by_run.values()): raise ValueError('Paired runs and >=100 resamples required')
    ids=sorted(pairs_by_run); rng=random.Random(seed); draws=[]
    for _ in range(samples):
        means=[]
        for _ in ids:
            values=pairs_by_run[ids[rng.randrange(len(ids))]]
            means.append(statistics.mean(values[rng.randrange(len(values))] for _ in values))
        draws.append(statistics.mean(means))
    draws.sort(); means=[statistics.mean(pairs_by_run[k]) for k in ids]
    return dict(mean=statistics.mean(means),between_run_std=statistics.stdev(means) if len(means)>1 else None,
        ci95=[draws[int(.025*(samples-1))],draws[int(.975*(samples-1))]],seed=seed,samples=samples,
        independent_runs=len(ids),scope='equal-weight independent runs, then paired repeats',
        conclusion_scope='DESCRIPTIVE_ONLY' if len(ids)<3 else 'THREE_OR_MORE_INDEPENDENT_RUNS')
