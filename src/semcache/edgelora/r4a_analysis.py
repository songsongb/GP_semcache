"""Measured paired-block TTFT/E2E contrasts; fidelity and Native gates separate."""
import csv
import random
import statistics

from .communication import require
from .r3a_artifacts import check_parity,read,sha,unpack,write
from .r3c_analysis import percentile
from .r4a_evidence import flags


def blocked_contrast(cache,full,metric,*,fixture=False):
    require(cache and len(cache)==len(full),'matched pairs required')
    refs={r['comparison_id']:r for r in full};require(len(refs)==len(full),'duplicate comparator')
    pairs=[]
    for row in cache:
        require(row['comparison_id'] in refs,'unmatched latency pair');base=refs[row['comparison_id']]
        require(row['policy']==base['policy'] and row['generated_tokens']==base['generated_tokens'],'output-length/policy timing mismatch')
        pairs.append(dict(comparison_id=row['comparison_id'],block=row['block'],delta_ms=row[metric]-base[metric]))
    blocks=sorted({p['block'] for p in pairs});means=[statistics.mean(p['delta_ms'] for p in pairs if p['block']==b) for b in blocks]
    rng=random.Random(42);draws=[statistics.mean(rng.choices(means,k=len(means))) for _ in range(10000)]
    lo,hi=percentile(draws,.025),percentile(draws,.975);fixed=all(r['policy']['mode']=='fixed' for r in cache+full)
    observed=not fixture and fixed and len(cache)>=8 and len(blocks)>=2 and hi<0 and all(p['delta_ms']<0 for p in pairs)
    return dict(metric=metric,delta_mean_ms=statistics.mean(p['delta_ms'] for p in pairs),ci95_low_ms=lo,ci95_high_ms=hi,
        benefit_observed=observed,status='CACHE_FASTER_DIAGNOSTIC' if observed else 'UNRESOLVED_OR_UNFAVORABLE',
        raw_paired_deltas=pairs,block_mean_deltas_ms=means,pairs=len(pairs),repeat_blocks=len(blocks),resamples=10000,seed=42,
        timing_comparable=fixed,uncertainty='whole repeat-block bootstrap; two blocks is weak conditional evidence, not independent deployments')


def summarize(rows):
    return dict(n=len(rows),**{metric:dict(mean=statistics.mean(r[metric] for r in rows),median=statistics.median(r[metric] for r in rows),
        p90=percentile([r[metric] for r in rows],.9)) for metric in ('ttft_ms','e2e_wall_ms')},
        sequential_requests_per_second=len(rows)*1000/sum(r['e2e_wall_ms'] for r in rows),
        generated_tokens_per_second=sum(r['generated_tokens'] for r in rows)*1000/sum(r['e2e_wall_ms'] for r in rows),
        rate_scope='matched sequential request walls excluding outside-wall reference/evidence exports; not concurrent serving')


def analyze(directory,prepared_directory,rows,tokens,worker,*,fixture=False):
    directory=__import__('pathlib').Path(directory);prepared_directory=__import__('pathlib').Path(prepared_directory)
    per_request={r['request_id']:[s for s in tokens if s['request_id']==r['request_id']] for r in rows}
    measured=[r for r in rows if r['phase']=='measured'];contrasts=[];by_cell={};fidelity=[]
    native=read(prepared_directory/'native_index.json');native_by={(r['cell_id'],r['sample_id']):r for r in native['generations']}
    native_all=bool(worker['native_equivalent_prefill']) and not fixture;native_checks=[]
    def values(row,root):
        path=root/row['logits_file'];require(sha(path)==row['logits_sha256'],'stored logit bytes SHA mismatch');return unpack(path.read_bytes(),'float32')
    for cell in worker['completed_cells']:
        block=[r for r in measured if r['cell_id']==cell];baseline=[r for r in block if r['mode']=='NO_REUSE'];by_cell[cell]={}
        for mode in ('NO_REUSE','COLD_MISS','WARM_HIT'):by_cell[cell][mode]=summarize([r for r in block if r['mode']==mode])
        for mode in ('COLD_MISS','WARM_HIT'):
            selected=[r for r in block if r['mode']==mode]
            for metric in ('ttft_ms','e2e_wall_ms'):contrasts.append(dict(cell_id=cell,mode=mode,**blocked_contrast(selected,baseline,metric,fixture=fixture)))
            refs={r['comparison_id']:r for r in baseline}
            for r in selected:
                b=refs[r['comparison_id']];equal=(r['generated_token_ids']==b['generated_token_ids'] and r['generated_text']==b['generated_text'] and
                    r['logits_sha256']==b['logits_sha256'] and
                    [s['qkv_sha256'] for s in per_request[r['request_id']]]==[s['qkv_sha256'] for s in per_request[b['request_id']]])
                fidelity.append(dict(request_id=r['request_id'],reference_request_id=b['request_id'],bitwise_equal=equal))
                require(equal,'distributed full-recompute/reuse fidelity failed')
        cold={r['comparison_id']:r for r in block if r['mode']=='COLD_MISS'}
        by_cell[cell]['cold_fill_plus_one_resident_amortized_e2e_delta_ms']=statistics.mean(
            (cold[r['comparison_id']]['e2e_wall_ms']+r['e2e_wall_ms'])/2-next(b['e2e_wall_ms'] for b in baseline if b['comparison_id']==r['comparison_id'])
            for r in block if r['mode']=='WARM_HIT')
        by_cell[cell]['amortization_scope']='two measured served requests (cold+resident) / 2 versus one reused matched NO_REUSE observation; not an independently measured two-NO_REUSE batch'
        if not fixture:
            for r in baseline:
                ref=native_by[cell,r['sample_id']];a,b=values(r,directory),values(ref,prepared_directory)
                width=r['logits_shape'][-1];checks=[]
                for i in range(min(r['generated_tokens'],ref['generated_tokens'])):
                    same_history=r['generated_token_ids'][:i]==ref['generated_token_ids'][:i]
                    metric=check_parity(a[i*width:(i+1)*width],b[i*width:(i+1)*width],[1,1,width])
                    checks.append(dict(step=i,same_input_history=same_history,strict_metrics=metric))
                passed=r['generated_token_ids']==ref['generated_token_ids'] and all(c['same_input_history'] and c['strict_metrics']['passed'] for c in checks)
                native_all=native_all and passed
                native_checks.append(dict(request_id=r['request_id'],passed=passed,steps=checks,
                    note='after histories diverge, free-run logit differences are not matched-input numerical attribution'))
    decision=dict(execution_status='COMPLETE',R4A_PHYSICAL_E2E_PASS=not fixture,R4A_RAW_QKV_E2E_FIDELITY_PASS=not fixture and all(r['bitwise_equal'] for r in fidelity),
        R4A_TTFT_BENEFIT_OBSERVED=any(c['benefit_observed'] for c in contrasts if c['mode']=='WARM_HIT' and c['metric']=='ttft_ms'),
        R4A_E2E_BENEFIT_OBSERVED=any(c['benefit_observed'] for c in contrasts if c['mode']=='WARM_HIT' and c['metric']=='e2e_wall_ms'),
        NATIVE_PEFT_STRICT_PARITY_PASS=native_all,native_equivalent=native_all,R4A_GOLD_BLEU_AVAILABLE=False,
        fixture_checks_passed=fixture,completed_cells=worker['completed_cells'],by_cell=by_cell,native_generation_checks=native_checks,
        fidelity_checks=fidelity,full_deployment_lifecycle='NOT_ESTIMABLE',**flags(fixture))
    decision['correctness_qualified_speedup']=bool(native_all and decision['R4A_RAW_QKV_E2E_FIDELITY_PASS'] and decision['R4A_E2E_BENEFIT_OBSERVED'])
    return contrasts,decision


def export(directory,contrasts,decision):
    write(directory/'r4a_decision.json',decision)
    fields=('cell_id','mode','metric','delta_mean_ms','ci95_low_ms','ci95_high_ms','pairs','repeat_blocks','benefit_observed','status')
    with (directory/'r4a_latency_contrasts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for c in contrasts:writer.writerow({k:c[k] for k in fields})
