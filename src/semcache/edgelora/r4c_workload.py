"""Separate proactive warming, natural discovery, and generation policies."""
from .communication import digest, require
from .r4c_prompts import prefix_of, validate_prepared

LABEL = 'CONTROLLED_PROACTIVE_PREFIX_WARMING'


def policy(special, mode, budget):
    require(mode in ('quality', 'fixed') and type(budget) is int and 1 <= budget <= 128, 'invalid bounded greedy policy')
    require(set(special) == {'bos_token_id','eos_token_id','pad_token_id'} and all(type(t) is int and t >= 0 for t in special.values()), 'invalid special tokens')
    return dict(version='R4C_GREEDY_V1', mode=mode, max_new_tokens=budget, do_sample=False,
                stop_at_eos=mode == 'quality', decode_skip_special_tokens=True,
                decode_clean_up_tokenization_spaces=False, **special)


def termination(p, generated):
    require(generated and len(generated) <= p['max_new_tokens'], 'invalid generated count')
    return 'EOS' if p['stop_at_eos'] and generated[-1] == p['eos_token_id'] else 'LENGTH_CAP' if len(generated) == p['max_new_tokens'] else None


def schedule(prompts, special, *, prefix_length=128, mode='quality', strict_count=1):
    require(0 <= strict_count <= len(prompts), 'invalid strict cohort size')
    tasks = []
    for i, target in enumerate(prompts):
        validate_prepared(target); source = prefix_of(target, prefix_length)
        pair = f'C{i:04d}'; profile = 'STRICT_AUDIT' if i < strict_count else 'SCALABLE_EVALUATION'
        for suffix, cache_mode, prompt, p in (
                ('P', 'COLD_MISS', source, policy(special, 'fixed', 1)),
                ('N', 'NO_REUSE', target, policy(special, mode, target['max_new_tokens'])),
                ('H', 'PARTIAL_PREFIX_HIT', target, policy(special, mode, target['max_new_tokens']))):
            tasks.append(dict(request_id=pair+'_'+suffix, pair_id=pair, sample_id=target['sample_id'], mode=cache_mode,
                              prompt=prompt, policy=p, prefix_length=prefix_length, evidence_profile=profile,
                              workload_label=LABEL, source_request_id=pair+'_P', dataset=target['dataset']))
    require(len(tasks)<=126 and sum(t['policy']['max_new_tokens'] for t in tasks)<=8192,'bounded C1 pilot supports <=42 targets / 8192 forwards per process')
    return tasks


def natural_discovery(prompts, *, minimum=16, layers=None, hidden=None, element_bytes=None,raw_cache_bytes_per_token=None):
    require(minimum > 0, 'positive natural discovery boundary required')
    pairs = []; touched = set(); unique={}
    bytes_per_token=raw_cache_bytes_per_token if raw_cache_bytes_per_token is not None else None if layers is None else 3*layers*hidden*element_bytes
    # Bounded offline pair table, never claims physical cache hits.
    require(len(prompts) <= 4096, 'bounded natural discovery supports at most 4096 prepared prompts')
    for i, a in enumerate(prompts):
        for b in prompts[i+1:]:
            n = 0
            for x,y in zip(a['token_ids'],b['token_ids']):
                if x != y: break
                n += 1
            n = min(n, len(a['token_ids'])-1, len(b['token_ids'])-1)
            if n >= minimum:
                touched.update((a['sample_id'],b['sample_id']))
                pairs.append(dict(a=a['sample_id'], b=b['sample_id'], prefix_tokens=n,
                    a_reusable_fraction=n/len(a['token_ids']), b_reusable_fraction=n/len(b['token_ids']),
                    raw_qkv_capacity_bytes=None if bytes_per_token is None else bytes_per_token*n))
                unique[tuple(a['token_ids'][:n])]=n
    return dict(category='NATURAL_EXACT_PREFIX_DISCOVERY_ONLY', physical_hits_executed=False,
        records_inspected=len(prompts), minimum_prefix_tokens=minimum, eligible_pairs=len(pairs), matches=pairs,
        raw_cache_bytes_per_token=bytes_per_token,independently_owned_discovered_prefix_capacity_bytes=None if bytes_per_token is None else sum(unique.values())*bytes_per_token,
        records_without_useful_overlap=[p['sample_id'] for p in prompts if p['sample_id'] not in touched],
        limitation='Template tokens may dominate overlap. No natural hit-rate or traffic model inferred.')
