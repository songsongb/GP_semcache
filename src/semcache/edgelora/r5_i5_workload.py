"""Tokenizer-exact preparation, deterministic selection and independent bounded blocks."""
from copy import deepcopy
import math
from .communication import digest, require
from .r4c_workload import policy
from .r5_i5_data import LABELS, load_dataset
from .r5_i5_evidence import fresh, seal
from .r3a_artifacts import write, sha

CONDITIONS = ('FULL_RECOMPUTE', 'SEMANTIC_NO_REUSE', 'SEMCACHE_INTEGRATED')
ORDERS = ((0,1,2), (2,1,0), (1,2,0), (0,2,1), (2,0,1), (1,0,2))
SCALES = dict(SMOKE=10, SMALL=100, MEDIUM=500, LARGE=1000)
BUDGETS = dict(coqa=32, snips=16, multiwoz=64, qasper=32, **{'2wikimqa':32, 'multi_news':96})
TEMPLATE = 'I5_PROMPTS_V1'


SCALE_SEMANTICS = 'INDEPENDENT_BLOCK_DATASET_SCALE'


def routing(sample, seed=42):
    identity=[sample['dataset'],sample['version'],sample['group_id'] if sample['dataset'] in ('coqa','multiwoz') else sample['id'],seed]
    user=int(digest(identity),16)%30+1
    return dict(user_id=f'User_{user:02d}',adapter_id='Adapter_'+'ABC'[(user-1)//10])


def turn_order(sample):
    value=sample.get('turn_id',0)
    try:return (0,int(value),sample['id'])
    except (ValueError,TypeError):return (1,str(value),sample['id'])


def pack_groups(selected, limit):
    """Contiguous groups, next-fit whole groups; oversized groups chunk at limit."""
    require(1<=limit<=20,'packing limit outside existing safety bound')
    groups={}
    for p in selected:
        s=p['sample'];groups.setdefault((s['dataset'],s['version'],s['group_id']),[]).append(p)
    packed=[];current=[]
    for group in groups.values():
        group.sort(key=lambda p:turn_order(p['sample']))
        if len(group)>limit:
            if current:packed.append(current);current=[]
            packed.extend(group[i:i+limit] for i in range(0,len(group),limit))
        else:
            if current and len(current)+len(group)>limit:packed.append(current);current=[]
            current.extend(group)
    if current:packed.append(current)
    return packed


def continuity(cohorts):
    locations={}
    for i,cohort in enumerate(cohorts):
        for p in cohort:locations.setdefault(p['sample']['group_id'],set()).add(i)
    return [dict(group_id=g,block_ordinals=sorted(bs),reason='GROUP_EXCEEDS_BLOCK_LIMIT',cache_survives=False)
            for g,bs in locations.items() if len(bs)>1]


def parts(s):
    name = s['dataset']
    if name == 'coqa':
        return 'Read the story and answer the current question briefly.\nStory:\n', '\n' + s['history'] + '\nQuestion: ' + s['question'] + '\nAnswer:'
    if name == 'snips':
        return 'Classify the intent. Output exactly one label from: ' + ', '.join(LABELS) + '.\n', 'Utterance: ' + s['question'] + '\nIntent:'
    if name == 'multiwoz':
        return 'Track the complete dialogue belief state. Output only a JSON object of slot names to string values. Omit unset slots. Allowed slots: ' + ', '.join(s['ontology']) + '.\n', s['history'] + '\nState:'
    if name == 'multi_news':
        return 'Summarize the news passages.\nNews:\n', '\n' + s['question'] + '\nConcise summary:'
    return 'Answer using the passages. Give only the answer; say unanswerable if unsupported.\nPassages:\n', '\nQuestion: ' + s['question'] + '\nAnswer:'


def prepare_prompt(s, tok, context_limit, budget, *, allow_truncation=False, input_limit=None):
    require(type(budget) is int and 1 <= budget <= 128, 'I5 supports 1..128 generation tokens')
    require(type(context_limit) is int and 1 < context_limit <= 2048, 'I5 OPT positional limit required')
    limit = min(input_limit or context_limit-budget, context_limit-budget)
    before, after = parts(s); enc = lambda t: list(tok.encode(t, add_special_tokens=False))
    context = s['context']; cids = enc(context); original = enc(before+context+after)
    require(len(enc(before+after)) <= limit, 'INELIGIBLE_FIXED_FIELDS: instruction/question/history exceed context; never truncated')
    keep = len(cids); retained = context
    if len(original) > limit:
        require(allow_truncation and context, 'INELIGIBLE_CONTEXT: complete prompt plus generation exceeds context')
        while True:
            head, tail = (keep+1)//2, keep//2
            retained = tok.decode(cids[:head], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            retained += '\n[...context omitted...]\n'
            if tail: retained += tok.decode(cids[-tail:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            ids = enc(before+retained+after)
            if len(ids) <= limit: break
            require(keep > 0, 'INELIGIBLE_CONTEXT: omission marker exceeds budget')
            keep -= max(1, min(keep, len(ids)-limit))
    text = before+retained+after; ids = enc(text)
    require(ids and len(ids)+budget <= context_limit, 'context budget exceeded')
    return dict(sample=deepcopy(s), input_text=text, token_ids=ids, token_ids_sha256=digest(ids),
        template_version=TEMPLATE, max_new_tokens=budget, context_limit=context_limit,
        original_context_tokens=len(cids), retained_context_tokens=keep, removed_context_tokens=len(cids)-keep,
        original_prompt_tokens=len(original), retained_prompt_tokens=len(ids), truncated=keep<len(cids),
        evaluation_label='LONGBENCH_DERIVED_CONTEXT_LIMITED_EVALUATION' if s['version']=='LongBench-v1' and keep<len(cids)
                         else 'FULL_CONTEXT_ELIGIBLE', truncation_policy='context-only token head/tail v1; complete prompt retokenized; fixed fields untouched')


def select(samples, tok, *, count, seed, context_limit, budget, prefix=None, allow_truncation=False, input_limit=None):
    eligible = []; excluded = []
    for s in samples:
        try:
            p = prepare_prompt(s, tok, context_limit, budget, allow_truncation=allow_truncation, input_limit=input_limit)
            require(prefix is None or 0 < prefix < len(p['token_ids']), 'PREFIX_HAS_NO_NONEMPTY_SUFFIX')
            eligible.append(p)
        except ValueError as exc:
            excluded.append(dict(id=s['id'], reason=str(exc)))
    # Group order hash; turns follow source order within a conversation.
    eligible.sort(key=lambda p:(digest([seed, p['sample']['group_id']]), turn_order(p['sample'])))
    report = dict(requested=count, total=len(samples), eligible=len(eligible), exclusions=excluded,
        selected_ids=[p['sample']['id'] for p in eligible[:count]], not_selected_ids=[p['sample']['id'] for p in eligible[count:]],
        ordering='SHA256(seed,conversation/group); numeric turn ID; original ID; no controller HIT annotations', seed=seed,
        partial_selected_groups=[dict(group_id=g,eligible_turns=sum(p['sample']['group_id']==g for p in eligible),selected_turns=sum(p['sample']['group_id']==g for p in eligible[:count]),reason='EXACT_SAMPLE_COUNT_SELECTION_BOUNDARY') for g in sorted({p['sample']['group_id'] for p in eligible[:count]}) if sum(p['sample']['group_id']==g for p in eligible)>sum(p['sample']['group_id']==g for p in eligible[:count])])
    if len(eligible) < count:
        raise EligibilityError(report)
    return eligible[:count], report


class EligibilityError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f'INSUFFICIENT_ELIGIBLE_DATA: requested={report["requested"]}, eligible={report["eligible"]}; no substitution/duplication')


def requests(prompts, *, track, prefix, offset=0, special, routing_seed=42):
    require(track in ('NATURAL', 'CONTROLLED'), 'invalid workload track')
    tasks = []
    for i,p in enumerate(prompts):
        n = offset+i; route=routing(p['sample'],routing_seed)
        base = dict(input_text=p['input_text'], token_ids=p['token_ids'], prepared=p,
                    dataset=p['sample']['dataset'], sample_id=p['sample']['id'], group_id=p['sample']['group_id'],
                    **route, pair_id=f'P{n:06d}')
        if track == 'CONTROLLED':
            require(type(prefix) is int and 0 < prefix < len(p['token_ids']), 'invalid exact prefix')
            # Source IDs are sliced, never decode/re-encoded. Text is semantic input only.
            tasks.append(dict(base, request_id=f'P{n:06d}S', token_ids=p['token_ids'][:prefix],
                input_text=p['source_semantic_text'], role='PRIMING_SOURCE', scored=False,
                policy=policy(special, 'fixed', 1), strict=p['strict']))
        tasks.append(dict(base, request_id=f'P{n:06d}T', role='SCORED_TARGET', scored=True,
            policy=policy(special, 'quality', p['max_new_tokens']), strict=p['strict']))
    for i,t in enumerate(tasks): t['request_sequence_number']=i
    return tasks


def resources(tasks, *, layers=32, hidden=2560, capacity=512*2**20, evidence_budget=8*2**30,
              seconds_per_forward=5., strict_count=0):
    require(0 < capacity <= 2**30 and 128*2**20 <= evidence_budget <= 16*2**30, 'bounded cache/evidence required')
    bpt = 3*layers*hidden*2; max_input=max(len(t['token_ids']) for t in tasks)
    max_total=max(len(t['token_ids'])+t['policy']['max_new_tokens'] for t in tasks)
    forwards=sum(t['policy']['max_new_tokens'] for t in tasks) + 3  # 3 one-token uncached model warmups
    # Conservative storage estimate: last logits + verbose TCP/context ledgers + sampled full tensors.
    per_forward_json=2*layers*(32768 + max_input*100)
    disk = forwards*(50272*4+per_forward_json) + sum(16*layers*hidden*len(t['token_ids'])*2 for t in tasks if t['strict'])
    model=2700000000*2; kv=2*layers*hidden*max_total*2
    activation=4*2**30 + 32*max_input*max_input*2 + max_input*50272*4
    estimate=dict(raw_qkv_bytes_per_token=bpt, per_adapter_capacity_bytes=capacity, total_cache_capacity_bytes=3*capacity,
        max_source_tokens_fitting=capacity//bpt, maximum_input_tokens=max_input, maximum_context_tokens=max_total,
        model_fp16_estimate_bytes=model, active_kv_estimate_bytes=kv, transient_raw_qkv_estimate_bytes=max_input*bpt,
        activation_workspace_reserve_bytes=activation, gpu_estimate_bytes=model+3*capacity+kv+max_input*bpt+activation,
        cpu_memory_reserve_bytes=32*2**30, estimated_evidence_bytes=disk, evidence_budget_bytes=evidence_budget,
        expected_maximum_forwards=forwards, duration_estimate_seconds=forwards*seconds_per_forward,
        duration_estimate_source='user-configurable planning assumption; replace with measured smoke forwards',
        cache_eligible_requests=sum(len(t['token_ids'])*bpt<=capacity for t in tasks),
        longest_priming_source_bytes=max((len(t['token_ids'])*bpt for t in tasks if not t['scored']), default=0),
        strict_requests=sum(t['strict'] for t in tasks), gpu_limit_bytes=24*2**30)
    require(estimate['longest_priming_source_bytes'] <= capacity, 'CACHE_CAPACITY: priming prefix cannot fit')
    require(estimate['gpu_estimate_bytes'] < 22*2**30, 'GPU_RESOURCE_ESTIMATE exceeds conservative 22 GiB')
    require(disk < evidence_budget, 'EVIDENCE_RESOURCE_ESTIMATE exceeds block budget; reduce block/budget/generation')
    return estimate


def prepare_experiment(registry, tok, tokenizer_identity, context_limit, special, output, *, scale='SMOKE',
        overrides=None, budgets=None, tracks=('NATURAL',), prefixes=(64,128,256,512), repetitions=2,
        block_size=10, seed=42, strict_per_block=1, capacity=512*2**20, evidence_budget=8*2**30,
        allow_truncation=False, fixture=False, input_limit=None, seconds_per_forward=5.):
    out=fresh(output, registry.values()); blocks=[]; dataset_records={}; overrides=overrides or {}; budgets=budgets or {}
    require(scale in SCALES and 1<=block_size<=20 and 1<=repetitions<=12 and 1<=strict_per_block<=block_size, 'invalid bounded experiment design')
    require(tracks and set(tracks)<= {'NATURAL','CONTROLLED'} and all(p in (64,128,256,512,1024) for p in prefixes), 'unsupported track/prefix')
    try:
        for name,path in registry.items():
            samples,m=load_dataset(path, fixture=fixture); require(name==m['dataset'], 'dataset registry key mismatch')
            count=overrides.get(name,SCALES[scale]); require(type(count) is int and 0<count<=1000, 'dataset count outside 1..1000')
            dataset_records[name]=dict(m, prepared_path=str(path), prepared_manifest_sha256=sha(__import__('pathlib').Path(path)/'i5_output_manifest.json'), selections=[])
            for track in tracks:
                for prefix in (prefixes if track=='CONTROLLED' else (None,)):
                    selected,selection=select(samples,tok,count=count,seed=seed,context_limit=context_limit,
                        budget=budgets.get(name,BUDGETS[name]),prefix=prefix,allow_truncation=allow_truncation,input_limit=input_limit)
                    dataset_records[name]['selections'].append(dict(selection,track=track,prefix=prefix,
                        truncation_count=sum(p['truncated'] for p in selected),truncation_rate=sum(p['truncated'] for p in selected)/count,
                        lengths=[{k:p[k] for k in ('original_context_tokens','retained_context_tokens','removed_context_tokens','retained_prompt_tokens','truncated')} for p in selected]))
                    cohorts=pack_groups(selected,block_size);breaks=continuity(cohorts)
                    for rep in range(repetitions):
                        start=0
                        for ordinal,packed in enumerate(cohorts):
                            cohort=deepcopy(packed)
                            # Predetermined hash-ranked group selection; source and target both strictly captured.
                            chosen=set(sorted(range(len(cohort)),key=lambda i:digest([seed,cohort[i]['sample']['group_id']]))[:strict_per_block])
                            for i,p in enumerate(cohort):
                                p['strict']=i in chosen
                                if prefix is not None: p['source_semantic_text']=tok.decode(p['token_ids'][:prefix], skip_special_tokens=False,clean_up_tokenization_spaces=False)
                            tasks=requests(cohort,track=track,prefix=prefix,offset=start,special=special,routing_seed=seed)
                            block=dict(dataset=name,track=track,prefix_length=prefix,repetition=rep,scale=scale,
                                scale_semantics=SCALE_SEMANTICS,routing_seed=seed,packing='GROUP_NEXT_FIT_V1',
                                continuity_breaks=[b for b in breaks if ordinal in b['block_ordinals']],dataset_continuity_break=ordinal>0,
                                cache_lifetime_requests=len(tasks),physical_requests_per_condition=len(tasks),cache_resets_per_condition=1,
                                scored_unique_samples=len(cohort), repeated_evaluation_samples=0 if rep==0 else len(cohort),
                                priming_requests=sum(not t['scored'] for t in tasks), tasks=tasks,
                                condition_order=[CONDITIONS[i] for i in ORDERS[2*((rep//2+ordinal)%3)+rep%2]],
                                reset='new ES/model/CPU UD sessions and empty cache/index per condition', warmup='one uncached one-token request per adapter; no admission; separate timings')
                            block['resources']=resources(tasks,capacity=capacity,evidence_budget=evidence_budget,seconds_per_forward=seconds_per_forward)
                            block['block_id']=digest(block); blocks.append(block);start+=len(cohort)
        config=dict(schema='I5_EXPERIMENT_V1',scale_semantics=SCALE_SEMANTICS,routing_algorithm='SHA256_CANONICAL_DATASET_VERSION_GROUP_SEED_MOD30_V1',scale=scale,seed=seed,blocks=blocks,tokenizer_identity=tokenizer_identity,
            context_limit=context_limit,special_tokens=special,template_version=TEMPLATE,TEST_FIXTURE_ONLY=fixture,
            conditions=CONDITIONS,clusters=1,policy='LRU_FIT_COMPLETE_SOURCE_V1',max_entries_per_adapter=64,
            shadow_equivalence='same metadata eligibility/LRU bytes and virtual HIT no-admission; tensor failures may diverge; analyzer checks source-decision equivalence',
            balanced_order_cycle=ORDERS,replications=repetitions,block_size=block_size)
        config['cache_lifetime_assessment']=dict(continuous_20_requests='SUPPORTED_WITHIN_ONE_BLOCK_SUBJECT_TO_RESOURCE_PREFLIGHT',
            longer_sessions='NOT_SUPPORTED_BY_CURRENT_20_TARGET_BLOCK_BOUND',cross_job_live_cache='NOT_SUPPORTED',
            future_work='Continuous 1000-request cache study; validated residency/proof serialization or a single bounded allocation required; never restore partial live cache',
            startup_cost='model reload and three CPU UD restarts per condition/block; measured separately from matched latency')
        config['scale_summary']=dict(unique_scored_samples=len({(b['dataset'],t['sample_id']) for b in blocks for t in b['tasks'] if t['scored']}),
            repetitions=repetitions,independent_blocks=len(blocks),cache_resets_all_conditions=3*len(blocks),
            maximum_cache_lifetime_requests=max(b['cache_lifetime_requests'] for b in blocks))
        config['experiment_id']=digest(config)
        write(out/'i5_dataset_manifest.json',dataset_records);write(out/'i5_workload_manifest.json',config)
        seal(out); return config
    except EligibilityError as exc:
        write(out/'i5_eligibility_failure.json',exc.report);seal(out);raise
