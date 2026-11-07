"""CPU natural opportunity audit and explicitly controlled I5 pilot preparation.

No model forward, encoder, cache policy or physical eligibility is implemented here.
"""
from collections import Counter,defaultdict,OrderedDict
from copy import deepcopy
from pathlib import Path
import json
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,read,write,sha
from .r5_i5_evidence import fresh,seal
from .r5_i5r2_tensors import Evidence
from .r5_i5r3a import provenance
from .r5_i5_workload import prepare_prompt,select,routing,parts,pack_groups,BUDGETS,TEMPLATE
from .r5_multi_artifacts import validate,sharing
from .r5_i5r1_plan import source_identity

DATASETS=('qasper','2wikimqa','multi_news','snips','multiwoz')
NAMESPACE=ROOT/'results/r5/i5r/parallel'
CAPACITY=512*2**20

def output(path,protected=()):
    p=Path(path).resolve()
    require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'parallel experiment namespace required')
    require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps input')
    require(not p.exists(),'fresh output required');p.mkdir(parents=True);return p

def assets(registry,multi):
    paths=read(registry);require(set(paths)<=set(DATASETS) and paths,'explicit dataset registry required')
    m,pm,specs,users=validate(multi)
    require(not m['TEST_FIXTURE_ONLY'] and len(users['users'])==30,'real expanded 30-user preparation required')
    from .r4c_prepare import opt_tokenizer
    tok=opt_tokenizer(pm);require(tok.supported_context_length==2048 and tok.identity['raw_cache_bytes_per_token']==491520,'verified OPT tokenizer required')
    return paths,m,pm,specs,users,tok

def grams(ids):
    out=defaultdict(list)
    for i in range(len(ids)-2):out[tuple(ids[i:i+3])].append(i)
    return out

def lcp(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n

def longest_contiguous(a,b):
    # Suffix automaton: O(len(a)+len(b)); never enumerates occurrence pairs.
    states=[dict(next={},link=-1,length=0)];last=0
    for x in a:
        cur=len(states);states.append(dict(next={},link=0,length=states[last]['length']+1));p=last
        while p>=0 and x not in states[p]['next']:states[p]['next'][x]=cur;p=states[p]['link']
        if p>=0:
            q=states[p]['next'][x]
            if states[p]['length']+1==states[q]['length']:states[cur]['link']=q
            else:
                clone=len(states);states.append(dict(next=dict(states[q]['next']),link=states[q]['link'],length=states[p]['length']+1))
                while p>=0 and states[p]['next'].get(x)==q:states[p]['next'][x]=clone;p=states[p]['link']
                states[q]['link']=states[cur]['link']=clone
        last=cur
    v=n=best=0
    for x in b:
        while v and x not in states[v]['next']:v=states[v]['link'];n=states[v]['length']
        if x in states[v]['next']:v=states[v]['next'][x];n+=1
        else:v=n=0
        best=max(best,n)
    return best

def token_origins(p,tok):
    """Offsets when supported; otherwise only byte-verified fixed prefix labels."""
    text=p['input_text'];s=p['sample'];before,after=parts(s)
    labels=['UNKNOWN']*len(p['token_ids'])
    # Do not recover missing offsets by pretending independent tokenization aligns.
    try:
        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        require(list(encoded['input_ids'])==p['token_ids'],'offset tokenizer differs')
        context_end=len(text)-len(after)
        marker='\n[...context omitted...]\n';marker_span=None
        if p.get('truncated') and text.count(marker)==1:
            lo=text.index(marker);marker_span=(lo,lo+len(marker))
        content_spans=[]
        for field in ('question','history'):
            value=s.get(field,'')
            if value and after.count(value)==1:
                start=context_end+after.index(value);content_spans.append((start,start+len(value),field.upper()))
        for i,(a,b) in enumerate(encoded['offset_mapping']):
            if b<=a:continue
            if b<=len(before):labels[i]='TEMPLATE'
            elif a>=len(before) and b<=context_end:
                if marker_span and a<marker_span[1] and b>marker_span[0]:labels[i]='TEMPLATE' if marker_span[0]<=a and b<=marker_span[1] else 'CROSS_BOUNDARY'
                else:labels[i]='CONTEXT'
            elif a<len(before)<b or a<context_end<b:labels[i]='CROSS_BOUNDARY'
            else:
                overlaps=[(lo,hi,kind) for lo,hi,kind in content_spans if a<hi and b>lo]
                if len(overlaps)==1 and overlaps[0][0]<=a and b<=overlaps[0][1]:labels[i]=overlaps[0][2]
                elif overlaps:labels[i]='CROSS_BOUNDARY'
                elif a>=context_end:labels[i]='TEMPLATE'
    except (TypeError,NotImplementedError,KeyError,AttributeError):
        n=lcp(p['token_ids'],list(tok.encode(before,add_special_tokens=False)))
        decoded=tok.decode(p['token_ids'][:n],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if before.startswith(decoded):labels[:n]=['TEMPLATE']*n
    return labels

def window_origin(labels,pos):
    kinds=set(labels[pos:pos+3])
    return next(iter(kinds)) if len(kinds)==1 else 'CROSS_BOUNDARY' if 'UNKNOWN' not in kinds else 'UNKNOWN'

def hypothetical_residency(prompts,users,block_ids):
    stores=defaultdict(OrderedDict);resident={};evictions=0
    for i,t in enumerate(prompts):
        if block_ids is not None and (i==0 or block_ids[i]!=block_ids[i-1]):stores.clear()
        aid=t['route']['adapter_id'];cache=stores[aid];resident[i]=set(cache)
        matches=[k for k,s in cache.items() if 0<len(s['token_ids'])<len(t['token_ids']) and s['token_ids']==t['token_ids'][:len(s['token_ids'])] and sharing(users,s['route']['user_id'],t['route']['user_id'],aid,aid) is None]
        if matches:cache.move_to_end(max(matches,key=lambda k:len(cache[k]['token_ids'])))
        elif len(t['token_ids'])*491520<=CAPACITY:
            while cache and (sum(len(s['token_ids'])*491520 for s in cache.values())+len(t['token_ids'])*491520>CAPACITY or len(cache)>=64):cache.popitem(last=False);evictions+=1
            cache[i]=t
    return resident,evictions

def audit_prompts(prompts,users,block_size=5):
    require(0<len(prompts)<=400 and 1<=block_size<=20,'bounded natural workload requires 1..400 records; explicit separate sampling needed for larger data')
    require(len({p['sample']['id'] for p in prompts})==len(prompts),'duplicate natural sample IDs')
    packed=pack_groups(prompts,block_size);block_map={p['sample']['id']:i for i,b in enumerate(packed) for p in b}
    block_ids=[block_map[p['sample']['id']] for p in prompts]
    indexes=[grams(p['token_ids']) for p in prompts];postings=defaultdict(set)
    short,se= hypothetical_residency(prompts,users,block_ids)
    long,le= hypothetical_residency(prompts,users,None)
    rows=[];candidate_requests=set()
    for j,t in enumerate(prompts):
        # Inverted index identifies candidate sources; no occurrence Cartesian product.
        candidate=set().union(*(postings[g] for g in indexes[j])) if indexes[j] else set()
        for i,s in enumerate(prompts[:j]):
            x,y=s['token_ids'],t['token_ids'];common=lcp(x,y);strict=0<len(x)<len(y) and common==len(x)
            authorized=sharing(users,s['route']['user_id'],t['route']['user_id'],s['route']['adapter_id'],t['route']['adapter_id']) is None
            shared=set(indexes[i])&set(indexes[j]) if i in candidate else set()
            occurrences=same=non_template=0;origins=Counter();positions=Counter();examples=[]
            for g in sorted(shared):
                a,b=indexes[i][g],indexes[j][g];n=len(a)*len(b);occurrences+=n;same+=len(set(a)&set(b))
                ca=Counter(window_origin(s['origins'],p) for p in a);cb=Counter(window_origin(t['origins'],p) for p in b)
                template=ca['TEMPLATE']*cb['TEMPLATE'];non_template+=n-template
                for ka,na in ca.items():
                    for kb,nb in cb.items():origins[ka+'->'+kb]+=na*nb
                for pos in b:positions['target_'+('prefix' if pos<3 else 'suffix' if pos>=len(y)-5 else 'middle')]+=len(a)
                for pos in a:positions['source_'+('prefix' if pos<3 else 'suffix' if pos>=len(x)-5 else 'middle')]+=len(b)
                if len(examples)<8:examples.append(dict(gram=list(g),source_positions=a,target_positions=b))
            checks=dict(strict_prefix=strict,authorized=authorized,source_capacity=len(x)*491520<=CAPACITY,source_completed_before_target='ASSUMED_SEQUENTIAL_NOT_OBSERVED',same_short_block=block_ids[i]==block_ids[j],
                        hypothetical_short_resident=i in short[j],hypothetical_sequential_resident=i in long[j])
            if shared:candidate_requests.add(j)
            rows.append(dict(source_id=s['sample']['id'],target_id=t['sample']['id'],source_index=i,target_index=j,source_route=s['route'],target_route=t['route'],source_tokens=len(x),target_tokens=len(y),source_shorter=len(x)<len(y),equal_length=len(x)==len(y),full_sequence_equal=x==y,
                common_prefix_tokens=common,strict_reuse_fraction=len(x)/len(y) if strict else 0,partial_common_prefix_fraction=common/len(y),longest_contiguous_tokens=longest_contiguous(x,y) if shared else common,
                raw_w3_candidate=bool(shared),unique_matching_3grams=len(shared),matching_occurrence_pairs=occurrences,same_position_occurrences=same,shifted_position_occurrences=occurrences-same,
                non_template_only_occurrences=non_template,proven_content_occurrences=sum(v for k,v in origins.items() if all(x in ('CONTEXT','QUESTION','HISTORY') for x in k.split('->'))),origin_occurrence_pairs=dict(origins),position_occurrence_pairs=dict(positions),position_bins='prefix:start<3; suffix:start>=sequence_length-5; middle:other; precedence prefix',position_examples=examples,position_examples_coverage='FIRST_8_GRAMS; full positions reproducible from recorded token IDs',
                **checks,rejections=[k for k in ('strict_prefix','authorized','source_capacity','same_short_block') if not checks[k]],semantic_acceptance='NOT_EVALUATED',physical_hit='NOT_EVALUATED'))
        for g in indexes[j]:postings[g].add(j)
    eligible=lambda r:r['strict_prefix'] and r['authorized'] and r['source_capacity']
    return dict(scope='STATIC_ONLY',requests=len(prompts),chronological_pairs=len(rows),strict_prefix_pairs=sum(r['strict_prefix'] for r in rows),equal_full_prompt_pairs=sum(r['full_sequence_equal'] for r in rows),raw_w3_candidate_pairs=sum(r['raw_w3_candidate'] for r in rows),requests_with_raw_w3_candidates=len(candidate_requests),candidate_request_denominator=len(prompts),non_template_candidate_pairs=sum(r['non_template_only_occurrences']>0 for r in rows),
        rates=dict(exact_prefix_pairs=sum(r['strict_prefix'] for r in rows)/len(rows) if rows else None,raw_w3_candidate_pairs=sum(r['raw_w3_candidate'] for r in rows)/len(rows) if rows else None,raw_w3_candidate_requests=len(candidate_requests)/len(prompts),pair_denominator=len(rows),request_denominator=len(prompts)),
        prefix_eligibility=dict(unrestricted_authorized_capacity_pairs=sum(eligible(r) for r in rows),short_lifetime_hypothetically_resident_pairs=sum(eligible(r) and r['hypothetical_short_resident'] for r in rows),sequential_hypothetically_resident_pairs=sum(eligible(r) and r['hypothetical_sequential_resident'] for r in rows)),
        hypothetical_LRU=dict(assumptions='All sources complete; every eligible miss admitted; no semantic gate; never measured residency/HIT',short_block_size=block_size,short_evictions=se,sequential_evictions=le),independent_original_contexts=len({digest_bytes(p['sample']['context'].encode()) for p in prompts}),natural_actual_hit_status='NOT_EVALUATED',rows=rows)

def cpu_compute(registry,multi,input_limit=1280,block_size=5,seed=42,max_samples=200):
    require(1<=max_samples<=400,'explicit natural audit sample bound 1..400 required')
    paths,m,pm,specs,users,tok=assets(registry,multi);results={};provenance_records={}
    for name in DATASETS:
        if name not in paths or not Path(paths[name]).is_dir():results[name]=dict(status='DATASET_NOT_READY',acquisition='Use 118_prepare_r5_i5.py dataset with actual local source, immutable revision and split');continue
        samples,manifest=provenance(paths[name],name)
        original=Path(manifest['source_path']);upstream='UNAVAILABLE; recorded source hashes only'
        if original.exists():
            from .r4c_dataset import file_hashes
            require(file_hashes(original)==manifest['source_files'],'available upstream source hashes differ');upstream='AVAILABLE_SHA256_VERIFIED'
        provenance_records[name]=dict(manifest=manifest,inventory=Evidence(paths[name]).verify(),upstream=upstream)
        prepared=[];excluded=[]
        for s in samples:
            try:
                p=prepare_prompt(s,tok.tokenizer,2048,BUDGETS[name],allow_truncation=True,input_limit=input_limit);p['route']=routing(s,seed);p['origins']=token_origins(p,tok.tokenizer);prepared.append(p)
            except ValueError as exc:excluded.append(dict(id=s['id'],reason=str(exc)))
        # Actual existing I5 deterministic selection order, retaining all eligible records.
        full_eligible=len(prepared);selection=None
        if prepared:
            ordered,selection=select(samples,tok.tokenizer,count=min(max_samples,len(prepared)),seed=seed,context_limit=2048,budget=BUDGETS[name],allow_truncation=True,input_limit=input_limit)
            by={p['sample']['id']:p for p in prepared};prepared=[by[p['sample']['id']] for p in ordered]
        r=audit_prompts(prepared,users,block_size) if prepared else dict(scope='STATIC_ONLY',requests=0,rows=[])
        r.update(status='DATASET_READY' if prepared else 'NO_ELIGIBLE_REQUEST',original_samples=len(samples),full_eligible_count=full_eligible,selected_sample_count=len(prepared),selection=selection,excluded=excluded,truncated_count=sum(p['truncated'] for p in prepared),
                 prompts=[dict(sample=p['sample'],route=p['route'],token_ids=p['token_ids'],origins=p['origins'],prompt_sha256=digest(p['token_ids']),truncated=p['truncated']) for p in prepared],
                 controlled_recommendations=[dict(classification='CONTROLLED_EXACT_PREFIX',sample_id=p['sample']['id'],prefix_sha256=digest(p['token_ids'][:512]),prefix_tokens=512) for p in prepared if len(p['token_ids'])>512][:3],
                 gpu_recommendation='CONTROLLED_PILOT_ONLY_NO_NATURAL_HIT_CLAIM' if any(len(p['token_ids'])>512 for p in prepared) else 'NO_GO')
        results[name]=r
    return dict(results=results,provenance=provenance_records,model=pm['identity'],tokenizer=tok.identity,multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),template=TEMPLATE,budgets=BUDGETS,input_limit=input_limit,block_size=block_size,seed=seed,GPU_NOT_RUN=True)

def cpu(registry,multi,output_dir,input_limit=1280,block_size=5,seed=42,max_samples=200):
    result=cpu_compute(registry,multi,input_limit,block_size,seed,max_samples)
    out=output(output_dir,(registry,multi,*read(registry).values()));config=dict(registry=str(Path(registry).resolve()),multi=str(Path(multi).resolve()),input_limit=input_limit,block_size=block_size,seed=seed,max_samples=max_samples)
    write(out/'experiment2_config.json',dict(arguments=config,source=source_identity(),registry_sha256=sha(registry)))
    # Per-pair JSONL avoids the historical 16MiB per-JSON limit.
    import csv
    for name,r in result['results'].items():
        with (out/(name+'_pairs.jsonl')).open('w') as f:
            for row in r.pop('rows',[]):f.write(json.dumps(row,sort_keys=True)+'\n')
        with (out/(name+'_prompts.jsonl')).open('w') as f:
            for row in r.pop('prompts',[]):f.write(json.dumps(row,sort_keys=True)+'\n')
    write(out/'experiment2_summary.json',result)
    with (out/'experiment2_summary.csv').open('w') as f:
        fields=['dataset','status','requests','chronological_pairs','strict_prefix_pairs','raw_w3_candidate_pairs','requests_with_raw_w3_candidates','non_template_candidate_pairs'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for k,v in result['results'].items():w.writerow(dict(dataset=k,**{a:v.get(a,'NOT_EVALUATED') for a in fields[1:]}))
    lines=['# Natural reuse audit','','STATIC_ONLY. Raw w3 candidates are not semantic acceptance or physical HITs. Unknown provenance remains unknown; non-template-only counts include unknowns and are not proven content-only counts. Residency assumes admission/semantic success. Controlled recommendations are excluded from natural counts.','','| Dataset | Status | Eligible requests | Earlier/later pairs | Strict prefix pairs | Raw w3 pairs | Raw w3 requests | Independent contexts | GPU recommendation |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for name,r in result['results'].items():lines.append('| '+' | '.join(str(x) for x in (name,r['status'],*[r.get(k,'NOT_EVALUATED') for k in ('requests','chronological_pairs','strict_prefix_pairs','raw_w3_candidate_pairs','requests_with_raw_w3_candidates','independent_original_contexts','gpu_recommendation')]))+' |')
    lines+=['','Counts use eligible original requests; pair denominator is chronological m*(m-1)/2. Repetitions are not independent contexts. Position/provenance breakdowns and every independently evaluated eligibility check are in pair JSONL. Full token IDs/origins are recorded in prompt JSONL. Semantic and physical runtime outcomes remain NOT_EVALUATED.']
    (out/'experiment2_report.md').write_text('\n'.join(lines)+'\n')
    seal(out);return dict(status='CPU_AUDIT_COMPLETE',datasets={k:v['status'] for k,v in result['results'].items()},GPU_NOT_RUN=True)

def cpu_verify(audit,output_dir):
    origin=Path(audit).resolve();dest=Path(output_dir).resolve()
    require(not dest.is_relative_to(origin) and not origin.is_relative_to(dest),'verification output overlaps immutable audit')
    ev=Evidence(audit);receipt=ev.verify();cfg=ev.json('experiment2_config.json')
    require(cfg['source']==source_identity() and sha(cfg['arguments']['registry'])==cfg['registry_sha256'],'CPU audit source/registry differs')
    # Re-execute the full deterministic audit into a fresh sibling; compare all
    # exported records, not merely a trusted summary or free-form PASS flag.
    out=Path(output_dir);result=cpu(output_dir=out,**cfg['arguments'])
    expected=Evidence(out);expected.verify()
    require(ev.inventory==expected.inventory,'independent CPU recomputation differs')
    return dict(status='INDEPENDENT_CPU_VERIFY_PASS',input_inventory=receipt,**{k:v for k,v in result.items() if k!='status'})
