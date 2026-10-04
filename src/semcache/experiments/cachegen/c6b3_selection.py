"""Model-free B3 v2 reselection; immutable semantic input and offset-only audit."""
import json
from collections import defaultdict, Counter
from pathlib import Path
from semcache.experiments.cachegen import c6b3_multiwoz as b
from semcache.simulation.multi_user import logical_user_assignment, digest

RULE = ('Current-user offset-contained exact-w3 only; earliest source/span per target; '
        'eight targets per history-depth bucket 1/2/3/4+; deterministic augmenting '
        'matching in bucket and workload order enforces 32 distinct target conversations')


def annotate(rows, tokenizer):
    if not tokenizer.is_fast:
        raise ValueError('Fast pinned tokenizer with offset_mapping required')
    output=[]
    for row in rows:
        text=row['prompt_text']; current=row['current_user_text']
        suffix='User: '+current+'\nAssistant:'
        if not text.endswith(suffix) or text!=row['query_text'] or row['prompt_version']!=b.VERSION:
            raise ValueError('Prompt serialization mismatch')
        start=len(text)-len(suffix)+len('User: '); end=start+len(current)
        encoded=tokenizer(text,add_special_tokens=True,truncation=False,return_offsets_mapping=True)
        if encoded['input_ids']!=row['token_ids']:
            raise ValueError('Frozen token IDs differ from pinned tokenizer')
        eligible=[i for i,(a,z) in enumerate(encoded['offset_mapping']) if start<=a<z<=end]
        depth=len(row['history_source_ids'])
        if depth!=min(int(row['turn_id'])//2,4) or row['history_policy']!={'last_complete_exchanges':4}:
            raise ValueError('Frozen K=4 history mismatch')
        output.append(dict(row,current_user_char_span=[start,end],current_user_token_indices=eligible,
                           history_depth=depth))
    return output


def select(rows):
    users=logical_user_assignment(rows,2,42,'multiwoz')
    index=defaultdict(list); candidates={k:{} for k in range(1,5)}
    for ti,t in enumerate(rows):
        if (t['semantic_execution_provenance']!='MEASURED' or t['model_revision']!=b.m9.MODEL_REVISION
            or t['tokenizer_id']!=f'{b.m9.MODEL_ID}@{b.m9.MODEL_REVISION}'
            or t['semantic_assignment_source']!=b.m9.assignment_source('multiwoz') or not t['reference_text'].strip()):
            raise ValueError('Invalid measured workload provenance/reference')
        allowed=set(t['current_user_token_indices']); specials={0,1,2}|set(t.get('special_token_ids',[]))
        windows=[(p,tuple(t['token_ids'][p:p+3])) for p in sorted(allowed)
                 if p>=3 and {p,p+1,p+2}<=allowed and not specials.intersection(t['token_ids'][p:p+3])]
        hits=[]
        for tp,ids in windows:
            for si,sp in index[t['cluster_id'],ids]:
                s=rows[si]
                if s['source_id']!=t['source_id'] and s['token_ids']!=t['token_ids']:
                    hits.append((si,sp,tp,ids));break
        depth=t['history_depth']
        if hits and depth>=1 and int(t['turn_id'])>0:
            si,sp,tp,ids=min(hits); s=rows[si]
            user=lambda i:{'user_000':'user_a','user_001':'user_b'}[users[i]]
            e=dict(episode_id=f'multiwoz:{si}:{ti}:{sp}:{tp}',dataset='multiwoz',
                source_index=si,target_index=ti,source_id=s['source_id'],target_id=t['source_id'],
                source_user=user(si),target_user=user(ti),cross_user=users[si]!=users[ti],
                source_start=sp,target_start=tp,token_ids=list(ids),cluster=t['cluster_id'],
                cache_key=[t['cluster_id'],list(ids)],selected_hit_count=1,
                admission='cold single-entry; normal admission must succeed',
                reference_text=t['reference_text'],reference_sha256=t['reference_sha256'],
                target_turn_id=t['turn_id'],history_depth=depth,domain=t.get('domain_or_intent'),
                m9b_strict_eligible=False,m9b_strict_diagnostic_only=True)
            candidates[min(depth,4)].setdefault(t['conversation_id'],e)
        for sp,ids in windows:index[t['cluster_id'],ids].append((ti,sp))
    # Bipartite matching avoids a greedy bucket consuming another bucket's only dialogues.
    owners={}; assigned={}
    def match(slot,seen):
        for cid,e in candidates[slot[0]].items():
            if cid in seen:continue
            seen.add(cid)
            if cid not in owners or match(owners[cid],seen):
                owners[cid]=slot;assigned[slot]=e;return True
        return False
    counts={k:len(v) for k,v in candidates.items()}
    for bucket in range(1,5):
        for n in range(8):
            if not match((bucket,n),set()):
                raise ValueError(f'Cannot satisfy 8 per bucket with distinct conversations; candidate conversation counts={counts}')
    return [assigned[slot] for slot in sorted(assigned)],counts


def run(args):
    if not args.existing_plan or not args.semantic_input:
        raise ValueError('Reselect requires --existing-plan and --semantic-input')
    if args.output_dir.exists():raise ValueError('New output directory required')
    old=json.loads((args.existing_plan/'manifest.json').read_text())
    if old['history_k']!=4 or old['max_sequence_length']!=384 or old['prompt_version']!=b.VERSION or old['bleu_protocol']!=b.BLEU:
        raise ValueError('Frozen formulation/protocol mismatch')
    if b.m9.file_hash(args.source)!=old['source_sha256']:raise ValueError('Source hash mismatch')
    for path,sha in old['output_hashes'].items():
        if b.m9.file_hash(path)!=sha:raise ValueError('Prior artifact hash mismatch')
    if old['output_hashes'].get(str(args.semantic_input.resolve()))!=b.m9.file_hash(args.semantic_input):
        raise ValueError('Semantic input not bound to original manifest')
    raw=b.m9.read_raw(args.source,'multiwoz'); expected=b.reconstruct(raw,4)
    rows=b.m9.read_raw(args.semantic_input,'multiwoz')
    if len(rows)!=len(expected):raise ValueError('Row count mismatch')
    for r,e in zip(rows,expected):
        for key in ('source_id','conversation_id','prompt_text','query_text','current_user_text','reference_text','history_source_ids','turn_id','prompt_sha256','reference_sha256'):
            if r[key]!=e[key]:raise ValueError('Frozen prompt/source mismatch: '+key)
    tokenizer,metadata=b.tokenizer_only()
    if metadata!=old['tokenizer']:raise ValueError('Tokenizer provenance mismatch')
    annotated=annotate(rows,tokenizer)
    episodes,candidate_counts=select(annotated)
    counts=b.lengths(rows,tokenizer)
    if any(c['total_tokens']>384 for c in counts):raise ValueError('Frozen length bound exceeded')
    selection,training=b.plan(rows,counts,episodes=episodes,selection_rule=RULE)
    hold=set(training['holdout_conversation_ids']);train=set(training['training_conversation_ids'])
    a,c=training['users'].values()
    if hold&train or set(a['conversation_ids'])&set(c['conversation_ids']):raise ValueError('Conversation leakage')
    windows=Counter(tuple(e['token_ids']) for e in episodes)
    audit=dict(episode_count=len(episodes),distinct_target_conversations=selection['target_conversation_count'],
        target_turn_zero_count=sum(int(e['target_turn_id'])==0 for e in episodes),
        history_depth_distribution=dict(Counter(e['history_depth'] for e in episodes)),
        unique_exact_w3_windows=len(windows),most_frequent_window_count=max(windows.values()),
        template_scaffold_window_count=0,unique_source_conversations=len({e['source_conversation_id'] for e in episodes}),
        user_directions=dict(Counter(e['source_user']+' -> '+e['target_user'] for e in episodes)),
        cross_user_count=selection['cross_user_count'],domain_coverage=dict(Counter(str(e['domain']) for e in episodes)),
        train_holdout_overlap=0,training_conversations_single_user=True,all_eval_conversations_excluded=True,
        candidate_conversation_counts=candidate_counts)
    if audit['episode_count']!=32 or audit['distinct_target_conversations']!=32 or audit['target_turn_zero_count']:
        raise ValueError('Selection audit failed')
    for e in episodes:
        for side in ('source','target'):
            r=annotated[e[side+'_index']];p=e[side+'_start']
            if not {p,p+1,p+2}<=set(r['current_user_token_indices']):raise ValueError('Scaffold hit')
            e[side+'_current_user_char_span']=r['current_user_char_span']
            e[side+'_current_user_token_indices']=r['current_user_token_indices']
    selection.update(selection_sha256=digest(episodes),audit=audit)
    training.update(history_k=4,max_sequence_length=384)
    outputs={'current_user_spans.json':[dict(source_id=r['source_id'],history_depth=r['history_depth'],
        current_user_char_span=r['current_user_char_span'],current_user_token_indices=r['current_user_token_indices']) for r in annotated],
             'evaluation_selection.json':selection,'training_plan.json':training,'selection_audit.json':audit,
             'length_stats.json':json.loads((args.existing_plan/'length_stats.json').read_text())}
    for name,value in outputs.items():b.write(args.output_dir/name,value)
    manifest=dict(old,selection_version='current_user_content_v2',prior_manifest_sha256=b.m9.file_hash(args.existing_plan/'manifest.json'),
        semantic_input_sha256=b.m9.file_hash(args.semantic_input),clustering_rerun=False,
        git={key:b.subprocess.check_output(['git',*cmd],text=True).strip() for key,cmd in
             [('branch',['branch','--show-current']),('commit',['rev-parse','HEAD']),('status',['status','--porcelain'])]},
        output_hashes={str((args.output_dir/name).resolve()):b.m9.file_hash(args.output_dir/name) for name in outputs})
    manifest['output_hashes'][str(args.semantic_input.resolve())]=b.m9.file_hash(args.semantic_input)
    b.write(args.output_dir/'manifest.json',manifest)
    print(json.dumps(audit,indent=2))
