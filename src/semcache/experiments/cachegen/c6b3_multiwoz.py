"""B3-0 history preparation. No OPT weights, training, or task evaluation."""
import argparse
import json
import math
import os
import subprocess
import platform
from collections import defaultdict, Counter
from pathlib import Path
from semcache.experiments import m9b_semantic_workload as m9
from semcache.experiments.user_assignment import assign_users
from semcache.experiments.cachegen.c6_quality import select, RULE, BLEU
from semcache.simulation.multi_user import digest

VERSION = 'c6b3_multiwoz_history_v1'
SCOPE = dict(task_capability_stage='C6-B3-0', paper_bleu_claimed=False,
             protocol_provenance='REPRODUCTION_CHOICE', bleu_protocol=BLEU,
             training_performed=False, quality_evaluation_performed=False)


def reconstruct(rows, k):
    if type(k) is not int or not 0 <= k <= 4:
        raise ValueError('History K must be explicitly chosen from 0..4')
    previous = defaultdict(list)
    seen = set()
    output = []
    splits = {}
    for i, row in enumerate(rows):
        cid = row.get('conversation_id')
        if not isinstance(cid, str) or not cid or row.get('dataset') != 'multiwoz':
            raise ValueError('Need MultiWOZ conversation identity')
        if row['source_id'] in seen or row.get('global_query_index') != i:
            raise ValueError('Duplicate identity or invalid stable workload order')
        seen.add(row['source_id'])
        for field in ('query_text', 'reference_text'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing {field}')
        if cid in splits and splits[cid] != row['source_split']:
            raise ValueError('Ambiguous conversation across source splits')
        splits[cid] = row['source_split']
        tid = str(row['metadata']['turn_id'])
        if not tid.isdecimal() or int(tid) != 2*len(previous[cid]):
            raise ValueError('Require complete ordered prepared user turns 0,2,4,...; missing context')
        history = previous[cid][-k:] if k else []
        lines = ['Dialogue:']
        for old in history:
            lines.extend(['User: '+old['query_text'], 'Assistant: '+old['reference_text']])
        lines.extend(['User: '+row['query_text'], 'Assistant:'])
        prompt = '\n'.join(lines)
        output.append(dict(row, query_text=prompt, prompt_text=prompt,
                           current_user_text=row['query_text'], turn_id=tid,
                           prompt_version=VERSION, history_policy=dict(last_complete_exchanges=k),
                           history_source_ids=[r['source_id'] for r in history],
                           prompt_sha256=digest(prompt), reference_sha256=digest(row['reference_text'])))
        previous[cid].append(row)
    return output


def stats(values, bound):
    values = sorted(values)
    if not values:
        raise ValueError('Empty population')
    return {**{f'p{p}': values[max(0, math.ceil(len(values)*p/100)-1)] for p in (50,90,95,99)},
            'max': max(values), 'count': len(values), 'fraction_within_bound': sum(v<=bound for v in values)/len(values)}


def lengths(rows, tokenizer):
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must provide EOS')
    result = []
    for row in rows:
        prompt = tokenizer(row['prompt_text'], add_special_tokens=True, truncation=False)['input_ids']
        completion = tokenizer(' '+row['reference_text'], add_special_tokens=False, truncation=False)['input_ids']
        result.append(dict(source_id=row['source_id'], prompt_tokens=len(prompt),
                           completion_tokens=len(completion)+1, total_tokens=len(prompt)+len(completion)+1))
    return result


def analyze(rows, tokenizer, bound):
    result = {}
    for k in range(5):
        measured = lengths(reconstruct(rows,k),tokenizer)
        result[str(k)] = {field: stats([v[field] for v in measured],bound)
                         for field in ('prompt_tokens','total_tokens')}
    return result


def plan(rows, counts, seed=42):
    episodes = select(rows, 'multiwoz', 32, seed)
    holdout = set()
    for episode in episodes:
        for side in ('source','target'):
            row = rows[episode[side+'_index']]
            episode[side+'_conversation_id'] = row['conversation_id']
            episode[side+'_prompt_sha256'] = row['prompt_sha256']
            episode[side+'_reference_sha256'] = row['reference_sha256']
            holdout.add(row['conversation_id'])
    train = [r for r in rows if r['conversation_id'] not in holdout]
    if not train:
        raise ValueError('No training conversations remain')
    assignments = assign_users(train,2,seed,'seeded_group_balanced')
    count_map = {r['source_id']:r for r in counts}
    users = {}
    for logical, name in [('user_000','user_a'),('user_001','user_b')]:
        subset = [r for r,u in zip(train,assignments) if u==logical]
        users[name] = dict(row_ids=[r['source_id'] for r in subset],
            conversation_ids=sorted({r['conversation_id'] for r in subset}),
            row_count=len(subset), conversation_count=len({r['conversation_id'] for r in subset}),
            domains=dict(Counter(str(r.get('domain_or_intent')) for r in subset)),
            expected_tokens=sum(count_map[r['source_id']]['total_tokens'] for r in subset),
            supervised_tokens=sum(count_map[r['source_id']]['completion_tokens'] for r in subset))
    train_ids=[r['source_id'] for r in train]
    train_cids=sorted({r['conversation_id'] for r in train})
    return dict(**SCOPE, episodes=episodes, selection_sha256=digest(episodes), selection_rule=RULE,
                target_conversation_count=len({e['target_conversation_id'] for e in episodes}),
                cross_user_count=sum(e['cross_user'] for e in episodes)), dict(**SCOPE,
        users=users, seed=seed, train_row_ids=train_ids, train_row_ids_sha256=digest(train_ids),
        training_conversation_ids=train_cids, training_conversation_ids_sha256=digest(train_cids),
        holdout_conversation_ids=sorted(holdout), holdout_conversation_ids_sha256=digest(sorted(holdout)),
        assignment='seeded_group_balanced on training conversations after holdout exclusion',
        objective='prompt labels -100; separately tokenized leading ASCII space + response + EOS supervised',
        adapter=dict(r=8,lora_alpha=8,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj'],
                     bias='none',task_type='CAUSAL_LM',base_model_frozen=True))


def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2)


def tokenizer_only():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    from semcache.models.tokenizer_provenance import resolve_tokenizer_provenance, validate_tokenizer_snapshot
    asset=cached_file(m9.MODEL_ID,'tokenizer_config.json',revision=m9.MODEL_REVISION,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(str(Path(asset).parent),local_files_only=True)
    metadata=resolve_tokenizer_provenance(tokenizer,model_source_id=m9.MODEL_ID,
        tokenizer_source_id=m9.MODEL_ID,model_requested_revision=m9.MODEL_REVISION,
        tokenizer_requested_revision=m9.MODEL_REVISION,model_resolved_revision=None,
        model_revision_explicit=True,tokenizer_revision_explicit=True)
    validate_tokenizer_snapshot(dict(model_id=m9.MODEL_ID,model_revision=m9.MODEL_REVISION,**metadata))
    return tokenizer,metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--stage',choices=['analyze','generate','validate'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--analysis',type=Path)
    p.add_argument('--history-k',type=int,choices=range(5))
    p.add_argument('--max-sequence-length',type=int,default=256)
    p.add_argument('--semantic-output',type=Path)
    a=p.parse_args(argv)
    if a.stage=='validate':
        manifest=json.loads((a.output_dir/'manifest.json').read_text())
        if m9.file_hash(a.source)!=manifest['source_sha256']:
            raise ValueError('Source hash mismatch')
        for path,sha in manifest['output_hashes'].items():
            if m9.file_hash(path)!=sha: raise ValueError('Artifact hash mismatch: '+path)
        print('Validated source and all frozen output hashes; no model loaded')
        return
    if a.max_sequence_length<1: raise ValueError('Invalid sequence bound')
    if a.output_dir.exists(): raise ValueError('Output directory must be new')
    rows=m9.read_raw(a.source,'multiwoz')
    reconstruct(rows,0)  # audit before loading even a tokenizer
    tokenizer,token_metadata=tokenizer_only()
    base=dict(**SCOPE,source_path=str(a.source.resolve()),source_sha256=m9.file_hash(a.source),
              model_id=m9.MODEL_ID,model_revision=m9.MODEL_REVISION,tokenizer=token_metadata,
              prompt_version=VERSION,max_sequence_length=a.max_sequence_length,
              git={key:subprocess.check_output(['git',*cmd],text=True).strip() for key,cmd in
                   [('branch',['branch','--show-current']),('commit',['rev-parse','HEAD']),('status',['status','--porcelain'])]},
              python_version=platform.python_version(),completion_serialization='one leading ASCII space + original reference response',
              source_audit='SERAPH schema verified externally; contiguous per-conversation turn audit enforced at runtime')
    if a.stage=='analyze':
        write(a.output_dir/'length_stats.json',dict(base,statistics=analyze(rows,tokenizer,a.max_sequence_length)))
        print('Saved K=0..4 nearest-rank length statistics; choose K before generation')
        return
    if a.history_k is None or a.analysis is None or a.semantic_output is None:
        raise ValueError('Generation requires --history-k, --analysis and --semantic-output')
    analysis=json.loads(a.analysis.read_text())
    for key in ('source_sha256','model_revision','tokenizer','prompt_version','max_sequence_length'):
        if analysis[key]!=base[key]: raise ValueError('Analysis provenance mismatch: '+key)
    if a.semantic_output.exists() or Path(str(a.semantic_output)+'.manifest.json').exists():
        raise ValueError('Semantic output must be new')
    prompts=reconstruct(rows,a.history_k)
    counts=lengths(prompts,tokenizer)
    # No length-based row filtering or implicit truncation; increase bound explicitly if needed.
    if any(c['total_tokens']>a.max_sequence_length for c in counts):
        raise ValueError('Selected K exceeds bound; choose a smaller K or analyze a larger explicit bound. No rows removed.')
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.utils.seed import seed_everything
    seed_everything(42)
    encoder=TinyBERTSemanticEncoder(revision=m9.ENCODER_REVISION,pooling='masked_mean',max_length=512,
                                  device='cpu',dtype='float32',local_files_only=True)
    semantic,diagnostics=m9.convert_rows(prompts,'multiwoz',tokenizer,encoder,
        dict(tokenizer=token_metadata,semantic_encoder=encoder.metadata,execution_provenance='MEASURED'))
    for row in semantic: row['special_token_ids']=list(tokenizer.all_special_ids)
    selection,training=plan(semantic,counts)
    training.update(history_k=a.history_k,max_sequence_length=a.max_sequence_length,
                    response_length_statistics=stats([v['completion_tokens'] for v in counts],a.max_sequence_length))
    a.semantic_output.parent.mkdir(parents=True,exist_ok=True)
    with a.semantic_output.open('x',encoding='utf8') as stream: stream.write(m9.serialize(semantic))
    outputs={a.output_dir/'evaluation_selection.json':selection,a.output_dir/'training_plan.json':training,
             a.output_dir/'length_stats.json':analysis}
    for path,value in outputs.items(): write(path,value)
    manifest=dict(base,history_k=a.history_k,analysis_sha256=m9.file_hash(a.analysis),
        semantic_encoder=encoder.metadata,clustering=diagnostics,seed=42,
        physical_safety_contract='PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER',
        output_hashes={str(path.resolve()):m9.file_hash(path) for path in [*outputs,a.semantic_output]})
    write(a.output_dir/'manifest.json',manifest)
    print(json.dumps(dict(training_users=training['users'],selection_count=32,
        target_conversations=selection['target_conversation_count'],cross_user_count=selection['cross_user_count'])))
